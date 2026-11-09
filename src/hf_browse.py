"""HF GGUF browse proxy — server-side search for GGUF models.

Uses Hugging Face Hub HTTP API directly (no huggingface_hub dep required for search)
so we can filter library=gguf and return only GGUF siblings.

Results are cached 5 min to stay friendly to HF rate limits.
Supports HF_TOKEN env for private/gated models.

Called by app.py endpoints:
  GET /api/hf/search          (fmt=gguf)
  GET /api/hf/models/{author}/{name}/gguf
"""
import asyncio
import datetime as _dt
import json
import os
import re
import threading
import time
from urllib.parse import quote

import httpx

from runtime_paths import data_dir

HF_API = "https://huggingface.co/api"
HF_ROOT = "https://huggingface.co"
CACHE_TTL = 300
# Full GGUF publisher index: HF retired the `skip` parameter in favour of
# the Link: rel="next" cursor chain. Walking the whole GGUF listing is
# surprisingly cheap (measured ~220 pages of 1000, 219k models, ~37k
# authors) so the "All companies" list is built from a complete index
# instead of a bounded crawl. Built in a background thread and persisted
# so app restarts resume where they left off.
AUTHOR_INDEX_PATH = os.path.join(data_dir(), "generated", "hf_snapshot",
                                 "gguf-authors.json")
AUTHOR_INDEX_TTL = 24 * 3600
AUTHOR_PAGE_SIZE = 1000
_author_index_lock = threading.Lock()
_author_index = {"authors": {}, "cursor": None, "done": False,
                 "models": 0, "updated": None, "running": False,
                 "error": None}
_author_index_loaded = False

# repo_id -> GGUF file count, so collection GGUF filtering (and repeat
# owners) is a one-time HF lookup per repo, persisted across restarts.
REPO_GGUF_CACHE_PATH = os.path.join(data_dir(), "generated", "hf_snapshot",
                                    "gguf-repos.json")
REPO_GGUF_TTL = 7 * 86400
_repo_gguf_lock = threading.Lock()
_repo_gguf = {}
_repo_gguf_loaded = False
_cache = {}  # key -> (expires_at, data)

_QUANT_RE = re.compile(r"(Q\d+_K_M|Q\d+_K_S|Q\d+_K_L|Q\d+_0|Q\d+_1|Q\d+|F16|F32|BF16|IQ\d+_[A-Z]+)", re.IGNORECASE)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(GB|MB)", re.IGNORECASE)

def _token():
    # vault first (single encrypted .env), then operator plaintext env, then legacy YAML
    try:
        import secrets_vault as _vault
        t = _vault.get_hf_token()
        if t:
            return t
    except Exception:
        pass
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    # try settings.yaml
    try:
        import yaml
        from pathlib import Path
        p = Path(data_dir()) / "config" / "settings.yaml"
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            v = (data.get("hf_token") or "").strip()
            if v:
                return v
    except Exception:
        pass
    return None

_ANON_HEADERS = {"User-Agent": "llama-rpc-farm/1.0"}

def _headers():
    h = dict(_ANON_HEADERS)
    t = _token()
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h

async def _get_public(client, url, params=None):
    """GET with the configured token, retrying anonymously if rejected.

    Fine-grained HF tokens are entity-scoped (e.g. `collection.read` only
    for the token owner's own collections). HF answers 403 for another
    owner's PUBLIC collection even though anonymous reads are allowed —
    this is why public browsing worked in a browser but failed in the app.
    Retry without Authorization on 401/403; if the anonymous request also
    fails, return the original token response so private/own content keeps
    its real error. Non-auth errors (404/5xx) are returned as-is.
    """
    resp = await client.get(url, params=params)
    if resp.status_code in (401, 403) and _token():
        async with httpx.AsyncClient(
                timeout=client.timeout, headers=dict(_ANON_HEADERS),
                follow_redirects=True) as anon:
            anon_resp = await anon.get(url, params=params)
            if anon_resp.status_code < 400:
                return anon_resp
    return resp

def _cache_get(key):
    ent = _cache.get(key)
    if ent and ent[0] > time.time():
        return ent[1]
    if ent:
        _cache.pop(key, None)
    return None

def _cache_set(key, data):
    _cache[key] = (time.time() + CACHE_TTL, data)
    # simple LRU cap
    if len(_cache) > 200:
        oldest = sorted(_cache.items(), key=lambda kv: kv[1][0])[0][0]
        _cache.pop(oldest, None)

def _parse_quant(filename):
    m = _QUANT_RE.search(filename or "")
    return m.group(1).upper() if m else ""

def _is_gguf(filename):
    return (filename or "").lower().endswith(".gguf")

_avatar_cache = {}  # author -> (expires, url)


# ---------------- full GGUF publisher index ----------------
def _index_next_cursor(link_header):
    """Extract the cursor from an HF `Link: <...>; rel="next"` header."""
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header or "")
    if not m:
        return None
    try:
        return httpx.URL(m.group(1)).params.get("cursor")
    except Exception:
        return None


def _merge_author_batch(state, batch):
    """Accumulate one page of models into the author dict (pure helper)."""
    authors = state["authors"]
    for m in batch or []:
        mid = m.get("id") or ""
        if not mid:
            continue
        a = m.get("author") or (mid.split("/")[0] if "/" in mid else "")
        if not a:
            continue
        ent = authors.setdefault(a, {"count": 0, "downloads": 0, "likes": 0})
        ent["count"] += 1
        ent["downloads"] += m.get("downloads") or 0
        ent["likes"] += m.get("likes") or 0
        state["models"] += 1


def _load_author_index():
    global _author_index_loaded
    if _author_index_loaded:
        return
    with _author_index_lock:
        if _author_index_loaded:
            return
        try:
            if os.path.exists(AUTHOR_INDEX_PATH):
                with open(AUTHOR_INDEX_PATH, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                _author_index.update({
                    "authors": blob.get("authors") or {},
                    "cursor": blob.get("cursor"),
                    "done": bool(blob.get("done")),
                    "models": int(blob.get("models") or 0),
                    "updated": blob.get("updated"),
                    "error": None,
                })
        except Exception:
            pass
        _author_index_loaded = True


def _save_author_index_locked():
    try:
        os.makedirs(os.path.dirname(AUTHOR_INDEX_PATH), exist_ok=True)
        tmp = AUTHOR_INDEX_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "authors": _author_index["authors"],
                "cursor": _author_index["cursor"],
                "done": _author_index["done"],
                "models": _author_index["models"],
                "updated": _author_index["updated"],
            }, f)
        os.replace(tmp, AUTHOR_INDEX_PATH)
    except Exception:
        pass


def _load_repo_gguf_cache():
    global _repo_gguf_loaded
    if _repo_gguf_loaded:
        return
    with _repo_gguf_lock:
        if _repo_gguf_loaded:
            return
        try:
            if os.path.exists(REPO_GGUF_CACHE_PATH):
                with open(REPO_GGUF_CACHE_PATH, "r", encoding="utf-8") as f:
                    _repo_gguf.update(json.load(f) or {})
        except Exception:
            pass
        _repo_gguf_loaded = True


def _repo_gguf_get(repo_id):
    _load_repo_gguf_cache()
    ent = _repo_gguf.get(repo_id.lower())
    if not ent:
        return None
    try:
        age = (_dt.datetime.now(_dt.timezone.utc)
               - _dt.datetime.fromisoformat(ent.get("t") or "")).total_seconds()
        if age > REPO_GGUF_TTL:
            return None
    except Exception:
        return None
    return ent.get("n")


def _repo_gguf_set(repo_id, count):
    with _repo_gguf_lock:
        _load_repo_gguf_cache()
        _repo_gguf[repo_id.lower()] = {
            "n": int(count),
            "t": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(REPO_GGUF_CACHE_PATH), exist_ok=True)
            tmp = REPO_GGUF_CACHE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_repo_gguf, f)
            os.replace(tmp, REPO_GGUF_CACHE_PATH)
        except Exception:
            pass


async def _repo_gguf_count(repo_id):
    """Number of .gguf siblings in a repo (None = unknown / check failed)."""
    repo_id = (repo_id or "").strip().strip("/")
    if "/" not in repo_id:
        return None
    cached = _repo_gguf_get(repo_id)
    if cached is not None:
        return cached
    author, name = repo_id.split("/", 1)
    try:
        async with httpx.AsyncClient(timeout=20, headers=_headers(),
                                     follow_redirects=True) as client:
            resp = await client.get(f"{HF_API}/models/{quote(author)}/{quote(name)}")
        if resp.status_code == 404:
            n = 0
        elif resp.status_code in (401, 403):
            return None  # private/unknown: don't hide the collection
        else:
            resp.raise_for_status()
            sibs = resp.json().get("siblings") or []
            n = sum(1 for s in sibs if _is_gguf(s.get("rfilename")))
    except Exception:
        return None
    _repo_gguf_set(repo_id, n)
    return n


def _index_is_stale():
    updated = _author_index.get("updated")
    if not updated:
        return True
    try:
        age = (_dt.datetime.now(_dt.timezone.utc)
               - _dt.datetime.fromisoformat(updated)).total_seconds()
        return age > AUTHOR_INDEX_TTL
    except Exception:
        return True


def _crawl_author_index():
    """Walk the whole GGUF listing via the cursor chain, persisting progress."""
    import httpx as _httpx
    url = HF_API + "/models"
    fails = 0
    try:
        while True:
            with _author_index_lock:
                if _author_index["done"]:
                    break
                cursor = _author_index["cursor"]
            params = {"filter": "gguf", "limit": str(AUTHOR_PAGE_SIZE),
                      "sort": "downloads", "direction": "-1"}
            if cursor:
                params["cursor"] = cursor
            try:
                with _httpx.Client(timeout=90, headers=_headers(),
                                   follow_redirects=True) as client:
                    resp = client.get(url, params=params)
                if resp.status_code == 429:
                    ra = resp.headers.get("retry-after") or "10"
                    try:
                        wait = min(float(ra), 60.0)
                    except ValueError:
                        wait = 10.0
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    fails += 1
                    if fails > 5:
                        with _author_index_lock:
                            _author_index["error"] = f"HTTP {resp.status_code}"
                        return
                    time.sleep(5 * fails)
                    continue
                if resp.status_code == 400 and cursor:
                    # Expired/invalid cursor: restart the scan cleanly rather
                    # than resume into the middle of a dead search state.
                    with _author_index_lock:
                        _author_index.update({"authors": {}, "models": 0,
                                              "cursor": None})
                    continue
                resp.raise_for_status()
                batch = resp.json() or []
                fails = 0
            except Exception as e:
                fails += 1
                with _author_index_lock:
                    _author_index["error"] = str(e)
                if fails > 5:
                    return
                time.sleep(5 * fails)
                continue
            with _author_index_lock:
                before = _author_index["models"]
                _merge_author_batch(_author_index, batch)
                link = resp.headers.get("link", "")
                nxt = _index_next_cursor(link)
                _author_index["cursor"] = nxt
                if not batch or not nxt or len(batch) < AUTHOR_PAGE_SIZE:
                    _author_index["done"] = True
                _author_index["updated"] = _dt.datetime.now(
                    _dt.timezone.utc).isoformat()
                _author_index["error"] = None
                _save_author_index_locked()
                advanced = _author_index["models"] - before
            if not advanced:
                with _author_index_lock:
                    _author_index["done"] = True
                    _save_author_index_locked()
                break
            time.sleep(0.05)
    finally:
        with _author_index_lock:
            _author_index["running"] = False
            _save_author_index_locked()


def ensure_author_index(force=False):
    """Start (or resume) the background index build; return a snapshot."""
    global _author_index_loaded
    _load_author_index()
    start = False
    with _author_index_lock:
        if force and _author_index["done"]:
            _author_index.update({"authors": {}, "models": 0, "cursor": None,
                                  "done": False, "updated": None})
        if not _author_index["running"]:
            if not _author_index["done"] or _index_is_stale():
                if _author_index["done"] and _index_is_stale():
                    # stale but complete: do a clean rebuild
                    _author_index.update({"authors": {}, "models": 0,
                                          "cursor": None, "done": False})
                _author_index["running"] = True
                _author_index["error"] = None
                start = True
        snap = {
            "authors": dict(_author_index["authors"]),
            "done": _author_index["done"],
            "running": _author_index["running"],
            "models": _author_index["models"],
            "error": _author_index["error"],
            "updated": _author_index["updated"],
        }
    if start:
        threading.Thread(target=_crawl_author_index, daemon=True).start()
    return snap


async def list_gguf_companies(limit: int = 20, search: str = "", pipeline_tag: str = "", quant: str = "", recommended: bool = False):
    """Curated big companies when recommended, else top by downloads — reflective of task/quant."""
    CURATED = [
        "openai", "google", "Qwen", "deepseek-ai", "microsoft",
        "meta-llama", "mistralai", "huihui-ai", "orcarouter",
        "unsloth", "ornith-ai",
    ]
    norm = {c.lower(): c for c in CURATED}
    limit = max(1, min(limit, 5000))
    key = f"companies:curated:{search}:{pipeline_tag}:{quant}:{limit}:{recommended}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    # All companies: serve from the complete GGUF publisher index (background
    # cursor crawl) instead of the old bounded top-downloads crawl. Search is
    # an exhaustive substring match over every indexed author, so any company
    # visible on Hugging Face is reachable no matter where it ranks.
    if not recommended:
        snap = ensure_author_index()
        needle = (search or "").strip().lower()
        rows = []
        for author, ent in snap["authors"].items():
            if needle and needle not in author.lower():
                continue
            rows.append({"author": author,
                         "count": ent.get("count") or 0,
                         "downloads": ent.get("downloads") or 0,
                         "likes": ent.get("likes") or 0})
        rows.sort(key=lambda x: (-x["count"], -x["downloads"]))
        result = {"companies": rows[:limit], "totalUnique": len(rows),
                  "curated": False, "scanned_models": snap["models"],
                  "scanning": bool(snap["running"] and not snap["done"]),
                  "complete": bool(snap["done"]), "error": snap["error"]}
        if snap["done"]:
            _cache_set(key, result)  # never cache a partial, still-growing list
        return result
    import asyncio
    if search and search.strip():
        needle = search.strip().lower()
        matched = [c for c in CURATED if needle in c.lower()]
        seen = {}
        try:
            data = await search_gguf(query=search or "", sort="downloads", limit=50, page=0, quant=quant or "", pipeline_tag=pipeline_tag or "", author="")
            for m in data.get("models") or []:
                a = m.get("author") or ""
                if not a or needle not in a.lower():
                    continue
                if a not in seen:
                    seen[a] = {"author": a, "count": 0, "downloads": 0, "likes": 0}
                seen[a]["count"] += 1
                seen[a]["downloads"] += m.get("downloads") or 0
        except Exception:
            pass
        # Recommended only: emit authors that actually returned GGUF results
        # (a curated name with no GGUF is not a usable recommendation).
        merged = {}
        for a, v in seen.items():
            merged[a.lower()] = {**v, "curated": a.lower() in norm}
        top = sorted(merged.values(), key=lambda x: (0 if x.get("curated") else 1, -x["downloads"]))[:limit]
        async def enrich(t):
            av = await get_avatar(t["author"])
            if av:
                t["avatarUrl"] = av
            return t
        if top:
            await asyncio.gather(*[enrich(t) for t in top])
        result = {"companies": top, "totalUnique": len(merged), "curated": True}
        _cache_set(key, result)
        return result
    # no search → curated when recommended, else top by downloads (every possible company)
    if recommended:
        seen = {}
        for author in CURATED:
            try:
                data = await search_gguf(query="", sort="downloads", limit=50, page=0, quant=quant or "", pipeline_tag=pipeline_tag or "", author=author)
                models = data.get("models") or []
                cnt = len(models)
                if cnt == 50:
                    try:
                        data2 = await search_gguf(query="", sort="downloads", limit=50, page=1, quant=quant or "", pipeline_tag=pipeline_tag or "", author=author)
                        if (data2.get("models") or []):
                            cnt = 50
                    except Exception:
                        pass
                if cnt:
                    m = models[0]
                    seen[author] = {"author": author, "count": cnt, "downloads": m.get("downloads") or 0, "likes": m.get("likes") or 0, "curated": True}
            except Exception:
                pass  # never recommend an author we couldn't verify has GGUF
            await asyncio.sleep(0.05)
        top = sorted(seen.values(), key=lambda x: (-x["count"], -x["downloads"]))[:limit]
        async def enrich2(t):
            av = await get_avatar(t["author"])
            if av:
                t["avatarUrl"] = av
            return t
        if top:
            await asyncio.gather(*[enrich2(t) for t in top])
        result = {"companies": top, "totalUnique": len(seen), "curated": True}
        _cache_set(key, result)
        return result

async def list_gguf_pipeline_tags(limit: int = 200):
    """Distinct pipeline_tags for GGUF models — for task filter dropdown (all llama.cpp capable types)."""
    key = f"pipeline_tags:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    seen = set()
    for page in range(4):
        try:
            data = await search_gguf(query="", sort="downloads", limit=50, page=page, quant="", pipeline_tag="", author="")
        except Exception:
            break
        for m in data.get("models") or []:
            pt = m.get("pipeline_tag") or "none"
            seen.add(pt or "none")
        if len(seen) > 20:
            break
        await asyncio.sleep(0.1)
    for must in ["text-generation", "feature-extraction", "image-text-to-text", "image-to-text", "text-to-image", "any-to-any", "automatic-speech-recognition", "text-to-speech"]:
        seen.add(must)
    order = ["text-generation", "feature-extraction", "image-text-to-text", "image-to-text", "text-to-image", "any-to-any", "automatic-speech-recognition", "text-to-speech"]
    sorted_tags = sorted(seen, key=lambda x: (order.index(x) if x in order else 99, x))
    result = {"pipeline_tags": sorted_tags}
    _cache_set(key, result)
    return result

async def list_collections(owner: str, limit: int = 20, pipeline_tag: str = "",
                           quant: str = "", gguf_only: bool = True):
    """List HF collections for an owner — task/quant-aware, GGUF-only by default.

    HF's collections listing ignores `skip` (it returns the same owner set),
    so a single request is both sufficient and avoids duplicate rows.
    With gguf_only (default) a collection is shown only when at least one of
    its model repos actually contains .gguf files; non-GGUF source-weight
    collections (e.g. Qwen3.8) simply don't appear in the picker.
    """
    owner = (owner or "").strip()
    if not owner:
        return {"collections": []}
    limit = max(1, min(limit, 100))
    # include task/quant/gguf in cache key
    key = f"collections:{owner}:{limit}:{pipeline_tag}:{quant}:gguf{int(gguf_only)}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = f"https://huggingface.co/api/collections"
    all_cols = []
    async with httpx.AsyncClient(timeout=15, headers=_headers(), follow_redirects=True) as client:
        params = {"owner": owner, "limit": str(limit)}
        resp = await _get_public(client, url, params=params)
        if resp.status_code != 404:
            resp.raise_for_status()
            all_cols = resp.json() or []
    # filter reflective of task: only collections with at least one model matching pipeline_tag
    if pipeline_tag and pipeline_tag != "all":
        filtered = []
        for c in all_cols:
            items = c.get("items") or []
            # check if any item's pipeline_tag matches (or collection itself is for that task)
            has = any((it.get("pipeline_tag") or "") == pipeline_tag for it in items if it.get("type") == "model")
            # also check tags for task keyword as fallback
            if not has:
                # check collection title/description for task hint
                txt = ((c.get("title") or "") + " " + (c.get("description") or "")).lower()
                if pipeline_tag.lower() in txt:
                    has = True
            if has:
                filtered.append(c)
        all_cols = filtered
    if gguf_only and all_cols:
        # Verify each collection has at least one GGUF repo. Repo verdicts are
        # persisted 7d, so this is a one-time cost per repo, not per dropdown.
        uniq = []
        for c in all_cols:
            for it in c.get("items") or []:
                if it.get("type") == "model" and it.get("id"):
                    uniq.append(it.get("id"))
        uniq = list(dict.fromkeys(uniq))
        sem = asyncio.Semaphore(6)

        async def _check(mid):
            async with sem:
                return mid, await _repo_gguf_count(mid)

        verdicts = dict(await asyncio.gather(*[_check(m) for m in uniq])) if uniq else {}
        kept = []
        for c in all_cols:
            ids = [it.get("id") for it in (c.get("items") or [])
                   if it.get("type") == "model" and it.get("id")]
            if not ids:
                continue
            vals = [verdicts.get(i) for i in ids]
            # keep if any repo has GGUF; fail-open if a check was inconclusive
            if any((v or 0) > 0 for v in vals) or any(v is None for v in vals):
                kept.append(c)
        all_cols = kept
    # trim to limit and shape
    out = []
    for c in all_cols[:limit]:
        out.append({
            "slug": c.get("slug") or "",
            "title": c.get("title") or c.get("slug") or "",
            "description": c.get("description") or "",
            "lastUpdated": c.get("lastUpdated") or "",
            "itemCount": len(c.get("items") or []),
        })
    result = {"collections": out, "gguf_only": bool(gguf_only)}
    _cache_set(key, result)
    return result

async def get_collection(slug: str, gguf_only: bool = True):
    """Fetch collection detail by slug (e.g. unsloth/unsloth-dynamic-...).

    Collection items carry no sibling info (tags are empty), so a
    safetensors source repo and its GGUF quant look identical. With
    gguf_only (default) each item's model detail is checked concurrently
    (cached 5 min, so repeat views are instant) and only repos that
    actually contain .gguf files are returned, with ggufCount attached.
    Items whose detail check fails are kept with ggufCount -1 (unknown)
    rather than hidden over a transient blip.
    Returns items filtered to GGUF."""
    slug = (slug or "").strip().strip("/")
    if not slug or "/" not in slug:
        return {"slug": slug, "items": [], "title": ""}
    key = f"collection:{slug}:gguf{int(gguf_only)}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = f"https://huggingface.co/api/collections/{quote(slug, safe='/')}"
    async with httpx.AsyncClient(timeout=15, headers=_headers(), follow_redirects=True) as client:
        resp = await _get_public(client, url)
        resp.raise_for_status()
        c = resp.json()
        items = c.get("items") or []
        filtered = []
        for it in items:
            if it.get("type") != "model":
                continue
            mid = it.get("id") or ""
            tags = it.get("tags") or []
            author = it.get("author") or (mid.split("/")[0] if "/" in mid else "")
            filtered.append({
                "id": mid,
                "author": author,
                "downloads": it.get("downloads") or 0,
                "likes": it.get("likes") or 0,
                "pipeline_tag": it.get("pipeline_tag") or "",
                "tags": tags,
                "private": it.get("private") or False,
            })
        if gguf_only and filtered:
            # collection items carry no sibling info, so verify each repo's
            # model detail concurrently (each cached 5 min). Keep only repos
            # with real .gguf files; fail-open (ggufCount -1) on check errors.
            sem = asyncio.Semaphore(8)

            async def _check(it):
                mid = it.get("id") or ""
                if "/" not in mid:
                    return None
                author, name = mid.split("/", 1)
                try:
                    async with sem:
                        detail = await get_gguf_files(author, name)
                    n = detail.get("ggufCount") or 0
                except Exception:
                    n = -1
                if n == 0:
                    return None
                it["ggufCount"] = n
                return it

            checked = await asyncio.gather(*[_check(it) for it in filtered])
            filtered = [it for it in checked if it is not None]
        else:
            for it in filtered:
                it["ggufCount"] = it.get("ggufCount", 0)
        result = {
            "slug": c.get("slug") or slug,
            "title": c.get("title") or slug,
            "description": c.get("description") or "",
            "owner": c.get("owner") or {},
            "items": filtered,
            "gguf_only": gguf_only,
        }
        _cache_set(key, result)
        return result

async def get_avatar(author: str):
    """Fetch author/org avatar URL from HF page. Cached 24h."""
    if not author or "/" in author:
        return None
    key = f"avatar:{author.lower()}"
    ent = _avatar_cache.get(key)
    if ent and ent[0] > time.time():
        return ent[1]
    url = f"https://huggingface.co/{quote(author)}"
    try:
        async with httpx.AsyncClient(timeout=10, headers={"User-Agent": "llama-rpc-farm/1.0"}, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            text = resp.text
            # look for avatarUrl in embedded JSON (page may HTML-escape quotes)
            m = re.search(r'(?:"|&quot;)avatarUrl(?:"|&quot;)\s*:\s*(?:"|&quot;)([^"&<]+)(?:"|&quot;)', text)
            if m:
                avatar = m.group(1).replace("\\u002F", "/").replace("\\/", "/")
                # HF escapes slashes; ensure https
                avatar = avatar.replace("&amp;", "&").split("&quot;")[0].split('"')[0].strip()
                if avatar.startswith("https://"):
                    _avatar_cache[key] = (time.time() + 86400, avatar)
                    return avatar
            # fallback: direct cdn pattern (stop at HTML entities/whitespace too)
            m2 = re.search(r'https://cdn-avatars\.huggingface\.co[^"&\s<\\]+', text)
            if m2:
                avatar = m2.group(0).replace("&amp;", "&").split("&quot;")[0].split('"')[0].strip()
                if avatar.startswith("https://"):
                    _avatar_cache[key] = (time.time() + 86400, avatar)
                    return avatar
    except Exception:
        pass
    return None

async def search_gguf(query: str = "", sort: str = "downloads", limit: int = 20, page: int = 0, quant: str = "", pipeline_tag: str = "", author: str = "", recommended: bool = False, cursor: str = ""):
    """
    Search HF Hub for GGUF models.
    Uses GET /api/models?filter=gguf&search=...&sort=...&limit=... plus HF's
    Link: rel="next" cursor (the `skip` parameter is deprecated upstream).
    Chunked: only enough HF rows to fill the requested window are fetched.
    Returns {models, total, page, limit, has_more, exact_total, next_cursor,
    paged_by} — total is exact only when exact_total is true (recommended/
    snapshot path); otherwise page with next_cursor. Cursor paging is
    batch-aligned and never skips rows: an empty quant-filtered batch advances
    the cursor and keeps fetching instead of dropping models.
    Each model: {id, author, modelId, downloads, likes, lastModified, tags, pipeline_tag, siblings:[{rfilename,size,quant}], ggufCount}
    """
    cursor = (cursor or "").strip() or None
    key = f"search:{query}:{sort}:{limit}:{page}:{quant}:{pipeline_tag}:{author}:{recommended}:{cursor}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # HF search: library=gguf alone is ignored when search is set, so use filter=gguf
    # HF API expects repeated filter= params, not comma
    filters = ["gguf"]
    if pipeline_tag and pipeline_tag != "all":
        filters.append(pipeline_tag)
    if quant:
        # quant not a HF tag, filtered client-side
        pass
    # HF API valid sorts are downloads/likes/lastModified — trending is frontend-only, map to likes
    api_sort = sort if sort in ("downloads", "likes", "lastModified") else "downloads"
    if sort == "trending":
        api_sort = "likes"
    limit = max(1, min(limit, 50))
    page = max(0, page)
    # NOTE: `limit`/`page` are the UI window. HF rows are fetched 100 at a
    # time and filtered locally (gguf/quant are post-filters), so one UI page
    # may span several HF fetches — but we stop as soon as the requested
    # window is fillable instead of crawling the whole result set.
    need = (page + 1) * limit
    params = {
        "sort": api_sort,
        "limit": str(limit),
        "direction": "-1",
        "full": "false",
        "config": "false",
    }
    # httpx will encode list as ?filter=gguf&filter=text-generation
    params["filter"] = filters
    if query and query.strip():
        params["search"] = query.strip()
    if author:
        params["author"] = author
    # NOTE: never send HF's deprecated `skip` here. Deep paging rides the
    # Link: rel="next" cursor below (cursor pagination is the supported path
    # and cannot silently drop rows the way offset paging did).

    url = HF_API + "/models"
    async with httpx.AsyncClient(timeout=20, headers=_headers(), follow_redirects=True) as client:
        # Combined-filter pagination: query/pipeline_tag/author are handled by HF
        # server-side; quant + recommended are post-fetch filters.
        # EXPENSIVE-FETCH FIX: maintain a persistent on-disk snapshot of every
        # curated author's GGUF models. On each search, only fetch what changed
        # since the snapshot (small delta), then serve all filtering/sorting/
        # pagination locally from the snapshot — no more 11-author full crawls.
        CURATED = ["openai","google","Qwen","deepseek-ai","microsoft","meta-llama","mistralai","huihui-ai","orcarouter","unsloth","ornith-ai"]
        CURATED_SET = {c.lower() for c in CURATED}

        # sort key for merged global ordering (respects the selected sort)
        def _sort_key(m):
            if api_sort == "likes":
                return m.get("likes") or 0
            if api_sort == "lastModified":
                return m.get("lastModified") or ""
            return m.get("downloads") or 0

        async def _get_with_retry(p):
            """One HF GET with sort-fallback + transient-error retries.

            Returns (batch, sort_used, next_cursor). Retries 429/5xx/timeouts
            3x with backoff; a 400 on a non-downloads sort falls back to
            downloads once (HF rejects some sort+filter combos). Never mutates
            p. `next_cursor` comes from HF's Link: rel="next" header — HF
            deprecated `skip`, so this is the only supported deep-paging
            primitive and the one that guarantees no rows are skipped.
            """
            last = None
            for attempt in range(3):
                q = dict(p)
                try:
                    resp = await client.get(url, params=q)
                    resp.raise_for_status()
                    return (resp.json(), q.get("sort"),
                            _index_next_cursor(resp.headers.get("link", "")))
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code if e.response is not None else 0
                    if code == 400 and q.get("sort") != "downloads":
                        q = dict(q)
                        q["sort"] = "downloads"
                        resp = await client.get(url, params=q)
                        resp.raise_for_status()
                        return (resp.json(), "downloads",
                                _index_next_cursor(resp.headers.get("link", "")))
                    if code in (429, 500, 502, 503) and attempt < 2:
                        last = e
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    raise
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last = e
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    raise
            raise last

        # post-fetch predicate (gguf siblings + quant). HF handles
        # query/pipeline_tag/author server-side; these run locally per batch.
        def _passes_post_filters(m):
            siblings = m.get("siblings") or []
            ggufs = [s for s in siblings if _is_gguf(s.get("rfilename"))]
            if not ggufs:
                return False
            if quant:
                q = quant.strip().upper()
                ggufs = [s for s in ggufs if q in (s.get("rfilename") or "").upper()]
                if not ggufs:
                    return False
            return True

        async def _fetch_cursor_page(params_base, cursor, limit):
            """Fetch cursor-aligned HF batches until a non-empty GGUF window.

            Cursor pagination cannot address a row inside an HF page, so a
            result page is aligned to whole HF batches. A quant filter can
            make a batch yield zero rows; we advance the cursor and keep
            fetching (bounded) instead of skipping those raw rows — that is
            what guarantees no models are lost between pages.
            Returns (passing, next_cursor, exhausted).
            """
            passing = []
            nxt = cursor
            exhausted = False
            batch_size = max(1, min(limit, 100))
            for _ in range(8):
                p = dict(params_base)
                p["limit"] = str(batch_size)
                if nxt:
                    p["cursor"] = nxt
                batch, _, next_cursor = await _get_with_retry(p)
                if not batch:
                    exhausted = True
                    nxt = next_cursor
                    break
                passing.extend(m for m in batch if _passes_post_filters(m))
                nxt = next_cursor
                if not nxt:
                    exhausted = True
                    break
                if passing:
                    break
                await asyncio.sleep(0.05)
            return passing, nxt, exhausted

        # ---- persistent snapshot store ----
        import json as _json
        SNAPSHOT_DIR = os.path.join(data_dir(), "generated", "hf_snapshot")
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        SNAPSHOT_SCOPE = "gguf-all"

        def _snap_path(a):
            return os.path.join(SNAPSHOT_DIR, f"{a.lower()}.json")

        async def _get_author_snapshot(a, base_params):
            """Return full model list for one author. Loads snapshot from disk,
            then fetches only pages of NEW/CHANGED models since the snapshot
            was taken (delta), merging into the snapshot. Full re-crawl only
            if snapshot is missing or older than SNAPSHOT_TTL."""
            import datetime as _dt
            path = _snap_path(a)
            snap = {}
            snap_time = None
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        blob = _json.load(f)
                    snap = blob.get("models") or {}
                    snap_time = blob.get("time")
                    # Snapshots built before the scope marker were fetched
                    # with whatever task/search filter was active at the
                    # time, so they miss models. Rebuild them once.
                    if blob.get("scope") != SNAPSHOT_SCOPE:
                        snap, snap_time = {}, None
                except Exception:
                    snap, snap_time = {}, None
            snap_age = None
            if snap_time:
                try:
                    snap_age = ( _dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(snap_time) ).total_seconds()
                except Exception:
                    snap_age = None
            # full refresh weekly (SNAPSHOT_TTL); otherwise delta-only.
            # Snapshot refresh still needs the author's full list (it backs
            # local pagination for all later pages), so it walks the cursor
            # chain to the end via _get_with_retry.
            # The snapshot backs every recommended query, so it holds the
            # author's full GGUF list. Task and search filters are applied
            # locally afterwards, never baked into the snapshot.
            base_params.pop("search", None)
            base_params["filter"] = ["gguf"]
            if snap_age is None or snap_age > 7 * 86400:
                fresh = []
                cur = None
                while True:
                    p = {**base_params, "author": a, "limit": "100"}
                    if cur:
                        p["cursor"] = cur
                    batch, _, nxt = await _get_with_retry(p)
                    if not batch:
                        break
                    fresh.extend(batch)
                    if not nxt:
                        break
                    cur = nxt
                    await asyncio.sleep(0.05)
                snap = {m.get("id"): m for m in fresh if m.get("id")}
            else:
                # delta: fetch first page(s) sorted by lastModified and merge
                # anything newer than our snapshot time until we hit known ids
                p = {**base_params, "author": a, "sort": "lastModified"}
                cur = None
                merged_new = 0
                while True:
                    p2 = dict(p)
                    p2["limit"] = "100"
                    if cur:
                        p2["cursor"] = cur
                    batch, _, nxt = await _get_with_retry(p2)
                    if not batch:
                        break
                    # stop when this page is entirely models we already have
                    # unchanged — match on full id (author/model-name) AND
                    # lastModified, so new uploads or edits are always caught
                    def _known_unchanged(m):
                        mid = m.get("id") or ""
                        if mid not in snap:
                            return False
                        old = snap[mid]
                        # id match + name match + same lastModified = unchanged
                        return (old.get("lastModified") == m.get("lastModified")
                                and (old.get("modelId") or old.get("id")) == (m.get("modelId") or m.get("id")))
                    known = sum(1 for m in batch if _known_unchanged(m))
                    for m in batch:
                        mid = m.get("id")
                        if mid:
                            snap[mid] = m
                    merged_new += len(batch)
                    if known == len(batch):
                        break  # whole page already known and unchanged — done
                    if not nxt:
                        break
                    cur = nxt
                    await asyncio.sleep(0.05)
            # persist snapshot
            try:
                with open(path, "w", encoding="utf-8") as f:
                    _json.dump({"time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                "scope": SNAPSHOT_SCOPE, "models": snap}, f)
            except Exception:
                pass
            return list(snap.values())

        def _shape(m):
            siblings = m.get("siblings") or []
            ggufs = [s for s in siblings if _is_gguf(s.get("rfilename"))]
            if quant:
                q = quant.strip().upper()
                ggufs = [s for s in ggufs if q in (s.get("rfilename") or "").upper()]
            enriched = [{
                "rfilename": s.get("rfilename") or "",
                "size": s.get("size"),
                "quant": _parse_quant(s.get("rfilename") or ""),
                "lfs": s.get("lfs"),
            } for s in ggufs]
            enriched.sort(key=lambda x: (x.get("size") or 0), reverse=True)
            gated = bool(m.get("gated") or m.get("private") and m.get("gated") is not False)
            if isinstance(m.get("gated"), str):
                gated = m.get("gated") != "false"
            return {
                "id": m.get("id") or m.get("modelId") or "",
                "modelId": m.get("modelId") or m.get("id") or "",
                "author": m.get("author") or (m.get("id") or "").split("/")[0] if "/" in (m.get("id") or "") else "",
                "downloads": m.get("downloads") or 0,
                "likes": m.get("likes") or 0,
                "lastModified": m.get("lastModified") or m.get("sha") or "",
                "pipeline_tag": m.get("pipeline_tag") or "",
                "tags": m.get("tags") or [],
                "siblings": enriched,
                "ggufCount": len(enriched),
                "gated": gated,
                "private": bool(m.get("private")),
            }

        if recommended and not author:
            # fan out per curated author using cached snapshots (cheap after
            # first build). Snapshots live locally, so exact totals + slicing
            # here cost nothing — no HF crawl per page.
            merged = []
            for a in CURATED:
                merged.extend(await _get_author_snapshot(a, params))
            merged.sort(key=_sort_key, reverse=True)
            # HF applies task/search server-side on the normal path; the
            # snapshot holds everything, so match them here the same way:
            # task = pipeline_tag or a tag, search = substring of the id.
            task = pipeline_tag if pipeline_tag and pipeline_tag != "all" else ""
            terms = (query or "").lower().split()

            def _matches(m):
                if task and m.get("pipeline_tag") != task and task not in (m.get("tags") or []):
                    return False
                mid = (m.get("id") or "").lower()
                return all(t in mid for t in terms)

            passing = [m for m in merged if _matches(m) and _passes_post_filters(m)]
            total = len(passing)
            start = page * limit
            models = [_shape(m) for m in passing[start:start + limit]]
            result = {"models": models, "total": total, "page": page,
                      "limit": limit, "has_more": start + limit < total,
                      "exact_total": True, "next_cursor": None,
                      "paged_by": "page"}
            _cache_set(key, result)
            return result

        # Cursor path: one cursor-aligned batch (or a few when a quant filter
        # is sparse). Never uses the deprecated skip, so deep pages cannot
        # silently drop models the way skip-based offset paging did.
        passing, next_cursor, exhausted = await _fetch_cursor_page(
            dict(params), cursor, limit)
        models = [_shape(m) for m in passing]
        result = {"models": models, "total": None, "page": page,
                  "limit": limit, "has_more": bool(next_cursor),
                  "exact_total": False, "next_cursor": next_cursor,
                  "cursor": cursor, "paged_by": "cursor"}
        _cache_set(key, result)
        return result

async def check_file_access(repo_id: str, filename: str = ""):
    """Test whether the configured token can actually download from a repo.

    Metadata (siblings list) is public even for gated repos, so a 1-byte
    range GET on a real file is the definitive check. Cheap: ~1 byte.
    Returns {repo, file, token_set, access, status, reason} where access is
    "granted" | "denied".
    """
    repo_id = (repo_id or "").strip().strip("/")
    if not filename:
        detail = await get_gguf_files(*repo_id.split("/", 1))
        sibs = detail.get("siblings") or []
        if not sibs:
            return {"repo": repo_id, "file": "", "token_set": bool(_token()),
                    "access": "denied", "status": 0,
                    "reason": "no GGUF files to test"}
        filename = sibs[0].get("rfilename") or ""
    token = _token()
    url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename)}"
    headers = {"User-Agent": "llama-rpc-farm/1.0", "Range": "bytes=0-0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url)
    except Exception as e:
        return {"repo": repo_id, "file": filename, "token_set": bool(token),
                "access": "denied", "status": 0, "reason": f"request failed: {e}"}
    code = resp.status_code
    if code in (200, 206):
        return {"repo": repo_id, "file": filename, "token_set": bool(token),
                "access": "granted", "status": code, "reason": "ok"}
    if code == 401:
        reason = ("invalid or missing token" if token
                  else "no token saved — add one in Settings → Defaults")
    elif code == 403:
        reason = "token lacks access — open the HF page and click “Agree and access repository”"
    elif code == 404:
        reason = "file not found"
    else:
        reason = f"unexpected status {code}"
    return {"repo": repo_id, "file": filename, "token_set": bool(token),
            "access": "denied", "status": code, "reason": reason}

async def get_gguf_files(author: str, name: str):
    """Fetch single model detail and return GGUF sibling list."""
    model_id = f"{author}/{name}"
    key = f"detail:{model_id}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = f"{HF_API}/models/{quote(author)}/{quote(name)}"
    async with httpx.AsyncClient(timeout=20, headers=_headers(), follow_redirects=True) as client:
        # Siblings carry per-file sizes only with blobs=true; without it
        # every size is null and the UI shows "0 MB".
        resp = await client.get(url, params={"blobs": "true"})
        resp.raise_for_status()
        m = resp.json()
        siblings = m.get("siblings") or []
        ggufs = []
        for s in siblings:
            fn = s.get("rfilename") or ""
            if not _is_gguf(fn):
                continue
            ggufs.append({
                "rfilename": fn,
                "size": s.get("size"),
                "quant": _parse_quant(fn),
                "lfs": s.get("lfs"),
            })
        ggufs.sort(key=lambda x: (x.get("size") or 0), reverse=True)
        gated = bool(m.get("gated") or m.get("private") and m.get("gated") is not False)
        if isinstance(m.get("gated"), str):
            gated = m.get("gated") != "false"
        result = {
            "id": m.get("id") or model_id,
            "modelId": m.get("modelId") or m.get("id") or model_id,
            "author": m.get("author") or author,
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "lastModified": m.get("lastModified") or "",
            "pipeline_tag": m.get("pipeline_tag") or "",
            "tags": m.get("tags") or [],
            "cardData": m.get("cardData") or {},
            "siblings": ggufs,
            "ggufCount": len(ggufs),
            "gated": gated,
            "private": bool(m.get("private")),
        }
        _cache_set(key, result)
        return result
