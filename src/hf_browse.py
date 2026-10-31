"""HF GGUF browse proxy — server-side search for GGUF models.

Uses Hugging Face Hub HTTP API directly (no huggingface_hub dep required for search)
so we can filter library=gguf and return only GGUF siblings.

Results are cached 5 min to stay friendly to HF rate limits.
Supports HF_TOKEN env for private/gated models.

Called by app.py endpoints:
  GET /api/hf/search          (fmt=gguf | safetensors)
  GET /api/hf/models/{author}/{name}/gguf
  GET /api/hf/models/{author}/{name}/safetensors
"""
import asyncio
import os
import time
import re
from urllib.parse import quote

import httpx

HF_API = "https://huggingface.co/api"
CACHE_TTL = 300
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
        p = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            v = (data.get("hf_token") or "").strip()
            if v:
                return v
    except Exception:
        pass
    return None

def _headers():
    h = {"User-Agent": "llama-rpc-farm/1.0"}
    t = _token()
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h

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


def _is_safetensor_file(filename):
    return (filename or "").lower().endswith(".safetensors")


def _has_safetensor_siblings(m):
    """True when the repo row carries at least one real .safetensors file.

    HF's filter=safetensors can still return GGUF repos (they often declare
    the safetensors library tag), so search results are post-filtered on
    actual sibling filenames — same defense search_gguf applies for GGUFs.
    """
    return any(_is_safetensor_file(s.get("rfilename"))
               for s in (m.get("siblings") or []))

_avatar_cache = {}  # author -> (expires, url)

async def list_gguf_companies(limit: int = 20, search: str = "", pipeline_tag: str = "", quant: str = "", recommended: bool = False):
    """Curated big companies when recommended, else top by downloads — reflective of task/quant."""
    CURATED = [
        "openai", "google", "Qwen", "deepseek-ai", "microsoft",
        "meta-llama", "mistralai", "huihui-ai", "orcarouter",
        "unsloth", "ornith-ai",
    ]
    norm = {c.lower(): c for c in CURATED}
    limit = max(1, min(limit, 50))
    key = f"companies:curated:{search}:{pipeline_tag}:{quant}:{limit}:{recommended}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
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
        merged = {}
        if recommended:
            for c in matched:
                merged[c.lower()] = {"author": c, "count": 0, "downloads": 0, "likes": 0, "curated": True}
            for a, v in list(seen.items()):
                lk = a.lower()
                if lk in merged:
                    merged[lk]["count"] = v["count"]
                    merged[lk]["downloads"] = v["downloads"]
        else:
            for c in matched:
                merged[c.lower()] = {"author": c, "count": 0, "downloads": 0, "likes": 0, "curated": True}
            for a, v in seen.items():
                lk = a.lower()
                if lk not in merged:
                    merged[lk] = v
                else:
                    merged[lk]["count"] = v["count"]
                    merged[lk]["downloads"] = v["downloads"]
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
                else:
                    seen[author] = {"author": author, "count": 0, "downloads": 0, "likes": 0, "curated": True}
            except Exception:
                seen[author] = {"author": author, "count": 0, "downloads": 0, "likes": 0, "curated": True}
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
    else:
        seen = {}
        per_page = 50
        pages_needed = max(4, (limit // 10) + 2)
        pages_needed = min(pages_needed, 20)
        for page in range(pages_needed):
            try:
                data = await search_gguf(query="", sort="downloads", limit=per_page, page=page, quant=quant or "", pipeline_tag=pipeline_tag or "", author="")
            except Exception:
                break
            models = data.get("models") or []
            if not models:
                break
            for m in models:
                a = m.get("author") or ""
                if not a:
                    continue
                if a not in seen:
                    seen[a] = {"author": a, "count": 0, "downloads": 0, "likes": 0}
                seen[a]["count"] += 1
                seen[a]["downloads"] += m.get("downloads") or 0
                seen[a]["likes"] += m.get("likes") or 0
            if len(seen) >= limit and page >= 2:
                break
            if page < pages_needed - 1:
                await asyncio.sleep(0.15)
        top = sorted(seen.values(), key=lambda x: x["downloads"], reverse=True)[:limit]
        async def enrich3(t):
            av = await get_avatar(t["author"])
            if av:
                t["avatarUrl"] = av
            return t
        if top:
            await asyncio.gather(*[enrich3(t) for t in top])
        result = {"companies": top, "totalUnique": len(seen), "curated": False}
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

async def list_collections(owner: str, limit: int = 20, pipeline_tag: str = "", quant: str = ""):
    """List HF collections for an owner — reflective of task/quant, paginated for full list."""
    owner = (owner or "").strip()
    if not owner:
        return {"collections": []}
    limit = max(1, min(limit, 100))
    # include task/quant in cache key
    key = f"collections:{owner}:{limit}:{pipeline_tag}:{quant}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = f"https://huggingface.co/api/collections"
    # paginate via limit+skip to get full list (not just greatest hits)
    all_cols = []
    per_page = min(limit, 50)
    pages = (limit + per_page - 1) // per_page
    async with httpx.AsyncClient(timeout=15, headers=_headers(), follow_redirects=True) as client:
        for page in range(pages):
            skip = page * per_page
            params = {"owner": owner, "limit": str(per_page)}
            if skip:
                params["skip"] = str(skip)
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break
            all_cols.extend(raw)
            if len(raw) < per_page:
                break
            await asyncio.sleep(0.05)
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
    result = {"collections": out}
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
        resp = await client.get(url)
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

async def search_gguf(query: str = "", sort: str = "downloads", limit: int = 20, page: int = 0, quant: str = "", pipeline_tag: str = "", author: str = "", recommended: bool = False):
    """
    Search HF Hub for GGUF models.
    Uses GET /api/models?library=gguf&search=...&sort=...&limit=...&skip=...
    Then filters siblings to .gguf and optionally quant.
    Chunked: only enough HF rows to fill the requested page window are
    fetched (next chunk on next page). Returns {models, total, page, limit,
    has_more, exact_total} — total is exact only when exact_total is true
    (recommended/snapshot path, or HF exhausted); otherwise use has_more.
    Each model: {id, author, modelId, downloads, likes, lastModified, tags, pipeline_tag, siblings:[{rfilename,size,quant}], ggufCount}
    """
    key = f"search:{query}:{sort}:{limit}:{page}:{quant}:{pipeline_tag}:{author}:{recommended}"
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
    # NOTE: no HF `skip` here — _fetch_window pages HF from offset 0 and
    # slices the UI window locally. A leftover skip used to double-offset
    # (HF skipped raw rows AND we sliced filtered rows), eating models on
    # every page>=1 and corrupting internal page=1 callers (company counts).

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

            Returns (batch, sort_used). Retries 429/5xx/timeouts 3x with
            backoff; a 400 on a non-downloads sort falls back to downloads
            once (HF rejects some sort+filter combos). Never mutates p.
            """
            last = None
            for attempt in range(3):
                q = dict(p)
                try:
                    resp = await client.get(url, params=q)
                    resp.raise_for_status()
                    return resp.json(), q.get("sort")
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code if e.response is not None else 0
                    if code == 400 and q.get("sort") != "downloads":
                        q = dict(q)
                        q["sort"] = "downloads"
                        resp = await client.get(url, params=q)
                        resp.raise_for_status()
                        return resp.json(), "downloads"
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

        async def _fetch_window(params_base, need):
            """Fetch HF pages until `need` post-filtered rows are collected
            (or HF is exhausted). Returns (passing_raw, exhausted).

            Chunked: page 0 typically costs 1-2 HF fetches instead of the
            full crawl the old _fetch_all did. Exact totals are therefore
            unknown unless exhausted — callers must use has_more.
            """
            passing = []
            skip = 0
            exhausted = False
            while len(passing) < need:
                p = dict(params_base)
                p["limit"] = "100"
                if skip:
                    p["skip"] = str(skip)
                batch, _ = await _get_with_retry(p)
                if not batch:
                    exhausted = True
                    break
                for m in batch:
                    if _passes_post_filters(m):
                        passing.append(m)
                        if len(passing) >= need + 1:
                            break
                if len(batch) < 100:
                    exhausted = True
                    break
                if len(passing) >= need + 1:
                    break
                skip += 100
                await asyncio.sleep(0.05)
            return passing, exhausted

        # ---- persistent snapshot store ----
        import json as _json
        SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated", "hf_snapshot")
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

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
            # local pagination for all later pages), so it pages through
            # everything — but via _get_with_retry (no caller-dict mutation).
            # Defensive: drop any UI-page `skip` so snapshots always cover
            # the author's full list from HF offset 0.
            base_params = {k: v for k, v in base_params.items() if k != "skip"}
            if snap_age is None or snap_age > 7 * 86400:
                fresh = []
                skip = 0
                while True:
                    p = {**base_params, "author": a, "limit": "100"}
                    if skip:
                        p["skip"] = str(skip)
                    batch, _ = await _get_with_retry(p)
                    if not batch:
                        break
                    fresh.extend(batch)
                    if len(batch) < 100:
                        break
                    skip += 100
                    await asyncio.sleep(0.05)
                snap = {m.get("id"): m for m in fresh if m.get("id")}
            else:
                # delta: fetch first page(s) sorted by lastModified and merge
                # anything newer than our snapshot time until we hit known ids
                p = {**base_params, "author": a, "sort": "lastModified"}
                skip = 0
                merged_new = 0
                while True:
                    p2 = dict(p)
                    p2["limit"] = "100"
                    if skip:
                        p2["skip"] = str(skip)
                    batch, _ = await _get_with_retry(p2)
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
                    if len(batch) < 100:
                        break
                    skip += 100
                    await asyncio.sleep(0.05)
            # persist snapshot
            try:
                with open(path, "w", encoding="utf-8") as f:
                    _json.dump({"time": _dt.datetime.now(_dt.timezone.utc).isoformat(), "models": snap}, f)
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
            passing = [m for m in merged if _passes_post_filters(m)]
            total = len(passing)
            start = page * limit
            models = [_shape(m) for m in passing[start:start + limit]]
            result = {"models": models, "total": total, "page": page,
                      "limit": limit, "has_more": start + limit < total,
                      "exact_total": True}
            _cache_set(key, result)
            return result

        # Chunked path: fetch only enough HF rows to fill the requested
        # window (usually 1-2 HF fetches for page 0). Exact total unknown
        # unless HF is exhausted — frontend must use has_more for paging.
        passing, exhausted = await _fetch_window(dict(params), need)
        start = page * limit
        window = passing[start:start + limit]
        models = [_shape(m) for m in window]
        has_more = (len(passing) > start + limit) or not exhausted
        total = len(passing) if exhausted else None
        result = {"models": models, "total": total, "page": page,
                  "limit": limit, "has_more": has_more,
                  "exact_total": exhausted}
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
        resp = await client.get(url)
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


# ---------------- safetensors (vLLM) browse ----------------

async def search_safetensors(query: str = "", sort: str = "downloads",
                             limit: int = 20, page: int = 0,
                             pipeline_tag: str = "", author: str = "",
                             recommended: bool = False):
    """Search HF Hub for source/safetensors repos (what `vllm serve` eats).

    Same chunked contract as search_gguf: {models, total, page, limit,
    has_more, exact_total}. Repos are post-filtered on real .safetensors
    siblings — HF's filter=safetensors also matches GGUF repos that merely
    declare the library tag, so filenames are the source of truth.
    Each model: {id, author, modelId, downloads, likes, lastModified, tags,
    pipeline_tag, safetensorsCount, gated, private} — siblings are NOT
    carried (search never includes per-file sizes; the detail endpoint
    with blobs=true is the size source).
    """
    key = f"search-st:{query}:{sort}:{limit}:{page}:{pipeline_tag}:{author}:{recommended}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # filter=safetensors matched zero GGUF leaks in the top-10 downloads
    # probe (library=safetensors DID leak one), so prefer it; the sibling
    # post-filter below is the real guarantee either way.
    filters = ["safetensors"]
    if pipeline_tag and pipeline_tag != "all":
        filters.append(pipeline_tag)
    api_sort = sort if sort in ("downloads", "likes", "lastModified") else "downloads"
    if sort == "trending":
        api_sort = "likes"
    limit = max(1, min(limit, 50))
    page = max(0, page)
    need = (page + 1) * limit
    params = {
        "sort": api_sort,
        "limit": "100",
        "direction": "-1",
        "full": "false",
        "config": "false",
        "filter": filters,
    }
    if query and query.strip():
        params["search"] = query.strip()
    if author:
        params["author"] = author

    url = HF_API + "/models"

    def _passes(m):
        return _has_safetensor_siblings(m)

    def _shape(m):
        gated = bool(m.get("gated") or m.get("private") and m.get("gated") is not False)
        if isinstance(m.get("gated"), str):
            gated = m.get("gated") != "false"
        return {
            "id": m.get("id") or m.get("modelId") or "",
            "modelId": m.get("modelId") or m.get("id") or "",
            "author": m.get("author") or ((m.get("id") or "").split("/")[0]
                                          if "/" in (m.get("id") or "") else ""),
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "lastModified": m.get("lastModified") or m.get("sha") or "",
            "pipeline_tag": m.get("pipeline_tag") or "",
            "tags": m.get("tags") or [],
            "safetensorsCount": 1,
            "gated": gated,
            "private": bool(m.get("private")),
        }

    passing = []
    skip = 0
    exhausted = False
    async with httpx.AsyncClient(timeout=20, headers=_headers(),
                                 follow_redirects=True) as client:
        while len(passing) < need:
            p = dict(params)
            if skip:
                p["skip"] = str(skip)
            try:
                resp = await client.get(url, params=p)
                resp.raise_for_status()
                batch = resp.json()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 400 and p.get("sort") != "downloads":
                    # some sort+filter combos are rejected — fall back once
                    p2 = dict(p)
                    p2["sort"] = "downloads"
                    resp = await client.get(url, params=p2)
                    resp.raise_for_status()
                    batch = resp.json()
                elif code in (429, 500, 502, 503):
                    raise
                else:
                    raise
            if not batch:
                exhausted = True
                break
            for m in batch:
                if _passes(m):
                    passing.append(m)
                    if len(passing) >= need + 1:
                        break
            if len(batch) < 100:
                exhausted = True
                break
            if len(passing) >= need + 1:
                break
            skip += 100
            await asyncio.sleep(0.05)

    if recommended and not author:
        # Recommended for safetensors reuses the curated list but relies on
        # HF-side author filtering per author, fetched cheaply page-at-a-time
        # (no snapshot machinery — source repos are far fewer per author).
        merged = []
        async with httpx.AsyncClient(timeout=20, headers=_headers(),
                                     follow_redirects=True) as client:
            for a in ["openai", "google", "Qwen", "deepseek-ai", "microsoft",
                      "meta-llama", "mistralai", "huihui-ai", "orcarouter",
                      "unsloth", "ornith-ai"]:
                pa = dict(params)
                pa["author"] = a
                pa["limit"] = "100"
                try:
                    resp = await client.get(url, params=pa)
                    resp.raise_for_status()
                    for m in resp.json() or []:
                        if _passes(m):
                            merged.append(m)
                except Exception:
                    continue
        merged.sort(key=lambda m: m.get(api_sort if api_sort != "likes" else "likes") or 0,
                    reverse=True)
        total = len(merged)
        start = page * limit
        models = [_shape(m) for m in merged[start:start + limit]]
        result = {"models": models, "total": total, "page": page,
                  "limit": limit, "has_more": start + limit < total,
                  "exact_total": True}
        _cache_set(key, result)
        return result

    start = page * limit
    window = passing[start:start + limit]
    models = [_shape(m) for m in window]
    has_more = (len(passing) > start + limit) or not exhausted
    total = len(passing) if exhausted else None
    result = {"models": models, "total": total, "page": page,
              "limit": limit, "has_more": has_more,
              "exact_total": exhausted}
    _cache_set(key, result)
    return result


async def get_safetensors_files(author: str, name: str):
    """Fetch single repo detail and return its .safetensors sibling list.

    Sizes come from ?blobs=true (search never carries per-file sizes).
    Returns the same envelope as get_gguf_files with safetensorsCount.
    """
    model_id = f"{author}/{name}"
    key = f"detail-st:{model_id}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = f"{HF_API}/models/{quote(author)}/{quote(name)}"
    async with httpx.AsyncClient(timeout=20, headers=_headers(),
                                 follow_redirects=True) as client:
        resp = await client.get(url, params={"blobs": "true"})
        resp.raise_for_status()
        m = resp.json()
    st_files = []
    for s in (m.get("siblings") or []):
        fn = s.get("rfilename") or ""
        if not _is_safetensor_file(fn):
            continue
        st_files.append({
            "rfilename": fn,
            "size": s.get("size"),
            "lfs": s.get("lfs"),
        })
    st_files.sort(key=lambda x: (x.get("size") or 0), reverse=True)
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
        "siblings": st_files,
        "safetensorsCount": len(st_files),
        "gated": gated,
        "private": bool(m.get("private")),
        "config": m.get("config"),
    }
    _cache_set(key, result)
    return result
