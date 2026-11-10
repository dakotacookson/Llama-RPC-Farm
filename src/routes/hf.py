"""Hugging Face downloads, search, and image proxying."""
import asyncio
import os
import time

import pydantic
from fastapi import APIRouter, HTTPException

import config_loader as cfg
import engine_registry
import gguf_inspect
import path_resolver
from runtime_paths import data_dir
from hf_icons import _load_model_icons, _save_model_icons
import hf_browse
import hf_downloader

BASE_DIR = data_dir()

router = APIRouter()


# ---------------- hf downloads ----------------

class DownloadIn(pydantic.BaseModel):
    link: str
    store: str = "llm-models"
    all_shards: bool = True  # split model (-00001-of-00002): fetch every shard
    repo: bool = False  # whole repository: link is hf://owner/name


@router.get("/api/downloads")
async def list_downloads():
    return hf_downloader.all_jobs()


@router.get("/api/downloads/stores")
async def download_stores():
    return hf_downloader.list_stores()


@router.post("/api/downloads")
async def start_download(d: DownloadIn):
    # Engine-gate (DEC-07, T3 F4): refuse out-of-vocabulary formats
    # BEFORE dispatch. The UI's manifest downloads.formats already keeps
    # the right pane visible, so this 409 is a safety net for direct API
    # calls.
    # Format inference: repo=True (hf://owner/name) or a non-.gguf
    # filename is the HF-hub "hf" vocabulary; *.gguf is "gguf".
    engine = engine_registry.get_engine()
    allowed = engine.ui_manifest().get("downloads", {}).get("formats", [])
    fmt = "gguf"
    if d.repo or not str(d.link or "").lower().endswith(".gguf"):
        fmt = "hf"
    if fmt not in allowed:
        raise HTTPException(
            status_code=409,
            detail=f"the {engine.name} engine does not download "
                   f"'{fmt}' models (allowed: {', '.join(allowed) or 'none'})")
    try:
        return engine.acquire(d.link, d.store, repo=d.repo,
                              all_shards=d.all_shards)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/downloads/{job_id}/cancel")
async def cancel_download(job_id: str):
    return hf_downloader.cancel(job_id)


# ---------------- hf browse (GGUF search, proxy to Hugging Face) ----------------
import hf_browse  # local helper

# Network-bound HF reads are cached briefly: tab switches and repeated
# searches should not re-hit the HF API. Live access checks stay uncached.
_HF_CACHE = {}
_HF_CACHE_TTL = 120.0


async def _hf_cached(key, fetch):
    now = time.time()
    hit = _HF_CACHE.get(key)
    if hit and now - hit[0] < _HF_CACHE_TTL:
        return hit[1]
    result = await fetch()
    if len(_HF_CACHE) > 500:
        _HF_CACHE.clear()
    _HF_CACHE[key] = (now, result)
    return result

@router.get("/api/hf/search")
async def hf_search(
    query: str = "",
    sort: str = "downloads",
    limit: int = 20,
    page: int = 0,
    quant: str = "",
    pipeline_tag: str = "",
    author: str = "",
    recommended: bool = False,
    cursor: str = "",
):
    key = ("search", query, sort, limit, page, quant, pipeline_tag,
           author, recommended, cursor)
    try:
        return await _hf_cached(key, lambda: hf_browse.search_gguf(
            query=query, sort=sort, limit=limit, page=page,
            quant=quant, pipeline_tag=pipeline_tag, author=author,
            recommended=recommended, cursor=cursor))
    except Exception as e:
        # surface HF errors as 502 so UI can show hint
        raise HTTPException(status_code=502, detail=str(e))

@router.get("/api/hf/models/{author}/{name}/gguf")
async def hf_gguf_files(author: str, name: str):
    try:
        return await _hf_cached(("gguf", author, name),
                                lambda: hf_browse.get_gguf_files(author, name))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.get("/api/hf/access/{author}/{name}")
async def hf_access(author: str, name: str, file: str = ""):
    """Live check: can the saved token download from this (possibly gated) repo?"""
    try:
        return await hf_browse.check_file_access(f"{author}/{name}", file)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.get("/img-proxy")
async def img_proxy(url: str):
    """Same-origin image fetch. The browser blocks (OpaqueResponseBlocking)
    or taints some cross-origin images (HF CDN avatars serve restrictive
    headers); proxying through this keeps <img> loads quiet and cacheable.
    Only http(s) URLs, image responses only."""
    from urllib.parse import urlparse as _urlparse
    import httpx as _httpx
    from fastapi.responses import Response as _ImgResponse
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="url must be http(s)")
    try:
        async with _httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)[:200])
    ctype = (resp.headers.get("content-type") or "").lower().split(";")[0].strip()
    body = resp.content
    if not ctype.startswith("image/"):
        # Some CDNs answer "*/*" or "application/octet-stream" for avatars.
        # Sniff magic bytes before rejecting so valid images still proxy.
        _magic_ok = (
            body.startswith(b"\x89PNG") or body.startswith(b"\xff\xd8\xff")
            or body.startswith(b"GIF87a") or body.startswith(b"GIF89a")
            or body.startswith(b"RIFF") or body[:4] == b"\x00\x00\x01\x00"
            or body.strip().startswith(b"<svg") or body.strip().startswith(b"<?xml")
        )
        if not _magic_ok:
            raise HTTPException(status_code=415, detail="not an image")
        ctype = "image/png" if body.startswith(b"\x89PNG") else \
                "image/jpeg" if body.startswith(b"\xff\xd8\xff") else \
                "image/gif" if body.startswith(b"GIF8") else \
                "image/svg+xml" if b"<svg" in body[:512] else \
                "application/octet-stream"
    return _ImgResponse(content=body, media_type=ctype,
                        headers={"Cache-Control": "public, max-age=86400"})


# ---- model icon resolution (cached permanently in config/model_icons.json) ----
@router.get("/api/models/{model_id}/icon")
async def model_icon(model_id: str, path: str = ""):
    """Resolve an icon for a model and cache it permanently. Tries to find the
    real HF repo from the model's GGUF metadata (basename/name), then the HF
    author avatar. Falls back to the cube placeholder (empty icon) if nothing
    resolves. Cached in config/model_icons.json.

    `path`: optional store:// or raw GGUF path — lets UNCONFIGURED scan rows
    (Models tab -> Source list) resolve the SAME icons as configured models.
    Cached under a `file:<basename>` key so every quant/shard of the same
    file shares one entry and one HF lookup."""
    # direct file-path lookup (scan rows): key by GGUF basename so the
    # whole model family (quants, shards) shares one icon entry
    if path:
        fname = os.path.basename(path)
        key = f"file:{fname}"
        icons = _load_model_icons()
        if key in icons:
            return {"model_id": key, "icon": icons[key]}
        url = await _resolve_icon_url(icons, key, base_path=path)
        return {"model_id": key, "icon": url}

    icons = _load_model_icons()
    if model_id in icons:
        return {"model_id": model_id, "icon": icons[model_id]}

    url = await _resolve_icon_url(icons, model_id)
    return {"model_id": model_id, "icon": url}


class IconResolveItem(pydantic.BaseModel):
    key: str
    path: str = ""


class IconResolveIn(pydantic.BaseModel):
    items: list[IconResolveItem] = []


@router.post("/api/models/icons/resolve")
async def resolve_model_icons(body: IconResolveIn):
    """Batch icon resolution: one browser request for many models/scan rows."""
    icons = _load_model_icons()
    out, pending = {}, []
    for item in body.items:
        key = (item.key or "").strip()
        if not key:
            continue
        if key in icons:
            out[key] = icons[key]
        else:
            pending.append((key, item.path or ""))
    if pending:
        sem = asyncio.Semaphore(4)

        async def one(key, path):
            async with sem:
                try:
                    url = await _resolve_icon_url(
                        icons, key, base_path=path, persist=False)
                except Exception:
                    url = ""
                return key, url

        for key, url in await asyncio.gather(*(one(k, p) for k, p in pending)):
            out[key] = url
        try:
            _save_model_icons(icons)
        except Exception:
            pass
    return {"icons": out}


async def _resolve_icon_url(icons, cache_key, base_path="", persist=True):
    """Shared icon-resolution pipeline: GGUF metadata basename -> HF search
    -> author avatar. Persists to config/model_icons.json under cache_key."""
    url = None
    # 1) explicit path (scan rows) or the model config's GGUF path
    if not base_path:
        model = cfg.get_model(cache_key)
        base_path = (model or {}).get("base_path", "")
    # 2) derive a search hint from the GGUF FILE NAME (not the display name).
    #    Prefer the GGUF metadata basename (e.g. "Huihui-Qwen3.6") which is the
    #    most searchable; fall back to the file name with quant/shard stripped.
    hint = None
    if base_path:
        try:
            resolved = path_resolver.resolve(base_path)
            meta = gguf_inspect.metadata(resolved)
            hint = meta.get("general.basename") or meta.get("general.name")
        except Exception:
            hint = None
    if not hint and base_path:
        fname = os.path.basename(base_path)
        if fname.lower().endswith(".gguf"):
            fname = fname[:-5]
        import re as _re
        fname = _re.sub(r"-(Q\d+_K?_\w+|Q\d+_\d+|F16|BF16|IQ\d+_\w+|GGUF)$", "", fname)
        hint = fname
    # 3) try to find the real HF repo by searching the hint, then get its avatar
    if hint:
        try:
            res = await hf_browse.search_gguf(query=str(hint), sort="downloads", limit=5)
            for m in (res.get("models") or []):
                author = (m.get("author") or "").strip()
                if author:
                    url = await hf_browse.get_avatar(author)
                    if url:
                        break
        except Exception:
            url = None
    # 4) fall back to the cube placeholder (empty icon -> frontend shows cube)
    if not url:
        url = ""

    icons[cache_key] = url
    if persist:
        _save_model_icons(icons)
    return url

@router.get("/api/hf/companies")
async def hf_companies(search: str = "", limit: int = 20, pipeline_tag: str = "", quant: str = "", recommended: bool = False):
    try:
        return await _hf_cached(
            ("companies", limit, search, pipeline_tag, quant, recommended),
            lambda: hf_browse.list_gguf_companies(
                limit=limit, search=search, pipeline_tag=pipeline_tag,
                quant=quant, recommended=recommended))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.get("/api/hf/pipeline-tags")
async def hf_pipeline_tags(limit: int = 200):
    try:
        return await _hf_cached(("pipeline-tags", limit),
                                lambda: hf_browse.list_gguf_pipeline_tags(limit=limit))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.get("/api/hf/collections")
async def hf_collections(owner: str, limit: int = 20, pipeline_tag: str = "",
                         quant: str = "", gguf_only: bool = True):
    try:
        return await _hf_cached(
            ("collections", owner, limit, pipeline_tag, quant, gguf_only),
            lambda: hf_browse.list_collections(
                owner=owner, limit=limit, pipeline_tag=pipeline_tag,
                quant=quant, gguf_only=gguf_only))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.get("/api/hf/collection")
async def hf_collection(slug: str, gguf_only: bool = True):
    try:
        return await _hf_cached(("collection", slug, gguf_only),
                                lambda: hf_browse.get_collection(slug, gguf_only=gguf_only))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
