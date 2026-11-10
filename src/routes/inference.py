"""Embeddings, ASR/audio, decision, and vision smoke tests."""
import os
import re
import threading
import time

import pydantic
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

import access
import config_loader as cfg
import engine_registry
import local_node
import model_parser
import path_resolver
import swap_manager
from web_shared import (_access_filter_models, _gateway_headers,
                        _gateway_url)

router = APIRouter()

# /api/scan walks the model store (recursive). The UI auto-refreshes, so
# cache the directory walk briefly: a slow/large store must not be re-walked
# on every poll. Dirty refreshes pass refresh=1 to bypass this.
_SCAN_CACHE = {}
_SCAN_CACHE_TTL = 60.0
_SCAN_CACHE_LOCK = threading.Lock()


def _scan_cache_get(key, refresh):
    if refresh:
        return None
    now = time.time()
    with _SCAN_CACHE_LOCK:
        hit = _SCAN_CACHE.get(key)
    if hit and now - hit[0] < _SCAN_CACHE_TTL:
        return hit[1], hit[2]
    return None


def _scan_cache_put(key, models, projectors):
    with _SCAN_CACHE_LOCK:
        _SCAN_CACHE[key] = (time.time(), models, projectors)


# ---------------- embeddings test ----------------
@router.get("/api/embeddings/models")
async def list_embedding_models():
    # capability detection is the engine adapter's vocabulary (G-6):
    # llama.cpp keeps the "embed"-substring heuristic.
    engine = engine_registry.get_engine()
    return _access_filter_models(
        engine.filter_embedding_models(cfg.list_models()))


class EmbedIn(pydantic.BaseModel):
    model: str
    input: str | list[str]
    # optional second input for similarity UI, handled client-side


@router.post("/api/embeddings")
async def create_embeddings(e: EmbedIn, request: Request):
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running — Launch All first")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = _gateway_url("/v1/embeddings")  # GATEWAY: key-auth + usage tracking for ALL farm-proxied generate paths
    def do():
        import urllib.request, json as _json, urllib.error
        payload = _json.dumps({"model": e.model, "input": e.input}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers=_gateway_headers(request),
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                body = r.read().decode()
                return _json.loads(body)
        except urllib.error.HTTPError as he:
            b = he.read().decode()[:2000]
            try:
                j = _json.loads(b)
                raise HTTPException(status_code=he.code, detail=j.get("error", {}).get("message", b))
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=he.code, detail=b)
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))
    return await run_in_threadpool(do)




# ---------------- audio (ASR) test ----------------
@router.get("/api/audio/models")
async def list_audio_models():
    # capability detection is the engine adapter's vocabulary (G-8)
    engine = engine_registry.get_engine()
    return _access_filter_models(
        engine.filter_audio_models(cfg.list_models()))


@router.get("/api/decision/models")
async def list_decision_models():
    """TypeSafe decision models (llama.cpp /v1/systemone). Detected from
    the GGUF's <arch>.decision.type — the same gate llama.cpp uses —
    via the engine adapter, so a future engine just filters its own
    vocabulary."""
    engine = engine_registry.get_engine()
    return _access_filter_models(
        engine.filter_decision_models(cfg.list_models()))


class AudioIn(pydantic.BaseModel):
    model: str
    audio: str  # data URI (data:audio/wav;base64,...) from browser MediaRecorder
    language: str = "English"  # force ASR language ("None" = auto-detect)


@router.post("/api/audio")
async def audio_test(v: AudioIn, request: Request):
    """Transcribe browser-captured audio via OpenAI-style input_audio
    (base64 data URI) in a chat completion. Audio never touches disk."""
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running — Launch All first")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = _gateway_url("/v1/chat/completions")  # GATEWAY: key-auth + usage tracking for ALL farm-proxied generate paths
    def do():
        import urllib.request, json as _json, urllib.error
        msgs = []
        # Language forcing is engine vocabulary (G-8): the llama.cpp adapter
        # emits the Qwen3-ASR "language X<asr_text>" system message (matches
        # qwen-asr's force_language; as a system message it lands right
        # before the generation prompt — verified equivalent in
        # llama-server). Other engines return None — sending the prefix
        # there would be a no-op or leak into the transcript.
        sys_msg = engine_registry.get_engine().asr_language_system_message(v.language)
        if sys_msg:
            msgs.append({"role": "system", "content": sys_msg})
        msgs.append({"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": v.audio, "format": "wav"}},
        ]})
        payload = _json.dumps({"model": v.model, "messages": msgs}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers=_gateway_headers(request),
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                resp = _json.loads(r.read().decode())
        except urllib.error.HTTPError as he:
            b = he.read().decode()[:2000]
            try:
                j = _json.loads(b)
                raise HTTPException(status_code=he.code, detail=j.get("error", {}).get("message", b))
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=he.code, detail=b)
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))
        # Qwen3-ASR emits "language <X><asr_text>" before the transcript —
        # strip it, keep the detected language as a side field.
        try:
            msg = resp["choices"][0]["message"]
            raw = msg.get("content") or ""
            m = re.match(r"^\s*language\s+([A-Za-z]+)\s*<asr_text>\s*", raw)
            if m:
                msg["content"] = raw[m.end():]
                msg["language"] = m.group(1)
        except Exception:
            pass
        return resp
    return await run_in_threadpool(do)


# ---------------- vision test ----------------
@router.get("/api/vision/models")
async def list_vision_models():
    # capability detection is the engine adapter's vocabulary (G-9):
    # llama.cpp filters by the mmproj projector flag.
    engine = engine_registry.get_engine()
    return _access_filter_models(
        engine.filter_vision_models(cfg.list_models()))


class VisionIn(pydantic.BaseModel):
    model: str
    prompt: str
    image: str  # data URI (data:image/png;base64,...) built client-side


@router.post("/api/vision")
async def vision_test(v: VisionIn, request: Request):
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running — Launch All first")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = _gateway_url("/v1/chat/completions")  # GATEWAY: key-auth + usage tracking for ALL farm-proxied generate paths
    def do():
        import urllib.request, json as _json, urllib.error
        payload = _json.dumps({
            "model": v.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": v.prompt},
                {"type": "image_url", "image_url": {"url": v.image}},
            ]}],
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers=_gateway_headers(request),
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                resp = _json.loads(r.read().decode())
        except urllib.error.HTTPError as he:
            b = he.read().decode()[:2000]
            try:
                j = _json.loads(b)
                raise HTTPException(status_code=he.code, detail=j.get("error", {}).get("message", b))
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=he.code, detail=b)
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))
        # Qwen3-ASR emits "language <X><asr_text>" before the transcript —
        # strip it, keep the detected language as a side field.
        try:
            msg = resp["choices"][0]["message"]
            raw = msg.get("content") or ""
            m = re.match(r"^\s*language\s+([A-Za-z]+)\s*<asr_text>\s*", raw)
            if m:
                msg["content"] = raw[m.end():]
                msg["language"] = m.group(1)
        except Exception:
            pass
        return resp
    return await run_in_threadpool(do)


def _tags_by_base_path(models, engine):
    """Same model-type filters as /api/models, keyed by base_path. Used by
    /api/scan so configured source rows inherit mmproj/vision and other
    capabilities the file scan alone cannot know."""
    def _ids(fn):
        try:
            return {m.get("model_id") for m in (fn(models) or [])}
        except Exception:
            return set()
    emb_ids = _ids(engine.filter_embedding_models)
    aud_ids = _ids(engine.filter_audio_models)
    vis_ids = _ids(engine.filter_vision_models)
    dec_ids = _ids(engine.filter_decision_models)
    out = {}
    for m in models:
        mid = m.get("model_id") or ""
        base = mid.split("--")[0]
        arch = (m.get("architecture") or "").lower()
        flags = m.get("flags") or {}
        tags = []
        if m.get("decision_type") or base in dec_ids:
            tags.append("decision")
        else:
            is_speech = base in aud_ids
            if is_speech:
                tags.append("speech")
            if not is_speech and (base in vis_ids or "mmproj" in flags):
                tags.append("vision")
            if base in emb_ids or "bert" in arch or "embed" in arch:
                tags.append("embedding")
            if not tags or tags == ["vision"]:
                tags.append("chat")
        out[m.get("base_path")] = tags
    return out


@router.get("/api/scan")
async def scan_models(path: str, recursive: bool = False,
                      refresh: bool = False):
    if not path:
        raise HTTPException(status_code=400, detail="path query param required")
    # resolve store:// alias for THIS platform, then scan locally
    local = path_resolver.resolve(path)
    key = (local, bool(recursive))
    cached = _scan_cache_get(key, refresh)
    if cached is None:
        models, projectors = await run_in_threadpool(
            model_parser.scan_directory, local, recursive
        )
        _scan_cache_put(key, models, projectors)
    else:
        models, projectors = cached
    # Configured-ness comes from the FULL config set — BEFORE the view
    # filter. A restricted viewer's /api/models only lists their view's
    # models, so comparing against it would show every other config's
    # GGUF as "unconfigured" (and offer to double-configure it). The
    # wizard rule instead: a config existing ANYWHERE on this farm means
    # configured — hidden from the model source for restricted viewers
    # (they never needed it; it's not theirs to touch). The exception is
    # a freshly downloaded GGUF with NO config at all: those stay
    # visible + configurable so the restricted user can wire them up.
    all_base_paths = set()
    import yaml as _yaml
    for fname in os.listdir(cfg.MODELS_DIR):
        if not fname.endswith(".yaml"):
            continue
        if "__tune_tmp" in fname or "__ab_single" in fname:
            continue
        try:
            with open(os.path.join(cfg.MODELS_DIR, fname), "r",
                      encoding="utf-8") as f:
                d = _yaml.safe_load(f) or {}
        except (OSError, _yaml.YAMLError):
            continue
        if d.get("base_path"):
            all_base_paths.add(d["base_path"])
        for v in (d.get("variants") or []):
            if isinstance(v, dict) and v.get("base_path"):
                all_base_paths.add(v["base_path"])
    # Collect draft paths from model configs (as store:// strings); reuse
    # this same list for capability tags below instead of re-reading YAML.
    configured_models = cfg.list_models()
    draft_paths = set()
    for m in configured_models:
        flags = m.get("flags") or {}
        model_draft = flags.get("model-draft")
        if model_draft and model_draft.lower().endswith('.gguf'):
            draft_paths.add(model_draft)
        for variant in m.get("variants", []):
            if not isinstance(variant, dict):
                continue
            v_flags = variant.get("flags") or {}
            v_model_draft = v_flags.get("model-draft")
            if v_model_draft and v_model_draft.lower().endswith('.gguf'):
                draft_paths.add(v_model_draft)
    # Filter out models that are either referenced as a draft or have MTP in their filename
    filtered_models = []
    for model in models:
        base_path = model["base_path"]
        if base_path and base_path.lower().endswith('.gguf'):
            # Extract the filename (the last component of the path)
            # For store://llm-models/file.gguf, we want 'file.gguf'
            filename = base_path.split('/')[-1]
            if base_path in draft_paths or 'mtp' in filename.lower():
                # Skip this model because it is either referenced as a draft or has MTP in the filename
                continue
        filtered_models.append(model)
    # Engine-gate (DEC-07): /api/scan is the wizard's model source list —
    # only models in the ACTIVE engine's downloads vocabulary are
    # offered. llama.cpp serves GGUF (hf entries hidden). Projectors are
    # editor chrome, not engine-qualified model sources.
    engine = engine_registry.get_engine()
    allowed = engine.ui_manifest().get("downloads", {}).get("formats", [])
    filtered_models = [
        m for m in filtered_models
        if m.get("format", "gguf") in allowed
    ]
    # report paths back in store-alias form so configs stay portable
    # (configured-ness below compares AFTER aliasing — configs store
    # store:// paths, scan rows are absolute until here)
    for m in filtered_models:
        m["base_path"] = path_resolver.to_alias(m["base_path"])
        for f in m["files"].values():
            f["path"] = path_resolver.to_alias(f["path"])
    # Wizard source rule (per-IP): a config existing ANYWHERE on this
    # farm = configured. For restricted viewers those rows leave the
    # source list entirely (not theirs to re-configure — and their own
    # view-filtered /api/models would have mislabeled them anyway);
    # trusted operators keep the row, stamped configured=True so the UI
    # shows state without re-deriving it against a view-filtered list.
    kept = []
    for model in filtered_models:
        is_cfgd = model.get("base_path") in all_base_paths
        if is_cfgd and not access.gate_open():
            continue
        model["configured"] = is_cfgd
        kept.append(model)
    filtered_models = kept
    for p in projectors:
        p["base_path"] = path_resolver.to_alias(p["base_path"])
    # attach architecture for grouping in the wizard source list
    # Configured source rows inherit the SAME capability tags as the model
    # list (mmproj -> vision, embed -> embedding, etc.). File-scan heuristics
    # alone cannot see mmproj because projectors are scanned separately.
    try:
        configured_tags = _tags_by_base_path(
            engine.filter_models(configured_models), engine)
    except Exception:
        configured_tags = {}
    # Same cold-start cost as /api/models: warm headers concurrently.
    cfg._bulk_model_meta([m.get("base_path") for m in filtered_models])
    for m in filtered_models:
        m["architecture"] = cfg._get_model_architecture(m["base_path"])
        # TypeSafe decision models surface in the wizard the same way
        m["decision_type"] = cfg._get_decision_type(m["base_path"])
        if m["base_path"] in configured_tags:
            m["model_tags"] = configured_tags[m["base_path"]]
            continue
        mid = (m.get("model_id") or m.get("base_path") or "").lower()
        arch = (m.get("architecture") or "").lower()
        tags = []
        if m["decision_type"]:
            tags.append("decision")
        else:
            is_speech = "asr" in mid or "whisper" in mid or "audio" in mid
            if is_speech:
                tags.append("speech")
            if "embed" in mid or "bert" in arch:
                tags.append("embedding")
            if not is_speech and ("vision" in mid or "vl" in arch
                                  or "-vl" in mid):
                tags.append("vision")
            if not tags or tags == ["vision"]:
                tags.append("chat")
        m["model_tags"] = tags
    return {"models": filtered_models, "projectors": projectors}
