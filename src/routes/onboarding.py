"""First-five-minutes onboarding: recipes, pairing, clients, safety, metrics."""
import json

import pydantic
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

import access
import config_loader as cfg
import local_node
import onboarding
import swap_manager
from web_shared import (_access_filter_models, _gateway_headers,
                        _gateway_url, _raw_is_private)

router = APIRouter()


def _control():
    return ((access.gate_open() and access.view_allow() is None)
            or access.is_admin(None))


def _require_control():
    if not _control():
        raise HTTPException(404, "not found")


def _identity_ok():
    """Any logged-in/scoped caller may see their own client configs."""
    return bool(access.gate_open() or access.current_key_record()
                or access.session_user())


def _require_identity():
    if not _identity_ok():
        raise HTTPException(404, "not found")


def _local_name():
    for m in cfg.machines.all():
        if local_node.is_local(m):
            return m.get("name")
    return ""


@router.get("/api/onboarding/recipes")
async def recipes():
    return {"recipes": onboarding.list_recipes(),
            "hardware": onboarding.fleet_hardware(),
            "local_machine": _local_name()}


class RecipeApplyIn(pydantic.BaseModel):
    machine: str = ""
    launch: bool = True


@router.post("/api/onboarding/recipes/{recipe_id}/apply")
async def recipe_apply(recipe_id: str, body: RecipeApplyIn = RecipeApplyIn()):
    _require_control()
    try:
        onboarding.get_recipe(recipe_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from None
    try:
        job_id = onboarding.apply_recipe(
            recipe_id, body.machine or None, launch=body.launch)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"ok": True, "job_id": job_id}


@router.get("/api/onboarding/recipes/jobs/{job_id}")
async def recipe_job(job_id: str):
    _require_control()
    j = onboarding._job(job_id)
    if not j:
        raise HTTPException(404, "unknown recipe job")
    return j


@router.post("/api/onboarding/access-key")
async def create_key():
    _require_control()
    try:
        token = onboarding.create_access_key("quickstart")
    except Exception as e:
        raise HTTPException(400, str(e)) from None
    return {"ok": True, "token": token, "name": "quickstart"}


@router.get("/api/onboarding/pairing-command")
async def pairing_command():
    _require_control()
    try:
        return onboarding.pairing_commands()
    except Exception as e:
        raise HTTPException(400, str(e)) from None


def _model_options():
    out = []
    seen = set()
    try:
        import engine_registry
        engine = engine_registry.get_engine()
        models = _access_filter_models(cfg.list_models())
        def _ids(fn):
            try:
                return {m.get("model_id") for m in (fn(models) or [])}
            except Exception:
                return set()
        decis = _ids(engine.filter_decision_models)
        embeds = _ids(engine.filter_embedding_models)
        audio = _ids(engine.filter_audio_models)
        vision = _ids(engine.filter_vision_models)
        for m in models:
            mid = (m.get("model_id") or "").strip()
            if not mid or mid in seen:
                continue
            if not access.gate_open() and _raw_is_private(mid.split("--")[0]):
                continue
            seen.add(mid)
            kinds = []
            if mid in decis:
                kinds.append("decision")
            if mid in embeds:
                kinds.append("embeddings")
            if mid in audio:
                kinds.append("audio")
            if mid in vision:
                kinds.append("vision")
            if not kinds or (mid not in decis and mid not in embeds
                             and mid not in audio):
                # Vision models are also text chat models. Decision,
                # embeddings and ASR are their own endpoint families.
                kinds.insert(0, "chat")
            kind = kinds[0]
            out.append({"id": mid, "name": m.get("name") or mid,
                        "kind": kind, "kinds": kinds})
    except Exception:
        pass
    return out


def _sample_model():
    opts = _model_options()
    return opts[0]["id"] if opts else "your-model"


@router.get("/api/onboarding/clients")
async def clients(request: Request, model: str = ""):
    _require_identity()
    host = request.url.hostname or "127.0.0.1"
    base = f"http://{host}:8090"
    models = _model_options()
    chosen = (model or "").strip()
    chat_default = next((m["id"] for m in models
                         if "chat" in (m.get("kinds") or [])), None)
    if not chosen or not any(m["id"] == chosen for m in models):
        chosen = chat_default or (models[0]["id"] if models else "your-model")
    m = next((x for x in models if x["id"] == chosen), {})
    kinds = m.get("kinds") or [m.get("kind") or "chat"]
    return {"base_url": base, "model": chosen, "models": models,
            "kind": m.get("kind") or "chat", "kinds": kinds,
            "clients": onboarding.client_configs(base, chosen, kinds)}


class TestIn(pydantic.BaseModel):
    model: str = ""
    prompt: str = "Reply with OK."


@router.post("/api/onboarding/test")
async def test_endpoint(t: TestIn, request: Request):
    _require_identity()
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(503, "llama-swap is not running")
    model = (t.model or _sample_model()).strip()
    mopt = next((m for m in _model_options() if m["id"] == model), {})
    kinds = mopt.get("kinds") or [mopt.get("kind") or "chat"]
    if "audio" in kinds or "vision" in kinds:
        raise HTTPException(
            400, "Use the Audio/Vision test tabs — this model needs a "
                 "media payload, not a text-only probe.")
    if "decision" in kinds:
        path = "/v1/systemone"
        payload = {"model": model, "state": t.prompt,
                   "questions": {"route": {
                       "type": "choice", "instructions": "Pick one",
                       "criteria": {"yes": 1, "no": 2}}}}
    elif "embeddings" in kinds:
        path = "/v1/embeddings"
        payload = {"model": model, "input": t.prompt}
    else:
        path = "/v1/chat/completions"
        payload = {"model": model,
                   "messages": [{"role": "user", "content": t.prompt}],
                   "max_tokens": 8, "stream": False}

    def _run():
        import time
        import urllib.error
        import urllib.request
        base = _gateway_url("")
        headers = _gateway_headers(request)
        req = urllib.request.Request(
            base + "/v1/models", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=15) as r:
            models = json.loads(r.read().decode()).get("data", [])
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            base + path, data=body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST")
        chat = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    chat = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code in (500, 502, 503) and attempt < 2:
                    time.sleep(1.5)
                    continue
                raise
        if path == "/v1/embeddings":
            reply = f"{len(chat.get('data') or [])} embedding(s)"
        elif path == "/v1/systemone":
            reply = json.dumps(chat.get("answers") or {})[:120]
        else:
            m = (chat.get("choices") or [{}])[0].get("message") or {}
            reply = m.get("content") or m.get("reasoning_content") or ""
        return {"ok": True, "model": model, "models": len(models),
                "kind": kinds[0], "reply": str(reply)[:120]}

    try:
        return await run_in_threadpool(_run)
    except Exception as e:                              # noqa: BLE001
        raise HTTPException(502, str(e)) from None


@router.post("/api/onboarding/local-store")
async def local_store():
    _require_control()
    try:
        root = onboarding.ensure_default_store()
    except Exception as e:
        raise HTTPException(400, str(e)) from None
    return {"ok": True, "root": root, "alias": onboarding._store_alias(),
            "platform": __import__("path_resolver").LOCAL_PLATFORM}


@router.get("/api/onboarding/safety")
async def safety():
    _require_control()
    return onboarding.safety_report()


class MarkIn(pydantic.BaseModel):
    event: str
    extra: dict | None = None


@router.post("/api/onboarding/mark")
async def mark(m: MarkIn):
    _require_control()
    return onboarding.mark(m.event, m.extra)


@router.get("/api/onboarding/metrics")
async def metrics():
    _require_control()
    return onboarding.metrics_summary()


@router.get("/api/onboarding/status")
async def status():
    _require_control()
    return {
        "local_machine": _local_name(),
        "recipes": len(onboarding.RECIPES),
        "keys": len(access.access_keys()),
        "safety": onboarding.safety_report(),
    }
