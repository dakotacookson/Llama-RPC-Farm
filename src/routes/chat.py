"""Native chat test endpoint (streaming proxy to llama-swap/gateway)."""
import json

import pydantic
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

import access
import config_loader as cfg
import engine_registry
import swap_manager
from web_shared import (_access_filter_models, _gateway_headers,
                        _gateway_url, _raw_is_private, LOGIN_COOKIE)

router = APIRouter()


# ---------------- chat test (native, no chainlit) ----------------
# The Chat tab serves its UI from ui/ and talks to these two endpoints;
# the app forwards to llama-swap server-side (same-origin for the
# browser, no CORS, no second service). Loads stream: the first request
# on an unloaded model triggers the load (minutes to 1h+ on big RAM
# crams), tokens stream as the upstream sends them.


class ChatIn(pydantic.BaseModel):
    model: str
    messages: list[dict] = []
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""


@router.get("/api/chat/models")
async def chat_models(request: Request):
    """Models the Chat tab can test: swap-known ids with loaded flags,
    merged with configured (not-yet-pushed) ids — loaded vs will-load,
    same split the old picker showed. Embeddings excluded (they use
    /v1/embeddings, not chat)."""
    st = swap_manager.status()
    running = bool(st.get("running"))
    out, loaded = [], set()
    if running:
        try:
            def _fetch():
                import urllib.request
                import json as _json
                req = urllib.request.Request(
                    _gateway_url("/models"),
                    headers=_gateway_headers(request))
                with urllib.request.urlopen(req, timeout=10) as r:
                    return _json.loads(r.read().decode()).get("data", [])
            data = await run_in_threadpool(_fetch)
            for m in data:
                mid = m.get("id") or ""
                if not mid or "embed" in mid.lower():
                    continue
                out.append(mid)
                if (m.get("status") or {}).get("value") == "loaded":
                    loaded.add(mid)
        except Exception as e:
            print(f"[chat] swap model list failed: {e}")
    try:
        engine = engine_registry.get_engine()
        configured = _access_filter_models(
            engine.filter_models(cfg.list_models()))
    except ValueError:
        configured = _access_filter_models(cfg.list_models())
    for m in configured:
        mid = m.get("model_id") or ""
        if not mid or "embed" in mid.lower() or mid in out:
            continue
        out.append(mid)
    # access.py: private + key-allowlist filtering. The gateway list is
    # already key-filtered; the configured fallback was filtered above.
    # Re-apply the private gate for the daemon list (defense in depth).
    if not access.gate_open():
        allow = access.view_allow()
        out = [mid for mid in out
               if not _raw_is_private(mid.split("--")[0])]
        if not allow:
            out = []
    return {"running": running,
            "models": [{"id": mid, "loaded": mid in loaded} for mid in out]}


@router.post("/api/chat")
async def chat_complete(c: ChatIn, request: Request = None):
    """Stream a chat completion as SSE (OpenAI chunks passed through
    verbatim). Long-lived on purpose: no read timeout (a first-load
    cram can take 1h+); client disconnects cancel the upstream pull."""
    mid = (c.model or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="model is required")
    # access.py: same "never existed" rule as the models table — a
    # restricted viewer gets the 404-shaped error even if the swap
    # daemon currently serves the model for a trusted user. View-
    # scoped users additionally may only chat their allowlist (mid or
    # mid--variant tokens; the plain mid covers its variants).
    if not access.gate_open():
        base_mid = mid.split("--")[0]
        if _raw_is_private(base_mid):
            raise HTTPException(status_code=404, detail=f"model '{mid}' not found")
        allow = access.view_allow()
        # EXACT token match — variants are separate grants
        if allow is not None and mid not in allow:
            raise HTTPException(status_code=404, detail=f"model '{mid}' not found")
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503,
                            detail="swap not running — Launch All first")
    # upstream = THE FARM GATEWAY (:8090): every chat crosses key-auth
    # + usage tracking; the caller's identity (cookie/Bearer) forwards
    # so the usage row attributes to THEM, not "console"
    base = _gateway_url("")
    import httpx as _httpx
    msgs = ([{"role": "system", "content": c.system_prompt}]
            if (c.system_prompt or "").strip() else []) + (c.messages or [])
    payload = {"model": mid, "messages": msgs, "stream": True,
               "temperature": c.temperature, "max_tokens": c.max_tokens}

    async def gen():
        timeout = _httpx.Timeout(connect=10.0, read=None, write=60.0,
                                 pool=60.0)
        async with _httpx.AsyncClient(timeout=timeout) as client:
            # identity forward: cookie/Bearer from the CALLER rides to
            # the gateway — usage attributes / grants apply per-user
            fwd = {}
            sid = (request.cookies.get(LOGIN_COOKIE) if request else "") or ""
            if sid:
                fwd["cookie"] = f"{LOGIN_COOKIE}={sid}"
            tok = access.request_key(request) if request else ""
            if tok:
                fwd["authorization"] = f"Bearer {tok}"
            try:
                async with client.stream(
                        "POST", f"{base}/v1/chat/completions",
                        json=payload, headers=fwd) as up:
                    if up.status_code != 200:
                        try:
                            body = (await up.aread()).decode()[:2000]
                        except Exception:
                            body = ""
                        try:
                            detail = json.loads(body).get(
                                "error", {}).get("message", body)
                        except Exception:
                            detail = body or f"upstream {up.status_code}"
                        # SSE error frame so the UI renders it inline
                        yield f"data: {json.dumps({'error': detail})}\n\n"
                        return
                    async for line in up.aiter_lines():
                        if line.strip():
                            yield line + "\n"
            except _httpx.ConnectError as e:
                yield f"data: {json.dumps({'error': f'swap unreachable: {e}'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
