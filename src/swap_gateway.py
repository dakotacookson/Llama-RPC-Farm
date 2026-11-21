"""Swap gateway: a view-aware reverse proxy in front of llama-swap :8080.

Listens on :8090. Forwards /v1/* verbatim EXCEPT:
- /v1/models + /models: the response's data[] is filtered to the models
  the calling client's IP is allowed to see (settings.yaml views:
  allowlist, exactly the seam the farm UI's access.py uses — same file,
  same semantics, no second source of truth).
- /v1/* generate endpoints: a model outside the caller's allowlist gets
  a 404-shaped "model not found" error — never a hint that other models
  exist.
- Trusted IPs (loopback anchors + settings.yaml admin_ips:): everything
  passes through unfiltered.

Configured IPs with NO view see the FULL list (unrestricted — a foreign
IP that isn't bound to any view is treated as a farm guest; tighten by
binding their IP to an empty-models view or dropping them at the
firewall).

Streaming: responses are passed through as bytes. /v1/models bodies are
small JSON so they're buffered and rewritten.

Run:  .venv/bin/python src/swap_gateway.py   (systemd: swap-gateway.service)
"""
import ipaddress
import json
import os
import sys

import time

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

import access as farm_access
from runtime_paths import data_dir
BASE_DIR = data_dir()
SETTINGS_FILE = os.path.join(BASE_DIR, "config", "settings.yaml")

LISTEN_PORT = int(os.environ.get("GATEWAY_PORT", "8090"))
UPSTREAM = os.environ.get("GATEWAY_UPSTREAM", "http://127.0.0.1:8080")
_TRUSTED_ANCHORS = {"127.0.0.1", "::1"}
_EXTRA_TRUSTED_PROXIES = {
    x.strip() for x in (os.environ.get("FARM_TRUSTED_PROXIES") or "").split(",")
    if x.strip()
}
_REQUIRE_KEYS = str(os.environ.get("FARM_GATEWAY_REQUIRE_KEYS") or "").lower() in (
    "1", "true", "yes", "on")
SETTINGS_KEY = "views"
KEYS_SETTINGS_KEY = "access_keys"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# The FARM UI (served on :8199) posts its test tabs' calls to THIS
# gateway (:8090) so every generate path crosses key-auth + usage
# tracking — a browser needs CORS for the cross-origin hop.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware,
                   allow_origin_regex=r"https?://(localhost(:\d+)?|127\.0\.0\.1(:\d+)?|10\.0\.0\.\d+(:\d+)?|100\.\d+\.\d+\.\d+(:\d+)?|\[?[a-f0-9:]+\]?(:\d+)?)",
                   allow_credentials=True,   # session cookie rides along
                   allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["authorization", "x-api-key",
                                  "content-type", "anthropic-version",
                                  "anthropic-beta",
                                  "anthropic-dangerous-direct-browser-access"])
# Local models on big contexts can stall for many minutes between streamed
# chunks (huge prefill, RPC backends, memory pressure). The read timeout is
# *idle-gap* based; the old 30s read surfaced as client-side "connection
# reset" during long 27B/RPC prefills, and 30 min still cut first-load crams
# that can legitimately take an hour or more. No read deadline: client
# disconnects cancel the upstream pull, and the gateway is LAN/VPN-only.
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(None, connect=30.0, read=None, write=600.0))


def _ip_or_net(value):
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        pass
    try:
        return ipaddress.ip_network(value, strict=False)
    except ValueError:
        return None


def _load_settings():
    import yaml
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}



def views_table():
    raw = _load_settings().get(SETTINGS_KEY) or {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for name, ent in raw.items():
        if not isinstance(ent, dict):
            continue
        ips = ent.get("ips") or []
        models = ent.get("models") or []
        if isinstance(ips, str):
            ips = [ips]
        if isinstance(models, str):
            models = [models]
        ips = {str(i).strip() for i in ips if str(i).strip()}
        models = {str(m).strip() for m in models if str(m).strip()}
        if ips:
            out[str(name).strip()] = {"ips": ips, "models": models}
    return out


def _trusted_proxy(ip: str) -> bool:
    if not ip:
        return False
    if ip in _TRUSTED_ANCHORS or ip in _EXTRA_TRUSTED_PROXIES:
        return True
    try:
        import ipaddress
        addr = ipaddress.ip_address(ip)
        for item in _EXTRA_TRUSTED_PROXIES:
            try:
                if addr in ipaddress.ip_network(item, strict=False):
                    return True
            except ValueError:
                pass
    except ValueError:
        pass
    return False


def client_ip(request: Request) -> str:
    """Real client IP; forwarded headers are trusted only from a proxy."""
    c = getattr(request, "client", None)
    actual = (c.host if c and c.host else "") or ""
    if not _trusted_proxy(actual):
        return actual
    for header in ("x-forwarded-for", "x-real-ip"):
        val = (request.headers.get(header) or "").strip()
        if val:
            return val.split(",")[0].strip()
    return actual


def request_key(request: Request) -> str:
    """Access key from Authorization: Bearer / X-Api-Key (first found).
    NOTE: llama-swap ALSO validates this key against its own apiKeys —
    the gateway consumes the identity BEFORE forwarding; unknown keys
    still pass llama-swap's own check when it lists the same key."""
    hdr = (request.headers.get("authorization") or "").strip()
    if hdr.lower().startswith("bearer "):
        hdr = hdr[7:].strip()
    if not hdr:
        hdr = (request.headers.get("x-api-key") or "").strip()
    return hdr


def request_key_name(request: Request) -> str | None:
    """Identity NAME for usage tracking / attribution — never the raw
    token. Header AND login-session identities resolve to the access-key
    record name, so quotas/usage stay consistent for browser users."""
    tok = request_token(request)
    if tok:
        rec = access_keys().get(tok)
        return rec["name"] if rec else None
    user = _session_user(request)
    if user:
        return user
    if client_ip(request) in _TRUSTED_ANCHORS:
        return "console"
    return None


def request_key_record(request: Request):
    tok = request_token(request)
    return access_keys().get(tok) if tok else None


def request_token(request: Request) -> str:
    """The EFFECTIVE access-key token for this request: header first,
    else the login-session cookie's mapped key. The gateway needs the
    TOKEN (not just the name) for model-allowlist checks — a cookie
    login must carry the same grants as its key."""
    tok = request_key(request)
    if tok:
        return tok
    import secrets_vault as _v
    user = _session_user(request)
    if not user:
        return ""
    rec = _v.get_web_login(user)
    if rec and rec.get("key"):
        try:
            return _v.decrypt_value(rec["key"])
        except Exception:
            return ""
    return ""


def _session_user(request: Request) -> str | None:
    """Farm login user from the session cookie, or None."""
    sid = request.cookies.get("farm_session") or ""
    if not sid:
        return None
    try:
        import secrets_vault as _v
        u = _v.get_web_session(sid)
        return u or None
    except Exception:
        return None


def access_keys():
    raw = _load_settings().get(KEYS_SETTINGS_KEY) or {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for tok, ent in raw.items():
        tok = str(tok).strip()
        if not tok:
            continue
        if not isinstance(ent, dict):
            ent = {}
        models = ent.get("models") or []
        if isinstance(models, str):
            models = [models]
        models = {str(m).strip() for m in models if str(m).strip()}
        out[tok] = {"name": str(ent.get("name") or tok),
                    "models": models,
                    "admin": bool(ent.get("admin", False)),
                    "limits": farm_access.normalize_limits(
                        ent.get("limits"))}
    return out



def _unauthorized():
    from fastapi import HTTPException
    return HTTPException(401, "invalid or missing API key")


def allow_for_ip(request: Request):
    """Identity-by-key: allowlist for the presented key. None =
    unrestricted (no keys configured, unscoped key, admin key without
    model grants, loopback console). IP-allow args kept for compat.

    STRICT mode (operator decision): once keys exist, a request without
    a resolvable VALID key gets (models=[], generate=401) — farm parity
    with :8199. Unknown/malformed keys are refusals, not "lists open". """
    keys = access_keys()
    if not keys:
        if _REQUIRE_KEYS and request_key_name(request) != "console":
            return set()
        return None  # pre-Access farm state: open
    tok = request_token(request)
    rec = keys.get(tok) if tok else None
    if rec is not None and not rec.get("models"):
        # known key with no model grants = unscoped: unrestricted, the
        # same rule as access.restrict_scope_from_key on :8199. The
        # first-run admin key is created exactly like this
        # (models: [], admin_unrestricted: true).
        return None
    if rec is None:
        # loopback console (the farm itself proxying server-side) stays
        # unrestricted: the FARM's embeddings/audio/vision proxies cross
        # this gateway without a key by design — tracking attributes
        # them to "console"
        if request_key_name(request) == "console":
            return None
        # any OTHER keyless/unknown caller: zero models + 401
        return set()
    return rec["models"]


GATEWAY_PATHS = ("v1/", "models", "health", "upstream")
GENERATE_PATHS = ("/v1/chat/completions", "/v1/completions",
                  "/v1/embeddings", "/v1/responses", "/v1/rerank",
                  "/v1/audio/", "/v1/images/", "/v1/systemone",
                  "/v1/messages")
# count_tokens is model-scoped (same key/allowlist gate) but not a
# generation: no usage row.
MODEL_PATHS = GENERATE_PATHS + ("/v1/messages/count_tokens",)


def _is_model_path(req_path: str) -> bool:
    """Key/allowlist scope applies to this request."""
    p = (req_path or "").rstrip("/")
    for pref in MODEL_PATHS:
        if p == pref.rstrip("/"):
            return True
        if pref.endswith("/") and p.startswith(pref):
            return True
        if not pref.endswith("/") and p.startswith(pref + "/"):
            return True
    return False


def _is_generate_path(req_path: str) -> bool:
    """Request produces (or would produce) generated tokens for usage."""
    p = (req_path or "").rstrip("/")
    if p == "/v1/messages/count_tokens":
        return False
    for pref in GENERATE_PATHS:
        if p == pref.rstrip("/"):
            return True
        if pref.endswith("/") and p.startswith(pref):
            return True
        if not pref.endswith("/") and p.startswith(pref + "/"):
            return True
    return False


def _mid_allowed(mid: str, allow) -> bool:
    """EXACT token match. A bare mid grants ONLY the base entry; a
    variant ("mid--label") appears iff its own token is granted. Peer-
    qualified names ("peer/model") compare their tail. (Legacy behavior
    admitted a base's whole variant family with one bare token — that
    leaked variants the operator never selected.)"""
    if mid in allow:
        return True
    if "/" in mid:
        tail = mid.rsplit("/", 1)[1]
        return _mid_allowed(tail, allow)
    return False


def _anthropic_usage(u) -> dict:
    """Anthropic usage -> {prompt_tokens, completion_tokens}.

    Anthropic's `input_tokens` EXCLUDES cache-read/cache-creation tokens
    (llama.cpp reports e.g. input_tokens=4, cache_read_input_tokens=12
    for the same prompt OpenAI counts as ~17). Sum them so farm usage is
    comparable across protocols.
    """
    if not isinstance(u, dict):
        return {}
    prompt = (int(u.get("input_tokens") or 0)
              + int(u.get("cache_read_input_tokens") or 0)
              + int(u.get("cache_creation_input_tokens") or 0))
    out = int(u.get("output_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": out}


def _quota_error(req_path: str, blocked: dict):
    """429 in the caller's protocol shape, with retry/limit headers."""
    period = blocked.get("period") or "quota"
    used = blocked.get("used") or 0
    limit = blocked.get("limit") or 0
    retry = max(1, int(blocked.get("reset_in") or 60))
    msg = (f"{period} request limit reached ({used}/{limit}); "
           f"retry after {retry}s")
    headers = {
        "Retry-After": str(retry),
        "X-RateLimit-Limit-Requests": str(limit),
        "X-RateLimit-Remaining-Requests": "0",
        "X-RateLimit-Reset-Requests": str(int(blocked.get("reset_at") or 0)),
    }
    if (req_path or "").startswith("/v1/messages"):
        return JSONResponse(status_code=429, headers=headers, content={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": msg}})
    from fastapi import HTTPException
    raise HTTPException(status_code=429, detail=msg, headers=headers)


def _gate_error(req_path: str, status: int, message: str):
    """401/404 in the caller's protocol shape.

    Anthropic SDKs parse `{"type":"error","error":{...}}`; OpenAI clients
    already get the FastAPI `{"detail": ...}` shape from every other farm
    endpoint. Returns a JSONResponse for Anthropic, raises otherwise.
    """
    if (req_path or "").startswith("/v1/messages"):
        etype = {401: "authentication_error",
                 404: "not_found_error"}.get(status, "api_error")
        return JSONResponse(status_code=status, content={
            "type": "error",
            "error": {"type": etype, "message": message}})
    from fastapi import HTTPException
    raise HTTPException(status_code=status, detail=message)



_CANON_MODELS: list = []
_CANON_TS: float = 0.0


async def canonical_model(mid: str) -> str:
    """Full canonical model id for a caller-provided string. The farm's
    rule: usage buckets carry the FULL deployment name, never a client
    shorthand — llama-swap's /v1/models is the authority (exact id first,
    case-insensitive fallback). Unresolvable strings pass through
    unchanged (still recorded — a request happened)."""
    global _CANON_MODELS, _CANON_TS
    mid = (mid or "").strip()
    if not mid:
        return mid
    now = time.time()
    if not _CANON_MODELS or now - _CANON_TS > 30:
        try:
            data = await client.get("/v1/models", timeout=4)
            _CANON_MODELS = [m.get("id") or "" for m in
                             (data.json().get("data") or [])]
            _CANON_TS = now
        except Exception:
            pass
    for c in _CANON_MODELS:
        if c and c == mid:
            return c
    lm = mid.lower()
    for c in _CANON_MODELS:
        if c and c.lower() == lm:
            return c
    # partial match (client sent a prefix / short alias)
    for c in _CANON_MODELS:
        if c and (c.lower().startswith(lm) or lm in c.lower()):
            return c
    return mid


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE",
                                         "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    if not path.startswith(GATEWAY_PATHS) and path not in (
            "", "docs", "ui"):
        # non-API paths (UI, logs stream etc.) are a farm-facing surface —
        # gateway only exposes the OpenAI API + health probes
        raise _not_found()
    allow = allow_for_ip(request)

    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "connection")}

    req_path = "/" + path
    _scope = _is_model_path(req_path)
    _gen = _is_generate_path(req_path)
    # STRICT key gate: once keys exist, no valid key means model-scoped
    # paths answer 401 outright (never a forwarded 500) and model lists
    # come back empty — the caller learns the protocol, not the farm.
    if allow is not None and not allow:
        if _scope:
            return _gate_error(req_path, 401,
                               "invalid or missing API key")
        if req_path.rstrip("/") in ("/v1/models", "/models"):
            return JSONResponse(status_code=200, content={
                "object": "list", "data": []})
        # other GETs (health etc.) still pass through — cheap, harmless

    # TRUTH 1: usage tracking rides the IDENTITY on every generate path
    # (console included) — independent of scoping/enforcement.
    # TRUTH 2: enforcement (401 / model gate) applies ONLY to scoped keys.
    if _scope and allow is not None and len(allow) == 0:
        return _gate_error(req_path, 401, "invalid or missing API key")
    payload = {}
    mid = ""
    if _scope:
        try:
            payload = json.loads(body or b"{}")
        except Exception:
            payload = {}
        mid = await canonical_model(
            (payload.get("model") or "").strip())
        if allow is not None and not _mid_allowed(mid, allow):
            return _gate_error(req_path, 404,
                               f"model '{mid}' not found")

    # usage tracking identity (key NAME) — resolve BEFORE the hop; the
    # collector never sees raw tokens
    import usage_collector as _uc
    _urec = request_key_record(request)
    _uname = request_key_name(request)
    if _gen and _urec:
        blocked = _uc.check_quota(_urec["name"], _urec.get("limits"))
        if blocked:
            return _quota_error(req_path, blocked)
    _t0 = time.monotonic()

    upstream_req = client.build_request(
        request.method, f"/{path}", content=body or None,
        headers=headers, params=request.query_params)
    upstream_resp = await client.send(upstream_req, stream=True)
    _dur = time.monotonic() - _t0

    # record usage for generate paths: tokens when the response carries
    # the OpenAI usage block (buffered JSON); streams are estimated from
    # content length as they stream past (see counting below)
    if _gen and _uname:
        _ct = upstream_resp.headers.get("content-type", "")
        print(f"[usage] branch ct={_ct!r} name={_uname} mid={mid}", flush=True)
        if _ct.startswith("application/json"):
            try:
                data = await upstream_resp.aread()
                j = json.loads(data)
                u = j.get("usage") or {}
                tm = j.get("timings") or {}
                if tm and (tm.get("predicted_per_second") or 0) > 0:
                    # exact server-measured decode (prefill excluded)
                    _toks = {"prompt_tokens": tm.get("prompt_n")
                             or u.get("prompt_tokens") or 0,
                             "completion_tokens": tm.get("predicted_n")
                             or u.get("completion_tokens") or 0}
                    _dur = round((tm.get("predicted_n") or 0) /
                                 (tm.get("predicted_per_second") or 1), 3)
                else:
                    _toks = {"prompt_tokens": (u.get("prompt_tokens")
                                               or u.get("input_tokens") or 0),
                             "completion_tokens": (
                                 u.get("completion_tokens")
                                 or u.get("output_tokens") or 0)}
                _uc.record(_uname, path, mid,
                           upstream_resp.status_code, _dur, _toks)
                # rebuild the response (already buffered)
                from fastapi.responses import Response as _Resp
                out_headers = {k.decode() if isinstance(k, bytes) else k:
                               v.decode() if isinstance(v, bytes) else v
                               for k, v in upstream_resp.headers.items()
                               if k.lower() not in (
                                   "content-length", "transfer-encoding",
                                   "connection")}
                await upstream_resp.aclose()
                return _Resp(content=data,
                             status_code=upstream_resp.status_code,
                             headers=out_headers,
                             media_type="application/json")
            except Exception:
                pass
        else:
            # streaming: count content deltas in the background closer
            return StreamingResponse(
                _counting_stream(upstream_resp, {
                    "name": _uname, "path": path, "model": mid,
                    "status": upstream_resp.status_code, "dur": _dur}),
                status_code=upstream_resp.status_code,
                headers={k: v for k, v in upstream_resp.headers.items()
                         if k.lower() not in (
                             "content-length", "transfer-encoding",
                             "connection")},
                background=_CloseBg(upstream_resp))

    # model list: rewrite the JSON body to the caller's view
    if allow is not None and req_path.rstrip("/") in ("/v1/models", "/models"):
        data = await upstream_resp.aread()
        try:
            parsed = json.loads(data)
            parsed["data"] = [m for m in (parsed.get("data") or [])
                              if _mid_allowed((m.get("id") or ""), allow)]
            payload = (json.dumps(parsed, separators=(",", ":")) + "\n")\
                .encode()
            out_headers = {k.decode() if isinstance(k, bytes) else k:
                           v.decode() if isinstance(v, bytes) else v
                           for k, v in upstream_resp.headers.items()
                           if k.lower() not in (
                               "content-length", "transfer-encoding",
                               "connection")}
            await upstream_resp.aclose()
            return Response(content=payload, status_code=upstream_resp.status_code,
                            headers=out_headers, media_type="application/json")
        except Exception:
            pass  # non-JSON upstream body falls through verbatim

    return StreamingResponse(
        upstream_resp.aiter_bytes(),
        status_code=upstream_resp.status_code,
        headers={k: v for k, v in upstream_resp.headers.items()
                 if k.lower() not in ("content-length", "transfer-encoding",
                                      "connection")},
        background=_CloseBg(upstream_resp))


async def _counting_stream(resp, meta):
    """Wrap the upstream byte stream: count generated output and record
    ONE usage row when the stream closes.

    Handles OpenAI SSE (`choices[].delta.content` / `reasoning_content`),
    llama.cpp's final `timings` block, and Anthropic Messages SSE
    (`message_start` / `content_block_delta` / `message_delta`). Exact
    token counts come from the server when it reports them; otherwise the
    row is estimate-flagged.
    """
    chars = 0
    saw_usage = None
    saw_anthropic = {}
    saw_timings = None
    final_status = meta["status"]
    _t_first = None    # first CONTENT/reasoning delta = decode start
    _t_last = None     # last chunk = decode end (excludes prefill)

    def _mark(text):
        nonlocal chars, _t_first
        if not text:
            return
        if _t_first is None:
            _t_first = time.monotonic()
        chars += len(text)

    async for chunk in resp.aiter_bytes():
        _t_last = time.monotonic()
        try:
            for line in chunk.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    continue
                try:
                    j = json.loads(raw)
                    # llama.cpp streams a FINAL chunk carrying per-run
                    # TIMINGS (predicted_per_second etc.) — the true
                    # decode rate for THAT run, prefill excluded.
                    tm = j.get("timings") or {}
                    if tm:
                        saw_timings = tm
                    u = j.get("usage")
                    if u and u.get("completion_tokens"):
                        saw_usage = u
                    typ = j.get("type")
                    if typ == "message_start":
                        au = _anthropic_usage(
                            (j.get("message") or {}).get("usage"))
                        if au.get("prompt_tokens"):
                            saw_anthropic["prompt_tokens"] = au["prompt_tokens"]
                        if au.get("completion_tokens"):
                            saw_anthropic["completion_tokens"] = \
                                au["completion_tokens"]
                    elif typ == "message_delta":
                        au = _anthropic_usage(j.get("usage"))
                        if au.get("prompt_tokens"):
                            saw_anthropic["prompt_tokens"] = au["prompt_tokens"]
                        if au.get("completion_tokens"):
                            saw_anthropic["completion_tokens"] = \
                                au["completion_tokens"]
                    elif typ == "content_block_start":
                        cb = j.get("content_block") or {}
                        _mark(cb.get("text") or cb.get("thinking") or "")
                    elif typ == "content_block_delta":
                        dlt = j.get("delta") or {}
                        if dlt.get("type") in ("text_delta",
                                               "thinking_delta"):
                            _mark(dlt.get("text")
                                  or dlt.get("thinking") or "")
                    # OpenAI-style chunks (and llama.cpp's mixed shape)
                    for ch in (j.get("choices") or []):
                        d = ch.get("delta") or {}
                        _mark(d.get("content") or "")
                        _mark(d.get("reasoning_content") or "")
                except Exception:
                    chars += max(0, len(raw) - 80)
        except Exception:
            pass
        yield chunk

    tokens = None
    est = True
    if saw_timings and (saw_timings.get("predicted_per_second") or 0) > 0:
        # server-reported per-run decode rate beats every estimate:
        # predicted_n / predicted_per_second give the EXACT tokens and
        # decode seconds that llama.cpp measured (prefill excluded)
        tokens = {
            "prompt_tokens": (saw_timings.get("prompt_n") or 0),
            "completion_tokens": (saw_timings.get("predicted_n") or 0)}
        _dur_decode = (saw_timings.get("predicted_n") or 0) / \
            (saw_timings.get("predicted_per_second") or 1)
        est = False
    elif saw_anthropic:
        tokens = {
            "prompt_tokens": saw_anthropic.get("prompt_tokens") or 0,
            "completion_tokens": (
                saw_anthropic.get("completion_tokens")
                or max(0, chars // 4))}
        # output_tokens from message_delta is exact; a stream cut before
        # that event falls back to the character estimate.
        est = not bool(saw_anthropic.get("completion_tokens"))
        _dur_decode = round((_t_last - _t_first), 3) if (
            _t_first and _t_last and _t_last > _t_first) else 0.0
    elif saw_usage:
        tokens = {"prompt_tokens": (saw_usage.get("prompt_tokens")
                                    or saw_usage.get("input_tokens") or 0),
                  "completion_tokens": (
                      saw_usage.get("completion_tokens")
                      or saw_usage.get("output_tokens") or 0)}
        est = False
        _dur_decode = round((_t_last - _t_first), 3) if (
            _t_first and _t_last and _t_last > _t_first) else 0.0
    else:
        tokens = {"prompt_tokens": 0,
                  "completion_tokens": max(0, chars // 4)}
        _dur_decode = round((_t_last - _t_first), 3) if (
            _t_first and _t_last and _t_last > _t_first) else 0.0
    try:
        import usage_collector as _uc
        _uc.record(meta["name"], meta["path"], meta["model"],
                   final_status, _dur_decode, tokens, estimated=est)
        print(f"[usage-close] {meta['name']}/{meta['model']} "
              f"chars={chars} tok={tokens}", flush=True)
    except Exception as e:
        print(f"[usage-close-ERR] {e}", flush=True)


class _CloseBg:
    """Ensure the streamed upstream response is released."""
    def __init__(self, resp):
        self._resp = resp

    async def __call__(self):
        try:
            await self._resp.aclose()
        except Exception:
            pass


def _not_found():
    from fastapi import HTTPException
    return HTTPException(404, "not found")


def _model_not_found(mid: str):
    from fastapi import HTTPException
    label = mid or "(missing)"
    return HTTPException(404, f"model '{label}' not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, reload=False)