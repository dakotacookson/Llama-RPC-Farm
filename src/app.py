"""FastAPI app: web UI + JSON API for Llama RPC Farm.

Run: python src/app.py  (or uvicorn app:app --port 8199)
"""
import asyncio
import json
import math
import re
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException, WebSocket, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pydantic
from concurrent.futures import ThreadPoolExecutor

from fastapi.concurrency import run_in_threadpool

import config_loader as cfg
import engine_registry
import settings_persister
import vllm_manager
import gpu_vendor
import hf_downloader
import local_node
import swap_manager
import model_parser
import path_resolver
import stats_collector
import rpc_manager
import gguf_inspect

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "ui", "templates"))
templates.env.auto_reload = True  # pick up index.html edits without a restart

app = FastAPI(title="Llama RPC Farm")




@app.on_event("startup")
def seed_local_master():
    """The machine running this app is the default master node."""
    local_node.ensure_local_master()
    # DEC-17: one-time stamp of `engine` on legacy group entries (both
    # collections) so existing groups keep rendering under their engine.
    # Best-effort: a locked/read-only config dir never blocks boot.
    try:
        cfg.migrate_group_engines()
    except Exception:
        pass
    # Engine reconcile at boot: whatever engine settings.yaml names is the
    # source of truth, so the swap host's config file (per-engine, DEC-03)
    # must be regenerated from the CURRENT model configs before anyone
    # points a client at :8080 — otherwise a daemon left over from the
    # previous engine (e.g. config.vllm.yaml after switching back to
    # llama.cpp) keeps serving its stale entries until a manual Launch
    # All. Best-effort + backgrounded: SSH/config trouble must never
    # block boot (same policy as migrate_group_engines above).
    import threading as _thr

    def _reconcile_swap():
        import swap_manager as _sm
        # access.py: background housekeeping is always trusted, so the
        # reconcile sees every model (including `private: true` ones)
        # and keeps their swap entries pushed while the HTTP gate is
        # what actually hides them from foreign viewers.
        import access as _access
        with _access.trust_current_thread():
            try:
                report = _sm.push_scripts()
                if report.get("error"):
                    print(f"[boot-reconcile] push failed: {report['error']}")
                    return
                # Only a daemon that is ALREADY running needs the reload; a
                # stopped one picks the fresh config up at its next launch
                # (do_stop_swap/do_start in launch()). reload_daemon_async
                # skips when a reload is already in progress.
                if _sm.status().get("running"):
                    res = _sm.reload_daemon_async()
                    if res.get("error"):
                        print(f"[boot-reconcile] daemon reload failed: "
                              f"{res['error']}")
                print("[boot-reconcile] swap config pushed for active engine")
            except Exception as e:
                print(f"[boot-reconcile] skipped: {e}")

    _thr.Thread(target=_reconcile_swap, daemon=True,
                name="boot-swap-reconcile").start()


class NoCacheStaticFiles(StaticFiles):
    """Static assets must always revalidate — a stale cached app.js made
    every button/comet look broken after UI fixes."""
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp

app.mount("/static", NoCacheStaticFiles(directory=os.path.join(BASE_DIR, "ui", "static")), name="static")


# ---------------- client-IP view gate ----------------
# access.py decision, wired here: the app stays fully usable from every
# IP — what changes per peer is VISIBILITY of `private: true` configs.
# Raw-ASGI wrapper (not BaseHTTPMiddleware) so one pass covers http +
# websocket scopes. Starlette runs each request in its own asyncio task,
# so the contextvar the wrapper flips is scoped to that request alone —
# concurrent requests and background tasks never see each other's gate.
import access

class IpGateMiddleware:
    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        access.restrict_if_foreign(scope)
        try:
            await self.app(scope, receive, send)
        finally:
            # request-local hygiene: in production each request is a
            # fresh task (contextvar dies with it), but a reset keeps
            # the gate clean even when callers drive scopes manually
            # (tests, in-process probes).
            access.set_view_allow(None)
            access.set_restricted(False)

app.add_middleware(IpGateMiddleware)


# ---------------- web logins (key identity via username/password) ----
# Secrets in the vault (.env): LOGIN_<NAME>_PASS = salted SHA-256 hash,
# LOGIN_<NAME>_KEY = Fernet-encrypted access-key token (AES-128-CBC +
# HMAC-256 under the hood — same machinery as machine SSH passwords),
# WEB_SESSION_<sid> = encrypted "{user}|{created}" records. Sessions
# live in the vault too, so logins SURVIVE daemon restarts. Cookie is
# HttpOnly, SameSite=Lax; the API key never needs to be pasted into a
# browser — logging in IS the key adoption.

LOGIN_COOKIE = "farm_session"


@app.get("/api/auth/me")
async def auth_me(request: Request, response: Response):
    """Identity report for the UI boot: full resolution already happened
    in the middleware (Bearer key OR session cookie → key). Reports the
    resolved key identity + whether a web-login session backs it (so
    the login chip shows the USER, not just the key name)."""
    # no-store: Firefox bfcache/normal cache must never serve a stale
    # identity report — a cached "unauthenticated" me = phantom login loop
    response.headers["Cache-Control"] = "no-store, max-age=0"
    import secrets_vault as _vault
    tok = access.request_key(request)
    sid = request.cookies.get(LOGIN_COOKIE) or ""
    user = access.session_user()
    name = access.key_display_name()
    admin = bool(access.is_admin(None))
    scoped = access.view_allow() is not None
    # trust_via must reflect the REAL peer: the loopback console gets
    # its label only when the socket peer actually IS loopback — an
    # unauthenticated remote request must NOT inherit the console label
    # (the UI's login-first gate keys off exactly this field — labeling
    # everything loopback silently disabled the login screen).
    peer = ((request.scope.get("client") or (None,))[0] or "")
    is_console = peer in access._TRUSTED_ANCHORS
    return {"name": name, "user": user, "admin": admin, "scoped": scoped,
            "models": sorted(access.view_allow() or []),
            "logged_in": bool(user) or bool(tok),
            "trust_via": "session-cookie" if user else
                         ("key" if tok else
                          ("loopback" if is_console else "unauthenticated")),
            "has_logins": bool(_vault.list_web_logins())}


@app.get("/api/auth/my-key")
async def auth_my_key(request: Request):
    """THE requester's own raw access key — for the User page's copy
    button (any logged-in user needs the exact token their session is
    using to configure API clients; the list endpoint deliberately
    only masks).

    Rules that make this safe:
    - never answers unless the request carries a resolvable identity
      (Bearer/X-Api-Key header or web-login session cookie). The token
      returned is ALWAYS the requester's own — `_identity_token`
      resolves only the caller's header/session, so there is no way to
      ask for anyone else's key. Console-only loopback callers have no
      key to copy — unauthenticated asks get a 404-shaped refusal.
    - no-store: the raw token must never sit in any cache."""
    import secrets_vault as _vault
    tok = _identity_token(request)
    if not tok:
        raise HTTPException(404, "not found")
    rec = access.key_record(tok)
    if not rec:
        raise HTTPException(404, "not found")
    return JSONResponse(
        status_code=200,
        content={"name": rec.get("name") or "", "key": tok},
        headers={"Cache-Control": "no-store, max-age=0"})


def _identity_token(request: Request) -> str:
    """Token for THIS request: Bearer/X-Api-Key header first, then the
    web-login session cookie (resolving to the login's mapped key)."""
    tok = access.request_key(request)
    if tok:
        return tok
    sid = request.cookies.get(LOGIN_COOKIE) or ""
    if not sid:
        return ""
    import secrets_vault as _vault
    user = _vault.get_web_session(sid)
    if not user:
        return ""
    rec = _vault.get_web_login(user)
    if rec and rec.get("key"):
        import access as _a
        # session maps to the login's access key — resolve like a key
        return _vault.decrypt_value(rec["key"]) if rec["key"].startswith(
            "gAAA") or True else rec["key"]
    return ""


def _resolve_identity(token: str):
    """token -> (key_name, admin?) by direct key match, else the
    session-resolved one (already a raw token)."""
    rec = access.key_record(token)
    if rec is not None:
        return rec.get("name"), rec.get("admin")
    return None, None


class LoginIn(pydantic.BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def auth_login(l: LoginIn, response: Response):
    """Web login: verify against the vault's salted hash, set a
    session cookie that maps to the login's access key — identity for
    the whole UI on the next request. Restart-safe by design."""
    import secrets_vault as _vault
    user = (l.username or "").strip()
    rec = _vault.get_web_login(user)
    if not rec or not _vault.verify_password(l.password or "", rec["hash"]):
        raise HTTPException(401, "invalid username or password")
    # transparent hash upgrade: legacy SHA-256 rows re-hash to PBKDF2 on
    # successful login — write ONLY the _PASS field (set_web_login with a
    # None key token would leave the mapping untouched, but touching only
    # the password entry is surgical and can never disturb KEY/ADMIN)
    import secrets_vault as _sv
    if not (rec["hash"] or "").startswith("pbkdf2$"):
        _vault.set_env_value(
            f"{_vault._login_env_key(user)}_PASS",
            _vault.hash_password(l.password))
    sid = secrets.token_urlsafe(24)
    _vault.set_web_session(sid, user)
    response.set_cookie(LOGIN_COOKIE, sid, max_age=30 * 24 * 3600,
                        httponly=True, samesite="lax",
                        path="/")
    return {"ok": True, "user": user}


@app.post("/api/auth/logout")
async def auth_logout(request: Request, response: Response):
    sid = request.cookies.get(LOGIN_COOKIE) or ""
    if sid:
        import secrets_vault as _vault
        _vault.delete_web_session(sid)
    response.delete_cookie(LOGIN_COOKIE, path="/")
    return {"ok": True}


class ChangePasswordIn(pydantic.BaseModel):
    username: str = ""   # empty = own login (admin convenience)
    current: str = ""    # required when changing ANOTHER login's password
    new: str = ""


@app.post("/api/auth/change-password")
async def auth_change_password(p: ChangePasswordIn, request: Request):
    """Password change — ADMIN ONLY (operator decision): usernames,
    passwords, key mappings and roles are admin-handled, always. The
    admin may change ANY login's password (own included) by naming the
    target; the verifier is the admin's OWN session/key identity."""
    if not (access.is_admin(None) or (request.scope.get("client") or
                                      (None,))[0] in access._TRUSTED_ANCHORS):
        raise HTTPException(404, "not found")
    import secrets_vault as _vault
    target = (p.username or "").strip() or access.session_user()
    if not target:
        raise HTTPException(400, "no target user — pass username")
    if not p.new or len(p.new) < 8:
        raise HTTPException(400, "new password must be at least 8 characters")
    rec = _vault.get_web_login(target)
    if not rec:
        raise HTTPException(404, f"no login named '{target}'")
    # changing ANOTHER login requires the CURRENT password of that login
    # (verifies the admin isn't a hijacked session casually rotating
    # others; own password change needs own current password)
    if p.username and p.username != access.session_user():
        if not _vault.verify_password(p.current or "", rec["hash"]):
            raise HTTPException(401,
                                f"current password of '{target}' is wrong")
    _vault.set_env_value(
        f"{_vault._login_env_key(target)}_PASS",
        _vault.hash_password(p.new))
    return {"ok": True, "user": target}


# ---------------- user page (System tab · logins) ----------------
class LoginUpdateIn(pydantic.BaseModel):
    password: str = ""          # empty = keep current
    access_key_token: str = ""  # empty = keep current mapping
    admin: bool | None = None


def _require_console_or_admin():
    """User-page writes are control surfaces: loopback console or an
    admin key/session only (the middleware resolved identity ctx)."""
    if access.is_admin(None) or access.admin_sees_everything():
        return
    raise HTTPException(404, "not found")


# ---------------- usage telemetry (User page graphs) ----------------
@app.get("/api/usage")
async def usage_me(period: str = "all", count: int = 14,
                   who: str = "", request: Request = None):
    """FULL usage for THIS identity — every request ever made under the
    caller's key, bucketed for the User page:
      period=all               → grand totals (requests, hours, tokens)
      period=day|month|year    → last `count` buckets as a graph series
    Plus per-model totals + average decode tok/s (Σtok_out/Σdur).
    Strictly per-identity: each key reads ONLY its own usage file —
    no shared/console traffic is folded in (console rows belong to the
    console identity alone). Admins may ?who= another identity.
    Identity comes ONLY from the middleware context. No prompt/body
    text is ever stored — only tokens, statuses, durations."""
    import usage_collector as _uc
    import access as _a
    if who.strip():
        if not (_a.is_admin(None) or _a.admin_sees_everything()):
            raise HTTPException(403, "admins only")
        target = who.strip()
    else:
        target = _a.key_display_name() or _a.session_user() or "(none)"
    period = period if period in ("all", "day", "month", "year") else "all"
    count = max(1, min(int(count) or 14, 60))
    agg = _uc.aggregate(target, period=period, count=count)
    return {"identity": target, "period": period, "count": count,
            **agg}


@app.get("/api/usage/by-model")
async def usage_by_model(days: int = 30, who: str = "",
                         request: Request = None):
    """'Usage by model' tab feed: per-model daily series (30d window) +
    all-time totals + true avg speed + last-used. Strictly
    identity-scoped: admins may ?who=; everyone else sees ONLY their
    own rows — console traffic belongs to the console identity alone
    and is never folded into other users' cards."""
    import usage_collector as _uc
    import access as _a
    if who.strip():
        if not (_a.is_admin(None) or _a.admin_sees_everything()):
            raise HTTPException(403, "admins only")
        target = who.strip()
        return {"identity": target,
                **_uc.by_model_daily(target, days=max(1, min(int(days) or 30, 90)))}
    me = _a.key_display_name() or _a.session_user() or "(none)"
    days_n = max(1, min(int(days) or 30, 90))
    mine = _uc.by_model_daily(me, days=days_n) if me != "(none)" else {}
    return {"identity": me, **mine}


@app.get("/api/usage/tracked")
async def usage_tracked(request: Request = None):
    """Identities with usage history — admin/console surface for the
    user page's admin variant (everyone else: just themselves, only if
    they HAVE history)."""
    import usage_collector as _uc
    import access as _a
    all_names = _uc.key_files()
    me = _a.key_display_name() or _a.session_user()
    if _a.is_admin(None) or _a.admin_sees_everything():
        return {"tracked": all_names, "me": me or ""}
    return {"tracked": [me] if me in all_names else [],
            "me": me or ""}


@app.get("/api/users")
async def users_list(request: Request):
    """Login accounts for the User tab (admin/console only). Passwords
    NEVER leave the vault; key mappings masked."""
    _require_console_or_admin()
    import secrets_vault as _vault
    keys = access.access_keys()
    bytok = {}
    for tok, e in keys.items():
        bytok[e["name"]] = {"masked": _mask_token(tok), "admin": e["admin"]}
    me_user = access.session_user()
    out = []
    for name in _vault.list_web_logins():
        rec = _vault.get_web_login(name) or {}
        # resolve the login's KEY MAPPING by raw token -> key name
        # (the raw token NEVER leaves the vault)
        mapped = None
        if rec.get("key"):
            mapped = _vault.decrypt_value(rec["key"])
        krec = access.key_record(mapped) if mapped else None
        out.append({
            "name": name,
            "admin": bool(rec.get("admin")),
            "key": (krec or {}).get("name") or None,
            "unrestricted": bool((krec or {}).get(
                "unrestricted", (krec or {}).get("admin_unrestricted", True))),
        })
    return {"users": out, "me": me_user}


class UserUpsertIn(pydantic.BaseModel):
    username: str
    password: str = ""           # required when creating
    access_key_name: str = ""    # map by KEY NAME (not raw token)
    admin: bool = False


@app.post("/api/users")
async def users_upsert(u: UserUpsertIn, request: Request):
    """Create or update a login. Mapping by key NAME — the raw token
    never appears in any request or response."""
    _require_console_or_admin()
    import secrets_vault as _vault
    name = (u.username or "").strip()
    if not name:
        raise HTTPException(400, "username required")
    rec = _vault.get_web_login(name)
    pwd = u.password
    if not rec and not pwd:
        raise HTTPException(400, "new user needs a password")
    # resolve key NAME -> raw token (mapping only; token never echoed)
    tok = ""
    if u.access_key_name:
        keys = access.access_keys()
        match = [t for t, e in keys.items() if e["name"] == u.access_key_name]
        if not match:
            raise HTTPException(400,
                                f"no access key named '{u.access_key_name}'")
        tok = match[0]
    admin = (u.admin if u.admin is not None
             else bool((rec or {}).get("admin")))
    _vault.set_web_login(name, pwd or None, tok or None, admin)
    return {"ok": True, "user": name, "created": not rec}


class UserDeleteIn(pydantic.BaseModel):
    username: str


@app.post("/api/users/delete")
async def users_delete(u: UserDeleteIn, request: Request):
    """Remove a login (vault fields only — the access KEY stays; keys
    are managed on the Access page)."""
    _require_console_or_admin()
    import secrets_vault as _vault
    name = (u.username or "").strip()
    if not name:
        raise HTTPException(400, "username required")
    if name == access.session_user():
        raise HTTPException(400, "cannot delete your own session's login")
    env = _vault.ENV_PATH
    for suffix in ("_PASS", "_KEY", "_ADMIN"):
        _vault.delete_env_value(f"LOGIN_{name.upper().replace('-', '_')}"
                                f"{suffix}")
    return {"ok": True, "deleted": name}


class FirstLoginIn(pydantic.BaseModel):
    """Bootstrap: the FIRST admin login may create itself while no
    logins exist — after that, logins only via Access tab (admin)."""
    username: str
    password: str
    key_name: str = "console"


@app.post("/api/auth/bootstrap")
async def auth_bootstrap(l: FirstLoginIn, response: Response):
    import secrets_vault as _vault, access as _a
    if _vault.list_web_logins():
        # logins exist: this endpoint is admin/console-only from now on
        if not (access.gate_open() and access.view_allow() is None):
            raise HTTPException(404, "not found")
    name = (l.username or "").strip()
    if not name or not l.password:
        raise HTTPException(400, "username and password required")
    # admin key for the login (reuse existing by name or create)
    tok = access.generate_token()
    current = _load_settings_yaml_preserving(SETTINGS_FILE)
    keys = current.get(_a.KEYS_SETTINGS_KEY) or {}
    if not isinstance(keys, dict):
        keys = {}
    keys[tok] = {"name": (l.key_name or name).strip(), "models": [],
                 "admin": True, "admin_unrestricted": True}
    current[_a.KEYS_SETTINGS_KEY] = keys
    _dump_settings_yaml_preserving(current, SETTINGS_FILE)
    _vault.set_web_login(name, l.password, tok, admin=True)
    sid = secrets.token_urlsafe(24)
    _vault.set_web_session(sid, name)
    response.set_cookie(LOGIN_COOKIE, sid, max_age=30 * 24 * 3600,
                        httponly=True, samesite="lax", path="/")
    return {"ok": True, "user": name, "key_created": True}


# ---------------- pages ----------------
import time as _time_mod
_DAEMON_BOOT = str(int(_time_mod.time()))  # per-boot unique

def _asset_version():
    """Cache buster: file mtimes + daemon boot time. The boot component
    guarantees a fresh URL after every daemon restart, so Firefox can
    never serve a stale cached app.js against a new index.html."""
    try:
        paths = [os.path.join(BASE_DIR, "ui", "static", f)
                 for f in ("app.js", "style.css")]
        mt = str(int(max(os.path.getmtime(p) for p in paths)))
    except OSError:
        mt = "0"
    return f"{mt}-{_DAEMON_BOOT}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Never cache the shell page: a stale cached index.html with mismatched
    # element IDs used to silently break every button below the mismatch.
    # (app.js itself is versioned via ?v=mtime.)
    resp = templates.TemplateResponse(
        request, "index.html", {"asset_v": _asset_version()})
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# ---------------- machines ----------------
class ExposeIn(pydantic.BaseModel):
    mode: str = "both"      # both | vram | ram
    gpus: list[str] = []    # empty = all GPUs


class MachineIn(pydantic.BaseModel):
    name: str
    ip: str
    username: str
    password: str = ""
    port: int = 22
    role: str = "peer"
    os: str = "linux"
    gpu_vendor: str = "nvidia"   # nvidia | amd | cpu
    rpc_port: int = 50052
    gpu: str = ""
    ram_total_gb: float | None = None
    llama_bin: str = ""
    expose: ExposeIn = ExposeIn()
    # RDMA transport for llama.cpp RPC on this machine. When False (default)
    # the rpc-server is started with GGML_RPC_NO_RDMA=1 so a wedged RoCE QP
    # can never stall tensor traffic. When True, rdma_ip must be set — it is
    # the RoCE NIC address that both the RPC client connects to AND the GID
    # matcher uses (the IP must exist in the NIC's GID table for
    # auto-negotiation to engage).
    rdma: bool = False
    rdma_ip: str = ""
    # optional explicit pinning (discovered by machine detect); empty = let
    # llama.cpp auto-match the RoCE v2 GID from the socket's local address
    rdma_dev: str = ""
    rdma_gid: str = ""
    # rdma_cm: use rdma_cm QP path resolution (ConnectX-3/mlx4). Unset = upstream
    # raw ibv_modify_qp path. TCP always available via rdma=False.
    rdma_cm: bool = False
    # DEC-03 (AUDIT-T6 §1.6): per-machine swap-config overrides, same shape
    # as vllm_delivery — empty = inherit the global per-engine path
    # (swap.yaml config_path / config_path_vllm), non-empty = pin THIS
    # machine's llama-swap config file for that engine.
    swap_config_path: str = ""
    swap_config_path_vllm: str = ""


def _vllm_capability(machine):
    """HW-PARITY audit (t_a80c72a6 §3b) — full hardware capability gate for
    one machine record, shared by /api/machines status and list.

    Returns (capable, reason, info):
      capable  bool  — may select/use this host with the vLLM engine
      reason   str   — why not, verbatim for the UI tooltip ("" when capable)
      info     str   — caveat for CAPABLE hosts ("" mostly): shown as an
                       info note, never a grey-out.

    incapable (audit Section 3b): windows (vLLM needs WSL2), macOS (the farm
    never installs vLLM-Metal), intel/CPU vendors (llama.cpp SYCL/CPU only).
    capable: linux + nvidia. capable WITH caveat: linux + amd (ROCm wheels:
    py3.12 only, wheels.vllm.ai/rocm, glibc>=2.39 x86_64, HIP verified
    post-install) and linux with gpu_vendor unset (install gates at runtime
    on nvidia-smi/rocm-smi presence — not pre-greyed, just annotated).
    Vendor is normalized via gpu_vendor.normalize_vendor ('rocm'/'hip' ->
    'amd', 'sycl'/'xpu' -> 'intel', ...) so YAML spelling cannot dodge the
    gate. Derived response-only; never persisted to the machine store."""
    os_type = (machine.get("os") or "linux").strip().lower()
    vendor = gpu_vendor.normalize_vendor(machine.get("gpu_vendor"))
    if os_type == "windows":
        return False, "vLLM requires Linux (WSL2 for Windows hosts)", ""
    if os_type == "macos":
        return False, ("vLLM does not support macOS in the farm — vLLM-Metal "
                       "is a separate package the farm does not install; use "
                       "llama.cpp (Metal) or the MLX runtime"), ""
    if vendor == "intel":
        return False, ("vLLM does not support this host via the farm "
                       "installer — llama.cpp (SYCL) required"), ""
    if vendor == "cpu":
        return False, ("vLLM does not support CPU-only hosts — llama.cpp "
                       "(CPU build) required"), ""
    if os_type == "linux" and vendor in ("nvidia", "amd", ""):
        if vendor == "amd":
            return True, "", ("ROCm wheels: Python 3.12 only, from "
                              "wheels.vllm.ai/rocm (0.30.0 ships rocm723, "
                              "glibc >= 2.39, x86_64). Installer verifies "
                              "the HIP backend after install.")
        if vendor == "":
            return True, "", ("gpu_vendor unknown — vLLM install requires "
                              "nvidia-smi or rocm-smi on the host; install "
                              "will gate at runtime.")
        return True, "", ""
    # unreachable per the branches above; conservative default
    return False, "vLLM requires Linux with an nvidia or amd GPU", ""


@app.get("/api/machines")
async def list_machines():
    """Return machine list with full hardware vllm capability on each
    entry (HW-PARITY audit §3b): vllm_capable + vllm_capability_reason +
    vllm_capability_info. Incapable hosts (windows/macos/intel/cpu) get
    false + the reason string the UI shows in its info tooltip; ROCm and
    unknown-vendor linux hosts stay capable but carry an info caveat.
    (ARD §5 — research pitfall.) Derived from the stored os + gpu_vendor
    fields only; records are copied so neither the flags nor resolved
    secrets leak into the machine store/YAML."""
    # never leak resolved passwords to the browser — in-memory only
    out = []
    for m in cfg.machines.all():
        m = dict(m)  # copy: the flags are response-only
        capable, reason, info = _vllm_capability(m)
        m["vllm_capable"] = capable
        m["vllm_capability_reason"] = reason
        m["vllm_capability_info"] = info
        m.pop("password", None)
        out.append(m)
    return out


def _ssh_port_guard(machine):
    """Fail fast with an actionable error when a new (non-local) machine
    refuses SSH — typically a fresh host with the SSH port closed.
    Returns None when fine or when the check doesn't apply (local node)."""
    try:
        if local_node.is_local(machine):
            return None
    except Exception:
        pass
    from ssh_manager import SSHManager
    ssh = SSHManager(
        machine.get("ip", ""), machine.get("username", ""),
        machine.get("password", ""), port=machine.get("port", 22),
    )
    return ssh.ssh_port_error(timeout=4)


class FingerprintIn(pydantic.BaseModel):
    ip: str
    port: int = 22
    username: str = "zazi"
    os: str = "linux"


@app.post("/api/machines/fingerprint")
async def machine_fingerprint(p: FingerprintIn):
    """SSH host-key fingerprint WITHOUT login (key exchange is pre-auth).
    Used by the add-machine wizard so first-time hosts are shown + trusted
    on first use; returns whether we already know this host."""
    import ssh_manager
    try:
        fp = await run_in_threadpool(
            ssh_manager.host_fingerprint, p.ip, p.port)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    known = ssh_manager.load_known_hosts().get(
        ssh_manager._host_id(p.ip, p.port))
    return {**fp, "ip": p.ip, "port": p.port,
            "known": bool(known),
            "stored_fingerprint": (known or {}).get("fingerprint")}


# DEC-03 (AUDIT-T6 §1.6): validated per-machine swap-config overrides —
# runs on the machine save path (add/update) so a bad value never lands
# in machines.yaml (same strictness as vllm_delivery).
def _validate_machine_swap_config(payload):
    for engine in ("llama_cpp", "vllm"):
        cfg.machine_swap_config_override(payload, engine)


@app.post("/api/machines")
async def add_machine(m: MachineIn):
    try:
        payload = m.model_dump()
        if not payload.get("password"):
            # fresh add with key-auth (no password): nothing to fill
            pass
        _validate_machine_swap_config(payload)
        detail = _ssh_port_guard(payload)
        if detail:
            raise HTTPException(status_code=400, detail=detail)
        saved = cfg.machines.add(payload)
        saved = dict(saved)
        saved.pop("password", None)
        return saved
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/machines/{name}")
async def update_machine(name: str, m: MachineIn):
    try:
        payload = m.model_dump()
        if not payload.get("password"):
            # blank = keep vault password: resolve it for the pre-save
            # SSH guard so edits don't falsely fail auth
            try:
                import secrets_vault as _vault
                existing = cfg.machines.get(name) or {}
                payload["password"] = _vault.get_machine_password(existing)
            except Exception:
                pass
        _validate_machine_swap_config(payload)
        detail = _ssh_port_guard(payload)
        if detail:
            raise HTTPException(status_code=400, detail=detail)
        saved = cfg.machines.update(name, m.model_dump())
        saved = dict(saved)
        saved.pop("password", None)
        return saved
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/machines/{name}")
async def delete_machine(name: str):
    try:
        import secrets_vault as _vault
        target = cfg.machines.get(name) or {}
        _vault.delete_env_value(_vault.machine_env_key(target.get("id") or name))
    except Exception:
        pass
    return {"remaining": cfg.machines.delete(name)}


@app.post("/api/machines/{name}/detect")
async def detect_hardware(name: str):
    """Discover GPUs/cores/RAM on a machine and persist into its config,
    so expose defaults to everything available with no manual entry."""
    machine = cfg.machines.require(name)
    stats = await run_in_threadpool(stats_collector.machine_stats, machine)
    if stats["status"] != "online":
        detail = _ssh_port_guard(machine)
        raise HTTPException(
            status_code=503,
            detail=detail or f"{name} is offline (SSH port open but "
                             f"login/command failed — check credentials)")
    gpus = [gpu_vendor.gpu_token(machine, g["index"]) for g in stats["gpus"]]
    updates = {
        "gpu": ",".join(gpus),
        "ram_total_gb": math.floor(stats["ram_total_gb"]) if stats["ram_total_gb"] else None,
        "detected": {
            "gpu_names": [g["name"] for g in stats["gpus"]],
            "compute_cap": [g.get("compute_cap") for g in stats["gpus"]],
            "gpu_uuids": [g.get("uuid") for g in stats["gpus"]],
            "vram_total_gb": round(sum(g["mem_total_gb"] for g in stats["gpus"]), 1),
            "cores": stats["cores"],
            "ram_total_gb": math.floor(stats["ram_total_gb"]),
        },
    }
    # RDMA auto-detect: when the machine has rdma=True, discover the RoCE NIC
    # (device name + GID index + IP) via ibv_devinfo + the sysfs GID table,
    # and persist them so the user doesn't have to type them.
    is_local = False
    try:
        is_local = local_node.is_local(machine)
    except Exception:
        pass
    rdma_info = {}
    if machine.get("rdma"):
        rdma_info = await run_in_threadpool(_detect_rdma_nic, machine, is_local)
        if rdma_info:
            updates["rdma_dev"] = rdma_info.get("dev", "")
            updates["rdma_gid"] = rdma_info.get("gid", "")
            if rdma_info.get("ip"):
                updates["rdma_ip"] = rdma_info["ip"]
    updates = {k: v for k, v in updates.items() if v is not None}
    result = cfg.machines.update(name, updates)
    result["synced_models"] = cfg.sync_models_to_hardware()
    if rdma_info:
        result["rdma_detected"] = rdma_info
    return result


def _detect_rdma_nic(machine, is_local):
    """Discover the machine's RoCE NIC: active ibverbs device, its active
    port, the RoCE v2 GID index on that port that maps to a local IP, and
    that IP. Returns {} when no RDMA device is found.
    ONE bundled remote command (single SSH round trip) — a shell script
    scans all devices/ports/GIDs locally and emits "dev|port|gid|ip"."""
    remote_script = (
        'for dev in $(ibv_devices 2>/dev/null | awk "NR>2{print \\$1}"); do '
        'for port in 1 2; do '
        'st=$(cat /sys/class/infiniband/$dev/ports/$port/state 2>/dev/null); '
        '[[ "$st" != *ACTIVE* ]] && continue; '
        'gids=/sys/class/infiniband/$dev/ports/$port/gids; '
        'for gf in $(ls $gids 2>/dev/null); do '
        'gid=$(cat $gids/$gf 2>/dev/null); '
        'gt=$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/types/$gf 2>/dev/null); '
        'nd=$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/ndevs/$gf 2>/dev/null); '
        'norm=$(echo "$gid" | tr -d :); '
        '[[ ${#norm} -ne 32 ]] && continue; [[ "$gt" != *v2* ]] && continue; '
        '[[ "${norm:20:4}" != ffff ]] && continue; '
        'ip=$((16#${norm:24:2})).$((16#${norm:26:2})).$((16#${norm:28:2})).$((16#${norm:30:2})); '
        'echo "$dev|$port|$gf|$ip|$nd"; '
        'done; done; done'
    )
    if is_local:
        import subprocess
        try:
            out = subprocess.run(["bash", "-c", remote_script], capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            # no bash on Windows (RDMA tooling is Linux-only anyway)
            return {}
    else:
        try:
            from ssh_manager import SSHManager
            ssh = SSHManager(machine["ip"], machine["username"], machine.get("password", ""), port=machine.get("port", 22))
            code, out, _ = ssh.run(remote_script, timeout=20)
        except Exception:
            return {}
    for line in (out or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        dev, port, gid, ip = parts[0], parts[1], parts[2], parts[3]
        if ip.startswith("10.0.0.") or (len(parts) > 4 and parts[4]):
            return {"dev": dev, "port": port, "gid": gid, "ip": ip}
    return {}


class ProbeIn(pydantic.BaseModel):
    ip: str
    username: str = "zazi"
    password: str = ""
    port: int = 22
    os: str = "linux"
    gpu_vendor: str = "nvidia"
    rpc_port: int = 50052


@app.post("/api/machines/probe")
async def probe_hardware(p: ProbeIn):
    """Detect hardware from raw form inputs -- works for machines that
    aren't saved yet. Returns gpu list + RAM + detected block; persists
    nothing."""
    machine = {"name": p.ip, **p.model_dump()}
    # Explicit SSH-port check first so a fresh host with port 22 closed
    # gets "SSH rejected: port 22 not open ..." instead of "unreachable".
    detail = _ssh_port_guard(machine)
    if detail:
        raise HTTPException(status_code=503, detail=detail)

    def run():
        s = stats_collector.machine_stats(machine)
        if s["status"] != "online":
            return {"online": False}
        return {
            "online": True,
            "gpu": ",".join(gpu_vendor.gpu_token(machine, g["index"]) for g in s["gpus"]),
            "ram_total_gb": float(math.floor(s["ram_total_gb"])) if s["ram_total_gb"] else None,
            "cores": s["cores"],
            "detected": {
                "gpu_names": [g["name"] for g in s["gpus"]],
                "compute_cap": [g.get("compute_cap") for g in s["gpus"]],
                "gpu_uuids": [g.get("uuid") for g in s["gpus"]],
                "vram_total_gb": round(sum(g["mem_total_gb"] for g in s["gpus"]), 1),
                "cores": s["cores"],
                "ram_total_gb": math.floor(s["ram_total_gb"]),
            },
        }
    result = await run_in_threadpool(run)
    if not result["online"]:
        raise HTTPException(
            status_code=503,
            detail=f"{p.ip} is unreachable over SSH (port {p.port} open but "
                   f"login/command failed) — check username/password and that "
                   f"'echo ok' works via plain ssh")
    result["detected"] = result.get("detected") or {}
    return result


class DiscoverIn(pydantic.BaseModel):
    subnet: str = ""          # empty = local /24
    port: int = 22
    username: str = ""        # empty = try creds from saved machines
    password: str = ""
    include_known: bool = False  # hide already-added machines by default
    ping: bool = True         # ICMP sweep: show live hosts without SSH too


@app.get("/api/discover/subnets")
async def discover_subnets():
    """Local /24 networks of this machine — for the discovery dropdown."""
    import discovery
    return {"subnets": discovery.local_subnets(),
            "default": discovery.local_subnet() or ""}


@app.post("/api/discover")
async def discover_hosts(p: DiscoverIn):
    """Scan a subnet: ICMP ping for live hosts, TCP sweep for open SSH
    port, banner-grab the OS pre-auth (no login needed). SSH login is
    only optional enrichment (hostname/GPUs). Persists nothing."""
    import discovery

    creds = [(p.username, p.password)] if p.username else None

    def run():
        return discovery.discover(
            subnet=p.subnet or None, port=p.port, credentials=creds,
            include_known=p.include_known, ping=p.ping)

    result = await run_in_threadpool(run)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/machines/detect-all")
async def detect_all():
    """Detect hardware on every online machine in parallel."""
    machines = cfg.machines.all()

    def detect(m):
        try:
            stats = stats_collector.machine_stats(m)
        except Exception:
            return None
        if stats["status"] != "online":
            return None
        gpus = [gpu_vendor.gpu_token(m, g["index"]) for g in stats["gpus"]]
        upd = {
            "gpu": ",".join(gpus),
            "detected": {
                "gpu_names": [g["name"] for g in stats["gpus"]],
            "compute_cap": [g.get("compute_cap") for g in stats["gpus"]],
                "gpu_uuids": [g.get("uuid") for g in stats["gpus"]],
                "vram_total_gb": round(sum(g["mem_total_gb"] for g in stats["gpus"]), 1),
                "cores": stats["cores"],
                "ram_total_gb": math.floor(stats["ram_total_gb"]),
            },
        }
        if stats["ram_total_gb"]:
            upd["ram_total_gb"] = math.floor(stats["ram_total_gb"])
        return m["name"], upd

    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        results = list(pool.map(detect, machines))
    updated = []
    for r in results:
        if r:
            cfg.machines.update(r[0], r[1])
            updated.append(r[0])
    synced = cfg.sync_models_to_hardware() if updated else []
    return {"updated": updated, "synced_models": synced}


# ---------------- background hardware watcher ----------------
# Polls every machine's full stats on a slow timer and persists CHANGES into
# machines.yaml (GPU count/names, VRAM, RAM). Swapped cards show up in the
# Machines page + visualizer without manual "Detect Hardware" clicks, and
# model configs get reconciled to the new slot layout automatically.
_hw_watcher_state = {"running": False, "last": {}}
HW_WATCH_INTERVAL_S = 120  # low frequency: SSH round-trip per machine


def _hw_fingerprint(machine):
    d = machine.get("detected") or {}
    return (
        ",".join(d.get("gpu_names") or []),
        d.get("vram_total_gb"),
        d.get("ram_total_gb"),
        d.get("cores"),
    )


def hw_watch_once():
    """One watcher pass: collect full stats from all online machines,
    persist any hardware delta into machines.yaml, sync model configs.
    Returns {"changed": [...], "synced_models": [...] }."""
    machines = cfg.machines.all()
    changed = []

    def probe(m):
        try:
            return m["name"], stats_collector.machine_stats(m)
        except Exception:
            return m["name"], None

    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        results = list(pool.map(probe, machines))
    for name, stats in results:
        if not stats or stats.get("status") != "online":
            continue
        try:
            upd = {
                "gpu": ",".join(gpu_vendor.gpu_token(
                    cfg.machines.require(name), g["index"])
                    for g in stats["gpus"]),
                "detected": {
                    "gpu_names": [g["name"] for g in stats["gpus"]],
            "compute_cap": [g.get("compute_cap") for g in stats["gpus"]],
                    "gpu_uuids": [g.get("uuid") for g in stats["gpus"]],
                    "vram_total_gb": round(sum(g["mem_total_gb"] for g in stats["gpus"]), 1),
                    "cores": stats["cores"],
                    "ram_total_gb": math.floor(stats["ram_total_gb"]),
                },
            }
            if stats["ram_total_gb"]:
                upd["ram_total_gb"] = math.floor(stats["ram_total_gb"])
        except (ValueError, KeyError, TypeError):
            continue
        cur = cfg.machines.get(name)
        if not cur:
            continue
        if _hw_fingerprint(cur) != _hw_fingerprint(upd):
            cfg.machines.update(name, upd)
            changed.append(name)
    synced = cfg.sync_models_to_hardware() if changed else []
    return {"changed": changed, "synced_models": synced}


def _hw_watcher_loop():
    while _hw_watcher_state["running"]:
        try:
            res = hw_watch_once()
            if res["changed"]:
                print(f"[hw-watcher] hardware changed on: {res['changed']}"
                      + (f" — synced {len(res['synced_models'])} model config(s)"
                         if res["synced_models"] else ""))
        except Exception as e:
            print(f"[hw-watcher] pass failed: {e}")
        time.sleep(HW_WATCH_INTERVAL_S)


@app.on_event("startup")
async def start_hw_watcher():
    if _hw_watcher_state["running"]:
        return
    _hw_watcher_state["running"] = True
    threading.Thread(target=_hw_watcher_loop, daemon=True,
                     name="hw-watcher").start()


@app.get("/api/hardware/changes")
async def hardware_changes():
    """Manual re-check right now (same logic as the background watcher).
    Returns which machines changed + how many model configs were synced."""
    return await run_in_threadpool(hw_watch_once)


@app.get("/api/machines/{name}/status")
async def machine_status(name: str):
    try:
        st = await run_in_threadpool(rpc_manager.machine_status, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # B5a2 + DEC-06: the serving signal is engine-owned. In swap-only
    # modes (vLLM) the adapter probes /health on the llama-swap listen
    # port and the payload is tagged so the UI renders it engine-aware.
    # llama.cpp mode returns the machine_status dict exactly as before.
    try:
        engine = engine_registry.get_engine()
    except ValueError:
        return st  # unreadable settings: keep the llama.cpp payload
    if not engine.capable("rpc_cluster"):
        try:
            machine = cfg.machines.require(name)
            port = swap_manager.load_settings().get("listen_port") or 8080
            st[engine.engine_id] = engine.probe(machine, port,
                                                local_node.is_local(machine))
        except Exception as e:
            st[engine.engine_id] = {"running": False,
                                    "detail": f"probe failed: {e}"}
        st["engine"] = engine.engine_id
    return st


@app.post("/api/machines/{name}/rpc/start")
async def start_rpc(name: str):
    # G-3 + DEC-06: starting ggml-rpc-server processes is an
    # rpc_cluster-capability concept — vLLM serves over HTTP from one
    # host, there is nothing to start. Explicit 409 (same shape as the
    # /api/launch vLLM refusal), never a silent no-op; the stop endpoints
    # stay reachable for cleanup.
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if not engine.capable("rpc_cluster"):
        raise HTTPException(
            status_code=409,
            detail="vLLM mode does not use RPC servers — start the model "
                   "through llama-swap instead (Launch All / /api/swap/*)")
    try:
        return await run_in_threadpool(rpc_manager.start_rpc, name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/machines/{name}/rpc/stop")
async def stop_rpc(name: str):
    try:
        return await run_in_threadpool(rpc_manager.stop_rpc, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/rpc/stop-all")
async def stop_all_rpc():
    """Kill leftover rpc-servers on every machine (both OS types)."""
    # B16 + DEC-06: rpc-server cleanup is rpc_cluster-capability only,
    # and both rpc-server START paths are already G-3 gated — under a
    # swap-only engine no peers ever spawn, so the only cleanup that
    # exists is the swap daemon itself (swap_manager.stop,
    # POST /api/swap/stop). Explicit 409 (same shape as /api/launch),
    # never a silent no-op; stale peers from a pre-vLLM llama.cpp session
    # are cleared by switching the engine back and running this once.
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if not engine.capable("rpc_cluster"):
        raise HTTPException(
            status_code=409,
            detail="vLLM mode does not use RPC clustering — serve models "
                   "through llama-swap (Launch All / /api/swap/*)")
    names = [m["name"] for m in cfg.machines.all()]
    with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
        results = list(pool.map(rpc_manager.stop_rpc, names))
    return results


@app.post("/api/start-all")
async def start_all():
    """Start everything: RPC servers on every machine, then the
    llama-swap daemon with saved settings. Always 200 — per-part
    failures live inside the payload so the UI can show details.
    G-3: under vLLM the RPC phase is skipped (llama.cpp-only concept —
    vLLM serves over HTTP from the swap host); the payload reports
    rpc: {skipped: true} so the UI shows an honest summary."""
    def run():
        try:
            engine = engine_registry.get_engine()
        except ValueError as e:
            raise ValueError(f"engine settings unreadable: {e}") from None
        if not engine.capable("rpc_cluster"):
            rpc_results = [{"skipped": True,
                            "reason": "vLLM mode: no RPC servers to start"}]
        else:
            names = [m["name"] for m in cfg.machines.all()]
            with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
                rpc_results = list(pool.map(rpc_manager.start_rpc, names))
        try:
            swap_report = swap_manager.launch(None, None, None, None)
        except ValueError as e:
            swap_report = {"failed": True, "error": str(e)}
        return {"rpc": rpc_results, "swap": swap_report}
    return await run_in_threadpool(run)


# ---------------- groups ----------------
class GroupIn(pydantic.BaseModel):
    name: str
    description: str = ""
    machines: list[str] = []


class SwapGroupIn(pydantic.BaseModel):
    name: str
    description: str = ""
    members: list[str] = []
    swap: bool = False
    exclusive: bool = False
    persistent: bool = False


def _active_engine_name():
    """DEC-17: the engine whose group world the API serves. Resolved once
    per request from the registry (settings.yaml `engine`); ValueError
    from a bad settings value propagates as a 500 — the settings PUT path
    already rejects bad engines, so it never happens in practice."""
    return engine_registry.get_engine().name


def _group_has_private_member(group):
    """access.py: a group whose membership includes a `private: true`
    model disappears for restricted viewers — one of its members reads
    as non-existent (`mid--variant` labels still reference their base
    id), so the whole group reads as never existing. Trusted IPs see
    every group."""
    if access.gate_open():
        return False
    for mem in (group.get("members") or []):
        if _raw_is_private(str(mem).split("--")[0]):
            return True
    return False


def _swap_group_write_guard(data, new_group, existing_name=None):
    """Serve-group writes from a restricted viewer: members must ALL be
    visible to them (no smuggling a foreign model into a group), and
    update/delete must target a group they can already see. Trusted
    users pass untouched. 404-shaped on every refusal (never a hint)."""
    if access.gate_open():
        return
    allow = access.view_allow()
    if not new_group:
        # visibility of the TARGET group under the same filter the
        # listing applies — invisible group = never existed
        for grp in _engine_groups(cfg.swap_groups):
            if grp.get("name") == existing_name:
                break
        else:
            raise HTTPException(404, "not found")
    if data is None:
        return
    members = data.get("members") or []
    for mem in members:
        token = str(mem)
        base_mid = token.split("--")[0]
        if _raw_is_private(base_mid):
            raise HTTPException(404, "not found")
        # EXACT token match — bare mid grants the base only; a variant
        # member requires its own mid--label grant
        if allow is not None and token not in allow:
            raise HTTPException(404, "not found")


def _raw_is_private(model_id):
    """Read the raw YAML flag for an unresolved (gate-hidden) member —
    get_model returns None under the gate, so the question 'is it hidden
    because private' needs the flag, not the config."""
    try:
        with open(os.path.join(cfg.MODELS_DIR, f"{model_id}.yaml"),
                  "r", encoding="utf-8") as f:
            import yaml as _yaml
            return bool((_yaml.safe_load(f) or {}).get("private"))
    except Exception:
        return False


def _engine_groups(collection):
    """DEC-17: only the ACTIVE engine's group entries. Missing `engine`
    (legacy YAML) reads as llama_cpp — same default the collection
    stamps on write, so un-migrated files keep rendering under
    llama_cpp and never appear under vllm. access.py: groups holding a
    private member additionally vanish in a restricted view."""
    engine = _active_engine_name()
    out = []
    for g in [g for g in collection.all()
              if (g.get("engine") or "llama_cpp") == engine]:
        if _group_has_private_member(g):
            continue
        out.append(g)
    return out


def _stamp_group_engine(item: dict) -> dict:
    """Migration (DEC-17): stamp the ACTIVE engine on a group being
    created/updated. New groups are born engine-qualified; an update
    re-stamps only when the stored engine is missing (legacy entry
    touched under vllm stays llama_cpp — hidden, never migrated away
    from its birth engine)."""
    item.setdefault("engine", _active_engine_name())
    return item


def _reject_cross_engine(collection, name: str):
    """DEC-17 safety net (unreachable from the UI — GET only lists the
    active engine's groups): mutating a group that belongs to the OTHER
    engine fails loudly instead of silently editing it. Raises
    HTTPException 409."""
    g = collection.get(name)
    if g is not None and (g.get("engine") or "llama_cpp") != _active_engine_name():
        raise HTTPException(
            status_code=409,
            detail=f"group '{name}' belongs to the "
                   f"{g.get('engine')} engine (active: "
                   f"{_active_engine_name()})")


@app.get("/api/groups")
async def list_groups():
    return _engine_groups(cfg.groups)


@app.post("/api/groups")
async def add_group(g: GroupIn):
    # DEC-17: new machine groups are stamped with the active engine.
    data = _stamp_group_engine(g.model_dump())
    try:
        return cfg.groups.add(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/groups/{name}")
async def update_group(name: str, g: GroupIn):
    _reject_cross_engine(cfg.groups, name)  # DEC-17 safety net
    # re-stamp only fills a MISSING engine (legacy entry) — an update
    # never moves a group between engines.
    data = _stamp_group_engine(g.model_dump())
    try:
        return cfg.groups.update(name, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/groups/{name}")
async def delete_group(name: str):
    _reject_cross_engine(cfg.groups, name)  # DEC-17 safety net
    return {"remaining": cfg.groups.delete(name)}


# ---------------- serve groups (llama-swap model co-serving) ----------------
@app.get("/api/swap-groups")
async def list_swap_groups():
    groups = _engine_groups(cfg.swap_groups)
    # access.py: restricted viewers see ONLY their own members — each
    # group's membership is filtered through the same gate the model
    # list uses (private flag + view allowlist), and groups that end up
    # with no visible members vanish entirely (a foreign serve group
    # reads as never existing, matching the model-table semantics).
    if access.gate_open():
        return groups
    allow = access.view_allow()
    visible = []
    for g in groups:
        members = []
        for mem in (g.get("members") or []):
            token = str(mem)
            base_mid = token.split("--")[0]
            if _raw_is_private(base_mid):
                continue
            # EXACT token match (see _swap_group_write_guard note)
            if allow is not None and token not in allow:
                continue
            members.append(mem)
        if not members:
            continue
        visible.append({**g, "members": members})
    return visible


@app.post("/api/swap-groups")
async def add_swap_group(g: SwapGroupIn):
    # DEC-17: new serve groups are stamped with the active engine (the
    # user's vllm groups born under vllm show up under vllm only).
    data = _stamp_group_engine(g.model_dump())
    _swap_group_write_guard(data, new_group=True)
    try:
        saved = cfg.swap_groups.add(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Serve-group truth: a deployment in 2+ swap:true groups only takes
    # effect in the first — warn loudly at save time instead of letting
    # the generator silently drop the loser (one group per model).
    return {"group": saved, "warnings": swap_manager.duplicate_swap_claims()}


@app.put("/api/swap-groups/{name}")
async def update_swap_group(name: str, g: SwapGroupIn):
    _reject_cross_engine(cfg.swap_groups, name)  # DEC-17 safety net
    data = _stamp_group_engine(g.model_dump())
    _swap_group_write_guard(data, new_group=False, existing_name=name)
    try:
        saved = cfg.swap_groups.update(name, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"group": saved, "warnings": swap_manager.duplicate_swap_claims()}


@app.delete("/api/swap-groups/{name}")
async def delete_swap_group(name: str):
    _reject_cross_engine(cfg.swap_groups, name)  # DEC-17 safety net
    if not access.gate_open():
        # restricted deletion must target a group they can fully see —
        # same visibility rule the listing shows them (404 otherwise)
        _swap_group_write_guard(None, new_group=False, existing_name=name)
    return {"remaining": cfg.swap_groups.delete(name)}


@app.get("/api/groups/{name}/status")
async def group_status(name: str):
    try:
        return await run_in_threadpool(rpc_manager.group_status, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------- models ----------------
class VariantIn(pydantic.BaseModel):
    label: str
    servers: list[str] | None = None
    group: str | None = None
    device_order: str | None = None
    device_order_policy: str | None = None
    tensor_split_mode: str | None = None
    tensor_split: str | None = None
    fit_margin: int | None = None
    fit_ram_allow: int | None = None
    llama_bin: str | None = None
    expose: dict | None = None
    flags: dict | None = None
    # A2: per-variant vllm: overrides — shallow dict merge over the
    # primary's section (same partial-override semantics as flags), so a
    # variant on a weak-GPU machine can carry its own gmu/cpu-offload.
    vllm: dict | None = None
    pinned: bool | None = None
    ttl: int | None = None


class ModelConfigIn(pydantic.BaseModel):
    model_id: str
    original_model_id: str = ""  # set when renaming: old file is removed after save
    name: str = ""
    base_path: str = ""
    tensor_split_mode: str = "auto"   # auto | fit | split (manual is legacy -> fit)
    tensor_split: str = ""            # explicit layer weights for split mode (e.g. 4,4,4,2,2,2); unused by auto/fit
    fit_margin: int = 1024           # MiB reserve per device when mode=fit
    fit_ram_allow: int = 1024      # MiB of each RAM pool fit may use; rest reserved
    llama_bin: str = "~/llama.cpp/build/bin/llama-server"
    servers: list[str] = []
    group: str = ""                  # optional group affinity — pins model to group's machines
    pinned: bool = False             # keep-warm: never auto-unload when pinned
    ttl: int = 0                     # auto-unload after ttl seconds idle (0 = swap default, pinned forces 0)
    device_order: str = ""
    device_order_policy: str = "as-selected"  # as-selected | vram-first
    expose: dict = {}
    flags: dict = {}
    vllm: dict | None = None  # per-model vllm: section (additive; None = absent)
    # custom runtime (external server, e.g. winbox .bat passthrough):
    # round-trips verbatim so custom models are editable/deletable saved
    # configs like any other model; validated below on save.
    runtime: str = "llama"
    custom: dict | None = None
    variants: list[VariantIn] = []
    reset: bool = False  # true = stop all RPC/llama processes + daemon before pushing, so the edit applies to a clean cluster


@app.get("/api/models")
async def list_saved_models(full: bool = False):
    # DEC-11 (T4-G6): the Models table lists only models the ACTIVE
    # engine serves — adapter filter (llama.cpp: GGUF; vLLM: HF dirs),
    # the same downloads.vocabulary gate /api/scan applies to sources.
    # full=1: the ACCESS-TAB grant grid needs the WHOLE catalog (base +
    # every variant) so the operator can grant anything — allowed only
    # for control identities (trusted console / admin key). Scoped
    # viewers get the view-filtered table (?full=1 silently ignored).
    if full and ((access.gate_open() and access.view_allow() is None)
                 or access.is_admin(None)):
        import os as _os
        import yaml as _yaml
        out = []
        for fname in sorted(_os.listdir(cfg.MODELS_DIR)):
            if not fname.endswith(".yaml"):
                continue
            if "__tune_tmp" in fname or "__ab_single" in fname:
                continue
            try:
                with open(_os.path.join(cfg.MODELS_DIR, fname), "r",
                          encoding="utf-8") as f:
                    d = _yaml.safe_load(f) or {}
            except Exception:
                continue
            d["_file"] = fname
            if "variants" not in d or d["variants"] is None:
                d["variants"] = []
            d["architecture"] = cfg._get_model_architecture(d.get("base_path"))
            out.append(d)
        return out
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return engine.filter_models(cfg.list_models())


@app.get("/api/rpc-map")
async def rpc_map_all():
    """RPC<N> -> {machine, resource} for every model, exactly as the
    generated scripts count devices (peer order = --rpc order; per peer:
    selected GPUs by index, then CPU iff RAM selected). The visualizer
    resolves sched-log RPC tokens through this instead of guessing."""
    # B16 + DEC-06: RPC numbering is an rpc_cluster concept — there is no
    # RPC<N> map under a swap-only engine (no sched-log RPC tokens exist
    # to resolve). Explicit 409 (same shape as /api/launch), never an
    # empty misleading map.
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if not engine.capable("rpc_cluster"):
        raise HTTPException(
            status_code=409,
            detail="vLLM mode does not use RPC clustering — serve models "
                   "through llama-swap (Launch All / /api/swap/*)")
    out = {}
    host = swap_manager.get_host()
    for m in cfg.list_models():
        mid = m.get("model_id")
        servers = m.get("servers") or []
        if not servers or not mid:
            continue
        try:
            master, peers = rpc_manager._split_servers(
                [host["name"]] + [s for s in servers if s != host["name"]],
                host["name"])
        except (ValueError, KeyError):
            continue
        # launch-mode parity with the swap plan (swap_manager
        # _resolve_launch_mode): a local-mode deployment runs llama-server
        # on ONE box with no RPC peers — no RPC<N> tokens exist in its
        # sched logs, so the visualizer must see an empty map, not the
        # peer-chaining map the old code invented.
        def _mode(deploy):
            return swap_manager._resolve_launch_mode(
                deploy, host["name"],
                [s for s in (deploy.get("servers") or servers)
                 if s != host["name"]])
        variants = {"": {} if _mode(m) == "local"
                    else rpc_manager.compute_rpc_map(m, master, peers)}
        for v in (m.get("variants") or []):
            label = (v.get("label") or "").strip()
            if not label:
                continue
            merged = rpc_manager._resolve_variant(m, label)
            try:
                vmaster, vpeers = rpc_manager._split_servers(
                    [host["name"]] + [s for s in (merged.get("servers") or servers)
                                      if s != host["name"]], host["name"])
                variants[label] = {} if _mode(merged) == "local" \
                    else rpc_manager.compute_rpc_map(merged, vmaster, vpeers)
            except (ValueError, KeyError):
                continue
        out[mid] = {"file": (m.get("base_path") or "").split("/").pop(),
                    "maps": variants}
    return out


@app.get("/api/models/icons")
async def model_icons_all():
    """Return all permanently-cached model icons (config/model_icons.json)."""
    return _load_model_icons()


@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return m


@app.post("/api/models")
async def save_model(m: ModelConfigIn, request: Request):
    mid = (m.model_id or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="model_id is required")
    # View auto-join: a restricted (view-scoped) user who just downloaded
    # and configured a NEW model keeps access to it — save_model files
    # the config, so this adds the fresh mid to their own view's
    # allowlist immediately (both wizard steps ride ONE request, so the
    # same request's contextvar gate identifies the viewer). Trusted
    # users are never view-scoped; nothing happens for them.
    import access as _access
    if not _access.gate_open() and not cfg.get_model(mid):
        ip = _access.request_ip(request)
        view = _access.view_for_ip(ip)
        if view is not None:
            view_name = _view_name_for_ip(ip)
            if view_name:
                current = _load_settings_yaml_preserving(SETTINGS_FILE)
                views = current.get(_access.VIEWS_SETTINGS_KEY) or {}
                entry = views.get(view_name) or {}
                models_list = [str(x) for x in (entry.get("models") or [])]
                if mid not in models_list:
                    models_list.append(mid)
                    entry["models"] = models_list
                    views[view_name] = entry
                    current[_access.VIEWS_SETTINGS_KEY] = views
                    _dump_settings_yaml_preserving(current, SETTINGS_FILE)
    # manual tensor-split removed: fit mode supersedes it (llama-server's
    # fitter assigns whole layers by live free memory at load). Legacy
    # manual payloads are coerced so old clients keep working.
    if m.tensor_split_mode == "manual":
        m.tensor_split_mode = "fit"
        m.tensor_split = ""
    # Variant-write protection (strict mode): the editor round-trips the
    # variant list it was SHOWN — and for a scoped key that list is
    # narrowed to their grants (config_loader._filtered_variants). Saving
    # it back verbatim would silently DELETE ungranted variants from the
    # config (observed shape: the zazi key editing dad's model wipes
    # --dad). Rule: a scoped (restricted) editor keeps on-disk variants
    # their token does not cover, posted entries replace granted ones.
    # Gate-open editors (operator console / unrestricted) keep the old
    # replace-everything semantics — a trusted user's empty variant list
    # means "remove all variants".
    disk_variants = []
    if not _access.gate_open() and not (m.original_model_id or "").strip():
        try:
            import yaml as _yaml
            with open(os.path.join(cfg.MODELS_DIR, f"{mid}.yaml"),
                      "r", encoding="utf-8") as _f:
                disk_variants = (_yaml.safe_load(_f) or {}).get(
                    "variants") or []
        except Exception:
            disk_variants = []
        allow = _access.view_allow()
        if allow is not None:
            posted = {((v.label or "")).strip() for v in (m.variants or [])}
            kept = [v for v in disk_variants
                    if ((v.get("label") or "").strip() not in posted
                        and f"{mid}--{(v.get('label') or '').strip()}"
                        not in allow)]
            m.variants = list(m.variants or []) + kept
    # optional group affinity: must exist if set
    if m.group:
        g = cfg.groups.get(m.group)
        if not g:
            raise HTTPException(status_code=400, detail=f"group '{m.group}' not found")
        # if servers explicitly set, they must be subset of group's machines (warn, not block — group may be intent)
        if m.servers:
            allowed = set(g.get("machines") or [])
            unknown = [s for s in m.servers if s not in allowed]
            if unknown:
                raise HTTPException(status_code=400, detail=f"servers {unknown} not in group '{m.group}' (group has {sorted(allowed)})")
    # validate variants: labels unique, non-empty, servers subset of group if group set
    if m.variants:
        seen = set()
        for v in m.variants:
            lab = (v.label or "").strip()
            if not lab:
                raise HTTPException(status_code=400, detail="variant label is required")
            if "/" in lab or "\\" in lab or lab in (".", ".."):
                raise HTTPException(status_code=400, detail=f"variant label '{lab}' contains invalid characters")
            if lab in seen:
                raise HTTPException(status_code=400, detail=f"duplicate variant label '{lab}'")
            seen.add(lab)
            if v.group:
                gv = cfg.groups.get(v.group)
                if not gv:
                    raise HTTPException(status_code=400, detail=f"variant '{lab}': group '{v.group}' not found")
                if v.servers:
                    allowed = set(gv.get("machines") or [])
                    unknown = [s for s in v.servers if s not in allowed]
                    if unknown:
                        raise HTTPException(status_code=400, detail=f"variant '{lab}': servers {unknown} not in group '{v.group}' (group has {sorted(allowed)})")
            # servers must exist (if any). DEC-10: the vllm variant modal
            # saves variants with NO servers selection (single-host engine
            # — the hardware step carries no machines), so v.servers may
            # be None; treat it as empty instead of crashing the save.
            valid_machines = {mm["name"] for mm in cfg.machines.all()}
            unknown_v = [s for s in (v.servers or []) if s not in valid_machines]
            if unknown_v:
                raise HTTPException(status_code=400, detail=f"variant '{lab}': unknown machine(s): {', '.join(unknown_v)}")
    try:
        data = m.model_dump()
        # runtime: custom (external server) — validate the spec now (400)
        # so a bad custom config can never be filed; the plan-time branch
        # re-validates and skips with a warning as second defense. A
        # custom model needs no GGUF: base_path stays empty by design.
        if (data.get("runtime") or "") == "custom":
            spec = data.get("custom") or {}
            swap_manager.validate_custom_spec(spec, where=f"model '{mid}'")
            if not data.get("custom"):
                data.pop("custom", None)
            # winbox-bat is single-Windows-box by construction — fail fast
            # at save (400) instead of skipping silently at plan time
            if (spec.get("kind") or "") == "winbox-bat":
                try:
                    host_name = swap_manager.load_settings().get("host")
                except Exception:
                    host_name = None
                peers = [s for s in (data.get("servers") or [])
                         if s != host_name]
                if len(peers) != 1:
                    raise ValueError(
                        f"model '{mid}': custom winbox-bat needs exactly one "
                        f"target box in servers (got {peers})")
                try:
                    box = cfg.machines.require(peers[0])
                except ValueError:
                    raise ValueError(
                        f"model '{mid}': unknown machine '{peers[0]}'")
                if (box.get("os", "linux")) != "windows":
                    raise ValueError(
                        f"model '{mid}': custom winbox-bat needs a windows "
                        f"box, '{peers[0]}' is {box.get('os', 'linux')}")
        elif data.get("custom") is None:
            data.pop("custom", None)
        if not data.get("runtime") or data.get("runtime") == "llama":
            data.pop("runtime", None)  # default stays unfiled (legacy files)
        orig = (data.pop("original_model_id", "") or "").strip()
        # engine-adapter validation (B5a2, ARD §4): in vllm mode the
        # adapter checks the PAYLOAD before any file write — errors
        # reject the save (400), warnings ride along in the response.
        # llama.cpp mode keeps the byte-identical save path: its
        # validate_model_config is disk-based (POST /api/models/{id}/test
        # validates the stored config against real hardware), so calling
        # it here would 400 every brand-new model ("not found" on disk)
        # and validate stale data on edits.
        try:
            engine = engine_registry.get_engine()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        # DEC-09: the B11c save-path auto-translation is retired per the
        # user directive (zero cross-engine warnings — separation, not
        # better warnings). Each engine saves its own vocabulary verbatim;
        # a NEW vllm model missing its `vllm:` section gets the
        # live-validated farm defaults (see vllm_new_model_defaults)
        # filled in here — the editor shows the same prefill, so the
        # user never starts from an empty card. An EDIT that clears the
        # section still fails loudly at launch, unchanged.
        validation_warnings = []
        if engine.name == "vllm":
            if not data.get("vllm") and not cfg.get_model(mid):
                data["vllm"] = vllm_manager.vllm_new_model_defaults()
            # G-10: the Engine×OS gate — vLLM refuses Windows/macOS hosts
            # (and ROCm hosts below Python 3.12). The model's deployment
            # targets explicit `servers`, else its group's machines, else
            # every machine; adapter errors reject the save (400) so a
            # model can never be filed with hosts the engine can't serve.
            hosts = (m.servers
                     or (cfg.groups.get(m.group) or {}).get("machines")
                     or [mm["name"] for mm in cfg.machines.all()])
            by_name = {mm["name"]: mm for mm in cfg.machines.all()}
            host_errors = engine.validate_hosts(
                [by_name[n] for n in hosts if n in by_name])
            if host_errors:
                raise HTTPException(
                    status_code=400,
                    detail="model config invalid: " + "; ".join(host_errors))
            issues = engine.validate_model_config(data)
            errors = [i["msg"] for i in issues if i.get("level") == "error"]
            validation_warnings = [i["msg"] for i in issues
                                   if i.get("level") == "warn"]
            if errors:
                raise HTTPException(
                    status_code=400,
                    detail="model config invalid: " + "; ".join(errors))
        saved = cfg.save_model(mid, data)
        if orig and orig != mid:
            # rename: remove the orphaned old file (plus its seed, if any)
            try:
                cfg.delete_model(orig)
                import tuner as _tuner
                import os as _os
                _sp = _tuner._seed_path(orig)
                if _os.path.exists(_sp):
                    _os.remove(_sp)
            except Exception:
                pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # optional hard-reset: user asked that a model edit resets everything.
    # Stop the daemon + every RPC/llama process cluster-wide BEFORE pushing,
    # so the regenerated config/scripts apply to a clean state (no stale
    # peers keeping the old device map alive). Then push + restart daemon.
    if m.reset:
        reset_result = {"stopped": [], "errors": []}
        try:
            def _stop_all(mm_name):
                try:
                    return {"machine": mm_name,
                            "r": rpc_manager.stop_rpc(mm_name)}
                except Exception as e:
                    return {"machine": mm_name, "error": str(e)}
            # daemon first, then all peers in parallel; idempotent + fast
            try:
                reset_result["stopped"].append(swap_manager.stop())
            except Exception as e:
                reset_result["errors"].append(f"swap stop: {e}")
            names = [mm["name"] for mm in cfg.machines.all()]
            with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
                reset_result["stopped"].extend(pool.map(_stop_all, names))
        except Exception as e:
            reset_result["errors"].append(str(e))
    # auto-push to swap so newly saved models/variants become routable
    # immediately: regenerate + write config only (no restart — the daemon
    # watches the config file and reloads routes live). The outcome rides
    # along in the response so the UI reports it instead of guessing.
    push_result = {"pushed": False, "models": [], "warnings": []}
    reloading = False
    reset_result = None
    try:
        st = swap_manager.status()
        if st.get("running"):
            try:
                pr = await run_in_threadpool(swap_manager.push_scripts)
                push_result = {"pushed": True, "models": pr.get("models", []),
                               "warnings": pr.get("warnings", [])}
                # stop + restart the daemon in the background (peers stay
                # up) so the new routes are live immediately; the UI polls
                # /api/swap/reload-status for the outcome
                import asyncio as _asyncio

                async def _auto_reload():
                    try:
                        await run_in_threadpool(swap_manager.reload_daemon_async)
                    except Exception as e:
                        print(f"[auto-push] reload failed after save {mid}: {e}")

                _asyncio.create_task(_auto_reload())
                reloading = True
            except Exception as e:
                push_result = {"pushed": False, "error": str(e)}
                print(f"[auto-push] push failed after save {mid}: {e}")
        else:
            push_result = {"pushed": False, "reason": "swap not running — Launch All to apply"}
    except Exception as e:
        push_result = {"pushed": False, "error": str(e)}
        print(f"[auto-push] status check failed: {e}")
    if isinstance(saved, dict):
        saved = {**saved, "_swap_push": push_result, "_swap_reload": reloading,
                 "_reset": reset_result if m.reset else None,
                 "_validation_warnings": validation_warnings}
    return saved


@app.post("/api/models/cleanup")
async def cleanup_models():
    """Delete any blank/invalid model configs (model_id empty or file is .yaml).
    Returns deleted filenames. Safe to call anytime."""
    import glob as _glob
    deleted = []
    errors = []
    # .yaml is the blank-id file; also catch any file where loaded model_id is empty
    for path in _glob.glob(os.path.join(cfg.MODELS_DIR, "*.yaml")):
        try:
            import yaml as _yaml
            with open(path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
            mid = (data.get("model_id") or "").strip()
            fname = os.path.basename(path)
            # blank file or mismatched id
            if not mid or fname == ".yaml" or fname == ".yml":
                os.remove(path)
                deleted.append(fname)
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")
    return {"deleted": deleted, "errors": errors, "remaining": len(cfg.list_models())}


def _file_entry(path):
    exists = os.path.exists(path)
    return {"path": path, "exists": exists,
            "size_gb": round(os.path.getsize(path) / 1e9, 1) if exists else 0}


@app.delete("/api/models/scan-file")
async def delete_scanned_file(delete_file: bool = False, path: str = ""):
    """Delete a scanned artifact file (store:// form) from the store.
    delete_file=False previews (exists/size); the path must resolve
    inside a configured store so arbitrary paths are rejected."""
    if not path or not path.startswith(path_resolver.PREFIX):
        raise HTTPException(status_code=400, detail="store:// path required")
    local = path_resolver.resolve(path)
    store_name = path[len(path_resolver.PREFIX):].split("/")[0]
    root = path_resolver.resolve(f"store://{store_name}")
    norm_local = os.path.normpath(local)
    norm_root = os.path.normpath(root)
    if not norm_local.startswith(norm_root):
        raise HTTPException(status_code=400, detail="path outside store roots")

    files = await run_in_threadpool(
        lambda: {"files": [_file_entry(local)]})
    if not delete_file:
        return files
    target = files["files"][0]
    if not target["exists"]:
        return {"deleted": [], "errors": [f"{local}: not found"]}
    try:
        await run_in_threadpool(os.remove, local)
    except OSError as e:
        return {"deleted": [], "errors": [str(e)]}
    return {"deleted": [local], "errors": []}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    out = {"deleted": cfg.delete_model(model_id)}
    # stale seeds must not haunt a future model with the same id
    try:
        import tuner, os
        sp = tuner._seed_path(model_id)
        if os.path.exists(sp):
            os.remove(sp)
            out["seed_removed"] = True
    except Exception:
        pass
    return out


@app.post("/api/models/{model_id}/test")
async def test_model(model_id: str):
    """Instant static config check (no side effects). Live verification
    is covered by Tune, which loads and benches the real artifact."""
    # B16 + DEC-06: engine-routed validation. llama.cpp mode keeps the
    # disk-based rpc_manager.validate_model; vllm mode routes to the
    # engine adapter's config validator (vllm_manager.validate_model_config
    # — pure function of the stored model config, same issues shape).
    # Semantics match the B5a2 pre-save gate: errors -> 400 with the issue
    # messages, warnings ride along under ok:true. rpc_manager is NEVER
    # touched in vllm mode.
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if not engine.capable("rpc_cluster"):
        model = cfg.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404,
                                detail=f"model '{model_id}' not found")
        issues = engine.validate_model_config(model)
        errors = [i["msg"] for i in issues if i.get("level") == "error"]
        if errors:
            raise HTTPException(status_code=400,
                                detail="model config invalid: "
                                       + "; ".join(errors))
        return JSONResponse(
            {"model_id": model_id, "ok": True,
             "issues": [i for i in issues if i.get("level") == "warn"]})
    report = await run_in_threadpool(rpc_manager.validate_model, model_id)
    status = 200 if report.get("ok") else 500
    return JSONResponse(report, status_code=status)


# ---------------- auto-tune (math first, live-verify after) ----------------
def _refuse_tuner_in_vllm():
    """B16: the tuner is llama.cpp-only (VRAM/KV math over RPC device
    maps; its live-verify loads llama-server). ui_hints.hidden_panels
    already hides every tuner panel under vLLM — the API must refuse too,
    so a stale/hidden UI or a direct call can't run llama.cpp math.
    Explicit 409 (same shape as /api/launch), never a silent no-op."""
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if engine.name == "vllm":
        raise HTTPException(
            status_code=409,
            detail="Tuner is llama.cpp-specific (VRAM/KV math); "
                   "hidden in vLLM mode")


class TuneEstimateIn(pydantic.BaseModel):
    base_path: str = ""
    servers: list[str] = []
    device_order: str = ""
    cache_type: str = "q8_0"
    mmproj: str = ""


@app.post("/api/tune/estimate")
async def tune_estimate_form(t: TuneEstimateIn):
    """Math-only estimate for the model form (unsaved): hardware MUST be
    selected in the form (servers + device_order chips). 0 loads."""
    _refuse_tuner_in_vllm()
    import tuner
    try:
        master = None
        if t.servers:
            try:
                master = rpc_manager._split_servers(t.servers)[0]
            except ValueError:
                master = None
        target_os = (master or {}).get("os", "linux") if master else "linux"
        est = await run_in_threadpool(
            tuner.estimate, t.base_path, t.servers, t.device_order,
            target_os, t.cache_type, t.mmproj)
        return est
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models/{model_id}/tune/estimate")
async def tune_estimate_saved(model_id: str):
    _refuse_tuner_in_vllm()
    import tuner
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    try:
        master = rpc_manager._split_servers(m.get("servers") or [])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    flags = m.get("flags", {}) or {}
    try:
        return await run_in_threadpool(
            tuner.estimate, m.get("base_path") or m.get("path", ""),
            m.get("servers") or [], m.get("device_order") or "",
            master.get("os", "linux"),
            flags.get("cache-type-k") or "q8_0", flags.get("mmproj", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/models/{model_id}/tune/defaults")
async def tune_apply_defaults(model_id: str):
    """Apply safe defaults (q8 KV, MTP on iff file has draft tensors,
    vram-first fit) to a saved model. No loads."""
    _refuse_tuner_in_vllm()
    import tuner
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    if not m.get("servers"):
        raise HTTPException(status_code=400,
                            detail="no servers selected — pick hardware in the form first")
    try:
        master = rpc_manager._split_servers(m.get("servers"))[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    d = await run_in_threadpool(
        tuner.safe_defaults, m.get("base_path") or m.get("path", ""),
        m.get("servers"), m.get("device_order") or "",
        master.get("os", "linux"), m.get("flags", {}) or {})
    m["tensor_split_mode"] = d["tensor_split_mode"]
    m["fit_margin"] = d["fit_margin"]
    m["fit_ram_allow"] = d["fit_ram_allow"]
    m["device_order_policy"] = d["device_order_policy"]
    flags = m.get("flags", {}) or {}
    flags.update(d["flags"])
    m["flags"] = flags
    saved = cfg.save_model(model_id, m)
    return {"saved": saved["model_id"] if isinstance(saved, dict) else model_id,
            "defaults": d}


class TuneStartIn(pydantic.BaseModel):
    target_tps: float | None = None  # None = topology default (25 local, 20 RPC)
    try_mtp_off: bool = True
    port: int = 18399


@app.post("/api/models/{model_id}/tune/start")
async def tune_start(model_id: str, t: TuneStartIn):
    import tuner
    if "__tune_tmp" in model_id or "__ab_single" in model_id:
        raise HTTPException(
            status_code=400,
            detail=f"'{model_id}' is a tuner scratch config — save the form "
                   f"first (it auto-saves under the proper model id) then tune")
    if cfg.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found (did you delete it? re-save the form first)")
    _refuse_tuner_in_vllm()
    try:
        return await run_in_threadpool(
            tuner.start_tune, model_id, t.target_tps, t.port, t.try_mtp_off)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models/{model_id}/tune/status")
async def tune_status(model_id: str):
    _refuse_tuner_in_vllm()
    import tuner
    return tuner.job_status(model_id)


@app.post("/api/models/{model_id}/tune/cancel")
async def tune_cancel(model_id: str):
    _refuse_tuner_in_vllm()
    import tuner
    return await run_in_threadpool(tuner.cancel_tune, model_id)


@app.get("/api/models/{model_id}/tune/drift")
async def tune_drift(model_id: str):
    """Seed-vs-live config drift (the 5→24 incident class). No loads."""
    _refuse_tuner_in_vllm()
    import tuner
    return await run_in_threadpool(tuner.drift_check, model_id)


@app.post("/api/models/{model_id}/tune/seed")
async def tune_seed(model_id: str):
    """Freeze the current saved config as the known-good seed.
    Future tunes measure it first and must beat it to replace it."""
    _refuse_tuner_in_vllm()
    import tuner
    try:
        return await run_in_threadpool(tuner.snapshot_seed, model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class DeleteModelIn(pydantic.BaseModel):
    delete_file: bool = False


@app.delete("/api/models/{model_id}/file")
async def delete_model_file(model_id: str, d: DeleteModelIn):
    """Model file deletion: removes the artifact file(s) the model
    config points at (base_path gguf + mmproj/mtp companions) from the
    store, then removes the config itself. With delete_file=False it
    only previews what would be deleted."""
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")

    def collect():
        seen = set()
        raws = [m.get("base_path")]
        flags = m.get("flags") or {}
        raws += [flags.get(k) for k in ("mmproj", "mtp") if flags.get(k)]
        files = []
        for raw in raws:
            if not raw:
                continue
            path = path_resolver.resolve(raw)
            if path not in seen:
                seen.add(path)
                files.append(_file_entry(path))
        return files

    files = await run_in_threadpool(collect)
    if not d.delete_file:
        return {"files": files}

    def really_delete():
        deleted, errors = [], []
        for f in files:
            if not f["exists"]:
                continue
            try:
                os.remove(f["path"])
                deleted.append(f["path"])
            except OSError as e:
                errors.append(f"{f['path']}: {e}")
        if not errors:
            cfg.delete_model(model_id)
        return {"deleted": deleted, "errors": errors,
                "config_removed": not errors}
    return await run_in_threadpool(really_delete)


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


@app.get("/api/chat/models")
async def chat_models():
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
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{st['port']}/models",
                        timeout=10) as r:
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
        configured = engine.filter_models(cfg.list_models())
    except ValueError:
        configured = cfg.list_models()
    for m in configured:
        mid = m.get("model_id") or ""
        if not mid or "embed" in mid.lower() or mid in out:
            continue
        out.append(mid)
    # access.py: a `private: true` model is not chattable for restricted
    # viewers — drop it from the picker AND refuse a direct POST (same
    # "never existed" semantics as the launch path). The swap-running
    # list above may still carry the id when the daemon serves it for a
    # trusted user — the gate below re-checks before streaming. View-
    # scoped users are ALSO limited to their allowlist tokens (mid or
    # mid--variant), same filter the models table renders by.
    if not access.gate_open():
        allow = access.view_allow()
        out = [mid for mid in out
               if not _raw_is_private(mid.split("--")[0])]
        if allow is not None:
            # EXACT token match — variants are separate grants
            out = [mid for mid in out if mid in allow]
    return {"running": running,
            "models": [{"id": mid, "loaded": mid in loaded} for mid in out]}


@app.post("/api/chat")
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
    base = "http://127.0.0.1:8090"
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


# ---------------- flow visualizer (live llama-swap proxy log stream) ----------------
import re as _re

_VIS_LOG_RE = _re.compile(
    r'Request\s+(\S+)\s+"(\S+)\s+(\S+)\s+HTTP/\d\.\d"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+([\d.]+)(\w+)'
)


def _parse_proxy_line(line):
    """Parse a llama-swap proxy log line into a structured event dict, or None."""
    m = _VIS_LOG_RE.search(line)
    if not m:
        return None
    client, method, path, status, size, ua, dur, unit = m.groups()
    # normalize duration to seconds
    mult = {"µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}.get(unit.lower(), 1.0)
    return {
        "client": client,
        "method": method,
        "path": path,
        "status": int(status),
        "bytes": int(size),
        "ua": ua,
        "duration_s": round(float(dur) * mult, 6),
    }


_UPSTREAM_CHUNK_RE = _re.compile(
    r'http: streamed chunk: data: \{"choices":\[\{"finish_reason":null,.*?"model":"([^"]+)"'
)
# GGML_SCHED_DEBUG=2 node line: node #N (op): name (size) [backend cause] use=..,c=..: srcs...
# backend is %5.5s, cause is %8.8s (may be empty -> trailing spaces)
_SCHED_NODE_RE = _re.compile(
    r'node\s*#\s*\d+\s*\([^)]*\):\s*([^ ]+)\s*\([^)]*\)\s*\[([^ ]+)\s*([^ ]*)\]\s*use=\d+,c=\d+:(.*)'
)
# a src tensor in the tail: " name (size) [backend cause]"
_SCHED_SRC_RE = _re.compile(r'\s+([^ ]+)\s*\([^)]*\)\s*\[([^ ]+)\s*([^ ]*)\]')

# vLLM engine stats line (every VLLM_LOG_STATS_INTERVAL=10s, upstream log
# stream carries it because vLLM runs INSIDE llama-swap):
# "(APIServer pid=..) INFO .. [loggers.py:259] Engine 000: Avg prompt
# throughput: 2.3 tokens/s, Avg generation throughput: 0.8 tokens/s,
# Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache
# hit rate: 50.0%" — the visualizer's vLLM serving-activity signal (the
# llama.cpp "http: streamed chunk" token stream has no vLLM equivalent).
_VLLM_STATS_RE = _re.compile(
    r'Engine\s+\d+:\s+Avg prompt throughput:\s*([\d.]+) tokens/s,'
    r'\s*Avg generation throughput:\s*([\d.]+) tokens/s,'
    r'\s*Running:\s*(\d+) reqs,\s*Waiting:\s*(\d+) reqs,'
    r'\s*GPU KV cache usage:\s*([\d.]+)%,\s*Prefix cache hit rate:\s*([\d.]+)%'
)
# llama-swap tags model context in the same stream:
# "[INFO] <model-id> Health check passed ..." — binds the engine pid to a
# served model so stats pulses can attribute to the right node.
_VLLM_MODEL_TAG_RE = _re.compile(r'\[INFO\]\s*<([^>]+)>')


def _parse_sched_line(line):
    """Parse a GGML_SCHED_DEBUG=2 node line into a structured event, or None."""
    m = _SCHED_NODE_RE.search(line)
    if not m:
        return None
    name, backend, cause, tail = m.groups()
    srcs = []
    for sm in _SCHED_SRC_RE.finditer(tail):
        srcs.append({"name": sm.group(1), "backend": sm.group(2), "cause": sm.group(3)})
    return {"type": "sched", "tensor": name, "backend": backend,
            "cause": cause, "srcs": srcs}


def _parse_upstream_line(line):
    """Extract a flow event from one upstream log line.

    llama.cpp: the verbose "http: streamed chunk" line carries the model
    path per token chunk. vLLM: the 10s engine-stats line (throughput,
    KV usage, prefix-cache hit) — vLLM never logs per-chunk lines, so the
    stats line is its serving-activity signal. Both return {"model": id}
    shaped events; vLLM adds {"type":"vllm_stats", ...} so the frontend
    can label the source and drive the animation from real throughput.
    """
    m = _UPSTREAM_CHUNK_RE.search(line)
    if m:
        return {"model": m.group(1)}
    m = _VLLM_STATS_RE.search(line)
    if m:
        return {"type": "vllm_stats", "model": None,
                "prompt_tps": float(m.group(1)),
                "gen_tps": float(m.group(2)),
                "running": int(m.group(3)),
                "waiting": int(m.group(4)),
                "kv_pct": float(m.group(5)),
                "cache_hit_pct": float(m.group(6))}
    m = _VLLM_MODEL_TAG_RE.search(line)
    if m:
        # model context line (llama-swap "<id> ..." tag) — remember which
        # model the next stats line belongs to
        return {"type": "vllm_model_tag", "model": m.group(1)}
    return None


@app.websocket("/ws/flow/upstream")
async def flow_upstream_ws(websocket: WebSocket):
    """Tail llama-swap's upstream log stream and push real-time flow events
    to the browser: llama.cpp token-chunk events (model path) and vLLM
    10s engine-stats events (throughput / KV / cache). vLLM stats carry
    no model id — the "<id>" tag lines live on the PROXY stream, so a
    side tail of /logs/stream/proxy supplies the attribution (the
    frontend also falls back to the single loaded model)."""
    import httpx, asyncio
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    url = f"http://{host}:{port}/logs/stream/upstream"
    proxy_url = f"http://{host}:{port}/logs/stream/proxy"
    # Identity gate (access.py — the middleware resolved this ws scope's
    # Bearer key / session cookie exactly like a page request). A scoped
    # viewer must not watch token streams of models outside their grants:
    # every chunk event names the model path, which would leak ungranted
    # ids AND their live/loaded state. Gate-open viewers (trusted console,
    # unrestricted key) stream everything as before. Unresolvable
    # identities (anonymous ws on loopback never happens through the UI;
    # a scoped caller with allow None = unknown key) see nothing.
    _flow_allow = None if access.gate_open() else access.view_allow()
    if _flow_allow is None and not access.gate_open():
        await websocket.close(code=1008)
        return

    def _event_allowed(ev):
        """Scoped-key filter: model path → base mid (strip variant label —
        llama.cpp streams the plain base GGUF path for variants) must sit
        in the caller's allowlist with the container rule (_mid_allowed
        semantics — a granted mid--variant token lights its base's
        stream). No allow = open."""
        if _flow_allow is None:
            return True
        from swap_gateway import _mid_allowed
        raw = (ev.get("model") or "").strip()
        base = raw.split("/")[-1].split("--")[0]
        return bool(base) and _mid_allowed(base, _flow_allow)
    # llama-swap replays buffered history on connect — prime silently so a
    # page reload doesn't flare up "serving" state from old token chunks
    priming = True
    last_tag = None  # most recent "<model-id>" tag (from the proxy stream)
    async def tail_proxy_tags():
        """vLLM attribution: llama-swap logs "[INFO] <model-id> ..." on the
        proxy stream whenever a model starts/stops/health-checks. Stats
        lines on the upstream stream have no model id, so this task keeps
        `last_tag` fresh; the latest tag is overwhelmingly the serving
        model (swaps are exclusive in the vllm group)."""
        nonlocal last_tag
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", proxy_url) as resp:
                    async for line in resp.aiter_lines():
                        m = _VLLM_MODEL_TAG_RE.search(line)
                        if m:
                            last_tag = m.group(1)
        except Exception:
            pass
    tag_task = asyncio.create_task(tail_proxy_tags())
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as resp:
                async def unprime_later():
                    nonlocal priming
                    await asyncio.sleep(1.5)
                    priming = False
                primer = asyncio.create_task(unprime_later())
                try:
                    async for line in resp.aiter_lines():
                        ev = _parse_upstream_line(line)
                        if not ev:
                            continue
                        if ev.get("type") == "vllm_model_tag":
                            last_tag = ev.get("model")
                            continue
                        if ev.get("type") == "vllm_stats":
                            # stats carry no model id — bind them to the
                            # most recent <id> tag (proxy stream watcher)
                            ev["model"] = last_tag
                        if ev and not priming and _event_allowed(ev):
                            try:
                                await websocket.send_text(json.dumps(ev))
                            except Exception:
                                primer.cancel()
                                return
                finally:
                    primer.cancel()
    except Exception:
        pass
    tag_task.cancel()
    try:
        await websocket.close()
    except Exception:
        pass


@app.websocket("/ws/flow/sched")
async def flow_sched_ws(websocket: WebSocket):
    """Tail llama-swap's upstream (llama-server) log stream and push parsed
    GGML_SCHED_DEBUG=2 per-tensor backend-assignment events to the browser.
    Also tails every RPC peer's rpc-server.log so single-RPC models (e.g.
    winbox-only) show their real per-GPU work, not just the host's proxy."""
    import httpx, asyncio
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    # tail host upstream + every peer's rpc log in parallel
    peers = [m for m in cfg.machines.all() if not local_node.is_local(m)]

    async def tail_host():
        url = f"http://{host}:{port}/logs/stream/upstream"
        # llama-swap REPLAYS its buffered log history on connect — without
        # a priming pass, opening the Visualizer flares up every GPU with
        # hours-old activity. Consume silently first, only forward lines
        # that arrive after the stream settles.
        priming = True
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    async def unprime_later():
                        nonlocal priming
                        await asyncio.sleep(1.5)
                        priming = False
                    primer = asyncio.create_task(unprime_later())
                    try:
                        async for line in resp.aiter_lines():
                            ev = _parse_sched_line(line)
                            if ev and not priming:
                                try:
                                    await websocket.send_text(json.dumps(ev))
                                except Exception:
                                    primer.cancel()
                                    return
                    finally:
                        primer.cancel()
        except Exception:
            pass

    async def tail_peer(machine):
        import local_node as _ln
        is_win = machine.get("os", "linux") == "windows"
        is_local = _ln.is_local(machine)
        seen = set()
        # first poll returns the CURRENT log tail (hours-old lines) — seed
        # the dedup set with it silently so reload doesn't replay old
        # activity as fresh events
        seeded = False
        while True:
            try:
                log = await run_in_threadpool(
                    rpc_manager._tail_all_rpc_logs, machine, is_win, is_local)
                if not seeded:
                    seeded = True
                    seen.update((log or "").splitlines())
                    await asyncio.sleep(1)
                    continue
                for line in (log or "").splitlines():
                    if line in seen:
                        continue
                    seen.add(line)
                    # keep the set bounded (drop OLDEST lines only —
                    # seen.clear() replayed the whole tail every second,
                    # flooding the browser with duplicate events)
                    if len(seen) > 500:
                        seen = set(sorted(seen)[-500:])
                    ev = _parse_sched_line(line)
                    if ev:
                        try:
                            await websocket.send_text(json.dumps(ev))
                        except Exception:
                            return
            except Exception:
                pass
            await asyncio.sleep(1)

    try:
        await asyncio.gather(tail_host(), *[tail_peer(m) for m in peers])
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass


@app.websocket("/ws/flow")
async def flow_ws(websocket: WebSocket):
    """Tail llama-swap's proxy log stream (SSE over HTTP) and push parsed request
    events to the browser in real time. Identity gate (access.py): a
    scoped viewer must not watch other identities' request telemetry —
    client IPs, user agents, request shapes are household telemetry the
    same rule /api/swap/loaded applies to loaded lists."""
    import httpx
    if not access.gate_open() and access.view_allow() is None:
        await websocket.accept()
        await websocket.close(code=1008)
        return
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    url = f"http://{host}:{port}/logs/stream/proxy"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as resp:
                async for line in resp.aiter_lines():
                    ev = _parse_proxy_line(line)
                    if ev:
                        try:
                            await websocket.send_text(json.dumps(ev))
                        except Exception:
                            return
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
    try:
        await websocket.close()
    except Exception:
        pass


@app.websocket("/ws/logs/stream")
async def logs_stream_ws(websocket: WebSocket):
    """Logs tab: live raw tail of llama-swap's proxy + upstream log streams
    (SSE over HTTP), pushed to the browser as plain lines. Untouched text —
    the visualizer parses structured events from the same feeds; this tab
    shows them verbatim for real-time debugging."""
    # B15 verified: the source here is llama-swap's own /logs/stream/* on the
    # swap host:port (swap_manager.status) — vLLM runs INSIDE llama-swap, so
    # its proxy/upstream log lines ride the same streams. Engine-agnostic;
    # no change. (Per-model parse layers: flow's token-chunk regex keys on
    # "model":"<id>" which vLLM's OpenAI API also emits; sched events are
    # GGML_SCHED_DEBUG llama.cpp-only and simply never match under vLLM.)
    import httpx, asyncio
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    sent = 0
    dead = asyncio.Event()

    async def stream_one(kind, url):
        nonlocal sent
        # llama-swap replays buffered history on connect — consume the
        # first burst silently (like the visualizer's priming) so opening
        # the tab doesn't dump hours of stale lines; only forward lines
        # that arrive after the stream settles.
        priming = True
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    async def unprime_later():
                        nonlocal priming
                        await asyncio.sleep(1.5)
                        priming = False
                    primer = asyncio.create_task(unprime_later())
                    try:
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if priming:
                                continue
                            try:
                                await websocket.send_text(json.dumps(
                                    {"src": kind, "line": line}))
                                sent += 1
                            except Exception:
                                dead.set()
                                return
                    finally:
                        primer.cancel()
        except Exception:
            return

    async def keepalive():
        # periodic ping keeps the ws from idling out on silent proxies and
        # gives the browser a liveness signal for the status dot
        while not dead.is_set():
            try:
                await websocket.send_text(json.dumps({"ping": True}))
            except Exception:
                return
            await asyncio.sleep(10)

    try:
        await asyncio.gather(
            stream_one("proxy", f"http://{host}:{port}/logs/stream/proxy"),
            stream_one("upstream", f"http://{host}:{port}/logs/stream/upstream"),
            keepalive())
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass


# ---------------- hf downloads ----------------

class DownloadIn(pydantic.BaseModel):
    link: str
    store: str = "llm-models"
    all_shards: bool = True  # split model (-00001-of-00002): fetch every shard
    repo: bool = False  # whole safetensors repo (vLLM format): link is hf://owner/name


@app.get("/api/downloads")
async def list_downloads():
    return hf_downloader.all_jobs()


@app.get("/api/downloads/stores")
async def download_stores():
    return hf_downloader.list_stores()


@app.post("/api/downloads")
async def start_download(d: DownloadIn):
    # Engine-gate (DEC-07, T3 F4): refuse out-of-vocabulary formats
    # BEFORE dispatch. The UI's manifest downloads.formats already keeps
    # the right pane visible, so this 409 is a safety net for direct API
    # calls — same refusal pattern as /api/launch in vLLM mode.
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


@app.post("/api/downloads/{job_id}/cancel")
async def cancel_download(job_id: str):
    return hf_downloader.cancel(job_id)


# ---------------- hf browse (GGUF search, proxy to Hugging Face) ----------------
import hf_browse  # local helper

@app.get("/api/hf/search")
async def hf_search(
    query: str = "",
    sort: str = "downloads",
    limit: int = 20,
    page: int = 0,
    quant: str = "",
    pipeline_tag: str = "",
    author: str = "",
    recommended: bool = False,
    fmt: str = "gguf",
):
    # Engine-gate (DEC-07): the format chips are a UI convenience; the
    # server refuses searches for a format the ACTIVE engine cannot
    # download/serve (manifest downloads.formats) so the panes never
    # offer models that would 409 at POST /api/downloads.
    try:
        engine = engine_registry.get_engine()
    except Exception:
        engine = None
    if engine is not None:
        allowed = engine.ui_manifest().get("downloads", {}).get("formats", [])
        # UI chip vocabulary ("safetensors") vs manifest format key ("hf")
        want = {"safetensors": "hf"}.get((fmt or "gguf").lower(),
                                         (fmt or "gguf").lower())
        if want not in allowed:
            raise HTTPException(
                status_code=409,
                detail=f"the {engine.name} engine does not browse "
                       f"'{want}' models (allowed: {', '.join(allowed) or 'none'})")
    try:
        if (fmt or "gguf").lower() == "safetensors":
            return await hf_browse.search_safetensors(
                query=query, sort=sort, limit=limit, page=page,
                pipeline_tag=pipeline_tag, author=author,
                recommended=recommended
            )
        return await hf_browse.search_gguf(
            query=query, sort=sort, limit=limit, page=page,
            quant=quant, pipeline_tag=pipeline_tag, author=author, recommended=recommended
        )
    except Exception as e:
        # surface HF errors as 502 so UI can show hint
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/hf/models/{author}/{name}/gguf")
async def hf_gguf_files(author: str, name: str):
    try:
        return await hf_browse.get_gguf_files(author, name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/hf/models/{author}/{name}/safetensors")
async def hf_safetensors_files(author: str, name: str):
    """Detail for a source/safetensors repo: safetensors siblings + sizes."""
    try:
        return await hf_browse.get_safetensors_files(author, name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/hf/access/{author}/{name}")
async def hf_access(author: str, name: str, file: str = ""):
    """Live check: can the saved token download from this (possibly gated) repo?"""
    try:
        return await hf_browse.check_file_access(f"{author}/{name}", file)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/img-proxy")
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
MODEL_ICONS_FILE = os.path.join(BASE_DIR, "config", "model_icons.json")


def _sanitize_icon_url(v):
    """Coerce a cached icon value to a clean https URL or ''.

    Old cache entries were polluted by the HF avatar scraper
    (URL + '&quot;, ...' JSON leakage). Strip at the first HTML
    entity/quote/whitespace and reject non-http(s) values (e.g.
    dicts from the former double-wrapped _resolve_icon_url return).
    """
    if not isinstance(v, str):
        return ""
    v = v.replace("&amp;", "&").strip()
    for sep in ("&quot;", '"', "'", " ", "<", "\\n", "\\t"):
        if sep in v:
            v = v.split(sep)[0]
    v = v.strip()
    if not (v.startswith("https://") or v.startswith("http://") or v == ""):
        return ""
    return v


def _load_model_icons():
    if not os.path.exists(MODEL_ICONS_FILE):
        return {}
    try:
        with open(MODEL_ICONS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        return {}
    # sanitize on load so one polluted entry can't break the UI
    cleaned = {}
    dirty = False
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        sv = _sanitize_icon_url(v)
        cleaned[k] = sv
        if sv != v:
            dirty = True
    if dirty:
        try:
            _save_model_icons(cleaned)
        except Exception:
            pass
    return cleaned


def _save_model_icons(icons):
    try:
        tmp = MODEL_ICONS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(icons, f, indent=2)
        os.replace(tmp, MODEL_ICONS_FILE)
    except Exception:
        pass


@app.get("/api/models/{model_id}/icon")
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


async def _resolve_icon_url(icons, cache_key, base_path=""):
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
    _save_model_icons(icons)
    return url

@app.get("/api/hf/companies")
async def hf_companies(search: str = "", limit: int = 20, pipeline_tag: str = "", quant: str = "", recommended: bool = False):
    try:
        return await hf_browse.list_gguf_companies(limit=limit, search=search, pipeline_tag=pipeline_tag, quant=quant, recommended=recommended)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/hf/pipeline-tags")
async def hf_pipeline_tags(limit: int = 200):
    try:
        return await hf_browse.list_gguf_pipeline_tags(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/hf/collections")
async def hf_collections(owner: str, limit: int = 20, pipeline_tag: str = "", quant: str = ""):
    try:
        return await hf_browse.list_collections(owner=owner, limit=limit, pipeline_tag=pipeline_tag, quant=quant)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/hf/collection")
async def hf_collection(slug: str, gguf_only: bool = True):
    try:
        return await hf_browse.get_collection(slug, gguf_only=gguf_only)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------------- custom themes (user-built palettes) ----------------
# User themes live in config/custom_themes.json: {id: {"name": str, "tokens": {css-var: #hex}}}.
# The frontend applies them as inline custom properties over the base dark
# tokens, so no CSS regeneration is needed and previews apply instantly.

CUSTOM_THEMES_FILE = os.path.join(BASE_DIR, "config", "custom_themes.json")
CUSTOM_THEME_TOKENS = (
    "--bg", "--surface", "--raised", "--hover", "--inset",
    "--border", "--border-soft", "--border-strong",
    "--text", "--text-2", "--muted",
    "--accent", "--ribbon", "--ok", "--warn", "--bad",
)
# numeric (non-color) theme settings: ribbon FX multipliers
CUSTOM_THEME_NUMERIC = ("--ribbon-glow", "--ribbon-sat")


def _load_custom_themes():
    if not os.path.exists(CUSTOM_THEMES_FILE):
        return {}
    try:
        with open(CUSTOM_THEMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_custom_themes(themes):
    tmp = CUSTOM_THEMES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(themes, f, indent=2)
    os.replace(tmp, CUSTOM_THEMES_FILE)


@app.get("/api/themes/custom")
async def list_custom_themes():
    """All user-built themes (name + token map each)."""
    return {"themes": _load_custom_themes()}


class CustomThemeIn(pydantic.BaseModel):
    id: str = ""      # empty = derive from name
    name: str = "Custom"
    tokens: dict = {}


@app.post("/api/themes/custom")
async def save_custom_theme(t: CustomThemeIn):
    """Create or replace a user theme. Only known tokens + #rrggbb values
    are stored; anything else is rejected."""
    name = (t.name or "").strip()[:40] or "Custom"
    tid = (t.id or "").strip().lower()
    if not tid:
        tid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "custom"
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", tid):
        raise HTTPException(status_code=400, detail="theme id must match [a-z0-9-]")
    if not isinstance(t.tokens, dict):
        raise HTTPException(status_code=400, detail="tokens must be an object")
    clean = {}
    for k, v in t.tokens.items():
        if k in CUSTOM_THEME_NUMERIC:
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"token '{k}' must be a number")
            if not 0 <= f <= 2.5:
                raise HTTPException(status_code=400, detail=f"token '{k}' must be 0-2.5")
            clean[k] = str(f)
            continue
        if k not in CUSTOM_THEME_TOKENS:
            raise HTTPException(status_code=400, detail=f"unknown token '{k}'")
        if not isinstance(v, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            raise HTTPException(status_code=400, detail=f"token '{k}' must be #rrggbb")
        clean[k] = v.lower()
    if len(clean) < len(CUSTOM_THEME_TOKENS) + len(CUSTOM_THEME_NUMERIC):
        missing = [k for k in (*CUSTOM_THEME_TOKENS, *CUSTOM_THEME_NUMERIC) if k not in clean]
        raise HTTPException(status_code=400, detail=f"missing tokens: {', '.join(missing)}")
    themes = _load_custom_themes()
    themes[tid] = {"name": name, "tokens": clean}
    try:
        _save_custom_themes(themes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"id": tid, "name": name, "tokens": clean}


@app.delete("/api/themes/custom/{tid}")
async def delete_custom_theme(tid: str):
    themes = _load_custom_themes()
    if tid not in themes:
        raise HTTPException(status_code=404, detail=f"custom theme '{tid}' not found")
    del themes[tid]
    try:
        _save_custom_themes(themes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": tid, "remaining": list(themes)}


# ---------------- app settings ----------------

SETTINGS_FILE = os.path.join(BASE_DIR, "config", "settings.yaml")
ACCORDION_FILE = os.path.join(BASE_DIR, "config", "accordion_state.json")


def _load_accordion_state():
    if not os.path.exists(ACCORDION_FILE):
        return {}
    try:
        with open(ACCORDION_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_accordion_state(state):
    tmp = ACCORDION_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, ACCORDION_FILE)


FLOW_LAYOUT_FILE = os.path.join(BASE_DIR, "config", "flow_layout.json")


def _load_flow_layout():
    if not os.path.exists(FLOW_LAYOUT_FILE):
        return {}
    try:
        with open(FLOW_LAYOUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_flow_layout(layout):
    try:
        tmp = FLOW_LAYOUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)
        os.replace(tmp, FLOW_LAYOUT_FILE)
    except Exception:
        pass


@app.get("/api/flow/layout")
async def get_flow_layout():
    return _load_flow_layout()


class FlowLayoutIn(pydantic.BaseModel):
    layout: dict


@app.post("/api/flow/layout")
async def set_flow_layout(inp: FlowLayoutIn):
    _save_flow_layout(inp.layout)
    return {"ok": True}


@app.get("/api/accordion")
async def get_accordion_state():
    return _load_accordion_state()


class AccordionIn(pydantic.BaseModel):
    key: str
    open: bool


@app.post("/api/accordion")
async def set_accordion_state(a: AccordionIn):
    state = _load_accordion_state()
    state[a.key] = a.open
    _save_accordion_state(state)
    return {"ok": True, "key": a.key, "open": a.open}


# ---------------- settings.yaml persister (comment-preserving) --------------
# B9c (DEC-14): the round-trip pair now lives in settings_persister.py so
# vllm_installer / vllm_uninstaller can write settings.yaml through the
# SAME comment-preserving persister instead of plain yaml.safe_dump.

def _view_name_for_ip(ip: str):
    """Reverse lookup: which configured view (settings.yaml views:) an IP
    maps to, by ACCESS-TABLE name — used by the save-model auto-join (a
    restricted user configuring a new model keeps seeing it)."""
    ip = (ip or "").strip()
    for name, ent in _access_views_table().items():
        if ip in ent["ips"]:
            return name
    return None


def _access_views_table():
    import access as _a
    return _a._views_table()


def _load_settings_yaml_preserving(path):
    """settings.yaml as a mutable mapping, comments attached."""
    return settings_persister.load_yaml_preserving(path)


def _dump_settings_yaml_preserving(data, path):
    """Write settings.yaml back, comments intact. Atomic via tmp+replace."""
    settings_persister.dump_yaml_preserving(data, path)


@app.get("/api/settings")
async def get_settings():
    import yaml
    defaults = {"model_dir": "store://llm-models",
                "llama_bin": "~/llama.cpp/build/bin",
                "theme_family": "nord", "theme_mode": "dark"}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        defaults.update(data)
    # engine schema (ARD §2): inject engine default + full vllm: block so
    # the frontend never None-checks. A typo in settings.yaml (bad engine,
    # unknown vllm key) must NOT fall back silently — ValueError -> 400
    # with the valid values in the detail.
    try:
        engine_registry.cfg.inject_engine_defaults(defaults)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    # Identity-scoped settings: theme state is PER-KEY (access_keys ui:
    # block); engine/vllm/model_dir stay farm-wide and are admin-console
    # material. Scoped (non-admin) callers get their ui overlay + no
    # topology write surface (POST refuses engine from them).
    _key_ui = {}
    try:
        rec = access.current_key_record()
        if rec and isinstance(rec.get("ui"), dict):
            _key_ui = rec["ui"]
    except Exception:
        _key_ui = {}
    if _key_ui:
        defaults.update({k: v for k, v in _key_ui.items()
                         if k in ("theme_family", "theme_mode")})
    # never leak the raw token to the browser — expose only set-state.
    # Source of truth is the encrypted vault (.env); legacy YAML read for compat.
    token = str(defaults.pop("hf_token", "") or "")
    try:
        import secrets_vault as _vault
        defaults["hf_token_set"] = _vault.hf_token_set() or bool(token.strip())
    except Exception:
        defaults["hf_token_set"] = bool(token.strip())
    # UI field manifest (AUDIT-T6 §1.5, DEC-02) — the complete
    # engine-owned inventory the frontend renders from. The legacy
    # ui_hints view rides along DEPRECATED (its last JS consumer is the
    # flag-presets reader; Card 15 retires the key): ui_manifest() is
    # now the contract, app.js reads appSettings.ui_manifest.
    engine_adapter = engine_registry.get_engine(defaults)
    defaults["ui_manifest"] = engine_adapter.ui_manifest()
    # deprecated shim (retires with the ui_hints key in Card 15): derived
    # from the same manifest so the two views cannot drift
    defaults["ui_hints"] = engine_adapter.ui_hints()
    return defaults


class AppSettingsIn(pydantic.BaseModel):
    model_dir: str = "store://llm-models"
    llama_bin: str = "~/llama.cpp/build/bin"
    theme_family: str = "nord"
    theme_mode: str = "dark"
    hf_token: str | None = None  # only a non-empty value updates the stored token
    clear_hf_token: bool = False
    engine: str | None = None  # llama_cpp | vllm — None leaves the stored value
    vllm: dict | None = None  # partial block; merged over the stored one


async def _restart_everything_for_engine(new_engine: str):
    """Engine changed: restart llama-swap (engine-qualified config), kill all
    model servers, and restart this web app so every reader picks up the new
    engine. Same detached-restart machinery as POST /api/restart."""
    import subprocess as _sp
    import shlex as _shlex
    app_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(app_path))
    py = sys.executable
    # llama-swap settings: engine-qualified config path per DEC-03
    try:
        s = swap_manager.load_settings()
    except Exception:
        s = {}
    sw_port = s.get("listen_port") or 8080
    raw_cfg = s.get("config_path") or "~/.config/llama-swap/config.yaml"
    raw_cfg = os.path.expanduser(raw_cfg)
    cfg_engine = "llama_cpp" if new_engine == "llama_cpp" else "vllm"
    try:
        eff, _ = swap_manager.engine_swap_config_path(cfg_engine, raw_cfg)
    except Exception:
        eff = raw_cfg
    log = os.path.join(tempfile.gettempdir(), "llama-web.log")
    if os.name == "nt":
        ps = (
            "Start-Sleep 1; "
            "Get-Process llama-swap,llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; "
            "$c = Get-NetTCPConnection -LocalPort 8199 -State Listen -ErrorAction SilentlyContinue; "
            "if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }; "
            "Start-Sleep 1; "
            f"Start-Process -FilePath '{py}' -ArgumentList '\"{app_path}\"' "
            f"-WindowStyle Hidden -RedirectStandardOutput '{log}' -RedirectStandardError '{log}.err'"
        )
        _sp.Popen(["powershell", "-NoProfile", "-Command", ps],
                  start_new_session=True, close_fds=True)
    else:
        q = _shlex.quote
        # 1) Re-provision the daemon the SAME way the app does it: swap_manager
        #    .launch() regenerates the engine-qualified config (config.yaml vs
        #    config.vllm.yaml), rewrites the systemd unit's ExecStart, and
        #    restarts it — beating the unit's Restart=always properly instead
        #    of racing it with pkill+setsid (which lost the race: systemd
        #    resurrected the OLD-config daemon within 5s).
        # stop FIRST so the old daemon frees the configured port (the
        # launch path rolls to 8081+ when the port is still held by the
        # dying daemon — persisting the wrong port into swap.yaml)
        try:
            swap_manager.stop()
            import time as _tm
            _tm.sleep(2)
        except Exception as e:
            print(f"[engine-flip] swap stop failed: {e}", file=sys.stderr)
        # restore the configured port if a previous flip rolled it
        try:
            st = swap_manager.load_settings()
            st["listen_port"] = 8080
        except Exception:
            pass
        try:
            swap_manager.launch(None, 8080, None, None)
        except Exception as e:
            print(f"[engine-flip] swap relaunch failed: {e}", file=sys.stderr)
        # 2) Kill the old engine's model servers (they serve stale configs).
        # 3) Restart this web app detached so every reader picks up the new
        #    engine (same machinery as POST /api/restart).
        cmd = (
            "sleep 1; "
            "pkill llama-server 2>/dev/null || true; "
            "pkill -f 'vllm serve' 2>/dev/null || true; "
            "(fuser -k 8199/tcp > /dev/null 2>&1 || true); "
            "sleep 1; "
            f"setsid -f {q(py)} {q(app_path)} > {q(log)} 2>&1 "
            f"|| nohup {q(py)} {q(app_path)} > {q(log)} 2>&1 &"
        )
        _sp.Popen(["/bin/bash", "-c", cmd], start_new_session=True, close_fds=True)
    return {
        "status": "restarting",
        "message": f"Engine switched to {new_engine} — restarting llama-swap "
                   f"(config: {eff}) and the web app (~3s at http://127.0.0.1:8199)",
    }


@app.post("/api/settings")
async def save_settings(s: AppSettingsIn, r: Request):
    # Identity split: THEMES are per-key (any identity sets their own
    # ui); ENGINE/model_dir/llama_bin/vllm/hf_token are farm-wide state —
    # only trusted console or an ADMIN key may write those. A scoped
    # non-admin POST carrying engine/vllm/etc. is refused 404-shaped.
    _control_identity = (access.gate_open() and access.view_allow() is None) \
        or access.is_admin(None)
    _key_name_for_ui = access.key_display_name()
    if not _control_identity and (s.engine is not None or s.vllm is not None
                                  or s.hf_token or s.clear_hf_token):
        raise HTTPException(404, "not found")
    # merge over the CURRENT FILE so unknown/future keys survive the write;
    # loaded via the comment-preserving persister (B9c) — a plain
    # yaml.safe_load here is what stripped every comment on theme switches
    current = _load_settings_yaml_preserving(SETTINGS_FILE)
    engine_changed = False
    update = s.model_dump()
    tok = (update.pop("hf_token", None) or "").strip()
    clear_tok = update.pop("clear_hf_token", False)
    engine = update.pop("engine", None)
    vllm_in = update.pop("vllm", None)
    # per-key UI state: scoped identities write theme_family/theme_mode
    # into THEIR access_keys ui: block instead of the shared file —
    # farm-wide state untouched.
    if not _control_identity and _key_name_for_ui and \
            (update.get("theme_family") or update.get("theme_mode")):
        _want = {k: update[k] for k in ("theme_family", "theme_mode")
                 if update.get(k)}
        _keys = current.get(access.KEYS_SETTINGS_KEY)
        if not isinstance(_keys, dict):
            _keys = {}
        for _t, _e in _keys.items():
            if isinstance(_e, dict) and _e.get("name") == _key_name_for_ui:
                ui = _e.get("ui") or {}
                ui.update(_want)
                _e["ui"] = ui
                break
        current[access.KEYS_SETTINGS_KEY] = _keys
        update.pop("theme_family", None)
        update.pop("theme_mode", None)
    current.update(update)
    if engine is not None:
        # enum check with the valid values in the message (never a silent
        # fallback — ARD §2); ValueError -> 400
        try:
            cfg.validate_engine(engine, where="POST /api/settings")
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        engine_changed = engine != (current.get("engine") or cfg.DEFAULT_ENGINE)
        current["engine"] = engine
        # DEC-08: the t_90def951 engine-save flag-mapping report (which
        # llama.cpp flags would bridge into vllm: keys) is retired along
        # with the launch-path bridge — vLLM reads only the per-model
        # `vllm:` section now, so there is nothing to preview. A model
        # without a `vllm:` section fails loudly at launch time with
        # "no vllm config for this model". DEC-09 retires the reverse
        # (B11c save-path translation) too — no cross-engine mapping
        # or warning groups remain anywhere.
    if vllm_in is not None:
        if not isinstance(vllm_in, dict):
            raise HTTPException(400, "vllm must be an object")
        merged = dict(current.get("vllm") or {})
        merged.update(vllm_in)
        probe = {"engine": current.get("engine", cfg.DEFAULT_ENGINE), "vllm": merged}
        try:
            cfg.inject_engine_defaults(probe)  # unknown keys / bad delivery -> ValueError
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        current["vllm"] = probe["vllm"]
    # normalize the merged settings (injects defaults for missing keys) so
    # the file on disk always round-trips through the same schema the
    # loader enforces; failure -> 400, file left untouched
    try:
        cfg.inject_engine_defaults(current)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    # never persist raw token in YAML — source of truth is the encrypted
    # vault (.env), legacy YAML token ignored/removed
    current.pop("hf_token", None)
    import secrets_vault as _vault
    if clear_tok:
        _vault.set_hf_token("")
    elif tok:
        _vault.set_hf_token(tok)
    # don't echo the raw token back; write via the comment-preserving
    # persister (B9c) so unrelated keys, key order, and all comments
    # survive — never a wholesale rewrite from the request dict
    # DEC-15: the _engine_save_warnings channel is retired with the
    # cross-engine save-path translation (DEC-09) — no warning groups
    # remain. Keep the scrub-pop one release long: a pre-DEC-15 build
    # may have persisted the key into settings.yaml, and the persister
    # rewrites the whole file from this dict, so dropping it here
    # scrubs the stale copy from disk.
    current.pop("_engine_save_warnings", None)
    _dump_settings_yaml_preserving(current, SETTINGS_FILE)
    out = dict(current)
    out.pop("hf_token", None)  # response-only: never echoed
    out["hf_token_set"] = _vault.hf_token_set()
    if engine_changed:
        # Engine flip: the running daemons still serve the OLD engine's
        # configs/scripts. Full restart (same as /api/restart) — but the
        # app process restarting itself re-reads settings.yaml on boot,
        # so the new engine takes effect everywhere in one shot.
        resp = await _restart_everything_for_engine(engine)
        # keep the settings echo (engine/vllm block/hf_token_set) in the
        # same response — the client renders the saved state AND the
        # restart notice; a bare {"status": "restarting"} lost the
        # normalized settings the caller just saved.
        merged = dict(out)
        merged.update(resp)
        return merged
    return out


# ---------------- model views (per-IP visibility; Access tab) ------------
class ViewIn(pydantic.BaseModel):
    """One view-policy entry. `ips` are the client IPs landing in this
    view (Tailscale / LAN, exact literals); `models` are allowlist
    tokens (model_id or model_id--variant). Both replace-on-save."""
    ips: list[str] = []
    models: list[str] = []


@app.get("/api/views")
async def list_views(r: Request):
    """Access-key table (the Access tab identity model). Control surfaces
    (this endpoint + siblings) require trusted context (loopback console)
    or an `admin: true` key. Legacy name kept so the existing UI route
    strings work."""
    if not access.gate_open() and not access.is_admin(None):
        raise HTTPException(404, "not found")
    # NEVER echo full tokens back — masked; the full value is handed to
    # the operator exactly ONCE at generation.
    keys = access.access_keys()
    return {"keys": [
        {"name": e["name"], "token_masked": _mask_token(tok),
         "admin": e["admin"], "models": sorted(e["models"]),
         "unrestricted": e.get("unrestricted", True)}
        for tok, e in keys.items()]}


def _access_save_autopush():
    """Access-key save hook: key grants map to model allowlist tokens, but
    the ROUTING half (which deployment ids exist in llama-swap at all) is
    the pushed swap config — a key granted a never-pushed variant ("mid
    --dad") stayed invisible until the next manual push (observed live:
    gateway filtered it out of /v1/models because llama-swap's config had
    no such key). So every key mutation regenerates + writes the swap
    config in the background; the running daemon watches the file and
    picks routes up live (same semantic as save_model's auto-push, minus
    the reload — a config-only watch reload serves list changes; a stop/
    start here would yank loaded models out from under users). Fire-
    and-forget: Access saves never block on or fail from the push."""
    def _push():
        try:
            st = swap_manager.status()
            if not st.get("running"):
                return
            pr = swap_manager.push_scripts()
            if pr.get("error"):
                print(f"[access-autopush] push failed: {pr['error']}")
                return
            print(f"[access-autopush] swap config re-pushed "
                  f"({len(pr.get('models', []))} models)")
        except Exception as e:
            print(f"[access-autopush] failed: {e}")
    import threading as _threading
    _threading.Thread(target=_push, daemon=True,
                      name="access-save-autopush").start()


@app.post("/api/views/keys/reveal")
async def key_reveal(r: Request):
    """ADMIN REVEAL: the requester is an admin identity — they may see
    any key's FULL token (they hand keys to users, revoke them, rotate
    them). Keyed by NAME (the token itself must never need to be known
    to request it). Never logs; no-store."""
    if not (access.is_admin(None) or
            (r.scope.get("client") or (None,))[0] in access._TRUSTED_ANCHORS):
        raise HTTPException(404, "not found")
    import json as _json
    try:
        body = _json.loads((await r.body()) or b"{}")
    except Exception:
        body = {}
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "key name required")
    # reveal by name — find tokens by their record's name
    keys = access.access_keys()
    for tok, e in keys.items():
        if e["name"] == name:
            return JSONResponse(
                status_code=200,
                content={"name": name, "key": tok},
                headers={"Cache-Control": "no-store, max-age=0"})
    raise HTTPException(404, f"no key named '{name}'")


def _mask_token(tok: str) -> str:
    if len(tok) <= 8:
        return tok[:4] + "…"
    return tok[:8] + "…" + tok[-4:]


class KeyIn(pydantic.BaseModel):
    name: str = ""
    models: list[str] = []
    admin: bool = False
    unrestricted: bool | None = None  # None = legacy default (true)


@app.post("/api/views/keys/generate")
async def key_generate(k: KeyIn, r: Request):
    """Create an access key: random token, returned ONCE in this
    response — never echoed back by the list endpoint."""
    if not access.gate_open() and not access.is_admin(None):
        raise HTTPException(404, "not found")
    tok = access.generate_token()
    current = _load_settings_yaml_preserving(SETTINGS_FILE)
    keys = current.get(access.KEYS_SETTINGS_KEY) or {}
    if not isinstance(keys, dict):
        keys = {}
    ent = {"name": (k.name or "key").strip() or "key",
           "models": [m.strip() for m in k.models if m.strip()],
           "admin": bool(k.admin),
           "admin_unrestricted": True if k.unrestricted is None
                                 else bool(k.unrestricted)}
    keys[tok] = ent
    current[access.KEYS_SETTINGS_KEY] = keys
    _dump_settings_yaml_preserving(current, SETTINGS_FILE)
    _access_save_autopush()
    return {"ok": True, "token": tok, "name": ent["name"],
            "admin": ent["admin"], "models": ent["models"],
            "unrestricted": ent["admin_unrestricted"]}


@app.post("/api/views/keys/{token_masked}/update")
async def key_update(token_masked: str, k: KeyIn, r: Request):
    """Update a key by its MASKED id (the list never echoes raw tokens;
    update/delete ride the mask, matching what the operator copied)."""
    if not access.gate_open() and not access.is_admin(None):
        raise HTTPException(404, "not found")
    current = _load_settings_yaml_preserving(SETTINGS_FILE)
    keys = current.get(access.KEYS_SETTINGS_KEY) or {}
    for tok in list(keys):
        if _mask_token(tok) != token_masked:
            continue
        ent = keys[tok] if isinstance(keys[tok], dict) else {}
        models = [m.strip() for m in k.models if m.strip()]
        keys[tok] = {"name": (k.name or ent.get("name") or "key").strip()
                     or "key", "models": models, "admin": bool(k.admin),
                     "admin_unrestricted": ent.get("admin_unrestricted", True)
                     if k.unrestricted is None else bool(k.unrestricted)}
        current[access.KEYS_SETTINGS_KEY] = keys
        _dump_settings_yaml_preserving(current, SETTINGS_FILE)
        _access_save_autopush()
        return {"ok": True, "updated": token_masked}
    raise HTTPException(404, "not found")


@app.post("/api/views/keys/revoke")
async def key_revoke(rq: KeyIn, r: Request):
    """Revoke by masked token (body: {"name": masked_token})."""
    if not access.gate_open() and not access.is_admin(None):
        raise HTTPException(404, "not found")
    current = _load_settings_yaml_preserving(SETTINGS_FILE)
    keys = current.get(access.KEYS_SETTINGS_KEY) or {}
    target = (rq.name or "").strip()
    for tok in list(keys):
        if _mask_token(tok) == target:
            keys.pop(tok, None)
            current[access.KEYS_SETTINGS_KEY] = keys  # empty {} stays (no keys = open mode)
            _dump_settings_yaml_preserving(current, SETTINGS_FILE)
            _access_save_autopush()
            return {"ok": True, "revoked": target}
    raise HTTPException(404, "not found")


@app.post("/api/views/whoami")
async def views_whoami(
    r: Request,
    x_diag_ip: str | None = None,
    x_forwarded_for: str | None = None,
):
    """(docs below)"""
    """Identity report for THIS request: resolved key name, admin flag,
    scope. The Access tab + boot gate consume it. X-Diag-IP live-check:
    asker must be trusted (loopback) or admin-key context."""
    if x_diag_ip:
        if not access.gate_open() and not access.is_admin(None):
            raise HTTPException(404, "not found")
        # live-check now answers for a KEY context simulation
        rec = access.key_record(x_diag_ip.strip())
        if rec is None:
            return {"name": None, "admin": False, "scoped": False,
                    "models": [], "trusted": False, "unknown_key": True}
        return {"name": rec["name"], "admin": rec["admin"],
                "scoped": bool(rec["models"]),
                "models": sorted(rec["models"]),
                "trusted": rec["admin"] and not rec["models"]}
    return {"name": access.key_display_name(),
            "admin": bool(access.is_admin(None)),
            "scoped": access.view_allow() is not None,
            "models": sorted(access.view_allow() or []),
            "trusted": access.gate_open() and access.view_allow() is None}


@app.post("/api/install-llama-cpp")
async def install_llama_cpp():
    """Detect existing llama.cpp installation and set bin path.
    On Linux: checks ~/llama.cpp/build/bin/llama-server
    On Windows: checks %USERPROFILE%/llama.cpp/build/bin/llama-server.exe
    Does NOT build - just configures the path if already present.
    Use llama-install.sh or system package manager for full build.
    """
    from pathlib import Path
    is_win = sys.platform == "win32" or os.name == "nt"
    home = Path.home()
    llama_bin = None
    if is_win:
        d = home / "llama.cpp"
        if d.is_dir():
            cand = d / "build" / "bin" / "llama-server.exe"
            if cand.exists():
                llama_bin = str(cand)
    else:
        d = home / "llama.cpp"
        if d.is_dir():
            cand = d / "build" / "bin" / "llama-server"
            if cand.exists():
                llama_bin = str(cand)
    if llama_bin is None:
        return JSONResponse(status_code=200, content={"status": "no-installation", "message": "llama.cpp not found on this system - install via https://github.com/ggml-org/llama-install.sh (curl https://llama.app/install.sh | sh) or system package manager"})
    settings = _load_settings_yaml_preserving(SETTINGS_FILE)
    settings["llama_bin"] = llama_bin
    _dump_settings_yaml_preserving(settings, SETTINGS_FILE)
    return {"status": "success", "llama_bin": llama_bin, "message": "llama.cpp path configured"}


def _detect_bin_on_machine(machine):
    """Check if llama.cpp bin already exists on a machine. Returns
    (found:bool, bin_path:str|None, detail:str). Never installs — just
    probes, so existing working systems are untouched."""
    import subprocess as _sp
    from pathlib import Path as _Path
    is_win = (machine.get("os", "linux") == "windows")
    is_local = local_node.is_local(machine)
    # candidate bin paths to probe — configured + defaults
    candidates = []
    configured = (machine.get("llama_bin") or "").strip()
    if configured:
        # configured may be a dir or full bin path
        if configured.endswith("llama-server") or configured.endswith("llama-server.exe"):
            candidates.append(configured)
        else:
            candidates.append(configured.rstrip("/\\") + ("/llama-server.exe" if is_win else "/llama-server"))
    if is_win:
        candidates += [
            "%USERPROFILE%\\llama.cpp\\build\\bin\\llama-server.exe",
            "%USERPROFILE%\\llama.cpp\\llama-server.exe",
            "%USERPROFILE%/llama.cpp/build/bin/llama-server.exe",
            "%USERPROFILE%/llama.cpp/llama-server.exe",
            "~/llama.cpp/build/bin/llama-server.exe",
            "~/llama.cpp/llama-server.exe",
        ]
    else:
        candidates += [
            "~/llama.cpp/build/bin/llama-server",
            "/usr/local/bin/llama-server",
            "/usr/bin/llama-server",
        ]
    # dedupe preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    for cand in uniq:
        # expand for local check
        if is_local:
            try:
                if is_win:
                    expanded = os.path.expandvars(os.path.expanduser(cand))
                    if _Path(expanded).exists():
                        return True, cand, f"found at {cand}"
                else:
                    expanded = os.path.expanduser(cand)
                    if _Path(expanded).exists():
                        return True, cand, f"found at {cand}"
            except Exception:
                continue
        else:
            # remote probe via SSH
            try:
                from ssh_manager import SSHManager
                ssh = SSHManager(machine["ip"], machine["username"], machine["password"], port=machine.get("port", 22))
                if is_win:
                    # PowerShell Test-Path, handles %USERPROFILE% and ~
                    ps = f'powershell -NoProfile -Command "if (Test-Path \\"{cand}\\") {{ echo FOUND }} else {{ echo MISSING }}"'
                    code, out, _ = ssh.run(ps, timeout=15)
                else:
                    # expand ~ on remote side
                    code, out, _ = ssh.run(f'test -f {cand} && echo FOUND || echo MISSING', timeout=15)
                if "FOUND" in (out or ""):
                    return True, cand, f"found at {cand}"
            except Exception as e:
                # treat SSH failure as not found but surface in detail
                continue
    return False, None, "no bin found (checked: " + ", ".join(uniq) + ")"


class InstallAllIn(pydantic.BaseModel):
    install_missing: bool = False  # actually install on missing nodes
    machines: list[str] | None = None  # subset to check/install; None = all


@app.post("/api/install-llama-cpp/all")
async def install_llama_cpp_all(p: InstallAllIn | None = None):
    """Check every configured machine for an existing llama.cpp bin.
    Existing installs are SKIPPED (not reinstalled) — preserves working
    systems. Missing/offline nodes are reported so the UI can act.
    With install_missing=true, missing nodes get the correct official
    prebuilt for their machine type installed over SSH."""
    install_missing = bool(p and p.install_missing)
    machines = cfg.machines.all()
    # modal subset: only check/install the ticked machines (None = all)
    only = {n for n in (p.machines if p and p.machines else []) if n}
    if only:
        machines = [m for m in machines if m.get("name") in only]
    if not machines:
        return JSONResponse(status_code=200, content={"results": [], "message": "no machines configured"})

    def check_one(m):
        name = m.get("name", "?")
        ip = m.get("ip", "")
        os_type = m.get("os", "linux")
        is_local = local_node.is_local(m)
        # quick reachability for remote
        if not is_local:
            try:
                from ssh_manager import SSHManager
                ssh = SSHManager(m["ip"], m["username"], m["password"], port=m.get("port", 22))
                reachable = ssh.is_reachable()
            except Exception:
                reachable = False
            if not reachable:
                return {"machine": name, "ip": ip, "os": os_type, "status": "offline",
                        "skipped": True, "reason": "offline/unreachable — skipped",
                        "bin": m.get("llama_bin") or None}
        found, bin_path, detail = _detect_bin_on_machine(m)
        if found:
            # ensure machine record has a bin if it was empty
            if not (m.get("llama_bin") or "").strip():
                try:
                    cfg.machines.update(name, {"llama_bin": bin_path})
                except Exception:
                    pass
            return {"machine": name, "ip": ip, "os": os_type, "status": "already-installed",
                    "skipped": True, "reason": detail, "bin": bin_path}
        else:
            return {"machine": name, "ip": ip, "os": os_type, "status": "missing",
                    "skipped": False, "reason": detail, "bin": None}

    # run checks in parallel (threadpool, since SSH is blocking)
    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        results = list(pool.map(check_one, machines))

    # optionally install the right prebuilt on missing nodes (sequentially;
    # installs are heavy downloads, and SSH fan-out of 600s curls is risky)
    if install_missing:
        import llamacpp_installer
        for r in results:
            if r.get("status") != "missing":
                continue
            m = next((mm for mm in machines if mm.get("name") == r["machine"]), None)
            if not m:
                continue
            res = llamacpp_installer.install_on(m)
            r["status"] = "installed" if res.get("ok") else "install-failed"
            r["skipped"] = bool(res.get("ok"))
            r["reason"] = res.get("detail") or res.get("error", "")
            r["bin"] = res.get("bin")

    skipped = sum(1 for r in results if r.get("skipped") and r["status"] in ("already-installed", "offline", "installed"))
    missing = [r for r in results if r["status"] == "missing"]
    return {
        "results": results,
        "summary": f"{len(results)} nodes checked: {skipped} already installed/offline (skipped), {len(missing)} missing",
        "missing": [r["machine"] for r in missing],
    }


@app.post("/api/install-llama-cpp/install")
async def install_llama_cpp_machine(name: str = ""):
    """Install the right official llama.cpp prebuilt for a machine type
    (CUDA/ROCm/Metal/CPU, linux/windows/macos) over SSH or locally,
    then point the machine's llama_bin at it."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    machine = cfg.machines.require(name)
    import llamacpp_installer
    result = await run_in_threadpool(llamacpp_installer.install_on, machine)
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)


# ---------- per-machine llama.cpp UPDATE jobs (machine-page button) ----------
# Installs are slow multi-hundred-MB downloads (winbox's CUDA zip crawls
# at ~1.4MB/s — 15+ min), far past any sane synchronous request timeout,
# so the update button runs as a polled background job. One update at a
# time per machine; the registry is tiny and process-local like
# hf_downloader's jobs.
_LLAMA_UPDATES: dict = {}
_LLAMA_UPDATES_LOCK = threading.Lock()
_LATEST_TAG_CACHE = {"tag": None, "ts": 0.0}
# per-machine llama-server version probe (SSH/bash, ~seconds) — cached
# so the machine grid can re-ask on every render without hammering
_LLAMA_VER_CACHE: dict = {}
_LLAMA_VER_TTL = 600


def _llama_build_no(version_str):
    """'version: 0.7.0-dev (build 11869, commit d327bf1f0)' -> 11869
    ('build N' wins — the first bare integer would be the 0 in 0.7.0).
    Bare release tags ('b11869') fall back to first-integer."""
    m = (re.search(r"build (\d+)", version_str or "")
         or re.search(r"(\d+)", version_str or ""))
    return int(m.group(1)) if m else None


def _llama_version_on(machine):
    """First line of '<llama_bin>/llama-server --version' on a machine
    (local bash or raw SSH). POSIX: leading ~ expands via $HOME on the
    target; Windows: cmd.exe expands %USERPROFILE% inside the path."""
    lb = (machine.get("llama_bin") or "").strip().rstrip("/\\")
    if not lb:
        return None
    is_win = machine.get("os", "linux") == "windows"
    try:
        from llamacpp_installer import _machine_run
        run, _ = _machine_run(machine)
        if is_win:
            cmd = f'"{lb}\\llama-server.exe" --version 2>&1'
        else:
            cmd = rf'p="{lb}/llama-server"; p="${{p/#\~/$HOME}}"; "$p" --version 2>&1'
        code, out = run(cmd, timeout=30)
    except Exception:
        return None
    for line in (out or "").splitlines():
        if line.strip():
            return line.strip()
    return None


def _llama_version_payload(machine):
    """Current build vs latest upstream for one machine, TTL-cached —
    shared by the version endpoint and the update job's skip check."""
    name = machine.get("name", "?")
    now = time.time()
    with _LLAMA_UPDATES_LOCK:
        hit = _LLAMA_VER_CACHE.get(name)
    if hit and now - hit["ts"] < _LLAMA_VER_TTL:
        return hit
    version = _llama_version_on(machine)
    latest = _latest_llama_tag()
    build = _llama_build_no(version or "")
    latest_build = _llama_build_no(latest)
    payload = {"version": version, "build": build,
               "latest": latest, "latest_build": latest_build,
               "on_latest": bool(build and latest_build
                                 and build >= latest_build),
               "ts": now}
    with _LLAMA_UPDATES_LOCK:
        _LLAMA_VER_CACHE[name] = payload
    return payload


def _latest_llama_tag():
    """Newest upstream release tag that carries bin assets (5-min cache —
    the machine grid re-asks on every detail open and GitHub's API is
    rate-limited)."""
    import llamacpp_installer
    now = time.time()
    with _LLAMA_UPDATES_LOCK:
        if _LATEST_TAG_CACHE["tag"] and now - _LATEST_TAG_CACHE["ts"] < 300:
            return _LATEST_TAG_CACHE["tag"]
    tag, _assets, _base = llamacpp_installer._fetch_latest_assets()
    with _LLAMA_UPDATES_LOCK:
        _LATEST_TAG_CACHE.update(tag=tag, ts=now)
    return tag


@app.get("/api/llama-cpp-latest")
async def llama_cpp_latest():
    """Newest upstream llama.cpp release with binaries, for the update
    badges (compared against each machine's llama_version build number).
    Failures are 200 + latest=null + detail so a GitHub hiccup only
    blanks the badge, never errors the modal."""
    try:
        tag = await run_in_threadpool(_latest_llama_tag)
    except Exception as e:
        return {"latest": None, "latest_build": None,
                "detail": f"cannot reach GitHub releases: {e}"}
    return {"latest": tag, "latest_build": _llama_build_no(tag)}


@app.get("/api/machines/{name}/llama-cpp-version")
async def llama_cpp_version_machine(name: str):
    """Current llama-server build on one machine vs the latest upstream
    release — lets the UI swap the Update button for an 'on latest'
    chip. TTL-cached (SSH probe + GitHub tag are both slow-ish), so the
    machine grid can re-ask on every render without hammering anything."""
    machine = cfg.machines.require(name)
    return await run_in_threadpool(_llama_version_payload, machine)


@app.post("/api/machines/{name}/update-llama-cpp")
async def update_llama_cpp_machine(name: str):
    """One-click llama.cpp UPDATE for one machine (machine-page button).

    Background job that: (1) stops that machine's llama processes —
    Windows refuses to overwrite locked exes (observed live on winbox),
    and Linux keeps serving the old inode anyway; (2) installs the
    newest prebuilt via llamacpp_installer (auto-detects os / gpu_vendor
    / arch / glibc, uses the stored SSH credentials); (3) resurrects
    llama-swap if this machine hosts it (stop_rpc's legacy stop-all
    kills it). Poll GET /api/llama-cpp-updates/{job_id} for progress."""
    machine = cfg.machines.require(name)
    with _LLAMA_UPDATES_LOCK:
        for j in _LLAMA_UPDATES.values():
            if j["machine"] == name and j["state"] in ("stopping", "installing"):
                return JSONResponse(
                    status_code=409,
                    content={"detail": f"update already running on {name}",
                             "job_id": j["id"],
                             "status_url": f"/api/llama-cpp-updates/{j['id']}"})
        job_id = secrets.token_hex(6)
        job = {"id": job_id, "machine": name, "state": "stopping",
               "log": [], "result": None,
               "started": time.time(), "finished": None}
        _LLAMA_UPDATES[job_id] = job

    def _run_update():
        import llamacpp_installer
        import rpc_manager
        import swap_manager
        import urllib.request
        was_swap_up = False
        try:
            # Already current? Skip the whole update (probe the running
            # binary's build vs the newest release tag) — a no-op
            # rebuild still costs a fetch + cmake pass, and the UI
            # shows 'on latest' for these machines anyway.
            vp = _llama_version_payload(machine)
            if vp.get("on_latest"):
                job["log"].append(
                    f"{name} already on the latest release "
                    f"({vp.get('latest')}) — nothing to do")
                job["result"] = {"ok": True, "machine": name,
                                 "detail": f"already on the latest release "
                                           f"({vp.get('latest')})"}
                job["state"] = "done"
            else:
                # Fully dynamic routing: probe the machine itself (what
                # binary dir exists, is a git source tree + CMakeCache
                # behind it, which flags that cache carries) and pick the
                # path from the ANSWER — never from path-string guessing.
                # New machines classify correctly the moment they're added.
                job["state"] = "updating"
                probe = llamacpp_installer.probe_setup(
                    machine, log=lambda m: job["log"].append(str(m)))
                kind = probe.get("kind", "prebuilt")
                if kind == "source":
                    fl = ", ".join(f"{k}={v}" for k, v in
                                   sorted((probe.get("flags") or {}).items())[:6])
                    job["log"].append(
                        f"{name}: source build detected"
                        + (f" ({fl})" if fl else "")
                        + " → in-place rebuild, custom flags preserved")
                    res = llamacpp_installer.update_source_on(
                        machine, probe, log=lambda m: job["log"].append(str(m)))
                else:
                    if kind == "missing":
                        job["log"].append(
                            f"{name}: no usable llama.cpp found → fresh "
                            "prebuilt install (os/vendor/arch/glibc-picked)")
                    else:
                        job["log"].append(
                            f"{name}: prebuilt install → stopping llama "
                            "processes first (Windows locks running exes)")
                        try:
                            if name == (swap_manager.get_host() or {}).get("name"):
                                port = int(swap_manager.load_settings().get("listen_port") or 8080)
                                with urllib.request.urlopen(
                                        f"http://127.0.0.1:{port}/health", timeout=3) as r:
                                    was_swap_up = (r.status == 200)
                                job["log"].append(
                                    f"llama-swap on {name}: "
                                    + ("running — will restart after the update"
                                       if was_swap_up else "not running"))
                        except Exception:
                            was_swap_up = False
                        job["log"].append(f"stopping llama processes on {name} …")
                        try:
                            rep = rpc_manager.stop_rpc(name)
                            job["log"].append("stop_rpc: " + json.dumps(rep)[:400])
                        except Exception as e:
                            job["log"].append(f"stop_rpc failed (continuing anyway): {e}")
                    job["state"] = "installing"
                    res = llamacpp_installer.install_on(
                        machine, log=lambda m: job["log"].append(str(m)))
                job["result"] = res
                job["state"] = "done" if res.get("ok") else "failed"
        except Exception as e:
            job["state"] = "failed"
            job["result"] = {"ok": False, "detail": f"update crashed: {e}"}
        finally:
            if was_swap_up:
                try:
                    job["log"].append("restarting llama-swap on the swap host …")
                    rep = swap_manager.launch(None, None, None, None)
                    job["log"].append("llama-swap relaunch: "
                                      + json.dumps(rep)[:400])
                except Exception as e:
                    job["log"].append(
                        f"llama-swap relaunch failed: {e} "
                        "(restart it from the app)")
            # successful update means the version probe changed — drop
            # the cached one so the grid immediately shows 'on latest'
            with _LLAMA_UPDATES_LOCK:
                _LLAMA_VER_CACHE.pop(name, None)
            job["finished"] = time.time()

    threading.Thread(target=_run_update, daemon=True).start()
    return {"job_id": job_id, "status_url": f"/api/llama-cpp-updates/{job_id}"}


@app.get("/api/llama-cpp-updates/{job_id}")
async def llama_cpp_update_status(job_id: str):
    """Poll a running/finished per-machine llama.cpp update job."""
    with _LLAMA_UPDATES_LOCK:
        job = _LLAMA_UPDATES.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown update job id")
    out = dict(job)
    out["log"] = job["log"][-60:]
    return out


@app.post("/api/install-llama-swap")
async def install_llama_swap(name: str = ""):
    """Install the official llama-swap binary on a machine if not already
    present (local machines run locally — no SSH), then point swap
    settings' binary at it. Mirrors the llama.cpp installer."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    machine = cfg.machines.require(name)
    import swap_installer
    found, bin_path = await run_in_threadpool(swap_installer.detect_on, machine)
    if found:
        return {"ok": True, "machine": name, "bin": bin_path,
                "detail": f"already installed at {bin_path} (skipped)"}
    result = await run_in_threadpool(swap_installer.install_on, machine)
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)


@app.post("/api/machines/{name}/install-vllm")
async def install_vllm_machine(name: str = ""):
    """Install vLLM on ONE machine via vllm_installer (venv delivery —
    PyPI wheel into the dedicated venv; docker delivery refuses inside
    the installer). Returns the run_steps report, same shape as the
    llama.cpp install endpoint (ok=true -> 200, failed -> 502)."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    try:
        machine = cfg.machines.require(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    import vllm_installer
    result = await run_in_threadpool(vllm_installer.install_on, machine)
    # Passthrough: the caller gets the full run_steps report (machine,
    # delivery, target, steps[], ok, failed, detail) — same contract as
    # the llamacpp install endpoint. A refusal (windows, docker, no GPU
    # tooling) is a FAILED INSTALL, not a server error: the report rides
    # back with a 502 so the UI can show the installer's own message.
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)


@app.post("/api/install-vllm/all")
async def install_vllm_all(p: InstallAllIn | None = None):
    """Batch vLLM install: iterate configured machines and run
    vllm_installer on each (sequentially — installs are heavy downloads;
    mirroring the llama.cpp batch pattern). Non-linux machines are
    skipped up front (vLLM is linux-only) and collect the installer's
    refusal report instead of running anything on them."""
    import vllm_installer
    machines = cfg.machines.all()
    # modal subset (mirrors the llama.cpp batch endpoint): only the
    # ticked machines are checked/installed; None = all configured
    only = {n for n in (p.machines if p and p.machines else []) if n}
    if only:
        machines = [m for m in machines if m.get("name") in only]
    if not machines:
        return JSONResponse(status_code=200,
                            content={"results": [],
                                     "message": "no machines configured"})
    results = []
    for m in machines:
        name = m.get("name", "?")
        os_type = (m.get("os") or "linux").strip().lower()
        if os_type != "linux":
            results.append({"machine": name, "os": os_type,
                            "status": "skipped",
                            "ok": False,
                            "detail": ("vLLM is linux-only — install on "
                                       "a linux host (or a WSL2 instance "
                                       "added as a linux machine)")})
            continue
        res = await run_in_threadpool(vllm_installer.install_on, m)
        res["status"] = "installed" if res.get("ok") else "install-failed"
        results.append(res)
    installed = sum(1 for r in results if r.get("status") == "installed")
    failed = sum(1 for r in results if r.get("status") == "install-failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    return {
        "results": results,
        "summary": (f"{len(results)} nodes: {installed} installed, "
                    f"{failed} failed, {skipped} skipped (non-linux)"),
        "missing": [r["machine"] for r in results
                    if r.get("status") == "install-failed"],
    }


@app.get("/api/machines/{name}/vllm-plan")
async def vllm_plan_machine(name: str):
    """B13 pre-flight method report for ONE machine: WHICH delivery method
    the installer would choose and WHY (per-machine vllm_delivery override
    wins over the settings global), plus the OS/hardware gates it would
    hit — BEFORE any install runs. Read-only (vllm_installer.plan never
    executes commands). 200 with the plan dict; a failed plan (windows,
    docker override in the venv-only wave, bad override value, unknown
    machine) is 200 too with ok=false + error — it is an ANSWER to "what
    would happen", not a server error. 404 only for an unknown machine."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="machine name required")
    try:
        machine = cfg.machines.require(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    import vllm_installer
    plan = await run_in_threadpool(vllm_installer.plan, machine)
    # plan() output only contains installer facts (no machine record fields
    # ride along), so no password can leak — assert-and-return keeps the
    # contract explicit for future fields.
    plan = dict(plan)
    plan.pop("password", None)
    return JSONResponse(status_code=200, content=plan)


# ---------------------------------------------------------------------------
# Unified installer endpoints (UX-1): probe/plan/execute layer in app
# ---------------------------------------------------------------------------
# Wraps src/installer/{probe,execute}.py. Remote machines run the probe
# ladder + executor over an SSHManager-backed RunCmd; local machines run
# RealRunCmd locally. NO GPU drivers are ever installed (executor refuses);
# llama.cpp serving behavior is untouched (this only adds new endpoints).


class ToolsInstallIn(pydantic.BaseModel):
    machine: str
    tools: list[str] = []          # empty = all tools the plan offers
    role: str = "main"             # 'main' | 'client'
    dry_run: bool = False


def _machine_env(machine):
    """(SystemInfo, RunCmd) for a machine record: local machines probe
    locally; remote machines run the same read-only probe ladder over
    SSH (vllm_installer._make_runner pattern). OS/gpu_vendor come from
    the machine record (config truth), the rest from the remote probes."""
    from installer.env import SystemInfo, RealFS, RunResult
    from installer.probe import LLCPP_BACKEND, VLLM_BACKEND
    import gpu_vendor as _gv
    import local_node as _ln

    os_type = (machine.get("os") or _ln.local_os() or "linux").strip().lower()
    if os_type == "darwin":
        os_type = "macos"
    vendor = _gv.normalize_vendor(machine.get("gpu_vendor"))
    is_local = _ln.is_local(machine)

    if is_local:
        from installer.env import RealRunCmd as _RRC
        run = _RRC()
    else:
        from ssh_manager import SSHManager

        class _SSHRunCmd:
            """RunCmd adapter: SSHManager.run -> (code, out, err) ->
            RunResult. Windows peers get a PowerShell wrapper so the
            probe ladder's shell syntax parses (cmd.exe default shell)."""

            def __init__(self, m):
                self._ssh = SSHManager(m["ip"], m.get("username", ""),
                                       m.get("password", ""),
                                       port=m.get("port", 22))
                self._win = os_type == "windows"

            def run(self, cmd, timeout=30.0):
                real = cmd
                if self._win:
                    real = ('powershell -NoProfile -Command "'
                            + cmd.replace('"', '\\"') + '"')
                try:
                    code, out, err = self._ssh.run(real, timeout=timeout)
                except Exception as e:
                    return RunResult(cmd, 255, "", f"ssh error: {e}")
                return RunResult(cmd, code, out or "", err or "")

        run = _SSHRunCmd(machine)

    # detect the rest of SystemInfo with the shared probe ladder (cheap,
    # read-only; same code path bootstrap.py uses locally)
    from installer.env import detect_system_info
    info = detect_system_info(run)
    # record facts WIN over probed facts where the record is authoritative:
    # the machine card's os/gpu_vendor is what the operator declared.
    info.os = os_type
    info.gpu_vendor = vendor or info.gpu_vendor
    return info, run


class _ToolsProgress:
    """Per-step install progress collector: Executor.log lines are
    appended as NDJSON events; the StreamingResponse consumer reads
    them as they land (same per-machine progress pattern as
    /api/stats/stream — per-EVENT, not per-machine)."""

    def __init__(self):
        import queue
        self.q = queue.Queue()
        self._done = object()

    def log(self, msg):
        self.q.put(("log", msg))

    def finish(self, summary):
        self.q.put(("done", summary))

    def stream(self):
        import queue
        while True:
            try:
                kind, payload = self.q.get(timeout=300)
            except queue.Empty:
                yield json.dumps({"event": "timeout"}) + "\n"
                return
            if kind == "log":
                yield json.dumps({"event": "log", "message": payload}) + "\n"
            else:
                yield json.dumps({"event": "done",
                                  "summary": payload}) + "\n"
                return


@app.post("/api/tools/install")
async def tools_install(p: ToolsInstallIn):
    """Unified install: plan for ONE machine (installer.probe), execute
    it (installer.execute), streaming per-step progress as NDJSON
    ({event: log|done}). Engine-gated: the plan carries per-tool
    skip/refuse notes for both engine modes; llama.cpp behavior is
    untouched (this endpoint only ADDS surfaces). Windows machines get
    a vLLM refusal with WSL guidance (probe parity matrix)."""
    machine = p.machine
    try:
        m = cfg.machines.require(machine)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    role = p.role if p.role in ("main", "client") else "main"
    progress = _ToolsProgress()

    def run_install():
        try:
            info, run = _machine_env(m)
            from installer.env import RealFS
            from installer.probe import probe as installer_probe
            from installer.execute import Executor

            fs = RealFS() if local_node.is_local(m) else _RemoteFS(run, is_win=getattr(run, '_win', False))
            # probe's parity matrix refuses windows/non-offered vLLM
            # backends as PlanNote (WSL guidance verbatim); engine mode
            # rides in settings, not in this endpoint's contract.
            plan = installer_probe(info, fs, run, role=role)
            if p.tools:
                keep = set(p.tools)
                # keep only the requested tools' steps; settings/manifest
                # steps for OTHER tools are dropped with their tool
                from installer.probe import (InstallTool, RepairTool,
                                             SkipTool, PlanNote)
                plan = [s for s in plan
                        if getattr(s, "name", None) in keep
                        or (type(s).__name__ not in
                            ("InstallTool", "RepairTool", "SkipTool",
                             "PlanNote"))]
            ex = Executor(log=progress.log, dry_run=p.dry_run,
                          info=info,
                          manifest_path=os.path.join(
                              os.path.expanduser("~"),
                              ".llama-rpc-farm", "installer-state.yaml"),
                          runner=lambda cmd: run.run(cmd))
            summary = ex.execute(plan)
            ex.write_manifest(summary, detected={})
            progress.finish(summary)
        except Exception as e:                          # noqa: BLE001
            from installer.execute import failure_envelope
            progress.finish({"failures": [failure_envelope("installer", e)],
                             "installed": [], "skipped": [],
                             "repaired": [], "settings": [], "notes": [],
                             "manifest": {}})

    import threading
    t = threading.Thread(target=run_install, daemon=True)
    t.start()
    return StreamingResponse(_tools_ndjson(progress),
                             media_type="application/x-ndjson")


async def _tools_ndjson(progress):
    """Drain the progress queue into NDJSON lines (async bridge so the
    event loop stays free while the install thread works)."""
    import queue as _q
    while True:
        try:
            kind, payload = await run_in_threadpool(
                lambda: progress.q.get(timeout=300))
        except _q.Empty:
            yield json.dumps({"event": "timeout"}) + "\n"
            return
        if kind == "log":
            yield json.dumps({"event": "log", "message": payload}) + "\n"
        else:
            yield json.dumps({"event": "done", "summary": payload}) + "\n"
            return


class _RemoteFS:
    """FS adapter for remote machines: existence/version checks over SSH
    (vllm_installer's runner pattern). Only what probe.py touches."""

    def __init__(self, run, is_win=False):
        self._run = run
        self._win = is_win

    def exists(self, path):
        return self.isfile(path)

    def isfile(self, path):
        p = self.expanduser(path)
        if self._win:
            cmd = ('powershell -NoProfile -Command '
                   '"if (Test-Path \\"' + p + '\\") { echo Y } '
                   'else { exit 1 }"')
        else:
            cmd = '[ -f "' + p + '" ] && echo Y || exit 1'
        r = self._run.run(cmd)
        return r.ok

    def expanduser(self, path):
        # expand ~ on the REMOTE side so the path is correct there
        if not path.startswith("~"):
            return path
        r = self._run.run("echo " + path)
        out = (r.stdout or "").strip()
        return out if out.startswith(("/", "C:", "%")) else path

    def read_file(self, path):
        r = self._run.run('cat "' + self.expanduser(path) + '"')
        if not r.ok:
            raise FileNotFoundError(path)
        return r.stdout
    def file_exists_and_runs(self, binary, version_arg="--version"):
        p = self.expanduser(binary)
        cmd = (f'if [ -f "{p}" ]; then "{p}" {version_arg} 2>&1; '
               f'else echo __MISSING__; exit 127; fi')
        r = self._run.run(cmd)
        if "__MISSING__" in (r.stdout or ""):
            return RunResult(f"{binary} {version_arg}", 127, "", "missing")
        return RunResult(f"{binary} {version_arg}", r.exit_code,
                         r.stdout or "", r.stderr or "")


@app.post("/api/tools/install/sync")
async def tools_install_sync(p: ToolsInstallIn):
    """Non-streaming variant of /api/tools/install: same plan+execute,
    but the full summary is returned in one body (for tests + simple
    clients). Same engine gates, same refusal semantics."""
    machine = p.machine
    try:
        m = cfg.machines.require(machine)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    from installer.env import RealFS
    from installer.probe import probe as installer_probe
    from installer.execute import Executor
    info, run = _machine_env(m)
    fs = RealFS() if local_node.is_local(m) else _RemoteFS(run, is_win=getattr(run, '_win', False))
    plan = installer_probe(info, fs, run, role=p.role if p.role in ("main", "client") else "main")
    summary = await run_in_threadpool(
        _execute_plan, plan, info, p.dry_run)
    ok = not summary.get("failures")
    return JSONResponse(status_code=200 if ok else 502, content=summary)


def _execute_plan(plan, info, dry_run):
    from installer.execute import Executor
    ex = Executor(log=lambda m: None, dry_run=dry_run, info=info,
                  manifest_path=os.path.join(
                      os.path.expanduser("~"),
                      ".llama-rpc-farm", "installer-state.yaml"))
    summary = ex.execute(plan)
    ex.write_manifest(summary, detected={})
    return summary


@app.post("/api/machines/{name}/uninstall-vllm")
async def uninstall_vllm_machine(name: str = ""):
    """Uninstall vLLM from ONE machine via vllm_uninstaller (venv
    delivery — removes the farm-managed venv and clears settings
    vllm.venv_bin; docker delivery refuses inside the uninstaller).
    Returns the run_steps report, same shape as the install endpoint
    (ok=true -> 200, failed -> 502). B12."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    try:
        machine = cfg.machines.require(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    import vllm_uninstaller
    result = await run_in_threadpool(vllm_uninstaller.uninstall_on, machine)
    # Passthrough: the caller gets the full run_steps report. A refusal
    # (docker, non-farm venv path) is a FAILED uninstall, not a server
    # error: the report rides back with a 502 so the UI can show the
    # uninstaller's own message.
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)


@app.post("/api/machines/{name}/uninstall-llamacpp")
async def uninstall_llamacpp_machine(name: str = ""):
    """Uninstall llama.cpp from ONE machine via llamacpp_uninstaller
    (removes ONLY the farm-managed install dir the installer created —
    path-guarded, never a source build, never anything outside the
    guarded root) and clears the machine's llama_bin pointer.
    Returns the run_steps report, same shape as the vllm uninstall
    endpoint (ok=true -> 200, failed -> 502). DEC5 (T5-1)."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    try:
        machine = cfg.machines.require(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    import llamacpp_uninstaller
    result = await run_in_threadpool(llamacpp_uninstaller.uninstall_on,
                                     machine)
    # Passthrough: the caller gets the full run_steps report. A refusal
    # (foreign llama_bin path) is a FAILED uninstall, not a server
    # error: the report rides back with a 502 so the UI can show the
    # uninstaller's own message.
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)


# ---------------- GET /api/tools/verify (UX-3, design §8) --------------------
# The old REST install endpoints (install-llama-cpp{,/all,/install},
# install-llama-swap, machines/{name}/install-vllm, install-vllm/all,
# machines/{name}/vllm-plan, uninstall-vllm, uninstall-llamacpp) are
# RETIRED: the app never executes installs anymore — src/bootstrap.py owns
# them (copy-paste one-liner from the UI, no exec from the app). What
# survives is the read-only status probe: per tool {found, path, version,
# expected} from installer.probe (the same pure logic the installer uses,
# via RealFS/RealRunCmd). A settings path that exists counts as found even
# when the well-known locations miss (the settings record is probed first,
# §6 prefill is the source of truth once the installer ran).


@app.get("/api/tools/verify")
async def tools_verify():
    """Read-only per-tool status: llama.cpp / llama-swap / vLLM.
    Each entry: {found, path, version, expected, configured} where
    `configured` is the settings-recorded path (may be stale). Probes are
    local-only (this host) and read-only — nothing is ever installed,
    removed, or written here."""
    from installer.env import RealFS, RealRunCmd
    from installer import probe as installer_probe
    import yaml as _yaml

    run = RealRunCmd()
    fs = RealFS()

    def _settings_key(path, key):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
        except Exception:
            return None
        cur = data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur if isinstance(cur, str) else None

    settings_bin = _settings_key(SETTINGS_FILE, "llama_bin")
    venv_bin = _settings_key(SETTINGS_FILE, "vllm.venv_bin")
    try:
        import swap_manager
        swap_bin = (swap_manager.load_settings() or {}).get("binary")
    except Exception:
        swap_bin = None

    def _entry(found, path, version, expected, configured):
        return {"found": bool(found), "path": path, "version": version,
                "expected": expected, "configured": configured or None}

    out = {}
    # llama.cpp — well-known prebuilt/source locations, then settings record
    p, ver = installer_probe.probe_llama_cpp(fs, run, role="main")
    if not p and settings_bin:
        cand = settings_bin
        if not (cand.endswith("llama-server") or cand.endswith("llama-server.exe")):
            cand = cand.rstrip("/\\") + "/llama-server"
        if fs.isfile(cand):
            r = fs.file_exists_and_runs(cand)
            p, ver = cand, (installer_probe._version_from_output(
                r.stdout + " " + r.stderr) if r.ok else None)
    out["llama.cpp"] = _entry(p, p, ver, "b10000+", settings_bin)
    # llama-swap
    p, ver = installer_probe.probe_swap(fs, run)
    if not p and swap_bin and swap_bin != "llama-swap" and fs.isfile(swap_bin):
        r = fs.file_exists_and_runs(swap_bin)
        p, ver = swap_bin, (installer_probe._version_from_output(
            r.stdout + " " + r.stderr) if r.ok else None)
    out["llama-swap"] = _entry(p, p, ver, "255+", swap_bin)
    # vLLM
    p, ver = installer_probe.probe_vllm(fs, run)
    if not p and venv_bin and fs.isfile(venv_bin):
        r = fs.file_exists_and_runs(venv_bin)
        p, ver = venv_bin, (installer_probe._version_from_output(
            r.stdout + " " + r.stderr) if r.ok else None)
    out["vllm"] = _entry(p, p, ver, "0.30.0", venv_bin)
    return out



@app.post("/api/restart")
async def restart_app():
    """Stop llama-swap + RPC on all nodes and restart this web app.
    POSIX: bash + setsid (detached). Windows: PowerShell. The new
    process survives the parent's exit on every platform.
    Frontend should poll / after ~3s to confirm it's back.
    Resolves python + app path from this file's location so any
    project-root .venv is used (no hardcoded paths).
    """
    import subprocess as _sp
    import shlex as _shlex
    app_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(app_path))
    candidates = [
        os.path.join(project_root, ".venv", "bin", "python"),
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        sys.executable,
    ]
    py = next((c for c in candidates if c and os.path.exists(c)), sys.executable)
    log = os.path.join(tempfile.gettempdir(), "llama-web.log")
    if os.name == "nt":
        # Windows: PowerShell — kill daemons + port owner, then start
        # detached (Start-Process survives the parent).
        ps = (
            "Start-Sleep 1; "
            "Get-Process llama-swap,llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; "
            "$c = Get-NetTCPConnection -LocalPort 8199 -State Listen -ErrorAction SilentlyContinue; "
            "if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }; "
            "Start-Sleep 1; "
            f"Start-Process -FilePath '{py}' -ArgumentList '\"{app_path}\"' "
            f"-WindowStyle Hidden -RedirectStandardOutput '{log}' -RedirectStandardError '{log}.err'"
        )
        _sp.Popen(["powershell", "-NoProfile", "-Command", ps],
                  start_new_session=True, close_fds=True)
    else:
        # POSIX (Linux + macOS): bash exists on both. macOS lacks fuser
        # and setsid, so prefer them but fall back to nohup/&.
        q = _shlex.quote
        # restart llama-swap too, with the saved settings (port + config)
        try:
            s = swap_manager.load_settings()
        except Exception:
            s = {}
        sw_port = s.get("listen_port") or 8080
        sw_cfg = os.path.expanduser(s.get("config_path") or "~/.config/llama-swap/config.yaml")
        sw_bin = s.get("binary") or "llama-swap"
        sw_home = os.path.expanduser("~/.local/bin")
        sw_bin_full = sw_bin if os.path.sep in sw_bin else (
            shutil.which(sw_bin) or os.path.join(sw_home, sw_bin))
        cmd = (
            "exec >> " + q(os.path.expanduser(
                "~/.config/llama-swap/restart-trace.log")) + " 2>&1; "
            "echo ===restart $(date -u +%H:%M:%S) pid $$; "
            "sleep 1; "
            "pkill llama-swap 2>/dev/null || true; "
            "pkill llama-server 2>/dev/null || true; "
            # BRACKET TRICK: this bash's own cmdline CONTAINS the literal
            # 'vllm serve' text, so `pkill -f 'vllm serve'` matches and
            # kills ITSELF (observed: restart bash died right here, swap
            # never started). The [v] character class breaks self-match.
            "pkill -f '[v]llm serve' 2>/dev/null || true; "
            "(fuser -k 8199/tcp > /dev/null 2>&1 || true); "
            # Observed: right after fuser -k, the dying process can still
            # hold :8199 for up to ~9s (SIGTERM shutdown + TIME_WAIT-ish
            # linger). The freshly spawned app's singleton guard waits 10s
            # for the bind and exits(2) if held — race lost = dead UI until
            # manual start. Poll for the port going free BEFORE setsid, so
            # the handoff never races (max 15s).
            "for i in $(seq 0 30); do "
            "  python3 -c \"import socket;s=socket.socket();\\"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);\\"
            "s.bind(('0.0.0.0',8199));s.close()\" 2>/dev/null && break; "
            "  sleep 0.5; "
            "done; "
            "echo step-before-systemctl alive=$$; "
            # swap comes back via its systemd --user unit when one exists
            # (Restart=always survives crashes AND direct kills); setsid
            # bare-start only as fallback for installs without the unit.
            # A setsid HERE fights the unit (two daemons race for :8080)
            # and a systemd STOP by any path leaves swap dead — observed
            # after restart: unit inactive, nothing brought it back.
            "systemctl --user start llama-swap || echo systemctl-start-failed rc=$?; "
            "systemctl --user is-active --quiet llama-swap || "
            f"setsid -f {q(sw_bin_full)} --config {q(sw_cfg)} --listen 127.0.0.1:{sw_port} --watch-config "
            f">> {q(os.path.expanduser('~/.config/llama-swap/llama-swap-8080.log'))} 2>&1 || true; "
            "echo step-after-swap is-active=$(systemctl --user is-active llama-swap); "
            f"setsid -f {q(py)} {q(app_path)} > {q(log)} 2>&1 "
            f"|| nohup {q(py)} {q(app_path)} > {q(log)} 2>&1 &"
        )
        _sp.Popen(["/bin/bash", "-c", cmd], start_new_session=True, close_fds=True)
    # remote-proxy llama-servers (launch-mode=local models) run on their
    # own boxes — stop them BEFORE restarting, or they keep serving while
    # the app/swap cycle under them. Fire-and-forget: restart must not
    # block on SSH; the daemon comes back in ~3s either way.
    import threading as _th
    def _stop_remote_llama():
        # host EXCLUDED: the restart bash's pkill+setsid brings llama-swap
        # back up on this box — a stop_rpc here would race and kill the
        # freshly restarted daemon (observed: swap 000 after restart).
        import rpc_manager as _rm
        import local_node as _ln
        names = [m["name"] for m in cfg.machines.all()
                 if not _ln.is_local(m)]
        with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
            list(pool.map(_rm.stop_rpc, names))
    _th.Thread(target=_stop_remote_llama, daemon=True).start()
    return {"status": "restarting", "message": "Restarting web app - it will be back in ~3s at http://127.0.0.1:8199"}


# ---------------- file explorer (cross-platform) ----------------

def _fs_list_local(raw_path, show_hidden=False):
    import pathlib
    # handle store:// alias for model dir; resolve to real path but remember alias
    disp_path = raw_path
    if raw_path and raw_path.startswith(path_resolver.PREFIX):
        try:
            resolved = path_resolver.resolve(raw_path)
            raw_path = resolved
        except Exception:
            pass
    # expand ~ and env, normalize
    if not raw_path or raw_path.strip() == "":
        raw_path = os.path.expanduser("~")
    else:
        raw_path = os.path.expanduser(os.path.expandvars(raw_path))
    raw_path = os.path.abspath(raw_path) if not raw_path.startswith("/") else os.path.normpath(raw_path)
    # if path is file, show its dir
    if os.path.isfile(raw_path):
        raw_path = os.path.dirname(raw_path)
    exists = os.path.exists(raw_path)
    is_dir = os.path.isdir(raw_path) if exists else False
    parent = os.path.dirname(raw_path) if raw_path != "/" else "/"
    entries = []
    if is_dir:
        try:
            with os.scandir(raw_path) as it:
                for entry in it:
                    if not show_hidden and entry.name.startswith("."):
                        continue
                    try:
                        is_d = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        is_d = False
                    entries.append({"name": entry.name, "path": os.path.join(raw_path, entry.name), "is_dir": is_d})
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=f"permission denied: {raw_path}")
        # dirs first, then alpha
        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    # alias for portability (e.g. /mnt/... -> store://llm-models/...)
    try:
        alias = path_resolver.to_alias(raw_path)
    except Exception:
        alias = raw_path
    return {"path": raw_path, "alias": alias, "display_path": disp_path, "parent": parent, "exists": exists, "is_dir": is_dir, "os": "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux", "entries": entries}


def _fs_list_remote(machine, raw_path, show_hidden=False):
    from ssh_manager import SSHManager, SSHError
    is_win = machine.get("os", "linux") == "windows"
    if not raw_path or raw_path.strip() == "":
        raw_path = "~" if not is_win else "%USERPROFILE%"
    # store:// on remote: resolve for that machine's OS (best-effort: use local stores.yaml)
    if raw_path.startswith(path_resolver.PREFIX):
        try:
            raw_path = path_resolver.resolve_for(raw_path, machine.get("os", "linux"))
        except Exception:
            pass
    ssh = SSHManager(machine["ip"], machine["username"], machine["password"], port=machine.get("port", 22))
    if not is_win:
        # Use python if available for robust listing, else fallback to ls
        exp_path = raw_path
        # embed path as python repr to avoid argv/-- confusion and space handling
        import json as _json_local
        repr_path = repr(exp_path)
        code, out, err = ssh.run(f'python3 -c "import os, json; p=os.path.expanduser({repr_path}); print(json.dumps({{\'exists\': os.path.exists(p), \'is_dir\': os.path.isdir(p), \'parent\': os.path.dirname(os.path.abspath(p)), \'real\': p}}))"', timeout=15)
        meta = None
        if code == 0 and out.strip():
            try:
                import json as _json
                meta = _json.loads(out.strip().splitlines()[-1])
            except Exception:
                meta = None
        if meta and meta.get("exists"):
            real = meta["real"]
            parent = meta["parent"]
            is_dir = meta["is_dir"]
            if not is_dir:
                real = parent
                is_dir = True
                repr_real = repr(real)
                code, out, _ = ssh.run(f'python3 -c "import os; print(os.path.dirname(os.path.abspath(os.path.expanduser({repr_real}))))"', timeout=10)
                parent = out.strip().splitlines()[-1] if code == 0 else os.path.dirname(real)
            # list entries — try python json, else ls -p fallback
            repr_real2 = repr(real)
            code, out, _ = ssh.run(f'python3 -c "import os, json; p={repr_real2}; print(json.dumps([{{\'name\': n, \'is_dir\': os.path.isdir(os.path.join(p,n))}} for n in os.listdir(p) if not n.startswith(\".\") or {str(show_hidden).lower()}]))" 2>/dev/null || ls -1 -p "{real}"', timeout=15)
            entries = []
            if code == 0 and out.strip():
                # try json first
                try:
                    import json as _json
                    parsed = _json.loads(out.strip().splitlines()[-1])
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        for e in parsed:
                            entries.append({"name": e["name"], "path": os.path.join(real, e["name"]), "is_dir": bool(e["is_dir"])})
                    else:
                        raise ValueError()
                except Exception:
                    # ls -p fallback
                    for line in out.splitlines():
                        name = line.strip()
                        if not name:
                            continue
                        if name.startswith(".") and not show_hidden:
                            continue
                        is_d = name.endswith("/")
                        clean = name.rstrip("/")
                        entries.append({"name": clean, "path": os.path.join(real, clean), "is_dir": is_d})
                entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
            try:
                alias = path_resolver.to_alias(real)
            except Exception:
                alias = real
            return {"path": real, "alias": alias, "parent": parent, "exists": True, "is_dir": True, "os": "linux", "entries": entries}
        # fallback: try ls directly
        code, out, _ = ssh.run(f'ls -ld "{exp_path}" 2>&1', timeout=10)
        if code != 0:
            raise HTTPException(status_code=404, detail=f"path not found or unreachable: {raw_path}")
        # expand via echo
        code, out, _ = ssh.run(f'echo "{exp_path}"', timeout=10)
        real = out.strip() or exp_path
        try:
            alias = path_resolver.to_alias(real)
        except Exception:
            alias = real
        return {"path": real, "alias": alias, "parent": os.path.dirname(real), "exists": True, "is_dir": True, "os": "linux", "entries": []}
    else:
        # windows remote via powershell
        # normalize slashes
        ps_path = raw_path.replace("'", "''")
        # if empty or ~, use $env:USERPROFILE
        if raw_path.strip() in ("~", "%USERPROFILE%", "$env:USERPROFILE"):
            ps_path = "$env:USERPROFILE"
        code, out, err = ssh.run(f'powershell -NoProfile -Command "$p=\\"{ps_path}\\"; $e=Test-Path $p; $isDir=(Test-Path $p -PathType Container); $real=(Resolve-Path $p -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path -ErrorAction SilentlyContinue); if (-not $real) {{ $real=$p }}; $parent=Split-Path $real -Parent; if (-not $parent) {{ $parent=$real }}; Write-Output (@{{exists=$e; isDir=$isDir; real=$real; parent=$parent}} | ConvertTo-Json -Compress)"', timeout=15)
        if code != 0 or not out.strip():
            raise HTTPException(status_code=500, detail=f"windows browse failed: {err or out}")
        import json as _json
        try:
            meta = _json.loads(out.strip().splitlines()[-1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"parse failed: {e} – {out}")
        exists = bool(meta.get("exists"))
        is_dir = bool(meta.get("isDir"))
        real = meta.get("real") or raw_path
        parent = meta.get("parent") or os.path.dirname(real)
        if not exists:
            raise HTTPException(status_code=404, detail=f"path not found: {raw_path}")
        if not is_dir:
            real = parent
        # list
        code, out, err = ssh.run(f'powershell -NoProfile -Command "Get-ChildItem -Force -Path \\"{ps_path}\\" | Select-Object Name,FullName,PSIsContainer | ConvertTo-Json -Compress"', timeout=15)
        entries = []
        if code == 0 and out.strip():
            try:
                data = _json.loads(out.strip())
                if isinstance(data, dict):
                    data = [data]
                for e in data or []:
                    name = e.get("Name")
                    if not name:
                        continue
                    if name.startswith(".") and not show_hidden:
                        continue
                    entries.append({"name": name, "path": e.get("FullName") or os.path.join(real, name), "is_dir": bool(e.get("PSIsContainer"))})
                entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
            except Exception:
                pass
        try:
            alias = path_resolver.to_alias(real)
        except Exception:
            alias = real
        return {"path": real, "alias": alias, "parent": parent, "exists": exists, "is_dir": is_dir, "os": "windows", "entries": entries}


@app.get("/api/fs/browse")
async def fs_browse(path: str = "", machine: str = "", show_hidden: bool = False):
    """Browse filesystem — local host if machine=, else remote via SSH.
    Works on both Linux and Windows (machine.os decides). Query: path, machine (name), show_hidden."""
    if machine:
        m = cfg.machines.get(machine)
        if not m:
            raise HTTPException(status_code=404, detail=f"machine '{machine}' not found")
        # remote
        try:
            return await run_in_threadpool(_fs_list_remote, m, path, show_hidden)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        try:
            return await run_in_threadpool(_fs_list_local, path, show_hidden)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ---------------- model scanning ----------------

# short-lived stats cache so the 1s overview poll stays snappy (full SSH
# collection per machine is slow, esp. Windows). The detail modal forces a
# fresh collection on demand.
_stats_cache = {}  # key -> {"ts": ..., "data": ...}
_STATS_TTL = 5.0  # seconds


def _active_engine_is_vllm():
    """Engine toggle read for the stats/overview surfaces. Best-effort:
    a registry hiccup means llama_cpp behavior (no filtering) rather
    than an error page."""
    try:
        from engine_registry import load_settings as _engine_load
        return (_engine_load().get("engine") or "llama_cpp") == "vllm"
    except ImportError:
        return False


@app.get("/api/stats")
async def get_stats(light: bool = False):
    """Get cluster-wide machine stats (GPU, CPU, RAM). Cached briefly so the
    auto-refresh poll doesn't re-SSH every second. light=true skips the slow
    full SSH collection and returns reachability + identity instantly (the
    overview uses this; the detail modal does a full collection on demand).
    Cache is keyed by light so the two never overwrite each other.
    Engine-aware (vLLM mode): hosts the engine cannot use (windows/macos/
    intel/cpu — _vllm_capability) are EXCLUDED from data, totals and the
    machine count entirely, so the Cluster Overview only accounts for
    vLLM-serving hosts. Under llama_cpp nothing is filtered."""
    key = "light" if light else "full"
    now = time.time()
    ent = _stats_cache.get(key)
    if ent is not None and now - ent["ts"] < _STATS_TTL:
        return ent["data"]

    def collect_all():
        machines = cfg.machines.all()
        fn = stats_collector.machine_stats_light if light else stats_collector.machine_stats
        with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
            data = list(pool.map(fn, machines))
        # DEC-17: every stats entry carries the server's vLLM hardware
        # verdict (same fields as /api/machines) so the overview machine
        # cards can dim incapable hosts under vllm WITHOUT re-deriving
        # os/vendor rules client-side (B14). Stamped response-only, never
        # persisted. Under llama_cpp the UI ignores the fields entirely.
        by_name = {m["name"]: m for m in machines}
        for s in data:
            m = by_name.get(s.get("name"))
            if m is not None:
                capable, reason, info = _vllm_capability(m)
                s["vllm_capable"] = capable
                s["vllm_capability_reason"] = reason
                s["vllm_capability_info"] = info
        # vLLM mode drops incapable hosts from the overview wholesale —
        # their VRAM/RAM/cores must not inflate cluster totals and their
        # cards must not imply they could ever serve (B16). llama_cpp
        # keeps the verbatim full list.
        if _active_engine_is_vllm():
            data = [s for s in data if s.get("vllm_capable") is not False]
        return {"data": data, "totals": stats_collector.cluster_totals(data)}
    result = await run_in_threadpool(collect_all)
    _stats_cache[key] = {"ts": time.time(), "data": result}
    return result


@app.get("/api/stats/stream")
async def get_stats_stream():
    """Per-machine stats, streamed as NDJSON: one JSON line per machine
    AS ITS collection completes, then a final {totals: ...} line.

    The dashboard consumes this so one slow/offline machine can't drag
    the others: each card paints the moment its own data lands instead
    of waiting for the slowest host (the old /api/stats blocked until
    every SSH collection finished — 5s+ on Windows blobs, 6s+ on offline
    boxes). Same short cache as /api/stats per machine."""
    from fastapi.responses import StreamingResponse

    machines = cfg.machines.all()
    per_machine_cache = _stats_cache.setdefault("per-machine", {})
    now = time.time()
    # vLLM mode: incapable hosts (windows/mac/intel/cpu) never stream and
    # never enter totals — the overview accounts only for vLLM-capable
    # machines (B16; /api/stats parity)
    if _active_engine_is_vllm():
        machines = [m for m in machines
                    if _vllm_capability(m)[0] is not False]

    def gen():
        data = {}
        fresh = {}
        with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
            futures = {pool.submit(stats_collector.machine_stats, m): m
                       for m in machines}
            for fut in futures:
                m = futures[fut]
                try:
                    s = fut.result()
                except Exception:
                    s = stats_collector._empty(m.get("os", "linux"))
                s["name"] = m["name"]
                s["ip"] = m["ip"]
                # DEC-17: server's vLLM hardware verdict on every streamed
                # entry too (same fields as /api/machines; response-only).
                capable, reason, info = _vllm_capability(m)
                s["vllm_capable"] = capable
                s["vllm_capability_reason"] = reason
                s["vllm_capability_info"] = info
                data[m["name"]] = s
                fresh[m["name"]] = time.time()
                yield json.dumps(s) + "\n"
        # merge into the per-machine cache, preserving newer entries
        for name, ts in fresh.items():
            old = per_machine_cache.get(name)
            if not old or now - old["ts"] > _STATS_TTL:
                per_machine_cache[name] = {"ts": ts, "data": data[name]}
        totals = stats_collector.cluster_totals(list(data.values()))
        yield json.dumps({"totals": totals, "ts": time.time()}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _parse_win_disk(raw):
    """Parse PowerShell Get-PSDrive JSON into (free, total) human strings
    (GB or TB depending on size). Picks the drive with the most free space."""
    if not raw:
        return "", ""
    try:
        data = json.loads(raw)
    except Exception:
        return "", ""
    items = data if isinstance(data, list) else [data]
    if not items:
        return "", ""
    best = max(items, key=lambda d: d.get("Free") or 0)
    free = best.get("Free") or 0
    used = best.get("Used") or 0
    return _fmt_free(free), _fmt_free(free + used)


def _fmt_free(bytes_free):
    gb = bytes_free / 1024 ** 3
    if gb >= 1024:
        return f"{gb / 1024:.1f} TB"
    return f"{gb:.0f} GB"


def _fmt_free_kb(kb_str):
    """df -k free space (KB) -> human GB/TB string."""
    try:
        return _fmt_free(int(kb_str) * 1024)
    except (ValueError, TypeError):
        return ""


def _parse_posix_blob(detail, blob):
    """Parse the combined 'K:...\nC:...\n...' blob from a posix detail query."""
    if not blob:
        return
    for line in blob.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if key == "K":
            detail["kernel"] = val
        elif key == "C":
            detail["cpu_model"] = val
        elif key == "U":
            detail["uptime"] = val
        elif key == "I":
            detail["net"]["ip"] = val
        elif key == "S":
            detail["net"]["speed"] = val
        elif key == "D":
            detail["disk"]["free"] = _fmt_free_kb(val)
            detail["disk"]["mount"] = "/"
        elif key == "DT":
            detail["disk"]["total"] = _fmt_free_kb(val)
        elif key == "L":
            detail["llama_version"] = val


_SPEED_CACHE_FILE = os.path.join(BASE_DIR, "config", "net_speed_cache.json")
_SPEED_CACHE_TTL = 3600  # 1 hour


def _load_speed_cache():
    if not os.path.exists(_SPEED_CACHE_FILE):
        return {}
    try:
        with open(_SPEED_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_speed_cache(cache):
    try:
        tmp = _SPEED_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, _SPEED_CACHE_FILE)
    except Exception:
        pass


def _measure_ssh_speed(ssh, mb=50):
    """Measure real throughput over the existing SSH connection by having the
    remote stream `mb` MB to stdout and timing the read. Returns Mb/s."""
    import time as _t
    start = _t.time()
    try:
        code, out, _ = ssh.run(f"dd if=/dev/zero bs=1M count={mb} 2>/dev/null",
                               timeout=60)
    except Exception:
        return None
    elapsed = _t.time() - start
    if elapsed <= 0 or code != 0:
        return None
    return round(mb * 8 / elapsed)


@app.get("/api/machines/{name}/detail")
async def machine_detail(name: str):
    """Deep system info for one machine: OS/kernel, CPU model, disk, network,
    uptime, llama.cpp version, and the live stats. Used by the dashboard card
    detail modal."""
    try:
        machine = cfg.machines.require(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"machine '{name}' not found")
    is_local = local_node.is_local(machine)
    machine_os = machine.get("os", "linux")

    # reuse ONE SSH connection for all commands (opening one per command was
    # slow — each handshake is ~1s+). Local runs use subprocess directly.
    ssh = None
    if not is_local:
        try:
            from ssh_manager import SSHManager
            ssh = SSHManager(machine["ip"], machine["username"],
                             machine["password"], port=machine.get("port", 22))
        except Exception:
            ssh = None

    def _run(cmd, timeout=20):
        if is_local:
            try:
                p = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=timeout)
                return p.stdout.strip()
            except Exception:
                return ""
        if ssh is None:
            return ""
        try:
            code, out, _ = ssh.run(cmd, timeout=timeout)
            return out.strip() if code == 0 else ""
        except Exception:
            return ""

    detail = {"name": name, "is_local": is_local, "os": machine_os,
              "ip": machine.get("ip", ""), "role": machine.get("role", "peer"),
              "gpu_vendor": machine.get("gpu_vendor", "nvidia"),
              "rpc_port": machine.get("rpc_port", 50052),
              "llama_bin": machine.get("llama_bin", ""),
              "mac": (machine.get("mac") or "")}

    # ---- combine all detail queries into ONE round-trip per OS ----
    detail["net"] = {"ip": "", "speed": ""}
    detail["disk"] = {"free": "", "mount": ""}
    # RDMA transport info: how this machine's RPC connection is served
    detail["rdma"] = {
        "enabled": bool(machine.get("rdma")),
        "transport": ("RDMA (rdma_cm, native)" if machine.get("rdma") and machine.get("rdma_cm")
                      else "RDMA (auto-negotiate)" if machine.get("rdma")
                      else "TCP"),
        "ip": machine.get("rdma_ip", ""),
        "dev": machine.get("rdma_dev", ""),
        "gid": machine.get("rdma_gid", ""),
    }
    detail["disk"] = {"free": "", "mount": ""}
    if machine_os == "windows":
        blob = _run(
            "powershell -NoProfile -Command \""
            "$o=@{};"
            "$o.kernel=(ver);"
            "$o.cpu=(Get-CimInstance Win32_Processor).Name;"
            "$o.uptime=((Get-Date)-(Get-CimInstance Win32_OperatingSystem).LastBootUpTime).ToString();"
            "$o.ip=((Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*'} | Select-Object -First 1).IPAddress);"
            "$o.speed=((Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object -First 1).LinkSpeed);"
            "$o.disk=(Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Free -gt 0} | Select-Object Name,Free,Used | ConvertTo-Json -Compress);"
            "$o.cores=(Get-CimInstance Win32_Processor | Measure-Object -Sum NumberOfLogicalProcessors).Sum;"
            "$o.coreload=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object {$_.Name -ne '_Total'} | Select-Object -ExpandProperty PercentProcessorTime | ForEach-Object {[math]::Round($_,1)} | ConvertTo-Json -Compress);"
            "$o.llama=((& (Join-Path $env:USERPROFILE 'llama.cpp\\bin\\llama-server.exe') --version 2>&1 | Select-String 'version:' | Select-Object -First 1).ToString());"
            "if (-not $o.llama) { $o.llama=((& (Join-Path $env:USERPROFILE 'llama.cpp\\llama-server.exe') --version 2>&1 | Select-String 'version:' | Select-Object -First 1).ToString()); }"
            "$o | ConvertTo-Json -Compress\"", timeout=30)
        try:
            o = json.loads(blob)
            detail["kernel"] = o.get("kernel", "")
            detail["cpu_model"] = o.get("cpu", "")
            detail["uptime"] = o.get("uptime", "")
            detail["net"]["ip"] = o.get("ip", "")
            detail["net"]["speed"] = o.get("speed", "")
            detail["disk"]["free"], detail["disk"]["total"] = _parse_win_disk(o.get("disk", ""))
            detail["llama_version"] = o.get("llama", "")
            # coreload comes back as a JSON string; parse it into a list
            cl = o.get("coreload")
            if isinstance(cl, str):
                try:
                    detail["cpu_cores"] = json.loads(cl)
                except Exception:
                    detail["cpu_cores"] = []
            else:
                detail["cpu_cores"] = cl or []
            detail["cores"] = o.get("cores", "")
        except Exception:
            pass
    elif machine_os == "macos":
        blob = _run(
            "echo \"K:$(uname -a)\";"
            "echo \"C:$(sysctl -n machdep.cpu.brand_string)\";"
            "echo \"U:$(uptime)\";"
            "echo \"I:$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)\";"
            "echo \"S:$(networksetup -getmedia en0 2>/dev/null | grep -i 'current' | sed 's/.*: //')\";"
            "echo \"D:$(df -k / | awk 'NR==2 {print $4}')\";"
            "echo \"DT:$(df -k / | awk 'NR==2 {print $2}')\";"
            "echo \"L:$(\\\"{bin}/llama-server\\\" --version 2>&1 || "
            "\\\"{bin}/llama-cli\\\" --version 2>&1)\"".format(
                bin=machine.get('llama_bin', '') or '.'), timeout=30)
        _parse_posix_blob(detail, blob)
    else:
        _bin = machine.get('llama_bin', '')
        blob = _run(
            f"K=$(uname -a); "
            f"C=$(lscpu | grep 'Model name' | cut -d: -f2 | sed 's/^ *//'); "
            f"U=$(uptime -p); "
            f"I=$(hostname -I 2>/dev/null | grep -oF '{machine['ip']}' | head -n1); "
            f"S=$(for i in $(ls /sys/class/net | grep -v '^lo'); do "
            f"d=$(readlink /sys/class/net/$i/device/driver 2>/dev/null); "
            f"case \"$i\" in veth*|br-*|docker*|virbr*|tailscale*) continue;; esac; "
            f"case \"$d\" in *bridge*|*veth|*tun|*tap) continue;; esac; "
            f"s=$(cat /sys/class/net/$i/speed 2>/dev/null); "
            f"if [ -n \"$s\" ] && [ \"$s\" != \"-1\" ] && [ \"$s\" -gt 0 ]; then "
            f"if [ \"$s\" -ge 1000 ]; then echo \"$((s/1000)) Gb/s\"; "
            f"else echo \"${{s}} Mb/s\"; fi; break; fi; "
            f"e=$(ethtool $i 2>/dev/null | grep -i speed | sed 's/.*: *//;s/ *$//'); "
            f"if [ -n \"$e\" ] && [ \"$e\" != \"Unknown!\" ]; then echo \"$e\"; break; fi; "
            f"case \"$d\" in *virtio*) echo \"virtio (speed n/a)\"; break;; esac; done); "
            f"D=$(df -k / | tail -n1 | awk '{{print $4}}'); "
            f"DT=$(df -k / | tail -n1 | awk '{{print $2}}'); "
            f"L=$(\"{_bin}/llama-server\" --version 2>&1 || \"{_bin}/llama-cli\" --version 2>&1); "
            f"echo \"K:$K\"; echo \"C:$C\"; echo \"U:$U\"; echo \"I:$I\"; "
            f"echo \"S:$S\"; echo \"D:$D\"; echo \"DT:$DT\"; echo \"L:$L\"", timeout=30)
        _parse_posix_blob(detail, blob)

    # ---- real throughput over SSH (cached 1h) ----
    # virtio/virtual NICs don't report a link speed, so measure what the
    # machine can actually push to the cluster over its SSH link. For the
    # local machine, measure to the first reachable peer.
    speed_ssh = None
    if not is_local and ssh is not None:
        speed_ssh = ssh
    elif is_local:
        for peer in cfg.machines.all():
            if peer.get("name") == name or local_node.is_local(peer):
                continue
            try:
                from ssh_manager import SSHManager
                speed_ssh = SSHManager(peer["ip"], peer["username"],
                                        peer["password"],
                                        port=peer.get("port", 22))
                break
            except Exception:
                continue
    # only measure over SSH when the reported link speed is missing/unknown
    # (virtio/virtual NICs). Machines with a real reported speed keep it.
    reported = str(detail["net"].get("speed") or "").strip()
    need_measure = (not reported or "n/a" in reported.lower()
                    or "unknown" in reported.lower())
    if need_measure and speed_ssh is not None:
        cache = _load_speed_cache()
        ent = cache.get(name)
        now = time.time()
        if ent and now - ent.get("ts", 0) < _SPEED_CACHE_TTL:
            detail["net"]["speed"] = ent.get("mbps")
        else:
            mbps = _measure_ssh_speed(speed_ssh)
            if mbps:
                detail["net"]["speed"] = mbps
                cache[name] = {"mbps": mbps, "ts": now}
                _save_speed_cache(cache)
        if detail["net"]["speed"]:
            detail["net"]["speed"] = f"{detail['net']['speed']} Mb/s (SSH)"

    # live stats ride along so the modal shows current numbers
    try:
        detail["stats"] = stats_collector.machine_stats(machine)
    except Exception:
        detail["stats"] = None
    return detail


class SafeDefaultsIn(pydantic.BaseModel):
    path: str = ""
    group: str = ""
    servers: list[str] = []


@app.get("/api/safe-defaults")
async def safe_defaults(path: str = "", group: str = "", servers: str = ""):
    """Safe defaults for a new model — hardware-aware. Uses live free VRAM/RAM
    (total−used) and GGUF header (is_gemma, has_mtp, layers) to pick ctx, batch, cache, etc.
    Only called from Scan → Configure (not Scan)."""
    import os as _os
    # resolve hardware set
    target_machines = []
    if group:
        g = cfg.groups.get(group)
        if g:
            for nm in g.get("machines") or []:
                try:
                    target_machines.append(cfg.machines.require(nm))
                except Exception:
                    pass
    elif servers:
        for nm in [s.strip() for s in servers.split(",") if s.strip()]:
            try:
                target_machines.append(cfg.machines.require(nm))
            except Exception:
                pass
    if not target_machines:
        target_machines = cfg.machines.all()
    # live stats for those machines
    def get_stats():
        with ThreadPoolExecutor(max_workers=len(target_machines) or 1) as pool:
            data = list(pool.map(stats_collector.machine_stats, target_machines))
        return data, stats_collector.cluster_totals(data)
    try:
        per_machine, totals = await run_in_threadpool(get_stats)
    except Exception:
        totals = {"vram": {"total": 0, "used": 0}, "ram": {"total": 0, "used": 0}}
        per_machine = []
    free_vram = max(0, totals.get("vram", {}).get("total", 0) - totals.get("vram", {}).get("used", 0))
    free_ram = max(0, totals.get("ram", {}).get("total", 0) - totals.get("ram", {}).get("used", 0))
    # heuristic: master cores for threads
    try:
        master = next((m for m in target_machines if m.get("role")=="master"), target_machines[0])
        threads = int((master.get("detected") or {}).get("cores") or master.get("cores") or 8)
    except Exception:
        threads = 8
    threads = max(4, min(16, threads))
    # batch heuristic: more VRAM → larger batch
    batch = 1024
    ubatch = 1024
    # ctx heuristic: keep KV under ~30% of free (VRAM+0.5*RAM) minus model size
    # need model header?
    ctx_size = 16384
    cache_k = "q8_0"
    cache_v = "q8_0"
    is_gemma = False
    is_embed = False
    is_decision = False
    has_mtp = False
    has_mmproj = False
    if path:
        # embed detection: filename contains embed/nomic/bge/gte/minilm
        # or model type is embedding (heuristic)
        pl = path.lower()
        if any(t in pl for t in ("embed", "bge", "gte", "minilm")):
            is_embed = True
        try:
            real = path_resolver.resolve(path)
            real_exp = _os.path.expanduser(real)
            # TypeSafe decision models (laya/clef/...): detected from the
            # GGUF's <arch>.decision.type — the same gate llama.cpp uses.
            # These serve /v1/systemone only (no generation), so chat-side
            # heuristics (KV estimator, gemma cache) must not run for them.
            if gguf_inspect.get_decision_type(real_exp):
                is_decision = True
            # find companion mmproj in same dir
            d = _os.path.dirname(real_exp)
            if _os.path.isdir(d):
                for fn in _os.listdir(d):
                    if fn.startswith("mmproj") and fn.endswith(".gguf"):
                        has_mmproj = True
                        break
            # read header for gemma/mtp
            if _os.path.exists(real_exp):
                try:
                    meta = gguf_inspect.metadata(real_exp)
                    # is_gemma: check arch
                    for k in meta.keys():
                        if "gemma" in str(k).lower() or "gemma" in str(meta.get(k,"")).lower():
                            is_gemma = True
                    # also check model name
                    if "gemma" in path.lower():
                        is_gemma = True
                    has_mtp = gguf_inspect.has_mtp_layers_anywhere(real_exp)
                except Exception:
                    pass
        except Exception:
            pass
    if is_gemma:
        cache_k = "f16"
        cache_v = "f16"
    # ctx: if we can estimate KV bytes, pick safe (chat models only —
    # embed uses fixed small-window values below, never the estimator)
    if path and not is_embed and free_vram + free_ram > 0:
        try:
            real = path_resolver.resolve(path)
            real_exp = _os.path.expanduser(real)
            if _os.path.exists(real_exp):
                meta = gguf_inspect.metadata(real_exp)
                # estimate KV bytes per token: 2*layers*kv_heads*head_dim*bytes
                import struct as _struct
                n_layer = 0
                kv_heads = 0
                head_dim = 128
                for k,v in meta.items():
                    kl = str(k).lower()
                    if kl.endswith("block_count") and isinstance(v,int):
                        n_layer = v
                    if kl.endswith("head_count_kv") and isinstance(v,int):
                        kv_heads = v
                    if kl.endswith("key_length") and isinstance(v,int):
                        head_dim = v
                if n_layer and kv_heads:
                    # bytes for q8_0 vs f16
                    bpe = 1 if cache_k=="q8_0" else 2
                    kv_per_token = 2 * n_layer * kv_heads * head_dim * bpe
                    # model size
                    try:
                        w = sum(gguf_inspect.weight_bytes(p) for p in gguf_inspect.shard_paths(real_exp)) / 1e9
                    except Exception:
                        w = 8
                    free_for_kv = (free_vram + free_ram*0.3) - w - 3  # 3GB OS overhead
                    if free_for_kv > 2:
                        # choose ctx that fits 30% of free_for_kv
                        max_ctx = int((free_for_kv * 1e9 * 0.3) / max(kv_per_token,1))
                        # clamp safe 8192..32768
                        ctx_size = max(8192, min(32768, max_ctx // 1024 * 1024))
                    # HBM-rich hosts can go larger, but safe stays 16k
        except Exception:
            pass
    flags = {
        "ctx-size": ctx_size,
        "threads": threads,
        "flash-attn": "on",
        "cache-type-k": cache_k,
        "cache-type-v": cache_v,
        "batch-size": batch,
        "ubatch-size": ubatch,
        "parallel": 1,
        "jinja": True,
        "split-mode": "layer",
        "n-gpu-layers": 99,
        # verbose logging NO LONGER forced: -v floods thousands of
        # "node #N" sched lines per ubatch (severe slowdown at 27B scale).
        # Add -v/--log-timestamps/--log-prefix per-model only when testing.
    }
    if is_embed:
        # Embedding safe defaults (distinct from chat):
        # - whole prompt must fit in ONE ubatch (llama-server clamps
        #   n_batch down to n_ubatch otherwise), so keep the pair equal
        #   and sized to the window (upstream example serves with -ub 8192)
        # - no jinja/draft/reasoning: embedding-only endpoint, no sampling
        # - flash-attn left at default (auto), not forced on
        # - KV stays q8_0 (tiny for embed), full offload, single slot
        flags["embeddings"] = True
        flags.pop("jinja", None)
        flags.pop("flash-attn", None)
        ctx_size = 8192
        flags["ctx-size"] = 8192
        flags["batch-size"] = 8192
        flags["ubatch-size"] = 8192
        flags["parallel"] = 1
        # nomic v1.5 long-context recipe (2048 native -> 8192 via yarn)
        if "nomic" in path.lower():
            flags["rope-scaling"] = "yarn"
            flags["rope-freq-scale"] = 0.75
    if has_mmproj:
        # leave mmproj to user to pick — don't auto-fill, just hint via flags? keep unset
        pass
    if is_decision:
        # TypeSafe decision-model safe defaults (distinct from chat AND
        # embed). llama.cpp server docs (tools/server/README.md, System
        # One section): these models answer via /v1/systemone only — no
        # text generation, no sampling, no chat template.
        # - for laya/clef the WHOLE prompt is evaluated in one batch, so
        #   --ubatch-size must cover state+questions (>= 2048 headroom;
        #   long prompts are truncated to the model's max_head_tokens)
        # - ctx 8192 = laya's modern-bert.context_length
        # - no jinja/flash-attn/KV-quant/MTP: classifier head, not a
        #   generator; the decision temperatures are baked into the GGUF
        #   (<arch>.decision.temperature.*), sampling flags are no-ops
        # - full offload: 421M (laya) fits any GPU trivially
        for k in ("jinja", "flash-attn", "cache-type-k", "cache-type-v",
                  "batch-size", "ubatch-size", "ctx-size", "threads",
                  "split-mode"):
            flags.pop(k, None)
        flags.update({
            "ctx-size": 8192,
            "ubatch-size": 2048,
            "parallel": 1,
            "n-gpu-layers": 99,
        })
    # never auto-add draft-mtp unless has_mtp
    return {
        "path": path,
        "flags": flags,
        "hardware": {"free_vram": round(free_vram,1), "free_ram": round(free_ram,1), "threads": threads, "is_gemma": is_gemma, "is_embed": is_embed, "is_decision": is_decision, "has_mtp": has_mtp},
        "note": (f"TypeSafe decision model — serves /v1/systemone only (no chat generation)"
                 if is_decision else
                 f"safe for {free_vram:.1f} GB free VRAM + {free_ram:.1f} GB free RAM — tweak ctx/batch/cache based on live free")
    }


class LaunchIn(pydantic.BaseModel):
    model_id: str
    servers: list[str]
    port: int = 8080
    variant: str | None = None   # label of additional config; when set, use that variant's deployment
    master_name: str | None = None  # which server runs llama-server (Launch page "run on" picker)
    mode: str = "auto"           # rpc | local | auto — auto = generator heuristic decides from hardware


@app.post("/api/launch")
async def launch_model(l: LaunchIn):
    """Live launch: orchestrate everything from model config, no scripts.
    If variant is set, the variant's deployment is merged with the base model."""
    # B5a2 + DEC-06: the single-model launch path is engine-owned — the
    # adapter either runs it (llama.cpp: rpc_manager.launch_model) or
    # refuses it (vLLM serving routes through llama-swap /api/swap/*).
    # The refusal surfaces as 409, never a silent no-op, and fires before
    # any server-list validation (the mode gate is first).
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if l.mode not in ("rpc", "local", "auto"):
        raise HTTPException(status_code=400,
                            detail=f"invalid mode '{l.mode}' (rpc|local|auto)")
    if l.master_name and l.master_name not in l.servers:
        raise HTTPException(status_code=400,
                            detail=f"master '{l.master_name}' is not among the servers")
    # access.py: same "never existed" rule as /api/chat — launch is a
    # model-scoped action, so a restricted (strict-mode) viewer may only
    # launch/deploy tokens in their key's allowlist (EXACT match: a
    # variant grant does NOT widen to the base card; base launch = base
    # token on the grid, variant launch = mid--label). Unknown ids keep
    # the not-found path below (cfg.get_model in the engine).
    lmid = (l.model_id or "").strip()
    if not access.gate_open():
        base_mid = lmid.split("--")[0]
        if _raw_is_private(base_mid):
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
        allow = access.view_allow()
        if allow is not None and lmid not in allow:
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
    if not engine.capable("rpc_cluster"):
        try:
            engine.start_model(l.model_id, l.servers, l.port, l.variant)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
    valid = {m["name"] for m in cfg.machines.all()}
    unknown = [s for s in l.servers if s not in valid]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"unknown machine(s): {', '.join(unknown)}")
    if not l.servers:
        raise HTTPException(status_code=400, detail="pick at least one server")
    mode, master_name = l.mode, l.master_name
    if mode == "auto":
        # Generator decides: single-box-on-a-remote-peer deployments run
        # local on that box (no RPC hops); anything else stays rpc.
        # Unknown/ad-hoc ids keep the legacy rpc default.
        try:
            verdict = swap_manager.deployment_launch_mode(
                l.model_id, l.variant)
            mode = verdict["mode"]
            if mode == "local" and verdict["box"] and not master_name:
                master_name = verdict["box"]
        except ValueError:
            mode = "rpc"
    report = await run_in_threadpool(
        engine.start_model, l.model_id, l.servers, l.port, l.variant,
        master_name, mode)
    status = 500 if report.get("failed") else 200
    return JSONResponse(report, status_code=status)


@app.post("/api/launch/stop")
async def stop_launch(l: LaunchIn):
    """Stop a live-launched model on its servers."""
    # B15 + DEC-06: the stop path is engine-owned. llama.cpp:
    # rpc_manager.stop_model kills llama.cpp RPC ports. vLLM models are
    # served BY llama-swap, so the engine stop is llama-swap's model
    # unload: the adapter runs the entry's cmdStop (docker stop / pkill,
    # vllm_manager.build_cmd_stop) through the swap-host unload proxy —
    # exactly what the Swap tab's Evict does. Swap down -> 503.
    # access.py: same "never existed" rule as /api/launch — a scoped
    # strict-mode key may only stop tokens it is granted (stopping is a
    # model-scoped action; a variant grant does not cover the base key).
    # Unauthenticated/unknown keys under an open gate keep the legacy
    # no-op path (their stop is harmless: nothing they name resolves).
    lmid = (l.model_id or "").strip()
    if not access.gate_open():
        base_mid = lmid.split("--")[0]
        if _raw_is_private(base_mid):
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
        allow = access.view_allow()
        if allow is not None and lmid not in allow:
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if not engine.capable("rpc_cluster"):
        try:
            return await run_in_threadpool(
                engine.stop_model, l.model_id, l.servers, l.variant,
                None, l.port)
        except vllm_manager.SwapDownError as e:
            raise HTTPException(status_code=503, detail=str(e)) from None
    valid = {m["name"] for m in cfg.machines.all()}
    unknown = [s for s in l.servers if s not in valid]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"unknown machine(s): {', '.join(unknown)}")
    # Serve-group truth: a remote-proxy (single-box-on-a-peer) deployment
    # is owned by llama-swap — its server lives on the peer box, so
    # rpc_manager.stop_model (master llama-server + peer rpc-servers)
    # would miss it. Route through the swap unload proxy (cmdStop) so the
    # stop actually unloads it and the serve group frees the box.
    # Swap down -> 503, same as the Swap tab's Evict. Unknown ids keep
    # the legacy path.
    try:
        verdict = swap_manager.deployment_launch_mode(l.model_id, l.variant)
    except ValueError:
        verdict = None
    if verdict and verdict["mode"] == "local" and verdict["box"]:
        return await _proxy_swap_unload(verdict["swap_key"])
    return await run_in_threadpool(rpc_manager.stop_model, l.servers,
                                   l.model_id, l.variant, l.master_name,
                                   l.port)


@app.get("/api/launch/active")
async def active_launches():
    """Which configured model ports are serving right now (any machine).
    DEC-06: the per-machine serving probe is the adapter's active_model —
    llama.cpp checks the machine's RPC port, a swap-only engine probes
    HTTP /health on the llama-swap listen port (swap_manager
    listen_port — no hardcoded 8080)."""
    machines = cfg.machines.all()
    try:
        engine = engine_registry.get_engine()
    except ValueError:
        engine = engine_registry.get_engine({"engine": "llama_cpp"})
    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        return list(pool.map(engine.active_model, machines))


# ---------------- llama-swap: launch all ----------------
class SwapIn(pydantic.BaseModel):
    binary: str | None = None
    listen_port: int | None = None
    config_path: str | None = None
    # DEC-03 (AUDIT-T6 §1.6): vllm's swap-config file. Empty string clears
    # back to derive-from-config_path (engine-qualified sibling file).
    config_path_vllm: str | None = None
    host_name: str | None = None
    launch_config: dict | None = None


@app.get("/api/swap/settings")
async def swap_settings():
    try:
        return swap_manager.load_settings()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/swap/settings")
async def save_swap_settings(s: SwapIn):
    return swap_manager.save_settings(
        s.binary, s.listen_port, s.config_path, s.host_name,
        s.launch_config, config_path_vllm=s.config_path_vllm)


@app.get("/api/swap/preview")
async def swap_preview(config_path: str | None = None,
                       host_name: str | None = None):
    """Generated llama-swap config (no side effects)."""
    # F4 (t_90def951): engine×OS gate — the preview is the plan the launch
    # would push, so engine-impossible hosts surface here as warnings
    # (validate_hosts has no call site until now). Non-fatal: the plan is
    # still returned so the operator can see exactly what is wrong.
    try:
        engine = engine_registry.get_engine()
        host_errors = []
    except ValueError:
        engine = None
        host_errors = []
    if engine is not None:
        try:
            host_errors = engine.validate_hosts(
                [m for m in cfg.machines.all()])
        except Exception:
            host_errors = []
        if host_errors:
            print("[swap-preview] engine host gate: "
                  + "; ".join(host_errors))
    try:
        # DEC-06: the plan is the adapter's launch_plan (engine-owned
        # planner — swap_manager.launch_plan behind the adapter seam).
        plan = engine.launch_plan(config_path or None, host_name or None) \
            if engine is not None \
            else swap_manager.launch_plan(config_path or None,
                                          host_name or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if engine is not None and host_errors:
        plan = dict(plan)
        plan["warnings"] = list(plan.get("warnings") or []) + host_errors
    return plan


@app.post("/api/swap/push")
async def swap_push():
    """Regenerate + write swap config/scripts from current model configs.
    No peer/daemon restart; already-loaded instances keep serving until
    evicted, next load picks up the new scripts."""
    return await run_in_threadpool(swap_manager.push_scripts)


@app.get("/api/swap/reload-status")
async def swap_reload_status():
    """Outcome of the save-triggered daemon reload (polled by the UI)."""
    return dict(swap_manager.reload_state)


@app.post("/api/swap/launch")
async def swap_launch(s: SwapIn):
    """Generate + push config, start RPC peers, start llama-swap daemon."""
    try:
        report = await run_in_threadpool(
            swap_manager.launch, s.binary, s.listen_port, s.config_path,
            s.host_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    status = 500 if report.get("failed") else 200
    return JSONResponse(report, status_code=status)


@app.post("/api/swap/stop")
async def swap_stop():
    """Stop the llama-swap daemon AND any remote-proxy llama-servers
    (launch-mode=local models run llama-server on their own box — swap
    going down must not leave those orphans serving). stop_rpc(name)
    kills llama-swap/llama-server/rpc-server on every machine, both OS
    types; run it everywhere after the daemon is down."""
    out = await run_in_threadpool(swap_manager.stop)
    names = [m["name"] for m in cfg.machines.all()]
    with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
        out["machines"] = list(pool.map(rpc_manager.stop_rpc, names))
    return out


@app.get("/api/swap/status")
async def swap_status():
    return swap_manager.status()


@app.get("/api/swap/loaded")
async def swap_loaded_models(request: Request):
    """Currently loaded model(s) — FILTERED BY CALLER IDENTITY:
    admins/console see every loaded model; a key holder sees only the
    ones their key grants (whose model loaded/evicted when is household
    telemetry a non-admin never sees). Same llama-swap /v1/models read
    (status.value == "loaded"), empty when swap is down."""
    st = swap_manager.status()
    if not st.get("running"):
        return {"loaded": [], "running": False}
    def fetch():
        import urllib.request, json as _json
        import access as _a
        url = f"http://127.0.0.1:{st['port']}/v1/models" \
            if local_node.is_local(cfg.machines.require(st["host"])) \
            else f"http://{st.get('host_ip')}:{st['port']}/v1/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.load(r)
        ids = [m["id"] for m in data.get("data", [])
               if (m.get("status") or {}).get("value") == "loaded"]
        if _a.is_admin(None) or _a.admin_sees_everything():
            return ids
        # scoped caller: filter against THEIR grants. The middleware
        # stashed the raw record for this request — use it (no token
        # passes through user code)
        rec = _a.raw_key_record()
        if rec is None:
            return []
        allow = rec.get("models") or set()
        from swap_gateway import _mid_allowed
        return [i for i in ids if _mid_allowed(i, allow)]
    try:
        return {"loaded": await run_in_threadpool(fetch),
                "running": True, "host": st["host"], "port": st["port"]}
    except Exception as e:
        return {"loaded": [], "running": True, "error": str(e)}


@app.get("/api/flow/vllm-metrics")
async def flow_vllm_metrics():
    """Visualizer feed for engine=vllm: scrape the loaded upstream's
    /metrics (Prometheus text) and return the gauges + counters the flow
    diagram renders — per-model token counters (deltas computed from the
    previous scrape, monotonic within this app's lifetime), KV-cache
    usage, queue depth, prefix-cache hit rate. llama-swap exposes the
    upstream port per model; only loaded models answer, so unloaded ones
    report nulls. Same urllib-in-threadpool idiom as /api/swap/loaded."""
    global _FLOW_VLLM_METRICS_PREV
    st = swap_manager.status()
    if not st.get("running"):
        return {"models": {}, "running": False}

    def fetch():
        import urllib.request, json as _json
        base = (f"http://127.0.0.1:{st['port']}"
                if local_node.is_local(cfg.machines.require(st["host"]))
                else f"http://{st.get('host_ip')}:{st['port']}")
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            models = _json.load(r).get("data", [])
        loaded = [m["id"] for m in models
                  if (m.get("status") or {}).get("value") == "loaded"]
        out = {}
        for mid in loaded:
            # the upstream serves on its own port; llama-swap's /upstream/
            # <id> proxy path works without knowing that port
            try:
                with urllib.request.urlopen(
                        f"{base}/upstream/{mid}/metrics", timeout=5) as r:
                    body = r.read().decode()
                out[mid] = _parse_vllm_metrics(body)
            except Exception:
                out[mid] = None
        return out

    try:
        fresh = await run_in_threadpool(fetch)
    except Exception as e:
        return {"models": {}, "running": True, "error": str(e)}
    # per-model counters are deltas vs the previous scrape (tokens/sec is
    # the visualizer's animation driver; raw counters would never decay)
    result = {}
    # Identity gate (same rule as /api/swap/loaded): metrics per model
    # are serving telemetry — a scoped viewer sees only granted models
    # (deltas still computed from the FULL previous scrape so a granted
    # model's counters stay monotonic-correct across the filter).
    _flow_allow = None if access.gate_open() else access.view_allow()
    from swap_gateway import _mid_allowed as _mid_ok
    for mid, cur in fresh.items():
        if _flow_allow is not None and not (
                _mid_ok(mid, _flow_allow) or not cur):
            # ungranted model: drop its row entirely, but only when it is
            # serving (a None row tells the UI "loaded but quiet" — that
            # state must not leak either, so drop those too)
            continue
        if not cur:
            result[mid] = None
            continue
        prev = (_FLOW_VLLM_METRICS_PREV or {}).get(mid)
        result[mid] = cur
        if prev:
            result[mid]["prompt_tps"] = max(
                0.0, cur["prompt_tokens"] - prev["prompt_tokens"])
            result[mid]["gen_tps"] = max(
                0.0, cur["gen_tokens"] - prev["gen_tokens"])
        else:
            result[mid]["prompt_tps"] = 0.0
            result[mid]["gen_tps"] = 0.0
    _FLOW_VLLM_METRICS_PREV = fresh
    return {"models": result, "running": True,
            "host": st["host"], "port": st["port"]}


def _parse_vllm_metrics(body):
    """Scrape the handful of vLLM gauges the visualizer renders. Counter
    names are model_name-tagged (one scrape per loaded upstream port, so
    there is exactly one model per body)."""
    gauges = {"kv_cache_usage_perc": None, "num_requests_running": None,
              "num_requests_waiting": None, "prompt_tokens": None,
              "gen_tokens": None, "prefix_cache_hits": None,
              "prefix_cache_queries": None}
    prefixes = {
        "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
        "vllm:num_requests_running": "num_requests_running",
        "vllm:num_requests_waiting": "num_requests_waiting",
        "vllm:prompt_tokens_total{": "prompt_tokens",
        "vllm:generation_tokens_total{": "gen_tokens",
        "vllm:prefix_cache_hits_total{": "prefix_cache_hits",
        "vllm:prefix_cache_queries_total{": "prefix_cache_queries",
    }
    for line in (body or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, raw = line.rpartition(" ")
        if not name:
            continue
        for p, key in prefixes.items():
            if name.startswith(p):
                try:
                    gauges[key] = float(raw)
                except ValueError:
                    pass
                break
    if all(v is None for v in gauges.values()):
        return None
    for k in ("prompt_tokens", "gen_tokens", "prefix_cache_hits",
              "prefix_cache_queries"):
        gauges[k] = gauges[k] or 0.0
    return gauges


# last scrape's per-model counters (module state — /metrics counters are
# cumulative; the visualizer needs tokens-per-scrape deltas)
_FLOW_VLLM_METRICS_PREV = None


@app.post("/api/swap/load/{model_id}")
async def swap_load_model(model_id: str):
    """Load one model through the running swap daemon: vLLM models load
    on first request, so this fires a minimal warmup completion
    (max_tokens=1) at the daemon and waits for it. First load takes
    minutes (Ray cluster + weight load + warmup); the daemon-side
    timeout is generous for that reason. 404 for unknown models, 503
    when the daemon is down. Same urllib-in-threadpool idiom as
    swap_loaded_models (stdlib only, no extra deps)."""
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503,
                            detail="swap daemon is not running — Launch All first")
    import time as _time

    def warmup():
        import urllib.request, urllib.error, json as _json
        base = (f"http://127.0.0.1:{st['port']}"
                if local_node.is_local(cfg.machines.require(st["host"]))
                else f"http://{st.get('host_ip')}:{st['port']}")
        payload = _json.dumps(
            {"model": model_id,
             "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 1}).encode()
        req = urllib.request.Request(
            base + "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = _time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                body = _json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"daemon refused load: HTTP {e.code} "
                               f"{e.read(300).decode('utf-8', 'replace')}")
        choice = ((body.get("choices") or [{}])[0].get("message") or {})
        return {"ok": True, "model": model_id,
                "elapsed_s": round(_time.time() - t0, 1),
                "preview": (choice.get("content") or "")[:120]}
    try:
        return await run_in_threadpool(warmup)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(status_code=504,
                            detail=f"load timed out or daemon unreachable: {e}") from None


class PinIn(pydantic.BaseModel):
    pinned: bool = False


@app.post("/api/models/{model_id}/pin")
async def pin_model(model_id: str, p: PinIn):
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    m["pinned"] = bool(p.pinned)
    # ttl 0 = never for pinned, else default 600
    if p.pinned:
        m["ttl"] = 0
    return cfg.save_model(model_id, m)


class SwapDisableIn(pydantic.BaseModel):
    disabled: bool = False


@app.post("/api/models/{model_id}/swap-disable")
async def swap_disable_model(model_id: str, p: SwapDisableIn):
    """Launch-page disable switch: when true, the model (config-level,
    so ALL deployments inherit — variants and primary) is omitted from
    every push/preview of the swap config until re-enabled. Pure config
    flag; applying it to a running daemon takes the next push (the swap
    push / Launch All flow already ends in one)."""
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    m["swap_disabled"] = bool(p.disabled)
    return cfg.save_model(model_id, m)


@app.get("/api/models/{model_id}/swap-disable")
async def swap_disable_get(model_id: str):
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return {"model_id": model_id, "swap_disabled": bool(m.get("swap_disabled"))}


async def _proxy_swap_unload(model_id: str):
    """POST /api/models/unload/{model_id} on the swap host — llama-swap then
    runs the model entry's cmdStop, which under vLLM IS the engine stop
    (`docker stop farm-<id>` / pkill, vllm_manager.build_cmd_stop). Shared by
    /api/swap/evict/{model_id} and the vllm branch of /api/launch/stop (B15):
    one proxy body so the two entry points can never drift. No-op if swap down.
    DEC-06: the body lives in vllm_manager.swap_host_unload (the adapter's
    stop path calls the same function); app.py only maps SwapDownError -> 503."""
    try:
        return await run_in_threadpool(vllm_manager.swap_host_unload, model_id)
    except vllm_manager.SwapDownError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


@app.post("/api/swap/evict/{model_id}")
async def swap_evict(model_id: str):
    """Unload a single model from swap (frees its VRAM/RAM). Proxies to
    POST /api/models/unload/{model_id} on the swap host. No-op if swap down."""
    return await _proxy_swap_unload(model_id)


@app.post("/api/swap/evict-all")
async def swap_evict_all():
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = f"http://127.0.0.1:{st['port']}/api/models/unload" if is_local else f"http://{st.get('host_ip')}:{st['port']}/api/models/unload"
    def do():
        import urllib.request, urllib.error
        req = urllib.request.Request(url, data=b"", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return {"ok": True, "status": r.status, "body": r.read().decode()[:500]}
        except urllib.error.HTTPError as e:
            return {"ok": False, "status": e.code, "error": e.read().decode()[:500]}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return await run_in_threadpool(do)


# ---------------- embeddings test ----------------
@app.get("/api/embeddings/models")
async def list_embedding_models():
    # capability detection is the engine adapter's vocabulary (G-6):
    # llama.cpp keeps the "embed"-substring heuristic, vLLM extends it
    # with the known embedding families (bge/nomic/gte/…) + config.json
    # architectures. Same shape both ways.
    engine = engine_registry.get_engine()
    return engine.filter_embedding_models(cfg.list_models())


class EmbedIn(pydantic.BaseModel):
    model: str
    input: str | list[str]
    # optional second input for similarity UI, handled client-side


@app.post("/api/embeddings")
async def create_embeddings(e: EmbedIn):
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running — Launch All first")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = "http://127.0.0.1:8090/v1/embeddings"  # GATEWAY: key-auth + usage tracking for ALL farm-proxied generate paths
    def do():
        import urllib.request, json as _json, urllib.error
        payload = _json.dumps({"model": e.model, "input": e.input}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
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
@app.get("/api/audio/models")
async def list_audio_models():
    # capability detection is the engine adapter's vocabulary (G-8)
    engine = engine_registry.get_engine()
    return engine.filter_audio_models(cfg.list_models())


@app.get("/api/decision/models")
async def list_decision_models():
    """TypeSafe decision models (llama.cpp /v1/systemone). Detected from
    the GGUF's <arch>.decision.type — the same gate llama.cpp uses —
    via the engine adapter, so a future engine just filters its own
    vocabulary."""
    engine = engine_registry.get_engine()
    return engine.filter_decision_models(cfg.list_models())


class AudioIn(pydantic.BaseModel):
    model: str
    audio: str  # data URI (data:audio/wav;base64,...) from browser MediaRecorder
    language: str = "English"  # force ASR language ("None" = auto-detect)


@app.post("/api/audio")
async def audio_test(v: AudioIn):
    """Transcribe browser-captured audio via OpenAI-style input_audio
    (base64 data URI) in a chat completion. Audio never touches disk."""
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running — Launch All first")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = "http://127.0.0.1:8090/v1/chat/completions"  # GATEWAY: key-auth + usage tracking for ALL farm-proxied generate paths
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
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
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
@app.get("/api/vision/models")
async def list_vision_models():
    # capability detection is the engine adapter's vocabulary (G-9):
    # llama.cpp filters by the mmproj projector flag; vLLM vision models
    # are HF-format (no mmproj) and are detected from config.json
    # vision_config / architectures + known vision families.
    engine = engine_registry.get_engine()
    return engine.filter_vision_models(cfg.list_models())


class VisionIn(pydantic.BaseModel):
    model: str
    prompt: str
    image: str  # data URI (data:image/png;base64,...) built client-side


@app.post("/api/vision")
async def vision_test(v: VisionIn):
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running — Launch All first")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = "http://127.0.0.1:8090/v1/chat/completions"  # GATEWAY: key-auth + usage tracking for ALL farm-proxied generate paths
    def do():
        import urllib.request, json as _json, urllib.error
        payload = _json.dumps({
            "model": v.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": v.prompt},
                {"type": "image_url", "image_url": {"url": v.image}},
            ]}],
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
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


@app.get("/api/scan")
async def scan_models(path: str, recursive: bool = False):
    if not path:
        raise HTTPException(status_code=400, detail="path query param required")
    # resolve store:// alias for THIS platform, then scan locally
    local = path_resolver.resolve(path)
    models, projectors = await run_in_threadpool(
        model_parser.scan_directory, local, recursive
    )
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
        except (OSError, yaml.YAMLError):
            continue
        if d.get("base_path"):
            all_base_paths.add(d["base_path"])
        for v in (d.get("variants") or []):
            if isinstance(v, dict) and v.get("base_path"):
                all_base_paths.add(v["base_path"])
    # Collect draft paths from model configs (as store:// strings)
    draft_paths = set()
    for m in cfg.list_models():
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
    # offered. llama.cpp serves GGUF (hf entries hidden); vLLM serves HF
    # dirs (gguf entries hidden). Projectors are editor chrome, not
    # engine-qualified model sources — both engines keep them.
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
    for m in filtered_models:
        m["architecture"] = cfg._get_model_architecture(m["base_path"])
        # TypeSafe decision models surface in the wizard the same way
        m["decision_type"] = cfg._get_decision_type(m["base_path"])
    return {"models": filtered_models, "projectors": projectors}


# ---------------- cluster ops ----------------

if __name__ == "__main__":
    # SINGLETON GUARD: every restart path (/api/restart, engine save) spawns
    # this module detached while the old process was supposed to die — and
    # sometimes it doesn't (observed: 4 Oct-1 orphans lived for 3 days next
    # to the live instance). An orphan is not idle: its boot-reconcile
    # re-pushes a stale swap config BEHIND the live app's back, silently
    # un-deploying variants the operator saved (observed live: --dad
    # vanished from llama-swap between two pushes). Guard: any instance
    # that fails to BIND :8199 kills its own swap-write capability and
    # exits — only the port owner ever survives. The pre-import sys.path
    # setup above is shared, so this block is intentionally at the bottom
    # (after all definitions) yet still before uvicorn starts.
    import socket as _socket
    # Short bind-wait first: /api/restart kills the old port owner THEN
    # spawns us — the dying process may still hold the socket for a
    # moment. Retrying up to ~10s keeps the legal restart path working;
    # a process STILL holding :8199 after that is a long-lived orphan,
    # and that exact shape is what must never boot.
    _deadline = time.time() + 10.0
    _bound = False
    while True:
        _probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        # SO_REUSEADDR: after fuser -k the old app's CLOSED connections sit
        # in TIME_WAIT up to 60s (observed: bind blocked 20s+, /proc/net/tcp
        # full of state-06 :2007 rows) — plain bind() fails the whole wait.
        # uvicorn's own bind (config.py) sets REUSEADDR and succeeds, so the
        # guard must probe the same way or it self-aborts on healthy boots.
        # A LIVE listener still conflicts even with REUSEADDR (same-addr
        # rebinding of an active LISTEN socket is EADDRINUSE), so the
        # orphan-detection semantics are unchanged.
        _probe.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        try:
            _probe.bind(("0.0.0.0", 8199))
            _bound = True
        except OSError:
            _probe.close()
            if time.time() >= _deadline:
                break
            time.sleep(0.5)
            continue
        _probe.close()
        break
    if not _bound:
        print("[singleton] :8199 already owned by another instance — "
              "exiting (no orphan reconciles, no stale swap pushes)",
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8199, reload=False)
