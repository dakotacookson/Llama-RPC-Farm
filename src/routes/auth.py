"""Web logins, users, usage telemetry, and auth bootstrap."""
import secrets

import pydantic
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

import access
import settings_store
from settings_store import _dump_settings_yaml_preserving, _load_settings_yaml_preserving
from web_shared import LOGIN_COOKIE, _mask_token

router = APIRouter()


# ---------------- web logins (key identity via username/password) ----
# Secrets in the vault (.env): LOGIN_<NAME>_PASS = salted SHA-256 hash,
# LOGIN_<NAME>_KEY = Fernet-encrypted access-key token (AES-128-CBC +
# HMAC-256 under the hood — same machinery as machine credentials),
# WEB_SESSION_<sid> = encrypted "{user}|{created}" records. Sessions
# live in the vault too, so logins SURVIVE daemon restarts. Cookie is
# HttpOnly, SameSite=Lax; the API key never needs to be pasted into a
# browser — logging in IS the key adoption.

@router.get("/api/auth/me")
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
    has_logins = bool(_vault.list_web_logins())
    has_keys = bool(access.access_keys())
    return {"name": name, "user": user, "admin": admin, "scoped": scoped,
            "models": sorted(access.view_allow() or []),
            "logged_in": bool(user) or bool(tok),
            "trust_via": "session-cookie" if user else
                         ("key" if tok else
                          ("loopback" if is_console else "unauthenticated")),
            "has_logins": has_logins,
            "has_keys": has_keys,
            # First boot on a fresh install: no web login AND no access key.
            # The UI uses this to show the "create admin" setup form rather
            # than an unusable sign-in box.
            "needs_setup": (not has_logins and not has_keys)}


@router.get("/api/auth/my-key")
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


@router.post("/api/auth/login")
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
    return {"ok": True, "user": user, "session": sid}


@router.post("/api/auth/logout")
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


@router.post("/api/auth/change-password")
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
@router.get("/api/usage")
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


@router.get("/api/usage/by-model")
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


@router.get("/api/usage/tracked")
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


@router.get("/api/users")
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


@router.post("/api/users")
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


@router.post("/api/users/delete")
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


@router.post("/api/auth/bootstrap")
async def auth_bootstrap(l: FirstLoginIn, response: Response):
    import secrets_vault as _vault, access as _a
    has_logins = bool(_vault.list_web_logins())
    has_keys = bool(access.access_keys())
    if has_logins or has_keys:
        # This is a FIRST-RUN endpoint only once the farm has any identity
        # (a web login or an access key). After that it is admin/console-only.
        if not (access.gate_open() and access.view_allow() is None):
            raise HTTPException(404, "not found")
    name = (l.username or "").strip()
    if not name or not l.password:
        raise HTTPException(400, "username and password required")
    if len(l.password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    # admin key for the login (reuse existing by name or create)
    tok = access.generate_token()
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
    keys = current.get(_a.KEYS_SETTINGS_KEY) or {}
    if not isinstance(keys, dict):
        keys = {}
    keys[tok] = {"name": (l.key_name or name).strip(), "models": [],
                 "admin": True, "admin_unrestricted": True}
    current[_a.KEYS_SETTINGS_KEY] = keys
    _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
    _vault.set_web_login(name, l.password, tok, admin=True)
    sid = secrets.token_urlsafe(24)
    _vault.set_web_session(sid, name)
    response.set_cookie(LOGIN_COOKIE, sid, max_age=30 * 24 * 3600,
                        httponly=True, samesite="lax", path="/")
    return {"ok": True, "user": name, "session": sid, "key_created": True}
