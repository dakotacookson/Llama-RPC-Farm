"""API-key identity gate.

Identity = `Authorization: Bearer <key>` / `X-Api-Key` header, resolved
against the ACCESS KEYS table in config/settings.yaml (`access_keys:`
block) — edited from the Access tab, never hard-coded:

    access_keys:
      sk-dad-a1b2c3:            # the literal token clients send
        name: Dad               # label only
        models:                 # EXACT grants: mid or mid--variant
          - Huihui-Qwen3.6-35B-A3B-Q4_K-128k--dad
          - laya-q8-0
        admin: false            # true = control surfaces (Access tab...)

Same visibility mechanism as before: config_loader._is_private_visible
consults contextvars per request, set by app.py's IpGateMiddleware from
the resolved key. A key with `models:` sees ONLY those tokens (exact
grants — bare mid = base entry only, mid--label = that variant). Keys
with NO `models:` (or any request with NO key once keys exist) see
everything per the open-mode semantics. In-process callers (boot
reconcile, tuner, stats, SSH housekeeping) see everything — the gate is
opt-in hiding, never an accidental lockout.

Admin status is a KEY FLAG, not an IP: `admin: true` keys open control
surfaces (Access tab & its APIs) independent of what `models:` scopes.

Loopback stays trusted for the operator console (farm's own fetches —
tuner, boot reconcile, stats collectors — present no key by design).

`private: true` composes unchanged: such a model is invisible to every
key-scoped request regardless of grant membership (scoping can only
narrow, never widen past the private gate).

Behavior for missing/unknown keys matches the OLD unknown-IP behavior:
_restricted True (private gate closed), allow None (no list narrowing).
With NO keys configured at all: everything open (pre-Access farm).

Thread-model: gate state is a contextvar, so concurrent requests can't
see each other's scope, and worker threads spawned inside a scoped
request inherit it (run_in_threadpool copies context).
"""
import contextvars
import os
import secrets

VIEWS_SETTINGS_KEY = "views"            # legacy block, read for migration view
ADMIN_IPS_SETTINGS_KEY = "admin_ips"    # legacy, no longer grants trust
ADMINS_UNRESTRICTED_KEY = "admins_unrestricted"  # legacy flag, moot under keys
KEYS_SETTINGS_KEY = "access_keys"

# Always-trusted anchors: loopback (operator console, farm's own
# in-process fetches). Everything else identifies by KEY.
_TRUSTED_ANCHORS = {"127.0.0.1", "::1"}
TRUSTED_IPS = _TRUSTED_ANCHORS  # legacy name kept for callers only

# contextvar: exact-token allowlist for THIS request; None = unrestricted
_view_allow: contextvars.ContextVar = contextvars.ContextVar(
    "access_view_allow", default=None)

_restricted: contextvars.ContextVar = contextvars.ContextVar(
    "access_restricted", default=False)

_key_name: contextvars.ContextVar = contextvars.ContextVar(
    "access_key_name", default=None)

_key_admin: contextvars.ContextVar = contextvars.ContextVar(
    "access_key_admin", default=False)

# strict/unrestricted chip synced from the request's key record (see
# set_key_unrestricted) — admins_unrestricted legacy path reads it too
_key_unrestricted: contextvars.ContextVar = contextvars.ContextVar(
    "access_key_unrestricted", default=True)

# full raw record (incl. per-key `ui:` theme state) resolved at middleware
# time — contextvar so settings endpoints read THEIR caller's ui safely
def raw_key_record():
    """The raw key record resolved for THIS request (middleware), or
    None. Reader seam for identity-scoped surfaces (loaded-model list)."""
    return _raw_key_record.get()

_raw_key_record: contextvars.ContextVar = contextvars.ContextVar(
    "access_key_raw", default=None)


def current_key_record():
    """The caller's raw key record ({name, models, admin, ui}) or None
    (console/loopback, keyless, or in-process callers)."""
    return _raw_key_record.get()


def _load_settings():
    try:
        import yaml
        p = os.environ.get("FARM_SETTINGS_FILE") or _settings_path()
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _settings_path():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "config", "settings.yaml")


def access_keys():
    """access_keys: table normalized: {token: {name, models(set),
    admin(bool), unrestricted(bool)}}. Bad entries skipped — one typo
    never breaks the gate. The chip reads BOTH spellings it has been
    written under over time (admin_unrestricted legacy, unrestricted
    current) — absent = True (unrestricted), matching legacy keys."""
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
                    "unrestricted": bool(ent.get(
                        "unrestricted",
                        ent.get("admin_unrestricted", True)))}
    return out


def generate_token(prefix="sk-farm"):
    return f"{prefix}-{secrets.token_urlsafe(18)}"


def request_key(request) -> str:
    """Access key from Authorization: Bearer / X-Api-Key (first found)."""
    hdr = (request.headers.get("authorization") or "").strip()
    if hdr.lower().startswith("bearer "):
        hdr = hdr[7:].strip()
    if not hdr:
        hdr = (request.headers.get("x-api-key") or "").strip()
    return hdr


def request_token_from_scope(scope) -> str:
    """Token from a raw ASGI scope: Bearer/X-Api-Key header first, then
    the web-login session cookie (vault resolves login → access key).
    Cookie sessions are how browsers carry identity — same scoping, no
    manual key paste."""
    headers = {k.decode() if isinstance(k, bytes) else k:
               v.decode() if isinstance(v, bytes) else v
               for k, v in (scope.get("headers") or [])}
    auth = (headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        if tok:
            return tok
    tok = (headers.get("x-api-key") or "").strip()
    if tok:
        return tok
    cookie = (headers.get("cookie") or "")
    for part in cookie.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() == "farm_session":
                return _session_token(v.strip())
    return ""


def _session_token(sid: str) -> str:
    """sid -> the access-key token mapped to that login (vault)."""
    try:
        import secrets_vault as _vault
        user = _vault.get_web_session(sid)
        if not user:
            return ""
        _set_session_user(user)  # contextvar for /api/auth/me UI report
        rec = _vault.get_web_login(user)
        if rec and rec.get("key"):
            return rec["key"]  # already decrypted by get_web_login
    except Exception:
        pass
    return ""


# who the COOKIE session belongs to (key model has no user concept; the
# login layer does — surfaced by /api/auth/me only)
_session_user: contextvars.ContextVar = contextvars.ContextVar(
    "access_session_user", default=None)


def _set_session_user(user):
    return _session_user.set(user or None)


def session_user():
    """The logged-in username for THIS request (cookie sessions), or
    None when identity came from a direct key / no identity."""
    return _session_user.get()


def key_record(token: str):
    """{name, models, admin, unrestricted} for a token, or None. `ui` (per-key theme
    state) rides through untouched for the settings endpoints."""
    tok = (token or "").strip()
    raw = _load_settings().get(KEYS_SETTINGS_KEY) or {}
    if not isinstance(raw, dict):
        return None
    ent = raw.get(tok)
    if not isinstance(ent, dict):
        return None
    out = access_keys().get(tok) or {"name": tok, "models": set(),
                                     "admin": False}
    # the chip writes `admin_unrestricted` (legacy name for the same
    # switch); expose BOTH spellings so every consumer sees one truth
    unres = ent.get("unrestricted", ent.get("admin_unrestricted", True))
    out["unrestricted"] = bool(unres)
    out["admin_unrestricted"] = out["unrestricted"]
    out["ui"] = ent.get("ui") if isinstance(ent.get("ui"), dict) else {}
    return out


def request_ip(request):
    """Legacy helper (logging + 404-shaping callers)."""
    for header in ("x-forwarded-for", "x-real-ip"):
        val = (request.headers.get(header) or "").strip()
        if val:
            return val.split(",")[0].strip()
    client = getattr(request, "client", None)
    return (client.host if client and client.host else "") or ""


# ---- legacy IP-API shims: identity is key-first now; IPs grant nothing ----

def _ip_or_net(ip):
    """ipaddress object for a literal IP or CIDR, else None."""
    import ipaddress as _ipa
    try:
        return _ipa.ip_address(ip)
    except ValueError:
        pass
    try:
        return _ipa.ip_network(ip, strict=False)
    except ValueError:
        return None


def admin_ips():
    """LEGACY: kept for call-site compat. Grants nothing."""
    return set()


def admins_unrestricted():
    """LEGACY: per-key `admin:` flags replaced the global mode."""
    return False


def ip_trusted(ip):
    """Trusted = loopback anchors, a configured admin IP/CIDR, OR a
    non-IP-literal peer token ("testclient" from Starlette's TestClient,
    background threads' missing peer) — in-process callers are never
    accidentally locked out; a real socket request always carries an IP
    literal peer, so this branch can't be reached by a network client."""
    ip = (ip or "").strip()
    if not ip or ip in _TRUSTED_ANCHORS:
        return True
    obj = _ip_or_net(ip)
    if obj is None:
        return True  # not an IP literal -> in-process token, not a peer
    for admin in admin_ips():
        net = _ip_or_net(admin)
        if net is None:
            continue
        if obj == net or (hasattr(net, "netmask") and not hasattr(net, "ip")
                          and obj in net):
            return True
    return False


def request_trusted(request):
    return ip_trusted(request_ip(request))


def view_for_ip(ip):
    """LEGACY: IPs never scope. Returns None always."""
    return None


def view_for_key(token):
    """Grant record {name, models, admin} for a token, or None."""
    return key_record(token)


def set_view_allow(models):
    """Middleware hook: scope THIS request context. None/empty clears."""
    return _view_allow.set(set(models) if models else None)


def view_allow():
    """Exact-token allowlist for the current request, or None = open."""
    return _view_allow.get()


def set_restricted(value=True):
    return _restricted.set(bool(value))


def trust_current_thread():
    """Context manager for in-process background threads (boot
    reconcile, tuner, stats): force gate OPEN for this thread's
    context, restore on exit. Background housekeeping must always see
    every model — the HTTP gate is what hides scopes from requests."""
    from contextlib import contextmanager

    @contextmanager
    def _open():
        tok = _restricted.set(False)
        try:
            yield
        finally:
            _restricted.reset(tok)

    return _open()
def gate_open():
    return not _restricted.get()


def set_key_name(name):
    return _key_name.set(name or None)


def key_display_name():
    return _key_name.get()


def set_key_admin(value):
    return _key_admin.set(bool(value))


def set_key_unrestricted(value):
    """Sync the strict/unrestricted chip contextvar from the request's
    key record (restrict_scope_from_key). Default True = unrestricted,
    matching legacy keys that never wrote the chip."""
    return _key_unrestricted.set(bool(value))


def is_admin(request_or_record):
    """Admin check for CONTROL surfaces. Accepts the RESOLVED RECORD
    (dict) — control checks run after the middleware resolved context:
    the contextvar is checked first, then a record/token argument."""
    if _key_admin.get():
        return True
    if isinstance(request_or_record, dict):
        return bool(request_or_record.get("admin"))
    if isinstance(request_or_record, str):
        rec = key_record(request_or_record)
        if rec:
            return rec["admin"]
    return False


def admin_sees_everything():
    """True when THIS context is admin AND its unrestricted chip is on —
    the old 'trusted' meaning for whoami/Live-check rendering. Strict
    admins (chip off) get control surfaces but their model VISIBILITY
    stays scoped to the key's grants — the contextvar is synced in
    restrict_scope_from_key; a direct-record consumer passes the rec."""
    rec = _raw_key_record.get()
    if rec is not None:
        return bool(rec.get("admin")) and bool(rec.get(
            "unrestricted", rec.get("admin_unrestricted", True)))
    return bool(_key_admin.get()) and bool(_key_unrestricted.get())


def restrict_scope_from_key(scope, token: str):
    """THE identity seam: sets _view_allow/_restricted/_key_admin for
    this request scope from the presented key. Rules:
    - no keys configured ⇒ everything open (pre-Access farm state)
    - loopback peers with no key: operator console stays open
    - known key + models set ⇒ scoped context (restricted + allowlist);
      admin flag still opens control surfaces (contextvar)
    - known key, no models ⇒ unrestricted visibility
    - unknown/missing key once keys exist ⇒ restricted, allow None
      (identical to the old unknown-IP behavior)"""
    if scope.get("type") not in ("http", "websocket"):
        return
    keys = access_keys()
    # NOTE: do NOT clear the session-user contextvar here — the cookie
    # extraction (request_token_from_scope → _session_token) may have
    # just set it for THIS request; only identity-decision paths clear it
    _raw_key_record.set(None)
    if not keys:
        set_restricted(False)
        set_view_allow(None)
        set_key_admin(False)
        set_key_name(None)
        return
    t = (token or "").strip()
    # loopback console without a key stays open
    peer = ((scope.get("client") or (None,))[0] or "")
    if not t and peer in _TRUSTED_ANCHORS:
        set_restricted(False)
        set_view_allow(None)
        set_key_admin(True)
        set_key_name("console")
        return
    # (no unconditional clears here — the cookie extraction set the
    # session-user for THIS request; the decision paths below own it)
    _raw_key_record.set(None)
    if not t:
        set_restricted(True)
        set_view_allow(None)
        set_key_admin(False)
        set_key_name(None)
        return
    rec = key_record(t)
    if rec is None:
        set_restricted(True)
        set_view_allow(None)
        set_key_admin(False)
        set_key_name(None)
        _raw_key_record.set(None)
        _set_session_user(None)
        return
    _raw_key_record.set(rec)
    set_key_name(rec["name"])
    set_key_admin(rec["admin"])
    # strict-vs-unrestricted chip: sync the contextvar so
    # admin_sees_everything() reflects THE KEY's choice (the default is
    # unrestricted for legacy keys with no chip written; zazi-style
    # admin_unrestricted:false keys run strict = their model visibility
    # stays scoped even though control surfaces stay open)
    set_key_unrestricted(bool(rec.get(
        "unrestricted", rec.get("admin_unrestricted", True))))
    if rec["models"]:
        set_restricted(True)
        set_view_allow(rec["models"])
        # strict admin keeps the SCOPED view; unrestricted admin sees all
        if rec.get("admin") and admin_sees_everything():
            set_restricted(False)
            set_view_allow(None)
    else:
        set_restricted(False)
        set_view_allow(None)


def restrict_if_foreign(scope):
    """LEGACY ENTRY POINT kept for app.py's middleware — extracts the
    key from the scope and delegates to restrict_scope_from_key."""
    if scope.get("type") not in ("http", "websocket"):
        return
    restrict_scope_from_key(scope, request_token_from_scope(scope))