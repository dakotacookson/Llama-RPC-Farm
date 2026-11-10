"""Shared HTTP helpers used by more than one route module.

Kept dependency-light and free of route registration so both ``app.py`` and
``routes/*`` can import the same identity/gateway policy.
"""
import os

import access
import config_loader as cfg


def _access_filter_models(models):
    """Apply the caller's exact-token allowlist to a model list.

    gate_open (trusted console / unrestricted admin) sees everything.
    A restricted caller with no grants sees NOTHING; a scoped caller sees
    only granted base ids and granted variants.
    """
    if access.gate_open():
        return models
    allow = access.view_allow()
    if not allow:
        return []
    from swap_gateway import _mid_allowed
    out = []
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("model_id") or m.get("id") or "")
        variants = m.get("variants") or []
        keep_vars = [v for v in variants
                     if _mid_allowed(f"{mid}--{v.get('label')}", allow)]
        if _mid_allowed(mid, allow) or keep_vars:
            mm = dict(m)
            if variants:
                mm["variants"] = keep_vars
            out.append(mm)
    return out


LOGIN_COOKIE = "farm_session"


def _gateway_url(path=""):
    """Local :8090 gateway URL; GATEWAY_PORT overrides for isolated tests."""
    port = (os.environ.get("GATEWAY_PORT") or "8090").strip()
    return f"http://127.0.0.1:{port}{path}"


def _gateway_headers(request=None):
    """Identity headers for a server-side call to the :8090 gateway.

    The browser's farm session cookie resolves (inside the gateway) to the
    logged-in user's access key, so usage/grants attribute to them without
    ever exposing the key to JS. An explicit Bearer key also rides along.
    """
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        sid = request.cookies.get(LOGIN_COOKIE) if request is not None else ""
    except Exception:
        sid = ""
    if sid:
        h["Cookie"] = f"{LOGIN_COOKIE}={sid}"
    try:
        tok = access.request_key(request) if request is not None else ""
    except Exception:
        tok = ""
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


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


def _mask_token(tok: str) -> str:
    if len(tok) <= 8:
        return tok[:4] + "…"
    return tok[:8] + "…" + tok[-4:]
