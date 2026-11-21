"""Peer-agent token lifecycle: per-machine tokens, pairing, rotation.

Kept out of ``app.py`` so the HTTP layer does not own token policy. All
secret material stays in the vault; callers only pass machine records.

Policy: each peer gets its OWN token. The legacy farm-wide token is only
used to adopt agents paired before per-machine tokens existed, and is
rotated away by ``rotate_shared_tokens`` as soon as agents support
``/token/rotate``.
"""
import secrets

import local_node
import agent_client


def _new_agent_token():
    return secrets.token_urlsafe(32)


def _stored_token(machine):
    """This machine's own vault token (never the farm fallback)."""
    import secrets_vault as _vault
    try:
        return _vault.get_machine_agent_token(machine, allow_farm=False)
    except TypeError:
        try:
            return _vault.get_machine_agent_token(machine)
        except Exception:
            return ""
    except Exception:
        return ""


def _persist_agent_token(machine, token):
    mid = (machine or {}).get("id") or (machine or {}).get("name")
    if not mid or not token:
        return
    try:
        import secrets_vault as _vault
        _vault.set_machine_agent_token(mid, token)
    except Exception:
        pass


def _ensure_farm_agent_token():
    """Legacy farm token, retained only for adoption/migration."""
    import secrets_vault as _vault
    try:
        if not _vault.farm_agent_token_set():
            _vault.set_farm_agent_token(_new_agent_token())
        return _vault.get_farm_agent_token()
    except Exception:
        return ""


def _caps(machine, token=None, timeout=8):
    m = dict(machine)
    if token:
        m["_agent_token"] = token
    return agent_client.capabilities(m, timeout=timeout)


def _auto_agent_token(machine, requested=""):
    """Assign a unique token when a machine is added.

    Explicit token -> validate/enroll it. Existing per-machine token ->
    reuse after verification. A pre-existing farm-token-paired agent is
    adopted temporarily (the startup rotation pass replaces it). A fresh
    agent gets a brand-new unique token; the farm token is never used to
    enroll new peers.
    """
    if local_node.is_local(machine):
        return ""

    requested = (requested or "").strip()
    if requested:
        _ok, err = _ensure_agent_ready(machine, requested)
        if err:
            raise ValueError(f"peer agent rejected the token: {err}")
        return requested

    existing = _stored_token(machine)
    if existing:
        try:
            _caps(machine, existing)
            return existing
        except Exception:
            pass

    # Adopt an agent already paired with the old shared token. Do not pair
    # a fresh agent with it; rotation will replace this token.
    farm = _ensure_farm_agent_token()
    if farm:
        try:
            _caps(machine, farm)
            _persist_agent_token(machine, farm)
            return farm
        except Exception:
            pass

    tok = _new_agent_token()
    try:
        agent_client.pair(machine, tok)
        _caps(machine, tok)
    except Exception as e:
        if "already paired" in str(e).lower():
            raise ValueError(
                "agent is already paired with a different token — enter its "
                "token in the machine editor, or reinstall it to re-pair"
            ) from e
        raise ValueError(
            "could not auto-pair the peer agent — it may already be paired "
            "with another token; enter that token in the machine editor"
        ) from e
    _persist_agent_token(machine, tok)
    return tok


def _ensure_agent_ready(machine, token_override="", timeout=4):
    """Validate a peer agent's TOKEN, pairing only with a per-machine token.

    /health is deliberately unauthenticated, so it cannot prove the token is
    right. /capabilities requires auth and is the gate here.
    Returns (capabilities_or_None, error_or_None)."""
    m = dict(machine)
    if token_override:
        m["_agent_token"] = token_override
    try:
        out = _caps(m, token_override)
        return out, None
    except Exception as first:
        if token_override:
            # An explicitly supplied token can also enroll a fresh agent.
            try:
                agent_client.pair(m, token_override)
            except Exception as pe:
                if "already paired" in str(pe).lower():
                    return None, (
                        "agent is already paired with a different token — "
                        "enter its token in Configure")
                return None, str(first)
            try:
                out = _caps(m, token_override)
                _persist_agent_token(machine, token_override)
                return out, None
            except Exception as e:
                return None, str(e)

    existing = _stored_token(machine)
    if existing:
        try:
            return _caps(machine, existing), None
        except Exception:
            pass

    # Legacy adoption: an agent already paired with the farm token works,
    # but a fresh agent is never enrolled with the shared token.
    farm = _ensure_farm_agent_token()
    if farm:
        try:
            out = _caps(machine, farm)
            _persist_agent_token(machine, farm)
            return out, None
        except Exception:
            pass

    tok = _new_agent_token()
    try:
        agent_client.pair(machine, tok)
        out = _caps(machine, tok)
        _persist_agent_token(machine, tok)
        return out, None
    except Exception as e:
        if "already paired" in str(e).lower():
            return None, (
                "agent is already paired with a different token — enter its "
                "token in Configure, or reinstall it to re-pair")
        return None, str(e)


def _agent_guard(machine):
    """Fail fast with an actionable error when a new (non-local) machine's
    peer agent is unreachable. Agent-only: there is no shell preflight."""
    try:
        if local_node.is_local(machine):
            return None
    except Exception:
        pass
    ok, err = _ensure_agent_ready(machine)
    return None if ok else f"peer agent unreachable: {err}"


def migrate_missing_agent_tokens():
    """Copy the legacy farm token into machines that lack their own entry.

    This is a compatibility bridge for installs created before per-machine
    tokens. It keeps them reachable until rotate_shared_tokens runs.
    """
    import secrets_vault as _vault
    import config_loader as cfg
    farm = _vault.get_farm_agent_token()
    if not farm:
        return []
    adopted = []
    for machine in cfg.machines.all():
        try:
            if local_node.is_local(machine):
                continue
            if _vault.machine_agent_token_stored(machine):
                continue
            mid = machine.get("id") or machine.get("name")
            if mid:
                _vault.set_machine_agent_token(mid, farm)
                adopted.append(machine.get("name") or mid)
        except Exception:
            pass
    return adopted


def rotate_machine_token(machine, old_token=None):
    """Rotate one peer to a new unique token; returns a status dict.

    The vault is updated first so a lost response cannot lose the new
    credential; if neither the new token nor the old token works, restore
    the old value.
    """
    import secrets_vault as _vault
    mid = (machine or {}).get("id") or (machine or {}).get("name")
    old = old_token or _stored_token(machine)
    if not old:
        old = _vault.get_machine_agent_token(machine, allow_farm=True)
    if not mid or not old:
        raise ValueError("no existing agent token to rotate")
    new = _new_agent_token()
    _vault.set_machine_agent_token(mid, new)
    try:
        agent_client.rotate_token(machine, old, new)
        _caps(machine, new)
        return {"machine": machine.get("name") or mid, "ok": True}
    except Exception as e:
        # The agent may have persisted the new token before the response
        # was lost. Keep it if it authenticates.
        try:
            _caps(machine, new)
            return {"machine": machine.get("name") or mid, "ok": True}
        except Exception:
            _vault.set_machine_agent_token(mid, old)
            raise ValueError(f"token rotation failed: {e}") from e


def rotate_shared_tokens():
    """Replace every legacy farm token with a unique per-machine token.

    Best-effort: old agents without /token/rotate are left on the shared
    token and reported. Safe to run repeatedly; only shared entries rotate.
    """
    import secrets_vault as _vault
    import config_loader as cfg
    farm = _vault.get_farm_agent_token()
    if not farm:
        return {"rotated": [], "skipped": [], "failed": []}
    rotated, skipped, failed = [], [], []
    for machine in cfg.machines.all():
        name = machine.get("name") or machine.get("id")
        try:
            if local_node.is_local(machine):
                skipped.append({"machine": name, "reason": "local master"})
                continue
            stored = _vault.get_machine_agent_token(
                machine, allow_farm=False)
            if stored != farm:
                skipped.append({"machine": name, "reason": "unique token"})
                continue
            rotate_machine_token(machine, old_token=farm)
            rotated.append(name)
        except Exception as e:
            failed.append({"machine": name, "error": str(e)})
    return {"rotated": rotated, "skipped": skipped, "failed": failed}
