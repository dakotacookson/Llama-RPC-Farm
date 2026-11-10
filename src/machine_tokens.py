"""Peer-agent token lifecycle: farm token, per-machine pairing, readiness.

Kept out of ``app.py`` so the HTTP layer does not own token policy. All
secret material stays in the vault; callers only pass machine records.
"""
import local_node
import agent_client


def _ensure_farm_agent_token():
    """Farm token is app-managed: generated once and reused; never exposed
    for manual override in Settings."""
    import secrets as _secrets
    import secrets_vault as _vault
    try:
        if not _vault.farm_agent_token_set():
            _vault.set_farm_agent_token(_secrets.token_urlsafe(32))
        return _vault.get_farm_agent_token()
    except Exception:
        return ""


def _persist_farm_token_if_missing(machine):
    """A successfully reached agent with no per-machine vault token is using
    the farm token; persist it so the machine shows agent_token_set=true and
    its generated scripts get AGENT_TOKEN_<NAME>."""
    try:
        import secrets_vault as _vault
        mid = machine.get("id")
        if mid and not _vault.get_machine_agent_token(machine):
            farm = _ensure_farm_agent_token()
            if farm:
                _vault.set_machine_agent_token(mid, farm)
    except Exception:
        pass


def _auto_agent_token(machine, requested=""):
    """Auto-generate/assign a peer-agent token on first save.

    Uses an explicit token if given, else an existing per-machine vault
    token, else the farm token (auto-generated at app startup), pairing a
    fresh unpaired agent as needed. Returns the effective token.
    """
    if local_node.is_local(machine):
        return ""
    import secrets as _secrets
    import secrets_vault as _vault

    requested = (requested or "").strip()
    if requested:
        _ok, err = _ensure_agent_ready(machine, requested)
        if err:
            raise ValueError(
                f"peer agent rejected the token: {err}")
        return requested

    existing = _vault.get_machine_agent_token(machine)
    if existing:
        _ok, err = _ensure_agent_ready(machine, existing)
        if not err:
            return existing
    farm = _ensure_farm_agent_token()
    if farm:
        _ok, err = _ensure_agent_ready(machine)
        if not err:
            return farm
    tok = _secrets.token_urlsafe(32)
    m = dict(machine)
    m["_agent_token"] = tok
    try:
        agent_client.pair(m, tok)
        agent_client.capabilities(m, timeout=8)
        return tok
    except Exception as e:
        raise ValueError(
            "could not auto-pair the peer agent — it may already be paired "
            "with another token; enter that token in the machine editor") from e


def _ensure_agent_ready(machine, token_override="", timeout=4):
    """Validate a peer agent's TOKEN (not just reachability), auto-pairing a
    fresh agent with the farm token when possible.

    /health is deliberately unauthenticated, so it cannot prove the token is
    right. /capabilities requires auth and is the gate here.
    Returns (capabilities_or_None, error_or_None)."""
    m = dict(machine)
    if token_override:
        m["_agent_token"] = token_override
    try:
        out = agent_client.capabilities(m, timeout=8)
        if not token_override:
            _persist_farm_token_if_missing(machine)
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
                out = agent_client.capabilities(m, timeout=8)
                return out, None
            except Exception as e:
                return None, str(e)
        farm = _ensure_farm_agent_token()
        if farm:
            pair_err = None
            try:
                agent_client.pair(m, farm)
            except Exception as e:
                pair_err = str(e)
            m["_agent_token"] = farm
            try:
                out = agent_client.capabilities(m, timeout=8)
                _persist_farm_token_if_missing(machine)
                return out, None
            except Exception as e:
                if pair_err and "already paired" in pair_err.lower():
                    return None, (
                        "agent is already paired with a different token — "
                        "enter its token in Configure")
                return None, str(e)
        return None, str(first)


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
