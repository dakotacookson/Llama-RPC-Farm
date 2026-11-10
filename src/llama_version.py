"""llama.cpp build/version probing + latest-release cache."""
import re
import time

import agent_client
import local_node
from update_state import _LLAMA_UPDATES_LOCK


_LATEST_TAG_CACHE = {"tag": None, "ts": 0.0}
# per-machine llama-server version probe (agent/bash, ~seconds) — cached
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
    (local bash or the peer agent). POSIX: leading ~ expands via $HOME on the
    target; Windows: cmd.exe expands %USERPROFILE% inside the path."""
    import rpc_manager
    # Per-machine override wins; empty falls back to Settings' global
    # llama_bin (same resolver the launch path uses).
    lb = (rpc_manager._effective_bin_dir(machine) or "").strip().rstrip("/\\")
    if not lb:
        return None
    is_win = machine.get("os", "linux") == "windows"
    if (not local_node.is_local(machine)
            and agent_client.should_use_agent(machine)):
        try:
            path = (f"{lb}\\llama-server.exe" if is_win
                    else f"{lb}/llama-server")
            res = agent_client.install_verify(
                machine, path, ["--version"], timeout=35)
            if res.get("returncode") != 0:
                return None
            out = res.get("output") or ""
            for line in out.splitlines():
                if line.strip():
                    return line.strip()
            return None
        except Exception:
            return None
    try:
        from llamacpp_installer import _machine_run
        run, _ = _machine_run(machine)
        if is_win:
            cmd = f'"{lb}\\llama-server.exe" --version 2>&1'
        else:
            cmd = rf'p="{lb}/llama-server"; p="${{p/#\~/$HOME}}"; "$p" --version 2>&1'
        code, out = run(cmd, timeout=30)
        if code != 0:
            return None
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


def invalidate_llama_version(name):
    """Drop the cached version payload for one machine after an update."""
    with _LLAMA_UPDATES_LOCK:
        _LLAMA_VER_CACHE.pop(name, None)
