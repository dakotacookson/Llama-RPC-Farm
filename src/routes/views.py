"""Access views + per-view API keys (Access tab)."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pydantic
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import access
import agent_client
import config_loader as cfg
import local_node
import swap_manager
import settings_store
from settings_store import (_dump_settings_yaml_preserving,
                            _load_settings_yaml_preserving)
from llama_version import invalidate_llama_version
from machine_tokens import _ensure_agent_ready
from web_shared import _mask_token

router = APIRouter()


# ---------------- model views (per-IP visibility; Access tab) ------------
class ViewIn(pydantic.BaseModel):
    """One view-policy entry. `ips` are the client IPs landing in this
    view (Tailscale / LAN, exact literals); `models` are allowlist
    tokens (model_id or model_id--variant). Both replace-on-save."""
    ips: list[str] = []
    models: list[str] = []


@router.get("/api/views")
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


@router.post("/api/views/keys/reveal")
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


class KeyIn(pydantic.BaseModel):
    name: str = ""
    models: list[str] = []
    admin: bool = False
    unrestricted: bool | None = None  # None = legacy default (true)


@router.post("/api/views/keys/generate")
async def key_generate(k: KeyIn, r: Request):
    """Create an access key: random token, returned ONCE in this
    response — never echoed back by the list endpoint."""
    if not access.gate_open() and not access.is_admin(None):
        raise HTTPException(404, "not found")
    tok = access.generate_token()
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
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
    _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
    _access_save_autopush()
    return {"ok": True, "token": tok, "name": ent["name"],
            "admin": ent["admin"], "models": ent["models"],
            "unrestricted": ent["admin_unrestricted"]}


@router.post("/api/views/keys/{token_masked}/update")
async def key_update(token_masked: str, k: KeyIn, r: Request):
    """Update a key by its MASKED id (the list never echoes raw tokens;
    update/delete ride the mask, matching what the operator copied)."""
    if not access.gate_open() and not access.is_admin(None):
        raise HTTPException(404, "not found")
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
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
        _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
        _access_save_autopush()
        return {"ok": True, "updated": token_masked}
    raise HTTPException(404, "not found")


@router.post("/api/views/keys/revoke")
async def key_revoke(rq: KeyIn, r: Request):
    """Revoke by masked token (body: {"name": masked_token})."""
    if not access.gate_open() and not access.is_admin(None):
        raise HTTPException(404, "not found")
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
    keys = current.get(access.KEYS_SETTINGS_KEY) or {}
    target = (rq.name or "").strip()
    for tok in list(keys):
        if _mask_token(tok) == target:
            keys.pop(tok, None)
            current[access.KEYS_SETTINGS_KEY] = keys  # empty {} stays (no keys = open mode)
            _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
            _access_save_autopush()
            return {"ok": True, "revoked": target}
    raise HTTPException(404, "not found")


@router.post("/api/views/whoami")
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


@router.post("/api/install-llama-cpp")
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
    settings = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
    settings["llama_bin"] = llama_bin
    _dump_settings_yaml_preserving(settings, settings_store.SETTINGS_FILE)
    return {"status": "success", "llama_bin": llama_bin, "message": "llama.cpp path configured"}


def _detect_bin_on_machine(machine):
    """Check if llama.cpp bin already exists on a machine. Returns
    (found:bool, bin_path:str|None, detail:str). Never installs.
    Remote machines are probed through the peer agent's /files/stat."""
    from pathlib import Path as _Path
    is_win = (machine.get("os", "linux") == "windows")
    is_local = local_node.is_local(machine)
    candidates = []
    configured = (machine.get("llama_bin") or "").strip()
    if configured:
        if configured.endswith("llama-server") or configured.endswith("llama-server.exe"):
            candidates.append(configured)
        else:
            candidates.append(configured.rstrip("/\\") + ("/llama-server.exe" if is_win else "/llama-server"))
    if is_win:
        candidates += [
            "%USERPROFILE%\\llama.cpp\\build\\bin\\llama-server.exe",
            "%USERPROFILE%\\llama.cpp\\llama-server.exe",
            "~/llama.cpp/build/bin/llama-server.exe",
            "~/llama.cpp/llama-server.exe",
        ]
    else:
        candidates += [
            "~/llama.cpp/build/bin/llama-server",
            "/usr/local/bin/llama-server",
            "/usr/bin/llama-server",
        ]
    seen, uniq = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    if is_local:
        from installer.env import RealFS
        fs = RealFS()
        for cand in uniq:
            try:
                expanded = (os.path.expandvars(os.path.expanduser(cand))
                            if is_win else os.path.expanduser(cand))
                if _Path(expanded).exists():
                    r = fs.file_exists_and_runs(expanded)
                    if r.ok:
                        return True, cand, f"found at {cand}"
            except Exception:
                continue
        return False, None, "no runnable bin found (checked: " + ", ".join(uniq) + ")"
    for cand in uniq:
        probe = cand.replace("%USERPROFILE%", "~")
        try:
            res = agent_client.install_verify(
                machine, probe, ["--version"], timeout=35) or {}
            if res.get("returncode") == 0:
                return True, cand, f"found at {cand}"
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                return False, None, f"agent bin probe failed: {e}"
            continue
    return False, None, "no runnable bin found (checked: " + ", ".join(uniq) + ")"

class InstallAllIn(pydantic.BaseModel):
    install_missing: bool = False  # actually install on missing nodes
    machines: list[str] | None = None  # subset to check/install; None = all


@router.post("/api/install-llama-cpp/all")
async def install_llama_cpp_all(p: InstallAllIn | None = None):
    """Check every configured machine for an existing llama.cpp bin.
    Existing installs are SKIPPED (not reinstalled) — preserves working
    systems. Missing/offline nodes are reported so the UI can act.
    With install_missing=true, missing nodes get the correct official
    prebuilt for their machine type installed through the agent."""
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
            _caps, auth_err = _ensure_agent_ready(m)
            if auth_err:
                return {"machine": name, "ip": ip, "os": os_type, "status": "offline",
                        "skipped": True, "reason": auth_err,
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

    # run checks in parallel (threadpool, since agent calls are blocking)
    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        results = list(pool.map(check_one, machines))

    # optionally install the right prebuilt on missing nodes (sequentially;
    # installs are heavy downloads, and agent fan-out of 600s curls is risky)
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
            invalidate_llama_version(r["machine"])

    skipped = sum(1 for r in results if r.get("skipped") and r["status"] in ("already-installed", "offline", "installed"))
    missing = [r for r in results if r["status"] == "missing"]
    return {
        "results": results,
        "summary": f"{len(results)} nodes checked: {skipped} already installed/offline (skipped), {len(missing)} missing",
        "missing": [r["machine"] for r in missing],
    }


@router.post("/api/install-llama-cpp/install")
async def install_llama_cpp_machine(name: str = ""):
    """Install the right official llama.cpp prebuilt for a machine type
    (CUDA/ROCm/Metal/CPU, linux/windows/macos) through the agent or locally,
    then point the machine's llama_bin at it."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    machine = cfg.machines.require(name)
    import llamacpp_installer
    result = await run_in_threadpool(llamacpp_installer.install_on, machine)
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)
