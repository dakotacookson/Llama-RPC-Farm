"""Read-only installer/tool verification endpoints."""
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter

import config_loader as cfg
import local_node
import swap_manager
import settings_store

router = APIRouter()


# ---------------- GET /api/tools/verify (UX-3, design §8) --------------------
# The old REST install endpoints (install-llama-cpp{,/all,/install},
# install-llama-swap, uninstall-llamacpp) are
# RETIRED: the app never executes installs anymore — src/bootstrap.py owns
# them (copy-paste one-liner from the UI, no exec from the app). What
# survives is the read-only status probe: per tool {found, path, version,
# expected} from installer.probe (the same pure logic the installer uses,
# via RealFS/RealRunCmd). A settings path that exists counts as found even
# when the well-known locations miss (the settings record is probed first,
# §6 prefill is the source of truth once the installer ran).


@router.get("/api/tools/verify")
async def tools_verify():
    """Read-only per-tool status: llama.cpp / llama-swap.
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

    settings_bin = _settings_key(settings_store.SETTINGS_FILE, "llama_bin")
    try:
        import swap_manager
        swap_bin = (swap_manager.load_settings() or {}).get("binary")
    except Exception:
        swap_bin = None
    local_machine = next(
        (m for m in cfg.machines.all() if local_node.is_local(m)), {})
    configured_llcpp = ((local_machine.get("llama_bin") or "").strip()
                        or settings_bin)

    def _entry(found, path, version, expected, configured):
        return {"found": bool(found), "path": path, "version": version,
                "expected": expected, "configured": configured or None}

    def _configured_probe(configured, exe):
        """Probe EXACTLY the configured path (dir or file). Never substitute
        a well-known location — per-machine settings are authoritative."""
        if not configured:
            return "", None
        cand = os.path.expanduser(os.path.expandvars(str(configured)))
        if not cand.rstrip("/\\").endswith(exe):
            cand = os.path.join(cand, exe)
        if not fs.isfile(cand):
            return cand, None
        r = fs.file_exists_and_runs(cand)
        ver = (installer_probe._version_from_output(
            r.stdout + " " + r.stderr) if r.ok else None)
        return cand, ver

    out = {}
    # llama.cpp — configured local-machine path wins; otherwise probe the
    # well-known prebuilt/source locations.
    if configured_llcpp:
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        p, ver = _configured_probe(configured_llcpp, exe)
        found = bool(p and fs.isfile(p))
    else:
        p, ver = installer_probe.probe_llama_cpp(fs, run, role="main")
        found = bool(p)
    out["llama.cpp"] = _entry(found, p, ver, "b10000+", configured_llcpp)
    # llama-swap — an explicit path in swap.yaml wins; bare "llama-swap"
    # still resolves through PATH/well-known locations, as configured.
    if swap_bin and swap_bin != "llama-swap":
        exe = "llama-swap.exe" if os.name == "nt" else "llama-swap"
        p, ver = _configured_probe(swap_bin, exe)
        found = bool(p and fs.isfile(p))
    else:
        p, ver = installer_probe.probe_swap(fs, run)
        found = bool(p)
    out["llama-swap"] = _entry(found, p, ver, "255+", swap_bin)
    return out



@router.post("/api/restart")
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
    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(app_path)))
    candidates = [
        os.path.join(project_root, ".venv", "bin", "python"),
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        sys.executable,
    ]
    py = next((c for c in candidates if c and os.path.exists(c)), sys.executable)
    log = os.path.join(tempfile.gettempdir(), "llama-web.log")
    # POSIX: one code path for CLI and dashboard — restart.sh --full stops
    # llama-swap + every farm-managed llama/RPC process on all machines, then
    # restarts the daemon and this UI. Windows has no bash; the PowerShell
    # branch below remains the fallback (and covers installs without restart.sh).
    script = os.path.join(project_root, "restart.sh")
    if os.name != "nt" and os.path.exists(script):
        _sp.Popen(["/bin/bash", script, "--full"], start_new_session=True,
                  close_fds=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        return {"status": "restarting",
                "message": "Full restart started (llama-swap + all machines). "
                           "It will be back in ~5s at http://127.0.0.1:8199"}
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
            "(fuser -k 8199/tcp > /dev/null 2>&1 || true); "
            # Observed: right after fuser -k, the dying process can still
            # hold :8199 for up to ~9s (SIGTERM shutdown + TIME_WAIT-ish
            # linger). The freshly spawned app's singleton guard waits 10s
            # for the bind and exits(2) if held — race lost = dead UI until
            # manual start. Poll for the port going free BEFORE setsid, so
            # the handoff never races (max 15s).
            "for i in $(seq 0 30); do "
            "  python3 -c \"import socket;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
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
    # block on the agent; the daemon comes back in ~3s either way.
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
