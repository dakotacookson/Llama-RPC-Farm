"""Conservative orphan RPC-server reaper.

The generated start scripts stop the peers they started via an EXIT trap,
and RPC script entries now also carry an explicit cmdStop. This module is
the last line of defense: it periodically compares the models llama-swap
reports as loaded/loading (and the endpoints local llama-server processes
are actually using) against the ggml-rpc-server processes running on every
machine, then kills only clearly-orphaned, old servers.

Safety rules:
- Never runs when llama-swap is offline (can't know what is loaded).
- Never touches an endpoint referenced by a local llama-server --rpc list.
- Never touches an endpoint expected by a loaded/loading model deployment.
- Never touches a process younger than REAPER_GRACE (a slow CIFS load can
  spend many minutes between starting rpc-servers and opening the port).
- Only kills the specific per-port server (rpc_manager.stop_rpc(port)),
  never a blanket pkill, so co-served models are untouched.
- FARM_REAPER=0 disables it; FARM_REAPER_DRY_RUN=1 makes it log-only.
"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

import config_loader as cfg
import local_node
import rpc_manager
import swap_manager

INTERVAL = max(30, int(os.environ.get("FARM_REAPER_INTERVAL", "300") or 300))
GRACE = max(60, int(os.environ.get("FARM_REAPER_GRACE", "1800") or 1800))
ENABLED = str(os.environ.get("FARM_REAPER", "1")).lower() not in (
    "0", "false", "no", "off")
DRY_RUN = str(os.environ.get("FARM_REAPER_DRY_RUN", "")).lower() in (
    "1", "true", "yes", "on")


def _loaded_ids():
    """Model ids llama-swap currently reports loaded/loading, or None when
    the daemon can't be queried (reaper must then do nothing)."""
    try:
        st = swap_manager.status()
    except Exception:
        return None
    if not st.get("running"):
        return None
    url = f"http://{st.get('host_ip') or '127.0.0.1'}:{st.get('port') or 8080}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    out = set()
    for m in (data.get("data") or []):
        mid = (m.get("id") or "").strip()
        status = ((m.get("status") or {}).get("value") or "").strip().lower()
        if mid and status in ("loaded", "loading"):
            out.add(mid)
    return out


def _expected_endpoints(loaded):
    """peer_ip:port endpoints a loaded/loading deployment should own."""
    out = set()
    try:
        host = swap_manager.get_host()
        host_name = host["name"]
    except Exception:
        return out
    for dep in loaded:
        try:
            model, _base, _label = swap_manager.resolve_deployment(dep)
            servers = model.get("servers") or []
            names = [host_name] + [s for s in servers if s != host_name]
            master, peers = rpc_manager._split_servers(names, host_name)
            ports = rpc_manager._model_rpc_ports(model, peers)
        except Exception:
            continue
        by_name = {p.get("name"): p for p in peers}
        for (pname, _gname), port in ports.items():
            peer = by_name.get(pname)
            if peer and peer.get("ip"):
                out.add(f"{peer['ip']}:{port}")
    return out


def _active_endpoints():
    """Endpoints referenced by local llama-server --rpc args, or None when
    the host process table cannot be inspected (reaper must then skip)."""
    if os.name == "nt":
        return None
    try:
        proc = subprocess.run(
            ["bash", "-c",
             "ps -eo args= 2>/dev/null | grep '[l]lama-server' || true"],
            capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    out = set()
    for line in (proc.stdout or "").splitlines():
        m = re.search(r'--rpc\s+"?([^"\s]+)"?', line)
        if not m:
            continue
        for ep in m.group(1).split(","):
            ep = ep.strip()
            if ep:
                out.add(ep)
    return out


def run_once(dry_run=False):
    """One reaper pass. Returns a report; safe to call manually."""
    loaded = _loaded_ids()
    if loaded is None:
        return {"skipped": "llama-swap not running or /v1/models unreachable"}
    active = _active_endpoints()
    if active is None:
        return {"skipped": "cannot inspect local llama-server processes"}
    protected = _expected_endpoints(loaded) | active
    actions = []
    for machine in cfg.machines.all():
        name = machine.get("name") or ""
        ip = (machine.get("ip") or "").strip()
        if not name or not ip:
            continue
        for srv in rpc_manager.list_rpc_servers(name):
            ep = f"{ip}:{srv.get('port')}"
            if ep in protected:
                continue
            age = srv.get("age_s")
            if age is not None and age < GRACE:
                continue
            item = {"machine": name, "pid": srv.get("pid"),
                    "port": srv.get("port"), "endpoint": ep,
                    "age_s": age}
            if dry_run:
                item["action"] = "would_kill"
            else:
                try:
                    rpc_manager.stop_rpc(name, srv.get("port"))
                    item["action"] = "killed"
                except Exception as e:
                    item["action"] = "error"
                    item["error"] = str(e)
            actions.append(item)
    return {"loaded": sorted(loaded), "protected": sorted(protected),
            "actions": actions, "dry_run": bool(dry_run),
            "grace_s": GRACE}


def loop():
    if not ENABLED:
        print("[reaper] disabled (FARM_REAPER=0)")
        return
    print(f"[reaper] active — interval {INTERVAL}s, grace {GRACE}s"
          + (" (dry run)" if DRY_RUN else ""))
    # let the app/swap finish booting before the first pass
    time.sleep(30)
    while True:
        try:
            res = run_once(dry_run=DRY_RUN)
            if res.get("skipped"):
                pass
            else:
                acts = [a for a in res.get("actions", [])
                        if a.get("action") in ("killed", "would_kill", "error")]
                if acts:
                    print(f"[reaper] {len(acts)} orphan action(s): "
                          + json.dumps(acts)[:2000])
        except Exception as e:
            print(f"[reaper] pass failed: {e}")
        time.sleep(INTERVAL)


def start():
    if not ENABLED:
        return None
    t = threading.Thread(target=loop, daemon=True, name="rpc-orphan-reaper")
    t.start()
    return t
