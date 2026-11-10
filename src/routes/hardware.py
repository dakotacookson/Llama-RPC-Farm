"""Background hardware watcher + hardware-change reporting."""
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import config_loader as cfg
import gpu_vendor
import rpc_manager
import stats_collector
import swap_manager

router = APIRouter()


# ---------------- background hardware watcher ----------------
# Polls every machine's full stats on a slow timer and persists CHANGES into
# machines.yaml (GPU count/names, VRAM, RAM). Swapped cards show up in the
# Machines page + visualizer without manual "Detect Hardware" clicks, and
# model configs get reconciled to the new slot layout automatically.
_hw_watcher_state = {"running": False, "last": {}}
HW_WATCH_INTERVAL_S = 120  # low frequency: agent round-trip per machine


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
            # MAC-based IP healing: the agent's UDP responder reports its
            # current addresses, so DHCP re-leases fix themselves.
            import discovery
            discovery.pair_agents(include_known=True)
            healed = discovery.reconcile_machine_ips()
            if healed:
                print("[ip-sync] updated: " + ", ".join(
                    f"{n} {o}->{i}" for n, o, i in healed))
        except Exception as e:
            print(f"[ip-sync] reconcile failed: {e}")
        try:
            res = hw_watch_once()
            if res["changed"]:
                print(f"[hw-watcher] hardware changed on: {res['changed']}"
                      + (f" — synced {len(res['synced_models'])} model config(s)"
                         if res["synced_models"] else ""))
        except Exception as e:
            print(f"[hw-watcher] pass failed: {e}")
        time.sleep(HW_WATCH_INTERVAL_S)


@router.on_event("startup")
async def start_hw_watcher():
    if (os.environ.get("LLAMA_FARM_NO_HW_WATCH") or "").lower() in (
            "1", "true", "yes", "on"):
        # E2E/CI (or an operator pinning hardware by hand) needs the
        # declared machine record to win over host autodetection.
        return
    if _hw_watcher_state["running"]:
        return
    _hw_watcher_state["running"] = True
    threading.Thread(target=_hw_watcher_loop, daemon=True,
                     name="hw-watcher").start()


@router.get("/api/hardware/changes")
async def hardware_changes():
    """Manual re-check right now (same logic as the background watcher).
    Returns which machines changed + how many model configs were synced."""
    return await run_in_threadpool(hw_watch_once)


@router.get("/api/machines/{name}/status")
async def machine_status(name: str):
    try:
        st = await run_in_threadpool(rpc_manager.machine_status, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return st


@router.post("/api/machines/{name}/rpc/start")
async def start_rpc(name: str):
    try:
        return await run_in_threadpool(rpc_manager.start_rpc, name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/machines/{name}/rpc/stop")
async def stop_rpc(name: str):
    try:
        return await run_in_threadpool(rpc_manager.stop_rpc, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/rpc/stop-all")
async def stop_all_rpc():
    """Kill leftover rpc-servers on every machine (both OS types)."""
    names = [m["name"] for m in cfg.machines.all()]
    with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
        results = list(pool.map(rpc_manager.stop_rpc, names))
    return results


@router.post("/api/start-all")
async def start_all():
    """Start everything: RPC servers on every machine, then the
    llama-swap daemon with saved settings. Always 200 — per-part
    failures live inside the payload so the UI can show details."""
    def run():
        names = [m["name"] for m in cfg.machines.all()]
        with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
            rpc_results = list(pool.map(rpc_manager.start_rpc, names))
        try:
            swap_report = swap_manager.launch(None, None, None, None)
        except ValueError as e:
            swap_report = {"failed": True, "error": str(e)}
        return {"rpc": rpc_results, "swap": swap_report}
    return await run_in_threadpool(run)
