"""Model deployment launch/stop/active endpoints."""
from concurrent.futures import ThreadPoolExecutor

import pydantic
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import access
import config_loader as cfg
import engine_registry
import rpc_manager
import swap_manager
from swap_client import _proxy_swap_unload
from web_shared import _raw_is_private

router = APIRouter()


class LaunchIn(pydantic.BaseModel):
    model_id: str
    servers: list[str]
    port: int = 8080
    variant: str | None = None   # label of additional config; when set, use that variant's deployment
    master_name: str | None = None  # which server runs llama-server (Launch page "run on" picker)
    mode: str = "auto"           # rpc | local | auto — auto = generator heuristic decides from hardware


@router.post("/api/launch")
async def launch_model(l: LaunchIn):
    """Live launch: orchestrate everything from model config, no scripts.
    If variant is set, the variant's deployment is merged with the base model."""
    # B5a2 + DEC-06: the single-model launch path is engine-owned — the
    # adapter runs it (llama.cpp: rpc_manager.launch_model).
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if l.mode not in ("rpc", "local", "auto"):
        raise HTTPException(status_code=400,
                            detail=f"invalid mode '{l.mode}' (rpc|local|auto)")
    if l.master_name and l.master_name not in l.servers:
        raise HTTPException(status_code=400,
                            detail=f"master '{l.master_name}' is not among the servers")
    # access.py: same "never existed" rule as /api/chat — launch is a
    # model-scoped action, so a restricted (strict-mode) viewer may only
    # launch/deploy tokens in their key's allowlist (EXACT match: a
    # variant grant does NOT widen to the base card; base launch = base
    # token on the grid, variant launch = mid--label). Unknown ids keep
    # the not-found path below (cfg.get_model in the engine).
    lmid = (l.model_id or "").strip()
    if not access.gate_open():
        base_mid = lmid.split("--")[0]
        if _raw_is_private(base_mid):
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
        allow = access.view_allow()
        if allow is None or lmid not in allow:
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
    valid = {m["name"] for m in cfg.machines.all()}
    unknown = [s for s in l.servers if s not in valid]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"unknown machine(s): {', '.join(unknown)}")
    if not l.servers:
        raise HTTPException(status_code=400, detail="pick at least one server")
    mode, master_name = l.mode, l.master_name
    if mode == "auto":
        # Generator decides: single-box-on-a-remote-peer deployments run
        # local on that box (no RPC hops); anything else stays rpc.
        # Unknown/ad-hoc ids keep the legacy rpc default.
        try:
            verdict = swap_manager.deployment_launch_mode(
                l.model_id, l.variant)
            mode = verdict["mode"]
            if mode == "local" and verdict["box"] and not master_name:
                master_name = verdict["box"]
        except ValueError:
            mode = "rpc"
    try:
        report = await run_in_threadpool(
            engine.start_model, l.model_id, l.servers, l.port, l.variant,
            master_name, mode)
    except ValueError as e:
        return JSONResponse(
            {"model_id": l.model_id, "failed": True, "error": str(e),
             "steps": []}, status_code=500)
    status = 500 if report.get("failed") else 200
    return JSONResponse(report, status_code=status)


@router.post("/api/launch/stop")
async def stop_launch(l: LaunchIn):
    """Stop a live-launched model on its servers."""
    # B15 + DEC-06: the stop path is engine-owned. llama.cpp:
    # rpc_manager.stop_model kills llama.cpp RPC ports.
    # access.py: same "never existed" rule as /api/launch — a scoped
    # strict-mode key may only stop tokens it is granted (stopping is a
    # model-scoped action; a variant grant does not cover the base key).
    # Unauthenticated/unknown keys under an open gate keep the legacy
    # no-op path (their stop is harmless: nothing they name resolves).
    lmid = (l.model_id or "").strip()
    if not access.gate_open():
        base_mid = lmid.split("--")[0]
        if _raw_is_private(base_mid):
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
        allow = access.view_allow()
        # STRICT posture: unknown/unresolvable key = restricted, allow
        # None — may stop nothing (same rule as /api/launch).
        if allow is None or lmid not in allow:
            raise HTTPException(status_code=404,
                                detail=f"model '{lmid}' not found")
    valid = {m["name"] for m in cfg.machines.all()}
    unknown = [s for s in l.servers if s not in valid]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"unknown machine(s): {', '.join(unknown)}")
    # Serve-group truth: a remote-proxy (single-box-on-a-peer) deployment
    # is owned by llama-swap — its server lives on the peer box, so
    # rpc_manager.stop_model (master llama-server + peer rpc-servers)
    # would miss it. Route through the swap unload proxy (cmdStop) so the
    # stop actually unloads it and the serve group frees the box.
    # Swap down -> 503, same as the Swap tab's Evict. Unknown ids keep
    # the legacy path.
    # Stop BOTH possible owners: a live launch (llama-server on the
    # requested port) and, when the same deployment is served by
    # llama-swap, its remote-proxy entry. A live launch on a remote box is
    # indistinguishable by model id alone, so do the live port-targeted
    # stop first, then best-effort unload the swap entry.
    res = await run_in_threadpool(rpc_manager.stop_model, l.servers,
                                  l.model_id, l.variant, l.master_name,
                                  l.port)
    try:
        verdict = swap_manager.deployment_launch_mode(l.model_id, l.variant)
    except ValueError:
        verdict = None
    if verdict and verdict["mode"] == "local" and verdict["box"]:
        try:
            await _proxy_swap_unload(verdict["swap_key"])
        except Exception:
            pass
    return res


@router.get("/api/launch/active")
async def active_launches():
    """Which configured model ports are serving right now (any machine).
    DEC-06: the per-machine serving probe is the adapter's active_model —
    llama.cpp checks the machine's RPC port."""
    machines = cfg.machines.all()
    try:
        engine = engine_registry.get_engine()
    except ValueError:
        engine = engine_registry.get_engine({"engine": "llama_cpp"})
    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        return list(pool.map(engine.active_model, machines))
