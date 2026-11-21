"""Enable/disable optional mutual-TLS transport for llama.cpp RPC."""
import pydantic
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

import access
import agent_client
import config_loader as cfg
import local_node
import rpc_tls
import swap_manager

router = APIRouter()


def _require_control():
    if not ((access.gate_open() and access.view_allow() is None)
            or access.is_admin(None)):
        raise HTTPException(404, "not found")


class SecurityIn(pydantic.BaseModel):
    reload: bool = True


def _peers():
    return [m for m in cfg.machines.all() if not local_node.is_local(m)]


def _reload_swap():
    try:
        if swap_manager.status().get("running"):
            swap_manager.reload_daemon_async()
    except Exception:
        pass


@router.get("/api/rpc/security/status")
async def status():
    _require_control()
    return {"settings": rpc_tls.status(),
            "machines": [{"name": m.get("name"), "ip": m.get("ip")}
                         for m in _peers()]}


@router.post("/api/rpc/security/enable")
async def enable(body: SecurityIn = SecurityIn()):
    _require_control()
    rpc_tls.ensure_ca()
    rpc_tls.issue_master_cert()
    failures = []
    installed = []
    for m in _peers():
        name = m.get("name")
        try:
            bundle = rpc_tls.cert_bundle(m)
            agent_client.tls_install(m, bundle["cert"], bundle["key"],
                                     bundle["ca"])
            hs = agent_client.tls_https_start(m)
            if not hs.get("running"):
                raise RuntimeError(
                    f"agent HTTPS failed: {hs.get('error') or hs}")
            st = agent_client.tls_status(m)
            if int(st.get("tunnel_port") or 0) != 5399:
                raise RuntimeError(
                    "agent does not support the single-port RPC tunnel; "
                    "update the agent first")
            try:
                cfg.machines.update(name, {"agent_tls": True,
                                           "agent_tls_port": 8197})
            except Exception:
                pass
            installed.append(name)
        except Exception as e:
            failures.append({"machine": name, "error": str(e)})
    if failures:
        return {"ok": False, "enabled": False, "installed": installed,
                "failures": failures,
                "detail": "security left OFF; fix the failures and retry"}
    rpc_tls.set_enabled(True)
    refresh = await run_in_threadpool(rpc_tls.refresh_local_proxies)
    try:
        await run_in_threadpool(swap_manager.push_scripts)
    except Exception as e:
        failures.append({"machine": "*", "error": f"push: {e}"})
    if body.reload:
        _reload_swap()
    return {"ok": not failures, "enabled": True, "installed": installed,
            "failures": failures, "refresh": refresh}


@router.post("/api/rpc/security/disable")
async def disable(body: SecurityIn = SecurityIn()):
    _require_control()
    rpc_tls.set_enabled(False)
    rpc_tls.stop_all_local_proxies()
    stopped = []
    for m in _peers():
        try:
            agent_client.tls_stop(m)
            stopped.append(m.get("name"))
        except Exception:
            pass
    try:
        await run_in_threadpool(swap_manager.push_scripts)
    except Exception:
        pass
    if body.reload:
        _reload_swap()
    return {"ok": True, "enabled": False, "tls_stopped": stopped}


@router.post("/api/rpc/security/refresh")
async def refresh():
    _require_control()
    return await run_in_threadpool(rpc_tls.refresh_local_proxies)
