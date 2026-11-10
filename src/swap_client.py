"""Low-level llama-swap HTTP helpers shared by route modules and startup."""

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

import config_loader as cfg
import local_node
import swap_manager


def _swap_api_url(st, path):
    host = ("127.0.0.1"
            if local_node.is_local(cfg.machines.require(st["host"]))
            else st.get("host_ip"))
    return f"http://{host}:{st['port']}{path}"


def _swap_api_get(st, path, timeout=5):
    import urllib.request, json as _json
    with urllib.request.urlopen(_swap_api_url(st, path), timeout=timeout) as r:
        return _json.load(r)


def _swap_host_unload(model_id: str):
    """POST /api/models/unload/{model_id} on the swap host — llama-swap then
    runs the model entry's cmdStop. Shared by /api/swap/evict/{model_id}
    and the remote-proxy branch of /api/launch/stop: one proxy body so the
    two entry points can never drift. Swap down raises ValueError."""
    st = swap_manager.status()
    if not st.get("running"):
        raise ValueError("swap not running")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = (f"http://127.0.0.1:{st['port']}/api/models/unload/{model_id}"
           if is_local else
           f"http://{st.get('host_ip')}:{st['port']}/api/models/unload/{model_id}")
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return {"ok": True, "status": r.status,
                    "body": r.read().decode()[:500]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code,
                "error": e.read().decode()[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _proxy_swap_unload(model_id: str):
    """Proxy an unload to the swap host. No-op if swap down."""
    try:
        return await run_in_threadpool(_swap_host_unload, model_id)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
