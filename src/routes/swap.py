"""llama-swap launch/stop/pin/unload actions."""
from concurrent.futures import ThreadPoolExecutor

import pydantic
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import config_loader as cfg
import engine_registry
import local_node
import swap_captures
import rpc_manager
import swap_manager
from swap_client import _proxy_swap_unload, _swap_api_get
from web_shared import _gateway_headers, _gateway_url

router = APIRouter()


# ---------------- llama-swap: launch all ----------------
class SwapIn(pydantic.BaseModel):
    binary: str | None = None
    listen_port: int | None = None
    config_path: str | None = None
    host_name: str | None = None
    launch_config: dict | None = None


@router.get("/api/swap/settings")
async def swap_settings():
    try:
        return swap_manager.load_settings()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/swap/settings")
async def save_swap_settings(s: SwapIn):
    return swap_manager.save_settings(
        s.binary, s.listen_port, s.config_path, s.host_name,
        s.launch_config)


@router.get("/api/swap/preview")
async def swap_preview(config_path: str | None = None,
                       host_name: str | None = None):
    """Generated llama-swap config (no side effects)."""
    # F4 (t_90def951): engine×OS gate — the preview is the plan the launch
    # would push, so engine-impossible hosts surface here as warnings
    # (validate_hosts has no call site until now). Non-fatal: the plan is
    # still returned so the operator can see exactly what is wrong.
    try:
        engine = engine_registry.get_engine()
        host_errors = []
    except ValueError:
        engine = None
        host_errors = []
    if engine is not None:
        try:
            host_errors = engine.validate_hosts(
                [m for m in cfg.machines.all()])
        except Exception:
            host_errors = []
        if host_errors:
            print("[swap-preview] engine host gate: "
                  + "; ".join(host_errors))
    try:
        # DEC-06: the plan is the adapter's launch_plan (engine-owned
        # planner — swap_manager.launch_plan behind the adapter seam).
        plan = engine.launch_plan(config_path or None, host_name or None) \
            if engine is not None \
            else swap_manager.launch_plan(config_path or None,
                                          host_name or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if engine is not None and host_errors:
        plan = dict(plan)
        plan["warnings"] = list(plan.get("warnings") or []) + host_errors
    return plan


@router.post("/api/swap/push")
async def swap_push():
    """Regenerate + write swap config/scripts from current model configs.
    No peer/daemon restart; already-loaded instances keep serving until
    evicted, next load picks up the new scripts."""
    return await run_in_threadpool(swap_manager.push_scripts)


@router.get("/api/swap/reload-status")
async def swap_reload_status():
    """Outcome of the save-triggered daemon reload (polled by the UI)."""
    return dict(swap_manager.reload_state)


@router.post("/api/swap/launch")
async def swap_launch(s: SwapIn):
    """Generate + push config, start RPC peers, start llama-swap daemon."""
    try:
        report = await run_in_threadpool(
            swap_manager.launch, s.binary, s.listen_port, s.config_path,
            s.host_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    status = 500 if report.get("failed") else 200
    return JSONResponse(report, status_code=status)


@router.post("/api/swap/stop")
async def swap_stop():
    """Stop the llama-swap daemon AND any remote-proxy llama-servers
    (launch-mode=local models run llama-server on their own box — swap
    going down must not leave those orphans serving). stop_rpc(name)
    kills llama-swap/llama-server/rpc-server on every machine, both OS
    types; run it everywhere after the daemon is down."""
    out = await run_in_threadpool(swap_manager.stop)
    names = [m["name"] for m in cfg.machines.all()]
    with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
        out["machines"] = list(pool.map(rpc_manager.stop_rpc, names))
    return out


@router.get("/api/swap/status")
async def swap_status():
    return swap_manager.status()


@router.get("/api/swap/loaded")
async def swap_loaded_models(request: Request):
    """Currently loaded model(s) — FILTERED BY CALLER IDENTITY:
    admins/console see every loaded model; a key holder sees only the
    ones their key grants (whose model loaded/evicted when is household
    telemetry a non-admin never sees). Same llama-swap /v1/models read
    (status.value == "loaded"), empty when swap is down."""
    st = swap_manager.status()
    if not st.get("running"):
        return {"loaded": [], "running": False}
    def fetch():
        import urllib.request, json as _json
        import access as _a
        url = f"http://127.0.0.1:{st['port']}/v1/models" \
            if local_node.is_local(cfg.machines.require(st["host"])) \
            else f"http://{st.get('host_ip')}:{st['port']}/v1/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.load(r)
        ids = [m["id"] for m in data.get("data", [])
               if (m.get("status") or {}).get("value") == "loaded"]
        if (_a.gate_open() and _a.view_allow() is None) \
                or _a.admin_sees_everything():
            return ids
        # scoped caller: filter against THEIR grants. The middleware
        # stashed the raw record for this request — use it (no token
        # passes through user code)
        rec = _a.raw_key_record()
        if rec is None:
            return []
        allow = rec.get("models") or set()
        from swap_gateway import _mid_allowed
        return [i for i in ids if _mid_allowed(i, allow)]
    try:
        return {"loaded": await run_in_threadpool(fetch),
                "running": True, "host": st["host"], "port": st["port"]}
    except Exception as e:
        return {"loaded": [], "running": True, "error": str(e)}


# ---------------- llama-swap activity & performance ----------------
# llama-swap records every proxied request (/api/metrics/activity); the
# farm shows it on its own Activity tab. This endpoint only READS it.

def _caller_model_filter():
    """None when the caller sees every model (console / unrestricted
    admin / open gate); else a predicate over model ids built from the
    caller's grants — the same rule /api/swap/loaded applies."""
    import access as _a
    if (_a.gate_open() and _a.view_allow() is None) \
            or _a.admin_sees_everything():
        return None
    rec = _a.raw_key_record()
    if rec is None:
        return lambda mid: False
    allow = rec.get("models") or set()
    from swap_gateway import _mid_allowed
    return lambda mid: _mid_allowed(str(mid or ""), allow)


def _speed_stats(rows):
    """Speeds over successful rows, in two honest forms:
    median_tps     - median of per-request input tok/s: how fast the model
                     works on a request; one cold-start load (a 60 s first
                     request) cannot drag it down.
    throughput_tps - input tokens / wall-clock time from the first
                     request's start to the last one's end: what the
                     server actually delivered, counting idle gaps and
                     requests that ran at the same time correctly.
    Summing per-request durations (the old figure) did neither."""
    from datetime import datetime
    import statistics as _stats
    ok = [r for r in rows if (r.get("status") or 0) < 400
          and (r.get("input_tokens") or 0) > 0 and (r.get("duration_ms") or 0) > 0]
    tps = [r["input_tps"] for r in ok if r.get("input_tps")]
    starts, ends = [], []
    for r in ok:
        try:
            end = datetime.fromisoformat(str(r["ts"])).timestamp()
        except (TypeError, ValueError):
            continue
        ends.append(end)
        starts.append(end - r["duration_ms"] / 1000.0)
    span = (max(ends) - min(starts)) if ends else 0
    tokens = sum(r["input_tokens"] for r in ok)
    return {"median_tps": round(_stats.median(tps), 1) if tps else None,
            "throughput_tps": round(tokens / span, 1) if span > 0 else None,
            "window_s": round(span, 1) if span else None}


def _activity_row(r):
    """One llama-swap activity row, trimmed to what the tab shows.
    Embedding responses carry no llama-server timings (speeds are -1),
    so input tokens per second is derived from the request duration."""
    tok = r.get("tokens") or {}
    def _n(k):
        v = tok.get(k)
        return v if isinstance(v, (int, float)) and v >= 0 else None
    dur = r.get("duration_ms") or 0
    inp, out = _n("input_tokens"), _n("output_tokens")
    return {
        "id": r.get("id"), "ts": r.get("timestamp"), "src": r.get("src"),
        "model": r.get("model"), "task": r.get("task"),
        "path": r.get("req_path"), "status": r.get("resp_status_code"),
        "duration_ms": dur, "input_tokens": inp, "output_tokens": out,
        "cache_tokens": _n("cache_tokens"),
        "prompt_tps": _n("prompt_per_second"),
        "gen_tps": _n("tokens_per_second"),
        "input_tps": (round(inp / (dur / 1000.0), 1)
                      if inp and dur > 0 else None),
        "capture_id": r.get("capture_id") if r.get("has_capture") else None,
        "kept": bool(r.get("capture_id")
                     and swap_captures.get(r.get("capture_id"))),
    }


@router.get("/api/swap/activity")
async def swap_activity(limit: int = 200):
    """Recent llama-swap requests plus per-model totals over them.
    Scoped keys see only rows for models they are granted, without the
    client address."""
    st = swap_manager.status()
    if not st.get("running"):
        return {"running": False, "rows": [], "models": []}
    limit = max(1, min(int(limit or 200), 1000))
    allowed = _caller_model_filter()

    def fetch():
        data = _swap_api_get(st, f"/api/metrics/activity?limit={limit}&page=1")
        rows = [_activity_row(r) for r in (data.get("data") or [])]
        if allowed is not None:
            rows = [dict(r, src=None, capture_id=None, kept=False)
                    for r in rows if allowed(r["model"])]
        per, per_rows = {}, {}
        for r in rows:
            per_rows.setdefault(r["model"] or "?", []).append(r)
            m = per.setdefault(r["model"] or "?", {
                "model": r["model"] or "?", "requests": 0, "errors": 0,
                "input_tokens": 0, "output_tokens": 0, "duration_ms": 0,
                "gen_tps": [], "task": r["task"]})
            m["requests"] += 1
            if (r["status"] or 0) >= 400:
                m["errors"] += 1
            m["input_tokens"] += r["input_tokens"] or 0
            m["output_tokens"] += r["output_tokens"] or 0
            m["duration_ms"] += r["duration_ms"] or 0
            if r["gen_tps"]:
                m["gen_tps"].append(r["gen_tps"])
        models = []
        for m in per.values():
            g = m.pop("gen_tps")
            m.update(_speed_stats(per_rows[m["model"]]))
            m["gen_tps"] = round(sum(g) / len(g), 1) if g else None
            models.append(m)
        models.sort(key=lambda m: -m["requests"])
        return {"running": True, "host": st["host"], "rows": rows,
                "models": models, "speed": _speed_stats(rows),
                "total": data.get("total"),
                "scoped": allowed is not None}
    try:
        return await run_in_threadpool(fetch)
    except Exception as e:
        return {"running": True, "rows": [], "models": [], "error": str(e)}


@router.get("/api/swap/capture/{capture_id}")
async def swap_capture(capture_id: str):
    """Headers and body of one proxied request, for the Activity tab's
    detail view. Failed requests come from the farm's own copy
    (swap_captures); others are fetched live while llama-swap still has
    them. Credentials are redacted. Request bodies can hold other users'
    text, so only callers that see every model may read them."""
    if _caller_model_filter() is not None:
        raise HTTPException(status_code=403,
                            detail="request details are admin-only")
    kept = swap_captures.get(capture_id)
    if kept:
        return dict(kept, source="kept")
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="llama-swap is not running")
    from urllib.parse import quote as _q

    def fetch():
        return _swap_api_get(st, f"/api/captures/{_q(capture_id, safe='')}")
    try:
        cap = await run_in_threadpool(fetch)
    except Exception:
        cap = None
    if not isinstance(cap, dict) or "error" in cap:
        raise HTTPException(
            status_code=404,
            detail="llama-swap no longer keeps this request (it drops "
                   "captures within about 30 s; the farm keeps failed "
                   "requests only)")
    return dict(swap_captures.summarize(cap, {"capture_id": capture_id}),
                source="live")


@router.post("/api/swap/load/{model_id}")
async def swap_load_model(model_id: str, request: Request):
    """Load one model through the running swap daemon: this fires a minimal
    warmup completion (max_tokens=1) at the daemon and waits for it. First
    load takes a while (weight load + warmup); the daemon-side timeout is
    generous for that reason. 404 for unknown models, 503 when the daemon
    is down. Same urllib-in-threadpool idiom as swap_loaded_models
    (stdlib only, no extra deps)."""
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503,
                            detail="swap daemon is not running — Launch All first")
    import time as _time

    def warmup():
        import urllib.request, urllib.error, json as _json
        # Go through the farm gateway (:8090) with the logged-in user's
        # identity (session cookie → access key), exactly like API clients.
        base = _gateway_url("")
        is_embed = False
        is_decision = False
        try:
            engine = engine_registry.get_engine()
            base_id = model_id.split("--")[0]
            emb_ids = {m.get("model_id") for m in
                       (engine.filter_embedding_models(cfg.list_models()) or [])}
            dec_ids = {m.get("model_id") for m in
                       (engine.filter_decision_models(cfg.list_models()) or [])}
            is_embed = base_id in emb_ids
            is_decision = base_id in dec_ids
        except Exception:
            pass
        if is_decision:
            payload = _json.dumps({
                "model": model_id, "state": "warmup",
                "questions": {"route": {"type": "choice",
                                        "instructions": "Pick one",
                                        "criteria": {"yes": 1, "no": 2}}},
            }).encode()
            path = "/v1/systemone"
        elif is_embed:
            payload = _json.dumps(
                {"model": model_id, "input": "hi"}).encode()
            path = "/v1/embeddings"
        else:
            payload = _json.dumps(
                {"model": model_id,
                 "messages": [{"role": "user", "content": "hi"}],
                 "max_tokens": 1}).encode()
            path = "/v1/chat/completions"
        req = urllib.request.Request(
            base + path, data=payload,
            headers=_gateway_headers(request), method="POST")
        t0 = _time.time()
        body = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    body = _json.load(r)
                break
            except urllib.error.HTTPError as e:
                detail = e.read(300).decode("utf-8", "replace")
                # App startup reloads llama-swap (stop+start); a load that
                # lands in that window gets a transient 5xx. Retry briefly.
                if e.code in (500, 502, 503) and attempt < 2:
                    _time.sleep(2)
                    continue
                raise RuntimeError(
                    f"daemon refused load: HTTP {e.code} {detail}")
        if body is None:
            raise RuntimeError("daemon refused load: no response")
        choice = ((body.get("choices") or [{}])[0].get("message") or {})
        preview = (choice.get("content") or "")
        if not preview and body.get("data"):
            preview = "(embedding ok)"
        return {"ok": True, "model": model_id,
                "elapsed_s": round(_time.time() - t0, 1),
                "preview": preview[:120]}
    try:
        return await run_in_threadpool(warmup)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(status_code=504,
                            detail=f"load timed out or daemon unreachable: {e}") from None


class PinIn(pydantic.BaseModel):
    pinned: bool = False


@router.post("/api/models/{model_id}/pin")
async def pin_model(model_id: str, p: PinIn):
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    m["pinned"] = bool(p.pinned)
    # ttl 0 = never for pinned, else default 600
    if p.pinned:
        m["ttl"] = 0
    return cfg.save_model(model_id, m)


class SwapDisableIn(pydantic.BaseModel):
    disabled: bool = False


@router.post("/api/models/{model_id}/swap-disable")
async def swap_disable_model(model_id: str, p: SwapDisableIn):
    """Launch-page disable switch: when true, the model (config-level,
    so ALL deployments inherit — variants and primary) is omitted from
    every push/preview of the swap config until re-enabled. Pure config
    flag; applying it to a running daemon takes the next push (the swap
    push / Launch All flow already ends in one)."""
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    m["swap_disabled"] = bool(p.disabled)
    return cfg.save_model(model_id, m)


@router.get("/api/models/{model_id}/swap-disable")
async def swap_disable_get(model_id: str):
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return {"model_id": model_id, "swap_disabled": bool(m.get("swap_disabled"))}


@router.post("/api/swap/evict/{model_id}")
async def swap_evict(model_id: str):
    """Unload a single model from swap (frees its VRAM/RAM). Proxies to
    POST /api/models/unload/{model_id} on the swap host. No-op if swap down."""
    return await _proxy_swap_unload(model_id)


@router.post("/api/swap/evict-all")
async def swap_evict_all():
    st = swap_manager.status()
    if not st.get("running"):
        raise HTTPException(status_code=503, detail="swap not running")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = f"http://127.0.0.1:{st['port']}/api/models/unload" if is_local else f"http://{st.get('host_ip')}:{st['port']}/api/models/unload"
    def do():
        import urllib.request, urllib.error
        req = urllib.request.Request(url, data=b"", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return {"ok": True, "status": r.status, "body": r.read().decode()[:500]}
        except urllib.error.HTTPError as e:
            return {"ok": False, "status": e.code, "error": e.read().decode()[:500]}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return await run_in_threadpool(do)
