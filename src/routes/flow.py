"""Live llama-swap flow / log WebSocket stream.

Parses llama-swap proxy + scheduler log lines and streams normalized events
to the visualizer. Owns no app state beyond the log parsers.
"""
import json
import os
import re
import time

from fastapi import APIRouter, WebSocket
from fastapi.concurrency import run_in_threadpool

import access
import config_loader as cfg
import local_node
import rpc_manager
import swap_manager

router = APIRouter()


# ---------------- flow visualizer (live llama-swap proxy log stream) ----------------
import re as _re

_VIS_LOG_RE = _re.compile(
    r'Request\s+(\S+)\s+"(\S+)\s+(\S+)\s+HTTP/\d\.\d"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+([\d.]+)(\w+)'
)


def _parse_proxy_line(line):
    """Parse a llama-swap proxy log line into a structured event dict, or None."""
    m = _VIS_LOG_RE.search(line)
    if not m:
        return None
    client, method, path, status, size, ua, dur, unit = m.groups()
    # normalize duration to seconds
    mult = {"µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}.get(unit.lower(), 1.0)
    return {
        "client": client,
        "method": method,
        "path": path,
        "status": int(status),
        "bytes": int(size),
        "ua": ua,
        "duration_s": round(float(dur) * mult, 6),
    }


_UPSTREAM_CHUNK_RE = _re.compile(
    r'http: streamed chunk: data: \{"choices":\[\{"finish_reason":null,.*?"model":"([^"]+)"'
)
# GGML_SCHED_DEBUG=2 node line: node #N (op): name (size) [backend cause] use=..,c=..: srcs...
# backend is %5.5s, cause is %8.8s (may be empty -> trailing spaces)
_SCHED_NODE_RE = _re.compile(
    r'node\s*#\s*\d+\s*\([^)]*\):\s*([^ ]+)\s*\([^)]*\)\s*\[([^ ]+)\s*([^ ]*)\]\s*use=\d+,c=\d+:(.*)'
)
# a src tensor in the tail: " name (size) [backend cause]"
_SCHED_SRC_RE = _re.compile(r'\s+([^ ]+)\s*\([^)]*\)\s*\[([^ ]+)\s*([^ ]*)\]')



def _parse_sched_line(line):
    """Parse a GGML_SCHED_DEBUG=2 node line into a structured event, or None."""
    m = _SCHED_NODE_RE.search(line)
    if not m:
        return None
    name, backend, cause, tail = m.groups()
    srcs = []
    for sm in _SCHED_SRC_RE.finditer(tail):
        srcs.append({"name": sm.group(1), "backend": sm.group(2), "cause": sm.group(3)})
    return {"type": "sched", "tensor": name, "backend": backend,
            "cause": cause, "srcs": srcs}


# Common model-load failures seen in llama.cpp / llama-swap upstream logs.
# Order matters: the most specific cause wins (OOM before the MTP-context
# error it caused, missing file before the generic loader error).
_LOAD_FAULT_RULES = (
    ("oom", re.compile(
        r"cudaMalloc failed|CUDA error: out of memory|out of memory|"
        r"failed to allocate .*buffer|failed to allocate .*buffers|"
        r"compute buffer allocation failed|failed to allocate CUDA",
        re.IGNORECASE),
     "CUDA out of memory. Lower ubatch/context, disable MTP, or free VRAM and retry."),
    ("port", re.compile(
        r"address already in use|failed to bind|couldn't bind|"
        r"port .* is held by another process",
        re.IGNORECASE),
     "Port already in use. Stop the stale server or let the farm roll the port."),
    ("missing_model", re.compile(
        r"failed to open GGUF file|No such file or directory|"
        r"failed to load model from|model file not found|"
        r"failed to load model .*not found",
        re.IGNORECASE),
     "Model file is missing or unreadable at the configured store path."),
    ("mtp", re.compile(
        r"failed to (?:create|load) (?:the )?MTP(?: context)?|"
        r"doesn't contain MTP layers|"
        r"\bMTP\b[^\n]{0,80}\b(?:fail(?:ed|ure)?|error)\b|"
        r"\b(?:fail(?:ed|ure)?|error)\b[^\n]{0,80}\bMTP\b",
        re.IGNORECASE),
     "MTP/speculative context failed. Disable MTP or use a model with MTP layers."),
    ("mmproj", re.compile(
        r"failed to (?:load|open) (?:the )?"
        r"(?:mmproj|clip(?:[ _-]?model)?|projector)(?: model)?|"
        r"(?:couldn't|could not|unable to) (?:load|open) (?:the )?"
        r"(?:mmproj|clip(?:[ _-]?model)?|projector)|"
        r"\b(?:mmproj|clip_model_load|projector)\b[^\n]{0,80}"
        r"\b(?:fail(?:ed|ure)?|error)\b|"
        r"\b(?:fail(?:ed|ure)?|error)\b[^\n]{0,80}"
        r"\b(?:mmproj|clip_model_load|projector)\b",
        re.IGNORECASE),
     "Vision projector (mmproj) failed to load or does not match the model."),
    ("rpc_connect", re.compile(
        r"Failed to connect to \S+:\d+", re.IGNORECASE),
     "Cannot reach an RPC peer. Check that its rpc-server is running and "
     "that the peer's firewall allows the port from this host."),
    ("load_error", re.compile(
        r"failed to load model|error loading model|"
        r"exiting due to model loading error",
        re.IGNORECASE),
     "Model failed to load. See Logs for the exact upstream error."),
)

_WRAPPER_GEN_RE = re.compile(r">>> wrapper-gen:\s*(\S+)")
_LOAD_MODEL_RE = re.compile(r"load_model: loading model '([^']+)'")
_START_FAIL_RE = re.compile(r"starting\s+(.+?)\s+failed:\s*(.*)")
# Load finished cleanly: stop watching this model's output for startup
# faults (streamed request/response bodies can otherwise look like errors).
_MODEL_READY_RE = re.compile(
    r"model loaded|listening on https?://|server is listening", re.IGNORECASE)
# Request/response chatter (not startup diagnostics). Never treat these as
# model-load failures — model output is arbitrary text.
_TRAFFIC_RE = re.compile(
    r"streamed chunk|\"choices\"|\"object\"\s*:\s*\"(?:chat|list)|"
    r"^\s*(?:POST|GET) /|\]\s*Request ", re.IGNORECASE)


def _parse_load_fault(line):
    """Return (code, human_reason) for a recognizable load failure, else None."""
    for code, rx, reason in _LOAD_FAULT_RULES:
        if rx.search(line or ""):
            return code, reason
    return None


def _load_model_map():
    """Cheap basename -> model_id map (YAML only; no GGUF header reads)."""
    import yaml as _yaml
    out = {}
    try:
        names = os.listdir(cfg.MODELS_DIR)
    except Exception:
        return out
    for fn in names:
        if not fn.endswith(".yaml") or "__tune_tmp" in fn or "__ab_single" in fn:
            continue
        try:
            with open(os.path.join(cfg.MODELS_DIR, fn), "r",
                      encoding="utf-8") as f:
                d = _yaml.safe_load(f) or {}
        except Exception:
            continue
        bp = os.path.basename(str(d.get("base_path") or ""))
        if bp:
            out[bp] = d.get("model_id") or fn[:-5]
    return out


def _resolve_fault_model(raw, mmap):
    raw = str(raw or "").strip().strip("'\"")
    if not raw:
        return ""
    return mmap.get(os.path.basename(raw), raw)


@router.websocket("/ws/swap/faults")
async def swap_faults_ws(websocket: WebSocket):
    """Persistent top-right load-failure notices.

    Tails llama-swap's proxy+upstream log streams, recognizes the common
    llama.cpp/llama-swap startup failures (OOM, port, missing model, MTP,
    mmproj, generic load errors), and emits ONE structured event per model
    per short cooldown. The browser renders it as a sticky notice until the
    user dismisses it. History replay is primed away (~1.5 s)."""
    import httpx, asyncio
    await websocket.accept()
    st = swap_manager.status()
    if not st.get("running"):
        await websocket.close()
        return
    gate_open = access.gate_open()
    allow = access.view_allow()
    if not gate_open and allow is None:
        await websocket.close(code=1008)
        return
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    urls = {
        "proxy": f"http://{host}:{port}/logs/stream/proxy",
        "upstream": f"http://{host}:{port}/logs/stream/upstream",
    }
    mmap = _load_model_map()
    priming_until = time.time() + 1.5
    state = {"model": "", "loading": False, "loading_model": "",
             "hint_ts": 0.0, "last_emit": {}, "specific": {},
             "last_code": {}, "pending": {}}
    dead = asyncio.Event()

    def _allowed(model):
        if gate_open or allow is None:
            return True
        if not model:
            return False
        from swap_gateway import _mid_allowed
        return (_mid_allowed(model, allow)
                or _mid_allowed(model.split("--")[0], allow))

    async def _emit(model, code, reason, detail):
        if not _allowed(model):
            return
        now = time.time()
        key = model or "?"
        prev_code = state["last_code"].get(key)
        # a specific cause always overrides a generic "exited prematurely"
        override = (code != "generic" and prev_code == "generic")
        if not override and now - state["last_emit"].get(key, 0) < 6:
            return
        state["last_emit"][key] = now
        state["last_code"][key] = code
        if code != "generic":
            state["specific"][key] = now
        try:
            await websocket.send_text(json.dumps({
                "type": "load_failure",
                "model": model,
                "code": code,
                "reason": reason,
                "detail": (detail or "").strip()[:600],
                "ts": now,
            }))
        except Exception:
            dead.set()

    async def tail(kind, url):
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    async for line in resp.aiter_lines():
                        if dead.is_set():
                            return
                        if not line:
                            continue
                        # model attribution hints
                        m = (_WRAPPER_GEN_RE.search(line)
                             or _LOAD_MODEL_RE.search(line))
                        if m:
                            state["model"] = _resolve_fault_model(
                                m.group(1), mmap)
                            state["loading"] = True
                            state["loading_model"] = state["model"]
                            state["hint_ts"] = time.time()
                        if time.time() < priming_until:
                            continue
                        # Startup finished — everything after this is normal
                        # serving output (including arbitrary model text).
                        if _MODEL_READY_RE.search(line):
                            state["loading"] = False
                            # Startup recovered (llama.cpp/llama-swap may
                            # retry after a transient allocation failure).
                            # A held fault is therefore FALSE — drop it.
                            state["pending"].pop(
                                state["loading_model"] or "?", None)
                            continue
                        # llama-swap's own "starting <model> failed" line
                        sm = _START_FAIL_RE.search(line)
                        if sm:
                            model = _resolve_fault_model(sm.group(1), mmap)
                            key = model or "?"
                            now = time.time()
                            pend = state["pending"].pop(key, None)
                            if pend and now - pend["ts"] <= 300:
                                # Attach the specific cause we saw during
                                # this attempt (OOM/MTP/mmproj…) to the
                                # authoritative failure line.
                                await _emit(model, pend["code"],
                                            pend["reason"], pend["detail"])
                            elif now - state["specific"].get(key, 0) >= 120:
                                await _emit(
                                    model, "generic",
                                    "Model failed to load (upstream exited prematurely).",
                                    sm.group(2) or line)
                            continue
                        # Only interpret startup-log lines as load faults, and
                        # only while the model we saw start is still loading.
                        # This stops unrelated logs / streamed responses from
                        # being blamed on whichever model was mentioned last.
                        if not state["loading"]:
                            continue
                        if _TRAFFIC_RE.search(line):
                            continue
                        if time.time() - state["hint_ts"] > 900:
                            state["loading"] = False
                            continue
                        fault = _parse_load_fault(line)
                        if fault:
                            code, reason = fault
                            key = state["loading_model"] or "?"
                            # Hold the cause; do not notify yet. If the
                            # model becomes ready we discard it (this was
                            # the false "CUDA out of memory" while a model
                            # loaded fine); a real failure surfaces via
                            # llama-swap's "starting … failed" line.
                            prev = state["pending"].get(key)
                            if not prev or prev.get("code") == "generic":
                                state["pending"][key] = {
                                    "code": code, "reason": reason,
                                    "detail": line, "ts": time.time()}
                            continue
        except Exception:
            pass

    async def watch_client():
        while not dead.is_set():
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=10)
                if msg.get("type") == "websocket.disconnect":
                    dead.set()
                    return
            except asyncio.TimeoutError:
                continue
            except Exception:
                dead.set()
                return

    tasks = [asyncio.create_task(tail("proxy", urls["proxy"])),
             asyncio.create_task(tail("upstream", urls["upstream"])),
             asyncio.create_task(watch_client())]
    try:
        _done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception:
        for t in tasks:
            t.cancel()
    try:
        await websocket.close()
    except Exception:
        pass


def _parse_upstream_line(line):
    """Extract a flow event from one upstream log line.

    llama.cpp: the verbose "http: streamed chunk" line carries the model
    path per token chunk. Returns {"model": id} shaped events.
    """
    m = _UPSTREAM_CHUNK_RE.search(line)
    if m:
        return {"model": m.group(1)}
    return None


@router.websocket("/ws/flow/upstream")
async def flow_upstream_ws(websocket: WebSocket):
    """Tail llama-swap's upstream log stream and push real-time flow events
    to the browser: llama.cpp token-chunk events (model path)."""
    import httpx, asyncio
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    url = f"http://{host}:{port}/logs/stream/upstream"
    # Identity gate (access.py — the middleware resolved this ws scope's
    # Bearer key / session cookie exactly like a page request). A scoped
    # viewer must not watch token streams of models outside their grants:
    # every chunk event names the model path, which would leak ungranted
    # ids AND their live/loaded state. Gate-open viewers (trusted console,
    # unrestricted key) stream everything as before. Unresolvable
    # identities (anonymous ws on loopback never happens through the UI;
    # a scoped caller with allow None = unknown key) see nothing.
    _flow_allow = None if access.gate_open() else access.view_allow()
    if _flow_allow is None and not access.gate_open():
        await websocket.close(code=1008)
        return

    def _event_allowed(ev):
        """Scoped-key filter: model path → base mid (strip variant label —
        llama.cpp streams the plain base GGUF path for variants) must sit
        in the caller's allowlist with the container rule (_mid_allowed
        semantics — a granted mid--variant token lights its base's
        stream). No allow = open."""
        if _flow_allow is None:
            return True
        from swap_gateway import _mid_allowed
        raw = (ev.get("model") or "").strip()
        base = raw.split("/")[-1].split("--")[0]
        return bool(base) and _mid_allowed(base, _flow_allow)
    # llama-swap replays buffered history on connect — prime silently so a
    # page reload doesn't flare up "serving" state from old token chunks
    priming = True
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as resp:
                async def unprime_later():
                    nonlocal priming
                    await asyncio.sleep(1.5)
                    priming = False
                primer = asyncio.create_task(unprime_later())
                try:
                    async for line in resp.aiter_lines():
                        ev = _parse_upstream_line(line)
                        if not ev:
                            continue
                        if ev and not priming and _event_allowed(ev):
                            try:
                                await websocket.send_text(json.dumps(ev))
                            except Exception:
                                primer.cancel()
                                return
                finally:
                    primer.cancel()
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass


@router.websocket("/ws/flow/sched")
async def flow_sched_ws(websocket: WebSocket):
    """Tail llama-swap's upstream (llama-server) log stream and push parsed
    GGML_SCHED_DEBUG=2 per-tensor backend-assignment events to the browser.
    Also tails every RPC peer's rpc-server.log so single-RPC models (e.g.
    winbox-only) show their real per-GPU work, not just the host's proxy."""
    import httpx, asyncio
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    # tail host upstream + every peer's rpc log in parallel
    peers = [m for m in cfg.machines.all() if not local_node.is_local(m)]

    async def tail_host():
        url = f"http://{host}:{port}/logs/stream/upstream"
        # llama-swap REPLAYS its buffered log history on connect — without
        # a priming pass, opening the Visualizer flares up every GPU with
        # hours-old activity. Consume silently first, only forward lines
        # that arrive after the stream settles.
        priming = True
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    async def unprime_later():
                        nonlocal priming
                        await asyncio.sleep(1.5)
                        priming = False
                    primer = asyncio.create_task(unprime_later())
                    try:
                        async for line in resp.aiter_lines():
                            ev = _parse_sched_line(line)
                            if ev and not priming:
                                try:
                                    await websocket.send_text(json.dumps(ev))
                                except Exception:
                                    primer.cancel()
                                    return
                    finally:
                        primer.cancel()
        except Exception:
            pass

    async def tail_peer(machine):
        import local_node as _ln
        is_win = machine.get("os", "linux") == "windows"
        is_local = _ln.is_local(machine)
        seen = set()
        # first poll returns the CURRENT log tail (hours-old lines) — seed
        # the dedup set with it silently so reload doesn't replay old
        # activity as fresh events
        seeded = False
        while True:
            try:
                log = await run_in_threadpool(
                    rpc_manager._tail_all_rpc_logs, machine, is_win, is_local)
                if not seeded:
                    seeded = True
                    seen.update((log or "").splitlines())
                    await asyncio.sleep(1)
                    continue
                for line in (log or "").splitlines():
                    if line in seen:
                        continue
                    seen.add(line)
                    # keep the set bounded (drop OLDEST lines only —
                    # seen.clear() replayed the whole tail every second,
                    # flooding the browser with duplicate events)
                    if len(seen) > 500:
                        seen = set(sorted(seen)[-500:])
                    ev = _parse_sched_line(line)
                    if ev:
                        try:
                            await websocket.send_text(json.dumps(ev))
                        except Exception:
                            return
            except Exception:
                pass
            await asyncio.sleep(1)

    try:
        await asyncio.gather(tail_host(), *[tail_peer(m) for m in peers])
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass


@router.websocket("/ws/flow")
async def flow_ws(websocket: WebSocket):
    """Tail llama-swap's proxy log stream (SSE over HTTP) and push parsed request
    events to the browser in real time. Identity gate (access.py): a
    scoped viewer must not watch other identities' request telemetry —
    client IPs, user agents, request shapes are household telemetry the
    same rule /api/swap/loaded applies to loaded lists."""
    import httpx
    if not access.gate_open() and access.view_allow() is None:
        await websocket.accept()
        await websocket.close(code=1008)
        return
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    url = f"http://{host}:{port}/logs/stream/proxy"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as resp:
                async for line in resp.aiter_lines():
                    ev = _parse_proxy_line(line)
                    if ev:
                        try:
                            await websocket.send_text(json.dumps(ev))
                        except Exception:
                            return
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
    try:
        await websocket.close()
    except Exception:
        pass


@router.websocket("/ws/logs/stream")
async def logs_stream_ws(websocket: WebSocket):
    """Logs tab: live raw tail of llama-swap's proxy + upstream log streams
    (SSE over HTTP), pushed to the browser as plain lines. Untouched text —
    the visualizer parses structured events from the same feeds; this tab
    shows them verbatim for real-time debugging."""
    # B15 verified: the source here is llama-swap's own /logs/stream/* on the
    # swap host:port (swap_manager.status). (Per-model parse layers: flow's
    # token-chunk regex keys on "model":"<id>"; sched events are
    # GGML_SCHED_DEBUG llama.cpp-only.)
    import httpx, asyncio
    await websocket.accept()
    st = swap_manager.status()
    host = st.get("host_ip") or "127.0.0.1"
    port = st.get("port") or 8080
    sent = 0
    dead = asyncio.Event()

    async def stream_one(kind, url):
        nonlocal sent
        # llama-swap replays buffered history on connect — consume the
        # first burst silently (like the visualizer's priming) so opening
        # the tab doesn't dump hours of stale lines; only forward lines
        # that arrive after the stream settles.
        priming = True
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    async def unprime_later():
                        nonlocal priming
                        await asyncio.sleep(1.5)
                        priming = False
                    primer = asyncio.create_task(unprime_later())
                    try:
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if priming:
                                continue
                            try:
                                await websocket.send_text(json.dumps(
                                    {"src": kind, "line": line}))
                                sent += 1
                            except Exception:
                                dead.set()
                                return
                    finally:
                        primer.cancel()
        except Exception:
            return

    async def keepalive():
        # periodic ping keeps the ws from idling out on silent proxies and
        # gives the browser a liveness signal for the status dot
        while not dead.is_set():
            try:
                await websocket.send_text(json.dumps({"ping": True}))
            except Exception:
                return
            await asyncio.sleep(10)

    try:
        await asyncio.gather(
            stream_one("proxy", f"http://{host}:{port}/logs/stream/proxy"),
            stream_one("upstream", f"http://{host}:{port}/logs/stream/upstream"),
            keepalive())
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass
