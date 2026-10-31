# Flow visualizer & Live Logs

Both tabs consume the same llama-swap feeds; the visualizer parses
structured events, the Logs tab shows raw lines.

## Data sources

llama-swap exposes three HTTP log streams:

| Stream | Content |
|---|---|
| `/logs/stream/proxy` | Per-request lines: `Request <client> "METHOD path HTTP/x" status bytes "ua" duration` |
| `/logs/stream/upstream` | Everything the upstream llama-server prints (incl. `http: streamed chunk` token lines and GGML_SCHED_DEBUG tensor lines) |
| peer `rpc-server-<port>.log` files | The peer side of the same sched debug output (needed for single-RPC models where the host has nothing to schedule) |

**History replay:** llama-swap replays its buffered log history on every
connect. Every consumer must **prime silently** (~1.5 s) before
forwarding/displaying, or opening the page flares up hours-old activity.
Implemented server-side in each WebSocket (`priming` flag) and again
client-side (ignore first 1.5 s after open). Belt and braces on purpose.

## WebSockets (src/app.py)

| Endpoint | Parses | Emits |
|---|---|---|
| `/ws/flow` | proxy request lines | `{client, method, path, status, bytes, ua, duration_s}` |
| `/ws/flow/upstream` | `http: streamed chunk ... "model": "..."` | `{model}` (GGUF path or id) |
| `/ws/flow/sched` | GGML_SCHED_DEBUG node lines + every peer's rpc log tail | `{type: sched, tensor, backend, cause, srcs}` |
| `/ws/logs/stream` | raw lines from proxy+upstream | `{src: "proxy"\|"upstream", line}` + `{ping: true}` keepalives |

Regexes live at the top of the flow section in `app.py`
(`_VIS_LOG_RE`, `_UPSTREAM_CHUNK_RE`, `_SCHED_NODE_RE`, `_SCHED_SRC_RE`).
GGML_SCHED_DEBUG=2 node line shape:

```
node #N (OP): name (size) [BACKEND cause] use=..,c=..: src1 (..) [B c] src2 ...
```

The `%5.5s`/`%8.8s` upstream formatting is why backend/cause may carry
trailing spaces — don't "clean up" the regex tolerances.

## Visualizer attribution policy

Sched lines carry NO model id. `handleSchedEvent` (app.js) attributes
each event to a model by policy:

1. A model still streaming (upstream chunk < SERVING_WINDOW old) owns
   the sched.
2. Nothing streaming + single loaded model → it owns (load-time events
   must light edges too).
3. Multiple loaded, none streaming → most-recently-loaded.

Backend token handling:

- `RPC*` → exact lookup in `flowRpcMaps` (from `/api/rpc-map`) →
  `{machine, resource}`; resource `RAM` marks the RAM cell, otherwise
  the GPU index. Unknown token → nothing lights.
- `CUDA*/ROCm*/SYCL*/Metal` → host machine (first server), GPU index
  capped to detected count.
- `cpu` → every machine whose rpc_map has a RAM entry (else host).

Activity entries decay after 1.5 s (render) / 60 s (memory prune); a
500 ms poll keeps loaded-model state and topology fresh.

HOST-ONLY models (single server, master role) barely emit upstream
chunks, so a 400 ms refresh keeps their GPU cells lit for 2.5 s after
each token.

## Variant / base-model mapping

`buildFlowModelMaps(models)` builds filename → servers maps. Variants
share the base model's GGUF filename, so variant metadata must NOT
overwrite base-filename keys (`vfIsBase` guard) — it once lit aibox3
while only aibox1 was serving. `flowFileToId` may map the filename to a
variant key (`id--label`) but the servers map keeps the base entry.

## The Logs tab (`initLogsTab`)

- Buffer: 2000 lines in memory; 600 rows in the DOM (append-only, pruned
  from top). Filter matches highlight rows (`log-hit`).
- Source checkboxes (proxy/upstream) + search: literal by default,
  `/regex/` syntax for regex mode.
- Pause/clear/copy/auto-scroll controls reuse the visualizer button
  style (`flow-ctrl-btn`). Manual scroll pauses auto-scroll; the button
  re-enables.
- Backend keepalive pings every 10 s double as the liveness signal for
  the status pill.

## Debug toggles

- `localStorage.setItem("flow-debug-all-active", "1")` — preview mode:
  every edge/GPU lit as if streaming (hidden checkbox in the visualizer
  legend).
- `window.__flowDebugGet()` — dump of visualizer state (loaded models,
  maps, activity) for console debugging.