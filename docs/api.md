# HTTP & WebSocket API

All routes live in `src/app.py`. Prefix: `http://localhost:8199`. JSON
unless noted. Errors are FastAPI `HTTPException` (4xx) or `{"error": ...}`.

## Machines

| Method + Path | Purpose |
|---|---|
| `GET /api/machines` | List machines (password replaced by `password_set` flag; response-only `vllm_capable` per entry — `false` for Windows hosts so the UI can grey them out in vllm mode) |
| `POST /api/machines` | Add machine (wizard payload) |
| `PUT /api/machines/{name}` | Update machine |
| `DELETE /api/machines/{name}` | Remove machine |
| `POST /api/machines/fingerprint` | Hardware fingerprint of one machine (pre-save probe) |
| `POST /api/machines/{name}/detect` | Detect hardware (GPUs/cores/RAM) and persist into `detected` |
| `POST /api/machines/detect-all` | Detect hardware on every machine |
| `POST /api/machines/probe` | Probe hardware on an unsaved machine (wizard) |
| `GET /api/machines/{name}/status` | Reachability + rpc-server running + IP sync. In vllm mode the payload gains `"engine": "vllm"` plus a `vllm` probe: `{"running": bool, "detail": str}` from `GET http://<ip>:<swap_port>/health` (detail strings: `vLLM /health 200`, `vLLM /health 503 (starting)`, `vLLM /health unreachable`, `probe failed: ...`). llama.cpp mode returns the payload exactly as before |
| `POST /api/machines/{name}/rpc/start` | Start rpc-server(s) on that machine |
| `POST /api/machines/{name}/rpc/stop` | Stop rpc-server(s); body may carry `port` for per-port stop |
| `POST /api/rpc/stop-all` | Stop every llama process on all machines |
| `POST /api/start-all` | Start RPC on all machines + llama-swap |
| `GET /api/machines/{name}/detail` | Machine detail card (kernel, cpu, disks, llama version, RDMA) |
| `GET /api/stats` | Cluster stats snapshot (blocking: waits for every machine's collection; 5s cache) |
| `GET /api/stats/stream` | **Per-machine stats as NDJSON**: one JSON line per machine as its collection completes, then a final `{totals, ts}` line. Dashboard consumes this so one slow/offline machine can't delay the others — each card paints the moment its own data lands |

## Discovery

| `GET /api/discover/subnets` | Candidate subnets for scanning |
| `POST /api/discover` | Scan a subnet for SSH hosts (fingerprint + OS probe) |
| `GET /api/hardware/changes` | Hardware fingerprint diff polling |

## Groups (machine clusters)

| `GET/POST /api/groups`, `PUT/DELETE /api/groups/{name}` | CRUD |
| `GET /api/groups/{name}/status` | Per-machine serving status |

## Models

| Method + Path | Purpose |
|---|---|
| `GET /api/models` | All model configs (with variants). In vllm mode each config may carry a per-model `vllm:` section — `max_model_len`, `max_num_seqs`, `tensor_parallel_size`, `pipeline_parallel_size`, `kv_cache_dtype` (`auto` never stored — it's the default), `cpu_offload_gb`, `quantization`, `tokenizer`, `extra_args` (list of strings; the sanctioned escape hatch for Ray/distributed-executor flags). Unknown keys are rejected on save (`400`), numbers coerced to int, absent = engine default |
| `POST /api/models` | Save model (wizard save); the `vllm:` section is validated + normalized in place (`config_loader.apply_vllm_section`), llama.cpp keys (`flags`, device chips) pass through untouched, an empty/absent `vllm:` is dropped so pre-vLLM configs never gain the key |
| `GET/PUT/DELETE /api/models/{model_id}` | Read/update/delete one config |
| `POST /api/models/{model_id}/test` | Live load test (single model) |
| `POST /api/models/cleanup` | Remove configs whose GGUF file no longer exists |
| `GET /api/models/scan-file` | Resolve a single path → model info |
| `GET /api/scan?path=&recursive=` | Directory GGUF scan (shards collapsed) |
| `GET /api/models/{model_id}/file` | GGUF file info + metadata |
| `POST /api/models/{model_id}/icon` | Set icon |
| `POST /api/models/{model_id}/pin` | Toggle pin |
| `GET /api/rpc-map` | **RPC<N> → {machine, resource} for every model** (visualizer ground truth) |
| `GET /api/safe-defaults` | Math-derived flags for a path (tuner.estimate) |

## Tuning

| `GET /api/tune/estimate`, `GET /api/models/{id}/tune/estimate` | Pure-math VRAM fit estimate |
| `GET /api/models/{id}/tune/defaults` | Safe defaults (estimate-based) |
| `POST /api/models/{id}/tune/start` | Start live benchmark search (background thread) |
| `GET /api/models/{id}/tune/status` | Progress + trials |
| `POST /api/models/{id}/tune/cancel` | Cancel |
| `GET /api/models/{id}/tune/drift` | Historical best-vs-now comparison |
| `POST /api/models/{id}/tune/seed` | Seed a config as tune baseline |

## Launch (single model)

| `POST /api/launch` | Stop old instances on its servers → RPC peers up → llama-server up; returns steps. **In vllm mode: `409`** — "vLLM mode does not use live RPC-cluster launch — serve models through llama-swap instead (Launch All / /api/swap/* endpoints)". Unreadable settings → `400` with the engine error |
| `POST /api/launch/stop` | Stop a model's processes |
| `GET /api/launch/active` | Per-machine serving booleans (cheap) |

## llama-swap (daemon)

| Method + Path | Purpose |
|---|---|
| `GET/POST /api/swap/settings` | Host, binary, port, config path |
| `GET /api/swap/preview` | Generate yaml + scripts without writing (dry run) |
| `POST /api/swap/push` | Push config + scripts to host |
| `POST /api/swap/launch` | Full orchestration (generate → push → start → health check) |
| `POST /api/swap/stop` | Stop daemon (peers stay up) |
| `GET /api/swap/status` | Running + port |
| `GET /api/swap/loaded` | Currently loaded model ids (from llama-swap) |
| `POST /api/swap/evict/{model_id}`, `/api/swap/evict-all` | Evict loaded models |
| `GET /api/swap/reload-status` | Async reload progress |

## Serve groups (model co-serving)

| `GET/POST /api/swap-groups`, `PUT/DELETE /api/swap-groups/{name}` | CRUD |
| Members are model ids; `swap`/`exclusive`/`persistent` map to llama-swap group behavior |

## Downloads / HF

| `GET /api/downloads` | Download job list |
| `POST /api/downloads` | Start download (`link` = `hf://repo/file`, `store` alias, `all_shards`) |
| `POST /api/downloads/{job_id}/cancel` | Cancel |
| `GET /api/downloads/stores` | `store://` aliases |
| `GET /api/hf/search` | GGUF search (quant filters, author snapshots) |
| `GET /api/hf/models/{author}/{name}/gguf` | Repo file list |
| `GET /api/hf/access/{author}/{name}` | Gated-repo access check (uses HF token) |
| `GET /api/hf/companies` · `pipeline-tags` · `collections` · `collection` | Browse facets |
| `GET /img-proxy` | Cached image proxy for HF assets |

## UI state

| `GET/POST /api/flow/layout` | Visualizer node positions |
| `GET/POST /api/accordion` | Collapsible panel open/closed state |
| `GET/POST /api/settings` | App settings (model_dir, llama_bin, theme) plus the engine schema: responses carry `engine`, a fully-defaulted `vllm:` block (`delivery: venv\|docker`, `venv_bin`, `docker_image`, `default_gpu_memory_utilization`), `hf_token_set`, and `ui_hints` (adapter vocabulary: labels, hidden panels, flag presets). POST accepts optional `engine` (enum-checked, `400` with valid values on a typo) and a partial `vllm:` dict merged over the stored block (unknown keys / bad delivery / gpu-memory-utilization outside (0, 1] → `400`); writes go through the comment-preserving persister so unrelated keys and comments survive. Raw `hf_token` is never returned — only `hf_token_set` |
| `GET/POST /api/themes/custom`, `DELETE /api/themes/custom/{tid}` | Custom themes |

## Installer

| `POST /api/install-llama-cpp` | Detect existing local install, configure path |
| `POST /api/install-llama-cpp/install?name=` | Install the right prebuilt on one machine |
| `POST /api/install-llama-cpp/all` | Check all machines, optionally install missing |
| `POST /api/install-llama-swap?name=` | Install llama-swap on a machine if missing (local = no SSH); updates swap settings' binary |
| `POST /api/restart` | Restart this web app (POSIX/Windows aware) |

vLLM install is not HTTP-driven: `vllm_installer.install(machine)` /
`plan(machine)` (same `steps.run_steps` report shape as the llama.cpp
installer) run from code — gates → delivery (venv or docker pull) →
verify → settings `vllm.venv_bin` update.

## Misc

| `GET /api/fs/browse` | File explorer (local or per-machine, OS-aware) |
| `GET /api/embeddings|audio|vision/models` + `POST /api/embeddings` etc. | Test-tab inference probes |
| `GET /api/chat/models` + streaming `POST /api/chat` | Native chat test (model list with loaded flags, SSE chat completions via swap) |

## WebSockets

| Path | Purpose |
|---|---|
| `/ws/flow` | Parsed proxy-request events (visualizer left column) |
| `/ws/flow/upstream` | Token-chunk events (model currently streaming) |
| `/ws/flow/sched` | Per-tensor backend events from GGML_SCHED_DEBUG logs + peer rpc-server logs |
| `/ws/logs/stream` | **Raw** proxy+upstream lines for the Logs tab (`{src, line}`), primed ~1.5s |
| `/ws/socket.io/*` | Chainlit chat (proxied) |

All log-feed WebSockets must tolerate llama-swap's history replay on
connect — the backend primes silently; clients ignore the first ~1.5s.

## Auth model

None on the API itself (LAN tool). Secrets never cross it — machine
passwords appear only as `password_set: true/false`; the HF token as
`hf_token_set`.