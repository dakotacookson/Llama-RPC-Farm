# Architecture

## What the app is

A single FastAPI process (`src/app.py`) that:

1. Serves the web UI (`ui/templates/index.html` + `ui/static/*`) on :8199.
2. Exposes a JSON API the UI drives.
3. Manages remote machines **only** through the peer agent — starting/
   stopping `ggml-rpc-server` on peers, launching `llama-server`, pushing
   configs and files. Agents are installed from their own packages.
4. Owns the llama-swap daemon lifecycle and its generated config.
5. Subscribes to llama-swap's HTTP log streams and re-publishes them as
   WebSockets for the visualizer and live-logs tabs.
6. Serves a native Chat tab (vanilla JS + `GET /api/chat/models` / streaming `POST /api/chat`, forwarded server-side to llama-swap) — no separate chat service.

## How it works, in one breath

The machine running the UI is the master; everything else is a peer. Model
configs live as YAML, and each deployment chooses its topology: single-box
local, multi-box RPC, peer-local, custom runtime or MLX. The app generates
the exact launch plan/scripts (RPC peers first, llama-server last, or the
peer agent start for local deployments), and llama-swap fronts it all so
clients always hit one endpoint. The app also runs the `:8090` view/access
gateway in-process — no separate service to install. Secrets (agent tokens,
HF token) live in an encrypted vault on your box, never in configs, never in
scripts.

## Module map (src/)

| Module | Responsibility |
|--------|----------------|
| `app.py` | All HTTP/WS routes. Thin: validates, calls modules, returns. Also: flow-log parsing (`_parse_sched_line` etc.), UI state endpoints (accordion/flow layout), machine detail. |
| `config_loader.py` | YAML CRUD store classes (`machines`, `groups`, `models`, `swap_groups`), model helpers (`list_models`, variant resolution), hardware fingerprinting, model↔hardware sync. Owns the engine enum vocabulary. |
| `engine_registry.py` | Engine adapter seam (`llama_cpp`): `load_settings` applies the engine schema (enum — bad values are startup errors, never silent fallback); `get_engine(settings)` is the one seam app.py dispatches on. Adapters implement the `EngineAdapter` Protocol (build_command / probe / swap_entries / validate_model_config / validate_hosts / ui_hints); `LlamaCppAdapter` is pure delegation to rpc_manager/swap_manager. |
| `rpc_manager.py` | The biggest module (~2.6k lines): RPC server start/stop/status, port rolling, per-port state files, `build_llama_command`, `compute_rpc_map`, bash/PowerShell script generation, RDMA gating, llama-swap host bin resolution. |
| `swap_manager.py` | llama-swap daemon: settings, `generate()` (builds full config.yaml plan from model configs), push, `launch()` orchestration, `stop()`, `reload_daemon()`, systemd/win/mac service paths, backup + validation. |
| `tuner.py` | Math estimate (VRAM fit, KV bytes, split weights) + staged live benchmark search. Writes tuning results back into model configs. |
| `llamacpp_installer.py` | Downloads the right llama.cpp prebuilt per OS/vendor/arch/glibc and updates `llama_bin`. |
| `swap_installer.py` | Downloads the official llama-swap prebuilt per OS/arch, skip-if-present, updates swap settings' binary. Single-host tool — no fleet install. |
| `stats_collector.py` | GPU/CPU/RAM stats: local or peer-agent; three OS branches (linux `/proc`+nvidia-smi, Windows PowerShell, macOS sysctl/top). |
| `discovery.py` | LAN identity helpers: UDP peer-agent discovery + ARP/MAC healing. No network sweep. |
| `local_node.py` | Which machine am I? `is_local()` decides master vs peer everywhere. |
| `gpu_vendor.py` | Vendor normalization + device token prefixes (`CUDA0`, `ROCm0`, `SYCL0`, `Metal`). |
| `model_parser.py` | Directory scan for GGUF; collapses `-000XX-of-000YY` shards into one model. |
| `gguf_inspect.py` | Minimal GGUF metadata reader (arch, tensor count, MTP layers, byte sizes) without deps. |
| `path_resolver.py` | `store://` alias ↔ real path per platform (`config/stores.yaml`). |
| `hf_browse.py` / `hf_downloader.py` | Hugging Face GGUF search/collections + download jobs (single + shard fan-out), proxied through the app. |
| `secrets_vault.py` | Fernet vault in git-ignored `.env` (`FARM_SECRET_KEY`, `FARM_SALT`, `HF_TOKEN_ENC`, `AGENT_TOKEN_ENC`, `MACHINE_<ID>_AGENT_TOKEN_ENC`, mode 0600); agent tokens/HF token decrypted only at point of use, never written back to YAML or sent to the browser (`agent_token_set` / `hf_token_set` flags only). SHA-256+salt `hash_password`/`verify_password` for one-way fingerprints (login passwords only; no remote-login password transport). |
| `steps.py` | Shared step-runner for launch reports (used by both single-model launch and swap launch). |
| `reaper.py` | Conservative orphan `ggml-rpc-server` reaper: protects loaded/loading models, active `--rpc` endpoints, and young processes; kills only per-port orphans. |

## Two clusters of ownership (don't blur them)

**RPC cluster** (`rpc_manager`): per-peer `ggml-rpc-server` processes,
`--rpc` endpoint list, RPC<N> numbering, generated `start-*.sh` scripts.

**Swap daemon** (`swap_manager`): llama-swap itself — generates
`config.yaml` from every model config, pushes scripts + sidecar to the
host, launches via systemd/detached/Win32_Process.

Both produce step reports via `steps.run_steps`; files are pushed through
the peer agent (`/files/write`) or written locally.

## Engine registry (llama.cpp)

`engine_registry.py` is the only seam app.py branches on. Settings
`engine: llama_cpp` (default, backward compatible when the key is
missing); a typo is a startup error listing valid values. The adapter
implements the Protocol so callers get stable shapes:

| Operation | llama.cpp |
|---|---|
| command | RPC wrapper script / `llama-server --rpc` |
| probe | TCP port-open check |
| serving | multi-machine RPC cluster |
| stop | process group |
| hosts | any |
| kv cache | q4_0/q8_0 |

## Request flow (typical "Launch All")

```
UI (Launch tab) → POST /api/swap/launch
  → swap_manager.launch()
      → generate()          # builds plan: yaml + per-model scripts
      → backup + write yaml to host
      → push_scripts()      # SFTP scripts + swap-secrets.env sidecar
      → validate binary
      → stop leftovers
      → _start_linux_service / _start_windows_* (daemon start)
      → health check (port probe)
  ← steps report → UI renders
```

Single-model live launch: `POST /api/launch` → `rpc_manager` starts each
peer's rpc-server (`start_rpc`), then `_start_llama_server` on the master,
then waits for the HTTP port.

## Serving flow (runtime)

```
client → llama-swap :8080 (proxy)
  → matches model id → starts/keeps warm its wrapper script
    → script starts RPC peers through each agent (/rpc/ensure) + llama-server --rpc host:port,host:port
  → llama-server logs GGML_SCHED_DEBUG per-tensor lines + upstream chunks
    → app tails /logs/stream/{proxy,upstream} → WebSockets → UI
```

## Frontend structure (ui/static/app.js)

Single global JS file, organized by tab: dashboard, machines, groups,
visualizer (`initFlowVisualizer`), logs (`initLogsTab`), models wizard
(`mowiz*`), downloads, tests (chat/vision/audio/embeddings), launch,
serve groups, settings. Tab activation dispatches via `data-tab` buttons
in `nav`. State kept in module-level maps (`flowModelMachines`,
`flowRpcMaps`, ...). Rendering is innerHTML-based with RAF coalescing.

**Bar animations:** CSS transitions on `.bar`/`.metric-fill` (width +
color, 0.8s) only work on persistent DOM nodes. Any polling renderer that
rebuilds cards must patch fills in place (`_upsertMachineCard` for
dashboard cards, per-job patching in `loadDownloads`) — full innerHTML
rebuilds make bars jump-cut. Overview totals use `style.width` on
persistent elements, so they transition naturally.

**Dashboard status colors:** machine cards carry `online` (green border
+ top line + glowing dot) or `offline` (red border + pulsing top bar +
red dot) classes — a down machine must read as DOWN, not just faded.
Motion (pulsing) is reserved for offline; healthy cards are static green.

## State copies (important)

- `config/*.yaml` — source of truth, edited by the app in-process.
- `~/.config/llama-swap/` on the host — generated output, overwritten on
  every push. Never hand-edit.
- In-memory: the running app holds parsed YAML; external edits need a
  restart to be seen.