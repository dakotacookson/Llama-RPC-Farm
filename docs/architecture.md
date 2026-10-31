# Architecture

## What the app is

A single FastAPI process (`src/app.py`) that:

1. Serves the web UI (`ui/templates/index.html` + `ui/static/*`) on :8199.
2. Exposes a JSON API the UI drives.
3. Manages remote machines over SSH (paramiko) — starting/stopping
   `ggml-rpc-server` on peers, launching `llama-server`, pushing configs.
4. Owns the llama-swap daemon lifecycle and its generated config.
5. Subscribes to llama-swap's HTTP log streams and re-publishes them as
   WebSockets for the visualizer and live-logs tabs.
6. Serves a native Chat tab (vanilla JS + `GET /api/chat/models` / streaming `POST /api/chat`, forwarded server-side to llama-swap) — no separate chat service.

## Module map (src/)

| Module | Responsibility |
|--------|----------------|
| `app.py` | All HTTP/WS routes. Thin: validates, calls modules, returns. Also: flow-log parsing (`_parse_sched_line` etc.), UI state endpoints (accordion/flow layout), machine detail. |
| `config_loader.py` | YAML CRUD store classes (`machines`, `groups`, `models`, `swap_groups`), model helpers (`list_models`, variant resolution), hardware fingerprinting, model↔hardware sync. Owns the engine schema vocabulary (enum + `vllm:` block defaults/validation) and the per-model `vllm:` round-trip (`sanitize_vllm_section` / `apply_vllm_section`). |
| `engine_registry.py` | Engine adapter seam (`llama_cpp` \| `vllm`): `load_settings` applies the engine schema (enum + normalized `vllm:` block — bad values are startup errors, never silent fallback); `get_engine(settings)` is the one seam app.py dispatches on. Adapters implement the `EngineAdapter` Protocol (build_command / probe / swap_entries / validate_model_config / validate_hosts / ui_hints); `LlamaCppAdapter` is pure delegation to rpc_manager/swap_manager, `VllmAdapter` registers lazily from vllm_manager. |
| `rpc_manager.py` | The biggest module (~2.6k lines): RPC server start/stop/status, port rolling, per-port state files, `build_llama_command`, `compute_rpc_map`, bash/PowerShell script generation, RDMA gating, llama-swap host bin resolution. |
| `swap_manager.py` | llama-swap daemon: settings, `generate()` (builds full config.yaml plan from model configs), push, `launch()` orchestration, `stop()`, `reload_daemon()`, systemd/win/mac service paths, backup + validation. Engine-aware at entry-emit time only: when settings `engine: vllm`, each entry emits the vllm cmd/cmdStop/`/health` via `vllm_manager.swap_entry_parts` (thin start script on linux; RPC peers ignored with a warning) — everything downstream (backup, write, push, launch, health) stays engine-agnostic. |
| `tuner.py` | Math estimate (VRAM fit, KV bytes, split weights) + staged live benchmark search. Writes tuning results back into model configs. |
| `llamacpp_installer.py` | Downloads the right llama.cpp prebuilt per OS/vendor/arch/glibc and updates `llama_bin`. |
| `vllm_installer.py` | Installs vLLM on ONE machine (local or SSH) as a `steps.run_steps` report, mirroring llamacpp_installer: gates (OS linux-only — Windows/macOS refused before any command; nvidia-smi must answer; AMD needs rocm-smi + Python 3.12), then delivery per settings `vllm:` — venv (uv when available, else `python3 -m venv`, pinned wheel `0.18.0`, ROCm hosts get the ROCm wheel index) or docker (`docker pull` of the configured image). Post-install verify: `vllm --version`; on AMD a backend probe asserts `torch.version.hip` (the silent-CUDA-fallback guard). Success points settings `vllm.venv_bin` at the installed binary. `plan()` is the no-side-effect preview. The experimental vllm-gguf-plugin is never auto-installed. |
| `vllm_manager.py` | All vLLM knowledge: `build_vllm_command` (venv `vllm serve` or `docker run`, multi-line shell like rpc_manager), `build_cmd_stop` (`docker stop farm-<id>` / `pkill -f "vllm serve.*<id>"` — llama-swap stops via cmdStop, not the process group), `swap_entry_parts` (cmd/cmdStop/checkEndpoint=`/health`/env — empty by design: device pinning travels inside cmd), HTTP probing (`probe_health` /health, `probe_identity` /v1/models served-name match, `probe_capacity` /metrics gauges), static validation (single host, fp8-only kv cache, GGUF needs `vllm.tokenizer`, TP-vs-GPU-count cross-check), the engine×OS gate (`validate_hosts`: no native Windows, no macOS, ROCm wheels need Python 3.12), and `ui_hints()` (labels/hidden panels/flag presets). Farm device chips map to `CUDA_VISIBLE_DEVICES` index lists. |
| `swap_installer.py` | Downloads the official llama-swap prebuilt per OS/arch, skip-if-present, updates swap settings' binary. Single-host tool — no fleet install. |
| `stats_collector.py` | GPU/CPU/RAM stats: local or SSH, three OS branches (linux `/proc`+nvidia-smi, Windows PowerShell, macOS sysctl/top). |
| `ssh_manager.py` | paramiko wrapper: `run`, `push_file`, `push_text` (os-aware rename), `check_port`, known_hosts fingerprint store. |
| `discovery.py` | LAN host discovery (arp + SSH identification), OS/MAC probing per platform. |
| `local_node.py` | Which machine am I? `is_local()` decides master vs peer everywhere. |
| `gpu_vendor.py` | Vendor normalization + device token prefixes (`CUDA0`, `ROCm0`, `SYCL0`, `Metal`). |
| `model_parser.py` | Directory scan for GGUF; collapses `-000XX-of-000YY` shards into one model. |
| `gguf_inspect.py` | Minimal GGUF metadata reader (arch, tensor count, MTP layers, byte sizes) without deps. |
| `path_resolver.py` | `store://` alias ↔ real path per platform (`config/stores.yaml`). |
| `hf_browse.py` / `hf_downloader.py` | Hugging Face GGUF search/collections + download jobs (single + shard fan-out), proxied through the app. |
| `secrets_vault.py` | Fernet vault in git-ignored `.env` (`FARM_SECRET_KEY`, `FARM_SALT`, `HF_TOKEN_ENC`, `MACHINE_<ID>_PASSWORD_ENC`, mode 0600); passwords/HF token decrypted only at point of use, never written back to YAML or sent to the browser (`password_set` / `hf_token_set` flags only). SHA-256+salt `hash_password`/`verify_password` for one-way fingerprints. |
| `steps.py` | Shared step-runner for launch reports (used by both single-model launch and swap launch). |

## Two clusters of ownership (don't blur them)

**RPC cluster** (`rpc_manager`): per-peer `ggml-rpc-server` processes,
`--rpc` endpoint list, RPC<N> numbering, generated `start-*.sh` scripts.

**Swap daemon** (`swap_manager`): llama-swap itself — generates
`config.yaml` from every model config, pushes scripts + sidecar to the
host, launches via systemd/detached/Win32_Process.

Both produce step reports via `steps.run_steps`; both push files via
`ssh_manager.push_text`.

## Engine registry (llama.cpp | vLLM)

`engine_registry.py` is the only seam app.py branches on. Settings
`engine: llama_cpp` (default, backward compatible when the key is
missing) or `vllm`; a typo is a startup error listing valid values.
Each adapter implements the same Protocol so callers get identical
shapes:

| Operation | llama.cpp | vLLM |
|---|---|---|
| command | RPC wrapper script / `llama-server --rpc` | `vllm serve <model> --host 0.0.0.0 --port "$PORT"` or `docker run` |
| probe | TCP port-open check | HTTP `GET /health` (200 = serving; vLLM binds the port long before it can answer) |
| serving | multi-machine RPC cluster | single host only — TP for multi-GPU within the host |
| stop | process group | `cmdStop` (`docker stop` / `pkill -f`) |
| hosts | any | linux only (no native Windows; macOS needs vLLM-Metal; ROCm wheels need Python 3.12) |
| kv cache | q4_0/q8_0 | fp8 only |

vLLM scope decisions baked into the code: `servers` must be exactly one
machine (a peer chip or extra server is refused/warned — in vllm mode
there is no on-demand multi-machine equivalent of `--rpc`); device
chips map to `CUDA_VISIBLE_DEVICES` index lists (RAM chips map to
nothing — CPU offload is `--cpu-offload-gb`); the per-model `env:` is
always empty because pinning travels inside cmd (venv prefix / docker
`-e`), so preview and runtime can never drift. The experimental
vllm-gguf-plugin is opt-in only; GGUF on vLLM requires
`vllm.tokenizer` pointing at the base HF directory.

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
then waits for the HTTP port. In vllm mode this endpoint answers **409**
— live RPC-cluster launch is a llama.cpp-only concept; serve models
through llama-swap (`POST /api/swap/launch`) instead, where `generate()`
emits the vllm cmd/cmdStop/`/health` entries.

## Serving flow (runtime)

```
client → llama-swap :8080 (proxy)
  → matches model id → starts/keeps warm its wrapper script
    → script starts RPC peers (sshpass) + llama-server --rpc host:port,host:port
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