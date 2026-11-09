# Configuration reference

Everything under `config/`. All files are YAML (or JSON for UI state) and
are CRUD'd by the running app at :8199. The app holds parsed state in
memory — hand edits while it's serving require a restart to be visible.

## machines.yaml — the fleet

Per machine:

```yaml
- id: aibox1            # legacy identifier (see warning)
  name: aibox1          # THE key used by ~all code
  ip: 127.0.0.1         # localhost = the machine running the app
  role: master          # master | peer; master = this app's box
  os: linux             # linux | windows | macos
  rpc_port: 50052       # legacy base; per-model ports are rolled at launch
  gpu: CUDA0,CUDA1,CUDA2   # device tokens (gpu_vendor.py prefixes)
  llama_bin: /home/you/llama.cpp/build/bin
  expose:
    mode: both          # vram | ram | both — what this box shares to RPC
    gpus: []
  detected:             # written by Hardware Detect; not hand-edited
    gpu_names: [...]
    vram_total_gb: 36.0
    cores: 10
    ram_total_gb: 27
    arch: x86_64        # written by llamacpp_installer probes
  ram_total_gb: 27
  gpu_vendor: nvidia    # nvidia | amd | intel | apple | cpu
  rdma: false           # RDMA/RoCE transport (opt-in; TCP default)
  rdma_ip: / rdma_dev: / rdma_gid: / rdma_cm: false
  transport: agent      # accepted for compatibility; runtime is agent-only
  agent_port: 8198      # peer-agent HTTP port
  # agent_token is write-only in the Add/Edit form; it is stored in the
  # Fernet vault, never in machines.yaml. There is no SSH/remote-login
  # password: all peer work goes through the agent.
```

**Warning:** `id` and `name` can diverge (historical rename: id `aibox2`,
name `aibox3`). Code keys on `name` almost everywhere — do not "fix" one
without the other.

## stores.yaml — portable model paths

```yaml
stores:
  llm-models:
    windows: \\10.0.0.75\models\llm-models
    linux: /mnt/cifs/llm-models
```

Model configs reference files as `store://llm-models/<file>`; the resolver
(`path_resolver.py`) maps to the right root for the OS of the machine the
path will be used on. Add a `macos:` key when a Mac joins. Never store
absolute model paths in model configs.

## settings.yaml — app settings

`model_dir` (scan root, store:// alias), `llama_bin` (global default bin
dir; machine `llama_bin` overrides), `theme_family`/`theme_mode`.

### engine selection (schema in `src/engine_registry.py` / `config_loader.py`)

```yaml
engine: llama_cpp        # llama_cpp
```

- A missing `engine` key means `llama_cpp` (backward compatible); a
  typo is a **startup error** listing the valid values — never a
  silent fallback.
- Writes from POST /api/settings merge over the current file through
  the comment-preserving persister (B9c) — comments and unknown keys
  survive.

## models/*.yaml — one file per model

```yaml
- model_id: qwen3-8-27b-uncensored-q4-k-m-128k
  name: ...
  base_path: store://llm-models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf
  runtime: llama            # llama | mlx (Apple, HF repo id)
  servers: [aibox1]         # machines this model deploys across
  device_order: aibox1-CUDA0,aibox1-CUDA1,aibox1-CUDA2
  device_order_policy: as-selected   # or vram-first
  tensor_split_mode: fit    # fit | split | auto (fit supersedes legacy manual)
  fit_margin: 1024
  fit_ram_allow: 1024
  flags:                    # key -> value; emitted as --key value
    ctx-size: 128000
    flash-attn: on
    jinja: true
    cache-type-k: q8_0
  expose:                   # per-server VRAM/RAM sharing for RPC
    aibox1: {mode: vram, gpus: [CUDA0, CUDA1, CUDA2]}
  pinned: false
  ttl: 0                    # llama-swap auto-unload after idle
  variants: [...]           # additional configs, see models-and-variants.md
```

Flags are stored verbatim and rendered as CLI flags; string values are
`shlex.quote`d at command-build time (injection tests enforce this).

## groups.yaml — machine groups (deployment affinity)

```yaml
- id: full-cluster
  name: full-cluster
  description: ...
  machines: [aibox1, aibox3, aibox2, winbox]
```

Used by the model editor's hardware picker and presets. Note machine
clusters (`groups.yaml`) vs **serve groups** (`swap_groups.yaml`) are
different concepts: machines vs models.

## swap_groups.yaml — llama-swap groups (models)

```yaml
- name: mid Range
  members: [model-a, model-b]   # model_ids
  swap: true        # llama-swap swaps these (one loaded at a time)
  exclusive: true   # loads of these evict everything else
  persistent: false # llama-swap persistent group
```

Generated into llama-swap's `groups:` config section on push.

## JSON state files (app-owned, don't hand-edit)

| File | Contents |
|---|---|
| `accordion_state.json` | Which collapsible panels are open (persisted UI state) |
| `flow_layout.json` | Visualizer node drag positions (per model node) |
| `model_icons.json` | Model → icon URL cache |
| `net_speed_cache.json` | Bandwidth probe results (7-day TTL) |
| `swap.yaml` | llama-swap host pointer (`host_name`); never contains secrets |

## The second copy (on the host)

`swap_manager` pushes everything to `~/.config/llama-swap/` on the host
machine (`%USERPROFILE%\.config\llama-swap\` on Windows). That copy plus
`start-*.sh` scripts and the `swap-secrets.env` sidecar live there. It is
regenerated on every push — treat it as build output.