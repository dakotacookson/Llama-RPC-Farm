# RPC model: numbering, devices, ports, scripts

This is the most subtle part of the system. One rule above all: **RPC
numbering has ONE source of truth** — `rpc_manager.compute_rpc_map()`
(mirrored by `_rpc_legend` for human-readable logs). Everything else
(visualizer attribution, generated scripts, `--rpc` endpoint order)
must derive from it, never from a second scheme.

## How RPC numbering works

`--rpc` takes a comma list of `ip:port` endpoints. Numbering follows
exactly that order:

```
for peer in peers (servers order, master first):
    for group in peer groups (gpu group, then cpu group iff RAM):
        for device in group devices (GPUs by index, then CPU):
            assign next RPC<N>
```

- `RPC0=aibox3-CUDA0, RPC1=aibox3-CUDA1, RPC2=aibox3-RAM` etc.
- `compute_rpc_map()` returns `{model_id: {file, maps: {variant: {
  RPC0: {machine, resource}, ...}}}` served at `GET /api/rpc-map`.
- The generated script logs the same mapping as a legend line so
  llama-server errors mentioning `RPC3` can be traced to a box.
- The visualizer resolves `RPC<N>` tokens from sched logs via this map.
  **Unknown ids light nothing** — no modulo guessing.

## Device tokens

Owned by `src/gpu_vendor.py`:

| Vendor | Token form | Notes |
|---|---|---|
| NVIDIA | `CUDA0`, `CUDA1`, ... | `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set for stable ordering |
| AMD | `ROCm0`, ... | llama.cpp renamed HIP0→ROCm0 at b9748; `HIP*` still accepted |
| Intel | `SYCL0`, ... | |
| Apple | `Metal` (no index) | the Metal backend registers ONE device named exactly that |
| CPU/RAM | `CPU` in scripts, displayed as `RAM` | llama-server's own CPU device token is `CPU` |

Chips in the UI are `<machine>-<token>` (e.g. `aibox1-CUDA0`,
`aibox3-RAM`); `_parse_chip` splits them. `is_gpu_token` accepts
`CUDA/ROCm/HIP/Metal/SYCL` prefixes.

## Ports

- Legacy base port 50052 still exists but per-model ports are **rolled**
  at launch: `_model_rpc_port(model_id, peer, group)` hashes model+peer
  (+group) onto a port range, `_roll_port_if_busy` bumps past anything
  already listening (including Windows Hyper-V/WSL excluded ranges).
- Per-port state files (`rpc.state.<port>`) record the device set each
  rpc-server was started with, so co-served models on different ports
  never clobber each other's ensure-checks.
- Logs follow the same naming: `rpc-server-<port>.log` in `$HOME` /
  `%USERPROFILE%`.

## Peer groups (`_peer_server_groups`)

A peer can contribute up to two rpc-servers:

1. **GPU server** — its selected GPUs (expose config / device_order).
2. **CPU server** — iff its RAM chip is selected (`-d CPU`, or default
   CPU-only when no GPUs selected).

This exists because llama.cpp serializes multi-backend work per
rpc-server connection; the CPU backend gets its own port so it never
blocks GPU tensors behind a single client connection.

## Generated scripts (`_render_rpc`, `generate_script`)

Written to `generated/`, chmod 0700, LF endings. Structure:

```
stop_node_rpc_linux / stop_node_rpc_win   # per-port targeted kills only
start_node_rpc_linux   # sshpass + setsid||nohup ggml-rpc-server -p $port --device ...
start_node_rpc_win     # scp a per-peer .ps1, run via powershell
main: start RPC peers, wait /dev/tcp, then llama-server with --rpc/--device
trap → teardown on exit; peers may belong to co-served models (kills are
per-port, never blanket pkill)
```

Rendered from an f-string: literal `{`/`}` in the bash output must be
written `{{`/`}}` in the source — the same string also embeds PowerShell
`{ }` blocks, which is the #1 source of escaping bugs here.

Env baked into scripts: `GGML_SCHED_DEBUG=3` + `GGML_RPC_DEBUG=3`
(visualizer + log tails ground truth), `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`GGML_CUDA_P2P=1`, and RDMA gating (`GGML_RPC_NO_RDMA=1` default;
`GGML_RDMA_DEV/GGML_RDMA_GID/USE_CM` when enabled).

`$SSH_PASS` must be in the environment (see secrets.md) — the script
fails fast with a clear message otherwise.

## llama-server command (`build_llama_command`)

- Model path resolved per target OS via `store://`.
- Flags emitted in stored order; `-v --log-timestamps --log-prefix`
  injected (visualizer's upstream fallback needs verbose chunk logs).
- String flag values are `shlex.quote`d.
- `--device` from device_order (chip → real token mapping; `--device` on
  the master embeds `RPC<N>` tokens for peers).
- fit mode: `--fit on --fit-target ...`; a RAM device in the order
  downgrades fit to explicit computed split weights (llama.cpp's fitter
  has an unsigned-underflow loop on RAM-last-zero-layers; see the long
  comment in `build_llama_command`).
- `--tensor-split`/`--n-gpu-layers` are skipped in fit mode (the fitter
  owns them; user-set ngl makes it abort).

## Where things can silently disagree

| Symptom | Cause to check |
|---|---|
| Visualizer lights wrong machines | variant overwrote base filename→servers map (app.js `buildFlowModelMaps`) |
| RPC tokens unresolved in visualizer | `/api/rpc-map` missing for that model/variant |
| Peer shows as down while serving | rdma_cm peer has no TCP listener → port probe false; check state files |
| Two models collide on one peer | both rolled onto same port → `_roll_port_if_busy` should prevent; check `rpc.state.*` staleness |