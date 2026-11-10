# Deployment topologies

Each model deployment picks the topology that suits its hardware —
automatically (`launch_mode: auto`) or explicitly. All of them sit behind
the same llama-swap endpoint, so clients keep one OpenAI-compatible URL.

| Topology | Where inference runs | Use it when |
|---|---|---|
| **Single box** | one machine, no RPC | the model fits on one box. CUDA, ROCm, Vulkan, Metal and CPU-only are all first-class |
| **Multi-box RPC cluster** | host machine, with `--rpc` to peers | the model is too big for one box; pool GPUs and host RAM across machines |
| **Peer-local (remote-local)** | a peer, started through its agent | compute belongs on another box, but clients still use one farm endpoint (llama-swap proxies to it) |
| **Custom runtime** | an external OpenAI-compatible server the agent starts/stops | you already have a non-llama.cpp stack (for example a Windows `.bat` runtime) |
| **MLX** | Apple Silicon via `mlx_lm.server` | Mac-native serving without RPC or device flags |

How the automatic choice works:

- explicit `launch_mode: local|rpc` on the model (or variant) wins;
- `auto` picks **local** when every selected device chip belongs to one
  non-Windows peer box;
- anything involving the host, multiple boxes, or a Windows box stays an
  **RPC** chain (Windows remotes cannot read SMB model files in a
  remote-launch session);
- MLX is always local to one Mac.

## MLX, Metal, and custom runtimes

MLX is a **separate runtime, not a llama.cpp backend**: it cannot join a
model's `--rpc` device graph or take part in tensor splitting. It can sit
alongside llama.cpp/RPC models behind llama-swap as its own model entry, but
a single model is either MLX (one Mac) or llama.cpp (which may span boxes).
The same applies to **custom runtimes**: they are proxied standalone OpenAI
servers, not RPC peers.

To put an Apple Silicon GPU into a llama.cpp cluster, use the **Metal**
backend (`Metal` device token) — that is llama.cpp's own Apple GPU backend,
still llama.cpp — not MLX. MLX and llama.cpp/Metal are two different ways
to serve on a Mac; only the latter can join the RPC graph.

## Heterogeneous multi-node

Master and peers can be any mix of Linux, Windows and macOS. The master
orchestrates; each peer runs its native `ggml-rpc-server` (or `llama-server`
for peer-local deployments) through the peer agent. There is no SSH and no
shared shell — every peer operation is a typed agent call, so Windows
PowerShell, macOS process quirks and Linux all behave the same.

The tensor placement itself is llama.cpp's scheduler: it distributes model
layers/tensors across the selected devices, and the farm supplies the
`--rpc`, `--device` and `--tensor-split`/`--fit` plan it needs. The Flow
Visualizer then reads the real per-tensor GGML sched log lines and shows
which GPU on which machine is executing each tensor, live while the model
runs. One OpenAI-compatible endpoint can therefore span several boxes and
several GPUs without hiding what the scheduler is actually doing.

Internals: [rpc-model.md](rpc-model.md). Cross-platform process details:
[cross-platform.md](cross-platform.md). Model/variant config:
[models-and-variants.md](models-and-variants.md).

## Combinations that work

These are the shapes the farm is designed around, drawn from a working mixed
fleet (Linux master + Linux peers + Windows box + CPU laptop):

1. **One big model across two boxes (RPC)** — host with 3 GPUs, peer with 3
   GPUs (or 2 GPUs + shared RAM). `servers: [host, peer]`,
   `device_order` lists every selected chip, `tensor_split_mode: fit` (or
   explicit `split` weights). llama-server runs on the host, peers run
   `ggml-rpc-server`, clients see one model.
2. **Same weights, two placements (base + variant)** — base on the host
   (`servers: [aibox1]`) for everyday use; a variant on a peer
   (`servers: [aibox3]`, `launch_mode: local`) so llama-server runs there and
   llama-swap proxies to it. Different access keys can expose base vs
   variant to different people.
3. **Small models per box, one endpoint** — a 4B on one peer
   (remote-local), ASR + embeddings on a CPU/RAM box, and an external native
   server on Windows, all listed by the same llama-swap instance.
4. **Take-turns group on one box** — several 27–35B models that do not fit
   together; `swap: true` so loading one evicts the group's others.
5. **Exclusive "clear the stage" group** — a model that needs all of a
   box's VRAM; `swap: true, exclusive: true` so loading it evicts everything.
6. **Always-on co-serving group** — ASR + embeddings + a small decision model
   on a dedicated box; `persistent: true` keeps them all loaded.
7. **CPU/RAM-only node** — a laptop with no usable GPU: `device_order:
   box-RAM`, `n-gpu-layers: 0`. Good for embeddings, ASR and small decision
   models.
8. **Windows peer** — for big models, use it as an **RPC peer** (Windows only
   runs `ggml-rpc-server`; the host reads the GGUF). For an existing native
   server, use `runtime: custom` with an agent-managed `.bat`. Avoid
   remote-local `llama-server` on Windows for SMB model paths.

## Combinations to avoid

- **MLX + RPC peers in one model** — MLX is single-Mac; peers are ignored.
- **A persistent group whose members do not all fit in VRAM together** —
  every member stays loaded; the last one will fail.
- **The same model in two `swap: true` groups** — only the first group wins.
- **`tensor_split_mode: split` with a weight count that differs from
  `--device`** — layers get mis-assigned.
- **`fit` + a RAM device when you expect pure fit** — the farm downgrades to
  a computed split (llama.cpp's RAM-last fitter bug).
- **Custom runtime with more than one server** — it manages exactly one
  external OpenAI server.

The app shows these as non-blocking warnings when you save a model, machine
group, or serve group, so you see the problem at configuration time instead
of at first load.
