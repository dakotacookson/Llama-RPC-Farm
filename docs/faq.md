# FAQ

### Do I need multiple machines?

No. The farm runs solo on one box — one GPU, several GPUs, or CPU/RAM only.
Add peers later; model configs and the endpoint stay the same.

### Does RPC make inference slower?

Yes. A model split across machines is slower per token than the same model
on one box. That is the trade-off: use RPC for models that do not fit
anywhere else, and keep models that fit on one machine local. Real measured
runs: [benchmarks.md](benchmarks.md).

### Do the GPUs have to match?

No. llama.cpp's scheduler distributes layers/tensors across the selected
devices, and the farm supplies the `--rpc`, `--device`, and
`--tensor-split`/`--fit` plan. Mixed vendors and mixed VRAM sizes work, but
the slowest card/link can become the straggler.

### Does it work on Windows and macOS?

Yes. The app runs on Linux, Windows, and macOS; peers can be any mix, all
reached through the typed peer agent (no SSH, no remote shell).

- **Windows peers:** use them as **RPC peers** for big models — the host
  reads the GGUF, Windows runs `ggml-rpc-server`. For an existing native
  server, use `runtime: custom`. Avoid remote-local `llama-server` on
  Windows with SMB model paths.
- **Apple Silicon:** use the **Metal** backend to join a llama.cpp RPC
  cluster. MLX is a separate single-Mac runtime and cannot join RPC.

More: [topologies.md](topologies.md), [cross-platform.md](cross-platform.md).

### Is it secure?

For a **trusted LAN or VPN**, yes. llama.cpp's RPC server is
unauthenticated by design, so never port-forward the RPC ports, UI, gateway,
or agent port to the internet. The farm adds token-authenticated peer
agents and per-key model access at the `:8090` gateway. Full threat model:
[../SECURITY.md](../SECURITY.md).

### Does it modify my existing llama.cpp or llama-swap configs?

No. The farm keeps its own state in `config/*.yaml` and pushes its generated
output to its own swap directory (`~/.config/llama-swap/`). It drives stock
llama.cpp binaries and talks to llama-swap over HTTP.

### Why not Ollama?

Ollama is excellent for pulling and running models on one machine. Llama RPC
Farm is an orchestration layer: it pools machines over llama.cpp RPC, adds
llama-swap's on-demand multi-model serving, a dashboard/visualizer, and
per-user keys. It drives stock llama.cpp builds, so every upstream backend
and flag is available as-is. You can run both on the same network.

### Why not exo / LocalAI / vLLM?

Different focus. exo builds its own runtime/device mesh; LocalAI/vLLM are
full serving stacks with their own model runners. The farm deliberately does
**not** invent a runtime — it orchestrates stock llama.cpp + llama-swap and
shows you exactly where every tensor runs. Comparison:
[../README.md#how-is-this-different](../README.md#how-is-this-different).

### What model formats can it serve?

llama.cpp entries must be **GGUF**. The Hugging Face browser also indexes
HF/safetensors repos, but those are catalog entries, not launchable by
llama.cpp. MLX entries use an HF repo id and are served by `mlx_lm.server`.

### Does every machine need the model file?

For **RPC** deployments the host reads the GGUF; peers only run
`ggml-rpc-server`. For **peer-local** deployments the peer runs
`llama-server`, so the model path must resolve on that peer's OS via
`store://` ([configuration.md](configuration.md)).

### Can it use system RAM when VRAM runs out?

Yes. Select the machine's RAM chip in the device order; it becomes a CPU
device in the llama.cpp plan. Note the `fit` + RAM-device caveat in
[topologies.md](topologies.md#combinations-to-avoid).

### How do multiple models share one endpoint?

llama-swap starts a model on the first request and unloads it after its TTL
(or never, if pinned). Serve groups add take-turns, exclusive, and
always-loaded behavior. See
[models-and-variants.md](models-and-variants.md#serve-groups-llama-swap).

### Why is a model slow / failing to load?

Run the tuner, check its VRAM estimate, and look at the launch/load logs.
The tuner is math-first and then benchmarks live:
[tuner.md](tuner.md). Common causes: a bad tensor split, too much context
for available VRAM, a RAM device in pure-fit mode, or a peer that is
offline/flaky.

### Can I expose it to the public internet?

Not supported. Put it behind a VPN or an authenticated reverse proxy. See
[../SECURITY.md](../SECURITY.md).

### Where do I ask questions?

[Discussions](https://github.com/dakotacookson/Llama-RPC-Farm/discussions)
for questions and show-and-tell; [Issues](https://github.com/dakotacookson/Llama-RPC-Farm/issues)
for bugs and hardware reports.
