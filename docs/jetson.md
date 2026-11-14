# NVIDIA Jetson (Tegra, Linux arm64)

A Jetson is a normal Linux `aarch64` peer. The peer agent and the farm
control plane work; what differs is the **GPU**.

## GPU support by JetPack generation

| Board / stack | CUDA | GPU in the farm |
|---|---|---|
| Jetson Nano (Tegra X1), TX1/TX2 — JetPack 4.6 / L4T R32 | 10.2 | ❌ **CPU-only** |
| Jetson Xavier — JetPack 5 / L4T R35 | 11.4 | ✅ potentially (detected as `nvidia`) |
| Jetson Orin — JetPack 5/6 | 11.4–12.x | ✅ potentially (detected as `nvidia`) |

Current llama.cpp CUDA needs **CUDA ≥ 11** (`cuda_bf16.h`) and **C++17**
nvcc support. JetPack 4.6's CUDA 10.2 has neither, and there is no newer
JetPack for the Nano/TX (EOL). Upstream CUDA prebuilts target CUDA 13.x
and desktop GPUs, so they are unusable too. The peer agent detects this
(`gpu_unsupported`) and the farm forces **CPU-only** routing: no CUDA
install, no `--device CUDA0`, no GPU chips.

## First install

1. Run the agent one-liner (or the `arm64.deb`):

   ```bash
   curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/install-remote-agent.sh | bash
   ```

   On Ubuntu 18.04 / Jetson this bootstraps **Python 3.11** with a static
   `uv` (the official uv build needs glibc 2.28; the musl build runs on
   2.27) and creates a seeded venv, so `pip` works for agent updates.

2. The agent reports `board`, `arch`, `gpu_unsupported`/`cpu_only`. Add
   the machine in the farm and run **Detect hardware**; it will show
   `gpu_vendor: cpu` and no GPU rows.

3. Install the engine. The arm64 prebuilt needs glibc ≥ 2.38, so the farm
   **builds from source** on JetPack 4. For the source build this stack
   needs:

   ```bash
   sudo apt update && sudo apt install -y g++-8 gcc-8
   ```

   The agent prefers `g++-8` automatically, adds `-lstdc++fs` (required by
   gcc 8 for `<filesystem>`), and downloads a portable CMake ≥ 3.14 if the
   system one is older (Ubuntu 18.04 ships 3.10).

4. Select the Jetson's **RAM** chip in a model's hardware placement. It
   joins as a CPU/RAM RPC peer (`-d CPU`, `-ngl 0`).

## RPC notes

- `ggml-rpc-server` exposes one `CPU` device; the farm numbers it as an
  `RPC<N>` peer with resource `RAM`.
- Use the same llama.cpp build as the host (RPC checks the protocol major
  version at handshake). The farm's source bootstrap clones the same
  release tag.
- TCP only; no RDMA. Keep RPC on the LAN/VPN.

## Verification

```bash
~/.llama-peer-agent/venv/bin/llama-peer-agent --version
~/llama.cpp/build/bin/ggml-rpc-server -H 127.0.0.1 -p 50200   # shows CPU
```
