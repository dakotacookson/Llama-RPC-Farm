# GPU backend E2E

The normal rolling CI runs `tests/e2e/single_node.py` on standard
GitHub-hosted runners. Those are **CPU-only** and exercise the CPU llama.cpp
path (installer flow + Qwen3-0.6B + gateway chat). Real GPU backend testing
needs GPU hardware in the runner.

## What GitHub-hosted runners provide

| Runner | GPU | Notes |
|---|---|---|
| `ubuntu-latest`, `windows-latest` | none | CPU only |
| `macos-latest` | Apple Silicon CPU | Metal GPU passthrough for arbitrary workloads is not guaranteed |
| **GPU larger runner** | **1× NVIDIA Tesla T4 (16 GB)** | Linux/Windows; GitHub Team/Enterprise Cloud; paid per minute; create it in a runner group and use its **runner name** in `runs-on` |

GitHub does **not** host AMD ROCm or Intel XPU runners. Those require a
self-hosted runner (your own box) or a cloud VM registered as a self-hosted
runner.

## Running the backend E2E

Use the manual **GPU backend E2E** workflow
(`.github/workflows/gpu-e2e.yml`) and pass:

- `runner_labels` — JSON array, e.g. `["gpu-t4"]` for a GitHub T4 runner, or
  `["self-hosted","linux","x64","gpu-rocm"]` for an AMD box.
- `gpu_vendor` — `nvidia`, `amd`, `intel`, `apple`, or `auto`.
- `n_gpu_layers` — usually `99`.

The workflow runs the source app/agent (`--source`) so no packaging step is
needed. It uses the app's real `/api/tools/install/sync` flow to fetch the
matching backend build, verifies `llama-server` + `ggml-rpc-server`, then loads
Qwen3-0.6B through llama-swap and the `:8090` gateway.

On a self-hosted machine you can also run it directly:

```bash
python tests/e2e/single_node.py --source --gpu-vendor nvidia
python tests/e2e/single_node.py --source --gpu-vendor amd
python tests/e2e/single_node.py --source --gpu-vendor intel
python tests/e2e/single_node.py --source --gpu-vendor apple   # macOS Metal
```

CPU (used by rolling CI):

```bash
python tests/e2e/single_node.py --source --gpu-vendor cpu
```

## Registering self-hosted runners

On the GPU box, install the standard GitHub Actions runner and give it labels
that match the target, for example:

- NVIDIA Linux: `self-hosted`, `linux`, `x64`, `gpu-nvidia`
- AMD ROCm Linux: `self-hosted`, `linux`, `x64`, `gpu-rocm`
- Intel XPU Linux: `self-hosted`, `linux`, `x64`, `gpu-intel`
- Apple Metal: `self-hosted`, `macos`, `arm64`, `gpu-metal`
- Windows NVIDIA: `self-hosted`, `windows`, `x64`, `gpu-nvidia`

The runner user must be able to run `nvidia-smi` / `rocm-smi` / `xpu-smi`
and start GPU workloads.

## Notes

- GitHub GPU larger runners are T4 only; they cover CUDA, not ROCm/XPU.
- The E2E declares the backend via the machine record and disables host
  hardware autodetection, so the asset selected matches the runner.
- `--gpu-vendor auto` detects the local vendor with the app's own
  `gpu_vendor.detect_vendor_local()`.
