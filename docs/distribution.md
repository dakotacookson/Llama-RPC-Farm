# Distribution channels

Templates live under `packaging/`. They are starting points: update
versions/hashes for each release before submitting upstream.

## Docker / GHCR

Images are built by `.github/workflows/docker.yml`:

- `ghcr.io/dakotacookson/llama-rpc-farm:latest`
- `ghcr.io/dakotacookson/llama-rpc-farm-agent:latest`

Profiles:

```bash
docker compose --profile demo up --build          # CPU demo, auto quickstart
docker compose --profile demo-wizard up --build   # CPU demo, setup wizard visible
docker compose --profile solo up                  # normal single box
docker compose --profile gpu up                   # NVIDIA
docker compose --profile gpu-rocm up              # AMD ROCm
docker compose --profile gpu-intel up             # Intel SYCL/Vulkan
PEER_AGENT_TOKEN=... docker compose --profile agent up -d
PEER_AGENT_TOKEN=... docker compose --profile agent-nvidia up -d
PEER_AGENT_TOKEN=... docker compose --profile agent-rocm up -d
```

### GPU / hardware passthrough

Docker only passes through what the host exposes:

- **NVIDIA** — install `nvidia-container-toolkit`, configure Docker, then
  verify `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
  The `gpu` and `agent-nvidia` profiles use Compose device reservations.
- **AMD ROCm** — host needs the `amdgpu` driver + ROCm runtime + the
  `video`/`render` groups. The `gpu-rocm`/`agent-rocm` profiles expose
  `/dev/kfd` and `/dev/dri`, add those groups, and disable seccomp.
  Set `HSA_OVERRIDE_GFX_VERSION` for unsupported consumer cards (e.g.
  `10.3.0` for some RDNA2 parts).
- **Intel SYCL/Vulkan** — expose `/dev/dri` and install the matching
  Mesa/Vulkan or oneAPI userspace. The `gpu-intel` profile does the device
  mapping; the engine installer chooses the SYCL prebuilt.
- **Vulkan on any vendor** — expose `/dev/dri` and the Vulkan ICD; the
  installer falls back to Vulkan when CUDA/ROCm is unavailable.
- **Apple Metal** — Docker Desktop cannot pass Metal into a Linux
  container. Use the native macOS `.pkg` or `scripts/quickstart.sh`.

For a **peer agent** container, use `network_mode: host` (already set) and
the same device mappings as the farm. Add the peer to the farm, then let
the farm install engines through the agent. Mount the model store into the
agent container if you use shared storage.

The demo container is CPU-only by design (it seeds `gpu_vendor: cpu`) so it
works on any machine and in CI. Use `demo-wizard` to test the first-run
wizard / recipe / Connect flows without auto-downloading a model.

## Homebrew (macOS)

Template: `packaging/homebrew/llama-rpc-farm.rb`.

```bash
brew install --cask ./packaging/homebrew/llama-rpc-farm.rb
```

For a tap, copy the cask into `homebrew-llama-rpc-farm/Casks/` and publish
that repository.

## winget (Windows)

Template: `packaging/winget/DakotaCookson.LlamaRPCFarm.yaml`.

Update `PackageVersion` and `InstallerSha256` per release, then submit with
`wingetcreate` or a PR to `microsoft/winget-pkgs`.

## Proxmox VE

Template: `packaging/proxmox/install-llama-rpc-farm.sh`.

Run inside a Debian/Ubuntu VM or LXC:

```bash
curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/packaging/proxmox/install-llama-rpc-farm.sh | bash
```

For GPU/RPC peers, install the peer agent on each host; do not expect a
container to see GPUs unless the host passes them through.

## Unraid

Template: `packaging/unraid/llama-rpc-farm.xml`.

Add it to Community Applications, or install manually via
Docker → Add Container → Template URL. Persist `/data` to
`/mnt/user/appdata/llama-rpc-farm`.

## CasaOS

Template: `packaging/casaos/docker-compose.yml`.

Use CasaOS → App Store → Install a customized app, or drop the compose
file into `/DATA/AppData/llama-rpc-farm`.

## TrueNAS SCALE

Use the Docker/Compose path with a dataset mounted at `/data`. TrueNAS
users should enable the NVIDIA container runtime for GPU recipes.

## Release checklist

- [ ] Bump `VERSION` when starting a new minor/major line.
- [ ] Verify release assets and checksums.
- [ ] Update winget version + SHA256.
- [ ] Update Homebrew cask if the package name changes.
- [ ] Confirm GHCR tags point at the new release.
- [ ] Re-run the demo smoke test.
