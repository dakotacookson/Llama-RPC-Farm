# Raspberry Pi (Linux arm64)

Raspberry Pi 4 and 5 (and other 64-bit Arm boards) run the same stock
llama.cpp as every other machine. There is **no VideoCore GPU backend**:
a Pi is a CPU-only host. Use it as a small single-box endpoint or as an
extra CPU/RAM peer in a larger RPC model.

## Supported OS versions

Upstream llama.cpp ships Linux arm64 prebuilts
(`llama-<tag>-bin-ubuntu-arm64.tar.gz`) built on Ubuntu 24.04. Those
binaries import `GLIBC_2.38`, which decides the install route:

| OS | glibc | Route |
|---|---|---|
| Raspberry Pi OS **Trixie** (Debian 13) | 2.41 | Prebuilt (CPU) |
| **Ubuntu 24.04** arm64 | 2.39 | Prebuilt (CPU) |
| Raspberry Pi OS **Bookworm** (Debian 12) | 2.36 | **Source build** (prebuilt cannot load) |

The farm detects this automatically: `prebuilt_blocker()` sees a Linux
arm64 host with glibc < 2.38 and selects the source build instead of
installing a binary that would fail at exec with
`GLIBC_2.38 not found`. No manual switch is needed.

Use a **64-bit** OS (`uname -m` = `aarch64`). 32-bit Raspberry Pi OS is
not supported by the arm64 prebuilts and is not recommended.

The same rule applies to the farm app and agent installers: the arm64
`.deb` binaries are built on Ubuntu 24.04 and target Trixie / Ubuntu
24.04. On Bookworm, install the app with the one-line **source** installer
(`install-remote.sh`) and the agent with the universal wheel
(`install-remote-agent.sh`); both run on Bookworm's glibc 2.36.

## Add a Pi as a peer

1. Install the peer agent on the Pi. The universal wheel is pure Python
   and works on **every** Pi OS version (including Bookworm) with no
   build step:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/install-remote-agent.sh | bash
   ```

   The `llama-rpc-farm-agent_<version>_arm64.deb` asset is also available,
   but it is a PyInstaller build produced on Ubuntu 24.04, so use it on
   **Trixie / Ubuntu 24.04**; on Bookworm prefer the wheel. The agent
   listens on `:8198` and answers UDP discovery on `:8199`; systemd user
   lingering starts it at boot.

2. In the farm UI, **Machines → Add**, enter the Pi's IP and agent token,
   then **Detect hardware**. The card shows the board model
   (`Raspberry Pi 5 Model B Rev 1.0`) and arch `aarch64`.

3. Install the engine from the machine card. On Bookworm the farm builds
   llama.cpp from source in the background (this can take a while on a
   Pi — the build clamps `-j` to available RAM); on Trixie/Ubuntu it
   downloads the arm64 prebuilt. For the source route install the
   toolchain first:

   ```bash
   sudo apt update && sudo apt install -y git cmake build-essential
   ```

4. Select the Pi's **RAM** chip in a model's hardware placement. The Pi
   exposes a single `CPU` device, which the farm numbers as an `RPC<N>`
   peer with resource `RAM`.

## RPC notes

- `ggml-rpc-server` exposes a single `CPU` device when no accelerator is
  present, so a Pi joins the cluster exactly like any other CPU/RAM peer.
- Use the same llama.cpp build as the host. RPC checks the protocol major
  version at handshake; a mismatched build fails to connect.
- TCP only — no RDMA on a Pi. Keep RPC on the LAN/VPN; it is
  unauthenticated by design.
- Performance: a Pi contributes RAM/CPU capacity but is network-bound and
  slow as a compute device. If a model fits on one box, keep it there.

## Troubleshooting

- **`GLIBC_2.38 not found`** — the prebuilt was installed on Bookworm.
  Re-run the engine install; the farm will use the source build. (If you
  installed outside the farm, rebuild from source.)
- **Source build runs out of memory** — the installer clamps `-j` using
  available RAM. Close other services or add swap; use a Pi 4/5 with at
  least 4 GB.
- **Pi shows as x86_64** — hardware was detected before the arch was
  known. Run **Detect hardware** again; it persists `arch` and `board`
  from the agent.
- **Thermal throttling / low tok/s** — use active cooling and a good PSU;
  expect a few tok/s on 1B–3B Q4 models on a Pi 5.
