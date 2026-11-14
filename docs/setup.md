# Setup & installation guide

End-to-end path from a bare folder to a working farm: web UI up, machines
added, models visible, llama-swap serving, clients hitting the gateway. The app's own config lives under
the project folder (or the platform data dir for packaged builds); engines
and the agent are installed to standard user locations.

## 0. Prerequisites

| Where you need it | Requirement |
|---|---|
| Master (this app's box) | Python 3.11+, a terminal, network reach to every peer (Tailscale or LAN) |
| Peers | The peer agent package (`.deb` / `.pkg` / `setup.exe`) — no remote login needed |
| Models | llama.cpp `llama-server` built (or installed later via the UI) + GGUF files on a shared path |
| Serving clients | Any OpenAI-compatible client (opencode, SillyTavern, curl, …) |
| Optional | `curl` (used by run.sh / run.bat health checks) |

The machine running the app is **always the master**; everything else is a
peer. OS of master/peers can mix Linux / Windows / macOS.

**Raspberry Pi (Linux arm64):** a Pi is a CPU-only peer. Trixie (glibc
2.41) and Ubuntu 24.04 (2.39) use the upstream arm64 prebuilt; Bookworm
(glibc 2.36) cannot load it and the farm automatically builds from
source. Use the `arm64.deb` agent package (or the universal wheel). See
[raspberry-pi.md](raspberry-pi.md).

**NVIDIA Jetson (JetPack 4.6 / Nano/TX):** GPU offload is unsupported by
current llama.cpp (CUDA 10.2); the agent detects it and forces CPU-only.
Source engine builds need `g++-8`; [jetson.md](jetson.md).

## 1. Install

One-shot, idempotent, safe to re-run. It picks a Python 3.11+ (downloading
a local build via `uv` on POSIX if none exists, no sudo), creates `.venv`,
and installs `requirements.txt`.

```bash
# Linux / macOS
./install.sh

# Windows (from cmd.exe)
install.bat
```

This installs the app and Python requirements only. Engines and llama-swap
are installed from the UI after the app starts (step 2's setup checklist).

## 2. Start the web UI

```bash
./run.sh        # Linux / macOS — detached, survives logout
run.bat         # Windows
```

Open `http://localhost:8199`. Stop later with
`fuser -k 8199/tcp` (Linux) or the dashboard's Restart button.

For a quick smoke test see [examples.md](examples.md)

### First launch: create the admin account

A fresh install has **no default credentials** and no access keys, so the
UI opens a **Create admin account** dialog instead of an unusable sign-in
box. Pick a username and a password (8+ characters); this creates the
first web login plus its admin access key and signs the browser in.

- The dialog appears whenever `/api/auth/me` reports `needs_setup: true`
  (no web logins and no access keys).
- `POST /api/auth/bootstrap` is the same first-run path; once any login or
  access key exists it is admin/console-only.
- Later logins/keys are managed from the app (Access/User pages). There is
  no built-in `admin/admin` account to forget to change.

### First-run setup checklist

The Dashboard shows a **Setup needed** panel whenever a machine is missing
something: the peer agent, llama.cpp, `ggml-rpc-server`, or llama-swap on the
swap host. Each row has an install button (engines via the app/peer agent,
llama-swap on its host). The Machines page shows the same status as badges on
each machine row.

### Tabs

| Tab | What it does |
|-----|--------------|
| Dashboard | Live status of every machine: reachability, RPC server, GPU VRAM/temp |
| Machines | Add/edit machines (IP, agent port, agent token, GPUs); start/stop RPC servers |
| Groups | Cluster groups of machines for multi-AI setups |
| Models | Scan directories for GGUF files (shards shown as ONE model); per-model: servers, variants, tensor split, flags key/value editor |
| Launch | Load a model through the selected engines/peers and watch live state |
| Downloads | Search Hugging Face, filter GGUF models, and download into a `store://` path |

## 3. Secrets vault (`.env`)

All sensitive values live in ONE git-ignored `.env` at the project root:

```bash
cp .env.example .env
.venv/bin/python scripts/migrate_secrets_to_env.py   # one-time YAML -> vault
```

- `FARM_SECRET_KEY` + `FARM_SALT` — auto-generated vault key material
- `HF_TOKEN_ENC` — Hugging Face token (set it in the UI, it is encrypted on save)
- `MACHINE_<ID>_AGENT_TOKEN_ENC` — per-machine peer-agent tokens (set via
  the Machines tab; stored Fernet-encrypted, never in the browser or YAML)

All `config/*.yaml` beyond the shipped templates (`settings.example.yaml`
and similar) are **per-install user data and git-ignored** — a fresh clone
contains nothing specific to anyone's setup, and your machine names, model
lists, access keys, and LAN paths never leave your box through this repo.

## 4. Add machines (Machines tab)

Install the agent package on the peer, then enter: name, IP, agent port
(default 8198), agent token, OS, and share mode (`vram` / `ram` / `both`) plus which
GPU device tokens to share (`CUDA0`, `ROCm1`, … — see `src/gpu_vendor.py`).
`Hardware Detect` fills in GPU names, VRAM, core/RAM totals. The app's own
machine is auto-registered as master.

**Auto-discovery is near zero-touch.** The agent's UDP responder reports
identity **and** hardware (GPU vendor/names, VRAM, RAM, cores) for
NVIDIA, AMD ROCm, Intel SYCL and Apple GPUs, so a discovered agent
prefills the form. Agent tokens are never broadcast (that would hand the
peer to anyone on the LAN). The app manages one farm token automatically
in the encrypted vault — there is no Settings field for it. Fresh agents in
their pairing window enroll against it; per-machine tokens are generated
when needed.

**Packaged binaries self-register.** On first run the app/agent installs a
login service (systemd user unit on Linux, LaunchAgent on macOS, Scheduled
Task on Windows) so they come back after reboot/login, and the Windows agent
adds inbound firewall rules for TCP 8198 + UDP 8199. An agent with no token
starts in a 24-hour **pairing window**; the app generates a farm token and
enrolls it automatically (set `PEER_AGENT_PAIRING=0` to disable, or
`PEER_AGENT_PAIRING_WINDOW=<seconds>` to change it). Opt out of auto-start
with `LLAMA_FARM_NO_SERVICE=1` / `PEER_AGENT_NO_SERVICE=1`; run either binary
with `--install-service` / `--uninstall-service` to manage it manually.

Machine identity is the MAC, not the IP — DHCP drift self-heals when the
new IP answers on the agent port (only for machines with a stored MAC).

## 5. Point model paths at shared storage (stores.yaml)

```yaml
stores:
  llm-models:
    windows: \\10.0.0.75\models\llm-models
    linux: /mnt/cifs/llm-models
```

Model configs then use `store://llm-models/<file.gguf>` — never absolute
paths. Add a `macos:` key when a Mac joins.

New downloads are bundled into a per-model folder
(`store://llm-models/<model-stem>/`): the selected weights (with all
shards), every `mmproj`/MTP/draft GGUF in the repo, config/tokenizer
metadata (from the repo, or from its declared `base_model` when the GGUF
repo omits them), LICENSE/README files, and a `farm-bundle.json` manifest.
Existing flat files keep working; only new downloads use bundles.

## 6. Install llama.cpp + RPC (UI: Install engines)

Use the Dashboard **Setup needed** panel or the Machines page per-machine
**Install engines / Reinstall engines** action. The app installs the right
prebuilt through the peer agent (or locally on the master), then verifies
`llama-server` and `ggml-rpc-server` actually run. You can also point
`llama_bin` at an existing build dir, e.g.
`/home/you/llama.cpp/build/bin`. Linux CUDA builds need glibc ≥ 2.38
(Ubuntu 24.04+); Vulkan builds only need 2.34 and are the automatic
fallback. Peers run `ggml-rpc-server` from the same build.

## 7. Start RPC servers + launch models (Launch tab)

Launching a model starts the required RPC servers, then llama-server. Every peer is started through
its peer agent's typed endpoints (`/rpc/ensure`, `/rpc/stop`), with ports
rolled per model if busy (base 50052 is legacy), then the `llama-server`
command runs with your per-model flags. The script contains no secrets;
agent tokens come from the pushed `swap-secrets.env` sidecar.

## 8. Install llama-swap and serve

1. **Install** — when the swap host is missing llama-swap, the Dashboard
   **Setup needed** panel and the Machines page show **Install llama-swap**.
   It downloads the right official prebuilt for the platform
   (`~/.local/bin/llama-swap` POSIX,
   `%USERPROFILE%\llama-swap\llama-swap.exe` Windows).
2. **Push** — saving a model or variant pushes `config/swap.yaml` +
   per-model start scripts to the swap host
   (`~/.config/llama-swap/`). Do not hand-edit that pushed copy; edit
   configs in the app.
3. **Serve** — llama-swap listens on `:8080` and starts/stops model
   servers on demand. It launches scripts as a systemd **user** unit
   (`llama-swap.service`, `Restart=on-failure`) when available; secrets
   reach model commands via a 0600 `swap-secrets.env` sidecar (host-side
   only, git-ignored).

## 9. Gateway (:8090) — built in

The view/access-key reverse proxy is **embedded in the main app** and
starts automatically on `:8090` with the UI. No separate service is
needed, and packaged `.deb`/`.pkg`/`setup.exe` builds include it. Clients point
at the gateway instead of `:8080`.

Environment overrides: `GATEWAY_PORT` (default `8090`), `GATEWAY_HOST`
(default `0.0.0.0`), `GATEWAY_UPSTREAM` (default `http://127.0.0.1:8080`).

**Gateway lifecycle follows the app.** On startup, if a stale standalone
`swap_gateway.py` process still answers on `:8090` (for example from an
older checkout), the app stops it and binds its embedded gateway, so
gateway code can never outlive the app and serve stale logic. An unrelated
service holding the port is never killed — the app logs it and stays off
`:8090`.

If you intentionally run a separate gateway service, start the app with
`FARM_GATEWAY_DISABLE=1`; then the app never touches `:8090`.

## 10. Connect clients

Point any OpenAI-compatible client at the gateway (or llama-swap) and the
model list is served per view:

```
baseURL: http://<master-ip>:8090/v1     # or http://<master-ip>:8080/v1
apiKey:  <an access key from the access/key settings>
```

Example (opencode `opencode.jsonc`):

```json
"provider": {
  "llamaswap": {
    "npm": "@ai-sdk/openai-compatible",
    "options": { "baseURL": "http://127.0.0.1:8090/v1", "apiKey": "<key>",
                 "timeout": 1800000, "headerTimeout": 300000,
                 "chunkTimeout": 600000 }
  }
}
```

Use generous read/chunk timeouts — local models on large contexts can sit
minutes between streamed chunks (prefill, RPC backends, memory pressure).

## Troubleshooting

- **UI won't come up** — check the log (`/tmp/llama-web.log` POSIX,
  `%TEMP%\llama-web.log` Windows); free port 8199 and re-run run.sh.
- **Peer-agent failures** — service/task stopped, token missing/rotated, or
  firewall blocking TCP 8198/UDP 8199. There is no shell fallback: every
  configured machine must have a reachable agent, and installing/repairing
  it is a one-time platform-package step.
- **Model path not found on a peer** — the `store://` alias lacks a key
  for that machine's OS; add it in `stores.yaml`.
- **Port 50052 family busy on a peer** — per-model ports roll
  automatically; check the generated script's port line.
- **Agent token missing** — set it in the Machines tab (stored in the
  vault); generated scripts read it from `swap-secrets.env`.
- **llama-swap exits and stays down after a kill** — a bare SIGTERM exits
  "cleanly" and `Restart=on-failure` skips respawn; bounce the unit
  (`systemctl --user restart llama-swap`) — `./restart.sh --full` does
  this, then restarts the UI.

More depth: [architecture](architecture.md), [secrets](secrets.md),
[cross-platform](cross-platform.md), [configuration](configuration.md),
[rpc-model](rpc-model.md), [api](api.md).

## Peer agent (recommended)

Linux:

```bash
PEER_AGENT_TOKEN='long-random-token' ./scripts/install-peer-agent.sh
```

Windows: install the wheel into
`%USERPROFILE%\.llama-peer-agent\venv`, write the token env file, and
create a boot-triggered Scheduled Task (`LlamaPeerAgent`) that runs the
venv entry point. The master's Machines page shows current→latest agent
version and can update existing agents in place.
