# Optional peer agent (design plan)

Status: **agent-only** in `peer_agent/` (agent 1.0.0). RPC + native
llama-server lifecycle, remote-local models, multi-host RPC scripts,
custom Windows runtimes, process/status views, logs,
files, service control, installers, stats/tuner probes, discovery, and
in-place agent updates are agent-backed. Remote shell access, password
transport, and paramiko were removed; agents are installed from their own
`.deb`/`.pkg`/`setup.exe`.
Separate release artifacts are built automatically on every `main` push by
`.github/workflows/rolling-main.yml`; the agent is published in each
versioned rolling release, with `main-latest` pointing at the newest build.

## Why

Shell-based remote execution works for bootstrap, but every action is a
new connection with OS-specific shell/quoting. The remaining sharp edges:

- `cmdStop`/cleanup historically needed the right environment (sidecar)
- Windows remote-local launches cannot read SMB model paths
- process/metrics queries differ across Linux, Windows, macOS
- a dropped connection can leave an action half-done

A tiny peer agent makes those deterministic with typed operations.

## Non-goals

- No arbitrary remote shell. Parity comes from typed, validated
  operations (processes, files, services, logs, metrics), not shell.
- No model-file transfer/streaming (models stay on the shared store).
- No change to llama.cpp, llama-swap, or generated start/stop semantics.
- The agent is required for every configured peer; there is no shell-only
  mode.

## Implemented (current)

- `peer_agent/` — dependency-free wheel + sdist (`llama-rpc-farm-agent`)
  with `/health`, `/version`, `/capabilities`, `/info`, `/status`,
  `/processes`, `/metrics`, `/system/detail`, `/system/rdma`, `/speedtest`,
  `/logs/tail`, `/logs/list`, `/files/*`, `/service/*`,
  `/install/*`, `/build/*`, `/custom/start`, `/custom/stop`,
  `/agent/update`, `/rpc/list`, `/rpc/ensure`, `/rpc/stop`,
  `/llama/list`, `/llama/status`, `/llama/start`, `/llama/stop`, and
  `/llama/stop-all`; Bearer-token auth.
- UDP discovery: agents answer `LLAMA_FARM_DISCOVER_V1` on port `8199`
  with identity-only metadata; `src/agent_discovery.py` collects replies.
- `src/agent_client.py` — master-side client. `transport()` always returns
  `agent`; legacy `transport:` values are accepted but ignored.
- `rpc_manager.start_rpc` / `stop_rpc(port)` / `list_rpc_servers` use the
  agent when selected, so the orphan reaper automatically gets the same
  process list on Linux and Windows.
- Native remote-local models use the agent too: the generated
  `start-<model>.sh` calls `/llama/start` (payload JSON pushed next to the
  scripts), holds until the port closes, and `cmdStop` calls
  `/llama/stop`. The same state file (`swap.port.<key>`) records rolled
  ports for cmdStop.
- `scripts/set-agent-token.py` — stores per-machine (or `--farm`) tokens
  in the vault; the token is never written to YAML.
- Machine config fields: `transport` and `agent_port` (defaults
  `auto` / `8198`).
- `machine_status` uses `/status`, RPC log tails use `/logs/tail`, and
  full stop-everything (`stop_rpc(port=None)`) uses `/llama/stop-all`
  when the agent is selected.
- Live Launch-page remote-local runs (`_start_llama_server` /
  `stop_llama_server_port`) also use `/llama/start` / `/llama/stop` when
  the agent is selected.
- `src/peer_transport.py` is the transport seam; its only implementation
  is `AgentTransport` (there is no shell runtime transport).
- Generated RPC-cluster scripts use the agent's `/rpc/ensure`,
  `/rpc/stop`, `/port/open`, and `/rpc/list` endpoints on every OS. RPC
  `cmdStop` is agent-aware.
- Remote-local models and custom Windows `.bat` runtimes use typed agent
  endpoints (`/llama/*`, `/custom/*`).
- Agent updates are release-driven: the Machines page compares the running
  version with the highest published GitHub release. Pip-installed agents
  use `/agent/update` with the release wheel; standalone packaged agents
  use `/agent/update-binary` to replace and restart their executable.
- Linux installs enable the systemd user unit with linger; Windows uses
  a boot-triggered Scheduled Task (`LlamaPeerAgent`). Both survive reboot.
- Stats/tuner use `/metrics`, `/system/detail`, and `/speedtest`;
  machine detail and the filesystem browser use `/system/detail` and
  `/files/*`; discovery merges UDP agent hits into scan results.

## Coexistence model

Per machine:

```yaml
transport: auto        # accepted for compatibility; behaviour is agent-only
agent_port: 8198
# agent token lives in the vault, never in YAML
```

Resolution:

1. Every configured peer is reached through the peer agent.
   `agent_client.transport()` always returns `agent` and
   `should_use_agent()` always returns true; the legacy `transport:`
   value no longer changes behaviour.
2. There is no shell transport helper; any remote command would fail.
3. First-time agent installation/recovery on a host with no agent is an
   explicit out-of-band operation (the platform package), not a fallback
   the app silently takes.

Every operation is idempotent. Start scripts and llama-swap config use
the same typed agent endpoints on every OS.

## Agent shape

- One small process per peer, shipped either as a pip package or a
  standalone PyInstaller binary.
- Binds to the LAN/Tailscale interface only. Loopback plus an allowlist.
- Token auth (`Authorization: Bearer <agent-token>`), token stored in a
  peer-local 0600 file; master stores it in the Fernet vault.
- No arbitrary command endpoint. All arguments validated against the same
  rules as the master's generated scripts.

### Endpoints (v1)

| Method + path | Purpose |
|---|---|
| `GET /health` | version, os/arch, uptime |
| `GET /version` | protocol version + agent version + `packaged` flag |
| `GET /capabilities` | binaries, model roots, RPC/llama availability |
| `GET /info` | hostname, machine id/MAC, addresses, agent port, OS/arch |
| `GET /status` | RPC/llama/swap processes, counts, GPU rows |
| `GET /processes` | typed process summary |
| `GET /metrics` | CPU/RAM/GPU metrics |
| `GET /system/detail` | kernel/CPU/uptime/network/disk/llama version |
| `GET /speedtest?mb=N` | bounded throughput stream (1..256 MB) |
| `GET /system/rdma` | RoCE device/port/GID/IP discovery |
| `POST /install/artifact`, `/install/verify` | typed installs |
| `POST /agent/update` | pip wheel self-update + restart |
| `POST /agent/update-binary` | standalone binary self-replace + restart |
| `GET /build/probe`, `POST /build/update` | llama.cpp source probe/update |
| `GET /logs/tail?name=&lines=` | bounded tail of known log files only |
| `GET /logs/list` | known log filenames |
| `POST /files/write` | bounded atomic write under allowed roots |
| `POST /files/mkdir`, `/files/chmod`, `/files/backup` | bounded file ops |
| `GET /files/stat`, `/files/list` | bounded path inspection |
| `POST /service/start`, `/service/stop`, `/service/validate` | llama-swap control plane |
| `GET /service/status` | daemon status |
| `GET /which?name=` | allowlisted binary lookup |
| `GET /port/open?port=N` | TCP port probe |
| `POST /rpc/ensure` | `{port, devices, env}` → start or reuse |
| `POST /rpc/stop` | `{port}` → kill only that `ggml-rpc-server` |
| `GET /rpc/list` | `[{pid, port, age_s, cmd}]` (feeds the orphan reaper) |
| `POST /llama/start` | `{port, args, env, state_key}` → start/reuse `llama-server` |
| `POST /llama/stop` | `{port, state_key}` → kill that port + recorded rolled ports |
| `POST /llama/stop-all` | stop llama-swap, llama-server, and all RPC servers |
| `POST /custom/start`, `/custom/stop` | typed custom Windows `.bat` runtime |
| `POST /agent/update` | in-place upgrade from a verified wheel; restarts itself |
| `GET /llama/list` | running native `llama-server` processes |
| `GET /llama/status?port=N` | 200 while loading/running, 404 when gone |

There is no shell fallback, no sshpass, and no password transport. All
normal configured-machine operations are agent-backed; installing the
agent is a one-time platform-package step.

Mutating calls are validated and return the same shape as the master's
step reports. Stop-all is explicit and restricted to the known llama
process names; per-port operations remain the preferred path.

## Master integration

`src/peer_transport.py` provides the seam. `AgentTransport` calls
`src/agent_client.py`; it is the only implementation.

```python
class PeerTransport:
    def status(machine) -> dict: ...
    def stop_all(machine) -> dict: ...
    def logs_tail(machine, name, lines) -> dict: ...
    def processes(machine) -> dict: ...
```

Current call sites using the seam or agent branches:

- lifecycle: `start_rpc`, `stop_rpc`, `stop_llama_server_port`,
  `list_rpc_servers`, `_start_llama_server`
- status: `machine_status`
- logs: `_tail_log`
- generated remote-local start/stop scripts
- orphan reaper (`rpc_list` + `rpc_stop`)

New operations must go through `PeerTransport`; there is no direct
remote-shell call site.

## Install / lifecycle

- Install from the release packages (`.deb`, `.pkg`, `setup.exe`):
  - Linux: systemd user unit `llama-peer-agent.service`, enabled with
    linger so it starts at boot without a login.
  - Windows: boot-triggered Scheduled Task `LlamaPeerAgent`, running the
    venv entry point through a token-loading `.cmd` wrapper.
  - macOS: launchd LaunchAgent installed by the `.pkg` postinstall.
- Agent reads a 0600 env file for the token; never writes secrets to logs.
  The farm token itself is auto-generated and managed by the app.
- `GET /health` returns the protocol version; the master warns on drift but
  keeps working when the contract is compatible.
- Discovery binds UDP `8199` (configurable with `PEER_AGENT_DISCOVERY_PORT`
  or `--discovery-port`). The probe/reply contains identity only; enrollment
  still uses the vault token. A firewall must allow the LAN/VPN UDP port.
- File/service endpoints are limited to `$HOME`, `PEER_AGENT_MODEL_ROOTS`,
  and `PEER_AGENT_FILE_ROOTS` (colon-separated on POSIX). Add the swap
  config/script roots there if they live outside home.
- Uninstall removes the unit, token file, and firewall rule only.

## Security

- LAN/VPN only; document “do not port-forward”.
- Token auth on every endpoint; constant-time compare.
- IP allowlist optional (CIDR list from `config/settings.yaml`).
- Request body size limits, strict JSON schema, path allowlist for logs.
- No secrets in responses; no shell interpolation of user strings.
- Audit log for mutating calls (timestamp, caller IP, action, result).
- HTTPS optional; recommend a Tailscale/VPN boundary first.

## Parity roadmap

1. **Lifecycle/status (done)**: RPC + native llama-server, remote-local,
   RPC-cluster scripts/cmdStop, custom Windows runtimes, process/status,
   log tails, stop-all, `PeerTransport` seam.
2. **Discovery (done)**: `/info`, UDP responder, master-side collector,
   Add Machine auto-fill and connection detection.
3. **Agent operations (done)**: files, service control, installers,
   build probe/update, RDMA, stats/tuner probes, custom runtimes.
4. **Agent lifecycle (done)**: Linux linger/systemd, Windows boot task,
   macOS LaunchAgent, release wheel/binary updates, version display and
   update button.
5. **Packaging (done)**: signed/standalone agent and app installers for
   Linux, Windows, and macOS, plus raw agent binaries and a universal wheel.
6. **Metrics (done)**: `/metrics` agent endpoint and dashboard probes.
7. **macOS parity (done)**: launchd installer and live validation.

Every slice is agent-backed; there is no parallel shell runtime path.

## Release packaging

Source stays in this monorepo; releases are rolling artifacts built on
every `main` push:

- agent: Linux `.deb`, macOS `.pkg`, Windows Inno Setup `setup.exe`,
  raw per-platform standalone binaries, and a universal pip wheel, plus
  SHA-256 checksums in each versioned rolling release
- app: the same three installer formats in the same release

`main-latest` is a moving pointer to the newest versioned release; older
builds are retained under their own version tags.

See `docs/release.md` for the platform workflows. Bump
`peer_agent/pyproject.toml` and `llama_peer_agent.__version__` when the
agent protocol changes. Protocol compatibility is reported by `/version`;
the master accepts protocol 1 only and reports incompatible/missing
agents as offline.

## Testing

- Unit tests against the same stubbed transport fixture used today.
- Contract tests: one fake agent server that implements the v1 endpoints.
- Fallback tests: agent healthy/down/version-mismatch → correct transport.
- Reaper safety tests: loaded model, loading model, active `--rpc` endpoint,
  young orphan, real orphan.
- Discovery contract test with a loopback UDP responder.
- No live-host tests in the default suite.

## Risks / mitigations

| Risk | Mitigation |
|---|---|
| Extra network service | LAN/VPN bind, token auth, firewall, optional HTTPS |
| Agent drift | version in `/health`, compatibility check, explicit in-place `/agent/update` |
| Split-brain actions | per-port operations preferred; explicit typed stop-all only |
| Windows service complexity | scheduled task fallback first, service later |
| Security regression | no shell endpoint; whitelist args; audit log |
