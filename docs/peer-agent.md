# Optional peer agent (design plan)

Status: **plan only — not implemented.** SSH remains the default and the
fallback. The agent is an optional reliability upgrade, not a replacement.

## Why

SSH works for bootstrap and most lifecycle actions, but every action is a
new connection with OS-specific shell/quoting. The remaining sharp edges:

- `cmdStop`/cleanup needs the right environment (`SSH_PASS` sidecar)
- Windows remote-local launches cannot read SMB model paths
- process/metrics queries differ across Linux, Windows, macOS
- a dropped SSH connection can leave an action half-done

A tiny peer agent makes those deterministic without changing the
architecture for users who prefer SSH.

## Non-goals

- No model-file transfer/streaming (models stay on the shared store).
- No remote shell. Only whitelisted lifecycle/metrics operations.
- No change to llama.cpp, llama-swap, or generated start/stop semantics.
- No requirement to install it. A farm can remain SSH-only forever.

## Coexistence model

Per machine:

```yaml
transport: auto        # auto | ssh | agent
agent_port: 8198
# agent token lives in the vault, never in YAML
```

Resolution:

1. `ssh` → existing SSH path, unchanged.
2. `agent` → require a healthy agent; fail loudly if unavailable.
3. `auto` (default) → use the agent when it answers `/health` and the
   configured version is compatible; otherwise fall back to SSH.

Every operation is idempotent, so the master can retry over the other
transport. Start scripts and llama-swap config stay transport-agnostic.

## Agent shape

- Single small process per peer (`python -m peer_agent` initially; a
  static binary later if needed).
- Binds to the LAN/Tailscale interface only. Loopback plus an allowlist.
- Token auth (`Authorization: Bearer <agent-token>`), token stored in a
  peer-local 0600 file; master stores it in the Fernet vault.
- No arbitrary command endpoint. All arguments validated against the same
  rules as the master's generated scripts.

### Endpoints (v1)

| Method + path | Purpose |
|---|---|
| `GET /health` | version, os/arch, uptime, swap/llama presence |
| `GET /capabilities` | GPU vendor/indices, visible model store paths, binaries, RPC support |
| `POST /rpc/ensure` | `{port, devices, env}` → start or reuse; returns `{running, pid, reused}` |
| `POST /rpc/stop` | `{port}` → kill only that `ggml-rpc-server` |
| `GET /rpc/list` | `[{pid, port, age_s, cmd}]` (feeds the orphan reaper) |
| `POST /llama/start` | start a remote-local `llama-server` (cmd/flags/port/state key) |
| `POST /llama/stop` | kill by port + clear roll state |
| `GET /processes` | llama-server / llama-swap / rpc-server summary |
| `GET /logs/tail?name=&lines=` | bounded tail of known log files only |
| `GET /metrics` | Prometheus text (GPU/VRAM/temp, process counts) |
| `GET /version` | protocol version + build hash |

All mutating calls take an idempotency key and return the same shape the
master's step reports already use.

## Master integration

Add a transport seam analogous to `engine_registry`, e.g.:

```python
class PeerTransport(Protocol):
    def rpc_ensure(machine, port, devices, env) -> dict: ...
    def rpc_stop(machine, port) -> dict: ...
    def rpc_list(machine) -> list: ...
    def llama_start(machine, spec) -> dict: ...
    def llama_stop(machine, port, state_key) -> dict: ...
    def processes(machine) -> dict: ...
```

- `SshTransport` = today's `rpc_manager` / `ssh_manager` calls.
- `AgentTransport` = HTTP calls.
- `AutoTransport` = health probe, then delegate; log the chosen transport.

The orphan reaper uses `rpc_list` + `rpc_stop`, so it benefits automatically.

## Install / lifecycle

- Install from the existing installer UI/CLI:
  - Linux/macOS: systemd user unit `llama-peer-agent.service` (or launchd).
  - Windows: scheduled task or service running the agent exe.
- Agent reads a 0600 env file for the token; never writes secrets to logs.
- `GET /health` returns the protocol version; the master warns on drift but
  keeps working when the contract is compatible.
- Uninstall removes the unit, token file, and firewall rule only.

## Security

- LAN/VPN only; document “do not port-forward”.
- Token auth on every endpoint; constant-time compare.
- IP allowlist optional (CIDR list from `config/settings.yaml`).
- Request body size limits, strict JSON schema, path allowlist for logs.
- No secrets in responses; no shell interpolation of user strings.
- Audit log for mutating calls (timestamp, caller IP, action, result).
- HTTPS optional; recommend a Tailscale/VPN boundary first.

## Rollout phases

1. **Skeleton**: health/capabilities/version + installer + token vault wiring.
2. **RPC lifecycle**: ensure/stop/list + `SshTransport`/`AgentTransport` seam;
   reaper consumes agent list.
3. **Remote-local llama-server**: start/stop/log tail; fixes the Windows SMB
   limitation for remote-local deployments.
4. **Metrics + process view**: Prometheus endpoint and dashboard integration.
5. **Windows + macOS parity**: service/scheduled-task installers, path rules.
6. **Optional binary build**: PyInstaller/Nuitka single-file executable.

Each phase keeps SSH fully functional, so no user is forced to migrate.

## Testing

- Unit tests against the same stubbed transport fixture used today.
- Contract tests: one fake agent server that implements the v1 endpoints.
- Fallback tests: agent healthy/down/version-mismatch → correct transport.
- Reaper safety tests: loaded model, loading model, active `--rpc` endpoint,
  young orphan, real orphan.
- No live-host tests in the default suite.

## Risks / mitigations

| Risk | Mitigation |
|---|---|
| Extra network service | LAN/VPN bind, token auth, firewall, optional HTTPS |
| Agent drift | version in `/health`, compatibility check, SSH fallback |
| Split-brain actions | idempotency keys + per-port operations, never blanket kill |
| Windows service complexity | scheduled task fallback first, service later |
| Security regression | no shell endpoint; whitelist args; audit log |
