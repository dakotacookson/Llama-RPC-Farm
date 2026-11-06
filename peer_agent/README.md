# llama-rpc-farm-agent

Optional tiny agent for Llama RPC Farm peers. It is **not required** — the
farm keeps working over SSH. Use it when you want deterministic lifecycle
control, consistent process listings on Linux/Windows, and less per-action
SSH.

## What it does

- `GET /health`, `GET /version`, `GET /capabilities`, `GET /info`
- `GET /status`, `GET /processes`, `GET /metrics`
- `GET /system/detail`, `GET /system/rdma`, `GET /speedtest?mb=N`
- `POST /ray/start|stop`, `POST /install/artifact|verify|vllm`
- `GET /build/probe`, `POST /build/update`
- `GET /logs/tail?name=&lines=`, `GET /logs/list`
- `POST /files/write|mkdir|chmod|backup`, `GET /files/stat|list`
- `POST /service/start|stop|validate`, `GET /service/status`
- `GET /which?name=`, `GET /port/open?port=N`
- `GET /rpc/list`
- `POST /rpc/ensure` — start or reuse a `ggml-rpc-server` on one port
- `POST /rpc/stop` — kill only that port's `ggml-rpc-server`
- `GET /llama/list`, `GET /llama/status?port=N`
- `POST /llama/start` — start/reuse a native `llama-server`
- `POST /llama/stop` — kill only that port's `llama-server`
- `POST /llama/stop-all` — stop llama-swap, llama-server, and RPC servers

All HTTP endpoints require `Authorization: Bearer <token>`.

UDP discovery answers `LLAMA_FARM_DISCOVER_V1` on port `8199` with
identity-only metadata (hostname, machine id/MAC, addresses, agent port,
OS/arch). Disable with `--no-discovery` or
`PEER_AGENT_DISCOVERY=0`.

## Install

```bash
python3 -m pip install llama_rpc_farm_agent-*.whl
# or from this folder:
python3 -m pip install .
```

## Run

```bash
export PEER_AGENT_TOKEN='replace-with-a-long-random-token'
llama-peer-agent --host 0.0.0.0 --port 8198
```

Optional:

```bash
export PEER_AGENT_RPC_BIN="$HOME/llama.cpp/build/bin/ggml-rpc-server"
export PEER_AGENT_LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
export PEER_AGENT_LOG_DIR="$HOME"
export PEER_AGENT_STATE_DIR="$HOME/.cache/llama-wrapper"
export PEER_AGENT_MODEL_ROOTS="/mnt/models"
export PEER_AGENT_FILE_ROOTS="/srv/llama-swap"   # optional extra roots
export PEER_AGENT_DISCOVERY_PORT="8199"
```

Bind only to your LAN/VPN. Do not port-forward this service. The token is
the boundary; the agent exposes no arbitrary shell endpoint.

## Master-side setup

Set the same token on the farm master:

```bash
.venv/bin/python scripts/set-agent-token.py <machine-name> '<token>'
```

Then set `transport: auto` (default), `transport: agent` (require agent),
or `transport: ssh` (never use agent) on that machine. `auto` probes
`/health` and falls back to SSH if the agent is unreachable.

Native RDMA/`rdma_cm` peers should stay on `transport: ssh` for now: the
agent's readiness check is TCP-based.

## systemd (Linux)

Copy `llama_peer_agent/templates/llama-peer-agent.service` to
`~/.config/systemd/user/`, create
`~/.config/llama-peer-agent/agent.env` with `PEER_AGENT_TOKEN=...`, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now llama-peer-agent
```
