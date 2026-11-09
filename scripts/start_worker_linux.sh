#!/usr/bin/env bash
set -euo pipefail

# Usage: ./start_worker_linux.sh <worker-ip|name|id> [rpc-port] [agent-port]
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Project Python missing; run the repository installer first." >&2
  exit 1
fi

exec "$PYTHON" - "$ROOT" "$@" <<'PY'
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(sys.argv.pop(1), "src"))

import agent_client as agent
import config_loader as cfg

parser = argparse.ArgumentParser(description="Start a CPU RPC worker using the app's agent credentials.")
parser.add_argument("worker", help="Configured machine IP, name, or ID")
parser.add_argument("rpc_port", nargs="?", default=os.environ.get("RPC_PORT"))
parser.add_argument("agent_port", nargs="?", default=os.environ.get("AGENT_PORT"))
args = parser.parse_args()

def port(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise ValueError("ports must be between 1 and 65535")
    return number

def find_machine():
    matches = [m for m in cfg.machines.all()
               if args.worker in (m.get("ip"), m.get("name"), m.get("id"))]
    if len(matches) != 1:
        raise ValueError("Worker must match exactly one configured machine; check the Machines page.")
    return dict(matches[0])

def diagnostics(machine, rpc_port):
    print("\n--- agent diagnostics ---", file=sys.stderr)
    for label, call in (
        ("processes", lambda: agent.status(machine, timeout=5)),
        ("rpc servers", lambda: agent.rpc_list(machine, timeout=5)),
        ("rpc-server.log", lambda: agent.logs_tail(machine, "rpc-server.log", 50, timeout=5)),
    ):
        try:
            print(f"{label}: {call()}", file=sys.stderr)
        except Exception as exc:
            print(f"{label}: unavailable ({exc})", file=sys.stderr)

def main():
    machine = find_machine()
    rpc_port = port(args.rpc_port or machine.get("rpc_port") or 50052)
    machine["agent_port"] = port(args.agent_port or machine.get("agent_port") or 8198)
    ip = machine.get("ip")
    if not ip:
        raise ValueError("Configured machine has no IP address")

    print(f"Checking agent on {ip}:{machine['agent_port']}...", flush=True)
    agent.health(machine, timeout=5.0)
    print(f"Agent is healthy. Ensuring CPU RPC worker on {ip}:{rpc_port}...", flush=True)

    result = {}
    error = []
    def ensure():
        try:
            result.update(agent.rpc_ensure(machine, rpc_port, devices=["CPU"], timeout=30.0))
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=ensure, daemon=True)
    worker.start()
    worker.join(timeout=35.0)

    if worker.is_alive():
        print("ERROR: /rpc/ensure did not return within 35 seconds.", file=sys.stderr)
        diagnostics(machine, rpc_port)
        return 1
    if error:
        raise error[0]

    print(f"Agent returned: {result}", flush=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if agent.port_open(machine, rpc_port, timeout=5.0).get("open"):
            print(f"RPC worker ready on {ip}:{rpc_port}")
            return 0
        time.sleep(1)

    print(f"ERROR: RPC worker did not open port {rpc_port} within 30 seconds.", file=sys.stderr)
    diagnostics(machine, rpc_port)
    return 1

try:
    sys.exit(main())
except (agent.AgentError, ValueError, RuntimeError, OSError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
except KeyboardInterrupt:
    sys.exit(130)
PY
