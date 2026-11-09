#!/usr/bin/env bash
set -euo pipefail

# Agent-only worker shutdown.
# Usage: ./stop_all.sh <worker-ip|name|id> [rpc-port] [agent-port]

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Project Python missing; run the repository installer first." >&2
  exit 1
fi

exec "$PYTHON" - "$ROOT" "$@" <<'PY'
import argparse
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(sys.argv.pop(1), "src"))

import agent_client as agent
import config_loader as cfg

parser = argparse.ArgumentParser(
    description="Stop the local llama-server and remote CPU RPC worker."
)
parser.add_argument("worker", help="Configured machine IP, name, or ID")
parser.add_argument("rpc_port", nargs="?", default=os.environ.get("RPC_PORT"))
parser.add_argument("agent_port", nargs="?", default=os.environ.get("AGENT_PORT"))
args = parser.parse_args()

def port(value, default):
    number = int(value or default)
    if not 1 <= number <= 65535:
        raise ValueError("ports must be between 1 and 65535")
    return number

def find_machine():
    matches = [m for m in cfg.machines.all()
               if args.worker in (m.get("ip"), m.get("name"), m.get("id"))]
    if len(matches) != 1:
        raise ValueError(
            "Worker must match exactly one configured machine; check the Machines page."
        )
    return dict(matches[0])

machine = find_machine()
rpc_port = port(args.rpc_port, machine.get("rpc_port") or 50052)
machine["agent_port"] = port(
    args.agent_port, machine.get("agent_port") or 8198
)
http_port = int(os.environ.get("HTTP_PORT") or 8084)

print(f"Stopping local llama-server on port {http_port}...")
try:
    out = subprocess.run(
        ["lsof", "-ti", f"TCP:{http_port}", "-sTCP:LISTEN"],
        capture_output=True, text=True, check=False,
    ).stdout
except FileNotFoundError:
    print("WARN: lsof not found — cannot locate the local listener; "
          "stop it manually (e.g. Ctrl-C).")
    out = ""
pids = [int(p) for p in out.split() if p.strip().isdigit()]
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"WARN: could not stop pid {pid}: {exc}")
if pids:
    time.sleep(1.0)  # let it exit, then force any survivor
    for pid in pids:
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    print(f"Stopped local llama-server ({len(pids)} process(es)).")
else:
    print("No local llama-server is listening on that port.")

print(f"Stopping remote RPC worker on {machine['ip']}:{rpc_port} via agent...")
result = agent.rpc_stop(machine, rpc_port)
print(f"Agent returned: {result}")
print("Done.")
PY
