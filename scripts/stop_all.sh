#!/usr/bin/env bash
set -euo pipefail

WORKER_IP="${1:?Usage: ./stop_all.sh <worker-ip> <worker-user> [rpc-port]}"
WORKER_USER="${2:?Usage: ./stop_all.sh <worker-ip> <worker-user> [rpc-port]}"
RPC_PORT="${3:-${RPC_PORT:-50052}}"
HTTP_PORT="${HTTP_PORT:-8084}"

echo "Stopping local llama-server on port $HTTP_PORT..."
lsof -ti TCP:"$HTTP_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true

echo "Stopping remote RPC worker on $WORKER_IP:$RPC_PORT..."
ssh -o BatchMode=yes "$WORKER_USER@$WORKER_IP" \
  "pkill -f 'ggml-rpc-server.*-p $RPC_PORT' || true"

echo "Done."
