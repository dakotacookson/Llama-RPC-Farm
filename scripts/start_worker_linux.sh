#!/usr/bin/env bash
set -euo pipefail

WORKER_IP="${1:?Usage: ./start_worker_linux.sh <worker-ip> <worker-user> [rpc-port]}"
WORKER_USER="${2:?Usage: ./start_worker_linux.sh <worker-ip> <worker-user> [rpc-port]}"
RPC_PORT="${3:-${RPC_PORT:-50052}}"

WORKER_RPC_BIN="${WORKER_RPC_BIN:-$HOME/llama.cpp/build/bin/ggml-rpc-server}"
WORKER_LOG="${WORKER_LOG:-/tmp/ggml-rpc-server.log}"

SSH_TARGET="$WORKER_USER@$WORKER_IP"

echo "Checking existing RPC worker on $SSH_TARGET:$RPC_PORT..."

if ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_TARGET" \
  "pgrep -f 'ggml-rpc-server.*-p $RPC_PORT' >/dev/null"; then
  echo "RPC worker already running on $WORKER_IP:$RPC_PORT"
else
  echo "Starting RPC worker on $WORKER_IP:$RPC_PORT..."
  ssh -o BatchMode=yes "$SSH_TARGET" \
    "nohup '$WORKER_RPC_BIN' -H 0.0.0.0 -p $RPC_PORT --device CPU \
      >'$WORKER_LOG' 2>&1 & echo \$!"
fi

echo "Streaming worker log: $SSH_TARGET:$WORKER_LOG"
ssh -o BatchMode=yes "$SSH_TARGET" \
  "tail -n 50 -F '$WORKER_LOG'"
