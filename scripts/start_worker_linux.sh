#!/usr/bin/env bash
set -euo pipefail

WORKER_IP="${1:?Usage: ./start_worker_linux.sh <worker-ip> <worker-user> [rpc-port]}"
WORKER_USER="${2:?Usage: ./start_worker_linux.sh <worker-ip> <worker-user> [rpc-port]}"
RPC_PORT="${3:-${RPC_PORT:-50052}}"

WORKER_RPC_BIN="${WORKER_RPC_BIN:-$HOME/llama-rpc-worker/build/bin/ggml-rpc-server}"
WORKER_LOG="${WORKER_LOG:-/tmp/ggml-rpc-server.log}"

SSH_TARGET="$WORKER_USER@$WORKER_IP"
export WORKER_START_FILE="${WORKER_START_FILE:-/tmp/llama_rpc_worker_start.epoch}"
date +%s > "$WORKER_START_FILE"

wait_for_rpc() {
  local tries="${RPC_WAIT_SECONDS:-60}"
  for ((i=1; i<=tries; i++)); do
    if nc -z -w 2 "$WORKER_IP" "$RPC_PORT" >/dev/null 2>&1; then
      echo "RPC worker ready on $WORKER_IP:$RPC_PORT"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: RPC worker not reachable on $WORKER_IP:$RPC_PORT after ${tries}s" >&2
  return 1
}

echo "Checking RPC worker on $SSH_TARGET:$RPC_PORT..."

if nc -z -w 2 "$WORKER_IP" "$RPC_PORT" >/dev/null 2>&1; then
  echo "RPC worker already running on $WORKER_IP:$RPC_PORT"
else
  echo "Starting RPC worker on $WORKER_IP:$RPC_PORT..."
  ssh -o BatchMode=yes "$SSH_TARGET" \
    "nohup '$WORKER_RPC_BIN' -H 0.0.0.0 -p $RPC_PORT --device CPU \
      >'\"\$HOME/$WORKER_LOG\"' 2>&1 & echo \$!"
  wait_for_rpc
fi

echo "Streaming worker log: $SSH_TARGET:$WORKER_LOG"
ssh -o BatchMode=yes "$SSH_TARGET" \
  "tail -n 0 -F '$WORKER_LOG'" | \
  perl -MTime::HiRes=time -ne 'BEGIN { open my $f, "<", $ENV{WORKER_START_FILE}; chomp(my $s = <$f>); $start = $s } { my $e = time() - $start; $| = 1; my $c = /error|failed|fatal/i ? 31 : /closed|closing|recv returned 0/i ? 33 : /graph_compute|graph_recompute/i ? 35 : /accepted/i ? 32 : 0; printf "\e[36m%02d:%02d.%03d\e[0m \e[%dm%s\e[0m", int($e/60), int($e)%60, ($e-int($e))*1000, $c, $_ }'
