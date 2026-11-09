#!/usr/bin/env bash
set -euo pipefail

WORKER_IP="${1:?Usage: ./run_cli.sh <worker-ip> <worker-user>}"
WORKER_USER="${2:?Usage: ./run_cli.sh <worker-ip> <worker-user>}"

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL_FILE="${MODEL_FILE:-Qwen_Qwen3.5-0.8B-Q4_K_M.gguf}"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/bin/llama-server}"
RPC_PORT="${RPC_PORT:-50052}"
HTTP_PORT="${HTTP_PORT:-8084}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: llama-server not found or not executable: $LLAMA_SERVER" >&2
  exit 1
fi

if [[ ! -s "$MODEL_PATH" ]]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  echo "Run ./get_model.sh first, or set MODEL_DIR/MODEL_FILE." >&2
  exit 1
fi

echo "Starting llama-server"
echo "Model: $MODEL_PATH"
echo "RPC worker: $WORKER_IP:$RPC_PORT"
echo "HTTP endpoint: http://127.0.0.1:$HTTP_PORT"
echo "Press Ctrl-C to stop."

exec "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --ctx-size 8192 \
  -ngl 99 \
  --flash-attn on \
  --jinja \
  --rpc "$WORKER_IP:$RPC_PORT" \
  --device "MTL0,RPC0" \
  --host 127.0.0.1 \
  --port "$HTTP_PORT"
