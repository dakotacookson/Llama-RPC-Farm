#!/usr/bin/env bash
set -euo pipefail

# Shared-model smoke test:
# - llama-server runs locally on this Mac
# - the model file stays local to this Mac
# - tensor work is dispatched to the Linux worker's ggml-rpc-server
#
# Usage: ./run_shared_dummy.sh <worker-ip>
# Override: MODEL_PATH, MODEL_FILE, MODEL_DIR, RPC_PORT, HTTP_PORT,
#           CTX_SIZE, GPU_LAYERS, THREADS, LLAMA_SERVER

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT/scripts"

WORKER_IP="${1:?Usage: ./run_shared_dummy.sh <worker-ip>}"
RPC_PORT="${RPC_PORT:-50052}"
HTTP_PORT="${HTTP_PORT:-8084}"
CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:-99}"
THREADS="${THREADS:-4}"

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
DEFAULT_MODEL_FILE="$(sed -n 's/^FILE="\([^"]*\)"$/\1/p' "$SCRIPT_DIR/get_dummy_model.sh" | head -n1)"
if [[ -z "$DEFAULT_MODEL_FILE" ]]; then
  echo "ERROR: Could not determine the default model filename from $SCRIPT_DIR/get_dummy_model.sh" >&2
  exit 1
fi
MODEL_FILE="${MODEL_FILE:-$DEFAULT_MODEL_FILE}"

case "$MODEL_FILE" in
  /*|*/*) MODEL_PATH="$MODEL_FILE" ;;
  *) MODEL_PATH="$MODEL_DIR/$MODEL_FILE" ;;
esac
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_FILE}"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/bin/llama-server}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: RPC-capable llama-server not found or not executable: $LLAMA_SERVER" >&2
  echo "Set LLAMA_SERVER=/path/to/llama-server and retry." >&2
  exit 1
fi

if [[ ! -s "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  echo "Run ./get_dummy_model.sh first, or set MODEL_PATH=/path/to/model.gguf and retry." >&2
  exit 1
fi

echo "Ensuring RPC worker on $WORKER_IP:$RPC_PORT..."
"$SCRIPT_DIR/start_worker_linux.sh" "$WORKER_IP" "$RPC_PORT"

echo "Starting local llama-server"
echo "Model:         $MODEL_PATH"
echo "Server binary: $LLAMA_SERVER"
echo "RPC worker:    $WORKER_IP:$RPC_PORT"
echo "HTTP endpoint: http://127.0.0.1:$HTTP_PORT"
echo "Press Ctrl-C to stop."

exec "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --ctx-size "$CTX_SIZE" \
  -ngl "$GPU_LAYERS" \
  --flash-attn on \
  --jinja \
  --rpc "$WORKER_IP:$RPC_PORT" \
  --device "MTL0,RPC0" \
  --host 127.0.0.1 \
  --port "$HTTP_PORT"
