#!/usr/bin/env bash
set -euo pipefail

# Single-Mac Metal launcher for llama-server.
# Usage:
#   ./run_mac_local.sh
#   MODEL_FILE=my-model.gguf ./run_mac_local.sh
#   MODEL_FILE=my-model.gguf HTTP_PORT=8085 ./run_mac_local.sh
#   MODEL_FILE="$HOME/GGUF_models/my-model.gguf" ./run_mac_local.sh
# Bare filenames use MODEL_DIR; paths containing / are used as supplied.

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL_FILE="${MODEL_FILE:-Qwen_Qwen3.5-0.8B-Q4_K_M.gguf}"
case "$MODEL_FILE" in
  /*|*/*) MODEL_PATH="$MODEL_FILE" ;;
  *) MODEL_PATH="$MODEL_DIR/$MODEL_FILE" ;;
esac

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/bin/llama-server}"
HOST="${HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8084}"
CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:-99}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: llama-server not found or not executable: $LLAMA_SERVER" >&2
  exit 1
fi

if [[ ! -s "$MODEL_PATH" ]]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  echo 'Set MODEL_FILE to an absolute path, a path relative to the working directory, or a filename under MODEL_DIR.' >&2
  exit 1
fi

echo "Starting llama-server (local Metal)"
echo "Model:         $MODEL_PATH"
echo "Server binary: $LLAMA_SERVER"
echo "HTTP endpoint: http://$HOST:$HTTP_PORT"
echo "Context:       $CTX_SIZE tokens"
echo "GPU layers:    $GPU_LAYERS"
echo "Press Ctrl-C to stop."

exec "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --ctx-size "$CTX_SIZE" \
  -ngl "$GPU_LAYERS" \
  --flash-attn on \
  --jinja \
  --host "$HOST" \
  --port "$HTTP_PORT"
