#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
FILE="Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
URL="https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/$FILE"

mkdir -p "$MODEL_DIR"
TARGET="$MODEL_DIR/$FILE"

if [[ -s "$TARGET" ]]; then
  echo "Already downloaded: $TARGET"
  exit 0
fi

echo "Downloading $FILE to $TARGET"
curl -L --fail --retry 3 --progress-bar \
  -o "$TARGET.part" "$URL"

mv "$TARGET.part" "$TARGET"
echo "Done: $TARGET"
