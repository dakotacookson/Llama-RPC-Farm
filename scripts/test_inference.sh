#!/usr/bin/env bash
set -euo pipefail

LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8084}"
PROMPT="${1:?Usage: ./test_inference.sh \"<prompt>\" [max_tokens]}"
MAX_TOKENS="${2:-32}"

curl -sS --max-time 180 \
  "http://$LLAMA_HOST:$LLAMA_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n \
    --arg prompt "$PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{
      messages: [{role:"user", content:$prompt}],
      max_tokens: $max_tokens,
      stream: false
    }')" \
| jq -r '.choices[0].message.content // .error // .'
