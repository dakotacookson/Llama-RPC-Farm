#!/usr/bin/env bash
set -euo pipefail

LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8084}"
PROMPT="${1:?Usage: ./test_inference.sh \"<prompt>\" [max_tokens]}"
MAX_TOKENS="${2:-25}"
REQUEST_START=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -e 'print clock_gettime(CLOCK_MONOTONIC)')

TIMER_PID=""
cleanup() {
  if [[ -n "$TIMER_PID" ]]; then
    kill "$TIMER_PID" 2>/dev/null || true
    wait "$TIMER_PID" 2>/dev/null || true
    TIMER_PID=""
    printf '\r\033[2K' >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -t 2 ]]; then
  perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC,sleep -e '$|=1; my $start=clock_gettime(CLOCK_MONOTONIC); while (1) { my $ms=int((clock_gettime(CLOCK_MONOTONIC)-$start)*1000); printf "\r\e[2KRunning... \e[34m%02d:%02d.%03d\e[0m", int($ms/60000), int($ms/1000)%60, $ms%1000; sleep 0.1 }' >&2 &
  TIMER_PID=$!
fi
RESPONSE=$(curl -sS --max-time 180 \
  "http://$LLAMA_HOST:$LLAMA_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n \
    --arg prompt "$PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{
      messages: [{role:"user", content:$prompt}],
      max_tokens: $max_tokens,
      stream: false,
      chat_template_kwargs: {enable_thinking: false}
    }')" \
)
cleanup
ELAPSED=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -e 'printf "%.3f", clock_gettime(CLOCK_MONOTONIC) - shift' "$REQUEST_START")
RATE_COLOR=""
RESET=""
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RATE_COLOR=$'\033[94m'
  RESET=$'\033[0m'
fi
printf '%s\n' "$RESPONSE" | jq -r --arg elapsed "$ELAPSED" --arg color "$RATE_COLOR" --arg reset "$RESET" '"Request elapsed: " + (if $color == "" then "" else "\u001b[34m" end) + $elapsed + $reset + " s\n", (.choices[0].message.content // .choices[0].message.reasoning_content // .error.message // ""), ("\n---\nTokens/sec: " + $color + "\(.timings.predicted_per_second // "n/a")" + $reset)'
