#!/usr/bin/env bash
set -euo pipefail

# Local OpenAI-compatible smoke test. Uses python3 (already required by
# the app) for JSON + timing instead of perl/jq, so the repo has no extra
# CLI dependencies.

LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8084}"
PROMPT="${1:?Usage: ./test_inference.sh \"<prompt>\" [max_tokens]}"
MAX_TOKENS="${2:-25}"

RESPONSE_FILE="$(mktemp)"
TIMER_PID=""

stop_timer() {
  if [[ -n "$TIMER_PID" ]]; then
    kill "$TIMER_PID" 2>/dev/null || true
    wait "$TIMER_PID" 2>/dev/null || true
    TIMER_PID=""
    printf '\r\033[2K' >&2
  fi
}

cleanup() {
  stop_timer
  rm -f "$RESPONSE_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PAYLOAD="$(python3 - "$PROMPT" "$MAX_TOKENS" <<'PY'
import json
import sys

print(json.dumps({
    "messages": [{"role": "user", "content": sys.argv[1]}],
    "max_tokens": int(sys.argv[2]),
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}))
PY
)"

# Live timer while the request is in flight (TTY only).
if [[ -t 2 ]]; then
  python3 - <<'PY' >&2 &
import sys
import time

start = time.monotonic()
while True:
    ms = int((time.monotonic() - start) * 1000)
    sys.stderr.write(
        "\r\033[2KRunning... \033[34m%02d:%02d.%03d\033[0m"
        % (ms // 60000, (ms // 1000) % 60, ms % 1000))
    sys.stderr.flush()
    time.sleep(0.1)
PY
  TIMER_PID=$!
fi

META="$(curl -sS --max-time 180 \
  -o "$RESPONSE_FILE" -w '%{http_code} %{time_total}' \
  "http://$LLAMA_HOST:$LLAMA_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD")"
HTTP_STATUS="${META%% *}"
ELAPSED="${META##* }"
stop_timer

if [[ "$HTTP_STATUS" == "503" ]]; then
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    printf '\033[31mLoading model\033[0m, please wait\n' >&2
  else
    printf 'Loading model, please wait\n' >&2
  fi
  exit 0
fi

python3 - "$RESPONSE_FILE" "$ELAPSED" "$HTTP_STATUS" <<'PY'
import json
import sys

path, elapsed, status = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
except Exception as exc:  # noqa: BLE001
    print(f"Invalid JSON response (HTTP {status}): {exc}", file=sys.stderr)
    sys.exit(1)

choice = (data.get("choices") or [{}])[0] or {}
message = choice.get("message") or {}
text = (message.get("content")
        or message.get("reasoning_content")
        or (data.get("error") or {}).get("message")
        or "")
tps = (data.get("timings") or {}).get("predicted_per_second", "n/a")
print(f"Request elapsed: {elapsed} s")
print(text)
print("\n---")
print(f"Tokens/sec: {tps}")
PY
