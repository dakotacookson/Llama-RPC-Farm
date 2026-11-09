# Minimal CLI RPC smoke-test plan

## Goal

Establish the simplest end-to-end local-model workflow:

```text
Download model → start Linux RPC worker → start local llama-server → send prompt → stop
```

This plan deliberately avoids the Farm UI, peer-agent pairing, model registration, llama-swap, and Farm API keys. It uses direct SSH and a directly launched `llama-server`.

## Prerequisites

Before running any script on **Mac - Master**:

- The Linux worker is reachable by IP.
- SSH key authentication to the worker works.
- The worker has `ggml-rpc-server` installed.
- The master has `llama-server` installed.
- The test GGUF is downloaded locally.

No Farm daemon, UI, agent token, model registration, or API key is required for this smoke test.

## File layout

```text
scripts/
  PLAN_cli.md
  get_model.sh
  start_worker_linux.sh
  run_cli.sh
  test_inference.sh
  stop_all.sh
```

## Runtime contract

| Script | Invocation | Stays attached? | Responsibility |
|---|---|---:|---|
| `get_model.sh` | `./get_model.sh` | No | Download one small GGUF |
| `start_worker_linux.sh` | `./start_worker_linux.sh <ip> <user>` | Yes | Start and tail remote RPC worker |
| `run_cli.sh` | `./run_cli.sh <ip> <user>` | Yes | Start and stream local llama-server |
| `test_inference.sh` | `./test_inference.sh "<prompt>" [max_tokens]` | No | Send one completion request |
| `stop_all.sh` | `./stop_all.sh <ip> <user>` | No | Stop local server and remote worker |

## Derived values

No usernames, home directories, model paths, or machine IPs are hardcoded.

- Local user: `$(id -un)`
- Model directory: `${MODEL_DIR:-$HOME/models}`
- Local llama-server: `${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}`
- Worker RPC binary: `${WORKER_RPC_BIN:-$HOME/llama.cpp/build/bin/ggml-rpc-server}`
- Worker RPC port: `${RPC_PORT:-50052}`
- Local HTTP port: `${HTTP_PORT:-8084}`

The worker username must be supplied explicitly; it cannot be assumed to equal the master username.

## Execution order (do not commit names)
Terminal A and B can be left running with prompt output

#### Terminal A
```bash
./get_model.sh

```
```bash
./start_worker_linux.sh 100.121.205.5 joshua
```
#### Terminal B
```bash
./run_cli.sh 100.121.205.5 joshua
```

Then test them from a third terminal 
```bash
./test_inference.sh "The capital of Austria is"
```

Stop with:

```bash
./stop_all.sh 100.121.205.5 joshua
```

## `get_model.sh`

Downloads a small Q4_K_M GGUF into the default model directory. It uses `curl` with redirect following, retry, and a `.part` file to avoid treating a partial download as complete.

```bash
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
```

## `start_worker_linux.sh`

Starts the RPC worker over SSH and remains attached while tailing its log. This is the manual SSH fallback; it intentionally does not use the Farm peer agent.

```bash
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
```

## `run_cli.sh`

Starts the local `llama-server` in the foreground and streams its output. It does not perform an inference test; use `test_inference.sh` in another terminal.

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKER_IP="${1:?Usage: ./run_cli.sh <worker-ip> <worker-user>}" 
WORKER_USER="${2:?Usage: ./run_cli.sh <worker-ip> <worker-user>}"

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL_FILE="${MODEL_FILE:-Qwen_Qwen3.5-0.8B-Q4_K_M.gguf}"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
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
  --device "Metal,RPC0" \
  --host 127.0.0.1 \
  --port "$HTTP_PORT"
```

## `test_inference.sh`

Sends one OpenAI-compatible chat completion to the locally running `llama-server`. It is a thin, reusable wrapper around `curl`.

```bash
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
```

## `stop_all.sh`

Stops the local `llama-server` and the remote RPC worker. It does not kill unrelated processes.

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKER_IP="${1:?Usage: ./stop_all.sh <worker-ip> <worker-user> [rpc-port]}"
WORKER_USER="${2:?Usage: ./stop_all.sh <worker-ip> <worker-user> [rpc-port]}"
RPC_PORT="${3:-${RPC_PORT:-50052}}"

echo "Stopping local llama-server on port $HTTP_PORT..."
lsof -ti TCP:"${HTTP_PORT:-8084}" -sTCP:LISTEN | xargs kill 2>/dev/null || true

echo "Stopping remote RPC worker on $WORKER_IP:$RPC_PORT..."
ssh -o BatchMode=yes "$WORKER_USER@$WORKER_IP" \
  "pkill -f 'ggml-rpc-server.*-p $RPC_PORT' || true"

echo "Done."
```

## What is intentionally omitted

- Farm web UI
- Peer-agent installation and pairing
- Farm machine registration
- Farm model registration
- llama-swap
- Farm API keys
- Hardcoded usernames or absolute paths
- Automatic RPC port selection
- Device-map computation

## When to move beyond this

Use the Farm-managed path when you need:

- Multiple models and automatic loading/unloading.
- A stable OpenAI-compatible endpoint.
- Per-user access keys and usage tracking.
- Automatic RPC port selection and device mapping.
- A dashboard, logs, and lifecycle controls.

At that point, register machines and models through the Farm API/UI, start llama-swap, and use its gateway rather than direct `llama-server` ports.
