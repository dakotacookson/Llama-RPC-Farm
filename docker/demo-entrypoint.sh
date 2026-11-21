#!/usr/bin/env bash
# Demo container: start the farm, install engines, load the tiny model,
# verify a completion, then keep the UI running.
set -euo pipefail
cd /app
mkdir -p /data/config /data/models

# Demo is CPU-only: seed a local master so the engine resolver never picks
# a CUDA build on a host that has no GPU runtime.
HOST="${HOSTNAME:-demo}"
if [ ! -f /data/config/machines.yaml ]; then
cat > /data/config/machines.yaml <<MACHINES
machines:
- name: ${HOST}
  ip: 127.0.0.1
  role: master
  os: linux
  gpu_vendor: cpu
  gpu: ''
  rpc_port: 50052
  expose:
    mode: both
    gpus: []
MACHINES
fi

if [ ! -f /data/config/settings.yaml ]; then
  if [ "${FARM_DEMO_AUTO:-1}" = "1" ]; then
    WIZARD_DONE=true
  else
    WIZARD_DONE=false
  fi
  cat > /data/config/settings.yaml <<SETTINGS
model_dir: store://llm-models
llama_bin: ~/llama.cpp/bin
engine: llama_cpp
storage_mode: shared
setup_wizard_done: ${WIZARD_DONE}
SETTINGS
fi

python src/app.py &
APP_PID=$!
trap 'kill -TERM "$APP_PID" 2>/dev/null || true' TERM INT

for _ in $(seq 1 90); do
  if python - <<'PY'
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8199/api/machines", timeout=2)
except Exception:
    raise SystemExit(1)
PY
  then break; fi
  sleep 1
done

if [ "${FARM_DEMO_AUTO:-1}" = "1" ]; then
  python src/quickstart.py --recipe "${FARM_DEMO_RECIPE:-demo-0.6b}" || {
    echo "[demo] quickstart failed; keeping the UI up for inspection" >&2
  }
else
  echo "[demo] wizard profile: open http://127.0.0.1:8199 and walk the setup" >&2
fi

wait "$APP_PID"
