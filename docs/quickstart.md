# Five-minute quickstart

Goal: from nothing to a verified OpenAI/Anthropic completion on one
machine, with no YAML and no manual engine setup.

## One command (Linux / macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/quickstart.sh | bash
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/quickstart.ps1 | iex
```

It is safe to re-run. An existing checkout is reused; an existing app is
reused if it is already listening.

What it does:

1. Clones/updates `~/Llama-RPC-Farm` (or `%USERPROFILE%\Llama-RPC-Farm`).
2. Runs the idempotent installer (Python venv + requirements) when needed.
3. Starts the UI on `:8199` (and the embedded gateway on `:8090`).
4. Installs stock llama.cpp + `ggml-rpc-server` + llama-swap on this host.
5. Downloads the `demo-0.6b` recipe and writes the model config.
6. Launches llama-swap, warms the model, and makes a real completion.
7. Prints the endpoints and a generated access key.

Expected output ends with:

```
UI:         http://127.0.0.1:8199
OpenAI:     http://127.0.0.1:8090/v1
Anthropic:  http://127.0.0.1:8090
Model:      qwen3-0-6b-demo
Access key: sk-farm-...
```

## Choose a different starter model

```bash
./scripts/quickstart.sh --recipe qwen3-4b
./scripts/quickstart.sh --recipe qwen3-8b
./scripts/quickstart.sh --recipe qwen3-32b
./scripts/quickstart.sh --recipe nomic-embed
./scripts/quickstart.sh --recipe qwen3-asr
```

Recipes are in `src/onboarding.py`. Add your own by appending a dict with
`id`, `repo`, `file`, `model_id`, `kind`, `size_gb`, and `flags`.

Skip steps when debugging:

```bash
./scripts/quickstart.sh --no-install      # engines already installed
./scripts/quickstart.sh --no-launch       # download + configure only
./scripts/quickstart.sh --json            # machine-readable result
```

## Docker

```bash
docker compose --profile demo up --build          # CPU demo, auto quickstart
docker compose --profile demo-wizard up --build   # CPU demo, walk the wizard
docker compose --profile solo up                  # normal single box
docker compose --profile gpu up                   # NVIDIA
docker compose --profile gpu-rocm up              # AMD ROCm
docker compose --profile gpu-intel up             # Intel SYCL/Vulkan
```

The demo service runs the same quickstart inside the container and keeps
the UI up afterwards. `demo-wizard` skips the automatic model install so
you can test the setup wizard, recipes, Connect page, and safety check by
hand. Config, models, and keys live in the `farm-data` volume. GPU
passthrough requirements (NVIDIA toolkit, ROCm `/dev/kfd`/`/dev/dri`,
Intel/Vulkan `/dev/dri`, and why Metal must run natively) are in
`docs/distribution.md`.

## What “ready” looks like

- `GET :8090/v1/models` returns the recipe model.
- A chat completion through `:8090` returns content (or reasoning content
  for reasoning-only models).
- `/api/onboarding/safety` reports no `bad` checks.
- The setup wizard’s final screen shows OpenAI, Anthropic, model, and key.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `engine install failed` | Check network access to GitHub releases; re-run the command. |
| Model download is slow | It is a real Hugging Face download; the demo is ~0.4 GB. |
| `llama-swap is not running` | Re-run the quickstart; it launches the daemon. |
| No output in chat | Pick a chat recipe; `nomic-embed` is embeddings-only. |
| Ports already in use | Stop the other farm instance or pass a different `FARM_URL`. |

## Local funnel metrics

The app records local-only timing in `config/onboarding_metrics.json`:
setup shown, storage saved, engines installed, recipe applied, first
model ready, first chat. Nothing is sent off the machine. Query it with:

```bash
curl http://127.0.0.1:8199/api/onboarding/metrics
```
