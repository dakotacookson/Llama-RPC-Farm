# farmctl — command-line client

`farmctl` is part of the main app. It is **API-only**: it talks to the
running app on `:8199` over HTTP and never reads `config/`, the vault, or
peers directly. That means it sees the same in-memory state as the UI and
can report on every machine the farm manages.

```
./farmctl <command> [options]
```

Windows: `farmctl.bat`. From a source checkout you can also run
`.venv/bin/python src/farmctl.py`.

## Auth and target

The app trusts loopback (the machine running the farm), so on the master
no key is needed. From another host:

```bash
./farmctl --url http://10.0.0.10:8199 --key sk-farm-... status
# or
export FARMCTL_URL=http://10.0.0.10:8199
export FARMCTL_KEY=sk-farm-...
./farmctl status
```

Global options:

| Option | Meaning |
|---|---|
| `--url URL` | Farm base URL (default `http://127.0.0.1:8199`, env `FARMCTL_URL`/`FARM_URL`) |
| `--key KEY` | Access key (env `FARMCTL_KEY`/`FARM_API_KEY`/`FARM_ACCESS_KEY`) |
| `--json` | Machine-readable output |
| `--timeout SEC` | Per-request timeout (default 120s) |

## Commands

### `status`

One row per configured machine: role, OS, online/offline, RPC running,
GPU count, VRAM, max temp, RAM, CPU — plus the llama-swap daemon state and
currently loaded models. Per-GPU detail follows the table.

```bash
./farmctl status
./farmctl --json status | jq '.stats.totals'
```

APIs: `/api/machines`, `/api/stats`, `/api/swap/status`,
`/api/swap/loaded`.

### `models`

Lists configured models and their variants (effective servers, context,
split mode). Scoped keys only see the models they are granted.

```bash
./farmctl models
./farmctl --json models | jq '.[].model_id'
```

API: `/api/models`.

### `start`

Generates/pushes the llama-swap config and starts the **daemon only**.
llama-swap loads models on demand, so this does **not** load every
configured model. It is the CLI equivalent of the dashboard's
"Launch All" button.

```bash
./farmctl start
```

API: `POST /api/swap/launch`. Prints the launch step report and exits
nonzero if any step fails.

### `stop`

Stops the llama-swap daemon and any farm-managed `llama-server` / RPC
processes, so nothing is left orphaned.

```bash
./farmctl stop
```

API: `POST /api/swap/stop`.

### `launch --model ID`

Loads and warms **one** model through llama-swap, waiting for the load.
Use `model_id--variant-label` for a variant.

```bash
./farmctl launch --model qwen3-8-27b-uncensored-q4-k-m-128k
./farmctl launch --model my-model--small --load-timeout 1800
```

API: `POST /api/swap/load/{model_id}`.

### `unload ID`

Unloads one model from llama-swap, freeing its VRAM/RAM.

```bash
./farmctl unload qwen3-8-27b-uncensored-q4-k-m-128k
```

API: `POST /api/swap/evict/{model_id}`.

### `doctor`

Per-machine readiness checklist (peer agent, `llama-server`,
`ggml-rpc-server`, llama-swap on the swap host) plus daemon state.
Exits nonzero if anything is missing or the daemon is down.

```bash
./farmctl doctor
./farmctl --json doctor | jq '.readiness.machines'
```

APIs: `/api/readiness`, `/api/swap/status`, `/api/machines`.

### `snapshot`

Writes a redacted JSON support snapshot: machines, models, RPC map, swap
settings/status/loaded, stats, settings, and readiness. Password and
token fields are stripped. `--fast` skips the readiness probe; `--out -`
writes to stdout.

```bash
./farmctl snapshot
./farmctl snapshot --fast --out /tmp/farm.json
```

APIs: `/api/machines`, `/api/models?full=1`, `/api/rpc-map`,
`/api/swap/*`, `/api/stats`, `/api/settings`, `/api/readiness`.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Request failed, command reported failure, or app unreachable |
| `2` | Usage error (bad/missing arguments) |
| `130` | Interrupted (Ctrl-C) |

## Notes

- `--json` is accepted before or after the subcommand.
- The CLI never talks to a peer agent directly; every operation goes
  through the app's API, so peer and auth policy live in one place.
- `farmctl` does not hot-reload anything. After editing `src/*.py`, restart
  the app (`./restart.sh --ui`) before expecting new CLI-facing routes to
  exist.
