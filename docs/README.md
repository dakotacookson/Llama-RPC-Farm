# Llama RPC Farm — Documentation Index

Web UI (FastAPI, port 8199) that manages a llama.cpp RPC cluster over SSH:
a Linux master runs llama-swap + llama-server, peers run `ggml-rpc-server`.
Frontend is vanilla JS; all state lives in `config/*.yaml`.

## Reading order

| Doc | What it covers |
|-----|----------------|
| [setup.md](setup.md) | Full setup & installation guide: prerequisites, vault, machines, stores, llama.cpp build, llama-swap, gateway, client config, troubleshooting |
| [architecture.md](architecture.md) | Module map, request flow, data flow, what owns what |
| [api.md](api.md) | Every HTTP + WebSocket endpoint (105 routes), grouped by tab |
| [configuration.md](configuration.md) | All `config/` files: what each stores, who writes it, safe-to-edit rules |
| [secrets.md](secrets.md) | The Fernet vault, `.env` keys, launch paths (`farm-run.sh`, sidecar), injection tests |
| [cross-platform.md](cross-platform.md) | Linux/Windows/macOS handling: remote exec patterns, process mgmt, paths, installer + glibc rules |
| [rpc-model.md](rpc-model.md) | RPC numbering, `compute_rpc_map`, device tokens, ports, generated scripts |
| [visualizer.md](visualizer.md) | Flow visualizer + Live Logs: data sources, sched parsing, attribution policy, priming |
| [tuner.md](tuner.md) | Auto-tune: math estimate, staged live search, safe defaults |
| [models-and-variants.md](models-and-variants.md) | Model configs, flags, device_order, variants, serve groups |
| [development.md](development.md) | Running, testing, the string-assertion test style, common pitfalls (see also `AGENTS.md`) |

## Quick orientation

```
src/app.py        FastAPI entrypoint (routes; logic lives in the modules below)
src/rpc_manager.py   RPC servers, llama-server commands, script generation
src/swap_manager.py  llama-swap daemon lifecycle + config push
src/tuner.py         model auto-tune (estimate + live benchmark search)
ui/static/app.js     single-page frontend (vanilla JS, ~6.5k lines)
config/              runtime state (YAML, CRUD'd by the app at :8199)
```

Agents: read `../AGENTS.md` first — it holds the traps and gotchas.