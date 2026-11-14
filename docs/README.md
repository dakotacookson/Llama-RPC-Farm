# Llama RPC Farm — Documentation Index

Web UI (FastAPI, port 8199) that manages a llama.cpp RPC cluster through
peer agents: a Linux master runs llama-swap + llama-server, peers run
`ggml-rpc-server`. Agents ship as separate platform packages.
Frontend is vanilla JS; all state lives in `config/*.yaml`.

## Reading order

| Doc | What it covers |
|-----|----------------|
| [setup.md](setup.md) | Full setup & installation guide: prerequisites, vault, machines, stores, llama.cpp build, llama-swap, gateway, client config, troubleshooting |
| [raspberry-pi.md](raspberry-pi.md) | Raspberry Pi / Linux arm64: Bookworm vs Trixie install routes, agent, CPU-only RPC peers |
| [jetson.md](jetson.md) | NVIDIA Jetson/Tegra: GPU support by JetPack generation, CPU-only RPC on JetPack 4 (Nano/TX) |
| [topologies.md](topologies.md) | Single-box, RPC, peer-local, custom, MLX; deployment recipes and anti-patterns |
| [access.md](access.md) | Users, API keys, per-key model grants, gateway, client config |
| [benchmarks.md](benchmarks.md) | Real measured runs, how to reproduce them, and a template for submitting yours |
| [testing.md](testing.md) | In-app test bench and the development/e2e test suite |
| [faq.md](faq.md) | RPC speed, matching GPUs, Windows/macOS, security, common questions |
| [architecture.md](architecture.md) | Module map, request flow, data flow, what owns what |
| [api.md](api.md) | Every HTTP + WebSocket endpoint (105 routes), grouped by tab |
| [cli.md](cli.md) | `farmctl`: headless CLI over the same API — status, models, start/stop, launch/unload one model, doctor, snapshot |
| [configuration.md](configuration.md) | All `config/` files: what each stores, who writes it, safe-to-edit rules |
| [secrets.md](secrets.md) | The Fernet vault, agent tokens, `.env` keys, sidecar, injection tests |
| [../SECURITY.md](../SECURITY.md) | Threat model, trusted-network rules, vulnerability reporting, hardening |
| [cross-platform.md](cross-platform.md) | Linux/Windows/macOS handling: remote exec patterns, process mgmt, paths, installer + glibc rules |
| [rpc-model.md](rpc-model.md) | RPC numbering, `compute_rpc_map`, device tokens, ports, generated scripts |
| [visualizer.md](visualizer.md) | Flow visualizer + Live Logs: data sources, sched parsing, attribution policy, priming |
| [tuner.md](tuner.md) | Auto-tune: math estimate, staged live search, safe defaults |
| [models-and-variants.md](models-and-variants.md) | Model configs, flags, device_order, variants, serve groups |
| [development.md](development.md) | Running, testing, the string-assertion test style, common pitfalls (see also `AGENTS.md`) |
| [peer-agent.md](peer-agent.md) | Peer agent: agent-first transport, endpoints, discovery, auto-start, in-place updates, packaging roadmap |
| [release.md](release.md) | Rolling `main-latest` build pipelines and native artifacts |
| [windows-signing.md](windows-signing.md) | Windows SmartScreen, code-signing options, and how to run an unsigned build safely |
| [gpu-e2e.md](gpu-e2e.md) | GPU backend E2E: hosted runner limits, self-hosted labels, ROCm/XPU/Metal |

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