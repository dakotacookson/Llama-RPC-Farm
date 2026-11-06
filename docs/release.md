# Release pipelines

Two tag-driven GitHub Actions workflows publish separate release artifacts.
Both build on Ubuntu, Windows, and macOS runners.

A third workflow, **Rolling main build** (`release-main.yml`), runs on every
push to `main` (and via `workflow_dispatch`). It builds **both** the app and
the peer agent on all three runners, then refreshes a single prerelease named
`main-latest` (tag `main-latest`, moved to the newest `main` commit). Artifacts
are versioned `0.0.0-main.<short-sha>`. This is a convenience snapshot, not an
official release — tag workflows remain the supported release path.

## Agent — tags `agent-v*`

```bash
git tag agent-v0.3.4
git push origin agent-v0.3.4
```

Artifacts:
- Linux: standalone `llama-peer-agent` binary, `.tar.gz`, `.deb`
- Windows: standalone `llama-peer-agent.exe`, `.zip`
- macOS: standalone binary, `.tar.gz`, `.dmg`
- SHA-256 checksums per platform

The tag must match `peer_agent/pyproject.toml` and
`llama_peer_agent.__version__`.

## App — tags `app-v*`

```bash
git tag app-v1.0.0
git push origin app-v1.0.0
```

Artifacts:
- Linux: standalone `llama-rpc-farm` binary, `.tar.gz`, `.deb`
- Windows: `llama-rpc-farm.exe`, `.zip`
- macOS: standalone binary, `.tar.gz`, `.dmg`
- SHA-256 checksums per platform

The app is built with PyInstaller. Bundled assets include `ui/` and a local
agent wheel under `agent-wheel/`; the release workflow builds that wheel from
`peer_agent/` before packaging.

## Runtime data

Packaged builds never write into the bundle. On first run they create:
- Linux: `~/.config/llama-rpc-farm/`
- macOS: `~/Library/Application Support/LlamaRPCFarm/`
- Windows: `%APPDATA%\\LlamaRPCFarm\\`

`LLAMA_FARM_DATA_DIR` overrides this location. `LLAMA_FARM_HOST` and
`LLAMA_FARM_PORT` override the bind address/port (default `0.0.0.0:8199`).

## Local release smoke test

```bash
python -m pip wheel --no-deps -w agent-wheel ./peer_agent
pyinstaller --noconfirm --clean packaging/app.spec
LLAMA_FARM_DATA_DIR=/tmp/llama-farm-data LLAMA_FARM_PORT=8187 ./dist/llama-rpc-farm
```
