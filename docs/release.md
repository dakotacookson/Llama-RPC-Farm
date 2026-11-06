# Release pipelines

Releases are rolling **`main-latest`** prereleases built automatically on
every push to `main` (and manually via `workflow_dispatch`). There are no
tag-driven release workflows.

## Platform workflows

One workflow per platform, each building **both** the app and the peer agent:

| Workflow | Runner | Artifacts |
|---|---|---|
| `main-build-linux.yml` | `ubuntu-latest` | `llama-rpc-farm`, `llama-peer-agent`, `.deb`, `.tar.gz`, SHA-256 |
| `main-build-macos.yml` | `macos-latest` | `llama-rpc-farm`, `llama-peer-agent`, `.dmg`, `.tar.gz`, SHA-256 |
| `main-build-windows.yml` | `windows-latest` | `llama-rpc-farm.exe`, `llama-peer-agent.exe`, `.zip`, SHA-256 |

The Linux workflow also runs the pre-release checks (py_compile,
`node --check`, `bash -n`, pytest) before packaging.

## Publishing

Each platform workflow calls the reusable `publish-main-latest.yml`, which:

1. Moves the `main-latest` tag to the current `main` commit.
2. Creates the `main-latest` prerelease if missing.
3. Uploads that platform's assets with `--clobber`.

Publishing is serialized with the `main-latest-release` concurrency group,
so all three platforms accumulate into one prerelease without racing.

Artifacts are versioned `0.0.0-main.<short-sha>`.

## Local release smoke test

```bash
python -m pip wheel --no-deps -w agent-wheel ./peer_agent
pyinstaller --noconfirm --clean packaging/app.spec
LLAMA_FARM_DATA_DIR=/tmp/llama-farm-data LLAMA_FARM_PORT=8187 ./dist/llama-rpc-farm
```

## Runtime data

Packaged builds never write into the bundle. On first run they create:
- Linux: `~/.config/llama-rpc-farm/`
- macOS: `~/Library/Application Support/LlamaRPCFarm/`
- Windows: `%APPDATA%\\LlamaRPCFarm\\`

`LLAMA_FARM_DATA_DIR` overrides this location. `LLAMA_FARM_HOST` and
`LLAMA_FARM_PORT` override the bind address/port (default `0.0.0.0:8199`).
