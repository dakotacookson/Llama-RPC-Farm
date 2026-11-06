# Release pipelines

Releases are rolling **`main-latest`** prereleases built automatically on
every push to `main` (and manually via `workflow_dispatch`). There are no
tag-driven release workflows.

## Versioning

The repo-root `VERSION` file is the single source of truth for the base
version. Builds append the `main` suffix and short commit:

```
<VERSION>-main.<short-sha>      e.g. 0.1.0-main.100de53
```

Bump `VERSION` when you want a new base version. Artifact filenames and
`.deb` control versions use the computed string, so every download is
traceable to a commit.

## Orchestration

`rolling-main.yml` is the only workflow triggered by `main` pushes. It:

1. Reads `VERSION` and computes `<VERSION>-main.<short-sha>`.
2. Runs three platform **build jobs** in parallel in the same workflow:
   - `linux` — linux-x64, also runs the pre-release checks
     (py_compile, `node --check`, `bash -n`, pytest)
   - `macos` — macos-arm64
   - `windows` — windows-x64
3. Runs a **single** publish job only after **all three** builds succeed.

Each build job produces both the app and the peer agent:

| Platform | App artifacts | Agent artifacts |
|---|---|---|
| Linux | `.deb` | `.deb` |
| macOS | `.pkg` | `.pkg` |
| Windows | `setup.exe` | `setup.exe` |

Each platform also emits a SHA-256 checksum file.

## Publishing

The publish job runs once, after all builds pass:

1. Deletes any previous `main-latest` release (so stale artifacts from old
   commits never accumulate).
2. Moves the `main-latest` tag to the current `main` commit.
3. Creates a fresh `main-latest` prerelease with all platform assets and a
   generated body: version, commit link, changelog since the previous roll,
   download names per platform, checksum note, and quick-start instructions.

Because the publish step is gated on `needs: [linux, macos, windows]` and
the workflow serializes on the `rolling-main` concurrency group, a failed
build leaves the previous release untouched — no partial releases.

## First-run automation

Packaged binaries self-register a login service (systemd user unit,
launchd LaunchAgent, or Windows Scheduled Task) and the Windows agent adds
inbound rules for TCP 8198 + UDP 8199. Unpaired agents open a bounded
pairing window and are enrolled by the app with its farm token. Opt out with
`LLAMA_FARM_NO_SERVICE=1` / `PEER_AGENT_NO_SERVICE=1`.

## Local release smoke test

```bash
python -m pip wheel --no-deps -w agent-wheel ./peer_agent
pyinstaller --noconfirm --clean packaging/app.spec
LLAMA_FARM_DATA_DIR=/tmp/llama-farm-data LLAMA_FARM_PORT=8187 ./dist/llama-rpc-farm
```

## Runtime data

Packaged builds never write into the bundle. On first run they create the
platform data dir plus `config/`, `config/models/`, `config/usage/`,
`generated/`, and minimal default config files (`settings.yaml`,
`machines.yaml`, `groups.yaml`, `swap_groups.yaml`, `stores.yaml`,
`swap.yaml`). The standalone peer agent creates its log/state dirs at
startup. On first run they create:
- Linux: `~/.config/llama-rpc-farm/`
- macOS: `~/Library/Application Support/LlamaRPCFarm/`
- Windows: `%APPDATA%\\LlamaRPCFarm\\`

`LLAMA_FARM_DATA_DIR` overrides this location. `LLAMA_FARM_HOST` and
`LLAMA_FARM_PORT` override the bind address/port (default `0.0.0.0:8199`).
