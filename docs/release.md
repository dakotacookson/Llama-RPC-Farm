# Release pipelines

Every push to `main` (and manual `workflow_dispatch`) builds rolling
prereleases: an **immutable versioned release per build** plus a moving
**`main-latest`** pointer to the newest one. Historical builds are retained,
so any past build remains downloadable. There are no tag-driven release
workflows.

## Versioning

The repo-root `VERSION` file is the floor for the version line, in
`MAJOR.MINOR.PATCH` form. Each successful build automatically receives the
next patch version on that line and is published as an immutable tag:

```
release-1.0.0, release-1.0.1, release-1.0.2, ...
```

The next number is computed at build time from already-published
`release-*` tags, so a number is never reused and nothing is overwritten.
Bump `VERSION` (e.g. to `1.1.0`) to start a new minor line; the next build
becomes `release-1.1.0`. Artifact filenames and `.deb` control versions use
the bare computed version (e.g. `1.0.0`), so every download is traceable to
a release.

## Orchestration

`rolling-main.yml` is the only workflow triggered by `main` pushes. It:

1. Reads `VERSION` and computes the next `X.Y.Z` from published
   `release-*` tags.
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

1. Creates an **immutable per-build prerelease** tagged `release-<version>`
   (e.g. `release-1.0.0`) and uploads every platform asset to it, with a
   generated body: version, commit link, changelog since the previous roll,
   download names per platform, checksum note, and quick-start instructions.
   Older versioned releases are never deleted, so every historical build
   stays downloadable.
2. Recreates `main-latest` as a **moving pointer** to the newest build. It
   carries no assets; its body links to the newest `release-<version>`
   release. This keeps `https://github.com/<repo>/releases/tag/main-latest`
   stable without duplicating artifacts on the pointer.

Because the publish step is gated on `needs: [linux, macos, windows]` and
the workflow serializes on the `rolling-main` concurrency group, a failed
build leaves the previous release untouched — no partial releases.

## First-run automation

Packaged binaries self-register a login service (systemd user unit,
launchd LaunchAgent, or Windows Scheduled Task) and the Windows agent adds
inbound rules for TCP 8198 + UDP 8199. Unpaired agents open a bounded
pairing window and are enrolled by the app with its farm token. Opt out with
`LLAMA_FARM_NO_SERVICE=1` / `PEER_AGENT_NO_SERVICE=1`.

## Signing & notarization

Builds are signed automatically when the repository has the signing secrets.
Without them the workflow still builds and publishes **unsigned** artifacts.

### macOS (removes Gatekeeper warnings)

Required secrets:
- `MACOS_CERT_P12_BASE64` — base64 of a `.p12` containing **Developer ID Application** and **Developer ID Installer** certificates
- `MACOS_CERT_PASSWORD`
- `MACOS_APP_SIGN_ID` — e.g. `Developer ID Application: Your Name (TEAMID)`
- `MACOS_INSTALLER_SIGN_ID` — e.g. `Developer ID Installer: Your Name (TEAMID)`
- `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD` — for `notarytool`

The workflow signs the binaries with the hardened runtime
(`packaging/entitlements.plist`), signs the `.pkg` with `productsign`,
submits it to `notarytool --wait`, and staples the ticket.

Requires a paid Apple Developer Program membership.

### Windows (reduces/removes SmartScreen warnings)

Required secrets:
- `WINDOWS_CERT_PFX_BASE64` — base64 of a code-signing `.pfx`
- `WINDOWS_CERT_PASSWORD`

The workflow signs the inner `.exe` files and the Inno Setup installers with
`signtool` (SHA-256 + timestamp). A new standard OV certificate may still
warn until SmartScreen reputation accrues; an **EV** certificate or
Microsoft Trusted Signing improves this immediately. HSM/cloud signing
(Azure Trusted Signing, DigiCert KeyLocker) needs a different signtool
invocation than the `.pfx` path used here.

Without the two secrets the build is unsigned and SmartScreen shows
"Windows protected your PC". See [windows-signing.md](windows-signing.md)
for the safe "More info → Run anyway" path, hash verification, and the
certificate options (including free SignPath signing for open source).

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
