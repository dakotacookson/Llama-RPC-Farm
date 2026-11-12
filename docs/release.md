# Release pipelines

Every push to `main` (and manual `workflow_dispatch`) builds rolling
releases: an **immutable versioned release per build** plus a moving
**`main-latest`** prerelease pointer to the newest one. The newest
versioned build is also marked as the repository's **latest full release**.
Historical builds are retained, so any past build remains downloadable.
There are no tag-driven release workflows.

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

Every build job runs `scripts/stamp_version.py` with the computed version
before packaging. That writes the same `X.Y.Z` into `VERSION`,
`peer_agent/pyproject.toml`, and `peer_agent/llama_peer_agent/__init__.py`.
The app does **not** use local source versions to decide peer updates: it
live-checks GitHub releases and uses the highest published `release-*` tag.

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
| Linux | `.deb` | `.deb` + `llama-peer-agent-linux-x86_64` |
| macOS | `.pkg` | `.pkg` + `llama-peer-agent-macos-arm64` |
| Windows | `setup.exe` | `setup.exe` + `llama-peer-agent-windows-x64.exe` |

The Linux job also publishes the universal agent wheel
`llama_rpc_farm_agent-<version>-py3-none-any.whl` and the app source wheel
`llama_rpc_farm-<version>-py3-none-any.whl`. The app wheel lets source
installs update in place even when they are not git checkouts. Each
platform also emits a SHA-256 checksum file.

## Publishing

The publish job runs once, after all builds pass:

1. Creates an **immutable per-build release** tagged `release-<version>`
   (e.g. `release-1.0.0`) and uploads every platform asset to it, with a
   generated body: version, commit link, changelog since the previous roll,
   download names per platform, checksum note, and quick-start instructions.
   Older versioned releases are never deleted, so every historical build
   stays downloadable.
2. Recreates `main-latest` as a **moving prerelease pointer** to the newest
   build. It carries no assets; its body links to the newest
   `release-<version>` release. This keeps
   `https://github.com/<repo>/releases/tag/main-latest` stable without
   duplicating artifacts on the pointer.
3. Marks the newest versioned `release-<version>` as the repository's
   latest full release (`prerelease=false`, `make_latest=true`).

Because the publish step is gated on `needs: [linux, macos, windows]` and
the workflow serializes on the `rolling-main` concurrency group, a failed
build leaves the previous release untouched — no partial releases.

## App source updates

Source installs download the app source wheel from the newest release and
copy its tracked tree (`src`, `ui`, `peer_agent`, `scripts`, `packaging`,
and the top-level launcher files) over the install. Git checkouts keep the
dirty-tree and newer-than-release guards. Releases without the wheel asset
fall back to GitHub's immutable source archive.

## Agent updates

The app resolves the newest release that carries the machine's raw agent
asset (or the universal wheel). Standalone packaged agents update through
`/agent/update-binary`; pip-installed agents update through
`/agent/update` with the release wheel. The installed release is recorded
per machine so an asset whose internal build number lags the release tag
does not create an update loop.

## Component update detection

A rolling release is produced for every `main` push, but the app and the
peer agent usually did not both change. Comparing each component to the
release *tag* made every release look like an update for both sides.

Each release therefore publishes `release-manifest.json`:

```json
{ "release_version": "2.0.0", "app_commit": "…", "agent_commit": "…" }
```

The `prepare` job computes those with
`git log -1 --format=%H -- <component paths>` (full-history checkout):

- **app** — the last commit touching `src`, `ui`, `packaging`, `scripts`,
  `requirements.txt`, the run/install/restart scripts, or `.github/workflows`
- **agent** — the last commit touching `peer_agent/`

`scripts/stamp_build_info.py` bakes those commits into generated,
git-ignored `build_info` modules. The app compares its baked `APP_COMMIT`
to the manifest; the peer agent reports `build_commit` from `/version` and
the master compares that to the manifest's `agent_commit`. When the
commits match, the component is **on latest even if the release tag is
newer**; an app-only release therefore never prompts agents, and an
agent-only release never prompts the app. Docs/tests-only pushes do not
change the app commit.

Older releases/agents without a manifest or `build_commit` fall back to
the previous tag comparison until they are updated once.

## First-run automation

Packaged binaries self-register a login service (systemd user unit,
launchd LaunchAgent, or Windows Scheduled Task) and the Windows agent adds
inbound rules for TCP 8198 + UDP 8199. Unpaired agents open a bounded
pairing window and are enrolled by the app with its auto-managed farm
token (stored only in the encrypted vault; there is no Settings override).
Opt out with
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
python scripts/build_app_wheel.py --out dist
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
