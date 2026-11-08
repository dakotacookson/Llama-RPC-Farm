# Secrets model

## Where secrets live

ONE git-ignored `.env` at the project root. Nothing sensitive lives in
YAML, generated scripts, API responses, or git.

| Key | Meaning |
|---|---|
| `FARM_SECRET_KEY` + `FARM_SALT` | Vault key material (auto-generated on first run) |
| `HF_TOKEN_ENC` | Hugging Face token, Fernet-encrypted |
| `MACHINE_<ID>_AGENT_TOKEN_ENC` | Per-machine peer-agent token, Fernet-encrypted (id uppercased, non-alnum → `_`) |
| `AGENT_TOKEN_ENC` | Farm-wide agent token, auto-generated and managed by the app (no Settings override) |

Crypto (`src/secrets_vault.py`): SHA-256+salt fingerprints
(`hash_password`/`verify_password`) for verification-only cases; Fernet
(AES+HMAC) for values the app must use. Values are decrypted in memory at
the point of use only — agent call, HF call, script sidecar write — and
never written back or sent to the browser (`password_set`/`hf_token_set`
flags only).

One-time migration from the legacy plaintext YAML field:

```bash
cp .env.example .env
.venv/bin/python scripts/migrate_secrets_to_env.py
```

Machine records no longer carry login passwords; only the agent token is
stored (encrypted in `.env`).

## Launch paths (agent-only)

1. **Generated scripts** — `generated/*.sh` contain no passwords. They call
   each peer's agent over HTTP with `AGENT_TOKEN_<NAME>` values supplied by
   the runtime sidecar.
2. **llama-swap launches** — push writes a `swap-secrets.env` sidecar
   (0600, host-side only) next to the scripts; each model cmd sources it
   before exec. The pushed `config.yaml` never holds secrets.
3. **Tuner trials** — use the same agent endpoints for remote work.

There is no password transport: `sshpass`, `paramiko`, and `$SSH_PASS` were
removed. Agents are installed from their own `.deb`/`.pkg`/`setup.exe`.

## Injection defenses (enforced by tests)

- Model flag values are `shlex.quote`d when `build_llama_command` renders
  them — `tests/test_shell_injection.py` parametrizes over
  `;`/backtick/`&&` payloads and fails on any unquoted metacharacter.
- Machine IPs/usernames/paths in generated scripts are validated/quoted —
  `tests/test_script_injection.py`.
- Keep it that way: any new place a user-controlled string enters a shell
  command needs the same treatment.

## Operations

- Rotate: edit/delete per-machine keys or the HF token via the app, or use
  `secrets_vault` (`set_machine_agent_token` / `set_hf_token` /
  `set_env_value`). The farm token (`AGENT_TOKEN_ENC`) is app-managed:
  regenerated automatically if missing, never exposed in Settings.
- `.env` must be `0600`-ish protected; it is NOT tracked by git.
- Before distributing the repo: rotate/remove real credentials and tokens.