# Secrets model

## Where secrets live

ONE git-ignored `.env` at the project root. Nothing sensitive lives in
YAML, generated scripts, API responses, or git.

| Key | Meaning |
|---|---|
| `FARM_SECRET_KEY` + `FARM_SALT` | Vault key material (auto-generated on first run) |
| `HF_TOKEN_ENC` | Hugging Face token, Fernet-encrypted |
| `MACHINE_<ID>_PASSWORD_ENC` | Per-machine SSH password, Fernet-encrypted (id uppercased, non-alnum → `_`) |

Crypto (`src/secrets_vault.py`): SHA-256+salt fingerprints
(`hash_password`/`verify_password`) for verification-only cases; Fernet
(AES+HMAC) for values the app must use. Values are decrypted in memory at
the point of use only — SSH connect, HF call, script sidecar write — and
never written back or sent to the browser (`password_set`/`hf_token_set`
flags only).

One-time migration from the legacy plaintext YAML field:

```bash
cp .env.example .env
.venv/bin/python scripts/migrate_secrets_to_env.py
```

The legacy `password:` field in `machines.yaml` still exists in some
installs; `_inject_machine_secret` prefers the vault value and treats the
YAML field as stale. New/edited machines always write to the vault.

## The three launch paths (all vault-fed, nothing baked)

1. **Manual scripts** — generated `generated/*.sh` contain NO passwords
   and require `$SSH_PASS` in the environment:

   ```bash
   scripts/farm-run.sh aibox1 generated/start-my-model-RPC.sh 8080
   # wrapper decrypts the machine password from .env and execs
   # equivalent: SSH_PASS='...' generated/start-my-model-RPC.sh
   ```

2. **llama-swap launches** — push writes a `swap-secrets.env` sidecar
   (0600, host-side only) next to the scripts; each model cmd sources it
   before exec. The pushed `config.yaml` never holds secrets.

3. **Tuner trials** — the trial runner injects `SSH_PASS` into the
   subprocess environment (never argv, never disk).

Same rule elsewhere: the tuner bandwidth probe uses `sshpass -e`
(env-based, invisible in `ps`); live SSH is paramiko with the in-memory
vault-resolved password.

## Injection defenses (enforced by tests)

- Model flag values are `shlex.quote`d when `build_llama_command` renders
  them — `tests/test_shell_injection.py` parametrizes over
  `;`/backtick/`&&` payloads and fails on any unquoted metacharacter.
- Machine IPs/usernames/paths in generated scripts are validated/quoted —
  `tests/test_script_injection.py`.
- Keep it that way: any new place a user-controlled string enters a shell
  command needs the same treatment.

## Operations

- Rotate: edit/delete keys via the app (Machine editor → set password) or
  `secrets_vault.set_machine_password` / `set_hf_token` / `set_env_value`.
- `.env` must be `0600`-ish protected; it is NOT tracked by git.
- Before distributing the repo: rotate/remove real passwords and tokens.
  See `THIRD-PARTY-LICENSES.md` for the LGPL-2.1 (paramiko) obligations.