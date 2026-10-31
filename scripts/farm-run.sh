#!/usr/bin/env bash
# farm-run.sh — launch a generated script with the SSH password supplied
# from the encrypted vault at runtime (never baked into generated/*.sh).
#
# Usage:
#   scripts/farm-run.sh <machine-id-or-name> <generated-script> [args...]
#   scripts/farm-run.sh aibox1 generated/start-my-model-RPC.sh 8080
#
# Resolves <machine> via config/machines.yaml, decrypts its vault password
# (MACHINE_<ID>_PASSWORD_ENC in .env), exports SSH_PASS, then execs the
# script. The password lives only in that process's environment — same-user
# readable via /proc, but never on disk in plaintext and never in `ps`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

if [ "$#" -lt 2 ]; then
  echo "usage: farm-run.sh <machine-id-or-name> <generated-script> [args...]" >&2
  exit 2
fi
MACHINE="$1"; SCRIPT="$2"; shift 2

if [ ! -f "$SCRIPT" ] && [ -f "$ROOT/$SCRIPT" ]; then
  SCRIPT="$ROOT/$SCRIPT"
fi
[ -f "$SCRIPT" ] || { echo "ERROR: script not found: $SCRIPT" >&2; exit 2; }

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

SSH_PASS="$("$PY" - "$ROOT" "$MACHINE" <<'EOF'
import os, sys
root, want = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "src"))
import config_loader as cfg
import secrets_vault as vault
m = cfg.machines.get(want)
if m is None:  # try by id as well as name
    for cand in cfg.machines.all():
        if cand.get("id") == want:
            m = cand
            break
if m is None:
    print(f"unknown machine: {want}", file=sys.stderr)
    sys.exit(3)
pw = vault.get_machine_password(m)
if not pw:
    print(f"no vault password for machine: {want} (set it in Machines UI)", file=sys.stderr)
    sys.exit(4)
print(pw, end="")
EOF
)" || exit "$?"
export SSH_PASS
exec "$SCRIPT" "$@"
