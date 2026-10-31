#!/usr/bin/env python3
"""One-time migration: move plaintext secrets from YAML configs into the
single encrypted .env vault (Fernet, key in .env).

Moves:
  config/machines.yaml[*].password  -> .env MACHINE_<ID>_PASSWORD_ENC
  config/settings.yaml:hf_token     -> .env HF_TOKEN_ENC

Then clears the YAML fields so nothing sensitive stays in configs.
Idempotent: skips machines whose password is already empty/vaulted.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import yaml  # noqa: E402
import secrets_vault as vault  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    moved_machines, moved_hf = 0, 0

    mpath = os.path.join(BASE, "config", "machines.yaml")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        machines = data.get("machines") or []
        for m in machines:
            pw = (m.get("password") or "")
            if pw and not str(pw).startswith("env:"):
                mid = m.get("id") or m.get("name")
                vault.set_machine_password(mid, pw)
                m["password"] = ""  # cleared: vault is the source of truth
                moved_machines += 1
        tmp = mpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, mpath)

    spath = os.path.join(BASE, "config", "settings.yaml")
    if os.path.exists(spath):
        with open(spath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        tok = (data.get("hf_token") or "").strip()
        if tok:
            vault.set_hf_token(tok)
            data.pop("hf_token", None)
            moved_hf = 1
            tmp = spath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp, spath)

    print(f"moved {moved_machines} machine password(s), hf_token moved={bool(moved_hf)}")
    print(f"vault: {vault.ENV_PATH} (HF set={vault.hf_token_set()})")


if __name__ == "__main__":
    main()
