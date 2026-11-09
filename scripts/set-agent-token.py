#!/usr/bin/env python3
"""Set a peer-agent Bearer token in the Fernet vault.

Usage:
  .venv/bin/python scripts/set-agent-token.py <machine-name> [token]

The token is never printed. Omit it to be prompted. The farm-wide token is
auto-managed by the app and is not settable here.
"""
import argparse
import getpass
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config_loader as cfg          # noqa: E402
import secrets_vault as vault        # noqa: E402


def _find_machine(name):
    machine = cfg.machines.get(name)
    if machine is not None:
        return machine
    for candidate in cfg.machines.all():
        if str(candidate.get("id") or "") == name:
            return candidate
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Set a Llama RPC Farm peer-agent token")
    ap.add_argument("machine", nargs="?", help="machine name or id")
    ap.add_argument("token", nargs="?", help="token (omit to prompt)")
    ap.add_argument("--clear", action="store_true", help="delete the stored token")
    args = ap.parse_args(argv)
    if not args.machine:
        ap.error("machine is required")
    token = "" if args.clear else (args.token or getpass.getpass("Agent token: ").strip())
    if not args.clear and not token:
        print("error: empty token", file=sys.stderr)
        return 2
    machine = _find_machine(args.machine)
    if machine is None:
        print(f"error: unknown machine '{args.machine}'", file=sys.stderr)
        return 2
    mid = machine.get("id") or machine.get("name") or args.machine
    vault.set_machine_agent_token(mid, "" if args.clear else token)
    print(f"agent token {'cleared' if args.clear else 'set'} for "
          f"{machine.get('name') or mid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
