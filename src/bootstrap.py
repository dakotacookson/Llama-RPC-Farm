#!/usr/bin/env python3
# bootstrap.py — installer entry (design §1/§3/§4).
#
# Runs AFTER the shell entry (install.sh / install.ps1 / install.bat) has
# provisioned uv + a python venv. Owns: detect -> probe -> plan -> execute,
# per role (main / client), with --verbose teeing to install.log and the
# report-setup subcommand printing a redacted detection summary for bug
# reports. Windows entry runs `python bootstrap.py ...` (install.bat).
#
# Role flows (§3): main = app venv + llama-swap + llama.cpp + (vLLM opt)
# + settings prefill; client = engine runtime only (llama.cpp / vLLM+ray),
# NO app, NO llama-swap, NO settings writes.

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from installer.env import RealFS, RealRunCmd, detect_system_info   # noqa: E402
from installer.probe import probe                                   # noqa: E402
from installer.execute import Executor                              # noqa: E402

LOG_DIR = "~/.llama-rpc-farm"
LOG_FILE = os.path.join(LOG_DIR, "install.log")

# Pinned requirements (§2). Keep in one place; upgrade bumps these.
DEFAULT_PINS = {
    "llama.cpp": {"version": "b10934"},
    "llama-swap": {"version": "255"},
    "vllm": {"version": "0.30.0"},
}


class TeeLog:
    """print-like logger: stdout (--verbose) + install.log tee (§4),
    every line timestamped with a per-step-ish PASS/FAIL/SKIP prefix kept
    from the caller (probe/executor already emit SKIP/FAIL)."""

    def __init__(self, verbose=False, path=None):
        self.verbose = verbose
        self.path = os.path.expanduser(path) if path else None
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        if self.verbose:
            print(msg)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def cmd_install(args) -> int:
    log = TeeLog(verbose=args.verbose or args.report,
                 path=args.log_file)
    run, fs = RealRunCmd(), RealFS()
    info = detect_system_info(run)
    log(f"DETECT os={info.os} arch={info.arch} gpu={info.gpu_vendor} "
        f"driver={info.driver_version} glibc={info.glibc} wsl={info.wsl}")
    if args.verbose:
        log(f"vram_gb={info.vram_gb}")

    # GPU-driver hard rule (§4): we NEVER install drivers. If NVIDIA is
    # expected but nvidia-smi is absent the executor refuses below via
    # notes; nothing system-level is ever touched here.
    manifest = _read_manifest(log)
    plan = probe(info, fs, run, role=args.role, pinned=DEFAULT_PINS,
                 with_vllm=not args.no_vllm,
                 with_vllm_metal=args.with_vllm_metal,
                 force_settings=args.force_settings,
                 manifest=manifest)
    if args.dry_run:
        log("dry-run plan:")
        for s in plan:
            log(f"  {s}")
        return 0
    ex = Executor(log=log, offline=args.offline, info=info,
                  manifest_path=os.path.join(LOG_DIR, "installer-state.yaml"))
    summary = ex.execute(plan)
    ex.write_manifest(summary, detected=_detected_block(info))
    if summary["failures"]:
        for env in summary["failures"]:
            log("FAIL " + env["error"])
        print(json.dumps({"ok": False, "failures": summary["failures"]},
                         indent=2))
        return 1
    if not summary["installed"] and not summary["repaired"]:
        log("Nothing to do — all tools present and up to date.")
    else:
        log(f"DONE installed={summary['installed']} "
            f"repaired={summary['repaired']} skipped={summary['skipped']}")
    if args.role == "main":
        log("Next: start the app with ./run.sh (POSIX) or run.bat (Windows); "
            "binary paths are prefilled in config/.")
    return 0


def _redact(p: str) -> str:
    """Replace the home prefix with ~ so no username ever appears in a
    public report (design §4 redaction rule)."""
    home = os.path.expanduser("~")
    if p and p.startswith(home + os.sep):
        return "~" + p[len(home):]
    return p


def cmd_report_setup(args) -> int:
    """Redacted detection summary for pasting into a GitHub issue (§4)."""
    run, fs = RealRunCmd(), RealFS()
    info = detect_system_info(run)
    lines = [
        "llama-rpc-farm setup report",
        f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"os: {info.os}  arch: {info.arch}  wsl: {info.wsl}",
        f"gpu: {info.gpu_vendor}  driver: {info.driver_version}",
        f"glibc: {'.'.join(map(str, info.glibc)) if info.glibc else 'n/a'}",
        f"vram_gb: {info.vram_gb}",
    ]
    for tool, candidates in (
            ("llama-swap", ["~/.local/bin/llama-swap"]),
            ("llama.cpp", ["~/llama.cpp/bin/llama-server",
                           "~/llama.cpp/build/bin/llama-server"]),
            ("vllm", ["~/vllm-env/bin/vllm"])):
        found = None
        for c in candidates:
            p = os.path.expanduser(c)
            if os.path.isfile(p):
                found = p
                break
        lines.append(f"{tool}: "
                     f"{'FOUND ' + _redact(found) if found else 'not found'}")
    print("\n".join(lines))
    return 0


def _read_manifest(log):
    path = os.path.expanduser(os.path.join(LOG_DIR, "installer-state.yaml"))
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _detected_block(info):
    return {"os": info.os, "arch": info.arch, "gpu": info.gpu_vendor,
            "driver": info.driver_version,
            "vram_gb": info.vram_gb, "wsl": info.wsl}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bootstrap.py",
                                description="Llama-RPC-Farm installer")
    sub = p.add_subparsers(dest="cmd")
    for name in ("install", "upgrade"):
        sp = sub.add_parser(name)
        sp.add_argument("--role", choices=["main", "client"], default="main")
        sp.add_argument("--with-vllm", dest="no_vllm", action="store_false",
                        default=True, help="install vLLM venv (default on)")
        sp.add_argument("--no-vllm", dest="no_vllm", action="store_true",
                        help="skip the vLLM venv")
        sp.add_argument("--with-vllm-metal", action="store_true",
                        help="opt-in vLLM-Metal plugin (macOS)")
        sp.add_argument("--unattended", action="store_true")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--force-settings", action="store_true",
                        help="overwrite valid settings paths (§6 c)")
        sp.add_argument("--offline", action="store_true",
                        help="install from the bundled wheelhouse/ (§1)")
        sp.add_argument("--verbose", "-v", action="store_true")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--log-file", default=None,
                        help="override install.log path")
        sp.set_defaults(func=cmd_install, report=False)
    rs = sub.add_parser("report-setup",
                        help="print a redacted detection summary")
    rs.set_defaults(func=cmd_report_setup, verbose=False, report=False)
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())