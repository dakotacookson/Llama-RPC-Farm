"""Self-service install: run the app at login and open the needed ports.

Packaged builds call `install()` on first start (unless LLAMA_FARM_NO_SERVICE
is set). Source checkouts can opt in with `--install-service`. Everything is
best-effort: a missing systemd/launchd/schtasks or non-admin shell logs a
warning instead of breaking the UI.
"""
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "llama-rpc-farm"
APP_LABEL = "com.llamarpc.farm.app"
WIN_TASK = "LlamaRPCFarm"
TCP_PORTS = (8199, 8090, 8080)


def _exec_start() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    script = os.path.abspath(sys.argv[0])
    return f'"{sys.executable}" "{script}"'


def _log(msg: str) -> None:
    print(f"[selfservice] {msg}")


def _run(args, timeout=30):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"{' '.join(map(str, args))}: {e}")
        return None


def _state_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home()) / "LlamaRPCFarm"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LlamaRPCFarm"
    return Path(os.environ.get("XDG_CONFIG_HOME")
                or (Path.home() / ".config")) / "llama-rpc-farm"


def _linux_unit() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME")
                or (Path.home() / ".config"))
    return base / "systemd" / "user" / f"{APP_NAME}.service"


def _linux_install() -> None:
    unit = _linux_unit()
    unit.parent.mkdir(parents=True, exist_ok=True)
    logs = _state_dir()
    logs.mkdir(parents=True, exist_ok=True)
    unit.write_text(f"""\
[Unit]
Description=Llama RPC Farm
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={_exec_start()}
Restart=on-failure
RestartSec=3
WorkingDirectory={Path.home()}

[Install]
WantedBy=default.target
""", encoding="utf-8")
    if not shutil.which("systemctl"):
        _log("systemctl not found; service file written but not enabled")
        return
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", f"{APP_NAME}.service"])
    user = os.environ.get("USER") or ""
    if user:
        _run(["loginctl", "enable-linger", user])
    _log(f"systemd user service installed: {unit}")


def _linux_uninstall() -> None:
    if shutil.which("systemctl"):
        _run(["systemctl", "--user", "disable", "--now",
              f"{APP_NAME}.service"])
    try:
        _linux_unit().unlink()
    except OSError:
        pass


def _mac_plist() -> Path:
    return (Path.home() / "Library" / "LaunchAgents"
            / f"{APP_LABEL}.plist")


def _mac_install() -> None:
    logs = Path.home() / "Library" / "Logs" / "LlamaRPCFarm"
    logs.mkdir(parents=True, exist_ok=True)
    plist = _mac_plist()
    plist.parent.mkdir(parents=True, exist_ok=True)
    args = ([sys.executable] if getattr(sys, "frozen", False)
            else [sys.executable, os.path.abspath(sys.argv[0])])
    data = {
        "Label": APP_LABEL,
        "ProgramArguments": args,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logs / "app.log"),
        "StandardErrorPath": str(logs / "app.err.log"),
    }
    with open(plist, "wb") as f:
        plistlib.dump(data, f)
    # Register only: the plist loads at next login. Do NOT bootstrap now —
    # the current process is already running and a second copy would fight
    # for :8199.
    _log(f"launchd agent installed (starts at next login): {plist}")


def _mac_uninstall() -> None:
    uid = os.getuid()
    _run(["launchctl", "bootout", f"gui/{uid}/{APP_LABEL}"])
    try:
        _mac_plist().unlink()
    except OSError:
        pass


def _windows_firewall(tcp=(), udp=(), tag="Llama RPC Farm") -> None:
    netsh = shutil.which("netsh")
    if not netsh:
        return
    rules = [("TCP", p) for p in tcp] + [("UDP", p) for p in udp]
    for proto, port in rules:
        name = f"{tag} {proto} {port}"
        check = _run([netsh, "advfirewall", "firewall", "show", "rule",
                      f"name={name}"])
        if check is not None and check.returncode == 0:
            continue
        res = _run([netsh, "advfirewall", "firewall", "add", "rule",
                    f"name={name}", "dir=in", "action=allow",
                    f"protocol={proto}", f"localport={port}"])
        if res is None or res.returncode != 0:
            _log(f"firewall rule failed (needs admin?): {name}")


def _windows_install() -> None:
    _windows_firewall(tcp=TCP_PORTS, tag="Llama RPC Farm App")
    schtasks = shutil.which("schtasks")
    if not schtasks:
        _log("schtasks not found; no auto-start task created")
        return
    cmd = _exec_start()
    # Register for next login only; do not /Run (would duplicate the
    # already-running process and fight for the port).
    _run([schtasks, "/Create", "/SC", "ONLOGON", "/RL", "HIGHEST",
          "/TN", WIN_TASK, "/TR", cmd, "/F"])
    _log(f"scheduled task installed (starts at next login): {WIN_TASK}")


def _windows_uninstall() -> None:
    schtasks = shutil.which("schtasks")
    if schtasks:
        _run([schtasks, "/Delete", "/TN", WIN_TASK, "/F"])


def install() -> None:
    if os.name == "nt":
        _windows_install()
    elif sys.platform == "darwin":
        _mac_install()
    else:
        _linux_install()


def uninstall() -> None:
    if os.name == "nt":
        _windows_uninstall()
    elif sys.platform == "darwin":
        _mac_uninstall()
    else:
        _linux_uninstall()


def maybe_auto_install(argv=None) -> None:
    """Auto-install on packaged first run; never on a source checkout."""
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--uninstall-service" in argv:
        uninstall()
        raise SystemExit(0)
    disabled = (os.environ.get("LLAMA_FARM_NO_SERVICE") or "").lower() in (
        "1", "true", "yes", "on")
    if "--install-service" in argv:
        install()
        raise SystemExit(0)
    if disabled:
        return
    if getattr(sys, "frozen", False):
        install()
