"""Self-service install for the peer agent: login service + firewall rules.

Packaged agents call `maybe_auto_install()` on start. Registration only —
the agent process is already running, so we never start a duplicate. The
service picks the token from the agent.env file written by pairing/install.
"""
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "llama-peer-agent"
LABEL = "com.llamarpc.farm.agent"
WIN_TASK = "LlamaPeerAgent"
AGENT_PORT = 8198
DISCOVERY_PORT = 8199


def _log(msg: str) -> None:
    print(f"[selfservice] {msg}")


def _run(args, timeout=30):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"{' '.join(map(str, args))}: {e}")
        return None


def _command():
    """(executable, [args]) for services/tasks."""
    if getattr(sys, "frozen", False):
        return sys.executable, ["--host", "0.0.0.0",
                                "--port", str(AGENT_PORT)]
    return sys.executable, ["-m", "llama_peer_agent", "--host", "0.0.0.0",
                            "--port", str(AGENT_PORT)]


def _linux_unit() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME")
                or (Path.home() / ".config"))
    return base / "systemd" / "user" / f"{APP_NAME}.service"


def _linux_install() -> None:
    unit = _linux_unit()
    unit.parent.mkdir(parents=True, exist_ok=True)
    exe, args = _command()
    cmd = " ".join([f'"{exe}"'] + [f'"{a}"' if " " in a else a for a in args])
    unit.write_text(f"""\
[Unit]
Description=Llama RPC Farm peer agent
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={cmd}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
""", encoding="utf-8")
    if not shutil.which("systemctl"):
        _log("systemctl not found; unit written, not enabled")
        return
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", f"{APP_NAME}.service"])
    user = os.environ.get("USER") or ""
    if user:
        _run(["loginctl", "enable-linger", user])
    _log(f"systemd user service registered (starts at next login): {unit}")


def _linux_uninstall() -> None:
    if shutil.which("systemctl"):
        _run(["systemctl", "--user", "disable", "--now",
              f"{APP_NAME}.service"])
    try:
        _linux_unit().unlink()
    except OSError:
        pass


def _mac_plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _mac_install() -> None:
    logs = Path.home() / "Library" / "Logs" / "llama-peer-agent"
    logs.mkdir(parents=True, exist_ok=True)
    exe, args = _command()
    plist = _mac_plist()
    plist.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "Label": LABEL,
        "ProgramArguments": [exe] + args,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logs / "agent.log"),
        "StandardErrorPath": str(logs / "agent.err.log"),
    }
    with open(plist, "wb") as f:
        plistlib.dump(data, f)
    _log(f"launchd agent registered (starts at next login): {plist}")


def _mac_uninstall() -> None:
    uid = os.getuid()
    _run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"])
    try:
        _mac_plist().unlink()
    except OSError:
        pass


def _windows_firewall() -> None:
    netsh = shutil.which("netsh")
    if not netsh:
        return
    for proto, port in (("TCP", AGENT_PORT), ("UDP", DISCOVERY_PORT)):
        name = f"Llama RPC Farm Agent {proto} {port}"
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
    _windows_firewall()
    schtasks = shutil.which("schtasks")
    if not schtasks:
        _log("schtasks not found; no auto-start task created")
        return
    exe, args = _command()
    tr = f'"{exe}" ' + " ".join(args)
    _run([schtasks, "/Create", "/SC", "ONLOGON", "/RL", "HIGHEST",
          "/TN", WIN_TASK, "/TR", tr, "/F"])
    _log(f"scheduled task registered (starts at next login): {WIN_TASK}")


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
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--uninstall-service" in argv:
        uninstall()
        raise SystemExit(0)
    if "--install-service" in argv:
        install()
        raise SystemExit(0)
    disabled = (os.environ.get("PEER_AGENT_NO_SERVICE") or "").lower() in (
        "1", "true", "yes", "on")
    if disabled:
        return
    if getattr(sys, "frozen", False):
        install()
