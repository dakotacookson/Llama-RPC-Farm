"""Typed llama-swap service control for the agent."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import files, processes
from .config import AgentConfig

SERVICE = "llama-swap"
DAEMON_ARGS = ("--watch-config",)
_ALLOWED = {"llama-swap", "llama-server", "ggml-rpc-server",
            "llama-swap.exe", "llama-server.exe",
            "ggml-rpc-server.exe"}


class AgentServiceError(RuntimeError):
    pass


def _name(name) -> str:
    name = str(name or "").strip()
    if name != SERVICE:
        raise AgentServiceError("unsupported service")
    return name


def _binary(binary) -> str:
    binary = str(binary or "").strip()
    if not binary:
        raise AgentServiceError("binary is required")
    base = os.path.basename(binary)
    if base not in _ALLOWED:
        raise AgentServiceError("binary is not allowed")
    if os.path.isabs(binary):
        if not os.path.isfile(binary):
            raise AgentServiceError("binary not found")
        return binary
    found = shutil.which(binary)
    if not found:
        raise AgentServiceError("binary not found on PATH")
    return found


def _unit(binary: str, config_path: str, port: int) -> str:
    return (
        "[Unit]\n"
        "Description=llama-swap (managed by Llama RPC Farm)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        f'ExecStart={binary} --config "{config_path}" --listen :{port} '
        + " ".join(DAEMON_ARGS) + '\n'
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _popen(config: AgentConfig, binary: str, config_path: str, port: int):
    log = Path(config.log_dir).expanduser() / "llama-swap.log" if config.log_dir \
        else Path.home() / "llama-swap.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "stdout": open(log, "w", encoding="utf-8"),
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [binary, "--config", config_path, "--listen", f":{port}",
         *DAEMON_ARGS], **kwargs)
    return proc.pid


def start(config: AgentConfig, name, binary, config_path, port) -> dict:
    _name(name)
    binary = _binary(binary)
    config_path = str(files._path(config, config_path))
    try:
        port = int(port)
    except (TypeError, ValueError) as e:
        raise AgentServiceError("port must be an integer") from e
    if not 1 <= port <= 65535:
        raise AgentServiceError("port must be in 1..65535")
    if processes.list_swap_processes():
        return {"method": "already-running", "pid": None,
                "running": True}
    if os.name != "nt" and sys.platform != "darwin":
        unit = Path.home() / ".config" / "systemd" / "user" / "llama-swap.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(_unit(binary, config_path, port), encoding="utf-8")
        try:
            subprocess.run(["systemctl", "--user", "daemon-reload"],
                           capture_output=True, timeout=30)
            proc = subprocess.run(
                ["systemctl", "--user", "enable", "--now", "llama-swap"],
                capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                return {"method": "systemd", "pid": None, "running": True}
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        pid = _popen(config, binary, config_path, port)
    except OSError as e:
        raise AgentServiceError(f"failed to start llama-swap: {e}") from e
    time.sleep(1)
    return {"method": "detached-process", "pid": pid,
            "running": True}


def stop(config: AgentConfig, name) -> dict:
    _name(name)
    if os.name != "nt" and sys.platform != "darwin":
        try:
            subprocess.run(["systemctl", "--user", "stop", "llama-swap"],
                           capture_output=True, timeout=20)
            subprocess.run(["systemctl", "--user", "disable", "llama-swap"],
                           capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            pass
    return processes.stop_swap(config)


def status(config: AgentConfig, name) -> dict:
    _name(name)
    procs = processes.list_swap_processes()
    return {"service": SERVICE, "running": bool(procs),
            "process": bool(procs), "pids": [p["pid"] for p in procs],
            "ports": [p["port"] for p in procs if p.get("port")]}


def validate(config: AgentConfig, binary, config_path) -> dict:
    binary = _binary(binary)
    config_path = str(files._path(config, config_path))
    try:
        proc = subprocess.run(
            [binary, "--config", config_path, "--validate"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise AgentServiceError(f"validate failed: {e}") from e
    return {"returncode": proc.returncode,
            "output": ((proc.stdout or "") + (proc.stderr or ""))[-800:]}


def which(name) -> dict:
    name = str(name or "").strip()
    if name not in _ALLOWED:
        raise AgentServiceError("binary is not allowed")
    return {"name": name, "path": shutil.which(name)}
