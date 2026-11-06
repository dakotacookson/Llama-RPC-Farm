"""Typed custom Windows .bat runtime start/stop (one-off runtimes)."""
import os
import subprocess
import time
from pathlib import Path

from .config import AgentConfig


class AgentCustomError(RuntimeError):
    pass


def _pid_path(config: AgentConfig, port: int) -> Path:
    base = Path(config.state_dir).expanduser() if config.state_dir else (
        Path.home() / ".cache" / "llama-wrapper")
    return base / f"custom.{int(port)}.pid"


def _port_pid(port: int):
    ps = ("$p=(Get-NetTCPConnection -LocalPort " + str(int(port))
          + " -State Listen -ErrorAction SilentlyContinue | "
          "Select-Object -First 1).OwningProcess; if($p){[int]$p}")
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (proc.stdout or "").strip().splitlines()
    try:
        return int(lines[-1]) if lines else None
    except ValueError:
        return None


def _kill_tree(pid: int):
    try:
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                       capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pass


def stop(config: AgentConfig, port: int, state_key=None) -> dict:
    if os.name != "nt":
        raise AgentCustomError("custom runtime is Windows-only")
    port = int(port)
    pids = set()
    owner = _port_pid(port)
    if owner:
        pids.add(owner)
    try:
        pids.add(int(_pid_path(config, port).read_text().strip()))
    except (OSError, ValueError):
        pass
    for pid in pids:
        _kill_tree(pid)
    deadline = time.time() + 5
    while time.time() < deadline:
        if _port_pid(port) is None:
            break
        time.sleep(0.2)
    owner = _port_pid(port)
    if owner:
        _kill_tree(owner)
    try:
        _pid_path(config, port).unlink()
    except OSError:
        pass
    return {"port": port, "killed": _port_pid(port) is None,
            "pids": sorted(pids)}


def start(config: AgentConfig, bat: str, port: int, log=None,
          workdir=None) -> dict:
    if os.name != "nt":
        raise AgentCustomError("custom runtime is Windows-only")
    bat = str(bat or "").strip()
    if not bat or not os.path.isabs(bat) or not bat.lower().endswith(".bat"):
        raise AgentCustomError("bat must be an absolute .bat path")
    if not os.path.isfile(bat):
        raise AgentCustomError("bat file not found")
    try:
        port = int(port)
    except (TypeError, ValueError) as e:
        raise AgentCustomError("port must be an integer") from e
    if not 1 <= port <= 65535:
        raise AgentCustomError("port must be in 1..65535")
    stop(config, port)
    if log:
        logp = Path(log)
    else:
        logp = Path.home() / f"custom-{port}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    if workdir:
        cwd = str(workdir)
    else:
        cwd = str(Path(bat).parent)
    logf = open(logp, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            ["cmd.exe", "/c", bat], cwd=cwd,
            stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, close_fds=True,
            creationflags=(subprocess.DETACHED_PROCESS
                           | subprocess.CREATE_NEW_PROCESS_GROUP))
    except OSError as e:
        logf.close()
        raise AgentCustomError(f"failed to start bat: {e}") from e
    finally:
        try:
            logf.close()
        except Exception:
            pass
    pidp = _pid_path(config, port)
    pidp.parent.mkdir(parents=True, exist_ok=True)
    pidp.write_text(str(proc.pid), encoding="utf-8")
    return {"pid": proc.pid, "port": port, "log": str(logp)}
