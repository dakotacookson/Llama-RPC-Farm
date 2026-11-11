"""Typed custom Windows .bat runtime start/stop (one-off runtimes)."""
import ntpath
import os
import re
import subprocess
import time
from pathlib import Path

from .config import AgentConfig

_MATCH_RE = re.compile(r"^[A-Za-z0-9_.: \\-]+$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.: \\*?+\-]+$")


class AgentCustomError(RuntimeError):
    pass


def _state_base(config: AgentConfig) -> Path:
    return (Path(config.state_dir).expanduser() if config.state_dir
            else Path.home() / ".cache" / "llama-wrapper")


def _pid_path(config: AgentConfig, port: int) -> Path:
    return _state_base(config) / f"custom.{int(port)}.pid"


def _bat_path(config: AgentConfig, port: int) -> Path:
    return _state_base(config) / f"custom.{int(port)}.bat"


def _derive_stop_bat(bat: str) -> str:
    """Sibling STOP-<core>.bat: run-iq3_s.bat -> STOP-iq3_s.bat."""
    bat = str(bat or "").strip()
    if not bat:
        return ""
    folder, name = ntpath.split(bat)
    stem, ext = ntpath.splitext(name)
    core = re.sub(r"^(run|start)-", "", stem, flags=re.IGNORECASE) or stem
    return ntpath.join(folder, f"STOP-{core}{ext or '.bat'}")


def _clean_terms(values, pattern=False):
    out = []
    rx = _NAME_RE if pattern else _MATCH_RE
    for value in values or []:
        value = str(value or "").strip()
        if value and rx.fullmatch(value):
            out.append(value)
    return out


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


def _matching_pids(match_subs, name_pats) -> set:
    """PIDs whose command line contains a stop_match substring, or whose
    process name matches a stop_names pattern."""
    match_subs = _clean_terms(match_subs)
    name_pats = _clean_terms(name_pats, pattern=True)
    conds = [f"(' ' + $_.CommandLine + ' ') -like '*{s}*'"
             for s in match_subs]
    conds += [f"$_.Name -like '{p}'" for p in name_pats]
    if not conds:
        return set()
    ps = ("$p = Get-CimInstance Win32_Process | Where-Object { "
          + " -or ".join(conds)
          + " } | Select-Object -ExpandProperty ProcessId; if($p){$p}")
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=25)
    except (OSError, subprocess.SubprocessError):
        return set()
    pids = set()
    for line in (proc.stdout or "").strip().splitlines():
        try:
            pids.add(int(line.strip()))
        except ValueError:
            pass
    return pids


def _kill_tree(pid: int):
    try:
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                       capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pass


def stop(config: AgentConfig, port: int, state_key=None,
         stop_bat=None, stop_match=None, stop_names=None) -> dict:
    """Stop a custom runtime, preferring its STOP-*.bat.

    The bat is run first; then port/PID and optional process-pattern kills
    act as a safety net (some stacks detach, respawn, or use a name that
    the bat's filter misses).
    """
    if os.name != "nt":
        raise AgentCustomError("custom runtime is Windows-only")
    port = int(port)
    bat_state = _bat_path(config, port)
    stop_bat = str(stop_bat or "").strip()
    if not stop_bat:
        try:
            start_bat = bat_state.read_text(encoding="utf-8").strip()
        except OSError:
            start_bat = ""
        stop_bat = _derive_stop_bat(start_bat)
    stop_bat_ran = False
    stop_output = ""
    if stop_bat and os.path.isfile(stop_bat) \
            and stop_bat.lower().endswith(".bat"):
        try:
            proc = subprocess.run(
                ["cmd.exe", "/c", stop_bat],
                cwd=str(Path(stop_bat).parent),
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=45)
            stop_bat_ran = True
            stop_output = ((proc.stdout or "") + (proc.stderr or ""))[-500:]
        except (OSError, subprocess.SubprocessError) as e:
            stop_output = str(e)
        deadline = time.time() + 10
        while time.time() < deadline:
            if _port_pid(port) is None:
                break
            time.sleep(0.25)

    pids = set()
    owner = _port_pid(port)
    if owner:
        pids.add(owner)
    try:
        pids.add(int(_pid_path(config, port).read_text().strip()))
    except (OSError, ValueError):
        pass
    pids.update(_matching_pids(stop_match, stop_names))
    for pid in pids:
        _kill_tree(pid)
    deadline = time.time() + 8
    while time.time() < deadline:
        if _port_pid(port) is None:
            break
        time.sleep(0.2)
    owner = _port_pid(port)
    if owner:
        _kill_tree(owner)
        pids.add(owner)
    for path in (_pid_path(config, port), bat_state):
        try:
            path.unlink()
        except OSError:
            pass
    return {"port": port, "killed": _port_pid(port) is None,
            "pids": sorted(pids), "stop_bat": stop_bat,
            "stop_bat_ran": stop_bat_ran,
            **({"stop_output": stop_output} if stop_output else {})}


def start(config: AgentConfig, bat: str, port: int, log=None,
          workdir=None, stop_bat=None, stop_match=None,
          stop_names=None) -> dict:
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
    stop(config, port, stop_bat=stop_bat or _derive_stop_bat(bat),
         stop_match=stop_match, stop_names=stop_names)
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
    bat_state = _bat_path(config, port)
    bat_state.write_text(bat, encoding="utf-8")
    return {"pid": proc.pid, "port": port, "log": str(logp)}
