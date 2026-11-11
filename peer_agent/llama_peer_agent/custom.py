"""Typed custom-runtime start/stop scripts (any OS, any engine).

The farm does not know what an external runtime is (Strata, SGLang, ...).
It only starts a user-provided start script and, on stop, prefers a
user-provided (or sibling-derived) stop script. Port/PID/pattern kills are
a safety net for stacks that detach or ignore their stop script.
"""
import fnmatch
import ntpath
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
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


def _script_path(config: AgentConfig, port: int) -> Path:
    return _state_base(config) / f"custom.{int(port)}.script"


def _legacy_bat_path(config: AgentConfig, port: int) -> Path:
    return _state_base(config) / f"custom.{int(port)}.bat"


def _is_windows_path(path: str) -> bool:
    return os.name == "nt" or bool(re.match(r"^[A-Za-z]:[\\/]", path or ""))


def _pathmod(path: str):
    return ntpath if _is_windows_path(path) else posixpath


def _derive_stop_script(script: str) -> str:
    """Find a sibling STOP-<core> script, preferring the start extension."""
    script = str(script or "").strip()
    if not script:
        return ""
    pm = _pathmod(script)
    folder, name = pm.split(script)
    stem, ext = pm.splitext(name)
    core = re.sub(r"^(run|start|stop)-", "", stem, flags=re.IGNORECASE) or stem
    if core.lower() in ("run", "start"):
        core = ""
    mid = f"-{core}" if core else ""
    candidates = []
    for prefix in ("STOP", "stop"):
        for e in (ext, ".sh", ".bat", ""):
            cand = pm.join(folder, f"{prefix}{mid}{e}")
            if cand not in candidates:
                candidates.append(cand)
    for cand in candidates:
        try:
            if os.path.isfile(cand):
                return cand
        except OSError:
            pass
    return candidates[0] if candidates else ""


def _clean_terms(values, pattern=False):
    out = []
    rx = _NAME_RE if pattern else _MATCH_RE
    for value in values or []:
        value = str(value or "").strip()
        if value and rx.fullmatch(value):
            out.append(value)
    return out


def _port_pid(port: int):
    if os.name == "nt":
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
    for cmd in (["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                ["fuser", "-n", "tcp", str(port)]):
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        for token in re.split(r"\s+", (proc.stdout or "").strip()):
            if token.isdigit():
                return int(token)
    if shutil.which("ss"):
        try:
            proc = subprocess.run(["ss", "-ltnp"], capture_output=True,
                                  text=True, timeout=10)
            for line in (proc.stdout or "").splitlines():
                if re.search(rf":{int(port)}\s", line):
                    match = re.search(r"pid=(\d+)", line)
                    if match:
                        return int(match.group(1))
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def _matching_pids(match_subs, name_pats) -> set:
    match_subs = _clean_terms(match_subs)
    name_pats = _clean_terms(name_pats, pattern=True)
    if not match_subs and not name_pats:
        return set()
    pids = set()
    if os.name == "nt":
        conds = [f"(' ' + $_.CommandLine + ' ') -like '*{s}*'"
                 for s in match_subs]
        conds += [f"$_.Name -like '{p}'" for p in name_pats]
        ps = ("$p = Get-CimInstance Win32_Process | Where-Object { "
              + " -or ".join(conds)
              + " } | Select-Object -ExpandProperty ProcessId; if($p){$p}")
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, text=True, timeout=25)
        except (OSError, subprocess.SubprocessError):
            return pids
        for line in (proc.stdout or "").strip().splitlines():
            try:
                pids.add(int(line.strip()))
            except ValueError:
                pass
        return pids
    try:
        proc = subprocess.run(["ps", "-eo", "pid=,comm=,args="],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return pids
    for line in (proc.stdout or "").splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, comm, args = int(parts[0]), parts[1], parts[2]
        if any(fnmatch.fnmatch(comm, pat) for pat in name_pats) \
                or any(sub in args for sub in match_subs):
            pids.add(pid)
    return pids


def _kill_tree(pid: int):
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                           capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            pass
        return
    try:
        pgid = os.getpgid(int(pid))
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            return
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
    except OSError:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass


def _script_command(script: str) -> list:
    script = str(script)
    if os.name == "nt":
        ext = os.path.splitext(script)[1].lower()
        if ext == ".ps1":
            return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", script]
        if ext in (".bat", ".cmd"):
            return ["cmd.exe", "/c", script]
        return [script]
    if os.access(script, os.X_OK):
        return [script]
    ext = os.path.splitext(script)[1].lower()
    if ext == ".py":
        return [sys.executable, script]
    return ["bash", script]


def _read_start_script(config: AgentConfig, port: int) -> str:
    for path in (_script_path(config, port), _legacy_bat_path(config, port)):
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


def stop(config: AgentConfig, port: int, state_key=None,
         stop_script=None, stop_bat=None, stop_match=None,
         stop_names=None) -> dict:
    port = int(port)
    provided = str(stop_script or stop_bat or "").strip()
    start_script = _read_start_script(config, port)
    candidates = []
    for base in (provided, start_script, provided):
        if not base:
            continue
        derived = _derive_stop_script(base)
        for cand in (base, derived):
            if cand and cand not in candidates:
                candidates.append(cand)
    stop_script = next((c for c in candidates if os.path.isfile(c)),
                       candidates[0] if candidates else "")
    stop_script_ran = False
    stop_output = ""
    if stop_script and os.path.isfile(stop_script):
        try:
            proc = subprocess.run(
                _script_command(stop_script),
                cwd=os.path.dirname(stop_script) or None,
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=45)
            stop_script_ran = True
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
    for path in (_pid_path(config, port), _script_path(config, port),
                 _legacy_bat_path(config, port)):
        try:
            path.unlink()
        except OSError:
            pass
    return {"port": port, "killed": _port_pid(port) is None,
            "pids": sorted(pids), "stop_script": stop_script,
            "stop_script_ran": stop_script_ran,
            **({"stop_output": stop_output} if stop_output else {})}


def start(config: AgentConfig, script: str, port: int, log=None,
          workdir=None, stop_script=None, stop_bat=None,
          stop_match=None, stop_names=None) -> dict:
    script = str(script or "").strip()
    if not script or not os.path.isabs(script):
        raise AgentCustomError("start script must be an absolute path")
    if not os.path.isfile(script):
        raise AgentCustomError("start script not found")
    try:
        port = int(port)
    except (TypeError, ValueError) as e:
        raise AgentCustomError("port must be an integer") from e
    if not 1 <= port <= 65535:
        raise AgentCustomError("port must be in 1..65535")
    stop(config, port, stop_script=stop_script or stop_bat
         or _derive_stop_script(script),
         stop_match=stop_match, stop_names=stop_names)
    if log:
        logp = Path(log)
    else:
        logp = Path.home() / f"custom-{port}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    cwd = str(workdir) if workdir else (os.path.dirname(script) or None)
    logf = open(logp, "a", encoding="utf-8")
    try:
        kwargs = {
            "cwd": cwd, "stdout": logf, "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL, "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(_script_command(script), **kwargs)
    except OSError as e:
        logf.close()
        raise AgentCustomError(f"failed to start script: {e}") from e
    finally:
        try:
            logf.close()
        except Exception:
            pass
    pidp = _pid_path(config, port)
    pidp.parent.mkdir(parents=True, exist_ok=True)
    pidp.write_text(str(proc.pid), encoding="utf-8")
    _script_path(config, port).write_text(script, encoding="utf-8")
    return {"pid": proc.pid, "port": port, "log": str(logp)}
