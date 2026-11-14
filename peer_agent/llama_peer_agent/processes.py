"""OS process management for the peer agent (RPC servers only)."""
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import files
from .config import AgentConfig
from .platform_info import board_model, tegra_gpu_unsupported_reason

_PORT_RE = re.compile(r"(?:^|\s)(?:-p|--port)\s+(\d+)(?:\s|$)")
_LISTEN_RE = re.compile(r"--listen[=\s]+[^\s]*:(\d+)")
_DEVICE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


class AgentProcessError(RuntimeError):
    pass


def rpc_binary(config: AgentConfig) -> str:
    explicit = (config.rpc_bin or "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else ""
    name = "ggml-rpc-server.exe" if os.name == "nt" else "ggml-rpc-server"
    found = shutil.which(name)
    if found:
        return found
    for cand in (
        Path.home() / "llama.cpp" / "build" / "bin" / name,
        Path.home() / "llama.cpp" / "bin" / name,
        Path.home() / ".local" / "bin" / name,
    ):
        if cand.is_file():
            return str(cand)
    return ""


def llama_binary(config: AgentConfig) -> str:
    explicit = (config.llama_bin or "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else ""
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    found = shutil.which(name)
    if found:
        return found
    for cand in (
        Path.home() / "llama.cpp" / "build" / "bin" / name,
        Path.home() / "llama.cpp" / "bin" / name,
        Path.home() / ".local" / "bin" / name,
    ):
        if cand.is_file():
            return str(cand)
    return ""


def capabilities(config: AgentConfig) -> dict:
    rpc_bin_path = rpc_binary(config)
    llama_bin_path = llama_binary(config)
    return {
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "rpc_available": bool(rpc_bin_path),
        "rpc_binary": rpc_bin_path or None,
        "llama_available": bool(llama_bin_path),
        "llama_binary": llama_bin_path or None,
        "model_roots": list(config.model_roots),
    }


def _state_path(config: AgentConfig, port: int) -> Path:
    base = Path(config.state_dir).expanduser() if config.state_dir else (
        Path.home() / ".cache" / "llama-wrapper")
    return base / f"rpc.state.{int(port)}"


def _log_path(config: AgentConfig, port: int) -> Path:
    base = Path(config.log_dir).expanduser() if config.log_dir else Path.home()
    name = "rpc-server.log" if int(port) == 50052 else f"rpc-server-{int(port)}.log"
    return base / name


def _llama_state_path(config: AgentConfig, state_key: str) -> Path:
    base = Path(config.state_dir).expanduser() if config.state_dir else (
        Path.home() / ".cache" / "llama-wrapper")
    return base / f"swap.port.{state_key}"


def _llama_log_path(config: AgentConfig, port: int) -> Path:
    base = Path(config.log_dir).expanduser() if config.log_dir else Path.home()
    return base / f"llama-server-{int(port)}.log"


def _port_open(port: int, timeout=0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _read_state(config: AgentConfig, port: int) -> str:
    try:
        return _state_path(config, port).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_state(config: AgentConfig, port: int, want: str):
    path = _state_path(config, port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(want, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _tail(path: Path, lines=20) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _is_rpc_cmd(cmd: str) -> bool:
    base = os.path.basename((cmd or "").split()[0]) if cmd else ""
    return base in ("ggml-rpc-server", "ggml-rpc-server.exe")


def _parse_etime(text: str):
    text = (text or "").strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    try:
        parts = [int(x) for x in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def _proc_basename(cmd: str) -> str:
    return os.path.basename((cmd or "").split()[0]) if cmd else ""


def _list_processes(names, require_port=True) -> list[dict]:
    names = {n.lower() for n in names}
    if os.name == "nt":
        ps = (
            "Get-CimInstance Win32_Process | Where-Object { "
            "$_.Name -eq 'ggml-rpc-server.exe' -or "
            "$_.Name -eq 'llama-server.exe' -or "
            "$_.Name -eq 'llama-swap.exe' } | "
            "ForEach-Object { [pscustomobject]@{ Pid=$_.ProcessId; "
            "Age=[int]((Get-Date)-$_.CreationDate).TotalSeconds; "
            "Name=$_.Name; Cmd=$_.CommandLine } } | ConvertTo-Json -Compress")
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return []
        raw = (proc.stdout or "").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            return []
        if isinstance(data, dict):
            data = [data]
        out = []
        for item in data if isinstance(data, list) else []:
            name = str(item.get("Name") or "").lower()
            if name not in names:
                continue
            cmd = item.get("Cmd") or ""
            m = _PORT_RE.search(cmd) or _LISTEN_RE.search(cmd)
            if not m and require_port:
                continue
            try:
                out.append({"pid": int(item.get("Pid") or 0),
                            "port": int(m.group(1)) if m else None,
                            "age_s": int(item.get("Age")) if item.get("Age") is not None else None,
                            "cmd": cmd})
            except (TypeError, ValueError):
                continue
        return out

    macos = sys.platform == "darwin"
    cmd = ("ps -axo pid=,etime=,command=" if macos
           else "ps -eo pid=,etimes=,args=")
    try:
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        age = _parse_etime(parts[1]) if macos else (
            int(parts[1]) if parts[1].isdigit() else None)
        cmdline = parts[2]
        if _proc_basename(cmdline).lower() not in names:
            continue
        m = _PORT_RE.search(cmdline) or _LISTEN_RE.search(cmdline)
        if not m and require_port:
            continue
        out.append({"pid": pid, "port": int(m.group(1)) if m else None,
                    "age_s": age, "cmd": cmdline})
    return out


def list_rpc_processes() -> list[dict]:
    """Running ggml-rpc-server processes on THIS host."""
    return _list_processes(("ggml-rpc-server", "ggml-rpc-server.exe"))


def list_llama_processes() -> list[dict]:
    """Running llama-server processes on THIS host."""
    return _list_processes(("llama-server", "llama-server.exe"))


def list_swap_processes() -> list[dict]:
    """Running llama-swap daemon processes on THIS host."""
    return _list_processes(("llama-swap", "llama-swap.exe"),
                           require_port=False)


def gpu_rows() -> list[str]:
    """NVIDIA CSV rows normalized for the metrics payload."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.used,"
             "temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines()
            if line.strip()]


def process_summary(config: AgentConfig) -> dict:
    """Process view used by the master status page."""
    rpc = list_rpc_processes()
    llama = list_llama_processes()
    swap = list_swap_processes()
    return {
        "rpc": rpc,
        "llama": llama,
        "swap": swap,
        "gpus": gpu_rows(),
        "rpc_running": bool(rpc),
        "llama_running": bool(llama),
        "swap_running": bool(swap),
        "counts": {"rpc": len(rpc), "llama": len(llama),
                   "swap": len(swap)},
    }


def _kill_pid(pid: int, force=False):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, text=True, timeout=20)
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def stop_rpc_port(config: AgentConfig, port: int) -> dict:
    port = int(port)
    pids = [p["pid"] for p in list_rpc_processes() if p["port"] == port]
    for pid in pids:
        _kill_pid(pid, force=False)
    deadline = time.time() + 5
    while time.time() < deadline:
        alive = {p["pid"] for p in list_rpc_processes()
                 if p["port"] == port}
        if not alive:
            break
        time.sleep(0.2)
    alive = {p["pid"] for p in list_rpc_processes() if p["port"] == port}
    for pid in alive:
        _kill_pid(pid, force=True)
    if alive:
        time.sleep(0.3)
    alive = {p["pid"] for p in list_rpc_processes() if p["port"] == port}
    try:
        _state_path(config, port).unlink()
    except OSError:
        pass
    # stale RPC tensor cache (same cleanup as the script path)
    try:
        shutil.rmtree(Path.home() / ".cache" / "llama.cpp" / "rpc",
                      ignore_errors=True)
    except Exception:
        pass
    return {"port": port, "killed": not bool(alive), "pids": pids}


def _safe_state_key(key: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-"
                   for c in str(key or "")).strip("-")
    if not safe:
        raise AgentProcessError("state_key is required")
    return safe[:120]


def stop_llama_port(config: AgentConfig, port: int,
                    state_key=None) -> dict:
    """Kill llama-server on this port plus any ports in its state file."""
    ports = {int(port)}
    if state_key:
        try:
            state = _llama_state_path(config, _safe_state_key(state_key))
            for line in state.read_text(encoding="utf-8").splitlines():
                try:
                    ports.add(int(line.strip()))
                except ValueError:
                    pass
        except OSError:
            pass
    pids = []
    for p in ports:
        for item in list_llama_processes():
            if item["port"] == p:
                pids.append(item["pid"])
    for pid in set(pids):
        _kill_pid(pid, force=False)
    deadline = time.time() + 5
    while time.time() < deadline:
        alive = {i["pid"] for i in list_llama_processes()
                 if i["port"] in ports}
        if not alive:
            break
        time.sleep(0.2)
    alive = {i["pid"] for i in list_llama_processes() if i["port"] in ports}
    for pid in alive:
        _kill_pid(pid, force=True)
    if alive:
        time.sleep(0.3)
    alive = {i["pid"] for i in list_llama_processes() if i["port"] in ports}
    if state_key:
        try:
            _llama_state_path(config, _safe_state_key(state_key)).unlink()
        except OSError:
            pass
    return {"port": int(port), "ports": sorted(ports),
            "killed": not bool(alive), "pids": sorted(set(pids))}


def validate_llama_start(port, args, env, state_key, llama_bin=None):
    if llama_bin is not None and (not isinstance(llama_bin, str)
                                  or "\x00" in llama_bin):
        raise AgentProcessError("llama_bin must be a string")
    try:
        port = int(port)
    except (TypeError, ValueError) as e:
        raise AgentProcessError("port must be an integer") from e
    if not 1 <= port <= 65535:
        raise AgentProcessError("port must be in 1..65535")
    if not isinstance(args, list) or not args:
        raise AgentProcessError("args must be a non-empty list of strings")
    if len(args) > 512:
        raise AgentProcessError("too many args")
    clean_args = []
    for a in args:
        if not isinstance(a, str) or not a or "\x00" in a:
            raise AgentProcessError("args must be non-empty strings")
        if a in ("--host", "--port") or a.startswith(("--host=", "--port=")):
            raise AgentProcessError("--host/--port are managed by the agent")
        clean_args.append(a)
    if not (("-m" in clean_args) or ("--model" in clean_args)):
        raise AgentProcessError("args must include -m <model> or --model <model>")
    if not isinstance(env, dict):
        raise AgentProcessError("env must be an object")
    clean_env = {}
    for k, v in env.items():
        k = str(k)
        if not _ENV_KEY_RE.match(k):
            raise AgentProcessError(f"invalid env key: {k!r}")
        if not isinstance(v, (str, int, float, bool)):
            raise AgentProcessError(f"env value for {k} must be scalar")
        clean_env[k] = str(v)
    state = _safe_state_key(state_key) if state_key else None
    return port, clean_args, clean_env, state


def start_llama(config: AgentConfig, port: int, args: list, env: dict,
                state_key=None, llama_bin=None) -> dict:
    port, args, env, state_key = validate_llama_start(
        port, args, env, state_key, llama_bin)
    existing = [p for p in list_llama_processes() if p["port"] == port]
    if _port_open(port):
        if existing:
            return {"running": True, "loading": False, "reused": True,
                    "port": port, "pid": existing[0]["pid"],
                    "log": str(_llama_log_path(config, port))}
        return {"running": False, "loading": False, "reused": False,
                "port": port, "error": "port is held by a foreign process"}
    if existing:
        # a previous attempt is still loading (port not open yet)
        return {"running": False, "loading": True, "reused": True,
                "port": port, "pid": existing[0]["pid"],
                "log": str(_llama_log_path(config, port))}

    bin_path = llama_binary(config)
    override = (llama_bin or "").strip()
    if override:
        # The master knows which binary it probed and wants served; use
        # exactly that instead of re-running discovery (which can pick a
        # stale legacy path). Restrict to an allowed root and the known
        # binary name: this is a typed operation, not a shell.
        try:
            p = files._path(config, override)
        except files.AgentFileError as e:
            raise AgentProcessError(f"llama_bin override invalid: {e}") from e
        if not p.is_file() or p.name not in ("llama-server",
                                             "llama-server.exe"):
            raise AgentProcessError(
                "llama_bin override must be an existing llama-server binary")
        bin_path = str(p)
    if not bin_path:
        return {"running": False, "loading": False, "reused": False,
                "port": port,
                "error": "llama-server binary not found; pass --llama-bin "
                         "or install llama.cpp"}
    log_path = _llama_log_path(config, port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [bin_path] + args + ["--host", "0.0.0.0", "--port", str(port)]
    full_env = dict(os.environ)
    full_env.update(env)
    full_env.setdefault("GGML_SCHED_DEBUG", "3")
    full_env.setdefault("GGML_RPC_DEBUG", "3")
    try:
        logf = open(log_path, "w", encoding="utf-8")
    except OSError as e:
        return {"running": False, "loading": False, "reused": False,
                "port": port, "error": f"cannot open log {log_path}: {e}"}
    try:
        kwargs = {
            "stdout": logf, "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL, "env": full_env, "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as e:
        logf.close()
        return {"running": False, "loading": False, "reused": False,
                "port": port, "error": f"failed to start {bin_path}: {e}"}
    finally:
        try:
            logf.close()
        except Exception:
            pass
    if state_key:
        path = _llama_state_path(config, state_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(port), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    time.sleep(1)
    if proc.poll() is not None:
        return {"running": False, "loading": False, "reused": False,
                "port": port, "pid": proc.pid, "log": str(log_path),
                "error": "llama-server exited during startup",
                "log_tail": _tail(log_path)}
    return {"running": _port_open(port), "loading": not _port_open(port),
            "reused": False, "port": port, "pid": proc.pid,
            "log": str(log_path)}


def llama_status(port: int) -> dict:
    port = int(port)
    procs = [p for p in list_llama_processes() if p["port"] == port]
    open_ = _port_open(port)
    return {"port": port, "running": open_, "loading": bool(procs) and not open_,
            "pid": procs[0]["pid"] if procs else None}


def validate_ensure(port, devices, env):
    try:
        port = int(port)
    except (TypeError, ValueError) as e:
        raise AgentProcessError("port must be an integer") from e
    if not 1 <= port <= 65535:
        raise AgentProcessError("port must be in 1..65535")
    if devices is None:
        devices = []
    if not isinstance(devices, list):
        raise AgentProcessError("devices must be a list of strings")
    clean_devices = []
    for d in devices:
        d = str(d).strip()
        if not d:
            continue
        if not _DEVICE_RE.match(d):
            raise AgentProcessError(f"invalid device token: {d!r}")
        clean_devices.append(d)
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise AgentProcessError("env must be an object")
    clean_env = {}
    for k, v in env.items():
        k = str(k)
        if not _ENV_KEY_RE.match(k):
            raise AgentProcessError(f"invalid env key: {k!r}")
        if not isinstance(v, (str, int, float, bool)):
            raise AgentProcessError(f"env value for {k} must be scalar")
        clean_env[k] = str(v)
    return port, clean_devices, clean_env


def ensure_rpc(config: AgentConfig, port: int, devices: list,
               env: dict) -> dict:
    port, devices, env = validate_ensure(port, devices, env)
    want = ",".join(devices) if devices else "default"
    if _port_open(port) and _read_state(config, port) == want:
        return {"running": True, "reused": True, "port": port,
                "pid": None, "log": str(_log_path(config, port))}

    stop_rpc_port(config, port)
    time.sleep(0.5)
    bin_path = rpc_binary(config)
    if not bin_path:
        return {"running": False, "reused": False, "port": port,
                "error": "ggml-rpc-server binary not found; pass --rpc-bin "
                         "or install llama.cpp"}

    log_path = _log_path(config, port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args = [bin_path, "-H", "0.0.0.0", "-p", str(port),
            "-t", str(os.cpu_count() or 4)]
    if devices:
        args += ["--device", ",".join(devices)]
    full_env = dict(os.environ)
    full_env.update(env)
    full_env.setdefault("GGML_SCHED_DEBUG", "3")
    full_env.setdefault("GGML_RPC_DEBUG", "3")
    try:
        logf = open(log_path, "w", encoding="utf-8")
    except OSError as e:
        return {"running": False, "reused": False, "port": port,
                "error": f"cannot open log {log_path}: {e}"}
    try:
        kwargs = {
            "stdout": logf, "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL, "env": full_env, "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(args, **kwargs)
    except OSError as e:
        logf.close()
        return {"running": False, "reused": False, "port": port,
                "error": f"failed to start {bin_path}: {e}"}
    finally:
        try:
            logf.close()
        except Exception:
            pass

    deadline = time.time() + 30
    while time.time() < deadline:
        if _port_open(port):
            _write_state(config, port, want)
            return {"running": True, "reused": False, "port": port,
                    "pid": proc.pid, "log": str(log_path)}
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    return {"running": False, "reused": False, "port": port,
            "pid": proc.pid, "log": str(log_path),
            "error": "rpc-server did not open its port",
            "log_tail": _tail(log_path)}


def stop_all(config: AgentConfig) -> dict:
    """Stop llama-swap first, then llama-server, then every RPC server.

    This is the parity operation for the master's legacy stop-everything
    path. It is deliberately explicit (no arbitrary process names) and
    reports anything that survived.
    """
    groups = (
        (("llama-swap", "llama-swap.exe"), False),
        (("llama-server", "llama-server.exe"), True),
        (("ggml-rpc-server", "ggml-rpc-server.exe"), True),
    )
    all_pids = []
    for names, require_port in groups:
        pids = [p["pid"] for p in _list_processes(names, require_port)]
        all_pids.extend(pids)
        for pid in pids:
            _kill_pid(pid, force=False)
        deadline = time.time() + 4
        while time.time() < deadline:
            if not _list_processes(names, require_port):
                break
            time.sleep(0.2)
        for item in _list_processes(names, require_port):
            _kill_pid(item["pid"], force=True)
        time.sleep(0.2)
    remaining = {
        "rpc": list_rpc_processes(),
        "llama": list_llama_processes(),
        "swap": list_swap_processes(),
    }
    left = sum(len(v) for v in remaining.values())
    return {"killed": left == 0, "pids": sorted(set(all_pids)),
            "remaining": remaining}


_LOG_NAME_RE = re.compile(
    r"^(?:rpc-server(?:-\d+)?|llama-server(?:-\d+)?|llama-swap)\.log$")


def _log_dir(config: AgentConfig) -> Path:
    return (Path(config.log_dir).expanduser() if config.log_dir
            else Path.home())


def logs_tail(config: AgentConfig, name: str, lines: int = 20) -> dict:
    """Tail a known log file only. Rejects paths, traversal, and unknown names."""
    name = str(name or "").strip()
    if not name or os.path.basename(name) != name or not _LOG_NAME_RE.match(name):
        raise AgentProcessError("unknown log name")
    try:
        lines = int(lines)
    except (TypeError, ValueError) as e:
        raise AgentProcessError("lines must be an integer") from e
    if not 1 <= lines <= 500:
        raise AgentProcessError("lines must be in 1..500")
    if name.startswith("rpc-server"):
        m = re.search(r"-(\d+)\.log$", name)
        path = _log_path(config, int(m.group(1)) if m else 50052)
    elif name.startswith("llama-server"):
        m = re.search(r"-(\d+)\.log$", name)
        path = _llama_log_path(config, int(m.group(1))) if m else (
            _log_dir(config) / "llama-server.log")
    else:
        path = _log_dir(config) / "llama-swap.log"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"name": name, "path": str(path), "lines": [], "text": ""}
    tail = text.splitlines()[-lines:]
    return {"name": name, "path": str(path), "lines": tail,
            "text": "\n".join(tail)}




def info(config: AgentConfig) -> dict:
    """Stable identity + addresses for discovery and enrollment."""
    node = uuid.getnode()
    mac = ":".join(f"{(node >> i) & 0xff:02x}" for i in range(40, -1, -8))
    hostname = socket.gethostname()
    addresses = []
    try:
        for addr in socket.gethostbyname_ex(hostname)[2]:
            if addr not in addresses:
                addresses.append(addr)
    except OSError:
        pass
    return {
        "hostname": hostname,
        "home": str(Path.home()),
        "machine_id": mac,
        "mac": mac,
        "addresses": addresses,
        "agent_port": int(config.port),
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "board": board_model(),
        "cpu_only": bool(tegra_gpu_unsupported_reason()),
        "gpu_unsupported": tegra_gpu_unsupported_reason(),
        "model_roots": list(config.model_roots),
        "rpc_available": bool(rpc_binary(config)),
        "llama_available": bool(llama_binary(config)),
    }


def stop_swap(config: AgentConfig) -> dict:
    """Stop only the llama-swap daemon."""
    names = ("llama-swap", "llama-swap.exe")
    pids = [p["pid"] for p in _list_processes(names, require_port=False)]
    for pid in pids:
        _kill_pid(pid, force=False)
    deadline = time.time() + 5
    while time.time() < deadline:
        if not _list_processes(names, require_port=False):
            break
        time.sleep(0.2)
    for item in _list_processes(names, require_port=False):
        _kill_pid(item["pid"], force=True)
    left = _list_processes(names, require_port=False)
    return {"killed": not bool(left), "pids": sorted(set(pids)),
            "remaining": left}


def port_open(port) -> dict:
    try:
        port = int(port)
    except (TypeError, ValueError) as e:
        raise AgentProcessError("port must be an integer") from e
    if not 1 <= port <= 65535:
        raise AgentProcessError("port must be in 1..65535")
    return {"port": port, "open": _port_open(port)}


def logs_list(config: AgentConfig) -> dict:
    """Known log files in the agent log directory."""
    base = _log_dir(config)
    names = []
    try:
        for path in base.iterdir():
            if path.is_file() and _LOG_NAME_RE.match(path.name):
                names.append(path.name)
    except OSError:
        pass
    return {"logs": sorted(names)}
