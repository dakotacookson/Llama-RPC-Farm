"""Typed Ray worker start/stop for multi-host vLLM."""
import os
import re
import shutil
import subprocess
from pathlib import Path

from . import files
from .config import AgentConfig

_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


class AgentRayError(RuntimeError):
    pass


def _ray_bin(raw) -> str:
    raw = str(raw or "").strip()
    if not raw:
        raise AgentRayError("ray_bin is required")
    base = os.path.basename(raw)
    if base not in ("ray", "ray.exe"):
        raise AgentRayError("ray_bin must be a ray executable")
    if os.path.isabs(raw):
        if not os.path.isfile(raw):
            raise AgentRayError("ray_bin not found")
        return raw
    found = shutil.which(raw)
    if not found:
        raise AgentRayError("ray not found on PATH")
    return found


def _iface_for(address: str) -> str:
    if not address:
        return ""
    try:
        proc = subprocess.run(
            ["ip", "route", "get", address], capture_output=True,
            text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    parts = (proc.stdout or "").split()
    if "dev" in parts:
        i = parts.index("dev")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _path(config: AgentConfig, raw, default):
    return str(files._path(config, raw or default))


def start(config: AgentConfig, ray_bin, address, num_gpus=0, env=None,
          master_ip="", log=None, pid_file=None, workdir=None) -> dict:
    ray_bin = _ray_bin(ray_bin)
    address = str(address or "").strip()
    if not address or ":" not in address:
        raise AgentRayError("address must be host:port")
    try:
        num_gpus = int(num_gpus or 0)
    except (TypeError, ValueError) as e:
        raise AgentRayError("num_gpus must be an integer") from e
    if not 0 <= num_gpus <= 64:
        raise AgentRayError("num_gpus out of range")
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise AgentRayError("env must be an object")
    clean_env = {}
    for k, v in env.items():
        k = str(k)
        if not _ENV_RE.match(k):
            raise AgentRayError(f"invalid env key: {k!r}")
        if not isinstance(v, (str, int, float, bool)):
            raise AgentRayError(f"env value for {k} must be scalar")
        clean_env[k] = str(v)
    iface = _iface_for(master_ip or address.split(":", 1)[0])
    if iface:
        clean_env.setdefault("GLOO_SOCKET_IFNAME", iface)
        clean_env.setdefault("NCCL_SOCKET_IFNAME", iface)
    log = _path(config, log, str(Path.home() / "ray-worker.log"))
    pid_file = _path(config, pid_file, str(Path.home() / "ray-worker.pid"))
    if workdir:
        workdir = _path(config, workdir, str(Path.home()))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    full_env.update(clean_env)
    full_env["PATH"] = os.path.dirname(ray_bin) + os.pathsep + full_env.get(
        "PATH", "")
    subprocess.run([ray_bin, "stop", "--force"], capture_output=True,
                   text=True, timeout=30)
    args = [ray_bin, "start", f"--address={address}",
            f"--num-gpus={num_gpus}"]
    kwargs = {"stdout": open(log, "w", encoding="utf-8"),
              "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
              "env": full_env, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(args, cwd=workdir or None, **kwargs)
    Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
    Path(pid_file).write_text(str(proc.pid), encoding="utf-8")
    return {"pid": proc.pid, "log": log, "pid_file": pid_file,
            "address": address, "num_gpus": num_gpus}


def stop(config: AgentConfig, ray_bin, pid_file=None, timeout=30) -> dict:
    ray_bin = _ray_bin(ray_bin)
    killed = False
    if pid_file:
        try:
            p = Path(_path(config, pid_file, ""))
            pid = int(p.read_text(encoding="utf-8").strip())
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True, timeout=15)
            else:
                os.kill(pid, 15)
            killed = True
        except (OSError, ValueError):
            pass
    try:
        subprocess.run([ray_bin, "stop", "--force"], capture_output=True,
                       text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        pass
    return {"stopped": True, "pid_killed": killed}
