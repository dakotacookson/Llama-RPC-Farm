"""Client for the optional peer agent.

The agent is a SEPARATE, optional service on peers. SSH remains the
default and permanent fallback: machine `transport` is `auto` (probe the
agent, else SSH), `ssh` (never use the agent), or `agent` (require it).

Protocol v1 endpoints (see docs/peer-agent.md):
  GET  /health, /version, /capabilities, /info, /status, /processes,
       /rpc/list, /llama/list, /llama/status, /logs/tail
  POST /rpc/ensure {port, devices, env}
  POST /rpc/stop   {port}
  POST /llama/start {port, args, env, state_key}
  POST /llama/stop {port, state_key}
  POST /llama/stop-all
All requests carry `Authorization: Bearer <agent token>`; the token comes
from the Fernet vault (per-machine, then farm-wide), never from YAML.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import secrets_vault

PROTOCOL_VERSION = 1
DEFAULT_AGENT_PORT = 8198
_HEALTH_TTL = 15.0
_health_cache: dict[str, tuple[float, bool]] = {}
# Never route LAN/VPN agent calls through an environment HTTP proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class AgentError(RuntimeError):
    """Agent unreachable, unauthorized, or protocol-incompatible."""


def transport(machine) -> str:
    value = str((machine or {}).get("transport") or "auto").strip().lower()
    return value if value in ("auto", "ssh", "agent") else "auto"


def agent_port(machine) -> int:
    try:
        value = int((machine or {}).get("agent_port") or DEFAULT_AGENT_PORT)
    except (TypeError, ValueError):
        value = DEFAULT_AGENT_PORT
    return value if 1 <= value <= 65535 else DEFAULT_AGENT_PORT


def _token(machine) -> str:
    try:
        return secrets_vault.get_machine_agent_token(machine)
    except Exception:
        return (os.environ.get("FARM_AGENT_TOKEN")
                or os.environ.get("AGENT_TOKEN") or "").strip()


def _base(machine) -> str:
    ip = str((machine or {}).get("ip") or "").strip()
    if not ip:
        raise AgentError("machine has no IP")
    return f"http://{ip}:{agent_port(machine)}"


def _request(machine, method: str, path: str, body=None, timeout=6.0):
    token = _token(machine)
    if not token:
        raise AgentError("agent token is not configured for this machine")
    data = None
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_base(machine) + path, data=data,
                                 method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read(1_048_576)
            return json.loads(raw.decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read(1000).decode("utf-8", "replace")
        except Exception:
            detail = ""
        raise AgentError(f"agent HTTP {e.code}: {detail or e.reason}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise AgentError(f"agent request failed: {e}") from e


def health(machine, timeout=2.0) -> dict:
    out = _request(machine, "GET", "/health", timeout=timeout)
    proto = out.get("protocol_version")
    if proto is not None:
        try:
            proto = int(proto)
        except (TypeError, ValueError):
            raise AgentError(f"agent returned invalid protocol: {proto!r}")
        if proto != PROTOCOL_VERSION:
            raise AgentError(
                f"agent protocol {proto} != supported {PROTOCOL_VERSION}")
    return out


def version(machine, timeout=3.0) -> dict:
    return _request(machine, "GET", "/version", timeout=timeout)


def capabilities(machine, timeout=5.0) -> dict:
    return _request(machine, "GET", "/capabilities", timeout=timeout)


def info(machine, timeout=5.0) -> dict:
    return _request(machine, "GET", "/info", timeout=timeout)


def status(machine, timeout=8.0) -> dict:
    return _request(machine, "GET", "/status", timeout=timeout)


def metrics(machine, timeout=12.0) -> dict:
    return _request(machine, "GET", "/metrics", timeout=timeout)


def system_detail(machine, timeout=40.0) -> dict:
    return _request(machine, "GET", "/system/detail", timeout=timeout)


def rdma(machine, timeout=25.0) -> dict:
    return _request(machine, "GET", "/system/rdma", timeout=timeout)


def speedtest(machine, mb=50, timeout=120.0):
    """Measure throughput from the agent's /speedtest stream (Mb/s)."""
    import time as _time
    token = _token(machine)
    if not token:
        raise AgentError("agent token is not configured for this machine")
    req = urllib.request.Request(
        _base(machine) + f"/speedtest?mb={int(mb)}",
        headers={"Authorization": f"Bearer {token}"})
    start = _time.monotonic()
    total = 0
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise AgentError(f"agent speedtest failed: {e}") from e
    elapsed = _time.monotonic() - start
    if elapsed <= 0 or total <= 0:
        return None
    return round((total * 8) / elapsed / 1_000_000)


def processes(machine, timeout=8.0) -> dict:
    return _request(machine, "GET", "/processes", timeout=timeout)


def logs_tail(machine, name: str, lines: int = 20,
              timeout=8.0) -> dict:
    query = urllib.parse.urlencode({"name": str(name),
                                    "lines": int(lines)})
    return _request(machine, "GET", f"/logs/tail?{query}", timeout=timeout)


def stop_all(machine, timeout=45.0) -> dict:
    return _request(machine, "POST", "/llama/stop-all", {}, timeout=timeout)


def logs_list(machine, timeout=8.0) -> list:
    out = _request(machine, "GET", "/logs/list", timeout=timeout)
    logs = out.get("logs") if isinstance(out, dict) else None
    return logs if isinstance(logs, list) else []


def service_status(machine, name="llama-swap", timeout=8.0) -> dict:
    query = urllib.parse.urlencode({"name": name})
    return _request(machine, "GET", f"/service/status?{query}",
                    timeout=timeout)


def service_start(machine, name="llama-swap", binary="llama-swap",
                  config_path="", port=8080, timeout=60.0) -> dict:
    return _request(machine, "POST", "/service/start", {
        "name": name, "binary": binary, "config_path": config_path,
        "port": int(port),
    }, timeout=timeout)


def service_stop(machine, name="llama-swap", timeout=30.0) -> dict:
    return _request(machine, "POST", "/service/stop", {"name": name},
                    timeout=timeout)


def service_validate(machine, binary="llama-swap", config_path="",
                     timeout=65.0) -> dict:
    return _request(machine, "POST", "/service/validate", {
        "binary": binary, "config_path": config_path,
    }, timeout=timeout)


def which(machine, name: str, timeout=8.0) -> dict:
    query = urllib.parse.urlencode({"name": name})
    return _request(machine, "GET", f"/which?{query}", timeout=timeout)


def port_open(machine, port: int, timeout=8.0) -> dict:
    query = urllib.parse.urlencode({"port": int(port)})
    return _request(machine, "GET", f"/port/open?{query}", timeout=timeout)


def file_write(machine, path, text, mode=None, backup=False,
               create_parents=False, timeout=30.0) -> dict:
    body = {"path": str(path), "text": str(text), "backup": bool(backup),
            "create_parents": bool(create_parents)}
    if mode is not None:
        body["mode"] = int(mode)
    return _request(machine, "POST", "/files/write", body,
                    timeout=timeout)


def file_mkdir(machine, path, timeout=15.0) -> dict:
    return _request(machine, "POST", "/files/mkdir", {"path": str(path)},
                    timeout=timeout)


def file_chmod(machine, path, mode, timeout=15.0) -> dict:
    return _request(machine, "POST", "/files/chmod", {
        "path": str(path), "mode": int(mode)}, timeout=timeout)


def file_backup(machine, path, stamp=None, timeout=20.0) -> dict:
    return _request(machine, "POST", "/files/backup", {
        "path": str(path), "stamp": stamp}, timeout=timeout)


def file_stat(machine, path, timeout=8.0) -> dict:
    query = urllib.parse.urlencode({"path": str(path)})
    return _request(machine, "GET", f"/files/stat?{query}", timeout=timeout)


def file_list(machine, path, hidden=False, timeout=8.0) -> dict:
    query = urllib.parse.urlencode({"path": str(path),
                                    "hidden": "1" if hidden else "0"})
    return _request(machine, "GET", f"/files/list?{query}", timeout=timeout)


def ray_start(machine, ray_bin, address, num_gpus=0, env=None,
              master_ip="", log=None, pid_file=None, workdir=None,
              timeout=60.0) -> dict:
    return _request(machine, "POST", "/ray/start", {
        "ray_bin": ray_bin, "address": address, "num_gpus": int(num_gpus),
        "env": dict(env or {}), "master_ip": master_ip, "log": log,
        "pid_file": pid_file, "workdir": workdir,
    }, timeout=timeout)


def ray_stop(machine, ray_bin, pid_file=None, timeout=40.0) -> dict:
    return _request(machine, "POST", "/ray/stop", {
        "ray_bin": ray_bin, "pid_file": pid_file,
    }, timeout=timeout)


def install_artifact(machine, urls, dest, archive="tar.gz",
                     strip_components=0, member="", mode=None,
                     timeout=1800.0) -> dict:
    body = {"urls": urls if isinstance(urls, list) else [urls],
            "dest": dest, "archive": archive,
            "strip_components": int(strip_components),
            "member": member}
    if mode is not None:
        body["mode"] = int(mode)
    return _request(machine, "POST", "/install/artifact", body,
                    timeout=timeout)


def install_verify(machine, path, args=None, timeout=130.0) -> dict:
    return _request(machine, "POST", "/install/verify", {
        "path": path, "args": args or ["--version"],
    }, timeout=timeout)


def vllm_install(machine, python="python3", venv_root="~/vllm-env",
                 version="0.30.0", rocm=False, timeout=2000.0) -> dict:
    return _request(machine, "POST", "/install/vllm", {
        "python": python, "venv_root": venv_root,
        "version": version, "rocm": bool(rocm),
    }, timeout=timeout)


def build_probe(machine, llama_bin, timeout=30.0) -> dict:
    query = urllib.parse.urlencode({"llama_bin": llama_bin})
    return _request(machine, "GET", f"/build/probe?{query}",
                    timeout=timeout)


def build_update(machine, src_root, build_dir, bin_path, jobs=None,
                 timeout=7300.0) -> dict:
    return _request(machine, "POST", "/build/update", {
        "src_root": src_root, "build_dir": build_dir,
        "bin_path": bin_path, "jobs": jobs,
    }, timeout=timeout)


def rpc_list(machine, timeout=8.0) -> list:
    out = _request(machine, "GET", "/rpc/list", timeout=timeout)
    servers = out.get("servers") if isinstance(out, dict) else None
    return servers if isinstance(servers, list) else []


def rpc_ensure(machine, port: int, devices=None, env=None,
               timeout=40.0) -> dict:
    return _request(machine, "POST", "/rpc/ensure", {
        "port": int(port),
        "devices": list(devices or []),
        "env": dict(env or {}),
    }, timeout=timeout)


def rpc_stop(machine, port: int, timeout=20.0) -> dict:
    return _request(machine, "POST", "/rpc/stop", {"port": int(port)},
                    timeout=timeout)


def llama_list(machine, timeout=8.0) -> list:
    out = _request(machine, "GET", "/llama/list", timeout=timeout)
    servers = out.get("servers") if isinstance(out, dict) else None
    return servers if isinstance(servers, list) else []


def llama_start(machine, port: int, args: list, env=None,
                state_key=None, timeout=30.0) -> dict:
    return _request(machine, "POST", "/llama/start", {
        "port": int(port),
        "args": [str(a) for a in (args or [])],
        "env": dict(env or {}),
        "state_key": state_key,
    }, timeout=timeout)


def llama_stop(machine, port: int, state_key=None, timeout=20.0) -> dict:
    return _request(machine, "POST", "/llama/stop", {
        "port": int(port), "state_key": state_key,
    }, timeout=timeout)


def _cache_key(machine) -> str:
    return str((machine or {}).get("name")
               or (machine or {}).get("id")
               or (machine or {}).get("ip") or "")


def should_use_agent(machine) -> bool:
    """Per-machine transport decision. `auto` probes /health (15s cache)."""
    mode = transport(machine)
    if mode == "ssh":
        return False
    # native RDMA/CM peers have no TCP listener for the agent's readiness
    # probe, so auto stays on SSH; an explicit `agent` choice is honoured
    # and the agent must implement its own probe.
    if (machine or {}).get("rdma_cm") and mode == "auto":
        return False
    if mode == "agent":
        return True
    key = _cache_key(machine)
    now = time.monotonic()
    hit = _health_cache.get(key)
    if hit and hit[0] > now:
        return hit[1]
    try:
        health(machine, timeout=1.5)
        ok = True
    except AgentError:
        ok = False
    _health_cache[key] = (now + _HEALTH_TTL, ok)
    return ok
