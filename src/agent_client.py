"""Client for the peer agent — the ONLY runtime transport to peers.

The agent runs on every configured peer. `transport()` always returns
"agent" and `should_use_agent()` always returns true; the legacy
legacy `transport:` values are accepted for config compatibility but
no longer change behaviour. There is no remote shell fallback.

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
import base64
import hashlib
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
    """Agent-only architecture: every configured peer is reached over the
    peer agent. The legacy `transport:` value is accepted for config
    compatibility but no longer changes behaviour."""
    return "agent"


def agent_port(machine) -> int:
    try:
        value = int((machine or {}).get("agent_port") or DEFAULT_AGENT_PORT)
    except (TypeError, ValueError):
        value = DEFAULT_AGENT_PORT
    return value if 1 <= value <= 65535 else DEFAULT_AGENT_PORT


def _token(machine) -> str:
    override = str((machine or {}).get("_agent_token") or "").strip()
    if override:
        return override
    try:
        tok = secrets_vault.get_machine_agent_token(machine)
        if tok:
            return tok
    except Exception:
        pass
    # Farm-wide token (vault): paired agents accept this when no
    # per-machine token is stored. This is what lets a freshly discovered
    # agent be health-checked/paird without restarting the app.
    try:
        tok = secrets_vault.get_farm_agent_token()
        if tok:
            return tok
    except Exception:
        pass
    return (os.environ.get("FARM_AGENT_TOKEN")
            or os.environ.get("AGENT_TOKEN") or "").strip()


def _base(machine) -> str:
    ip = str((machine or {}).get("ip") or "").strip()
    if not ip:
        raise AgentError("machine has no IP")
    return f"http://{ip}:{agent_port(machine)}"


def _request(machine, method: str, path: str, body=None, timeout=6.0,
             token=None):
    token = token or _token(machine)
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


def pair(machine, token, timeout=8.0) -> dict:
    """Enroll an unpaired agent (its /pair window) with the farm token."""
    data = json.dumps({"token": token}, separators=(",", ":")).encode()
    req = urllib.request.Request(
        _base(machine) + "/pair", data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read(1000).decode("utf-8", "replace")
        except Exception:
            detail = ""
        raise AgentError(f"pair HTTP {e.code}: {detail or e.reason}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise AgentError(f"pair request failed: {e}") from e


def health(machine, timeout=2.0, token=None) -> dict:
    out = _request(machine, "GET", "/health", timeout=timeout, token=token)
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


def custom_start(machine, bat, port, log=None, workdir=None,
                 timeout=60.0) -> dict:
    return _request(machine, "POST", "/custom/start", {
        "bat": bat, "port": int(port), "log": log, "workdir": workdir,
    }, timeout=timeout)


def custom_stop(machine, port, state_key=None, timeout=60.0) -> dict:
    return _request(machine, "POST", "/custom/stop", {
        "port": int(port), "state_key": state_key,
    }, timeout=timeout)


def agent_update(machine, wheel_bytes, version="", timeout=200.0) -> dict:
    data = bytes(wheel_bytes)
    return _request(machine, "POST", "/agent/update", {
        "wheel_b64": base64.b64encode(data).decode("ascii"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "version": version,
    }, timeout=timeout)


def agent_update_binary(machine, url, version="", sha256="", timeout=300.0) -> dict:
    return _request(machine, "POST", "/agent/update-binary", {
        "url": url, "version": version, "sha256": sha256,
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
    """Agent-only: always true. There is no remote shell transport —
    agents are installed from their own platform packages."""
    return True
