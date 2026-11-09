"""Minimal token-authenticated HTTP server for the peer agent."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import logging
import os
import platform
import signal
import threading
import time
import urllib.parse
import base64
import hashlib
import subprocess
import sys
import tempfile

from . import PROTOCOL_VERSION, __version__
from .config import AgentConfig, ensure_dirs
from . import processes
from . import discovery
from . import files
from . import service
from . import metrics
from . import system
from . import install
from . import custom
from . import build

_START_TS = time.time()
_MAX_BODY = 1_048_576
_PAIRING_DEADLINE = 0.0


def _pairing_open() -> bool:
    return bool(_PAIRING_DEADLINE and time.time() < _PAIRING_DEADLINE)


def _pairing_enabled() -> bool:
    return (os.environ.get("PEER_AGENT_PAIRING") or "1").lower() not in (
        "0", "false", "no", "off")


def _pairing_window() -> int:
    try:
        return max(30, int(os.environ.get("PEER_AGENT_PAIRING_WINDOW") or 600))
    except ValueError:
        return 86400
log = logging.getLogger("llama_peer_agent")


class AgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, config: AgentConfig):
        self.config = config
        super().__init__(address, handler)


class AgentHandler(BaseHTTPRequestHandler):
    server_version = f"llama-peer-agent/{__version__}"

    def _authorized(self) -> bool:
        expected = self.server.config.token
        if not expected:
            return False
        header = (self.headers.get("Authorization") or "").strip()
        if not header.lower().startswith("bearer "):
            return False
        got = header[7:].strip()
        return hmac.compare_digest(got, expected)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as e:
            raise ValueError("invalid Content-Length") from e
        if length < 0 or length > _MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("JSON object required")
        return obj

    def _dispatch_get(self, path: str):
        config = self.server.config
        if path == "/health":
            return 200, {
                "ok": True,
                "agent_version": __version__,
                "protocol_version": PROTOCOL_VERSION,
                "uptime_s": round(time.time() - _START_TS, 1),
                "os": platform.system().lower(),
                "arch": platform.machine(),
            }
        if path == "/version":
            import sys as _sys
            return 200, {"agent_version": __version__,
                         "protocol_version": PROTOCOL_VERSION,
                         "packaged": bool(getattr(_sys, "frozen", False))}
        if path == "/capabilities":
            return 200, processes.capabilities(config)
        if path == "/info":
            return 200, processes.info(config)
        if path == "/metrics":
            return 200, metrics.collect()
        if path == "/system/detail":
            return 200, system.detail(config)
        if path == "/system/rdma":
            return 200, system.rdma()
        if path == "/build/probe":
            query = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                return 200, build.probe(
                    config, (query.get("llama_bin") or [""])[0])
            except (files.AgentFileError, build.AgentBuildError) as e:
                return 400, {"error": str(e)}
        if path == "/status":
            return 200, processes.process_summary(config)
        if path == "/processes":
            return 200, processes.process_summary(config)
        if path == "/logs/tail":
            query = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else "")
            name = (query.get("name") or [""])[0]
            try:
                lines = int((query.get("lines") or ["20"])[0])
                return 200, processes.logs_tail(config, name, lines)
            except (ValueError, processes.AgentProcessError) as e:
                return 400, {"error": str(e)}
        if path == "/logs/list":
            return 200, processes.logs_list(config)
        if path == "/port/open":
            query = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                return 200, processes.port_open((query.get("port") or [""])[0])
            except processes.AgentProcessError as e:
                return 400, {"error": str(e)}
        if path == "/which":
            query = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                return 200, service.which((query.get("name") or [""])[0])
            except service.AgentServiceError as e:
                return 400, {"error": str(e)}
        if path == "/service/status":
            query = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                return 200, service.status(
                    config, (query.get("name") or [""])[0])
            except service.AgentServiceError as e:
                return 400, {"error": str(e)}
        if path == "/files/stat":
            query = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                return 200, files.stat(
                    config, (query.get("path") or [""])[0])
            except files.AgentFileError as e:
                return 400, {"error": str(e)}
        if path == "/files/list":
            query = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else "")
            hidden = (query.get("hidden") or ["0"])[0].lower() in (
                "1", "true", "yes")
            try:
                return 200, files.list_dir(
                    config, (query.get("path") or [""])[0], hidden)
            except files.AgentFileError as e:
                return 400, {"error": str(e)}
        if path == "/rpc/list":
            return 200, {"servers": processes.list_rpc_processes()}
        if path == "/llama/list":
            return 200, {"servers": processes.list_llama_processes()}
        if path == "/llama/status":
            try:
                port = int((self.path.split("?", 1)[1] if "?" in self.path
                            else "").split("port=", 1)[-1].split("&", 1)[0])
            except (TypeError, ValueError):
                return 400, {"error": "port query parameter required"}
            if not 1 <= port <= 65535:
                return 400, {"error": "port must be in 1..65535"}
            st = processes.llama_status(port)
            if not (st["running"] or st["loading"]):
                return 404, {"error": "no llama-server on that port"}
            return 200, st
        return 404, {"error": "not found"}

    def _dispatch_post(self, path: str, body: dict):
        global _PAIRING_DEADLINE
        config = self.server.config
        if path == "/pair":
            if (config.token or "").strip():
                return 400, {"error": "already paired"}
            if not _pairing_open():
                return 400, {"error": "pairing window closed"}
            tok = str(body.get("token") or "").strip()
            if len(tok) < 16:
                return 400, {"error": "token must be at least 16 chars"}
            from .config import save_token
            try:
                save_token(tok)
            except OSError as e:
                return 500, {"error": f"could not persist token: {e}"}
            config.token = tok
            _PAIRING_DEADLINE = 0.0
            log.warning("paired successfully; token persisted")
            return 200, {"ok": True, "paired": True}
        if path == "/files/write":
            try:
                mode = body.get("mode")
                return 200, files.write(
                    config, body.get("path"), body.get("text") or "",
                    mode=int(mode) if mode is not None else None,
                    backup=bool(body.get("backup")),
                    create_parents=bool(body.get("create_parents")))
            except (TypeError, ValueError, files.AgentFileError) as e:
                return 400, {"error": str(e)}
        if path == "/files/mkdir":
            try:
                return 200, files.mkdir(config, body.get("path"))
            except files.AgentFileError as e:
                return 400, {"error": str(e)}
        if path == "/files/chmod":
            try:
                return 200, files.chmod(
                    config, body.get("path"), body.get("mode"))
            except (TypeError, ValueError, files.AgentFileError) as e:
                return 400, {"error": str(e)}
        if path == "/files/backup":
            try:
                return 200, files.backup(
                    config, body.get("path"), body.get("stamp"))
            except files.AgentFileError as e:
                return 400, {"error": str(e)}
        if path == "/service/start":
            try:
                return 200, service.start(
                    config, body.get("name"), body.get("binary"),
                    body.get("config_path"), body.get("port"))
            except service.AgentServiceError as e:
                return 400, {"error": str(e)}
        if path == "/service/stop":
            try:
                return 200, service.stop(config, body.get("name"))
            except service.AgentServiceError as e:
                return 400, {"error": str(e)}
        if path == "/service/validate":
            try:
                return 200, service.validate(
                    config, body.get("binary"), body.get("config_path"))
            except service.AgentServiceError as e:
                return 400, {"error": str(e)}
        if path == "/custom/start":
            try:
                return 200, custom.start(
                    config, body.get("bat"), body.get("port"),
                    body.get("log"), body.get("workdir"))
            except custom.AgentCustomError as e:
                return 400, {"error": str(e)}
        if path == "/custom/stop":
            try:
                return 200, custom.stop(
                    config, body.get("port"), body.get("state_key"))
            except custom.AgentCustomError as e:
                return 400, {"error": str(e)}
        if path == "/install/artifact":
            try:
                return 200, install.artifact(
                    config, body.get("urls"), body.get("dest"),
                    body.get("archive") or "tar.gz",
                    body.get("strip_components") or 0,
                    body.get("member") or "", body.get("mode"))
            except (TypeError, ValueError, install.AgentInstallError) as e:
                return 400, {"error": str(e)}
        if path == "/install/verify":
            try:
                return 200, install.verify(
                    config, body.get("path"), body.get("args"))
            except install.AgentInstallError as e:
                return 400, {"error": str(e)}
        if path == "/agent/update":
            return self._agent_update(body)
        if path == "/agent/update-binary":
            return self._agent_update_binary(body)
        if path == "/build/update":
            try:
                return 200, build.update(
                    config, body.get("src_root"), body.get("build_dir"),
                    body.get("bin_path"), body.get("jobs"))
            except (files.AgentFileError, build.AgentBuildError) as e:
                return 400, {"error": str(e)}
        if path == "/rpc/ensure":
            try:
                port, devices, env = processes.validate_ensure(
                    body.get("port"), body.get("devices") or [],
                    body.get("env") or {})
                result = processes.ensure_rpc(config, port, devices, env)
            except processes.AgentProcessError as e:
                return 400, {"error": str(e)}
            return 200, result
        if path == "/rpc/stop":
            try:
                port = int(body.get("port"))
            except (TypeError, ValueError):
                return 400, {"error": "port must be an integer"}
            if not 1 <= port <= 65535:
                return 400, {"error": "port must be in 1..65535"}
            return 200, processes.stop_rpc_port(config, port)
        if path == "/llama/start":
            try:
                port, args, env, state = processes.validate_llama_start(
                    body.get("port"), body.get("args") or [],
                    body.get("env") or {}, body.get("state_key"))
                result = processes.start_llama(
                    config, port, args, env, state)
            except processes.AgentProcessError as e:
                return 400, {"error": str(e)}
            ok = bool(result.get("running") or result.get("loading")
                      or result.get("reused"))
            return (200 if ok else 500), result
        if path == "/llama/stop-all":
            return 200, processes.stop_all(config)
        if path == "/llama/stop":
            try:
                port = int(body.get("port"))
            except (TypeError, ValueError):
                return 400, {"error": "port must be an integer"}
            if not 1 <= port <= 65535:
                return 400, {"error": "port must be in 1..65535"}
            try:
                return 200, processes.stop_llama_port(
                    config, port, body.get("state_key"))
            except processes.AgentProcessError as e:
                return 400, {"error": str(e)}
        return 404, {"error": "not found"}

    def _restart_after_update(self, config):
        import os
        import time
        time.sleep(1.5)
        try:
            if getattr(sys, "frozen", False):
                argv = [sys.executable, "--host", config.host,
                        "--port", str(config.port)]
            else:
                argv = [sys.executable, "-m", "llama_peer_agent",
                        "--host", config.host, "--port", str(config.port)]
            os.execv(sys.executable, argv)
        except Exception as e:
            log.error("agent self-restart failed: %s", e)

    def _agent_update(self, body: dict):
        config = self.server.config
        raw = str(body.get("wheel_b64") or "")
        digest = str(body.get("sha256") or "").strip().lower()
        version = str(body.get("version") or "").strip()
        if not raw or not digest:
            return 400, {"error": "wheel_b64 and sha256 are required"}
        try:
            blob = base64.b64decode(raw, validate=True)
        except Exception as e:
            return 400, {"error": f"invalid wheel_b64: {e}"}
        if len(blob) > 8 * 1024 * 1024:
            return 400, {"error": "wheel too large"}
        actual = hashlib.sha256(blob).hexdigest()
        if actual != digest:
            return 400, {"error": "sha256 mismatch"}
        import os
        from pathlib import Path
        d = Path.home() / ".llama-peer-agent"
        d.mkdir(parents=True, exist_ok=True)
        name = f"llama_rpc_farm_agent-{version or 'update'}-py3-none-any.whl"
        wheel = d / name
        wheel.write_bytes(blob)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "--force-reinstall", "--no-deps", str(wheel)],
                capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as e:
            return 500, {"error": f"pip install failed: {e}"}
        if proc.returncode != 0:
            return 500, {"error": ((proc.stdout or "") + (proc.stderr or ""))[-500:]}
        threading.Thread(target=self._restart_after_update,
                         args=(config,), daemon=True).start()
        return 200, {"ok": True, "installed": version or name,
                     "restarting": True}

    def _agent_update_binary(self, body: dict):
        """Standalone agents replace their own executable from a GitHub
        release asset, then re-exec. Wheels are only for pip installs."""
        config = self.server.config
        if not getattr(sys, "frozen", False):
            return 400, {"error": "not a standalone agent"}
        import os
        import shutil
        import tempfile
        import urllib.request
        from pathlib import Path
        from .install import _url

        try:
            url = _url(body.get("url"))
        except Exception as e:
            return 400, {"error": f"invalid url: {e}"}
        version = str(body.get("version") or "").strip()
        expected = str(body.get("sha256") or "").strip().lower()
        dest = Path(sys.executable).resolve()
        try:
            fd, tmp = tempfile.mkstemp(prefix=".agent-update-",
                                       dir=str(dest.parent))
            os.close(fd)
            with urllib.request.urlopen(url, timeout=180) as resp, \
                    open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            size = os.path.getsize(tmp)
            if size < 1024 * 1024 or size > 200 * 1024 * 1024:
                raise RuntimeError(f"unexpected agent size: {size} bytes")
            if expected:
                import hashlib
                with open(tmp, "rb") as f:
                    actual = hashlib.sha256(f.read()).hexdigest()
                if actual != expected:
                    raise RuntimeError("sha256 mismatch")
            os.chmod(tmp, 0o755)
            backup = dest.with_name(dest.name + ".old")
            try:
                os.replace(dest, backup)
            except OSError:
                pass
            os.replace(tmp, dest)
        except Exception as e:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            return 500, {"error": f"agent binary update failed: {e}"}
        threading.Thread(target=self._restart_after_update,
                         args=(config,), daemon=True).start()
        return 200, {"ok": True, "installed": version or dest.name,
                     "restarting": True}

    def _send_speedtest(self):
        query = urllib.parse.parse_qs(
            self.path.split("?", 1)[1] if "?" in self.path else "")
        try:
            mb = int((query.get("mb") or ["50"])[0])
        except (TypeError, ValueError):
            self._send_json(400, {"error": "mb must be an integer"})
            return
        if not 1 <= mb <= 256:
            self._send_json(400, {"error": "mb must be in 1..256"})
            return
        total = mb * 1024 * 1024
        chunk = b"\0" * (1024 * 1024)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(total))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        left = total
        try:
            while left > 0:
                n = min(left, len(chunk))
                self.wfile.write(chunk[:n])
                left -= n
        except OSError:
            pass

    def _handle(self, method: str):
        path = (self.path or "/").split("?", 1)[0].rstrip("/") or "/"
        if not self._authorized():
            allowed = path in ("/health", "/version") or (
                path == "/pair" and _pairing_open())
            if not allowed:
                self._send_json(401, {"error": "unauthorized"})
                return
        if method == "GET" and path == "/speedtest":
            self._send_speedtest()
            return
        try:
            if method == "GET":
                status, payload = self._dispatch_get(path)
            else:
                try:
                    body = self._read_json()
                except (ValueError, json.JSONDecodeError) as e:
                    self._send_json(400, {"error": str(e)})
                    return
                status, payload = self._dispatch_post(path, body)
            self._send_json(status, payload)
        except processes.AgentProcessError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            log.exception("request failed: %s %s", method, path)
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)


def create_server(config: AgentConfig) -> AgentHTTPServer:
    global _PAIRING_DEADLINE
    if not (config.token or "").strip():
        if not _pairing_enabled():
            raise ValueError("PEER_AGENT_TOKEN is required")
        _PAIRING_DEADLINE = time.time() + _pairing_window()
        log.warning("no token — PAIRING MODE open for %ss",
                    _pairing_window())
    if not 0 <= int(config.port) <= 65535:
        raise ValueError("port must be in 0..65535 (0 = ephemeral)")
    return AgentHTTPServer((config.host, int(config.port)),
                           AgentHandler, config)


def serve(config: AgentConfig):
    ensure_dirs(config)
    server = create_server(config)
    config.port = int(server.server_address[1])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def _shutdown(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _shutdown)
            except (OSError, ValueError):
                pass
    disc = None
    if config.discovery:
        try:
            disc = discovery.start(config)
            log.info("discovery responder on udp %s:%s",
                     config.discovery_host, config.discovery_port)
        except OSError as e:
            log.warning("discovery disabled: %s", e)
    log.info("llama-peer-agent %s protocol=%s listening on %s:%s",
             __version__, PROTOCOL_VERSION, config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if disc is not None:
            thread, sock, stop_event = disc
            stop_event.set()
            try:
                sock.close()
            except OSError:
                pass
            thread.join(timeout=2)
        log.info("llama-peer-agent stopped")
