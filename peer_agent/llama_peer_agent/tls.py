"""Mutual-TLS front for ggml-rpc-server on a peer.

The RPC server itself stays plain TCP on its existing port. This module
adds a second listener (tls_port_for(port)) that requires a CA-signed
client certificate and forwards decrypted bytes to 127.0.0.1:port.
"""
from __future__ import annotations

import os
import socket
import ssl
import threading
from pathlib import Path

from .config import AgentConfig

TLS_PORT_BASE = 53000
TLS_PORT_RANGE = 500
TLS_PORT_LEGACY_BASE = 53500
PROXY_DIR = ".llama-peer-agent/tls"


def tls_dir(config: AgentConfig) -> Path:
    root = (Path(config.state_dir).expanduser() if config.state_dir
            else Path.home() / ".cache" / "llama-wrapper")
    d = root / "tls"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tls_port_for(rpc_port: int) -> int:
    p = int(rpc_port)
    if 51000 <= p < 51000 + TLS_PORT_RANGE:
        return TLS_PORT_BASE + (p - 51000)
    return TLS_PORT_LEGACY_BASE + (p % TLS_PORT_RANGE)


def cert_paths(config: AgentConfig):
    d = tls_dir(config)
    return d / "server.crt", d / "server.key", d / "ca.crt"


def install(config: AgentConfig, cert, key, ca) -> dict:
    crt, key_path, ca_path = cert_paths(config)
    for path, data in ((crt, cert), (key_path, key), (ca_path, ca)):
        if not isinstance(data, (str, bytes)) or not str(data).strip():
            raise ValueError("cert, key, and ca are required PEM strings")
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return {"ok": True, "cert": str(crt), "key": str(key_path),
            "ca": str(ca_path)}


def status(config: AgentConfig) -> dict:
    crt, key, ca = cert_paths(config)
    return {"cert": crt.is_file(), "key": key.is_file(), "ca": ca.is_file(),
            "proxies": sorted(_PROXIES)}


def _context(config: AgentConfig):
    crt, key, ca = cert_paths(config)
    if not (crt.is_file() and key.is_file() and ca.is_file()):
        raise RuntimeError("TLS certs not installed on this peer")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(crt), str(key))
    ctx.load_verify_locations(str(ca))
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def _pump(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


class _Proxy:
    def __init__(self, rpc_port, context):
        self.rpc_port = int(rpc_port)
        self.listen_port = tls_port_for(rpc_port)
        self.context = context
        self._sock = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._sock is not None:
            return self
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", self.listen_port))
        s.listen(64)
        s.settimeout(0.5)
        self._sock = s
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name=f"rpc-tls-{self.listen_port}")
        self._thread.start()
        return self

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn):
        target = None
        try:
            conn = self.context.wrap_socket(conn, server_side=True)
            target = socket.create_connection(
                ("127.0.0.1", self.rpc_port), timeout=10)
            threading.Thread(target=_pump, args=(conn, target),
                             daemon=True).start()
            threading.Thread(target=_pump, args=(target, conn),
                             daemon=True).start()
        except Exception:
            for s in (conn, target):
                try:
                    if s:
                        s.close()
                except OSError:
                    pass

    def stop(self):
        self._stop.set()
        try:
            if self._sock:
                self._sock.close()
        except OSError:
            pass
        self._sock = None


_PROXIES = {}
_LOCK = threading.Lock()


def ensure_proxy(config: AgentConfig, rpc_port: int) -> dict:
    rpc_port = int(rpc_port)
    with _LOCK:
        existing = _PROXIES.get(rpc_port)
        if existing is not None:
            return {"running": True, "reused": True,
                    "tls_port": existing.listen_port}
        p = _Proxy(rpc_port, _context(config))
        p.start()
        _PROXIES[rpc_port] = p
        return {"running": True, "reused": False, "tls_port": p.listen_port}


def stop_proxy(rpc_port: int) -> dict:
    with _LOCK:
        p = _PROXIES.pop(int(rpc_port), None)
    if p:
        p.stop()
    return {"stopped": bool(p), "tls_port": tls_port_for(int(rpc_port))}


def stop_all() -> dict:
    with _LOCK:
        items = list(_PROXIES.items())
        _PROXIES.clear()
    for _port, p in items:
        p.stop()
    return {"stopped": len(items)}
