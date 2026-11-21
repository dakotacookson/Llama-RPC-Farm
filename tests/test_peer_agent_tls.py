"""Peer-agent mTLS front: cert install and proxy roundtrip."""
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

import rpc_tls
from llama_peer_agent.config import AgentConfig
from llama_peer_agent import tls as agent_tls


def _cert_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(rpc_tls, "CERT_DIR", str(tmp_path))
    monkeypatch.setattr(rpc_tls, "CA_CRT", str(tmp_path / "ca.crt"))
    monkeypatch.setattr(rpc_tls, "CA_KEY", str(tmp_path / "ca.key"))
    monkeypatch.setattr(rpc_tls, "MASTER_CRT", str(tmp_path / "master.crt"))
    monkeypatch.setattr(rpc_tls, "MASTER_KEY", str(tmp_path / "master.key"))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _echo_server():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]

    def _run():
        conn, _ = s.accept()
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                conn.sendall(data)
        finally:
            conn.close()
            s.close()
    threading.Thread(target=_run, daemon=True).start()
    return port


def _cfg(tmp_path):
    return AgentConfig(token="t", state_dir=str(tmp_path / "state"),
                       log_dir=str(tmp_path / "logs"))


def test_install_cert_bundle(tmp_path, monkeypatch):
    _cert_paths(monkeypatch, tmp_path)
    rpc_tls.ensure_ca()
    bundle = rpc_tls.cert_bundle({"name": "peer1", "ip": "127.0.0.1"})
    cfg = _cfg(tmp_path)
    out = agent_tls.install(cfg, bundle["cert"], bundle["key"], bundle["ca"])
    assert out["ok"] is True
    st = agent_tls.status(cfg)
    assert st["cert"] and st["key"] and st["ca"]


def test_agent_tls_proxy_roundtrip(tmp_path, monkeypatch):
    _cert_paths(monkeypatch, tmp_path)
    rpc_tls.ensure_ca()
    rpc_tls.issue_master_cert()
    rpc_tls.issue_peer_cert({"name": "peer1", "ip": "127.0.0.1"})
    cfg = _cfg(tmp_path)
    bundle = rpc_tls.cert_bundle({"name": "peer1", "ip": "127.0.0.1"})
    agent_tls.install(cfg, bundle["cert"], bundle["key"], bundle["ca"])

    rpc_port = _free_port()
    tls_port = _free_port()
    monkeypatch.setattr(agent_tls, "tls_port_for", lambda p: tls_port)
    echo = _echo_server()
    # The agent proxy forwards to the RPC port; point it at the echo server
    # by using a fake rpc port equal to the echo port's value only for the
    # target lookup: ensure_proxy uses the rpc_port both as key/target.
    monkeypatch.setattr(agent_tls._Proxy, "_handle",
                        lambda self, conn: _handle_to(self, conn, echo))
    out = agent_tls.ensure_proxy(cfg, rpc_port)
    assert out["running"] is True and out["tls_port"] == tls_port

    client_listen = _free_port()
    client = rpc_tls.TLSProxy(
        "127.0.0.1", client_listen, "127.0.0.1", tls_port,
        rpc_tls.client_context(), server_side=False).start()
    try:
        sock = None
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", client_listen),
                                                timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        assert sock is not None
        with sock:
            sock.sendall(b"agent-secure-rpc")
            assert sock.recv(64) == b"agent-secure-rpc"
    finally:
        client.stop()
        agent_tls.stop_proxy(rpc_port)


def _handle_to(self, conn, target_port):
    """Proxy handler with a test target instead of self.rpc_port."""
    target = None
    try:
        conn = self.context.wrap_socket(conn, server_side=True)
        target = socket.create_connection(("127.0.0.1", target_port),
                                          timeout=10)
        threading.Thread(target=agent_tls._pump, args=(conn, target),
                         daemon=True).start()
        threading.Thread(target=agent_tls._pump, args=(target, conn),
                         daemon=True).start()
    except Exception:
        for s in (conn, target):
            try:
                if s:
                    s.close()
            except OSError:
                pass
