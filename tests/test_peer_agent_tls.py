"""Peer-agent mTLS front: cert install and proxy roundtrip."""
import os
import socket
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

import rpc_tls
from llama_peer_agent.config import AgentConfig
from llama_peer_agent import tls as agent_tls
from llama_peer_agent import server as agent_server


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


def test_agent_https_listener(tmp_path, monkeypatch):
    _cert_paths(monkeypatch, tmp_path)
    rpc_tls.ensure_ca()
    rpc_tls.issue_master_cert()
    rpc_tls.issue_peer_cert({"name": "peer1", "ip": "127.0.0.1"})
    cfg = _cfg(tmp_path)
    bundle = rpc_tls.cert_bundle({"name": "peer1", "ip": "127.0.0.1"})
    agent_tls.install(cfg, bundle["cert"], bundle["key"], bundle["ca"])
    assert agent_tls.https_context(cfg) is not None

    import http.client
    port = _free_port()
    monkeypatch.setattr(agent_tls, "https_port", lambda: port)
    out = agent_server._start_https(cfg)
    assert out["running"] is True
    try:
        conn = http.client.HTTPSConnection(
            "127.0.0.1", port, context=rpc_tls.client_context(), timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        conn.close()
    finally:
        agent_server._stop_https()


def test_agent_tls_tunnel_roundtrip(tmp_path, monkeypatch):
    _cert_paths(monkeypatch, tmp_path)
    rpc_tls.ensure_ca()
    rpc_tls.issue_master_cert()
    rpc_tls.issue_peer_cert({"name": "peer1", "ip": "127.0.0.1"})
    cfg = _cfg(tmp_path)
    bundle = rpc_tls.cert_bundle({"name": "peer1", "ip": "127.0.0.1"})
    agent_tls.install(cfg, bundle["cert"], bundle["key"], bundle["ca"])

    rpc_port = _free_port()
    tunnel_port = _free_port()
    monkeypatch.setattr(agent_tls, "TUNNEL_PORT", tunnel_port)
    echo = _echo_server()
    # tunnel maps the requested target port to 127.0.0.1:<target>; point
    # the target at the echo server by using an allowed RPC-range port.
    # We run the echo server on an allowed port-like socket instead:
    target_port = 51000 + (rpc_port % 500)
    # start a second echo listener on the exact target port if possible
    try:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", target_port))
        srv.listen(1)
    except OSError:
        pytest.skip("target RPC-range port unavailable")
    threading.Thread(target=lambda: _echo_on(srv), daemon=True).start()
    out = agent_tls.ensure_proxy(cfg, rpc_port)
    assert out["running"] is True and out["tls_port"] == tunnel_port

    client_listen = _free_port()
    client = rpc_tls.TLSProxy(
        "127.0.0.1", client_listen, "127.0.0.1", tunnel_port,
        rpc_tls.client_context(), server_side=False,
        route_port=target_port).start()
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
        agent_tls.stop_proxy()


def _echo_on(srv):
    conn, _ = srv.accept()
    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            conn.sendall(data)
    finally:
        conn.close()
        srv.close()
