"""Optional mTLS RPC transport: certs, ports, and proxy roundtrip."""
import socket
import threading
import time

import pytest

import rpc_manager
import rpc_tls


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


def test_enabled_setting_roundtrip(monkeypatch, tmp_path):
    import settings_store
    p = tmp_path / "settings.yaml"
    p.write_text("engine: llama_cpp\n")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", str(p))
    assert rpc_tls.enabled() is False
    rpc_tls.set_enabled(True)
    assert rpc_tls.enabled() is True
    rpc_tls.set_enabled(False)
    assert rpc_tls.enabled() is False


def test_tls_port_mapping_is_stable():
    assert rpc_tls.TUNNEL_PORT == 5399
    assert rpc_tls.tls_port_for(51000) == 53000
    assert rpc_tls.tls_port_for(51499) == 53499
    assert rpc_tls.tls_port_for(50052) >= 53500
    assert rpc_tls.tls_port_for(51001) == rpc_tls.tls_port_for(51001)


def test_cert_generation(monkeypatch, tmp_path):
    _cert_paths(monkeypatch, tmp_path)
    ca = rpc_tls.ensure_ca()
    assert ca.endswith("ca.crt")
    rpc_tls.issue_master_cert()
    bundle = rpc_tls.cert_bundle({"name": "peer1", "ip": "127.0.0.1"})
    assert "BEGIN CERTIFICATE" in bundle["cert"]
    assert "BEGIN PRIVATE KEY" in bundle["key"] or "BEGIN RSA PRIVATE KEY" in bundle["key"]
    assert "BEGIN CERTIFICATE" in bundle["ca"]
    # contexts load the generated material
    assert rpc_tls.client_context() is not None
    assert rpc_tls.server_context("peer1") is not None


def test_mtls_proxy_roundtrip(monkeypatch, tmp_path):
    _cert_paths(monkeypatch, tmp_path)
    rpc_tls.ensure_ca()
    rpc_tls.issue_master_cert()
    rpc_tls.issue_peer_cert({"name": "peer1", "ip": "127.0.0.1"})

    echo_port = _echo_server()
    server_listen = _free_port()
    client_listen = _free_port()
    server = rpc_tls.TLSProxy("127.0.0.1", server_listen, "127.0.0.1",
                              echo_port, rpc_tls.server_context("peer1"),
                              server_side=True).start()
    client = rpc_tls.TLSProxy("127.0.0.1", client_listen, "127.0.0.1",
                              server_listen, rpc_tls.client_context(),
                              server_side=False).start()
    try:
        deadline = time.time() + 5
        sock = None
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", client_listen),
                                                timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        assert sock is not None
        with sock:
            sock.sendall(b"secure-rpc")
            assert sock.recv(64) == b"secure-rpc"
    finally:
        client.stop()
        server.stop()


def test_secure_build_uses_local_proxy_and_same_ports(stub_cfg, monkeypatch):
    model = {"model_id": "secure-model", "flags": {},
             "base_path": "store://llm-models/x.gguf"}
    master = {"name": "aibox1", "ip": "10.0.0.1", "role": "master",
              "os": "linux", "gpu": ""}
    peer = {"name": "peer1", "ip": "10.0.0.2", "role": "peer",
            "os": "linux", "gpu": "CUDA0",
            "detected": {"gpu_names": ["GPU"], "vram_total_gb": 8}}
    stub_cfg.update({"aibox1": master, "peer1": peer})
    ports = rpc_manager._model_rpc_ports(model, [peer])
    port = ports[("peer1", "all")]

    monkeypatch.setattr(rpc_tls, "enabled", lambda: False)
    plain = rpc_manager.build_llama_command(
        model, ["aibox1", "peer1"], master_name="aibox1")
    assert f"10.0.0.2:{port}" in plain

    monkeypatch.setattr(rpc_tls, "enabled", lambda: True)
    secure = rpc_manager.build_llama_command(
        model, ["aibox1", "peer1"], master_name="aibox1")
    assert f"127.0.0.1:{port}" in secure
    assert f"10.0.0.2:{port}" not in secure
