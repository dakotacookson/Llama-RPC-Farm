"""Model replication: streaming file endpoints + per-machine path resolution."""
import io
import json
import os
import sys
import threading
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

from llama_peer_agent.config import AgentConfig  # noqa: E402
from llama_peer_agent import files, server as agent_server  # noqa: E402
import agent_client  # noqa: E402
import local_node  # noqa: E402
import model_sync  # noqa: E402


def _cfg(tmp):
    return AgentConfig(token="test-token", host="127.0.0.1", port=0,
                       file_roots=[str(tmp)])


def test_write_stream_and_read_chunks(tmp_path):
    c = _cfg(tmp_path)
    dst = tmp_path / "sub" / "model.gguf"
    res = files.write_stream(c, str(dst), io.BytesIO(b"hello world"),
                             11, mode=0o644, create_parents=True)
    assert res["bytes"] == 11 and dst.read_bytes() == b"hello world"
    path, size = files.read_info(c, str(dst))
    assert size == 11
    assert b"".join(files.read_chunks(c, str(dst), 4)) == b"hello world"


def test_write_stream_rejects_short_body(tmp_path):
    c = _cfg(tmp_path)
    dst = tmp_path / "short.gguf"
    try:
        files.write_stream(c, str(dst), io.BytesIO(b"abc"), 10,
                           create_parents=True)
    except files.AgentFileError as e:
        assert "short upload" in str(e)
    else:
        raise AssertionError("short body accepted")


def test_agent_file_endpoints_roundtrip(tmp_path):
    c = _cfg(tmp_path)
    srv = agent_server.create_server(c)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    hdr = {"Authorization": "Bearer test-token"}
    try:
        dst = tmp_path / "a.bin"
        url = (base + "/files/write_bin?path=" + urllib.request.quote(str(dst))
               + "&create_parents=1")
        req = urllib.request.Request(url, data=b"payload", method="POST",
                                     headers={**hdr,
                                              "Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert json.loads(r.read().decode())["bytes"] == 7
        assert dst.read_bytes() == b"payload"
        req = urllib.request.Request(
            base + "/files/read?path=" + urllib.request.quote(str(dst)),
            headers=hdr)
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.read() == b"payload"
    finally:
        srv.shutdown()
        srv.server_close()


def test_resolve_model_path_default_dir(monkeypatch):
    monkeypatch.setattr(local_node, "is_local", lambda m: True)
    model_sync._HOME_CACHE.clear()
    m = {"name": "pi", "os": "linux", "model_dir": ""}
    settings = {"model_sync": {"enabled": True,
                               "default_dir": "~/llama-models"}}
    p = model_sync.resolve_model_path(
        m, "store://llm-models/foo/bar.gguf", settings)
    assert p.endswith("/llama-models/foo/bar.gguf")


def test_resolve_model_path_machine_override(monkeypatch):
    monkeypatch.setattr(local_node, "is_local", lambda m: True)
    model_sync._HOME_CACHE.clear()
    m = {"name": "pi", "os": "linux", "model_dir": "/data/models"}
    p = model_sync.resolve_model_path(
        m, "store://llm-models/a.gguf",
        {"model_sync": {"enabled": False, "default_dir": "~/ignored"}})
    assert p == "/data/models/a.gguf"


def test_resolve_model_path_remote_home(monkeypatch):
    monkeypatch.setattr(local_node, "is_local", lambda m: False)
    monkeypatch.setattr(agent_client, "info",
                        lambda m: {"home": "/home/pi"})
    model_sync._HOME_CACHE.clear()
    m = {"name": "pi", "os": "linux", "model_dir": ""}
    p = model_sync.resolve_model_path(
        m, "store://llm-models/a.gguf",
        {"model_sync": {"enabled": True, "default_dir": "~/llama-models"}})
    assert p == "/home/pi/llama-models/a.gguf"


def test_resolve_model_path_windows(monkeypatch):
    monkeypatch.setattr(local_node, "is_local", lambda m: False)
    monkeypatch.setattr(agent_client, "info",
                        lambda m: {"home": r"C:\Users\z"})
    model_sync._HOME_CACHE.clear()
    m = {"name": "win", "os": "windows", "model_dir": ""}
    p = model_sync.resolve_model_path(
        m, "store://llm-models/a.gguf",
        {"model_sync": {"enabled": True, "default_dir": "~/llama-models"}})
    assert p == r"C:\Users\z\llama-models\a.gguf"


def test_sync_disabled_keeps_store(monkeypatch):
    monkeypatch.setattr(local_node, "is_local", lambda m: True)
    model_sync._HOME_CACHE.clear()
    m = {"name": "pi", "os": "linux", "model_dir": ""}
    monkeypatch.setattr(model_sync.path_resolver, "resolve_for",
                        lambda path, osname: "/STORE/" + path.split("/", 3)[-1])
    p = model_sync.resolve_model_path(
        m, "store://llm-models/a.gguf",
        {"model_sync": {"enabled": False, "default_dir": "~/ignored"}})
    assert p == "/STORE/a.gguf"


def test_ensure_model_local_copy(monkeypatch, tmp_path):
    import config_loader as cfg
    src_dir = tmp_path / "store"
    src_dir.mkdir(parents=True)
    src = src_dir / "a.gguf"
    src.write_bytes(b"model-bytes")
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    model = {"model_id": "m", "base_path": "store://llm-models/a.gguf",
             "servers": ["target"], "launch_mode": "local",
             "device_order": "target-RAM"}
    target = {"name": "target", "ip": "10.0.0.9", "os": "linux",
              "model_dir": str(target_dir)}
    monkeypatch.setattr(cfg, "get_model", lambda mid: model)
    monkeypatch.setattr(cfg.machines, "all", lambda: [target])
    monkeypatch.setattr(model_sync, "target_machine",
                        lambda mid, label, merged: target)
    monkeypatch.setattr(local_node, "is_local", lambda m: True)
    monkeypatch.setattr(model_sync.path_resolver, "resolve_for",
                        lambda path, osname: str(src))
    settings = {"model_sync": {"enabled": True,
                               "default_dir": str(target_dir)}}
    log = []
    res = model_sync.ensure_model("m", log=log.append, settings=settings)
    assert (target_dir / "a.gguf").read_bytes() == b"model-bytes"
    assert res["copied"] and not res["errors"]


def _start_agent(tmp_path):
    c = _cfg(tmp_path)
    srv = agent_server.create_server(c)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, port


def test_copy_local_to_remote_and_back(monkeypatch, tmp_path):
    srv, port = _start_agent(tmp_path)
    peer = {"name": "peer", "ip": "127.0.0.1", "agent_port": port,
            "_agent_token": "test-token", "os": "linux"}
    master = {"name": "master", "ip": "10.0.0.1", "os": "linux"}
    monkeypatch.setattr(local_node, "is_local",
                        lambda m: m.get("name") == "master")
    src = tmp_path / "src.gguf"
    src.write_bytes(b"x" * 4096)
    remote_dst = tmp_path / "remote.gguf"
    try:
        model_sync.copy_between(master, str(src), peer, str(remote_dst),
                                4096, lambda m: None)
        assert remote_dst.read_bytes() == b"x" * 4096
        back = tmp_path / "back.gguf"
        model_sync.copy_between(peer, str(remote_dst), master, str(back),
                                4096, lambda m: None)
        assert back.read_bytes() == b"x" * 4096
    finally:
        srv.shutdown()
        srv.server_close()


def test_copy_remote_to_remote(monkeypatch, tmp_path):
    srv, port = _start_agent(tmp_path)
    a = {"name": "a", "ip": "127.0.0.1", "agent_port": port,
         "_agent_token": "test-token", "os": "linux"}
    b = {"name": "b", "ip": "127.0.0.1", "agent_port": port,
         "_agent_token": "test-token", "os": "linux"}
    monkeypatch.setattr(local_node, "is_local", lambda m: False)
    src = tmp_path / "a.gguf"
    src.write_bytes(b"y" * 8192)
    dst = tmp_path / "b.gguf"
    try:
        model_sync.copy_between(a, str(src), b, str(dst), 8192,
                                lambda m: None)
        assert dst.read_bytes() == b"y" * 8192
    finally:
        srv.shutdown()
        srv.server_close()
