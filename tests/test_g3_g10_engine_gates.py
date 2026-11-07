"""G-3 + G-10: llama.cpp regression coverage for RPC controls + launch.

- POST /api/machines/{name}/rpc/start delegates to rpc_manager.start_rpc.
- POST /api/start-all starts RPC on every machine then launches
  llama-swap.
- POST /api/models has NO host gate (llama.cpp serves Windows peers via
  RPC).
- /api/launch and /api/launch/stop keep their llama.cpp behavior,
  including the remote-proxy swap unload.

Heavy externals (peer transport, real process starts) are monkeypatched; the REAL
config_loader owns machines.yaml against tmp files.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402
import config_loader  # noqa: E402


def _model_saved(model_id):
    """Path of a saved model file inside the (string) MODELS_DIR."""
    return os.path.join(config_loader.MODELS_DIR, f"{model_id}.yaml")


# ---------------- fixtures ------------------------------------------------

@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Same contract as test_engine_endpoints.settings_file: BOTH settings
    globals point into tmp; machines.yaml gets one linux master."""
    import engine_registry
    eng = tmp_path / "settings.yaml"
    eng.write_text("")
    monkeypatch.setattr(app_module, "SETTINGS_FILE", str(eng))
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", str(eng))
    import access
    monkeypatch.setattr(access, "_settings_path", lambda: str(eng))

    machines = tmp_path / "machines.yaml"
    machines.write_text(
        "machines:\n"
        "- id: aibox1\n"
        "  name: aibox1\n"
        "  ip: 127.0.0.1\n"
        "  role: master\n"
        "  os: linux\n"
        "  rpc_port: 50052\n"
        "  gpu: CUDA0,CUDA1\n"
        "  gpu_vendor: nvidia\n"
    )
    monkeypatch.setattr(config_loader.machines, "path", str(machines))
    import rpc_manager
    monkeypatch.setattr(rpc_manager, "cfg", config_loader)

    # models dir -> tmp (save_model resolves through it; app.cfg IS
    # config_loader, one setattr covers both names)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    yield str(eng)


@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def _add_windows_machine():
    """Extend the fixture machines.yaml with a windows peer (winbox)."""
    with open(config_loader.machines.path, "a", encoding="utf-8") as f:
        f.write(
            "- id: winbox\n"
            "  name: winbox\n"
            "  ip: 10.0.0.124\n"
            "  role: peer\n"
            "  os: windows\n"
            "  rpc_port: 50052\n"
            "  gpu: CUDA0\n"
            "  gpu_vendor: nvidia\n"
        )


def _add_group(name, machines):
    """Point cfg.groups at a one-group map (save_model resolves groups)."""
    monkey_target = config_loader
    monkey_target.groups = type(
        "G", (), {"all": staticmethod(lambda: [
            {"name": name, "machines": machines, "description": ""}]),
            "get": staticmethod(lambda n: {"name": name, "machines": machines}
                                if n == name else None)})()



class _LcppEngine:
    name = "llama_cpp"

    # DEC-04: save_model dispatches the per-model section round-trip
    # through the owning adapter; stub doubles carry the protocol members.
    def apply_model_section(self, data, where="model"):
        return data

    def sanitize_model_section(self, raw, where="model"):
        if raw is None:
            return None
        return dict(raw)

    def capable(self, capability):
        return capability == "rpc_cluster"


def _engine(engine_obj, monkeypatch):
    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: engine_obj)


def _swap_down(monkeypatch):
    """swap not running -> save/start-all skip push/launch I/O paths."""
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    monkeypatch.setattr(app_module.swap_manager, "launch",
                        lambda *a, **k: {"failed": True,
                                         "error": "not under test"})


# ---------------- Part A: /api/machines/{name}/rpc/start -------------------

def test_rpc_start_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode: the endpoint still delegates to rpc_manager.start_rpc
    and returns its payload verbatim."""
    _engine(_LcppEngine(), monkeypatch)
    calls = []

    def fake_start(name):
        calls.append(name)
        return {"machine": name, "running": True, "port": 50052}

    monkeypatch.setattr(app_module.rpc_manager, "start_rpc", fake_start)
    r = client.post("/api/machines/aibox1/rpc/start")
    assert r.status_code == 200, r.text
    assert r.json() == {"machine": "aibox1", "running": True, "port": 50052}
    assert calls == ["aibox1"]



# ---------------- Part A: /api/start-all -----------------------------------

def test_start_all_llama_cpp_starts_rpc_then_swap(client, settings_file,
                                                 monkeypatch):
    """llama.cpp mode: RPC starts on EVERY machine, then llama-swap
    launches — the original behavior, unchanged."""
    _engine(_LcppEngine(), monkeypatch)
    rpc_names = []
    monkeypatch.setattr(app_module.rpc_manager, "start_rpc",
                        lambda name: rpc_names.append(name) or
                        {"machine": name, "running": True, "port": 50052})
    launches = []
    monkeypatch.setattr(app_module.swap_manager, "launch",
                        lambda *a, **k: launches.append(a) or
                        {"host": "aibox1", "port": 8199})
    r = client.post("/api/start-all")
    assert r.status_code == 200, r.text
    body = r.json()
    assert rpc_names == ["aibox1"]           # the fixture's one machine
    assert len(launches) == 1
    assert body["rpc"][0]["machine"] == "aibox1"
    assert body["swap"]["host"] == "aibox1"



# ---------------- Part B: validate_hosts in POST /api/models ---------------

def _model_payload(**over):
    base = {
        "model_id": "qwen-3-8b",
        "name": "Qwen3 8B",
        "base_path": "/models/Qwen3-8B",
        "servers": ["aibox1"],
    }
    base.update(over)
    return base







def test_save_model_llama_cpp_never_host_gated(client, settings_file,
                                               monkeypatch):
    """llama.cpp mode: validate_hosts is never consulted — a windows peer
    in the farm does NOT block a model save (llama.cpp serves it fine via
    RPC). Adapter shape stays the pre-G-10 one."""
    _engine(_LcppEngine(), monkeypatch)
    _swap_down(monkeypatch)
    _add_windows_machine()
    monkeypatch.setattr(app_module.rpc_manager, "validate_model",
                        lambda mid: {"issues": []})
    r = client.post("/api/models", json=_model_payload())
    assert r.status_code == 200, r.text
    assert (os.path.exists(_model_saved("qwen-3-8b")))


# ---------------- B15: /api/launch/stop is engine-aware ---------------------



def test_launch_stop_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode, unconfigured id: the heuristic can't resolve the
    deployment, so stop keeps the legacy path — rpc_manager.stop_model
    with the same arguments, no swap proxying."""
    _engine(_LcppEngine(), monkeypatch)
    calls = {}

    def fake_stop(servers, model_id=None, variant=None, master_name=None,
                  launch_port=None):
        calls["args"] = (servers, model_id, variant, master_name,
                         launch_port)
        return [{"machine": servers[0], "stopped": True}]

    monkeypatch.setattr(app_module.rpc_manager, "stop_model", fake_stop)
    monkeypatch.setattr(app_module, "_proxy_swap_unload",
                        lambda model_id: (_ for _ in ()).throw(
                            AssertionError("llama_cpp stop must not proxy")))
    r = client.post("/api/launch/stop",
                    json={"model_id": "Qwen3-8B", "servers": ["aibox1"],
                          "port": 8080})
    assert r.status_code == 200, r.text
    assert calls["args"] == (["aibox1"], "Qwen3-8B", None, None, 8080)


def test_launch_stop_remote_proxy_unloads_via_swap(client, settings_file,
                                                   monkeypatch):
    """Serve-group truth: a remote-proxy (single-box-on-a-peer) llama.cpp
    deployment stops via the swap unload proxy (cmdStop) — the server
    lives on the peer box, so rpc_manager.stop_model would miss it."""
    _engine(_LcppEngine(), monkeypatch)
    proxied = []
    monkeypatch.setattr(
        app_module.swap_manager, "deployment_launch_mode",
        lambda mid, variant=None, host_name=None: {
            "mode": "local", "box": "aibox3", "peers": ["aibox3"],
            "swap_key": "qwen3-8b--DAD"})
    stopped = []
    monkeypatch.setattr(app_module.rpc_manager, "stop_model",
                        lambda *a, **k: stopped.append((a, k)) or [])

    async def fake_unload(model_id):
        proxied.append(model_id)
        return {"ok": True}

    monkeypatch.setattr(app_module, "_proxy_swap_unload", fake_unload)
    _add_peer_aibox3()
    r = client.post("/api/launch/stop",
                    json={"model_id": "qwen3-8b", "servers": ["aibox3"],
                          "port": 8080})
    assert r.status_code == 200, r.text
    assert stopped, "live-port stop must run first"
    assert proxied == ["qwen3-8b--DAD"]


def _add_peer_aibox3():
    """Extend the fixture machines.yaml with the aibox3 peer."""
    with open(config_loader.machines.path, "a", encoding="utf-8") as f:
        f.write(
            "- id: aibox3\n"
            "  name: aibox3\n"
            "  ip: 10.0.0.98\n"
            "  role: peer\n"
            "  os: linux\n"
            "  rpc_port: 50052\n"
            "  gpu: CUDA0,CUDA1,CUDA2\n"
            "  gpu_vendor: nvidia\n"
        )


def test_launch_auto_resolves_remote_box_to_local(client, settings_file,
                                                    monkeypatch):
    """mode=auto (the new default): a single-box-on-a-peer deployment
    launches local ON that box — the generator heuristic decides (no RPC
    hops), and run-on follows the verdict when the caller pinned no
    master."""
    captured = {}

    class _AutoEngine(_LcppEngine):
        def start_model(self, model_id, servers, port, variant=None,
                        master_name=None, mode="rpc"):
            captured.update(model_id=model_id, servers=servers, port=port,
                            variant=variant, master_name=master_name,
                            mode=mode)
            return {"ok": True, "failed": False}

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                         lambda s=None: _AutoEngine())
    monkeypatch.setattr(
        app_module.swap_manager, "deployment_launch_mode",
        lambda mid, variant=None, host_name=None: {
            "mode": "local", "box": "aibox3", "peers": ["aibox3"],
            "swap_key": "qwen3-8b--DAD"})
    _add_peer_aibox3()
    r = client.post("/api/launch",
                    json={"model_id": "qwen3-8b", "servers": ["aibox3"],
                          "port": 8080, "mode": "auto"})
    assert r.status_code == 200, r.text
    assert captured["mode"] == "local"
    assert captured["master_name"] == "aibox3"


def test_launch_explicit_rpc_still_passes_through(client, settings_file,
                                                  monkeypatch):
    """Explicit mode=rpc is never second-guessed: the operator's pin wins
    over the heuristic (same rule the generator's launch_mode field has)."""
    captured = {}

    class _AutoEngine(_LcppEngine):
        def start_model(self, model_id, servers, port, variant=None,
                        master_name=None, mode="rpc"):
            captured.update(master_name=master_name, mode=mode)
            return {"ok": True, "failed": False}

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                         lambda s=None: _AutoEngine())
    monkeypatch.setattr(app_module.swap_manager, "deployment_launch_mode",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("explicit mode must not resolve")))
    _add_peer_aibox3()
    r = client.post("/api/launch",
                    json={"model_id": "qwen3-8b", "servers": ["aibox3"],
                          "port": 8080, "mode": "rpc",
                          "master_name": "aibox3"})
    assert r.status_code == 200, r.text
    assert captured == {"master_name": "aibox3", "mode": "rpc"}


def test_swap_evict_shares_proxy_helper(client, settings_file, monkeypatch):
    """The B15 refactor kept /api/swap/evict/{model_id} behavior: same
    proxy helper, same 503-when-down contract."""
    _engine(_LcppEngine(), monkeypatch)
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/swap/evict/Qwen3-8B")
    assert r.status_code == 503, r.text
    monkeypatch.setattr(app_module.rpc_manager, "stop_model",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("no rpc path here")))


def test_swap_load_warmup_contract(client, settings_file, monkeypatch):
    """POST /api/swap/load/{id}: 503 when the daemon is down; otherwise
    a minimal warmup completion through the daemon (max_tokens=1) that
    triggers llama-swap's on-first-request load. The urllib call is
    stubbed — no real daemon needed."""
    import io
    import json as _json
    import urllib.request
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/swap/load/Qwen3-8B")
    assert r.status_code == 503, r.text

    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": True, "host": "aibox1",
                                 "host_ip": "127.0.0.1", "port": 8080})
    monkeypatch.setattr(app_module.local_node, "is_local",
                        lambda m: True, raising=False)
    seen = {}

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return _json.dumps(
                {"choices": [{"message": {"content": "hi"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = _json.loads(req.data.decode())
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    r = client.post("/api/swap/load/Qwen3-8B")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["model"] == "Qwen3-8B"
    assert seen["url"] == "http://127.0.0.1:8090/v1/chat/completions"
    assert seen["body"]["max_tokens"] == 1
