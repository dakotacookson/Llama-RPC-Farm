"""G-3 + G-10: engine gating of RPC controls + validate_hosts wiring.

- POST /api/machines/{name}/rpc/start: llama.cpp mode unchanged
  (delegates to rpc_manager.start_rpc); vllm mode -> 409 with a clear
  detail, mirroring the /api/launch vLLM refusal — never a silent no-op
  and never a 500.
- POST /api/start-all: llama.cpp mode starts RPC on every machine then
  launches llama-swap (unchanged); vllm mode SKIPS the RPC phase (the
  payload reports rpc: [{skipped: true, reason}]) but still launches the
  daemon.
- POST /api/models (vllm branch): engine.validate_hosts gates the save —
  Windows/macOS hosts (and ROCm hosts below Python 3.12) reject with 400.
  Host scope: explicit servers -> group machines -> all machines.
  llama.cpp mode has NO host gate (its adapter returns []).

Heavy externals (SSH, real process starts) are monkeypatched; the REAL
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


class _VllmEngine:
    """Minimal vllm adapter: delegate validate_hosts to the REAL
    vllm_manager implementation so the Engine×OS gate is exercised
    end-to-end, and stub the model-config validator."""
    name = "vllm"

    # DEC-04: save_model dispatches the per-model section round-trip
    # through the owning adapter; stub doubles carry the protocol members.
    def apply_model_section(self, data, where="vllm"):
        return data

    def sanitize_model_section(self, raw, where="vllm"):
        if raw is None:
            return None
        return dict(raw)

    def validate_model_config(self, cfg):
        return []

    def capable(self, capability):
        return capability == "swap_stop"

    def stop_model(self, model_id, servers, variant=None):
        import vllm_manager
        return vllm_manager.VllmAdapter.stop_model(
            self, model_id, servers, variant)

    def validate_hosts(self, machines):
        import vllm_manager
        return vllm_manager.validate_hosts(machines)


class _LcppEngine:
    name = "llama_cpp"

    # DEC-04: save_model dispatches the per-model section round-trip
    # through the owning adapter; stub doubles carry the protocol members.
    def apply_model_section(self, data, where="vllm"):
        return data

    def sanitize_model_section(self, raw, where="vllm"):
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


def test_rpc_start_vllm_refused_409(client, settings_file, monkeypatch):
    """vllm mode: 409 with a pointer to the llama-swap path — and
    rpc_manager.start_rpc is NEVER called (no llama.cpp processes)."""
    _engine(_VllmEngine(), monkeypatch)
    calls = []
    monkeypatch.setattr(app_module.rpc_manager, "start_rpc",
                        lambda name: calls.append(name))
    r = client.post("/api/machines/aibox1/rpc/start")
    assert r.status_code == 409, r.text
    assert "vLLM" in r.json()["detail"]
    assert calls == []


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


def test_start_all_vllm_skips_rpc_phase(client, settings_file, monkeypatch):
    """vllm mode: NO rpc_manager.start_rpc call at all; the payload marks
    the phase skipped and llama-swap still launches."""
    _engine(_VllmEngine(), monkeypatch)
    rpc_names = []
    monkeypatch.setattr(app_module.rpc_manager, "start_rpc",
                        lambda name: rpc_names.append(name))
    launches = []
    monkeypatch.setattr(app_module.swap_manager, "launch",
                        lambda *a, **k: launches.append(a) or
                        {"host": "aibox1", "port": 8199})
    r = client.post("/api/start-all")
    assert r.status_code == 200, r.text
    body = r.json()
    assert rpc_names == []                            # RPC phase skipped
    assert body["rpc"] == [{"skipped": True,
                            "reason": "vLLM mode: no RPC servers to start"}]
    assert len(launches) == 1                        # swap still launched
    assert body["swap"]["host"] == "aibox1"


# ---------------- Part B: validate_hosts in POST /api/models ---------------

def _model_payload(**over):
    base = {
        "model_id": "Qwen3-8B-vllm",
        "name": "Qwen3 8B (vLLM)",
        "base_path": "/models/Qwen3-8B",
        "servers": ["aibox1"],
        "vllm": {"max_model_len": 32768},
    }
    base.update(over)
    return base


def test_save_model_vllm_linux_hosts_pass(client, settings_file,
                                          monkeypatch):
    """vllm mode, linux servers: validate_hosts returns no errors — the
    save proceeds exactly as before (200, file written)."""
    _engine(_VllmEngine(), monkeypatch)
    _swap_down(monkeypatch)
    r = client.post("/api/models", json=_model_payload())
    assert r.status_code == 200, r.text
    assert (os.path.exists(_model_saved("Qwen3-8B-vllm")))


def test_save_model_vllm_windows_server_rejected_400(client, settings_file,
                                                     monkeypatch):
    """vllm mode, explicit servers include a Windows host: the REAL
    validate_hosts gate rejects the save with 400 naming the host — and
    no model file is written."""
    _engine(_VllmEngine(), monkeypatch)
    _swap_down(monkeypatch)
    _add_windows_machine()
    r = client.post("/api/models", json=_model_payload(
        servers=["aibox1", "winbox"]))
    assert r.status_code == 400, r.text
    assert "winbox" in r.json()["detail"]
    assert "Windows" in r.json()["detail"]
    assert not os.path.exists(_model_saved("Qwen3-8B-vllm"))


def test_save_model_vllm_group_machines_are_gated(client, settings_file,
                                                  monkeypatch):
    """servers empty + group set: the group's machines are the host scope —
    a windows group member rejects the save."""
    _engine(_VllmEngine(), monkeypatch)
    _swap_down(monkeypatch)
    _add_windows_machine()
    _add_group("winfarm", ["aibox1", "winbox"])
    r = client.post("/api/models", json=_model_payload(
        servers=[], group="winfarm"))
    assert r.status_code == 400, r.text
    assert "winbox" in r.json()["detail"]
    assert not os.path.exists(_model_saved("Qwen3-8B-vllm"))


def test_save_model_vllm_defaults_to_all_machines(client, settings_file,
                                                  monkeypatch):
    """servers empty + no group: every machine is the host scope."""
    _engine(_VllmEngine(), monkeypatch)
    _swap_down(monkeypatch)
    _add_windows_machine()
    r = client.post("/api/models", json=_model_payload(servers=[]))
    assert r.status_code == 400, r.text
    assert "winbox" in r.json()["detail"]


def test_save_model_vllm_explicit_servers_skip_untargeted_winbox(
        client, settings_file, monkeypatch):
    """vllm mode targeting ONLY linux hosts while a windows peer exists:
    the save passes — the gate covers the model's deployment targets,
    not the whole farm."""
    _engine(_VllmEngine(), monkeypatch)
    _swap_down(monkeypatch)
    _add_windows_machine()
    r = client.post("/api/models", json=_model_payload(servers=["aibox1"]))
    assert r.status_code == 200, r.text


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
    assert (os.path.exists(_model_saved("Qwen3-8B-vllm")))


# ---------------- B15: /api/launch/stop is engine-aware ---------------------

def test_launch_stop_vllm_never_calls_rpc_stop(client, settings_file,
                                               monkeypatch):
    """vllm mode: POST /api/launch/stop must NOT touch rpc_manager.stop_model
    (that kills llama.cpp RPC ports — the wrong stop path). The vllm stop is
    llama-swap's model unload: the swap-host unload proxy fires (llama-swap
    then executes the entry's cmdStop — vllm_manager.build_cmd_stop)."""
    _engine(_VllmEngine(), monkeypatch)
    proxied = []

    def fake_proxy(model_id):
        proxied.append(model_id)
        return {"ok": True, "status": 200, "body": ""}

    def boom(*a, **k):
        raise AssertionError(
            "rpc_manager.stop_model must not run in vllm mode")

    monkeypatch.setattr(app_module.rpc_manager, "stop_model", boom)
    monkeypatch.setattr(app_module.vllm_manager, "swap_host_unload",
                        fake_proxy)
    r = client.post("/api/launch/stop",
                    json={"model_id": "Qwen3-8B-vllm",
                          "servers": ["aibox1"], "port": 8080})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "status": 200, "body": ""}
    assert proxied == ["Qwen3-8B-vllm"]


def test_launch_stop_vllm_swap_down_is_503(client, settings_file,
                                           monkeypatch):
    """vllm mode with the swap daemon down: explicit 503 (never a silent
    no-op and never the llama.cpp kill path)."""
    _engine(_VllmEngine(), monkeypatch)
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    monkeypatch.setattr(app_module.rpc_manager, "stop_model",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("wrong stop path")))
    r = client.post("/api/launch/stop",
                    json={"model_id": "Qwen3-8B-vllm",
                          "servers": ["aibox1"], "port": 8080})
    assert r.status_code == 503, r.text
    assert "swap not running" in r.json()["detail"]


def test_launch_stop_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode, unconfigured id: the heuristic can't resolve the
    deployment, so stop keeps the legacy path — rpc_manager.stop_model
    with the same arguments, no swap proxying."""
    _engine(_LcppEngine(), monkeypatch)
    calls = {}

    def fake_stop(servers, model_id=None, variant=None, master_name=None):
        calls["args"] = (servers, model_id, variant, master_name)
        return [{"machine": servers[0], "stopped": True}]

    monkeypatch.setattr(app_module.rpc_manager, "stop_model", fake_stop)
    monkeypatch.setattr(app_module, "_proxy_swap_unload",
                        lambda model_id: (_ for _ in ()).throw(
                            AssertionError("llama_cpp stop must not proxy")))
    r = client.post("/api/launch/stop",
                    json={"model_id": "Qwen3-8B", "servers": ["aibox1"],
                          "port": 8080})
    assert r.status_code == 200, r.text
    assert calls["args"] == (["aibox1"], "Qwen3-8B", None, None)


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
    monkeypatch.setattr(app_module.rpc_manager, "stop_model",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("remote-proxy stop must proxy")))

    async def fake_unload(model_id):
        proxied.append(model_id)
        return {"ok": True}

    monkeypatch.setattr(app_module, "_proxy_swap_unload", fake_unload)
    _add_peer_aibox3()
    r = client.post("/api/launch/stop",
                    json={"model_id": "qwen3-8b", "servers": ["aibox3"],
                          "port": 8080})
    assert r.status_code == 200, r.text
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
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["body"]["max_tokens"] == 1


def test_vllm_launch_view_has_per_model_cards():
    """UI contract: #vllm-launch-view carries per-model launch cards
    (config subtitle + Load/Evict actions), not prose alone — the
    launch-grid idiom in vLLM vocabulary."""
    src = _ui_source("ui/static/app.js")
    assert "loadVllmLaunchCards" in src
    assert "vllm-launch-grid" in src
    assert 'data-act="vllm-load"' in src
    assert 'data-act="vllm-evict"' in src
    assert "/api/swap/load/" in src


# ---------------- Part C: FIX-A7 UI gate vocabulary -------------------------
# The A7 audit (t_fe83e6b0 comment 119) verdict: both reported defects are
# UI-gate placement. The frontend has no test harness, so these tests pin
# the CONTRACTS the frontend consumes: (1) the template must NOT tag
# #start-all with the blanket data-lcpp-only hide; (2) the vLLM adapter's
# ui_hints must hide the per-model launch grid (rpc_launch) — the same
# data-panel mechanism applyEngineHints drives from GET /api/settings.


def _ui_source(rel):
    """Read a UI source file straight off the repo tree (the template and
    app.js are plain text; the UI contract is greppable)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, rel), encoding="utf-8") as f:
        return f.read()


def test_start_all_button_not_lcpp_gated_in_template():
    """FIX-A7 (a): #start-all must NOT carry data-lcpp-only — under vLLM
    the button stays VISIBLE (its /api/start-all backend skips the RPC
    phase and still launches llama-swap; hiding it orphaned the one-click
    start). The old attribute made applyEngineHints hide it wholesale."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = open(os.path.join(here, "ui", "templates", "index.html"),
               encoding="utf-8").read()
    start_all_line = next(l for l in tpl.splitlines()
                          if 'id="start-all"' in l)
    assert "data-lcpp-only" not in start_all_line, start_all_line
    # and no other element silently inherited the gate either — the
    # attribute is retired from the template entirely
    assert "data-lcpp-only" not in tpl
    # the JS side no longer consumes it (relabel replaces the blanket hide)
    appjs = open(os.path.join(here, "ui", "static", "app.js"),
                 encoding="utf-8").read()
    assert "data-lcpp-only" not in appjs


def test_start_all_relabel_is_engine_aware_in_apply_engine_hints():
    """FIX-A7 (a): applyEngineHints must set an honest vLLM label — the
    action under vLLM is 'Start llama-swap' (RPC phase skipped), never a
    bare 'Start All' that implies RPC servers."""
    appjs = _ui_source(os.path.join("ui", "static", "app.js"))
    hints_fn = appjs.split("function applyEngineHints", 1)[1]
    # the relabel lives INSIDE applyEngineHints (runs on every settings
    # load AND on engine change via POST /api/settings) ...
    assert 'startAll.textContent = "Start llama-swap"' in hints_fn
    assert '#start-all' in hints_fn
    # ... and the click handler restores the same engine-aware label
    # instead of a hardcoded "Start All" (it fires after every action)
    restore = appjs.split('"#start-all"', 1)[1]
    assert 'engine === "vllm" ? "Start llama-swap" : "Start All"' in restore


def test_vllm_hides_rpc_launch_grid_panel():
    """FIX-A7 (b): the Launch tab per-model grid is rpc_launch data-panel
    vocabulary, and VllmAdapter.ui_hints() lists it in hidden_panels —
    under vLLM the grid hides instead of rendering RPC launch cards whose
    POST /api/launch dead-ends in the designed 409."""
    import vllm_manager
    hints = vllm_manager.VllmAdapter().ui_hints()
    assert "rpc_launch" in hints["hidden_panels"]
    # llama.cpp keeps the grid: its adapter hides nothing
    from engine_registry import LlamaCppAdapter
    assert LlamaCppAdapter().ui_hints()["hidden_panels"] == []
    # the template marks BOTH the grid and its hint paragraph so the
    # whole "start a single model directly" surface disappears together
    tpl = _ui_source(os.path.join("ui", "templates", "index.html"))
    marked = [l for l in tpl.splitlines() if 'data-panel="rpc_launch"' in l]
    assert any('id="launch-grid"' in l for l in marked), marked
    assert any("Or start a single model directly" in l for l in marked), marked


def test_settings_endpoint_rides_rpc_launch_hint(client, settings_file,
                                                 monkeypatch):
    """GET /api/settings under a stored engine=vllm must carry rpc_launch
    in hidden_panels — the frontend's applyEngineHints consumes THIS
    payload (appSettings.ui_manifest), not the adapter directly. Uses the
    REAL VllmAdapter so the full ride-through is exercised. DEC-02: the
    payload carries the engine's ui_manifest (the contract) with the
    deprecated ui_hints view riding along."""
    import yaml as _yaml  # app's own dependency; proves availability
    import vllm_manager as _vm
    _engine(_vm.VllmAdapter(), monkeypatch)
    with open(settings_file, "w", encoding="utf-8") as f:
        f.write(_yaml.safe_dump({"engine": "vllm"}))
    body = client.get("/api/settings").json()
    assert body["engine"] == "vllm"
    assert "rpc_launch" in body["ui_hints"]["hidden_panels"]
    # DEC-02 manifest ride-through: the complete inventory arrives too,
    # listing rpc_launch as hidden + the swap_only launch mode
    man = body["ui_manifest"]
    assert "rpc_launch" in man["hidden_panels"]
    assert man["launch"]["mode"] == "swap_only"
    assert man["model_editor"]["sections"] == [{"id": "vllm_section"}]
