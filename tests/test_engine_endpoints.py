"""B5a2: engine-aware status / models-save / launch endpoints in app.py.

Covers ARD §4 (endpoint changes) for the three B5a2 surfaces:
- GET /api/machines/{name}/status: llama.cpp mode returns the exact
  machine_status shape (byte-identical, no extra keys); vllm mode adds
  engine + vllm {running, detail} from the adapter probe on the
  llama-swap listen port.
- POST /api/models: vllm-mode adapter validation gates the save —
  errors -> 400 (no file written), warnings ride along in the response;
  the per-model vllm: section round-trips through save/load; llama.cpp
  mode keeps the byte-identical save path (no disk-based validation).
- POST /api/launch: vllm mode -> 409 with explanation (RPC-cluster
  launch is llama.cpp-only; vLLM routes through llama-swap).

Heavy externals (SSH probing, real HTTP /health) are monkeypatched; the
REAL config_loader round-trips the model YAML against tmp files.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402


# ---------------- fixtures ------------------------------------------------

@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Redirect BOTH settings files into the pytest tmp dir so tests
    never read or write the repo's real config/. app_module.SETTINGS_FILE
    is what POST /api/settings writes; engine_registry.SETTINGS_FILE is
    what get_engine() reads — one tmp path for both keeps them coherent."""
    import engine_registry
    eng = tmp_path / "settings.yaml"
    eng.write_text("")  # empty -> default engine (llama_cpp)
    # BOTH module globals MUST point here: a test that POSTs /api/settings
    # writes through app_module.SETTINGS_FILE — if that still points at
    # the repo file, the suite corrupts the user's real settings.yaml.
    monkeypatch.setattr(app_module, "SETTINGS_FILE", str(eng))
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", str(eng))

    swap = tmp_path / "swap.yaml"
    swap.write_text("listen_port: 8123\n")
    import swap_manager
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE", str(swap))

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
        "  detected:\n"
        "    gpu_names: [GPU0, GPU1]\n"
        "    vram_total_gb: 24.0\n"
        "    ram_total_gb: 64\n"
        "  expose: {mode: both, gpus: []}\n"
    )
    import config_loader
    # _YamlCollection reads self.path at call time -> point the machines
    # collection at the tmp file (no module-global filename attr exists)
    monkeypatch.setattr(config_loader.machines, "path", str(machines))
    import rpc_manager
    monkeypatch.setattr(rpc_manager, "cfg", config_loader)

    # models dir -> tmp (save_model/get_model resolve through it)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(app_module.cfg, "MODELS_DIR", str(models_dir))

    # groups: app's save_model checks cfg.groups.get — empty map
    monkeypatch.setattr(config_loader, "groups",
                        types.SimpleNamespace(all=lambda: [],
                                              get=lambda n: None))
    yield str(eng)


@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def _set_engine(client, engine, tmp_file):
    """Persist the engine via the REAL POST /api/settings (B5a1)."""
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(f"engine: {engine}\n")
    # force a fresh read (get_engine() with no args loads the file)
    import engine_registry


# ---------------- GET /api/machines/{name}/status ---------------------------

def test_machine_status_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode: the endpoint returns rpc_manager.machine_status
    output EXACTLY as before — no engine key, no probe call."""
    probe_calls = []

    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "llama_cpp"

        def capable(self, capability):
            return capability == "rpc_cluster"

        def probe(self, *a, **k):
            probe_calls.append(a)
            raise AssertionError("llama_cpp mode must not probe /health")

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())

    def fake_status(name):
        return {"name": name, "ip": "127.0.0.1", "reachable": True,
                "rpc_running": True, "gpus": [], "is_local": True,
                "is_master": True}

    monkeypatch.setattr(app_module.rpc_manager, "machine_status", fake_status)
    r = client.get("/api/machines/aibox1/status")
    assert r.status_code == 200
    body = r.json()
    assert body == {"name": "aibox1", "ip": "127.0.0.1", "reachable": True,
                    "rpc_running": True, "gpus": [], "is_local": True,
                    "is_master": True}
    assert "engine" not in body and "vllm" not in body
    assert probe_calls == []


def test_machine_status_vllm_probe_shape(client, settings_file, monkeypatch):
    """vllm mode: status gains engine='vllm' + vllm {running, detail},
    probed on the llama-swap listen_port from swap settings (default
    8080 when unset)."""
    seen = {}

    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "vllm"
        engine_id = "vllm"

        def capable(self, capability):
            return capability == "swap_stop"

        def probe(self, machine, port, is_local):
            seen["machine"] = machine
            seen["port"] = port
            seen["is_local"] = is_local
            return {"running": True, "detail": "vLLM /health 200"}

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(
        app_module.rpc_manager, "machine_status",
        lambda name: {"name": name, "ip": "127.0.0.1", "reachable": True,
                      "rpc_running": False, "gpus": [], "is_local": True,
                      "is_master": True})
    r = client.get("/api/machines/aibox1/status")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "vllm"
    assert body["vllm"] == {"running": True, "detail": "vLLM /health 200"}
    # port came from swap_manager.load_settings()["listen_port"]
    assert seen["port"] == 8123
    assert seen["machine"]["name"] == "aibox1"
    # machine dict resolved from the machines map, not the raw name
    assert "ip" in seen["machine"]


def test_machine_status_vllm_probe_failure_is_reported(client, settings_file,
                                                        monkeypatch):
    """A crashing probe must not 500 the status endpoint — the machine
    is still reachable; the probe detail says what failed."""
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "vllm"
        engine_id = "vllm"

        def capable(self, capability):
            return capability == "swap_stop"

        def probe(self, *a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(
        app_module.rpc_manager, "machine_status",
        lambda name: {"name": name, "ip": "127.0.0.1", "reachable": True,
                      "rpc_running": False, "gpus": [], "is_local": True,
                      "is_master": True})
    r = client.get("/api/machines/aibox1/status")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "vllm"
    assert body["vllm"]["running"] is False
    assert "probe failed" in body["vllm"]["detail"]


def test_machine_status_unknown_machine_404(client, settings_file, monkeypatch):
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "llama_cpp"

        def capable(self, capability):
            return capability == "rpc_cluster"

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())

    def boom(name):
        raise ValueError(f"machine '{name}' not found")

    monkeypatch.setattr(app_module.rpc_manager, "machine_status", boom)
    r = client.get("/api/machines/nope/status")
    assert r.status_code == 404


# ---------------- POST /api/models ------------------------------------------

def _model_payload(**over):
    base = {
        "model_id": "Qwen3-8B-vllm",
        "name": "Qwen3 8B (vLLM)",
        "base_path": "/models/Qwen3-8B",
        "servers": ["aibox1"],
        "device_order": "aibox1-CUDA0",
        "flags": {"ctx-size": 32768},
        "vllm": {"max_model_len": 32768, "max_num_seqs": 256},
    }
    base.update(over)
    return base


def test_save_model_vllm_section_roundtrip(client, settings_file, monkeypatch):
    """The per-model vllm: section survives the save -> load round-trip
    through the REAL config_loader (B4 vocabulary), and the response
    carries the saved section back."""
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "vllm"
        engine_id = "vllm"

        def capable(self, capability):
            return capability == "swap_stop"

        def validate_model_config(self, cfg):
            return []

        # G-10: the save flow now runs the Engine×OS gate before the
        # config validator; the fixture farm has only the linux aibox1.
        def validate_hosts(self, machines):
            return []

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    # swap not running -> save skips the push/reload path (no daemon I/O)
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/models", json=_model_payload())
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["vllm"] == {"max_model_len": 32768, "max_num_seqs": 256}
    assert saved["_validation_warnings"] == []
    # round-trip: GET /api/models/{id} returns the same section
    g = client.get("/api/models/Qwen3-8B-vllm")
    assert g.status_code == 200
    assert g.json()["vllm"] == {"max_model_len": 32768, "max_num_seqs": 256}


def test_save_model_vllm_error_rejects_400(client, settings_file, monkeypatch):
    """vllm-mode adapter errors reject the save with 400 BEFORE any file
    write — nothing lands in the models dir."""
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "vllm"
        engine_id = "vllm"

        def capable(self, capability):
            return capability == "swap_stop"

        def validate_hosts(self, machines):
            return []  # G-10 host gate passes: fixture farm is linux-only

        def validate_model_config(self, cfg):
            return [{"level": "error",
                      "msg": "vLLM serves one machine (got 2: aibox1, b2)"}]

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/models", json=_model_payload())
    assert r.status_code == 400
    assert "one machine" in r.json()["detail"]
    # no file was written
    import config_loader
    files = os.listdir(config_loader.MODELS_DIR)
    assert files == [], f"rejected save must not write files, got {files}"


def test_save_model_vllm_warns_ride_along(client, settings_file, monkeypatch):
    """Adapter warnings do NOT block the save; they ride along in the
    response under _validation_warnings."""
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "vllm"
        engine_id = "vllm"

        def capable(self, capability):
            return capability == "swap_stop"

        def validate_hosts(self, machines):
            return []  # G-10 host gate passes: fixture farm is linux-only

        def validate_model_config(self, cfg):
            return [{"level": "warn", "msg": "GGUF on vLLM is experimental"}]

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/models", json=_model_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["_validation_warnings"] == ["GGUF on vLLM is experimental"]
    g = client.get("/api/models/Qwen3-8B-vllm")
    assert g.status_code == 200  # save went through


def test_save_model_real_adapter_rejects_multi_server(client, settings_file,
                                                      monkeypatch):
    """The REAL VllmAdapter.validate_model_config (not a stub) rejects a
    multi-machine servers list that lacks chip coverage — a multi-host
    config is legal only with GPU chips selected on every host
    (device_order); 'b2' has none, so the save is refused with the
    per-host chip error."""
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    # write engine: vllm into the real settings file the registry reads
    with open(settings_file, "w", encoding="utf-8") as f:
        f.write("engine: vllm\n")
    r = client.post("/api/models",
                    json=_model_payload(servers=["aibox1", "b2"]))
    assert r.status_code == 400
    assert "b2" in r.json()["detail"]


def test_save_model_llama_cpp_path_unchanged(client, settings_file,
                                              monkeypatch):
    """llama.cpp mode: NO adapter validation runs on save (its checks are
    disk-based, POST /api/models/{id}/test owns them) — a payload with
    zero servers saves fine, byte-identical to pre-B5a2 behavior. The
    vllm: key is absent from both request and response."""
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "llama_cpp"

        def capable(self, capability):
            return capability == "rpc_cluster"

        def validate_model_config(self, cfg):
            raise AssertionError(
                "llama_cpp save path must not call validate_model_config")

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/models", json=_model_payload(vllm=None, servers=[]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "_validation_warnings" in body
    assert body.get("vllm") is None


def test_save_model_real_adapter_rejects_bad_kv_dtype(client, settings_file,
                                                      monkeypatch):
    """Unknown vllm keys and bad kv_cache_dtype values are rejected by
    the REAL adapter (fp8-only pitfall from research)."""
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    with open(settings_file, "w", encoding="utf-8") as f:
        f.write("engine: vllm\n")
    r = client.post("/api/models", json=_model_payload(
        vllm={"kv_cache_dtype": "q8_0", "max_model_len": 4096}))
    assert r.status_code == 400
    assert "kv_cache_dtype" in r.json()["detail"]
    # unknown key
    r2 = client.post("/api/models", json=_model_payload(
        vllm={"nope": 1}))
    assert r2.status_code == 400
    assert "unknown" in r2.json()["detail"]


# ---------------- DEC-09: save-path translation retired -----------------
# B11c's auto-translation (engine_config_translate.to_vllm / to_llamacpp
# on POST /api/models) is deleted per the user directive: zero
# cross-engine warning groups — separation, not better warnings. Each
# engine saves its own vocabulary verbatim. Replaced below by tests that
# pin the new contract: no translation, no _translation_warnings key.


def test_save_model_vllm_no_translation_no_vllm_section(
        client, settings_file, monkeypatch):
    """DEC-09: engine=vllm + llama.cpp flags + no vllm: section -> the
    flags are saved verbatim, NOTHING is translated into a vllm: section
    (the loud failure happens at launch, not save), and the response
    carries no _translation_warnings key."""
    class _Engine:
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "vllm"
        engine_id = "vllm"

        def capable(self, capability):
            return capability == "swap_stop"

        def validate_model_config(self, cfg):
            return []

        def validate_hosts(self, machines):
            return []

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/models", json=_model_payload(
        flags={"ctx-size": 16384, "ngl": 99}, vllm=None))
    assert r.status_code == 200, r.text
    body = r.json()
    # no translation happened — but a NEW vllm model without a section
    # now gets the live-validated farm defaults filled in (see
    # vllm_manager.vllm_new_model_defaults); llama.cpp flags stay
    # verbatim and no warnings ride along.
    assert body.get("vllm") == {
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "dtype": "float32",
        "enforce_eager": True,
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",
        "parallel_layout": "pp",
    }
    assert body["flags"]["ctx-size"] == 16384
    assert "_translation_warnings" not in body
    assert body["_validation_warnings"] == []


def test_save_model_llama_cpp_no_translation_from_vllm_section(
        client, settings_file, monkeypatch):
    """DEC-09: engine=llama_cpp + vllm: section + no llama.cpp flags ->
    to_llamacpp no longer runs; flags stay empty and the vllm: section is
    preserved verbatim (not translated, not stripped)."""
    class _Engine:
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "llama_cpp"

        def capable(self, capability):
            return capability == "rpc_cluster"

        def validate_model_config(self, cfg):
            raise AssertionError("llama_cpp save must not validate")

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/models", json=_model_payload(
        flags={}, servers=[],
        vllm={"max_model_len": 8192, "max_num_seqs": 64}))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["flags"] == {}
    assert body["vllm"]["max_model_len"] == 8192
    assert "_translation_warnings" not in body
    assert body["_validation_warnings"] == []


# ---------------- POST /api/launch ------------------------------------------

def test_launch_vllm_mode_returns_409(client, settings_file, monkeypatch):
    """vllm mode: POST /api/launch answers 409 with an explanation
    pointing at the llama-swap route — rpc_manager.launch_model is
    never called."""
    def boom(*a, **k):
        raise AssertionError("launch_model must not run in vllm mode")

    monkeypatch.setattr(app_module.rpc_manager, "launch_model", boom)
    with open(settings_file, "w", encoding="utf-8") as f:
        f.write("engine: vllm\n")
    r = client.post("/api/launch",
                    json={"model_id": "Qwen3-8B-vllm",
                          "servers": ["aibox1"], "port": 8080})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "llama-swap" in detail


def test_launch_llama_cpp_mode_still_works(client, settings_file, monkeypatch):
    """llama.cpp mode: the 409 gate is inert — launch still reaches
    rpc_manager.launch_model with the same arguments."""
    calls = {}

    def fake_launch(model_id, servers, port, variant=None, master_name=None,
                    mode="rpc"):
        calls["args"] = (model_id, servers, port, variant, master_name,
                         mode)
        return {"ok": True, "failed": False}

    monkeypatch.setattr(app_module.rpc_manager, "launch_model", fake_launch)
    r = client.post("/api/launch",
                    json={"model_id": "Qwen3-8B", "servers": ["aibox1"],
                          "port": 8080})
    assert r.status_code == 200
    # unconfigured id: auto falls back to the legacy rpc default
    assert calls["args"] == ("Qwen3-8B", ["aibox1"], 8080, None, None, "rpc")


def test_launch_vllm_409_preempts_validation(client, settings_file):
    """The 409 fires before any server-list validation: even a bogus
    server name gets 409 in vllm mode (the mode gate is first)."""
    with open(settings_file, "w", encoding="utf-8") as f:
        f.write("engine: vllm\n")
    r = client.post("/api/launch",
                    json={"model_id": "x", "servers": ["ghost"], "port": 1})
    assert r.status_code == 409


# ---------------- settings round-trip keeps the engine switch working -------

def test_engine_switch_via_settings_endpoint(client, settings_file):
    """B5a1 surface still works after B5a2 (regression): POST
    /api/settings persists engine, and GET reflects it — the same
    settings the 409/status/models gates read."""
    r = client.post("/api/settings", json={"engine": "vllm"})
    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "vllm"
    g = client.get("/api/settings")
    assert g.json()["engine"] == "vllm"


# ---------------- GET /api/machines (B8/B14: vllm capability payload) --------

def _add_windows_machine(settings_file, monkeypatch):
    """Extend the fixture machines.yaml with a windows peer (winbox)."""
    import config_loader
    machines = config_loader.machines.path
    with open(machines, "a", encoding="utf-8") as f:
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
    return machines


def _add_machine(machines_file, **kw):
    """Append one machine entry to the fixture machines.yaml; unknown
    fields default like the add-machine wizard does (os linux, vendor
    nvidia). Returns the file path."""
    fields = {"role": "peer", "os": "linux", "rpc_port": 50052, **kw}
    lines = [f"- id: {fields['name']}", f"  name: {fields['name']}",
             f"  ip: {fields.get('ip', '10.0.0.199')}"]
    for k, v in fields.items():
        if k in ("name", "ip"):
            continue
        if isinstance(v, str):
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {v}")
    with open(machines_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return machines_file


def test_machine_list_flags_windows_not_vllm_capable(client, settings_file,
                                                     monkeypatch):
    """B8: a windows-os machine in the list reports vllm_capable=false
    (vLLM has no native Windows support) while linux machines stay true —
    the flag the UI uses to grey out Windows hosts when engine=vllm."""
    _add_windows_machine(settings_file, monkeypatch)
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["winbox"]["vllm_capable"] is False
    assert by_name["aibox1"]["vllm_capable"] is True


def test_machine_list_flag_is_response_only(client, settings_file,
                                            monkeypatch):
    """B8: the flag must not leak into the machine store / YAML on disk —
    it's derived at response time from os, never persisted."""
    machines_file = _add_windows_machine(settings_file, monkeypatch)
    before = open(machines_file, encoding="utf-8").read()
    r = client.get("/api/machines")
    assert r.status_code == 200
    assert all("vllm_capable" in m for m in r.json())
    # disk byte-identical after the call
    assert open(machines_file, encoding="utf-8").read() == before
    # and the in-memory store is untouched too
    import config_loader
    assert all("vllm_capable" not in m for m in config_loader.machines.all())


def test_machine_list_flag_present_in_llama_cpp_mode(client, settings_file,
                                                     monkeypatch):
    """B8: the flag rides along in llama.cpp mode too — the UI greys out
    only when engine=vllm, but the payload contract is engine-independent
    (os-derived, no engine read needed)."""
    _add_windows_machine(settings_file, monkeypatch)
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["winbox"]["vllm_capable"] is False
    assert by_name["aibox1"]["vllm_capable"] is True


def test_machine_list_os_normalization(client, settings_file, monkeypatch):
    """B8: os casing/whitespace variants ('Windows', ' WINDOWS ') are
    still caught — the gate must not depend on the YAML's exact spelling."""
    import config_loader
    machines_file = _add_windows_machine(settings_file, monkeypatch)
    text = open(machines_file, encoding="utf-8").read()
    text = text.replace("  os: windows\n", "  os: ' Windows '\n")
    open(machines_file, "w", encoding="utf-8").write(text)
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["winbox"]["vllm_capable"] is False


# ---------------- B14: full hardware capability payload (audit §3b) ----------

def test_machine_list_payload_shape_has_reason_and_info(client,
                                                        settings_file,
                                                        monkeypatch):
    """B14 payload shape: every entry carries vllm_capable plus
    vllm_capability_reason ('' when capable) and vllm_capability_info
    ('' unless caveated) — reason strings surface verbatim in the payload."""
    _add_windows_machine(settings_file, monkeypatch)
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    # incapable: reason populated verbatim, info empty
    assert by_name["winbox"]["vllm_capable"] is False
    assert by_name["winbox"]["vllm_capability_reason"] == \
        "vLLM requires Linux (WSL2 for Windows hosts)"
    assert by_name["winbox"]["vllm_capability_info"] == ""
    # capable: reason empty
    assert by_name["aibox1"]["vllm_capable"] is True
    assert by_name["aibox1"]["vllm_capability_reason"] == ""
    assert by_name["aibox1"]["vllm_capability_info"] == ""


def test_machine_list_macos_not_capable_with_reason(client, settings_file,
                                                    monkeypatch):
    """B14: a macOS machine reports vllm_capable=false — the farm never
    installs vLLM-Metal — with the audit's reason verbatim."""
    import config_loader
    machines_file = config_loader.machines.path
    _add_machine(machines_file, name="macbox", os="macos",
                 gpu_vendor="apple")
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["macbox"]["vllm_capable"] is False
    assert by_name["macbox"]["vllm_capability_reason"] == (
        "vLLM does not support macOS in the farm — vLLM-Metal is a separate "
        "package the farm does not install; use llama.cpp (Metal) or the "
        "MLX runtime")
    assert by_name["macbox"]["vllm_capability_info"] == ""


def test_machine_list_nvidia_linux_capable(client, settings_file,
                                           monkeypatch):
    """B14: linux + nvidia stays fully capable, no reason, no info."""
    import config_loader
    _add_machine(config_loader.machines.path, name="nvbox",
                 gpu_vendor="nvidia")
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["nvbox"]["vllm_capable"] is True
    assert by_name["nvbox"]["vllm_capability_reason"] == ""
    assert by_name["nvbox"]["vllm_capability_info"] == ""


def test_machine_list_rocm_capable_with_info_caveat(client, settings_file,
                                                    monkeypatch):
    """B14: linux + amd (ROCm) stays CAPABLE — never greyed out — but the
    payload carries the ROCm wheels caveat in vllm_capability_info for the
    tooltip."""
    import config_loader
    _add_machine(config_loader.machines.path, name="amdbbox",
                 gpu_vendor="amd")
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["amdbbox"]["vllm_capable"] is True
    assert by_name["amdbbox"]["vllm_capability_reason"] == ""
    info = by_name["amdbbox"]["vllm_capability_info"]
    assert "Python 3.12" in info
    assert "wheels.vllm.ai/rocm" in info
    assert "HIP" in info


def test_machine_list_intel_not_capable_with_reason(client, settings_file,
                                                    monkeypatch):
    """B14: linux + intel gpu_vendor -> vllm_capable=false, llama.cpp
    (SYCL) reason verbatim."""
    import config_loader
    _add_machine(config_loader.machines.path, name="intelbox",
                 gpu_vendor="intel")
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["intelbox"]["vllm_capable"] is False
    assert by_name["intelbox"]["vllm_capability_reason"] == (
        "vLLM does not support this host via the farm installer — "
        "llama.cpp (SYCL) required")


def test_machine_list_cpu_only_not_capable_with_reason(client, settings_file,
                                                       monkeypatch):
    """B14: linux + cpu-only vendor -> vllm_capable=false, llama.cpp
    (CPU build) reason verbatim."""
    import config_loader
    _add_machine(config_loader.machines.path, name="cpubox",
                 gpu_vendor="cpu")
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["cpubox"]["vllm_capable"] is False
    assert by_name["cpubox"]["vllm_capability_reason"] == (
        "vLLM does not support CPU-only hosts — llama.cpp (CPU build) "
        "required")


def test_machine_list_unknown_vendor_capable_with_runtime_note(client,
                                                               settings_file,
                                                               monkeypatch):
    """B14: linux with gpu_vendor unset ('') stays CAPABLE (not greyed)
    but vllm_capability_info tells the user the install gates at runtime
    on nvidia-smi / rocm-smi."""
    import config_loader
    _add_machine(config_loader.machines.path, name="vagubox",
                 gpu_vendor="")
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["vagubox"]["vllm_capable"] is True
    assert by_name["vagubox"]["vllm_capability_reason"] == ""
    info = by_name["vagubox"]["vllm_capability_info"]
    assert "nvidia-smi" in info and "rocm-smi" in info
    assert "gate at runtime" in info


def test_machine_list_vendor_normalization_catches_rocm_spelling(client,
                                                                 settings_file,
                                                                 monkeypatch):
    """B14: vendor spelling variants normalize — gpu_vendor 'rocm' maps to
    amd (capable + ROCm info), 'sycl'/'xpu' to intel (incapable) — the
    gate cannot be dodged by YAML spelling."""
    import config_loader
    machines_file = config_loader.machines.path
    _add_machine(machines_file, name="rocmspelled", gpu_vendor="rocm")
    _add_machine(machines_file, name="xpubox", gpu_vendor="xpu")
    r = client.get("/api/machines")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["rocmspelled"]["vllm_capable"] is True
    assert "ROCm wheels" in by_name["rocmspelled"]["vllm_capability_info"]
    assert by_name["xpubox"]["vllm_capable"] is False
    assert "SYCL" in by_name["xpubox"]["vllm_capability_reason"]


def test_machine_list_capability_flags_never_persisted(client, settings_file,
                                                       monkeypatch):
    """B14: the three capability fields are response-only — derived at
    response time, never written to the machine store/YAML on disk."""
    import config_loader
    machines_file = _add_machine(config_loader.machines.path,
                                 name="probebox", gpu_vendor="cpu")
    before = open(machines_file, encoding="utf-8").read()
    r = client.get("/api/machines")
    assert r.status_code == 200
    assert all("vllm_capability_reason" in m for m in r.json())
    # disk byte-identical after the call
    assert open(machines_file, encoding="utf-8").read() == before
    # and the in-memory store is untouched too
    assert all("vllm_capable" not in m
               for m in config_loader.machines.all())

_CANNED_REPORT = {
    "machine": "aibox1",
    "delivery": "venv",
    "target": "~/vllm-env/bin/vllm",
    "steps": [
        {"step": "gate:os", "detail": "os linux: supported"},
        {"step": "verify:existing", "detail": "~/vllm-env/bin/vllm not "
                                             "usable yet (no output)"},
        {"step": "gate:gpu", "detail": "nvidia driver ok: 550.54.15, RTX 3090"},
        {"step": "venv:create", "detail": "venv ready: ~/vllm-env via uv"},
        {"step": "venv:install", "detail": "installed vllm==0.30.0 into "
                                           "~/vllm-env"},
        {"step": "verify:version", "detail": "vLLM 0.30.0"},
        {"step": "settings:update", "detail": "settings vllm.venv_bin -> "
                                              "~/vllm-env/bin/vllm"},
    ],
    "failed": False,
    "ok": True,
    "detail": "install complete",
}

_WINDOWS_REFUSAL_REPORT = {
    "machine": "winbox",
    "delivery": "venv",
    "steps": [{"step": "gate:os",
               "failed": True,
               "detail": "vLLM cannot run natively on Windows — install "
                         "WSL2, add the WSL instance to the farm as a "
                         "linux machine, and install vLLM there"}],
    "failed": True,
    "ok": False,
    "detail": "vLLM cannot run natively on Windows — install WSL2, add "
              "the WSL instance to the farm as a linux machine, and "
              "install vLLM there",
}


def _patch_vllm_installer(monkeypatch, report):
    """Fake vllm_installer module injected into sys.modules — the endpoint
    imports it lazily inside the handler, so the import must resolve to
    the fake. Records the machine dicts it was called with."""
    import types
    calls = []

    def fake_install_on(machine, *a, **k):
        calls.append(machine)
        return dict(report, machine=(machine or {}).get("name",
                                                        report["machine"]))

    fake = types.ModuleType("vllm_installer")
    fake.install_on = fake_install_on
    monkeypatch.setitem(sys.modules, "vllm_installer", fake)
    return calls


def test_install_vllm_returns_canned_report(client, settings_file,
                                            monkeypatch):
    """B10a single-machine endpoint: mocked installer's run_steps report
    passes through verbatim with 200 on ok=True."""
    calls = _patch_vllm_installer(monkeypatch, _CANNED_REPORT)
    r = client.post("/api/machines/aibox1/install-vllm")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["machine"] == "aibox1"
    assert body["delivery"] == "venv"
    assert [s["step"] for s in body["steps"]] == [
        s["step"] for s in _CANNED_REPORT["steps"]]
    assert body["detail"] == "install complete"
    # the machine dict from the store reached the installer
    assert len(calls) == 1 and calls[0]["name"] == "aibox1"


def test_install_vllm_windows_refusal_report_not_500(client, settings_file,
                                                     monkeypatch):
    """A windows machine is a failed install (ok=false -> 502) whose
    body is the installer's refusal report — never a 500/raise."""
    _add_windows_machine(settings_file, monkeypatch)
    _patch_vllm_installer(monkeypatch, _WINDOWS_REFUSAL_REPORT)
    r = client.post("/api/machines/winbox/install-vllm")
    assert r.status_code == 502
    body = r.json()
    assert body["ok"] is False
    assert "WSL2" in body["detail"]
    assert body["steps"][0]["step"] == "gate:os"


def test_install_vllm_unknown_machine_404(client, settings_file,
                                          monkeypatch):
    calls = _patch_vllm_installer(monkeypatch, _CANNED_REPORT)
    r = client.post("/api/machines/ghost/install-vllm")
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]
    assert calls == []  # installer never ran


def test_install_vllm_blank_name_404(client, settings_file, monkeypatch):
    """A blank (%20) name passes the emptiness guard as ' ' but fails the
    machine lookup -> 404; the installer never runs. (httpx normalizes
    a truly empty path segment, so %20 is the closest expressible case.)"""
    calls = _patch_vllm_installer(monkeypatch, _CANNED_REPORT)
    r = client.post("/api/machines/%20/install-vllm")
    assert r.status_code == 404
    assert calls == []


_CANNED_UNINSTALL_REPORT = {
    "machine": "aibox1",
    "delivery": "venv",
    "target": "~/vllm-env/bin/vllm",
    "steps": [
        {"step": "guard:path", "detail": "venv root ~/vllm-env is "
                                         "farm-managed"},
        {"step": "venv:remove", "detail": "venv ~/vllm-env removed"},
        {"step": "settings:clear", "detail": "settings vllm.venv_bin "
                                             "cleared (was "
                                             "~/vllm-env/bin/vllm)"},
    ],
    "failed": False,
    "ok": True,
    "detail": "vLLM uninstall complete",
}


def _patch_vllm_uninstaller(monkeypatch, report):
    """Fake vllm_uninstaller module injected into sys.modules — the
    endpoint imports it lazily inside the handler. Records the machine
    dicts it was called with."""
    import types
    calls = []

    def fake_uninstall_on(machine, *a, **k):
        calls.append(machine)
        return dict(report, machine=(machine or {}).get("name",
                                                        report["machine"]))

    fake = types.ModuleType("vllm_uninstaller")
    fake.uninstall_on = fake_uninstall_on
    monkeypatch.setitem(sys.modules, "vllm_uninstaller", fake)
    return calls


def test_uninstall_vllm_returns_canned_report(client, settings_file,
                                              monkeypatch):
    """B12 single-machine endpoint: mocked uninstaller's run_steps report
    passes through verbatim with 200 on ok=True."""
    calls = _patch_vllm_uninstaller(monkeypatch, _CANNED_UNINSTALL_REPORT)
    r = client.post("/api/machines/aibox1/uninstall-vllm")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["machine"] == "aibox1"
    assert body["delivery"] == "venv"
    assert [s["step"] for s in body["steps"]] == [
        s["step"] for s in _CANNED_UNINSTALL_REPORT["steps"]]
    assert body["detail"] == "vLLM uninstall complete"
    # the machine dict from the store reached the uninstaller
    assert len(calls) == 1 and calls[0]["name"] == "aibox1"


def test_uninstall_vllm_failure_report_not_500(client, settings_file,
                                               monkeypatch):
    """ok=false from the uninstaller (e.g. foreign-venv guard refusal)
    maps to 502 with the refusal report body — never a 500/raise."""
    refusal = {
        "machine": "aibox1",
        "delivery": "venv",
        "steps": [{"step": "guard:path", "failed": True,
                   "detail": "refusing to remove venv"}],
        "failed": True,
        "ok": False,
        "detail": "refusing to remove venv",
    }
    _patch_vllm_uninstaller(monkeypatch, refusal)
    r = client.post("/api/machines/aibox1/uninstall-vllm")
    assert r.status_code == 502
    body = r.json()
    assert body["ok"] is False
    assert "refusing" in body["detail"]


def test_uninstall_vllm_unknown_machine_404(client, settings_file,
                                            monkeypatch):
    calls = _patch_vllm_uninstaller(monkeypatch, _CANNED_UNINSTALL_REPORT)
    r = client.post("/api/machines/ghost/uninstall-vllm")
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]
    assert calls == []  # uninstaller never ran


def test_install_vllm_all_mixed_fleet(client, settings_file, monkeypatch):
    """B10a batch endpoint: linux machines get the installer (canned
    report + status tag), non-linux machines are skipped with the
    linux-only message, and the summary counts add up."""
    _add_windows_machine(settings_file, monkeypatch)
    calls = _patch_vllm_installer(monkeypatch, _CANNED_REPORT)
    r = client.post("/api/install-vllm/all")
    assert r.status_code == 200, r.text
    body = r.json()
    by_name = {res["machine"]: res for res in body["results"]}
    assert set(by_name) == {"aibox1", "winbox"}
    # linux machine: the installer ran with the real machine dict
    assert by_name["aibox1"]["status"] == "installed"
    assert by_name["aibox1"]["ok"] is True
    assert by_name["aibox1"]["detail"] == "install complete"
    assert [c["name"] for c in calls] == ["aibox1"]
    # windows machine: skipped up front, no installer call
    assert by_name["winbox"]["status"] == "skipped"
    assert by_name["winbox"]["ok"] is False
    assert "linux-only" in by_name["winbox"]["detail"]
    assert "1 installed" in body["summary"]
    assert "1 skipped" in body["summary"]
    assert body["missing"] == []


def test_install_vllm_all_reports_failures(client, settings_file,
                                           monkeypatch):
    """A failed install surfaces in summary + missing (the llama.cpp
    batch response keys), with the installer's own detail preserved."""
    bad = dict(_CANNED_REPORT, ok=False, failed=True,
               detail="nvidia-smi is missing or fails — install/verify "
                      "the NVIDIA driver before installing vLLM")
    bad["steps"] = [dict(s) for s in _CANNED_REPORT["steps"]]
    bad["steps"][2] = {"step": "gate:gpu", "failed": True,
                       "detail": bad["detail"]}
    _patch_vllm_installer(monkeypatch, bad)
    r = client.post("/api/install-vllm/all")
    assert r.status_code == 200, r.text  # batch itself succeeds; the
    # per-machine report carries the failure
    body = r.json()
    assert body["results"][0]["status"] == "install-failed"
    assert "nvidia-smi" in body["results"][0]["detail"]
    assert body["missing"] == ["aibox1"]
    assert "1 failed" in body["summary"]


def test_install_vllm_all_no_machines(client, settings_file, monkeypatch):
    """Empty fleet: 200 with an empty result list (llama.cpp batch
    contract)."""
    import config_loader
    monkeypatch.setattr(config_loader.machines, "all", lambda: [])
    r = client.post("/api/install-vllm/all")
    assert r.status_code == 200
    assert r.json() == {"results": [],
                        "message": "no machines configured"}


def test_tools_verify_shape_and_not_found(client, settings_file, monkeypatch):
    """GET /api/tools/verify: read-only status for the three tools with
    the §8 contract shape {found, path, version, expected, configured}.
    Empty tmp settings -> nothing configured, nothing found (tmp home has
    no tool binaries — the RealFS probe runs against the test HOME)."""
    import os
    # RealFS expanduser resolves through os.path.expanduser; pin HOME so
    # the well-known locations (~/.local/bin/...) land in the tmp dir.
    # (monkeypatch fixture: HOME is restored after the test — an unmanaged
    # MonkeyPatch here leaked into later tests and broke installer-probe.)
    monkeypatch.setenv("HOME", os.path.dirname(settings_file))
    r = client.get("/api/tools/verify")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"llama.cpp", "llama-swap", "vllm"}
    for tool, entry in body.items():
        assert set(entry) == {"found", "path", "version", "expected",
                              "configured"}, (tool, entry)
        assert entry["found"] is False, (tool, entry)
    assert body["llama.cpp"]["configured"] is None
    assert body["vllm"]["expected"] == "0.30.0"
    assert body["llama-swap"]["expected"] == "255+"
    assert "b10000" in body["llama.cpp"]["expected"]


def test_tools_verify_settings_record_counts(client, settings_file,
                                               monkeypatch):
    """A configured path that EXISTS counts as found even when the
    well-known locations miss — the settings record (§6 prefill) is
    probed as fallback and its --version becomes the version."""
    # write a fake vllm binary + point settings at it
    import os, stat, yaml as _yaml
    # pin HOME like the shape test so well-known locations miss and the
    # settings record is the only thing the probe can find (fixture-
    # scoped: restored after the test, no leakage into installer-probe)
    monkeypatch.setenv("HOME", os.path.dirname(settings_file))
    bindir = os.path.join(os.path.dirname(settings_file), "bin")
    os.makedirs(bindir, exist_ok=True)
    fake = os.path.join(bindir, "vllm")
    with open(fake, "w") as f:
        f.write("#!/bin/sh\necho 'vLLM 0.30.0'\n")
    os.chmod(fake, 0o755)
    with open(settings_file, "w") as f:
        _yaml.safe_dump({"vllm": {"venv_bin": fake}}, f)
    r = client.get("/api/tools/verify")
    assert r.status_code == 200
    v = r.json()["vllm"]
    assert v["found"] is True, v
    assert v["path"] == fake
    assert v["version"] == "0.30.0"
    assert v["configured"] == fake


def test_install_vllm_all_machines_subset(client, settings_file,
                                          monkeypatch):
    """The install modal's subset: `machines` limits check/install to
    the ticked hosts; unticked machines are never touched (no installer
    call, absent from results)."""
    _add_windows_machine(settings_file, monkeypatch)
    calls = _patch_vllm_installer(monkeypatch, _CANNED_REPORT)
    r = client.post("/api/install-vllm/all",
                    json={"install_missing": True, "machines": ["winbox"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [res["machine"] for res in body["results"]] == ["winbox"]
    assert calls == []  # skipped up front, installer never ran
    # unknown names filter everything out -> empty contract
    r = client.post("/api/install-vllm/all",
                    json={"machines": ["no-such-box"]})
    assert r.status_code == 200
    assert r.json() == {"results": [],
                        "message": "no machines configured"}


def test_install_vllm_single_real_installer_contract(client, settings_file,
                                                     monkeypatch):
    """The REAL vllm_installer.install_on (not a stub) runs against the
    endpoint: machine dict from the store + tmp settings file, skip probe
    first — end-to-end wiring proof (FakeRunner-level, no network)."""
    import vllm_installer as real_mod

    class _FakeRunner:
        """run() -> (code, out) scripted like tests/test_vllm_installer."""
        def __init__(self):
            self.calls = []

        def run(self, cmd, timeout=60):
            self.calls.append(cmd)
            if "nvidia-smi" in cmd:
                return (0, "550.54.15, RTX 3090")
            if cmd.endswith("--version"):
                # skip probe (1st) fails, verify (2nd) succeeds
                return ((1, "no") if len(self.calls) == 2
                        else (0, real_mod.VLLM_VERSION_PIN))
            if "echo yes" in cmd:
                return (0, "no")
            if "python3 -m venv" in cmd:
                return (0, "done")
            if "-m pip install" in cmd and "vllm==" in cmd:
                return (0, "ok")
            return (0, "")

    runner = _FakeRunner()
    monkeypatch.setattr(real_mod, "_make_runner", lambda machine: runner.run)
    import engine_registry
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", settings_file)
    r = client.post("/api/machines/aibox1/install-vllm")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["delivery"] == "venv"
    # the skip probe ran against the configured venv_bin (fresh settings ->
    # empty venv -> the probe misses and the install path would follow);
    # this test pins the ENDPOINT wiring: the real installer ran with the
    # machine dict from the store and the tmp settings file (no crash, no
    # docker branch).
    assert body["skipped"] is True
    assert body["detail"] == "skipped (already present)"
    assert any('"$HOME/vllm-env/bin/vllm" --version' in c
               for c in runner.calls)
