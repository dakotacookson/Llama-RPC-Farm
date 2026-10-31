"""Launch-page strict-mode gating: POST /api/launch + POST
/api/launch/stop enforce the same key-grant allowlist as /api/chat and
the models table (access.py view_allow — EXACT token match). A scoped
key (models set in its `access_keys:` record) must (a) 404 "never
existed" a launch/stop of a model outside its grants — including the
BASE key of a model where only one variant is granted — and (b) keep
working for granted tokens.

Driven through the REAL identity seam, no hand-set contextvars: the
access_keys table comes from the isolated settings.yaml (FARM_SETTINGS_
FILE -> access.py's own reader, same file the engine reads) and the
Bearer header crosses app's IpGateMiddleware -> restrict_scope_from_key,
exactly like a live key-holding browser.

Unknown ids keep the engine's not-found path; open-gate (no keys
configured) callers are unchanged."""
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest

import app as app_module


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Same isolation shape as tests/test_engine_endpoints.py: BOTH
    settings globals into tmp (app_module writes, engine_registry reads),
    swap settings + machines + models dir isolated — never touches the
    repo's real config/. access.py reads ITS settings via the
    FARM_SETTINGS_FILE env var (its own reader), pointed at the same tmp
    file so the key table lives with the engine table."""
    import types
    import engine_registry
    eng = tmp_path / "settings.yaml"
    eng.write_text("engine: llama_cpp\n")
    monkeypatch.setattr(app_module, "SETTINGS_FILE", str(eng))
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", str(eng))
    # access.py owns its settings path (env override + repo-relative
    # default) — point the env at the same tmp file
    monkeypatch.setenv("FARM_SETTINGS_FILE", str(eng))

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
    monkeypatch.setattr(config_loader.machines, "path", str(machines))
    import rpc_manager
    monkeypatch.setattr(rpc_manager, "cfg", config_loader)
    monkeypatch.setattr(app_module, "rpc_manager", rpc_manager)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(app_module.cfg, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(config_loader, "groups",
                        types.SimpleNamespace(all=lambda: [],
                                              get=lambda n: None))
    yield str(eng)


@pytest.fixture
def client(settings_file, monkeypatch):
    from fastapi.testclient import TestClient
    _engine_llcpp(monkeypatch)
    return TestClient(app_module.app)


def _engine_llcpp(monkeypatch):
    """llama.cpp adapter double: capable(rpc_cluster), launch/stop land in
    spies (the real engine start would SSH out)."""
    class _Lcpp:
        name = "llama_cpp"

        def capable(self, c):
            return c == "rpc_cluster"

        # launch/stop go through the adapter seam (DEC-06) — route them
        # into rpc_manager so the test's monkeypatched spy fires
        def start_model(self, model_id, servers, port, variant=None,
                        master_name=None, mode="rpc"):
            import rpc_manager
            return rpc_manager.launch_model(model_id, servers, port,
                                            variant, master_name, mode)

        def stop_model(self, model_id, servers, variant=None,
                       master_name=None, port=8080):
            import rpc_manager
            return rpc_manager.stop_model(servers, model_id, variant,
                                          master_name, port)

        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            return None if raw is None else dict(raw)
    import engine_registry
    monkeypatch.setattr(engine_registry, "get_engine",
                        lambda s=None: _Lcpp())


def _write_key(path, token, name, models, admin=False, unrestricted=True):
    """Append/replace the access_keys table in the isolated settings.yaml —
    the REAL config the middleware's key_record() reads."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("access_keys", {})[token] = {
        "name": name, "models": models, "admin": admin,
        "admin_unrestricted": unrestricted}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


DAD_KEY = "sk-dad-test-0000000000000000000000000000000"

# model configs derived from the models table, not hardcoded grids: a
# base config carrying a dad variant (the variant is granted, the base
# is NOT) and a plain granted model
BASE_MID = "huihui-a3b"


def _seed_models():
    """Write model configs through the REAL config_loader (same shape the
    Models tab saves): base + one variant, plus a plain model."""
    import config_loader
    config_loader.save_model(BASE_MID, {
        "model_id": BASE_MID,
        "name": "Huihui A3B",
        "base_path": "/models/huihui.gguf",
        "servers": ["aibox1"],
        "variants": [{
            "label": "dad",
            "servers": ["aibox1"],
            "tensor_split_mode": "split",
            "tensor_split": "4,7,4",
        }],
    })
    config_loader.save_model("ornith-9b", {
        "model_id": "ornith-9b",
        "name": "Ornith",
        "base_path": "/models/ornith.gguf",
        "servers": ["aibox1"],
    })


@pytest.fixture
def scoped_dad(settings_file):
    """Dad's key record exactly as the operator configures it in the
    Access tab: strict chip (admin_unrestricted: false), ONE variant
    grant (not the base) and one plain mid."""
    _write_key(settings_file, DAD_KEY, "Dad",
               [f"{BASE_MID}--dad", "ornith-9b"],
               admin=False, unrestricted=False)


def _hdr():
    return {"Authorization": f"Bearer {DAD_KEY}"}


def test_launch_respects_scoped_allowlist(client, scoped_dad):
    """A non-granted model launches as 'never existed' (404) — same
    semantics as POST /api/chat."""
    import config_loader
    _seed_models()
    r = client.post("/api/launch", headers=_hdr(), json={
        "model_id": "laya-q8-0", "servers": ["aibox1"]})
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "model 'laya-q8-0' not found"


def test_launch_variant_grant_never_widens_to_base(client, scoped_dad):
    """Dad granted huihui-a3b--dad (variant token): the BASE key stays
    unlaunchable — exact-token rule, no family widening."""
    _seed_models()
    r = client.post("/api/launch", headers=_hdr(), json={
        "model_id": BASE_MID, "servers": ["aibox1"]})
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == f"model '{BASE_MID}' not found"


def test_launch_granted_variant_ok(client, scoped_dad, monkeypatch):
    """Granted tokens pass through to the engine (the model id arrives
    pre-split by launch_model's own --label handling)."""
    _seed_models()
    calls = {}

    def fake_launch(model_id, servers, port, variant=None, master_name=None,
                    mode="rpc"):
        calls["args"] = (model_id, servers, port)
        return {"ok": True, "failed": False}

    monkeypatch.setattr(app_module.rpc_manager, "launch_model", fake_launch)
    r = client.post("/api/launch", headers=_hdr(), json={
        "model_id": f"{BASE_MID}--dad", "servers": ["aibox1"], "port": 8098})
    assert r.status_code == 200, r.text
    assert calls["args"] == (f"{BASE_MID}--dad", ["aibox1"], 8098)


def test_stop_respects_scoped_allowlist(client, scoped_dad):
    """Stop mirrors launch: a non-granted model is 404 'never existed'."""
    _seed_models()
    r = client.post("/api/launch/stop", headers=_hdr(), json={
        "model_id": "laya-q8-0", "servers": ["aibox1"]})
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "model 'laya-q8-0' not found"


def test_stop_variant_grant_never_widens_to_base(client, scoped_dad):
    _seed_models()
    r = client.post("/api/launch/stop", headers=_hdr(), json={
        "model_id": BASE_MID, "servers": ["aibox1"]})
    assert r.status_code == 404, r.text


def test_launch_private_model_hidden_for_scoped(client, scoped_dad):
    """A private model outside the key's grants: 404 before any machine
    validation — raw YAML flag read (_raw_is_private), same as chat."""
    _seed_models()
    import config_loader
    saved = config_loader.MODELS_DIR
    with open(os.path.join(saved, "dark-scar.yaml"), "w",
              encoding="utf-8") as f:
        f.write("model_id: dark-scar\nprivate: true\nservers:\n- aibox1\n")
    r = client.post("/api/launch", headers=_hdr(), json={
        "model_id": "dark-scar", "servers": ["aibox1"]})
    assert r.status_code == 404, r.text


def test_unknown_key_restricted_no_allowlist_still_launchless(client,
                                                              settings_file):
    """STRICT posture: once keys exist, an unknown key is restricted with
    allow None — launch refuses everything with 404 (farm parity with
    the gateway's anonymous-refusal; control stays forbidden)."""
    _write_key(settings_file, "sk-other-key-000000000000000000000000000",
               "Other", ["whatever"])
    _seed_models()
    r = client.post("/api/launch", headers={
        "Authorization": "Bearer sk-not-a-real-key"}, json={
        "model_id": "ornith-9b", "servers": ["aibox1"]})
    assert r.status_code == 404, r.text


def test_unrestricted_and_console_unchanged(client, settings_file,
                                            monkeypatch):
    """No keys configured (pre-Access farm state, gate open): the launch
    reaches the engine exactly as before — nothing 404s on grant
    grounds."""
    _seed_models()
    calls = {}

    def fake_launch(model_id, servers, port, variant=None, master_name=None,
                    mode="auto"):
        calls["args"] = (model_id, servers)
        raise ValueError("model 'x' not found")

    monkeypatch.setattr(app_module.rpc_manager, "launch_model", fake_launch)
    r = client.post("/api/launch", json={
        "model_id": "ghost-model", "servers": ["aibox1"]})
    assert r.status_code == 500, r.text  # engine ValueError -> failed report
    assert calls["args"] == ("ghost-model", ["aibox1"])