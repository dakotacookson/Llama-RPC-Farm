"""Tests for the engine registry seam (ARD §1) and the settings engine
schema (ARD §2). Pure logic: no network, no real machines.

Covers the B1 checkpoint:
- default engine injected when the settings key is missing
- invalid engine -> startup ValueError listing valid values (no silent fallback)
- vllm block normalization/validation (defaults, unknown keys, delivery, gmu)
- LlamaCppAdapter delegation to rpc_manager/swap_manager (zero behavior change)
- get_engine() seam: unknown-but-valid engine names rejected until registered
"""
import os

import pytest
import yaml

import config_loader as cfg
import engine_registry
import rpc_manager
from engine_registry import (
    DEFAULT_ENGINE,
    LlamaCppAdapter,
    get_engine,
    load_settings,
)


@pytest.fixture
def settings_file(tmp_path):
    """Path of a scratch settings.yaml for load_settings(path=...)."""
    return str(tmp_path / "settings.yaml")


def _write_settings(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


# ---------------- default engine injected when key missing ----------------

def test_default_engine_injected_when_missing(settings_file):
    """Existing installs have no engine key — the default must be
    injected. DEC-01 (engine-qualified blocks): llama_cpp mode materializes
    NO vllm vocabulary — a llama.cpp settings file carries only shared
    farm keys (spec §1.4/§3)."""
    _write_settings(settings_file, {"model_dir": "store://llm-models",
                                    "llama_bin": "~/llama.cpp/build/bin"})
    out = load_settings(settings_file)
    assert out["engine"] == DEFAULT_ENGINE == "llama_cpp"
    # engine-qualified: NO vllm block in llama_cpp mode
    assert "vllm" not in out
    # unrelated keys survive untouched
    assert out["llama_bin"] == "~/llama.cpp/build/bin"


def test_missing_settings_file_behaves_like_empty(settings_file):
    out = load_settings(settings_file)  # file never written
    assert out["engine"] == "llama_cpp"
    assert "vllm" not in out


def test_engine_llama_cpp_round_trips(settings_file):
    _write_settings(settings_file, {"engine": "llama_cpp"})
    out = load_settings(settings_file)
    assert out["engine"] == "llama_cpp"


# ---------------- invalid engine -> startup error, no fallback ------------

@pytest.mark.parametrize("bad", ["vllm ", "llama-cpp", "VLLM", "llama",
                                 "engine_x", ""])
def test_invalid_engine_raises_listing_valid_values(settings_file, bad):
    """A typo must be a startup error listing valid values — never a
    silent fallback to the default."""
    _write_settings(settings_file, {"engine": bad})
    with pytest.raises(ValueError) as ei:
        load_settings(settings_file)
    msg = str(ei.value)
    assert bad in msg
    assert "llama_cpp" in msg and "vllm" in msg


def test_invalid_engine_raises_even_via_get_engine(settings_file):
    _write_settings(settings_file, {"engine": "vllmm"})
    with pytest.raises(ValueError):
        load_settings(settings_file)


# ---------------- vllm block validation ----------------

def test_vllm_block_partial_overrides_keep_rest_of_defaults(settings_file):
    _write_settings(settings_file, {"engine": "vllm",
                                    "vllm": {"delivery": "docker"}})
    out = load_settings(settings_file)
    assert out["engine"] == "vllm"
    assert out["vllm"]["delivery"] == "docker"
    assert out["vllm"]["venv_bin"] == "~/vllm-env/bin/vllm"  # default kept


def test_vllm_unknown_key_rejected(settings_file):
    # DEC-01: foreign vllm vocabulary only validates when vllm is ACTIVE
    _write_settings(settings_file, {"engine": "vllm",
                                    "vllm": {"deliveri": "venv"}})
    with pytest.raises(ValueError) as ei:
        load_settings(settings_file)
    assert "deliveri" in str(ei.value)


def test_vllm_invalid_delivery_rejected(settings_file):
    _write_settings(settings_file, {"engine": "vllm",
                                    "vllm": {"delivery": "snap"}})
    with pytest.raises(ValueError) as ei:
        load_settings(settings_file)
    assert "venv" in str(ei.value) and "docker" in str(ei.value)


def test_vllm_bad_gpu_memory_utilization_rejected(settings_file):
    for bad in (0, 1.5, -0.1, "high", True):
        _write_settings(settings_file,
                        {"engine": "vllm",
                         "vllm": {"default_gpu_memory_utilization": bad}})
        with pytest.raises(ValueError):
            load_settings(settings_file)


def test_vllm_non_mapping_block_rejected(settings_file):
    _write_settings(settings_file, {"engine": "vllm", "vllm": ["venv"]})
    with pytest.raises(ValueError):
        load_settings(settings_file)


def test_foreign_vllm_block_tolerated_in_llama_cpp_mode(settings_file):
    """DEC-01 (spec §3): per-model/foreign vocabulary on disk is
    dead-but-tolerated — a llama.cpp load must not validate (or fail on)
    the vllm block; it passes through untouched."""
    _write_settings(settings_file, {"engine": "llama_cpp",
                                    "vllm": {"deliveri": "venv"}})
    out = load_settings(settings_file)
    assert out["vllm"] == {"deliveri": "venv"}


# ---------------- config_loader helpers (schema vocabulary) ----------------

def test_validate_engine_accepts_enum_and_rejects_typo():
    assert cfg.validate_engine("llama_cpp") == "llama_cpp"
    assert cfg.validate_engine("vllm") == "vllm"
    with pytest.raises(ValueError):
        cfg.validate_engine("vllam")


def test_inject_engine_defaults_mutates_and_returns():
    s = {}
    out = cfg.inject_engine_defaults(s)
    assert out is s
    assert s["engine"] == "llama_cpp"
    # engine-qualified: llama_cpp injects no vllm block (DEC-01)


def test_inject_engine_defaults_preserves_partial_vllm_block():
    s = {"engine": "vllm", "vllm": {"venv_bin": "/opt/vllm/bin/vllm"}}
    cfg.inject_engine_defaults(s)
    assert s["vllm"]["venv_bin"] == "/opt/vllm/bin/vllm"
    assert s["vllm"]["delivery"] == "venv"


def test_inject_engine_defaults_delegates_to_active_adapter():
    """DEC-01: inject_engine_defaults shrinks to engine enum + dispatch —
    the vllm block validation runs on the adapter the registry returns."""
    import vllm_manager
    s = {"engine": "vllm"}
    calls = []
    orig = vllm_manager.VllmAdapter.settings_section

    def spy(self, raw_settings):
        calls.append(dict(raw_settings))
        return orig(self, raw_settings)

    vllm_manager.VllmAdapter.settings_section = spy
    try:
        cfg.inject_engine_defaults(s)
    finally:
        vllm_manager.VllmAdapter.settings_section = orig
    assert calls == [{"engine": "vllm"}]
    assert s["vllm"]["delivery"] == "venv"  # adapter applied the defaults


def test_inject_engine_defaults_llamacpp_settings_section_noop():
    """llama_cpp owns no settings block: its settings_section is identity."""
    adapter = LlamaCppAdapter()
    raw = {"llama_bin": "x"}
    assert adapter.settings_section(raw) is raw
    assert raw == {"llama_bin": "x"}


def test_adapter_ownership_members():
    """DEC-01 identity/file-ownership members on both adapters."""
    llamacpp = LlamaCppAdapter()
    assert llamacpp.engine_id == "llama_cpp"
    assert llamacpp.config_section is None
    import vllm_manager
    va = vllm_manager.VllmAdapter()
    assert va.engine_id == "vllm"
    assert va.config_section == "vllm"


# ---------------- get_engine: registry seam ----------------

def test_get_engine_returns_llamacpp_adapter_by_default():
    adapter = get_engine({"engine": "llama_cpp"})
    assert isinstance(adapter, LlamaCppAdapter)
    assert adapter.name == "llama_cpp"


def test_get_engine_defaults_when_settings_none(monkeypatch):
    monkeypatch.setattr("engine_registry.load_settings",
                        lambda: {"engine": "llama_cpp"})
    assert get_engine().name == "llama_cpp"


def test_get_engine_rejects_invalid_engine_value():
    with pytest.raises(ValueError):
        get_engine({"engine": "llama_cpppp"})


def test_get_engine_returns_vllm_adapter():
    """B2 registered VllmAdapter: 'vllm' resolves to it, and the adapter
    is the Protocol-conformant class from vllm_manager."""
    import vllm_manager

    adapter = get_engine({"engine": "vllm"})
    assert adapter.name == "vllm"
    assert isinstance(adapter, vllm_manager.VllmAdapter)


def test_get_engine_rejects_unknown_engine_with_options():
    """A name passing the enum but missing from _REGISTRY (defensive: only
    possible if a config names an engine the checkout lacks the module
    for) must error listing the registered options — never silently
    fall back to llama_cpp."""
    import engine_registry

    saved = engine_registry._REGISTRY.pop("vllm", None)
    try:
        with pytest.raises(ValueError) as ei:
            get_engine({"engine": "vllm"})
        assert "no adapter" in str(ei.value)
        assert "options:" in str(ei.value)
    finally:
        if saved is not None:
            engine_registry._REGISTRY["vllm"] = saved


# ---------------- LlamaCppAdapter delegation (zero behavior change) --------

def test_adapter_build_command_delegates(stub_cfg, monkeypatch):
    calls = {}
    def fake_build(model_cfg, servers, master_name=None):
        calls["args"] = (model_cfg, list(servers), master_name)
        return "llama-server --port ${PORT}"
    monkeypatch.setattr(rpc_manager, "build_llama_command", fake_build)
    model = {"model_id": "m", "flags": {},
             "base_path": "store://llm-models/x.gguf"}
    out = LlamaCppAdapter().build_command(model, ["aibox1"], master_name="aibox1")
    assert out == "llama-server --port ${PORT}"
    assert calls["args"] == (model, ["aibox1"], "aibox1")


def test_adapter_build_command_matches_direct_call(stub_cfg):
    """Delegation must be byte-identical to calling rpc_manager directly."""
    model = {"model_id": "m", "flags": {"ctx-size": 4096},
             "base_path": "store://llm-models/x.gguf"}
    master = {"name": "aibox1", "ip": "10.0.0.98", "username": "u",
              "password": "p", "os": "linux", "rpc_port": 50052,
              "gpu": "", "role": "master",
              "detected": {"gpu_names": [], "vram_total_gb": 0,
                           "ram_total_gb": 16}}
    stub_cfg.update({master["name"]: master})
    via_adapter = LlamaCppAdapter().build_command(model, ["aibox1"],
                                                  master_name="aibox1")
    direct = rpc_manager.build_llama_command(model, ["aibox1"],
                                             master_name="aibox1")
    assert via_adapter == direct


def test_adapter_probe_delegates_to_port_open(monkeypatch):
    seen = {}
    def fake_port_open(machine, port, is_local):
        seen["args"] = (machine, port, is_local)
        return True
    monkeypatch.setattr(rpc_manager, "_port_open", fake_port_open)
    out = LlamaCppAdapter().probe({"name": "aibox1"}, 50052, True)
    assert out == {"running": True, "detail": "rpc port open"}
    assert seen["args"] == ({"name": "aibox1"}, 50052, True)


def test_adapter_probe_reports_closed(monkeypatch):
    monkeypatch.setattr(rpc_manager, "_port_open", lambda m, p, l: False)
    out = LlamaCppAdapter().probe({"name": "aibox1"}, 50052, False)
    assert out == {"running": False, "detail": "rpc port closed"}


def test_adapter_validate_model_config_returns_issues(monkeypatch):
    def fake_validate(model_id):
        assert model_id == "m"
        return {"model_id": model_id, "ok": True,
                "issues": [{"level": "warn", "msg": "ctx high"}]}
    monkeypatch.setattr(rpc_manager, "validate_model", fake_validate)
    out = LlamaCppAdapter().validate_model_config({"model_id": "m"})
    assert out == [{"level": "warn", "msg": "ctx high"}]


def test_adapter_validate_hosts_is_permissive():
    """llama.cpp accepts any farm host — no engine×OS restriction."""
    assert LlamaCppAdapter().validate_hosts(
        [{"name": "w", "os": "windows"}, {"name": "l", "os": "linux"}]) == []


def test_adapter_ui_manifest_shape():
    """AUDIT-T6 §1.5 (DEC-02): the adapter's COMPLETE field inventory —
    every editor section listed (nothing hidden), RPC-cluster launch,
    GGUF downloads. ui_hints() is a deprecated projection of it."""
    man = LlamaCppAdapter().ui_manifest()
    assert man["model_editor"]["sections"] == [
        {"id": "flags"}, {"id": "tensor_split"}, {"id": "rpc_peers"},
        {"id": "rpc_launch"}, {"id": "tuner"}]
    assert man["hidden_panels"] == []
    assert man["hidden_tabs"] == []  # llama.cpp owns every test tab (G-8)
    assert man["labels"]["peers"] == "RPC peers"
    assert man["flag_presets"]["ctx_size"]["flag"] == "--ctx-size"
    assert man["launch"] == {"mode": "rpc_cluster", "probe_label": "RPC port"}
    assert man["downloads"] == {"formats": ["gguf"], "browse_source": "gguf_search"}
    # the deprecated shim projects the manifest (cannot drift)
    assert LlamaCppAdapter().ui_hints() == {
        "labels": man["labels"],
        "hidden_panels": man["hidden_panels"],
        "hidden_tabs": man["hidden_tabs"],
        "flag_presets": man["flag_presets"],
    }


def test_adapter_ui_hints_shape():
    """Legacy view (deprecated, rides along until Card 15): the same four
    keys ui_hints() always returned, now derived from ui_manifest()."""
    hints = LlamaCppAdapter().ui_hints()
    assert set(hints) == {"labels", "hidden_panels", "hidden_tabs", "flag_presets"}
    assert hints["labels"]["peers"] == "RPC peers"
    assert hints["hidden_panels"] == []
    assert hints["hidden_tabs"] == []  # llama.cpp owns every test tab (G-8)
    assert hints["flag_presets"]["ctx_size"]["flag"] == "--ctx-size"


def test_vllm_adapter_ui_manifest_has_pp_preset():
    """A1 fix: UI parity — a pipeline-parallel preset alongside TP.
    AUDIT-T6 §1.5 (DEC-02): the manifest is the engine-owned contract;
    the deprecated ui_hints view projects from it."""
    import vllm_manager
    man = vllm_manager.ui_manifest()
    presets = man["flag_presets"]
    assert presets["parallelism"]["flag"] == "--tensor-parallel-size"
    assert presets["parallelism_pp"]["flag"] == "--pipeline-parallel-size"
    # complete inventory: vllm_section is the only editor surface the
    # engine owns; llama.cpp panels/tabs are hidden; swap_only launch,
    # HF downloads
    assert man["model_editor"]["sections"] == [{"id": "vllm_section"}]
    assert set(man["hidden_panels"]) == {"tuner", "tensor_split", "rpc_peers", "rpc_launch"}
    assert man["hidden_tabs"] == ["audio"]
    assert man["launch"] == {"mode": "swap_only", "probe_label": "HTTP /health",
                             "multi_host": True}
    assert man["downloads"] == {"formats": ["hf"], "browse_source": "hf_repos"}
    # shim projection stays in sync with the manifest
    hints = vllm_manager.VllmAdapter().ui_hints()
    assert hints["hidden_panels"] == man["hidden_panels"]
    assert hints["hidden_tabs"] == man["hidden_tabs"]


def test_adapter_swap_entries_delegates_to_generate(monkeypatch):
    """swap_entries must be pure delegation to swap_manager.launch_plan
    (renamed from generate, DEC-06) with the same (config_path,
    host_name) contract."""
    calls = {}
    import swap_manager
    def fake_generate(config_path=None, host_name=None):
        calls["args"] = (config_path, host_name)
        return {"host": host_name, "yaml": "models: {}", "models": [],
                "scripts": [], "_scripts": {}}
    monkeypatch.setattr(swap_manager, "launch_plan", fake_generate)
    plan = LlamaCppAdapter().swap_entries("/cfg/swap.yaml", "aibox1")
    assert plan["host"] == "aibox1"
    assert calls["args"] == ("/cfg/swap.yaml", "aibox1")


def test_settings_file_default_points_at_repo_config():
    """The registry's default settings path must be the repo config file."""
    import engine_registry as er
    assert er.SETTINGS_FILE.endswith(os.path.join("config", "settings.yaml"))
    assert os.path.exists(er.SETTINGS_FILE)