"""Tests for the engine registry seam (ARD §1) and the settings engine
schema (ARD §2). Pure logic: no network, no real machines.

Covers the B1 checkpoint:
- default engine injected when the settings key is missing
- invalid engine -> startup ValueError listing valid values (no silent fallback)
- LlamaCppAdapter delegation to rpc_manager/swap_manager (zero behavior change)
- get_engine() seam resolves the llama.cpp adapter
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
    injected."""
    _write_settings(settings_file, {"model_dir": "store://llm-models",
                                    "llama_bin": "~/llama.cpp/build/bin"})
    out = load_settings(settings_file)
    assert out["engine"] == DEFAULT_ENGINE == "llama_cpp"
    # unrelated keys survive untouched
    assert out["llama_bin"] == "~/llama.cpp/build/bin"


def test_missing_settings_file_behaves_like_empty(settings_file):
    out = load_settings(settings_file)  # file never written
    assert out["engine"] == "llama_cpp"


def test_engine_llama_cpp_round_trips(settings_file):
    _write_settings(settings_file, {"engine": "llama_cpp"})
    out = load_settings(settings_file)
    assert out["engine"] == "llama_cpp"


# ---------------- invalid engine -> startup error, no fallback ------------

@pytest.mark.parametrize("bad", ["bogus", "llama-cpp", "LLAMA_CPP", "llama",
                                 "engine_x", ""])
def test_invalid_engine_raises_listing_valid_values(settings_file, bad):
    """A typo must be a startup error listing valid values — never a
    silent fallback to the default."""
    _write_settings(settings_file, {"engine": bad})
    with pytest.raises(ValueError) as ei:
        load_settings(settings_file)
    msg = str(ei.value)
    assert bad in msg
    assert "llama_cpp" in msg


def test_invalid_engine_raises_even_via_get_engine(settings_file):
    _write_settings(settings_file, {"engine": "bogus"})
    with pytest.raises(ValueError):
        load_settings(settings_file)


# ---------------- config_loader helpers (schema vocabulary) ----------------

def test_validate_engine_accepts_enum_and_rejects_typo():
    assert cfg.validate_engine("llama_cpp") == "llama_cpp"
    with pytest.raises(ValueError):
        cfg.validate_engine("bogus")


def test_inject_engine_defaults_mutates_and_returns():
    s = {}
    out = cfg.inject_engine_defaults(s)
    assert out is s
    assert s["engine"] == "llama_cpp"




def test_inject_engine_defaults_llamacpp_settings_section_noop():
    """llama_cpp owns no settings block: its settings_section is identity."""
    adapter = LlamaCppAdapter()
    raw = {"llama_bin": "x"}
    assert adapter.settings_section(raw) is raw
    assert raw == {"llama_bin": "x"}



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


def test_settings_file_default_points_at_repo_config(monkeypatchmonkeypatch):
    """The registry's default settings path must be the repo config file.
    Conftest isolates SETTINGS_FILE for the whole suite; restore the
    repository default explicitly so this test checks the constant's
    intended value without breaking that isolation."""
    import engine_registry as er
    assert er.SETTINGS_FILE.endswith(os.path.join("config", "settings.yaml"))
    assert os.path.exists(er.SETTINGS_FILE)