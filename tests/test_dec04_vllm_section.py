"""DEC-04 (AUDIT-T6 Wave 2, CARD 4): vllm section ownership via adapter.

config_loader's sanitize/apply now delegate to the owning adapter
(vllm_manager.VllmAdapter / LlamaCppAdapter no-ops). Tests: delegation
shapes/errors identical, and the round-trip isolation contract — a vllm
save must not alter the llama_cpp vocabulary and vice versa.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import config_loader as cfg
import engine_registry


# ---------------- delegation: same shapes, same errors ---------------------

def test_sanitize_delegates_to_adapter():
    """cfg.sanitize_vllm_section reaches VllmAdapter.sanitize_model_section
    with identical results (numbers coerced, auto kv dropped, unknown
    keys rejected)."""
    assert cfg.sanitize_vllm_section(None) is None
    assert cfg.sanitize_vllm_section({}) == {}
    out = cfg.sanitize_vllm_section({
        "max_model_len": 32768, "gpu_memory_utilization": 0.35,
        "kv_cache_dtype": "auto"})
    assert out == {"max_model_len": 32768, "gpu_memory_utilization": 0.35}
    with pytest.raises(ValueError, match="unknown vllm key"):
        cfg.sanitize_vllm_section({"nope": 1})
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        cfg.sanitize_vllm_section({"kv_cache_dtype": "q8_0"})


def test_apply_delegates_to_adapter_and_owner_is_vllm():
    """apply stores/normalizes in place and the section adapter is the
    VllmAdapter instance (engine_id/config_section identity)."""
    data = {"flags": {"ctx-size": 8192}, "vllm": {"max_model_len": 8192}}
    cfg.apply_vllm_section(data)
    assert data["vllm"] == {"max_model_len": 8192}
    assert data["flags"] == {"ctx-size": 8192}  # llama.cpp keys untouched
    empty = {"vllm": None}
    cfg.apply_vllm_section(empty)
    assert "vllm" not in empty
    adapter = engine_registry.get_engine_section_adapter()
    assert adapter.engine_id == "vllm"
    assert adapter.config_section == "vllm"


# ---------------- round-trip isolation (the CARD's core contract) ----------

def _save(tmp_path, monkeypatch, payload):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(cfg, "MODELS_DIR", str(models_dir))
    os.makedirs(models_dir, exist_ok=True)
    saved = cfg.save_model(payload["model_id"], dict(payload))
    reloaded = cfg.get_model(payload["model_id"])
    return saved, reloaded


def test_vllm_save_preserves_llama_cpp_section(tmp_path, monkeypatch):
    """A vllm-mode save (adapter owns the vllm: section) must not alter
    the llama.cpp vocabulary (flags/device placement) in the round-trip."""
    payload = {
        "model_id": "iso-vllm",
        "name": "iso vllm",
        "base_path": "/models/x",
        "flags": {"ctx-size": 4096, "ngl": 99},
        "tensor_split": "4,4",
        "vllm": {"max_model_len": 16384, "max_num_seqs": 128},
    }
    monkeypatch.setattr(engine_registry, "load_settings",
                        lambda path=None: {"engine": "vllm"})
    saved, reloaded = _save(tmp_path, monkeypatch, payload)
    assert reloaded["flags"] == {"ctx-size": 4096, "ngl": 99}
    assert reloaded["tensor_split"] == "4,4"
    assert reloaded["vllm"] == {"max_model_len": 16384, "max_num_seqs": 128}
    # llama.cpp vocabulary round-trips intact through the adapter save
    assert saved["flags"] == {"ctx-size": 4096, "ngl": 99}
    assert saved["tensor_split"] == "4,4"


def test_llama_cpp_save_preserves_vllm_section(tmp_path, monkeypatch):
    """A llama.cpp-mode save (no-op section handler) must not alter the
    vllm: block — it passes through untouched (additive round-trip)."""
    payload = {
        "model_id": "iso-lcpp",
        "name": "iso lcpp",
        "base_path": "/models/x",
        "flags": {"ctx-size": 8192},
        "vllm": {"gpu_memory_utilization": 0.5, "kv_cache_dtype": "fp8"},
    }
    monkeypatch.setattr(engine_registry, "load_settings",
                        lambda path=None: {"engine": "llama_cpp"})
    saved, reloaded = _save(tmp_path, monkeypatch, payload)
    assert reloaded["flags"] == {"ctx-size": 8192}
    assert reloaded["vllm"] == {"gpu_memory_utilization": 0.5,
                                "kv_cache_dtype": "fp8"}


def test_llama_cpp_apply_is_strict_noop(tmp_path, monkeypatch):
    """The llama.cpp apply_model_section never mutates the dict (in place
    OR by normalizing) — the vllm: section it happens to carry is exactly
    preserved, and a section-free save gains no key."""
    adapter = engine_registry.get_engine({"engine": "llama_cpp"})
    data = {"vllm": {"weird": {"nested": 1}}}
    out = adapter.apply_model_section(data)
    assert out is data
    assert data["vllm"] == {"weird": {"nested": 1}}
    clean = {"flags": {"ngl": 9}}
    assert adapter.apply_model_section(clean) is clean
    assert "vllm" not in clean