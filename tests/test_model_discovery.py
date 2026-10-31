"""Model discovery tests: GGUF scan regression + HF/safetensors dirs (B4).

scan_directory is the model wizard's source list — new formats must not
disturb existing GGUF grouping, and HF/safetensors directories (vLLM's
native format, ARD §3.4) must collapse into ONE entry whose shards use
the same -000XX-of-000YY convention as GGUF splits. Fixtures are real
temp directories: no monkeypatching the filesystem module under test.
"""
import os

import pytest

import model_parser as mp


@pytest.fixture
def tree(tmp_path):
    """A scratch model root; tests build layouts under it."""
    return tmp_path


def _touch(path, size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _by_name(models):
    return {m["name"]: m for m in models}


# ---------------- GGUF regression (existing behavior) ----------------------

def test_gguf_shard_collapse_unchanged(tree):
    root = tree / "gguf-root"
    root.mkdir()
    for i in range(1, 4):
        (root / f"Qwen3-8B-0000{i}-of-00003.gguf").write_bytes(b"a" * 10)
    models, projectors = mp.scan_directory(str(root))
    assert len(models) == 1
    m = models[0]
    assert m["name"] == "Qwen3-8B"
    assert m["format"] == "gguf"
    assert m["shard_count"] == 3
    assert m["is_sharded"] is True
    assert projectors == []


def test_gguf_single_file_unsharded(tree):
    (tree / "tiny.gguf").write_bytes(b"a" * 5)
    models, _ = mp.scan_directory(str(tree))
    assert len(models) == 1
    assert models[0]["format"] == "gguf"
    assert models[0]["is_sharded"] is False


def test_mmproj_still_separate(tree):
    (tree / "model.gguf").write_bytes(b"a")
    (tree / "mmproj-model.gguf").write_bytes(b"b")
    models, projectors = mp.scan_directory(str(tree))
    assert [m["name"] for m in models] == ["model"]
    assert [p["name"] for p in projectors] == ["mmproj-model.gguf"]


# ---------------- HF / safetensors directory detection ---------------------

def test_hf_dir_is_one_model(tree):
    """config.json + safetensors = ONE model; base_path is the directory
    (what `vllm serve` takes), named after the directory."""
    d = tree / "Qwen3-27B-HF"
    d.mkdir()
    (d / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (d / "model.safetensors").write_bytes(b"w" * 100)
    (d / "tokenizer.json").write_text("{}")
    models, _ = mp.scan_directory(str(tree))
    assert len(models) == 1
    m = models[0]
    assert m["name"] == "Qwen3-27B-HF"
    assert m["format"] == "hf"
    assert m["base_path"] == str(d)
    assert m["shard_count"] == 1
    assert m["is_sharded"] is False
    # the weight file is listed inside the entry
    assert "model.safetensors" in m["files"]


def test_hf_sharded_safetensors_collapse(tree):
    """model-00001-of-00002.safetensors shards collapse into one entry —
    same -000XX-of-000YY pattern as GGUF splits."""
    d = tree / "Qwen3-32B"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"a" * 40)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"b" * 60)
    (d / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    models, _ = mp.scan_directory(str(tree))
    assert len(models) == 1
    m = models[0]
    assert m["format"] == "hf"
    assert m["is_sharded"] is True
    assert m["shard_count"] == 2
    assert m["total_size_bytes"] == 100
    assert set(m["files"]) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_hf_dir_without_safetensors_is_not_a_model(tree):
    """config.json alone (e.g. a tokenizer cache) must not be detected."""
    d = tree / "not-a-model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    models, _ = mp.scan_directory(str(tree))
    assert models == []


def test_safetensors_without_config_json_is_not_a_model(tree):
    """Loose safetensors without config.json is not identifiable as an HF
    model (vLLM reads the architecture from config.json) — not detected."""
    d = tree / "loose"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    models, _ = mp.scan_directory(str(tree))
    assert models == []


def test_hf_stray_safetensors_not_counted_as_shards(tree):
    """adapter_model.safetensors (LoRA) doesn't share the weight shard base
    name and must not inflate the entry's shard count."""
    d = tree / "with-adapter"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"a" * 30)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"b" * 30)
    (d / "adapter_model.safetensors").write_bytes(b"c" * 5)
    models, _ = mp.scan_directory(str(tree))
    assert len(models) == 1
    assert models[0]["shard_count"] == 2
    assert "adapter_model.safetensors" not in models[0]["files"]


def test_hf_and_gguf_side_by_side(tree):
    """Both formats list together, sorted by name."""
    d = tree / "B-HF-model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"w")
    (tree / "A-gguf-model.gguf").write_bytes(b"g")
    models, _ = mp.scan_directory(str(tree))
    assert [(m["name"], m["format"]) for m in models] == [
        ("A-gguf-model", "gguf"), ("B-HF-model", "hf")]


def test_hf_dir_skips_gguf_inside_it(tree):
    """A GGUF dropped inside an HF dir belongs to that one model dir —
    it must not appear as a separate entry (non-recursive scan)."""
    d = tree / "hf-with-gguf"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"w")
    (d / "converted.gguf").write_bytes(b"g")
    models, _ = mp.scan_directory(str(tree))
    assert len(models) == 1
    assert models[0]["format"] == "hf"


# ---------------- recursive scan -------------------------------------------

def test_recursive_hf_nested_and_no_descend(tree):
    """Recursive: HF dirs at any depth are one entry each; their contents
    (including nested GGUF) are not re-scanned as separate models."""
    a = tree / "farm" / "Alpha-8B"
    a.mkdir(parents=True)
    (a / "config.json").write_text("{}")
    for i in range(1, 4):
        (a / f"model-0000{i}-of-00003.safetensors").write_bytes(b"ab")
    (tree / "farm" / "Beta.gguf").write_bytes(b"g")
    (tree / "farm" / "loose-also.gguf").write_bytes(b"h")

    models, _ = mp.scan_directory(str(tree), recursive=True)
    got = _by_name(models)
    assert set(got) == {"Alpha-8B", "Beta", "loose-also"}
    assert got["Alpha-8B"]["format"] == "hf"
    assert got["Alpha-8B"]["shard_count"] == 3
    assert got["Beta"]["format"] == "gguf"


def test_recursive_gguf_still_grouped_per_root(tree):
    root = tree / "farm"
    root.mkdir()
    (root / "shard-00001-of-00002.gguf").write_bytes(b"a")
    (root / "shard-00002-of-00002.gguf").write_bytes(b"b")
    models, _ = mp.scan_directory(str(root), recursive=True)
    assert len(models) == 1
    assert models[0]["shard_count"] == 2


def test_scan_missing_path_returns_empty():
    assert mp.scan_directory("/nonexistent/path/xyz") == ([], [])


# ---------------- config_loader: vllm section round-trip -------------------

from config_loader import (  # noqa: E402
    apply_vllm_section,
    sanitize_vllm_section,
    save_model,
    get_model,
    VLLM_MODEL_KEYS,
)


def test_sanitize_full_section():
    out = sanitize_vllm_section({
        "max_model_len": 32768.0,      # floats coerced to int
        "max_num_seqs": 256,
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "fp8",
        "cpu_offload_gb": 20,
        "quantization": "fp8",
        "tokenizer": "/mnt/store/qwen3-hf",
        "extra_args": [" --enable-prefix-caching ", "--swap-space 4"],
    })
    assert out == {
        "max_model_len": 32768,
        "max_num_seqs": 256,
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "fp8",
        "cpu_offload_gb": 20,
        "quantization": "fp8",
        "tokenizer": "/mnt/store/qwen3-hf",
        "extra_args": ["--enable-prefix-caching", "--swap-space 4"],
    }


def test_sanitize_empty_and_none():
    assert sanitize_vllm_section(None) is None
    assert sanitize_vllm_section({}) == {}
    # "auto" is the default — not stored
    assert sanitize_vllm_section({"kv_cache_dtype": "auto"}) == {}


def test_sanitize_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown vllm key"):
        sanitize_vllm_section({"bogus_flag": 1})


def test_sanitize_pipeline_parallel_size_round_trips():
    """A1 fix: pipeline_parallel_size is a first-class schema key — no
    longer rejected as unknown; coerced to int like its TP sibling."""
    out = sanitize_vllm_section({"pipeline_parallel_size": 2.0})
    assert out == {"pipeline_parallel_size": 2}
    with pytest.raises(ValueError, match="must be a number"):
        sanitize_vllm_section({"pipeline_parallel_size": "2x"})


def test_sanitize_bad_kv_dtype_rejected():
    with pytest.raises(ValueError, match="fp8"):
        sanitize_vllm_section({"kv_cache_dtype": "q8_0"})


def test_sanitize_bad_numbers_and_types_rejected():
    with pytest.raises(ValueError, match="must be a number"):
        sanitize_vllm_section({"max_model_len": "32k"})
    with pytest.raises(ValueError, match="must be a number"):
        sanitize_vllm_section({"tensor_parallel_size": True})
    with pytest.raises(ValueError, match="must be a string"):
        sanitize_vllm_section({"quantization": 8})
    with pytest.raises(ValueError, match="non-empty strings"):
        sanitize_vllm_section({"extra_args": ["ok", ""]})
    with pytest.raises(ValueError, match="must be a mapping"):
        sanitize_vllm_section("fp8")


def test_sanitize_keeps_zero_offload():
    """0 is a meaningful, valid value here (storage layer) — omitting it
    from the command line is vllm_manager's job (degradation matrix)."""
    assert sanitize_vllm_section({"cpu_offload_gb": 0}) == {"cpu_offload_gb": 0}


def test_apply_section_additive_flags_untouched():
    """The core round-trip guarantee: a config carrying BOTH engines'
    sections keeps llama.cpp fields byte-identical."""
    data = {"model_id": "m", "flags": {"ctx-size": 32768, "n-gpu-layers": 99},
            "tensor_split_mode": "fit", "device_order": "aibox1-CUDA0",
            "vllm": {"max_model_len": 8192}}
    apply_vllm_section(data)
    assert data["flags"] == {"ctx-size": 32768, "n-gpu-layers": 99}
    assert data["tensor_split_mode"] == "fit"
    assert data["device_order"] == "aibox1-CUDA0"
    assert data["vllm"] == {"max_model_len": 8192}


def test_apply_section_empty_removes_key():
    data = {"model_id": "m", "flags": {}, "vllm": {}}
    apply_vllm_section(data)
    assert "vllm" not in data
    data = {"model_id": "m", "flags": {}}
    apply_vllm_section(data)
    assert "vllm" not in data


def test_vllm_model_keys_match_vllm_manager():
    """Single source of truth: vllm_manager re-exports config_loader's
    vocabulary (B4 moved it there; command assembly must stay in sync)."""
    import vllm_manager
    assert vllm_manager.VLLM_MODEL_KEYS == VLLM_MODEL_KEYS
    # 0.30: fp8 KV needs SM89+; the unquantized dtypes are the fallback on
    # older consumer GPUs (config_loader owns the vocabulary).
    assert vllm_manager.VALID_KV_CACHE_DTYPES == (
        "auto", "fp8", "half", "bfloat16", "float16")


def test_save_model_round_trip(tmp_path, monkeypatch):
    """save_model normalizes the vllm: section; get_model reads it back
    intact, llama.cpp fields untouched. Uses a temp config dir so the
    repo's real config/models/ is never written."""
    import config_loader as cfg
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(cfg, "MODELS_DIR", str(models_dir))
    data = {"model_id": "roundtrip-test", "name": "RT",
            "base_path": "/mnt/store/Qwen3-27B-HF",
            "flags": {"ctx-size": 4096},
            "vllm": {"max_model_len": 16384, "kv_cache_dtype": "fp8",
                     "cpu_offload_gb": 0, "extra_args": ["--swap-space 4"]}}
    saved = save_model("roundtrip-test", data)
    assert saved["vllm"] == {"max_model_len": 16384, "kv_cache_dtype": "fp8",
                             "cpu_offload_gb": 0,
                             "extra_args": ["--swap-space 4"]}
    loaded = get_model("roundtrip-test")
    assert loaded["vllm"]["max_model_len"] == 16384
    assert loaded["flags"] == {"ctx-size": 4096}
    # saving the loaded config again is idempotent
    import copy
    again = save_model("roundtrip-test", copy.deepcopy(loaded))
    assert again["vllm"] == loaded["vllm"]
    # unknown keys in the section are rejected at save time
    with pytest.raises(ValueError, match="unknown vllm key"):
        save_model("roundtrip-test", {"model_id": "roundtrip-test",
                                      "vllm": {"nope": 1}})


def test_save_model_without_vllm_key_stays_clean(tmp_path, monkeypatch):
    """Pre-vLLM configs must not gain a vllm: key on an unrelated save."""
    import config_loader as cfg
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(cfg, "MODELS_DIR", str(models_dir))
    save_model("legacy-model", {"model_id": "legacy-model",
                                "flags": {"ctx-size": 8192}})
    loaded = get_model("legacy-model")
    assert "vllm" not in loaded
    assert loaded["flags"] == {"ctx-size": 8192}