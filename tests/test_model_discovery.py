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
