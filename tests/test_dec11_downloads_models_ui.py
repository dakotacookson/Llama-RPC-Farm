"""DEC-11 (engine-aware downloads + models list UI, closes T4-G5/G6):

- GET /api/models: the Models table lists only models the ACTIVE engine
  serves — llama_cpp hides 'hf' (safetensors repo) models, vllm hides
  'gguf' models. Format inference mirrors rpc_manager.validate_model's
  reverse gate (no explicit `format` -> gguf iff base_path ends .gguf).
- Adapter filter_models members: llama.cpp keeps gguf, vllm keeps hf.
- UI pane rendering (node-level contract tested at applyEngineHints'
  source level is not possible here; the backend manifest drives it —
  the /api/settings ui_manifest.downloads.formats list is what the
  renderer consumes, already covered by DEC-02 tests).

Same stub contract as test_dec07_acquisition_gate (SSH/HF faked).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402
import config_loader  # noqa: E402
import engine_registry  # noqa: E402


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    eng = tmp_path / "settings.yaml"
    eng.write_text("")  # empty -> default engine (llama_cpp)
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

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(app_module.cfg, "MODELS_DIR", str(models_dir))

    monkeypatch.setattr(config_loader, "groups",
                        __import__("types").SimpleNamespace(
                            all=lambda: [], get=lambda n: None))
    yield str(eng)


@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def _write_model(models_dir, mid, base_path, fmt=None):
    body = (
        f"model_id: {mid}\n"
        f"name: {mid}\n"
        f"base_path: {base_path}\n"
        "servers:\n- aibox1\n"
    )
    if fmt is not None:
        body += f"format: {fmt}\n"
    (models_dir / f"{mid}.yaml").write_text(body, encoding="utf-8")


def _set_engine(engine, tmp_file):
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(f"engine: {engine}\n")


# ---------------- adapter filter_models --------------------------------------

def test_llamacpp_filter_models_keeps_gguf_hides_hf():
    ad = engine_registry.get_engine({"engine": "llama_cpp"})
    models = [
        {"model_id": "gg1", "base_path": "store://llm-models/m.gguf"},
        {"model_id": "hf1", "base_path": "store://llm-models/Qwen3",
         "format": "hf"},
        # no explicit format + non-.gguf path -> inferred hf -> hidden
        {"model_id": "hf2", "base_path": "store://llm-models/Mistral"},
    ]
    out = ad.filter_models(models)
    assert [m["model_id"] for m in out] == ["gg1"]


def test_vllm_filter_models_keeps_hf_hides_gguf():
    ad = engine_registry.get_engine({"engine": "vllm"})
    models = [
        {"model_id": "gg1", "base_path": "store://llm-models/m.gguf"},
        {"model_id": "hf1", "base_path": "store://llm-models/Qwen3",
         "format": "hf"},
        {"model_id": "gg2", "base_path": "store://llm-models/m.gguf",
         "format": "gguf"},
        # no explicit format + non-.gguf path -> inferred hf -> kept
        {"model_id": "hf2", "base_path": "store://llm-models/Mistral"},
    ]
    out = ad.filter_models(models)
    assert sorted(m["model_id"] for m in out) == ["hf1", "hf2"]


# ---------------- GET /api/models engine gate --------------------------------

def test_api_models_llamacpp_lists_gguf_only(client, settings_file):
    import pathlib
    models_dir = pathlib.Path(config_loader.MODELS_DIR)
    _write_model(models_dir, "ggmodel",
                 "store://llm-models/m.gguf", fmt="gguf")
    _write_model(models_dir, "hfmodel",
                 "store://llm-models/Qwen3", fmt="hf")
    r = client.get("/api/models")
    assert r.status_code == 200, r.text
    ids = [m.get("model_id") for m in r.json()]
    assert "ggmodel" in ids
    assert "hfmodel" not in ids


def test_api_models_vllm_lists_hf_only(client, settings_file):
    _set_engine("vllm", settings_file)
    import pathlib
    models_dir = pathlib.Path(config_loader.MODELS_DIR)
    _write_model(models_dir, "ggmodel",
                 "store://llm-models/m.gguf", fmt="gguf")
    _write_model(models_dir, "hfmodel",
                 "store://llm-models/Qwen3", fmt="hf")
    r = client.get("/api/models")
    assert r.status_code == 200, r.text
    ids = [m.get("model_id") for m in r.json()]
    assert "hfmodel" in ids
    assert "ggmodel" not in ids