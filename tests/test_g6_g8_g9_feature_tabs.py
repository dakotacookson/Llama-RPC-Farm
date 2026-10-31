"""G-6/G-8/G-9: engine-adapter capability vocabulary (feature tabs).

The engine adapters OWN capability detection — app.py just dispatches:
- filter_embedding_models / filter_vision_models / filter_audio_models
  per engine (llama.cpp heuristics byte-identical to the old inline code;
  vLLM reads HF config.json + known families).
- asr_language_system_message: llama.cpp emits the Qwen3-ASR "language
  X<asr_text>" suffix; vLLM returns None (no system message, tab hidden).
- ui_hints().hidden_tabs: engine-exclusive test tabs hidden by the
  frontend (Audio tab under vLLM, G-8).

Heavy externals (real model stores) are not needed: HF-format checks read
config.json written into tmp dirs; config_loader.list_models is stubbed.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402
import engine_registry  # noqa: E402
import vllm_manager  # noqa: E402
from engine_registry import LlamaCppAdapter  # noqa: E402
from test_engine_endpoints import settings_file  # noqa: E402,F401

VllmAdapter = vllm_manager.VllmAdapter


def _m(mid, **kw):
    """Model dict shaped like config_loader.list_models() entries."""
    m = {"model_id": mid, "name": mid, "flags": {},
         "base_path": f"store://llm-models/{mid}", "device_order": "",
         "expose": {}}
    m.update(kw)
    return m


# ---------------- llama.cpp filters: byte-identical heuristics ----------

def test_llcpp_embedding_filter_keeps_substring_heuristic():
    """llama.cpp keeps the ORIGINAL inline heuristic byte-identical:
    'embed' substring in id/name only. It misses BGE-style names — that
    gap is exactly what the vLLM filter extends (G-6)."""
    models = [
        _m("nomic-embed-text-v1.5", name="Nomic Embed"),
        _m("BAAI/bge-large-en-v1.5", name="bge"),  # no 'embed' anywhere
        _m("qwen2.5-0.5b-instruct"),
        _m("EmbedGemma", name="chat model"),  # id hit
    ]
    out = LlamaCppAdapter().filter_embedding_models(models)
    assert [m["model_id"] for m in out] == ["nomic-embed-text-v1.5", "EmbedGemma"]


def test_llcpp_audio_filter_flags_and_id():
    models = [
        _m("qwen3-asr-1.7b", flags={"audio": True}),
        _m("whisper-large-v3"),
        _m("qwen2.5-0.5b-instruct"),
    ]
    out = LlamaCppAdapter().filter_audio_models(models)
    assert [m["model_id"] for m in out] == ["qwen3-asr-1.7b", "whisper-large-v3"]


def test_llcpp_vision_filter_requires_mmproj_flag():
    models = [
        _m("qwen2.5-vl-7b", flags={"mmproj": True}),
        _m("llava-1.6", flags={"mmproj": True}),
        _m("qwen2.5-0.5b-instruct", flags={"mmproj": False}),
        _m("plain-model"),
    ]
    out = LlamaCppAdapter().filter_vision_models(models)
    assert [m["model_id"] for m in out] == ["qwen2.5-vl-7b", "llava-1.6"]


# ---------------- vLLM filters: HF config.json + families ---------------

@pytest.fixture
def hf_store(tmp_path, monkeypatch):
    """Point path_resolver's stores.yaml at a tmp store dir; helper to
    write config.json for a model dir. vllm_manager reads stores.yaml at
    call time via _load_stores(), so patching STORES_FILE suffices."""
    import path_resolver

    root = tmp_path / "llm-models"
    root.mkdir()
    stores = tmp_path / "stores.yaml"
    stores.write_text(f"stores:\n  llm-models:\n    linux: {root}\n")
    monkeypatch.setattr(path_resolver, "STORES_FILE", str(stores))
    return root


def _hf_model(root, mid, config=None, hf=True):
    d = root / mid.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    if hf:
        (d / "config.json").write_text(json.dumps(config or {}))
    return _m(mid)


def _write_hf_config(root, model_id, config):
    d = root / model_id.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))


def test_vllm_vision_filter_reads_vision_config(hf_store):
    root = hf_store
    models = [
        _m("Qwen2.5-VL-7B-Instruct"),
        _m("qwen2.5-0.5b-instruct"),
    ]
    # write configs: vision_config present on the VL model
    _write_hf_config(root, "Qwen2.5-VL-7B-Instruct", {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "vision_config": {"depth": 32}})
    _write_hf_config(root, "qwen2.5-0.5b-instruct",
                     {"architectures": ["Qwen2ForCausalLM"]})
    out = VllmAdapter().filter_vision_models(models)
    assert [m["model_id"] for m in out] == ["Qwen2.5-VL-7B-Instruct"]


def test_vllm_vision_filter_known_families_without_config(hf_store):
    # config.json unreadable -> family substring still catches it
    models = [_m("llava-hf/llava-1.5-7b-hf"), _m("qwen2.5-0.5b-instruct")]
    out = VllmAdapter().filter_vision_models(models)
    assert [m["model_id"] for m in out] == ["llava-hf/llava-1.5-7b-hf"]


def test_vllm_embedding_filter_config_plus_families(hf_store):
    root = hf_store
    _write_hf_config(root, "bge-m3", {"architectures": ["XLMRobertaModel"]})
    _write_hf_config(root, "qwen2.5-0.5b-instruct",
                     {"architectures": ["Qwen2ForCausalLM"]})
    models = [
        _m("bge-m3"),                        # config.json arch hit
        _m("BAAI/bge-large-en-v1.5"),        # family substring hit
        _m("nomic-embed-text-v1.5"),         # family substring hit
        _m("qwen2.5-0.5b-instruct"),
    ]
    out = VllmAdapter().filter_embedding_models(models)
    assert [m["model_id"] for m in out] == [
        "bge-m3", "BAAI/bge-large-en-v1.5", "nomic-embed-text-v1.5"]


def test_vllm_audio_filter_empty_but_tab_hidden():
    """vLLM has no ASR models yet — filter returns [] and the tab is
    hidden via ui_hints.hidden_tabs (the actual G-8 mechanism)."""
    assert VllmAdapter().filter_audio_models([_m("qwen2.5-0.5b-instruct")]) == []
    assert "audio" in VllmAdapter().ui_hints()["hidden_tabs"]
    assert LlamaCppAdapter().ui_hints()["hidden_tabs"] == []


# ---------------- asr_language_system_message (G-8) ----------------------

def test_llcpp_asr_language_message_format():
    msg = LlamaCppAdapter().asr_language_system_message("de")
    assert "de" in msg and "asr_text" in msg


def test_vllm_asr_language_message_is_none():
    assert VllmAdapter().asr_language_system_message("de") is None


def test_llcpp_asr_language_empty_is_none():
    assert LlamaCppAdapter().asr_language_system_message("") is None


# ---------------- endpoint dispatch --------------------------------------

@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def test_endpoints_dispatch_through_adapter(client, monkeypatch):
    """All three /api/<modality>/models endpoints call the engine adapter's
    filter — no inline heuristics left in app.py."""
    seen = []

    class _Engine:
        name = "llama_cpp"
        def filter_embedding_models(self, models):
            seen.append("emb")
            return [{"model_id": "e1"}]
        def filter_vision_models(self, models):
            seen.append("vis")
            return [{"model_id": "v1"}]
        def filter_audio_models(self, models):
            seen.append("aud")
            return [{"model_id": "a1"}]

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    for url, tag in [("/api/embeddings/models", "e"),
                     ("/api/vision/models", "v"),
                     ("/api/audio/models", "a")]:
        r = client.get(url)
        assert r.status_code == 200
        assert r.json() == [{"model_id": tag + "1"}]
    assert seen == ["emb", "vis", "aud"]


def test_audio_post_uses_engine_language_gate(client, monkeypatch):
    """POST /api/audio: the language system message comes from the
    adapter; a vLLM engine returns None -> no message is sent. Swap
    status is stubbed running so the handler reaches do(); the urlopen to
    the fake swap port then fails with 500, but the adapter was consulted
    first — that's the contract under test."""
    captured = {}

    class _Engine:
        name = "vllm"
        def asr_language_system_message(self, language):
            captured["language"] = language
            return None
        def filter_audio_models(self, models):
            return []

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    import swap_manager
    monkeypatch.setattr(swap_manager, "status", lambda: {
        "running": True, "host": "aibox1", "port": 59999,
        "host_ip": "127.0.0.1"})
    r = client.post("/api/audio", json={
        "model": "asr", "audio": "data:audio/wav;base64,AAAA", "language": "fr"})
    assert r.status_code in (500, 502)  # request built then connection refused
    assert captured.get("language") == "fr"


def test_settings_endpoint_exposes_hidden_tabs(client, monkeypatch):
    """GET /api/settings rides the active engine's ui_hints through so the
    frontend can hide engine-exclusive tabs."""
    class _Engine:
        name = "vllm"
        # DEC-01 protocol members (settings endpoint injects via the adapter)
        engine_id = "vllm"
        config_section = "vllm"
        def settings_section(self, raw_settings):
            return raw_settings
        def ui_manifest(self):
            # DEC-02 contract member: the manifest is what GET /api/settings
            # injects; ui_hints rides along as the deprecated shim
            return {"labels": {}, "hidden_panels": [], "hidden_tabs": ["audio"],
                    "model_editor": {"sections": []}, "flag_presets": {},
                    "launch": {"mode": "swap_only", "probe_label": "HTTP /health"},
                    "downloads": {"formats": ["hf"], "browse_source": "hf_repos"}}
        def ui_hints(self):
            return {"labels": {}, "hidden_panels": [], "hidden_tabs": ["audio"],
                    "flag_presets": {}}
        def validate_hosts(self, machines):
            return []

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    hints = body.get("ui_hints") or {}
    assert hints.get("hidden_tabs") == ["audio"]
    # DEC-02: the manifest is the injected contract
    assert (body.get("ui_manifest") or {}).get("hidden_tabs") == ["audio"]