"""DEC-07 (acquisition engine gate): adapter acquire/browse/
validate_downloaded members + the engine-gate on POST /api/downloads and
GET /api/scan, and the reverse format gate in rpc_manager.validate_model.

- Adapter members: LlamaCppAdapter.acquire refuses repo=True and
  non-GGUF links; VllmAdapter.acquire refuses bare .gguf links; both
  delegate to the right hf_downloader starter otherwise. browse()
  proxies gguf vs safetensors search. validate_downloaded() checks the
  engine's servable shape (.gguf file vs config.json+safetensors dir).
- POST /api/downloads: in llama.cpp mode a GGUF link works and an HF
  repo download 409s; in vLLM mode an HF repo works and a GGUF 409s.
  The gate fires BEFORE any hf_downloader call (no job is created).
- GET /api/scan: hf entries disappear in llama.cpp mode, gguf entries
  disappear in vLLM mode, projectors survive both.
- rpc_manager.validate_model (reverse gate): hf-format model under
  llama.cpp is an error; gguf under vLLM only warns.

Heavy externals (HF network, subprocess downloads) are monkeypatched —
no real downloads, no network.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402
import config_loader  # noqa: E402
import engine_registry  # noqa: E402


# ---------------- fixtures (same contract as test_engine_endpoints) ---------

@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    import engine_registry
    eng = tmp_path / "settings.yaml"
    eng.write_text("")  # empty -> default engine (llama_cpp)
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
    )
    monkeypatch.setattr(config_loader.machines, "path", str(machines))
    import rpc_manager
    monkeypatch.setattr(rpc_manager, "cfg", config_loader)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(app_module.cfg, "MODELS_DIR", str(models_dir))

    monkeypatch.setattr(config_loader, "groups",
                        __import__("types").SimpleNamespace(
                            all=lambda: [], get=lambda n: None))

    # download store pointing into tmp
    store_root = tmp_path / "llm-models"
    store_root.mkdir()
    stores = tmp_path / "stores.yaml"
    stores.write_text(f"stores:\n  llm-models: {store_root}\n")
    import path_resolver
    monkeypatch.setattr(path_resolver, "STORES_FILE", str(stores))

    yield str(eng)


@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def _set_engine(engine, tmp_file):
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(f"engine: {engine}\n")


# ---------------- adapter members -------------------------------------------

class _Recording:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def test_llamacpp_adapter_acquire_refuses_repo_and_nongguf(monkeypatch):
    ad = engine_registry.get_engine({"engine": "llama_cpp"})
    with pytest.raises(ValueError):
        ad.acquire("hf://Qwen/Qwen3-0.6B", "llm-models", repo=True)
    with pytest.raises(ValueError):
        ad.acquire("hf://Qwen/Qwen3-0.6B/model.safetensors", "llm-models")


def test_llamacpp_adapter_acquire_routes_start_and_start_shards(monkeypatch):
    import hf_downloader
    calls = {}

    def fake_start(link, store):
        calls["start"] = (link, store)
        return {"id": "s1"}

    def fake_shards(link, store):
        calls["shards"] = (link, store)
        return {"id": "s2"}

    monkeypatch.setattr(hf_downloader, "start", fake_start)
    monkeypatch.setattr(hf_downloader, "start_shards", fake_shards)
    ad = engine_registry.get_engine({"engine": "llama_cpp"})
    assert ad.acquire("hf://o/r/file.gguf", "llm-models",
                       all_shards=False)["id"] == "s1"
    assert calls["start"] == ("hf://o/r/file.gguf", "llm-models")
    # default all_shards=True keeps the historical split-set behavior
    assert ad.acquire("hf://o/r/base-00001-of-00002.gguf",
                      "llm-models")["id"] == "s2"
    assert calls["shards"] == ("hf://o/r/base-00001-of-00002.gguf",
                               "llm-models")


def test_vllm_adapter_acquire_refuses_bare_gguf(monkeypatch):
    ad = engine_registry.get_engine({"engine": "vllm"})
    with pytest.raises(ValueError):
        ad.acquire("hf://o/r/file.gguf", "llm-models")


def test_vllm_adapter_acquire_routes_repo_and_shards(monkeypatch):
    import hf_downloader
    calls = {}

    def fake_repo(repo_id, store):
        calls["repo"] = (repo_id, store)
        return {"id": "r1"}

    def fake_shards(link, store):
        calls["shards"] = (link, store)
        return {"id": "r2"}

    monkeypatch.setattr(hf_downloader, "start_repo", fake_repo)
    monkeypatch.setattr(hf_downloader, "start_shards", fake_shards)
    ad = engine_registry.get_engine({"engine": "vllm"})
    assert ad.acquire("hf://Qwen/Qwen3-0.6B", "llm-models",
                       repo=True)["id"] == "r1"
    assert calls["repo"] == ("Qwen/Qwen3-0.6B", "llm-models")
    assert ad.acquire("hf://o/r/m-00001-of-00002.safetensors",
                      "llm-models")["id"] == "r2"


def _run(coro):
    import asyncio
    return asyncio.new_event_loop().run_until_complete(coro)


def test_llamacpp_adapter_browse_gguf(monkeypatch):
    import hf_browse
    calls = {}

    async def fake_gguf(**kw):
        calls.update(kw)
        return {"models": ["gg"]}

    async def fake_st(**kw):
        raise AssertionError("safetensors search must not run in llama mode")
    monkeypatch.setattr(hf_browse, "search_gguf", fake_gguf)
    monkeypatch.setattr(hf_browse, "search_safetensors", fake_st)
    ad = engine_registry.get_engine({"engine": "llama_cpp"})
    res = _run(ad.browse(query="qwen", quant="q4", author="a"))
    assert res == {"models": ["gg"]}
    assert calls["query"] == "qwen" and calls["quant"] == "q4"


def test_vllm_adapter_browse_safetensors(monkeypatch):
    import hf_browse
    calls = {}

    async def fake_st(**kw):
        calls.update(kw)
        return {"models": ["st"]}

    async def fake_gguf(**kw):
        raise AssertionError("gguf search must not run in vllm mode")
    monkeypatch.setattr(hf_browse, "search_safetensors", fake_st)
    monkeypatch.setattr(hf_browse, "search_gguf", fake_gguf)
    ad = engine_registry.get_engine({"engine": "vllm"})
    res = _run(ad.browse(query="qwen"))
    assert res == {"models": ["st"]}
    assert calls["query"] == "qwen"


def test_validate_downloaded_shapes(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    hf_dir = tmp_path / "HFModel"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")

    import path_resolver
    monkeypatch.setattr(path_resolver, "resolve", lambda p: str(tmp_path / p))

    lc = engine_registry.get_engine({"engine": "llama_cpp"})
    assert lc.validate_downloaded("model.gguf") == []
    errs = lc.validate_downloaded("HFModel")
    assert errs and errs[0]["level"] == "error"

    vm = engine_registry.get_engine({"engine": "vllm"})
    errs = vm.validate_downloaded("model.gguf")
    assert errs and errs[0]["level"] == "error"
    errs = vm.validate_downloaded("HFModel")  # config.json, no weights
    assert errs and errs[0]["level"] == "error"
    (hf_dir / "model.safetensors").write_bytes(b"ST")
    assert vm.validate_downloaded("HFModel") == []


# ---------------- POST /api/downloads engine gate ----------------------------

def test_downloads_gguf_ok_llamacpp_mode(client, settings_file, monkeypatch):
    import hf_downloader
    monkeypatch.setattr(hf_downloader, "start_shards",
                        lambda l, s: {"id": "j1", "status": "downloading"})
    r = client.post("/api/downloads", json={
        "link": "hf://o/r/file.gguf", "store": "llm-models"})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "j1"


def test_downloads_repo_409_llamacpp_mode(client, settings_file, monkeypatch):
    import hf_downloader
    fired = []
    monkeypatch.setattr(hf_downloader, "start_repo",
                        lambda *a: fired.append(a) or {})
    r = client.post("/api/downloads", json={
        "link": "hf://Qwen/Qwen3-0.6B", "store": "llm-models",
        "repo": True})
    assert r.status_code == 409
    assert "hf" in r.json()["detail"]
    assert not fired  # gate fires BEFORE any downloader call


def test_downloads_repo_ok_vllm_mode(client, settings_file, monkeypatch):
    _set_engine("vllm", settings_file)
    import hf_downloader
    monkeypatch.setattr(hf_downloader, "start_repo",
                        lambda rid, s: {"id": "j2", "status": "downloading"})
    r = client.post("/api/downloads", json={
        "link": "hf://Qwen/Qwen3-0.6B", "store": "llm-models",
        "repo": True})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "j2"


def test_downloads_gguf_409_vllm_mode(client, settings_file, monkeypatch):
    _set_engine("vllm", settings_file)
    import hf_downloader
    fired = []
    monkeypatch.setattr(hf_downloader, "start_shards",
                        lambda *a: fired.append(a) or {})
    monkeypatch.setattr(hf_downloader, "start",
                         lambda *a: fired.append(a) or {})
    r = client.post("/api/downloads", json={
        "link": "hf://o/r/file.gguf", "store": "llm-models"})
    assert r.status_code == 409
    assert "gguf" in r.json()["detail"].lower()
    assert not fired


# ---------------- GET /api/hf/search format gate -----------------------------

def test_hf_search_gate(client, settings_file, monkeypatch):
    import hf_browse

    async def fake_gguf(**kw):
        return {"models": ["gg"]}

    async def fake_st(**kw):
        return {"models": ["st"]}
    monkeypatch.setattr(hf_browse, "search_gguf", fake_gguf)
    monkeypatch.setattr(hf_browse, "search_safetensors", fake_st)

    # llama.cpp: gguf search fine, safetensors 409
    r = client.get("/api/hf/search", params={"query": "q", "fmt": "gguf"})
    assert r.status_code == 200
    r = client.get("/api/hf/search",
                   params={"query": "q", "fmt": "safetensors"})
    assert r.status_code == 409

    # vllm: safetensors fine, gguf 409
    _set_engine("vllm", settings_file)
    r = client.get("/api/hf/search",
                   params={"query": "q", "fmt": "safetensors"})
    assert r.status_code == 200
    r = client.get("/api/hf/search", params={"query": "q", "fmt": "gguf"})
    assert r.status_code == 409


# ---------------- GET /api/scan engine gate ----------------------------------

def _make_scan_target(tmp_path, monkeypatch):
    root = tmp_path / "scan"
    root.mkdir()
    (root / "model.gguf").write_bytes(b"GGUF")
    hf = root / "HFModel"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    (hf / "m.safetensors").write_bytes(b"ST")
    import model_parser
    return root, model_parser


def test_scan_filters_by_engine(client, settings_file, tmp_path, monkeypatch):
    root, _ = _make_scan_target(tmp_path, monkeypatch)
    import path_resolver
    monkeypatch.setattr(path_resolver, "resolve", lambda p: str(root))
    monkeypatch.setattr(path_resolver, "to_alias", lambda p: str(p))

    # llama.cpp: only the gguf entry
    r = client.get("/api/scan", params={"path": str(root)})
    assert r.status_code == 200, r.text
    fmts = {m.get("format") for m in r.json()["models"]}
    assert fmts == {"gguf"}

    # vllm: only the hf entry
    _set_engine("vllm", settings_file)
    r = client.get("/api/scan", params={"path": str(root)})
    assert r.status_code == 200
    fmts = {m.get("format") for m in r.json()["models"]}
    assert fmts == {"hf"}


# ---------------- reverse format gate in rpc_manager.validate_model ---------

def _write_model(tmp_path, monkeypatch, model_id, body):
    path = tmp_path / "models" / f"{model_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_validate_model_hf_under_llamacpp_is_error(tmp_path, monkeypatch,
                                                   settings_file):
    import rpc_manager
    _write_model(tmp_path, monkeypatch, "m-hf", (
        "model_id: m-hf\n"
        "base_path: store://llm-models/HFModel\n"
        "format: hf\n"
        "servers: [aibox1]\n"
        "device_order: aibox1-CUDA0\n"
    ))
    report = rpc_manager.validate_model("m-hf")
    msgs = [i["msg"] for i in report["issues"] if i["level"] == "error"]
    assert any("hf" in m or "safetensors" in m for m in msgs), report
    assert not report["ok"]


def test_validate_model_gguf_under_vllm_only_warns(tmp_path, monkeypatch,
                                                   settings_file):
    _set_engine("vllm", settings_file)
    import rpc_manager
    _write_model(tmp_path, monkeypatch, "m-gguf", (
        "model_id: m-gguf\n"
        "base_path: store://llm-models/model.gguf\n"
        "servers: [aibox1]\n"
        "device_order: aibox1-CUDA0\n"
    ))
    report = rpc_manager.validate_model("m-gguf")
    errs = [i for i in report["issues"] if i["level"] == "error"]
    warns = [i for i in report["issues"] if i["level"] == "warn"
             and "experimental" in i["msg"]]
    # the gguf-under-vllm gate must be a WARNING, never an error
    assert warns, report
    gguf_errs = [e for e in errs if "not servable" in e["msg"]]
    assert not gguf_errs


def test_validate_model_gguf_under_llamacpp_stays_clean(tmp_path, monkeypatch,
                                                        settings_file):
    import rpc_manager
    _write_model(tmp_path, monkeypatch, "m-ok", (
        "model_id: m-ok\n"
        "base_path: store://llm-models/model.gguf\n"
        "servers: [aibox1]\n"
        "device_order: aibox1-CUDA0\n"
    ))
    report = rpc_manager.validate_model("m-ok")
    gate_issues = [i for i in report["issues"]
                   if "not servable" in i["msg"] or "experimental" in i["msg"]]
    assert not gate_issues, report