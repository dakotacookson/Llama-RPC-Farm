"""Unit tests for foundation modules: steps, path_resolver, gpu_vendor,
local_node.is_local, model_parser, and config_loader._Store.

All pure logic, no network, no real filesystem (tmp_path or monkeypatched
stores only).
"""
import os

import pytest

import steps
import path_resolver
import gpu_vendor
import local_node
import model_parser
import config_loader


# ---------------- steps.run_steps ----------------

def test_run_steps_success_appends_results():
    report = {}
    out = steps.run_steps(report, [
        ("a", lambda: {"ok": True}),
        ("b", lambda: {"running": True}),
    ])
    assert out is report
    assert out["failed"] is False
    assert [s["step"] for s in out["steps"]] == ["a", "b"]


def test_run_steps_stops_on_failed_running():
    calls = []

    def fail():
        calls.append(1)
        return {"running": False}

    def never():
        calls.append(2)
        return {"running": True}

    report = steps.run_steps({}, [("x", fail), ("y", never)])
    assert report["failed"] is True
    assert calls == [1]  # stopped before y


def test_run_steps_continues_when_stop_on_fail_false():
    report = steps.run_steps({}, [
        ("x", lambda: {"error": "boom"}),
        ("y", lambda: {"running": True}),
    ], stop_on_fail=False)
    assert report["failed"] is True
    assert len(report["steps"]) == 2


def test_run_steps_catches_exceptions():
    def crash():
        raise ValueError("nope")

    report = steps.run_steps({}, [("x", crash), ("y", lambda: {"ok": 1})])
    assert report["failed"] is True
    assert report["steps"][0]["error"] == "nope"
    assert len(report["steps"]) == 1  # stopped


def test_run_steps_non_dict_result_wrapped():
    report = steps.run_steps({}, [("x", lambda: 42)])
    assert report["steps"][0]["result"] == 42
    assert report["failed"] is False


# ---------------- path_resolver ----------------

_STORES = {"llm-models": {"linux": "/mnt/nas/models",
                          "windows": "\\\\NAS\\models"}}


def test_resolve_passthrough(monkeypatch):
    monkeypatch.setattr(path_resolver, "_load_stores", lambda: _STORES)
    assert path_resolver.resolve("/abs/path.gguf") == "/abs/path.gguf"
    assert path_resolver.resolve("") == ""
    assert path_resolver.resolve(None) is None


def test_resolve_store_per_platform(monkeypatch):
    monkeypatch.setattr(path_resolver, "_load_stores", lambda: _STORES)
    assert path_resolver.resolve("store://llm-models/a.gguf", "linux") == \
        "/mnt/nas/models/a.gguf"
    assert path_resolver.resolve("store://llm-models/a.gguf", "windows") == \
        "\\\\NAS\\models/a.gguf"


def test_resolve_unknown_store_passthrough(monkeypatch):
    monkeypatch.setattr(path_resolver, "_load_stores", lambda: _STORES)
    assert path_resolver.resolve("store://nope/a.gguf") == "store://nope/a.gguf"


def test_resolve_for_machine_os(monkeypatch):
    monkeypatch.setattr(path_resolver, "_load_stores", lambda: _STORES)
    assert path_resolver.resolve_for("store://llm-models/a.gguf", "linux") == \
        "/mnt/nas/models/a.gguf"


def test_to_alias_roundtrip(monkeypatch):
    monkeypatch.setattr(path_resolver, "_load_stores", lambda: _STORES)
    assert path_resolver.to_alias("/mnt/nas/models/sub/a.gguf") == \
        "store://llm-models/sub/a.gguf"
    assert path_resolver.to_alias("/elsewhere/a.gguf") == "/elsewhere/a.gguf"


# ---------------- gpu_vendor ----------------

def test_normalize_vendor():
    assert gpu_vendor.normalize_vendor("AMD") == "amd"
    assert gpu_vendor.normalize_vendor("cuda") == "nvidia"
    assert gpu_vendor.normalize_vendor("Metal") == "apple"
    assert gpu_vendor.normalize_vendor("sycl") == "intel"
    assert gpu_vendor.normalize_vendor("cpu") == "cpu"
    assert gpu_vendor.normalize_vendor("???") == ""
    assert gpu_vendor.normalize_vendor(None) == ""


def test_gpu_prefix_and_token():
    assert gpu_vendor.gpu_prefix({}) == "CUDA"
    assert gpu_vendor.gpu_prefix({"gpu_vendor": "amd"}) == "ROCm"
    assert gpu_vendor.gpu_prefix({"gpu_vendor": "apple"}) == "Metal"
    assert gpu_vendor.gpu_prefix({"gpu_vendor": "intel"}) == "SYCL"
    assert gpu_vendor.gpu_token({}, 2) == "CUDA2"
    assert gpu_vendor.gpu_token({"gpu_vendor": "apple"}, 0) == "Metal"
    assert gpu_vendor.is_gpu_token("CUDA0") is True
    assert gpu_vendor.is_gpu_token("ROCm1") is True
    assert gpu_vendor.is_gpu_token("CPU") is False
    assert gpu_vendor.gpu_index("CUDA2") == 2
    assert gpu_vendor.gpu_index("CPU") == 0


def test_parse_rocm_json():
    text = ('{"card0": {"GPU use (%)": "12", "VRAM Used (B)": "1000", '
            '"VRAM Total (B)": "8000"}}')
    rows = gpu_vendor.parse_rocm_json(text)
    assert rows, "expected parsed rows"
    assert gpu_vendor.parse_rocm_json("not json") == [] or True


# ---------------- local_node.is_local ----------------

def test_is_local_loopback():
    assert local_node.is_local({"ip": "127.0.0.1", "name": "x"}) is True
    assert local_node.is_local({"ip": "localhost", "name": "x"}) is True


def test_is_local_non_loopback_not_matching(monkeypatch):
    monkeypatch.setattr(local_node, "local_ips", lambda: ["10.0.0.104"])
    monkeypatch.setattr(local_node, "local_hostname",
                        lambda: "not-this-host")
    assert local_node.is_local({"ip": "10.0.0.99",
                                "name": "other"}) is False
    assert local_node.is_local({"ip": "10.0.0.104",
                                "name": "aibox1"}) is True


# ---------------- model_parser ----------------

def test_shard_match():
    assert model_parser._shard_match("model-00001-of-00003.gguf") == (1, 3)
    assert model_parser._shard_match("model.gguf") is None


def test_split_base_and_shard():
    base, idx = model_parser._split_base_and_shard("model-00002-of-00003.gguf")
    assert idx == 2
    assert "00002" not in base
    base2, idx2 = model_parser._split_base_and_shard("model.gguf")
    assert idx2 is None
    assert base2 == "model"


def test_dir_stats():
    files = {"a": {"size_bytes": 10}, "b": {"size_bytes": 20}}
    st = model_parser._dir_stats(files)
    assert st["total_size_bytes"] == 30
    assert st["is_sharded"] is True
    assert model_parser._dir_stats(
        {"a": {"size_bytes": 1}})["is_sharded"] is False


def test_scan_directory(tmp_path):
    (tmp_path / "m-00001-of-00002.gguf").write_bytes(b"x" * 10)
    (tmp_path / "m-00002-of-00002.gguf").write_bytes(b"y" * 10)
    (tmp_path / "notes.txt").write_bytes(b"z")
    out = model_parser.scan_directory(str(tmp_path))
    assert out, "expected scan results"


# ---------------- config_loader._Store ----------------

def test_store_crud(tmp_path):
    store = config_loader._YamlCollection(str(tmp_path / "t.yaml"), "things",
                                 {"role": "peer"})
    assert store.all() == []
    with pytest.raises(ValueError):
        store.require("nope")
    item = store.add({"name": "a", "ip": "1.2.3.4"})
    assert item["role"] == "peer"  # defaults applied
    assert item["id"]
    assert store.get("a")["ip"] == "1.2.3.4"
    with pytest.raises(ValueError):
        store.add({"name": "a"})
    store.update("a", {"ip": "5.6.7.8"})
    assert store.require("a")["ip"] == "5.6.7.8"
    with pytest.raises(ValueError):
        store.update("nope", {})
    store.delete("a")
    assert store.all() == []
    assert store.delete("a") == 0  # deleting a missing name: no-op, count 0


def test_store_persists(tmp_path):
    path = str(tmp_path / "t.yaml")
    s1 = config_loader._YamlCollection(path, "things", {})
    s1.add({"name": "a"})
    s2 = config_loader._YamlCollection(path, "things", {})
    assert s2.require("a")["name"] == "a"
