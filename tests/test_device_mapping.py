"""Unit tests for the device-order pipeline: chip parsing, per-peer
resources, RPC map/legend/labels, --device translation, auto order,
model validation, and variant merging.

These lock the SINGLE-SOURCE-OF-TRUTH contract: the --device RPC ids,
the rpc-server backends, and the visualizer map can never drift apart.
"""
import types

import pytest
import yaml

from conftest import _FakeMachines
import rpc_manager


def _master(name="aibox1"):
    return {"name": name, "ip": "10.0.0.104", "os": "linux",
            "username": "zazi", "gpu": "CUDA0,CUDA1",
            "gpu_vendor": "nvidia"}


def _peer(name="aibox3", ip="10.0.0.98", gpus=("CUDA0", "CUDA1"),
          os="linux"):
    return {"name": name, "ip": ip, "os": os, "username": "zazi",
            "gpu": ",".join(gpus),
            "gpu_vendor": "nvidia" if os == "linux" else "",
            "detected": {"gpu_names": ["NVIDIA GeForce RTX 2060 SUPER",
                                       "NVIDIA GeForce RTX 2060"],
                         "ram_total_gb": 16}}


# ---------------- _peer_selected_resources ----------------

def test_selected_resources_chips():
    order = "aibox1-CUDA0,aibox3-CUDA1,aibox3-RAM"
    gpus, ram = rpc_manager._peer_selected_resources(
        order, "aibox3", _peer())
    assert gpus == ["CUDA1"]
    assert ram is True


def test_selected_resources_falls_back_to_all_peer_gpus():
    gpus, ram = rpc_manager._peer_selected_resources(
        "", "aibox3", _peer())
    assert gpus == ["CUDA0", "CUDA1"]
    assert ram is False


def test_selected_resources_gpUless_peer_gets_cpu():
    gpus, ram = rpc_manager._peer_selected_resources(
        "", "tiny", {"name": "tiny", "gpu": "", "detected": {}})
    assert gpus == [] and ram is True


# ---------------- _peer_server_devices ----------------

def test_server_devices_gpu_and_ram():
    devs = rpc_manager._peer_server_devices(
        "aibox3-CUDA1,aibox3-RAM", _peer())
    assert devs == ["CUDA1", "CPU"]


def test_server_devices_default():
    assert rpc_manager._peer_server_devices("", _peer()) == \
        ["CUDA0", "CUDA1"]


# ---------------- compute_rpc_map / legend / label ----------------

def test_rpc_map_numbering():
    peers = [_peer("aibox3", "10.0.0.98"),
             _peer("winbox", "10.0.0.124", gpus=("CUDA1",), os="windows")]
    m = {"device_order": "aibox1-CUDA0,aibox3-CUDA0,aibox3-CUDA1,winbox-CUDA1"}
    rpc_map = rpc_manager.compute_rpc_map(m, _master(), peers)
    assert rpc_map == {
        "RPC0": {"machine": "aibox3", "resource": "CUDA0"},
        "RPC1": {"machine": "aibox3", "resource": "CUDA1"},
        "RPC2": {"machine": "winbox", "resource": "CUDA1"},
    }


def test_rpc_map_ram_last():
    peers = [_peer()]
    m = {"device_order": "aibox3-CUDA0,aibox3-RAM"}
    rpc_map = rpc_manager.compute_rpc_map(m, _master(), peers)
    assert rpc_map["RPC0"]["resource"] == "CUDA0"
    assert rpc_map["RPC1"] == {"machine": "aibox3", "resource": "RAM"}


def test_rpc_map_empty_single_node():
    assert rpc_manager.compute_rpc_map({}, _master(), []) == {}


def test_rpc_legend_format():
    peers = [_peer()]
    leg = rpc_manager._rpc_legend("aibox3-CUDA0,aibox3-RAM",
                                  _master(), peers)
    assert leg == "RPC0=aibox3-CUDA0,RPC1=aibox3-RAM"


def test_peer_backend_label():
    label = rpc_manager._peer_backend_label(
        "aibox3-CUDA0", _peer())
    assert "CUDA0" in label and "RTX 2060 SUPER" in label
    # shell-safe: no quotes/dollar/backticks
    assert '"' not in label and "$" not in label and "`" not in label


# ---------------- _device_order_to_llama ----------------

def test_device_order_to_llama_basic():
    peers = [_peer("aibox3", "10.0.0.98"),
             _peer("winbox", "10.0.0.124", gpus=("CUDA1",), os="windows")]
    out = rpc_manager._device_order_to_llama(
        "aibox1-CUDA0,aibox1-CUDA1,aibox3-CUDA0,winbox-CUDA1",
        _master(), peers, "as-selected")
    assert out.startswith("CUDA0,CUDA1,")
    assert "RPC0" in out and "RPC1" in out


def test_device_order_master_ram_emits_no_token():
    """Master RAM needs no token (host memory is implicit to
    llama-server); peer RAM becomes an RPC id."""
    out = rpc_manager._device_order_to_llama(
        "aibox1-RAM,aibox3-CUDA0,aibox3-RAM", _master(), [_peer()],
        "as-selected")
    assert "CPU" not in out.split(",")
    assert out.count("RPC") == 2  # peer CUDA0 + peer RAM


# ---------------- _auto_device_order ----------------

def test_auto_device_order_includes_all(monkeypatch):
    master = _master()
    peers = [_peer()]
    monkeypatch.setattr(rpc_manager, "_peer_selected_resources",
                        lambda order, name, peer: (["CUDA0"], False))
    order = rpc_manager._auto_device_order(master, peers)
    # device tokens (master CUDA ids + RPC ids), not machine names
    assert "RPC0" in order and "CUDA" in order


# ---------------- validate_model ----------------

def _cfg_with(model, machines):
    fake = types.SimpleNamespace(
        machines=types.SimpleNamespace(
            require=lambda n: next(m for m in machines
                                   if m["name"] == n),
            all=lambda: machines),
        get_model=lambda mid: model if model["model_id"] == mid else None)
    return fake


def test_validate_missing_model(monkeypatch):
    monkeypatch.setattr(rpc_manager, "cfg",
                        _cfg_with({"model_id": "x"}, []))
    rep = rpc_manager.validate_model("nope")
    assert rep["ok"] is False
    assert rep["issues"]


def test_validate_no_servers(monkeypatch):
    monkeypatch.setattr(rpc_manager, "cfg",
                        _cfg_with({"model_id": "m", "servers": []}, []))
    rep = rpc_manager.validate_model("m")
    assert rep["ok"] is False
    assert any("servers" in i["msg"] for i in rep["issues"])


def test_validate_ok_model(monkeypatch):
    master = _master()
    peer = _peer()
    model = {"model_id": "m", "servers": ["aibox1", "aibox3"],
             "base_path": "/m.gguf", "flags": {},
             "device_order": "aibox1-CUDA0,aibox3-CUDA0"}
    fake = _cfg_with(model, [master, peer])
    monkeypatch.setattr(rpc_manager, "cfg", fake)
    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: True))
    rep = rpc_manager.validate_model("m")
    assert rep["model_id"] == "m"
    assert isinstance(rep["issues"], list)


# ---------------- _resolve_variant ----------------

def test_resolve_variant_none_returns_base():
    m = {"model_id": "m", "ctx": 1}
    assert rpc_manager._resolve_variant(m, None) is m
    assert rpc_manager._resolve_variant(m, "") is m


def test_resolve_variant_merges():
    m = {"model_id": "m", "base_path": "/b.gguf", "name": "M",
         "ctx": 1, "flags": {"threads": 4, "batch-size": 512},
         "variants": [{"label": "fast", "flags": {"threads": 16},
                       "servers": ["aibox1"]}]}

    v = rpc_manager._resolve_variant(m, "fast")
    assert v["flags"] == {"threads": 16, "batch-size": 512}
    assert v["servers"] == ["aibox1"]
    assert v["base_path"] == "/b.gguf"  # base preserved
    assert v["_variant_label"] == "fast"
    assert "variants" not in v
    assert m["flags"] == {"threads": 4, "batch-size": 512}  # no mutation


def test_resolve_variant_unknown_raises():
    with pytest.raises(ValueError):
        rpc_manager._resolve_variant({"model_id": "m", "variants": []},
                                     "nope")


# ---------------- variant vllm overrides (A2) -------------------------------
