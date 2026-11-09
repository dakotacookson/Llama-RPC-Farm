"""Placement recommender: pick the best single box or multi-box RPC
spread from GGUF size + detected hardware. Pure selection logic — GGUF
reads and safe-defaults are stubbed, so no filesystem/network is touched.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import placement  # noqa: E402


class _FakeMachines:
    def __init__(self, machines):
        self._machines = machines

    def all(self):
        return list(self._machines)


def _machine(name, gpu_count, gpu_gb, ram_gb, role="peer", os_type="linux"):
    return {
        "name": name,
        "role": role,
        "os": os_type,
        "gpu": ",".join(f"CUDA{i}" for i in range(gpu_count)),
        "detected": {
            "gpu_names": [f"GPU {i}" for i in range(gpu_count)],
            "vram_total_gb": gpu_gb * gpu_count,
            "ram_total_gb": ram_gb,
            "cores": 12,
        },
    }


def _need(gb, ctx=32768, mmproj=0.0, has_mtp=False):
    return {
        "need_gb": gb,
        "need_bytes": int(gb * 1024 ** 3),
        "weights_gb": gb,
        "kv_gb": 0.0,
        "ssm_gb": 0.0,
        "mmproj_gb": mmproj,
        "draft_gb": 0.0,
        "ctx": ctx,
        "model_max_ctx": 131072,
        "arch": "llama",
        "has_mtp": has_mtp,
    }


def _patch(monkeypatch, machines, need):
    monkeypatch.setattr(placement.cfg, "machines", _FakeMachines(machines))
    monkeypatch.setattr(placement, "_model_needs", lambda *a, **k: need)
    monkeypatch.setattr(
        placement.tuner, "safe_defaults",
        lambda *a, **k: {"flags": {
            "flash-attn": "on", "jinja": True, "cache-type-k": "q8_0",
            "ctx-size": 8192, "-v": True, "log-prefix": True,
        }})


def test_single_box_when_one_machine_fits(monkeypatch):
    machines = [
        _machine("aibox1", 4, 12, 32, role="master"),
        _machine("aibox3", 2, 12, 32),
    ]
    _patch(monkeypatch, machines, _need(20))
    r = placement.recommend("store://llm-models/m.gguf")
    assert r["single_box"] is True
    assert r["servers"] == ["aibox1"]
    # two 12 GB cards are enough (10.3 GB usable each); no need to use all 4
    assert r["device_order"].split(",") == ["aibox1-CUDA0", "aibox1-CUDA1"]
    assert r["fits"] is True
    assert r["host"] == "aibox1"


def test_multi_box_rpc_when_no_single_box_fits(monkeypatch):
    machines = [
        _machine("aibox1", 2, 12, 32, role="master"),
        _machine("aibox3", 2, 12, 32),
    ]
    _patch(monkeypatch, machines, _need(30))
    r = placement.recommend("store://llm-models/m.gguf")
    assert r["single_box"] is False
    assert r["servers"][0] == "aibox1"       # master hosts llama-server
    assert "aibox3" in r["servers"]
    assert r["fits"] is True
    assert r["device_order"].count("CUDA") == 3


def test_host_ram_spill_uses_implicit_memory(monkeypatch):
    machines = [_machine("aibox1", 1, 8, 64, role="master")]
    _patch(monkeypatch, machines, _need(30))
    r = placement.recommend("store://llm-models/m.gguf")
    assert r["single_box"] is True
    # host RAM is implicit to llama-server: no -RAM chip is emitted
    assert r["device_order"] == "aibox1-CUDA0"
    assert r["fits"] is True
    assert any("host RAM spill" in x for x in r["reasons"])


def test_peer_ram_spill_emits_rpc_cpu_chip(monkeypatch):
    machines = [
        _machine("aibox1", 1, 12, 8, role="master"),
        _machine("aibox3", 1, 12, 64),
    ]
    _patch(monkeypatch, machines, _need(30))
    r = placement.recommend("store://llm-models/m.gguf")
    assert "aibox3-RAM" in r["device_order"].split(",")
    assert r["expose"]["aibox3"]["mode"] == "both"
    assert r["fits"] is True


def test_reports_shortfall(monkeypatch):
    machines = [_machine("aibox1", 1, 8, 4, role="master")]
    _patch(monkeypatch, machines, _need(40))
    r = placement.recommend("store://llm-models/m.gguf")
    assert r["fits"] is False
    assert any("may fail to load" in w for w in r["warnings"])


def test_servers_allow_list_filters_machines(monkeypatch):
    machines = [
        _machine("aibox1", 1, 12, 32, role="master"),
        _machine("bigbox", 4, 24, 64),
    ]
    _patch(monkeypatch, machines, _need(20))
    allr = placement.recommend("store://llm-models/m.gguf")
    assert allr["servers"] == ["bigbox"]  # fastest/biggest wins farm-wide
    only = placement.recommend("store://llm-models/m.gguf",
                               servers=["aibox1"])
    assert only["servers"] == ["aibox1"]


def test_windows_peer_warns(monkeypatch):
    machines = [
        _machine("aibox1", 0, 0, 8, role="master"),
        _machine("winbox", 2, 12, 64, os_type="windows"),
    ]
    _patch(monkeypatch, machines, _need(20))
    r = placement.recommend("store://llm-models/m.gguf")
    assert "winbox" in r["servers"]
    assert any("Windows box" in w for w in r["warnings"])


def test_cpu_only_box_still_recommends(monkeypatch):
    machines = [_machine("laptop", 0, 0, 64, role="master")]
    _patch(monkeypatch, machines, _need(8))
    r = placement.recommend("store://llm-models/m.gguf")
    assert r["servers"] == ["laptop"]
    assert r["device_order"] == "laptop-RAM"
    assert r["flags"]["n-gpu-layers"] == 0
    assert r["flags"]["threads"] == 12
    assert r["fits"] is True


def test_baseline_flags_are_clean_and_ctx_locked(monkeypatch):
    machines = [_machine("aibox1", 2, 12, 32, role="master")]
    _patch(monkeypatch, machines, _need(12, ctx=65536, has_mtp=True))
    r = placement.recommend("store://llm-models/m.gguf")
    f = r["flags"]
    assert f["ctx-size"] == 65536
    assert "-v" not in f and "log-prefix" not in f
    assert f["spec-type"] == "draft-mtp"
    assert f["n-gpu-layers"] == 99
    assert r["model_max_ctx"] == 131072


def test_multi_gpu_offers_split_reference(monkeypatch):
    machines = [_machine("aibox1", 3, 12, 32, role="master")]
    _patch(monkeypatch, machines, _need(30))
    r = placement.recommend("store://llm-models/m.gguf")
    assert r["tensor_split_mode"] == "fit"
    assert len(r["suggested_split"]) == 3
    assert all(w >= 1 for w in r["suggested_split"])
