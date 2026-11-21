"""Virtual load-balanced models over llama-swap replica entries."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import load_balance as lb


def _catalog(monkeypatch):
    cat = {
        "public-model": {
            "public": "public-model",
            "strategy": "least_inflight",
            "replicas": [
                {"key": "public-model__lb-a", "public": "public-model",
                 "label": "a", "weight": 2, "max_inflight": 2},
                {"key": "public-model__lb-b", "public": "public-model",
                 "label": "b", "weight": 1, "max_inflight": 2},
            ],
        }
    }
    monkeypatch.setattr(lb, "_catalog", lambda: cat)
    monkeypatch.setattr(lb, "_INFLIGHT", {})
    monkeypatch.setattr(lb, "_RR", {})
    return cat


def test_replica_key_and_collapse(monkeypatch):
    _catalog(monkeypatch)
    assert lb.replica_key("public-model", "a") == "public-model__lb-a"
    assert lb.public_id("public-model__lb-b") == "public-model"
    assert lb.collapse(["public-model__lb-a", "public-model__lb-b"]) == [
        "public-model"]


def test_least_inflight_prefers_weighted_light_replica(monkeypatch):
    _catalog(monkeypatch)
    # first pick is weighted: a has 2x capacity, so its score is 0.5 vs 1
    first = lb.choose("public-model")
    assert first["key"] == "public-model__lb-a"
    lb.release(first)


def test_round_robin_cycles(monkeypatch):
    cat = _catalog(monkeypatch)
    cat["public-model"]["strategy"] = "round_robin"
    a = lb.choose("public-model")
    b = lb.choose("public-model")
    assert a["key"] != b["key"]
    lb.release(a)
    lb.release(b)


def test_rewrite_model_list_hides_internals_and_exposes_public(monkeypatch):
    _catalog(monkeypatch)
    parsed = {"object": "list", "data": [
        {"id": "public-model__lb-a", "object": "model",
         "status": {"value": "loaded"}},
        {"id": "public-model__lb-b", "object": "model",
         "status": {"value": "unloaded"}},
        {"id": "other", "object": "model"},
    ]}
    out = lb.rewrite_model_list(parsed)
    ids = [m["id"] for m in out["data"]]
    assert "public-model" in ids
    assert "public-model__lb-a" not in ids
    assert "public-model__lb-b" not in ids
    assert "other" in ids
    pub = next(m for m in out["data"] if m["id"] == "public-model")
    assert pub["status"]["value"] == "loaded"


def test_groups_for_model_splits_by_machine():
    model = {"model_id": "m",
             "device_order": "aibox1-CUDA0,aibox1-CUDA1,aibox3-CUDA0"}
    groups = lb.groups_for_model(model)
    by = {g["machine"]: g for g in groups}
    assert set(by) == {"aibox1", "aibox3"}
    assert by["aibox1"]["device_order"] == "aibox1-CUDA0,aibox1-CUDA1"
    assert by["aibox1"]["key"] == "m__lb-aibox1"
    assert by["aibox1"]["weight"] == 2
