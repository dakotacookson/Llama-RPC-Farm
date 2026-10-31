"""Unit tests for vllm_manager probing: /health, /v1/models, /metrics.

No network: _http_get is stubbed (it's the one I/O seam) to return
canned (status, body) pairs, so the probe logic — the part that decides
running/not-running and parses payloads — is exercised exactly, the way
test_rpc_lifecycle stubs the SSH layer.
"""
import json

import vllm_manager
from vllm_manager import probe_capacity, probe_health, probe_identity

MACHINE = {"name": "aibox1", "ip": "10.0.0.98"}


def _stub(monkeypatch, responses):
    """responses: url-suffix -> (status, body). Unlisted URLs get
    (None, '') i.e. connection refused."""
    calls = []

    def fake_get(url, timeout=3):
        calls.append(url)
        for suffix, resp in responses.items():
            if url.endswith(suffix):
                return resp
        return None, ""

    monkeypatch.setattr(vllm_manager, "_http_get", fake_get)
    return calls


# ---------------- /health (liveness) ---------------------------------------

def test_health_200_means_running(monkeypatch):
    _stub(monkeypatch, {"/health": (200, "")})
    out = probe_health(MACHINE, 8000, True)
    assert out == {"running": True, "detail": "vLLM /health 200"}


def test_health_503_is_starting_not_running(monkeypatch):
    """vLLM binds the port early; 503 must read as starting, never as
    up — same signal the swap checkEndpoint relies on."""
    _stub(monkeypatch, {"/health": (503, "")})
    out = probe_health(MACHINE, 8000, False)
    assert out["running"] is False
    assert "503" in out["detail"] and "starting" in out["detail"]


def test_health_unreachable(monkeypatch):
    _stub(monkeypatch, {})
    out = probe_health(MACHINE, 8000, True)
    assert out["running"] is False
    assert "unreachable" in out["detail"]


def test_health_uses_machine_ip_and_health_path(monkeypatch):
    calls = _stub(monkeypatch, {"/health": (200, "")})
    probe_health(MACHINE, 8123, True)
    assert calls == ["http://10.0.0.98:8123/health"]


# ---------------- /v1/models (identity) -------------------------------------

def test_identity_lists_served_models(monkeypatch):
    body = json.dumps({"data": [{"id": "qwen3-8b"},
                                {"id": "qwen3-8b-fp8"}]})
    _stub(monkeypatch, {"/v1/models": (200, body)})
    out = probe_identity(MACHINE, 8000)
    assert out["ok"] is True
    assert out["models"] == ["qwen3-8b", "qwen3-8b-fp8"]
    assert out["match"] is None  # no expected id given -> no check


def test_identity_match_confirms_served_model_name(monkeypatch):
    body = json.dumps({"data": [{"id": "qwen3-8b"}]})
    _stub(monkeypatch, {"/v1/models": (200, body)})
    out = probe_identity(MACHINE, 8000, expected_model_id="qwen3-8b")
    assert out["match"] is True


def test_identity_mismatch_catches_wrong_served_model(monkeypatch):
    """The server came up but serves the wrong name (default id) — the
    identity check is what catches that /health alone never would."""
    body = json.dumps({"data": [{"id": "some-default-name"}]})
    _stub(monkeypatch, {"/v1/models": (200, body)})
    out = probe_identity(MACHINE, 8000, expected_model_id="qwen3-8b")
    assert out["ok"] is True
    assert out["match"] is False


def test_identity_non_json_body(monkeypatch):
    _stub(monkeypatch, {"/v1/models": (200, "<html>not json</html>")})
    out = probe_identity(MACHINE, 8000)
    assert out["ok"] is False
    assert "non-JSON" in out["detail"]


def test_identity_error_status(monkeypatch):
    _stub(monkeypatch, {"/v1/models": (500, "")})
    out = probe_identity(MACHINE, 8000)
    assert out["ok"] is False and out["models"] == []


# ---------------- /metrics (capacity) ----------------------------------------

def test_capacity_parses_vllm_gauges(monkeypatch):
    body = "\n".join([
        "# HELP vllm:gpu_cache_usage_perc GPU cache usage",
        "# TYPE vllm:gpu_cache_usage_perc gauge",
        "vllm:gpu_cache_usage_perc 0.42",
        "vllm:num_requests_waiting 3",
        "python_gc_objects_collected_total 123",  # not a vLLM gauge
    ])
    _stub(monkeypatch, {"/metrics": (200, body)})
    out = probe_capacity(MACHINE, 8000)
    assert out["ok"] is True
    assert out["metrics"] == {"vllm:gpu_cache_usage_perc": 0.42,
                              "vllm:num_requests_waiting": 3.0}
    assert "2 vLLM gauge(s)" in out["detail"]


def test_capacity_unreachable(monkeypatch):
    _stub(monkeypatch, {})
    out = probe_capacity(MACHINE, 8000)
    assert out["ok"] is False and out["metrics"] == {}


def test_capacity_nonnumeric_values_skipped(monkeypatch):
    body = "vllm:gpu_cache_usage_perc 0.42\nvllm:weird not-a-number\n"
    _stub(monkeypatch, {"/metrics": (200, body)})
    out = probe_capacity(MACHINE, 8000)
    assert out["metrics"] == {"vllm:gpu_cache_usage_perc": 0.42}


# ---------------- adapter probe delegates to /health -------------------------

def test_adapter_probe_uses_health(monkeypatch):
    """VllmAdapter.probe must answer the Protocol shape {"running",
    "detail"} via /health — not the RPC TCP check llama.cpp uses."""
    _stub(monkeypatch, {"/health": (200, "")})
    adapter = vllm_manager.VllmAdapter()
    out = adapter.probe(MACHINE, 8000, True)
    assert out == {"running": True, "detail": "vLLM /health 200"}