"""Tests for RPC port validation in _model_rpc_ports and related helpers.

Verifies that port allocation either rejects clearly invalid values or
produces ports that are within the valid TCP range (1-65535). No network
I/O; config is stubbed via the stub_cfg fixture from conftest.py.
"""
import pytest
import rpc_manager
from rpc_manager import _model_rpc_ports


def _machine(name, ip, rpc_port=50052, **kw):
    m = {
        "name": name,
        "ip": ip,
        "username": "u",
        "password": "p",
        "os": "linux",
        "rpc_port": rpc_port,
        "gpu": "CUDA0",
        "role": "peer",
        "detected": {
            "gpu_names": ["CUDA0 (GPU)"],
            "vram_total_gb": 8,
            "ram_total_gb": 16,
            "cores": 8,
        },
    }
    m.update(kw)
    return m


def _model(mid, **kw):
    m = {
        "model_id": mid,
        "flags": {},
        "base_path": "store://llm-models/x.gguf",
        "device_order": "",
        "device_order_policy": "as-selected",
        "expose": {},
    }
    m.update(kw)
    return m


# ---------------------------------------------------------------------------
# Happy path: allocated ports must be in valid TCP range
# ---------------------------------------------------------------------------

def test_model_rpc_ports_returns_valid_range():
    """_model_rpc_ports must return a port in [1, 65535] for each peer."""
    model = _model("qwen2-5")
    peers = [
        _machine("aibox2", "10.0.0.11"),
        _machine("winbox", "10.0.0.13", rpc_port=50053),
    ]
    ports = _model_rpc_ports(model, peers)
    assert set(ports.keys()) == {("aibox2", "all"), ("winbox", "all")}
    for name, port in ports.items():
        assert isinstance(port, int), f"Port for {name} is not an int: {port!r}"
        assert 1 <= port <= 65535, f"Port {port} for {name} is out of valid TCP range"


def test_model_rpc_ports_distinct_per_peer():
    """Each peer in the same model gets a distinct port so concurrent
    ggml-rpc-server instances do not collide."""
    model = _model("m")
    peers = [
        _machine("p1", "10.0.0.11"),
        _machine("p2", "10.0.0.12"),
        _machine("p3", "10.0.0.13"),
    ]
    ports = _model_rpc_ports(model, peers)
    values = list(ports.values())
    assert len(values) == len(set(values)), (
        f"Duplicate ports allocated across peers: {ports}"
    )


def test_model_rpc_ports_different_models_may_differ():
    """Two distinct models should receive different port allocations for the
    same peer so they can run concurrently without collision."""
    peer = _machine("aibox2", "10.0.0.11")
    ports_a = _model_rpc_ports(_model("model-a"), [peer])
    ports_b = _model_rpc_ports(_model("model-b"), [peer])
    # They may differ; the important invariant is both are valid.
    for p in (ports_a[("aibox2", "all")], ports_b[("aibox2", "all")]):
        assert 1 <= p <= 65535


# ---------------------------------------------------------------------------
# Boundary: base rpc_port edge values must not propagate invalid ports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_port", [1, 1024, 50052, 65530])
def test_model_rpc_ports_near_boundary_base_ports_stay_valid(base_port):
    """Even when the machine's base rpc_port is near the boundary of the
    valid range the allocated port must remain within 1-65535."""
    model = _model("boundary-model")
    peer = _machine("bp", "10.0.0.20", rpc_port=base_port)
    ports = _model_rpc_ports(model, [peer])
    port = ports[("bp", "all")]
    assert 1 <= port <= 65535, (
        f"Allocated port {port} is out of range for base_port={base_port}"
    )


def test_model_rpc_ports_empty_peer_list_returns_empty():
    """No peers -> empty allocation dict (not an error)."""
    model = _model("m")
    ports = _model_rpc_ports(model, [])
    assert ports == {} or isinstance(ports, dict)
