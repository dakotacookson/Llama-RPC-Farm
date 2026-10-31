"""Unit tests for the per-(model,peer) RPC port allocation and the
per-port state/log naming that underpins co-serving on a shared peer.

These are pure-logic: deterministic hashing, collision-freedom within a
model, distinct ports across models on the same peer, per-port state/log
paths, and that the base port keeps the legacy bare names for backward
compat with existing single-server installs.
"""
import rpc_manager
from rpc_manager import (
    _model_rpc_port, _model_rpc_ports,
    _rpc_state_path, _rpc_state_tmp, _rpc_log_name,
    _rpc_state_want,
)


# ---------------- port allocation ----------------

def test_port_deterministic():
    """Same (model, peer) always yields the same port."""
    assert _model_rpc_port("qwen-coder", "winbox") == _model_rpc_port("qwen-coder", "winbox")


def test_port_in_range():
    """Port is within [base, base+RANGE) — the high base (51000) keeps
    it clear of Windows' Hyper-V/WSL excluded port ranges."""
    for mid in ("a", "qwen2-5-coder-1-5b-instruct-q2-k", "x" * 200):
        for peer in ("winbox", "aibox2", "p"):
            p = _model_rpc_port(mid, peer)
            assert rpc_manager.RPC_PORT_BASE <= p < rpc_manager.RPC_PORT_BASE + rpc_manager.RPC_PORT_RANGE


def test_port_distinct_models_same_peer():
    """Two different models on the same peer get different ports (the
    whole point: co-served models can't share a single-client server)."""
    a = _model_rpc_port("qwen2-5-1-5b-instruct-q2-k", "winbox")
    b = _model_rpc_port("qwen2-5-coder-1-5b-instruct-q2-k", "winbox")
    assert a != b


def test_port_distinct_peers_same_model():
    """Same model, different peers -> different ports."""
    m = "qwen2-5-1-5b-instruct-q2-k"
    peers = ["aibox1", "aibox2", "aibox3", "winbox"]
    ports = [_model_rpc_port(m, p) for p in peers]
    assert len(set(ports)) == len(peers)


def test_ports_collision_free_within_model():
    """_model_rpc_ports guarantees no two peers of the same model collide."""
    model = {"model_id": "big-model"}
    peers = [{"name": f"node{i}"} for i in range(30)]
    ports = _model_rpc_ports(model, peers)
    vals = list(ports.values())
    assert len(set(vals)) == len(vals)
    # all in the high safe range
    assert all(rpc_manager.RPC_PORT_BASE <= v < rpc_manager.RPC_PORT_BASE + rpc_manager.RPC_PORT_RANGE
               for v in vals)


def test_ports_clear_windows_excluded_ranges():
    """Per-model ports must land ABOVE Windows' Hyper-V/WSL excluded
    port ranges (which typically occupy 50000-50500). A port in those
    ranges fails to bind with 'Failed to create server socket'."""
    for mid in ("qwen2-5-coder-1-5b-instruct-q2-k", "qwen2-5-1-5b-instruct-q2-k", "m"):
        for peer in ("winbox", "aibox1", "aibox2", "aibox3"):
            assert _model_rpc_port(mid, peer) >= 51000, (
                f"port {_model_rpc_port(mid, peer)} for {mid}/{peer} "
                f"is below the Windows-safe base 51000")


# ---------------- _is_excluded_port / _safe_port ----------------

def test_is_excluded_port_known_ranges():
    """Ports in the observed Windows Hyper-V/WSL excluded ranges are
    detected (these cause 'Failed to create server socket')."""
    for bad in (50058, 50102, 50157, 50200, 50400, 49700, 49950):
        assert rpc_manager._is_excluded_port(bad), bad


def test_is_excluded_port_safe_ports_pass():
    for ok in (50052, 51000, 51304, 52000, 60000):
        assert not rpc_manager._is_excluded_port(ok), ok


def test_safe_port_bumps_excluded_to_high_base():
    """A user-set machine rpc_port in a bad range is nudged to the safe
    high base so the rpc-server can actually bind."""
    assert rpc_manager._safe_port(50058) >= 51000
    assert rpc_manager._safe_port(50157) >= 51000
    # a safe port is returned unchanged
    assert rpc_manager._safe_port(50052) == 50052
    assert rpc_manager._safe_port(51304) == 51304


# ---------------- per-port state / log paths ----------------

def test_base_port_keeps_legacy_names():
    """Base port (50052) keeps the bare rpc.state / rpc-server.log names
    so an existing single-server install still reuses cleanly."""
    assert _rpc_state_path(50052, False) == ".cache/llama-wrapper/rpc.state"
    assert _rpc_log_name(50052) == "rpc-server.log"
    assert _rpc_state_tmp(50052, False) == "/tmp/llama-wrapper-rpc.state"
    assert _rpc_state_tmp(50052, True) == "$env:TEMP/llama-wrapper-rpc.state"


def test_per_port_names_include_port():
    p = 50157
    assert _rpc_state_path(p, False) == f".cache/llama-wrapper/rpc.state.{p}"
    assert _rpc_log_name(p) == f"rpc-server-{p}.log"
    assert _rpc_state_tmp(p, False) == f"/tmp/llama-wrapper-rpc.state.{p}"
    assert _rpc_state_tmp(p, True) == f"$env:TEMP/llama-wrapper-rpc.state.{p}"


# ---------------- _rpc_state_want ----------------

def test_state_want_gpu_list():
    m = {"expose": {"mode": "vram", "gpus": ["CUDA0", "CUDA1"]}}
    assert _rpc_state_want(m) == "CUDA0,CUDA1"


def test_state_want_ram_mode_no_gpus():
    m = {"expose": {"mode": "ram", "gpus": []}}
    assert _rpc_state_want(m) == "CPU"


def test_state_want_device_list_override():
    m = {"expose": {"mode": "vram", "gpus": ["CUDA0"]}}
    # explicit device_list wins over expose config
    assert _rpc_state_want(m, device_list=["CUDA0", "CPU"]) == "CUDA0,CPU"


def test_state_want_default():
    m = {"expose": {"mode": "both", "gpus": []}}
    assert _rpc_state_want(m) == "default"


# ---------------- _parse_chip / _gpu_sort_key ----------------

def test_parse_chip_splits_on_last_hyphen():
    assert rpc_manager._parse_chip("aibox1-CUDA0") == ("aibox1", "CUDA0")
    # machine names may contain hyphens -> split on LAST hyphen
    assert rpc_manager._parse_chip("my-box-2-RAM") == ("my-box-2", "RAM")


def test_parse_chip_bare_token():
    assert rpc_manager._parse_chip("CUDA0") == ("", "CUDA0")
    assert rpc_manager._parse_chip("") == ("", "")


def test_gpu_sort_key_numeric_order():
    names = ["CUDA10", "CUDA2", "CUDA1", "CPU"]
    assert sorted(names, key=rpc_manager._gpu_sort_key) == ["CPU", "CUDA1", "CUDA2", "CUDA10"]