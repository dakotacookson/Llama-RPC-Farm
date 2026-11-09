"""Non-blocking configuration advice: warn before a load fails."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config_advice as ca  # noqa: E402


def _has(lines, needle):
    return any(needle.lower() in ln.lower() for ln in lines)


def test_mlx_peer_and_device_warnings():
    m = {"runtime": "mlx", "servers": ["mac", "aibox1"],
         "device_order": "mac-Metal", "launch_mode": "rpc"}
    w = ca.model_warnings(m)
    assert _has(w, "single-mac")
    assert _has(w, "ignores device_order")
    assert _has(w, "launch_mode")


def test_custom_needs_one_box():
    assert _has(ca.model_warnings(
        {"runtime": "custom", "servers": ["a", "b"]}), "exactly one target")


def test_llama_local_with_peers_warns():
    w = ca.model_warnings({"servers": ["a", "b"], "launch_mode": "local"})
    assert _has(w, "ignores every server")


def test_llama_engine_checks_are_not_duplicated_here():
    # rpc_manager.validate_model owns split/fit/flag/path checks; advice
    # must not repeat them with different wording.
    m = {"servers": ["a", "b"], "launch_mode": "rpc",
         "tensor_split_mode": "fit",
         "device_order": "a-CUDA0,a-RAM"}
    assert not ca.model_warnings(m)
    m = {"servers": ["a", "b"], "launch_mode": "rpc",
         "tensor_split_mode": "split", "tensor_split": "4,4",
         "device_order": "a-CUDA0,b-CUDA0,b-CUDA1"}
    assert not ca.model_warnings(m)


def test_serve_group_conflicts_and_missing_members():
    models = {"a": {}, "b": {}}
    w = ca.serve_group_warnings(
        {"members": ["a", "ghost"], "exclusive": True, "persistent": True},
        models_by_id=models)
    assert _has(w, "unknown model")
    assert _has(w, "exclusive + persistent")
    w2 = ca.serve_group_warnings({"members": ["a", "b"], "persistent": True},
                                 models_by_id=models)
    assert _has(w2, "keep every member loaded")


def test_machine_group_warnings():
    machines = {"a": {"gpu_vendor": "nvidia"},
                "b": {"gpu_vendor": "amd"}}
    w = ca.machine_group_warnings(
        {"machines": ["a", "a", "ghost", "b"]}, machines_by_name=machines)
    assert _has(w, "unknown machine")
    assert _has(w, "duplicate machine")
    assert not _has(w, "mixed gpu vendors")  # mixed fleets are supported
