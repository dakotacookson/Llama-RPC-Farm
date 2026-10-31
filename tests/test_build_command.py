"""Unit tests for build_llama_command and the device-order -> --device
RPC mapping. These are the pieces most likely to silently break when
flags/peers/ports change, and they're pure string/logic (no network).

build_llama_command is imported from rpc_manager and exercised with
hand-built model dicts + peer lists so the output can be asserted
without any real machine or SSH.
"""
import shlex

import rpc_manager
from rpc_manager import build_llama_command, _split_servers, _device_order_to_llama


def _machine(name, ip, os="linux", **kw):
    m = {"name": name, "ip": ip, "username": "u", "password": "p",
         "os": os, "rpc_port": 50052, "gpu": "", "role": "peer",
         "detected": {"gpu_names": [], "vram_total_gb": 0, "ram_total_gb": 16}}
    m.update(kw)
    return m


# ---------------- build_llama_command: --rpc uses per-model ports ----------------

def test_rpc_uses_per_model_ports(stub_cfg, monkeypatch):
    """--rpc must list ip:port with per-(model,peer) ports, not the bare
    machine rpc_port for every peer."""
    model = {"model_id": "qwen2-5-1-5b-instruct-q2-k", "flags": {},
             "base_path": "store://llm-models/x.gguf"}
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("aibox2", "10.0.0.11", gpu="CUDA0")
    p2 = _machine("winbox", "10.0.0.13", os="windows", gpu="CUDA0")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    ports = rpc_manager._model_rpc_ports(model, [p1, p2])
    cmd = build_llama_command(model, [master["name"], p1["name"], p2["name"]],
                              master_name=master["name"])
    rpc_line = [l for l in cmd.splitlines() if "--rpc" in l][0]
    # each peer ip carries its own per-model port, never the bare 50052
    assert f"{p1['ip']}:{ports[('aibox2', 'all')]}" in rpc_line
    assert f"{p2['ip']}:{ports[('winbox', 'all')]}" in rpc_line
    assert "50052" not in rpc_line  # base port never used for per-model rpc


def test_no_rpc_when_single_node(stub_cfg, monkeypatch):
    """A model with no peers (master only) must not emit --rpc."""
    model = {"model_id": "single", "flags": {},
             "base_path": "store://llm-models/x.gguf"}
    master = _machine("aibox1", "10.0.0.98", role="master")
    stub_cfg.update({master["name"]: master})
    cmd = build_llama_command(model, [master["name"]], master_name=master["name"])
    assert "--rpc" not in cmd


# ---------------- build_llama_command: visualizer flags injected ----------------

def test_verbose_flags_injected_when_absent(stub_cfg):
    """The Flow Visualizer's upstream fallback needs -v; build_llama_command
    injects -v --log-timestamps --log-prefix when the model doesn't set them."""
    model = {"model_id": "m", "flags": {"ctx-size": 8192},
             "base_path": "store://llm-models/x.gguf"}
    master = _machine("aibox1", "10.0.0.98", role="master")
    stub_cfg.update({master["name"]: master})
    cmd = build_llama_command(model, [master["name"]], master_name=master["name"])
    flat = " ".join(cmd.split())
    assert "-v " in flat or flat.endswith("-v") or " -v " in flat
    assert "--log-timestamps" in flat
    assert "--log-prefix" in flat


def test_verbose_flags_not_duplicated_when_present(stub_cfg):
    """If the model already sets -v, don't add a second one."""
    model = {"model_id": "m",
             "flags": {"-v": True, "log-timestamps": True, "log-prefix": True},
             "base_path": "store://llm-models/x.gguf"}
    master = _machine("aibox1", "10.0.0.98", role="master")
    stub_cfg.update({master["name"]: master})
    cmd = build_llama_command(model, [master["name"]], master_name=master["name"])
    toks = shlex.split(cmd.replace("\\\n", " ").replace("${PORT}", "8080"))
    assert toks.count("-v") == 1
    assert toks.count("--log-timestamps") == 1
    assert toks.count("--log-prefix") == 1


# ---------------- build_llama_command: flags emitted verbatim ----------------

def test_flags_emitted_in_order(stub_cfg):
    model = {"model_id": "m",
             "flags": {"threads": 8, "ctx-size": 4096, "jinja": True},
             "base_path": "store://llm-models/x.gguf"}
    master = _machine("aibox1", "10.0.0.98", role="master")
    stub_cfg.update({master["name"]: master})
    cmd = build_llama_command(model, [master["name"]], master_name=master["name"])
    flat = cmd.replace("\\\n", " ")
    # user flags appear in stored order; bool True emits the bare flag
    assert "--threads 8" in flat
    assert "--ctx-size 4096" in flat
    assert "--jinja" in flat


def test_empty_string_flag_dropped(stub_cfg):
    """An empty-valued flag must be dropped (emitting it bare would
    swallow the next flag as its value)."""
    model = {"model_id": "m",
             "flags": {"cors-origins": "", "ctx-size": 1024},
             "base_path": "store://llm-models/x.gguf"}
    master = _machine("aibox1", "10.0.0.98", role="master")
    stub_cfg.update({master["name"]: master})
    cmd = build_llama_command(model, [master["name"]], master_name=master["name"])
    flat = cmd.replace("\\\n", " ")
    assert "--cors-origins" not in flat  # empty -> dropped
    assert "--ctx-size 1024" in flat


# ---------------- _split_servers ----------------

def test_split_servers_master_by_role(stub_cfg):
    a = _machine("a", "1.1.1.1", role="master")
    b = _machine("b", "2.2.2.2")
    c = _machine("c", "3.3.3.3")
    stub_cfg.update({m["name"]: m for m in (a, b, c)})
    master, peers = _split_servers([a["name"], b["name"], c["name"]])
    assert master["name"] == "a"
    assert [p["name"] for p in peers] == ["b", "c"]


def test_split_servers_master_name_override(stub_cfg):
    a = _machine("a", "1.1.1.1", role="master")
    b = _machine("b", "2.2.2.2")
    stub_cfg.update({m["name"]: m for m in (a, b)})
    # explicit master_name wins even if another has role:master
    master, peers = _split_servers([a["name"], b["name"]], master_name="b")
    assert master["name"] == "b"
    assert [p["name"] for p in peers] == ["a"]


def test_split_servers_falls_back_to_first(stub_cfg):
    a = _machine("a", "1.1.1.1")
    b = _machine("b", "2.2.2.2")
    stub_cfg.update({m["name"]: m for m in (a, b)})
    master, peers = _split_servers([a["name"], b["name"]])
    assert master["name"] == "a"
    assert [p["name"] for p in peers] == ["b"]


# ---------------- _device_order_to_llama / _map_devices ----------------

def test_device_order_peer_ram_becomes_rpc():
    """A peer RAM chip maps to its RPC<N> device (peer CPU served by the
    rpc-server); master RAM is implicit (no token)."""
    master = _machine("aibox1", "1.1.1.1", role="master", gpu="CUDA0")
    p1 = _machine("aibox2", "2.2.2.2", gpu="CUDA0")
    order = "aibox1-CUDA0,aibox2-RAM"
    out = _device_order_to_llama(order, master, [p1], "as-selected")
    assert "CUDA0" in out       # master GPU stays CUDA0
    assert "RPC0" in out        # peer RAM -> its RPC0 device
    assert "CPU" not in out     # master RAM would be implicit; not emitted


def test_peer_selected_resources_ram_and_gpus():
    master = _machine("aibox1", "1.1.1.1", role="master", gpu="CUDA0,CUDA1")
    order = "aibox1-CUDA0,aibox1-RAM"
    gpus, ram = rpc_manager._peer_selected_resources(order, "aibox1", master)
    assert gpus == ["CUDA0"]
    assert ram is True


def test_peer_selected_resources_no_chips_uses_all_gpus():
    peer = _machine("aibox2", "2.2.2.2", gpu="CUDA1,CUDA0")
    gpus, ram = rpc_manager._peer_selected_resources("", "aibox2", peer)
    assert gpus == ["CUDA0", "CUDA1"]  # sorted numerically
    assert ram is False


def test_peer_selected_resources_gpuless_peer_is_cpu():
    """A peer with no GPUs and no chips contributes CPU (matches the
    rpc-server default on CPU-only boxes)."""
    peer = _machine("cpu-box", "2.2.2.2", gpu="")
    gpus, ram = rpc_manager._peer_selected_resources("", "cpu-box", peer)
    assert gpus == []
    assert ram is True