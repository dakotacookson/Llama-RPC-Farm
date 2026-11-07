"""Unit tests for the generated start-* script (_render_rpc) and the
Windows .ps1 launcher (_win_launcher).

These parse the generated bash/ps1 as strings and assert the per-model-
port structure: Bash-3.2-safe per-peer lookups (macOS /bin/bash has no
associative arrays), agent-only lifecycle (no SSH/sshpass), per-port log
names, the --rpc list using per-node ports, and the .ps1 taking
-Port/-LogName.
"""
import rpc_manager
from rpc_manager import _render_rpc, _win_launcher, _model_rpc_ports


def _machine(name, ip, os="linux", **kw):
    m = {"name": name, "ip": ip, "username": "u", "password": "p",
         "os": os, "rpc_port": 50052, "gpu": "CUDA0", "role": "peer",
         "detected": {"gpu_names": ["CUDA0 (GPU)"], "vram_total_gb": 8,
                      "ram_total_gb": 16, "cores": 8}}
    m.update(kw)
    return m


def _model(mid, **kw):
    m = {"model_id": mid, "flags": {}, "base_path": "store://llm-models/x.gguf",
         "device_order": "", "device_order_policy": "as-selected",
         "expose": {}}
    m.update(kw)
    return m


# ---------------- _render_rpc: per-port structure ----------------

def test_render_has_rpc_ports_map(stub_cfg):
    model = _model("qwen2-5-1-5b-instruct-q2-k", device_order="aibox1-CUDA0,aibox2-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("aibox2", "10.0.0.11")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    script = _render_rpc(model, master, [p1, p2])
    ports = _model_rpc_ports(model, [p1, p2])
    # Bash 3.2-safe lookup function, not an associative array
    assert "rpc_port_of() {" in script
    assert "declare -A" not in script
    for p in (p1, p2):
        assert (f'"{p["ip"]}") echo "{ports[(p["name"], "all")]}" ;;'
                in script)


def test_render_rpc_list_uses_per_node_ports(stub_cfg):
    model = _model("m", device_order="aibox1-CUDA0,aibox2-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("aibox2", "10.0.0.11")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    script = _render_rpc(model, master, [p1, p2])
    ports = _model_rpc_ports(model, [p1, p2])
    # the --rpc line carries per-node ports
    rpc_line = [l for l in script.splitlines() if "--rpc" in l][0]
    assert f'{p1["ip"]}:{ports[("aibox2", "all")]}' in rpc_line
    assert f'{p2["ip"]}:{ports[("winbox", "all")]}' in rpc_line


def test_render_per_port_state_and_log(stub_cfg):
    model = _model("m", device_order="aibox1-CUDA0,aibox2-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("aibox2", "10.0.0.11")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    script = _render_rpc(model, master, [p1, p2])
    # per-port log names are derived at runtime from the node's port
    assert "rpc_log_for_port() {" in script
    assert 'rpc_log_of() { rpc_log_for_port "$(rpc_port_of "$1")"; }' in script
    assert "rpc-server-$1.log" in script
    # agent-only: no SSH/sshpass in generated launchers
    assert "sshpass" not in script
    assert "SSH_PASS" not in script

def test_render_is_valid_bash(stub_cfg):
    """The generated script must parse as bash (syntax check)."""
    import subprocess
    model = _model("m", device_order="aibox1-CUDA0,aibox2-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("aibox2", "10.0.0.11")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    script = _render_rpc(model, master, [p1, p2])
    proc = subprocess.run(["bash", "-n"], input=script,
                          text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_render_bash32_safe(stub_cfg):
    """macOS /bin/bash is 3.2: no associative arrays. The agent-only
    script must use case lookups and still parse as bash."""
    model = _model("m", device_order="node1-CUDA0,node2-CUDA0,winnode-CUDA0")
    master = _machine("node1", "10.0.0.11", role="master")
    p1 = _machine("node2", "10.0.0.12")
    p2 = _machine("winnode", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    script = _render_rpc(model, master, [p1, p2])
    assert "declare -A" not in script
    for fn in ("peer_devices()", "rpc_port_of()", "peer_label_of()",
               "agent_base_of()", "agent_token_of()", "agent_payload_of()"):
        assert fn in script, fn


def test_render_agent_token_uses_machine_name(stub_cfg):
    """The sidecar exports AGENT_TOKEN_<MACHINE_NAME>; keying the lookup
    by node IP (10_0_0_12) would send an empty bearer to /rpc/ensure."""
    model = _model("m", device_order="node1-CUDA0,node2-CUDA0")
    master = _machine("node1", "10.0.0.11", role="master")
    p1 = _machine("node2", "10.0.0.12")
    stub_cfg.update({m["name"]: m for m in (master, p1)})
    script = _render_rpc(model, master, [p1])
    assert "AGENT_TOKEN_NODE2" in script
    assert "AGENT_TOKEN_10_0_0_12" not in script
    # per-machine env must travel in the /rpc/ensure payload
    assert "CUDA_DEVICE_ORDER" in script
    assert "GGML_RPC_NO_RDMA" in script


def test_win_launcher_formats_exe_and_port():
    """The .ps1 must build the cmd with the exe path formatted (-f binds
    tighter than +) and the runtime $Port spliced in, not a broken
    '-f $exe' that only formats an empty string."""
    m = _machine("winbox", "1.1.1.1", os="windows")
    ps1 = _win_launcher(m, 50157, expose_args=" --device CUDA0", parametrized=True)
    # $base formats exe+port (expose_args baked into the format string);
    # $cmd = $base + extra + redirection
    assert "$base = ('cmd.exe / c" not in ps1  # sanity
    assert "' -f $exe, $Port)" in ps1
    assert "$cmd = $base + $ExtraArgs" in ps1
    # the runtime port is a parameter, not baked in
    assert "param([int]$Port = 50157" in ps1


def test_win_launcher_nonparametrized_bakes_port_and_logname():
    m = _machine("winbox", "1.1.1.1", os="windows")
    ps1 = _win_launcher(m, 50200)
    # non-parametrized bakes the port and the per-port log name
    assert "-p 50200" in ps1
    assert "rpc-server-50200.log" in ps1


# ---------------- CUDA device order reaches the peer ----------------

def test_rpc_env_exports_cuda_order_for_nvidia():
    """rpc-server env must pin PCI_BUS_ID so --device indexes match
    nvidia-smi on heterogeneous peers (3050+4080: fastest-first would
    put --device CUDA1 on the 3050). Unknown vendor defaults to NVIDIA;
    AMD/Intel hosts are exempt (NVIDIA-only var)."""
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in \
        rpc_manager._rpc_env_exports({"name": "winbox"})
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in \
        rpc_manager._rpc_env_exports({"name": "n", "gpu_vendor": "nvidia"})
    assert "CUDA_DEVICE_ORDER" not in \
        rpc_manager._rpc_env_exports({"name": "a", "gpu_vendor": "amd"})
    assert "CUDA_DEVICE_ORDER" not in \
        rpc_manager._rpc_env_exports({"name": "i", "gpu_vendor": "intel"})


def test_win_launcher_inlines_env_into_cmd():
    """Win32_Process.Create ignores the session's $env: (child inherits
    wmiprvse's environment), so vars must ride inline as `set K=V &&`
    in the cmd.exe line or they never reach ggml-rpc-server.exe."""
    m = _machine("winbox", "1.1.1.1", os="windows")
    for ps1 in (_win_launcher(m, 50157, expose_args=" --device CUDA1",
                              parametrized=True),
                _win_launcher(m, 50200)):
        assert "set CUDA_DEVICE_ORDER=PCI_BUS_ID && " in ps1
        assert "set GGML_RPC_NO_RDMA=1 && " in ps1


def test_render_rpc_envs_carry_cuda_order(stub_cfg):
    """Generated .sh RPC_ENVS entries must carry the order pin so both
    the Linux `export $envs` path and the Windows launcher get it."""
    model = _model("order-test")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1)})
    script = _render_rpc(model, master, [p1])
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in script


# ---------------- UUID identity beats index ordering ----------------

def test_cuda_visible_devices_helper():
    """Full UUID set (PCI order) -> CSV; anything partial/mismatched ->
    '' so no card is ever hidden by a partial CUDA_VISIBLE_DEVICES."""
    m = {"detected": {
        "gpu_names": ["RTX 3050", "RTX 4080"],
        "gpu_uuids": ["GPU-aaa", "GPU-bbb"]}}
    import gpu_vendor
    assert gpu_vendor.cuda_visible_devices(m) == "GPU-aaa,GPU-bbb"
    assert gpu_vendor.cuda_visible_devices({"detected": {
        "gpu_names": ["A", "B"], "gpu_uuids": ["GPU-aaa"]}}) == ""
    assert gpu_vendor.cuda_visible_devices({"detected": {
        "gpu_names": ["A"], "gpu_uuids": ["bogus"]}}) == ""
    assert gpu_vendor.cuda_visible_devices({}) == ""


def test_rpc_env_prefers_uuid_visible_over_order_pin():
    """UUIDs present -> CUDA_VISIBLE_DEVICES, no ORDER pin (old runtimes
    fail init under PCI_BUS_ID). No UUIDs -> ORDER pin (status quo)."""
    uuids = {"detected": {"gpu_names": ["S", "F"],
                          "gpu_uuids": ["GPU-aaa", "GPU-bbb"]}}
    env = rpc_manager._rpc_env_exports(dict({"name": "w"}, **uuids))
    assert "CUDA_VISIBLE_DEVICES=GPU-aaa,GPU-bbb" in env
    assert "CUDA_DEVICE_ORDER" not in env
    env2 = rpc_manager._rpc_env_exports({"name": "w"})
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in env2
    assert "CUDA_VISIBLE_DEVICES" not in env2