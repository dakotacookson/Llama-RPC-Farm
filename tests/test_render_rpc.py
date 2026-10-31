"""Unit tests for the generated start-* script (_render_rpc) and the
Windows .ps1 launcher (_win_launcher).

These parse the generated bash/ps1 as strings and assert the per-model-
port structure: a per-peer RPC_PORTS map, port-targeted pkill (not a
blanket pkill of all ggml-rpc-server), per-port rpc.state and log names,
the --rpc list using per-node ports, and the .ps1 taking -Port/-LogName.
"""
import rpc_manager
from rpc_manager import _render_rpc, _win_launcher, _model_rpc_ports, _rpc_log_name


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
    assert "declare -A RPC_PORTS" in script
    for p in (p1, p2):
        assert f'RPC_PORTS["{p["ip"]}"]="{ports[(p["name"], "all")]}"' in script


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


def test_render_kill_targets_port_not_blanket(stub_cfg):
    """The script must kill only the rpc-server on its port, not all
    ggml-rpc-server instances (co-served models on other ports survive)."""
    model = _model("m", device_order="aibox1-CUDA0,aibox2-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("aibox2", "10.0.0.11")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    script = _render_rpc(model, master, [p1, p2])
    # linux: pkill pattern includes -p $port with a trailing-space anchor
    # (port 5109 must never match a co-served model on 51096)
    assert 'pkill -u $LINUX_USER -f \\"ggml-[r]pc-server.*-p $port \\"' in script
    # never the old blanket pkill of all ggml-rpc-server
    assert "pkill -u $LINUX_USER -f ggml-[r]pc-server\" >/dev/null" not in script
    # windows: CIM filter by -p $port (anchored), not taskkill /F /IM
    assert "-like '* -p $port *'" in script
    assert "taskkill /F /IM ggml-rpc-server.exe" not in script


def test_render_per_port_state_and_log(stub_cfg):
    model = _model("m", device_order="aibox1-CUDA0,aibox2-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p1 = _machine("aibox2", "10.0.0.11")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    script = _render_rpc(model, master, [p1, p2])
    ports = _model_rpc_ports(model, [p1, p2])
    # state reads/writes use the per-port $port bash var (rpc.state.$port),
    # never the legacy bare rpc.state path
    assert "rpc.state.$port" in script
    assert "/.cache/llama-wrapper/rpc.state " not in script  # no bare legacy path
    # per-port log names appear in the RPC_LOGS map
    assert _rpc_log_name(ports[("aibox2", "all")]) in script
    assert _rpc_log_name(ports[("winbox", "all")]) in script
    assert "declare -A RPC_LOGS" in script


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


def test_render_no_unbound_var_under_set_u(stub_cfg):
    """The script uses 'set -u'. PowerShell tokens like $env:USERPROFILE
    and $_.CommandLine must be escaped (\\$env / \\$_) so bash doesn't try
    to expand them as bash vars under set -u (which crashed the script:
    'env: unbound variable'). Assert the raw $env / $_ never appear
    unescaped in the win SSH command strings."""
    model = _model("m", device_order="aibox1-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p2)})
    script = _render_rpc(model, master, [p2])
    # any line that's a double-quoted ssh powershell command must escape
    # $env: and $_. (the .ps1 heredoc body is quoted 'PS1' so $env is fine
    # there — only the inline ssh strings need the escape). Check the
    # win stop/state functions specifically.
    for line in script.splitlines():
        if "powershell -NoProfile -Command" in line and "Get-CimInstance" in line:
            # $_. must be escaped so bash leaves it for PowerShell
            assert "\\$_" in line, line
        if "powershell -NoProfile -Command" in line and "Remove-Item" in line:
            assert "\\$env:USERPROFILE" in line, line
        if "powershell -NoProfile -Command" in line and "Test-Path" in line and "rpc.state" in line:
            assert "\\$env:USERPROFILE" in line, line


def test_render_ps1_takes_port_and_logname_params(stub_cfg):
    """The Windows launcher must accept -Port/-LogName so a script with
    several Windows peers on different ports can invoke it per node."""
    model = _model("m", device_order="aibox1-CUDA0,winbox-CUDA0")
    master = _machine("aibox1", "10.0.0.98", role="master")
    p2 = _machine("winbox", "10.0.0.13", os="windows")
    stub_cfg.update({m["name"]: m for m in (master, p2)})
    script = _render_rpc(model, master, [p2])
    assert "param([int]$Port" in script
    assert "[string]$LogName" in script
    # start_rpc_win passes -Port $port -LogName $log
    assert "-Port $port -LogName $log" in script


# ---------------- _win_launcher: exe path formats correctly ----------------

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