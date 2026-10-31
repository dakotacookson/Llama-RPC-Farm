"""Unit tests for the busy-port roll-forward fallback.

A launch must not fail outright when the requested HTTP port is already
bound (leftover llama-server, an unrelated app like the node fixture on
5802). _roll_port_if_busy probes forward from the wanted port and
returns the first free one; _start_llama_server / swap launch paths use
it and report the ACTUAL bound port (port_rolled) so the UI can show it.

Generated start-*.sh scripts carry the same fallback at runtime
(roll_rpc_port + wait_for_rpc foreign-squatter detection), and every
llama-server / rpc-server launch carries the machine env + sched debug
flags so failures are diagnosable from the logs.
"""
import types

import pytest

import rpc_manager


def _fresh_free_port():
    """Find a port that is (a) actually bindable NOW and (b) outside the
    Windows-excluded ranges — 50052 grandfathered too.

    The naive ephemeral-bind trick (bind :0, read the port, release) is
    not enough: Linux's ephemeral range (32768-60999) overlaps _WIN_EXCLUDED
    (49698-50475), and when the OS hands a test a port inside those ranges
    _roll_port_if_busy legitimately skips the whole range (Windows peers
    can't bind there), breaking 'free port returned unchanged' assertions
    — an order/timing flake that hits a different test each run.
    """
    import socket
    from rpc_manager import _is_excluded_port
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if _is_excluded_port(port):
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))  # re-bind to confirm still free
            except OSError:
                continue
        return port
    pytest.fail("no free non-excluded ephemeral port found")


def _local_machine():
    return {"name": "aibox1", "ip": "127.0.0.1", "os": "linux",
            "username": "zazi", "rpc_port": 50052}


# ---------------- _roll_port_if_busy ----------------

def test_roll_returns_free_port_unchanged():
    """A free requested port is returned as-is, rolled=False."""
    m = _local_machine()
    free = _fresh_free_port()
    port, rolled = rpc_manager._roll_port_if_busy(m, free, True)
    assert port == free
    assert rolled is False


def test_roll_skips_busy_port():
    """A port held by another process is skipped; the next port wins."""
    m = _local_machine()
    import socket
    # held port may sit inside an excluded range; rolling forward from it
    # is fine — the assertion is just that a DIFFERENT port wins
    with socket.socket() as s1:
        s1.bind(("127.0.0.1", 0))
        p1 = s1.getsockname()[1]
        s1.listen(1)
        port, rolled = rpc_manager._roll_port_if_busy(m, p1, True)
        assert port != p1
        assert rolled is True


def test_roll_skips_windows_excluded_ranges():
    """A rolled port never lands in a Windows-excluded bind range."""
    m = {"name": "winbox", "ip": "10.0.0.13", "os": "windows",
         "username": "zazit", "rpc_port": 50052}

    class AlwaysOpen:
        def __call__(self, machine, port, is_local):
            return True  # every port "busy": rolling must exhaust and give up

    orig = rpc_manager._port_open
    rpc_manager._port_open = AlwaysOpen()
    try:
        port, rolled = rpc_manager._roll_port_if_busy(m, 50058, False)
        # exhausted tries: unchanged port, not rolled (caller tries as-is)
        assert port == 50058
        assert rolled is False
    finally:
        rpc_manager._port_open = orig


def test_roll_remote_uses_ssh_check(monkeypatch):
    """Remote machines probe via _port_open (SSH/rdma aware), not a
    local connect."""
    m = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
         "username": "zazi", "rpc_port": 50052}
    calls = []

    def fake_open(machine, port, is_local):
        calls.append((machine["name"], port, is_local))
        return port < 6000  # busy below 6000

    monkeypatch.setattr(rpc_manager, "_port_open", fake_open)
    port, rolled = rpc_manager._roll_port_if_busy(m, 5999, False)
    assert port == 6000
    assert rolled is True
    assert calls[0] == ("aibox2", 5999, False)


# ---------------- _start_llama_server reports the rolled port ----------------

def test_start_llama_server_rolls_busy_port(monkeypatch, tmp_path):
    """_start_llama_server picks a free port when the requested one is
    busy and reports it via port / port_rolled."""
    import socket
    m = _local_machine()
    # held port may sit inside an excluded range; the assertion is just
    # that a DIFFERENT, rolled port wins
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        s.listen(1)
        model = {"model_id": "test-model", "base_path": "/dev/null",
                 "flags": {}}

        captured = {}

        def fake_build(model, servers):
            return 'llama-server -m x --host 0.0.0.0 --port "$PORT"'

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return types.SimpleNamespace(pid=4242)

        monkeypatch.setattr(rpc_manager, "build_llama_command", fake_build)
        monkeypatch.setattr(rpc_manager, "local_node",
                            types.SimpleNamespace(is_local=lambda mm: True))
        monkeypatch.setattr(rpc_manager, "GENERATED_DIR", str(tmp_path))
        import subprocess as _sp
        monkeypatch.setattr(_sp, "Popen", fake_popen)

        launched = {}

        def fake_port_open(machine, port, is_local):
            # before launch: only the real socket is busy; after launch
            # (port recorded by fake_popen via captured cmd) the chosen
            # port reports open so the wait loop exits instantly
            chosen = launched.get("port")
            if chosen is not None and port == chosen:
                return True
            if port == busy:
                return True
            with socket.socket() as probe:
                try:
                    probe.connect(("127.0.0.1", port))
                    return True
                except OSError:
                    return False

        monkeypatch.setattr(rpc_manager, "_port_open", fake_port_open)

        orig_popen = fake_popen

        def popen_spy(cmd, **kw):
            res = orig_popen(cmd, **kw)
            for part in cmd if isinstance(cmd, list) else [cmd]:
                if f"--port " in str(part):
                    launched["port"] = int(str(part).split("--port ")[1].split()[0])
            return res

        monkeypatch.setattr(_sp, "Popen", popen_spy)

        res = rpc_manager._start_llama_server(
            model, m, [], busy)
        assert res["port"] != busy
        assert res["port_rolled"] is True
        assert res["running"] is True
        cmd_str = " ".join(captured["cmd"]) if isinstance(captured["cmd"], list) else captured["cmd"]
        assert f'--port {res["port"]}' in cmd_str
        assert f'--port {busy}' not in cmd_str


def test_start_llama_server_keeps_free_port(monkeypatch, tmp_path):
    """A free requested port is used verbatim; port_rolled falsy."""
    m = _local_machine()
    free = _fresh_free_port()
    model = {"model_id": "test-model", "base_path": "/dev/null",
             "flags": {}}

    captured = {}
    launched = {}

    def fake_build(model, servers):
        return 'llama-server -m x --host 0.0.0.0 --port "$PORT"'

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return types.SimpleNamespace(pid=4242)

    monkeypatch.setattr(rpc_manager, "build_llama_command", fake_build)
    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda mm: True))
    monkeypatch.setattr(rpc_manager, "GENERATED_DIR", str(tmp_path))
    import subprocess as _sp

    def popen_spy(cmd, **kw):
        res = fake_popen(cmd, **kw)
        for part in cmd if isinstance(cmd, list) else [cmd]:
            if "--port " in str(part):
                launched["port"] = int(str(part).split("--port ")[1].split()[0])
        return res

    monkeypatch.setattr(_sp, "Popen", popen_spy)
    monkeypatch.setattr(rpc_manager, "_port_open",
                        lambda machine, port, is_local:
                        port == launched.get("port"))

    res = rpc_manager._start_llama_server(model, m, [], free)
    assert res["port"] == free
    assert not res.get("port_rolled")
    assert res["running"] is True
    cmd_str = " ".join(captured["cmd"]) if isinstance(captured["cmd"], list) else captured["cmd"]
    assert f'--port {free}' in cmd_str


# ---------------- launch_model surfaces the rolled port ----------------

def test_launch_model_report_carries_actual_port(monkeypatch, stub_cfg):
    """The top-level report['port'] mirrors the serve step's ACTUAL
    port (possibly rolled), so the UI shows the live one."""
    machine = {"name": "aibox1", "ip": "127.0.0.1", "os": "linux",
               "username": "zazi", "rpc_port": 50052, "role": "master"}
    stub_cfg["aibox1"] = machine
    model = {"model_id": "m1", "base_path": "/dev/null", "flags": {},
             "servers": ["aibox1"]}

    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: True))
    monkeypatch.setattr(rpc_manager, "_split_servers",
                        lambda servers, master_name=None: (machine, []))
    from steps import run_steps as _orig  # noqa: F401  (report shape sanity)
    monkeypatch.setattr(rpc_manager, "stop_rpc",
                        lambda name, port=None: {"killed": True})
    monkeypatch.setattr(rpc_manager, "_start_llama_server",
                        lambda model, master, peers, port:
                        {"running": True, "pid": 1, "port": port + 1,
                         "port_rolled": True})

    def fake_get_model(mid):
        return model

    monkeypatch.setattr(rpc_manager.cfg, "get_model",
                        fake_get_model, raising=False)

    report = rpc_manager.launch_model("m1", ["aibox1"], 8080)
    assert report["port"] == 8081
    assert report.get("port_rolled") is True


# ---------------- debug env on every launch path ----------------

def test_rpc_env_exports_carries_sched_debug():
    """rpc-server launches must always run with GGML_SCHED_DEBUG /
    RPC_DEBUG so the peer logs show per-tensor placement."""
    env = rpc_manager._rpc_env_exports({"name": "aibox3"})
    assert "GGML_SCHED_DEBUG=3" in env
    assert "GGML_RPC_DEBUG=3" in env
    assert "GGML_RPC_NO_RDMA=1" in env


def test_rpc_env_exports_rdma_keeps_debug():
    """RDMA peers keep the debug vars alongside the RDMA pinning."""
    env = rpc_manager._rpc_env_exports(
        {"name": "r", "rdma": True, "rdma_dev": "mlx5_0", "rdma_gid": "3"})
    assert "GGML_SCHED_DEBUG=3" in env
    assert "GGML_RDMA_DEV=mlx5_0" in env
    assert "GGML_RDMA_GID=3" in env
    assert "GGML_RPC_NO_RDMA" not in env


def test_device_env_exports_carries_sched_debug():
    """llama-server launches (scripts AND live path) need the sched
    debug + device-order exports."""
    env = rpc_manager._device_env_exports({"name": "aibox1"})
    assert "GGML_SCHED_DEBUG=3" in env
    assert "GGML_RPC_DEBUG=3" in env
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in env


def test_start_llama_server_prepends_machine_env(monkeypatch, tmp_path):
    """The live launch path must run the server with the machine env
    (previously only the rendered scripts had it)."""
    m = _local_machine()
    model = {"model_id": "test-model", "base_path": "/dev/null",
             "flags": {}}
    captured = {}
    launched = {}

    def fake_build(model, servers):
        return 'llama-server -m x --host 0.0.0.0 --port "$PORT"'

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return types.SimpleNamespace(pid=4242)

    monkeypatch.setattr(rpc_manager, "build_llama_command", fake_build)
    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda mm: True))
    monkeypatch.setattr(rpc_manager, "GENERATED_DIR", str(tmp_path))
    import subprocess as _sp

    def popen_spy(cmd, **kw):
        res = fake_popen(cmd, **kw)
        for part in cmd if isinstance(cmd, list) else [cmd]:
            if "--port " in str(part):
                launched["port"] = int(str(part).split("--port ")[1].split()[0])
        return res

    monkeypatch.setattr(_sp, "Popen", popen_spy)
    monkeypatch.setattr(rpc_manager, "_port_open",
                        lambda machine, port, is_local:
                        port == launched.get("port"))

    res = rpc_manager._start_llama_server(model, m, [], 8099)
    assert res["running"] is True
    full = " ".join(captured["cmd"]) if isinstance(captured["cmd"], list) \
        else captured["cmd"]
    assert "GGML_SCHED_DEBUG=3" in full
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in full
    assert "GGML_SCHED_DEBUG=3" in res["command"]


# ---------------- rendered scripts: roll + preflight ----------------

def _render_rpc_script(stub_cfg):
    from rpc_manager import _render_rpc
    model = {"model_id": "roll-test", "flags": {},
             "base_path": "store://llm-models/x.gguf", "device_order": "",
             "device_order_policy": "as-selected", "expose": {}}
    master = {"name": "aibox1", "ip": "10.0.0.104", "username": "u",
              "password": "p", "os": "linux", "rpc_port": 50052,
              "gpu": "CUDA0", "role": "master",
              "detected": {"gpu_names": [], "vram_total_gb": 0,
                           "ram_total_gb": 16, "cores": 8}}
    p1 = dict(master, name="aibox3", ip="10.0.0.98", role="peer")
    p2 = dict(master, name="winbox", ip="10.0.0.124", os="windows",
              role="peer", username="w")
    stub_cfg.update({m["name"]: m for m in (master, p1, p2)})
    return _render_rpc(model, master, [p1, p2])


def test_render_rpc_has_roll_helpers(stub_cfg):
    """Generated scripts must carry the runtime port roll-forward:
    helpers + foreign-squatter detection in the wait loop."""
    script = _render_rpc_script(stub_cfg)
    for token in ("roll_rpc_port()", "tcp_probe()", "win_port_excluded()",
                  "rpc_log_for_port()", "ensure_node()",
                  "held by another process", "rolling to $port"):
        assert token in script, token
    import subprocess as _sp
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".sh",
                                     delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = _sp.run(["bash", "-n", path], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        os.unlink(path)


def test_render_rpc_roll_updates_live_maps(stub_cfg):
    """A roll must update RPC_PORTS + RPC_LOGS so the --rpc peer list
    (built after the waits) serves the rolled ports."""
    script = _render_rpc_script(stub_cfg)
    assert 'RPC_PORTS["$node"]="$port"' in script
    assert 'RPC_LOGS["$node"]="$(rpc_log_for_port "$port")"' in script
    # and the peer list is composed from the (possibly rolled) map
    assert "$node:${RPC_PORTS[$node]}" in script


def test_render_single_node_has_port_preflight(stub_cfg):
    """Single-node scripts must fail fast with a clear message when a
    foreign process holds llama-swap's assigned port."""
    model = {"model_id": "single-test", "flags": {},
             "base_path": "store://llm-models/x.gguf"}
    master = {"name": "aibox1", "ip": "10.0.0.104", "username": "u",
              "password": "p", "os": "linux", "rpc_port": 50052}
    stub_cfg.update({master["name"]: master})
    script = rpc_manager._render_single_node(model, master)
    assert "refusing to start" in script
    assert "GGML_SCHED_DEBUG=3" in script
    import subprocess as _sp
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".sh",
                                     delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = _sp.run(["bash", "-n", path], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        os.unlink(path)

# ---------------- script audit fixes: anchored ports, per-peer bins ----------------

def test_render_kill_patterns_anchored(stub_cfg):
    """Port-targeted kills must anchor the port (trailing space): a
    pattern for 5109 must never match a co-served model on 51096."""
    script = _render_rpc_script(stub_cfg)
    assert '-p $port \\"' in script
    # PowerShell match pads the command line with spaces so a port at the
    # END of the command line (no trailing arg) still matches
    assert "(' ' + \\$_.CommandLine + ' ') -like '* -p $port *'" in script
    # no unanchored port match remains in kill/pgrep lines
    for line in script.splitlines():
        if ("pkill" in line or "pgrep" in line) and "-p $port" in line:
            assert "-p $port " in line, line
        if "-like" in line and "-p $port" in line:
            assert "-p $port *" in line, line


def test_render_per_peer_bin_map(stub_cfg):
    """Each linux peer launches rpc-server from its OWN bin dir, with
    the master's layout only as fallback."""
    script = _render_rpc_script(stub_cfg)
    assert "declare -A RPC_BINS" in script
    assert 'RPC_BINS["10.0.0.98"]=' in script
    assert "${RPC_BINS[$node]:-" in script


def test_render_ps1_no_baked_device(stub_cfg):
    """The parametrized .ps1 must not bake a --device flag: per-node
    backends arrive at runtime via -ExtraArgs (a baked one would
    duplicate the flag on every Windows peer)."""
    from rpc_manager import _win_launcher
    m = {"name": "winbox", "ip": "10.0.0.13", "os": "windows",
         "username": "w", "llama_bin": "%USERPROFILE%\\llama.cpp",
         "expose": {"mode": "vram", "gpus": ["CUDA0"]}}
    ps1 = _win_launcher(m, 50052, expose_args="", parametrized=True)
    assert "--device" not in ps1.split("-ExtraArgs")[0]


def test_render_password_with_quote_stays_valid(stub_cfg):
    """A single-quote in the machine password must not break the
    generated script — passwords are vault-only now (never baked in),
    so even hostile values can't affect script syntax. The script stays
    valid bash and requires $SSH_PASS at runtime."""
    from rpc_manager import _render_rpc
    model = {"model_id": "q", "flags": {},
             "base_path": "store://llm-models/x.gguf", "device_order": "",
             "device_order_policy": "as-selected", "expose": {}}
    master = {"name": "aibox1", "ip": "10.0.0.104", "username": "u",
              "password": "p@ss'word", "os": "linux", "rpc_port": 50052,
              "gpu": "CUDA0", "role": "master",
              "detected": {"gpu_names": [], "vram_total_gb": 0,
                           "ram_total_gb": 16, "cores": 8}}
    p1 = dict(master, name="aibox3", ip="10.0.0.98", role="peer")
    stub_cfg.update({m["name"]: m for m in (master, p1)})
    script = _render_rpc(model, master, [p1])
    assert "p@ss'word" not in script
    assert "SSH_PASS:?" in script
    import subprocess as _sp
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".sh",
                                     delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = _sp.run(["bash", "-n", path], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        os.unlink(path)


def test_stop_rpc_port_pattern_anchored(monkeypatch, stub_cfg):
    """Live stop_rpc patterns must anchor the port too."""
    import subprocess as _sp
    captured = []

    def fake_run(cmd, is_win=False, timeout=20):
        captured.append(cmd if isinstance(cmd, str) else " ".join(cmd))
        return 1, "", ""

    monkeypatch.setattr(rpc_manager, "_run_local", fake_run)
    monkeypatch.setattr(rpc_manager, "local_node",
                        __import__("types").SimpleNamespace(
                            is_local=lambda m: True))
    m = {"name": "aibox1", "ip": "127.0.0.1", "os": "linux",
         "username": "zazi", "rpc_port": 50052}
    stub_cfg["aibox1"] = m
    rpc_manager.stop_rpc("aibox1", port=51096)
    assert any("-p 51096 " in c for c in captured), captured


def test_stop_llama_server_port_pattern_anchored(monkeypatch, stub_cfg):
    """Serve-group-safe stop must anchor the port (trailing space) and
    match ONLY llama-server — never rpc-servers, never 8080-in-80808."""
    captured = []

    def fake_run(cmd, is_win=False, timeout=20):
        captured.append(cmd if isinstance(cmd, str) else " ".join(cmd))
        return 1, "", ""

    monkeypatch.setattr(rpc_manager, "_run_local", fake_run)
    monkeypatch.setattr(rpc_manager, "local_node",
                        __import__("types").SimpleNamespace(
                            is_local=lambda m: True))
    m = {"name": "aibox3", "ip": "10.0.0.98", "os": "linux",
         "username": "zazi", "rpc_port": 50052}
    stub_cfg["aibox3"] = m
    rpc_manager.stop_llama_server_port("aibox3", port=8081)
    # end-or-space anchored: matches --port at cmdline END too (remote-local
    # launcher renders --port last; trailing-space-only never matched there,
    # observed live: cmdStop returned rc=1 on a healthy server, VRAM held)
    assert any("--port 8081( |$)" in c for c in captured), captured
    assert any("llama-[s]erver" in c for c in captured), captured
    assert not any("ggml" in c or "rpc-server" in c for c in captured), \
        captured


def test_remote_local_launch_stops_only_its_port(monkeypatch, stub_cfg):
    """_launch_model_remote_local pre-launch hygiene must be the targeted
    per-port stop — a blanket clear would kill co-served models on the
    same box (serve-group violation)."""
    import types as _types
    box = {"name": "aibox3", "ip": "10.0.0.98", "os": "linux",
           "username": "zazi", "rpc_port": 50052, "role": "peer",
           "gpu": "CUDA0", "gpu_vendor": "nvidia",
           "expose": {"mode": "vram", "gpus": ["CUDA0"]}}
    stub_cfg["aibox3"] = box
    monkeypatch.setattr(rpc_manager, "local_node",
                        _types.SimpleNamespace(is_local=lambda m: False))
    calls = []
    monkeypatch.setattr(rpc_manager, "stop_llama_server_port",
                        lambda name, port: calls.append((name, port)) or {
                            "machine": name, "killed": True})
    monkeypatch.setattr(rpc_manager, "_start_llama_server",
                        lambda *a, **k: {"running": True, "port": 8081})
    monkeypatch.setattr(rpc_manager, "stop_rpc",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("blanket stop must not run")))
    model = {"model_id": "m1", "servers": ["aibox3"],
             "base_path": "/m.gguf", "flags": {},
             "device_order": "aibox3-CUDA0"}
    out = rpc_manager._launch_model_remote_local(
        model, box, [], 8081)
    assert calls == [("aibox3", 8081)], calls
    assert any(s.get("step") == "stop:aibox3" for s in out["steps"])
