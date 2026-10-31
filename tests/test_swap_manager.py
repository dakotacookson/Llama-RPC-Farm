"""Tests for swap_manager.generate() — the B3 vLLM branch (ARD
t_76f3d1c0 §3.5) and the llama.cpp parity it must preserve.

vllm mode emits thin per-model start scripts hosting the vllm serve /
docker run command (cmdStop + /health probe on the entry, empty
rpc_map, no RPC fan-out) instead of RPC wrapper scripts. Everything
downstream of generate() is engine-agnostic. All IO (machines, model
configs, engine settings) is faked — no network, no real config files,
matching the approach of test_build_vllm_command.py.
"""
import os
import types

import yaml

import conftest
import config_loader as cfg
import engine_registry
import rpc_manager
import swap_manager

VENV_BLOCK = {"delivery": "venv", "venv_bin": "~/vllm-env/bin/vllm",
              "docker_image": "vllm/vllm-openai:latest",
              "default_gpu_memory_utilization": 0.9}
DOCKER_BLOCK = dict(VENV_BLOCK, delivery="docker")


def _machine(name, ip, **kw):
    m = {"name": name, "ip": ip, "username": "u", "password": "p",
         "os": "linux", "rpc_port": 50052, "gpu": "", "role": "peer",
         "gpu_vendor": "nvidia",
         "detected": {"gpu_names": ["NVIDIA RTX 3090"],
                      "vram_total_gb": 24, "ram_total_gb": 64}}
    m.update(kw)
    return m


def _model(mid="qwen3-8b", servers=None, **kw):
    m = {"model_id": mid, "name": mid, "flags": {},
         # .gguf suffix = llama.cpp-servable; the engine-vocabulary gate
         # in launch_plan skips HF-format models in llama.cpp mode.
         "base_path": "/mnt/store/" + mid + ".gguf",
         "servers": servers if servers is not None else ["aibox1"]}
    # DEC-08: launch reads only the native vllm: section; give every model
    # a minimal non-empty one (cpu_offload_gb 0 emits no flag) unless a
    # test overrides it via kw.
    m.setdefault("vllm", {"cpu_offload_gb": 0})
    m.update(kw)
    return m


def _wire(monkeypatch, tmp_path, models, engine_settings,
          extra_machines=()):
    by_name = {"aibox1": _machine("aibox1", "127.0.0.1", role="master")}
    for m in extra_machines:
        by_name[m["name"]] = m
    monkeypatch.setattr(swap_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(by_name),
        swap_groups=types.SimpleNamespace(all=lambda: []),
        list_models=lambda: list(models),
        # DEC-03: generate() resolves per-machine swap-config overrides
        # through cfg.machine_swap_config_override (real validator —
        # machines here are plain dicts, override keys default to "")
        machine_swap_config_override=cfg.machine_swap_config_override))
    # _split_servers resolves machine names through rpc_manager.cfg —
    # fake it too, or peer lookups would hit the real machine list.
    monkeypatch.setattr(rpc_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(by_name)))
    monkeypatch.setattr(engine_registry, "load_settings",
                        lambda *a, **k: dict(engine_settings))
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE",
                        str(tmp_path / "swap-settings.yaml"))


def _generate(monkeypatch, tmp_path, models, engine_settings,
              extra_machines=()):
    _wire(monkeypatch, tmp_path, models, engine_settings, extra_machines)
    return swap_manager.generate(str(tmp_path / "config.yaml"), "aibox1")


# ---------------- vllm mode: thin script entries (B3 checkpoint) ----------

def test_vllm_mode_snapshot(monkeypatch, tmp_path):
    model = _model()
    model["device_order"] = "aibox1-CUDA0"
    model["vllm"] = {"max_model_len": 32768}
    plan = _generate(monkeypatch, tmp_path, [model],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    assert plan["models"] == ["qwen3-8b"]
    assert plan["warnings"] == []
    assert plan["skipped"] == []
    assert plan["script_mids"] == ["qwen3-8b"]
    assert plan["scripts"] == ["start-qwen3-8b.sh"]
    script = plan["_scripts"]["start-qwen3-8b.sh"]
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1] == 'PORT="${1:-8080}"'
    assert any(line.startswith("exec ") for line in lines)
    # thin: no RPC fan-out, no sshpass block, no rpc-server machinery
    for banned in ("sshpass", "rpc-server", "ggml-rpc", "--rpc"):
        assert banned not in script
    assert "vllm serve" in script
    assert "CUDA_VISIBLE_DEVICES=0" in script
    assert '--served-model-name "qwen3-8b"' in script
    entry = plan["entries"][0]
    assert entry["mode"] == "script"
    assert entry["rpc_map"] == {}
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert emitted["cmd"] == (
        f"/bin/bash {plan['scripts_dir']}/start-qwen3-8b.sh ${{PORT}}")
    assert emitted["cmdStop"] == 'pkill -f "vllm serve.*qwen3-8b"'
    assert emitted["checkEndpoint"] == "/health"
    # env omitted when empty (vLLM carries env inside cmd by design); a
    # dict here would crash-loop the daemon (see kv-list test below)
    assert "env" not in emitted
    assert emitted["ttl"] == 0


def test_vllm_mode_docker_delivery_snapshot(monkeypatch, tmp_path):
    plan = _generate(monkeypatch, tmp_path, [_model()],
                     {"engine": "vllm", "vllm": DOCKER_BLOCK})
    script = plan["_scripts"]["start-qwen3-8b.sh"]
    assert "docker run --rm --gpus all" in script
    assert "-p ${PORT}:8000" in script
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert emitted["cmdStop"] == "docker stop farm-qwen3-8b"
    assert emitted["checkEndpoint"] == "/health"
    assert "env" not in emitted


def test_vllm_mode_multi_host_builds_cluster_script(monkeypatch, tmp_path):
    """Multi-host vLLM: peers pass through to swap_entry_parts — the
    entry carries the Ray cluster script and the swap-secrets.env wrap
    (worker nodes need $SSH_PASS, same sidecar the llama.cpp branch uses)."""
    peer = _machine("aibox2", "10.0.0.2")
    plan = _generate(monkeypatch, tmp_path,
                     [_model(servers=["aibox1", "aibox2"],
                             device_order="aibox1-CUDA0,aibox2-CUDA0")],
                     {"engine": "vllm", "vllm": VENV_BLOCK},
                     extra_machines=[peer])
    assert not any("peers ignored" in w for w in plan["warnings"])
    script = plan["_scripts"]["start-qwen3-8b.sh"]
    assert "rpc-server" not in script          # no llama.cpp fan-out
    assert "ray start --address=" in script    # cluster machinery instead
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert emitted["cmd"].startswith(
        f"/bin/bash -c 'source \"{plan['scripts_dir']}/swap-secrets.env\"")


def test_vllm_mode_single_host_keeps_thin_cmd(monkeypatch, tmp_path):
    """No peers: the vLLM entry stays the thin script cmd (no secrets
    wrap — single-node vLLM needs no SSH)."""
    plan = _generate(monkeypatch, tmp_path, [_model()],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert emitted["cmd"] == (
        f"/bin/bash {plan['scripts_dir']}/start-qwen3-8b.sh ${{PORT}}")


def test_vllm_mode_bad_model_skipped_not_fatal(monkeypatch, tmp_path):
    plan = _generate(monkeypatch, tmp_path,
                     [_model("bad;touch"), _model("good-model")],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    assert plan["models"] == ["good-model"]
    assert "bad;touch" in plan["skipped"]
    assert any("bad;touch" in w for w in plan["warnings"])
    assert plan["scripts"] == ["start-good-model.sh"]


def test_vllm_mode_env_emitted_as_kv_list(monkeypatch, tmp_path):
    """llama-swap's per-model env is a LIST of "K=V" strings: a dict makes
    the whole config fail to load (`cannot unmarshal !!map into
    []string`, daemon crash-loops). generate() must convert non-empty
    parts env to K=V strings and omit the key when empty."""
    import vllm_manager
    real_parts = vllm_manager.swap_entry_parts

    def parts_with_env(model_cfg, master, vllm_cfg=None, *, peers=None):
        d = real_parts(model_cfg, master, vllm_cfg, peers=peers)
        d["env"] = {"CUDA_VISIBLE_DEVICES": "0,1", "FOO_BAR": "baz"}
        return d

    import swap_manager  # patch the name generate() actually resolves
    monkeypatch.setattr(vllm_manager, "swap_entry_parts", parts_with_env)
    plan = _generate(monkeypatch, tmp_path, [_model()],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert emitted["env"] == ["CUDA_VISIBLE_DEVICES=0,1", "FOO_BAR=baz"]


def test_vllm_script_hoists_env_prefix_and_keeps_port(monkeypatch, tmp_path):
    """_vllm_script: `exec VAR=x cmd` is invalid bash (exit 127); env
    prefixes must be hoisted to export lines while $PORT resolution and
    the exec line survive."""
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "vllm", "vllm": VENV_BLOCK})
    script = swap_manager._vllm_script(
        'CUDA_VISIBLE_DEVICES=0,1 vllm serve "$PORT" --model /m')
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1] == 'PORT="${1:-8080}"'
    assert "export CUDA_VISIBLE_DEVICES=0,1" in script
    assert 'exec vllm serve "$PORT" --model /m' in script
    assert "exec CUDA_VISIBLE_DEVICES" not in script


def test_vllm_script_plain_cmd_untouched(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "vllm", "vllm": VENV_BLOCK})
    script = swap_manager._vllm_script('vllm serve "$PORT" --model /m')
    assert script.endswith('exec vllm serve "$PORT" --model /m\n')


def test_vllm_script_cluster_cmd_passes_through(monkeypatch, tmp_path):
    """_vllm_script: a multi-host cluster cmd is already a full bash
    program — it must NOT gain a wrapping `exec` header (`exec #!/bin/bash`
    is the load failure). Only the nvjitlink LD_LIBRARY_PATH line is
    injected so the final `vllm serve` still imports torch."""
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "vllm", "vllm": VENV_BLOCK})
    cluster = ('#!/bin/bash\nPORT="${1:-8080}"\n'
               'MASTER_IP="10.0.0.1"\nstart_workers\n'
               'exec ~/vllm-env/bin/vllm serve "/m" \\\n  --port "$PORT"\n')
    script = swap_manager._vllm_script(cluster)
    assert script.startswith("#!/bin/bash\n")
    assert "exec #!" not in script
    assert script.count("#!/bin/bash") == 1
    assert 'PORT="${1:-8080}"' in script
    assert 'exec ~/vllm-env/bin/vllm serve "/m"' in script
    import subprocess
    r = subprocess.run(["bash", "-n"], input=script, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


# ---------------- llama.cpp parity: branch adds nothing --------------------

def test_llama_cpp_mode_keeps_rpc_wrapper(monkeypatch, tmp_path):
    peer = _machine("aibox2", "10.0.0.2")
    plan = _generate(monkeypatch, tmp_path,
                     [_model(servers=["aibox1", "aibox2"],
                             launch_mode="rpc")],
                     {"engine": "llama_cpp"}, extra_machines=[peer])
    entry = plan["entries"][0]
    assert entry["mode"] == "script"
    assert entry["rpc_map"]  # RPC ids exist in llama.cpp mode
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert "cmdStop" not in emitted
    script = plan["_scripts"]["start-qwen3-8b.sh"]
    assert "--rpc" in script
    assert "vllm" not in script


def test_single_peer_auto_launch_mode_proxies_remote(monkeypatch, tmp_path):
    """launch_mode=auto (default): one peer box + no host chips -> run the
    whole deployment on that box LOCALLY (no RPC hops); llama-swap fronts
    it via proxy so the farm keeps one API. Entry: no cmd, proxy at the
    remote box, checkEndpoint none, cmdStop SSHes a port-targeted pkill."""
    peer = _machine("aibox2", "10.0.0.2")
    plan = _generate(monkeypatch, tmp_path,
                     [_model(servers=["aibox1", "aibox2"])],
                     {"engine": "llama_cpp"}, extra_machines=[peer])
    entry = plan["entries"][0]
    assert entry["mode"] == "remote-proxy"
    assert entry["rpc_map"] == {}
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    # cmd IS the remote launcher script: llama-swap runs it on load, it
    # SSHes to the box, starts llama-server there (reusing a live port),
    # waits for the port, exits — then llama-swap proxies to the box.
    assert "start-qwen3-8b.sh" in emitted["cmd"]
    assert "swap-secrets.env" in emitted["cmd"]
    # port: deterministic auto-assign (hash base 52000) since the model
    # pins no launch_mode_local_port
    assert emitted["proxy"].startswith("http://10.0.0.2:")
    assert int(emitted["proxy"].rsplit(":", 1)[1]) >= 52000
    # /health (not none): llama-server 503s "Loading model" during load —
    # llama-swap polls until 200 so the first client request WAITS
    # instead of surfacing bad-gateway
    assert emitted["checkEndpoint"] == "/health"
    assert "sshpass" in emitted["cmdStop"]
    stop_port = emitted["proxy"].rsplit(":", 1)[1]
    # end-or-space anchor: cmdline may END at --port (launcher renders it
    # last); trailing-space-only never matched there (observed live)
    assert f"--port {stop_port}( |$)" in emitted["cmdStop"]
    # the launcher script exists and carries the remote start logic
    script = plan["_scripts"]["start-qwen3-8b.sh"]
    assert "llama-server" in script
    assert "10.0.0.2" in script
    assert "--rpc" not in script  # local mode: no RPC tokens
    assert "already serving" in script  # idempotent reuse guard


def test_missing_engine_key_defaults_to_llama_cpp(monkeypatch, tmp_path):
    plan = _generate(monkeypatch, tmp_path, [_model()], {})
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert "cmdStop" not in emitted
    assert "vllm" not in plan["_scripts"]["start-qwen3-8b.sh"]


# ---------------- toggle never deletes/rewrites (ARD §8 risk 4) ------------

def test_engine_toggle_never_touches_old_scripts_or_rpc_state(
        monkeypatch, tmp_path):
    """Toggling to vllm must not delete or rewrite existing generated
    start-*.sh scripts or RPC state files: push_scripts writes only the
    files the current plan contains."""
    _wire(monkeypatch, tmp_path, [_model("new-model")],
          {"engine": "vllm", "vllm": VENV_BLOCK})
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": str(tmp_path / "config.yaml")}, f)
    old_script = tmp_path / "start-old-model.sh"
    old_script.write_text("#!/bin/bash\necho SENTINEL\n", encoding="utf-8")
    rpc_state = tmp_path / "rpc.state.50052"
    rpc_state.write_text("state\n", encoding="utf-8")

    out = swap_manager.push_scripts()

    written_names = {p.rsplit("/", 1)[-1] for p in out["written"]}
    # DEC-03: engine-qualified FILE — vllm mode pushes config.vllm.yaml
    # (sibling of llama.cpp's config.yaml, same dir) and never touches
    # the other engine's file.
    assert written_names == {"config.vllm.yaml", "start-new-model.sh"}
    assert old_script.read_text(encoding="utf-8") == \
        "#!/bin/bash\necho SENTINEL\n"
    assert rpc_state.read_text(encoding="utf-8") == "state\n"


# ---------------- DEC-13: swap push hardening (T2-G4/G8) ------------------

def test_scripts_dir_is_per_engine_subdir(monkeypatch, tmp_path):
    """T2-G8: generated scripts + sidecar land in swap/<engine>/ under the
    config dir — the two engines' script files never collide, and cmd
    paths inside the emitted yaml point at the subdir."""
    plan = _generate(monkeypatch, tmp_path, [_model()],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    assert plan["scripts_dir"] == f"{plan['swap_dir']}/swap/vllm"
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert plan["scripts_dir"] in emitted["cmd"]
    # llama_cpp mode gets its own subdir of the same dir
    plan2 = _generate(monkeypatch, tmp_path, [_model()],
                      {"engine": "llama_cpp"})
    assert plan2["scripts_dir"] == f"{plan2['swap_dir']}/swap/llama_cpp"


def test_push_scripts_backs_up_before_overwrite(monkeypatch, tmp_path):
    """T2-G4: push_scripts backs up an existing config it is about to
    overwrite (config.yaml.bak-<stamp>) and reports it in `backups`."""
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "llama_cpp"})
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": str(tmp_path / "config.yaml")}, f)
    # pre-existing operator config on the host (host is local: 127.0.0.1)
    old = tmp_path / "config.yaml"
    old.write_text("models: {old: {cmd: old}}\n", encoding="utf-8")

    out = swap_manager.push_scripts()

    assert out.get("validated") is True
    assert len(out["backups"]) == 1
    bak = out["backups"][0]
    assert bak.startswith(str(old) + ".bak-")
    with open(bak, encoding="utf-8") as f:
        assert f.read() == "models: {old: {cmd: old}}\n"
    # the config itself was replaced by the fresh plan
    with open(old, encoding="utf-8") as f:
        assert "qwen3-8b" in f.read()


def test_push_scripts_aborts_on_unparseable_plan(monkeypatch, tmp_path):
    """T2-G4: a plan whose yaml does not parse aborts the push with an
    error BEFORE any file is written — the daemon never gets a dead
    config."""
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "llama_cpp"})
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": str(tmp_path / "config.yaml")}, f)
    old = tmp_path / "config.yaml"
    old.write_text("models: {old: {cmd: old}}\n", encoding="utf-8")

    def broken_plan(*a, **k):
        return {"host": "aibox1", "swap_dir": str(tmp_path),
                "abs_config": str(old), "yaml": "\t{ not yaml ][",
                "_scripts": {}, "warnings": []}

    monkeypatch.setattr(swap_manager, "launch_plan", broken_plan)
    monkeypatch.setattr(swap_manager, "generate", broken_plan)

    out = swap_manager.push_scripts()

    assert "error" in out and "nothing written" in out["error"]
    assert out.get("validated") is False
    # original file untouched
    with open(old, encoding="utf-8") as f:
        assert f.read() == "models: {old: {cmd: old}}\n"


def test_model_ttl_wired_into_swap_entry(monkeypatch, tmp_path):
    """T2-G4: deploy_model['ttl'] reaches the emitted swap entry (was
    hard-coded 0); absent stays 0, negative clamps to 0."""
    plan = _generate(monkeypatch, tmp_path,
                     [_model("qwen3-8b", ttl=600),
                      _model("plain-model"),
                      _model("neg-model", ttl=-5)],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    emitted = yaml.safe_load(plan["yaml"])["models"]
    assert emitted["qwen3-8b"]["ttl"] == 600
    assert emitted["plain-model"]["ttl"] == 0
    assert emitted["neg-model"]["ttl"] == 0
    by_id = {e["model_id"]: e for e in plan["entries"]}
    assert by_id["qwen3-8b"]["ttl"] == 600
    assert by_id["neg-model"]["ttl"] == 0


# ================== DEC-03 (AUDIT-T6 §1.6): per-engine swap files =========

def test_vllm_generate_writes_config_vllm_yaml(monkeypatch, tmp_path):
    """vllm mode derives the sibling FILE config.vllm.yaml from llama.cpp's
    config_path (same dir, engine-qualified basename) — the two engines'
    routing tables never collide in one file."""
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "vllm", "vllm": VENV_BLOCK})
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "host_name": "aibox1"}, f)
    plan = swap_manager.generate(None, "aibox1")
    assert plan["abs_config"] == os.path.expanduser(
        "~/.config/llama-swap/config.vllm.yaml")
    assert plan["swap_dir"] == os.path.expanduser("~/.config/llama-swap")


def test_llama_cpp_generate_keeps_config_yaml(monkeypatch, tmp_path):
    """llama.cpp mode keeps the historical config.yaml verbatim — no
    qualification, existing installs unaffected (backward compat)."""
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "llama_cpp"})
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "host_name": "aibox1"}, f)
    plan = swap_manager.generate(None, "aibox1")
    assert plan["abs_config"] == os.path.expanduser(
        "~/.config/llama-swap/config.yaml")


def test_vllm_explicit_config_path_vllm_wins_verbatim(monkeypatch, tmp_path):
    """swap.yaml config_path_vllm (set explicitly) is taken VERBATIM — the
    operator pinned a file, the daemon must use exactly that file."""
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "vllm", "vllm": VENV_BLOCK})
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "config_path_vllm": "~/custom/vswap.yaml",
                        "host_name": "aibox1"}, f)
    plan = swap_manager.generate(None, "aibox1")
    assert plan["abs_config"] == os.path.expanduser(
        "~/custom/vswap.yaml")


def test_machine_swap_config_override_beats_global(monkeypatch, tmp_path):
    """machines.yaml swap_config_path_vllm on the host machine wins over
    the swap.yaml global (same shape as vllm_delivery) and stays verbatim."""
    host = _machine("aibox1", "127.0.0.1", role="master",
                    swap_config_path_vllm="~/mach/override.yaml")
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "vllm", "vllm": VENV_BLOCK}, extra_machines=[host])
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "host_name": "aibox1"}, f)
    plan = swap_manager.generate(None, "aibox1")
    assert plan["abs_config"] == os.path.expanduser(
        "~/mach/override.yaml")


def test_machine_swap_config_override_llama_cpp(monkeypatch, tmp_path):
    """llama.cpp's per-machine override key is swap_config_path — pinning
    it moves the daemon file; empty inherits the global."""
    host = _machine("aibox1", "127.0.0.1", role="master",
                    swap_config_path="~/mach/lcpp.yaml")
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "llama_cpp"}, extra_machines=[host])
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "host_name": "aibox1"}, f)
    plan = swap_manager.generate(None, "aibox1")
    assert plan["abs_config"] == os.path.expanduser(
        "~/mach/lcpp.yaml")


def test_machine_swap_config_override_bad_type_raises(monkeypatch, tmp_path):
    """A non-string machine override raises on generate (never a silent
    inherit pointing the daemon at the wrong file)."""
    host = _machine("aibox1", "127.0.0.1", role="master",
                    swap_config_path_vllm=5)
    _wire(monkeypatch, tmp_path, [_model()],
          {"engine": "vllm", "vllm": VENV_BLOCK}, extra_machines=[host])
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "host_name": "aibox1"}, f)
    with pytest.raises(ValueError, match="swap_config_path_vllm"):
        swap_manager.generate(None, "aibox1")


def test_engine_swap_config_path_unknown_engine_raises():
    with pytest.raises(ValueError, match="unknown engine"):
        swap_manager.engine_swap_config_path("mlx")


def test_engine_swap_config_path_resolution_matrix(monkeypatch, tmp_path):
    """Full resolution order: machine override > explicit param > global
    per-engine key (config_path_vllm) > derived engine-qualified default;
    llama.cpp always verbatim; vllm derive is idempotent."""
    host = _machine("aibox1", "127.0.0.1", role="master",
                    swap_config_path_vllm="~/mach/v-from-machine.yaml")
    _wire(monkeypatch, tmp_path, [], {}, extra_machines=[host])
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "config_path_vllm": "~/settings/v.yaml",
                        "host_name": "aibox1"}, f)
    p, fb = swap_manager.engine_swap_config_path("vllm", None, "aibox1")
    assert p == "~/mach/v-from-machine.yaml" and fb is False
    # machine override beats even an explicit param (preview/launch)
    p, _ = swap_manager.engine_swap_config_path(
        "vllm", "~/preview/x.yaml", "aibox1")
    assert p == "~/mach/v-from-machine.yaml"
    # drop the machine override: explicit param now wins verbatim
    host.pop("swap_config_path_vllm")
    p, _ = swap_manager.engine_swap_config_path(
        "vllm", "~/preview/x.yaml", "aibox1")
    assert p == "~/preview/x.yaml"
    # drop the param: the settings key wins verbatim
    p, _ = swap_manager.engine_swap_config_path("vllm", None, "aibox1")
    assert p == "~/settings/v.yaml"
    # drop the settings key: derive from llama.cpp's path (sibling file)
    with open(tmp_path / "swap-settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_path": "~/.config/llama-swap/config.yaml",
                        "host_name": "aibox1"}, f)
    p, _ = swap_manager.engine_swap_config_path("vllm", None, "aibox1")
    assert p == "~/.config/llama-swap/config.vllm.yaml"
    # llama.cpp: always the saved path verbatim
    p, _ = swap_manager.engine_swap_config_path("llama_cpp", None, "aibox1")
    assert p == "~/.config/llama-swap/config.yaml"


def test_engine_qualify_path_idempotent_and_bare():
    q = "~/.config/llama-swap/config.vllm.yaml"
    assert swap_manager._engine_qualify_path(q, "vllm") == q
    assert swap_manager._engine_qualify_path("config.yaml", "vllm") == \
        "config.vllm.yaml"
    assert swap_manager._engine_qualify_path("", "vllm") == ""
    assert swap_manager._engine_qualify_path(
        "%USERPROFILE%\\.config\\llama-swap\\config.yaml", "vllm") == (
        "%USERPROFILE%\\.config\\llama-swap\\config.vllm.yaml")


def test_load_save_settings_roundtrip_config_path_vllm(
        monkeypatch, tmp_path):
    """swap.yaml gains config_path_vllm (DEC-03): load surfaces it; save
    persists it; "" clears back to derive-from-config_path (None)."""
    settings_file = str(tmp_path / "swap.yaml")
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(swap_manager, "get_host",
                        lambda name=None: {"name": "aibox1", "ip": "1.2.3.4",
                                           "os": "linux"}, raising=False)
    out = swap_manager.save_settings(config_path="~/.config/llama-swap/config.yaml",
                                     config_path_vllm="~/custom/v.yaml")
    assert out["config_path_vllm"] == "~/custom/v.yaml"
    with open(settings_file, encoding="utf-8") as f:
        on_disk = yaml.safe_load(f)
    assert on_disk["config_path_vllm"] == "~/custom/v.yaml"
    # "" clears back to None (derive)
    out = swap_manager.save_settings(config_path_vllm="")
    assert out["config_path_vllm"] is None
    # load surfaces the key even when absent on disk (fresh installs)
    with open(settings_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"host_name": "aibox1"}, f)
    assert "config_path_vllm" in swap_manager.load_settings()

# ======================================================================
# qe-validation-test branch: secrets-vault / installer regression tests
# (helper names prefixed _qe_ to avoid collision with the vllm-branch helpers above)
# ======================================================================

import types

import pytest

import rpc_manager
import swap_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qe__machine(name, ip, os="linux", role="peer", **kw):
    m = {
        "name": name,
        "ip": ip,
        "username": "zazi",
        "password": "secret",
        "os": os,
        "rpc_port": 50052,
        "gpu": "",
        "role": role,
        "detected": {"gpu_names": [], "vram_total_gb": 0, "ram_total_gb": 16},
    }
    m.update(kw)
    return m


class _FakeSSH:
    """Records commands and returns a configurable exit code."""
    def __init__(self, exit_code=0):
        self.calls = []
        self._exit_code = exit_code

    def run(self, cmd, timeout=20):
        self.calls.append(cmd)
        stdout = "ok" if self._exit_code == 0 else ""
        stderr = "" if self._exit_code == 0 else "error"
        return self._exit_code, stdout, stderr

    def is_reachable(self):
        return True


# ---------------------------------------------------------------------------
# daemon start (launch path)
# ---------------------------------------------------------------------------

def test_start_linux_service_issues_llama_swap_command(monkeypatch):
    """_start_linux_service must run the llama-swap binary with a config
    and listen port (systemd enable or detached fallback)."""
    machine = _qe__machine("swapbox", "10.0.0.50")
    calls = []

    written = {}

    monkeypatch.setattr(swap_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: False),
                        raising=False)
    monkeypatch.setattr(swap_manager, "_remote_home",
                        lambda ssh: "/home/you", raising=False)
    monkeypatch.setattr(rpc_manager, "_ssh_for",
                        lambda m: _FakeSSH(exit_code=0), raising=False)
    monkeypatch.setattr(swap_manager, "_write_host",
                        lambda host, path, text: written.update({path: text}),
                        raising=False)
    monkeypatch.setattr(swap_manager, "_resolve_binary",
                        lambda host, b: b, raising=False)
    monkeypatch.setattr(swap_manager, "_run",
                        lambda host, cmd, timeout=30: calls.append(cmd) or (0, "", ""),
                        raising=False)

    result = swap_manager._start_linux_service(
        machine, "llama-swap", "/etc/llama-swap/config.yaml", 8080)

    # binary + config live in the systemd unit's ExecStart; the _run call
    # only needs to enable it
    unit_text = "".join(written.values())
    assert "ExecStart=" in unit_text and "--config" in unit_text, (
        f"systemd unit does not ExecStart llama-swap with a config: {written}")
    assert result.get("method") == "systemd", f"unexpected result: {result}"


def test_start_linux_service_failure_falls_back(monkeypatch):
    """When systemctl fails, the detached-process fallback must still run
    llama-swap (never silently succeed with nothing running)."""
    machine = _qe__machine("swapbox", "10.0.0.50")
    calls = []

    monkeypatch.setattr(swap_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: False),
                        raising=False)
    monkeypatch.setattr(swap_manager, "_remote_home",
                        lambda ssh: "/home/you", raising=False)
    monkeypatch.setattr(rpc_manager, "_ssh_for",
                        lambda m: _FakeSSH(exit_code=0), raising=False)
    monkeypatch.setattr(swap_manager, "_write_host",
                        lambda host, path, text: None, raising=False)
    monkeypatch.setattr(swap_manager, "_resolve_binary",
                        lambda host, b: b, raising=False)
    monkeypatch.setattr(swap_manager, "_run",
                        lambda host, cmd, timeout=30: calls.append(cmd) or (1, "", "err"),
                        raising=False)

    result = swap_manager._start_linux_service(
        machine, "llama-swap", "/etc/llama-swap/config.yaml", 8080)

    assert result.get("method") != "systemd", (
        f"claimed systemd success despite systemctl failure: {result}")
    assert any("llama-swap" in c for c in calls), (
        f"fallback start never ran llama-swap; captured: {calls}")


# ---------------------------------------------------------------------------
# stop (daemon lifecycle)
# ---------------------------------------------------------------------------

def test_stop_sends_kill_command(monkeypatch):
    """stop() must issue a command that terminates the llama-swap process
    (pkill / taskkill / killall) on the host machine."""
    machine = _qe__machine("swapbox", "10.0.0.50")
    calls = []

    monkeypatch.setattr(swap_manager, "_run",
                        lambda host, cmd, timeout=30: calls.append(cmd) or (0, "ok", ""),
                        raising=False)
    monkeypatch.setattr(swap_manager, "_wait_gone", lambda host, tries=6: True,
                        raising=False)
    monkeypatch.setattr(swap_manager, "get_host", lambda name=None: machine,
                        raising=False)
    monkeypatch.setattr(swap_manager, "load_settings",
                        lambda: {"host_name": "swapbox"},
                        raising=False)

    result = swap_manager.stop()

    kill_terms = ("pkill", "kill", "Stop-Process", "taskkill", "killall")
    assert any(
        any(term in c for term in kill_terms) for c in calls
    ), f"stop did not send any kill command; captured: {calls}"
    assert result.get("stopped") is True, f"stop did not report stopped: {result}"


def test_stop_targets_correct_machine(monkeypatch):
    """stop() must act on the host resolved from settings.host_name
    (get_host), not on some other configured machine."""
    machine_a = _qe__machine("swapbox-a", "10.0.0.50")
    machine_b = _qe__machine("swapbox-b", "10.0.0.51")
    run_hosts = []

    def fake_run(host, cmd, timeout=30):
        run_hosts.append(host["name"])
        return 0, "ok", ""

    monkeypatch.setattr(swap_manager, "_run", fake_run, raising=False)
    monkeypatch.setattr(swap_manager, "_wait_gone", lambda host, tries=6: True,
                        raising=False)
    monkeypatch.setattr(swap_manager, "get_host", lambda name=None: machine_b,
                        raising=False)
    monkeypatch.setattr(swap_manager, "load_settings",
                        lambda: {"host_name": "swapbox-b"},
                        raising=False)

    swap_manager.stop()

    assert "swapbox-b" in run_hosts, (
        f"stop did not target the configured host; ran on: {run_hosts}")
    assert "swapbox-a" not in run_hosts, (
        f"stop unexpectedly targeted the wrong machine; ran on: {run_hosts}")


# ---------------- deployment helpers: generator decides rpc vs local -------

def _wire_helpers(monkeypatch, tmp_path, models, extra_machines=()):
    by_name = {"aibox1": _machine("aibox1", "127.0.0.1", role="master",
                                  gpu="CUDA0,CUDA1")}
    for m in extra_machines:
        by_name[m["name"]] = m
    models_by_id = {m["model_id"]: m for m in models}
    monkeypatch.setattr(swap_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(by_name),
        swap_groups=types.SimpleNamespace(all=lambda: []),
        get_model=models_by_id.get,
        list_models=lambda: list(models),
        machine_swap_config_override=cfg.machine_swap_config_override))
    monkeypatch.setattr(rpc_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(by_name),
        get_model=models_by_id.get))
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE",
                        str(tmp_path / "swap-settings.yaml"))


def test_deployment_launch_mode_single_peer_chips_is_local(monkeypatch,
                                                           tmp_path):
    """Heuristic: every chip on one peer box -> local on that box, never
    an RPC chain (the DAD shape: all aibox3 CUDA, host aibox1)."""
    peer = _machine("aibox3", "10.0.0.98")
    model = _model("qwen3-8b", servers=["aibox1", "aibox3"],
                   device_order="aibox3-CUDA0,aibox3-CUDA1,aibox3-CUDA2")
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    verdict = swap_manager.deployment_launch_mode("qwen3-8b")
    assert verdict["mode"] == "local"
    assert verdict["box"] == "aibox3"
    assert verdict["swap_key"] == "qwen3-8b"


def test_deployment_launch_mode_spread_chips_is_rpc(monkeypatch, tmp_path):
    """Chips spanning host + peer -> classic RPC chain."""
    peer = _machine("aibox3", "10.0.0.98")
    model = _model("qwen3-8b", servers=["aibox1", "aibox3"],
                   device_order="aibox1-CUDA0,aibox3-CUDA0")
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    verdict = swap_manager.deployment_launch_mode("qwen3-8b")
    assert verdict["mode"] == "rpc"
    assert verdict["box"] is None


def test_deployment_launch_mode_explicit_pin_wins(monkeypatch, tmp_path):
    """Explicit launch_mode=rpc beats the single-peer hardware signal —
    the operator's pin always wins (same rule the generator honors)."""
    peer = _machine("aibox3", "10.0.0.98")
    model = _model("qwen3-8b", servers=["aibox1", "aibox3"],
                   device_order="aibox3-CUDA0", launch_mode="rpc")
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    assert swap_manager.deployment_launch_mode("qwen3-8b")["mode"] == "rpc"


def test_deployment_launch_mode_mlx_is_local(monkeypatch, tmp_path):
    """MLX runtime never rides RPC peers — decided local up-front so
    every consumer of the heuristic agrees with the generator."""
    peer = _machine("macmini", "10.0.0.99", gpu_vendor="apple", gpu="Metal")
    model = _model("qwen3-mlx", servers=["macmini"], runtime="mlx")
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    verdict = swap_manager.deployment_launch_mode("qwen3-mlx")
    assert verdict["mode"] == "local"


def test_deployment_swap_key_variant_and_combined_id(monkeypatch, tmp_path):
    """Variant swap keys sanitize the label (base--label); combined
    'base--label' ids resolve without an explicit variant arg."""
    model = _model("qwen3-8b", servers=["aibox3"])
    model["variants"] = [{"label": "DAD", "servers": ["aibox3"]}]
    _wire_helpers(monkeypatch, tmp_path, [model])
    assert swap_manager.deployment_swap_key("qwen3-8b", "DAD") == \
        "qwen3-8b--DAD"
    merged, base, label = swap_manager.resolve_deployment("qwen3-8b--DAD")
    assert (base, label) == ("qwen3-8b", "DAD")
    assert merged["servers"] == ["aibox3"]


def test_resolve_deployment_unknown_raises(monkeypatch, tmp_path):
    """Unknown model (or variant) -> ValueError: callers fall back to
    legacy behavior, never a silent mis-route."""
    _wire_helpers(monkeypatch, tmp_path, [_model("qwen3-8b")])
    import pytest as _pytest
    with _pytest.raises(ValueError):
        swap_manager.resolve_deployment("ghost-model")
    with _pytest.raises(ValueError):
        swap_manager.resolve_deployment("qwen3-8b", "ghost-variant")


# ---------------- duplicate swap claims are loud, not silent ---------------

def test_duplicate_swap_claims_flags_shared_member():
    groups = [
        {"name": "rpc-cluster", "swap": True, "engine": "llama_cpp",
         "members": ["qwen3-8b", "huihui-35b"]},
        {"name": "aibox1-only", "swap": True, "engine": "llama_cpp",
         "members": ["qwen3-8b", "ornith-9b"]},
    ]
    warnings = swap_manager.duplicate_swap_claims(groups)
    assert len(warnings) == 1
    assert "qwen3-8b" in warnings[0]
    assert "rpc-cluster" in warnings[0] and "aibox1-only" in warnings[0]


def test_duplicate_swap_claims_ignores_non_swap_and_cross_engine():
    groups = [
        {"name": "g1", "swap": True, "engine": "llama_cpp",
         "members": ["qwen3-8b"]},
        {"name": "g2", "swap": False, "engine": "llama_cpp",
         "members": ["qwen3-8b"]},
        {"name": "g3", "swap": True, "engine": "vllm",
         "members": ["qwen3-8b"]},
    ]
    assert swap_manager.duplicate_swap_claims(groups) == []


# ---------------- remote launcher: no reset loop on retry -------------------

def test_remote_launcher_skips_kill_when_starting(monkeypatch, tmp_path):
    """Retry-cascade guard: when a matching llama-server process is alive
    but its port isn't open yet (still loading), the launcher must NOT
    pkill it — every blind kill resets a multi-minute load to zero."""
    peer = _machine("aibox3", "10.0.0.98")
    model = _model("qwen3-8b", servers=["aibox1", "aibox3"],
                   device_order="aibox3-CUDA0")
    _wire(monkeypatch, tmp_path, [model], {"engine": "llama_cpp"},
          [peer])
    script = rpc_manager.render_remote_local_script(
        dict(model, device_order="aibox3-CUDA0"), peer, 8081)
    pkill_pos = script.find("pkill -u")
    guard_pos = script.find("already starting")
    assert guard_pos != -1 and pkill_pos != -1
    assert guard_pos < pkill_pos  # guard precedes the kill
    # end-or-space anchor: `--port N( |$)` — cmdline may END at the port
    assert "pgrep -u" in script and "--port 8081( |$)" in script


def test_remote_launcher_per_port_log(monkeypatch, tmp_path):
    """Co-served models (and successive attempts) must not share one
    remote llama-server.log — the log name carries the port."""
    peer = _machine("aibox3", "10.0.0.98")
    model = _model("qwen3-8b", servers=["aibox1", "aibox3"],
                   device_order="aibox3-CUDA0")
    _wire(monkeypatch, tmp_path, [model], {"engine": "llama_cpp"},
          [peer])
    script = rpc_manager.render_remote_local_script(
        dict(model, device_order="aibox3-CUDA0"), peer, 8081)
    assert "llama-server-8081.log" in script
    assert "> \\$HOME/llama-server.log " not in script
    import subprocess as _sp
    r = _sp.run(["bash", "-n"], input=script, capture_output=True,
                text=True)
    assert r.returncode == 0, r.stderr


# ---------------- rendered launchers must survive `set -u` ------------------
# Regression: the windows remote-local wrapper once emitted `\\$c`
# (double backslash) in its SSH probe lines — bash expanded the unset
# $c and died in ~250ms ("c: unbound variable"), so winbox models could
# never start. These tests EXECUTE the rendered wrappers with mocked
# ssh/sshpass/scp: the reuse path must be reached with no unbound
# variable (a 124 timeout proves it sat in the stay-alive loop instead
# of dying on line 7).

def _run_wrapper_with_mock_ssh(script_text, tmp_path):
    import os as _os
    import stat as _stat
    import subprocess as _sp
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    (fakebin / "sshpass").write_text(
        "#!/bin/bash\nif [ \"$1\" = \"-p\" ]; then shift 2; fi\nexec \"$@\"\n",
        encoding="utf-8")
    (fakebin / "ssh").write_text("#!/bin/bash\nexit 0\n",
                                 encoding="utf-8")
    (fakebin / "scp").write_text("#!/bin/bash\nexit 0\n",
                                 encoding="utf-8")
    for exe in ("sshpass", "ssh", "scp"):
        _os.chmod(fakebin / exe, 0o755)
    sh = tmp_path / "wrapper.sh"
    sh.write_text(script_text, encoding="utf-8")
    env = dict(_os.environ, PATH=f"{fakebin}{_os.pathsep}{_os.environ['PATH']}",
               SSH_PASS="x")
    # reuse path loops `sleep 5` forever: 124 == still alive in it.
    r = _sp.run(["timeout", "8", "bash", str(sh)], capture_output=True,
                text=True, env=env, timeout=30)
    return r


def test_windows_wrapper_reuse_probe_survives_set_u(monkeypatch, tmp_path):
    """Windows remote-local wrapper, port already serving: must print
    the reuse marker and idle (rc 124), never `c: unbound variable`."""
    peer = _machine("winbox", "10.0.0.124", os="windows")
    model = _model("nomic-1", servers=["aibox1", "winbox"],
                   device_order="winbox-CUDA0")
    _wire(monkeypatch, tmp_path, [model], {"engine": "llama_cpp"},
          [peer])
    script = rpc_manager.render_remote_local_script(
        dict(model, device_order="winbox-CUDA0"), peer, 52353)
    r = _run_wrapper_with_mock_ssh(script, tmp_path)
    assert "already serving — reusing" in r.stdout, r.stderr
    assert "unbound variable" not in r.stderr
    assert r.returncode == 124  # alive in stay-alive loop, not dead on line 7


def test_linux_wrapper_reuse_probe_survives_set_u(monkeypatch, tmp_path):
    """Same contract, POSIX branch (single backslash already correct —
    this locks the shape for both)."""
    peer = _machine("aibox3", "10.0.0.98")
    model = _model("qwen3-8b", servers=["aibox1", "aibox3"],
                   device_order="aibox3-CUDA0")
    _wire(monkeypatch, tmp_path, [model], {"engine": "llama_cpp"},
          [peer])
    script = rpc_manager.render_remote_local_script(
        dict(model, device_order="aibox3-CUDA0"), peer, 8080)
    r = _run_wrapper_with_mock_ssh(script, tmp_path)
    assert "already serving — reusing" in r.stdout, r.stderr
    assert "unbound variable" not in r.stderr
    assert r.returncode == 124


def test_no_double_escaped_remote_vars_in_wrappers(monkeypatch, tmp_path):
    """Static backstop for the class: no `\\\\$var` (which bash would
    expand under set -u) may appear in any rendered remote-local
    wrapper, linux or windows."""
    import subprocess as _sp
    for os_name, peer in (("linux", _machine("aibox3", "10.0.0.98")),
                          ("windows", _machine("winbox", "10.0.0.124",
                                              os="windows"))):
        model = _model("m1", servers=["aibox1", peer["name"]],
                       device_order=f"{peer['name']}-CUDA0")
        _wire(monkeypatch, tmp_path, [model], {"engine": "llama_cpp"},
              [peer])
        script = rpc_manager.render_remote_local_script(
            dict(model, device_order=f"{peer['name']}-CUDA0"), peer, 8080)
        assert "\\\\$c" not in script and "\\\\$env" not in script, os_name
        r = _sp.run(["bash", "-n"], input=script, capture_output=True,
                    text=True)
        assert r.returncode == 0, (os_name, r.stderr)


def test_windows_wrapper_ps1_run_uses_double_quoted_file(monkeypatch,
                                                         tmp_path):
    """Remote-local windows wrapper must invoke the uploaded ps1 via
    -File "<double-quoted path>": remote cmd.exe expands %USERPROFILE%
    and single quotes arrive doubled (''C:\\..''), which PowerShell
    rejects ("path's format is not supported", observed live). The
    double-quote idiom is the proven shape from the RPC launcher."""
    import os as _os
    import subprocess as _sp
    peer = _machine("winbox", "10.0.0.124", os="windows")
    model = _model("nomic-1", servers=["aibox1", "winbox"],
                   device_order="winbox-CUDA0")
    _wire(monkeypatch, tmp_path, [model], {"engine": "llama_cpp"},
          [peer])
    script = rpc_manager.render_remote_local_script(
        dict(model, device_order="winbox-CUDA0"), peer, 52352)
    assert '-File \\"%USERPROFILE%\\start-remote-local.ps1\\"' in script
    assert "-File '%USERPROFILE%" not in script
    # execution: all probes fail (port closed), scp + ps1-run succeed —
    # the recorded ps1-run invocation must carry the double-quoted -File.
    fakebin = tmp_path / "fakebin2"
    fakebin.mkdir(exist_ok=True)
    (fakebin / "sshpass").write_text(
        "#!/bin/bash\nif [ \"$1\" = \"-p\" ]; then shift 2; fi\nexec \"$@\"\n",
        encoding="utf-8")
    (fakebin / "ssh").write_text(
        "#!/bin/bash\necho \"$@\" >> \"$SSH_LOG\"\nexit 1\n",
        encoding="utf-8")
    (fakebin / "scp").write_text("#!/bin/bash\nexit 0\n",
                                 encoding="utf-8")
    for exe in ("sshpass", "ssh", "scp"):
        _os.chmod(fakebin / exe, 0o755)
    sh = tmp_path / "win-wrapper.sh"
    sh.write_text(script, encoding="utf-8")
    sshlog = tmp_path / "ssh.log"
    env = dict(_os.environ, PATH=f"{fakebin}{_os.pathsep}{_os.environ['PATH']}",
               SSH_PASS="x", SSH_LOG=str(sshlog))
    r = _sp.run(["timeout", "20", "bash", str(sh)], capture_output=True,
                text=True, env=env, timeout=40)
    assert "unbound variable" not in r.stderr
    runs = sshlog.read_text(encoding="utf-8")
    assert "ExecutionPolicy Bypass -File" in runs
    assert "-File '%USERPROFILE%" not in runs


def test_windows_ps1_template_is_ascii_only(monkeypatch, tmp_path):
    """Win PowerShell 5.1 -File reads BOM-less scripts as Windows-1252:
    any non-ASCII byte (e.g. an em-dash, whose 0x94 byte decodes as a
    literal double-quote) toggles string state and kills the parse with
    MissingEndCurlyBrace (observed live on winbox). The embedded ps1
    must therefore be pure ASCII. The surrounding bash wrapper runs on
    the Linux host and keeps its UTF-8 comments (Linux/Mac paths are
    proven working and stay untouched)."""
    peer = _machine("winbox", "10.0.0.124", os="windows")
    model = _model("nomic-1", servers=["aibox1", "winbox"],
                   device_order="winbox-CUDA0")
    _wire(monkeypatch, tmp_path, [model], {"engine": "llama_cpp"},
          [peer])
    script = rpc_manager.render_remote_local_script(
        dict(model, device_order="winbox-CUDA0"), peer, 52352)
    start = script.find("$ErrorActionPreference")
    end = script.find("PS1", start)
    assert start != -1 and end != -1
    ps1 = script[start:end]
    assert ps1.isascii(), [c for c in set(ps1) if ord(c) > 126]


def test_windows_single_peer_auto_forces_rpc(monkeypatch, tmp_path):
    """Windows remotes are never local-execution targets (no SMB over
    SSH-launch sessions): single-peer auto resolves rpc even with all
    chips on the Windows box."""
    peer = _machine("winbox", "10.0.0.124", os="windows")
    model = _model("nomic-1", servers=["aibox1", "winbox"],
                   device_order="winbox-CUDA0")
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    verdict = swap_manager.deployment_launch_mode("nomic-1")
    assert verdict["mode"] == "rpc"
    assert verdict["box"] is None


def test_windows_no_chip_order_also_forces_rpc(monkeypatch, tmp_path):
    """Same rule without explicit chips (e.g. winbox-only server list)."""
    peer = _machine("winbox", "10.0.0.124", os="windows")
    model = _model("asr-1", servers=["winbox"])
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    assert swap_manager.deployment_launch_mode("asr-1")["mode"] == "rpc"


def test_explicit_local_pin_still_wins_on_windows(monkeypatch, tmp_path):
    """An explicit launch_mode=local pin is an operator override and
    beats the Windows rule (visible breakage is the operator's choice)."""
    peer = _machine("winbox", "10.0.0.124", os="windows")
    model = _model("nomic-1", servers=["aibox1", "winbox"],
                   device_order="winbox-CUDA0", launch_mode="local")
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    assert swap_manager.deployment_launch_mode("nomic-1")["mode"] == "local"


def test_linux_single_peer_still_local(monkeypatch, tmp_path):
    """Guard: the Windows rule must not leak onto Linux peers."""
    peer = _machine("aibox3", "10.0.0.98")
    model = _model("qwen3-8b", servers=["aibox1", "aibox3"],
                   device_order="aibox3-CUDA0")
    _wire_helpers(monkeypatch, tmp_path, [model], [peer])
    verdict = swap_manager.deployment_launch_mode("qwen3-8b")
    assert verdict["mode"] == "local"
    assert verdict["box"] == "aibox3"


def test_winbox_entry_generates_rpc_wrapper(monkeypatch, tmp_path):
    """End to end: a winbox-only model plans as an RPC-chain script
    entry (no remote-proxy, no ps1 launcher, no cmdStop pkill)."""
    peer = _machine("winbox", "10.0.0.124", os="windows")
    model = _model("nomic-1", servers=["aibox1", "winbox"],
                   device_order="winbox-CUDA0")
    plan = _generate(monkeypatch, tmp_path, [model],
                     {"engine": "llama_cpp"}, [peer])
    assert plan["entries"][0]["mode"] == "script"
    assert "start-nomic-1.sh" in plan["_scripts"]
    assert "--rpc" in plan["_scripts"]["start-nomic-1.sh"]
