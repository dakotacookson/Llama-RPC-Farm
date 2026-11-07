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
    # script-mode RPC deployments carry an explicit cmdStop so peer
    # rpc-servers die even when the start-script EXIT trap is bypassed.
    # It must be a pushed bash script (sources the sidecar) — a raw curl
    # with ${AGENT_TOKEN_*} would be parsed by llama-swap as a macro.
    assert emitted["cmdStop"].startswith('/bin/bash "')
    stop_name = emitted["cmdStop"].split("/")[-1].strip('"')
    stop_script = plan["_scripts"][stop_name]
    assert "/rpc/stop" in stop_script and "swap-secrets.env" in stop_script
    assert "AGENT_TOKEN_" in stop_script
    script = plan["_scripts"]["start-qwen3-8b.sh"]
    assert "--rpc" in script
    assert "/rpc/ensure" in script
    assert "sshpass" not in script
    assert "vllm" not in script


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
    # Python-managed timeout avoids relying on GNU coreutils `timeout`,
    # which macOS does not provide by default.
    try:
        return _sp.run(["bash", str(sh)], capture_output=True,
                       text=True, env=env, timeout=8)
    except _sp.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return types.SimpleNamespace(returncode=124,
                                     stdout=stdout,
                                     stderr=stderr)


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
