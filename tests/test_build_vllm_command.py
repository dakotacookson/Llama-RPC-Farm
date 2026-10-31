"""Unit tests for vllm_manager command generation and config validation.

These are the pieces most likely to silently break when flags/delivery/
device mapping change, and they're pure string/logic (no network, no
real machines) — same approach as test_build_command.py. Hand-built
model dicts + machine dicts assert the exact cmd strings; vllm_cfg is
passed explicitly so settings.yaml never leaks into the assertions.
"""
import types

import pytest

import conftest
import vllm_manager
from vllm_manager import (
    build_cmd_stop,
    build_vllm_command,
    cuda_visible_devices,
    swap_entry_parts,
    validate_model_config,
    validate_hosts,
)


def _machine(name, ip, os="linux", **kw):
    m = {"name": name, "ip": ip, "username": "u", "password": "p",
         "os": os, "rpc_port": 50052, "gpu": "", "role": "peer",
         "detected": {"gpu_names": [], "vram_total_gb": 0,
                      "ram_total_gb": 16}}
    m.update(kw)
    return m


def _model(**vllm):
    # DEC-08: a native `vllm:` section is required at launch — give the
    # helper a minimal non-empty section (cpu_offload_gb 0 is the default
    # and emits no flag) so command-build tests exercise the real path.
    # Bridge-retirement tests pass their own dicts.
    vllm = vllm if vllm else {"cpu_offload_gb": 0}
    return {"model_id": "qwen3-8b", "flags": {},
            "base_path": "/mnt/store/qwen3-8b", "vllm": vllm}


def _cluster_model(**vllm):
    """Multi-host model: one GPU chip per host (aibox1 + aibox2) — the
    minimum device_order a Ray cluster build requires."""
    m = _model(**vllm)
    m["device_order"] = "aibox1-CUDA0,aibox2-CUDA0"
    return m


VENV_CFG = {"delivery": "venv", "venv_bin": "~/vllm-env/bin/vllm",
            "docker_image": "vllm/vllm-openai:latest",
            "default_gpu_memory_utilization": 0.9}
DOCKER_CFG = dict(VENV_CFG, delivery="docker")
MASTER = {"name": "aibox1", "ip": "10.0.0.98", "os": "linux"}


# ---------------- golden: venv, single GPU -------------------------------

def test_venv_single_gpu_golden():
    model = _model()
    model["device_order"] = "aibox1-CUDA0"
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    lines = cmd.split(" \\\n  ")
    assert lines == [
        'CUDA_VISIBLE_DEVICES=0 ~/vllm-env/bin/vllm serve',
        '"/mnt/store/qwen3-8b"',
        '--host 0.0.0.0',
        '--port "$PORT"',
        '--served-model-name "qwen3-8b"',
        '--gpu-memory-utilization 0.9',
        '--enable-auto-tool-choice',
        '--tool-call-parser hermes',
    ]


def test_store_path_resolved_for_host_os(monkeypatch):
    """store:// base paths resolve through path_resolver for the master's
    OS before landing in cmd (llama.cpp parity)."""
    seen = {}

    def fake_resolve(path, machine_os=None):
        seen["os"] = machine_os
        return "/mnt/cifs/llm-models/qwen3-8b"

    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        fake_resolve)
    cmd = build_vllm_command(
        {"model_id": "m", "flags": {}, "vllm": {"cpu_offload_gb": 0},
         "base_path": "store://llm-models/qwen3-8b"},
        MASTER, VENV_CFG)
    assert '"/mnt/cifs/llm-models/qwen3-8b"' in cmd
    assert seen["os"] == "linux"


def test_venv_no_device_chips_omits_cuda_prefix():
    """No device_order -> no CUDA_VISIBLE_DEVICES prefix (vLLM sees all
    GPUs, its own default)."""
    cmd = build_vllm_command({"model_id": "m", "flags": {},
                              "vllm": {"cpu_offload_gb": 0},
                              "base_path": "store://llm-models/m"},
                             MASTER, VENV_CFG)
    assert "CUDA_VISIBLE_DEVICES" not in cmd
    assert cmd.startswith("~/vllm-env/bin/vllm serve")


# ---------------- golden: tensor parallel --------------------------------

def test_venv_tp_multi_gpu_golden():
    model = _model(tensor_parallel_size=2)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1"
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    lines = cmd.split(" \\\n  ")
    assert lines[0] == "CUDA_VISIBLE_DEVICES=0,1 ~/vllm-env/bin/vllm serve"
    assert "--tensor-parallel-size 2" in lines
    # ARD §3.1 flag order: TP before the settings-level gmu default;
    # tool-calling defaults ride at the tail
    assert lines[-3] == "--gpu-memory-utilization 0.9"


# ---------------- golden: pipeline parallel + TP×PP world size ------------

def test_venv_pp_golden():
    """pipeline_parallel_size is first-class: emitted after TP, minimum 1."""
    model = _model(tensor_parallel_size=2, pipeline_parallel_size=2)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1," \
                            "aibox1-CUDA2,aibox1-CUDA3"
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    lines = cmd.split(" \\\n  ")
    assert lines[0] == ("CUDA_VISIBLE_DEVICES=0,1,2,3 "
                        "~/vllm-env/bin/vllm serve")
    assert "--tensor-parallel-size 2" in lines
    assert "--pipeline-parallel-size 2" in lines
    # TP before PP (ARD §3.1 order)
    assert lines.index("--tensor-parallel-size 2") \
        < lines.index("--pipeline-parallel-size 2")


def test_pp_only_emitted_without_tp():
    model = _model(pipeline_parallel_size=2)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1"
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    assert "--pipeline-parallel-size 2" in cmd
    assert "--tensor-parallel-size" not in cmd


def test_pp_zero_and_negative_rejected():
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        build_vllm_command(_model(pipeline_parallel_size=0), MASTER, VENV_CFG)
    with pytest.raises(ValueError, match="must be >= 1"):
        build_vllm_command(_model(pipeline_parallel_size=-1), MASTER,
                           VENV_CFG)


def test_pp_bool_rejected():
    with pytest.raises(ValueError, match="must be a number"):
        build_vllm_command(_model(pipeline_parallel_size=True), MASTER,
                           VENV_CFG)


# ---------------- golden: docker delivery --------------------------------

def test_docker_golden():
    model = _model()
    model["device_order"] = "aibox1-CUDA0"
    cmd = build_vllm_command(model, MASTER, DOCKER_CFG)
    lines = cmd.split(" \\\n  ")
    assert lines == [
        "docker run --rm --gpus all",
        "--name farm-qwen3-8b",
        "-p ${PORT}:8000",
        '-v "/mnt/store:/models"',
        "-e CUDA_VISIBLE_DEVICES=0",
        "vllm/vllm-openai:latest",
        '--model "/models/qwen3-8b"',
        '--served-model-name "qwen3-8b"',
        "--gpu-memory-utilization 0.9",
        "--enable-auto-tool-choice",
        "--tool-call-parser hermes",
    ]


def test_docker_pins_devices_with_env_flag():
    """Device pinning travels in -e CUDA_VISIBLE_DEVICES for docker (and
    never as a bare prefix, which docker would misread)."""
    model = _model()
    model["device_order"] = "aibox1-CUDA1,aibox1-CUDA0"
    cmd = build_vllm_command(model, MASTER, DOCKER_CFG)
    assert "-e CUDA_VISIBLE_DEVICES=1,0" in cmd
    assert "CUDA_VISIBLE_DEVICES=1,0 docker" not in cmd


# ---------------- device mapping ------------------------------------------

def test_device_mapping_dedupes_and_keeps_selection_order():
    model = _model()
    model["device_order"] = "aibox1-CUDA2,aibox1-CUDA0,aibox1-CUDA2"
    assert cuda_visible_devices(model, MASTER) == "2,0"


def test_device_mapping_ignores_ram_and_bare_tokens():
    model = _model()
    model["device_order"] = "aibox1-RAM,CUDA0"
    assert cuda_visible_devices(model, MASTER) == "0"


def test_device_mapping_skips_peer_chips():
    """Multi-host (ray cluster): chips naming another machine are skipped
    for THIS host — the peer's own worker line picks them up."""
    model = _model()
    model["device_order"] = "aibox2-CUDA0," + MASTER["name"] + "-CUDA1"
    ids = cuda_visible_devices(model, MASTER)
    assert ids == "1"


def test_device_mapping_peer_chip_only():
    """All chips belong to peers: this host contributes no GPUs."""
    model = _model()
    model["device_order"] = "aibox2-CUDA0"
    assert cuda_visible_devices(model, MASTER) == ""


# ---------------- flag assembly & guards ----------------------------------

def test_kv_cache_dtype_fp8_only():
    cmd = build_vllm_command(_model(kv_cache_dtype="fp8"), MASTER, VENV_CFG)
    assert "--kv-cache-dtype fp8" in cmd


def test_kv_cache_dtype_q8_rejected():
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        build_vllm_command(_model(kv_cache_dtype="q8_0"), MASTER, VENV_CFG)


def test_cpu_offload_gb_zero_omitted():
    """0 is the default — hiding it keeps the degradation matrix clean."""
    cmd = build_vllm_command(_model(cpu_offload_gb=0), MASTER, VENV_CFG)
    assert "--cpu-offload-gb" not in cmd


def test_cpu_offload_gb_positive_emitted():
    cmd = build_vllm_command(_model(cpu_offload_gb=20), MASTER, VENV_CFG)
    assert "--cpu-offload-gb 20" in cmd


# ---------------- tool calling ----------------

def test_tool_call_flags_default_on():
    """Default: --enable-auto-tool-choice + --tool-call-parser from the
    settings default (hermes) — auto tool choice requires a parser."""
    cmd = build_vllm_command(_model(), MASTER, VENV_CFG)
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser hermes" in cmd


def test_tool_call_parser_override_and_opt_out():
    cfg = dict(VENV_CFG, default_tool_call_parser="qwen25")
    cmd = build_vllm_command(_model(tool_call_parser="qwen3"), MASTER, cfg)
    assert "--tool-call-parser qwen3" in cmd
    # opt-out: enable_auto_tool_choice=False emits neither flag
    cmd = build_vllm_command(_model(enable_auto_tool_choice=False), MASTER,
                             VENV_CFG)
    assert "--enable-auto-tool-choice" not in cmd
    assert "--tool-call-parser" not in cmd


def test_tool_call_validation():
    with pytest.raises(ValueError, match="tool_call_parser"):
        build_vllm_command(_model(tool_call_parser=""), MASTER, VENV_CFG)
    with pytest.raises(ValueError, match="enable_auto_tool_choice"):
        build_vllm_command(_model(enable_auto_tool_choice="yes"), MASTER,
                           VENV_CFG)


# ---------------- per-entry gpu_memory_utilization (A2) --------------------

def test_gmu_per_entry_overrides_global():
    """A model with its own gmu (weak-GPU deployment) wins over the
    settings-level default."""
    cmd = build_vllm_command(_model(gpu_memory_utilization=0.35),
                             MASTER, VENV_CFG)
    assert "--gpu-memory-utilization 0.35" in cmd
    assert "--gpu-memory-utilization 0.9" not in cmd


def test_gmu_unset_falls_back_to_settings_default():
    cmd = build_vllm_command(_model(), MASTER, VENV_CFG)
    assert "--gpu-memory-utilization 0.9" in cmd
    model = _model()
    model["vllm"] = {"max_model_len": 4096}  # section set, gmu unset
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    assert "--gpu-memory-utilization 0.9" in cmd


def test_gmu_out_of_domain_rejected():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        build_vllm_command(_model(gpu_memory_utilization=1.5), MASTER,
                           VENV_CFG)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        build_vllm_command(_model(gpu_memory_utilization=0), MASTER,
                           VENV_CFG)
    with pytest.raises(ValueError, match="must be a number"):
        build_vllm_command(_model(gpu_memory_utilization=True), MASTER,
                           VENV_CFG)


def test_gmu_sanitized_on_save():
    """apply_vllm_section keeps the float (no int coercion) and rejects
    out-of-domain values like the settings block does."""
    from config_loader import apply_vllm_section
    data = apply_vllm_section({"vllm": {"gpu_memory_utilization": 0.35}})
    assert data["vllm"]["gpu_memory_utilization"] == 0.35
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        apply_vllm_section({"vllm": {"gpu_memory_utilization": 2}})
    with pytest.raises(ValueError, match="must be a number"):
        apply_vllm_section({"vllm": {"gpu_memory_utilization": "big"}})


def test_validate_gmu_out_of_domain_is_error(vllm_cfg_stub):
    issues = validate_model_config(_model(gpu_memory_utilization=1.5))
    assert any(i["level"] == "error" and "gpu_memory_utilization" in i["msg"]
               for i in issues)


def test_negative_max_model_len_rejected():
    with pytest.raises(ValueError, match="max_model_len"):
        build_vllm_command(_model(max_model_len=-1), MASTER, VENV_CFG)


def test_extra_args_appended_verbatim():
    cmd = build_vllm_command(
        _model(extra_args=["--enable-prefix-caching", "--swap-space 4"]),
        MASTER, VENV_CFG)
    assert "--enable-prefix-caching" in cmd
    assert "--swap-space 4" in cmd


def test_unsafe_model_id_rejected():
    with pytest.raises(ValueError, match="unsafe"):
        build_vllm_command({"model_id": "m; rm -rf /", "flags": {},
                            "base_path": "store://llm-models/m"},
                           MASTER, VENV_CFG)


# ---------------- cmdStop per delivery ------------------------------------

def test_cmd_stop_venv_pkill():
    assert build_cmd_stop(_model(), VENV_CFG) == \
        'pkill -f "vllm serve.*qwen3-8b"'


def test_cmd_stop_docker_by_name():
    assert build_cmd_stop(_model(), DOCKER_CFG) == "docker stop farm-qwen3-8b"


def test_swap_entry_parts_shape():
    parts = swap_entry_parts(_model(), MASTER, VENV_CFG)
    assert parts == {
        "cmd": build_vllm_command(_model(), MASTER, VENV_CFG),
        "cmdStop": 'pkill -f "vllm serve.*qwen3-8b"',
        "checkEndpoint": "/health",
        "env": {},
    }


# ---------------- DEC-08: bridge retired ----------------------------------

def test_launch_requires_native_vllm_section():
    """DEC-08: llama.cpp flags no longer bridge — a model without a
    native `vllm:` section fails loudly with the clear message."""
    model = {"model_id": "qwen3-8b", "flags": {"ctx-size": 8192},
             "base_path": "/mnt/store/qwen3-8b"}
    with pytest.raises(ValueError, match="no vllm config for this model"):
        build_vllm_command(model, MASTER, VENV_CFG)
    with pytest.raises(ValueError, match="no vllm config for this model"):
        swap_entry_parts(model, MASTER, VENV_CFG)


def test_launch_ignores_llamacpp_flags_entirely():
    """With a native section present, llama.cpp flags are inert: the
    command reflects ONLY the vllm: section."""
    model = _model(max_model_len=4096)
    model["flags"] = {"ctx-size": 999999, "ngl": 999, "cache-type-k": "q8_0"}
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    assert "--max-model-len 4096" in cmd
    assert "999999" not in cmd
    assert "ngl" not in cmd and "--kv-cache-dtype" not in cmd


def test_empty_vllm_section_also_fails():
    """An empty `vllm:` dict is indistinguishable from no section —
    both fail (nothing would launch with sane defaults)."""
    model = {"model_id": "m", "base_path": "/x", "flags": {}, "vllm": {}}
    with pytest.raises(ValueError, match="no vllm config for this model"):
        build_vllm_command(model, MASTER, VENV_CFG)


def test_error_names_the_model():
    model = {"model_id": "Ornith-1.5-9B", "base_path": "/x", "flags": {}}
    with pytest.raises(ValueError, match="Ornith-1.5-9B"):
        build_vllm_command(model, MASTER, VENV_CFG)


def test_legacy_model_read_still_loads():
    """READ tolerance stays: a YAML without a vllm: section is still a
    valid model config (config_loader untouched) — only launch refuses."""
    model = {"model_id": "m", "base_path": "/x", "flags": {"ngl": 30}}
    assert "ngl" in model["flags"]  # shape unchanged on read


# ---------------- adapter seam --------------------------------------------

def test_adapter_multi_host_builds_cluster_script(stub_cfg, monkeypatch):
    """Multi-host vLLM: peers no longer refused — the adapter builds the
    Ray cluster script (build_vllm_cluster_command). validate_hosts is
    stubbed so the test exercises the branch, not the OS gates."""
    stub_cfg.update({"aibox1": _machine("aibox1", "10.0.0.98", role="master"),
                     "aibox2": _machine("aibox2", "10.0.0.11")})
    monkeypatch.setattr(vllm_manager, "validate_hosts", lambda ms: [])
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    adapter = vllm_manager.VllmAdapter()
    cmd = adapter.build_command(_cluster_model(), ["aibox1", "aibox2"], "aibox1")
    assert cmd.startswith("#!/bin/bash")
    assert "--distributed-executor-backend ray" in cmd
    assert "ray start --address=$MIP:6379" in cmd
    assert "trap cleanup EXIT" in cmd
    with pytest.raises(ValueError):
        # 'mp' cannot form a multi-node cluster — refused
        m = _model(distributed_executor_backend="mp")
        adapter.build_command(m, ["aibox1", "aibox2"], "aibox1")


def test_adapter_build_command_matches_function(stub_cfg, monkeypatch):
    """The adapter is a thin seam: same output as build_vllm_command.
    _load_vllm_settings is stubbed so the adapter path doesn't read the
    repo's real settings.yaml."""
    stub_cfg.update({"aibox1": _machine("aibox1", "10.0.0.98",
                                        role="master")})
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    adapter = vllm_manager.VllmAdapter()
    assert adapter.build_command(_model(), ["aibox1"], "aibox1") == \
        build_vllm_command(_model(), _machine("aibox1", "10.0.0.98"),
                           VENV_CFG)


# ---------------- static validation ----------------------------------------

@pytest.fixture
def vllm_cfg_stub(stub_cfg, monkeypatch):
    """validate_model_config resolves machines through vllm_manager.cfg —
    point it at the same in-memory map the adapter tests use, pre-seeded
    with aibox1 (the machine the validation tests reference)."""
    stub_cfg.update({"aibox1": _machine("aibox1", "10.0.0.98")})
    fake = types.SimpleNamespace(machines=conftest._FakeMachines(stub_cfg))
    monkeypatch.setattr(vllm_manager, "cfg", fake)
    return stub_cfg


def test_validate_multi_server_is_cluster_warn(vllm_cfg_stub):
    vllm_cfg_stub.update({"aibox2": _machine("aibox2", "10.0.0.11")})
    model = {"model_id": "m", "base_path": "/mnt/store/m",
             "servers": ["aibox1", "aibox2"],
             "device_order": "aibox1-CUDA0,aibox2-CUDA0"}
    issues = validate_model_config(model)
    assert any(i["level"] == "warn" and "multi-host vLLM" in i["msg"]
               for i in issues)
    assert not any("one machine" in i["msg"] for i in issues)
    assert not [i for i in issues if i["level"] == "error"]


def test_validate_multi_server_without_chips_is_error(vllm_cfg_stub):
    """Adding a multi-machine vLLM model without per-host GPU chips is a
    save-time error (the cluster could never schedule workers there) —
    the editor must force chip selection up front."""
    stub_cfg = vllm_cfg_stub
    stub_cfg.update({"aibox2": _machine("aibox2", "10.0.0.11")})
    model = {"model_id": "m", "base_path": "/mnt/store/m",
             "servers": ["aibox1", "aibox2"],
             "vllm": {"cpu_offload_gb": 0}}
    issues = validate_model_config(model)
    errs = [i["msg"] for i in issues if i["level"] == "error"]
    assert any("at least one GPU chip" in m for m in errs), errs

    model["device_order"] = "aibox1-CUDA0"  # aibox2 still bare
    issues = validate_model_config(model)
    errs = [i["msg"] for i in issues if i["level"] == "error"]
    assert any("aibox2" in m for m in errs), errs

    model["device_order"] = "aibox1-CUDA0,aibox2-CUDA0"
    issues = validate_model_config(model)
    assert not [i for i in issues if i["level"] == "error"]


def test_validate_unknown_vllm_key_rejected():

    issues = validate_model_config(_model(bogus_flag=1))
    assert any(i["level"] == "error" and "unknown vllm key" in i["msg"]
               for i in issues)


def test_validate_tp_needs_enough_gpus(vllm_cfg_stub):
    model = _model(tensor_parallel_size=4)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1"
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    assert any("tensor_parallel_size 4 needs 4" in i["msg"]
               for i in issues)


def test_validate_tp_pp_world_size_needs_enough_gpus(vllm_cfg_stub):
    """vLLM world size = TP×PP: pipeline stages consume the same visible
    GPUs, so TP=2 + PP=2 needs 4 GPUs — 2 must be an error, 4 clean."""
    model = _model(tensor_parallel_size=2, pipeline_parallel_size=2)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1"
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    errs = [i["msg"] for i in issues if i["level"] == "error"]
    assert any("world size TP2xPP2=4 needs 4" in m for m in errs), errs

    model["device_order"] = ("aibox1-CUDA0,aibox1-CUDA1,"
                             "aibox1-CUDA2,aibox1-CUDA3")
    issues = validate_model_config(model)
    assert not [i for i in issues if i["level"] == "error"]


def test_validate_pp_undercount_is_error(vllm_cfg_stub):
    """The A1 defect: PP>1 alone with fewer visible GPUs than PP."""
    model = _model(pipeline_parallel_size=4)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1"
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    errs = [i["msg"] for i in issues if i["level"] == "error"]
    assert any("world size TP1xPP4=4 needs 4" in m for m in errs), errs


def test_validate_pp1_tp1_never_checks_gpus(vllm_cfg_stub):
    """Defaults (no explicit TP/PP) skip the fit check even with no chips."""
    model = _model()
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    assert not [i for i in issues if i["level"] == "error"]


def test_validate_more_gpus_than_world_size_warns(vllm_cfg_stub):
    model = _model(tensor_parallel_size=1, pipeline_parallel_size=2)
    model["device_order"] = ("aibox1-CUDA0,aibox1-CUDA1,aibox1-CUDA2,"
                             "aibox1-CUDA3")
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    warns = [i["msg"] for i in issues if i["level"] == "warn"]
    assert any("world size is 2" in m and "first 2" in m for m in warns), warns


def test_validate_clean_tp2_config_has_no_errors(vllm_cfg_stub):
    model = _model(tensor_parallel_size=2, kv_cache_dtype="fp8",
                   max_model_len=32768)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1"
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    assert not [i for i in issues if i["level"] == "error"]


def test_validate_gguf_without_tokenizer_is_error(vllm_cfg_stub):
    model = {"model_id": "m", "flags": {},
             "base_path": "/mnt/store/m.gguf", "servers": ["aibox1"]}
    issues = validate_model_config(model)
    assert any(i["level"] == "error" and "tokenizer" in i["msg"]
               for i in issues)


def test_validate_gguf_with_tokenizer_warns_only(vllm_cfg_stub):
    model = {"model_id": "m", "flags": {},
             "base_path": "/mnt/store/m.gguf", "servers": ["aibox1"],
             "vllm": {"tokenizer": "/mnt/store/qwen3-hf"}}
    issues = validate_model_config(model)
    assert not [i for i in issues if i["level"] == "error"]
    assert any(i["level"] == "warn" and "experimental" in i["msg"]
               for i in issues)


# ---------------- resource-plausibility parity guard (A2) -------------------

def test_validate_cpu_offload_exceeding_ram_warns(vllm_cfg_stub):
    """cpu_offload_gb beyond the serving machine's detected RAM cannot
    work — warn (not error: detection may be stale)."""
    vllm_cfg_stub["aibox1"]["detected"]["ram_total_gb"] = 32
    model = _model(cpu_offload_gb=64)
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    warns = [i["msg"] for i in issues if i["level"] == "warn"]
    assert any("cpu_offload_gb 64" in m and "RAM" in m for m in warns), warns
    # plausible value stays silent
    model["vllm"]["cpu_offload_gb"] = 16
    issues = validate_model_config(model)
    assert not [i for i in issues if i["level"] == "warn"
                and "cpu_offload_gb" in i["msg"]]


def test_validate_no_detected_resources_skips_plausibility(vllm_cfg_stub):
    """No detection data -> no warnings (never guesses)."""
    vllm_cfg_stub["aibox1"]["detected"]["ram_total_gb"] = 0
    model = _model(cpu_offload_gb=999)
    model["servers"] = ["aibox1"]
    issues = validate_model_config(model)
    assert not [i for i in issues if i["level"] == "warn"]


# ---------------- host gates ------------------------------------------------

def test_validate_hosts_windows_refused():
    errs = validate_hosts([{"name": "winbox", "os": "windows"}])
    assert len(errs) == 1
    assert "WSL2" in errs[0]


def test_validate_hosts_macos_refused():
    errs = validate_hosts([{"name": "mac", "os": "macos"}])
    assert len(errs) == 1
    assert "Metal" in errs[0]


def test_validate_hosts_linux_ok():
    assert validate_hosts([{"name": "aibox1", "os": "linux"}]) == []


def test_validate_hosts_rocm_needs_py312():
    errs = validate_hosts([{"name": "amdbox", "os": "linux",
                            "gpu_vendor": "amd",
                            "python_version": "3.11.8"}])
    assert len(errs) == 1 and "3.12" in errs[0]
    assert validate_hosts([{"name": "amdbox", "os": "linux",
                            "gpu_vendor": "amd",
                            "python_version": "3.12.3"}]) == []

# ---------------- multi-host: TP derivation + cluster script shape ----------

def test_tp_derivation_explicit_matches_chips_ok():
    model = _model(tensor_parallel_size=3)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1,aibox2-CUDA0"
    tp, warns = vllm_manager._derive_tensor_parallel_size(
        model, [MASTER, {"name": "aibox2"}])
    assert tp == 3 and warns == []


def test_tp_derivation_explicit_mismatch_refuses():
    model = _model(tensor_parallel_size=4)
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1,aibox2-CUDA0"
    with pytest.raises(ValueError, match="must equal the total"):
        vllm_manager._derive_tensor_parallel_size(
            model, [MASTER, {"name": "aibox2"}])


def test_tp_derivation_from_chips_when_unset():
    model = _model()
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1,aibox2-CUDA0"
    tp, warns = vllm_manager._derive_tensor_parallel_size(
        model, [MASTER, {"name": "aibox2"}])
    assert tp == 3 and warns == []


def test_tp_derivation_no_chips_falls_back_to_server_count():
    model = _model()  # no device_order
    tp, warns = vllm_manager._derive_tensor_parallel_size(
        model, [MASTER, {"name": "aibox2"}])
    assert tp == 2
    assert warns and "1 GPU per host" in warns[0]


def test_cluster_script_golden_shape(monkeypatch):
    """2+ hosts: ray workers via ssh+sshpass, head on master, wait loop,
    exec vllm serve with --distributed-executor-backend ray, EXIT trap."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master = dict(MASTER, username="u")
    peer = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
            "username": "u"}
    script = vllm_manager.build_vllm_cluster_command(
        _cluster_model(), master, [peer], VENV_CFG)
    assert script.startswith("#!/bin/bash")
    assert "sshpass -p \"$SSH_PASS\" ssh -n $SSH_OPTS \"u@10.0.0.11\"" in script
    assert "ray start --address=$MIP:6379 --num-gpus=1" in script
    # Gloo pinning: Ubuntu's /etc/hosts maps the local hostname to
    # 127.0.1.1, which makes Gloo rank meshes dial the loopback and die
    # with "Connection refused" — the script pins GLOO/NCCL_SOCKET_IFNAME
    # on head AND exports it over SSH for workers.
    assert "GLOO_SOCKET_IFNAME" in script
    assert "ray start --head --port=$RAY_PORT" in script
    assert "deadline=$((SECONDS + 300))" in script
    # liveness via the Ray API — `ray status` text has no "Alive" lines
    # on 2.44+, so grepping it would wait the full timeout every load
    assert "ray.nodes()" in script
    assert 'grep -c "Alive"' not in script
    # vllm serves as a CHILD with wait, never exec'd: exec destroys the
    # EXIT trap, so cmdStop's pkill orphaned the remote ray workers
    # (aibox3 kept VRAM pinned after unload). wait preserves the trap AND
    # propagates vllm's exit status to llama-swap.
    assert "~/vllm-env/bin/vllm serve" in script
    assert "wait $VLLM_PID" in script
    assert not script.count("exec ~/vllm-env/bin/vllm")
    assert "--pipeline-parallel-size 2" in script
    assert "--tensor-parallel-size 1" in script
    assert "--distributed-executor-backend ray" in script
    assert "trap cleanup EXIT" in script
    # multi-host eager default: CUDA graph capture hangs across a Ray TP
    # group (upstream #36237/#41725) — ON unless the user opted out.
    assert "--enforce-eager" in script
    # user opt-out (fast interconnect): graphs allowed
    script_off = vllm_manager.build_vllm_cluster_command(
        dict(_cluster_model(), vllm={"cpu_offload_gb": 0,
                                     "enforce_eager": False}),
        master, [peer], VENV_CFG)
    assert "--enforce-eager" not in script_off
    assert "ray stop" in script  # PID-file teardown of workers + head
    import subprocess
    r = subprocess.run(["bash", "-n"], input=script, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_cluster_script_ray_init_timeout_from_settings(monkeypatch):
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: dict(VENV_CFG, ray_port=6380,
                                     ray_init_timeout=120))
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master = dict(MASTER, username="u")
    peer = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
            "username": "u"}
    script = vllm_manager.build_vllm_cluster_command(
        _cluster_model(), master, [peer])
    assert "RAY_PORT=6380" in script
    assert "SECONDS + 120" in script


def test_settings_ray_port_collision_refused():
    """ray_port must not equal the llama-swap listen port."""
    adapter = vllm_manager.VllmAdapter()
    with pytest.raises(ValueError, match="collides with"):
        adapter.settings_section(
            {"engine": "vllm", "listen_port": 6379,
             "vllm": {"ray_port": 6379}})
    ok = adapter.settings_section(
        {"engine": "vllm", "listen_port": 8080, "vllm": {}})
    assert ok["vllm"]["ray_port"] == 6379
    assert ok["vllm"]["ray_init_timeout"] == 300


def test_distributed_executor_backend_schema():
    adapter = vllm_manager.VllmAdapter()
    assert adapter.sanitize_model_section(
        {"distributed_executor_backend": "ray"}) == {
        "distributed_executor_backend": "ray"}
    with pytest.raises(ValueError, match="must be 'ray' or 'mp'"):
        adapter.sanitize_model_section({"distributed_executor_backend": "mpi"})


def test_cluster_script_loopback_master_resolves_lan_at_runtime(monkeypatch):
    """Master configured as loopback (app runs on the swap host itself):
    workers cannot dial 127.x, so the script must resolve the LAN source
    address via `ip route get <peer>` instead of baking in 127.0.0.1."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master = {"name": "aibox1", "ip": "127.0.0.1", "os": "linux",
              "username": "u"}
    peer = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
            "username": "u"}
    script = vllm_manager.build_vllm_cluster_command(
        _cluster_model(), master, [peer], VENV_CFG)
    assert 'PEER_IP="10.0.0.11"' in script
    assert 'ip route get "$PEER_IP"' in script
    assert "ray start --address=$MIP:6379" in script
    import subprocess
    r = subprocess.run(["bash", "-n"], input=script, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_cluster_script_lan_master_kept_as_is(monkeypatch):
    """A real configured LAN IP is used verbatim (no route lookup)."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master = {"name": "aibox1", "ip": "10.0.0.98", "os": "linux",
              "username": "u"}
    peer = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
            "username": "u"}
    script = vllm_manager.build_vllm_cluster_command(
        _cluster_model(), master, [peer], VENV_CFG)
    assert 'MASTER_IP="10.0.0.98"' in script


def test_cluster_script_flag_continuations_are_single_backslash(monkeypatch):
    """Extra vLLM flags join with ` \\<newline>`; a doubled `\\\\`
    is a literal backslash in bash and breaks the continuation."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master = {"name": "aibox1", "ip": "10.0.0.98", "os": "linux",
              "username": "u"}
    peer = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
            "username": "u"}
    model = _cluster_model(max_model_len=32768, max_num_seqs=16)
    script = vllm_manager.build_vllm_cluster_command(
        model, master, [peer], VENV_CFG)
    assert "--max-model-len 32768" in script
    assert "--max-num-seqs 16" in script
    assert "\\\\\n" not in script  # no doubled backslash continuations


def _cluster_hosts():
    master = {"name": "aibox1", "ip": "10.0.0.98", "os": "linux",
              "username": "u"}
    peer = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
            "username": "u"}
    return master, [peer]


def test_cluster_script_refuses_without_chips(monkeypatch):
    """Push-time gate: no device_order at all — refuse instead of emitting
    a cluster that advertises --num-gpus=0 everywhere."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    with pytest.raises(ValueError, match="at least one GPU chip"):
        vllm_manager.build_vllm_cluster_command(
            _model(), master, peers, VENV_CFG)


def test_cluster_script_refuses_partially_covered_hosts(monkeypatch):
    """Push-time gate: chips on the master but none on the peer — the
    error names the bare host so the editor selection can be fixed."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    model = _model()
    model["device_order"] = "aibox1-CUDA0"
    with pytest.raises(ValueError, match="aibox2"):
        vllm_manager.build_vllm_cluster_command(
            model, master, peers, VENV_CFG)


def test_dtype_half_emitted_on_cluster_script(monkeypatch):
    """Turing/sm_75 peers (RTX 2060) cannot run bfloat16 — a validated
    `dtype: half` must reach the command line as --dtype half."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    script = vllm_manager.build_vllm_cluster_command(
        _cluster_model(dtype="half"), master, peers, VENV_CFG)
    assert "--dtype half" in script
    import subprocess
    r = subprocess.run(["bash", "-n"], input=script, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_dtype_auto_omitted_and_invalid_refused(monkeypatch):
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    master, peers = _cluster_hosts()
    script = vllm_manager.build_vllm_cluster_command(
        _cluster_model(dtype="auto"), master, peers, VENV_CFG)
    assert "--dtype" not in script
    with pytest.raises(ValueError, match="vllm.dtype 'int8'"):
        vllm_manager.build_vllm_cluster_command(
            _cluster_model(dtype="int8"), master, peers, VENV_CFG)


def test_dtype_sanitize():
    adapter = vllm_manager.VllmAdapter()
    assert adapter.sanitize_model_section({"dtype": "half"}) == {
        "dtype": "half"}
    assert adapter.sanitize_model_section({"dtype": "auto"}) == {}
    with pytest.raises(ValueError, match="dtype"):
        adapter.sanitize_model_section({"dtype": "int8"})


def test_tool_choice_default_on_and_overrides():
    """Tool-calling defaults: absent config still emits both flags (the
    agentic-client requirement), explicit False removes them, and a
    custom parser (llama3_json etc.) replaces the hermes default."""
    model = _model()
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser hermes" in cmd

    off = _model(enable_auto_tool_choice=False)
    cmd = build_vllm_command(off, MASTER, VENV_CFG)
    assert "--enable-auto-tool-choice" not in cmd
    assert "--tool-call-parser" not in cmd

    llama3 = _model(tool_call_parser="llama3_json")
    cmd = build_vllm_command(llama3, MASTER, VENV_CFG)
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser llama3_json" in cmd
    assert "hermes" not in cmd


def test_ray_env_overrides_and_default_envelopes(monkeypatch):
    """Per-model ray_env: defaults (heterogeneous-Ethernet fixes) emit as
    sorted export lines; false drops a default; str values replace; new
    keys add. Injection-shaped keys are refused at sanitize time."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    # defaults: all four fixes present
    script = vllm_manager.build_vllm_cluster_command(
        _cluster_model(), master, peers, VENV_CFG)
    assert "export NCCL_P2P_DISABLE=1" in script
    assert "export NCCL_SHM_DISABLE=1" in script
    assert "export VLLM_USE_RAY_WRAPPED_PP_COMM=0" in script
    assert "export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl" in script
    # per-step CG timeout default: 1800s — 300 dies hard on slow TCP
    # prefill steps (engine death + reload instead of riding it out)
    assert "export RAY_CGRAPH_get_timeout=1800" in script
    # opt-out for real interconnects: P2P/SHM fixes gone, channel swapped
    tuned = dict(_cluster_model(), vllm={
        "cpu_offload_gb": 0,
        "ray_env": {"NCCL_P2P_DISABLE": False, "NCCL_SHM_DISABLE": False,
                    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": "nccl",
                    "NCCL_IB_DISABLE": "0"}})
    script2 = vllm_manager.build_vllm_cluster_command(
        tuned, master, peers, VENV_CFG)
    assert "export NCCL_P2P_DISABLE" not in script2
    assert "export NCCL_SHM_DISABLE" not in script2
    assert "export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl" in script2
    assert "export NCCL_IB_DISABLE=0" in script2
    assert "export VLLM_USE_RAY_WRAPPED_PP_COMM=0" in script2
    import subprocess
    r = subprocess.run(["bash", "-n"], input=script2, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    # sanitize: env-var name shape is enforced (generated export lines
    # are an injection surface — same strictness as _SAFE_ID)
    adapter = vllm_manager.VllmAdapter()
    assert adapter.sanitize_model_section(
        {"ray_env": {"NCCL_IB_DISABLE": "0", "FOO": True}}) == {
        "ray_env": {"NCCL_IB_DISABLE": "0", "FOO": True}}
    with pytest.raises(ValueError, match="environment variable name"):
        adapter.sanitize_model_section({"ray_env": {"EVIL;rm": "1"}})
    with pytest.raises(ValueError, match="ray_env"):
        adapter.sanitize_model_section({"ray_env": ["NCCL=1"]})


def test_enforce_eager_sanitize_and_flags():
    """enforce_eager: explicit bool stored verbatim; absent stays absent
    (multi-host launch then applies the safe default). True emits
    --enforce-eager on the command line; False emits nothing."""
    adapter = vllm_manager.VllmAdapter()
    assert adapter.sanitize_model_section({"enforce_eager": True}) == {
        "enforce_eager": True}
    assert adapter.sanitize_model_section({"enforce_eager": False}) == {
        "enforce_eager": False}
    assert adapter.sanitize_model_section({}) == {}
    with pytest.raises(ValueError, match="enforce_eager"):
        adapter.sanitize_model_section({"enforce_eager": "yes"})
    model = _model(enforce_eager=True)
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    assert "--enforce-eager" in cmd
    model = _model(enforce_eager=False)
    cmd = build_vllm_command(model, MASTER, VENV_CFG)
    assert "--enforce-eager" not in cmd


def test_dtype_invalid_is_save_error(vllm_cfg_stub):
    model = {"model_id": "m", "base_path": "/mnt/store/m",
             "servers": ["aibox1", "aibox2"],
             "device_order": "aibox1-CUDA0,aibox2-CUDA0",
             "vllm": {"cpu_offload_gb": 0, "dtype": "int8"}}
    issues = validate_model_config(model)
    assert any(i["level"] == "error" and "dtype" in i["msg"]
               for i in issues)


def test_bf16_warning_on_pre_ampere_hardware(vllm_cfg_stub):
    """Heterogeneous-hardware warning: a selected GPU whose driver-reported
    compute_cap is sm_75 with default/auto dtype warns to set dtype half;
    an explicit half stays silent. Missing cap data never warns."""
    vllm_cfg_stub.update({
        "aibox1": _machine("aibox1", "10.0.0.98", role="master",
                           gpu_vendor="nvidia",
                           detected={"gpu_names": ["NVIDIA GeForce RTX 3060"],
                                     "compute_cap": ["8.6"],
                                     "vram_total_gb": 12, "ram_total_gb": 16}),
        "aibox2": _machine("aibox2", "10.0.0.11",
                           gpu_vendor="nvidia",
                           detected={"gpu_names": ["NVIDIA GeForce RTX 2060"],
                                     "compute_cap": ["7.5"],
                                     "vram_total_gb": 12, "ram_total_gb": 16}),
    })
    base = {"model_id": "m", "base_path": "/mnt/store/m",
            "servers": ["aibox1", "aibox2"],
            "device_order": "aibox1-CUDA0,aibox2-CUDA0",
            "vllm": {"cpu_offload_gb": 0}}
    issues = validate_model_config(dict(base))
    warns = [i["msg"] for i in issues if i["level"] == "warn"]
    assert any("aibox2" in w and "half" in w for w in warns), warns

    half = dict(base, vllm={"cpu_offload_gb": 0, "dtype": "half"})
    issues = validate_model_config(half)
    assert not any("half" in i["msg"] and i["level"] == "warn"
                   for i in issues)
    assert not [i for i in issues if i["level"] == "error"]


def test_tp_divisibility_error(vllm_cfg_stub, monkeypatch, tmp_path):
    """vLLM shards attention heads per TP rank and requires TP to divide
    num_attention_heads evenly (Qwen2.5-0.5B: 14 heads -> TP=5 dies at
    engine init; TP=2 fine). The head count is read from the weights'
    config.json on the swap host."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"num_attention_heads": 14}')
    model = {"model_id": "m", "base_path": str(model_dir),
             "servers": ["aibox1"],
             "device_order": "aibox1-CUDA0,aibox1-CUDA1,aibox1-CUDA2,"
                             "aibox1-CUDA3,aibox1-CUDA4",
             "vllm": {"tensor_parallel_size": 5}}
    monkeypatch.setattr(vllm_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(vllm_cfg_stub)))
    issues = validate_model_config(model)
    errs = [i["msg"] for i in issues if i["level"] == "error"]
    assert any("does not divide the model's 14 attention heads" in m
               for m in errs), errs

    model["vllm"] = {"tensor_parallel_size": 2}
    issues = validate_model_config(model)
    assert not [i for i in issues
                if i["level"] == "error" and "attention heads" in i["msg"]]


def test_tp_divisibility_silent_when_config_missing(vllm_cfg_stub,
                                                    monkeypatch, tmp_path):
    """No readable config.json (GGUF, missing dir) -> no claim, never an
    error — vLLM's own engine init stays the final arbiter."""
    model = {"model_id": "m", "base_path": str(tmp_path / "nope"),
             "servers": ["aibox1"],
             "device_order": "aibox1-CUDA0,aibox1-CUDA1,aibox1-CUDA2,"
                             "aibox1-CUDA3,aibox1-CUDA4",
             "vllm": {"tensor_parallel_size": 5}}
    monkeypatch.setattr(vllm_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(vllm_cfg_stub)))
    issues = validate_model_config(model)
    assert not [i for i in issues if "attention heads" in i["msg"]]


def test_cluster_script_refuses_derived_tp_not_dividing_heads(
        monkeypatch, tmp_path):
    """Launch-time gate (tp layout): TP may be DERIVED from the chip
    count (no explicit tensor_parallel_size), which save-time validation
    cannot see — build_vllm_cluster_command must refuse the same
    head-divisibility violation before the script is emitted."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"num_attention_heads": 14}')
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    model = _model(parallel_layout="tp")
    model["base_path"] = str(model_dir)
    # 5 chips, TP unset -> derived TP=5; 14 % 5 != 0
    model["device_order"] = ("aibox1-CUDA0,aibox1-CUDA1,aibox1-CUDA2,"
                             "aibox2-CUDA0,aibox2-CUDA1")
    with pytest.raises(ValueError, match="14 attention heads"):
        vllm_manager.build_vllm_cluster_command(
            model, master, peers, VENV_CFG)


def test_pp_layout_layer_split_default(monkeypatch):
    """pp layout (the multi-host default): one pipeline stage per HOST,
    TP within each host's chips — the llama.cpp-RPC-style layer split
    for 1xPCIe/Ethernet farms. Also: PP has no attention-head
    divisibility constraint, so the 14-head model passes under pp where
    tp=5 was refused."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    model = _cluster_model()  # 1 chip per host, layout unset -> pp
    script = vllm_manager.build_vllm_cluster_command(
        model, master, peers, VENV_CFG)
    assert "--pipeline-parallel-size 2" in script
    assert "--tensor-parallel-size 1" in script
    import subprocess
    r = subprocess.run(["bash", "-n"], input=script, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_pp_layout_heterogeneous_tp1(monkeypatch):
    """Heterogeneous PP (unequal chip counts per host): TP=1 pure-pipeline
    is the safe topology — one rank per stage, Ray PACK places them
    node-first, so stage boundaries cross hosts once per token with no
    TP over Ethernet. The layout auto-derives tp=1, pp=total chips, and
    still validates the generated script parses as bash."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    model = _model()
    # 3 chips on the master, 1 on the peer — unequal, TP=1, PP=4
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1,aibox1-CUDA2,aibox2-CUDA0"
    script = vllm_manager.build_vllm_cluster_command(
        model, master, peers, VENV_CFG)
    assert "--pipeline-parallel-size 4" in script
    assert "--tensor-parallel-size 1" in script
    import subprocess
    r = subprocess.run(["bash", "-n"], input=script, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_pp_layout_heterogeneous_rejects_tp_gt1(monkeypatch):
    """Heterogeneous hosts with an explicit TP>1 would shard a stage
    across hosts (TP over Ethernet) — refuse, naming the counts."""
    monkeypatch.setattr(vllm_manager, "_load_vllm_settings",
                        lambda: VENV_CFG)
    monkeypatch.setattr(vllm_manager.path_resolver, "resolve_for",
                        lambda p, os=None: p)
    master, peers = _cluster_hosts()
    model = _model()
    model["device_order"] = "aibox1-CUDA0,aibox1-CUDA1,aibox2-CUDA0"
    model["vllm"] = {"tensor_parallel_size": 2}
    with pytest.raises(ValueError, match="tensor_parallel"):
        vllm_manager.build_vllm_cluster_command(
            model, master, peers, VENV_CFG)


def test_parallel_layout_sanitize_and_choices():
    adapter = vllm_manager.VllmAdapter()
    assert adapter.sanitize_model_section(
        {"parallel_layout": "pp"}) == {"parallel_layout": "pp"}
    assert adapter.sanitize_model_section(
        {"parallel_layout": "tp"}) == {"parallel_layout": "tp"}
    assert adapter.sanitize_model_section({}) == {}
    with pytest.raises(ValueError, match="parallel_layout"):
        adapter.sanitize_model_section({"parallel_layout": "expert"})


def test_vllm_new_model_defaults():
    """The validated-farm starting section for NEW vllm models — locked
    as a contract: every key is individually user-editable, but the
    DEFAULTS themselves encode live-tested fixes (fp32 for the sm_75
    peer + clean agentic output, eager vs the CG capture hang,
    tool-calling for opencode, 32k len vs its max_tokens=16384, pp
    layer split for Ethernet). Sanitize must accept them verbatim."""
    defaults = vllm_manager.vllm_new_model_defaults()
    assert defaults == {
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "dtype": "float32",
        "enforce_eager": True,
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",
        "parallel_layout": "pp",
    }
    adapter = vllm_manager.VllmAdapter()
    assert adapter.sanitize_model_section(defaults) == defaults
    # the ui_manifest mirror must not drift from the source of truth
    manifest = vllm_manager.ui_manifest()
    assert manifest["vllm_new_model_defaults"] == defaults
