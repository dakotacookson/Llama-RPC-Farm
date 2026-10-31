"""Regression guard for the confirmed pytest exit-code-1 defect.

Each module listed in relevantSourcePaths must be importable without
raising ImportError or NameError when heavy optional deps (paramiko)
are stubbed, matching the conftest.py stub pattern already in use.

If a module cannot be imported this test fails immediately with a
precise error, replacing the opaque suite-level exit code 1 that
triggered the original finding.

No network I/O, no filesystem side-effects.
"""
import importlib
import sys
import types

import pytest

# Modules that conftest.py already stubs; extend here if new optional
# heavy deps are introduced.
_STUBBED_BEFORE_IMPORT = ["paramiko"]

# All source modules from relevantSourcePaths that are safe to import
# in a unit-test environment (no running server, no real filesystem).
# Modules that unconditionally start background threads or open sockets
# at import time are excluded with a comment.
_IMPORTABLE_MODULES = [
    "config_loader",
    "discovery",
    "gpu_vendor",
    "local_node",
    "model_parser",
    "path_resolver",
    "rpc_manager",
    "ssh_manager",
    "stats_collector",
    # app.py starts uvicorn at import time in some configurations — skip.
    # hf_browse / hf_downloader / llamacpp_installer / tuner / steps /
    # swap_manager / gguf_inspect may require heavy optional deps beyond
    # paramiko; they are listed as CANDIDATE below.
]

_CANDIDATE_MODULES = [
    "gguf_inspect",
    "hf_browse",
    "hf_downloader",
    "llamacpp_installer",
    "steps",
    "swap_manager",
    "tuner",
]


def _ensure_stubs():
    """Install lightweight stubs for known optional heavy deps so imports
    don't fail on missing wheels in the test environment."""
    for mod in _STUBBED_BEFORE_IMPORT:
        if mod not in sys.modules:
            try:
                importlib.import_module(mod)
            except ImportError:
                stub = types.ModuleType(mod)
                # Minimal surface used by ssh_manager / rpc_manager
                stub.SSHException = type("SSHException", (Exception,), {})
                stub.AutoAddPolicy = type("AutoAddPolicy", (), {})
                stub.SSHClient = type("SSHClient", (), {})
                sys.modules[mod] = stub


_ensure_stubs()


@pytest.mark.parametrize("module_name", _IMPORTABLE_MODULES)
def test_core_module_importable(module_name):
    """Core src/ module must import cleanly with stubs in place."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(
            f"ImportError importing '{module_name}': {exc}\n"
            "This is the regression guard for the exit-code-1 defect. "
            "Check that all required packages are present in the venv."
        )
    except Exception as exc:
        pytest.fail(
            f"Unexpected error importing '{module_name}': {type(exc).__name__}: {exc}"
        )
    assert mod is not None


@pytest.mark.parametrize("module_name", _CANDIDATE_MODULES)
def test_candidate_module_importable(module_name):
    """Optional/heavy src/ modules: skip if a known missing dep causes
    ImportError, but fail on NameError or unexpected runtime errors so
    structural breakage is still caught."""
    try:
        mod = importlib.import_module(module_name)
        assert mod is not None
    except ImportError as exc:
        pytest.skip(f"Optional dep missing for '{module_name}': {exc}")
    except Exception as exc:
        pytest.fail(
            f"Non-ImportError while importing '{module_name}': "
            f"{type(exc).__name__}: {exc}"
        )
