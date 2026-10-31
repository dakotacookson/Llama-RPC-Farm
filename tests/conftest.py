"""Pytest config: make `src/` importable and stub optional heavy deps.

Most tests target pure logic (port allocation, command building, device
mapping, script generation) and never touch the network. A few stub the
SSH layer to assert port-targeted kill/reuse behavior without a real host.
"""
import os
import sys
import types

# put src/ on the path so `import rpc_manager` works from the tests dir
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# paramiko / openai / chainlit are optional for the unit tests; the modules
# under test import paramiko at top of ssh_manager. Stub it so importing
# rpc_manager doesn't fail when paramiko isn't installed in the test env.
for _mod in ("paramiko",):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            import types
            stub = types.ModuleType(_mod)
            stub.SSHException = type("SSHException", (Exception,), {})
            stub.AutoAddPolicy = type("AutoAddPolicy", (), {})
            stub.SSHClient = type("SSHClient", (), {})
            sys.modules[_mod] = stub

import pytest


class _FakeMachines:
    """Resolves machine names to dicts from an in-memory map, mimicking
    config_loader's machines.require/get/all. Lets tests build peer
    lists by name without the real config files."""
    def __init__(self, by_name):
        self._by = by_name
    def require(self, name):
        if name not in self._by:
            raise ValueError(f"machine '{name}' not found")
        return self._by[name]
    def get(self, name):
        return self._by.get(name)
    def all(self):
        return list(self._by.values())


@pytest.fixture
def stub_cfg(monkeypatch):
    """Patch rpc_manager.cfg.machines with an in-memory map. Returns a
    dict you populate with machine-name -> machine-dict before calling
    code that resolves machines by name."""
    import rpc_manager
    by_name = {}
    fake = types.SimpleNamespace(machines=_FakeMachines(by_name))
    monkeypatch.setattr(rpc_manager, "cfg", fake)
    return by_name


@pytest.fixture(autouse=True)
def _default_engine_llama_cpp(monkeypatch, tmp_path):
    """Engine determinism: every test reads engine from an ISOLATED
    settings file defaulting to llama_cpp, not the farm's live
    config/settings.yaml.

    Before this fixture the engine leaked from the live settings — the
    suite silently changed behavior when the operator switched engines
    (vllm-era runs made llama_cpp-context tests delegate to the vllm
    adapter and vice versa). Tests that need engine=vllm write it into
    their own settings_file fixture (which monkeypatches SETTINGS_FILE
    AFTER this fixture, so the test's file wins). Tests that patch
    load_settings directly also win (their setattr runs later)."""
    import engine_registry
    default = tmp_path / "settings.yaml"
    default.write_text("engine: llama_cpp\n")
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", str(default))