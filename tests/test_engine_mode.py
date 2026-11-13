"""Per-machine llama.cpp install strategy: auto | prebuilt | source."""
import pytest

import config_loader as cfg
import llamacpp_installer as li
from routes.machines import EngineModeIn, MachineIn


def test_default_is_auto():
    assert cfg.MACHINE_DEFAULTS["engine_mode"] == "auto"


def test_machine_in_accepts_modes():
    assert MachineIn(name="m", ip="1.2.3.4", engine_mode="prebuilt").engine_mode \
        == "prebuilt"
    assert MachineIn(name="m", ip="1.2.3.4", engine_mode="SOURCE").engine_mode \
        == "source"
    # omitted stays None so an edit doesn't reset a saved choice
    assert MachineIn(name="m", ip="1.2.3.4").engine_mode is None


def test_machine_in_rejects_bad_mode():
    with pytest.raises(Exception):
        MachineIn(name="m", ip="1.2.3.4", engine_mode="turbo")


def test_source_build_is_target_scoped():
    assert "llama-server" in li.SOURCE_BUILD_TARGETS
    assert "ggml-rpc-server" in li.SOURCE_BUILD_TARGETS
    # tests/examples must not be in the hot rebuild set
    assert not any(t.startswith("test-") for t in li.SOURCE_BUILD_TARGETS)
