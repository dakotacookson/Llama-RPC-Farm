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


def test_backend_pin_and_decision():
    import llamacpp_installer as li

    master = {"os": "linux", "gpu_vendor": "nvidia",
              "detected": {"arch": "x86_64",
                           "compute_cap": ["8.6", "8.6", "8.6"]}}
    aibox3 = {"os": "linux", "gpu_vendor": "nvidia",
              "detected": {"arch": "x86_64",
                           "compute_cap": ["7.5", "6.0", "7.5"]}}
    aibox2 = {"os": "linux", "gpu_vendor": "nvidia",
              "detected": {"arch": "x86_64", "compute_cap": ["6.1"]}}
    hp = {"os": "linux", "gpu_vendor": "cpu",
          "detected": {"arch": "x86_64", "compute_cap": []},
          "engine_backend": "vulkan"}
    win = {"os": "windows", "gpu_vendor": "nvidia",
           "detected": {"arch": "x86_64", "compute_cap": [None, None]}}

    # glibc 2.35 blocks the CUDA prebuilt on master; aibox3/Aibox2 lack SM
    assert li.decide_engine_plan(
        master, probe={"kind": "source"}, glibc=(2, 35))["mode"] == "source"
    assert li.decide_engine_plan(
        master, probe={"kind": "source"}, glibc=(2, 39))["mode"] == "prebuilt"
    assert li.decide_engine_plan(
        aibox3, probe={"kind": "source"}, glibc=(2, 35))["mode"] == "source"
    assert li.decide_engine_plan(
        aibox2, probe={"kind": "source"}, glibc=(2, 43))["mode"] == "source"
    # HP is pinned to Vulkan, so a prebuilt is viable (CPU asset not forced)
    assert li.prebuilt_blocker(hp) == ""
    assert li.decide_engine_plan(
        hp, probe={"kind": "prebuilt"}, glibc=(2, 35))["mode"] == "prebuilt"
    # Windows has no source path; prebuilt is the only option
    assert li.decide_engine_plan(
        win, probe={"kind": "prebuilt"})["mode"] == "prebuilt"


def test_pick_assets_respects_backend_pin():
    import llamacpp_installer as li
    tag = "b999"
    assets = {
        f"llama-{tag}-bin-ubuntu-x64.tar.gz",
        f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz",
        f"llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz",
        f"cudart-llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz",
    }
    hp = {"os": "linux", "gpu_vendor": "cpu",
          "detected": {"arch": "x86_64"}, "engine_backend": "vulkan"}
    names, _d, note, _z = li._pick_assets(hp, tag, assets, glibc=(2, 35))
    assert names == [f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"]
    assert "Vulkan" in note
    hp["engine_backend"] = "cpu"
    names, _d, note, _z = li._pick_assets(hp, tag, assets, glibc=(2, 35))
    assert names == [f"llama-{tag}-bin-ubuntu-x64.tar.gz"]
    assert "CPU" in note
