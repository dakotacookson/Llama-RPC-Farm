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

    # Installed source stays source — there is no user switch.
    assert li.decide_engine_plan(
        master, probe={"kind": "source"}, glibc=(2, 35))["mode"] == "source"
    assert li.decide_engine_plan(
        master, probe={"kind": "source"}, glibc=(2, 39))["mode"] == "source"
    assert li.decide_engine_plan(
        aibox3, probe={"kind": "source"}, glibc=(2, 35))["mode"] == "source"
    assert li.decide_engine_plan(
        aibox2, probe={"kind": "source"}, glibc=(2, 43))["mode"] == "source"
    # No install: prebuilt only when the host can run it, else source.
    assert li.decide_engine_plan(
        master, probe={"kind": "missing"}, glibc=(2, 39))["mode"] == "prebuilt"
    assert li.decide_engine_plan(
        master, probe={"kind": "missing"}, glibc=(2, 35))["mode"] == "source"
    assert li.decide_engine_plan(
        aibox3, probe={"kind": "missing"}, glibc=(2, 35))["mode"] == "source"
    assert li.decide_engine_plan(
        aibox2, probe={"kind": "missing"}, glibc=(2, 43))["mode"] == "source"
    # Installed prebuilt stays prebuilt.
    assert li.prebuilt_blocker(hp) == ""
    assert li.decide_engine_plan(
        hp, probe={"kind": "prebuilt"}, glibc=(2, 35))["mode"] == "prebuilt"
    assert li.decide_engine_plan(
        win, probe={"kind": "prebuilt"})["mode"] == "prebuilt"
    assert li.decide_engine_plan(
        win, probe={"kind": "missing"})["mode"] == "prebuilt"


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


def test_source_build_args_follow_specs():
    import llamacpp_installer as li
    nv = {"gpu_vendor": "nvidia",
          "detected": {"compute_cap": ["8.6", "6.0", "7.5"]}}
    args = li.source_build_args(nv)
    assert "-DGGML_CUDA=ON" in args
    assert "-DGGML_CUDA_FA=ON" in args
    assert "-DGGML_RPC=ON" in args
    assert "-DCMAKE_CUDA_ARCHITECTURES=60;75;86" in args

    amd = {"gpu_vendor": "amd", "detected": {}}
    assert "-DGGML_HIP=ON" in li.source_build_args(amd)

    intel = {"gpu_vendor": "intel", "detected": {}}
    assert "-DGGML_SYCL=ON" in li.source_build_args(intel)

    cpu = {"gpu_vendor": "cpu", "detected": {}}
    cargs = li.source_build_args(cpu)
    assert "-DGGML_RPC=ON" in cargs
    assert not any(x in cargs for x in
                   ("-DGGML_CUDA=ON", "-DGGML_HIP=ON", "-DGGML_SYCL=ON"))
