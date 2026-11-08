"""Backend asset selection for every supported GPU vendor/OS.

These are hermetic installer tests (no GPUs, no downloads): they pin the
asset names the app would fetch for NVIDIA CUDA, AMD ROCm, Intel SYCL,
Vulkan fallback, macOS Metal, and CPU.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import llamacpp_installer as li  # noqa: E402
import swap_installer as si      # noqa: E402

TAG = "b12211"


def _assets():
    a = []
    for cud in ("13.4", "12.8"):
        a += [f"llama-{TAG}-bin-ubuntu-cuda-{cud}-x64.tar.gz",
              f"cudart-llama-{TAG}-bin-ubuntu-cuda-{cud}-x64.tar.gz"]
    a += [f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz",
          f"llama-{TAG}-bin-ubuntu-rocm-10.0-x64.tar.gz",
          f"llama-{TAG}-bin-ubuntu-sycl-fp16-x64.tar.gz",
          f"llama-{TAG}-bin-ubuntu-x64.tar.gz",
          f"llama-{TAG}-bin-win-cuda-12.4-x64.zip",
          f"cudart-llama-bin-win-cuda-12.4-x64.zip",
          f"llama-{TAG}-bin-win-vulkan-x64.zip",
          f"llama-{TAG}-bin-win-rocm-10.0-x64.zip",
          f"llama-{TAG}-bin-win-sycl-x64.zip",
          f"llama-{TAG}-bin-win-cpu-x64.zip",
          f"llama-{TAG}-bin-macos-arm64.tar.gz",
          f"llama-{TAG}-bin-macos-x64.tar.gz"]
    return a


def _m(os_name, vendor, arch="x86_64"):
    return {"os": os_name, "gpu_vendor": vendor,
            "detected": {"arch": arch}}


def _pick(machine, glibc=(2, 39), driver=None):
    return li._pick_assets(machine, TAG, _assets(), glibc=glibc,
                           driver_version=driver)


def test_linux_nvidia_new_glibc_picks_cuda_with_cudart():
    names, _dest, note, _zip = _pick(_m("linux", "nvidia"))
    assert names[0].endswith("ubuntu-cuda-13.4-x64.tar.gz")
    assert names[1].startswith("cudart-llama-")
    assert "CUDA 13.4" in note


def test_linux_nvidia_old_glibc_picks_vulkan():
    names, _dest, note, _zip = _pick(_m("linux", "nvidia"), glibc=(2, 35))
    assert names == [f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"]
    assert "Vulkan" in note


def test_linux_nvidia_cuda_asset_missing_falls_to_vulkan():
    assets = [a for a in _assets() if "-cuda-" not in a and "cudart" not in a]
    names, _dest, note, _zip = li._pick_assets(
        _m("linux", "nvidia"), TAG, assets, glibc=(2, 39))
    assert names == [f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"]


def test_linux_amd_picks_rocm():
    names, _d, note, _z = _pick(_m("linux", "amd"))
    assert names == [f"llama-{TAG}-bin-ubuntu-rocm-10.0-x64.tar.gz"]
    assert "ROCm" in note


def test_linux_intel_picks_sycl():
    names, _d, note, _z = _pick(_m("linux", "intel"))
    assert names == [f"llama-{TAG}-bin-ubuntu-sycl-fp16-x64.tar.gz"]
    assert "SYCL" in note


def test_linux_cpu_picks_cpu():
    names, _d, note, _z = _pick(_m("linux", "none"))
    assert names == [f"llama-{TAG}-bin-ubuntu-x64.tar.gz"]
    assert "CPU" in note


def test_windows_amd_intel_cpu():
    assert _pick(_m("windows", "amd"))[0][0].endswith(
        "win-rocm-10.0-x64.zip")
    assert _pick(_m("windows", "intel"))[0][0].endswith(
        "win-sycl-x64.zip")
    assert _pick(_m("windows", "none"))[0][0].endswith(
        "win-cpu-x64.zip")


def test_macos_metal_archs():
    assert _pick(_m("macos", "apple", "arm64"))[0][0].endswith(
        "bin-macos-arm64.tar.gz")
    assert _pick(_m("macos", "intel", "x86_64"))[0][0].endswith(
        "bin-macos-x64.tar.gz")


@pytest.mark.parametrize("os_name,arch,kind", [
    ("linux", "x86_64", "amd64"),
    ("linux", "arm64", "arm64"),
    ("macos", "x86_64", "amd64"),
    ("macos", "arm64", "arm64"),
    ("windows", "x86_64", "amd64"),
])
def test_swap_asset_names(os_name, arch, kind):
    tag = "v271"
    names = [
        f"llama-swap_271_{'darwin' if os_name == 'macos' else os_name}_{kind}"
        f"{'.zip' if os_name == 'windows' else '.tar.gz'}"
    ]
    name, _dest, ver = si._pick_asset(_m(os_name, "none", arch), tag, names)
    assert name == names[0]
    assert ver == "271"
