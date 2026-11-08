"""Windows NVIDIA driver fallback: old drivers use Vulkan/CPU, not CUDA."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import llamacpp_installer as li  # noqa: E402


def _machine():
    return {"os": "windows", "gpu_vendor": "nvidia",
            "detected": {"arch": "x86_64"}}


def _assets(tag="b1"):
    return [
        f"llama-{tag}-bin-win-cuda-12.4-x64.zip",
        f"cudart-llama-bin-win-cuda-12.4-x64.zip",
        f"llama-{tag}-bin-win-vulkan-x64.zip",
        f"llama-{tag}-bin-win-cpu-x64.zip",
    ]


def test_old_driver_picks_vulkan():
    names, dest, note, is_zip = li._pick_assets(
        _machine(), "b1", _assets(), driver_version=(551, 60))
    assert names == ["llama-b1-bin-win-vulkan-x64.zip"]
    assert "Vulkan" in note and is_zip is True


def test_minimum_driver_picks_cuda():
    names, _dest, note, _zip = li._pick_assets(
        _machine(), "b1", _assets(), driver_version=(551, 61))
    assert names[0] == "llama-b1-bin-win-cuda-12.4-x64.zip"
    assert names[1].startswith("cudart-llama-bin-win-cuda-12.4")
    assert "CUDA 12.4" in note


def test_old_driver_without_vulkan_asset_falls_back_cpu():
    assets = [a for a in _assets() if "vulkan" not in a]
    names, _dest, note, _zip = li._pick_assets(
        _machine(), "b1", assets, driver_version=(500, 1))
    assert names == ["llama-b1-bin-win-cpu-x64.zip"]
    assert "CPU" in note


def test_unknown_driver_keeps_cuda():
    names, _dest, _note, _zip = li._pick_assets(
        _machine(), "b1", _assets(), driver_version=None)
    assert names[0] == "llama-b1-bin-win-cuda-12.4-x64.zip"
