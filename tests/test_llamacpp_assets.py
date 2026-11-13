"""Backend selection must follow each machine's real specs, with sane
fallbacks when upstream rotates release assets."""
import llamacpp_installer as li

TAG = "b999"


def machine(os="linux", vendor="nvidia", arch="x86_64", names=None):
    return {"os": os, "gpu_vendor": vendor, "gpu": "",
            "detected": {"arch": arch, "gpu_names": names or []}}


def pick(m, assets, glibc=None, driver=None):
    names, _dest, note, _zip = li._pick_assets(m, TAG, set(assets), glibc,
                                               driver)
    return names, note


CUDA = [f"llama-{TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz",
        f"cudart-llama-{TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz"]
CUDA_OLD = [f"llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"]
VULKAN = f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"
GENERIC = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
ROCM = f"llama-{TAG}-bin-ubuntu-rocm-10.0-x64.tar.gz"
SYCL = f"llama-{TAG}-bin-ubuntu-sycl-fp16-x64.tar.gz"


def test_vendor_falls_back_to_detected_names():
    assert li._vendor_of({"gpu_vendor": "",
                          "detected": {"gpu_names": ["NVIDIA RTX 3080"]}}) \
        == "nvidia"
    assert li._vendor_of({"gpu_vendor": "",
                          "detected": {"gpu_names": ["AMD Radeon RX 7900"]}}) \
        == "amd"
    assert li._vendor_of({"gpu_vendor": "cpu",
                          "detected": {"gpu_names": []}}) == "cpu"


def test_nvidia_linux_prefers_cuda_on_new_glibc():
    names, note = pick(machine(), CUDA + [VULKAN, GENERIC], glibc=(2, 39))
    assert names == CUDA and "CUDA" in note


def test_nvidia_linux_old_glibc_uses_vulkan():
    names, note = pick(machine(), CUDA + [VULKAN, GENERIC], glibc=(2, 35))
    assert names == [VULKAN] and "Vulkan" in note


def test_nvidia_linux_falls_back_when_cuda_assets_absent():
    names, note = pick(machine(), [VULKAN, GENERIC], glibc=(2, 39))
    assert names == [VULKAN]


def test_amd_intel_cpu_macos_windows_matrix():
    assert pick(machine(vendor="amd"), [ROCM, VULKAN, GENERIC])[0] == [ROCM]
    assert pick(machine(vendor="intel"), [SYCL, GENERIC])[0] == [SYCL]
    assert pick(machine(vendor="cpu"), [GENERIC, VULKAN])[0] == [GENERIC]
    mac_arm = machine(os="macos", vendor="apple", arch="arm64")
    assert pick(mac_arm, [f"llama-{TAG}-bin-macos-arm64.tar.gz"])[0] \
        == [f"llama-{TAG}-bin-macos-arm64.tar.gz"]
    # Intel Mac with Apple vendor must still get x64 (regression)
    mac_intel = machine(os="macos", vendor="apple", arch="x86_64")
    assert pick(mac_intel, [f"llama-{TAG}-bin-macos-x64.tar.gz"])[0] \
        == [f"llama-{TAG}-bin-macos-x64.tar.gz"]
    win_cpu = machine(os="windows", vendor="cpu")
    assert pick(win_cpu, [f"llama-{TAG}-bin-win-cpu-x64.zip"])[0] \
        == [f"llama-{TAG}-bin-win-cpu-x64.zip"]


def test_missing_specific_asset_falls_back_to_generic():
    names, _ = pick(machine(vendor="amd"), [GENERIC])
    assert names == [GENERIC]


def test_device_summary_and_warning():
    assert li._device_summary("Available devices:\n  CUDA0: RTX 3080") \
        == "CUDA"
    assert li._device_summary("Available devices:\n  CPU: ...") == "CPU"
    assert li._device_summary("Available devices:\n  Vulkan0: Intel") \
        == "Vulkan"
    assert "expected CUDA" in li._device_warning(
        machine(vendor="nvidia"), "Vulkan0: Intel")
    assert "Vulkan" in li._device_warning(
        machine(vendor="cpu"), "Vulkan0: Intel")
    assert li._device_warning(machine(vendor="cpu"), "CPU") == ""
