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


def test_choose_release_skips_uploading_release():
    """The newest release can briefly contain only cudart assets while
    upstream is still uploading. It must be skipped for a release that
    actually has the machine's binary."""
    releases = [
        ("b12368", ["cudart-llama-b12368-bin-ubuntu-cuda-13.5-x64.tar.gz"],
         "base68"),
        ("b12367", [f"llama-b12367-bin-ubuntu-x64.tar.gz"], "base67"),
    ]
    sel = li._choose_release_for(machine(vendor="cpu"), releases)
    assert sel["tag"] == "b12367"
    assert sel["assets"] == ["llama-b12367-bin-ubuntu-x64.tar.gz"]
    assert sel["base"] == "base67"
    assert sel["missing"] == []


def test_choose_release_skips_release_missing_this_asset():
    """Asset rotation: a release can have CUDA but no CPU generic; a CPU
    host must fall back to the newest release that has its asset."""
    releases = [
        ("b2", [f"llama-b2-bin-ubuntu-cuda-13.5-x64.tar.gz"], "base2"),
        ("b1", [f"llama-b1-bin-ubuntu-x64.tar.gz"], "base1"),
    ]
    sel = li._choose_release_for(machine(vendor="cpu"), releases)
    assert sel["tag"] == "b1"
