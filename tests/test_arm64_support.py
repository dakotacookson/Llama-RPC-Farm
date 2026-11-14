"""Linux arm64 (Raspberry Pi) support: asset choice, glibc gate, agent
packaging, and board detection.

Bookworm ships glibc 2.36 and cannot load the upstream arm64 prebuilts
(they import GLIBC_2.38), so those hosts must build from source. Trixie
(2.41) and Ubuntu 24.04 (2.39) use the prebuilt.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

import llamacpp_installer as li  # noqa: E402

TAG = "b999"


def pi(os_name="linux", arch="aarch64", vendor="cpu"):
    return {"os": os_name, "gpu_vendor": vendor, "gpu": "",
            "detected": {"arch": arch, "gpu_names": []}}


def pick(m, assets, glibc=None):
    names, _dest, note, _zip = li._pick_assets(m, TAG, set(assets), glibc)
    return names, note


def test_arm64_linux_picks_arm64_assets():
    assets = [f"llama-{TAG}-bin-ubuntu-arm64.tar.gz",
              f"llama-{TAG}-bin-ubuntu-x64.tar.gz"]
    names, _ = pick(pi(), assets)
    assert names == [f"llama-{TAG}-bin-ubuntu-arm64.tar.gz"]
    vulkan = [f"llama-{TAG}-bin-ubuntu-vulkan-arm64.tar.gz"]
    names, _ = pick(pi(vendor="nvidia"), vulkan)
    assert names == vulkan


def test_arm64_glibc_gate_blocks_bookworm():
    # Bookworm (Debian 12) = 2.36 -> source build
    assert "glibc" in li.prebuilt_blocker(pi(), glibc=(2, 36))
    # Trixie (Debian 13) = 2.41 and Ubuntu 24.04 = 2.39 -> prebuilt
    assert li.prebuilt_blocker(pi(), glibc=(2, 41)) == ""
    assert li.prebuilt_blocker(pi(), glibc=(2, 39)) == ""
    # Exactly the documented minimum passes
    assert li.prebuilt_blocker(pi(), glibc=li.GLIBC_ARM64_MIN) == ""
    # x64 CPU has no arm64 gate
    assert li.prebuilt_blocker({"os": "linux", "gpu_vendor": "cpu",
                                "detected": {"arch": "x86_64"}},
                               glibc=(2, 35)) == ""


def test_arm64_engine_plan_follows_glibc():
    dec = li.decide_engine_plan(pi(), probe={"kind": "missing"},
                                glibc=(2, 36))
    assert dec["mode"] == "source"
    assert "glibc" in dec["prebuilt_blocker"]
    dec = li.decide_engine_plan(pi(), probe={"kind": "missing"},
                                glibc=(2, 41))
    assert dec["mode"] == "prebuilt"
    assert dec["prebuilt_blocker"] == ""


def test_arm64_agent_binary_assets():
    from routes.install import _agent_binary_asset

    assert _agent_binary_asset(pi()) == "llama-peer-agent-linux-arm64"
    assert _agent_binary_asset(
        {"os": "linux", "detected": {"arch": "x86_64"}}) \
        == "llama-peer-agent-linux-x86_64"
    assert _agent_binary_asset(
        {"os": "linux", "detected": {"arch": "arm64"}}) \
        == "llama-peer-agent-linux-arm64"


def test_board_model_reads_device_tree(monkeypatch, tmp_path):
    from llama_peer_agent import system

    model = tmp_path / "model"
    model.write_bytes(b"Raspberry Pi 5 Model B Rev 1.0\x00")
    monkeypatch.setattr(system, "_BOARD_MODEL_PATHS", (str(model),))
    assert system._board_model() == "Raspberry Pi 5 Model B Rev 1.0"


def test_board_model_empty_when_absent(monkeypatch):
    from llama_peer_agent import system

    monkeypatch.setattr(system, "_BOARD_MODEL_PATHS",
                        ("/nonexistent/device-tree/model",))
    assert system._board_model() == ""
