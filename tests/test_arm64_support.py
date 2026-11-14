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
    jetson = pi(vendor="nvidia")
    jetson["gpu"] = "CUDA0"
    jetson["detected"]["gpu_names"] = ["Jetson Orin"]
    names, _ = pick(jetson, vulkan)
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


def test_sub_gb_ram_is_not_floored_to_zero():
    from routes.machines import _ram_gb

    assert _ram_gb(0.4) == 0.4
    assert _ram_gb(27.9) == 27
    assert _ram_gb(None) == 0


def test_machine_in_accepts_mac_for_arp_healing():
    from routes.machines import MachineIn

    assert MachineIn(name="pi", ip="10.0.0.145",
                     mac="d8:3a:dd:a6:39:1a").mac \
        == "d8:3a:dd:a6:39:1a"
    assert MachineIn(name="pi", ip="10.0.0.145").mac is None


def test_linux_detail_echoes_glibc(monkeypatch):
    """Regression: the agent computed G but never echoed it, so the farm
    saw glibc='' and could not gate Bookworm's arm64 prebuilt."""
    from llama_peer_agent import system
    from llama_peer_agent.config import AgentConfig

    captured = {}

    def fake_run(cmd, timeout=30):
        captured["cmd"] = cmd
        return 1, "", ""

    monkeypatch.setattr(system, "_run", fake_run)
    system._linux(AgentConfig(token="t", host="127.0.0.1", port=8198))
    assert 'echo "G:$G"' in captured["cmd"]


def test_hw_watch_preserves_board_arch_and_sub_gb_ram(monkeypatch):
    """Regression: the background hardware watcher replaced `detected`
    every pass, wiping a Pi's board/arch and flooring 0.4 GB to 0."""
    from routes import hardware
    import config_loader as cfg
    import stats_collector

    pi = {"name": "pi", "ip": "10.0.0.145", "os": "linux",
          "gpu_vendor": "cpu", "ram_total_gb": 0.4,
          "detected": {"arch": "aarch64",
                       "board": "Raspberry Pi Zero 2 W Rev 1.0",
                       "gpu_names": [], "vram_total_gb": 0, "cores": 4,
                       "ram_total_gb": 0.4}}
    writes = {}

    def update(name, upd):
        writes[name] = upd
        return {**pi, **upd}

    monkeypatch.setattr(cfg.machines, "all", lambda: [pi])
    monkeypatch.setattr(cfg.machines, "get", lambda name: pi)
    monkeypatch.setattr(cfg.machines, "update", update)
    monkeypatch.setattr(cfg, "sync_models_to_hardware", lambda: [])
    monkeypatch.setattr(stats_collector, "machine_stats", lambda m: {
        "status": "online", "gpus": [], "cores": 8, "ram_total_gb": 0.4,
        "board": "Raspberry Pi Zero 2 W Rev 1.0", "arch": "aarch64"})

    hardware.hw_watch_once()

    det = writes["pi"]["detected"]
    assert det["board"] == "Raspberry Pi Zero 2 W Rev 1.0"
    assert det["arch"] == "aarch64"
    assert det["ram_total_gb"] == 0.4
    assert writes["pi"]["ram_total_gb"] == 0.4


def test_cpu_ram_budget_uses_selected_ram(monkeypatch):
    import tuner
    import config_loader as cfg

    pi = {"name": "pi", "ram_total_gb": 0.4,
          "detected": {"ram_total_gb": 0.4}}
    monkeypatch.setattr(cfg.machines, "all", lambda: [pi])
    assert tuner._cpu_ram_budget(["pi"], "pi-RAM") == 0.4 * tuner.RAM_USABLE_FRAC
    # no explicit chip -> all selected servers' RAM
    assert tuner._cpu_ram_budget(["pi"], "") == 0.4 * tuner.RAM_USABLE_FRAC


def test_tuner_estimate_cpu_only_uses_ram(monkeypatch, tmp_path):
    """A GPU-less Pi must still get a math estimate (from RAM), not
    'no GPUs selected'."""
    import tuner
    import config_loader as cfg

    pi = {"name": "pi", "ram_total_gb": 8,
          "detected": {"ram_total_gb": 8}}
    monkeypatch.setattr(cfg.machines, "all", lambda: [pi])
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr(tuner.path_resolver, "resolve_for",
                        lambda p, o: str(model))
    monkeypatch.setattr(tuner.gguf_inspect, "metadata", lambda p: {})
    monkeypatch.setattr(tuner, "_model_bytes", lambda p: 1024 * 1024)
    monkeypatch.setattr(tuner, "_kv_per_token", lambda *a, **k: 1024)
    monkeypatch.setattr(tuner.gguf_inspect, "ssm_state_bytes",
                        lambda *a, **k: 0)
    monkeypatch.setattr(tuner, "has_swa", lambda meta: False)
    monkeypatch.setattr(tuner, "virt_info", lambda m: {})

    est = tuner.estimate(str(model), ["pi"], "pi-RAM")
    assert est["gpus"] == []
    assert est["suggested_split"] == []
    assert est["usable_ram_gb"] > 0
    assert est["recommended_ctx"] >= 4096


def test_detected_vendor_corrects_gpuless_default():
    """A Pi added with the form-default 'nvidia' must become 'cpu' when
    detection finds no GPUs, or the installer would pick a CUDA asset."""
    from routes.machines import _detected_vendor

    assert _detected_vendor({"gpus": [], "gpu_vendor": ""},
                            {"gpu_vendor": "nvidia"}) == "cpu"
    assert _detected_vendor(
        {"gpus": [{"name": "NVIDIA GeForce RTX 3080"}], "gpu_vendor": ""},
        {}) == "nvidia"
    assert _detected_vendor({"gpus": [], "gpu_vendor": "amd"}, {}) == "amd"


def test_vendor_of_gpuless_arm_is_cpu():
    """An arm64 host with no GPUs is CPU-only even if gpu_vendor still
    says nvidia (form default before detection)."""
    assert li._vendor_of({
        "os": "linux", "gpu_vendor": "nvidia", "gpu": "",
        "detected": {"arch": "aarch64", "gpu_names": []},
    }) == "cpu"
    # x86 without a probe keeps the recorded vendor
    assert li._vendor_of({
        "os": "linux", "gpu_vendor": "nvidia", "gpu": "",
        "detected": {"arch": "x86_64", "gpu_names": []},
    }) == "nvidia"
    # arm64 with a real GPU keeps that vendor
    assert li._vendor_of({
        "os": "linux", "gpu_vendor": "nvidia", "gpu": "CUDA0",
        "detected": {"arch": "aarch64", "gpu_names": ["Jetson Orin"]},
    }) == "nvidia"
