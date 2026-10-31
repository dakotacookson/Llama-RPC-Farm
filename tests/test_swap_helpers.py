"""Unit tests for swap_manager's pure helpers: key dedup, env, dir/paths,
systemd unit, binary resolution, and settings round-trip.

No daemon, no SSH, no real config files (tmp settings path).
"""
import os

import pytest

import swap_manager


def _host(name="aibox1", os="linux", **kw):
    h = {"name": name, "ip": "127.0.0.1", "username": "zazi",
         "password": "p", "os": os, "gpu_vendor": "nvidia"}
    h.update(kw)
    return h


# ---------------- _swap_key ----------------

def test_swap_key_stable_and_deduped():
    taken = set()
    k1, w1 = swap_manager._swap_key({"model_id": "m"}, taken)
    assert (k1, w1) == ("m", None)
    k2, w2 = swap_manager._swap_key({"model_id": "m"}, taken)
    assert k2 == "m-2" and w2
    k3, _ = swap_manager._swap_key({"model_id": "m"}, taken)
    assert k3 == "m-3"


def test_swap_key_falls_back_to_name_and_file():
    taken = set()
    k, _ = swap_manager._swap_key({"name": "Pretty"}, taken)
    assert k == "Pretty"
    k2, _ = swap_manager._swap_key({"_file": "f"}, taken)
    assert k2 == "unknown" or k2 == "f"


# ---------------- _swap_env ----------------

def test_swap_env_nvidia(monkeypatch):
    import config_loader as cfg
    monkeypatch.setattr(cfg.machines, "require",
                        lambda n: _host(gpu_vendor="nvidia"))
    env = swap_manager._swap_env("aibox1")
    assert env["GGML_SCHED_DEBUG"] == "3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["GGML_CUDA_P2P"] == "1"


def test_swap_env_amd_and_intel(monkeypatch):
    import config_loader as cfg
    monkeypatch.setattr(cfg.machines, "require",
                        lambda n: _host(gpu_vendor="amd"))
    env = swap_manager._swap_env("aibox1")
    assert "CUDA_DEVICE_ORDER" not in env
    assert env["GGML_CUDA_P2P"] == "1"
    monkeypatch.setattr(cfg.machines, "require",
                        lambda n: _host(gpu_vendor="intel"))
    assert swap_manager._swap_env("aibox1")["SYCL_CACHE_PERSISTENT"] == "1"


# ---------------- paths ----------------

def test_expand_remote():
    assert swap_manager._expand_remote("~/c.yaml", "/home/z",
                                       "linux") == "/home/z/c.yaml"
    assert swap_manager._expand_remote(
        "%USERPROFILE%\\.config\\x", "C:\\U\\z", "windows") == \
        "C:\\U\\z\\.config\\x"
    assert swap_manager._expand_remote("/abs/x", "/home/z",
                                       "linux") == "/abs/x"


def test_expand_local(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/home/z" + p[1:])
    assert swap_manager._expand_local("~/c.yaml", "linux") == \
        "/home/z/c.yaml"


def test_swap_dir():
    assert swap_manager._swap_dir(_host(),
                                  "/home/z/.config/llama-swap/config.yaml") == \
        "/home/z/.config/llama-swap"
    assert swap_manager._swap_dir(
        _host(os="windows"),
        "C:\\U\\.config\\llama-swap\\config.yaml") == \
        "C:\\U\\.config\\llama-swap"


def test_host_config_path_local():
    h = _host()
    assert swap_manager._host_config_path(h, "~/.config/x.yaml") == \
        os.path.expanduser("~/.config/x.yaml")


def test_effective_config_path_keeps_matching_os():
    h = _host(os="linux")
    p, fell = swap_manager._effective_config_path(
        h, "/home/z/.config/llama-swap/config.yaml")
    assert fell is False
    p2, fell2 = swap_manager._effective_config_path(
        h, "%USERPROFILE%\\.config\\x.yaml")
    assert fell2 is True  # windows path on linux host -> host default


# ---------------- systemd unit ----------------

def test_systemd_unit():
    u = swap_manager._systemd_unit("/usr/bin/llama-swap",
                                   "/home/z/c.yaml", 8080)
    assert 'ExecStart=/usr/bin/llama-swap --config "/home/z/c.yaml" ' \
        '--listen :8080' in u
    assert "Restart=on-failure" in u
    assert "WantedBy=default.target" in u


# ---------------- _resolve_binary ----------------

def test_resolve_binary_local_absolute(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *a: None)
    assert swap_manager._resolve_binary(
        _host(), "/opt/ls/llama-swap") == "/opt/ls/llama-swap"


def test_resolve_binary_local_via_path(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda p: "/usr/bin/llama-swap" if p == "llama-swap"
                        else None)
    assert swap_manager._resolve_binary(_host(), "llama-swap") == \
        "/usr/bin/llama-swap"


# ---------------- settings round-trip ----------------

def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE",
                        str(tmp_path / "swap.yaml"))
    monkeypatch.setattr(swap_manager, "get_host", lambda n: _host())
    s = swap_manager.save_settings(binary="/b/ls", listen_port=8888,
                                    config_path="/c.yaml", host_name=None)
    assert s["listen_port"] == 8888
    assert s["binary"] == "/b/ls"
    s2 = swap_manager.load_settings()
    assert s2["listen_port"] == 8888
    # partial save keeps the rest
    swap_manager.save_settings(listen_port=8889)
    assert swap_manager.load_settings()["listen_port"] == 8889
    assert swap_manager.load_settings()["binary"] == "/b/ls"


def test_load_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE",
                        str(tmp_path / "missing.yaml"))
    monkeypatch.setattr(swap_manager, "get_host", lambda n: _host())
    s = swap_manager.load_settings()
    assert s["listen_port"] == 8080
    assert s["binary"] == "llama-swap"
