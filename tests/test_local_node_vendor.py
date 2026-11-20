"""Docker/ROCm installs can declare the GPU vendor explicitly."""
import types

import local_node


class _Machines:
    def __init__(self):
        self.rows = []
        self.added = None

    def all(self):
        return list(self.rows)

    def add(self, data):
        self.added = data
        self.rows.append(data)
        return data

    def update(self, name, data):
        return {**data, "name": name}


def _patch(monkeypatch):
    m = _Machines()
    fake = types.SimpleNamespace(machines=m)
    monkeypatch.setattr(local_node, "cfg", fake)
    monkeypatch.setattr(local_node, "local_ip", lambda: "10.0.0.5")
    monkeypatch.setattr(local_node, "local_hostname", lambda: "host")
    monkeypatch.setattr(local_node, "local_os", lambda: "linux")
    return m


def test_gpu_vendor_env_seeds_local_master(monkeypatch):
    m = _patch(monkeypatch)
    monkeypatch.setenv("LLAMA_FARM_GPU_VENDOR", "amd")
    local_node.ensure_local_master()
    assert m.added["gpu_vendor"] == "amd"


def test_invalid_gpu_vendor_env_is_ignored(monkeypatch):
    m = _patch(monkeypatch)
    monkeypatch.setenv("LLAMA_FARM_GPU_VENDOR", "bogus")
    local_node.ensure_local_master()
    assert "gpu_vendor" not in m.added


def test_no_gpu_vendor_env_keeps_probe_default(monkeypatch):
    m = _patch(monkeypatch)
    monkeypatch.delenv("LLAMA_FARM_GPU_VENDOR", raising=False)
    local_node.ensure_local_master()
    assert "gpu_vendor" not in m.added
