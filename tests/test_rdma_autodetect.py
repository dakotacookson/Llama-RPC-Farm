"""Hardware detect auto-enables RDMA when an ACTIVE RoCE NIC is present."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module          # noqa: E402
import routes.machines as machines_routes
import stats_collector
import config_loader as cl        # noqa: E402


def test_detect_hardware_auto_enables_rdma(tmp_path, monkeypatch):
    machines = tmp_path / "machines.yaml"
    machines.write_text(
        "machines:\n"
        "- id: n1\n"
        "  name: node1\n"
        "  ip: 127.0.0.1\n"
        "  role: peer\n"
        "  os: linux\n"
        "  gpu_vendor: nvidia\n"
        "  rdma: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cl.machines, "path", str(machines))
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(cl, "MODELS_DIR", str(models))
    monkeypatch.setattr(app_module.cfg, "MODELS_DIR", str(models))

    monkeypatch.setattr(
        stats_collector, "machine_stats",
        lambda m: {"status": "online",
                   "gpus": [{"index": 0, "name": "GPU", "mem_total_gb": 8,
                             "compute_cap": "8.6", "uuid": "GPU-x"}],
                   "cores": 8, "ram_total_gb": 16})
    monkeypatch.setattr(
        machines_routes, "_detect_rdma_nic",
        lambda machine, is_local: {"dev": "mlx5_0", "gid": "3",
                                   "ip": "10.0.0.9"})

    res = asyncio.run(machines_routes.detect_hardware("node1"))
    assert res["rdma"] is True
    assert res["rdma_dev"] == "mlx5_0"
    assert res["rdma_gid"] == "3"
    assert res["rdma_ip"] == "10.0.0.9"
    assert res["rdma_detected"]["dev"] == "mlx5_0"
