"""Unit tests for stats_collector parsing/aggregation and gguf_inspect
binary readers — the dashboard numbers must parse deterministically.

collect_* functions need live SSH and are covered by integration, not here.
"""
import struct

import pytest

import stats_collector
import gguf_inspect


# ---------------- _parse_gpus ----------------

def test_parse_gpus_nvidia_csv():
    out = ("0, NVIDIA GeForce RTX 3060, 12, 2048, 12288, 55\n"
           "1, NVIDIA GeForce RTX 3060, 0, 0, 12288, 40\n")
    gpus = stats_collector._parse_gpus(out)
    assert len(gpus) == 2
    assert gpus[0]["mem_used_gb"] == 2.0
    assert gpus[0]["mem_total_gb"] == 12.0
    assert gpus[0]["temp"] == 55
    assert gpus[1]["mem_pct"] == 0


def test_parse_gpus_skips_malformed():
    out = "garbage line\n0, GPU, x, y, z, w\n"
    assert stats_collector._parse_gpus(out) == []
    assert stats_collector._parse_gpus("") == []
    assert stats_collector._parse_gpus(None) == []


def test_parse_gpus_zero_total_no_crash():
    out = "0, GPU, 0, 0, 0, 40\n"
    gpus = stats_collector._parse_gpus(out)
    assert gpus[0]["mem_pct"] == 0


def test_parse_gpus_uuid_field():
    """8th CSV field (uuid) parses into gpus[].uuid; 6-7 field rows
    (older queries, AMD/Intel) yield uuid None — never a crash."""
    out = ("0, NVIDIA GeForce RTX 3050, 0, 100, 6144, 31, 8.6, "
           "GPU-cf9df052-a880-50de-71ed-387f0ebcdb11\n"
           "1, NVIDIA GeForce RTX 4080, 1, 2300, 16376, 57, 8.9, "
           "GPU-fb06e8e6-550f-fcfe-728a-8e8e4c964735\n")
    gpus = stats_collector._parse_gpus(out)
    assert gpus[0]["uuid"] == "GPU-cf9df052-a880-50de-71ed-387f0ebcdb11"
    assert gpus[1]["uuid"] == "GPU-fb06e8e6-550f-fcfe-728a-8e8e4c964735"
    old = stats_collector._parse_gpus(
        "0, NVIDIA GeForce RTX 3060, 12, 2048, 12288, 55\n")
    assert old[0]["uuid"] is None


# ---------------- cluster_totals ----------------

def _stat(gpus=(), ram_used=0, ram_total=0, cpu=0.0, cores=0,
          status="online"):
    return {"status": status, "gpus": list(gpus), "ram_used_gb": ram_used,
            "ram_total_gb": ram_total, "cpu_pct": cpu, "cores": cores}


def test_cluster_totals():
    stats = [
        _stat([{"mem_used_gb": 2.0, "mem_total_gb": 12.0}], 4, 16, 50.0, 8),
        _stat([{"mem_used_gb": 6.0, "mem_total_gb": 12.0}], 8, 32, 25.0, 8),
        _stat(status="offline"),  # excluded
    ]
    tot = stats_collector.cluster_totals(stats)
    assert tot["vram"] == {"used": 8.0, "total": 24.0, "pct": 33.3}
    assert tot["ram"] == {"used": 12, "total": 48, "pct": 25.0}
    assert tot["cpu"]["total"] == 16


def test_cluster_totals_empty():
    tot = stats_collector.cluster_totals([])
    assert tot["vram"]["pct"] == 0
    assert tot["cpu"]["pct"] == 0


# ---------------- gguf_inspect readers ----------------

def _gguf_bytes():
    # minimal GGUF header: magic + version + n_tensors + n_kv
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) +
            struct.pack("<Q", 0))


def test_gguf_read_helpers(tmp_path):
    import io
    f = io.BytesIO(_gguf_bytes())
    assert gguf_inspect._read(f, 4) == b"GGUF"
    assert gguf_inspect._read(f, 4) == struct.pack("<I", 3)


def test_gguf_metadata_rejects_non_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(Exception):
        gguf_inspect.metadata(str(bad))


def test_weight_bytes_missing_file():
    with pytest.raises(Exception):
        gguf_inspect.weight_bytes("/nonexistent/x.gguf")
