"""Unit tests for gpu_vendor capability helpers.

What the vLLM community actually does for the bf16 problem (upstream
PRs #1901/#17265, cuda.py `supported_dtypes`): check the DEVICE's
compute capability — `torch.cuda.get_device_capability()[0] < 8` — or
equivalently `nvidia-smi --query-gpu=compute_cap`. Product-name tables
rot every generation; the driver-reported sm never does. Unknown /
missing data -> None (no claim, never an error).
"""
import pytest

import gpu_vendor


@pytest.mark.parametrize("text,expected", [
    ("7.5", (7, 5)),
    ("8.6", (8, 6)),
    ("10.0", (10, 0)),
    (" 8.9 ", (8, 9)),
    ("N/A", None),
    ("", None),
    (None, None),
    ("8", None),
    ("8.6.1", None),
])
def test_parse_compute_cap(text, expected):
    assert gpu_vendor.parse_compute_cap(text) == expected


@pytest.mark.parametrize("major,minor,expected", [
    (6, 1, False),   # Pascal
    (7, 0, False),   # Volta
    (7, 5, False),   # Turing (aibox3)
    (8, 0, True),    # Ampere
    (8, 6, True),
    (8, 9, True),    # Ada
    (9, 0, True),    # Hopper
    (10, 0, True),   # Blackwell
])
def test_sm_supports_bf16(major, minor, expected):
    assert gpu_vendor.sm_supports_bf16(major, minor) is expected


@pytest.mark.parametrize("cap,expected", [
    ("7.5", False),
    ("8.6", True),
    (None, None),
    ("N/A", None),
])
def test_compute_cap_supports_bf16(cap, expected):
    assert gpu_vendor.compute_cap_supports_bf16(cap) is expected
