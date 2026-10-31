"""Unit tests for tuner._pick_port: bench ports must skip running tunes
and anything already listening, so two concurrent tunes never share a
port (benches would measure the wrong server).
"""
import socket

import tuner


def test_pick_port_free_returns_wanted():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    tuner._JOBS.clear()
    assert tuner._pick_port(free) == free


def test_pick_port_skips_listening():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        s.listen(1)
        tuner._JOBS.clear()
        got = tuner._pick_port(busy)
        assert got != busy


def test_pick_port_skips_running_tune_ports():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    tuner._JOBS.clear()
    tuner._JOBS["m1"] = {"state": "running", "port": free}
    try:
        assert tuner._pick_port(free) != free
    finally:
        tuner._JOBS.clear()


def test_pick_port_ignores_idle_tune_ports():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    tuner._JOBS.clear()
    tuner._JOBS["m1"] = {"state": "idle", "port": free}
    try:
        assert tuner._pick_port(free) == free
    finally:
        tuner._JOBS.clear()


def test_cancel_tune_unknown_model():
    res = tuner.cancel_tune("no-such-model")
    assert res["cancelled"] is False
