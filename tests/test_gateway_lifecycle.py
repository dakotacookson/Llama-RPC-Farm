"""Gateway lifecycle: the embedded :8090 gateway must follow the app.

A stale standalone `swap_gateway.py` used to keep serving old code because
startup saw the port answering and deferred to it. Startup now stops a
recognizably-farm gateway and takes the port; unrelated listeners stay.
"""
import types

import pytest

import app as app_module


@pytest.fixture(autouse=True)
def _reset_gateway_server():
    app_module._GATEWAY_SERVER = None
    yield
    app_module._GATEWAY_SERVER = None


def test_stop_stale_gateway_kills_farm_gateway(monkeypatch):
    state = {"open": True}
    killed = []
    monkeypatch.setattr(app_module, "_port_open", lambda p: state["open"])
    monkeypatch.setattr(app_module, "_port_listener_pids", lambda p: [1234])
    monkeypatch.setattr(
        app_module, "_pid_cmdline",
        lambda p: "/opt/farm/.venv/bin/python src/swap_gateway.py")
    monkeypatch.setattr(app_module, "_stop_gateway_units", lambda: None)

    def fake_terminate(pid):
        killed.append(pid)
        state["open"] = False

    monkeypatch.setattr(app_module, "_terminate_pid", fake_terminate)

    assert app_module._stop_stale_gateway(8090) is True
    assert killed == [1234]


def test_stop_stale_gateway_leaves_unrelated_listener(monkeypatch):
    killed = []
    monkeypatch.setattr(app_module, "_port_open", lambda p: True)
    monkeypatch.setattr(app_module, "_port_listener_pids", lambda p: [4242])
    monkeypatch.setattr(
        app_module, "_pid_cmdline",
        lambda p: "python3 -m http.server 8090")
    monkeypatch.setattr(app_module, "_terminate_pid",
                        lambda p: killed.append(p))

    assert app_module._stop_stale_gateway(8090) is False
    assert killed == []


def test_start_swap_gateway_takes_over_stale_listener(monkeypatch):
    import uvicorn

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("FARM_GATEWAY_DISABLE", raising=False)
    monkeypatch.setenv("GATEWAY_PORT", "18090")

    state = {"open": True}
    calls = {"stopped": [], "started": False}
    monkeypatch.setattr(app_module, "_port_open", lambda p: state["open"])

    def fake_stop(port):
        calls["stopped"].append(port)
        state["open"] = False
        return True

    monkeypatch.setattr(app_module, "_stop_stale_gateway", fake_stop)

    class _FakeServer:
        def __init__(self, cfg):
            self.should_exit = False
            calls["started"] = True

        def run(self):
            pass

    monkeypatch.setattr(
        uvicorn, "Config",
        lambda *a, **k: types.SimpleNamespace(host="127.0.0.1", port=18090))
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)

    app_module.start_swap_gateway()

    assert calls["stopped"] == [18090]
    assert calls["started"] is True
    assert app_module._GATEWAY_SERVER is not None


def test_start_swap_gateway_never_touches_port_when_disabled(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("FARM_GATEWAY_DISABLE", "1")
    probed = []
    monkeypatch.setattr(app_module, "_port_open",
                        lambda p: probed.append(p) or True)

    app_module.start_swap_gateway()

    assert probed == []
    assert app_module._GATEWAY_SERVER is None
