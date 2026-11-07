"""Peer agent phase 1: protocol server + master client + transport seam."""
import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

from llama_peer_agent.config import AgentConfig  # noqa: E402
from llama_peer_agent import server as agent_server  # noqa: E402
import agent_client  # noqa: E402
import peer_transport  # noqa: E402
import rpc_manager  # noqa: E402
import swap_manager  # noqa: E402
import stats_collector  # noqa: E402
import subprocess  # noqa: E402
import secrets_vault  # noqa: E402


TOKEN = "test-agent-token-000000000000000000"


@pytest.fixture
def live_agent():
    config = AgentConfig(token=TOKEN, host="127.0.0.1", port=0)
    srv = agent_server.create_server(config)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


@pytest.fixture
def live_agent_roots(tmp_path):
    config = AgentConfig(token=TOKEN, host="127.0.0.1", port=0,
                         file_roots=[str(tmp_path)], discovery=False)
    srv = agent_server.create_server(config)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, tmp_path
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _url(port, path):
    return f"http://127.0.0.1:{port}{path}"


def _get(port, path, token=TOKEN):
    req = urllib.request.Request(_url(port, path))
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode())


def _post(port, path, body, token=TOKEN):
    req = urllib.request.Request(
        _url(port, path), data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode())


def test_health_requires_token(live_agent):
    # /health and /version stay open (discovery/pairing); everything
    # else requires the bearer token.
    status, health = _get(live_agent, "/health", token=None)
    assert status == 200 and health["ok"] is True
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(live_agent, "/status", token=None)
    assert exc.value.code == 401


def test_health_version_capabilities(live_agent):
    status, health = _get(live_agent, "/health")
    assert status == 200
    assert health["ok"] is True
    assert health["protocol_version"] == 1
    status, version = _get(live_agent, "/version")
    assert status == 200 and version["protocol_version"] == 1
    status, caps = _get(live_agent, "/capabilities")
    assert status == 200
    assert {"os", "arch", "rpc_available", "rpc_binary",
            "model_roots"} <= set(caps)


def test_rpc_list_shape(live_agent, monkeypatch):
    monkeypatch.setattr(
        agent_server.processes, "list_rpc_processes",
        lambda: [{"pid": 42, "port": 51234, "age_s": 9, "cmd": "ggml-rpc-server -p 51234"}])
    status, body = _get(live_agent, "/rpc/list")
    assert status == 200
    assert body["servers"][0]["port"] == 51234


def test_rpc_ensure_and_stop_validation(live_agent, monkeypatch):
    calls = {}

    def fake_ensure(config, port, devices, env):
        calls["ensure"] = (port, devices, env)
        return {"running": True, "reused": False, "port": port, "pid": 123}

    def fake_stop(config, port):
        calls["stop"] = port
        return {"port": port, "killed": True, "pids": [123]}

    monkeypatch.setattr(agent_server.processes, "ensure_rpc", fake_ensure)
    monkeypatch.setattr(agent_server.processes, "stop_rpc_port", fake_stop)
    status, body = _post(live_agent, "/rpc/ensure", {
        "port": 51234, "devices": ["CUDA0"], "env": {"A": "1"}})
    assert status == 200 and body["running"] is True
    assert calls["ensure"] == (51234, ["CUDA0"], {"A": "1"})
    status, body = _post(live_agent, "/rpc/stop", {"port": 51234})
    assert status == 200 and body["killed"] is True
    assert calls["stop"] == 51234
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(live_agent, "/rpc/ensure", {"port": 0})
    assert exc.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(live_agent, "/rpc/ensure", {"port": 51234, "devices": ["--bad"]})
    assert exc.value.code == 400


def test_agent_client_health_and_transport(live_agent, monkeypatch):
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: TOKEN)
    machine = {"name": "peer1", "ip": "127.0.0.1",
               "agent_port": live_agent, "transport": "agent"}
    assert agent_client.should_use_agent(machine) is True
    assert agent_client.health(machine)["ok"] is True
    assert isinstance(agent_client.rpc_list(machine), list)
    assert agent_client.should_use_agent({**machine, "transport": "ssh"}) is True


def test_agent_client_auto_falls_back_when_down(monkeypatch):
    agent_client._health_cache.clear()
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: TOKEN)
    machine = {"name": "closed-peer", "ip": "127.0.0.1",
               "agent_port": 9, "transport": "auto"}
    # agent-only: transport selection no longer probes/falls back
    assert agent_client.should_use_agent(machine) is True
    with pytest.raises(agent_client.AgentError):
        agent_client.health(machine, timeout=0.2)


def test_rpc_manager_uses_agent_for_list_ensure_stop(stub_cfg, monkeypatch):
    stub_cfg["peer1"] = {
        "name": "peer1", "id": "peer1", "ip": "10.0.0.2",
        "username": "runner", "os": "linux", "transport": "agent",
        "agent_port": 8198, "rpc_port": 50052, "expose": {"mode": "both", "gpus": []},
    }
    monkeypatch.setattr(rpc_manager.agent_client, "should_use_agent",
                        lambda m: True)
    monkeypatch.setattr(rpc_manager.agent_client, "transport",
                        lambda m: "agent")
    monkeypatch.setattr(rpc_manager.agent_client, "rpc_list",
                        lambda m: [{"pid": 7, "port": 51234, "age_s": 1,
                                    "cmd": "ggml-rpc-server -p 51234"}])
    monkeypatch.setattr(rpc_manager.agent_client, "rpc_ensure",
                        lambda m, p, d, e: {"running": True, "reused": False,
                                            "port": p, "pid": 8})
    monkeypatch.setattr(rpc_manager.agent_client, "rpc_stop",
                        lambda m, p: {"port": p, "killed": True, "pids": [8]})
    monkeypatch.setattr(rpc_manager.agent_client, "stop_all",
                        lambda m: {"killed": True, "pids": [9],
                                   "remaining": {}})
    monkeypatch.setattr(rpc_manager.agent_client, "status",
                        lambda m: {"rpc_running": True, "gpus": [],
                                   "rpc": [{"port": 51234}]})
    assert rpc_manager.list_rpc_servers("peer1")[0]["port"] == 51234
    started = rpc_manager.start_rpc("peer1", 51234, device_list=["CUDA0"])
    assert started["running"] is True and started["port"] == 51234
    stopped = rpc_manager.stop_rpc("peer1", 51234)
    assert stopped["killed"] is True
    all_stopped = rpc_manager.stop_rpc("peer1")
    assert all_stopped["killed"] is True and all_stopped["port"] is None
    status = rpc_manager.machine_status("peer1")
    assert status["agent"] is True and status["rpc_running"] is True


def test_llama_list_and_status_endpoints(live_agent, monkeypatch):
    monkeypatch.setattr(agent_server.processes, "list_llama_processes",
                        lambda: [{"pid": 9, "port": 51991, "age_s": 4,
                                  "cmd": "llama-server -m /x --port 51991"}])
    status, body = _get(live_agent, "/llama/list")
    assert status == 200 and body["servers"][0]["port"] == 51991
    monkeypatch.setattr(agent_server.processes, "llama_status",
                        lambda port: {"port": port, "running": True,
                                      "loading": False, "pid": 9})
    status, body = _get(live_agent, "/llama/status?port=51991")
    assert status == 200 and body["running"] is True
    monkeypatch.setattr(agent_server.processes, "llama_status",
                        lambda port: {"port": port, "running": False,
                                      "loading": False, "pid": None})
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(live_agent, "/llama/status?port=51991")
    assert exc.value.code == 404


def test_llama_start_stop_endpoints(live_agent, monkeypatch):
    calls = {}

    def fake_start(config, port, args, env, state_key):
        calls["start"] = (port, args, env, state_key)
        return {"running": True, "loading": False, "reused": False,
                "port": port, "pid": 11}

    def fake_stop(config, port, state_key):
        calls["stop"] = (port, state_key)
        return {"port": port, "killed": True, "pids": [11]}

    monkeypatch.setattr(agent_server.processes, "start_llama", fake_start)
    monkeypatch.setattr(agent_server.processes, "stop_llama_port", fake_stop)
    status, body = _post(live_agent, "/llama/start", {
        "port": 51991, "args": ["-m", "/tmp/model.gguf"],
        "env": {"A": "1"}, "state_key": "group-a"})
    assert status == 200 and body["running"] is True
    assert calls["start"] == (51991, ["-m", "/tmp/model.gguf"], {"A": "1"},
                              "group-a")
    status, body = _post(live_agent, "/llama/stop", {
        "port": 51991, "state_key": "group-a"})
    assert status == 200 and body["killed"] is True
    assert calls["stop"] == (51991, "group-a")
    # validation: model arg required, managed host/port rejected
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(live_agent, "/llama/start", {"port": 51991, "args": ["--ctx-size", "1"]})
    assert exc.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(live_agent, "/llama/start", {
            "port": 51991, "args": ["-m", "/x", "--host", "1.2.3.4"]})
    assert exc.value.code == 400


def test_agent_client_llama_methods(live_agent, monkeypatch):
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: TOKEN)
    monkeypatch.setattr(agent_server.processes, "start_llama",
                        lambda c, p, a, e, s: {"running": True, "loading": False,
                                                "reused": False, "port": p})
    monkeypatch.setattr(agent_server.processes, "stop_llama_port",
                        lambda c, p, s: {"port": p, "killed": True, "pids": []})
    machine = {"name": "peer1", "ip": "127.0.0.1",
               "agent_port": live_agent, "transport": "agent"}
    assert agent_client.llama_start(
        machine, 51991, ["-m", "/tmp/x.gguf"], {}, state_key="k")["running"]
    assert isinstance(agent_client.llama_list(machine), list)
    assert agent_client.llama_stop(machine, 51991, state_key="k")["killed"]


def test_swap_process_list_allows_no_port(monkeypatch):
    class R:
        stdout = "123 10 llama-swap --config /x --listen :8080\n"
    monkeypatch.setattr(agent_server.processes.subprocess, "run",
                        lambda *a, **k: R())
    procs = agent_server.processes.list_swap_processes()
    assert procs[0]["pid"] == 123 and procs[0]["port"] == 8080


def test_info_status_process_log_endpoints(live_agent, monkeypatch, tmp_path):
    status, info = _get(live_agent, "/info")
    assert status == 200
    assert {"hostname", "machine_id", "agent_port", "os",
            "arch"} <= set(info)
    monkeypatch.setattr(
        agent_server.processes, "process_summary",
        lambda c: {"rpc": [], "llama": [], "swap": [],
                   "gpus": [], "rpc_running": False,
                   "llama_running": False, "swap_running": False})
    status, body = _get(live_agent, "/status")
    assert status == 200 and body["rpc_running"] is False
    status, body = _get(live_agent, "/processes")
    assert status == 200 and body["swap"] == []
    log = tmp_path / "rpc-server-51234.log"
    log.write_text("old\n" + "\n".join(f"line{i}" for i in range(30)))
    cfg = AgentConfig(token=TOKEN, host="127.0.0.1", port=0,
                      log_dir=str(tmp_path))
    assert len(agent_server.processes.logs_tail(
        cfg, "rpc-server-51234.log", 5)["lines"]) == 5
    with pytest.raises(agent_server.processes.AgentProcessError):
        agent_server.processes.logs_tail(cfg, "../secret.log", 5)


def test_logs_tail_endpoint(live_agent, monkeypatch):
    def fake_tail(c, n, l):
        if n != "rpc-server.log":
            raise agent_server.processes.AgentProcessError("unknown log name")
        return {"name": n, "lines": ["a"], "text": "a"}
    monkeypatch.setattr(agent_server.processes, "logs_tail", fake_tail)
    status, body = _get(live_agent, "/logs/tail?name=rpc-server.log&lines=1")
    assert status == 200 and body["text"] == "a"
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(live_agent, "/logs/tail?name=../secret&lines=1")
    assert exc.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(live_agent, "/logs/tail?name=rpc-server.log&lines=x")
    assert exc.value.code == 400


def test_stop_all_endpoint(live_agent, monkeypatch):
    monkeypatch.setattr(
        agent_server.processes, "stop_all",
        lambda c: {"killed": True, "pids": [1], "remaining": {}})
    status, body = _post(live_agent, "/llama/stop-all", {})
    assert status == 200 and body["killed"] is True


def test_agent_client_status_logs_stop_all(live_agent, monkeypatch):
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: TOKEN)
    monkeypatch.setattr(agent_server.processes, "process_summary",
                        lambda c: {"rpc_running": False, "rpc": []})
    monkeypatch.setattr(agent_server.processes, "logs_tail",
                        lambda c, n, l: {"name": n, "text": "x",
                                        "lines": ["x"]})
    monkeypatch.setattr(agent_server.processes, "stop_all",
                        lambda c: {"killed": True, "pids": []})
    machine = {"name": "peer1", "ip": "127.0.0.1",
               "agent_port": live_agent, "transport": "agent"}
    assert agent_client.status(machine)["rpc_running"] is False
    assert agent_client.logs_tail(machine, "rpc-server.log", 1)["text"] == "x"
    assert agent_client.stop_all(machine)["killed"] is True


def test_peer_transport_uses_agent(monkeypatch):
    machine = {"name": "peer1", "ip": "10.0.0.2", "transport": "agent"}
    monkeypatch.setattr(agent_client, "should_use_agent", lambda m: True)
    monkeypatch.setattr(agent_client, "status", lambda m: {"reachable": True})
    monkeypatch.setattr(agent_client, "stop_all", lambda m: {"killed": True})
    monkeypatch.setattr(agent_client, "logs_tail",
                        lambda m, n, l: {"text": "ok"})
    t = peer_transport.for_machine(machine)
    assert isinstance(t, peer_transport.AgentTransport)
    assert t.status(machine)["reachable"] is True
    assert t.stop_all(machine)["killed"] is True
    assert t.logs_tail(machine, "rpc-server.log")["text"] == "ok"


def test_file_endpoints(live_agent_roots):
    port, root = live_agent_roots
    dest = root / "scripts" / "start.sh"
    status, body = _post(port, "/files/write", {
        "path": str(dest), "text": "#!/bin/sh\necho hi\n",
        "mode": 0o755, "create_parents": True})
    assert status == 200 and body["bytes"] > 0
    assert oct(dest.stat().st_mode & 0o777) == "0o755"
    status, body = _get(port, "/files/stat?path=" + str(dest))
    assert status == 200 and body["exists"] and body["is_file"]
    status, body = _get(port, "/files/list?path=" + str(root / "scripts"))
    assert status == 200 and body["entries"][0]["name"] == "start.sh"
    status, body = _post(port, "/files/backup", {
        "path": str(dest), "stamp": "test"})
    assert status == 200 and body["backed_up"]
    assert (root / "scripts" / "start.sh.bak-test").exists()
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(port, "/files/write", {
            "path": "/etc/passwd", "text": "x"})
    assert exc.value.code == 400


def test_service_endpoints(live_agent, monkeypatch):
    monkeypatch.setattr(agent_server.service, "start",
                        lambda c, n, b, p, port: {"method": "systemd",
                                                  "running": True})
    monkeypatch.setattr(agent_server.service, "stop",
                        lambda c, n: {"killed": True, "pids": [1]})
    monkeypatch.setattr(agent_server.service, "status",
                        lambda c, n: {"running": True, "pids": [1]})
    monkeypatch.setattr(agent_server.service, "validate",
                        lambda c, b, p: {"returncode": 0, "output": "ok"})
    monkeypatch.setattr(agent_server.service, "which",
                        lambda n: {"name": n, "path": "/usr/bin/" + n})
    status, body = _post(live_agent, "/service/start", {
        "name": "llama-swap", "binary": "llama-swap",
        "config_path": "/tmp/config.yaml", "port": 8080})
    assert status == 200 and body["running"] is True
    status, body = _post(live_agent, "/service/stop", {"name": "llama-swap"})
    assert status == 200 and body["killed"] is True
    status, body = _get(live_agent, "/service/status?name=llama-swap")
    assert status == 200 and body["running"] is True
    status, body = _post(live_agent, "/service/validate", {
        "binary": "llama-swap", "config_path": "/tmp/config.yaml"})
    assert status == 200 and body["returncode"] == 0
    status, body = _get(live_agent, "/which?name=llama-swap")
    assert status == 200 and body["path"].endswith("llama-swap")


def test_swap_manager_agent_routing(monkeypatch):
    monkeypatch.setattr(swap_manager.local_node, "is_local", lambda h: False)
    monkeypatch.setattr(swap_manager.agent_client, "should_use_agent",
                        lambda h: True)
    monkeypatch.setattr(swap_manager.agent_client, "transport",
                        lambda h: "agent")
    calls = {}
    monkeypatch.setattr(swap_manager.agent_client, "file_write",
                        lambda h, p, t, mode=None: calls.update(
                            write=(p, t, mode)) or {"path": p})
    monkeypatch.setattr(swap_manager.agent_client, "file_mkdir",
                        lambda h, p: calls.update(mkdir=p) or {"path": p})
    monkeypatch.setattr(swap_manager.agent_client, "file_backup",
                        lambda h, p, s: calls.update(backup=(p, s)) or
                        {"backed_up": True, "backup": p + ".bak-" + s})
    monkeypatch.setattr(swap_manager.agent_client, "port_open",
                        lambda h, p: {"open": True})
    monkeypatch.setattr(swap_manager.agent_client, "service_status",
                        lambda h: {"running": True})
    monkeypatch.setattr(swap_manager.agent_client, "service_start",
                        lambda h, n, b, c, p: {"method": "systemd",
                                               "running": True})
    monkeypatch.setattr(swap_manager.agent_client, "which",
                        lambda h, n: {"path": "/usr/bin/llama-swap"})
    host = {"name": "peer1", "ip": "10.0.0.2", "os": "linux"}
    swap_manager._write_host(host, "/home/u/config.yaml", "x", mode=0o600)
    assert calls["write"] == ("/home/u/config.yaml", "x", 0o600)
    swap_manager._ensure_host_dir(host, "/home/u/scripts")
    assert calls["mkdir"] == "/home/u/scripts"
    assert swap_manager._backup_host_file(host, "/home/u/x", "t").endswith(
        ".bak-t")
    assert swap_manager._port_open(host, 8080) is True
    assert swap_manager._process_running(host) is True
    assert swap_manager._resolve_binary(host, "llama-swap") ==         "/usr/bin/llama-swap"
    assert swap_manager._start_linux_service(
        host, "llama-swap", "/home/u/config.yaml", 8080)["running"] is True


def test_metrics_endpoint_and_client(live_agent, monkeypatch):
    monkeypatch.setattr(agent_server.metrics, "collect",
                        lambda: {"status": "online", "os_type": "linux",
                                 "cpu_pct": 12.5, "cores": 8,
                                 "gpus": [], "ram_total_gb": 16,
                                 "ram_used_gb": 4})
    status, body = _get(live_agent, "/metrics")
    assert status == 200 and body["cpu_pct"] == 12.5
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: TOKEN)
    machine = {"name": "peer1", "ip": "127.0.0.1",
               "agent_port": live_agent, "transport": "agent"}
    assert agent_client.metrics(machine)["cores"] == 8


def test_stats_collector_uses_agent(monkeypatch):
    monkeypatch.setattr(stats_collector.local_node, "is_local",
                        lambda m: False)
    monkeypatch.setattr(stats_collector.agent_client, "should_use_agent",
                        lambda m: True)
    monkeypatch.setattr(stats_collector.agent_client, "metrics",
                        lambda m: {"status": "online", "os_type": "linux",
                                   "cpu_pct": 20.0, "cores": 4,
                                   "gpus": [], "ram_total_gb": 8,
                                   "ram_used_gb": 2, "load": "0.1"})
    monkeypatch.setattr(stats_collector.agent_client, "status",
                        lambda m: {"rpc_running": True})
    machine = {"name": "peer1", "ip": "10.0.0.2", "os": "linux"}
    st = stats_collector.machine_stats(machine)
    assert st["status"] == "online" and st["cpu_pct"] == 20.0
    assert st["rpc_running"] is True


def test_system_detail_and_speedtest(live_agent, monkeypatch):
    monkeypatch.setattr(agent_server.system, "detail",
                        lambda c: {"kernel": "Linux", "cores": 4,
                                   "net": {"ip": "10.0.0.2", "speed": "1 Gb/s"},
                                   "disk": {"free": "100", "total": "200"}})
    status, body = _get(live_agent, "/system/detail")
    assert status == 200 and body["kernel"] == "Linux"
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: TOKEN)
    machine = {"name": "peer1", "ip": "127.0.0.1",
               "agent_port": live_agent, "transport": "agent"}
    assert agent_client.system_detail(machine)["cores"] == 4
    req = urllib.request.Request(_url(live_agent, "/speedtest?mb=1"))
    req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=10) as r:
        assert len(r.read()) == 1024 * 1024
    assert agent_client.speedtest(machine, mb=1) > 0


def test_rdma_install_endpoints(live_agent, monkeypatch):
    monkeypatch.setattr(agent_server.system, "rdma",
                        lambda: {"dev": "mlx5_0", "ip": "10.0.0.5"})
    status, body = _get(live_agent, "/system/rdma")
    assert status == 200 and body["dev"] == "mlx5_0"
    monkeypatch.setattr(agent_server.install, "artifact",
                        lambda c, u, d, a, s, m, mode:
                        {"dest": d, "urls": u, "written": []})
    monkeypatch.setattr(agent_server.install, "verify",
                        lambda c, p, a: {"returncode": 0, "output": "ok"})
    status, body = _post(live_agent, "/install/artifact", {
        "urls": ["https://github.com/x/y/releases/download/v/asset.tgz"],
        "dest": "~/llama.cpp/bin", "archive": "tar.gz"})
    assert status == 200 and body["dest"] == "~/llama.cpp/bin"
    status, body = _post(live_agent, "/install/verify", {
        "path": "~/llama.cpp/bin/llama-server"})
    assert status == 200 and body["returncode"] == 0


def test_build_endpoints(live_agent, monkeypatch):
    monkeypatch.setattr(agent_server.build, "probe",
                        lambda c, lb: {"kind": "source", "llama_bin": lb})
    monkeypatch.setattr(agent_server.build, "update",
                        lambda c, s, b, b2, j: {"ok": True, "version": "x"})
    status, body = _get(live_agent, "/build/probe?llama_bin=~/llama.cpp/bin/llama-server")
    assert status == 200 and body["kind"] == "source"
    status, body = _post(live_agent, "/build/update", {
        "src_root": "/src", "build_dir": "/src/build",
        "bin_path": "/src/build/bin/llama-server"})
    assert status == 200 and body["ok"] is True
