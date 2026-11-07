"""Unit tests for the peer agent's typed modules (branch coverage focus)."""
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

from llama_peer_agent.config import AgentConfig  # noqa: E402
from llama_peer_agent import files, metrics, system, service, install  # noqa: E402
from llama_peer_agent import ray as agent_ray  # noqa: E402
from llama_peer_agent import build as agent_build  # noqa: E402


def cfg(tmp_path, **kw):
    return AgentConfig(token="t", host="127.0.0.1", port=8198,
                       file_roots=[str(tmp_path)], **kw)


def test_config_env_and_validation(monkeypatch):
    monkeypatch.setenv("PEER_AGENT_TOKEN", "tok")
    monkeypatch.setenv("PEER_AGENT_PORT", "9000")
    monkeypatch.setenv("PEER_AGENT_DISCOVERY", "0")
    monkeypatch.setenv("PEER_AGENT_FILE_ROOTS", "/a:/b")
    c = AgentConfig.from_env()
    assert c.token == "tok" and c.port == 9000
    assert c.discovery is False and c.file_roots == ["/a", "/b"]
    monkeypatch.setenv("PEER_AGENT_PORT", "bad")
    with pytest.raises(ValueError):
        AgentConfig.from_env()


def test_files_round_trip_and_guards(tmp_path):
    c = cfg(tmp_path)
    dest = tmp_path / "sub" / "x.sh"
    res = files.write(c, str(dest), "hello", mode=0o755,
                      create_parents=True, backup=False)
    assert res["bytes"] == 5 and dest.read_text() == "hello"
    assert oct(dest.stat().st_mode & 0o777) == "0o755"
    st = files.stat(c, str(dest))
    assert st["exists"] and st["is_file"]
    assert files.mkdir(c, str(tmp_path / "d"))["is_dir"]
    files.chmod(c, str(dest), 0o600)
    assert oct(dest.stat().st_mode & 0o777) == "0o600"
    b = files.backup(c, str(dest), "s1")
    assert b["backed_up"] and Path(b["backup"]).exists()
    assert files.backup(c, str(tmp_path / "missing"), "s")["backed_up"] is False
    (tmp_path / ".hidden").write_text("x")
    listing = files.list_dir(c, str(tmp_path), hidden=False)
    assert all(not e["name"].startswith(".") for e in listing["entries"])
    assert any(e["name"] == ".hidden" for e in
               files.list_dir(c, str(tmp_path), hidden=True)["entries"])
    with pytest.raises(files.AgentFileError):
        files.write(c, "/etc/passwd", "x")
    with pytest.raises(files.AgentFileError):
        files.write(c, "", "x")
    with pytest.raises(files.AgentFileError):
        files.write(c, str(tmp_path / "nope" / "x"), "x", create_parents=False)
    with pytest.raises(files.AgentFileError):
        files.chmod(c, str(tmp_path / "nope"), 0o600)
    with pytest.raises(files.AgentFileError):
        files.list_dir(c, str(dest))
    with pytest.raises(files.AgentFileError):
        files.mkdir(c, "/outside")


def test_files_write_backup_and_stat_missing(tmp_path):
    c = cfg(tmp_path)
    p = tmp_path / "a"
    p.write_text("old")
    res = files.write(c, str(p), "new", backup=True, mode=0o600)
    assert res["backup"] and Path(res["backup"]).read_text() == "old"
    assert files.stat(c, str(tmp_path / "none"))["exists"] is False


def test_metrics_helpers(monkeypatch):
    monkeypatch.setattr(metrics.shutil, "which", lambda n: None)
    assert metrics._gpus() == []
    monkeypatch.setattr(metrics.shutil, "which", lambda n: "/bin/nvidia-smi")
    class P:
        returncode = 0
        stdout = "0,GPU,10,1024,2048,40,8.6,GPU-1\n"
        stderr = ""
    monkeypatch.setattr(metrics, "_run", lambda *a, **k: (0, P.stdout, ""))
    rows = metrics._gpus()
    assert rows[0]["mem_total_gb"] == 2.0 and rows[0]["uuid"] == "GPU-1"
    assert isinstance(metrics._linux(), dict)
    monkeypatch.setattr(metrics, "_run",
                        lambda a, **k: (0, "8\n", "") if a[0] == "sysctl"
                        else (0, "", ""))
    m = metrics._macos()
    assert m["os_type"] == "macos"
    out = json.dumps({"cpu": 12, "cores": 8, "ram_total": 1024 * 1024,
                      "ram_free": 512 * 1024,
                      "gpus": "0,GPU,5,100,200,40\n"})
    monkeypatch.setattr(metrics, "_run", lambda *a, **k: (0, out, ""))
    w = metrics._windows()
    assert w["status"] == "online" and w["gpus"][0]["mem_pct"] == 50


def test_system_helpers(monkeypatch):
    assert system._pcie_tags("4,16\n") == "g4x16"
    assert system._rebar_values("1024\n") == "1024"
    assert system._parse_posix("K:x\nC:y\n") == {"K": "x", "C": "y"}
    blob = "K:k\nC:c\nU:u\nI:i\nS:s\nD:d\nDT:dt\nV:v\nCF:cf\nP:4,16\nR:1\nL:l\n"
    monkeypatch.setattr(system, "_run", lambda *a, **k: (0, blob, ""))
    d = system._linux(cfg(Path("/tmp")))
    assert d["pcie"] == "g4x16" and d["rebar"] == "1"
    monkeypatch.setattr(system, "_run", lambda *a, **k: (1, "", ""))
    assert system._linux(cfg(Path("/tmp"))) == {}
    monkeypatch.setattr(system, "_run", lambda *a, **k: (0, "K:k\n", ""))
    monkeypatch.setattr(system.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": "4\n", "returncode": 0})())
    assert system._macos(cfg(Path("/tmp")))["cores"] == 4
    out = json.dumps({"kernel": "w", "cpu": "c", "cores": 2,
                      "disk": [{"Free": 1}], "coreload": "[1,2]",
                      "llama": "v", "ip": "i", "speed": "s"})
    monkeypatch.setattr(system.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": out, "returncode": 0})())
    assert system._windows(cfg(Path("/tmp")))["cores"] == 2
    monkeypatch.setattr(system, "_run",
                        lambda *a, **k: (0, "mlx5_0|1|3|10.0.0.5|eth0\n", ""))
    assert system.rdma()["dev"] == "mlx5_0"


def test_service_helpers_and_start(monkeypatch, tmp_path):
    c = cfg(tmp_path)
    assert service._name("llama-swap") == "llama-swap"
    with pytest.raises(service.AgentServiceError):
        service._name("other")
    monkeypatch.setattr(service.shutil, "which", lambda n: "/bin/" + n)
    assert service._binary("llama-swap") == "/bin/llama-swap"
    with pytest.raises(service.AgentServiceError):
        service._binary("bash")
    with pytest.raises(service.AgentServiceError):
        service._binary("")
    assert "ExecStart=" in service._unit("b", "c", 1)
    monkeypatch.setattr(service.processes, "list_swap_processes", lambda: [])
    monkeypatch.setattr(service.processes, "stop_swap", lambda c: {"killed": True})
    monkeypatch.setattr(service.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0})())
    monkeypatch.setattr(service, "_popen", lambda *a, **k: 123)
    monkeypatch.setattr(service.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(service.os.path, "isfile", lambda p: True)
    binp = tmp_path / "llama-swap"
    binp.write_text("x")
    res = service.start(c, "llama-swap", str(binp),
                        str(tmp_path / "c.yaml"), 8080)
    assert res["running"] is True
    assert service.stop(c, "llama-swap")["killed"] is True
    assert service.status(c, "llama-swap")["service"] == "llama-swap"
    monkeypatch.setattr(service.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": "ok", "stderr": ""})())
    assert service.validate(c, "/bin/llama-swap",
                            str(tmp_path / "c.yaml"))["returncode"] == 0
    assert service.which("llama-swap")["path"].endswith("llama-swap")
    with pytest.raises(service.AgentServiceError):
        service.which("bash")


def test_install_url_and_verify(monkeypatch, tmp_path):
    with pytest.raises(install.AgentInstallError):
        install._url("http://example.com/x")
    assert install._url("https://github.com/a/b") == "https://github.com/a/b"
    c = cfg(tmp_path)
    binp = tmp_path / "llama-server"
    binp.write_text("x")
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": "v", "stderr": ""})())
    assert install.verify(c, str(binp))["returncode"] == 0
    with pytest.raises(install.AgentInstallError):
        install.verify(c, str(tmp_path / "missing"))
    with pytest.raises(install.AgentInstallError):
        install.verify(c, str(tmp_path / "other"))


def test_install_artifact_tar_zip_none(monkeypatch, tmp_path):
    c = cfg(tmp_path)
    tar_path = tmp_path / "x.tgz"
    with tarfile.open(tar_path, "w:gz") as t:
        data = b"hello"
        info = tarfile.TarInfo("pkg/bin/llama-server")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    zip_path = tmp_path / "x.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("pkg/llama-swap", "hi")
    data = {str(tar_path): tar_path.read_bytes(),
            str(zip_path): zip_path.read_bytes()}
    monkeypatch.setattr(install, "_url", lambda u: u)
    class Resp:
        def __init__(self, b): self.b = io.BytesIO(b)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n=-1): return self.b.read(n)
    monkeypatch.setattr(install.urllib.request, "urlopen",
                        lambda u, timeout=0: Resp(data[u]))
    dest = tmp_path / "out"
    res = install.artifact(c, [str(tar_path)], str(dest), "tar.gz",
                           strip_components=1)
    assert (dest / "bin" / "llama-server").read_text() == "hello"
    install.artifact(c, [str(zip_path)], str(dest), "zip",
                     strip_components=1, member="llama-swap")
    assert (dest / "llama-swap").read_text() == "hi"
    install.artifact(c, [str(tar_path)], str(dest), "none",
                     member="copy.tgz", mode=0o600)
    assert (dest / "copy.tgz").exists()
    with pytest.raises(install.AgentInstallError):
        install.artifact(c, [str(tar_path)], str(dest), "rar")


def test_install_tar_zip_escape(monkeypatch, tmp_path):
    c = cfg(tmp_path)
    tar_path = tmp_path / "bad.tgz"
    with tarfile.open(tar_path, "w:gz") as t:
        data = b"x"
        info = tarfile.TarInfo("../escape")
        info.size = 1
        t.addfile(info, io.BytesIO(data))
    class Resp:
        def __init__(self, b): self.b = io.BytesIO(b)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n=-1): return self.b.read(n)
    monkeypatch.setattr(install, "_url", lambda u: u)
    monkeypatch.setattr(install.urllib.request, "urlopen",
                        lambda u, timeout=0: Resp(tar_path.read_bytes()))
    with pytest.raises(install.AgentInstallError):
        install.artifact(c, ["https://github.com/x"], str(tmp_path / "o"),
                         "tar.gz")


def test_ray_helpers(monkeypatch, tmp_path):
    c = cfg(tmp_path)
    monkeypatch.setattr(agent_ray.shutil, "which", lambda n: "/bin/ray")
    monkeypatch.setattr(agent_ray.os.path, "isfile", lambda p: True)
    assert agent_ray._ray_bin("ray") == "/bin/ray"
    with pytest.raises(agent_ray.AgentRayError):
        agent_ray._ray_bin("/bin/bash")
    monkeypatch.setattr(agent_ray.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": "1.1.1.1 via 10.0.0.1 dev eth0",
                            "returncode": 0})())
    assert agent_ray._iface_for("1.1.1.1") == "eth0"
    assert agent_ray._iface_for("") == ""
    monkeypatch.setattr(agent_ray.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 99})())
    monkeypatch.setattr(agent_ray.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": "", "stderr": ""})())
    res = agent_ray.start(c, "/bin/ray", "10.0.0.1:6379", 2,
                          {"A": "1"}, "10.0.0.1", "~/l", "~/p", "~/")
    assert res["pid"] == 99
    assert agent_ray.stop(c, "/bin/ray", "~/p")["stopped"] is True
    with pytest.raises(agent_ray.AgentRayError):
        agent_ray.start(c, "/bin/ray", "", 2)
    with pytest.raises(agent_ray.AgentRayError):
        agent_ray.start(c, "/bin/ray", "h:1", 2, {"bad-key": 1})


def test_build_probe_and_update(monkeypatch, tmp_path):
    c = cfg(tmp_path)
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=ON\n")
    (build / "bin" / "llama-server").write_text("x")
    monkeypatch.setattr(agent_build.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": str(tmp_path) + "\n",
                            "stderr": ""})())
    p = agent_build.probe(c, str(build / "bin" / "llama-server"))
    assert p["kind"] == "source" and p["flags"]["GGML_CUDA"] == "ON"
    assert agent_build.probe(c, "")["kind"] == "missing"
    def fake_run(args, **k):
        cmd = " ".join(args)
        out = "v\n" if "--version" in cmd else ""
        return type("P", (), {"returncode": 0, "stdout": out,
                              "stderr": ""})()
    monkeypatch.setattr(agent_build.subprocess, "run", fake_run)
    res = agent_build.update(c, str(tmp_path), str(build),
                             str(build / "bin" / "llama-server"), 1)
    assert res["ok"] is True
    with pytest.raises(agent_build.AgentBuildError):
        agent_build.update(c, str(tmp_path / "nope"), str(build),
                           str(build / "bin" / "llama-server"))


def test_agent_client_wrappers(monkeypatch):
    import agent_client as ac
    monkeypatch.setenv("FARM_AGENT_TOKEN", "x")
    calls = []
    def fake(machine, method, path, body=None, timeout=6.0):
        calls.append((method, path, body))
        return {"ok": True, "servers": [], "logs": [], "returncode": 0}
    monkeypatch.setattr(ac, "_request", fake)
    m = {"ip": "10.0.0.1", "agent_port": 8198}
    ac.info(m); ac.status(m); ac.metrics(m); ac.processes(m)
    ac.logs_tail(m, "rpc-server.log", 5); ac.logs_list(m)
    ac.service_status(m); ac.service_start(m, config_path="/c", port=1)
    ac.service_stop(m); ac.service_validate(m, config_path="/c")
    ac.which(m, "llama-swap"); ac.port_open(m, 1)
    ac.file_write(m, "/x", "y", mode=0o600, backup=True,
                  create_parents=True)
    ac.file_mkdir(m, "/d"); ac.file_chmod(m, "/x", 0o600)
    ac.file_backup(m, "/x"); ac.file_stat(m, "/x"); ac.file_list(m, "/")
    ac.ray_start(m, "/r", "h:1", 1, {"A": "1"}, "h", "/l", "/p", "/w")
    ac.ray_stop(m, "/r", "/p")
    ac.install_artifact(m, ["https://github.com/x"], "/d")
    ac.install_verify(m, "/d/x"); ac.vllm_install(m)
    ac.build_probe(m, "/x"); ac.build_update(m, "/s", "/b", "/s/bin")
    ac.system_detail(m); ac.rdma(m)
    assert any(path.startswith("/file") for _, path, _ in calls)
    assert any(path.startswith("/ray") for _, path, _ in calls)
    with pytest.raises(ac.AgentError):
        ac._base({"ip": ""})
    monkeypatch.setenv("FARM_AGENT_TOKEN", "envtok")
    assert ac._token({"name": "x"}) == "envtok"
    monkeypatch.setattr(ac, "_request", lambda *a, **k: {"protocol_version": 2})
    with pytest.raises(ac.AgentError):
        ac.health(m)
    monkeypatch.setattr(ac, "_request", lambda *a, **k: {"protocol_version": "x"})
    with pytest.raises(ac.AgentError):
        ac.health(m)
    monkeypatch.setattr(ac, "_request", lambda *a, **k: {"protocol_version": 1})
    assert ac.health(m)["protocol_version"] == 1
    assert ac.should_use_agent({**m, "transport": "auto"}) is True


def test_peer_transport_both(monkeypatch):
    import peer_transport as pt
    class FakeAgent:
        def status(self, m): return {"agent": True}
        def stop_all(self, m): return {"killed": True}
        def logs_tail(self, m, n, l=20): return {"text": "a"}
        def processes(self, m): return {"rpc": []}
    monkeypatch.setattr(pt, "AgentTransport", FakeAgent)
    t = pt.for_machine({"name": "p"})
    assert t.status({})["agent"] is True
    assert t.stop_all({})["killed"] is True
    assert t.logs_tail({}, "x")["text"] == "a"
    assert t.processes({})["rpc"] == []

def test_main_cli(monkeypatch, capsys):
    from llama_peer_agent import __main__ as cli
    assert cli.main(["--version"]) == 0
    monkeypatch.setattr(cli.AgentConfig, "from_env",
                        lambda **k: (_ for _ in ()).throw(ValueError("bad")))
    assert cli.main([]) == 2
    monkeypatch.setattr(cli.AgentConfig, "from_env",
                        lambda **k: AgentConfig(token=""))
    monkeypatch.setattr(cli, "serve", lambda c: None)
    assert cli.main([]) == 0   # pairing mode allowed
    monkeypatch.setattr(cli.AgentConfig, "from_env",
                        lambda **k: AgentConfig(token="t"))
    monkeypatch.setattr(cli, "serve", lambda c: None)
    assert cli.main([]) == 0
    monkeypatch.setattr(cli, "serve", lambda c: (_ for _ in ()).throw(OSError("x")))
    assert cli.main([]) == 1


def test_discovery_error_paths(monkeypatch):
    from llama_peer_agent import discovery as d
    c = AgentConfig(token="t", host="127.0.0.1", port=1)
    sock = d.create_socket(c)
    stop = __import__("threading").Event()
    stop.set()
    d.serve(c, sock, stop)
    sock.close()
    class Bad:
        def recvfrom(self, n): raise OSError("x")
        def close(self): pass
    d.serve(c, Bad(), stop)
    assert d.payload(c)["magic"] == d.MAGIC.decode()


def test_process_helpers(monkeypatch, tmp_path):
    from llama_peer_agent import processes as p
    assert p._parse_etime("01:02:03") == 3723
    assert p._parse_etime("2-01:02:03") == 176523
    assert p._parse_etime("1:2") == 62
    assert p._parse_etime("bad") is None
    assert p._proc_basename("") == ""
    monkeypatch.setattr(p.shutil, "which", lambda n: "")
    monkeypatch.setattr(p.Path, "home", lambda: tmp_path)
    assert p.rpc_binary(AgentConfig(token="t")) == ""
    assert p.llama_binary(AgentConfig(token="t")) == ""
    (tmp_path / "llama.cpp" / "build" / "bin").mkdir(parents=True)
    (tmp_path / "llama.cpp" / "build" / "bin" / "ggml-rpc-server").write_text("")
    (tmp_path / "llama.cpp" / "build" / "bin" / "llama-server").write_text("")
    assert p.rpc_binary(AgentConfig(token="t")).endswith("ggml-rpc-server")
    assert p.llama_binary(AgentConfig(token="t")).endswith("llama-server")
    assert p.capabilities(AgentConfig(token="t"))["rpc_available"] is True
    assert p.port_open(1)["open"] in (True, False)
    with pytest.raises(p.AgentProcessError):
        p.port_open(0)
    with pytest.raises(p.AgentProcessError):
        p.port_open("x")
    c = AgentConfig(token="t", log_dir=str(tmp_path))
    (tmp_path / "rpc-server-51234.log").write_text("a\nb\n")
    assert p.logs_tail(c, "rpc-server-51234.log", 1)["lines"] == ["b"]
    assert p.logs_tail(c, "llama-server.log", 1)["lines"] == []
    with pytest.raises(p.AgentProcessError):
        p.logs_tail(c, "../x", 1)
    with pytest.raises(p.AgentProcessError):
        p.logs_tail(c, "rpc-server.log", 0)
    assert "rpc-server-51234.log" in p.logs_list(c)["logs"]
    assert p.process_summary(c)["counts"]["rpc"] >= 0
    assert isinstance(p.info(c)["hostname"], str)
    with pytest.raises(p.AgentProcessError):
        p.validate_llama_start(0, [], {}, None)
    with pytest.raises(p.AgentProcessError):
        p.validate_llama_start(1, ["--ctx-size"], {}, None)
    with pytest.raises(p.AgentProcessError):
        p.validate_ensure(0, [], {})
    with pytest.raises(p.AgentProcessError):
        p.validate_ensure(1, ["--bad"], {})
    with pytest.raises(p.AgentProcessError):
        p.validate_ensure(1, [], {"bad-key": 1})
    with pytest.raises(p.AgentProcessError):
        p.validate_llama_start(1, ["-m", "/x"], {"bad-key": 1}, None)
    assert p.validate_ensure(1, ["CUDA0"], {"A": "1"})[0] == 1
    assert p.validate_llama_start(1, ["-m", "/x"], {}, "k")[0] == 1


def test_server_all_endpoints(monkeypatch):
    # reuse the live fixture from test_peer_agent via pytest discovery? no:
    # this module does not import it. Build a tiny local server here.
    import threading
    from llama_peer_agent import server as srv
    conf = AgentConfig(token="t", host="127.0.0.1", port=0, discovery=False,
                       log_dir="/tmp", file_roots=["/tmp"])
    s = srv.create_server(conf)
    port = s.server_address[1]
    th = threading.Thread(target=s.serve_forever, daemon=True)
    th.start()
    import urllib.request, urllib.error
    def get(path):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
        req.add_header("Authorization", "Bearer t")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    def post(path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer t"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    try:
        assert get("/logs/list")[0] == 200
        assert get("/port/open?port=1")[0] == 200
        assert get("/which?name=llama-swap")[0] == 200
        assert post("/files/mkdir", {"path": "/tmp/agent-unit"})[0] == 200
        assert post("/files/write", {"path": "/tmp/agent-unit/x",
                                     "text": "x", "mode": 0o600})[0] == 200
        assert get("/files/stat?path=/tmp/agent-unit/x")[0] == 200
        assert get("/files/list?path=/tmp/agent-unit")[0] == 200
        assert post("/files/chmod", {"path": "/tmp/agent-unit/x",
                                     "mode": 0o644})[0] == 200
        assert post("/files/backup", {"path": "/tmp/agent-unit/x",
                                      "stamp": "u"})[0] == 200
        assert get("/system/rdma")[0] == 200
        assert get("/system/detail")[0] == 200
        assert get("/metrics")[0] == 200
        assert get("/build/probe?llama_bin=")[0] == 200
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/nope")
        assert e.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/logs/tail?name=../x")
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/port/open?port=0")
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/which?name=bash")
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/service/status?name=x")
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/files/stat?path=/etc/passwd")
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/files/write", {"path": "/etc/passwd", "text": "x"})
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/service/start", {"name": "x"})
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/ray/start", {"ray_bin": "bash", "address": "h:1"})
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/install/artifact", {"urls": ["http://x"], "dest": "/tmp"})
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/build/update", {"src_root": "/nope", "build_dir": "/nope",
                                   "bin_path": "/nope"})
        assert e.value.code == 400
        # /health is open (discovery/pairing); /status still needs auth
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        assert urllib.request.urlopen(req, timeout=5).status == 200
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/status",
                                   timeout=5)
        assert e.value.code == 401
        # invalid JSON / bad content length / speedtest validation
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/rpc/stop", body="{bad",
                     headers={"Authorization": "Bearer t",
                              "Content-Type": "application/json"})
        assert conn.getresponse().status == 400
        conn.close()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/speedtest?mb=0")
        req.add_header("Authorization", "Bearer t")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 400
    finally:
        s.shutdown(); s.server_close(); th.join(timeout=5)


def test_processes_start_stop_paths(monkeypatch, tmp_path):
    from llama_peer_agent import processes as p
    c = AgentConfig(token="t", state_dir=str(tmp_path), log_dir=str(tmp_path))
    monkeypatch.setattr(p, "_port_open", lambda port: True)
    monkeypatch.setattr(p, "list_llama_processes",
                        lambda: [{"pid": 1, "port": 8080, "cmd": "x"}])
    assert p.start_llama(c, 8080, ["-m", "/x"], {}, None)["reused"] is True
    monkeypatch.setattr(p, "list_llama_processes", lambda: [])
    assert "foreign process" in p.start_llama(c, 8080, ["-m", "/x"], {}, None)["error"]
    monkeypatch.setattr(p, "_port_open", lambda port: False)
    monkeypatch.setattr(p, "list_llama_processes",
                        lambda: [{"pid": 2, "port": 8080, "cmd": "x"}])
    assert p.start_llama(c, 8080, ["-m", "/x"], {}, None)["loading"] is True
    monkeypatch.setattr(p, "list_llama_processes", lambda: [])
    monkeypatch.setattr(p, "llama_binary", lambda c: "")
    assert "not found" in p.start_llama(c, 8080, ["-m", "/x"], {}, None)["error"]
    monkeypatch.setattr(p, "llama_binary", lambda c: "/bin/llama-server")
    monkeypatch.setattr(p.subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert "failed to start" in p.start_llama(c, 8080, ["-m", "/x"], {}, None)["error"]
    class Proc:
        pid = 3
        def poll(self): return 1
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **k: Proc())
    res = p.start_llama(c, 8080, ["-m", "/x"], {}, "k")
    assert res["running"] is False and "exited" in res["error"]
    class Proc2:
        pid = 4
        def poll(self): return None
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **k: Proc2())
    port_calls = [False, True]
    monkeypatch.setattr(p, "_port_open",
                        lambda port: port_calls.pop(0) if port_calls else True)
    res = p.start_llama(c, 8080, ["-m", "/x"], {}, "k")
    assert res["running"] is True
    assert p.llama_status(8080)["running"] is True

    # ensure_rpc
    monkeypatch.setattr(p, "list_rpc_processes", lambda: [])
    monkeypatch.setattr(p, "_port_open", lambda port: True)
    monkeypatch.setattr(p, "_read_state", lambda c, port: "default")
    assert p.ensure_rpc(c, 50052, [], {})["reused"] is True
    monkeypatch.setattr(p, "_read_state", lambda c, port: "other")
    monkeypatch.setattr(p, "rpc_binary", lambda c: "")
    assert "not found" in p.ensure_rpc(c, 50052, [], {})["error"]
    monkeypatch.setattr(p, "rpc_binary", lambda c: "/bin/ggml-rpc-server")
    monkeypatch.setattr(p, "_port_open", lambda port: False)
    class RProc:
        pid = 5
        def poll(self): return 1
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **k: RProc())
    assert p.ensure_rpc(c, 50052, [], {})["running"] is False
    monkeypatch.setattr(p, "_port_open", lambda port: True)
    class RProc2:
        pid = 6
        def poll(self): return None
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **k: RProc2())
    assert p.ensure_rpc(c, 50052, [], {})["running"] is True

    # stop paths
    procs = [{"pid": 7, "port": 50052, "cmd": "x"}]
    monkeypatch.setattr(p, "list_rpc_processes", lambda: procs)
    killed = []
    monkeypatch.setattr(p, "_kill_pid", lambda pid, force=False: killed.append(pid))
    p.stop_rpc_port(c, 50052)
    assert 7 in killed
    monkeypatch.setattr(p, "list_llama_processes", lambda: procs)
    p.stop_llama_port(c, 50052, "k")
    monkeypatch.setattr(p, "list_rpc_processes", lambda: [])
    monkeypatch.setattr(p, "list_llama_processes", lambda: [])
    monkeypatch.setattr(p, "list_swap_processes", lambda: [])
    monkeypatch.setattr(p, "_list_processes",
                        lambda names, require_port=True: [])
    assert p.stop_all(c)["killed"] is True
    assert p.stop_swap(c)["killed"] is True


def test_process_listing_and_gpu(monkeypatch):
    from llama_peer_agent import processes as p
    class R:
        stdout = " 101  12  ggml-rpc-server -p 50052\n 102  2 llama-server --port 8080\n"
        returncode = 0
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(p.sys, "platform", "linux")
    assert p.list_rpc_processes()[0]["port"] == 50052
    assert p.list_llama_processes()[0]["port"] == 8080
    monkeypatch.setattr(p.shutil, "which", lambda n: "/bin/nvidia-smi")
    class G:
        returncode = 0
        stdout = "0,GPU,10,100,200,40\n"
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: G())
    assert p.gpu_rows()[0].startswith("0,GPU")
    monkeypatch.setattr(p.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 1})())
    assert p.gpu_rows() == []
    monkeypatch.setattr(p.shutil, "which", lambda n: None)
    assert p.gpu_rows() == []
    # state helpers
    c = AgentConfig(token="t", state_dir=str(p.Path("/tmp")))
    p._write_state(c, 1, "x")
    assert p._read_state(c, 1) == "x"
    assert p._safe_state_key("a/b") == "a-b"
    with pytest.raises(p.AgentProcessError):
        p._safe_state_key("")


def test_agent_client_error_and_branch_paths(monkeypatch):
    import agent_client as ac
    import urllib.error, io
    assert ac.agent_port({"agent_port": "bad"}) == 8198
    assert ac.agent_port({"agent_port": 0}) == 8198
    assert ac.transport({"transport": "bad"}) == "agent"
    monkeypatch.setattr(ac.secrets_vault, "get_machine_agent_token",
                        lambda m: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setenv("AGENT_TOKEN", "a")
    assert ac._token({}) == "a"
    monkeypatch.delenv("AGENT_TOKEN")
    monkeypatch.setenv("FARM_AGENT_TOKEN", "f")
    assert ac._token({}) == "f"
    monkeypatch.delenv("FARM_AGENT_TOKEN")
    assert ac._token({}) == ""
    with pytest.raises(ac.AgentError):
        ac._request({"ip": "1.2.3.4"}, "GET", "/x")
    class HTTPErr(urllib.error.HTTPError):
        def __init__(self, body):
            super().__init__("http://x", 500, "bad", {}, None)
            self.body = body
        def read(self, n=-1):
            if self.body is None:
                raise OSError("x")
            return self.body
    class Opener:
        def __init__(self, exc): self.exc = exc
        def open(self, *a, **k): raise self.exc
    monkeypatch.setenv("FARM_AGENT_TOKEN", "f")
    monkeypatch.setattr(ac, "_OPENER", Opener(HTTPErr(b'{"e":1}')))
    with pytest.raises(ac.AgentError) as e:
        ac._request({"ip": "1.2.3.4"}, "GET", "/x")
    assert "agent HTTP" in str(e.value)
    monkeypatch.setattr(ac, "_OPENER", Opener(HTTPErr(None)))
    with pytest.raises(ac.AgentError):
        ac._request({"ip": "1.2.3.4"}, "GET", "/x")
    monkeypatch.setattr(ac, "_OPENER", Opener(OSError("down")))
    with pytest.raises(ac.AgentError):
        ac._request({"ip": "1.2.3.4"}, "GET", "/x")
    # wrappers not covered yet
    monkeypatch.setattr(ac, "_request", lambda *a, **k: {"ok": True})
    assert ac.version({})["ok"] and ac.capabilities({})["ok"]
    # speedtest paths
    monkeypatch.setattr(ac, "_token", lambda m: "")
    with pytest.raises(ac.AgentError):
        ac.speedtest({})
    monkeypatch.setattr(ac, "_token", lambda m: "t")
    class Resp:
        def __init__(self): self.n = 0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n):
            self.n += 1
            return b"x" * 1024 if self.n == 1 else b""
    class O:
        def open(self, *a, **k): return Resp()
    monkeypatch.setattr(ac, "_OPENER", O())
    assert ac.speedtest({"ip": "1.2.3.4"}) is not None
    class O2:
        def open(self, *a, **k): raise OSError("x")
    monkeypatch.setattr(ac, "_OPENER", O2())
    with pytest.raises(ac.AgentError):
        ac.speedtest({"ip": "1.2.3.4"})
    # should_use_agent cache and rdma
    assert ac.should_use_agent({"rdma_cm": True}) is True
    assert ac.should_use_agent({"name": "c", "ip": "1"}) is True
    assert ac._cache_key({"id": "i"}) == "i"
    assert ac._cache_key({"ip": "ip"}) == "ip"


def test_build_error_paths(monkeypatch, tmp_path):
    from llama_peer_agent import build as b
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    monkeypatch.setattr(b.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(b.AgentBuildError):
        b._run(["x"])
    assert b.probe(c, str(tmp_path / "no" / "bin" / "llama-server"))["kind"] == "missing"
    build = tmp_path / "build"
    build.mkdir()
    assert b.probe(c, str(build / "bin" / "llama-server"))["kind"] == "prebuilt"
    (build / "CMakeCache.txt").write_text("X\n")
    monkeypatch.setattr(b.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 1, "stdout": "", "stderr": "e"})())
    assert b.probe(c, str(build / "bin" / "llama-server"))["kind"] == "prebuilt"
    (build / "CMakeCache.txt").unlink()
    with pytest.raises(b.AgentBuildError):
        b.update(c, str(build), str(build), str(build / "x"))


def test_build_update_errors(monkeypatch, tmp_path):
    from llama_peer_agent import build as b
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    build = tmp_path / "b"
    build.mkdir()
    (build / "CMakeCache.txt").write_text("X\n")
    def run(args, **k):
        cmd = " ".join(args)
        if "fetch" in cmd: return type("P", (), {"returncode": 1, "stdout": "", "stderr": "f"})()
        if "merge-base" in cmd: return type("P", (), {"returncode": 1, "stdout": "", "stderr": ""})()
        if "status" in cmd: return type("P", (), {"returncode": 0, "stdout": " M x", "stderr": ""})()
        if "checkout" in cmd: return type("P", (), {"returncode": 1, "stdout": "", "stderr": "c"})()
        if "cmake" in cmd: return type("P", (), {"returncode": 1, "stdout": "out", "stderr": "err"})()
        if "--version" in cmd: return type("P", (), {"returncode": 0, "stdout": "v", "stderr": ""})()
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(b.subprocess, "run", run)
    with pytest.raises(b.AgentBuildError):
        b.update(c, str(tmp_path), str(build), str(build / "x"))
    # merge-base local-only
    monkeypatch.setattr(b.subprocess, "run",
                        lambda args, **k: type("P", (), {
                            "returncode": 1 if "merge-base" in " ".join(args) else 0,
                            "stdout": "", "stderr": ""})())
    with pytest.raises(b.AgentBuildError):
        b.update(c, str(tmp_path), str(build), str(build / "x"))
    # dirty
    monkeypatch.setattr(b.subprocess, "run",
                        lambda args, **k: type("P", (), {
                            "returncode": 0,
                            "stdout": " M x\n" if "status" in " ".join(args) else "",
                            "stderr": ""})())
    with pytest.raises(b.AgentBuildError):
        b.update(c, str(tmp_path), str(build), str(build / "x"))
    # checkout fail
    monkeypatch.setattr(b.subprocess, "run",
                        lambda args, **k: type("P", (), {
                            "returncode": 1 if "checkout" in " ".join(args) else 0,
                            "stdout": "", "stderr": "c"})())
    with pytest.raises(b.AgentBuildError):
        b.update(c, str(tmp_path), str(build), str(build / "x"))
    # build fail
    monkeypatch.setattr(b.subprocess, "run",
                        lambda args, **k: type("P", (), {
                            "returncode": 1 if "cmake" in " ".join(args) else 0,
                            "stdout": "", "stderr": "e"})())
    assert b.update(c, str(tmp_path), str(build),
                    str(build / "x"), 1)["ok"] is False


def test_install_more_paths(monkeypatch, tmp_path):
    from llama_peer_agent import install as ins
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    with pytest.raises(ins.AgentInstallError):
        ins.artifact(c, [], str(tmp_path / "o"))
    monkeypatch.setattr(ins, "_url", lambda u: u)
    with pytest.raises(ins.AgentInstallError):
        ins.artifact(c, ["https://github.com/x"], str(tmp_path / "o"), "rar")
    with pytest.raises(ins.AgentInstallError):
        ins.verify(c, str(tmp_path / "x"), args=[""])
    binp = tmp_path / "llama-server"
    binp.write_text("x")
    monkeypatch.setattr(ins.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(ins.AgentInstallError):
        ins.verify(c, str(binp))
    # vllm branches
    monkeypatch.setattr(ins.shutil, "which", lambda n: "/bin/" + n)
    monkeypatch.setattr(ins.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 1, "stdout": "", "stderr": "e"})())
    assert ins.vllm(c, "python3", "~/v", "1")["ok"] is False
    monkeypatch.setattr(ins.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": "", "stderr": ""})())
    # existing venv python path
    bindir = tmp_path / "v" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("")
    (bindir / "pip").write_text("")
    assert ins.vllm(c, "python3", str(tmp_path / "v"), "1")["ok"] is True
    with pytest.raises(ins.AgentInstallError):
        ins.vllm(c, str(tmp_path / "nope"), str(tmp_path / "v"))


def test_service_more_paths(monkeypatch, tmp_path):
    from llama_peer_agent import service as sv
    monkeypatch.setattr(sv.Path, "home", lambda: tmp_path)
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    monkeypatch.setattr(sv.shutil, "which", lambda n: "")
    with pytest.raises(sv.AgentServiceError):
        sv._binary("llama-swap")
    monkeypatch.setattr(sv.shutil, "which", lambda n: "/bin/" + n)
    monkeypatch.setattr(sv.os.path, "isfile", lambda p: False)
    with pytest.raises(sv.AgentServiceError):
        sv._binary("/bin/llama-swap")
    monkeypatch.setattr(sv.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(sv.processes, "list_swap_processes",
                        lambda: [{"pid": 1, "port": 1}])
    assert sv.start(c, "llama-swap", "/bin/llama-swap",
                    str(tmp_path / "c"), 1)["running"] is True
    monkeypatch.setattr(sv.processes, "list_swap_processes", lambda: [])
    monkeypatch.setattr(sv, "_popen", lambda *a, **k: 1)
    monkeypatch.setattr(sv.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 1})())
    assert sv.start(c, "llama-swap", "/bin/llama-swap",
                    str(tmp_path / "c"), 1)["pid"] == 1
    monkeypatch.setattr(sv, "_popen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(sv.AgentServiceError):
        sv.start(c, "llama-swap", "/bin/llama-swap",
                 str(tmp_path / "c"), 1)
    monkeypatch.setattr(sv.processes, "stop_swap", lambda c: {"killed": True})
    assert sv.stop(c, "llama-swap")["killed"] is True
    monkeypatch.setattr(sv.processes, "list_swap_processes",
                        lambda: [{"pid": 2, "port": 2}])
    assert sv.status(c, "llama-swap")["pids"] == [2]
    monkeypatch.setattr(sv.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(sv.AgentServiceError):
        sv.validate(c, "/bin/llama-swap", str(tmp_path / "c"))


def test_system_more_paths(monkeypatch, tmp_path):
    from llama_peer_agent import system as sy
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    monkeypatch.setattr(sy, "_run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        sy._linux(c)
    monkeypatch.setattr(sy, "_run", lambda *a, **k: (0, "", ""))
    assert sy._parse_posix("x") == {}
    out = json.dumps({"kernel": "w", "cpu": "c", "cores": 2,
                      "disk": {"Free": 1}, "llama": "v"})
    monkeypatch.setattr(sy.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": out, "returncode": 0})())
    assert sy._windows(c)["disk"]["free"] == "1"
    monkeypatch.setattr(sy.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": "bad", "returncode": 0})())
    assert sy._windows(c) == {}
    monkeypatch.setattr(sy.processes.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sy.os, "name", "nt")
    assert sy.detail(c) == {}
    monkeypatch.undo()


def test_ray_more_paths(monkeypatch, tmp_path):
    from llama_peer_agent import ray as r
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    monkeypatch.setattr(r.shutil, "which", lambda n: "")
    with pytest.raises(r.AgentRayError):
        r._ray_bin("ray")
    monkeypatch.setattr(r.shutil, "which", lambda n: "/bin/ray")
    monkeypatch.setattr(r.os.path, "isfile", lambda p: False)
    with pytest.raises(r.AgentRayError):
        r._ray_bin("/bin/ray")
    monkeypatch.setattr(r.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(r, "_ray_bin", lambda v: "/bin/ray")
    with pytest.raises(r.AgentRayError):
        r.start(c, "ray", "", 1)
    with pytest.raises(r.AgentRayError):
        r.start(c, "ray", "h:1", -1)
    with pytest.raises(r.AgentRayError):
        r.start(c, "ray", "h:1", 1, {"bad-key": 1})
    monkeypatch.setattr(r.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(r.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 1})())
    assert r.start(c, "ray", "h:1", 1, {}, "", "~/l", "~/p")["pid"] == 1
    assert r.stop(c, "ray", "~/p")["stopped"] is True
    assert r.stop(c, "ray", None)["stopped"] is True


def test_discovery_start_and_errors(monkeypatch):
    from llama_peer_agent import discovery as d
    c = AgentConfig(token="t", host="127.0.0.1", port=1,
                    discovery_host="127.0.0.1", discovery_port=0)
    sock = d.create_socket(c)
    thread, s2, stop = d.start(c)
    stop.set(); s2.close(); thread.join(timeout=2)
    sock.close()
    class Bad:
        def recvfrom(self, n): return (b"bad", ("1", 1))
        def sendto(self, *a): pass
        def close(self): pass
    stop = __import__("threading").Event()
    stop.set()
    d.serve(c, Bad(), stop)


def test_build_remaining(monkeypatch, tmp_path):
    from llama_peer_agent import build as b
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    build = tmp_path / "b"
    build.mkdir()
    cache = build / "CMakeCache.txt"
    cache.write_text("NOEQUALS\nGGML_CUDA:BOOL=ON\n")
    monkeypatch.setattr(b.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": str(tmp_path),
                            "stderr": ""})())
    assert b.probe(c, str(build / "bin" / "llama-server"))["flags"]["GGML_CUDA"] == "ON"
    monkeypatch.setattr(b.Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert b.probe(c, str(build / "bin" / "llama-server"))["kind"] == "source"
    monkeypatch.undo()
    # jobs None path
    def run(args, **k):
        cmd = " ".join(args)
        out = "v" if "--version" in cmd else ""
        return type("P", (), {"returncode": 0, "stdout": out,
                              "stderr": ""})()
    monkeypatch.setattr(b.subprocess, "run", run)
    assert b.update(c, str(tmp_path), str(build),
                    str(build / "bin" / "llama-server"), None)["ok"] is True


def test_files_remaining(monkeypatch, tmp_path):
    from llama_peer_agent import files as f
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    with pytest.raises(f.AgentFileError):
        f._path(c, "a\x00b")
    with pytest.raises(f.AgentFileError):
        f.write(c, str(tmp_path / "a"), 123)
    assert f.write(c, str(tmp_path / "rel"), "x")["path"].endswith("rel")
    chmod_calls = {"n": 0}
    def chmod(*a):
        chmod_calls["n"] += 1
        if chmod_calls["n"] > 1:
            raise OSError("x")
    monkeypatch.setattr(f.os, "chmod", chmod)
    assert f.write(c, str(tmp_path / "m"), "x", mode=0o600)["bytes"] == 1
    monkeypatch.undo()
    monkeypatch.setattr(f.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        f.write(c, str(tmp_path / "f"), "x")
    monkeypatch.undo()
    monkeypatch.setattr(f.Path, "resolve", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert f._roots(c) == []
    monkeypatch.undo()
    class BadEntry:
        name = "x"
        def is_dir(self): return False
        def is_file(self): raise OSError("x")
    class BadPath:
        def is_dir(self): return True
        def iterdir(self): return [BadEntry()]
    monkeypatch.setattr(f, "_path", lambda *a, **k: BadPath())
    assert f.list_dir(c, "/x")["entries"] == []


def test_metrics_remaining(monkeypatch):
    from llama_peer_agent import metrics as m
    monkeypatch.setattr(m.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert m._run(["x"])[0] == 1
    monkeypatch.setattr(m, "_run", lambda *a, **k: (1, "", ""))
    assert m._gpus() == []
    monkeypatch.setattr(m, "_run", lambda *a, **k: (0, "bad\n", ""))
    assert m._gpus() == []
    monkeypatch.setattr(m, "_run", lambda *a, **k: (0, "0,G,1,2\n", ""))
    assert m._gpus() == []
    # open failures
    import builtins
    real_open = builtins.open
    def bad_open(path, *a, **k):
        if str(path).startswith("/proc/") or path in ("/proc/stat", "/proc/loadavg", "/proc/meminfo"):
            raise OSError("x")
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", bad_open)
    assert m._linux()["status"] == "online"
    monkeypatch.undo()
    monkeypatch.setattr(m, "_run", lambda *a, **k: (1, "", ""))
    assert m._macos()["status"] == "online"
    assert m._windows()["status"] == "offline"


def test_service_remaining(monkeypatch, tmp_path):
    from llama_peer_agent import service as s
    c = AgentConfig(token="t", file_roots=[str(tmp_path)], log_dir=str(tmp_path))
    monkeypatch.setattr(s.shutil, "which", lambda n: "/bin/" + n)
    monkeypatch.setattr(s.os.path, "isfile", lambda p: True)
    with pytest.raises(s.AgentServiceError):
        s.start(c, "llama-swap", "/bin/llama-swap", str(tmp_path / "c"), "x")
    with pytest.raises(s.AgentServiceError):
        s.start(c, "llama-swap", "/bin/llama-swap", str(tmp_path / "c"), 0)
    monkeypatch.setattr(s.processes, "list_swap_processes",
                        lambda: [{"pid": 1, "port": 1}])
    assert s.start(c, "llama-swap", "/bin/llama-swap",
                   str(tmp_path / "c"), 1)["method"] == "already-running"
    monkeypatch.setattr(s.processes, "list_swap_processes", lambda: [])
    monkeypatch.setattr(s.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 2})())
    monkeypatch.setattr(s.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setattr(s, "Path", s.Path)
    assert s._popen(c, "/bin/llama-swap", str(tmp_path / "c"), 1) == 2
    monkeypatch.setattr(s.processes, "stop_swap", lambda c: {"killed": True})
    assert s.stop(c, "llama-swap")["killed"] is True


def test_system_remaining(monkeypatch, tmp_path):
    from llama_peer_agent import system as sy
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    assert sy._parse_posix("x") == {}
    monkeypatch.setattr(sy, "_run", lambda *a, **k: (0, "K:k\n", ""))
    monkeypatch.setattr(sy.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": "4", "returncode": 0})())
    assert sy._macos(c)["cores"] == 4
    out = json.dumps({"disk": [{"Free": 1}], "coreload": "[1]"})
    monkeypatch.setattr(sy.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": out, "returncode": 0})())
    assert sy._windows(c)["disk"]["free"] == "1"
    monkeypatch.setattr(sy.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "stdout": out, "returncode": 1})())
    assert sy._windows(c) == {}
    monkeypatch.setattr(sy, "_run", lambda *a, **k: (0, "bad\n", ""))
    assert sy.rdma() == {}


def test_ray_remaining(monkeypatch, tmp_path):
    from llama_peer_agent import ray as r
    c = AgentConfig(token="t", file_roots=[str(tmp_path)])
    monkeypatch.setattr(r, "_ray_bin", lambda v: "/bin/ray")
    monkeypatch.setattr(r.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert r._iface_for("1.1.1.1") == ""
    monkeypatch.setattr(r.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stdout": "x", "stderr": ""})())
    assert r._iface_for("1.1.1.1") == ""
    with pytest.raises(r.AgentRayError):
        r.start(c, "ray", "h:1", 1, {"A": object()})
    with pytest.raises(r.AgentRayError):
        r.start(c, "ray", "h:1", 1, "bad")
    monkeypatch.setattr(r.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 1})())
    assert r.start(c, "ray", "h:1", 1, {}, "", "~/l", "~/p")["pid"] == 1
    assert r.stop(c, "ray", "~/p")["stopped"] is True


def test_discovery_remaining(monkeypatch):
    from llama_peer_agent import discovery as d
    c = AgentConfig(token="t", host="127.0.0.1", port=1,
                    discovery_host="127.0.0.1", discovery_port=0)
    class S:
        def recvfrom(self, n): raise __import__("socket").timeout()
        def sendto(self, *a): pass
        def close(self): pass
    stop = __import__("threading").Event()
    stop.set()
    d.serve(c, S(), stop)
    class S2:
        def recvfrom(self, n): return (d.MAGIC, ("1", 1))
        def sendto(self, *a): raise OSError("x")
        def close(self): pass
    stop = __import__("threading").Event()
    # one iteration then stop via sendto? use monkeypatch event
    class Ev:
        def __init__(self): self.n = 0
        def is_set(self):
            self.n += 1
            return self.n > 1
    d.serve(c, S2(), Ev())


def test_processes_windows_and_macos_listing(monkeypatch):
    from llama_peer_agent import processes as p
    class R:
        def __init__(self, out="", code=0):
            self.stdout = out; self.returncode = code
    monkeypatch.setattr(p.os, "name", "nt")
    monkeypatch.setattr(p.subprocess, "run",
                        lambda *a, **k: R(json.dumps({
                            "Pid": 1, "Age": 2, "Name": "llama-server.exe",
                            "Cmd": "--port 8080"})))
    assert p.list_llama_processes()[0]["port"] == 8080
    monkeypatch.setattr(p.subprocess, "run",
                        lambda *a, **k: R('{"Pid":"bad","Name":"llama-server.exe"}'))
    assert p.list_llama_processes() == []
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: R("bad"))
    assert p.list_llama_processes() == []
    monkeypatch.setattr(p.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert p.list_llama_processes() == []
    monkeypatch.undo()
    monkeypatch.setattr(p.sys, "platform", "darwin")
    monkeypatch.setattr(p.subprocess, "run",
                        lambda *a, **k: R(" 1 01:02:03 llama-server --port 8080\n"
                                          "bad\n 2 bad llama-server\n"))
    assert p.list_llama_processes()[0]["port"] == 8080


def test_processes_stop_and_log_errors(monkeypatch, tmp_path):
    from llama_peer_agent import processes as p
    c = AgentConfig(token="t", log_dir=str(tmp_path), state_dir=str(tmp_path))
    # log open failure
    real_open = open
    def bad_open(path, *a, **k):
        if "rpc-server" in str(path) or "llama-server" in str(path):
            raise OSError("x")
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", bad_open)
    monkeypatch.setattr(p, "list_llama_processes", lambda: [])
    monkeypatch.setattr(p, "_port_open", lambda port: False)
    monkeypatch.setattr(p, "llama_binary", lambda c: "/bin/llama-server")
    assert "cannot open log" in p.start_llama(c, 8099, ["-m", "/x"], {}, None)["error"]
    monkeypatch.setattr(p, "rpc_binary", lambda c: "/bin/ggml-rpc-server")
    monkeypatch.setattr(p, "list_rpc_processes", lambda: [])
    assert "cannot open log" in p.ensure_rpc(c, 50099, [], {})["error"]
    monkeypatch.undo()
    # state file with bad line and unlink error
    path = p._llama_state_path(c, "k")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("bad\n")
    monkeypatch.setattr(p, "list_llama_processes", lambda: [])
    monkeypatch.setattr(p, "_kill_pid", lambda *a, **k: None)
    assert p.stop_llama_port(c, 1, "k")["killed"] is True
    # Windows kill path
    monkeypatch.setattr(p.os, "name", "nt")
    monkeypatch.setattr(p.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0})())
    p._kill_pid(1)
    monkeypatch.undo()
    # port open invalid
    with pytest.raises(p.AgentProcessError):
        p.port_open("x")


def test_discovery_payload_includes_hardware(monkeypatch):
    """UDP discovery carries hardware so a discovered agent prefills the form."""
    from llama_peer_agent import discovery, processes

    monkeypatch.setattr(processes, "info",
                        lambda c: {"mac": "de:ad:be:ef:12:34",
                                   "agent_port": 8198})
    monkeypatch.setattr(metrics, "collect", lambda: {
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "mem_total_gb": 24}],
        "ram_total_gb": 64, "cores": 16})
    discovery._hw_cache.update({"ts": 0.0, "data": {}})
    body = discovery.payload(None)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not body["hardware"]:
        time.sleep(0.05)
        body = discovery.payload(None)
    hw = body["hardware"]
    assert hw["gpu_vendor"] == "nvidia"
    assert hw["gpu_names"] == ["NVIDIA GeForce RTX 4090"]
    assert hw["vram_total_gb"] == 24
    assert hw["ram_total_gb"] == 64
    assert body["mac"] == "de:ad:be:ef:12:34"


def test_metrics_amd_intel_apple_windows(monkeypatch):
    import llama_peer_agent.metrics as mm
    monkeypatch.setattr(mm.shutil, "which", lambda n: n if n in (
        "rocm-smi", "xpu-smi") else None)
    monkeypatch.setattr(mm, "_run", lambda args, timeout=20, shell=False: (
        0,
        '{"card0":{"Card series":"Radeon RX 7900 XTX",'
        '"VRAM Total Memory (B)":"25753026560",'
        '"VRAM Total Used Memory (B)":"1073741824"}}'
        if "rocm-smi" in args else
        '{"device_list":[{"device_name":"Intel Arc A770",'
        '"memory_physical_size_byte":"17179869184"}]}', ""))
    rows = mm._amd_gpus()
    assert rows and rows[0]["name"] == "Radeon RX 7900 XTX"
    assert rows[0]["mem_total_gb"] == 24.0
    rows = mm._intel_gpus()
    assert rows and rows[0]["name"] == "Intel Arc A770"
    assert rows[0]["mem_total_gb"] == 16.0

    monkeypatch.setattr(mm.sys, "platform", "darwin")
    monkeypatch.setattr(mm, "_run", lambda *a, **k: (
        0, '{"SPDisplaysDataType":[{"sppci_model":"Apple M3 Pro",'
           '"spdisplays_vram":"18 GB"}]}', ""))
    rows = mm._apple_gpus()
    assert rows and rows[0]["name"] == "Apple M3 Pro"
    assert rows[0]["mem_total_gb"] == 18.0

    monkeypatch.setattr(mm.os, "name", "nt")
    monkeypatch.setattr(mm, "_run", lambda *a, **k: (
        0, '[{"Name":"AMD Radeon RX 6800","AdapterRAM":17179869184}]', ""))
    rows = mm._win_fallback_gpus()
    assert rows and rows[0]["name"] == "AMD Radeon RX 6800"
    assert mm._vendor_from_names([rows[0]["name"]]) == "amd"


def test_agent_client_token_override():
    import agent_client as ac
    assert ac._token({"name": "x", "_agent_token": "typed"}) == "typed"
