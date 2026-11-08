#!/usr/bin/env python3
"""Single-node end-to-end test for the *installed* app + agent packages.

Run on each release runner after installing the .deb/.pkg/setup.exe:

    python tests/e2e/single_node.py \
        --app /usr/local/bin/llama-rpc-farm \
        --agent /usr/local/bin/llama-peer-agent

It:
  1. downloads CPU llama.cpp + llama-swap + a tiny Qwen2.5-0.5B GGUF,
  2. starts the installed agent and checks /health,
  3. starts the installed app against a throwaway data dir,
  4. launches llama-swap through the app, loads the model through the
     real :8090 gateway, and asserts a chat completion,
  5. tears everything down.

No GPU is required (CPU backend). Designed for GitHub-hosted runners.
"""
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "e2e-agent-token-0123456789"
MODEL_REPO = "unsloth/Qwen3-0.6B-GGUF"
MODEL_FILE = "Qwen3-0.6B-Q4_K_M.gguf"
APP_PORT = 8199
GATEWAY_PORT = 8090
SWAP_PORT = 8080
AGENT_PORT = 18198


def _free_port(start, used):
    for port in range(int(start), int(start) + 200):
        if port in used:
            continue
        try:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            address = sock.getsockname()
            sock.close()
            if address[1] == port:
                used.add(port)
                return port
        except OSError:
            continue
    raise RuntimeError(f"no free port near {start}")


def log(msg):
    print(f"[e2e] {msg}", flush=True)


def _os_name():
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _arch():
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


def _is_win():
    return os.name == "nt"


def http(method, url, body=None, token=None, timeout=30):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read() or b"{}"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1200]
        raise RuntimeError(f"HTTP {e.code} {method} {url}: {detail}") from e
    try:
        return json.loads(raw.decode())
    except Exception:
        return {"raw": raw.decode("utf-8", "replace")}


def wait_json(url, timeout, token=None, what="", ok=lambda d: True):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        try:
            d = http("GET", url, token=token, timeout=5)
            if ok(d):
                return d
        except Exception as e:
            last = e
        time.sleep(1)
    raise RuntimeError(f"{what or url} did not become ready: {last}")


def download_model(work):
    from huggingface_hub import hf_hub_download
    log(f"model {MODEL_REPO}/{MODEL_FILE}")
    path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))
    # Copy out of the HF cache: cache entries are symlinks, and the app /
    # llama-swap run under an isolated HOME in this test.
    dest = work / "models" / MODEL_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(path, dest)
    return dest


def write_config(work, host, model_path):
    """Write the minimum config needed to boot the app.

    llama.cpp and llama-swap are intentionally NOT written here: the E2E
    test drives the app's own installer flow (/api/tools/install/sync) and
    lets the app persist their real paths.
    """
    import yaml
    data = work / "data"
    cfg_dir = data / "config"
    (cfg_dir / "models").mkdir(parents=True, exist_ok=True)
    (work / "swap").mkdir(parents=True, exist_ok=True)
    (work / "models").mkdir(parents=True, exist_ok=True)

    settings = {
        "model_dir": str(work / "models"),
        "llama_bin": "",
        "theme_family": "nord",
        "theme_mode": "dark",
        "engine": "llama_cpp",
    }
    (cfg_dir / "settings.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    machine = {
        "id": host, "name": host, "ip": "127.0.0.1", "username": "",
        "password": "", "port": 22, "role": "master", "os": _os_name(),
        "rpc_port": 50052, "gpu": "", "gpu_vendor": "cpu",
        "llama_bin": "", "ram_total_gb": 8,
        "expose": {"mode": "both", "gpus": []},
    }
    (cfg_dir / "machines.yaml").write_text(
        yaml.safe_dump({"machines": [machine]}, skip_keys=False) if False
        else yaml.safe_dump({"machines": [machine]}, sort_keys=False),
        encoding="utf-8")

    swap_cfg = {
        "binary": "llama-swap",
        "listen_port": SWAP_PORT,
        "config_path": str(work / "swap" / "config.yaml"),
        "host_name": host,
    }
    (cfg_dir / "swap.yaml").write_text(
        yaml.safe_dump(swap_cfg, sort_keys=False), encoding="utf-8")

    model = {
        "model_id": "e2e-tiny", "name": "e2e-tiny",
        "base_path": str(model_path),
        "tensor_split_mode": "auto", "tensor_split": "",
        "fit_margin": 1024, "fit_ram_allow": 1024,
        "servers": [host], "group": "", "pinned": False, "ttl": 0,
        "device_order": "", "device_order_policy": "as-selected",
        "expose": {},
        "flags": {"ctx-size": 2048, "n-gpu-layers": 0, "parallel": 1},
        "reset": False, "variants": [],
    }
    (cfg_dir / "models" / "e2e-tiny.yaml").write_text(
        yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return data


def stop_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if _is_win():
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


def isolate_env(work):
    """Keep the app/agent/engines away from the runner's real user dirs."""
    home = work / "home"
    (home / ".config").mkdir(parents=True, exist_ok=True)
    bindir = work / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "PATH": f"{bindir}{os.pathsep}" + (os.environ.get("PATH") or ""),
    }
    if _is_win():
        # The Windows installer paths use %USERPROFILE% / APPDATA / TEMP, not
        # HOME. Redirect them or the E2E installs engines into the runner's
        # real profile.
        for key, sub in (("USERPROFILE", "home"),
                         ("APPDATA", "home/AppData/Roaming"),
                         ("LOCALAPPDATA", "home/AppData/Local"),
                         ("TEMP", "tmp"), ("TMP", "tmp")):
            path = work / sub
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
    else:
        # swap_manager writes ~/.config/systemd/user/llama-swap.service and
        # calls systemctl --user. Point HOME at the sandbox and stub
        # systemctl so it falls back to a detached llama-swap process.
        for name in ("systemctl", "loginctl"):
            stub = bindir / name
            stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            stub.chmod(0o755)
    return env


def main():
    global APP_PORT, GATEWAY_PORT, SWAP_PORT, AGENT_PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--app-port", type=int, default=APP_PORT)
    ap.add_argument("--gateway-port", type=int, default=GATEWAY_PORT)
    ap.add_argument("--swap-port", type=int, default=SWAP_PORT)
    ap.add_argument("--agent-port", type=int, default=AGENT_PORT)
    args = ap.parse_args()
    used = set()
    APP_PORT = _free_port(args.app_port, used)
    GATEWAY_PORT = _free_port(args.gateway_port, used)
    SWAP_PORT = _free_port(args.swap_port, used)
    AGENT_PORT = _free_port(args.agent_port, used)

    work = Path(tempfile.mkdtemp(prefix="llama-farm-e2e-"))
    # Set the data-dir override BEFORE importing src modules so their
    # import-time config_loader never touches the repo's live config.
    os.environ["LLAMA_FARM_DATA_DIR"] = str(work / "data")
    sys.path.insert(0, str(ROOT / "src"))

    host = socket.gethostname().split(".")[0].lower()
    app_proc = agent_proc = None
    app_log = open(work / "app.log", "w", encoding="utf-8")
    agent_log = open(work / "agent.log", "w", encoding="utf-8")
    try:
        log(f"platform={_os_name()} arch={_arch()} host={host}")
        model_path = download_model(work)
        data = write_config(work, host, model_path)

        # ---- agent package ----
        isolated = isolate_env(work)
        agent_env = dict(os.environ, **isolated,
                         PEER_AGENT_TOKEN=TOKEN,
                         PEER_AGENT_PORT=str(AGENT_PORT),
                         PEER_AGENT_DISCOVERY_PORT=str(AGENT_PORT + 1),
                         PEER_AGENT_DISCOVERY="0",
                         PEER_AGENT_NO_SERVICE="1",
                         PEER_AGENT_PAIRING="0",
                         PEER_AGENT_LOG_DIR=str(work / "agent-logs"),
                         PEER_AGENT_STATE_DIR=str(work / "agent-state"))
        agent_proc = subprocess.Popen(
            [args.agent, "--host", "127.0.0.1", "--port", str(AGENT_PORT),
             "--token", TOKEN, "--no-discovery"],
            env=agent_env, stdout=agent_log, stderr=subprocess.STDOUT)
        health = wait_json(f"http://127.0.0.1:{AGENT_PORT}/health", 60,
                           token=TOKEN, what="agent /health")
        assert health.get("ok") is True, health
        ver = http("GET", f"http://127.0.0.1:{AGENT_PORT}/version", token=TOKEN)
        log(f"agent OK {ver.get('agent_version')} protocol={ver.get('protocol_version')}")

        # ---- app package ----
        app_env = dict(os.environ, **isolated,
                       LLAMA_FARM_DATA_DIR=str(data),
                       LLAMA_FARM_PORT=str(APP_PORT),
                       GATEWAY_PORT=str(GATEWAY_PORT),
                       GATEWAY_HOST="127.0.0.1",
                       GATEWAY_UPSTREAM=f"http://127.0.0.1:{SWAP_PORT}",
                       LLAMA_FARM_NO_SERVICE="1",
                       # Keep the E2E's declared CPU machine authoritative;
                       # otherwise the host's real GPUs get auto-detected and
                       # a CPU llama.cpp build is launched with --device CUDA*.
                       LLAMA_FARM_NO_HW_WATCH="1")
        app_proc = subprocess.Popen(
            [args.app], cwd=str(data), env=app_env, stdout=app_log,
            stderr=subprocess.STDOUT)
        wait_json(f"http://127.0.0.1:{APP_PORT}/api/machines", 90,
                  what="app UI")
        log("app UI OK")

        # ---- install llama.cpp + llama-swap THROUGH THE APP ----
        install = http(
            "POST", f"http://127.0.0.1:{APP_PORT}/api/tools/install/sync",
            body={"machine": host, "role": "main",
                  "tools": ["llama.cpp", "llama-swap"], "dry_run": False},
            timeout=1200)
        if install.get("failures"):
            raise RuntimeError(f"app install failed: {install['failures']}")
        installed = set(install.get("installed") or [])
        assert {"llama.cpp", "llama-swap"} <= installed, install
        log(f"app installed engines: {sorted(installed)}")
        verify = http("GET", f"http://127.0.0.1:{APP_PORT}/api/tools/verify")
        for tool in ("llama.cpp", "llama-swap"):
            assert verify[tool]["found"], verify[tool]
            log(f"{tool} -> {verify[tool]['path']} {verify[tool]['version']}")

        # ---- llama-swap via the app, load through the real gateway ----
        launch = http("POST", f"http://127.0.0.1:{APP_PORT}/api/swap/launch",
                      body={}, timeout=180)
        if launch.get("failed"):
            raise RuntimeError(f"swap launch failed: {launch}")
        wait_json(f"http://127.0.0.1:{APP_PORT}/api/swap/status", 90,
                  what="llama-swap running",
                  ok=lambda d: d.get("running"))
        log("llama-swap running")

        load = None
        last_err = None
        for attempt in range(1, 4):
            try:
                load = http(
                    "POST",
                    f"http://127.0.0.1:{APP_PORT}/api/swap/load/e2e-tiny",
                    body=None, timeout=600)
                if load.get("ok"):
                    break
                last_err = RuntimeError(f"model load failed: {load}")
            except Exception as e:
                last_err = e
            log(f"model load attempt {attempt} failed: {last_err}")
            time.sleep(3)
        if not load or not load.get("ok"):
            raise RuntimeError(f"model load failed after retries: {last_err}")
        log(f"model loaded in {load.get('elapsed_s')}s")

        models = http("GET", f"http://127.0.0.1:{GATEWAY_PORT}/v1/models")
        ids = [m.get("id") for m in models.get("data", [])]
        assert "e2e-tiny" in ids, ids
        log(f"gateway /v1/models sees e2e-tiny ({len(ids)} models)")

        chat = http("POST",
                    f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
                    body={"model": "e2e-tiny", "max_tokens": 64,
                          "messages": [{"role": "user", "content": "Say hi"}]},
                    timeout=300)
        msg = ((chat.get("choices") or [{}])[0].get("message") or {})
        answer = msg.get("content") or msg.get("reasoning_content") or ""
        assert isinstance(answer, str) and answer.strip(), chat
        log(f"chat OK: {answer[:80]!r}")

        # clean shutdown via the app, then terminate processes
        try:
            http("POST", f"http://127.0.0.1:{APP_PORT}/api/swap/stop",
                 body={}, timeout=120)
        except Exception:
            pass
        log("E2E PASSED")
        return 0
    except Exception as e:
        log(f"E2E FAILED: {e}")
        try:
            app_log.flush()
            agent_log.flush()
            print("\n--- app.log ---", flush=True)
            print((work / "app.log").read_text(errors="replace")[-6000:])
            print("\n--- agent.log ---", flush=True)
            print((work / "agent.log").read_text(errors="replace")[-3000:])
        except Exception:
            pass
        return 1
    finally:
        stop_tree(app_proc)
        stop_tree(agent_proc)
        app_log.close()
        agent_log.close()
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    _rc = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    # Never hang CI on a lingering library thread.
    os._exit(_rc)
