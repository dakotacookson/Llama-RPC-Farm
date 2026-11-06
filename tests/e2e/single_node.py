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
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "e2e-agent-token-0123456789"
MODEL_REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read() or b"{}"
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


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log(f"download {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "llama-farm-e2e"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def extract_tar(path, dest, strip=0):
    with tarfile.open(path, "r:gz") as tar:
        for item in tar.getmembers():
            parts = Path(item.name).parts[strip:] if strip else Path(item.name).parts
            if not parts:
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.issym():
                # upstream ships lib*.so.0 -> lib*.so.0.x.y symlinks; skipping
                # them leaves llama-server unable to find its shared libs.
                try:
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    os.symlink(item.linkname, target)
                except OSError:
                    pass
                continue
            if item.islnk():
                try:
                    os.link(dest.joinpath(*Path(item.linkname).parts[strip:])
                            if strip else dest.joinpath(*Path(item.linkname).parts),
                            target)
                except OSError:
                    pass
                continue
            if not item.isfile():
                continue
            src = tar.extractfile(item)
            if src is None:
                continue
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def extract_zip(path, dest, strip=0):
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts[strip:] if strip else Path(info.filename).parts
            if not parts:
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def find_binary(root, name):
    for p in Path(root).rglob(name):
        if p.is_file():
            return p
    return None


def install_llama_cpp(work):
    import llamacpp_installer as li
    machine = {"os": _os_name(), "gpu_vendor": "cpu",
               "detected": {"arch": _arch()}}
    tag, assets, base = li._fetch_latest_assets()
    names, dest, note, is_zip = li._pick_assets(machine, tag, assets, None)
    out = work / "llama-cpp"
    out.mkdir(parents=True, exist_ok=True)
    dl = work / "downloads"
    for name in names:
        p = download(f"{base}/{name}", dl / name)
        if is_zip:
            extract_zip(p, out, 0)
        else:
            extract_tar(p, out, 1)
    if not _is_win():
        for p in out.rglob("*"):
            if p.is_file():
                try:
                    os.chmod(p, 0o755)
                except OSError:
                    pass
    log(f"llama.cpp {tag}: {note}")
    return out


def install_swap(work):
    import swap_installer as si
    machine = {"os": _os_name(), "gpu_vendor": "cpu",
               "detected": {"arch": _arch()}}
    tag, assets, base = si._fetch_latest_assets()
    name, dest, note = si._pick_asset(machine, tag, assets)
    out = work / "llama-swap"
    out.mkdir(parents=True, exist_ok=True)
    p = download(f"{base}/{name}", work / "downloads" / name)
    if name.endswith(".zip"):
        extract_zip(p, out, 0)
    else:
        extract_tar(p, out, 0)
    binary = find_binary(out, "llama-swap.exe" if _is_win() else "llama-swap")
    if not binary:
        raise RuntimeError(f"llama-swap binary missing from {name}")
    if not _is_win():
        binary.chmod(0o755)
    log(f"llama-swap {tag}: {note}")
    return binary


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


def write_config(work, host, llama_dir, swap_bin, model_path):
    import yaml
    data = work / "data"
    cfg_dir = data / "config"
    (cfg_dir / "models").mkdir(parents=True, exist_ok=True)
    (work / "swap").mkdir(parents=True, exist_ok=True)

    settings = {
        "model_dir": str(work / "models"),
        "llama_bin": str(llama_dir),
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
        "llama_bin": str(llama_dir), "ram_total_gb": 8,
        "expose": {"mode": "both", "gpus": []},
    }
    (cfg_dir / "machines.yaml").write_text(
        yaml.safe_dump({"machines": [machine]}, sort_keys=False),
        encoding="utf-8")

    swap_cfg = {
        "binary": str(swap_bin),
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
        "llama_bin": str(llama_dir),
        "servers": [host], "group": "", "pinned": False, "ttl": 0,
        "device_order": "", "device_order_policy": "as-selected",
        "expose": {},
        "flags": {"ctx-size": 2048, "n-gpu-layers": 0, "parallel": 1},
        "vllm": None, "reset": False, "variants": [],
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
    """Keep the app/agent away from the runner's real user services."""
    home = work / "home"
    (home / ".config").mkdir(parents=True, exist_ok=True)
    bindir = work / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    if not _is_win():
        # swap_manager writes ~/.config/systemd/user/llama-swap.service and
        # calls systemctl --user. Point HOME at the sandbox and stub
        # systemctl so it falls back to a detached llama-swap process.
        for name in ("systemctl", "loginctl"):
            stub = bindir / name
            stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            stub.chmod(0o755)
    return {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "PATH": f"{bindir}{os.pathsep}" + (os.environ.get("PATH") or ""),
    }


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
        llama_dir = install_llama_cpp(work)
        swap_bin = install_swap(work)
        model_path = download_model(work)
        data = write_config(work, host, llama_dir, swap_bin, model_path)

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
                       LLAMA_FARM_NO_SERVICE="1")
        app_proc = subprocess.Popen(
            [args.app], env=app_env, stdout=app_log,
            stderr=subprocess.STDOUT)
        wait_json(f"http://127.0.0.1:{APP_PORT}/api/machines", 90,
                  what="app UI")
        log("app UI OK")

        # ---- llama-swap via the app, load through the real gateway ----
        launch = http("POST", f"http://127.0.0.1:{APP_PORT}/api/swap/launch",
                      body={}, timeout=180)
        if launch.get("failed"):
            raise RuntimeError(f"swap launch failed: {launch}")
        wait_json(f"http://127.0.0.1:{APP_PORT}/api/swap/status", 90,
                  what="llama-swap running",
                  ok=lambda d: d.get("running"))
        log("llama-swap running")

        load = http("POST",
                    f"http://127.0.0.1:{APP_PORT}/api/swap/load/e2e-tiny",
                    body=None, timeout=600)
        if not load.get("ok"):
            raise RuntimeError(f"model load failed: {load}")
        log(f"model loaded in {load.get('elapsed_s')}s")

        models = http("GET", f"http://127.0.0.1:{GATEWAY_PORT}/v1/models")
        ids = [m.get("id") for m in models.get("data", [])]
        assert "e2e-tiny" in ids, ids
        log(f"gateway /v1/models sees e2e-tiny ({len(ids)} models)")

        chat = http("POST",
                    f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
                    body={"model": "e2e-tiny", "max_tokens": 8,
                          "messages": [{"role": "user", "content": "Say hi"}]},
                    timeout=300)
        answer = ((chat.get("choices") or [{}])[0].get("message") or {}).get(
            "content")
        assert isinstance(answer, str), chat
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
    raise SystemExit(main())
