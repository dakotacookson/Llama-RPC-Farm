"""Static checks for distribution assets we cannot run on a bare-metal CI box.

These do not replace a real GPU smoke test; they catch the mistakes that
would make the Docker/Proxmox/Unraid/winget/Homebrew paths obviously broken.
"""
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(*parts):
    return os.path.join(ROOT, *parts)


def _read(*parts):
    with open(_path(*parts), "r", encoding="utf-8") as f:
        return f.read()


def test_dockerfiles_have_runtime_dependencies():
    farm = _read("Dockerfile")
    agent = _read("Dockerfile.agent")
    assert "libgomp1" in farm, "CPU llama.cpp needs libgomp at runtime"
    assert "LLAMA_FARM_MODELS_DIR=/data/models" in farm
    assert "src/app.py" in farm
    assert "libgomp1" in agent
    assert "llama_peer_agent" in agent
    assert "util-linux" in agent


def test_compose_profiles_and_gpu_mappings():
    data = yaml.safe_load(_read("docker-compose.yml"))
    services = data["services"]
    for name in ("farm", "demo", "demo-wizard", "gpu", "gpu-rocm",
                 "gpu-intel", "agent", "agent-nvidia", "agent-rocm"):
        assert name in services, f"missing compose service {name}"
        assert services[name].get("profiles"), f"{name} needs a profile"
    assert services["farm"]["profiles"] == ["solo"]

    assert services["demo"]["environment"]["FARM_DEMO_AUTO"] == "1"
    assert services["demo-wizard"]["environment"]["FARM_DEMO_AUTO"] == "0"
    assert services["demo-wizard"]["environment"][
        "LLAMA_FARM_GPU_VENDOR"] == "cpu"

    nvidia = services["gpu"]["deploy"]["resources"]["reservations"][
        "devices"][0]
    assert nvidia["driver"] == "nvidia"
    assert "gpu" in nvidia["capabilities"]

    rocm = services["gpu-rocm"]
    assert "/dev/kfd" in rocm["devices"]
    assert "/dev/dri" in rocm["devices"]
    assert {"video", "render"} <= set(rocm["group_add"])
    assert rocm["environment"]["LLAMA_FARM_GPU_VENDOR"] == "amd"

    intel = services["gpu-intel"]
    assert "/dev/dri" in intel["devices"]
    assert intel["environment"]["LLAMA_FARM_GPU_VENDOR"] == "intel"

    for name in ("agent", "agent-nvidia", "agent-rocm"):
        assert services[name]["network_mode"] == "host"
        assert "PEER_AGENT_TOKEN" in services[name]["environment"]


def test_compose_demo_does_not_require_agent_token():
    try:
        cfg = subprocess.run(
            ["docker", "compose", "--profile", "demo", "config"],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return  # Docker not installed on this runner
    if cfg.returncode != 0 and "docker: not found" in (cfg.stderr or ""):
        return
    assert cfg.returncode == 0, cfg.stderr[-1000:]


def test_quickstart_scripts_and_syntax():
    sh = _path("scripts", "quickstart.sh")
    ps = _path("scripts", "quickstart.ps1")
    assert os.path.isfile(sh) and os.access(sh, os.X_OK)
    assert os.path.isfile(ps)
    assert "src/quickstart.py" in _read("scripts", "quickstart.sh")
    assert "src\\quickstart.py" in _read("scripts", "quickstart.ps1")
    for script in ("scripts/quickstart.sh", "docker/demo-entrypoint.sh",
                   "packaging/proxmox/install-llama-rpc-farm.sh"):
        r = subprocess.run(["bash", "-n", _path(script)],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, f"{script}: {r.stderr}"


def test_ghcr_workflow_and_windows_build_version():
    docker_wf = yaml.safe_load(_read(".github", "workflows", "docker.yml"))
    assert "build" in docker_wf["jobs"]
    assert "demo-smoke" in docker_wf["jobs"]
    text = _read(".github", "workflows", "rolling-main.yml")
    assert "$env:BUILD_VERSION = $env:VERSION" in text
    assert "$env:APP_VERSION = $env:VERSION" in text


def test_packaging_templates_parse():
    yaml.safe_load(_read("packaging", "winget",
                         "DakotaCookson.LlamaRPCFarm.yaml"))
    yaml.safe_load(_read("packaging", "casaos", "docker-compose.yml"))
    ET.parse(_path("packaging", "unraid", "llama-rpc-farm.xml"))
    assert "sha256 :no_check" in _read("packaging", "homebrew",
                                       "llama-rpc-farm.rb")
    assert "install-llama-rpc-farm" in _read(
        "docs", "distribution.md")


def test_public_api_setup_assets():
    sh = _path("scripts", "setup-public-nginx.sh")
    assert os.path.isfile(sh) and os.access(sh, os.X_OK)
    text = _read("scripts", "setup-public-nginx.sh")
    assert "certbot certonly --webroot" in text
    assert "listen 443 ssl" in text
    assert "proxy_set_header X-Forwarded-For" in text
    assert "127.0.0.1:8090" in text
    assert "--env-file" in text
    assert "FARM_GATEWAY_REQUIRE_KEYS" in text
    r = subprocess.run(["bash", "-n", sh], capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 0, r.stderr
    assert os.path.isfile(_path("config", "runtime.env.example"))
    for script in ("run.sh", "restart.sh"):
        assert "config/runtime.env" in _read(script)
    assert "Tailscale" in _read("docs", "public-api.md")
    assert "443" in _read("docs", "public-api.md")


def test_onboarding_docs_exist():
    for rel in ("docs/quickstart.md", "docs/distribution.md"):
        assert os.path.isfile(_path(rel)), rel
    assert "five-minute" in _read("docs", "quickstart.md").lower()
    assert "demo-wizard" in _read("docs", "distribution.md")
