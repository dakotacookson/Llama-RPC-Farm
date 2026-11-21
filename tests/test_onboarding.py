"""Onboarding helpers: recipes, safety, pairing, clients, metrics."""
import os
import sys
import types

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import onboarding


def test_recipes_have_required_fields_and_unique_ids():
    ids = set()
    for r in onboarding.RECIPES:
        assert r["id"] not in ids
        ids.add(r["id"])
        assert r["repo"] and r["file"] and r["model_id"]
        assert r["kind"] in ("chat", "embeddings", "audio")


def test_recipe_fit_uses_vram_then_ram():
    hw = {"vram_gb": 8, "ram_gb": 64}
    assert onboarding._recipe_fits(
        {"min_vram_gb": 6, "min_ram_gb": 32}, hw)
    assert onboarding._recipe_fits(
        {"min_vram_gb": 24, "min_ram_gb": 16}, hw)   # RAM fallback
    assert not onboarding._recipe_fits(
        {"min_vram_gb": 24, "min_ram_gb": 128}, hw)


def test_client_configs_include_openai_and_anthropic():
    cfgs = {c["id"]: c for c in onboarding.client_configs(
        "http://farm:8090", "qwen3-4b")}
    assert "/v1/chat/completions" in cfgs["openai-curl"]["snippet"]
    assert "/v1/messages" in cfgs["anthropic-curl"]["snippet"]
    assert "ANTHROPIC_BASE_URL" in cfgs["claude-code"]["snippet"]
    assert cfgs["opencode"]["kind"] == "json"


def test_client_configs_are_model_type_aware():
    emb = onboarding.client_configs("http://farm:8090", "nomic", "embeddings")
    assert any("/v1/embeddings" in c["snippet"]
               or "embeddings.create" in c["snippet"] for c in emb)
    claude = next(c for c in emb if c["id"] == "claude-code")
    assert claude["compatible"] is False

    dec = onboarding.client_configs("http://farm:8090", "laya", "decision")
    assert any("/v1/systemone" in c["snippet"] for c in dec)
    assert next(c for c in dec if c["id"] == "claude-code")["compatible"] is False

    audio = onboarding.client_configs("http://farm:8090", "asr", "audio")
    assert any("input_audio" in c["snippet"] for c in audio)

    vision = onboarding.client_configs(
        "http://farm:8090", "vl", ["chat", "vision"])
    assert any("image_url" in c["snippet"] for c in vision)
    assert next(c for c in vision if c["id"] == "claude-code")["compatible"] is True


def test_safety_flags_no_keys_on_lan(monkeypatch):
    monkeypatch.setattr(onboarding.access, "access_keys", lambda: {})
    monkeypatch.setattr(onboarding, "_machine_count", lambda: 1)
    monkeypatch.setattr(onboarding, "_vpn_status",
                        lambda: {"found": [], "tailscale": False,
                                 "wireguard": False})
    monkeypatch.setattr(onboarding, "_listen_addresses",
                        lambda p: {"0.0.0.0"} if p in (8199, 8090)
                        else {"127.0.0.1"})
    rep = onboarding.safety_report()
    assert rep["ok"] is False
    keys = next(c for c in rep["checks"] if c["id"] == "keys")
    assert keys["level"] == "bad"
    assert any(c["level"] == "bad" for c in rep["checks"]
               if c["id"].startswith("port-"))


def test_safety_ok_with_keys_and_loopback(monkeypatch):
    monkeypatch.setattr(onboarding.access, "access_keys",
                        lambda: {"sk-one": {"name": "one"}})
    monkeypatch.setattr(onboarding, "_machine_count", lambda: 1)
    monkeypatch.setattr(onboarding, "_vpn_status",
                        lambda: {"found": [], "tailscale": False,
                                 "wireguard": False})
    monkeypatch.setattr(onboarding, "_listen_addresses", lambda p: {"127.0.0.1"})
    rep = onboarding.safety_report()
    assert rep["ok"] is True


def test_safety_detects_vpn_tunnel(monkeypatch):
    monkeypatch.setattr(onboarding.access, "access_keys",
                        lambda: {"sk-one": {"name": "one"}})
    monkeypatch.setattr(onboarding, "_listen_addresses", lambda p: {"127.0.0.1"})
    monkeypatch.setattr(onboarding, "_machine_count", lambda: 3)
    monkeypatch.setattr(onboarding, "_vpn_status",
                        lambda: {"found": ["WireGuard"], "tailscale": False,
                                 "wireguard": True})
    rep = onboarding.safety_report()
    vpn = next(c for c in rep["checks"] if c["id"] == "vpn")
    assert vpn["level"] == "ok"
    assert "WireGuard" in vpn["detail"]
    assert "port forwarding" in vpn["detail"]


def test_safety_no_vpn_multi_machine_warns_about_port_forwarding(monkeypatch):
    monkeypatch.setattr(onboarding.access, "access_keys",
                        lambda: {"sk-one": {"name": "one"}})
    monkeypatch.setattr(onboarding, "_listen_addresses", lambda p: {"127.0.0.1"})
    monkeypatch.setattr(onboarding, "_machine_count", lambda: 3)
    monkeypatch.setattr(onboarding, "_vpn_status",
                        lambda: {"found": [], "tailscale": False,
                                 "wireguard": False})
    rep = onboarding.safety_report()
    vpn = next(c for c in rep["checks"] if c["id"] == "vpn")
    assert vpn["level"] == "warn"
    assert "Do not port-forward" in vpn["detail"]
    rpc = next(c for c in rep["checks"] if c["id"] == "rpc")
    assert rpc["level"] == "warn"


def test_safety_single_machine_rpc_is_inactive(monkeypatch):
    monkeypatch.setattr(onboarding.access, "access_keys",
                        lambda: {"sk-one": {"name": "one"}})
    monkeypatch.setattr(onboarding, "_listen_addresses", lambda p: {"127.0.0.1"})
    monkeypatch.setattr(onboarding, "_machine_count", lambda: 1)
    monkeypatch.setattr(onboarding, "_vpn_status",
                        lambda: {"found": [], "tailscale": False,
                                 "wireguard": False})
    rep = onboarding.safety_report()
    vpn = next(c for c in rep["checks"] if c["id"] == "vpn")
    assert vpn["level"] == "info"
    rpc = next(c for c in rep["checks"] if c["id"] == "rpc")
    assert rpc["level"] == "ok"
    assert "no RPC peers" in rpc["detail"]


def test_pairing_commands_use_a_unique_one_machine_token():
    out = onboarding.pairing_commands()
    other = onboarding.pairing_commands()
    assert out["token"] != other["token"]
    assert len(out["token"]) >= 32
    assert "fleet" in out["note"]
    assert "PEER_AGENT_TOKEN" in out["unix"]
    assert out["token"] in out["unix"]
    assert "$env:PEER_AGENT_TOKEN" in out["windows"]


def test_create_access_key_writes_settings(monkeypatch, tmp_path):
    state = {"settings": {"access_keys": {}}}
    monkeypatch.setattr(onboarding, "_load_settings",
                        lambda: dict(state["settings"]))
    monkeypatch.setattr(onboarding, "_save_settings",
                        lambda d: state.update(settings=d))
    monkeypatch.setattr(onboarding.access, "generate_token",
                        lambda prefix="sk-farm": "sk-test")
    token = onboarding.create_access_key("quickstart")
    assert token == "sk-test"
    assert state["settings"]["access_keys"]["sk-test"]["name"] == "quickstart"


def test_metrics_mark_and_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding, "data_dir", lambda: str(tmp_path))
    onboarding.mark("setup_shown")
    onboarding.mark("first_model_ready", {"recipe": "demo-0.6b"})
    s = onboarding.metrics_summary()
    assert s["events"]["setup_shown"]
    assert s["events"]["first_model_ready"]
    assert s["time_to_first_model_s"] is not None
    assert os.path.exists(tmp_path / "config" / "onboarding_metrics.json")


def test_ensure_default_store_creates_root(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(onboarding, "_store_alias", lambda: "llm-models")
    monkeypatch.setattr(onboarding.path_resolver, "_load_stores",
                        lambda: {})
    monkeypatch.setattr(onboarding.path_resolver, "save_store",
                        lambda a, p, r: saved.update(alias=a, platform=p,
                                                     root=r))
    monkeypatch.setenv("LLAMA_FARM_MODELS_DIR", str(tmp_path / "models"))
    root = onboarding.ensure_default_store()
    assert root == str(tmp_path / "models")
    assert saved["alias"] == "llm-models"
    assert os.path.isdir(root)


def test_apply_worker_saves_config_and_warms(monkeypatch, tmp_path):
    saved = {}
    launch_calls = []

    monkeypatch.setattr(onboarding, "ensure_default_store", lambda: str(tmp_path))
    monkeypatch.setattr(onboarding, "_local_machine",
                        lambda name=None: {
                            "name": "host", "gpu": "CUDA0",
                            "detected": {"gpu_names": ["RTX"]}})
    monkeypatch.setattr(onboarding, "_store_alias", lambda: "llm-models")
    import hf_downloader
    monkeypatch.setattr(hf_downloader, "start_shards",
                        lambda link, store: {
                            "jobs": [],
                            "bundle": {"name": "bundle", "dir": str(tmp_path)}})
    monkeypatch.setattr(onboarding, "_wait_download", lambda res, jid: None)
    monkeypatch.setattr(onboarding, "_find_companion", lambda d, p: "")
    monkeypatch.setattr(onboarding.cfg, "save_model",
                        lambda mid, data: saved.update(mid=mid, data=data))
    monkeypatch.setattr(onboarding.swap_manager, "launch",
                        lambda: launch_calls.append(1) or {"failed": False})
    monkeypatch.setattr(onboarding, "_gateway_post",
                        lambda path, body: {"choices": [{"message":
                                                         {"content": "OK"}}]})
    monkeypatch.setattr(onboarding, "mark", lambda *a, **k: None)

    recipe = onboarding.get_recipe("demo-0.6b")
    onboarding._apply_worker("job-test", recipe, None, True)
    j = onboarding._job("job-test")
    assert j["done"] is True
    assert not j.get("error")
    assert saved["mid"] == recipe["model_id"]
    assert saved["data"]["base_path"].startswith("store://llm-models/bundle/")
    assert saved["data"]["flags"]["n-gpu-layers"] == 99
    assert launch_calls


def test_apply_worker_reports_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding, "ensure_default_store", lambda: str(tmp_path))
    monkeypatch.setattr(onboarding, "_local_machine",
                        lambda name=None: {"name": "host",
                                           "detected": {}})
    import hf_downloader
    monkeypatch.setattr(hf_downloader, "start_shards",
                        lambda link, store: (_ for _ in ()).throw(
                            RuntimeError("download boom")))
    recipe = onboarding.get_recipe("demo-0.6b")
    onboarding._apply_worker("job-fail", recipe, None, True)
    j = onboarding._job("job-fail")
    assert j["done"] is True
    assert "download boom" in (j["error"] or "")


def test_quickstart_cli_help():
    import subprocess
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src", "quickstart.py")
    r = subprocess.run([sys.executable, src, "--help"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "--recipe" in r.stdout


def test_safety_reports_rpc_mtls_enabled(monkeypatch):
    import rpc_tls
    monkeypatch.setattr(onboarding.access, "access_keys",
                        lambda: {"sk-one": {"name": "one"}})
    monkeypatch.setattr(onboarding, "_listen_addresses", lambda p: {"127.0.0.1"})
    monkeypatch.setattr(onboarding, "_machine_count", lambda: 3)
    monkeypatch.setattr(onboarding, "_vpn_status",
                        lambda: {"found": ["Tailscale"], "tailscale": True,
                                 "wireguard": False})
    monkeypatch.setattr(rpc_tls, "enabled", lambda: True)
    rep = onboarding.safety_report()
    rpc = next(c for c in rep["checks"] if c["id"] == "rpc")
    assert rpc["level"] == "ok"
    assert "mTLS" in rpc["detail"]


def test_safety_warns_when_peers_share_farm_token(monkeypatch):
    monkeypatch.setattr(onboarding.access, "access_keys",
                        lambda: {"sk-one": {"name": "one"}})
    monkeypatch.setattr(onboarding, "_listen_addresses", lambda p: {"127.0.0.1"})
    monkeypatch.setattr(onboarding, "_machine_count", lambda: 3)
    monkeypatch.setattr(onboarding, "_vpn_status",
                        lambda: {"found": ["Tailscale"], "tailscale": True,
                                 "wireguard": False})
    monkeypatch.setattr(onboarding, "_shared_agent_token_count", lambda: 2)
    rep = onboarding.safety_report()
    tok = next(c for c in rep["checks"] if c["id"] == "agent-tokens")
    assert tok["level"] == "warn"
    assert "farm token" in tok["detail"]


def test_wait_download_actually_waits_for_job_status(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(onboarding, "_job_update", lambda *a, **k: None)
    monkeypatch.setattr(onboarding.time, "sleep", lambda s: None)
    def state(_jid):
        calls["n"] += 1
        return ("downloading", None) if calls["n"] < 3 else ("done", None)
    monkeypatch.setattr(onboarding, "_job_download_state", state)
    import hf_downloader
    monkeypatch.setitem(hf_downloader.BUNDLES, "b1", {
        "name": "bundle", "status": "done"})
    onboarding._wait_download(
        {"jobs": [{"id": "j1"}], "bundle": {"name": "bundle"}}, "job")
    assert calls["n"] >= 3
