"""First-five-minutes onboarding helpers: recipes, safety, clients, pairing.

Kept out of routes so the quickstart CLI, the setup wizard, and the HTTP
layer share one implementation.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

import access
import config_loader as cfg
import local_node
import model_sync
import path_resolver
import settings_store
import swap_manager
from runtime_paths import data_dir

# ---------------------------------------------------------------------------
# Curated starter recipes
# ---------------------------------------------------------------------------

RECIPES = [
    {
        "id": "demo-0.6b",
        "name": "Demo — Qwen3 0.6B",
        "description": "Tiny CPU-friendly model. Proves the farm end to end; "
                       "runs on any machine.",
        "repo": "unsloth/Qwen3-0.6B-GGUF",
        "file": "Qwen3-0.6B-Q4_K_M.gguf",
        "model_id": "qwen3-0-6b-demo",
        "kind": "chat",
        "size_gb": 0.4,
        "min_vram_gb": 0,
        "min_ram_gb": 2,
        "tags": ["demo", "chat"],
        "flags": {"ctx-size": 4096, "parallel": 1, "jinja": True},
    },
    {
        "id": "qwen3-4b",
        "name": "Qwen3 4B",
        "description": "Daily-driver chat model for a single 6–8 GB GPU "
                       "or a CPU/RAM-only box.",
        "repo": "unsloth/Qwen3-4B-GGUF",
        "file": "Qwen3-4B-Q4_K_M.gguf",
        "model_id": "qwen3-4b",
        "kind": "chat",
        "size_gb": 2.5,
        "min_vram_gb": 6,
        "min_ram_gb": 8,
        "tags": ["chat", "daily"],
        "flags": {"ctx-size": 16384, "parallel": 1, "jinja": True},
    },
    {
        "id": "qwen3-8b",
        "name": "Qwen3 8B",
        "description": "Stronger single-box model; about 8 GB VRAM at "
                       "Q4_K_M with a moderate context.",
        "repo": "unsloth/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q4_K_M.gguf",
        "model_id": "qwen3-8b",
        "kind": "chat",
        "size_gb": 5.0,
        "min_vram_gb": 10,
        "min_ram_gb": 16,
        "tags": ["chat", "quality"],
        "flags": {"ctx-size": 32768, "parallel": 1, "jinja": True},
    },
    {
        "id": "qwen3-32b",
        "name": "Qwen3 32B",
        "description": "Big-model recipe for a pooled multi-box RPC farm "
                       "(or one 24 GB+ card with RAM spill).",
        "repo": "unsloth/Qwen3-32B-GGUF",
        "file": "Qwen3-32B-Q4_K_M.gguf",
        "model_id": "qwen3-32b",
        "kind": "chat",
        "size_gb": 20.0,
        "min_vram_gb": 24,
        "min_ram_gb": 32,
        "tags": ["chat", "big", "rpc"],
        "flags": {"ctx-size": 32768, "parallel": 1, "jinja": True},
    },
    {
        "id": "nomic-embed",
        "name": "Nomic Embed v1.5",
        "description": "Embeddings model for RAG and the Embeddings test tab.",
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "file": "nomic-embed-text-v1.5.f16.gguf",
        "model_id": "nomic-embed-text-v1-5",
        "kind": "embeddings",
        "size_gb": 0.6,
        "min_vram_gb": 0,
        "min_ram_gb": 2,
        "tags": ["embeddings", "rag"],
        "flags": {"embeddings": True, "ctx-size": 8192,
                  "batch-size": 8192, "ubatch-size": 8192},
    },
    {
        "id": "qwen3-asr",
        "name": "Qwen3 ASR 1.7B",
        "description": "Speech recognition with the matching mmproj; feeds "
                       "the Audio test tab.",
        "repo": "ggml-org/Qwen3-ASR-1.7B-GGUF",
        "file": "Qwen3-ASR-1.7B-bf16.gguf",
        "model_id": "qwen3-asr-1-7b-recipe",
        "kind": "audio",
        "size_gb": 4.0,
        "min_vram_gb": 6,
        "min_ram_gb": 8,
        "tags": ["audio", "asr"],
        "flags": {"ctx-size": 8192, "parallel": 1, "jinja": True,
                  "backend-sampling": True},
    },
]

_RECIPE_BY_ID = {r["id"]: r for r in RECIPES}
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()


def list_recipes():
    hw = fleet_hardware()
    out = []
    for r in RECIPES:
        item = dict(r)
        item["fit"] = _recipe_fits(r, hw)
        out.append(item)
    return out


def get_recipe(recipe_id):
    r = _RECIPE_BY_ID.get((recipe_id or "").strip())
    if not r:
        raise ValueError(f"unknown recipe '{recipe_id}'")
    return dict(r)


def fleet_hardware():
    """Aggregate detected VRAM/RAM across the whole farm."""
    vram = 0.0
    ram = 0.0
    gpus = []
    for m in cfg.machines.all():
        det = m.get("detected") or {}
        try:
            vram += float(det.get("vram_total_gb") or 0)
        except (TypeError, ValueError):
            pass
        try:
            ram += float(det.get("ram_total_gb")
                         or m.get("ram_total_gb") or 0)
        except (TypeError, ValueError):
            pass
        for i, name in enumerate(det.get("gpu_names") or []):
            gpus.append({"machine": m.get("name"), "index": i,
                         "name": name})
    return {"vram_gb": round(vram, 1), "ram_gb": round(ram, 1),
            "gpus": gpus}


def _recipe_fits(r, hw):
    need_v = float(r.get("min_vram_gb") or 0)
    need_r = float(r.get("min_ram_gb") or 0)
    if need_v and hw["vram_gb"] >= need_v:
        return True
    return hw["ram_gb"] >= need_r


def _load_settings():
    return settings_store._load_settings_yaml_preserving(
        settings_store.SETTINGS_FILE) or {}


def _save_settings(data):
    settings_store._dump_settings_yaml_preserving(
        data, settings_store.SETTINGS_FILE)


def _store_alias():
    settings = _load_settings()
    md = str(settings.get("model_dir") or "store://llm-models")
    if md.startswith(path_resolver.PREFIX):
        rest = md[len(path_resolver.PREFIX):]
        return rest.split("/", 1)[0] or "llm-models"
    return "llm-models"


def ensure_default_store():
    """Give a fresh install a local model root when stores.yaml is empty.

    The setup wizard normally asks for this; the quickstart/Docker demo
    must work with zero questions, so pick a sensible per-user directory.
    """
    alias = _store_alias()
    stores = path_resolver._load_stores()
    roots = stores.get(alias) or {}
    if roots.get(path_resolver.LOCAL_PLATFORM):
        return roots[path_resolver.LOCAL_PLATFORM]
    root = (os.environ.get("LLAMA_FARM_MODELS_DIR") or "").strip()
    if not root:
        root = os.path.join(os.path.expanduser("~"), "llama-models")
    os.makedirs(root, exist_ok=True)
    path_resolver.save_store(alias, path_resolver.LOCAL_PLATFORM, root)
    return root


def _local_machine(name=None):
    if name:
        return cfg.machines.require(name)
    for m in cfg.machines.all():
        if local_node.is_local(m):
            return m
    raise ValueError("no local machine configured")


def _gpu_box(machine):
    det = machine.get("detected") or {}
    return bool(det.get("gpu_names") or
                [g for g in (machine.get("gpu") or "").split(",") if g.strip()])


def _job(job_id):
    with _JOBS_LOCK:
        return dict(_JOBS.get(job_id) or {})


def _job_update(job_id, **kw):
    with _JOBS_LOCK:
        j = _JOBS.setdefault(job_id, {})
        j.update(kw)
        j["updated"] = time.time()


def _job_log(job_id, msg):
    with _JOBS_LOCK:
        j = _JOBS.setdefault(job_id, {})
        j.setdefault("log", []).append(str(msg)[:500])
        j["log"] = j["log"][-80:]


def apply_recipe(recipe_id, machine_name=None, launch=True):
    """Start a recipe install (download → config → launch → warmup)."""
    r = get_recipe(recipe_id)
    job_id = f"recipe-{int(time.time() * 1000)}-{r['id']}"[:80]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id, "recipe": r["id"], "name": r["name"],
            "phase": "starting", "done": False, "error": None,
            "model_id": r.get("model_id"), "started": time.time(),
            "log": [],
        }
    t = threading.Thread(target=_apply_worker,
                         args=(job_id, r, machine_name, bool(launch)),
                         daemon=True)
    t.start()
    return job_id


def _wait_download(res, job_id):
    jobs = res.get("jobs") or []
    ids = [j.get("id") for j in jobs if j.get("id")]
    deadline = time.time() + 7200
    while time.time() < deadline:
        states = [(_job_download_state(i)) for i in ids]
        if any(s == "downloading" for s in states):
            done = sum(1 for s in states if s == "done")
            _job_update(job_id, phase="downloading",
                        progress=round(done / max(1, len(states)), 2))
            time.sleep(1)
            continue
        err = next((e for s, e in states if s == "error"), None)
        if err:
            raise RuntimeError(err)
        return
    raise RuntimeError("download timed out after 2 hours")


def _job_download_state(job_id):
    import hf_downloader as _hf
    j = _hf.JOBS.get(job_id) or {}
    st = j.get("status")
    if st == "error":
        return "error", (j.get("error") or "download failed")
    if st in ("done", "cancelled"):
        return "done", None
    return "downloading", None


def _find_companion(bundle_dir, prefix):
    try:
        for root, _dirs, files in os.walk(bundle_dir):
            for f in files:
                if f.lower().startswith(prefix) and f.lower().endswith(".gguf"):
                    return os.path.relpath(os.path.join(root, f), bundle_dir)
    except OSError:
        pass
    return ""


def _gateway_post(path, payload, timeout=600):
    from web_shared import _gateway_url
    req = urllib.request.Request(
        _gateway_url(path),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def _apply_worker(job_id, r, machine_name, launch):
    try:
        machine = _local_machine(machine_name)
        host = machine.get("name")
        ensure_default_store()
        _job_log(job_id, f"machine: {host}")
        store = _store_alias()
        link = f"hf://{r['repo']}/{r['file']}"
        _job_update(job_id, phase="downloading", progress=0)
        _job_log(job_id, f"download: {link}")
        import hf_downloader
        res = hf_downloader.start_shards(link, store)
        _wait_download(res, job_id)
        bundle = res.get("bundle") or {}
        bname = bundle.get("name") or ""
        bdir = bundle.get("dir") or ""
        if not bname or not bdir:
            raise RuntimeError("download finished without a bundle path")
        _job_update(job_id, phase="saving")
        base = f"store://{store}/{bname}/{r['file']}"
        flags = dict(r.get("flags") or {})
        gpu = _gpu_box(machine)
        flags.setdefault("n-gpu-layers", 99 if gpu else 0)
        if r["kind"] == "embeddings":
            flags["embeddings"] = True
        mm = _find_companion(bdir, "mmproj")
        if mm:
            flags["mmproj"] = f"store://{store}/{bname}/{mm}"
        data = {
            "model_id": r["model_id"],
            "name": r["name"],
            "base_path": base,
            "tensor_split_mode": "fit" if gpu else "auto",
            "tensor_split": "",
            "fit_margin": 1024,
            "fit_ram_allow": 1024,
            "servers": [host],
            "group": "",
            "pinned": False,
            "ttl": 0,
            "device_order": "",
            "device_order_policy": "as-selected",
            "launch_mode": "auto",
            "expose": {},
            "flags": flags,
            "variants": [],
            "reset": False,
        }
        cfg.save_model(r["model_id"], data)
        _job_log(job_id, f"saved model {r['model_id']} -> {base}")
        if not launch:
            _job_update(job_id, phase="done", done=True,
                        result={"model_id": r["model_id"],
                                "base_path": base})
            return
        _job_update(job_id, phase="launching")
        report = swap_manager.launch()
        failed = report.get("failed") or []
        if failed:
            raise RuntimeError(f"llama-swap launch failed: {failed}")
        _job_log(job_id, "llama-swap launched")
        _job_update(job_id, phase="warming")
        last = None
        for attempt in range(4):
            try:
                if r["kind"] == "embeddings":
                    _gateway_post("/v1/embeddings",
                                  {"model": r["model_id"], "input": "hi"})
                else:
                    body = _gateway_post("/v1/chat/completions", {
                        "model": r["model_id"],
                        "messages": [{"role": "user",
                                      "content": "Reply with OK."}],
                        "max_tokens": 4, "stream": False})
                    if not body:
                        raise RuntimeError("warmup returned no response")
                last = None
                break
            except Exception as e:                      # noqa: BLE001
                last = e
                _job_log(job_id, f"warmup retry {attempt + 1}: {e}")
                if attempt == 1:
                    # The app's boot reconcile can race the first launch;
                    # regenerate/push once and let the daemon settle.
                    try:
                        swap_manager.launch()
                    except Exception:
                        pass
                time.sleep(2)
        if last is not None:
            raise RuntimeError(f"warmup failed after retries: {last}")
        _job_log(job_id, "warmup complete")
        mark("first_model_ready", {"recipe": r["id"]})
        _job_update(job_id, phase="done", done=True,
                    result={"model_id": r["model_id"],
                            "base_path": base})
    except Exception as e:                              # noqa: BLE001
        _job_log(job_id, f"FAILED: {e}")
        _job_update(job_id, phase="error", done=True, error=str(e))


# ---------------------------------------------------------------------------
# Access key for the ready screen / quickstart
# ---------------------------------------------------------------------------

def create_access_key(name="quickstart", models=None, admin=False):
    token = access.generate_token()
    current = _load_settings()
    keys = current.get("access_keys") or {}
    if not isinstance(keys, dict):
        keys = {}
    keys[token] = {
        "name": (name or "quickstart").strip() or "quickstart",
        "models": [m.strip() for m in (models or []) if str(m).strip()],
        "admin": bool(admin),
        "admin_unrestricted": True,
    }
    current["access_keys"] = keys
    _save_settings(current)
    return token


# ---------------------------------------------------------------------------
# Peer pairing command
# ---------------------------------------------------------------------------

def pairing_commands():
    """One unique enroll token per generated command.

    The token is bound to whichever one machine you run the command on;
    paste it into that machine's Agent token field when adding it. It is
    deliberately NOT the fleet-wide token.
    """
    import secrets
    tok = secrets.token_urlsafe(32)
    raw = "https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main"
    q = shlex.quote(tok)
    unix = (f"curl -fsSL {raw}/scripts/install-remote-agent.sh "
            f"| PEER_AGENT_TOKEN={q} bash")
    win = (f"$env:PEER_AGENT_TOKEN='{tok}'; "
           f"irm {raw}/scripts/install-remote-agent.ps1 | iex")
    return {
        "token": tok,
        "unix": unix,
        "windows": win,
        "note": "This command contains a one-machine token, not the fleet "
                "token. Keep it private, run it on one host, and paste the "
                "token into that machine's Agent token field when adding it.",
    }


# ---------------------------------------------------------------------------
# Client connection snippets
# ---------------------------------------------------------------------------

def client_configs(base_url="http://127.0.0.1:8090", model="your-model",
                   kinds="chat"):
    """Copy-paste snippets for the model's actual endpoint families.

    kinds may be a string or a list: chat, vision, audio, embeddings,
    decision. Vision models include chat. Chat clients stay visible for
    every model, but are marked compatible=false when they do not apply.
    """
    base = (base_url or "http://127.0.0.1:8090").rstrip("/")
    if isinstance(kinds, str):
        kinds = [kinds]
    kinds = {str(k or "").lower() for k in kinds if str(k or "").strip()}
    if not kinds:
        kinds = {"chat"}
    key = "<your-key>"
    def _mark(items, compatible=True):
        out = []
        for c in items:
            d = dict(c)
            d["compatible"] = compatible
            out.append(d)
        return out
    chat_clients = _mark([
        {"id": "openai-curl", "name": "OpenAI / curl", "kind": "http",
         "description": "Any OpenAI-compatible client.",
         "snippet": (
             f"curl {base}/v1/chat/completions \\\n"
             f"  -H 'Authorization: Bearer {key}' \\\n"
             f"  -H 'Content-Type: application/json' \\\n"
             f"  -d '{{\"model\":\"{model}\",\"messages\":"
             f"[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'")},
        {"id": "openai-python", "name": "OpenAI Python", "kind": "python",
         "description": "OpenAI SDK, pointed at the farm gateway.",
         "snippet": (
             "from openai import OpenAI\n"
             f"client = OpenAI(base_url='{base}/v1', api_key='{key}')\n"
             "r = client.chat.completions.create(\n"
             f"    model='{model}',\n"
             "    messages=[{'role': 'user', 'content': 'hi'}],\n"
             ")\nprint(r.choices[0].message.content)")},
        {"id": "anthropic-curl", "name": "Anthropic / curl", "kind": "http",
         "description": "Anthropic Messages API, same key and grants.",
         "snippet": (
             f"curl {base}/v1/messages \\\n"
             f"  -H 'x-api-key: {key}' \\\n"
             f"  -H 'anthropic-version: 2023-06-01' \\\n"
             f"  -H 'Content-Type: application/json' \\\n"
             f"  -d '{{\"model\":\"{model}\",\"max_tokens\":256,"
             f"\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'")},
        {"id": "claude-code", "name": "Claude Code", "kind": "env",
         "description": "Point Claude Code at the farm gateway.",
         "snippet": (
             f"export ANTHROPIC_BASE_URL={base}\n"
             f"export ANTHROPIC_AUTH_TOKEN={key}\n"
             f"export ANTHROPIC_MODEL={model}")},
        {"id": "opencode", "name": "opencode", "kind": "json",
         "description": "OpenAI-compatible provider config.",
         "snippet": json.dumps({
             "provider": {"llamaswap": {
                 "npm": "@ai-sdk/openai-compatible",
                 "options": {"baseURL": f"{base}/v1",
                             "apiKey": key,
                             "timeout": 1800000,
                             "headerTimeout": 300000,
                             "chunkTimeout": 600000}}}},
             indent=2)},
        {"id": "open-webui", "name": "Open WebUI", "kind": "env",
         "description": "Add an OpenAI connection with these values.",
         "snippet": (f"Base URL: {base}/v1\n"
                     f"API key: {key}\n"
                     f"Model: {model}")},
        {"id": "librechat", "name": "LibreChat", "kind": "yaml",
         "description": "Custom OpenAI-compatible endpoint.",
         "snippet": (f"endpoints:\n  custom:\n    - name: 'Llama RPC Farm'\n"
                     f"      apiKey: '{key}'\n"
                     f"      baseURL: '{base}/v1'\n"
                     f"      models:\n        - '{model}'")},
        {"id": "continue", "name": "Continue", "kind": "json",
         "description": "Continue / VS Code config.",
         "snippet": json.dumps({
             "models": [{"title": model, "provider": "openai",
                         "model": model, "apiBase": f"{base}/v1",
                         "apiKey": key}]}, indent=2)},
        {"id": "home-assistant", "name": "Home Assistant", "kind": "env",
         "description": "OpenAI Conversation integration.",
         "snippet": (f"Base URL: {base}/v1\n"
                     f"API key: {key}\n"
                     f"Model: {model}")},
    ])

    type_items = []
    if "embeddings" in kinds:
        type_items += _mark([
            {"id": "embeddings-curl", "name": "Embeddings / curl",
             "kind": "http",
             "description": "Embeddings endpoint; input may be a string or "
                            "a list of strings.",
             "snippet": (
                 f"curl {base}/v1/embeddings \\\n"
                 f"  -H 'Authorization: Bearer {key}' \\\n"
                 f"  -H 'Content-Type: application/json' \\\n"
                 f"  -d '{{\"model\":\"{model}\",\"input\":"
                 f"[\"first text\",\"second text\"]}}'")},
            {"id": "embeddings-python", "name": "Embeddings Python",
             "kind": "python",
             "description": "OpenAI SDK embeddings call.",
             "snippet": (
                 "from openai import OpenAI\n"
                 f"client = OpenAI(base_url='{base}/v1', api_key='{key}')\n"
                 f"r = client.embeddings.create(model='{model}', "
                 "input=['first text','second text'])\n"
                 "print(len(r.data[0].embedding))")},
            {"id": "open-webui-embeddings", "name": "Open WebUI RAG",
             "kind": "env", "description": "Use as the embedding model.",
             "snippet": (f"Base URL: {base}/v1\n"
                         f"API key: {key}\n"
                         f"Embedding model: {model}")},
        ])

    if "decision" in kinds:
        payload = json.dumps({
            "model": model,
            "state": "Should this request be routed to the fast model?",
            "questions": {"route": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": {"yes": 1, "no": 2}}},
        }, separators=(",", ":"))
        type_items += _mark([
            {"id": "decision-curl", "name": "TypeSafe decision / curl",
             "kind": "http",
             "description": "llama.cpp/TypeSafe /v1/systemone decision "
                            "endpoint — not chat.",
             "snippet": (
                 f"curl {base}/v1/systemone \\\n"
                 f"  -H 'Authorization: Bearer {key}' \\\n"
                 f"  -H 'Content-Type: application/json' \\\n"
                 f"  -d '{payload}'")},
            {"id": "decision-python", "name": "TypeSafe decision Python",
             "kind": "python",
             "description": "requests example for /v1/systemone.",
             "snippet": (
                 "import requests\n"
                 f"r = requests.post('{base}/v1/systemone',\n"
                 f"    headers={{'Authorization': 'Bearer {key}'}},\n"
                 f"    json={payload})\n"
                 "print(r.json())")},
        ])

    if "audio" in kinds:
        type_items += _mark([
            {"id": "audio-curl", "name": "Audio / ASR curl",
             "kind": "http",
             "description": "Send a base64 WAV data URI to the chat "
                            "endpoint's input_audio content block.",
             "snippet": (
                 f"curl {base}/v1/chat/completions \\\n"
                 f"  -H 'Authorization: Bearer {key}' \\\n"
                 f"  -H 'Content-Type: application/json' \\\n"
                 f"  -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":"
                 f"\"user\",\"content\":[{{\"type\":\"input_audio\","
                 f"\"input_audio\":{{\"data\":\"data:audio/wav;base64,"
                 f"REPLACE_WITH_BASE64\",\"format\":\"wav\"}}}}]}}]}}'")},
            {"id": "audio-python", "name": "Audio Python", "kind": "python",
             "description": "Base64 a WAV and send it as input_audio.",
             "snippet": (
                 "import base64, requests\n"
                 "audio = base64.b64encode(open('sample.wav','rb').read()).decode()\n"
                 f"r = requests.post('{base}/v1/chat/completions',\n"
                 f"    headers={{'Authorization': 'Bearer {key}'}},\n"
                 f"    json={{'model': '{model}', 'messages': [{{'role':"
                 " 'user', 'content': [{'type': 'input_audio',"
                 " 'input_audio': {'data': f'data:audio/wav;base64,{audio}',"
                 " 'format': 'wav'}}]}]})\n"
                 "print(r.json()['choices'][0]['message']['content'])")},
        ])

    if "vision" in kinds:
        type_items += _mark([
            {"id": "vision-curl", "name": "Vision / curl", "kind": "http",
             "description": "Chat completions with an image_url content "
                            "block (data URI or http URL).",
             "snippet": (
                 f"curl {base}/v1/chat/completions \\\n"
                 f"  -H 'Authorization: Bearer {key}' \\\n"
                 f"  -H 'Content-Type: application/json' \\\n"
                 f"  -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":"
                 f"\"user\",\"content\":[{{\"type\":\"text\",\"text\":"
                 f"\"Describe this image.\"}},{{\"type\":\"image_url\","
                 f"\"image_url\":{{\"url\":\"data:image/png;base64,"
                 f"REPLACE_WITH_BASE64\"}}}}]}}]}}'")},
            {"id": "vision-python", "name": "Vision Python", "kind": "python",
             "description": "OpenAI SDK with an image data URI.",
             "snippet": (
                 "import base64\nfrom openai import OpenAI\n"
                 "img = base64.b64encode(open('image.png','rb').read()).decode()\n"
                 f"client = OpenAI(base_url='{base}/v1', api_key='{key}')\n"
                 f"r = client.chat.completions.create(model='{model}',"
                 " messages=[{'role':'user','content':["
                 "{'type':'text','text':'Describe this image.'},"
                 "{'type':'image_url','image_url':{'url':"
                 "f'data:image/png;base64,{img}'}}]}])\n"
                 "print(r.choices[0].message.content)")},
        ])

    # Chat clients stay visible for every model so the page does not look
    # broken, but they are marked incompatible when the selected model is
    # an embeddings/decision/audio model.
    chat_compatible = ("chat" in kinds) or not any(
        k in kinds for k in ("embeddings", "decision", "audio"))
    return type_items + _mark(chat_clients, compatible=chat_compatible)


# ---------------------------------------------------------------------------
# Safety / exposure checks
# ---------------------------------------------------------------------------

def _listen_addresses(port):
    """Local bind addresses for a TCP port (best effort, all platforms)."""
    port = int(port)
    out = set()
    try:
        if os.name == "nt":
            cmd = ("powershell -NoProfile -Command "
                   f"\"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                   "-ErrorAction SilentlyContinue).LocalAddress\"")
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=8)
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if line:
                    out.add(line)
            return out
        if shutil.which("ss"):
            r = subprocess.run(["ss", "-ltn"], capture_output=True,
                               text=True, timeout=8)
            for line in (r.stdout or "").splitlines():
                cols = line.split()
                if len(cols) >= 4 and cols[3].endswith(f":{port}"):
                    out.add(cols[3].rsplit(":", 1)[0].strip("[]"))
        elif shutil.which("lsof"):
            r = subprocess.run(["lsof", "-nP", f"-iTCP:{port}",
                                "-sTCP:LISTEN"], capture_output=True,
                               text=True, timeout=8)
            for line in (r.stdout or "").splitlines()[1:]:
                cols = line.split()
                if len(cols) >= 9:
                    out.add(cols[8].rsplit(":", 1)[0].strip("[]"))
    except Exception:
        pass
    return out


_VPN_BINARIES = [
    ("tailscale", "Tailscale"),
    ("wg", "WireGuard"),
    ("wg-quick", "WireGuard"),
    ("wireguard-go", "WireGuard"),
    ("zerotier-cli", "ZeroTier"),
    ("netbird", "NetBird"),
    ("nebula", "Nebula"),
    ("openvpn", "OpenVPN"),
    ("tincd", "tinc"),
    ("cloudflared", "Cloudflare Tunnel"),
    ("ngrok", "ngrok"),
]
_VPN_IFACE_PREFIXES = ("tailscale", "wg", "zt", "netbird", "nebula",
                       "tun", "tap", "utun", "ppp", "ipsec", "vpn")


def _vpn_status():
    """Best-effort VPN/tunnel detection: installed clients + interface names."""
    found = []
    for binary, label in _VPN_BINARIES:
        if shutil.which(binary) and label not in found:
            found.append(label)
    names = []
    try:
        if os.name == "nt":
            cmd = ('powershell -NoProfile -Command '
                   '"(Get-NetAdapter).Name"')
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=8)
            names = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
        elif sys.platform == "darwin":
            r = subprocess.run(["ifconfig", "-l"], capture_output=True,
                               text=True, timeout=8)
            names = (r.stdout or "").split()
        else:
            net = "/sys/class/net"
            if os.path.isdir(net):
                names = os.listdir(net)
            else:
                r = subprocess.run(["ip", "-o", "link", "show"],
                                   capture_output=True, text=True,
                                   timeout=8)
                for line in (r.stdout or "").splitlines():
                    parts = line.split(":")
                    if len(parts) >= 2:
                        names.append(parts[1].strip().split("@")[0])
    except Exception:
        pass
    for n in names:
        low = n.lower()
        if any(low.startswith(p) for p in _VPN_IFACE_PREFIXES):
            label = ("Tailscale" if low.startswith("tailscale")
                     else "WireGuard" if low.startswith("wg")
                     else "VPN/tunnel")
            if label not in found:
                found.append(label)
    return {"found": found,
            "tailscale": "Tailscale" in found,
            "wireguard": "WireGuard" in found}


def _machine_count():
    try:
        return len(cfg.machines.all())
    except Exception:
        return 1


def _shared_agent_token_count():
    """Peers still using the legacy farm-wide token (blast-radius risk)."""
    try:
        import secrets_vault as vault
        farm = vault.get_farm_agent_token()
        if not farm:
            return 0
        count = 0
        for machine in cfg.machines.all():
            if local_node.is_local(machine):
                continue
            if vault.get_machine_agent_token(
                    machine, allow_farm=False) == farm:
                count += 1
        return count
    except Exception:
        return 0


def safety_report():
    checks = []
    keys = access.access_keys()
    public = {"0.0.0.0", "::", "*"}
    for port, label in ((8199, "farm UI"), (8090, "API gateway"),
                        (8198, "peer agent")):
        addrs = _listen_addresses(port)
        exposed = bool(addrs & public)
        if not addrs:
            checks.append({"id": f"port-{port}", "level": "info",
                           "title": f"{label} ({port})",
                           "detail": "not listening on this host"})
        elif exposed and port == 8198:
            checks.append({"id": f"port-{port}", "level": "ok",
                           "title": f"{label} ({port})",
                           "detail": "listening on all interfaces; agents "
                                     "require a bearer token. Keep it on a "
                                     "trusted LAN/VPN; HTTPS :8197 is the "
                                     "preferred agent transport."})
        elif exposed and not keys and port in (8090, 8199):
            checks.append({"id": f"port-{port}", "level": "bad",
                           "title": f"{label} ({port})",
                           "detail": "listening on a LAN interface with no "
                                     "access keys configured",
                           "fix": "Create an access key before letting other "
                                  "devices use the farm."})
        elif exposed:
            checks.append({"id": f"port-{port}", "level": "warn",
                           "title": f"{label} ({port})",
                           "detail": "reachable on the LAN/VPN (expected for "
                                     "a trusted network; never the internet). "
                                     "For public API clients use HTTPS 443 → "
                                     "nginx → 127.0.0.1:8090 (docs/public-api.md), "
                                     "not a forwarded 8090/8199."})
        else:
            checks.append({"id": f"port-{port}", "level": "ok",
                           "title": f"{label} ({port})",
                           "detail": "bound to loopback only"})
    vpn = _vpn_status()
    count = _machine_count()
    if vpn["found"]:
        checks.append({"id": "vpn", "level": "ok", "title": "VPN / tunnel",
                       "detail": "detected: " + ", ".join(vpn["found"]) +
                                 " — use it for remote access instead of "
                                 "port forwarding"})
    elif count <= 1:
        checks.append({"id": "vpn", "level": "info", "title": "VPN / tunnel",
                       "detail": "none detected. This is a single-machine "
                                 "farm; keep the UI and gateway on loopback "
                                 "or a trusted LAN, and do not port-forward "
                                 "them to the internet."})
    else:
        checks.append({"id": "vpn", "level": "warn", "title": "VPN / tunnel",
                       "detail": "none detected. Do not port-forward the "
                                 "UI, gateway, agent, or RPC ports: "
                                 "llama.cpp RPC is unauthenticated. Use a "
                                 "LAN, Tailscale/WireGuard, or another VPN."})
    if count <= 1:
        checks.append({"id": "rpc", "level": "ok", "title": "RPC exposure",
                       "detail": "single-machine farm: no RPC peers are "
                                 "configured, so RPC pooling is inactive."})
    else:
        try:
            import rpc_tls
            rpc_secure = bool(rpc_tls.enabled())
        except Exception:
            rpc_secure = False
        if rpc_secure:
            checks.append({"id": "rpc", "level": "ok", "title": "RPC exposure",
                           "detail": f"{count} machines configured. RPC mTLS "
                                     "is enabled; the tunnel port and raw RPC "
                                     "ports must still stay on the LAN/VPN and "
                                     "never be forwarded."})
        else:
            checks.append({"id": "rpc", "level": "warn", "title": "RPC exposure",
                           "detail": f"{count} machines configured. llama.cpp "
                                     "RPC servers are unauthenticated; keep them "
                                     "on a trusted LAN/VPN and never forward "
                                     "their ports."})
    if not keys:
        checks.append({"id": "keys", "level": "bad",
                       "title": "Access keys",
                       "detail": "no access keys exist; the gateway is open "
                                 "to every device that can reach it",
                       "fix": "Create one on the Access page."})
    else:
        checks.append({"id": "keys", "level": "ok",
                       "title": "Access keys",
                       "detail": f"{len(keys)} key(s) configured; stored in "
                                 "config/settings.yaml (owner-only)"})
    shared = _shared_agent_token_count()
    if shared:
        checks.append({"id": "agent-tokens", "level": "warn",
                       "title": "Agent tokens",
                       "detail": f"{shared} peer(s) still share the legacy "
                                 "farm token; one compromised peer can reach "
                                 "the others.",
                       "fix": "Rotate agent tokens: POST /api/machines/"
                              "rotate-agent-tokens (after agents update)."})
    elif _machine_count() > 1:
        checks.append({"id": "agent-tokens", "level": "ok",
                       "title": "Agent tokens",
                       "detail": "each peer has its own token"})
    bad = sum(1 for c in checks if c["level"] == "bad")
    warn = sum(1 for c in checks if c["level"] == "warn")
    return {"checks": checks, "bad": bad, "warn": warn,
            "ok": bad == 0}


# ---------------------------------------------------------------------------
# Local onboarding funnel (no external telemetry)
# ---------------------------------------------------------------------------

def _metrics_path():
    return os.path.join(data_dir(), "config", "onboarding_metrics.json")


def _load_metrics():
    try:
        with open(_metrics_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def mark(event, extra=None):
    event = (event or "").strip()
    if not event:
        return {}
    with _JOBS_LOCK:
        d = _load_metrics()
        now = time.time()
        d.setdefault("first_seen", now)
        events = d.setdefault("events", {})
        events.setdefault(event, now)
        if extra:
            d.setdefault("extra", {}).setdefault(event, extra)
        try:
            os.makedirs(os.path.dirname(_metrics_path()), exist_ok=True)
            tmp = _metrics_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, sort_keys=True)
            os.replace(tmp, _metrics_path())
        except OSError:
            pass
        return d


def metrics_summary():
    d = _load_metrics()
    ev = d.get("events") or {}
    first = d.get("first_seen")
    ready = ev.get("first_model_ready") or ev.get("first_chat")
    return {
        "events": ev,
        "time_to_first_model_s": round(ready - first, 1)
        if first and ready else None,
        "path": _metrics_path(),
    }
