"""Engine adapter seam: llama.cpp | vLLM behind one settings toggle.

Design per ARD t_76f3d1c0 (§1): a thin registry the app dispatches on.
`LlamaCppAdapter` is pure delegation to the existing rpc_manager /
swap_manager functions — code is NOT moved, behavior is byte-identical.
All vLLM knowledge will live in src/vllm_manager.py (B2); the registry is
the only seam app.py should branch on.
"""
import os
from typing import Protocol

import yaml

import config_loader as cfg
import local_node
import rpc_manager
# DEC-02 (AUDIT-T6 §1.5): the UI field manifest protocol + the ui_hints
# shim live in their own module (the spec's engine_registry hotspot
# workaround) — adapters mix in UiHintsMixin for the manifest contract.
import engine_manifest
from engine_manifest import UiHintsMixin

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(BASE_DIR, "config", "settings.yaml")

VALID_ENGINES = cfg.VALID_ENGINES
DEFAULT_ENGINE = cfg.DEFAULT_ENGINE
# DEC-01 (spec §1.4): the VLLM_* settings-block re-exports are retired —
# the vocabulary constants stay in config_loader (B4's schema home) and
# block validation lives on the owning adapter (VllmAdapter.settings_section).


def load_settings(path=None):
    """Load config/settings.yaml with the engine schema applied (ARD §2).

    Validates `engine` against the enum — a typo raises ValueError listing
    the valid values (startup error, never silent fallback). Missing engine
    injects the default (backward compat: existing installs have no key).
    Normalizes the `vllm:` block: missing -> defaults; unknown keys, bad
    delivery, or a non-numeric gpu-memory-utilization raise. Fresh dict.
    """
    file = path or SETTINGS_FILE
    data = {}
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{file}: expected a YAML mapping at top level, "
            f"got {type(data).__name__}")
    out = cfg.inject_engine_defaults(data)
    return out


class EngineAdapter(Protocol):
    """Operations the app dispatches on, per engine (ARD §1).

    llama.cpp is the reference implementation; the vLLM adapter (B2) must
    return the same shapes.

    Adapter v2 ownership members (AUDIT-T6 §1.3/§1.4, DEC-01): each engine
    owns its settings vocabulary and per-model section name.
    """

    name: str

    def settings_section(self, raw_settings: dict) -> dict:
        """Validate + inject defaults for THIS engine's block in
        settings.yaml (AUDIT-T6 §1.4).

        vllm: the `vllm:` block schema (delivery, venv_bin, docker_image,
        default_gpu_memory_utilization). llama_cpp: llama_bin + defaults
        (read-time warnings belong to install(); never 400 here).
        Mutates raw_settings in place and returns it, matching
        cfg.inject_engine_defaults' contract. Raises ValueError on
        unknown keys / invalid values — same strictness as today.
        """
        ...

    # -- identity / file ownership (Adapter v2, DEC-01) --
    engine_id: str
    config_section: "str | None"
    """Per-model YAML section name owned by this engine:
    None (llama.cpp: top-level vocabulary — flags{}, tensor_split,
    servers, variants) | "vllm" (the per-model `vllm:` block)."""

    def build_command(self, model_cfg, servers, master_name=None) -> str:
        """llama-swap `cmd:` for a model (multi-line shell, ${PORT})."""
        ...

    def probe(self, machine, port, is_local) -> dict:
        """Status probe -> {"running": bool, "detail": str}."""
        ...

    def swap_entries(self, config_path=None, host_name=None) -> dict:
        """llama-swap launch plan -> swap_manager.launch_plan() shape."""
        ...

    def validate_model_config(self, model_cfg) -> list:
        """Config issues -> [{"level": "error"|"warn", "msg": str}]."""
        ...

    def validate_hosts(self, machines) -> list:
        """Engine×OS gate -> [str] errors (vLLM refuses Windows hosts)."""
        ...

    def asr_language_system_message(self, language: str) -> str | None:
        """System-message prefix forcing ASR language -> str, or None when
        the engine has no equivalent (sending it would be a no-op or leak
        into the transcript)."""
        ...

    def filter_audio_models(self, models) -> list:
        """Audio/ASR-capable models from cfg.list_models() output."""
        ...

    def filter_vision_models(self, models) -> list:
        """Vision-capable models from cfg.list_models() output."""
        ...

    def filter_embedding_models(self, models) -> list:
        """Embedding-capable models from cfg.list_models() output."""
        ...

    def filter_models(self, models) -> list:
        """Models the ACTIVE engine can serve, from cfg.list_models()
        output (DEC-11, closes T4-G6). The Models table lists only these
        — the same downloads.vocabulary gate /api/scan applies to model
        sources, applied to configured models. Default: all models
        (an engine with no format restriction passes everything)."""
        ...

    # -- launch / stop / status (AUDIT-T6 §1.3, closes T2-G6/G7; DEC-06) --
    def capable(self, capability) -> bool:
        """Engine-capability check (spec dispatch rule §1.3): app.py asks
        the adapter instead of comparing engine names.
        "rpc_cluster" — RPC peers + single-model RPC launch/stop
                        (llama.cpp only; vLLM returns False).
        "swap_stop"   — model stop rides the llama-swap unload proxy
                        (vLLM only; llama.cpp returns False)."""
        ...

    def launch_plan(self, config_path=None, host_name=None) -> dict:
        """The llama-swap plan this engine would generate for a host —
        today swap_manager.launch_plan()'s body (host, yaml, models,
        scripts, warnings). vLLM emits thin per-model entries."""
        ...

    def start_model(self, model_id, servers, port, variant=None,
                    master_name=None, mode="rpc") -> dict:
        """Single-model live launch. llama.cpp: rpc_manager.launch_model.
        master_name forces which server runs llama-server; mode="local"
        is the single-box path (no RPC peers). vLLM refuses (ValueError)
        — serving goes through llama-swap."""
        ...

    def stop_model(self, model_id, servers, variant=None,
                   master_name=None) -> dict:
        """Stop a live-launched model. llama.cpp: rpc_manager.stop_model
        (kills RPC ports; master_name scopes the llama-server kill to the
        launch master). vLLM: llama-swap model unload (cmdStop) via
        the swap-host proxy."""
        ...

    def active_model(self, machine) -> dict:
        """Per-machine serving status for GET /api/launch/active.
        llama.cpp: TCP probe on the machine's RPC port.
        vLLM: HTTP /health probe on the swap listen port.
        Returns {"name": ..., "serving": bool} (+ engine extras)."""
        ...

    # -- acquisition (DEC-07, closes T5-3/T5-4/T5-5) --
    def acquire(self, link, store, repo=False, all_shards=True):
        """Start a download job in this engine's vocabulary and return the
        job dict. repo=True means `link` is hf://owner/name (whole HF-hub
        model). The app-level engine gate (POST /api/downloads) refuses
        out-of-vocabulary formats BEFORE dispatch; this member is the
        adapter-side backstop and must raise ValueError for the same
        shapes. Raises ValueError for a bad link/store (-> HTTP 400)."""
        ...

    async def browse(self, query="", sort="downloads", limit=20, page=0,
                     quant="", pipeline_tag="", author="",
                     recommended=False):
        """HF search in this engine's vocabulary: llama.cpp proxies GGUF
        quants, vLLM proxies safetensors/HF repos. Same result shape as
        hf_browse.search_gguf/search_safetensors."""
        ...

    def validate_downloaded(self, path) -> list:
        """Is `path` (store:// alias or raw path) a model this engine can
        serve? -> [{"level": "error"|"warn", "msg": str}]; [] = fine.
        llama.cpp wants a .gguf file; vLLM wants a directory holding
        config.json + *.safetensors."""
        ...

    # -- UI field manifest (AUDIT-T6 §1.5, DEC-02) --
    # The complete field inventory lives in src/engine_manifest.py (the
    # spec's engine_registry hotspot workaround): the Protocol slice is
    # engine_manifest.EngineUxProtocol, and UiHintsMixin provides the
    # deprecated ui_hints() shim + the llama.cpp-shaped default manifest.
    # `ui_hints` itself retires in Card 15 — no adapter adds members here
    # for it.


class LlamaCppAdapter(UiHintsMixin):
    """Pure delegation to rpc_manager / swap_manager — zero behavior change.

    Functions are called through the module (not imported bare) so tests
    can monkeypatch rpc_manager and still exercise the adapter.
    """

    name = "llama_cpp"
    # Adapter v2 ownership (DEC-01): llama.cpp vocabulary is top-level in
    # per-model YAML — there is no per-model section to own.
    engine_id = "llama_cpp"
    config_section = None

    def settings_section(self, raw_settings):
        """llama.cpp owns NO settings.yaml block of its own (AUDIT-T6
        §1.4): its vocabulary (llama_bin, model_dir) is shared farm
        chrome, so there is nothing engine-qualified to validate or
        default here. Kept as a no-op member so dispatch is uniform —
        read-time llama_bin validation stays with install(), never 400.
        """
        return raw_settings

    def sanitize_model_section(self, raw, where="vllm"):
        """llama.cpp has no per-model section of its own (DEC-04): the
        vocabulary IS the top-level model mapping, so there is nothing to
        sanitize. Kept as a no-op member so config_loader.save_model's
        adapter dispatch is uniform across engines — a llama.cpp save
        never touches a `vllm:` block and vice versa (round-trip
        isolation)."""
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a mapping")
        return dict(raw)

    def apply_model_section(self, data, where="vllm"):
        """No-op member for uniform dispatch (see sanitize_model_section):
        llama.cpp saves leave the model dict untouched."""
        return data

    def build_command(self, model_cfg, servers, master_name=None):
        return rpc_manager.build_llama_command(
            model_cfg, servers, master_name=master_name)

    def probe(self, machine, port, is_local):
        ok = rpc_manager._port_open(machine, port, is_local)
        return {"running": bool(ok),
                "detail": "rpc port open" if ok else "rpc port closed"}

    def swap_entries(self, config_path=None, host_name=None):
        """Delegates to the existing swap_manager.launch_plan() — the exact
        plan dict (host, yaml, models, scripts, warnings) the app uses
        today. vLLM entries land in B3 as a branch inside generate()."""
        import swap_manager
        return swap_manager.launch_plan(config_path, host_name)

    def validate_model_config(self, model_cfg):
        model_id = (model_cfg or {}).get("model_id", "")
        return rpc_manager.validate_model(model_id)["issues"]

    def validate_hosts(self, machines):
        """llama.cpp accepts any farm host — no engine×OS restriction."""
        return []

    def asr_language_system_message(self, language):
        # Qwen3-ASR: appending "language X<asr_text>" after the generation
        # prompt forces the language (matches qwen-asr's force_language).
        # As a system message it lands right before the generation prompt —
        # verified equivalent in llama-server. This is the SAME message the
        # audio endpoint has always sent (byte-identical payloads); other
        # engines return None via their own adapter.
        if language and language != "None":
            return f"language {language}<asr_text>"
        return None

    def filter_audio_models(self, models):
        # ASR models: audio-capable GGUFs (audio flag or asr/whisper in id).
        out = []
        for m in models:
            flags = m.get("flags") or {}
            mid = (m.get("model_id") or "").lower()
            if flags.get("audio") or "asr" in mid or "whisper" in mid:
                out.append(m)
        return out

    def filter_decision_models(self, models):
        # TypeSafe decision models (llama.cpp /v1/systemone): detected
        # from the GGUF's <arch>.decision.type metadata (the same gate
        # llama.cpp itself uses), not the filename — decision families
        # are new and name heuristics would miss them.
        out = []
        for m in models:
            if m.get("decision_type"):
                out.append(m)
        return out

    def filter_vision_models(self, models):
        # heuristic: model has an mmproj projector configured (required for
        # image input) — a llama.cpp projector concept. runtime: custom
        # models opt in explicitly via custom.vision (external servers
        # have no mmproj flag to detect).
        out = []
        for m in models:
            flags = m.get("flags") or {}
            if flags.get("mmproj"):
                out.append(m)
                continue
            if (m.get("runtime") or "") == "custom" and \
                    (m.get("custom") or {}).get("vision") is True:
                out.append(m)
        return out

    def filter_embedding_models(self, models):
        # heuristic: embed in id/name or known embed arch
        out = []
        for m in models:
            mid = m.get("model_id") or ""
            name = m.get("name") or ""
            if "embed" in mid.lower() or "embed" in name.lower():
                out.append(m)
        return out

    def filter_models(self, models):
        # DEC-11 (T4-G6): the Models table lists only llama.cpp-servable
        # models — GGUF files. An 'hf' (safetensors repo) model cannot be
        # served by llama-server RPC launch at all (hard error in
        # rpc_manager.validate_model), so it is not listed. Format
        # inference mirrors that gate: no explicit `format` -> 'gguf'
        # when base_path ends .gguf, else 'hf'. runtime: custom models
        # (external servers, no GGUF by design) are always listed — they
        # are saved configs like any other model.
        out = []
        for m in models:
            if (m.get("runtime") or "") == "custom":
                out.append(m)
                continue
            fmt = (m.get("format")
                   or ("gguf" if str(m.get("base_path") or "")
                       .lower().endswith(".gguf") else "hf"))
            if fmt == "gguf":
                out.append(m)
        return out

    # -- launch / stop / status (DEC-06, closes T2-G6/G7) --

    def capable(self, capability):
        caps = {"rpc_cluster": True, "swap_stop": False}
        return caps.get(capability, False)

    def launch_plan(self, config_path=None, host_name=None):
        """The llama-swap plan — swap_manager.launch_plan() is this engine's
        planner (unchanged delegation; DEC-06 keeps the function and
        routes it through the adapter so app.py never calls it directly)."""
        import swap_manager
        return swap_manager.launch_plan(config_path, host_name)

    def start_model(self, model_id, servers, port, variant=None,
                    master_name=None, mode="rpc"):
        return rpc_manager.launch_model(model_id, servers, port, variant,
                                        master_name=master_name, mode=mode)

    def stop_model(self, model_id, servers, variant=None, master_name=None,
                   launch_port=None):
        return rpc_manager.stop_model(servers, model_id, variant,
                                      master_name=master_name,
                                      launch_port=launch_port)

    def active_model(self, machine):
        """Per-machine serving probe for /api/launch/active. The machine
        dict carries its rpc_port; a missing/zero port is never serving.
        Shape matches the pre-DEC-06 response exactly ({"name",
        "serving"})."""
        port = (machine or {}).get("rpc_port") or 0
        is_local = local_node.is_local(machine or {})
        ok = bool(port) and rpc_manager._port_open(machine, port, is_local)
        return {"name": (machine or {}).get("name"), "serving": bool(ok)}

    # -- acquisition (DEC-07: llama.cpp vocabulary = GGUF) --

    def acquire(self, link, store, repo=False, all_shards=True):
        """llama.cpp downloads GGUF quants. repo=True (hf://owner/name,
        no filename) is the vLLM whole-repo vocabulary — refused here so
        the adapter is a backstop even if an app-level gate regresses
        (mirrors the vLLM start_model 409-refusal pattern from DEC-06)."""
        if repo:
            raise ValueError(
                "whole-repo downloads are not part of the llama.cpp "
                "engine — llama.cpp serves GGUF files (expected "
                "hf://owner/repo/file.gguf)")
        if not str(link or "").lower().endswith(".gguf"):
            raise ValueError(
                "llama.cpp engine downloads GGUF files — expected "
                "hf://owner/repo/file.gguf")
        import hf_downloader
        if all_shards:
            # split GGUF set: fetch every sibling shard in one job
            return hf_downloader.start_shards(link, store)
        return hf_downloader.start(link, store)

    async def browse(self, query="", sort="downloads", limit=20, page=0,
                     quant="", pipeline_tag="", author="",
                     recommended=False):
        """GGUF quant search (llama.cpp vocabulary). all_shards stays
        True-on-entry semantics: hf_downloader.start_shards fetches every
        shard of a split file — acquire() above keeps single-file start
        because the split-set is detected from the filename."""
        import hf_browse
        return await hf_browse.search_gguf(
            query=query, sort=sort, limit=limit, page=page,
            quant=quant, pipeline_tag=pipeline_tag, author=author,
            recommended=recommended)

    def validate_downloaded(self, path):
        """llama.cpp serves GGUF files; anything else is an error."""
        import path_resolver
        p = str(path or "")
        if not p.lower().endswith(".gguf"):
            return [{"level": "error",
                     "msg": f"llama.cpp engine serves GGUF files — "
                            f"'{p}' is not one"}]
        local = path_resolver.resolve(p)
        if not os.path.isfile(local):
            return [{"level": "error",
                     "msg": f"model file not found: {p}"}]
        return []

    def ui_manifest(self):
        """llama.cpp manifest (AUDIT-T6 §1.5): the mixin's default shape
        with this engine's labels + flag presets — every editor section
        listed (nothing hidden), generic flags editing, RPC-cluster
        launch, GGUF downloads."""
        manifest = UiHintsMixin.ui_manifest(self)
        manifest.update({
            "labels": {
                "server": "llama-server",
                "peers": "RPC peers",
                "parallelism": "RPC cluster",
                "probe": "RPC port",
            },
            # Tests tabs are engine-exclusive surfaces: the renderer hides
            # nav buttons + sections listed here (same mechanism as
            # hidden_panels). llama.cpp keeps all of them.
            "flag_presets": {
                "ctx_size": {"flag": "--ctx-size",
                             "placeholder": "e.g. 160000"},
                "parallel_seqs": {"flag": "-np", "placeholder": "e.g. 256"},
                "kv_cache": {"flag": "--cache-type-k",
                             "choices": ["f16", "q8_0", "q4_0"],
                             "note": None},
                "offload": {"flag": "-ngl", "placeholder": "e.g. 999"},
            },
        })
        return manifest


_REGISTRY: dict = {"llama_cpp": LlamaCppAdapter}


def _register_builtins():
    """Register the vLLM adapter lazily (B2).

    vllm_manager imports engine_registry's load_settings at call time, and
    rpc_manager/swap_manager (its imports) must not be dragged into the
    registry-import cycle — so the adapter registers via a deferred import
    here instead of vllm_manager importing this module at top level.
    """
    try:
        import vllm_manager
    except ImportError:
        return  # vLLM support absent (partial checkout); llama_cpp keeps working
    _REGISTRY[vllm_manager.ENGINE_VLLM] = vllm_manager.VllmAdapter


_register_builtins()


def get_engine_section_adapter():
    """Resolve the adapter that owns per-model section round-trip for the
    CURRENT engine (DEC-04). config_loader.save_model delegates its
    sanitize/apply here instead of hardcoding the vllm logic. LlamaCppAdapter
    owns no per-model section (it is the llama.cpp vocabulary itself), so it
    provides no-ops; vLLM owns the `vllm:` block."""
    return get_engine()

def get_engine(settings=None):
    """Resolve the engine adapter from settings (dict, or None = load file).

    One seam: app.py calls get_engine(settings) and dispatches on the
    adapter; no engine checks scattered past the registry boundary.
    """
    data = settings if settings is not None else load_settings()
    name = data.get("engine", DEFAULT_ENGINE) if isinstance(data, dict) \
        else DEFAULT_ENGINE
    if name not in VALID_ENGINES:
        raise ValueError(
            f"invalid engine '{name}' — must be one of "
            f"{', '.join(VALID_ENGINES)}")
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"engine '{name}' is recognized but has no adapter registered "
            f"yet (options: {', '.join(_REGISTRY)})") from None
    return cls()
