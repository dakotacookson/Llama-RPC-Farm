"""Read/write YAML config files: machines, groups, per-model configs.

All configs live in ./config next to the app. Nothing outside this
folder is ever written to.
"""
import os
import uuid
from datetime import datetime

import yaml

import path_resolver
import gguf_inspect

# Cache for MTP detection results keyed by resolved absolute path to avoid re-reading GGUF files
_mtp_cache = {}
# Cache for architecture detection
_arch_cache = {}
# Cache for TypeSafe decision-type detection (same pattern as _arch_cache)
_decision_cache = {}


def _get_model_architecture(gguf_path):
    """Return the model architecture from GGUF metadata."""
    if not gguf_path:
        return "unknown"
    if not gguf_path.lower().endswith('.gguf'):
        return "unknown"
    if gguf_path in _arch_cache:
        return _arch_cache[gguf_path]
    try:
        resolved = path_resolver.resolve(gguf_path)
        result = gguf_inspect.get_architecture(resolved)
    except Exception:
        result = "unknown"
    _arch_cache[gguf_path] = result
    return result


def _get_decision_type(gguf_path):
    """TypeSafe decision-model type ('laya', ...) or ''. Models carrying
    <arch>.decision.type serve only /v1/systemone (no text generation),
    so the editor/safe-defaults treat them like a distinct model class
    (same shape as embeddings/ASR, not chat)."""
    if not gguf_path:
        return ""
    if not gguf_path.lower().endswith('.gguf'):
        return ""
    if gguf_path in _decision_cache:
        return _decision_cache[gguf_path]
    try:
        resolved = path_resolver.resolve(gguf_path)
        result = gguf_inspect.get_decision_type(resolved)
    except Exception:
        result = ""
    _decision_cache[gguf_path] = result
    return result

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
MODELS_DIR = os.path.join(CONFIG_DIR, "models")

# ---------------- engine selection (settings.yaml) -------------------------
# Schema owned by src/engine_registry.py (load_settings / get_engine);
# these helpers re-export the vocabulary so callers can validate without
# importing the registry (which pulls in rpc_manager).
ENGINE_LLAMA_CPP = "llama_cpp"
ENGINE_VLLM = "vllm"
VALID_ENGINES = (ENGINE_LLAMA_CPP, ENGINE_VLLM)
DEFAULT_ENGINE = ENGINE_LLAMA_CPP

VLLM_DELIVERY_DEFAULT = "venv"
VALID_VLLM_DELIVERY = ("venv", "docker")
# B13 per-machine override: a machine record (config/machines.yaml) may set
# `vllm_delivery:` to pin ONE machine to a delivery method, overriding the
# global settings vllm.delivery (e.g. a host without python3.12 could be
# pinned to docker once docker ships). "" (the default) = inherit global.
VLLM_MACHINE_DELIVERY_KEY = "vllm_delivery"
VALID_MACHINE_VLLM_DELIVERY = ("", "venv", "docker")
VLLM_BLOCK_DEFAULTS = {
    "delivery": VLLM_DELIVERY_DEFAULT,
    "venv_bin": "~/vllm-env/bin/vllm",
    "docker_image": "vllm/vllm-openai:latest",
    "default_gpu_memory_utilization": 0.9,
    # multi-host Ray cluster knobs (design §2): gcs port + cluster-formation
    # timeout (vLLM multi-node-serving.sh's 300 s leader loop).
    "ray_port": 6379,
    "ray_init_timeout": 300,
    # Tool calling: auto tool choice requires a parser. "hermes" works for
    # the Qwen family (incl. GGUF builds); per-model override via
    # vllm.tool_call_parser, opt-out via vllm.enable_auto_tool_choice=False.
    "default_tool_call_parser": "hermes",
}
VLLM_ALLOWED_KEYS = set(VLLM_BLOCK_DEFAULTS)

# Per-model `vllm:` section (config/models/*.yaml, ARD §3.3). The schema
# vocabulary lives here (like the engine enum above) so config round-trips
# and save-path validation don't need vllm_manager (which pulls rpc_manager);
# vllm_manager re-exports these names.
VLLM_MODEL_KEYS = frozenset({
    "max_model_len", "max_num_seqs", "tensor_parallel_size",
    "pipeline_parallel_size",
    "kv_cache_dtype", "cpu_offload_gb", "quantization", "tokenizer",
    "extra_args",
    # A2: hardware-sensitive — heterogeneous machines need a different
    # budget per deployment (0.9 on the big GPU, 0.35 on the small one).
    # Falls back to settings' default_gpu_memory_utilization when unset.
    "gpu_memory_utilization",
    # multi-host (Ray cluster): ray|mp; mp is refused with peers for now
    # (vllm_manager.build_vllm_cluster_command) — reserved vocabulary.
    "distributed_executor_backend",
    # compute dtype for the model weights: Turing (sm_75, e.g. RTX 2060)
    # cannot run bfloat16 (needs sm_80+) — heterogeneous clusters must
    # force `half`. auto lets vLLM pick (bf16 on Ampere+).
    "dtype",
    # OpenAI tool-calling: --enable-auto-tool-choice + --tool-call-parser.
    # Default ON (parser 'hermes' — the Qwen/Most chat-template family):
    # opencode and every agentic client sends tools with
    # tool_choice="auto", which vLLM 400s without these flags.
    "enable_auto_tool_choice",
    "tool_call_parser",
    # Per-model Ray/NCCL transport env overrides for multi-host clusters.
    # A mapping of env-var -> str|int|bool. The cluster script emits every
    # key as `export KEY=VALUE` (VLLM_/NCCL_ prefixes are auto-copied to
    # workers by vLLM's ray/ray_env.py). Absent keys keep the built-in
    # heterogeneous-Ethernet defaults (see vllm_manager); users on
    # Infiniband/NVLink clusters override or drop them here, per model.
    "ray_env",
    # Multi-host parallel layout: "pp" (default) = pipeline-parallel across
    # hosts — layer-split, one activation pass per stage boundary, the
    # llama.cpp-RPC-style layout. Far better over 1xPCIe/Ethernet than TP
    # (which allreduces every layer). "tp" = tensor-parallel across hosts
    # — only sensible with Infiniband/NVLink-class interconnects.
    "parallel_layout",
    # CUDA graphs: default ON (eager) for multi-host Ray clusters —
    # capture hangs across heterogeneous Ethernet TP groups (upstream
    # #36237/#41725). High-speed interconnect users can turn it off.
    "enforce_eager",
    # Attention backend (--attention-backend, vLLM 0.18+): the vLLM
    # equivalent of llama.cpp's flash-attn toggle. flash_attn needs
    # sm_80+ (RTX 2060-class Turing cards cannot use it — vLLM falls
    # back to triton there); auto keeps vLLM's per-device pick.
    "attention_backend",
})
# research pitfall: vLLM's kv cache quantization supports fp8 on SM89+
# hardware ONLY (0.30's Triton attention backend raises on sm_86/sm_75:
# "native FP8 (fp8e4nv) requires SM89+"). On older consumer GPUs the
# supported set is the unquantized ones — auto/half/bfloat16/float16.
# fp8 stays first (the llama.cpp q4_0/q8_0 cache types still have no
# equivalent and must never reach the command line).
VALID_KV_CACHE_DTYPES = ("auto", "fp8", "half", "bfloat16", "float16")
VALID_COMPUTE_DTYPES = ("auto", "half", "float16", "bfloat16", "float32")


def validate_engine(engine, where="settings.yaml"):
    """Engine enum check. A typo must be a startup error listing the valid
    values — never a silent fallback to the default (ARD §2)."""
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"{where}: invalid engine '{engine}' — must be one of "
            f"{', '.join(VALID_ENGINES)}")
    return engine


def inject_engine_defaults(settings):
    """Normalize a settings dict in place: default engine when the key is
    missing (backward compat with existing installs) + each engine's OWN
    settings block defaulted so consumers never None-check it. Raises
    ValueError via validate_engine on a bad engine.

    Adapter v2 (AUDIT-T6 §1.4, DEC-01): the engine-qualified part is
    DELEGATED to the active adapter's settings_section() — today that is
    the `vllm:` block (owned by VllmAdapter); llama.cpp owns no block.
    Unknown-but-valid engine names keep the same strictness: a registered
    adapter is dispatched to; a recognized-but-unregistered engine raises
    (get_engine's message) — never a silent fallback.
    """
    if not isinstance(settings, dict):
        raise ValueError("settings must be a dict")
    settings.setdefault("engine", DEFAULT_ENGINE)
    validate_engine(settings["engine"])
    # Adapter v2 dispatch (AUDIT-T6 §1.4, DEC-01): the ACTIVE adapter owns
    # its settings block. Deferred import — engine_registry imports THIS
    # module at top level, so a module-level import here would be a cycle
    # (same pattern vllm_manager uses). Restored by UX-2 (t_cda5171e): the
    # installer-branch merge dropped the dispatch, so vllm-block validation
    # silently vanished and 15 engine-registry/roundtrip tests failed.
    import engine_registry
    engine_registry.get_engine(settings).settings_section(settings)
    return settings


class _YamlCollection:
    """A named list stored as one YAML file. Owns all CRUD."""

    def __init__(self, path, key, defaults):
        self.path = path
        self.key = key
        self.defaults = dict(defaults)
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _load(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get(self.key) or []

    def _save(self, items):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {self.key: items}, f, default_flow_style=False, sort_keys=False
            )

    def all(self):
        return self._load()

    def get(self, name):
        for item in self.all():
            if item["name"] == name:
                return item
        return None

    def require(self, name):
        item = self.get(name)
        if item is None:
            raise ValueError(f"{self.key[:-1]} '{name}' not found")
        return item

    def add(self, item):
        if self.get(item["name"]) is not None:
            raise ValueError(f"{item['name']} already exists")
        item.setdefault("id", uuid.uuid4().hex[:8])
        for k, v in self.defaults.items():
            item.setdefault(k, v)
        items = self.all()
        items.append(item)
        self._save(items)
        return item

    def update(self, name, updates):
        items = self.all()
        for i, item in enumerate(items):
            if item["name"] == name:
                items[i].update(updates)
                self._save(items)
                return items[i]
        raise ValueError(f"{name} not found")

    def delete(self, name):
        items = self.all()
        remaining = [i for i in items if i["name"] != name]
        self._save(remaining)
        return len(remaining)


MACHINE_DEFAULTS = {
    "ip": "",
    "username": "",
    "password": "",
    "port": 22,
    "mac": "",
    "role": "peer",        # "master" | "peer"
    "os": "linux",          # "linux" | "windows" - drives store:// path resolution
    "gpu_vendor": "nvidia", # "nvidia" | "amd" | "cpu" - drives device naming (CUDA0/ROCm0) + smi tool
    "rpc_port": 50052,
    "gpu": "",              # e.g. "CUDA0,CUDA1,CUDA2"
    "llama_bin": "",         # path to llama.cpp build/bin
    "expose": {"mode": "both", "gpus": []},  # what RPC exposes: both|vram|ram + gpu subset
    "rdma": False,          # enable RDMA transport for llama.cpp RPC on this machine
    "rdma_ip": "",           # RoCE NIC address (must be in the NIC's GID table); used when rdma=True
    "rdma_dev": "",          # optional pin: ibverbs device (e.g. rocep3s0); empty = auto
    "rdma_gid": "",          # optional pin: RoCE v2 GID index on that device; empty = auto
    "rdma_cm": False,        # use rdma_cm QP path resolution (mlx4/CX3); else raw path
    "swap_config_path": "",       # DEC-03: per-machine llama.cpp swap-config file ("" = inherit)
    "swap_config_path_vllm": "",  # DEC-03: per-machine vllm swap-config file ("" = inherit)
}
GROUP_DEFAULTS = {
    "description": "",
    "machines": [],        # list of machine names
    # DEC-17: engine-qualified groups. Both group collections carry an
    # `engine` field (llama_cpp | vllm) so each backend sees its own
    # world (dual-backend ideology: hidden-not-translated). Default
    # llama_cpp keeps existing engine-blind YAML loading exactly as
    # before — the current full-cluster group never vanishes.
    "engine": ENGINE_LLAMA_CPP,
}
# Serve groups (llama-swap MODEL groups: which models may run together).
# Distinct from machine groups above. Members are deployment labels:
# a model_id, or "model_id--variant-label" for variants.
SWAP_GROUP_DEFAULTS = {
    "description": "",
    "members": [],         # list of deployment labels
    "swap": False,         # False = members run together; True = one at a time
    "exclusive": False,    # True = loading a member unloads other groups
    "persistent": False,   # True = other groups can't unload this one
    # DEC-17: engine-qualified serve groups — same shape as GROUP_DEFAULTS.
    # Default llama_cpp keeps existing engine-blind YAML loading verbatim.
    "engine": ENGINE_LLAMA_CPP,
}

def migrate_group_engines():
    """DEC-17 migration: stamp `engine` on existing group entries so the
    engine-blind YAML keeps rendering under the engine it belongs to.

    Machine groups (groups.yaml): every existing group predates the
    split and was created under the llama.cpp world (RPC peer sets) —
    stamped llama_cpp; a group created under vllm is stamped on write
    by the API, so the only un-stamped entries are llama_cpp-era ones.

    Serve groups (swap_groups.yaml): engine inferred from member
    formats — a group whose members resolve to hf-format models is
    vLLM's (llama.cpp cannot serve safetensors), otherwise llama_cpp.
    Existing engine fields are NEVER overwritten. One-time per entry:
    after stamping, the file on disk carries the field (idempotent).
    """
    def _fmt_of_member(mid):
        m = get_model(mid)
        if not m:
            return None
        return (m.get("format")
                or ("gguf" if str(m.get("base_path") or "")
                    .lower().endswith(".gguf") else "hf"))

    def _migrate(collection, infer):
        items = collection.all()
        changed = False
        for g in items:
            if g.get("engine"):
                continue
            g["engine"] = infer(g)
            changed = True
        if changed:
            collection._save(items)

    def _machine_group_engine(g):
        return ENGINE_LLAMA_CPP

    def _serve_group_engine(g):
        fmts = [_fmt_of_member(mem) for mem in (g.get("members") or [])]
        hf = [f for f in fmts if f]
        if hf and all(f == "hf" for f in hf):
            return ENGINE_VLLM
        return ENGINE_LLAMA_CPP

    try:
        _migrate(groups, _machine_group_engine)
    except Exception:
        pass
    try:
        _migrate(swap_groups, _serve_group_engine)
    except Exception:
        pass


machines = _YamlCollection(
    os.path.join(CONFIG_DIR, "machines.yaml"), "machines", MACHINE_DEFAULTS
)


def _inject_machine_secret(m):
    """In-memory only: resolve vault password so SSH callers just work.

    The YAML on disk NEVER holds the secret after migration (it stays "").
    Nothing here writes to disk.
    """
    try:
        import secrets_vault as _vault
        m = dict(m)
        if not m.get("password"):
            pw = _vault.get_machine_password(m)
            if pw:
                m["password"] = pw
        m["password_set"] = bool((m.get("password") or "").strip())
        return m
    except Exception:
        m = dict(m)
        m["password_set"] = bool((m.get("password") or "").strip())
        return m


def _strip_machine_secret(item):
    """Split a machine payload: vault the password, return YAML-safe item."""
    item = dict(item)
    pw = item.pop("password", "")
    mid = item.get("id") or item.get("name") or ""
    if pw and not str(pw).startswith("env:"):
        try:
            import secrets_vault as _vault
            _vault.set_machine_password(mid, pw)
        except Exception:
            item["password"] = pw  # vault unavailable: keep legacy behavior
            return item
    item["password"] = ""
    return item


_orig_machines_all = machines.all
_orig_machines_get = machines.get
_orig_machines_require = machines.require
_orig_machines_add = machines.add
_orig_machines_update = machines.update


def _machines_all():
    return [_inject_machine_secret(m) for m in _orig_machines_all()]


def _machines_get(name):
    m = _orig_machines_get(name)
    return _inject_machine_secret(m) if m else None


def _machines_require(name):
    return _inject_machine_secret(_orig_machines_require(name))


def _machines_add(item):
    return _inject_machine_secret(_orig_machines_add(_strip_machine_secret(item)))


def _machines_update(name, updates):
    updates = dict(updates)
    if "password" in updates:
        pw = updates.pop("password")
        # non-empty = rotate secret; empty/missing = keep existing vault value
        if pw:
            try:
                import secrets_vault as _vault
                target = _orig_machines_get(name) or {}
                _vault.set_machine_password(target.get("id") or name, pw)
            except Exception:
                updates["password"] = pw
    return _inject_machine_secret(_orig_machines_update(name, updates))


machines.all = _machines_all
machines.get = _machines_get
machines.require = _machines_require
machines.add = _machines_add
machines.update = _machines_update


def validate_machine_vllm_delivery(machine):
    """Validate ONE machine record's `vllm_delivery` override (B13).

    Accepted values: "", "venv", "docker" — anything else raises (a typo
    must never silently inherit the global delivery, same strictness as
    the settings vllm.delivery enum). Empty/missing = inherit global.
    Called by resolve_vllm_delivery(); validation on the save path keeps
    bad values out of machines.yaml in the first place.
    """
    delivery = (machine or {}).get(VLLM_MACHINE_DELIVERY_KEY)
    if delivery is None:
        return ""
    if not isinstance(delivery, str):
        raise ValueError(
            f"machine vllm_delivery must be a string (one of "
            f"{', '.join(v or '(inherit global)' for v in VALID_MACHINE_VLLM_DELIVERY)}), "
            f"got {delivery!r}")
    delivery = delivery.strip().lower()
    if delivery not in VALID_MACHINE_VLLM_DELIVERY:
        raise ValueError(
            f"invalid machine vllm_delivery '{delivery}' — must be one of "
            f"{', '.join(v or '(inherit global)' for v in VALID_MACHINE_VLLM_DELIVERY)}")
    return delivery


def resolve_vllm_delivery(machine, global_delivery=None):
    """B13: the delivery method for ONE machine — per-machine
    `vllm_delivery` override wins over the settings `vllm.delivery` global.

    Returns (delivery, reason) where reason explains the choice in user
    words ("machine override", "global default venv", ...). Raises via
    validate_machine_vllm_delivery on a bad override. global_delivery may
    be passed by callers who already loaded the settings block; otherwise
    the default (venv) stands in — install()/plan() always pass the real
    block through vllm_installer._vllm_block().
    """
    override = validate_machine_vllm_delivery(machine)
    if override:
        return override, f"machine override (vllm_delivery: {override})"
    glob = global_delivery or VLLM_DELIVERY_DEFAULT
    return glob, f"global default ({glob})"


# DEC-03 (AUDIT-T6 §1.6): per-engine swap-config override keys for machine
# records — the machine-keyed mirror of swap.yaml's config_path /
# config_path_vllm two-key pattern (same shape as vllm_delivery above).
# "" / missing = inherit the global per-engine path.
SWAP_MACHINE_CONFIG_KEYS = {
    ENGINE_LLAMA_CPP: "swap_config_path",
    ENGINE_VLLM: "swap_config_path_vllm",
}


def machine_swap_config_override(machine, engine):
    """Validated per-machine swap-config path override for ONE engine
    (DEC-03, AUDIT-T6 §1.6).

    Returns "" when the machine inherits the global per-engine path, or a
    non-empty stripped path when it pins one. Raises ValueError on an
    unknown engine or a non-string override — same strictness as
    validate_machine_vllm_delivery: a typo must never silently inherit
    (a swapped-in garbage path would point the daemon at nothing).
    """
    key = SWAP_MACHINE_CONFIG_KEYS.get(engine)
    if key is None:
        raise ValueError(
            f"unknown engine '{engine}' — must be one of "
            f"{', '.join(VALID_ENGINES)}")
    val = (machine or {}).get(key)
    if val is None:
        return ""
    if not isinstance(val, str):
        raise ValueError(f"machine {key} must be a string path, got {val!r}")
    return val.strip()


groups = _YamlCollection(
    os.path.join(CONFIG_DIR, "groups.yaml"), "groups", GROUP_DEFAULTS
)
swap_groups = _YamlCollection(
    os.path.join(CONFIG_DIR, "swap_groups.yaml"), "swap_groups",
    SWAP_GROUP_DEFAULTS,
)


# ---------------- per-model configs (file-per-model, not a collection) ----
def _model_path(model_id):
    return os.path.join(MODELS_DIR, f"{model_id}.yaml")


def _is_private_visible(data, mid=None):
    """Model-visibility gate (access.py contextvar, set per request by
    the ASGI middleware from the VIEW POLICY in settings.yaml — same
    seam the private gate uses). Two layers, both request-scoped:
    - `private: true` models: trusted-request-only (existing behavior —
      darker-scarlett is the reference user).
    - guest view-policy: settings.yaml `views:` maps a view name to a
      list of model tokens; an IP bound to a view sees ONLY those
      models (matched mid or mid--variant). Everyone else unaffected.
    False positives are forbidden, false negatives never: a gate error
    keeps the model visible."""
    try:
        import access
        allow = access.view_allow()
        if allow is not None:
            if mid is None:
                mid = data.get("model_id") or data.get("name") \
                    or data.get("_file") or ""
            label = f'{mid}--{(data.get("_variant_label") or "").strip()}' \
                if data.get("_variant_label") else mid
            # EXACT token match, with one rule for containers: a BASE
            # config row is visible iff the base mid is granted OR any
            # granted mid--variant token belongs to it (the granted
            # variant rides inside its config — its other variants stay
            # hidden; scoping never widens past a granted exact token).
            if data.get("_variant_label"):
                if label not in allow:
                    return False
            elif mid not in allow and not any(
                    t.startswith(mid + "--") for t in allow):
                return False
        if not data.get("private"):
            return True
        return access.gate_open()
    except Exception:
        return True


def _is_mtp_draft(gguf_path):
    """Return True if the GGUF (or its shards) contains MTP draft tensors."""
    if not gguf_path:
        return False
    # Only check GGUF files; other formats (safetensors, etc.) don't have MTP draft tensors in this sense.
    if not gguf_path.lower().endswith('.gguf'):
        return False
    # Use cache to avoid re-reading the same file
    if gguf_path in _mtp_cache:
        return _mtp_cache[gguf_path]
    try:
        # Resolve store:// paths etc.
        resolved = path_resolver.resolve(gguf_path)
        result = gguf_inspect.has_mtp_layers_anywhere(resolved)
    except Exception:
        # If we can't read the file, assume it's not an MTP draft to avoid hiding valid models.
        result = False
    _mtp_cache[gguf_path] = result
    return result


def list_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    # First pass: collect all target base_paths (store:// strings) from model configs and variants
    target_base_paths = set()
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".yaml"):
            continue
        if "__tune_tmp" in fname or "__ab_single" in fname:
            continue
        try:
            with open(os.path.join(MODELS_DIR, fname), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            continue
        base_path = data.get("base_path")
        if base_path:
            target_base_paths.add(base_path)
        for variant in data.get("variants", []):
            v_base_path = variant.get("base_path")
            if v_base_path:
                target_base_paths.add(v_base_path)
    # Second pass: build the model list
    models = []
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".yaml"):
            continue
        if "__tune_tmp" in fname or "__ab_single" in fname:
            continue  # tuner scratch configs: never listed/tunable/deleted via UI
        try:
            with open(os.path.join(MODELS_DIR, fname), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            continue
        # Private-model gate (access.py): `private: true` configs vanish
        # for untrusted viewers — checked before anything else so no
        # trace (arch, decision_type, id) leaks into the listing.
        if not _is_private_visible(data):
            continue
        data["_file"] = fname
        if "variants" not in data or data["variants"] is None:
            data["variants"] = []
        base_path = data.get("base_path")
        # Skip if the base_path is a .gguf file and the basename contains "mtp" or "MTP" and it is not a target
        if base_path and base_path.lower().endswith('.gguf'):
            # Extract the filename (the last component of the path)
            # For store://llm-models/file.gguf, we want 'file.gguf'
            filename = base_path.split('/')[-1]
            if 'mtp' in filename.lower():
                # Check if this base_path is in the target_base_paths
                if base_path not in target_base_paths:
                    # Skip this model config because it is an MTP draft not used as a target
                    continue
        data["architecture"] = _get_model_architecture(base_path)
        # TypeSafe decision models (laya/clef/...): serve /v1/systemone
        # only — surfaces the class next to the architecture everywhere.
        data["decision_type"] = _get_decision_type(base_path)
        data["variants"] = _filtered_variants(data)
        models.append(data)
    return models


def _filtered_variants(data):
    """Strip variant entries the CURRENT REQUEST's key is not granted.
    The container rule in _is_private_visible widens BASE visibility to
    any granted mid--variant token (a granted variant rides inside its
    config) — but the embedded `variants:` list itself was returned
    whole, so a key granted only the BASE still saw the full variant
    family (observed: Dad's `--dad` variants visible to the zazi key on
    the Launch grid). Scoping never widens: variants are separate
    grants, matched EXACTLY (mid--label). Gate-open callers (trusted
    console / unrestricted chip) keep every variant. Gate errors keep
    the full list (fail-open, same rule as _is_private_visible)."""
    try:
        import access
        allow = access.view_allow()
        if allow is None:
            return data.get("variants") or []
        mid = data.get("model_id") or data.get("name") \
            or data.get("_file") or ""
        out = []
        for var in (data.get("variants") or []):
            label = (var.get("label") or "").strip()
            if not label:
                continue
            if f"{mid}--{label}" in allow:
                out.append(var)
        return out
    except Exception:
        return data.get("variants") or []


def get_model(model_id):
    path = _model_path(model_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return None
    # Private-model gate (access.py): a `private: true` config resolves
    # to None for untrusted callers — same result as a missing file, so
    # launch/chat/edit paths treat it as never existing.
    if not _is_private_visible(data):
        return None
    data["_file"] = f"{model_id}.yaml"
    if "variants" not in data or data["variants"] is None:
        data["variants"] = []
    # same per-request variant narrowing as list_models (a scoped key
    # granted the base must not see ungranted variants through the
    # detail/editor path)
    data["variants"] = _filtered_variants(data)
    return data


def save_model(model_id, data):
    mid = (model_id or "").strip()
    if not mid:
        raise ValueError("model_id is required — cannot save blank model")
    # guard against path traversal / empty filename like ".yaml"
    if mid in (".", "..") or "/" in mid or "\\" in mid:
        raise ValueError(f"invalid model_id '{mid}'")
    os.makedirs(MODELS_DIR, exist_ok=True)
    data.pop("_file", None)
    # additive vllm: section round-trip (ARD §3.3): validate + normalize in
    # place — flags and other llama.cpp keys pass through untouched.
    apply_vllm_section(data, where=f"model '{mid}'")
    # A2: variants may carry their own vllm: overrides (shallow-merged
    # over the primary at plan/launch time) — sanitize each with the same
    # schema so a bad per-variant value can't reach disk.
    for var in data.get("variants") or []:
        if isinstance(var, dict) and var.get("vllm") is not None:
            apply_vllm_section(
                var, where=f"model '{mid}' variant '{var.get('label') or '?'}'")
    # don't pollute existing single-config files with empty variants
    if not data.get("variants"):
        data.pop("variants", None)
    # ensure stored id is stripped
    data["model_id"] = mid
    data["updated"] = datetime.now().isoformat(timespec="seconds")
    with open(_model_path(mid), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return get_model(mid)


def delete_model(model_id):
    path = _model_path(model_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------- per-model `vllm:` section round-trip (ARD 3.3) ----------
# DEC-04: the sanitize/apply LOGIC moved to VllmAdapter
# (vllm_manager.VllmAdapter.sanitize_model_section / apply_model_section) --
# the adapter owns its per-model section (AUDIT-T6 1.3). These module
# functions are kept as thin delegates so existing call sites (save_model,
# tests) keep working without importing vllm_manager (which would drag
# rpc_manager into every config_loader import -- the lazy registry import
# avoids that).

def sanitize_vllm_section(raw, where="vllm"):
    """Delegate to the owning adapter (DEC-04). Same errors, same shapes."""
    import engine_registry
    return engine_registry.get_engine_section_adapter().sanitize_model_section(
        raw, where=where)


def apply_vllm_section(data, where="vllm"):
    """Delegate to the owning adapter (DEC-04). Same errors, same shapes."""
    import engine_registry
    return engine_registry.get_engine_section_adapter().apply_model_section(
        data, where=where)


def _machine_hw_slots(m):
    """Current GPU slot tokens for a machine config ('CUDA0', ...),
    plus whether RAM counts as a selectable resource."""
    import gpu_vendor
    det = m.get("detected") or {}
    n_gpus = len(det.get("gpu_names") or [])
    if not n_gpus:
        n_gpus = len([g for g in (m.get("gpu") or "").split(",") if g.strip()])
    pref = gpu_vendor.gpu_prefix(m)
    slots = [] if pref == "Metal" else [f"{pref}{i}" for i in range(n_gpus)]
    has_ram = bool((det.get("ram_total_gb") or m.get("ram_total_gb") or 0) > 0)
    return slots, has_ram


def sync_models_to_hardware():
    """After hardware detection, reconcile every saved model config's
    device_order / expose blocks with the CURRENT machine hardware slots.
    Dead slots (GPU count shrank / vendor changed) are dropped, new slots
    appended, order preserved, RAM entries kept iff RAM is available.
    Skips tuner scratch configs. Returns list of updated model ids."""
    import gpu_vendor
    hw = {}
    for m in machines.all():
        slots, has_ram = _machine_hw_slots(m)
        hw[m["name"]] = (slots, has_ram)
    updated = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".yaml") or "__tune_tmp" in fname or "__ab_single" in fname:
            continue
        path = os.path.join(MODELS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            continue
        dirty = False

        def fix_chips(chips):
            """Clamp a chip list to current hardware; returns new list."""
            out = []
            for c in chips:
                c = (c or "").strip()
                if not c or "-" not in c:
                    continue
                mn, res = c.rsplit("-", 1)
                slots, has_ram = hw.get(mn, ([], False))
                if res == "RAM":
                    if has_ram and c not in out:
                        out.append(c)
                elif res in slots and c not in out:
                    out.append(c)
                dirty = True
            # append missing GPU slots of machines already referenced
            for c in list(out):
                mn, res = c.rsplit("-", 1)
                if res == "RAM":
                    continue
                slots, _ = hw.get(mn, ([], False))
                for s in slots:
                    tok = f"{mn}-{s}"
                    if tok not in out:
                        out.append(tok)
                        dirty = True
            return out

        order = data.get("device_order")
        if order:
            chips = [c for c in order.split(",") if c.strip()]
            new_chips = fix_chips(chips)
            if new_chips != chips:
                data["device_order"] = ",".join(new_chips)
                dirty = True

        expose = data.get("expose")
        if isinstance(expose, dict):
            for mn in list(expose.keys()):
                block = expose.get(mn) or {}
                slots, _ = hw.get(mn, ([], False))
                if not slots and not (block.get("gpus")):
                    continue
                gpus = block.get("gpus") or []
                new_gpus = [g for g in gpus if g in slots]
                for s in slots:
                    if s not in new_gpus:
                        new_gpus.append(s)
                if new_gpus != gpus:
                    block["gpus"] = new_gpus
                    dirty = True

        # variants carry their own device_order / expose
        for var in data.get("variants") or []:
            if not isinstance(var, dict):
                continue
            vorder = var.get("device_order")
            if vorder:
                vchips = [c for c in vorder.split(",") if c.strip()]
                vnew = fix_chips(vchips)
                if vnew != vchips:
                    var["device_order"] = ",".join(vnew)
                    dirty = True
            vexpose = var.get("expose")
            if isinstance(vexpose, dict):
                for mn in list(vexpose.keys()):
                    block = vexpose.get(mn) or {}
                    slots, _ = hw.get(mn, ([], False))
                    if not slots and not (block.get("gpus")):
                        continue
                    gpus = block.get("gpus") or []
                    new_gpus = [g for g in gpus if g in slots]
                    for s in slots:
                        if s not in new_gpus:
                            new_gpus.append(s)
                    if new_gpus != gpus:
                        block["gpus"] = new_gpus
                        dirty = True

        if dirty:
            data.pop("_file", None)
            data["updated"] = datetime.now().isoformat(timespec="seconds")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            updated.append(data.get("model_id") or fname[:-5])
    return updated