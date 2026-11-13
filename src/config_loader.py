"""Read/write YAML config files: machines, groups, per-model configs.

All configs live in ./config next to the app. Nothing outside this
folder is ever written to.
"""
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import yaml

import path_resolver
from runtime_paths import data_dir
import gguf_inspect
from secret_fields import strip_secret_keys

# Cache for MTP detection results keyed by resolved absolute path to avoid re-reading GGUF files
_mtp_cache = {}
# One GGUF header read yields both values (architecture + decision type).
# Keyed by the config's store:// path; a network store has high per-read
# latency, so this cache is what makes the model list fast after boot.
_meta_cache = {}


def _get_model_meta(gguf_path):
    if not gguf_path or not gguf_path.lower().endswith('.gguf'):
        return "unknown", ""
    if gguf_path in _meta_cache:
        return _meta_cache[gguf_path]
    arch, dtype = "unknown", ""
    try:
        meta = gguf_inspect.metadata(path_resolver.resolve(gguf_path))
        arch = str(meta.get("general.architecture", "") or "").strip().lower() \
            or "unknown"
        if arch != "unknown":
            dtype = str(meta.get(f"{arch}.decision.type", "") or "") \
                .strip().lower()
    except Exception:
        pass
    _meta_cache[gguf_path] = (arch, dtype)
    return arch, dtype


def _bulk_model_meta(paths):
    """Warm the GGUF metadata cache for many models concurrently."""
    unique = [p for p in dict.fromkeys(paths)
              if p and p.lower().endswith('.gguf')]
    if not unique:
        return
    with ThreadPoolExecutor(max_workers=min(16, len(unique))) as pool:
        list(pool.map(_get_model_meta, unique))


def _get_model_architecture(gguf_path):
    return _get_model_meta(gguf_path)[0]


def _get_decision_type(gguf_path):
    """TypeSafe decision-model type ('laya', ...) or ''. Models carrying
    <arch>.decision.type serve only /v1/systemone (no text generation),
    so the editor/safe-defaults treat them like a distinct model class
    (same shape as embeddings/ASR, not chat)."""
    return _get_model_meta(gguf_path)[1]

BASE_DIR = data_dir()
CONFIG_DIR = os.path.join(BASE_DIR, "config")
MODELS_DIR = os.path.join(CONFIG_DIR, "models")
FLOW_LAYOUT_FILE = os.path.join(CONFIG_DIR, "flow_layout.json")

# ---------------- engine selection (settings.yaml) -------------------------
# Schema owned by src/engine_registry.py (load_settings / get_engine);
# these helpers re-export the vocabulary so callers can validate without
# importing the registry (which pulls in rpc_manager).
ENGINE_LLAMA_CPP = "llama_cpp"
VALID_ENGINES = (ENGINE_LLAMA_CPP,)
DEFAULT_ENGINE = ENGINE_LLAMA_CPP


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
    DELEGATED to the active adapter's settings_section() — llama.cpp owns
    no block. Unknown-but-valid engine names keep the same strictness: a
    registered adapter is dispatched to; a recognized-but-unregistered
    engine raises (get_engine's message) — never a silent fallback.
    """
    if not isinstance(settings, dict):
        raise ValueError("settings must be a dict")
    settings.setdefault("engine", DEFAULT_ENGINE)
    validate_engine(settings["engine"])
    # Adapter v2 dispatch (AUDIT-T6 §1.4, DEC-01): the ACTIVE adapter owns
    # its settings block. Deferred import — engine_registry imports THIS
    # module at top level, so a module-level import here would be a cycle.
    # Restored by UX-2 (t_cda5171e): the installer-branch merge dropped the
    # dispatch, so block validation silently vanished and 15
    # engine-registry/roundtrip tests failed.
    import engine_registry
    engine_registry.get_engine(settings).settings_section(settings)
    return settings


class _YamlCollection:
    """A named list stored as one YAML file. Owns all CRUD."""

    def __init__(self, path, key, defaults):
        self.path = path
        self.key = key
        self.defaults = dict(defaults)
        self._cache = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _load(self):
        try:
            st = os.stat(self.path)
        except OSError:
            return []
        key = (self.path, st.st_mtime_ns, st.st_size)
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get(self.key) or []
        self._cache = (key, items)
        return items

    def _save(self, items):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {self.key: items}, f, default_flow_style=False, sort_keys=False
            )
        try:
            st = os.stat(self.path)
            self._cache = ((self.path, st.st_mtime_ns, st.st_size), items)
        except OSError:
            self._cache = None

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
    # engine update/install strategy: "auto" (detect), "prebuilt" (fast
    # binary, switch away from a source build), "source" (rebuild in place)
    "engine_mode": "auto",
    # explicit prebuilt backend pin: auto|cpu|cuda|rocm|sycl|vulkan|metal.
    # "vulkan" keeps an Intel/CPU host on the Vulkan asset across updates.
    "engine_backend": "auto",
    "expose": {"mode": "both", "gpus": []},  # what RPC exposes: both|vram|ram + gpu subset
    "rdma": False,          # enable RDMA transport for llama.cpp RPC on this machine
    "rdma_ip": "",           # RoCE NIC address (must be in the NIC's GID table); used when rdma=True
    "rdma_dev": "",          # optional pin: ibverbs device (e.g. rocep3s0); empty = auto
    "rdma_gid": "",          # optional pin: RoCE v2 GID index on that device; empty = auto
    "rdma_cm": False,        # use rdma_cm QP path resolution (mlx4/CX3); else raw path
    "swap_config_path": "",       # DEC-03: per-machine llama.cpp swap-config file ("" = inherit)
    # Peer agent (docs/peer-agent.md). Runtime is agent-only; the legacy
    # legacy transport value is accepted but no longer changes behaviour.
    "transport": "auto",
    "agent_port": 8198,
}
GROUP_DEFAULTS = {
    "description": "",
    "machines": [],        # list of machine names
    # DEC-17: engine-qualified groups. Both group collections carry an
    # `engine` field so each backend sees its own world. Default
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

    Every existing group predates the engine split and was created under
    the llama.cpp world — stamped llama_cpp. Existing engine fields are
    NEVER overwritten. One-time per entry: after stamping, the file on
    disk carries the field (idempotent).
    """
    def _migrate(collection):
        items = collection.all()
        changed = False
        for g in items:
            if g.get("engine"):
                continue
            g["engine"] = ENGINE_LLAMA_CPP
            changed = True
        if changed:
            collection._save(items)

    try:
        _migrate(groups)
    except Exception:
        pass
    try:
        _migrate(swap_groups)
    except Exception:
        pass


machines = _YamlCollection(
    os.path.join(CONFIG_DIR, "machines.yaml"), "machines", MACHINE_DEFAULTS
)


_MACHINE_DERIVED_FIELDS = ("password_set", "agent_token_set")


def _read_safe_machine(m):
    """Return a machine record safe to hand to callers: no secret fields.

    The vault is the only secret store. A record in memory must never carry
    a plaintext password or token, so reads strip every secret-named field.
    Response-only flags (``password_set``/``agent_token_set``) are also
    dropped: they are recomputed from the vault at the API boundary.
    """
    m = strip_secret_keys(dict(m))
    for key in _MACHINE_DERIVED_FIELDS:
        m.pop(key, None)
    return m


def _strip_machine_secret(item):
    """Write-boundary scrub: vault secrets, then return a YAML-safe copy.

    Called for every item on save, so it must be idempotent: a password that
    already matches the vault is not re-encrypted (that used to rewrite
    .env once per machine on every save). A legacy/transient token is
    vaulted when this machine has no per-machine entry yet, then dropped.
    If the vault is unavailable the secret is kept in place rather than
    silently lost.
    """
    item = dict(item)
    mid = item.get("id") or item.get("name") or ""
    try:
        import secrets_vault as _vault
    except Exception:
        _vault = None

    kept = {}
    for key in ("_agent_token", "agent_token"):
        token = str(item.pop(key, "") or "").strip()
        if not token:
            continue
        if _vault is not None and mid:
            try:
                if _vault.machine_agent_token_stored(item):
                    continue
                _vault.set_machine_agent_token(mid, token)
                continue
            except Exception:
                pass
        kept[key] = token

    password = str(item.pop("password", "") or "")
    if password and not password.startswith("env:"):
        vaulted = False
        if _vault is not None and mid:
            try:
                if _vault.get_machine_password({"id": mid}) == password:
                    vaulted = True
                else:
                    _vault.set_machine_password(mid, password)
                    vaulted = True
            except Exception:
                pass
        if not vaulted:
            kept["password"] = password

    safe = strip_secret_keys(item)
    for key in _MACHINE_DERIVED_FIELDS:
        safe.pop(key, None)
    safe["password"] = kept.pop("password", "")
    safe.update(kept)
    return safe


_orig_machines_all = machines.all
_orig_machines_get = machines.get
_orig_machines_require = machines.require
_orig_machines_add = machines.add
_orig_machines_update = machines.update
_orig_machines_save = machines._save


def _machines_save(items):
    """The single machines.yaml write boundary.

    Every add/update/delete funnels through here, so a password or token
    can never be written back as plaintext. Legacy values are vaulted by
    ``_strip_machine_secret`` before the YAML is written.
    """
    _orig_machines_save([_strip_machine_secret(item) for item in items])


def _machines_all():
    return [_read_safe_machine(m) for m in _orig_machines_all()]


def _machines_get(name):
    m = _orig_machines_get(name)
    return _read_safe_machine(m) if m else None


def _machines_require(name):
    return _read_safe_machine(_orig_machines_require(name))


def _machines_add(item):
    # The save boundary vaults any password and drops secret fields, so this
    # is just the secret-free view of the stored record.
    return _read_safe_machine(_orig_machines_add(item))


def _machines_update(name, updates):
    updates = dict(updates)
    # Empty/missing password means "keep the existing vault value"; a
    # non-empty one is vaulted by the save boundary.
    if not str(updates.get("password") or "").strip():
        updates.pop("password", None)
    return _read_safe_machine(_orig_machines_update(name, updates))


def migrate_machine_secrets():
    """Force the write boundary when secrets are still at rest in YAML.

    Historical builds could persist a plaintext ``_agent_token`` (or a
    hand-edited ``agent_token``/``password``). A save vaults and strips
    them; this hook simply triggers that once at startup instead of waiting
    for the next machine edit. Idempotent and best-effort.
    """
    raw = _orig_machines_all()

    def has_secret(item):
        if str(item.get("password") or "").strip():
            return True
        return any(str(item.get(key) or "").strip()
                   for key in ("_agent_token", "agent_token"))

    if any(has_secret(item) for item in raw):
        try:
            machines._save(raw)
        except Exception:
            pass


machines.all = _machines_all
machines.get = _machines_get
machines.require = _machines_require
machines.add = _machines_add
machines.update = _machines_update
machines._save = _machines_save


# DEC-03 (AUDIT-T6 §1.6): per-engine swap-config override keys for machine
# records — the machine-keyed mirror of swap.yaml's config_path pattern.
# "" / missing = inherit the global per-engine path.
SWAP_MACHINE_CONFIG_KEYS = {
    ENGINE_LLAMA_CPP: "swap_config_path",
}


def machine_swap_config_override(machine, engine):
    """Validated per-machine swap-config path override for ONE engine
    (DEC-03, AUDIT-T6 §1.6).

    Returns "" when the machine inherits the global per-engine path, or a
    non-empty stripped path when it pins one. Raises ValueError on an
    unknown engine or a non-string override — a typo must never silently
    inherit (a swapped-in garbage path would point the daemon at nothing).
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


def _rename_model_machine_refs(value, old, new):
    """Rewrite machine references inside one model config.

    Targeted keys only (servers / device_order / expose): a model id or
    display name that happens to equal a machine name must never change.
    """
    changed = False
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "servers" and isinstance(item, list):
                nv = [new if str(x) == old else x for x in item]
                changed = changed or nv != item
                out[key] = nv
            elif key == "device_order" and isinstance(item, str):
                token = f"{old}-"
                nv = item.replace(token, f"{new}-") if token in item else item
                changed = changed or nv != item
                out[key] = nv
            elif key == "expose" and isinstance(item, dict):
                nv = {}
                for mk, mv in item.items():
                    nk = new if str(mk) == old else mk
                    changed = changed or nk != mk
                    nv[nk] = mv
                out[key] = nv
            else:
                nv, ch = _rename_model_machine_refs(item, old, new)
                changed = changed or ch
                out[key] = nv
        return out, changed
    if isinstance(value, list):
        out = []
        for item in value:
            nv, ch = _rename_model_machine_refs(item, old, new)
            changed = changed or ch
            out.append(nv)
        return out, changed
    return value, False


def rename_machine_references(old, new):
    """Cascade a machine rename into model configs, groups, and flow layout.

    Machine records are keyed by name; models/groups/layout reference that
    name. Without this, a rename silently orphans every deployment on the
    machine (swap generation skips them). Returns what changed.
    """
    old = str(old or "").strip()
    new = str(new or "").strip()
    if not old or not new or old == new:
        return {}
    result = {"models": [], "groups": 0, "flow_layout": 0}

    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".yaml") or "__tune_tmp" in fname                 or "__ab_single" in fname:
            continue
        path = os.path.join(MODELS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            continue
        updated, changed = _rename_model_machine_refs(data, old, new)
        if not changed:
            continue
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(updated, f, default_flow_style=False,
                           sort_keys=False)
        os.replace(tmp, path)
        result["models"].append(fname[:-5])

    items = groups.all()
    gchanged = False
    for g in items:
        machines = g.get("machines")
        if isinstance(machines, list) and old in machines:
            g["machines"] = [new if str(m) == old else m for m in machines]
            gchanged = True
    if gchanged:
        groups._save(items)
        result["groups"] = 1

    if os.path.exists(FLOW_LAYOUT_FILE):
        try:
            with open(FLOW_LAYOUT_FILE, "r", encoding="utf-8") as f:
                layout = json.load(f) or {}
            key = f"machine:{old}"
            nkey = f"machine:{new}"
            if key in layout and nkey not in layout:
                layout[nkey] = layout.pop(key)
                with open(FLOW_LAYOUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(layout, f, indent=2)
                result["flow_layout"] = 1
        except Exception:
            pass
    return result


def list_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    loaded = []
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
        loaded.append((fname, data))

    # MTP draft detection needs the FULL config set (before visibility
    # filtering): only a draft referenced as some model's target may stay.
    target_base_paths = set()
    for _fname, data in loaded:
        base_path = data.get("base_path")
        if base_path:
            target_base_paths.add(base_path)
        for variant in data.get("variants") or []:
            if isinstance(variant, dict) and variant.get("base_path"):
                target_base_paths.add(variant["base_path"])

    candidates = []
    for fname, data in loaded:
        if not _is_private_visible(data):
            continue
        data["_file"] = fname
        if "variants" not in data or data["variants"] is None:
            data["variants"] = []
        base_path = data.get("base_path")
        if base_path and base_path.lower().endswith('.gguf'):
            if ('mtp' in base_path.split('/')[-1].lower()
                    and base_path not in target_base_paths):
                continue
        candidates.append(data)

    # Header reads are the cold-start cost (network store): do them once,
    # concurrently, instead of two serial reads per model.
    _bulk_model_meta([d.get("base_path") for d in candidates])

    models = []
    for data in candidates:
        data["architecture"] = _get_model_architecture(data.get("base_path"))
        data["decision_type"] = _get_decision_type(data.get("base_path"))
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