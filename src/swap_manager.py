"""Launch All via llama-swap: build llama-swap's config.yaml dynamically
from our model configs, push it to the host machine, and manage the
llama-swap daemon (systemd on Linux, detached process on Windows).

llama-swap lazy-loads models (cmd runs on first request), so starting
the daemon is safe -- no VRAM is touched until traffic arrives.
"""
import os
import shutil
import time
from datetime import datetime

import yaml

import config_loader as cfg
import local_node
import rpc_manager
from steps import run_steps

SWAP_SETTINGS_FILE = os.path.join(cfg.CONFIG_DIR, "swap.yaml")
# runtime secrets sidecar (vault -> host at push time, 0600, never committed)
SIDECAR_NAME = "swap-secrets.env"

# Extra daemon flags: --watch-config makes llama-swap reload its routing
# table on config change, so a push-only refresh (no restart) is enough to
# make newly saved models/variants routable. Loaded models keep serving.
SWAP_DAEMON_ARGS = "--watch-config"


class _LiteralStr(str):
    pass


def _swap_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


class _SwapDumper(yaml.SafeDumper):
    pass


_SwapDumper.add_representer(_LiteralStr, _swap_representer)


# ---------------- host + settings ----------------
def get_host(name=None):
    """The machine llama-swap runs on. Explicit name wins (so a Linux
    box like aibox1 can host even when the app runs elsewhere);
    otherwise the local machine (which should be the master node)."""
    if name:
        return cfg.machines.require(name)
    for m in cfg.machines.all():
        if local_node.is_local(m):
            return m
    for m in cfg.machines.all():
        if m.get("role") == "master":
            return m
    raise ValueError("no local or master machine configured")


def default_config_path(host):
    if (host.get("os", "linux")) == "windows":
        return "%USERPROFILE%\\.config\\llama-swap\\config.yaml"
    return "~/.config/llama-swap/config.yaml"


# DEC-03 (AUDIT-T6 §1.6): per-engine swap-config files. The swap DIR is
# shared (one llama-swap daemon, one binary/port); the FILE is
# engine-qualified so each engine's entries/scripts/cmdStop shapes never
# collide — llama.cpp keeps the historical config.yaml, vllm gets
# config.vllm.yaml (same dir, sibling file).
VLLM_CONFIG_SUFFIX = "config.vllm.yaml"


def _engine_qualify_path(config_path, engine):
    """Engine-qualify ONE config path: llama.cpp keeps the path verbatim
    (backward compat — existing installs and machines.yaml overrides keep
    working); vllm replaces the basename with config.vllm.yaml in the
    same directory (~/x/config.yaml -> ~/x/config.vllm.yaml). Idempotent
    when the basename is already config.vllm.yaml. A bare filename (no
    directory) stays bare; empty stays empty."""
    if not config_path:
        return config_path
    if engine == "vllm":
        # split at the LAST separator of either kind: a
        # %USERPROFILE%\...\config.yaml string must qualify correctly even
        # when the app runs on POSIX (os.path.split alone only understands
        # the platform separator) — slice the ORIGINAL string so the
        # directory's separators stay exactly as the operator wrote them
        stripped = config_path.rstrip("/\\") or config_path
        cut = max(stripped.rfind("/"), stripped.rfind("\\"))
        base = stripped[cut + 1:] if cut >= 0 else stripped
        if base == VLLM_CONFIG_SUFFIX:
            return config_path
        return stripped[:cut + 1] + VLLM_CONFIG_SUFFIX if cut >= 0 \
            else VLLM_CONFIG_SUFFIX  # bare filename stays bare
    return config_path


# DEC-03: paths set EXPLICITLY for an engine (swap.yaml config_path_vllm /
# machines.yaml swap_config_path / swap_config_path_vllm) are taken
# verbatim — the operator pinned a file, the daemon must use exactly that
# file. Engine-qualification (deriving config.vllm.yaml from llama.cpp's
# path) applies only when vllm has no explicit path of its own.
def _engine_select_path(raw, explicit, engine):
    if explicit and raw:
        return raw
    return _engine_qualify_path(raw, engine)


def engine_swap_config_path(engine, config_path=None, host_name=None):
    """The ACTIVE engine's swap-config file for the current host (DEC-03,
    AUDIT-T6 §1.6). Returns (eff_path, fell_back).

    Resolution order: the host machine's `swap_config_path` /
    `swap_config_path_vllm` override (config_loader.machine_swap_config_
    override) wins when set; else the explicit config_path param; else the
    global per-engine path (swap.yaml `config_path` for llama.cpp /
    `config_path_vllm` for vllm, falling back to engine-qualifying
    llama.cpp's path when vllm's is unset — same dir, sibling file); else
    the host-OS default. Paths set explicitly for their engine
    (config_path_vllm, machine overrides) are
    taken verbatim; the DERIVED default (llama.cpp's path in vllm mode
    with no config_path_vllm) is engine-qualified — basename becomes
    config.vllm.yaml in the same dir, idempotent when already qualified.
    llama.cpp's path is always verbatim. Raises on unknown engine — never
    a silent fallback (the daemon would point at the wrong file).
    """
    if engine not in ("llama_cpp", "vllm"):
        raise ValueError(
            f"unknown engine '{engine}' — must be llama_cpp or vllm")
    settings = load_settings()
    host = get_host(host_name or settings["host_name"])
    explicit = bool(settings.get("config_path_vllm"))
    if config_path:
        raw = config_path
    elif engine == "vllm":
        raw = settings.get("config_path_vllm") or settings["config_path"]
    else:
        raw = settings["config_path"]
    eff, fell_back = _effective_config_path(host, raw)
    # per-machine override (same shape as vllm_delivery): a machine may
    # pin its own file, winning over the global per-engine path
    try:
        override = cfg.machine_swap_config_override(host, engine)
    except ValueError as e:
        raise ValueError(f"{host.get('name') or 'host'}: {e}") from e
    if override:
        explicit = True
        eff, fell_back = _effective_config_path(host, override)
    return _engine_select_path(eff, explicit, engine), fell_back


def load_settings():
    data = {}
    if os.path.exists(SWAP_SETTINGS_FILE):
        with open(SWAP_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    host = get_host(data.get("host_name"))
    return {
        "host": host["name"],
        "host_os": host.get("os", "linux"),
        # LOCAL host reports loopback: llama-swap binds 127.0.0.1:8080
        # (loopback-only since the swap-gateway lockdown), so any
        # consumer dialing st["host_ip"] must get an address that
        # actually answers. Remote hosts keep their config IP.
        "host_ip": "127.0.0.1" if local_node.is_local(host) else host["ip"],
        "host_name": data.get("host_name"),  # None = auto (local master)
        "binary": data.get("binary") or "llama-swap",
        "listen_port": data.get("listen_port") or 8080,
        "config_path": data.get("config_path") or default_config_path(host),
        # DEC-03: vllm's swap-config path (None = derive by engine-qualifying
        # config_path at generate time — engine_swap_config_path handles it)
        "config_path_vllm": data.get("config_path_vllm"),
        "launch_config": data.get("launch_config"),  # persisted launch settings
    }


def save_settings(binary=None, listen_port=None, config_path=None,
                  host_name=None, launch_config=None,
                  config_path_vllm=None):
    data = {}
    if os.path.exists(SWAP_SETTINGS_FILE):
        with open(SWAP_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if binary is not None:
        data["binary"] = binary
    if listen_port is not None:
        data["listen_port"] = int(listen_port)
    if config_path is not None:
        data["config_path"] = config_path
    if config_path_vllm is not None:
        # "" clears back to derive-from-config_path
        data["config_path_vllm"] = config_path_vllm or None
    if host_name is not None:
        data["host_name"] = host_name or None  # "" clears back to auto
    if launch_config is not None:
        data["launch_config"] = launch_config
    os.makedirs(os.path.dirname(SWAP_SETTINGS_FILE), exist_ok=True)
    with open(SWAP_SETTINGS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return load_settings()


# ---------------- config generation ----------------
def _swap_key(model, taken):
    """llama-swap model key: stable model_id first (so API `model` param matches),
    fallback to display name. Deduped so two models can never collide silently."""
    base = (model.get("model_id") or "").strip() or (model.get("name") or "").strip() \
        or model.get("_file", "unknown")
    key, i = base, 2
    while key in taken:
        key = f"{base}-{i}"
        i += 1
    if key != base:
        taken_warn = f"{base}: duplicate display name, using '{key}'"
    else:
        taken_warn = None
    taken.add(key)
    return key, taken_warn


def _effective_config_path(host, config_path):
    """config_path, or the host-OS default when the saved path belongs to
    the other OS (e.g. settings saved on Windows, app now on Linux)."""
    import re
    host_os = host.get("os", "linux")
    if config_path:
        is_posix_path = (config_path.startswith("~/")
                         or (config_path.startswith("/")
                             and not config_path.startswith("//")))
        is_win_path = ("\\" in config_path or "%" in config_path
                       or re.match(r"^[A-Za-z]:", config_path))
        if host_os in ("linux", "macos"):
            if is_win_path:
                return default_config_path(host), True
        else:
            if is_posix_path:
                return default_config_path(host), True
        return config_path, False
    return default_config_path(host), False


def _resolve_launch_mode(deploy_model, host_name, peers):
    """Decide rpc | local for a llama.cpp deployment.

    Explicit launch_mode (local/rpc) on the model or variant wins.
    launch_mode=auto (the default) picks from HARDWARE: when every
    device_order chip maps to a single peer machine — no host-local
    chips, no chips spread across boxes — the deployment can only ever
    use that one box's accelerators, so RPC chaining would add network
    hops for nothing. Return "local" there; anything multi-box or
    host-involved stays "rpc". Models with no device_order (or no peers)
    keep the classic path. Single-box verdicts targeting a WINDOWS box
    return "rpc": Windows remotes cannot read model files over SSH-launch
    sessions (no SMB credentials), so they only ever serve as RPC peers.
    """
    raw = str(deploy_model.get("launch_mode") or "auto").strip().lower()
    if raw in ("local", "rpc"):
        return raw
    if raw != "auto":
        raw = "auto"
    # MLX runtime never uses RPC peers (single machine only — the
    # generator forces the direct-command path downstream): decide local
    # up-front so every consumer of the heuristic agrees, whatever the
    # chips say.
    if (deploy_model.get("runtime") or "llama") == "mlx":
        return "local"
    if not peers:
        return "rpc"
    # Windows boxes can never be local-execution targets: over SSH they
    # run network-class logons with no SMB credentials, so a remotely
    # launched llama-server there cannot read store:// model files
    # (observed: winbox loads die instantly with missing-model). Force
    # the RPC chain instead — the host reads the model, the Windows box
    # only runs ggml-rpc-server (no model file needed). Explicit
    # launch_mode pins still win (operator override).
    if len(peers) == 1 and _is_windows_box(peers[0]):
        return "rpc"
    order = deploy_model.get("device_order") or ""
    import rpc_manager as _rm
    if not order.strip():
        # No explicit chips: a single-peer deployment whose server list
        # EXCLUDES the host is unambiguous — the config deliberately runs
        # only that box (e.g. qwen3-asr on winbox) → local. When the host
        # IS in servers, _auto_device_order includes the HOST's GPUs, so
        # a GPU-less-host deployment spans two boxes and must stay rpc
        # (observed: Q8_0-RPC models lost their aibox1 GPUs otherwise).
        # Only a GPU-less host is unambiguous: the sole peer box is the
        # only accelerator source -> local.
        import local_node as _ln
        master_obj = None
        try:
            master_obj = cfg.machines.require(host_name)
        except Exception:
            pass
        host_gpus = [d.strip() for d in ((master_obj or {}).get("gpu", "")
                                         or "").split(",") if d.strip()]
        return "local" if (len(peers) == 1
                           and (not host_gpus or host_name not in (
                               deploy_model.get("servers") or []))) else "rpc"
    machines_seen = set()
    for tok in order.split(","):
        tok = tok.strip()
        if not tok:
            continue
        mach, resource = _rm._parse_chip(tok)
        if mach and mach != host_name:
            machines_seen.add(mach)
        elif mach in ("", host_name):
            # host chip (bare CUDA token or host-qualified) → RPC path
            return "rpc"
        if resource == "RAM":
            # host RAM is implicit; a peer RAM chip means CPU tensors —
            # llama-server handles host RAM natively but a REMOTE box's
            # RAM is reachable only via RPC, so keep rpc for spreads
            pass
    if len(machines_seen) == 1 and not any(
            tok.strip() and (_rm._parse_chip(tok.strip())[0] in ("", host_name))
            for tok in order.split(",")):
        # Same Windows rule for the explicit-chips path: a single-box
        # verdict targeting a Windows box still means RPC.
        sole = next(iter(machines_seen))
        if _is_windows_box(sole):
            return "rpc"
        return "local"
    return "rpc"


def resolve_deployment(model_id, variant=None):
    """Merged deployment dict for one model deployment.

    Accepts a base model_id with an optional variant label, or a combined
    "base--label" id (the shape Launch-page cards and llama-swap keys use).
    Variant merge mirrors launch_plan()'s inline merge (delegated to
    rpc_manager._resolve_variant so the two can never drift). Returns
    (merged, base_id, label_or_None). Raises ValueError on unknown model
    or variant — callers map it to 4xx, never a silent fallback."""
    base_id, label = (model_id or "").strip(), (variant or "").strip() or None
    if label is None and "--" in base_id:
        cand_base, cand_label = base_id.split("--", 1)
        try:
            if cfg.get_model(cand_base) is not None:
                base_id, label = cand_base, cand_label.strip() or None
        except Exception:
            pass
    model = cfg.get_model(base_id)
    if model is None:
        raise ValueError(f"model '{model_id}' not found")
    if label:
        model = rpc_manager._resolve_variant(model, label)
    return model, base_id, label


def deployment_swap_key(model_id, variant=None):
    """The llama-swap model key for one deployment: base model_id, or
    base + '--' + sanitized variant label (same sanitize rule launch_plan
    uses for variant script/entry names). Matches the generated config's
    `models:` keys barring duplicate-label dedup collisions."""
    _merged, base_id, label = resolve_deployment(model_id, variant)
    if not label:
        return base_id
    safe = "".join(c if c.isalnum() or c in "-_" else "-"
                   for c in label).strip("-") or "variant"
    return f"{base_id}--{safe}"


def deployment_launch_mode(model_id, variant=None, host_name=None):
    """Heuristic launch-mode verdict for one deployment (goal: the
    generator decides rpc vs local from hardware, not the operator).

    Returns {"mode": "local"|"rpc", "box": <single-box machine or None>,
    "peers": [...], "swap_key": ...}. "local" + box = the whole
    deployment runs on that one box (fronted via the local-proxy
    entries); "rpc" = classic multi-box RPC chain on the swap host.
    host defaults to the swap host (same reference launch_plan uses)."""
    settings = load_settings()
    host = get_host(host_name or settings["host_name"])
    host_name = host["name"]
    merged, _base, _label = resolve_deployment(model_id, variant)
    servers = merged.get("servers") or []
    peers = [s for s in servers if s != host_name]
    mode = _resolve_launch_mode(merged, host_name, peers)
    box = None
    if mode == "local" and peers:
        # single remote box owns every chip: it is the only candidate.
        # (peers is normally exactly that box; first wins if several
        # slipped through — mirrors the generator's peer_objs[0].)
        box = peers[0]
    return {"mode": mode, "box": box, "peers": peers,
            "swap_key": deployment_swap_key(model_id, variant)}


def duplicate_swap_claims(groups=None):
    """Deployments claimed by more than one swap:true group of the same
    engine — the generator enforces one-group-per-model (first group
    wins), so the losers silently lose their mutual-exclusion. Returns
    [message] (empty = clean). Same-engine only: a llama.cpp group and a
    vLLM group never share a daemon config, so cross-engine overlap is
    harmless."""
    groups = groups if groups is not None else cfg.swap_groups.all()
    seen = {}   # (engine, member) -> group name
    dupes = []
    for sg in groups or []:
        if not sg.get("swap"):
            continue
        engine = sg.get("engine") or "llama_cpp"
        for mem in (sg.get("members") or []):
            key = (engine, mem)
            if key in seen and seen[key] != sg.get("name"):
                dupes.append(
                    f"'{mem}' is in swap groups '{seen[key]}' and "
                    f"'{sg.get('name')}' — only the first takes effect "
                    f"(one group per model); remove it from one")
            else:
                seen.setdefault(key, sg.get("name"))
    return dupes


def _is_windows_box(machine_name):
    """True when a machine is Windows (best-effort: unknown machines read
    as non-Windows so a stale config never flips a deployment to RPC by
    accident — the generator's own unknown-machine guard reports it)."""
    try:
        return (cfg.machines.require(machine_name) or {}).get(
            "os", "linux") == "windows"
    except Exception:
        return False


# ---------------- runtime: custom (user custom configs) ------------------
# A model with `runtime: custom` + a `custom:` section serves NO gguf from
# the farm — it is the USER's own serving config, passed through to
# llama-swap as an entry. Two kinds:
#   winbox-bat: the farm starts a .bat (e.g. D:\\Strata\\run-iq3_s.bat)
#     on a Windows box over SSH (kill-first, wait, stay-alive) and
#     proxies the OpenAI API to it; cmdStop kills the remote processes so
#     a swap restart/evict while loaded tears the server down with it.
#   raw: the user supplies the swap entry itself (cmd/proxy/cmdStop/
#     checkEndpoint/env + an optional script pushed as start-<mid>.sh).
#     The farm only namespaces + validates it — never edits the semantics.
# ISOLATION from machine-made configs (the whole point): swap keys go
# through the same dedup registry as generated models (no silent
# collision); script filenames derive from the unique model_id (one file
# per model — a custom script can never overwrite another model's);
# unknown keys/types fail loudly at save AND plan time, so a bad custom
# config skips with a warning instead of breaking the launch. Custom
# entries are engine-agnostic and emitted in both engine plans.
_CUSTOM_KINDS = ("winbox-bat", "raw")
_CUSTOM_SPEC_KEYS = {"kind", "bat", "port", "stop_match", "stop_names",
                     "proxy", "checkEndpoint", "cmd", "cmdStop", "env",
                     "script", "vision"}
_CUSTOM_KIND_KEYS = {
    "winbox-bat": {"kind", "bat", "port", "stop_match", "stop_names",
                   "vision"},
    "raw": {"kind", "proxy", "checkEndpoint", "cmd", "cmdStop", "env",
            "script", "vision"},
}
# placeholders a raw cmd/cmdStop may reference: the pushed script path
# and the vault sidecar (swap-secrets.env) path, resolved per plan.
_RAW_SCRIPT_TOKEN = "{SCRIPT}"
_RAW_SECRETS_TOKEN = "{SECRETS}"
_RAW_SCRIPT_LIMIT = 200_000

# stop_match/stop_names substrings land inside a PowerShell single-quoted
# -like pattern over SSH: keep them to a safe alphabet so a config typo
# can never break out of the quoting (injection tests cover this shape).
_CUSTOM_MATCH_RE = r"[A-Za-z0-9_.: \\-]+"


def validate_custom_spec(spec, where="custom"):
    """Shared shape validator for runtime: custom model configs.

    Raises ValueError on any bad spec — called at save time (API maps it
    to 400) and at plan time (caller maps it to skipped+warning), so a
    bad custom config can never reach disk silently OR break a launch.
    Server/box checks stay plan-time (they need live machine configs)."""
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: custom: section must be a mapping")
    kind = spec.get("kind")
    if kind not in _CUSTOM_KINDS:
        raise ValueError(
            f"{where}: unknown custom kind '{kind}' — must be one of "
            f"{list(_CUSTOM_KINDS)}")
    # vision marks image-capable external servers (listed for vision
    # tests); pure bool so a typo like "yes" fails loudly
    if "vision" in spec and not isinstance(spec["vision"], bool):
        raise ValueError(f"{where}: custom.vision must be true/false")
    unknown = sorted(set(spec) - _CUSTOM_KIND_KEYS[kind])
    if unknown:
        raise ValueError(
            f"{where}: kind '{kind}' rejects key(s) {unknown} — allowed: "
            f"{sorted(_CUSTOM_KIND_KEYS[kind])}")
    if kind == "winbox-bat":
        _validate_winbox_bat(spec, where)
    else:
        _validate_raw(spec, where)


def _validate_winbox_bat(spec, where):
    bat = (spec.get("bat") or "").strip()
    if not bat:
        raise ValueError(
            rf"{where}: custom.bat is required (e.g. 'D:\Strata\run-iq3_s.bat')")
    try:
        port = int(spec.get("port") or 8080)
    except (TypeError, ValueError):
        raise ValueError(
            f"{where}: custom.port '{spec.get('port')}' is not a number")
    if not 1 <= port <= 65535:
        raise ValueError(f"{where}: custom.port {port} out of range 1-65535")
    # alphabet gate for the SSH-embedded match lists (same shape as the
    # plan-time predicate builder — validated here so saves fail fast)
    _custom_stop_predicates(spec, bat)


def _validate_raw(spec, where):
    """Shape checks for kind raw (user-supplied swap entry). Types are
    strict; VALUES are the user's business — the farm passes them through
    (after {SCRIPT}/{SECRETS} substitution) without judging semantics."""
    import re as _re
    cmd = spec.get("cmd")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError(f"{where}: kind raw needs a non-empty custom.cmd")
    for k in ("cmdStop", "proxy", "checkEndpoint", "script"):
        v = spec.get(k)
        if v is not None and not isinstance(v, str):
            raise ValueError(f"{where}: custom.{k} must be a string")
    proxy = (spec.get("proxy") or "").strip()
    if proxy and not _re.fullmatch(r"https?://[^/\s]+(:\d+)?(/.*)?", proxy):
        raise ValueError(
            f"{where}: custom.proxy '{proxy}' must be an http(s) URL")
    ep = (spec.get("checkEndpoint") or "").strip()
    if ep and not ep.startswith("/"):
        raise ValueError(
            f"{where}: custom.checkEndpoint '{ep}' must start with '/'")
    if ep and not proxy:
        raise ValueError(
            f"{where}: custom.checkEndpoint needs custom.proxy (llama-swap "
            "only probes proxied upstreams)")
    env = spec.get("env")
    if env is not None:
        if not isinstance(env, dict):
            raise ValueError(f"{where}: custom.env must be a mapping")
        for k, v in env.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(
                    f"{where}: custom.env must map strings to strings")
            if not k.strip() or "=" in k or "\n" in k or "\n" in v:
                raise ValueError(
                    f"{where}: custom.env key '{k}' is not a clean KEY=VALUE")
    script = spec.get("script")
    if script is not None:
        if not script.strip():
            raise ValueError(
                f"{where}: custom.script is empty — omit it instead")
        if len(script) > _RAW_SCRIPT_LIMIT:
            raise ValueError(
                f"{where}: custom.script exceeds {_RAW_SCRIPT_LIMIT} chars")
    if _RAW_SCRIPT_TOKEN in (cmd + (spec.get("cmdStop") or "")) \
            and not script:
        raise ValueError(
            f"{where}: cmd references {_RAW_SCRIPT_TOKEN} but no "
            "custom.script is provided")


def _custom_stop_predicates(spec, bat):
    """PowerShell Where-Object kill predicates for a winbox-bat stack.

    Defaults derive from the .bat filename: the wrapper cmd.exe carries
    '<stem>.bat', and '<core>' (stem minus a leading run-/start-) matches
    the server + engine command lines (observed live on winbox: cmd.exe
    wrapper, python serve\\server.py --config strata-iq3_s.json,
    strata*.exe). stop_match/stop_names overrides append verbatim (same
    alphabet gate). Returns (condition_string, probe_substring)."""
    import re as _re
    stem = _re.sub(r".*[\\/]", "", bat)
    stem = _re.sub(r"\.[^.]*$", "", stem) or "custom"
    core = _re.sub(r"^(run|start)-", "", stem, flags=_re.IGNORECASE) or stem
    match_subs = list(spec.get("stop_match") or [core])
    name_pats = list(spec.get("stop_names") or ["strata*"])
    # name patterns may carry -like wildcards (* ?); match substrings may
    # not (they are wrapped in *...* here).
    for s in match_subs:
        if not isinstance(s, str) or not _re.fullmatch(
                r"[A-Za-z0-9_.: \\-]+", s):
            raise ValueError(
                f"custom.stop_match '{s}' must match {_CUSTOM_MATCH_RE}")
    for p in name_pats:
        if not isinstance(p, str) or not _re.fullmatch(
                r"[A-Za-z0-9_.: \\*?+\-]+", p):
            raise ValueError(f"custom.stop_names '{p}' has unsafe characters")
    conds = [f"(' ' + $_.CommandLine + ' ') -like '*{s}*'"
             for s in match_subs]
    conds += [f"$_.Name -like '{p}'" for p in name_pats]
    return " -or ".join(conds), match_subs[0]


def _render_winbox_bat_script(mid, peer, bat, port, probe, conds):
    """Launcher script for runtime: custom / kind winbox-bat.

    Written to the swap scripts_dir and executed by llama-swap's cmd on
    model load. Over SSH it: (1) KILLS any existing run of this stack
    first (stale server, half-loaded process, orphan from a dead swap
    daemon — same predicates as cmdStop) and waits for the port to close
    PLUS for the GPU VRAM to drain (port-closed does not mean VRAM-freed:
    the driver can hold the card for up to ~60s after the kill, and a
    fresh Strata engine started too soon OOMs on the MTP draft head with
    only ~800 MiB free — observed live on winbox), so every load starts
    clean; (2) uploads a .ps1 launcher (the proven -File shape from
    render_remote_local_script: cmd.exe-safe double-quoted path) that
    starts the .bat DETACHED via Win32_Process.Create with logging to
    %USERPROFILE%; (3) waits for the port (~2h budget, thin-link rule)
    with fast-fail when the server process dies pre-listen (port is the
    primary readiness signal — the cmd.exe wrapper PID stays alive even
    through a `pause` after a crash, so PID-alive alone means nothing);
    (4) stays alive as llama-swap's child until the port closes
    (cmdStop's kill closes it). Requires $SSH_PASS from the sidecar.
    ASCII ONLY in the ps1: Win PowerShell 5.1 -File reads BOM-less
    scripts as Windows-1252.
    """
    user = peer.get("username") or "root"
    ip = (peer.get("ip") or "").strip()
    box = peer["name"]
    import re as _re
    safe = _re.sub(r"[^A-Za-z0-9._-]", "-", mid).strip("-") or "custom"
    ps1_name = f"start-custom-{safe}.ps1"
    log_name = f"custom-{safe}-{port}.log"
    # PowerShell single-quoted string literal: only ' needs doubling
    bat_ps = "'" + bat.replace("'", "''") + "'"
    ps1 = f"""$ErrorActionPreference = 'Stop'
$port = {port}
# guard: the bash wrapper kills first, but if the port is somehow serving
# here (kill raced a fresh load), attach to it instead of stacking a
# second server on the same bind.
try {{ $c = New-Object Net.Sockets.TcpClient('127.0.0.1', $port); $c.Close(); Write-Output 'already serving'; exit 0 }} catch {{ }}
$log = Join-Path $env:USERPROFILE '{log_name}'
$cmd = 'cmd.exe /c "call ' + {bat_ps} + ' > "' + $log + '" 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = $cmd }}
if ($r.ReturnValue -ne 0) {{ Write-Error "Win32_Process.Create failed: $($r.ReturnValue)"; exit 1 }}
$wrapperPid = $r.ProcessId
Write-Output "started $wrapperPid"
for ($i = 0; $i -lt 2400; $i++) {{
  try {{ $c = New-Object Net.Sockets.TcpClient('127.0.0.1', $port); $c.Close(); Write-Output 'server up'; exit 0 }} catch {{ Start-Sleep -Seconds 2 }}
  if (($i % 20) -eq 0) {{
    $alive = @(Get-CimInstance Win32_Process | Where-Object {{ (' ' + $_.CommandLine + ' ') -like '*{probe}*' }})
    if ($alive.Count -eq 0) {{ Write-Error "server process died before opening port {port} - check ~/{log_name}"; exit 1 }}
    $wp = Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue
    if ($null -eq $wp) {{ Write-Error "launcher wrapper died before opening port {port} - check ~/{log_name}"; exit 1 }}
  }}
}}
Write-Error "port {port} did not come up in ~2h - check ~/{log_name}"
exit 1
"""
    # Embedded inside a bash double-quoted ssh argument, so the inner
    # PowerShell double quotes are backslash-escaped (\\" idiom — the
    # proven shape from render_remote_local_script).
    # NOTE: these run inside the launcher's bash double quotes under
    # `set -u` — every PowerShell $ MUST be backslash-escaped (\\$ idiom,
    # same as render_remote_local_script) or bash eats it as an (unset)
    # variable. cmdStop keeps BARE $ (llama-swap execs it directly, no
    # local shell), so the shared conds are escaped only here.
    probe_ps = (
        'powershell -NoProfile -Command \\"try { \\$c = New-Object '
        "Net.Sockets.TcpClient('127.0.0.1', " + str(port) + "); "
        '\\$c.Close(); exit 0 } catch { exit 1 }\\"')
    alive_ps = (
        'powershell -NoProfile -Command \\"Get-CimInstance Win32_Process | '
        "Where-Object { (' ' + \\$_.CommandLine + ' ') -like '*" + probe
        + "*' } | Select-Object -First 1\\\"")
    # Same kill as cmdStop (braces literal — plain string, not f-string).
    kill_ps = ('powershell -NoProfile -Command \\"Get-CimInstance '
               'Win32_Process | Where-Object { '
               + conds.replace("$", "\\$")
               + ' } | ForEach-Object { Stop-Process -Id '
               '\\$_.ProcessId -Force }\\"')
    return f"""#!/bin/bash
# runtime: custom (winbox-bat) launcher — server lives on {box}
set -uo pipefail
: "${{SSH_PASS:?SSH_PASS is not set — sourced from swap-secrets.env}}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
# FRESH START: kill any existing run of this stack first — a stale server,
# a half-loaded process, or an orphan from a dead swap daemon. Every load
# starts clean; the wait-for-up loop below attaches to the new server.
echo ">>> clearing any existing custom server on {box}:{port}"
sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
  "{kill_ps}" 2>/dev/null || true
sleep 3
# wait for the port to actually close (max ~90s) before starting the new
# run — else the fresh server races the dying one for the bind.
for _i in $(seq 1 45); do
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "{probe_ps}" 2>/dev/null || break
  sleep 2
done
# VRAM settle: port-closed does NOT mean VRAM-freed. The driver can hold
# the card for up to ~60s after the kill, and a fresh Strata engine
# started too soon OOMs on the MTP draft head (observed: 213 MiB needed,
# only ~800 MiB free right after a kill). Poll for strata compute
# processes to drain, then sleep once more for the release to settle.
# A dead .bat stuck on `pause` (no console under Win32_Process.Create)
# still matches the kill predicates above, so it is already gone here.
for _i in $(seq 1 30); do
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "nvidia-smi --query-compute-apps=process_name --format=csv,noheader" \\
    2>/dev/null | grep -qi strata || break
  sleep 2
done
sleep 15
echo ">>> starting custom server on {box}:{port} ({bat})"
tmp="$(mktemp /tmp/start-custom-XXXXXX.ps1)"
cat > "$tmp" <<'PS1'
{ps1}PS1
sshpass -p "$SSH_PASS" scp $SSH_OPTS -q "$tmp" "{user}@{ip}:{ps1_name}"
rc=$?
rm -f "$tmp"
[ $rc -ne 0 ] && {{ echo ">>> ps1 upload failed" >&2; exit 1; }}
# -File path in DOUBLE quotes (\\" idiom): remote cmd.exe expands
# %USERPROFILE% and hands child powershell.exe a double-quoted path —
# the proven shape from the RPC launcher. Single quotes arrive doubled
# (''C:\\...'') and -File rejects them.
sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
  "powershell -NoProfile -ExecutionPolicy Bypass -File \\"%USERPROFILE%\\{ps1_name}\\""
# Load/readiness budget: ~2h (2400 polls x ~3s). Only a dead server process
# (fast-fail below) or an explicit evict ends the wait.
for _i in $(seq 1 2400); do
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "{probe_ps}" 2>/dev/null && {{
    echo ">>> {box}:{port} is up"
    while sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
        "{probe_ps}" 2>/dev/null; do
      sleep 5
    done
    echo ">>> {box}:{port} closed — exiting"
    exit 0
  }}
  # fast-fail: server process died without ever opening the port — say so
  # now instead of idling the full ~2h with no context.
  if [ $((_i % 20)) -eq 0 ]; then
    sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
      "{alive_ps}" 2>/dev/null | grep -q . || {{
      echo "!!! {box}:{port} server process died before opening the port — check ~/{log_name} there" >&2
      exit 1
    }}
  fi
  sleep 2
done
echo "!!! {box}:{port} did not come up in ~2h — check ~/{log_name} there" >&2
exit 1
"""


def launch_plan(config_path=None, host_name=None):
    """Build the llama-swap launch plan from all model configs.

    DEC-06 (AUDIT-T6 §1.3): this is the planner the adapters expose as
    `launch_plan` — app.py reaches the plan ONLY through
    engine_registry.get_engine().launch_plan(); the function keeps its
    engine-aware entry-emit branch (ARD §3.5).

    The yaml key is the model's display name (set on the Models page),
    falling back to model_id. Master is always the host (which must be
    the master node); peers are the model's other servers.
    Each generate() run starts a fresh local-port registry: ports handed
    out during one plan must not constrain a different later plan.

    Linux host + peers: a per-model start-<id>.sh owns the RPC peers
    (spin up on load via sshpass, tear down on EXIT -- the proven
    start-rpc-qwen.sh shape) and cmd calls it, exactly like the live
    config. Single-node models and Windows hosts embed the direct
    llama-server command with the ${PORT} placeholder.
    rpc_manager._LOCAL_PORT_REGISTRY.clear()

    Returns a plan dict: host info, yaml text, per-model entries, the
    script files to push, and all resolved paths. launch() consumes
    this; it never re-derives anything.

    engine=vllm (ARD §3.5): the single branch below swaps the RPC
    wrapper-script emit for thin per-model vLLM scripts built from
    vllm_manager.swap_entry_parts (cmd/cmdStop/health); everything
    downstream (backup, write, push, launch, health) is engine-agnostic.
    """
    settings = load_settings()
    # Engine toggle (ARD §2): one read per generate() call. A bad engine
    # value raises here (startup error, never silent fallback); a missing
    # engine_registry module (partial checkout) keeps llama_cpp behavior.
    try:
        from engine_registry import load_settings as _engine_load
        engine_settings = _engine_load()
    except ImportError:
        engine_settings = {}
    engine = engine_settings.get("engine") or "llama_cpp"
    vllm_cfg = engine_settings.get("vllm") or {}
    host = get_host(host_name or settings["host_name"])
    host_name = host["name"]
    host_os = host.get("os", "linux")
    # DEC-03 (AUDIT-T6 §1.6): the daemon's config FILE is engine-qualified.
    # Resolution order (engine_swap_config_path): explicit config_path
    # param -> global per-engine path (swap.yaml config_path /
    # config_path_vllm) -> default; per-machine override
    # (machines.yaml swap_config_path / swap_config_path_vllm) wins over
    # the global; then the FILENAME is engine-qualified (vllm: basename
    # becomes config.vllm.yaml in the same dir; llama.cpp verbatim).
    eff_path, fell_back = engine_swap_config_path(
        engine, config_path, host_name)
    swap_dir = _swap_dir(host, eff_path)
    # DEC-13 (T2-G8): scripts live in a per-engine subdir so one engine's
    # start-*.sh can never overwrite the other's (the config FILE is
    # already engine-qualified from DEC-03; this closes the gap for the
    # script/sidecar side of the push).
    scripts_dir = _scripts_subdir(swap_dir, engine)
    abs_config = _host_config_path(host, eff_path)
    models = {}
    scripts = {}      # fname -> content (host-side, scripts_dir)
    script_mids = set()
    entries = []      # per-model: (key, model_id, mode)
    taken = set()
    warnings = []
    skipped = []
    # deployment label → SERVE GROUP name (e.g. "mid--DAD" → "aibox3-only").
    # Built BEFORE _process_one runs: the remote-local launcher writes its
    # actual-port state under the GROUP name when grouped (group-scoped
    # kills = the swap kick-out semantic) and cmdStop reads the same key
    # — resolution happens here at push time, nothing hard-coded in the
    # generated scripts. Same engine gate as the group block below (DEC-17).
    _label_to_group = {}
    _label_to_exclusive = set()
    for sg in cfg.swap_groups.all():
        if (sg.get("engine") or "llama_cpp") != engine:
            continue
        if sg.get("exclusive"):
            for mem in (sg.get("members") or []):
                _label_to_exclusive.add((mem or "").strip())
        for mem in (sg.get("members") or []):
            _label_to_group.setdefault((mem or "").strip(),
                                       (sg.get("name") or "").strip())
    def _deployment_key(base_mid, variant_label, taken_set):
        """Variant swap key: base + '--' + sanitized label, deduped via taken."""
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in (variant_label or "").strip())
        safe = safe.strip("-") or "variant"
        cand = f"{base_mid}--{safe}"
        # Use _swap_key dedup logic via taken
        tmp = {"model_id": cand}
        k, w = _swap_key(tmp, taken_set)
        return k, w

    def _process_one(deploy_model, mid_for_key, label_for_log, is_variant=False):
        # SERVE-GROUP key for the actual-port state file, by group kind:
        # - exclusive (take-turns) group: the GROUP name — cmdStop kills
        #   every port the group recorded, which IS the swap kick-out.
        # - co-serve group (swap:false, e.g. aibox2) or ungrouped: the
        #   deployment label itself — an explicit evict kills only its
        #   own server. Sharing the file here would let the LAST-launched
        #   member's port shadow the file and turn "evict Laya" into
        #   "kill nomic" (ports overwrite each other).
        # mid_for_key is the deployment label ("mid" / "mid--variant") —
        # resolved at push time, nothing hard-coded.
        _sg_key = _label_to_group.get(mid_for_key) \
            or (deploy_model.get("group") or "") or mid_for_key
        if mid_for_key not in _label_to_exclusive:
            # co-serve/ungrouped: scope the state file to THIS deployment,
            # not the group (see comment above)
            _sg_key = mid_for_key
        servers = deploy_model.get("servers") or []
        if not servers:
            skipped.append(mid_for_key)
            warnings.append(f"{mid_for_key}: no servers configured, skipped")
            return
        missing = []
        for s in servers:
            if s == host_name:
                continue
            try:
                cfg.machines.require(s)
            except ValueError:
                missing.append(s)
        if missing:
            skipped.append(mid_for_key)
            warnings.append(f"{mid_for_key}: unknown machine(s) {missing}, skipped")
            return
        if host_name not in servers:
            warnings.append(
                f"{mid_for_key}: host {host_name} not in servers, added as master")
        peers = [s for s in servers if s != host_name]
        # CUSTOM runtime (e.g. winbox-iq3_s): no GGUF served by the farm —
        # a remote box runs its own server; the farm only starts it over
        # SSH and proxies the OpenAI API to it. Handled before the
        # llama.cpp/vLLM branches (and the GGUF gate below) so a missing
        # base_path never reads as HF-format.
        if (deploy_model.get("runtime") or "") == "custom":
            try:
                _process_custom(deploy_model, mid_for_key, host_name,
                                peers, is_variant)
            except (ValueError, KeyError) as e:
                skipped.append(mid_for_key)
                warnings.append(f"{mid_for_key}: {e}, skipped")
            return
        # LAUNCH-MODE (llama.cpp): rpc | local | auto. AUTO is the default
        # and decides from HARDWARE, not user selection: if every selected
        # device chip belongs to ONE peer box (no host chips, no spread
        # across machines), the whole deployment runs there LOCALLY —
        # llama-server on that box, its own GPUs, zero RPC network hops —
        # and llama-swap fronts it via proxy so the farm keeps one API at
        # the host. Any mixed/multi-box order falls back to RPC chaining.
        # An explicit launch_mode: local / rpc on the model (or variant)
        # overrides the heuristic.
        local_mode = _resolve_launch_mode(deploy_model, host_name, peers)
        # Engine vocabulary mirror (DEC-11 in reverse): a model whose
        # checkpoint is HF-format (safetensors dir, format 'hf' or a
        # base_path not ending .gguf) has no llama.cpp presence —
        # llama-server cannot serve it, and the RPC wrapper the
        # llama.cpp branch would emit is garbage pointing at an HF dir.
        # vLLM mode already skips no-`vllm:`-section models at
        # swap_entry_parts; this is the mirror for llama.cpp mode.
        fmt = (deploy_model.get("format") or
               ("gguf" if str(deploy_model.get("base_path") or "")
                .lower().endswith(".gguf") else "hf"))
        if engine != "vllm" and fmt not in ("gguf", None, ""):
            skipped.append(mid_for_key)
            warnings.append(
                f"{mid_for_key}: HF-format checkpoint has no llama.cpp "
                "presence (no GGUF to serve) — skipped; serve it under "
                "the vLLM engine")
            return
        # key handling: primary uses original _swap_key on model, variants use deployment_key
        if is_variant:
            key, dup_warn = _deployment_key(mid_for_key.split("--")[0], deploy_model.get("_variant_label",""), taken)
        else:
            key, dup_warn = _swap_key(deploy_model, taken)
        if dup_warn:
            warnings.append(dup_warn)
        try:
            master, peer_objs = rpc_manager._split_servers(
                [host_name] + peers, host_name)
            # vLLM entries build their own cmd below; the llama-server
            # command is only needed on the llama.cpp path.
            if engine == "vllm":
                cmd = None
            else:
                cmd = rpc_manager.build_llama_command(
                    deploy_model, [host_name] + peers, master_name=host_name)
        except (ValueError, KeyError) as e:
            skipped.append(mid_for_key)
            warnings.append(f"{mid_for_key}: {e}, skipped")
            return
        # local mode with a REMOTE llama-server box: llama-swap's cmd IS
        # the launcher (SSHes over, starts/reuses llama-server on that
        # box, waits for the port) and the proxy fronts it. The launcher
        # only waits for TCP — model LOAD takes another 1-3 min, during
        # which llama-server answers /health with 503 "Loading model".
        # checkEndpoint=/health makes llama-swap poll until 200 (bounded
        # by healthCheckTimeout 7200) — the client's first request WAITS
        # for the load instead of surfacing a 502/503.
        if engine != "vllm" and local_mode == "local" and peer_objs:
            peer_m = peer_objs[0]
            # The remote llama-server's port: launch_mode_local_port when
            # pinned; else DETERMINISTIC AUTO-ASSIGN from the same hash
            # mechanism as RPC ports (md5(model_id\0peer) % range, base
            # 52000 — clear of the RPC range 51000-51499 and Windows
            # excluded ranges). Same inputs → same port, so the config
            # and a re-launched model reuse their own port, and N local
            # variants on one box never collide no matter how many there
            # are. The launcher rolls forward when busy, so the value is
            # the probe point, not gospel.
            raw_port = deploy_model.get("launch_mode_local_port")
            if raw_port:
                try:
                    remote_port = int(raw_port)
                except (TypeError, ValueError):
                    remote_port = rpc_manager.LOCAL_PORT_BASE
            else:
                # auto: deterministic + collision-free within this plan
                remote_port = rpc_manager.local_port_for(
                    mid_for_key, peer_m["name"])
            rpc_manager._LOCAL_PORT_REGISTRY.append(
                {"_local_port_of": remote_port})
            remote_ip = (peer_m.get("ip") or "").strip()
            remote_os = peer_m.get("os", "linux")
            if remote_os == "windows":
                # Windows llama-server box: render_remote_local_script
                # pushes a .ps1 launcher over SSH and drives it — same
                # start/stop semantics as the POSIX path.
                script_name = f"start-{mid_for_key}.sh"
                scripts[script_name] = rpc_manager.render_remote_local_script(
                    deploy_model, peer_m, remote_port, state_key=_sg_key)
                script_mids.add(mid_for_key)
                models[key] = {
                    "cmd": (
                        f"/bin/bash -c 'source \"{scripts_dir}/swap-secrets.env\" "
                        f"2>/dev/null; exec /bin/bash \"{scripts_dir}/{script_name}\"'"),
                    "proxy": f"http://{remote_ip}:{remote_port}",
                    # ACTUAL-PORT kill (rpc_manager.swap_state_killer_snippet):
                    # kills every --port recorded in the launcher's state
                    # file (config port when none) — a port that rolled
                    # past `remote_port` still dies on unload/swap. State
                    # key = the deployment's SERVE GROUP (group-scoped
                    # kick-out), or the model id when ungrouped — resolved
                    # at push time, never hard-coded.
                    "cmdStop": rpc_manager.swap_state_killer_snippet(
                        peer_m, _sg_key, remote_port,
                        os_type="windows"),
                    # /health: poll until the remote model finishes
                    # loading (llama-server 503s while loading)
                    "checkEndpoint": "/health",
                }
            else:
                # llama-swap's cmd IS the launcher: on model load it SSHes
                # to the box, reuses an already-serving port (config
                # reloads never restart a healthy server), else clears
                # stale processes and starts llama-server THERE with the
                # box's own GPUs (fit mode), waits for the port, exits 0 —
                # then llama-swap proxies every request to it.
                script_name = f"start-{mid_for_key}.sh"
                scripts[script_name] = rpc_manager.render_remote_local_script(
                    deploy_model, peer_m, remote_port, state_key=_sg_key)
                script_mids.add(mid_for_key)
                models[key] = {
                    "cmd": (
                        f"/bin/bash -c 'source \"{scripts_dir}/swap-secrets.env\" "
                        f"2>/dev/null; exec /bin/bash \"{scripts_dir}/{script_name}\"'"),
                    "proxy": f"http://{remote_ip}:{remote_port}",
                    # ACTUAL-PORT kill (see the windows branch above);
                    # state key = serve group (group-scoped kick-out)
                    "cmdStop": rpc_manager.swap_state_killer_snippet(
                        peer_m, _sg_key, remote_port),
                    # /health: poll until the remote model finishes
                    # loading (llama-server 503s while loading)
                    "checkEndpoint": "/health",
                }
            entries.append({"key": key, "model_id": mid_for_key,
                            "mode": "remote-proxy",
                            "pinned": bool(deploy_model.get("pinned")),
                            "ttl": max(0, int(deploy_model.get("ttl") or 0)),
                            "variant": deploy_model.get("_variant_label") or None,
                            "base_model": mid_for_key.split("--")[0],
                            "rpc_map": {}})
            return
        # MLX runtime never uses RPC peers: force the direct-command path
        if (deploy_model.get("runtime") or "llama") == "mlx" and peer_objs:
            warnings.append(
                f"{mid_for_key}: MLX runtime ignores RPC peers "
                f"(single machine only)")
            peer_objs = []
        if engine == "vllm":
            # ARD §3.5 — the ONE vLLM branch: entry-emit time only. The
            # llama-swap entry carries the vllm command (thin script on
            # linux; no RPC fan-out, no sshpass block, no device env
            # exports — pinning travels inside cmd via
            # CUDA_VISIBLE_DEVICES or docker -e). Multi-host: peer_objs
            # pass through to swap_entry_parts, which turns the cmd into
            # the Ray cluster script (multi-host TP/PP).
            import vllm_manager
            try:
                parts = vllm_manager.swap_entry_parts(
                    deploy_model, master, vllm_cfg, peers=peer_objs)
            except (ValueError, KeyError) as e:
                skipped.append(mid_for_key)
                warnings.append(f"{mid_for_key}: {e}, skipped")
                return
            # DEC-08: the launch path reads only the per-model `vllm:`
            # section; the t_90def951 flags bridge and its warnings
            # fan-out are retired (legacy configs fail loudly in
            # build_vllm_command instead).
            if host_os == "linux":
                script_name = f"start-{mid_for_key}.sh"
                # DEC-15: venv root comes from the generate()-time
                # vllm_cfg (engine_registry settings read) — the script
                # renderer no longer re-reads swap.yaml's unrelated
                # settings file for the vllm: block it doesn't contain
                # (that dead read fell back to the hardcoded default for
                # every non-default venv_bin).
                scripts[script_name] = _vllm_script(parts["cmd"], vllm_cfg)
                script_mids.add(mid_for_key)
                if peer_objs:
                    # multi-host (Ray cluster): worker nodes need sshpass
                    # via $SSH_PASS — same swap-secrets.env sidecar wrap
                    # the llama.cpp branch uses (single-node vLLM scripts
                    # need no SSH and skip the wrap, keeping the thin cmd).
                    models[key] = {
                        "cmd": f"/bin/bash -c 'source \"{scripts_dir}/swap-secrets.env\" 2>/dev/null; exec /bin/bash \"{scripts_dir}/{script_name}\" \"${{PORT}}\"'"}
                else:
                    models[key] = {
                        "cmd": f"/bin/bash {scripts_dir}/{script_name} ${{PORT}}"}
            else:
                vcmd = parts["cmd"].replace('"$PORT"', "${PORT}")
                if host_os == "windows":
                    models[key] = {"cmd": vcmd.replace(" \\\n  ", " ")}
                else:
                    models[key] = {"cmd": _LiteralStr(vcmd)}
            models[key]["cmdStop"] = parts["cmdStop"]
            models[key]["checkEndpoint"] = parts["checkEndpoint"]
            # llama-swap's per-model env is a LIST of "K=V" strings — a dict
            # here makes the whole config fail to load (`cannot unmarshal
            # !!map into []string`, daemon crash-loops). Convert; omit when
            # empty (vLLM entries carry env inside cmd by design).
            _entry_env = parts.get("env") or {}
            if _entry_env:
                models[key]["env"] = [
                    f"{k}={v}" for k, v in _entry_env.items()]
        elif host_os == "linux":
            script_name = f"start-{mid_for_key}.sh"
            if peer_objs:
                scripts[script_name] = rpc_manager._render_rpc(
                    deploy_model, master, peer_objs, cmd=cmd)
            else:
                scripts[script_name] = rpc_manager._render_single_node(
                    deploy_model, master)
            script_mids.add(mid_for_key)
            # vault-only secrets: multi-node scripts require $SSH_PASS at
            # runtime (see _render_rpc guard). llama-swap launches the cmd
            # directly with no env of ours, so source the pushed sidecar
            # (swap-secrets.env — written from the vault at push time,
            # 0600, never committed). Single-node scripts ignore it
            # harmlessly (source fails silently, they need no SSH).
            models[key] = {
                "cmd": f"/bin/bash -c 'source \"{scripts_dir}/swap-secrets.env\" 2>/dev/null; exec /bin/bash \"{scripts_dir}/{script_name}\" \"${{PORT}}\"'"}
        else:
            cmd = cmd.replace('"$PORT"', "${PORT}")
            if host_os == "windows":
                cmd = cmd.replace(" \\\n  ", " ")
                models[key] = {"cmd": cmd}
            else:
                models[key] = {"cmd": _LiteralStr(cmd)}
        # MLX models: mlx_lm.server has no /health endpoint, so point
        # llama-swap's probe at /v1/models instead
        if (deploy_model.get("runtime") or "llama") == "mlx":
            models[key]["checkEndpoint"] = "/v1/models"
        # DEC-13 (T2-G4): wire the model's ttl into the swap entry — the
        # key was previously read, zeroed, and dropped, so per-model
        # auto-unload config silently did nothing. Absent/None = 0
        # (llama-swap default: never unload), same as before; negative
        # values are clamped to 0 so a typo can't mean "unload instantly".
        try:
            ttl = max(0, int(deploy_model.get("ttl") or 0))
        except (TypeError, ValueError):
            ttl = 0
        pinned = bool(deploy_model.get("pinned"))
        models[key]["ttl"] = ttl
        # visualizer RPC map: exact RPC<N> -> (machine, resource) numbering
        rpc_map = rpc_manager.compute_rpc_map(deploy_model, master, peer_objs)
        entries.append({"key": key, "model_id": mid_for_key,
                        "mode": "script" if mid_for_key in script_mids else "direct",
                        "pinned": pinned, "ttl": ttl,
                        "variant": deploy_model.get("_variant_label") or None,
                        "base_model": mid_for_key.split("--")[0] if is_variant else mid_for_key,
                        "rpc_map": rpc_map})

    def _custom_key(deploy_model, mid_for_key, is_variant):
        """Swap key through the shared dedup registry — a custom model can
        never silently collide with a machine-made entry (same rule as
        every generated model)."""
        if is_variant:
            key, dup_warn = _deployment_key(
                mid_for_key.split("--")[0],
                deploy_model.get("_variant_label", ""), taken)
        else:
            key, dup_warn = _swap_key(deploy_model, taken)
        if dup_warn:
            warnings.append(dup_warn)
        return key

    def _custom_ttl_pinned(deploy_model):
        try:
            ttl = max(0, int(deploy_model.get("ttl") or 0))
        except (TypeError, ValueError):
            ttl = 0
        return ttl, bool(deploy_model.get("pinned"))

    def _custom_record(key, mid_for_key, deploy_model, mode, ttl, pinned,
                       is_variant):
        entries.append({"key": key, "model_id": mid_for_key,
                        "mode": mode,
                        "pinned": pinned, "ttl": ttl,
                        "variant": deploy_model.get("_variant_label") or None,
                        "base_model": mid_for_key.split("--")[0]
                        if is_variant else mid_for_key,
                        "rpc_map": {}})

    def _process_custom(deploy_model, mid_for_key, host_name, peers,
                        is_variant=False):
        """llama-swap entry for runtime: custom (user custom configs).

        Raises ValueError on any bad spec — the caller maps it to
        skipped+warning, never a silent fallback (and a bad custom model
        never affects any other model's entry).
        """
        spec = deploy_model.get("custom") or {}
        validate_custom_spec(spec, where=f"model '{mid_for_key}'")
        if (spec.get("kind") or "winbox-bat") == "raw":
            _process_custom_raw(deploy_model, mid_for_key, spec, is_variant)
            return
        bat = (spec.get("bat") or "").strip()
        port = int(spec.get("port") or 8080)
        if len(peers) != 1:
            raise ValueError(
                "custom winbox-bat is single-box only — "
                f"servers={deploy_model.get('servers')}")
        peer = cfg.machines.require(peers[0])
        if (peer.get("os", "linux")) != "windows":
            raise ValueError(
                f"custom winbox-bat needs a windows box, '{peers[0]}' "
                f"is {peer.get('os', 'linux')}")
        remote_ip = (peer.get("ip") or "").strip()
        if not remote_ip:
            raise ValueError(f"machine '{peers[0]}' has no ip configured")
        conds, probe = _custom_stop_predicates(spec, bat)
        key = _custom_key(deploy_model, mid_for_key, is_variant)
        script_name = f"start-{mid_for_key}.sh"
        scripts[script_name] = _render_winbox_bat_script(
            mid_for_key, peer, bat, port, probe, conds)
        script_mids.add(mid_for_key)
        user = peer.get("username") or "root"
        # Powershell braces are literal here (plain string, not f-string)
        # — the kill matches the live tree: wrapper cmd.exe carrying the
        # .bat name, server/engine processes via stop_match, engine exes
        # via stop_names. Port-anchored like the llama-server kills
        # elsewhere: only this stack's port is ever touched (the probe in
        # the launcher prevents cross-model kills by construction).
        ps_kill = (
            "Get-CimInstance Win32_Process | Where-Object { " + conds
            + " } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        models[key] = {
            "cmd": (
                f"/bin/bash -c 'source \"{scripts_dir}/swap-secrets.env\" "
                f"2>/dev/null; exec /bin/bash \"{scripts_dir}/{script_name}\"'"),
            "proxy": f"http://{remote_ip}:{port}",
            # /health: Strata answers 200 once the model is loaded (probed
            # live on winbox) — the client's first request waits for the
            # load instead of surfacing a 502/503.
            "checkEndpoint": "/health",
            "cmdStop": (
                'sshpass -p "$SSH_PASS" ssh -n '
                '-o StrictHostKeyChecking=no -o ConnectTimeout=5 '
                f"{user}@{remote_ip} "
                '"powershell -NoProfile -Command \\"' + ps_kill + '\\""'),
        }
        ttl, pinned = _custom_ttl_pinned(deploy_model)
        models[key]["ttl"] = ttl
        _custom_record(key, mid_for_key, deploy_model, "custom-proxy",
                       ttl, pinned, is_variant)

    def _process_custom_raw(deploy_model, mid_for_key, spec, is_variant=False):
        """Kind raw: the user's own swap entry, passed through verbatim
        (after {SCRIPT}/{SECRETS} substitution). The farm namespaces the
        key + script filename and validates shapes — it never edits the
        entry semantics, so machine-made configs are untouched by
        construction (additive dict entries only)."""
        key = _custom_key(deploy_model, mid_for_key, is_variant)
        script_name = f"start-{mid_for_key}.sh"
        script_body = spec.get("script")
        if script_body is not None:
            text = script_body if script_body.endswith("\n") \
                else script_body + "\n"
            scripts[script_name] = text
            script_mids.add(mid_for_key)

        def _sub(text):
            # placeholder paths resolve per plan (scripts_dir is the
            # per-engine push dir); unknown {TOKENS} pass through untouched
            # so user shell expansions like ${PORT} survive verbatim.
            out = (text or "")
            if script_body is not None:
                out = out.replace(
                    _RAW_SCRIPT_TOKEN, f"{scripts_dir}/{script_name}")
            out = out.replace(
                _RAW_SECRETS_TOKEN, f"{scripts_dir}/{SIDECAR_NAME}")
            return out

        entry = {"cmd": _sub(spec.get("cmd"))}
        if (spec.get("proxy") or "").strip():
            entry["proxy"] = (spec.get("proxy") or "").strip()
        if (spec.get("checkEndpoint") or "").strip():
            entry["checkEndpoint"] = (spec.get("checkEndpoint") or "").strip()
        if spec.get("cmdStop") is not None:
            entry["cmdStop"] = _sub(spec.get("cmdStop"))
        if spec.get("env") is not None:
            entry["env"] = [f"{k}={v}"
                            for k, v in (spec.get("env") or {}).items()]
        ttl, pinned = _custom_ttl_pinned(deploy_model)
        entry["ttl"] = ttl
        models[key] = entry
        _custom_record(key, mid_for_key, deploy_model, "custom-raw",
                       ttl, pinned, is_variant)

    for model in cfg.list_models():
        mid = model.get("model_id") or model.get("_file", "unknown")
        # swap_disabled gate: the operator hides a model from the swap
        # entirely (Launch page disable button). Skipped with warning so
        # the omission is visible in the preview; re-enabling drops the
        # flag and the next push serves it again.
        if model.get("swap_disabled"):
            warnings.append(
                f"{mid}: disabled for swap (re-enable on the Launch page), "
                "skipped")
            continue
        # primary deployment
        _process_one(model, mid, mid, is_variant=False)
        # variant deployments (if any) — each gets its own swap entry, inheriting base_path/name
        for var in (model.get("variants") or []):
            label = (var.get("label") or "").strip()
            if not label:
                continue
            # merge: primary base + variant overrides
            merged = dict(model)
            # remove variants to avoid recursion
            merged.pop("variants", None)
            # child overrides — only when variant explicitly provides non-None value
            for k in ("servers", "group", "device_order", "device_order_policy",
                      "tensor_split_mode", "tensor_split", "fit_margin", "fit_ram_allow",
                      "expose", "pinned", "ttl",
                      "launch_mode", "launch_mode_local_port"):
                if k in var and var[k] is not None:
                    merged[k] = var[k]
            # llama_bin: if variant specifies non-empty, use it, else keep primary
            if var.get("llama_bin") is not None and (var.get("llama_bin") or "").strip():
                merged["llama_bin"] = var["llama_bin"]
            # flags: merge variant flags over primary (partial override keeps primary defaults)
            if var.get("flags") is not None and isinstance(var["flags"], dict) and var["flags"]:
                base_flags = dict(merged.get("flags") or {})
                base_flags.update(var["flags"])
                merged["flags"] = base_flags
            # vllm (A2): same partial-override semantics as flags — shallow
            # dict merge over the primary's section, so a "weak-GPU variant"
            # (own servers/device_order + gmu/cpu_offload overrides) falls
            # out of existing machinery. Absent/empty inherits wholesale.
            if var.get("vllm") is not None and isinstance(var["vllm"], dict) and var["vllm"]:
                base_vllm = dict(merged.get("vllm") or {})
                base_vllm.update(var["vllm"])
                merged["vllm"] = base_vllm
            # ensure base fields remain
            merged["model_id"] = mid
            merged["base_path"] = model.get("base_path")
            merged["name"] = model.get("name") or mid
            merged["_variant_label"] = label
            var_mid = f"{mid}--{label}"
            # sanitize var_mid for script filename (filesystem safe)
            safe_mid = "".join(c if c.isalnum() or c in "-_." else "-" for c in var_mid)
            _process_one(merged, safe_mid, f"{mid} variant {label}", is_variant=True)
    doc = {
        "healthCheckTimeout": 7200,
        "env": _swap_env(host_name),
        "models": models,
    }
    # runtime secrets sidecar: pushed next to the scripts (0600) and sourced
    # by the wrapped model cmds above. Built from the in-memory
    # vault-resolved host password — the tracked swap.yaml never holds it,
    # and generated scripts never bake it. Only emitted when at least one
    # script actually requires $SSH_PASS (the _render_rpc guard). Lives in
    # the per-engine scripts_dir (DEC-13) so each engine has its own copy.
    # sidecar is also required when a remote-proxy entry's cmdStop SSHes
    # to the llama-server box (it needs $SSH_PASS at evict time) even
    # though that entry ships no script.
    # A raw custom entry (or its pushed script) may reference $SSH_PASS /
    # sshpass itself — same vault-fed sidecar, never baked secrets.
    def _needs_sidecar_text(*texts):
        return any(t and ("SSH_PASS" in t or "sshpass" in t)
                   for t in texts)

    if any(_needs_sidecar_text(c) for c in scripts.values()) or any(
            _needs_sidecar_text(e.get("cmd"), e.get("cmdStop"))
            for e in models.values()):
        import shlex as _shlex
        scripts[SIDECAR_NAME] = (
            "# llama-swap runtime secrets — generated at push time from the\n"
            "# encrypted vault (never committed, never baked into scripts).\n"
            "# Sourced by model launch commands before start-*.sh runs.\n"
            f"export SSH_PASS={_shlex.quote(host.get('password') or '')}\n"
        )
    # skipped with a warning; a deployment claimed twice stays in the
    # first group (llama-swap requires unique membership).
    try:
        label_to_key = {}
        for e in entries:
            for cand in {e["key"], e["model_id"]} | (
                    {f'{e["base_model"]}--{e["variant"]}'}
                    if e.get("variant") else set()):
                label_to_key.setdefault(cand, e["key"])
        claimed = {}
        swap_groups_out = {}
        for sg in cfg.swap_groups.all():
            # DEC-17: only the ACTIVE engine's serve groups participate —
            # the engine read at the top of generate() is the same value
            # /api/swap-groups filters on, so a llama.cpp RPC group
            # never lands in a vllm swap config (and vice versa). A
            # missing `engine` (legacy YAML) reads as llama_cpp.
            if (sg.get("engine") or "llama_cpp") != engine:
                continue
            keys = []
            for mem in (sg.get("members") or []):
                k = label_to_key.get(mem)
                if not k:
                    warnings.append(
                        f'serve group {sg.get("name")}: unknown deployment '
                        f"'{mem}', skipped")
                    continue
                if k in claimed:
                    warnings.append(
                        f"serve group {sg.get('name')}: '{mem}' already in "
                        f"'{claimed[k]}', skipped (one group per model)")
                    continue
                claimed[k] = sg.get("name")
                keys.append(k)
            if not keys:
                warnings.append(
                    f'serve group {sg.get("name")}: no resolvable members, '
                    f"skipped")
                continue
            swap_groups_out[sg["name"]] = {
                "swap": bool(sg.get("swap", False)),
                "exclusive": bool(sg.get("exclusive", False)),
                "persistent": bool(sg.get("persistent", False)),
                "members": keys,
            }
        if swap_groups_out:
            doc["groups"] = swap_groups_out
    except Exception as e:
        warnings.append(f"serve groups skipped: {e}")
    text = yaml.dump(doc, Dumper=_SwapDumper, default_flow_style=False,
                     sort_keys=False, allow_unicode=True)
    return {
        "host": host_name, "host_os": host_os,
        "abs_config": abs_config, "swap_dir": swap_dir,
        "scripts_dir": scripts_dir,
        "yaml": text, "models": sorted(models), "entries": entries,
        "scripts": sorted(scripts), "script_mids": sorted(script_mids),
        "skipped": skipped, "warnings": warnings, "_scripts": scripts,
    }


def _vllm_script(cmd, vllm_cfg=None):
    """Thin vLLM start script (ARD §3.5): shebang + PORT + exec of the
    vllm command — no RPC fan-out, no sshpass block, no device env
    exports (pinning travels inside cmd). cmd's $PORT / ${PORT}
    placeholders resolve from the script's PORT (arg 1, default 8080),
    mirroring rpc_manager._render_single_node.

    DEC-15: vllm_cfg is the generate()-time engine settings block
    (same object swap_entry_parts/build_vllm_command use) — never
    re-read from disk here.

    LD_LIBRARY_PATH: pip-installed torch needs the wheel's own
    nvjitlink lib to win over a system CUDA install (cusparse 12.9
    requires the 12_9 symbols only the wheel's copy exports); without
    it `vllm serve` dies at import (libcusparse undefined symbol) and
    llama-swap reports "upstream command exited prematurely". Sourced
    from the vllm_bin location so it stays correct for any venv root.
    """
    venv_bin = (vllm_cfg or {}).get("venv_bin") \
        or "~/vllm-env/bin/vllm"
    venv_root = os.path.dirname(os.path.dirname(os.path.expanduser(
        venv_bin)))
    nvjitlink = os.path.join(venv_root, "lib")
    nvjitlink_glob = os.path.join(
        nvjitlink, "python*/site-packages/nvidia/nvjitlink/lib")
    import glob as _glob
    matches = sorted(_glob.glob(nvjitlink_glob))
    ld_line = ""
    if matches:
        ld_line = ("export LD_LIBRARY_PATH=\"%s"
                   "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n" % matches[-1])
    # Multi-host cluster cmds are already complete bash programs
    # (shebang + Ray head/worker setup + exec vllm serve): wrapping them
    # in the thin `exec ...` header corrupts them (`exec #!/bin/bash`,
    # exit 127 path). Pass through, injecting only the nvjitlink
    # LD_LIBRARY_PATH line after the shebang/PORT header so the final
    # `vllm serve` still imports torch. Single-node plain cmds keep the
    # thin-wrap path below — same entry point, per-engine script body,
    # which is what keeps the llama-RPC <-> vLLM switch seamless.
    if cmd.lstrip().startswith("#!"):
        lines = cmd.splitlines(keepends=True)
        insert_at = 1
        if len(lines) > 1 and lines[1].startswith("PORT="):
            insert_at = 2
        if ld_line:
            lines.insert(insert_at, ld_line)
        return "".join(lines)
    # Hoist any env-prefix assignments (e.g. CUDA_VISIBLE_DEVICES=0,12)
    # out of the exec line: `exec VAR=x cmd` is not valid bash ("exec:
    # VAR=x: not found", exit 127). The vllm cmd carries the device
    # prefix inside build_vllm_command; it must become a separate
    # export statement so `exec` receives a real command name.
    exec_cmd = cmd
    exports = []
    while True:
        head, sep, rest = exec_cmd.partition(" ")
        if sep and "=" in head and not head.startswith(("/", "~", '"', "'")) \
                and all(c.isalnum() or c == "_" for c in
                        head.split("=", 1)[0]):
            exports.append(f"export {head}\n")
            exec_cmd = rest
        else:
            break
    body = "".join(exports) + f"exec {exec_cmd}"
    return f"""#!/bin/bash
PORT="${{1:-8080}}"
{ld_line}{body}
"""


def _swap_env(host_name):
    """llama-swap daemon env, per GPU vendor of the host machine.
    CUDA_DEVICE_ORDER is NVIDIA-only; on AMD hosts only P2P is set.
    GGML_SCHED_DEBUG=3 is always added (ground truth for visualizer)."""
    import gpu_vendor
    host = None
    try:
        host = cfg.machines.require(host_name)
    except Exception:
        pass
    v = gpu_vendor.normalize_vendor((host or {}).get("gpu_vendor"))
    base = {"GGML_SCHED_DEBUG": "3"}
    if v == "amd":
        return {**base, "GGML_CUDA_P2P": "1"}
    if v == "intel":
        return {**base, "SYCL_CACHE_PERSISTENT": "1"}
    return {**base, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "GGML_CUDA_P2P": "1"}


def _swap_dir(host, config_path):
    """Directory the config (and generated scripts) live in, host-side."""
    abs_path = _host_config_path(host, config_path)
    if host.get("os", "linux") == "windows":
        return abs_path.rsplit("\\", 1)[0] if "\\" in abs_path \
            else os.path.dirname(abs_path)
    import posixpath
    return posixpath.dirname(abs_path)


# ---------------- path helpers ----------------
def _remote_home(ssh, os_type="linux"):
    if os_type == "windows":
        # cmd.exe has no $HOME; PowerShell resolves the profile dir.
        code, out, _ = ssh.run(
            'powershell -NoProfile -Command "echo $env:USERPROFILE"',
            timeout=10)
        if code == 0 and out.strip():
            return out.strip().splitlines()[-1]
        raise ValueError("could not resolve %USERPROFILE% on remote Windows host")
    code, out, _ = ssh.run("echo $HOME", timeout=10)
    if code == 0 and out.strip():
        return out.strip().splitlines()[-1]
    raise ValueError("could not resolve $HOME on remote host")


def _expand_remote(path, home, os_type="linux"):
    if path.startswith("~/"):
        return home + path[1:]
    if os_type == "windows":
        # expand %USERPROFILE% (cmd) and $env:USERPROFILE (powershell) forms;
        # plain SSH lands in cmd.exe where these are NOT expanded for us.
        for var in ("%USERPROFILE%", "$env:USERPROFILE"):
            if var.lower() in path.lower():
                return path.lower().replace(var.lower(), home)
    return path


def _expand_local(path, os_type):
    if os_type == "windows":
        return os.path.expandvars(os.path.expanduser(path))
    return os.path.expanduser(path)


def _run(host, cmd, timeout=30):
    """Run a shell command on the host (local or SSH)."""
    is_local = local_node.is_local(host)
    is_win = host.get("os", "linux") == "windows"
    if is_local:
        return rpc_manager._run_local(cmd, is_win, timeout)
    return rpc_manager._ssh_for(host).run(cmd, timeout=timeout)


def _write_host(host, remote_path, text):
    """Write text to a file on the host (local or SFTP).

    Local writes are atomic (temp file + os.replace) so a stale
    root-owned file in a user-writable directory gets replaced cleanly
    instead of raising Errno 13. This is the fresh-install-safe path:
    no manual chown required.
    """
    is_local = local_node.is_local(host)
    if is_local:
        parent = os.path.dirname(remote_path) or "."
        os.makedirs(parent, exist_ok=True)
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp-swap-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, remote_path)
        except PermissionError as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            owner = "unknown"
            try:
                st = os.stat(remote_path)
                # pwd is POSIX-only; resolve the owner name best-effort
                if hasattr(os, "geteuid"):
                    import pwd
                    try:
                        owner = pwd.getpwuid(st.st_uid).pw_name
                    except KeyError:
                        owner = str(st.st_uid)
                else:
                    owner = str(st.st_uid)
            except OSError:
                st = None
            hint = (
                f"Permission denied writing '{remote_path}' "
                f"(owner: {owner}, you: {os.environ.get('USER', os.environ.get('USERNAME', '?'))}). "
                f"Fix: sudo chown $USER '{remote_path}' "
                f"or sudo chown -R $USER '{parent}'"
            )
            raise PermissionError(f"{e}. {hint}") from e
        return
    rpc_manager._ssh_for(host).push_text(
        text, remote_path, host.get("os", "linux"))


def _host_config_path(host, config_path):
    """Absolute config path on the host (expands ~ / %USERPROFILE%)."""
    is_local = local_node.is_local(host)
    if is_local:
        return _expand_local(config_path, host.get("os", "linux"))
    os_type = host.get("os", "linux")
    home = _remote_home(rpc_manager._ssh_for(host), os_type)
    return _expand_remote(config_path, home, os_type)


# ---------------- daemon control ----------------
def _port_open(host, port):
    if local_node.is_local(host):
        return rpc_manager._local_port_open(port)
    try:
        return rpc_manager._ssh_for(host).check_port(port)
    except Exception:
        return False


def _process_running(host):
    os_type = host.get("os", "linux")
    if os_type == "windows":
        code, out, _ = _run(
            host, 'powershell -NoProfile -Command "(Get-Process -Name '
            'llama-swap -ErrorAction SilentlyContinue).Count"', timeout=15)
        if code != 0:
            return False
        try:
            return int((out or "0").strip().splitlines()[-1]) > 0
        except (ValueError, IndexError):
            return False
    if os_type == "macos":
        # macOS has no pgrep; fall back to a port probe (llama-swap is
        # always a TCP listener, so this is sufficient)
        try:
            port = int(load_settings().get("listen_port") or 8080)
        except (ValueError, TypeError):
            port = 8080
        return bool(_port_open(host, port))
    code, out, _ = _run(
        host, 'pgrep -f "llama-[s]wap" >/dev/null 2>&1 && echo yes || echo no',
        timeout=15)
    return code == 0 and "yes" in out


def _start_windows_local(binary, config_path, port):
    import subprocess
    log = os.path.join(rpc_manager.GENERATED_DIR, "llama-swap.log")
    os.makedirs(rpc_manager.GENERATED_DIR, exist_ok=True)
    cmd = f'{binary} --config "{config_path}" --listen :{port} {SWAP_DAEMON_ARGS}'
    proc = subprocess.Popen(
        cmd, shell=True, stdout=open(log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS,
    )
    return proc.pid


def _start_windows_remote(host, binary, config_path, port):
    """Start llama-swap on a remote Windows host over SSH.

    Mirrors the RPC launcher: Win32_Process.Create detaches the process
    so it survives the SSH session. Paths with %USERPROFILE% / ~ are
    expanded target-side by PowerShell.
    """
    ssh = rpc_manager._ssh_for(host)
    # Expand config path target-side; keep quoting safe for PowerShell.
    ps = (
        "$ErrorActionPreference = 'Stop'; "
        f"$cfg = \"{config_path}\"; "
        "$cfg = $cfg -replace '%USERPROFILE%', $env:USERPROFILE "
        "-replace '\\$env:USERPROFILE', $env:USERPROFILE "
        "-replace '^~', $env:USERPROFILE; "
        f"$bin = \"{binary}\"; "
        "$bin = $bin -replace '%USERPROFILE%', $env:USERPROFILE "
        "-replace '\\$env:USERPROFILE', $env:USERPROFILE "
        "-replace '^~', $env:USERPROFILE; "
        "$log = Join-Path $env:USERPROFILE 'llama-swap.log'; "
        "$cmd = ('cmd.exe /c \"\"{0}\" --config \"{1}\" --listen :{2} " + SWAP_DAEMON_ARGS + " > \"{3}\" 2>&1\"' "
        "-f $bin, $cfg, "
        f"{int(port)}"
        ", $log); "
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        "-Arguments @{ CommandLine = $cmd }; "
        "if ($r.ReturnValue -ne 0) { Write-Error \"Win32_Process.Create failed: $($r.ReturnValue)\"; exit 1 }; "
        "Write-Output \"started pid $($r.ProcessId)\""
    )
    code, out, err = ssh.run(
        f'powershell -NoProfile -Command "{ps}"', timeout=30)
    if code != 0:
        return {"error": (err or out or f"exit {code}").strip()[-2000:]}
    pid = None
    for line in (out or "").splitlines():
        if "started pid" in line:
            try:
                pid = int("".join(c for c in line if c.isdigit()) or "0") or None
            except ValueError:
                pid = None
    return {"method": "win32-remote", "pid": pid}


def _resolve_binary(host, binary):
    """systemd ExecStart needs an absolute path; bare 'llama-swap' is
    rejected as a bad unit setting. Resolve via PATH on the target host,
    falling back to the literal value (setsid fallback still works)."""
    import shutil
    if local_node.is_local(host):
        expanded = os.path.expanduser(os.path.expandvars(binary))
        found = shutil.which(expanded)
        if found:
            return found
        if os.path.isabs(expanded):
            return expanded
        found = shutil.which(binary)
        return found or expanded
    # remote: ask the host
    code, out, _ = _run(host, f'command -v "{binary}" 2>/dev/null || echo ""',
                        timeout=10)
    if code == 0 and out.strip().startswith("/"):
        return out.strip().splitlines()[0].strip()
    return binary


def _systemd_unit(binary, config_path, port):
    return (
        "[Unit]\n"
        "Description=llama-swap (managed by Llama RPC Farm)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        f'ExecStart={binary} --config "{config_path}" --listen :{port} {SWAP_DAEMON_ARGS}\n'
        # on-failure (not always): a crash still resurrects the daemon,
        # but an intentional stop/kill stays stopped. With `always` every
        # pkill — including the restart script's and launch's own
        # stop step — was undone 5s later, looking like the daemon
        # "keeps restarting over and over".
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _start_linux_service(host, binary, config_path, port):
    """systemd user service; falls back to a detached background process.
    macOS has no systemctl and no setsid — nohup is the portable fallback."""
    is_local = local_node.is_local(host)
    ssh_home = None
    if not is_local:
        ssh_home = _remote_home(rpc_manager._ssh_for(host))
        unit_path = ssh_home + "/.config/systemd/user/llama-swap.service"
    else:
        unit_path = os.path.expanduser("~/.config/systemd/user/llama-swap.service")
    binary = _resolve_binary(host, binary)
    is_mac = host.get("os", "linux") == "macos"
    if not is_mac:
        _write_host(host, unit_path, _systemd_unit(binary, config_path, port))
        code, out, err = _run(
            host, "systemctl --user daemon-reload && "
            "systemctl --user enable --now llama-swap", timeout=30)
        if code == 0:
            return {"method": "systemd", "pid": None}
    # fallback: plain background process. setsid (util-linux) is missing on
    # macOS; nohup + & is universally available on POSIX.
    detach = "nohup" if is_mac else "setsid"
    log = "$HOME/llama-swap.log" if not is_local \
        else os.path.join(rpc_manager.GENERATED_DIR, "llama-swap.log")
    cmd = (f"{detach} {binary} --config \"{config_path}\" --listen :{port} {SWAP_DAEMON_ARGS} "
           f"</dev/null >{log} 2>&1 & echo \"PID=$!\"")
    if is_local:
        import subprocess
        os.makedirs(rpc_manager.GENERATED_DIR, exist_ok=True)
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=open(log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT, start_new_session=True)
        pid = proc.pid
    else:
        code, out, err = _run(host, cmd, timeout=15)
        pid = None
        if code == 0:
            for line in out.splitlines():
                if line.startswith("PID="):
                    pid = line[4:].strip()
    return {"method": f"{detach}-fallback", "pid": pid,
            "systemd_error": "" if is_mac else (err or out or "")[-300:]}


# DEC-06 compat alias: existing tests and external callers still speak
# `generate`; the name retires with the dead-key sweep (Card 15).
generate = launch_plan


def _scripts_subdir(swap_dir, engine):
    """Per-engine script subdir under the config dir (DEC-13, T2-G8):
    swap/<engine>/ — start-*.sh + sidecar for ONE engine never collide
    with the other engine's files, and toggling engines can never
    overwrite the inactive engine's scripts. Only linux hosts get
    scripts (windows embeds direct cmds), so a forward-slash join is
    safe even though swap_dir itself may be windows-styled."""
    return f"{swap_dir}/swap/{engine}"


def _ensure_host_dir(host, path):
    """Create a directory on the host (local or remote) if missing."""
    if local_node.is_local(host):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
        return
    if host.get("os", "linux") == "windows":
        _run(host, f'powershell -NoProfile -Command "'
              f'New-Item -ItemType Directory -Force -Path \\\\"{path}\\\\\""',
            timeout=15)
    else:
        _run(host, f'mkdir -p "{path}"', timeout=15)


def _backup_host_file(host, path, stamp):
    """Copy an existing host-side file to <path>.bak-<stamp> before it is
    overwritten (DEC-13, T2-G4). Returns the backup path, or None when the
    source doesn't exist. Never raises — backup is best-effort, the write
    itself is the hard gate."""
    is_local = local_node.is_local(host)
    bak = f"{path}.bak-{stamp}"
    if is_local:
        if not os.path.exists(path):
            return None
        import shutil
        try:
            shutil.copy2(path, bak)
            return bak
        except OSError:
            return None
    if host.get("os", "linux") == "windows":
        code, out, _ = _run(
            host, f'powershell -NoProfile -Command '
                  f'"if (Test-Path \\\\"{path}\\\\") {{ '
                  f'Copy-Item \\\\"{path}\\\\\" \\\\"{bak}\\\\\" -Force; '
                  f'echo BACKED_UP }}"', timeout=15)
        return bak if (code == 0 and "BACKED_UP" in (out or "")) else None
    code, out, _ = _run(
        host, f'if [ -f "{path}" ]; then cp "{path}" "{bak}" '
              f'&& echo BACKED_UP; fi', timeout=15)
    return bak if (code == 0 and "BACKED_UP" in (out or "")) else None


def push_scripts():
    """Regenerate + write the swap config and per-model scripts from the
    current wrapper model configs, WITHOUT touching peers or the daemon.
    For Tune/callers that just changed a config: makes the winner live
    for the NEXT load (already-loaded instances keep serving until
    evicted).

    DEC-13 (T2-G4): hardened push — validate BEFORE overwrite, and back
    up every existing file we are about to clobber. A plan whose YAML
    does not parse aborts the push with {"error": ...} and nothing is
    written; surviving operator configs/scripts get a .bak-<timestamp>
    copy first.
    Returns {"written": [...], "backups": [...], "validated": bool,
             "warnings": [...] } (or {"error": ...} on abort)."""
    settings = load_settings()
    gen = launch_plan(settings["config_path"], settings["host_name"])
    # validate BEFORE any overwrite: a plan that cannot parse must never
    # reach the host, or the daemon is left with a dead routing table.
    try:
        import yaml as _yaml
        _yaml.safe_load(gen["yaml"])
    except Exception as ye:
        return {"error": f"generated config does not parse; nothing "
                         f"written: {ye}", "validated": False,
                "warnings": gen.get("warnings", [])}
    host = cfg.machines.require(gen["host"])
    is_local = local_node.is_local(host)
    abs_config = gen["abs_config"]
    scripts_dir = gen.get("scripts_dir") or gen["swap_dir"]
    _ensure_host_dir(host, scripts_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = []
    # back up the config + every script we are about to overwrite
    for dest in [abs_config] + [f"{scripts_dir}/{fname}"
                                for fname in gen["_scripts"]]:
        bak = _backup_host_file(host, dest, stamp)
        if bak:
            backups.append(bak)
    _write_host(host, abs_config, gen["yaml"])
    written = [abs_config]
    for fname, content in gen["_scripts"].items():
        dest = scripts_dir + "/" + fname
        _write_host(host, dest, content)
        if is_local:
            try:
                # secrets sidecar is owner-only; scripts stay executable
                os.chmod(dest, 0o600 if fname == SIDECAR_NAME else 0o755)
            except OSError:
                pass
        elif fname == SIDECAR_NAME:
            try:
                _run(host, f"chmod 600 '{dest}'", timeout=15)
            except Exception:
                pass
        written.append(dest)
    return {"written": written, "backups": backups, "validated": True,
            "warnings": gen.get("warnings", []),
            "models": gen.get("models", [])}


def launch(binary=None, listen_port=None, config_path=None,
           host_name=None):
    """Full orchestration: generate, backup, write, validate, peers up,
    stop leftovers, start daemon, health-check. Returns a steps report.

    generate() produces the single plan (paths, scripts, modes);
    this function only executes it.
    """
    settings = load_settings()
    if binary:
        settings["binary"] = binary
    if listen_port:
        settings["listen_port"] = int(listen_port)
    if config_path:
        settings["config_path"] = config_path
    if host_name is not None:
        settings["host_name"] = host_name or None
    save_settings(settings["binary"], settings["listen_port"],
                  settings["config_path"], settings["host_name"])

    gen = launch_plan(settings["config_path"], settings["host_name"])
    host = cfg.machines.require(gen["host"])
    is_local = local_node.is_local(host)
    os_type = gen["host_os"]
    # Busy listen port (leftover daemon, unrelated app) must not fail the
    # launch outright: roll forward to the first free port on the host.
    # DEC-14: the roll is per-launch ONLY — never saved. Persisting the
    # rolled port made the default drift permanently (8080 -> 8081 ->
    # 8082...) every time a stale daemon held 8080 during a restart, and
    # every client pointed at the default then broke. The saved setting
    # stays the user's chosen port; a launch-time collision rolls for
    # that launch alone and the next clean restart lands back on the
    # configured port.
    port, port_rolled = rpc_manager._roll_port_if_busy(host, settings["listen_port"],
                                                       is_local)
    abs_config = gen["abs_config"]
    report = {"host": gen["host"], "port": port,
              "port_rolled": port_rolled,
              "config_path": abs_config, "steps": [],
              "models": gen["models"], "scripts": gen["scripts"],
              "warnings": gen["warnings"]}

    if not gen["models"]:
        report["steps"].append(
            {"step": "generate", "error": "no models generated"})
        report["failed"] = True
        return report

    def do_generate():
        return {"models": gen["models"], "scripts": gen["scripts"],
                "skipped": gen["skipped"]}

    def do_backup():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = abs_config + f".bak-{stamp}"
        if is_local:
            if not os.path.exists(abs_config):
                return {"backed_up": False, "note": "no existing config"}
            shutil.copy2(abs_config, bak)
            return {"backed_up": True, "backup": bak}
        if host.get("os", "linux") == "windows":
            # cmd.exe has no [ / cp — use PowerShell (Test-Path + Copy-Item)
            code, out, err = _run(
                host, f'powershell -NoProfile -Command '
                      f'"if (Test-Path \\"{abs_config}\\") {{ '
                      f'Copy-Item \\"{abs_config}\\" \\"{bak}\\" -Force; '
                      f'echo BACKED_UP }} else {{ echo NO_EXISTING }}"',
                timeout=15)
        else:
            code, out, err = _run(
                host, f'if [ -f "{abs_config}" ]; then cp "{abs_config}" '
                f'"{bak}" && echo BACKED_UP; else echo NO_EXISTING; fi',
                timeout=15)
        if code != 0:
            return {"error": (err or out or "")[-300:]}
        if "BACKED_UP" in out:
            return {"backed_up": True, "backup": bak}
        return {"backed_up": False, "note": "no existing config"}

    def do_write():
        # DEC-13 (T2-G8): scripts go to the per-engine scripts_dir
        # (swap/<engine>/), created on demand.
        _ensure_host_dir(host, gen.get("scripts_dir") or gen["swap_dir"])
        _write_host(host, abs_config, gen["yaml"])
        written = [abs_config]
        for fname, content in gen["_scripts"].items():
            dest = (gen.get("scripts_dir") or gen["swap_dir"]) + "/" + fname
            _write_host(host, dest, content)
            if not is_local:
                _run(host, f'chmod +x "{dest}"', timeout=15)
            else:
                try:
                    os.chmod(dest, 0o755)
                except OSError:
                    pass
            written.append(dest)
        return {"written": True, "files": written}

    def do_validate():
        cmd = f'{settings["binary"]} --config "{abs_config}" --validate'
        code, out, err = _run(host, cmd, timeout=60)
        tail = ((out or "") + (err or ""))[-800:]
        if code == 0:
            return {"valid": True, "output": tail}
        # Older llama-swap builds have no --validate flag. That usage
        # error means "can't validate", not "config invalid" -- fall back
        # to a YAML parse check so launch isn't blocked by binary version.
        if "flag provided but not defined: -validate" in tail:
            try:
                import yaml as _yaml
                _yaml.safe_load(gen["yaml"])
                return {"valid": True, "output": tail,
                        "note": "binary has no --validate; YAML parses OK"}
            except Exception as ye:
                return {"valid": False, "output": tail,
                        "error": f"YAML parse failed: {ye}"}
        return {"valid": False, "output": tail, "error": "config invalid"}

    # RPC peers up -- but only for direct-cmd models. Script-mode models
    # own their peers inside the .sh (spin up on load, down on exit),
    # so pre-starting them would just be torn down and redone.
    script_mids = set(gen["script_mids"])
    peer_names = []
    for model in cfg.list_models():
        mid = model.get("model_id")
        if mid in script_mids:
            continue
        if mid not in {e["model_id"] for e in gen["entries"]}:
            continue
        for s in model.get("servers") or []:
            if s != host["name"] and s not in peer_names:
                peer_names.append(s)

    def do_stop_swap():
        # Every Start click must first stop any running daemon: without
        # this each launch spawns another llama-swap (the new one rolls
        # to the next free port) and instances pile up. This also makes
        # switching between the vLLM and llama.cpp configs clean — the
        # old config's daemon is stopped before the new one starts.
        try:
            res = stop()
        except Exception as e:
            return {"stopped": False, "note": f"stop failed: {e}"}
        return {"stopped": res.get("stopped", False)}

    def do_start():
        if os_type == "windows":
            if not is_local:
                res = _start_windows_remote(host, settings["binary"], abs_config, port)
                if "error" in res:
                    return {"error": res["error"]}
                return res
            pid = _start_windows_local(
                settings["binary"], abs_config, port)
            return {"method": "detached-process", "pid": pid}
        return _start_linux_service(host, settings["binary"], abs_config,
                                    port)

    steps = [("generate", do_generate)]
    steps.append(("backup", do_backup))
    steps.append(("write", do_write))
    steps.append(("validate", do_validate))
    steps.append((f"stop:{host['name']}", rpc_manager.stop_rpc,
                   host["name"]))
    # Pre-start a base-port rpc-server on each peer for any DIRECT-cmd
    # model (non-script) that doesn't run ensure_rpc. ensure=True so a
    # running base-port server is reused and we never blanket-kill —
    # per-model rpc-servers (other ports) from co-served scripts survive.
    steps.extend((f"rpc:{p}", rpc_manager.start_rpc, p, None, None, None, True)
                 for p in peer_names)
    steps.append(("swap:stop", do_stop_swap))
    steps.append(("swap:start", do_start))
    run_steps(report, steps)
    if report["failed"]:
        return report

    waited, up = 0, False
    while waited < 60:
        if _port_open(host, port):
            up = True
            break
        time.sleep(2)
        waited += 2
    report["steps"].append(
        {"step": "swap:health", "running": up, "port": port,
         **({} if up else {"error": "port never opened"})})
    report["failed"] = not up
    return report


def stop():
    """Stop the llama-swap daemon on the host (peers stay up)."""
    host = get_host(load_settings()["host_name"])
    os_type = host.get("os", "linux")
    if os_type == "windows":
        code, out, err = _run(host, "taskkill /F /IM llama-swap.exe",
                              timeout=20)
        killed = _wait_gone(host)
        return {"host": host["name"], "stopped": killed,
                "detail": (out or err or "")[-300:]}
    if os_type == "macos":
        # no systemctl / pkill on macOS: kill by exact binary name (BSD ps
        # shows the command basename with `ps -axc -o pid,comm`)
        _run(host, 'killall llama-swap 2>/dev/null; '
                   'for p in $(ps -axc -o pid,comm | awk \'$2=="llama-swap"'
                   '{print $1}\'); do kill -9 "$p" 2>/dev/null; done; true',
             timeout=15)
        return {"host": host["name"], "stopped": _wait_gone(host)}
    _run(host, "systemctl --user stop llama-swap 2>/dev/null; "
               "systemctl --user disable llama-swap 2>/dev/null; true",
         timeout=20)
    _run(host, 'pkill -f "llama-[s]wap"; true', timeout=15)
    return {"host": host["name"], "stopped": _wait_gone(host)}


# Reload tracking for the save hook (polled by the UI).
reload_state = {"active": False, "started": None, "done": None, "error": None}


def reload_daemon():
    """Stop + start the swap daemon only (peers untouched) so a freshly
    pushed config takes effect immediately. Blocking; ~10-30s."""
    import time as _time
    settings = load_settings()
    host = get_host(settings["host_name"])
    os_type = host.get("os", "linux")
    is_local = local_node.is_local(host)
    stop()
    # roll past a busy port: stop() just freed ours, but something else
    # may still hold it (another daemon, an unrelated app). DEC-14: the
    # roll is per-launch only — never saved (see launch()'s note; the
    # saved port must stay the user's configured one or the default
    # walks 8080->8081->8082 across restarts and every client breaks).
    port, port_rolled = rpc_manager._roll_port_if_busy(
        host, settings["listen_port"], is_local)
    binary = settings["binary"]
    eff_path, _ = _effective_config_path(host, settings["config_path"])
    abs_config = _host_config_path(host, eff_path)
    if os_type == "windows":
        if not is_local:
            res = _start_windows_remote(host, binary, abs_config, port)
        else:
            res = {"pid": _start_windows_local(binary, abs_config, port)}
    else:
        res = _start_linux_service(host, binary, abs_config, port)
    waited, up = 0, False
    while waited < 60:
        if _port_open(host, port):
            up = True
            break
        _time.sleep(2)
        waited += 2
    return {"running": up, "port": port, "port_rolled": port_rolled,
            "start": res,
            **({} if up else {"error": "port never opened"})}


def reload_daemon_async():
    """Background entry for the save hook; skips if one is already running."""
    import time as _time
    if reload_state.get("active"):
        return {"skipped": True, "reason": "reload already in progress"}
    reload_state.update(active=True, started=_time.time(), done=None,
                        error=None)
    try:
        res = reload_daemon()
        reload_state.update(active=False, done=res)
        return res
    except Exception as e:
        reload_state.update(active=False, error=str(e))
        return {"error": str(e)}


def _wait_gone(host, tries=6):
    for _ in range(tries):
        if not _process_running(host):
            return True
        time.sleep(1)
    return not _process_running(host)


def status():
    """Is the swap daemon serving on the host?"""
    try:
        settings = load_settings()
    except ValueError as e:
        return {"running": False, "error": str(e)}
    host = get_host(settings["host_name"])
    port = settings["listen_port"]
    up = _port_open(host, port)
    return {"running": up, "host": host["name"], "port": port,
            # loopback for the local host (see load_settings note) so
            # the visualizer/log-tails reach a 127.0.0.1-bound swap
            "host_ip": "127.0.0.1" if local_node.is_local(host)
                       else host["ip"],
            "config_path": _host_config_path(host, settings["config_path"]),
            "process": _process_running(host) if up else False,
            "binary": settings["binary"]}
