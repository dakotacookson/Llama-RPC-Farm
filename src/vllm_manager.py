"""vLLM engine: command generation, probing, swap entries, validation.

Sibling of rpc_manager/swap_manager per the ARD (t_76f3d1c0 §1): all vLLM
knowledge lives here; rpc_manager stays closed to it. House style applies —
pure functions, string-exact commands, ValueError on invalid config (the
swap generator catches and skips with a warning, same as llama.cpp paths).

Scope (ARD §3): vLLM is single-host — one instance per machine (data-parallel
at farm level) or tensor-parallel within one multi-GPU host. There is no
on-demand multi-machine equivalent of llama.cpp --rpc, so `servers` must be
exactly one machine; multi-machine serving stays a llama.cpp concept.

Delivery comes from settings.yaml `vllm:` (schema owned by config_loader):
- venv:   <venv_bin> serve "<model>" --host 0.0.0.0 --port "$PORT" ...
- docker: docker run --rm --gpus all --name farm-<id> -p ${PORT}:8000 ...
with cmdStop per delivery (docker stop / pkill), because llama-swap stops
models by running cmdStop, not by killing the process group.

Device pinning: farm device chips ("aibox1-CUDA0") map to
CUDA_VISIBLE_DEVICES index lists — vLLM has no per-device token flags like
llama.cpp's `--device CUDA0`; the index env var is the equivalent. The
mapping travels inside cmd (prefix for venv, `-e` for docker) so it stays
visible in previews and identical across delivery modes; the per-model
`env:` dict is therefore always empty (B3 emits it untouched).
"""
import json
import os
import posixpath
import re

import config_loader as cfg
import gpu_vendor
import path_resolver
import rpc_manager
# DEC-02 (AUDIT-T6 §1.5): the UI field manifest protocol + the ui_hints
# shim live in src/engine_manifest.py — adapters mix in UiHintsMixin for
# the manifest contract (ui_hints() is derived from ui_manifest()).
from engine_manifest import UiHintsMixin

ENGINE_VLLM = cfg.ENGINE_VLLM


class SwapDownError(Exception):
    """The llama-swap daemon is not serving — the vLLM stop path (model
    unload via the swap host) cannot run. app.py maps this to 503."""


def swap_host_unload(model_id):
    """POST /api/models/unload/{model_id} on the swap host — llama-swap
    then runs the model entry's cmdStop, which under vLLM IS the engine
    stop (`docker stop farm-<id>` / pkill, build_cmd_stop). Shared by
    app.py's _proxy_swap_unload (/api/swap/evict) and the adapter's
    stop_model (/api/launch/stop vllm branch, B15 + DEC-06): one proxy
    body so the two entry points can never drift. Swap down -> SwapDownError."""
    import local_node
    import swap_manager
    st = swap_manager.status()
    if not st.get("running"):
        # parity with the pre-DEC-06 HTTPException(503) path
        raise SwapDownError("swap not running")
    host = cfg.machines.require(st["host"])
    is_local = local_node.is_local(host)
    url = (f"http://127.0.0.1:{st['port']}/api/models/unload/{model_id}"
           if is_local else
           f"http://{st.get('host_ip')}:{st['port']}/api/models/unload/{model_id}")
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return {"ok": True, "status": r.status,
                    "body": r.read().decode()[:500]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code,
                "error": e.read().decode()[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# keys allowed inside a model's `vllm:` section (ARD §3.3) — same strictness
# as the llama.cpp flag editor: unknown keys are rejected, never ignored.
# Schema vocabulary is owned by config_loader (B4); re-exported so existing
# importers keep working.
VLLM_MODEL_KEYS = cfg.VLLM_MODEL_KEYS

# research pitfall: vLLM's kv cache quantization supports fp8 only — the
# llama.cpp q4_0/q8_0 cache types have no equivalent and must never reach
# the command line.
VALID_KV_CACHE_DTYPES = cfg.VALID_KV_CACHE_DTYPES
VALID_COMPUTE_DTYPES = cfg.VALID_COMPUTE_DTYPES

# model_id lands in --served-model-name, the docker container name and the
# cmdStop pkill pattern — all injection surfaces. Strict allowlist instead
# of escaping: ids come from filenames and are already this shape.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Prometheus gauges surfaced to the Dashboard (ARD §3.2 "capacity").
_VLLM_METRIC_PREFIXES = (
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


def vllm_new_model_defaults():
    """The validated starting `vllm:` section for a NEW vLLM model.

    Live-verified on the 2-box farm (aibox1 3080Ti/3060s + aibox3 2060s,
    single-host AND multi-host, Qwen2.5 0.5B + 1.5B, opencode as the
    client). The editor pre-fills these; every value stays individually
    editable and the build-time flag generator treats them exactly like
    hand-set values:

    - dtype float32: the RTX 2060 peer cannot run vLLM's default bf16,
      and fp16 garbled opencode's agentic output on the 0.5B; fp32 gave
      clean generation. Users on uniform sm_80+ fleets can drop it.
    - enforce_eager True: CUDA graph capture hangs across a Ray group
      (upstream #36237/#41725) — live-reproduced on this farm.
    - enable_auto_tool_choice (with hermes parser): opencode and every
      agentic client sends tool_choice=auto; without the flags vLLM
      400s the whole request.
    - max_model_len 32768: opencode hardcodes max_tokens=16384; 8k
      context 400s on it ("cannot be greater than max_model_len").
    - max_num_seqs 16: modest batch for mixed consumer VRAM.
    - parallel_layout pp: layer split across hosts (llama.cpp-RPC
      style) — right for 1xPCIe/Ethernet; tp stays a per-model choice
      for Infiniband-class farms.
    """
    return {
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "dtype": "float32",
        "enforce_eager": True,
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",
        "parallel_layout": "pp",
    }


def _load_vllm_settings():
    """The normalized `vllm:` block from settings.yaml.

    Deferred import: engine_registry lazily imports THIS module to register
    the adapter, so a top-level import would be a cycle. At call time the
    registry is fully initialized.
    """
    from engine_registry import load_settings
    return load_settings().get("vllm") or {}


def _require_safe_id(model_cfg):
    """model_id, rejecting anything unsafe for cmd/cmdStop interpolation."""
    model_id = (model_cfg.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("model config has no model_id")
    if not _SAFE_ID.fullmatch(model_id):
        raise ValueError(
            f"model_id '{model_id}' contains characters unsafe for shell "
            "interpolation (allowed: letters, digits, dot, dash, underscore)")
    return model_id


# ---------------- device mapping (farm chips -> CUDA_VISIBLE_DEVICES) ------

def cuda_visible_devices(model_cfg, master):
    """Farm device chips -> CUDA_VISIBLE_DEVICES index list ("" = unset).

    "aibox1-CUDA0,aibox1-CUDA1" -> "0,1" (as-selected order, deduped).
    Bare GPU tokens ("CUDA0") belong to the host. RAM chips map to nothing:
    vLLM has no RAM device token (CPU offload is --cpu-offload-gb instead).
    Chips naming ANOTHER machine are refused — in vllm mode there are no
    peers, so a peer chip means the config was written for llama.cpp and
    silently running on fewer GPUs would explode VRAM on load.
    """
    master_name = (master or {}).get("name", "")
    ids = []
    for tok in (model_cfg.get("device_order") or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        machine, resource = rpc_manager._parse_chip(tok)
        if machine and machine != master_name:
            # Multi-host (ray cluster): chips may name peer machines.
            # This function is per-host: peer chips are skipped here —
            # the peer's own worker line calls it with the peer as master
            # and picks up that host's chips.
            continue
        if not resource or not gpu_vendor.is_gpu_token(resource):
            continue  # RAM chips / non-GPU tokens: no vLLM equivalent
        idx = gpu_vendor.gpu_index(resource)
        if idx not in ids:
            ids.append(idx)
    return ",".join(str(i) for i in ids)


# ---------------- command generation (ARD §3.1) ----------------------------

def build_vllm_command(model_cfg, master, vllm_cfg=None):
    """Build the `vllm serve` llama-swap cmd for one model (single host).

    DEC-08: the launch path reads ONLY the per-model `vllm:` section —
    the t_90def951 llama.cpp-flags bridge is retired. A model whose
    config has no `vllm:` section (a legacy YAML written for llama.cpp)
    fails loudly here instead of launching with silently wrong defaults.

    Multi-line shell joined with " \\\n  " exactly like
    rpc_manager.build_llama_command. Port follows the shared placeholder
    convention: `--port "$PORT"` (llama-swap sets PORT for the process;
    swap_manager rewrites it to ${PORT} for direct entries).

    docker delivery returns a `docker run` command: the container serves a
    fixed :8000 (mapped with -p ${PORT}:8000) and needs cmdStop — see
    build_cmd_stop.
    """
    vllm_cfg = vllm_cfg if vllm_cfg is not None else _load_vllm_settings()
    model_id = _require_safe_id(model_cfg)
    master = master or {}
    model_path = path_resolver.resolve_for(
        model_cfg.get("base_path") or model_cfg.get("path", ""),
        master.get("os", "linux"))
    if not model_path:
        raise ValueError("model config has no base_path")
    if not isinstance(model_cfg.get("vllm"), dict) or not model_cfg["vllm"]:
        raise ValueError(
            f"no vllm config for this model — '{model_id}' has no `vllm:` "
            "section; add one in the model editor (vLLM no longer reads "
            "llama.cpp flags)")
    ids = cuda_visible_devices(model_cfg, master)
    flags = _vllm_flag_parts(model_cfg, vllm_cfg, master.get("os", "linux"))
    if (vllm_cfg.get("delivery") or "venv") == "docker":
        return _docker_command(model_id, model_path, ids, flags, vllm_cfg)
    parts = [
        (f"CUDA_VISIBLE_DEVICES={ids} {vllm_cfg['venv_bin']} serve" if ids
         else f"{vllm_cfg['venv_bin']} serve"),
        f'"{model_path}"',
        "--host 0.0.0.0",
        '--port "$PORT"',
        f'--served-model-name "{model_id}"',
    ]
    parts.extend(flags)
    return " \\\n  ".join(parts)


def _vllm_flag_parts(model_cfg, vllm_cfg, host_os):
    """The value flags shared by both deliveries, in ARD §3.1 order.

    Raises ValueError on invalid values — a bad number must never reach
    the command line where vLLM would fail at load time (or worse, misparse).
    """
    v = model_cfg.get("vllm") or {}
    parts = []

    def num(key, flag, minimum, omit_zero=False):
        val = v.get(key)
        if val in (None, ""):
            return
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"vllm.{key} must be a number, got {val!r}")
        val = int(val)
        if omit_zero and val == 0:
            return  # offload 0 is the default; hide it (degradation matrix)
        if val < minimum:
            raise ValueError(f"vllm.{key} must be >= {minimum}, got {val}")
        parts.append(f"{flag} {val}")

    num("max_model_len", "--max-model-len", minimum=1)
    num("max_num_seqs", "--max-num-seqs", minimum=1)
    num("tensor_parallel_size", "--tensor-parallel-size", minimum=1)
    num("pipeline_parallel_size", "--pipeline-parallel-size", minimum=1)
    # Tool calling: auto tool choice requires a parser. Default parser is
    # "hermes" (works for Qwen family incl. GGUF builds); per-model override
    # via vllm.tool_call_parser; set vllm.enable_auto_tool_choice=False to
    # opt out.
    etc = v.get("enable_auto_tool_choice")
    if etc is not None and not isinstance(etc, bool):
        raise ValueError(
            "vllm.enable_auto_tool_choice must be a bool, got "
            f"{etc!r}")
    parser = v.get("tool_call_parser")
    if parser is not None and (not isinstance(parser, str)
                               or not parser.strip()):
        raise ValueError(
            "vllm.tool_call_parser must be a non-empty string, got "
            f"{parser!r}")
    # (validated above; emission lives in the canonical block below so
    # the flags are added exactly once).
    # A2: gpu_memory_utilization is per-model (heterogeneous machines).
    # The model's value wins; unset falls back to the settings-level
    # default. Validated (0, 1] at save time; the flag builder still
    # guards so a hand-edited config can't emit a broken command.
    gmu = v.get("gpu_memory_utilization")
    if gmu is None:
        gmu = vllm_cfg.get("default_gpu_memory_utilization")
    if gmu is not None:
        if isinstance(gmu, bool) or not isinstance(gmu, (int, float)) \
                or not 0 < float(gmu) <= 1:
            raise ValueError(
                "gpu_memory_utilization must be a number in (0, 1], "
                f"got {gmu!r}")
        parts.append(f"--gpu-memory-utilization {float(gmu)}")
    kv = v.get("kv_cache_dtype")
    if kv not in (None, "", "auto"):
        if kv not in VALID_KV_CACHE_DTYPES:
            raise ValueError(
                f"vllm.kv_cache_dtype '{kv}' is unsupported — allowed: "
                f"{list(VALID_KV_CACHE_DTYPES)} (fp8 needs SM89+; older "
                "consumer GPUs must use auto/half/bfloat16/float16)")
        parts.append(f"--kv-cache-dtype {kv}")
    dtype = v.get("dtype")
    if dtype not in (None, "", "auto"):
        if dtype not in VALID_COMPUTE_DTYPES:
            raise ValueError(
                f"vllm.dtype '{dtype}' is unsupported — allowed: "
                f"{list(VALID_COMPUTE_DTYPES)}")
        parts.append(f"--dtype {dtype}")
    # enforce_eager: user-controlled (default ON for multi-host — CUDA
    # graph capture hangs across Ray TP on Ethernet clusters, upstream
    # #36237/#41725). Stored as an explicit bool so the editor can offer
    # the checkbox; None/absent means "not chosen" and multi-host launch
    # then applies the safe default via the cluster script.
    if v.get("enforce_eager") is True:
        parts.append("--enforce-eager")
    elif v.get("enforce_eager") is False:
        pass  # explicit opt-out: graphs enabled (fast interconnects)
    # OpenAI tool-calling (default ON): agentic clients (opencode etc.)
    # send tools with tool_choice="auto", which vLLM refuses outright
    # ("auto tool choice requires --enable-auto-tool-choice and
    # --tool-call-parser") unless the server was launched with both
    # flags. Default parser is 'hermes' — the Qwen/Most template family
    # and the safest generic <tool_call> convention; per-model override
    # for the odd ones out (llama3_json, mistral, glm45, ...).
    if v.get("enable_auto_tool_choice") is not False:
        parts.append("--enable-auto-tool-choice")
        parser = v.get("tool_call_parser")
        if parser not in (None, "", "auto"):
            parts.append(f"--tool-call-parser {parser}")
        elif parser != "auto":  # absent -> the settings-level default parser
            parts.append(
                f"--tool-call-parser "
                f"{vllm_cfg.get('default_tool_call_parser', 'hermes')}")
    quant = v.get("quantization")
    if quant:
        if not isinstance(quant, str):
            raise ValueError("vllm.quantization must be a string")
        parts.append(f"--quantization {quant}")
    # attention backend (vLLM 0.18 `--attention-backend`, the vLLM
    # equivalent of llama.cpp's flash-attn toggle). The farm's friendly
    # aliases map to AttentionBackendEnum names in 0.18 (verified against
    # the live registry: FLASH_ATTN, TRITON_ATTN, FLASHINFER, ...).
    # flash_attn (FA2/3) needs sm_80+ — RTX 2060-class Turing cards
    # CANNOT use it (upstream vllm_flash_attn: "FA2 is only supported on
    # devices with compute capability >= 8"). auto/None (default) keeps
    # vLLM's own per-device pick — the flag is only emitted for an
    # explicit choice.
    attn = v.get("attention_backend")
    if attn not in (None, "", "auto"):
        alias_to_enum = {
            "flash_attn": "FLASH_ATTN",
            "triton": "TRITON_ATTN",
            "flashinfer": "FLASHINFER",
            "rocm_attn": "ROCM_ATTN",
            "flex_attention": "FLEX_ATTENTION",
            "torch_sdpa": "TORCH_SDPA",
        }
        if attn not in alias_to_enum:
            raise ValueError(
                f"vllm.attention_backend '{attn}' is unsupported — "
                f"allowed: {sorted(alias_to_enum)}")
        parts.append(f"--attention-backend {alias_to_enum[attn]}")
    num("cpu_offload_gb", "--cpu-offload-gb", minimum=0, omit_zero=True)
    tok = v.get("tokenizer")
    if tok:
        if not isinstance(tok, str):
            raise ValueError("vllm.tokenizer must be a path string")
        parts.append(
            f'--tokenizer "{path_resolver.resolve_for(tok, host_os)}"')
    extra = v.get("extra_args") or []
    if not isinstance(extra, list):
        raise ValueError("vllm.extra_args must be a list of strings")
    for a in extra:
        if not isinstance(a, str) or not a.strip():
            raise ValueError(
                "vllm.extra_args must be a list of non-empty strings")
        parts.append(a.strip())
    deb = v.get("distributed_executor_backend")
    if deb not in (None, "", "ray"):
        # ray is the default (and the only multi-node-capable backend the
        # farm supports) — 'mp' with peers is refused by the cluster
        # builder; single-host mp is vLLM's own default anyway.
        raise ValueError(
            f"vllm.distributed_executor_backend '{deb}' is unsupported — "
            "only 'ray' can form a multi-node cluster (mp needs a process "
            "on every node)")
    return parts


def _docker_command(model_id, model_path, ids, flags, vllm_cfg):
    """docker run shape from the research-validated llama-swap pattern:
    --gpus all + -e CUDA_VISIBLE_DEVICES for pinning, the model's parent
    directory mounted at /models, fixed container port 8000."""
    model_dir = posixpath.dirname(model_path)
    base = posixpath.basename(model_path)
    if not model_dir or not base or model_dir == model_path:
        raise ValueError(
            f"cannot derive a docker mount from model path '{model_path}'")
    parts = [
        "docker run --rm --gpus all",
        f"--name farm-{model_id}",
        "-p ${PORT}:8000",
        f'-v "{model_dir}:/models"',
    ]
    if ids:
        parts.append(f"-e CUDA_VISIBLE_DEVICES={ids}")
    parts.append(vllm_cfg.get("docker_image") or "vllm/vllm-openai:latest")
    parts.append(f'--model "/models/{base}"')
    parts.append(f'--served-model-name "{model_id}"')
    parts.extend(flags)
    return " \\\n  ".join(parts)


def build_cmd_stop(model_cfg, vllm_cfg=None):
    """llama-swap cmdStop for one vLLM model (ARD §3.1).

    docker: the container must be stopped by name (the proxy's child is
    `docker run`, killing it would orphan the server). venv: match the
    serve line — model_id is allowlist-safe (see _require_safe_id), so the
    pattern cannot be escaped.
    """
    vllm_cfg = vllm_cfg if vllm_cfg is not None else _load_vllm_settings()
    model_id = _require_safe_id(model_cfg)
    if (vllm_cfg.get("delivery") or "venv") == "docker":
        return f"docker stop farm-{model_id}"
    return f'pkill -f "vllm serve.*{model_id}"'


# ---------------- multi-host cluster script (multi-node TP/PP via Ray) ------

def _chips_per_host(model_cfg, servers):
    """GPU chips selected per host from device_order.

    Same token semantics as _derive_tensor_parallel_size: a
    `host-RESOURCE` token counts for that host, a bare GPU token counts
    for every host, a token for an unknown host counts nowhere. Returns
    {host_name: [tokens]} — hosts with an empty list have no GPU to
    offer the Ray cluster (they would start with --num-gpus=0 and vLLM
    could never schedule tensor-parallel workers on them).
    """
    order = model_cfg.get("device_order") or ""
    per_host = {(m or {}).get("name", ""): [] for m in servers}
    for tok in order.split(","):
        tok = tok.strip()
        if not tok:
            continue
        mach, resource = rpc_manager._parse_chip(tok)
        if not (resource and gpu_vendor.is_gpu_token(resource)):
            continue
        if mach:
            if mach in per_host:
                per_host[mach].append(tok)
        else:
            for chips in per_host.values():
                chips.append(tok)
    return per_host


def _require_chips_on_every_host(model_cfg, servers):
    """Multi-host gate: every host must own >=1 selected GPU chip.

    Without this the cluster script advertises --num-gpus=0 for the bare
    host and `vllm serve` can never place workers there — a silent
    cluster-formation failure. Raises ValueError naming the bare hosts
    (launch_plan turns it into a skip + warning; validate_model_config
    mirrors it as a save-time error so the editor enforces it too).
    """
    per_host = _chips_per_host(model_cfg, servers)
    bare = [name for name, chips in per_host.items() if not chips]
    if bare:
        raise ValueError(
            f"multi-host vLLM needs at least one GPU chip selected on "
            f"every host (device_order) — none on: "
            f"{', '.join(bare)}; pick chips in the model editor "
            f"(e.g. {bare[0]}-CUDA0)")


def _derive_tensor_parallel_size(model_cfg, servers):
    """Cluster TP size per design §2, in priority order:

    1. explicit `vllm.tensor_parallel_size` wins — but it must equal the
       total selected GPU chips across all servers when chip selection is
       known (device_order); a mismatched TP is a silent world-size bug,
       so refuse instead of guessing.
    2. else derive from the total selected GPU chips.
    3. else (no chip data at all) TP = len(servers) with a warning
       ("assuming 1 GPU per host").

    Returns (tp: int, issues: [str]) — issues are warnings, never errors.
    """
    v = model_cfg.get("vllm") or {}
    chips = []
    order = model_cfg.get("device_order") or ""
    for m in servers:
        name = (m or {}).get("name", "")
        for tok in order.split(","):
            tok = tok.strip()
            if not tok:
                continue
            mach, resource = rpc_manager._parse_chip(tok)
            if mach and mach != name:
                continue  # another host's chip (counted under that host)
            if resource and gpu_vendor.is_gpu_token(resource):
                chips.append(tok)
    explicit = v.get("tensor_parallel_size")
    if explicit not in (None, ""):
        if chips and int(explicit) != len(chips):
            raise ValueError(
                f"tensor_parallel_size {explicit} must equal the total "
                f"selected GPU chips across all hosts ({len(chips)}: "
                f"{model_cfg.get('device_order')})")
        return int(explicit), []
    if chips:
        return len(chips), []
    return len(servers), [
        f"no GPU chip selection — assuming tensor_parallel_size="
        f"{len(servers)} (1 GPU per host)"]


def build_vllm_cluster_command(model_cfg, master, peers, vllm_cfg=None):
    """Build the multi-host vLLM start script for one model (Ray cluster).

    Per design §1.1 (mirrors vLLM's own multi-node-serving.sh + the
    repo's _render_rpc pattern): ray workers are ssh'd (sshpass +
    $SSH_PASS from the sourced swap-secrets.env sidecar) onto every peer,
    the ray head starts on the master, the script polls `ray status`
    until all N nodes are alive (ray_init_timeout, default 300 s) and
    only then execs `vllm serve` with --distributed-executor-backend ray.
    The EXIT trap tears down ONLY the ray nodes this script itself
    started (PID files — orphans are the #1 leak risk).

    Linux-only by construction: validate_hosts refuses Windows/macOS
    before this is reached (windows/macOS hosts cannot join the ray
    cluster), so the peers list here is linux-only by the time we see it.
    """
    vllm_cfg = vllm_cfg if vllm_cfg is not None else _load_vllm_settings()
    model_id = _require_safe_id(model_cfg)
    master = master or {}
    peers = list(peers or [])
    host_os = master.get("os", "linux")
    model_path = path_resolver.resolve_for(
        model_cfg.get("base_path") or model_cfg.get("path", ""), host_os)
    if not model_path:
        raise ValueError("model config has no base_path")
    v = model_cfg.get("vllm") or {}
    if not isinstance(v, dict) or not v:
        raise ValueError(
            f"no vllm config for this model — '{model_id}' has no `vllm:` "
            "section; add one in the model editor (vLLM no longer reads "
            "llama.cpp flags)")
    backend = (v.get("distributed_executor_backend") or "ray").lower()
    if backend != "ray":
        raise ValueError(
            f"vllm.distributed_executor_backend '{backend}' is unsupported "
            "with multiple hosts — only 'ray' can form a multi-node "
            "cluster (mp needs a process on every node)")
    ray_port = int(vllm_cfg.get("ray_port") or 6379)
    ray_timeout = int(vllm_cfg.get("ray_init_timeout") or 300)
    _require_chips_on_every_host(model_cfg, [master] + peers)
    # Multi-host parallel layout (design: llama.cpp-RPC-style farms run
    # over 1xPCIe/Ethernet where per-layer TP allreduce is wasteful and
    # CG's cross-node TP hot path is the hang-prone one — upstream RFC
    # #35848). Default "pp": pipeline-parallel ACROSS hosts (stage per
    # host, one activation pass per boundary), TP *within* each host up
    # to its selected chip count. "tp" keeps the original all-chips TP
    # for users with real interconnects (Infiniband/NVLink).
    layout = (v.get("parallel_layout") or "pp").lower()
    if layout not in ("pp", "tp"):
        raise ValueError(
            f"vllm.parallel_layout '{layout}' is invalid — use 'pp' "
            "(pipeline across hosts, the Ethernet-friendly default) or "
            "'tp' (tensor-parallel across hosts, fast interconnects)")
    explicit_pp = v.get("pipeline_parallel_size")
    if layout == "tp":
        tp, _warns = _derive_tensor_parallel_size(
            model_cfg, [master] + peers)
        pp = int(explicit_pp) if explicit_pp not in (None, "") else 1
    else:
        # PP layout. Two valid topologies:
        #
        # uniform (the classic): one stage per HOST, TP shards within
        # each host across ITS chips — all hosts must contribute the
        # same chip count (vLLM's PP stages are uniform: world =
        # pp*tp, every stage runs tp_size workers; parallel_state.py
        # builds pp groups striding tp, so an unequal split would put
        # a stage's TP ranks on different hosts = TP over Ethernet).
        #
        # heterogeneous (TP=1, PP = total chips): rank layout is
        # contiguous and Ray packs single-GPU bundles node-first
        # (ray_utils initialize_ray_cluster, strategy="PACK"), so with
        # tp=1 every PP stage is exactly one rank = one whole GPU and
        # unequal per-host counts are fine — stage boundaries cross
        # hosts once per token, zero NCCL allreduce. This is the ONLY
        # clean layout when hosts have different GPU counts (e.g. 3+2).
        per_host = _chips_per_host(model_cfg, [master] + peers)
        widths = [len(c) for c in per_host.values()]
        total_chips = sum(widths)
        explicit_tp = v.get("tensor_parallel_size")
        if widths and len(set(widths)) > 1:
            # unequal chip counts: only TP=1 (pure pipeline, one GPU per
            # stage) is safe — any TP>1 would shard a stage across
            # hosts. TP=1 here is the heterogeneous default.
            if explicit_tp not in (None, "") and int(explicit_tp) != 1:
                raise ValueError(
                    f"hosts have different chip counts "
                    f"({ {k: len(c) for k, c in per_host.items()} }) — "
                    f"heterogeneous clusters only support tensor_"
                    f"parallel_size=1 (pure pipeline, one GPU per "
                    f"stage); got {explicit_tp}")
            tp = 1
            pp = total_chips
        else:
            tp = widths[0] if widths else 1
            if explicit_tp not in (None, "") and int(explicit_tp) != tp:
                raise ValueError(
                    f"tensor_parallel_size {explicit_tp} conflicts with "
                    f"the pp layout's per-host width {tp} (selected "
                    "chips per host); unset it or match the selection")
            pp = int(explicit_pp) if explicit_pp not in (None, "") \
                else len([master] + peers)
    # TP divisibility vs the model's attention heads, at the LAST moment
    # before the script is emitted: save-time validation only sees an
    # explicit tensor_parallel_size, but TP here may also be DERIVED from
    # the chip count — the exact hole that let a 14-head model launch
    # with TP=5 (fails only inside vLLM engine init).
    heads = _model_attention_heads(model_cfg, master)
    if heads and tp > 1 and heads % tp:
        raise ValueError(
            f"tensor_parallel_size {tp} does not divide the model's "
            f"{heads} attention heads evenly — vLLM requires "
            f"num_attention_heads % TP == 0; select a chip count / TP "
            f"that divides {heads}")
    master_bin = vllm_cfg.get("venv_bin") or "~/vllm-env/bin/vllm"
    master_ray_bin = os.path.join(
        os.path.dirname(os.path.dirname(os.path.expanduser(master_bin))),
        "bin", "ray")
    master_python = os.path.join(os.path.dirname(master_ray_bin), "python")
    flags = _vllm_flag_parts(model_cfg, vllm_cfg, host_os)
    master_ids = cuda_visible_devices(model_cfg, master)
    master_gpus = len([x for x in master_ids.split(",") if x]) \
        if master_ids else 0
    master_env_prefix = (
        "PYTORCH_ALLOC_CONF=expandable_segments:True "
        f"CUDA_VISIBLE_DEVICES={master_ids} "
                         if master_ids else "")

# per-peer launch lines + teardown, keyed by ip
    worker_starts = []
    worker_stops = []
    worker_pids = []
    for p in peers:
        ip = p.get("ip") or ""
        user = p.get("username") or "root"
        peer_bin = p.get("venv_bin") or (
            vllm_cfg.get("machine_vllm_venv_bin") or "")
        ray_bin = os.path.join(os.path.dirname(os.path.dirname(
            os.path.expanduser(peer_bin or master_bin))), "bin", "ray")
        peer_bin_dir = os.path.dirname(ray_bin)
        # $MASTER_IP resolves at runtime on the master (the loopback->LAN
        # rewrite above must run first); workers read it via the same
        # env var the head exported — pass the shell VARIABLE, not the
        # value, so the single-quoted remote command expands it there.
        ray_target = "$MASTER_IP"
        try:
            ids = cuda_visible_devices(model_cfg, p)
        except ValueError:
            ids = ""
        num_gpus = len([x for x in ids.split(",") if x]) if ids else 0
        log = f"$HOME/ray-worker-{model_id}.log"
        pidf = f"$HOME/ray-worker-{model_id}.pid"
        # Iface lookup for the peer's Gloo/NCCL pinning, positional-
        # parameter based: sed/awk programs need quoting that breaks
        # inside $() command-substitution reparsing (the peer sees bare
        # parens -> "syntax error near unexpected token '('"), and cut's
        # fixed field index is route-shape dependent ("via" gateways,
        # link routes). set -- + shift walks the words for the token
        # after "dev" — no quotes needed anywhere in the remote command.
        # Remote command quoting (verified live against a real peer):
        # the body is single-quoted so $PATH/$1/etc. expand on the PEER,
        # while MASTER_IP splices in via '"$MASTER_IP"' close-reopen —
        # the OUTER bash expands it before ssh runs, so the peer receives
        # a literal IP. All earlier attempts broke here: bare
        # double-quotes made the outer shell eat $(...) at the wrong
        # time or the peer hit $() reparsing with stripped quotes
        # ("syntax error near unexpected token '('").
        worker_starts.append(
            f"  sshpass -p \"$SSH_PASS\" ssh -n $SSH_OPTS \"{user}@{ip}\" \\\n"
            f"    'MIP='\"$MASTER_IP\"'; "
            f"set -- $(ip route get $MIP 2>/dev/null); "
            f"while [ $# -gt 1 ]; do [ $1 = dev ] && "
            f"{{ export GLOO_SOCKET_IFNAME=$2 NCCL_SOCKET_IFNAME=$2; "
            f"break; }}; shift; done; "
            f"export PATH={peer_bin_dir}:$PATH; "
            # PYTORCH_ALLOC_CONF must live HERE (worker-side), not in
            # ray_env: Ray only copies HF_/NCCL_/VLLM_/etc. prefixes to
            # workers, so a head-side export never reaches the peer's
            # vLLM worker process. expandable_segments lets the torch
            # allocator release reserved-but-unallocated pages — the
            # FP8 Marlin repack pass (gptq_marlin_repack, one 170MB
            # chunk per layer) OOM'd on a 12GB Turing card with 491MB
            # reserved-but-unallocated and only 127MB free. Also
            # CUDA_DEVICE_ORDER: heterogeneous nodes must map "GPU 0"
            # deterministically (FASTEST_FIRST default would reorder).
            + "export PYTORCH_ALLOC_CONF=expandable_segments:True; "
            + (f"export CUDA_VISIBLE_DEVICES={ids}; " if ids else "")
            + f"{ray_bin} stop --force >/dev/null 2>&1; "
            f"{ray_bin} start --address=$MIP:{ray_port} "
            f"--num-gpus={num_gpus} >> {log} 2>&1 & echo \\$! > {pidf}'")
        worker_stops.append(
            f'  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\\n'
            f"    '[ -f {pidf} ] && kill \"$(cat {pidf})\" >/dev/null "
            f"2>&1 || true; {ray_bin} stop --force >/dev/null 2>&1 "
            f"|| true'")
        worker_pids.append(ip)
    workers_start = "\n".join(worker_starts)
    workers_stop = "\n".join(worker_stops)

    flags_suffix = (" \\\n  " + " \\\n  ".join(flags)) if flags else ""
    master_ip = master.get("ip") or "127.0.0.1"
    first_peer_ip = (peers[0].get("ip") if peers else "") or ""

    # enforce_eager for multi-host: default ON unless the user explicitly
    # disabled it (high-speed interconnect). CUDA graph capture hangs
    # across a Ray TP group on Ethernet clusters (upstream #36237,
    # #41725 — capture stalls at 0% forever, both GPUs pinned at 100%).
    # Eager skips capture: no accuracy change, just slower decode.
    _eager = v.get("enforce_eager")
    eager_flag = "" if _eager is False else " --enforce-eager"
    # dedup: _vllm_flag_parts already emits --enforce-eager when True
    if _eager is True and any("--enforce-eager" in f for f in flags):
        eager_flag = ""

    # ray_env: per-model overrides of the multi-host transport envs.
    # DEFAULTS below are live-verified fixes for heterogeneous consumer
    # GPUs over Ethernet (aibox1 3080Ti/3060s + aibox3 2060s): without
    # them the first TP forward deadlocks (NCCL P2P/SHM negotiation hangs
    # cross-node — A/B-tested with torchrun: allreduce hangs without the
    # disables, completes with them). A well-built Infiniband/NVLink
    # cluster sets vllm.ray_env in the model config to drop them:
    #   ray_env: {NCCL_P2P_DISABLE: false, NCCL_SHM_DISABLE: false,
    #             enforce_eager: false}
    # Value false REMOVES the default key entirely; any other str/int
    # replaces it; extra keys are added verbatim (one export per line,
    # so the script stays grep-able and each key is independently
    # overridable). VLLM_/NCCL_/GLOO_ prefixed envs are auto-copied to
    # the ray workers by vLLM's ray/ray_env.py — no extra plumbing.
    ray_env_defaults = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "VLLM_USE_RAY_WRAPPED_PP_COMM": "0",
        # nccl (not shm): live-validated on the 2-box farm — shm channels
        # cannot carry GPU tensors cross-node; nccl needs cupy in the venv
        # (ray[cgraph]'s dependency). ray_non_carry_over note: NCCL debug
        # envs travel with the NCCL_ prefix automatically.
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": "nccl",
        # Per-STEP CG timeout (not per-request): default 300 dies hard on
        # a slow large-model prefill step over TCP (engine death + full
        # reload). 1800s costs nothing when steps are fast (the normal
        # case) and rides out pathological steps on 1xPCIe/TCP links.
        "RAY_CGRAPH_get_timeout": "1800",
        # Fragmentation guard (live-verified: the 27B INT8 PP load OOM'd
        # on a 12GB 2060 during engine profiling — 2.37GB transient
        # alloc with 2.24GB free, PyTorch's allocator holding
        # reserved-but-unallocated pages). Expandable segments let the
        # allocator return reserved pages, exactly the failure the vLLM
        # OOM message names. Cost: none observed; benefit on any
        # near-full card.
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    user_env = v.get("ray_env") or {}
    if user_env is not None and not isinstance(user_env, dict):
        raise ValueError("vllm.ray_env must be a mapping of "
                         "env-var -> value (use false to drop a default)")
    eff = dict(ray_env_defaults)
    for k, val in (user_env or {}).items():
        if val is False:
            eff.pop(k, None)  # explicit opt-out of a default
        elif val is True:
            eff[k] = "1"
        else:
            eff[str(k)] = str(val)
    # order-stable emission; values are env-var-safe (no quoting needed
    # for ints/enums — anything else would break `export KEY=VALUE`)
    ray_env_lines = "".join(
        f"export {k}={eff[k]}\n" for k in sorted(eff))

    # Heterogeneous stage balance (VLLM_RAY_BUNDLE_INDICES): Ray PACK
    # fills the master node's bundles first, so with a 3+1 chip split
    # ranks land [aibox1, aibox1, aibox1, aibox3] — the LAST rank (the
    # lm_head stage, 2.54GB fp16 vocab head on this 248K-vocab model)
    # would land on the peer's 12GB card and OOM (live-verified:
    # 9.31GB used + 2.37GB alloc = 11.68 > 11.56). The env remaps
    # rank->bundle so the peer's GPU takes a MIDDLE stage (no vocab
    # tower) and the big master cards take the head stages. Only
    # emitted for heterogeneous PP (uniform layouts keep vLLM's
    # default fill, which is already balanced). Read by the DRIVER
    # (ray_executor) — must be exported before `vllm serve`, NOT via
    # the ray_env worker-copy channel.
    bundle_indices_line = ""
    if layout == "pp" and widths and len(set(widths)) > 1 and tp == 1:
        master_n = widths[0]
        peer_bundles = list(range(master_n, total_chips))
        # interleave: rank0 = master bundle 0 (embed), then one peer
        # bundle per remaining master bundle slot, then the rest of the
        # master bundles (last = lm_head). For 3+1: "0,3,1,2".
        peer_bundles_iter = iter(peer_bundles)
        master_rest = list(range(1, master_n))
        stage_order = [0]
        # place peer stages as early as possible (peer cards take
        # KV-light mid-pipeline slots; master's bigger cards take the
        # tower stages). Round-robin until both pools drain.
        pi, mi = 0, 0
        while pi < len(peer_bundles) or mi < len(master_rest):
            if pi < len(peer_bundles):
                stage_order.append(peer_bundles[pi]); pi += 1
            if mi < len(master_rest):
                stage_order.append(master_rest[mi]); mi += 1
        bundle_indices_line = (
            "# Heterogeneous stage placement: peer GPUs take mid-pipeline\n"
            "# stages (no vocab towers); master's biggest cards take the\n"
            "# embed/lm_head stages (see comment above).\n"
            f"export VLLM_RAY_BUNDLE_INDICES={','.join(str(b) for b in stage_order)}\n")

    return f"""#!/bin/bash
# vLLM multi-host Ray cluster (generated) — model {model_id}
PORT="${{1:-8080}}"
MASTER_IP="{master_ip}"
PEER_IP="{first_peer_ip}"
# The app often runs ON the swap host, whose configured IP is loopback
# (machines.yaml 127.0.0.1): peers cannot dial that, so resolve the LAN
# source address used to reach the first peer. A real configured LAN IP
# is kept as-is — no config change needed when switching engines.
if [[ "$MASTER_IP" == 127.* || "$MASTER_IP" == "localhost" ]]; then
  _route_src=$(ip route get "$PEER_IP" 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\\1/p')
  [[ -n "$_route_src" ]] && MASTER_IP="$_route_src"
fi
RAY_PORT={ray_port}
N={1 + len(peers)}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
# Gloo (vLLM's CPU-side torch.distributed group alongside NCCL) resolves
# worker hostnames through /etc/hosts — Ubuntu maps the local hostname to
# 127.0.1.1, so ranks advertise a loopback address peers can't dial
# ("Gloo connectFullMesh ... Connection refused, remote=[127.0.1.1]").
# Pinning the LAN interface (the one routing to the first peer) makes
# every rank advertise a real address. Standard vLLM multi-node fix.
if [[ -n "$PEER_IP" ]]; then
  GLOO_IF=$(ip route get "$PEER_IP" 2>/dev/null | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p')
  [[ -n "$GLOO_IF" ]] && export GLOO_SOCKET_IFNAME="$GLOO_IF" NCCL_SOCKET_IFNAME="$GLOO_IF"
fi
# Ray CG transport fixes (vLLM 0.18 multi-node hangs, live-verified on
# this farm — see vllm_manager.ray_env_defaults for the full story).
# Every key is per-model overridable via `vllm.ray_env` in the model
# config; `false` as a value removes the env entirely for clusters with
# real interconnects (Infiniband/NVLink) where the fixes cost throughput.
{ray_env_lines}# flashinfer/triton JIT need the ninja binary at runtime. On peers the
# attention backend falls back to FLASHINFER when FA2 is unsupported
# (sm_75 GPUs like the RTX 2060), and its JIT build invokes `ninja` by
# bare name — the venv ships ninja in its own bin/, so put that on PATH
# for head AND workers (bare-name lookup, not just our vllm binary).
export PATH="$(dirname "{master_bin}"):$PATH"
# Delivery proof for stage-partition envs (VLLM_PP_LAYER_PARTITION must
# appear both here — the serve process env — and in vLLM's "Copying the
# following environment variables to workers" log line before any
# multi-stage load attempt can be trusted):
echo "[cluster] VLLM_PP_LAYER_PARTITION=${{VLLM_PP_LAYER_PARTITION:-<unset>}}" >&2

start_workers() {{
{workers_start}
}}

stop_workers() {{
{workers_stop}
}}

start_head() {{
  # Pre-start stale-cluster cleanup, same contract as the llama.cpp RPC
  # scripts' port-targeted peer kill: a previous load that died without
  # its EXIT trap (kill -9, host reboot, OOM) leaves the GCS
  # holding $RAY_PORT, and `ray start --head` then refuses with
  # "already running at ..." — every subsequent load fails until
  # manual cleanup. `ray stop --force` first; if the port is STILL
  # held (stop only works on a live session dir), kill whoever listens.
  {master_ray_bin} stop --force >/dev/null 2>&1 || true
  if ss -ltn 2>/dev/null | grep -q ":$RAY_PORT "; then
    fuser -k $RAY_PORT/tcp >/dev/null 2>&1 || true
    sleep 1
  fi
  # CUDA_VISIBLE_DEVICES pinning on the HEAD too: ray start --head
  # --num-gpus=N with all GPUs visible lets Ray pick ANY N (it takes
  # the first visible) — on aibox1 with 3 of 3 selected this is moot,
  # but a subset selection (e.g. CUDA1+CUDA2 only) would let Ray place
  # workers on CUDA0 (the excluded card). Same contract as the peer's
  # pinned export below (live-verified: an unpinned 1-GPU peer pulled
  # in a chip the model excluded and the rank OOM'd on it).
  {master_env_prefix}{master_ray_bin} start --head --port=$RAY_PORT --num-gpus={master_gpus} \\
>> $HOME/ray-head-{model_id}.log 2>&1 &
  echo $! > $HOME/ray-head-{model_id}.pid
}}

stop_head() {{
  [ -f $HOME/ray-head-{model_id}.pid ] && \\
kill "$(cat $HOME/ray-head-{model_id}.pid)" >/dev/null 2>&1 || true
  {master_ray_bin} stop --force >/dev/null 2>&1 || true
}}

cleanup() {{
  stop_workers
  stop_head
}}
trap cleanup EXIT

# ORDER: head FIRST, then workers. The head's GCS must be listening
# before peers dial --address=$MASTER_IP:6379; with ray 2.58 a worker
# that starts before the GCS exists can register a stale session or
# die with WrongClusterID when the head appears (live-verified: the
# peer joined "successfully" into nothing, the head's alive-loop then
# saw 1 node and aborted after the 300s deadline).
start_head
start_workers

# Wait for cluster formation (vLLM multi-node-serving.sh leader loop)
deadline=$((SECONDS + {ray_timeout}))
while (( SECONDS < deadline )); do
  # `ray status` text format varies by release (2.44 prints no "Alive"
# lines at all) — count live nodes through the Ray API instead.
# Address auto-resolves to the head started above (same host).
alive=$({master_python} -c "import ray; ray.init(address='auto', ignore_reinit_error=True, log_to_driver=False); print(sum(1 for n in ray.nodes() if n.get('Alive')))" 2>/dev/null || echo 0)
  (( alive >= N )) && break
  sleep 5
done
if (( alive < N )); then
  echo "!!! ray cluster never reached $N alive nodes (got $alive) — aborting" >&2
  echo "!!! hint: every host needs the same venv (~/vllm-env/bin/ray + vllm) — see ray-head-{model_id}.log here and ray-worker-{model_id}.log on each peer" >&2
  exit 1
fi

# vLLM serves as a CHILD, never exec'd: exec replaces the bash process
# image and silently destroys the EXIT trap, so cmdStop's pkill of the
# vllm process orphaned the remote ray workers (aibox3 kept its VRAM
# pinned after unload; only the start-side ray stop --force healed it
# on the NEXT load). With wait, vllm's SIGTERM makes wait return, the
# trap fires, and stop_workers/stop_head tear down the remote cluster.
# wait's exit status IS vllm's, so llama-swap still sees the true
# lifecycle result.
{bundle_indices_line}{master_bin} serve "{model_path}" \\
  --host 0.0.0.0 \\
  --port "$PORT" \\
  --served-model-name "{model_id}" \\
  --pipeline-parallel-size {pp} \\
  --tensor-parallel-size {tp} \\
  --distributed-executor-backend ray{eager_flag}{flags_suffix} &
VLLM_PID=$!
wait $VLLM_PID
"""


def swap_entry_parts(model_cfg, master, vllm_cfg=None, *, peers=None):
    r"""One llama-swap model entry for vLLM: cmd/cmdStop/checkEndpoint.

    B3's branch point: swap_manager.launch_plan() calls this at entry-emit
    time when settings engine=vllm, instead of the RPC wrapper-script path.
    checkEndpoint is /health (vLLM's readiness endpoint — the same signal
    probe_health uses). env is empty by design: device pinning travels
    inside cmd (prefix / -e) so preview and runtime can never drift.

    Multi-host (multi-host TP/PP, Ray cluster): when `peers` is non-empty
    the cmd becomes the cluster script from build_vllm_cluster_command.

    DEC-08: the t_90def951 bridge_warnings channel is retired along with
    the bridge itself — the entry carries only the launch fields.
    """
    vllm_cfg = vllm_cfg if vllm_cfg is not None else _load_vllm_settings()
    peers = list(peers or [])
    return {
        "cmd": (build_vllm_cluster_command(model_cfg, master, peers,
                                           vllm_cfg) if peers else
                build_vllm_command(model_cfg, master, vllm_cfg)),
        "cmdStop": build_cmd_stop(model_cfg, vllm_cfg),
        "checkEndpoint": "/health",
        "env": {},
    }


# ---------------- probing (ARD §3.2) ---------------------------------------

def _http_get(url, timeout=3):
    """GET url -> (status:int|None, body:str). status None = no HTTP answer
    ( refused/reset/timeout ). HTTP error codes ARE answers (503 = starting)
    and come back as (code, body). Isolated so tests can stub it."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return e.code, body
    except Exception:
        return None, ""


def probe_health(machine, port, is_local):
    """Liveness: GET /health -> 200 means serving. Replaces the RPC TCP
    port-open check as the running signal (ARD §3.2) — vLLM binds the port
    long before it can answer requests, so only HTTP proves readiness.
    is_local is accepted for adapter-shape parity; the machine's ip is used
    either way (127.0.0.1 via machine ip when local)."""
    ip = (machine or {}).get("ip") or "127.0.0.1"
    status, _ = _http_get(f"http://{ip}:{port}/health")
    if status == 200:
        return {"running": True, "detail": "vLLM /health 200"}
    if status is None:
        return {"running": False, "detail": "vLLM /health unreachable"}
    if status == 503:
        return {"running": False, "detail": "vLLM /health 503 (starting)"}
    return {"running": False, "detail": f"vLLM /health {status}"}


def probe_identity(machine, port, expected_model_id=None):
    """Identity: GET /v1/models -> served model ids; when expected_model_id
    is given, verifies the served-model-name actually routes (catches a
    server that came up serving the wrong or default-named model)."""
    ip = (machine or {}).get("ip") or "127.0.0.1"
    status, body = _http_get(f"http://{ip}:{port}/v1/models")
    if status != 200:
        return {"ok": False, "models": [], "match": None,
                "detail": f"/v1/models {status or 'unreachable'}"}
    try:
        data = json.loads(body or "{}")
        models = [str(m.get("id")) for m in (data.get("data") or [])
                  if isinstance(m, dict)]
    except (ValueError, AttributeError):
        return {"ok": False, "models": [], "match": None,
                "detail": "/v1/models returned non-JSON"}
    match = None if expected_model_id is None \
        else expected_model_id in models
    return {"ok": True, "models": models, "match": match,
            "detail": f"{len(models)} model(s) served"}


def probe_capacity(machine, port):
    """Capacity: GET /metrics -> KV-cache/queue gauges for the Dashboard
    panels (fed alongside stats_collector, which stays engine-agnostic)."""
    ip = (machine or {}).get("ip") or "127.0.0.1"
    status, body = _http_get(f"http://{ip}:{port}/metrics")
    if status != 200:
        return {"ok": False, "metrics": {},
                "detail": f"/metrics {status or 'unreachable'}"}
    wanted = {}
    for line in (body or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, raw = line.rpartition(" ")
        if not name:
            continue
        if any(name.startswith(p) for p in _VLLM_METRIC_PREFIXES):
            try:
                wanted[name] = float(raw)
            except ValueError:
                continue
    if not wanted:
        return {"ok": True, "metrics": {},
                "detail": "/metrics reachable, no vLLM gauges found"}
    return {"ok": True, "metrics": wanted,
            "detail": f"{len(wanted)} vLLM gauge(s)"}


# ---------------- validation (ARD §3.3, §5) --------------------------------

def _model_attention_heads(model_cfg, master=None):
    """num_attention_heads from the model's HF config.json, else None.

    Reads the config next to the weights on the SWAP HOST (master's OS
    path resolution) — vLLM's own head-divisibility check at engine init
    is the same source of truth. Returns None on any miss (GGUF dirs,
    missing file, unparsable JSON, non-local store the app can't read)
    so validation stays silent rather than guessing.
    """
    raw_path = model_cfg.get("base_path") or model_cfg.get("path") or ""
    if not raw_path or raw_path.lower().endswith(".gguf"):
        return None
    host_os = (master or {}).get("os", "linux")
    try:
        real = path_resolver.resolve_for(raw_path, host_os)
    except Exception:
        return None
    if not real:
        return None
    cfg_path = os.path.join(real, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    heads = (data or {}).get("num_attention_heads")
    if isinstance(heads, int) and not isinstance(heads, bool) and heads > 0:
        return heads
    return None


def validate_model_config(model_cfg):
    """Static vLLM config validation, no side effects -> issues list of
    {"level": "error"|"warn", "msg"} (same shape as
    rpc_manager.validate_model). Rules:
    - servers: exactly one machine in vllm mode (hard error — multi-machine
      is the llama.cpp RPC concept; TP is within-host)
    - vllm.kv_cache_dtype: auto|fp8 only (research pitfall: silent q4/q8)
    - unknown vllm keys rejected (same strictness as the kv editor)
    - GGUF base_path: experimental — requires vllm.tokenizer (base HF dir)
    """
    issues = []

    def err(msg):
        issues.append({"level": "error", "msg": msg})

    def warn(msg):
        issues.append({"level": "warn", "msg": msg})

    model_cfg = model_cfg or {}
    servers = model_cfg.get("servers") or []
    if not servers:
        err("no servers configured")
    elif len(servers) > 1:
        # Multi-host vLLM: allowed — Ray cluster (TP/PP across hosts).
        # Gate on host eligibility instead of the machine count: every
        # server must be a known machine, and validate_hosts() separately
        # refuses Windows/macOS (vLLM multi-host is Linux + ray only).
        known = [s for s in servers if cfg.machines.get(s)]
        unknown = [s for s in servers if not cfg.machines.get(s)]
        for u in unknown:
            err(f"unknown machine '{u}'")
        host_errors = validate_hosts(
            [cfg.machines.require(s) for s in known])
        for he in host_errors:
            err(he)
        try:
            _require_chips_on_every_host(
                model_cfg, [{"name": s} for s in servers])
        except ValueError as e:
            err(str(e))
        _v_multi = model_cfg.get("vllm") or {}
        layout = (_v_multi.get("parallel_layout") or "pp").lower()
        if layout == "tp":
            warn(f"multi-host vLLM: tensor-parallel across {len(servers)} "
                 f"machines over ray (master: {servers[0]}) — allreduce "
                 "every layer; only sensible on fast interconnects, "
                 "and the Ray CG hot path is hang-prone cross-node "
                 "(upstream RFC #35848). Consider parallel_layout: pp")
        else:
            warn(f"multi-host vLLM: pipeline-parallel across "
                 f"{len(servers)} machines over ray, TP within each "
                 f"host (master: {servers[0]})")
    try:
        _require_safe_id(model_cfg)
    except ValueError as e:
        err(str(e))
    master = None
    if servers:
        try:
            master = cfg.machines.get(servers[0])
        except Exception:
            master = None
        if master is None:
            err(f"unknown machine '{servers[0]}'")
    # GGUF base_path: experimental — requires vllm.tokenizer (base HF dir).
    # Checked BEFORE the vllm-block early-return: base_path alone decides,
    # and `vllm serve model.gguf` without a tokenizer fails at load time
    # even when the model sets no other vllm keys.
    if str(model_cfg.get("base_path") or "").lower().endswith(".gguf"):
        raw_v = model_cfg.get("vllm")
        tok = raw_v.get("tokenizer") if isinstance(raw_v, dict) else None
        if tok:
            warn("GGUF on vLLM is experimental (vllm-gguf-plugin) — "
                 "expect limited support")
        else:
            err("GGUF on vLLM needs vllm.tokenizer pointing at the base HF "
                "model directory (vllm-gguf-plugin is experimental)")
    v = model_cfg.get("vllm")
    if v is None:
        return issues  # engine-agnostic config: nothing vLLM-specific
    if not isinstance(v, dict):
        err("vllm: must be a mapping with keys "
            + ", ".join(sorted(VLLM_MODEL_KEYS)))
        return issues
    unknown = sorted(set(v) - VLLM_MODEL_KEYS)
    if unknown:
        err(f"unknown vllm key(s) {unknown} — allowed: "
            f"{sorted(VLLM_MODEL_KEYS)}")
    for key, minimum in (("max_model_len", 1), ("max_num_seqs", 1),
                         ("tensor_parallel_size", 1),
                         ("pipeline_parallel_size", 1), ("cpu_offload_gb", 0)):
        val = v.get(key)
        if val in (None, ""):
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            err(f"vllm.{key} must be a number, got {val!r}")
            continue
        if val < minimum:
            err(f"vllm.{key} must be >= {minimum}, got {val}")
    gmu = v.get("gpu_memory_utilization")
    if gmu not in (None, "") and (
            isinstance(gmu, bool) or not isinstance(gmu, (int, float))
            or not 0 < float(gmu) <= 1):
        err("vllm.gpu_memory_utilization must be a number in (0, 1], "
            f"got {gmu!r}")
    kv = v.get("kv_cache_dtype")
    if kv not in (None, "") and kv not in VALID_KV_CACHE_DTYPES:
        err(f"vllm.kv_cache_dtype '{kv}' is unsupported — allowed: "
            f"{list(VALID_KV_CACHE_DTYPES)} (fp8 needs SM89+ GPUs)")
    quant = v.get("quantization")
    if quant not in (None, "") and not isinstance(quant, str):
        err("vllm.quantization must be a string")
    tok = v.get("tokenizer")
    if tok not in (None, "") and not isinstance(tok, str):
        err("vllm.tokenizer must be a path string")
    dtype = v.get("dtype")
    if dtype not in (None, "", "auto") \
            and dtype not in VALID_COMPUTE_DTYPES:
        err(f"vllm.dtype '{dtype}' is unsupported — allowed: "
            f"{sorted(VALID_COMPUTE_DTYPES)}")
    # Heterogeneous-hardware dtype warning (never an error): vLLM
    # defaults to bfloat16 weights, which fails at LOAD time on GPUs
    # without bf16 arithmetic (NVIDIA sm < 8.0). The sm comes from the
    # driver itself (`nvidia-smi compute_cap`, stored per GPU in
    # detected.compute_cap at probe time) — never from product-name
    # tables. Missing data stays silent so new/rediscovered hardware
    # never trips validation.
    if dtype in (None, "", "auto", "bfloat16"):
        order = model_cfg.get("device_order") or ""
        bad = []
        for tok in order.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                mach, resource = rpc_manager._parse_chip(tok)
            except Exception:
                continue
            if not (resource and gpu_vendor.is_gpu_token(resource)):
                continue
            hosts = servers if not mach else [s for s in servers
                                              if s == mach]
            for host_name in hosts:
                try:
                    m = cfg.machines.get(host_name)
                except Exception:
                    continue
                caps = ((m or {}).get("detected") or {}).get(
                    "compute_cap") or []
                idx = gpu_vendor.gpu_index(resource)
                if idx >= len(caps):
                    continue
                ok = gpu_vendor.compute_cap_supports_bf16(caps[idx])
                if ok is False and host_name not in bad:
                    bad.append(f"{host_name}/sm_{caps[idx]}")
        if bad:
            warn(f"bfloat16 weights will not load on: {', '.join(bad)} "
                 f"— set vllm.dtype to 'half' for this model")
    extra = v.get("extra_args")
    if extra is not None and (
            not isinstance(extra, list)
            or any(not isinstance(a, str) or not a.strip() for a in extra)):
        err("vllm.extra_args must be a list of non-empty strings")
    # parallelism vs pinned devices: catch the mismatch statically instead
    # of at vLLM load time ("TP size > visible GPUs" is fatal there).
    # World size = TP×PP — vLLM's pipeline parallelism also partitions the
    # SAME visible GPUs (TP shards layers across GPUs, PP assigns pipeline
    # stages to GPUs), so a TP=2,PP=2 model needs 4 visible GPUs, not 2.
    tp = v.get("tensor_parallel_size")
    pp = v.get("pipeline_parallel_size")
    world = ((tp if isinstance(tp, int) and not isinstance(tp, bool) else 1)
             * (pp if isinstance(pp, int) and not isinstance(pp, bool) else 1))
    if world > 1 and master is not None:
        try:
            ids = cuda_visible_devices(model_cfg, master)
        except ValueError as e:
            err(str(e))
            ids = ""
        if ids:
            n = len(ids.split(","))
            pp_active = isinstance(pp, int) and not isinstance(pp, bool) \
                and pp > 1
            if n < world:
                # display effective sizes (unset == 1, vLLM's default)
                tp_d = tp if isinstance(tp, int) and not isinstance(tp, bool) \
                    else 1
                if pp_active:
                    err(f"parallelism world size TP{tp_d}xPP{pp}={world} "
                        f"needs {world} visible GPUs, CUDA_VISIBLE_DEVICES "
                        f"selects {n} ({ids})")
                else:
                    err(f"tensor_parallel_size {tp} needs {tp} visible "
                        f"GPUs, CUDA_VISIBLE_DEVICES selects {n} ({ids})")
            elif n > world:
                warn(f"CUDA_VISIBLE_DEVICES selects {n} GPUs but "
                     f"parallelism world size is {world} (TP={tp}, PP={pp})"
                     f" — vLLM will use the first {world}")
    # TP divisibility vs the model's architecture: vLLM shards attention
    # heads per TP rank and REQUIRES TP to divide num_attention_heads
    # evenly (e.g. Qwen2.5-0.5B has 14 heads -> TP=5 fails at engine
    # init with a pydantic ValidationError; TP=1,2,7,14 work). The head
    # count comes from the HF config.json next to the weights — silent
    # skip when absent (GGUF, unreadable path, non-standard layout).
    tp_eff = tp if isinstance(tp, int) and not isinstance(tp, bool) else 1
    if tp_eff > 1:
        heads = _model_attention_heads(model_cfg, master)
        if heads and heads % tp_eff:
            err(f"tensor_parallel_size {tp_eff} does not divide the model's"
                f" {heads} attention heads evenly — vLLM requires "
                f"num_attention_heads % TP == 0 (valid TPs divide {heads})")
    # A2 parity guard: the per-entry knobs are machine-blind numbers; a
    # value tuned for the farm's strong GPU can be fatal (OOM at load)
    # or pointless on a small one. Cross-check against the serving
    # machine's detected resources — warn (never error: detection can be
    # stale/absent, and vLLM's own loader is the final arbiter).
    if master is not None:
        detected = master.get("detected") or {}
        try:
            vram = float(detected.get("vram_total_gb") or 0)
        except (TypeError, ValueError):
            vram = 0.0
        try:
            ram = float(detected.get("ram_total_gb") or 0)
        except (TypeError, ValueError):
            ram = 0.0
        # GPU-memory budget vs. total VRAM: gmu=1.0 is legitimate for a
        # dedicated box, so only flag the clearly implausible (>1 or the
        # schema-forbidden <=0) — the save path already rejects those.
        if vram and isinstance(gmu, (int, float)) \
                and not isinstance(gmu, bool) and gmu > 1:
            warn(f"gpu_memory_utilization {gmu} exceeds 1.0 — vLLM will "
                 "fail at load (fraction of VRAM, not GiB)")
        # CPU offload vs. RAM: offloading more than the box has RAM
        # cannot work (weights stream from host memory).
        off = v.get("cpu_offload_gb")
        if ram and isinstance(off, (int, float)) \
                and not isinstance(off, bool) and off > 0 \
                and float(off) > ram:
            warn(f"cpu_offload_gb {off} exceeds the host's detected RAM "
                 f"({ram:g} GB) — vLLM cannot offload more weights than "
                 "host memory")
    return issues


def validate_hosts(machines):
    """Engine×OS gate -> [str] errors (ARD §5 gates).

    - windows: refuse natively-run vLLM with WSL2 guidance (research: no
      native Windows wheels; the installer must never even try)
    - macos: vLLM-Metal is a separate package the farm doesn't install —
      refuse rather than ship a broken venv
    - linux + ROCm: wheels are Python-3.12-only (silent-CUDA-fallback
      pitfall). Checked when the machine's detected python_version is
      known; the installer (B6) re-checks at runtime.
    """
    errors = []
    for m in machines or []:
        name = (m or {}).get("name", "?")
        os_type = ((m or {}).get("os") or "linux").strip().lower()
        if os_type == "windows":
            errors.append(
                f"{name}: vLLM cannot run natively on Windows — run it "
                "under WSL2 and add the WSL instance as a linux machine, "
                "or exclude this host in vllm mode")
        elif os_type == "macos":
            errors.append(
                f"{name}: vLLM on macOS needs the separate vLLM-Metal "
                "package (not installed by the farm) — run vLLM on a "
                "Linux host instead")
        elif gpu_vendor.normalize_vendor((m or {}).get("gpu_vendor")) == "amd":
            py = (m or {}).get("python_version") \
                or ((m or {}).get("detected") or {}).get("python_version")
            if py and not str(py).startswith("3.12"):
                errors.append(
                    f"{name}: ROCm vLLM wheels require Python 3.12 "
                    f"(host reports {py})")
    return errors


# ---------------- multimodal / embedding model filters (G-6, G-9) -----------

# HF-format vLLM models carry no mmproj (a llama.cpp projector concept) —
# vision capability is declared in the repo's config.json instead. Detection
# uses two signals, best-effort because the file may be absent:
#   1. config.json on the resolved model dir:
#      - "vision_config" present (LLaVA, Qwen-VL, MiniCPM-V, …), or
#      - architectures containing a known multimodal family
#   2. config.json unreadable/absent: model id/name prefix or substring
#      matching the common vision families.
_VISION_ARCH_FAMILIES = (
    "llava", "qwen2vl", "qwen2_5_vl", "qwen2-vl", "qwen2.5-vl", "qwen3vl",
    "qwen3-vl", "minicpmv", "minicpm-v", "internvl", "idefics", "pixtral",
    "paligemma", "gemma3", "mllama", "phi3_v", "phi3v", "phi4_multimodal",
    "molmo", "cogvlm", "smolvlm", "clip", "florence",
)

# vLLM embedding models (--task embed): the encoder families users actually
# name models after. "embed" substring stays the first signal (G-6's original
# heuristic); these catch BAAI/bge-*, nomic-ai/*, Alibaba/gte-*, e5, jina…
_EMBEDDING_ARCH_FAMILIES = (
    "bert", "xlm-roberta", "roberta", "gte", "bge", "e5", "jina",
    "arctic", "sentence-t5", "st5", "nomic", "minilm", "mpnet", "gte",
    "snowflake-arctic", "inf-retriever", "inf-retrieval", "bm25s",
)


def _hf_config(base_path):
    """Parsed config.json for a model's base_path, or None (missing,
    unreadable, or not HF format). Resolves store:// aliases; HF-format
    base_path is the model directory itself."""
    local = path_resolver.resolve(base_path)
    cfg_path = os.path.join(local, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _id_tokens(m):
    """Lowercased model_id + name + base_path tail for substring matching."""
    mid = (m.get("model_id") or "").lower()
    name = (m.get("name") or "").lower()
    tail = (m.get("base_path") or "").split("/")[-1].lower()
    return f"{mid} {name} {tail}"


def _family_match(text, families):
    """Substring match against a family tuple; '_' and '-' are treated as
    equivalent so 'Qwen2.5-VL' and 'qwen2_5_vl' both hit."""
    norm = text.replace("_", "-")
    return any(f in norm or f.replace("_", "-") in norm for f in families)


def filter_vision_models(models):
    """Vision-capable models for vLLM mode (G-9).

    HF-format vision models never carry mmproj, so capability comes from
    config.json (vision_config key or a known multimodal architecture).
    When config.json is absent/unreadable the id/name/path is matched
    against known vision families — best effort, documented as such.
    """
    out = []
    for m in models:
        text = _id_tokens(m)
        if _family_match(text, _VISION_ARCH_FAMILIES):
            out.append(m)
            continue
        config = _hf_config(m.get("base_path"))
        if config:
            if config.get("vision_config"):
                out.append(m)
                continue
            archs = " ".join(config.get("architectures") or []).lower()
            if _family_match(archs, _VISION_ARCH_FAMILIES):
                out.append(m)
    return out


def filter_embedding_models(models):
    """Embedding-capable models for vLLM mode (G-6).

    Keeps the original "embed" substring signal (it is still the most
    reliable) and extends it with the known embedding families — BGE /
    Nomic / GTE / E5 / Jina models name themselves without "embed".
    """
    out = []
    for m in models:
        text = _id_tokens(m)
        if "embed" in text:
            out.append(m)
            continue
        if _family_match(text, _EMBEDDING_ARCH_FAMILIES):
            out.append(m)
            continue
        config = _hf_config(m.get("base_path"))
        if config:
            archs = " ".join(config.get("architectures") or []).lower()
            if _family_match(archs, _EMBEDDING_ARCH_FAMILIES):
                out.append(m)
    return out


def filter_audio_models(models):
    """ASR models for vLLM mode.

    vLLM has no stable ASR serving path the farm can rely on (audio input
    via /v1/chat/completions input_audio is a llama-server capability; vLLM
    has no --task transcribe equivalent for these GGUF ASR models, and the
    language-forcing prompt is llama.cpp-specific). Keeping the same
    id/name heuristic as llama.cpp is harmless but the tab itself is
    hidden in vLLM mode via ui_hints.hidden_tabs — this filter only backs
    the endpoint contract.
    """
    out = []
    for m in models:
        flags = m.get("flags") or {}
        text = _id_tokens(m)
        if flags.get("audio") or "asr" in text or "whisper" in text:
            out.append(m)
    return out


def filter_models(models):
    """Servable models for vLLM mode (DEC-11, T4-G6).

    The Models table lists only models in the active engine's downloads
    vocabulary — vLLM serves 'hf' (safetensors repo dirs). A GGUF under
    vLLM is explicitly experimental (ARD §3.4), so it stays out of the
    table the same way /api/scan hides gguf entries in vLLM mode; the
    reverse gate in rpc_manager.validate_model still only warns, so an
    existing gguf config keeps validating — it just isn't listed.
    Format inference mirrors that gate: no explicit `format` -> 'hf'
    unless base_path ends in .gguf.
    """
    out = []
    for m in models:
        fmt = (m.get("format")
               or ("gguf" if str(m.get("base_path") or "")
                   .lower().endswith(".gguf") else "hf"))
        if fmt == "hf":
            out.append(m)
    return out


def asr_language_system_message(language):
    """None: the "language X<asr_text>" prefix is a llama-server prompt hack
    (verified ONLY against llama-server). Under vLLM it is meaningless or
    leaks into the transcript, so the vLLM path omits it entirely."""
    return None


# ---------------- frontend vocabulary (ARD §1, §4) --------------------------

def ui_manifest():
    """Field manifest for the frontend (AUDIT-T6 §1.5, DEC-02) — the
    complete per-engine inventory the editor/launch/downloads panes
    render from. Supersedes ui_hints(); the legacy {labels,
    hidden_panels, hidden_tabs, flag_presets} view is projected from
    this manifest by the adapters' UiHintsMixin shim. hidden_panels
    lists llama.cpp-only UI panels (tuner math and weight
    splitting are VRAM/KV-specific; rpc_peers is the wizard's multi-machine
    hardware hint; rpc_launch is the Launch tab's per-model RPC-cluster
    grid — under vLLM models start THROUGH llama-swap, and its direct
    /api/launch path refuses with 409, so the grid hides entirely). device
    chips stay visible — they map to CUDA_VISIBLE_DEVICES. hidden_tabs
    lists engine-exclusive test tabs: the Audio tab is llama.cpp-only (its
    ASR path is the llama-server input_audio + language-forcing prompt;
    vLLM cannot serve these GGUF ASR models), so it disappears entirely in
    vllm mode (G-8). model_editor.sections lists ONLY the vLLM-owned
    editor surface: the per-model `vllm:` card (the llama.cpp flags/
    tensor-placement panels are not part of this engine's vocabulary).
    launch is swap_only (models start THROUGH llama-swap; /api/launch
    409s) and downloads accept HF-hub formats only."""
    return {
        "labels": {
            "server": "vLLM server",
            "peers": "Tensor parallel",
            "parallelism": "Tensor parallel size",
            "probe": "HTTP /health",
        },
        "hidden_panels": ["tuner", "tensor_split", "rpc_peers", "rpc_launch"],
        "hidden_tabs": ["audio"],
        "model_editor": {
            "sections": [{"id": "vllm_section"}],
        },
        "flag_presets": {
            "ctx_size": {"flag": "--max-model-len",
                         "placeholder": "e.g. 32768"},
            "parallel_seqs": {"flag": "--max-num-seqs",
                              "placeholder": "e.g. 256"},
            "kv_cache": {"flag": "--kv-cache-dtype",
                         "choices": ["auto", "fp8"],
                         "note": "fp8 only — llama.cpp q4/q8 caches "
                                 "unsupported"},
            "offload": {"flag": "--cpu-offload-gb",
                        "placeholder": "e.g. 20"},
            "parallelism": {"flag": "--tensor-parallel-size",
                            "placeholder": "e.g. 2"},
            "parallelism_pp": {"flag": "--pipeline-parallel-size",
                               "placeholder": "e.g. 2",
                               "note": "pipeline stages — world size is "
                                       "TP×PP GPUs on one host"},
            "parallel_layout": {"flag": "--distributed-executor-backend",
                                "type": "choices",
                                "choices": ["pp", "tp"],
                                "default": "pp",
                                "note": "pp: pipeline across hosts "
                                        "(Ethernet-friendly, llama.cpp-RPC "
                                        "style); tp: tensor across hosts "
                                        "(Infiniband-class only)"},
            "quantization": {"flag": "--quantization",
                             "placeholder": "e.g. fp8, awq, gptq"},
            "attention_backend": {"flag": "--attention-backend",
                                  "choices": ["auto", "flash_attn",
                                              "triton", "flashinfer"],
                                  "default": "auto",
                                  "note": "vLLM's flash-attn toggle "
                                          "(llama.cpp flash-attn twin). "
                                          "flash_attn needs sm_80+ — "
                                          "Turing 2060-class cards fall "
                                          "back to triton (vLLM's own "
                                          "auto pick on this farm)"},
            "compute_dtype": {"flag": "--dtype",
                              "choices": ["auto", "half", "float16",
                                          "bfloat16", "float32"],
                              "note": "Turing/sm_75 GPUs (RTX 2060) cannot "
                                      "run bfloat16 — force half/float16"},
            "enforce_eager": {"flag": "--enforce-eager",
                              "type": "checkbox",
                              "default": True,
                              "note": "disable CUDA graphs (default ON "
                                      "multi-host: capture hangs across "
                                      "Ethernet TP; uncheck for Infiniband/"
                                      "fast interconnects)"},
            "auto_tool_choice": {"flag": "--enable-auto-tool-choice",
                                 "type": "checkbox",
                                 "default": True,
                                 "note": "allow tool_choice=\"auto\" "
                                         "(agentic clients need it) — "
                                         "adds --tool-call-parser"},
        },
        "launch": {"mode": "swap_only", "probe_label": "HTTP /health",
                   "multi_host": True},
        # live-validated starting values the editor pre-fills for a NEW
        # vllm model (mirror of vllm_new_model_defaults, consumed by
        # modelFormLoad) — keep the two in sync or the editor and the
        # save-time fill drift
        "vllm_new_model_defaults": vllm_new_model_defaults(),
        "downloads": {"formats": ["hf"], "browse_source": "hf_repos"},
    }


# ---------------- adapter (registered in engine_registry) ------------------

class VllmAdapter(UiHintsMixin):
    """The vLLM EngineAdapter. Returns the same shapes as LlamaCppAdapter
    (Protocol in engine_registry) so app.py dispatches identically.
    swap_entries delegates to swap_manager.launch_plan(), which is
    engine-aware at entry-emit time (its vllm branch calls
    swap_entry_parts above; ARD §3.5)."""

    name = ENGINE_VLLM
    # Adapter v2 ownership (DEC-01, AUDIT-T6 §1.3/§1.4): vLLM owns the
    # per-model `vllm:` section and the settings `vllm:` block.
    engine_id = ENGINE_VLLM
    config_section = "vllm"

    def settings_section(self, raw_settings):
        """Validate + default the settings `vllm:` block (moved verbatim
        from cfg.inject_engine_defaults — same errors, same strictness;
        the vocabulary constants stay in config_loader, B4's schema
        home). Mutates raw_settings in place and returns it.
        """
        vllm = raw_settings.get("vllm")
        if vllm is None:
            vllm = {}
        if not isinstance(vllm, dict):
            raise ValueError(
                "settings.yaml `vllm:` must be a mapping with keys "
                f"{sorted(cfg.VLLM_ALLOWED_KEYS)}")
        unknown = set(vllm) - cfg.VLLM_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"unknown vllm setting(s) {sorted(unknown)} — allowed: "
                f"{sorted(cfg.VLLM_ALLOWED_KEYS)}")
        delivery = vllm.get("delivery", cfg.VLLM_DELIVERY_DEFAULT)
        if delivery not in cfg.VALID_VLLM_DELIVERY:
            raise ValueError(
                f"invalid vllm delivery '{delivery}' — must be one of "
                f"{', '.join(cfg.VALID_VLLM_DELIVERY)}")
        gmu = vllm.get(
            "default_gpu_memory_utilization",
            cfg.VLLM_BLOCK_DEFAULTS["default_gpu_memory_utilization"])
        if not isinstance(gmu, (int, float)) or isinstance(gmu, bool) \
                or not 0 < float(gmu) <= 1:
            raise ValueError(
                "vllm.default_gpu_memory_utilization must be a number in "
                f"(0, 1], got {gmu!r}")
        block = dict(cfg.VLLM_BLOCK_DEFAULTS)
        block.update(vllm)
        block["delivery"] = delivery
        block["default_gpu_memory_utilization"] = float(gmu)
        # UX-1b: default_tool_call_parser must be a non-empty string —
        # the command builder falls back to it when a model sets no
        # per-model parser, so a bad value here breaks every launch.
        tcp = block.get("default_tool_call_parser")
        if not isinstance(tcp, str) or not tcp.strip():
            raise ValueError(
                "vllm.default_tool_call_parser must be a non-empty string, "
                f"got {tcp!r}")
        # multi-host Ray knobs: ints with sane bounds; ray_port must not
        # collide with the llama-swap listen port (ray gcs vs swap proxy).
        ray_port = block.get("ray_port", cfg.VLLM_BLOCK_DEFAULTS["ray_port"])
        if isinstance(ray_port, bool) or not isinstance(ray_port, (int, float)) \
                or not 1 <= int(ray_port) <= 65535:
            raise ValueError(
                f"vllm.ray_port must be a port number in [1, 65535], "
                f"got {ray_port!r}")
        block["ray_port"] = int(ray_port)
        try:
            swap_port = int(raw_settings.get("listen_port") or 8080)
        except (TypeError, ValueError):
            swap_port = 8080
        if block["ray_port"] == swap_port:
            raise ValueError(
                f"vllm.ray_port {block['ray_port']} collides with the "
                f"llama-swap listen port {swap_port} — pick another port")
        ray_timeout = block.get(
            "ray_init_timeout", cfg.VLLM_BLOCK_DEFAULTS["ray_init_timeout"])
        if isinstance(ray_timeout, bool) or not isinstance(ray_timeout, (int, float)) \
                or not int(ray_timeout) >= 1:
            raise ValueError(
                f"vllm.ray_init_timeout must be a positive number of "
                f"seconds, got {ray_timeout!r}")
        block["ray_init_timeout"] = int(ray_timeout)
        raw_settings["vllm"] = block
        return raw_settings

    def sanitize_model_section(self, raw, where="vllm"):
        """Normalize ONE model's `vllm:` block for storage; returns a fresh
        dict. DEC-04: the section round-trip logic MOVED HERE from
        config_loader (the adapter owns its per-model section — AUDIT-T6
        §1.3); config_loader delegates via engine_registry. Body is the
        verbatim former cfg.sanitize_vllm_section — same errors, same
        strictness; the vocabulary constants stay in config_loader (B4's
        schema home), so import cycles stay one-way (vllm_manager already
        imports config_loader)."""
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a mapping with keys "
                             f"{sorted(cfg.VLLM_MODEL_KEYS)}")
        unknown = sorted(set(raw) - cfg.VLLM_MODEL_KEYS)
        if unknown:
            raise ValueError(
                f"{where}: unknown vllm key(s) {unknown} — allowed: "
                f"{sorted(cfg.VLLM_MODEL_KEYS)}")
        out = {}
        for key in ("max_model_len", "max_num_seqs", "tensor_parallel_size",
                    "pipeline_parallel_size", "cpu_offload_gb"):
            val = raw.get(key)
            if val in (None, ""):
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{where}.{key} must be a number, got {val!r}")
            out[key] = int(val)
        # A2: gpu_memory_utilization is per-model now — a float in (0, 1]
        # (same domain as the settings-level default it shadows). Stored as
        # float so round-trips never coerce 0.5 -> 0.
        gmu = raw.get("gpu_memory_utilization")
        if gmu not in (None, ""):
            if isinstance(gmu, bool) or not isinstance(gmu, (int, float)) \
                    or not 0 < float(gmu) <= 1:
                raise ValueError(
                    f"{where}.gpu_memory_utilization must be a number in (0, 1], "
                    f"got {gmu!r}")
            out["gpu_memory_utilization"] = float(gmu)
        kv = raw.get("kv_cache_dtype")
        if kv not in (None, ""):
            if kv not in cfg.VALID_KV_CACHE_DTYPES:
                raise ValueError(
                    f"{where}.kv_cache_dtype '{kv}' is unsupported — "
                    f"allowed: {list(cfg.VALID_KV_CACHE_DTYPES)} "
                    "(fp8 needs SM89+ GPUs)")
            if kv != "auto":  # auto is the default; don't store a no-op
                out["kv_cache_dtype"] = kv
        for key in ("quantization", "tokenizer"):
            val = raw.get(key)
            if val in (None, ""):
                continue
            if not isinstance(val, str):
                raise ValueError(f"{where}.{key} must be a string")
            out[key] = val
        dtype = raw.get("dtype")
        if dtype not in (None, ""):
            if dtype not in VALID_COMPUTE_DTYPES:
                raise ValueError(
                    f"{where}.dtype '{dtype}' is unsupported — allowed: "
                    f"{sorted(VALID_COMPUTE_DTYPES)}")
            if dtype != "auto":  # auto is the default; don't store a no-op
                out["dtype"] = dtype
        # enforce_eager: explicit bool only (absent stays absent — the
        # multi-host launch then applies the safe default; single-host
        # vLLM keeps its own default of graphs enabled)
        ee = raw.get("enforce_eager")
        if ee is not None:
            if not isinstance(ee, bool):
                raise ValueError(
                    f"{where}.enforce_eager must be a boolean, got {ee!r}")
            out["enforce_eager"] = ee
        # tool-calling: enable flag is bool (absent = default ON), the
        # parser is a free string so new vLLM parsers need no code change
        tc = raw.get("enable_auto_tool_choice")
        if tc is not None:
            if not isinstance(tc, bool):
                raise ValueError(
                    f"{where}.enable_auto_tool_choice must be a boolean, "
                    f"got {tc!r}")
            out["enable_auto_tool_choice"] = tc
        tcp = raw.get("tool_call_parser")
        if tcp not in (None, ""):
            if not isinstance(tcp, str):
                raise ValueError(
                    f"{where}.tool_call_parser must be a string, got {tcp!r}")
            if tcp != "auto":  # auto is the default; don't store a no-op
                out["tool_call_parser"] = tcp
        # ray_env: mapping env-var -> str|int|bool; false = drop default.
        # Keys are validated to env-var shape — they land in generated
        # `export KEY=VALUE` lines, so a crafted key is an injection
        # surface (same rule as _SAFE_ID: strict allowlist).
        re_env = raw.get("ray_env")
        if re_env is not None:
            if not isinstance(re_env, dict):
                raise ValueError(
                    f"{where}.ray_env must be a mapping of env-var -> "
                    "value (use false to drop a default)")
            clean = {}
            for k, val in re_env.items():
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(k)):
                    raise ValueError(
                        f"{where}.ray_env key {k!r} is not a valid "
                        "environment variable name")
                if val is False or val is True:
                    clean[str(k)] = val
                elif isinstance(val, (int, str)):
                    clean[str(k)] = str(val) if isinstance(val, str) else val
                else:
                    raise ValueError(
                        f"{where}.ray_env[{k!r}] must be a string, int or "
                        f"boolean, got {val!r}")
            if clean:
                out["ray_env"] = clean
        deb = raw.get("distributed_executor_backend")
        if deb not in (None, ""):
            if deb not in ("ray", "mp"):
                raise ValueError(
                    f"{where}.distributed_executor_backend '{deb}' is "
                    "invalid — must be 'ray' or 'mp' (multi-host clusters "
                    "that form a multi-node cluster)")
            out["distributed_executor_backend"] = deb
        layout = raw.get("parallel_layout")
        if layout not in (None, ""):
            if layout not in ("pp", "tp"):
                raise ValueError(
                    f"{where}.parallel_layout '{layout}' is invalid — "
                    "'pp' (pipeline across hosts) or 'tp' (tensor "
                    "across hosts)")
            out["parallel_layout"] = layout
        # attention backend: vLLM 0.18+ --attention-backend (llama.cpp
        # flash-attn twin). auto (default) = vLLM's own per-device pick;
        # friendly aliases map to AttentionBackendEnum names (see the
        # flag builder's alias_to_enum). flash_attn on a mixed
        # sm_75/sm_86 farm would crash the Turing stage — storage keeps
        # the explicit user choice, the flag builder maps the name.
        attn = raw.get("attention_backend")
        if attn not in (None, ""):
            if not isinstance(attn, str) or attn not in (
                    "auto", "flash_attn", "triton", "flashinfer",
                    "rocm_attn", "flex_attention", "torch_sdpa"):
                raise ValueError(
                    f"{where}.attention_backend '{attn}' is unsupported "
                    "— allowed: auto, flash_attn, triton, flashinfer")
            if attn != "auto":  # auto is the default; don't store a no-op
                out["attention_backend"] = attn
        extra = raw.get("extra_args")
        if extra is not None:
            if not isinstance(extra, list) or any(
                    not isinstance(a, str) or not a.strip() for a in extra):
                raise ValueError(
                    f"{where}.extra_args must be a list of non-empty strings")
            out["extra_args"] = [a.strip() for a in extra if a.strip()]
        return out

    def apply_model_section(self, data, where="vllm"):
        """Validate + store the `vllm:` block of a model config dict IN
        PLACE. Verbatim former cfg.apply_vllm_section; see
        sanitize_model_section for the DEC-04 ownership move."""
        if not isinstance(data, dict):
            raise ValueError(f"{where}: model config must be a mapping")
        section = self.sanitize_model_section(data.get("vllm"), where=where)
        if section:
            data["vllm"] = section
        else:
            data.pop("vllm", None)
        return data

    def build_command(self, model_cfg, servers, master_name=None):
        """Multi-host-capable build: validate_hosts([master]+peers) ->
        TP derivation -> 1 host = build_vllm_command (unchanged) ->
        2+ hosts = build_vllm_cluster_command (Ray cluster script)."""
        master, peers = rpc_manager._split_servers(
            list(servers or []), master_name)
        errors = validate_hosts([master] + peers)
        if errors:
            raise ValueError("; ".join(errors))
        if not peers:
            return build_vllm_command(model_cfg, master)
        return build_vllm_cluster_command(model_cfg, master, peers)

    def probe(self, machine, port, is_local):
        return probe_health(machine, port, is_local)

    def swap_entries(self, config_path=None, host_name=None):
        import swap_manager
        return swap_manager.launch_plan(config_path, host_name)

    def validate_model_config(self, model_cfg):
        return validate_model_config(model_cfg)

    def validate_hosts(self, machines):
        return validate_hosts(machines)

    def asr_language_system_message(self, language):
        return asr_language_system_message(language)

    def filter_audio_models(self, models):
        return filter_audio_models(models)

    def filter_vision_models(self, models):
        return filter_vision_models(models)

    def filter_embedding_models(self, models):
        return filter_embedding_models(models)

    def filter_models(self, models):
        return filter_models(models)

    # -- acquisition (DEC-07: vLLM vocabulary = HF-hub / safetensors) --

    def acquire(self, link, store, repo=False, all_shards=True):
        """vLLM downloads HF-hub model repos (safetensors). A bare GGUF
        file link is the llama.cpp vocabulary — refused here (the
        adapter-side backstop; POST /api/downloads gates the same way
        BEFORE dispatch). repo=True: hf://owner/name whole repo; else a
        single safetensors file (or its shard set, all_shards)."""
        import hf_downloader
        if not repo and str(link or "").lower().endswith(".gguf"):
            raise ValueError(
                "GGUF file downloads are not part of the vLLM engine — "
                "vLLM serves HF model repos (expected hf://owner/name "
                "or a safetensors file)")
        if repo:
            repo_id = (link or "").strip()
            for prefix in ("hf://", "hf:/"):
                if repo_id.startswith(prefix):
                    repo_id = repo_id[len(prefix):]
            return hf_downloader.start_repo(repo_id, store)
        if all_shards:
            return hf_downloader.start_shards(link, store)
        return hf_downloader.start(link, store)

    async def browse(self, query="", sort="downloads", limit=20, page=0,
                     quant="", pipeline_tag="", author="",
                     recommended=False):
        """Safetensors/HF-repo search (vLLM vocabulary) — the same
        hf_browse.search_safetensors the frontend's vLLM pane uses, now
        reached through the adapter so app.py holds no engine names."""
        import hf_browse
        return await hf_browse.search_safetensors(
            query=query, sort=sort, limit=limit, page=page,
            pipeline_tag=pipeline_tag, author=author,
            recommended=recommended)

    def validate_downloaded(self, path):
        """vLLM serves a model DIRECTORY: config.json + at least one
        *.safetensors (the same detection model_parser uses)."""
        import path_resolver
        p = str(path or "")
        local = path_resolver.resolve(p)
        if not os.path.isdir(local):
            return [{"level": "error",
                     "msg": f"vLLM engine serves HF model directories — "
                            f"'{p}' is not a directory"}]
        files = set(os.listdir(local))
        if "config.json" not in files:
            return [{"level": "error",
                     "msg": f"'{p}' is missing config.json — not an HF "
                            f"model directory"}]
        if not any(f.endswith(".safetensors") for f in files):
            return [{"level": "error",
                     "msg": f"'{p}' has no *.safetensors weights"}]
        return []

    # -- launch / stop / status (DEC-06, closes T2-G6/G7) --

    def capable(self, capability):
        caps = {"rpc_cluster": False, "swap_stop": True}
        return caps.get(capability, False)

    def launch_plan(self, config_path=None, host_name=None):
        """swap_manager.launch_plan() is already engine-aware at entry-emit
        time (its vllm branch calls swap_entry_parts, ARD §3.5) — the
        delegation below IS the vllm launch plan."""
        import swap_manager
        return swap_manager.launch_plan(config_path, host_name)

    def start_model(self, model_id, servers, port, variant=None,
                    master_name=None, mode="rpc"):
        """vLLM has no live single-model RPC launch — serving goes
        through llama-swap (/api/swap/*). Refusal mirrors the pre-DEC-06
        /api/launch 409 detail; app.py turns it into HTTPException(409)."""
        raise ValueError(
            "vLLM mode does not use live RPC-cluster launch — "
            "serve models through llama-swap instead "
            "(Launch All / /api/swap/* endpoints)")

    def stop_model(self, model_id, servers, variant=None, master_name=None,
                   launch_port=None):
        """The vLLM stop path IS llama-swap's model unload: the entry's
        cmdStop (docker stop / pkill, build_cmd_stop) runs when the swap
        host proxies POST /api/models/unload/{model_id}. Returns the
        proxy payload shape app.py's _proxy_swap_unload produces.
        launch_port accepted for the shared adapter signature (unused —
        llama-swap owns the port knowledge here)."""
        return swap_host_unload(model_id)

    def active_model(self, machine):
        """GET /api/launch/active probe (replaces the hardcoded :8080
        socket check): HTTP /health on the swap listen port via the
        adapter's own probe (vllm_manager.probe_health)."""
        import swap_manager
        try:
            port = swap_manager.load_settings().get("listen_port") or 8080
        except Exception:
            port = 8080
        res = probe_health(machine, port, False)
        return {"name": (machine or {}).get("name"),
                "serving": bool(res.get("running"))}

    def ui_manifest(self):
        return ui_manifest()