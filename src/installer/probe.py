# probe.py — PURE decision logic: probe(state) -> Plan. (design §4/§5/§6, §10)
#
# ZERO subprocess/os.system/direct-disk calls live here. Everything arrives via
# the injected state (SystemInfo + FS + RunCmd results). This module answers:
# "given what we observed, what SHOULD the installer do?" — one step per tool.
#
# Backend parity rule (operator clarification): BOTH engines (llama.cpp AND
# vLLM) must be planned on every backend — CUDA, ROCm, MLX/vLLM-Metal, Vulkan,
# CPU. No plan may silently omit one engine on a backend where it is viable.
# Where a backend genuinely cannot host an engine (e.g. native Windows + vLLM),
# the plan emits an explicit PlanNote (refusal + documented path), never silence.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .env import FS, RunCmd, SystemInfo


# ---------------------------------------------------------------------------
# Typed plan steps
# ---------------------------------------------------------------------------

@dataclass
class SkipTool:
    name: str
    reason: str                      # 'present-current' | 'backend-refused' | 'not-offered'
    found_path: Optional[str] = None
    version: Optional[str] = None


@dataclass
class InstallTool:
    name: str
    source: str                      # asset name / url / pip spec
    version: str
    backend: Optional[str] = None    # e.g. 'cuda-13.3', 'rocm-10.0', 'vllm-metal'
    dest: Optional[str] = None
    extra: Dict = field(default_factory=dict)   # e.g. {'index_url': rocm index}


@dataclass
class RepairTool(InstallTool):
    """InstallTool re-run over the same dest because --version failed (§5)."""
    broken_version: Optional[str] = None


@dataclass
class WriteSettings:
    key: str
    value: str
    file: str                        # 'config/settings.yaml' | 'config/swap.yaml'
    merge_reason: str                # 'empty' | 'stale' | 'force'


@dataclass
class ManifestUpdate:
    tool: str
    entry: Dict


@dataclass
class PlanNote:
    """Informational block: capability statement, refusal, manual action."""
    kind: str                        # 'refused' | 'manual' | 'info' | 'warn'
    tool: Optional[str]
    message: str


Plan = List[object]


# ---------------------------------------------------------------------------
# Pinned requirements (mirrors design §2/§5; single source here)
# ---------------------------------------------------------------------------

# Engine backends per OS/vendor: the parity matrix. vLLM entries name how.
VLLM_BACKEND = {
    ("linux", "nvidia"):  {"offered": True,  "how": "pypi", "note": None},
    ("linux", "amd"):     {"offered": True,  "how": "rocm-index", "note": "py3.12-only wheels; torch.version.hip guard"},
    ("linux", "none"):    {"offered": True,  "how": "cpu", "note": "heavyweight on pure CPU; opt-in"},
    ("linux", "intel"):   {"offered": False, "how": None, "note": "vLLM XPU farm-unvalidated"},
    ("windows", "nvidia"):{"offered": False, "how": None, "note": "no native Windows; WSL2 is the documented path"},
    ("macos", "apple"):   {"offered": True,  "how": "vllm-metal", "note": "community vLLM-Metal plugin; explicit opt-in --with-vllm-metal"},
}

LLCPP_BACKEND = {
    ("linux", "nvidia"):  "cuda-13.3",
    ("linux", "amd"):     "rocm-10.0",
    ("linux", "none"):    "cpu",
    ("linux", "intel"):   "sycl",
    ("windows", "nvidia"):"cuda-13.3",
    ("windows", "none"):  "cpu",
    ("macos", "apple"):   "metal",
}

# Per-backend engine defaults (§10, operator clarification #3): flags that are
# CUDA-only vs universal, and the device-visibility env each backend uses.
ENGINE_DEFAULTS = {
    "llama.cpp": {
        "universal_flags": ["--n-gpu-layers", "--ctx-size"],
        "cuda_only_flags": [],
        "device_env": {"linux": {"nvidia": "CUDA_VISIBLE_DEVICES",
                                 "amd": "HIP_VISIBLE_DEVICES",
                                 "intel": "SYCL_DEVICE_FILTER"},
                       "macos": {"apple": None},     # Metal: no device env
                       "windows": {"nvidia": "CUDA_VISIBLE_DEVICES"}},
    },
    "vllm": {
        "universal_flags": ["--gpu-memory-utilization", "--max-model-len"],
        "cuda_only_flags": ["--enable-auto-tool-choice", "--tool-call-parser"],
        "device_env": {"linux": {"nvidia": "CUDA_VISIBLE_DEVICES",
                                 "amd": "HIP_VISIBLE_DEVICES"},
                       "macos": {"apple": None},     # vLLM-Metal: no util concept
                       "windows": {}},
    },
}


def tool_defaults(engine: str, info: SystemInfo) -> Dict:
    """Engine-adapter defaults for this backend (env var name or None)."""
    d = ENGINE_DEFAULTS.get(engine, {})
    env_map = d.get("device_env", {}).get(info.os, {})
    return {"device_env": env_map.get(info.gpu_vendor),
            "cuda_only_flags": d.get("cuda_only_flags", [])}


# ---------------------------------------------------------------------------
# Settings-prefill merge rules (§6)
# ---------------------------------------------------------------------------

SETTINGS_KEYS = {
    "llama.cpp": ("config/settings.yaml", "llama_bin"),
    "vllm":      ("config/settings.yaml", "vllm.venv_bin"),
    "llama-swap": ("config/swap.yaml", "binary"),
}


def _settings_merge_step(tool, found_path, current_value, fs, force=False):
    """One WriteSettings step or None, per §6 rules a/b/c."""
    if tool not in SETTINGS_KEYS or not found_path:
        return None
    path, key = SETTINGS_KEYS[tool]
    resolved = fs.expanduser(found_path)
    if force:
        reason = "force"
    elif not current_value:
        reason = "empty"
    elif not (fs.exists(current_value) if key != "vllm.venv_bin" else fs.exists(current_value)):
        reason = "stale"
    else:
        return None       # user-set value pointing at a valid binary — kept
    return WriteSettings(key=key, value=resolved, file=path, merge_reason=reason)


def _read_setting(fs, path, key):
    """Read a (possibly dotted) key from settings.yaml via pyyaml; None if absent."""
    if not fs.isfile(path):
        return None
    try:
        import yaml
        data = yaml.safe_load(fs.read_file(path)) or {}
    except Exception:
        return None
    cur = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, str) else None


# ---------------------------------------------------------------------------
# Per-tool presence probes (§5) — pure, against FS + RunCmd
# ---------------------------------------------------------------------------

def _version_from_output(out: str) -> Optional[str]:
    m = re.search(r"b\d{4,}", out)
    if m:
        return m.group(0)
    m = re.search(r"(\d+\.\d+\.\d+)", out)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d+)\b", out)     # llama-swap tag NUMBER style ("255")
    return m.group(1) if m else None


def probe_llama_cpp(fs, run, role="main") -> Tuple[Optional[str], Optional[str]]:
    """(found_path, version) or (None, None). Prebuilt dest or source build."""
    candidates = ["~/llama.cpp/bin/llama-server",
                  "~/llama.cpp/build/bin/llama-server"]
    if role == "client":
        candidates = [c.replace("llama-server", "ggml-rpc-server")
                      for c in candidates]
    for c in candidates:
        if fs.isfile(fs.expanduser(c)):
            r = fs.file_exists_and_runs(c)
            if r.ok:
                return c, _version_from_output(r.stdout + " " + r.stderr)
            return c, None          # exists but broken -> repair
    return None, None


def probe_swap(fs, run) -> Tuple[Optional[str], Optional[str]]:
    for c in ["~/.local/bin/llama-swap", "~/llama-swap/llama-swap.exe"]:
        if fs.isfile(fs.expanduser(c)):
            r = fs.file_exists_and_runs(c)
            if r.ok:
                return c, _version_from_output(r.stdout + " " + r.stderr)
            return c, None
    return None, None


def probe_vllm(fs, run) -> Tuple[Optional[str], Optional[str]]:
    for c in ["~/vllm-env/bin/vllm", "~/.local/bin/vllm"]:
        if fs.isfile(fs.expanduser(c)):
            r = fs.file_exists_and_runs(c)
            if r.ok:
                return c, _version_from_output(r.stdout + " " + r.stderr)
            return c, None
    return None, None


# ---------------------------------------------------------------------------
# The one entrypoint
# ---------------------------------------------------------------------------

def probe(info: SystemInfo, fs: FS, run: RunCmd,
          role: str = "main", pinned: Optional[Dict] = None,
          with_vllm: bool = True, with_vllm_metal: bool = False,
          force_settings: bool = False,
          manifest: Optional[Dict] = None) -> Plan:
    """Pure: decide skip/install/repair per tool + settings prefill steps."""
    pinned = pinned or {}
    manifest = manifest or {}
    plan: Plan = []

    def version_cmp(have, want):
        """Segmented compare: 'b10934' vs 'b99999' compares numeric bodies;
        dotted semver compares per-segment; mixed shapes fall back to ==."""
        h, w = str(have), str(want)
        hm, wm = re.fullmatch(r"b?(\d{4,})", h), re.fullmatch(r"b?(\d{4,})", w)
        if hm and wm:
            return int(hm.group(1)) >= int(wm.group(1))
        hs, ws = h.lstrip("b").split("."), w.lstrip("b").split(".")
        if len(hs) > 1 and len(ws) > 1:
            try:
                hp = [int(x) for x in hs]
                wp = [int(x) for x in ws]
                return hp >= wp
            except ValueError:
                pass
        return h == w

    # --- llama.cpp --------------------------------------------------------
    need_lcpp = pinned.get("llama.cpp", {}).get("version", "b10000")
    found, ver = probe_llama_cpp(fs, run, role)
    if found and ver and version_cmp(ver, need_lcpp):
        plan.append(SkipTool("llama.cpp", "present-current", found, ver))
    elif found and not ver:
        plan.append(PlanNote("warn", "llama.cpp",
                             f"{found} exists but --version fails — repair"))
        plan.append(RepairTool("llama.cpp", "prebuilt", need_lcpp,
                               backend=LLCPP_BACKEND.get((info.os, info.gpu_vendor), "cpu"),
                               dest=found, broken_version=None))
    else:
        plan.append(InstallTool("llama.cpp", "prebuilt", need_lcpp,
                                backend=LLCPP_BACKEND.get((info.os, info.gpu_vendor), "cpu"),
                                dest="~/llama.cpp/bin"))
    plan.append(ManifestUpdate("llama.cpp",
                {"version": ver or need_lcpp, "method": "prebuilt"}))

    # --- llama-swap (main only) -------------------------------------------
    if role == "main":
        need_swap = pinned.get("llama-swap", {}).get("version", "255")
        found, ver = probe_swap(fs, run)
        if found and ver and version_cmp(ver, need_swap):
            plan.append(SkipTool("llama-swap", "present-current", found, ver))
        elif found and not ver:
            plan.append(RepairTool("llama-swap", "prebuilt", need_swap,
                                   dest=found, broken_version=ver))
        else:
            plan.append(InstallTool("llama-swap", "prebuilt", need_swap,
                                    dest="~/.local/bin"))
        plan.append(ManifestUpdate("llama-swap", {"version": ver or need_swap}))

    # --- vLLM (parity: planned on every backend where viable) --------------
    v = VLLM_BACKEND.get((info.os, info.gpu_vendor))
    if not with_vllm:
        pass
    elif v and v["offered"]:
        how = v["how"]
        if how == "vllm-metal" and not with_vllm_metal:
            plan.append(SkipTool("vllm", "backend-refused-optin",
                                 note_msg := v["note"]))
        else:
            need_vllm = pinned.get("vllm", {}).get("version", "0.30.0")
            found, ver = probe_vllm(fs, run)
            if found and ver and version_cmp(ver, need_vllm):
                plan.append(SkipTool("vllm", "present-current", found, ver))
            elif found and not ver:
                plan.append(RepairTool("vllm", f"venv:{how}", need_vllm,
                                       dest=found, broken_version=ver))
            else:
                extra = {}
                if how == "rocm-index":
                    extra = {"index_url": "https://wheels.vllm.ai/rocm/",
                             "python": "3.12"}
                plan.append(InstallTool("vllm", f"venv:{how}", need_vllm,
                                        backend=how, dest="~/vllm-env",
                                        extra=extra))
            plan.append(ManifestUpdate("vllm", {"version": ver or need_vllm,
                                                "index": how}))
    elif v and not v["offered"]:
        plan.append(PlanNote("refused", "vllm", v["note"]))
    else:
        plan.append(PlanNote("refused", "vllm",
                             "backend not in parity matrix — treat as manual"))

    # --- settings prefill (§6): discovered path wins; user values kept -----
    if role == "main":
        for step in list(plan):
            found_path = getattr(step, "found_path", None)
            if found_path is None and isinstance(step, InstallTool):
                found_path = step.dest
            if found_path is None:
                continue
            tool = getattr(step, "name", None)
            if tool not in SETTINGS_KEYS:
                continue
            path, key = SETTINGS_KEYS[tool]
            cur = _read_setting(fs, path, key)
            s = _settings_merge_step(tool, found_path, cur, fs, force_settings)
            if s:
                plan.append(s)

    return plan