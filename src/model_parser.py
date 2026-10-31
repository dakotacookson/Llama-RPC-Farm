"""Scans model directories for GGUF files and HF/safetensors model dirs.

GGUF: a sharded file set like `model-00001-of-00003.gguf` or the user's
`part 01 of 03` naming is presented as ONE model entry with its shards
listed, ready to pass to llama-server --model.

HF/safetensors (vLLM's native format, ARD §3.4): a directory containing
config.json + at least one *.safetensors is ONE model — `vllm serve
<directory>`. Its -000XX-of-000YY safetensors shards collapse into that
single entry via the same shard pattern used for GGUF splits.

Every model entry carries `format`: "gguf" | "hf", so engine-aware
surfaces (model wizard, vLLM validation) can treat GGUF-on-vLLM as
experimental without re-statting the path.
"""
import os
import re

# shard patterns: llama.cpp split convention + human "part N of N"
SHARD_PATTERNS = [
    re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE),   # -00001-of-00003
    re.compile(r"-(\d{5})-of-(\d{5})\.safetensors$", re.IGNORECASE),
    re.compile(r"\s*part\s*0*(\d+)\s*of\s*0*(\d+)\s*\.gguf$", re.IGNORECASE),
    re.compile(r"\.part0*(\d+)of0*(\d+)\.gguf$", re.IGNORECASE),
]
MMPROJ_PREFIX = "mmproj-"
# HF repo root marker: present in every safetensors model directory
HF_CONFIG_MARKER = "config.json"
SAFETENSORS_EXT = ".safetensors"


def _shard_match(fname):
    for pattern in SHARD_PATTERNS:
        m = pattern.search(fname)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _dir_stats(files):
    total = sum(files[f]["size_bytes"] for f in files)
    return {
        "files": files,
        "shard_count": len(files),
        "total_size_bytes": total,
        "is_sharded": len(files) > 1,
    }


def _is_hf_model_dir(path):
    """True when `path` holds an HF/safetensors model: config.json plus at
    least one *.safetensors. A safetensors file alone is not identifiable
    (vLLM reads the architecture from config.json)."""
    try:
        if not os.path.isfile(os.path.join(path, HF_CONFIG_MARKER)):
            return False
        for name in os.listdir(path):
            if name.lower().endswith(SAFETENSORS_EXT):
                return True
    except OSError:
        return False
    return False


def _hf_entry(dirpath):
    """ONE model entry for an HF/safetensors directory.

    Shard collapse reuses the -000XX-of-000YY pattern: the dominant shard
    group (same base name) is the model's weights; stray safetensors files
    (e.g. adapter_model.safetensors) are not weight shards and are left out
    of the entry. base_path is the directory itself — what `vllm serve`
    takes.
    """
    groups = {}
    for name in sorted(os.listdir(dirpath)):
        if not name.lower().endswith(SAFETENSORS_EXT):
            continue
        base, _shard = _split_base_and_shard(name)
        groups.setdefault(base, []).append(name)
    # dominant group = the real weight shards (ties/strays: largest wins)
    shard_names = max(groups.values(), key=len) if groups else []
    files = {}
    for name in shard_names:
        full = os.path.join(dirpath, name)
        try:
            size = os.path.getsize(full)
        except OSError:
            size = 0
        files[name] = {"size_bytes": size, "path": full}
    return {
        "name": os.path.basename(os.path.normpath(dirpath)) or dirpath,
        "base_path": dirpath,
        "format": "hf",
        **_dir_stats(files),
    }


def scan_directory(path, recursive=False):
    """Scan a directory for GGUF models AND HF/safetensors model dirs.

    Returns (models, projectors). Sharded models are collapsed into a
    single entry. mmproj/mtp (multimodal projector) files are listed
    separately, not as models. HF model directories become one entry each
    with format="hf"; recursive scans do not descend into them (a model
    dir is one unit, its files are not separate models).
    """
    if not os.path.isdir(path):
        return [], []

    ggufs = []
    hf_dirs = []
    if recursive:
        for root, dirs, names in os.walk(path):
            if _is_hf_model_dir(root):
                hf_dirs.append(root)
                dirs[:] = []  # one model per dir; its contents are not models
                continue
            for name in names:
                if not name.lower().endswith(".gguf"):
                    continue
                full = os.path.join(root, name)
                ggufs.append((root, name, os.path.getsize(full)))
    else:
        entries = os.listdir(path)
        for name in entries:
            full = os.path.join(path, name)
            if os.path.isfile(full) and name.lower().endswith(".gguf"):
                ggufs.append((path, name, os.path.getsize(full)))
        # HF models live one level down in per-model subdirectories; the
        # scanned path itself may already be a model directory.
        for entry in [path] + [os.path.join(path, n) for n in entries]:
            if os.path.isdir(entry) and _is_hf_model_dir(entry):
                hf_dirs.append(entry)

    models = {}      # (root, base_name) -> {filename: {size, path}}
    projectors = []
    for root, name, size in ggufs:
        if name.startswith(MMPROJ_PREFIX) or name.startswith("mtp-"):
            projectors.append({
                "name": name, "base_path": os.path.join(root, name),
                "size_bytes": size, "shard_count": 1, "is_sharded": False,
            })
            continue
        base, shard = _split_base_and_shard(name)
        key = (root, base)
        models.setdefault(key, {})
        models[key][name] = {"size_bytes": size, "path": os.path.join(root, name)}

    results = []
    for (root, base), files in models.items():
        first = os.path.join(root, sorted(files)[0])
        entry = {
            "name": base,
            "base_path": first,
            "format": "gguf",
            **_dir_stats(files),
        }
        results.append(entry)
    results.extend(_hf_entry(d) for d in hf_dirs)
    results.sort(key=lambda m: m["name"].lower())
    projectors.sort(key=lambda p: p["name"].lower())
    return results, projectors


def _split_base_and_shard(fname):
    """Return (base_name, shard_index or None). Strips the shard suffix."""
    for pattern in SHARD_PATTERNS:
        m = pattern.search(fname)
        if m:
            base = fname[: m.start()]
            base = base.rstrip("-")
            return base, int(m.group(1))
    return os.path.splitext(fname)[0], None