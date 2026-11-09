"""Background Hugging Face downloads straight into a store location.

A download now lands in a PER-MODEL folder under the store:

    store://llm-models/<model-stem>/
        <weights>.gguf              # selected quant + all its shards
        mmproj-*.gguf               # ONE vision projector (F16 preferred)
        mtp-*.gguf / *draft*.gguf   # ONE MTP/draft head, matching quant only
        config.json / tokenizer*    # metadata for pure-Python use
        LICENSE* / README*          # license + model card
        farm-bundle.json            # manifest: source repo/revision/files

The selected file's split shards are fetched together (llama.cpp split
convention). Companion GGUFs are added conservatively: a single mmproj
and a single MTP/draft head whose quant matches the selection (repos
often ship one per quant plus "shared" variants; grabbing all of them
silently multiplies the download). HF metadata files needed to load the
model from Python (config/tokenizer/generation config, remote code,
license, README) are downloaded too; if the selected repo is a GGUF-only
quant repo that omits them, the HF `base_model` is consulted and just
its metadata (never its weights) lands under `hf-base/`.

`hf_hub_download` streams into a `.incomplete` file inside the bundle
folder and moves it into place, so bytes hit the store exactly once --
the ~/.cache/huggingface blob store is never touched.

Existing flat files are left alone: bundling applies to NEW downloads.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime

from huggingface_hub import get_hf_file_metadata, hf_hub_url

import path_resolver

# llama.cpp split convention: model-00001-of-00002.gguf (width varies)
SPLIT_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$", re.IGNORECASE)

MANIFEST_NAME = "farm-bundle.json"
BASE_METADATA_DIR = "hf-base"

# companion GGUF families we always pull with a model when present
COMPANION_TOKENS = ("mmproj", "mtp", "draft", "dflash", "eagle")

# weight/container extensions that must never be treated as metadata
_WEIGHT_EXTS = (
    ".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5",
    ".msgpack", ".onnx", ".tflite", ".ot", ".pkl", ".npz", ".npy",
)

_META_PREFIXES = (
    "config", "generation_config", "preprocessor_config",
    "processor_config", "video_preprocessor_config", "tokenizer", "vocab",
    "merges", "special_tokens", "added_tokens", "chat_template",
    "requirements", "configuration_", "modeling_", "tokenization_",
    "processing_", "image_processing_", "feature_extraction_",
    "license", "licence", "copying", "notice", "readme",
)
_META_EXTS = (
    ".json", ".txt", ".model", ".jinja", ".jinja2", ".py", ".md",
    ".yaml", ".yml", ".tiktoken", ".vocab", ".merges",
)

JOBS = {}
BUNDLES = {}
_PROCS = {}
_REPO_FILES_CACHE = {}
_REPO_INFO_CACHE = {}
_CACHE_TTL = 300

_CHILD = """\
import sys
from huggingface_hub import hf_hub_download
repo_id, filename, local_dir = sys.argv[1:4]
try:
    path = hf_hub_download(repo_id=repo_id, filename=filename,
                           local_dir=local_dir)
    print("OK:" + path, flush=True)
except Exception as e:
    print("ERR:" + str(e), flush=True)
    sys.exit(1)
"""

def parse_link(link):
    """hf://owner/repo/path/file.gguf -> (repo_id, filename).

    Tolerates pasted CLI forms: "hf download hf://owner/repo/file",
    "hf:/owner/..." (missing slash), and bare "owner/repo/file".
    """
    text = (link or "").strip()
    # strip a leading "hf download" / "huggingface-cli download" command
    for cmd in ("hf download", "huggingface-cli download"):
        if text.lower().startswith(cmd):
            text = text[len(cmd):].strip()
    if text.startswith("hf://"):
        text = text[len("hf://"):]
    elif text.startswith("hf:/"):
        text = text[len("hf:/"):]
    parts = [p for p in text.split("/") if p]
    # a trailing "--type model" style flag from a copied command
    parts = [p for p in parts if not p.startswith("--")]
    if len(parts) < 3:
        raise ValueError(
            "expected hf://owner/repo/filename "
            "(e.g. hf://orcarouter/Repo/file.gguf)")
    filename = "/".join(parts[2:]).split()[0]  # keep subfolders, drop CLI flags
    return "/".join(parts[:2]), filename


def list_stores():
    return sorted(path_resolver._load_stores().keys())


def _token():
    try:
        from hf_browse import _token as _hf_token
        return _hf_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or \
            os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _safe_rel(filename):
    """Repo-relative path safe to join under a bundle dir (no traversal)."""
    rel = str(filename or "").replace("\\", "/").strip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"invalid repo filename: {filename!r}")
    return "/".join(parts)


def _safe_folder(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or ""))
    return (name.strip(".-_") or "model")[:120]


def _bundle_name(filename):
    """Folder name for a model file: split stem, else basename stem."""
    base = os.path.basename(str(filename or ""))
    info = split_info(base)
    stem = info[0] if info else os.path.splitext(base)[0]
    return _safe_folder(stem)


def _expected_size(repo_id, filename):
    try:
        url = hf_hub_url(repo_id, filename)
        return get_hf_file_metadata(url, token=_token()).size
    except Exception:
        return None


def _incomplete_bytes(local_dir, filename=None):
    """Bytes downloaded so far, from the .incomplete files hf writes
    inside local_dir's own .cache folder (same volume, same disk).
    Scoped to this job's filename so concurrent jobs each report only
    their own progress."""
    cache = os.path.join(local_dir, ".cache", "huggingface", "download")
    # .incomplete files are named <short-hash("<filename>.metadata")>.<etag>.incomplete
    # (see LocalDownloadFilePaths.incomplete_path). While downloading
    # only the .lock exists, so match via the hash itself.
    if not filename:
        return _walk_incomplete(cache, None)
    from huggingface_hub._local_folder import _short_hash
    want = {_short_hash(os.path.basename(filename) + ".metadata")}
    return _walk_incomplete(cache, want)


def _walk_incomplete(cache, hashes):
    total = 0
    if not os.path.isdir(cache):
        return 0
    for root, _, files in os.walk(cache):
        for f in files:
            if not f.endswith(".incomplete"):
                continue
            if hashes and f.split(".")[0] not in hashes:
                continue
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _list_repo_files(repo_id):
    """Best-effort repo file listing, cached. [] on gated/offline errors."""
    now = time.time()
    hit = _REPO_FILES_CACHE.get(repo_id)
    if hit and now - hit[0] < _CACHE_TTL:
        return list(hit[1])
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(
            repo_id, repo_type="model", token=_token())
        _REPO_FILES_CACHE[repo_id] = (now, list(files))
        return list(files)
    except Exception:
        return []


def _repo_info(repo_id):
    """Best-effort model metadata (cardData/sha/tags), cached."""
    now = time.time()
    hit = _REPO_INFO_CACHE.get(repo_id)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, token=_token())
        _REPO_INFO_CACHE[repo_id] = (now, info)
        return info
    except Exception:
        return None


def _is_weight(name):
    return str(name or "").lower().endswith(_WEIGHT_EXTS)


_QUANT_RE = re.compile(
    r"(?:^|[-_.])(?:IQ\d+[_A-Z0-9]*|Q\d+(?:_[A-Z0-9]+)*|BF16|FP16|F16|F32)"
    r"(?:[-_.]|$)", re.IGNORECASE)


def _is_companion_gguf(name):
    """True for a companion GGUF (mmproj / MTP draft / draft head).

    Prefix forms (mmproj-*, mtp-*, draft*) are always companions. Infix
    "mtp"/"dflash"/"eagle" names count only when they do NOT also carry a
    quant token — otherwise they are another full-model quant the user did
    not select (e.g. "Model-DFlash2-MTP-Q8_0.gguf")."""
    base = os.path.basename(str(name or "")).lower()
    if not base.endswith(".gguf"):
        return False
    if base.startswith(("mmproj", "mtp-", "mtp_", "draft", "dflash",
                        "eagle")):
        return True
    if "mmproj" in base or "draft" in base:
        return True
    if any(t in base for t in ("mtp", "dflash", "eagle")):
        return not _QUANT_RE.search(base)
    return False


def _companion_kind(name):
    """'mmproj' | 'draft' | '' for a companion GGUF.

    `draft` covers MTP / DFlash / Eagle / Medusa heads: all are small
    side-models paired with the main weights, never another full quant.
    """
    base = os.path.basename(str(name or "")).lower()
    if not base.endswith(".gguf"):
        return ""
    if "mmproj" in base:
        return "mmproj"
    if base.startswith(("mtp-", "mtp_", "draft", "dflash", "eagle")):
        return "draft"
    if any(t in base for t in ("mtp", "dflash", "eagle", "draft")):
        return "draft"
    return ""


def _quant_tag(name):
    """Normalized quant tag from a filename ('UD-IQ1_S' -> 'IQ1S',
    'Q4_K_M' -> 'Q4KM', 'BF16' -> 'BF16'), or '' when none is present.
    Used to pair a draft/MTP head with the quant the user selected."""
    m = _QUANT_RE.search(os.path.basename(str(name or "")))
    if not m:
        return ""
    return re.sub(r"[^A-Z0-9]", "", m.group(0).upper())


def _select_companions(filename, repo_files, main):
    """Pick at most one mmproj + one matching draft/MTP head.

    Some repos (e.g. unsloth/Qwen3.8-Flash-Next-GGUF) ship an mmproj in
    two precisions and an MTP head PER QUANT plus 'shared' variants.
    Downloading every companion silently ballooned a 28 GB pick into
    tens of GB of draft heads the user never asked for. Rules:
      - never another full-model quant (exclude the selected shards)
      - mmproj: exactly one, preferring F16 (works on every backend)
      - MTP/draft: only one whose quant matches the selection; if the
        repo has exactly one draft file, take it; otherwise skip and
        warn (fetch a specific head manually for speculative decoding)
    """
    main_bases = {os.path.basename(f).lower() for f in (main or [])}
    sel_q = _quant_tag(filename)
    groups = {"mmproj": [], "draft": []}
    for f in repo_files or []:
        if not _is_companion_gguf(f):
            continue
        if f in (main or []) or os.path.basename(f).lower() in main_bases:
            continue
        kind = _companion_kind(f)
        if kind in groups:
            groups[kind].append(f)

    warnings = []
    chosen = []

    # mmproj: one file; F16 is the safest default, then BF16.
    if groups["mmproj"]:
        def _mm_key(f):
            b = os.path.basename(f).lower()
            pref = 0 if ("f16" in b and "bf16" not in b) else (
                1 if "bf16" in b else 2)
            return (pref, f.count("/"), len(b), b)
        pick = sorted(groups["mmproj"], key=_mm_key)[0]
        chosen.append(pick)
        if len(groups["mmproj"]) > 1:
            warnings.append(
                f"repo has {len(groups['mmproj'])} mmproj precision(s); "
                f"downloaded one ({os.path.basename(pick)})")

    # MTP/draft: only the matching quant (dedupe same basename across
    # folders — repos often keep a root copy and an MTP/ copy).
    drafts = groups["draft"]
    if drafts:
        def _draft_rank(f):
            b = os.path.basename(f).lower()
            return (1 if "shared" in b else 0, f.count("/"), len(b), b)
        matching = [f for f in drafts if sel_q and _quant_tag(f) == sel_q]
        if matching:
            by_base = {}
            for f in sorted(matching, key=_draft_rank):
                by_base.setdefault(os.path.basename(f).lower(), f)
            pick = sorted(by_base.values(), key=_draft_rank)[0]
            chosen.append(pick)
            if len(matching) > 1:
                warnings.append(
                    f"repo has {len(matching)} {sel_q} MTP/draft head(s); "
                    f"downloaded one ({os.path.basename(pick)})")
        elif len(drafts) == 1:
            chosen.append(drafts[0])
        else:
            tags = sorted({_quant_tag(f) for f in drafts if _quant_tag(f)})
            warnings.append(
                "repo MTP/draft heads exist for "
                f"{', '.join(tags) or 'other quants'} but not for "
                f"{sel_q or 'this quant'}; none downloaded — add one "
                "manually for speculative decoding")
    return chosen, warnings


def _is_metadata_file(name):
    """Small files needed to load/identify the model from Python."""
    base = os.path.basename(str(name or "")).lower()
    if not base or _is_weight(base):
        return False
    return (base.startswith(_META_PREFIXES)
            or base.endswith(_META_EXTS)
            or base in ("license", "licence", "copying", "notice"))


def _is_license_file(name):
    base = os.path.basename(str(name or "")).lower()
    return base.startswith(("license", "licence", "copying", "notice"))


def _has_config(files):
    return any(os.path.basename(f).lower() == "config.json" for f in files)


def _has_tokenizer(files):
    for f in files:
        b = os.path.basename(f).lower()
        if b.startswith(("tokenizer", "vocab", "merges", "spiece",
                         "special_tokens", "added_tokens")):
            return True
        if b.endswith((".model", ".tiktoken")):
            return True
    return False


def _base_model_ids(info):
    """Repo ids named by the HF card/tags as this model's base_model."""
    out = []

    def _add(value):
        if isinstance(value, str):
            v = value.strip()
            if v and "/" in v and v not in out:
                out.append(v)
        elif isinstance(value, list):
            for item in value:
                _add(item)
        elif isinstance(value, dict):
            _add(value.get("id") or value.get("modelId")
                 or value.get("repo_id"))

    if info is not None:
        card = getattr(info, "cardData", None) or {}
        if isinstance(card, dict):
            _add(card.get("base_model"))
        for tag in (getattr(info, "tags", None) or []):
            if isinstance(tag, str) and tag.startswith("base_model:"):
                # base_model:<relation>:owner/name  (or no relation segment)
                _add(tag.split(":", 2)[-1] if tag.count(":") >= 2
                     else tag.split(":", 1)[1])
    return out


def _license_meta(info):
    card = (getattr(info, "cardData", None) or {})
    if not isinstance(card, dict):
        return None
    lic = card.get("license") or card.get("license_name")
    if isinstance(lic, list):
        return ", ".join(str(x) for x in lic if x)
    return str(lic) if lic else None


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def _write_manifest(bundle):
    files = []
    if bundle.get("kind") == "repo":
        # one job covers the whole tree — record the actual files on disk
        for root, dirs, names in os.walk(bundle["dir"]):
            dirs[:] = [d for d in dirs if d != ".cache"]
            for name in names:
                if name == MANIFEST_NAME:
                    continue
                full = os.path.join(root, name)
                try:
                    rel = os.path.relpath(full, bundle["dir"]).replace(
                        os.sep, "/")
                    size = os.path.getsize(full)
                except OSError:
                    continue
                files.append({
                    "path": rel,
                    "repo": bundle.get("repo_id"),
                    "source_file": rel,
                    "kind": "repo-file",
                    "status": "done",
                    "size_bytes": size,
                    "error": None,
                })
    for jid in ([] if bundle.get("kind") == "repo"
                else bundle.get("job_ids") or []):
        job = JOBS.get(jid)
        if not job:
            continue
        try:
            rel = os.path.relpath(job["dest"], bundle["dir"]).replace(
                os.sep, "/")
        except Exception:
            rel = job.get("rel_path") or job.get("filename") or ""
        files.append({
            "path": rel,
            "repo": job.get("repo_id"),
            "source_file": job.get("filename"),
            "kind": job.get("kind") or "model",
            "status": job.get("status"),
            "size_bytes": job.get("done_bytes")
            if job.get("done_bytes") is not None else job.get("total_bytes"),
            "error": job.get("error"),
        })
    manifest = {
        "bundle_version": 1,
        "name": bundle.get("name"),
        "store": bundle.get("store"),
        "repo_id": bundle.get("repo_id"),
        "repo_revision": bundle.get("repo_revision"),
        "selected_file": bundle.get("selected_file"),
        "base_model": bundle.get("base_model"),
        "license": bundle.get("license"),
        "license_files": [
            f["path"] for f in files if _is_license_file(f["path"])
        ],
        "downloaded_at": bundle.get("created"),
        "files": files,
        "warnings": bundle.get("warnings") or [],
    }
    path = os.path.join(bundle["dir"], MANIFEST_NAME)
    _write_json(path, manifest)
    bundle["manifest"] = path
    return path


def _finalize_bundle(bundle_id):
    """Wait for every planned job, prune hf's .cache, write the manifest."""
    bundle = BUNDLES.get(bundle_id)
    if not bundle:
        return
    ids = list(bundle.get("job_ids") or [])
    while any(JOBS.get(j, {}).get("status") == "downloading" for j in ids):
        time.sleep(1)
    # hf leaves .cache/<...>/download metadata + hf-xet under the bundle;
    # keep the model folder clean for model_parser.
    shutil.rmtree(os.path.join(bundle["dir"], ".cache"), ignore_errors=True)
    statuses = {JOBS.get(j, {}).get("status") for j in ids}
    bundle["status"] = ("done" if statuses and statuses <= {"done"}
                        else "error")
    try:
        _write_manifest(bundle)
    except Exception as e:
        bundle["manifest_error"] = str(e)


def _start_file(repo_id, filename, store, bundle_dir, kind, local_subdir=""):
    """Start ONE file download inside a bundle folder."""
    rel = _safe_rel(filename)
    local_dir = (os.path.join(bundle_dir, local_subdir) if local_subdir
                 else bundle_dir)
    dest = os.path.join(local_dir, *rel.split("/"))
    job_id = uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "link": f"hf://{text_link(repo_id, filename)}",
        "repo_id": repo_id,
        "filename": filename,
        "rel_path": rel,
        "local_dir": local_dir,
        "store": store,
        "dest": dest,
        "status": "downloading",
        "kind": kind,
        "bundle": os.path.basename(bundle_dir),
        "bundle_dir": bundle_dir,
        "started": datetime.now().isoformat(timespec="seconds"),
        "total_bytes": _expected_size(repo_id, filename),
        "done_bytes": 0,
        "error": None,
        "finished": None,
    }
    if os.path.exists(dest):
        job["status"] = "done"
        job["total_bytes"] = os.path.getsize(dest)
        job["done_bytes"] = job["total_bytes"]
        job["error"] = "file already exists at destination"
        job["finished"] = datetime.now().isoformat(timespec="seconds")
        JOBS[job_id] = job
        return job
    JOBS[job_id] = job
    t = threading.Thread(target=_run, args=(job_id,), daemon=True)
    t.start()
    return job


def _plan_bundle(repo_id, filename, store, all_shards):
    """Resolve the bundle folder + every file that belongs to it."""
    root = path_resolver.resolve(f"store://{store}")
    if not root or root.startswith("store://"):
        raise ValueError(f"unknown store '{store}'")
    name = _bundle_name(filename)
    bundle_dir = os.path.join(root, name)
    os.makedirs(bundle_dir, exist_ok=True)

    main = sibling_shards(repo_id, filename) if all_shards else [filename]
    main = list(dict.fromkeys(main))

    repo_files = _list_repo_files(repo_id)
    warnings = []
    if not repo_files:
        warnings.append(
            "could not list repo files — companions/metadata were not "
            "discovered (main download still runs)")
    companions, comp_warnings = _select_companions(
        filename, repo_files, main)
    warnings += comp_warnings
    metadata = [f for f in repo_files if _is_metadata_file(f)]

    info = _repo_info(repo_id)
    base_ids = _base_model_ids(info)
    base_files = []
    base_model = base_ids[0] if base_ids else None
    if base_model and not (_has_config(repo_files) and _has_tokenizer(repo_files)):
        listed = _list_repo_files(base_model)
        if listed:
            base_files = [f for f in listed if _is_metadata_file(f)]
        else:
            warnings.append(
                f"base_model '{base_model}' could not be listed — "
                "tokenizer/config files may be missing")
    elif not (_has_config(repo_files) and _has_tokenizer(repo_files)):
        warnings.append(
            "repo has no config.json/tokenizer and no base_model was "
            "declared — pure-Python loading may need those files")
    if not any(_is_license_file(f) for f in (repo_files + base_files)):
        warnings.append(
            "no LICENSE/COPYING/NOTICE file found in the repo or base_model")

    return {
        "root": root,
        "name": name,
        "dir": bundle_dir,
        "main": main,
        "companions": companions,
        "metadata": metadata,
        "base_model": base_model,
        "base_files": base_files,
        "repo_info": info,
        "warnings": warnings,
    }


def _start_bundle(link, store, all_shards):
    repo_id, filename = parse_link(link)
    plan = _plan_bundle(repo_id, filename, store, all_shards)
    bundle_id = uuid.uuid4().hex[:8]
    bundle = {
        "id": bundle_id,
        "name": plan["name"],
        "dir": plan["dir"],
        "store": store,
        "repo_id": repo_id,
        "repo_revision": getattr(plan["repo_info"], "sha", None),
        "selected_file": filename,
        "base_model": plan["base_model"],
        "license": _license_meta(plan["repo_info"]),
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "downloading",
        "warnings": plan["warnings"],
        "model_files": plan["main"],
        "companion_files": plan["companions"],
        "metadata_files": plan["metadata"],
        "base_metadata_files": plan["base_files"],
        "job_ids": [],
    }
    BUNDLES[bundle_id] = bundle

    jobs = []
    seen = set()

    def _add(repo, f, kind, subdir=""):
        key = (repo, _safe_rel(f), subdir)
        if key in seen:
            return
        seen.add(key)
        job = _start_file(repo, f, store, plan["dir"], kind, subdir)
        jobs.append(job)
        bundle["job_ids"].append(job["id"])

    for f in plan["main"]:
        _add(repo_id, f, "model")
    for f in plan["companions"]:
        _add(repo_id, f, "companion")
    for f in plan["metadata"]:
        _add(repo_id, f, "metadata")
    if plan["base_model"]:
        for f in plan["base_files"]:
            _add(plan["base_model"], f, "base-metadata", BASE_METADATA_DIR)

    bundle["files"] = [j["dest"] for j in jobs]
    t = threading.Thread(target=_finalize_bundle, args=(bundle_id,),
                         daemon=True)
    t.start()
    return {
        "bundle": bundle,
        "jobs": jobs,
        "sharded": len(plan["main"]) > 1,
        "model_jobs": len(plan["main"]),
        "total": len(jobs),
    }


def start(link, store):
    """Download one GGUF (its shards + companions + metadata) into a bundle."""
    res = _start_bundle(link, store, all_shards=False)
    primary = dict(res["jobs"][0]) if res["jobs"] else {}
    primary["jobs"] = res["jobs"]
    primary["bundle"] = res["bundle"]
    primary["model_jobs"] = res["model_jobs"]
    return primary


def text_link(repo_id, filename):
    return f"{repo_id}/{filename}"


def split_info(filename):
    """(stem, index, total) for a split shard like base-00001-of-00002.gguf,
    else None. Width-agnostic so -1-of-2 and -00001-of-00002 both match."""
    m = SPLIT_RE.search(os.path.basename(filename or ""))
    if not m:
        return None
    stem = os.path.basename(filename)[:m.start()]
    return stem, int(m.group(1)), int(m.group(2))


def sibling_shards(repo_id, filename):
    """All shard filenames in the repo sharing this file's split stem.

    Lists the repo (auth via HF_TOKEN when set) and falls back to
    synthesizing -00001..N-of-0000N names (zero-padded like the input)
    when the listing fails, so a gated/offline listing never blocks the
    download — worst case we attempt the expected names directly.
    """
    info = split_info(filename)
    if not info:
        return [filename]
    stem, _, total = info
    width = len(SPLIT_RE.search(os.path.basename(filename)).group(1))
    expected = [f"{stem}-{i:0{width}d}-of-{total:0{width}d}.gguf"
                for i in range(1, total + 1)]
    listed = _list_repo_files(repo_id)
    if listed:
        want = {e.lower() for e in expected}
        # return full repo paths (shards sometimes sit in subfolders);
        # match on basename, keep numeric shard order
        found = [f for f in listed if f.split("/")[-1].lower() in want]
        found.sort(key=lambda f: int(
            SPLIT_RE.search(f.split("/")[-1]).group(1)))
        if found:
            return found
    return expected


def start_shards(link, store):
    """Download a model's full split set + companions + metadata.

    A non-split link behaves like start(). Returns
    {"jobs": [...], "sharded": bool, "total": N, "model_jobs": N,
     "bundle": {...}}.
    """
    res = _start_bundle(link, store, all_shards=True)
    return {
        "jobs": res["jobs"],
        "sharded": res["sharded"],
        "total": res["total"],
        "model_jobs": res["model_jobs"],
        "bundle": res["bundle"],
    }


def _run(job_id):
    job = JOBS[job_id]
    local_dir = job["local_dir"]
    # Each job gets its own child process (isolation: two concurrent
    # downloads must not share xet's chunk graph) and its own xet
    # cache. hf_xet is disabled anyway so bytes stream straight into
    # the .incomplete file inside local_dir -- never into ~/.cache.
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_XET_CACHE"] = os.path.join(local_dir, ".cache", "hf-xet")
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    _tok = _token()
    if _tok:
        env["HF_TOKEN"] = _tok
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _CHILD, job["repo_id"], job["filename"],
         local_dir],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, env=env)
    _PROCS[job_id] = proc
    lines = deque(maxlen=200)
    reader = threading.Thread(target=_pump, args=(proc, lines), daemon=True)
    reader.start()
    while proc.poll() is None:
        job["done_bytes"] = _incomplete_bytes(local_dir, job["filename"])
        job["log"] = list(lines)
        time.sleep(1)
    reader.join(timeout=5)
    job["done_bytes"] = _incomplete_bytes(local_dir, job["filename"])
    job["log"] = list(lines)
    if proc.returncode == 0:
        job["status"] = "done"
        try:
            size = os.path.getsize(job["dest"])
            job["total_bytes"] = size
            job["done_bytes"] = size
        except OSError:
            pass
        try:
            # detection only (never triggers another download)
            import config_loader
            config_loader._is_mtp_draft(job["dest"])
        except Exception:
            # Ignore errors in MTP detection; they should not affect download status
            pass
    elif job["status"] != "canceled":
        job["status"] = "error"
        job["error"] = (lines[-1] if lines else "") or \
            f"exit code {proc.returncode}"
    job["finished"] = datetime.now().isoformat(timespec="seconds")
    _PROCS.pop(job_id, None)


def _pump(proc, lines):
    for line in proc.stdout:
        lines.append(line.rstrip())


def cancel(job_id):
    job = JOBS.get(job_id)
    proc = _PROCS.get(job_id)
    if not job or not proc:
        return {"canceled": False}
    job["status"] = "canceled"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"canceled": True}


def all_jobs():
    return sorted(JOBS.values(), key=lambda j: j["started"], reverse=True)


def all_bundles():
    return sorted(BUNDLES.values(), key=lambda b: b.get("created") or "",
                  reverse=True)

def _open_paths():
    """Real paths of files currently held open by any process.

    Linux: /proc/<pid>/fd scan. macOS/other POSIX: `lsof -Fn` fallback.
    Windows: no scan — the age guard + active-job directory guard still
    protect running downloads there.
    """
    paths = set()
    if os.path.isdir("/proc"):
        try:
            pids = [p for p in os.listdir("/proc") if p.isdigit()]
        except OSError:
            pids = []
        for pid in pids:
            fd_dir = f"/proc/{pid}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    paths.add(os.path.realpath(os.path.join(fd_dir, fd)))
                except OSError:
                    pass
        if paths:
            return paths
    if shutil.which("lsof"):
        try:
            proc = subprocess.run(["lsof", "-Fn"], capture_output=True,
                                  text=True, timeout=20)
            for line in (proc.stdout or "").splitlines():
                if line.startswith("n") and len(line) > 1:
                    try:
                        paths.add(os.path.realpath(line[1:]))
                    except OSError:
                        pass
        except (OSError, subprocess.SubprocessError):
            pass
    return paths


def cleanup_stale_cache(min_age_seconds=3600, stores=None):
    """Remove stale HF download `.incomplete` / `.lock` files.

    Runs at app startup (and can be called manually). Only files older than
    `min_age_seconds` are removed, cache dirs belonging to an in-process
    active job are skipped wholesale, and any path currently open by a live
    process is skipped (Linux /proc, POSIX lsof) — so a running download is
    never touched. Scans each store root's `.cache/huggingface/download`
    plus one level of per-model bundle folders.
    """
    roots = []
    names = stores if stores is not None else list_stores()
    for name in names:
        try:
            root = path_resolver.resolve(f"store://{name}")
        except Exception:
            continue
        if root and not str(root).startswith("store://"):
            roots.append(root)
    roots = list(dict.fromkeys(roots))

    cache_dirs = []
    for root in roots:
        cache_dirs.append(os.path.join(
            root, ".cache", "huggingface", "download"))
        try:
            for entry in os.scandir(root):
                if entry.is_dir(follow_symlinks=False):
                    cache_dirs.append(os.path.join(
                        entry.path, ".cache", "huggingface", "download"))
        except OSError:
            pass

    open_paths = _open_paths()
    # Never touch a cache directory that belongs to a download this process
    # is still running — even if its mtime looks old (slow/blocked stream).
    active_cache_dirs = set()
    for job in JOBS.values():
        if job.get("status") != "downloading":
            continue
        for base in (job.get("bundle_dir"), job.get("local_dir")):
            if base and os.path.isdir(base):
                active_cache_dirs.add(os.path.realpath(os.path.join(
                    base, ".cache", "huggingface", "download")))
    now = time.time()
    result = {"removed_incomplete": 0, "removed_locks": 0,
              "removed_bytes": 0, "skipped_active": 0,
              "errors": []}
    for cache_dir in dict.fromkeys(cache_dirs):
        if not os.path.isdir(cache_dir):
            continue
        real_cache = os.path.realpath(cache_dir)
        if any(real_cache == d or real_cache.startswith(d + os.sep)
               for d in active_cache_dirs):
            result["skipped_active"] += 1
            continue
        for dirpath, _dirnames, files in os.walk(cache_dir):
            for fname in files:
                if not (fname.endswith(".incomplete")
                        or fname.endswith(".lock")):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    st = os.stat(path, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError as e:
                    result["errors"].append(f"{path}: {e}")
                    continue
                if now - st.st_mtime < min_age_seconds:
                    continue
                if os.path.realpath(path) in open_paths:
                    result["skipped_active"] += 1
                    continue
                try:
                    os.remove(path)
                except FileNotFoundError:
                    continue
                except OSError as e:
                    result["errors"].append(f"{path}: {e}")
                    continue
                if fname.endswith(".incomplete"):
                    result["removed_incomplete"] += 1
                    result["removed_bytes"] += st.st_size
                else:
                    result["removed_locks"] += 1
        # prune empty dirs left behind (keep the cache root itself)
        for dirpath, _dirnames, _files in os.walk(cache_dir, topdown=False):
            if dirpath == cache_dir:
                continue
            try:
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
            except OSError:
                pass
    return result
