"""Background Hugging Face downloads straight into a store location.

Parses hf:// links (hf://owner/repo/path/file.gguf), then runs the
download in a child process with local_dir set to the resolved store
root. huggingface_hub downloads to a .incomplete file inside that same
folder and moves it into place, so bytes stream to disk once -- the
~/.cache/huggingface blob store is never touched and the NAS sees one
sequential write instead of write+copy.
"""
import os
import re
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

JOBS = {}
_PROCS = {}

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

_REPO_CHILD = """\
import os, sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1:3]
# only weight/config/tokenizer files; the .cache dir snapshot_download
# writes inside local_dir is pruned at the end (keeps the store clean)
try:
    path = snapshot_download(
        repo_id=repo_id, local_dir=local_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                        "*.py", "*.jinja", "*.jinja2"],
        ignore_patterns=["*.gguf", "*.bin", "*.pt", "*.pth", "*.ckpt",
                         "*.h5", "*.msgpack", "*.onnx", "*.tflite",
                         "*.md", "*.gitattributes", "*.pdf"])
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


def _expected_size(repo_id, filename):
    try:
        url = hf_hub_url(repo_id, filename)
        return get_hf_file_metadata(url).size
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


def start(link, store):
    repo_id, filename = parse_link(link)
    root = path_resolver.resolve(f"store://{store}")
    if not root or root.startswith("store://"):
        raise ValueError(f"unknown store '{store}'")
    job_id = uuid.uuid4().hex[:8]
    dest = os.path.join(root, os.path.basename(filename))
    job = {
        "id": job_id, "link": f"hf://{text_link(repo_id, filename)}",
        "repo_id": repo_id, "filename": filename, "store": store,
        "dest": dest, "status": "downloading",
        "started": datetime.now().isoformat(timespec="seconds"),
        "total_bytes": _expected_size(repo_id, filename),
        "done_bytes": 0, "error": None, "finished": None,
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
        return [os.path.basename(filename)]
    stem, _, total = info
    width = len(SPLIT_RE.search(os.path.basename(filename)).group(1))
    expected = [f"{stem}-{i:0{width}d}-of-{total:0{width}d}.gguf"
                for i in range(1, total + 1)]
    try:
        from huggingface_hub import HfApi
        try:
            from hf_browse import _token as _hf_token
        except Exception:
            _hf_token = lambda: os.environ.get("HF_TOKEN") or \
                os.environ.get("HUGGING_FACE_HUB_TOKEN")
        listed = HfApi().list_repo_files(repo_id, repo_type="model",
                                         token=_hf_token())
        want = {e.lower() for e in expected}
        # return full repo paths (shards sometimes sit in subfolders);
        # match on basename, keep numeric shard order
        found = [f for f in listed if f.split("/")[-1].lower() in want]
        found.sort(key=lambda f: int(
            SPLIT_RE.search(f.split("/")[-1]).group(1)))
        if found:
            return found
    except Exception:
        pass
    return expected


def start_shards(link, store):
    """Start a download job per shard. A non-split link behaves exactly
    like start() and returns a single job. Returns
    {"jobs": [...], "sharded": bool, "total": N}."""
    repo_id, filename = parse_link(link)
    base = os.path.basename(filename)
    if not split_info(base):
        return {"jobs": [start(link, store)], "sharded": False, "total": 1}
    jobs = []
    for shard in sibling_shards(repo_id, base):
        jobs.append(start(f"hf://{repo_id}/{shard}", store))
    return {"jobs": jobs, "sharded": True, "total": len(jobs)}


def repo_size(repo_id):
    """Total bytes of the safetensors files snapshot_download would fetch.
    Best-effort: None when the listing fails (gated/offline)."""
    try:
        from huggingface_hub import HfApi
        try:
            from hf_browse import _token as _hf_token
        except Exception:
            _hf_token = lambda: os.environ.get("HF_TOKEN") or \
                os.environ.get("HUGGING_FACE_HUB_TOKEN")
        info = HfApi().model_info(repo_id, files_metadata=True,
                                  token=_hf_token())
        total = 0
        for s in (info.siblings or []):
            fn = s.rfilename or ""
            if fn.lower().endswith(".safetensors"):
                total += (s.size or 0)
        return total or None
    except Exception:
        return None


def start_repo(repo_id, store):
    """Download a whole HF/safetensors repo into its own subfolder of the
    store (config.json + *.safetensors = what model_parser and vllm serve
    expect). One job, one child process; progress tracks the .incomplete
    files snapshot_download writes under the repo subfolder."""
    repo_id = (repo_id or "").strip().strip("/")
    if repo_id.count("/") != 1:
        raise ValueError("expected owner/name repo id (e.g. Qwen/Qwen3-0.6B)")
    root = path_resolver.resolve(f"store://{store}")
    if not root or root.startswith("store://"):
        raise ValueError(f"unknown store '{store}'")
    job_id = uuid.uuid4().hex[:8]
    dest = os.path.join(root, repo_id.split("/")[-1])
    job = {
        "id": job_id, "link": f"hf://{repo_id}", "repo_id": repo_id,
        "filename": repo_id.split("/")[-1] + "/", "store": store,
        "dest": dest, "status": "downloading", "kind": "repo",
        "started": datetime.now().isoformat(timespec="seconds"),
        "total_bytes": repo_size(repo_id),
        "done_bytes": 0, "error": None, "finished": None,
    }
    if os.path.isdir(dest) and os.path.isfile(os.path.join(dest, "config.json")):
        job["status"] = "done"
        job["error"] = "model already exists at destination"
        job["finished"] = datetime.now().isoformat(timespec="seconds")
        JOBS[job_id] = job
        return job
    JOBS[job_id] = job
    t = threading.Thread(target=_run_repo, args=(job_id,), daemon=True)
    t.start()
    return job


def _run_repo(job_id):
    job = JOBS[job_id]
    local_dir = path_resolver.resolve(f"store://{job['store']}")
    repo_dir = os.path.join(local_dir, job["repo_id"].split("/")[-1])
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_XET_CACHE"] = os.path.join(repo_dir, ".cache", "hf-xet")
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    os.makedirs(repo_dir, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _REPO_CHILD, job["repo_id"], repo_dir],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, env=env)
    _PROCS[job_id] = proc
    lines = deque(maxlen=200)
    reader = threading.Thread(target=_pump, args=(proc, lines), daemon=True)
    reader.start()
    while proc.poll() is None:
        # progress = finished safetensors bytes + in-flight .incomplete bytes
        done = _safetensors_bytes(repo_dir)
        done += _incomplete_bytes(repo_dir)
        job["done_bytes"] = done
        job["log"] = list(lines)
        time.sleep(1)
    reader.join(timeout=5)
    job["done_bytes"] = max(job["done_bytes"], _safetensors_bytes(repo_dir))
    job["log"] = list(lines)
    # snapshot_download leaves a .cache folder inside the repo dir; prune it
    # so model_parser/vllm see a clean model dir
    import shutil
    shutil.rmtree(os.path.join(repo_dir, ".cache"), ignore_errors=True)
    if proc.returncode == 0:
        job["status"] = "done"
        job["total_bytes"] = _safetensors_bytes(repo_dir)
        job["done_bytes"] = job["total_bytes"]
    elif job["status"] != "canceled":
        job["status"] = "error"
        job["error"] = (lines[-1] if lines else "") or \
            f"exit code {proc.returncode}"
    job["finished"] = datetime.now().isoformat(timespec="seconds")
    _PROCS.pop(job_id, None)


def _safetensors_bytes(repo_dir):
    total = 0
    if not os.path.isdir(repo_dir):
        return 0
    for root, _, files in os.walk(repo_dir):
        if os.path.basename(root) == ".cache":
            continue
        for f in files:
            if f.lower().endswith(".safetensors"):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


def _run(job_id):
    job = JOBS[job_id]
    local_dir = path_resolver.resolve(f"store://{job['store']}")
    # Each job gets its own child process (isolation: two concurrent
    # downloads must not share xet's chunk graph) and its own xet
    # cache. hf_xet is disabled anyway so bytes stream straight into
    # the .incomplete file inside local_dir -- never into ~/.cache.
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_XET_CACHE"] = os.path.join(local_dir, ".cache", "hf-xet")
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
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