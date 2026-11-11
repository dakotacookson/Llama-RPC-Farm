#!/usr/bin/env python3
"""Build a universal source wheel for in-place farm-app updates.

This is intentionally a source-payload wheel, not a pip-installable
library: the updater unzips it and copies the tracked app tree (src, ui,
peer_agent, scripts, packaging) over an existing source install. It gives
non-git installs the same release asset path as git checkouts without
requiring a compiler or a platform-specific binary.
"""
import argparse
import hashlib
import os
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAYLOAD_DIRS = ("src", "ui", "peer_agent", "scripts", "packaging")
PAYLOAD_FILES = (
    "VERSION", "requirements.txt", "install.sh", "install.bat",
    "run.sh", "run.bat", "restart.sh", "restart-llama-swap.sh",
)
SKIP_DIRS = {"__pycache__", ".pytest_cache", "build", "dist", ".git"}


def _add(zf, root, path, record):
    arc = path.relative_to(root).as_posix()
    data = path.read_bytes()
    zf.writestr(arc, data)
    record.append((arc, hashlib.sha256(data).hexdigest(), len(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    version = args.version.strip()
    if not version:
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not version:
        raise SystemExit("VERSION is empty")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / ("llama_rpc_farm-%s-py3-none-any.whl" % version)
    record = []

    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in PAYLOAD_DIRS:
            base = ROOT / name
            if not base.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fn in filenames:
                    if fn.endswith((".pyc", ".pyo")):
                        continue
                    path = pathlib.Path(dirpath) / fn
                    _add(zf, ROOT, path, record)
        for name in PAYLOAD_FILES:
            path = ROOT / name
            if path.is_file():
                _add(zf, ROOT, path, record)

        dist = "llama_rpc_farm-%s.dist-info" % version
        metadata = (
            "Metadata-Version: 2.1\n"
            "Name: llama-rpc-farm\n"
            "Version: %s\n"
            "Summary: Llama RPC Farm source payload for in-place updates\n"
        ) % version
        wheel_meta = (
            "Wheel-Version: 1.0\n"
            "Generator: llama-rpc-farm-build\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        zf.writestr(dist + "/METADATA", metadata)
        zf.writestr(dist + "/WHEEL", wheel_meta)
        record.extend([
            (dist + "/METADATA",
             hashlib.sha256(metadata.encode()).hexdigest(),
             len(metadata.encode())),
            (dist + "/WHEEL",
             hashlib.sha256(wheel_meta.encode()).hexdigest(),
             len(wheel_meta.encode())),
            (dist + "/RECORD", "", ""),
        ])
        zf.writestr(dist + "/RECORD", "".join(
            "%s,sha256=%s,%s\n" % (arc, digest, size)
            if digest else "%s,,\n" % arc
            for arc, digest, size in record))
    print(wheel)


if __name__ == "__main__":
    main()
