"""Resolves store:// paths to platform-specific model storage roots.

Model configs store portable paths like `store://llm-models/foo.gguf`.
The roots live in config/stores.yaml with per-platform entries, so the
same configs work unmodified on Windows (UNC share) and on Linux
(CIFS mount) -- switching platforms is just moving the folder.
"""
import os
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORES_FILE = os.path.join(BASE_DIR, "config", "stores.yaml")

PREFIX = "store://"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
LOCAL_PLATFORM = "windows" if IS_WINDOWS else "macos" if IS_MACOS else "linux"


def _load_stores():
    if not os.path.exists(STORES_FILE):
        return {}
    with open(STORES_FILE, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("stores", {})


def resolve(path, platform=None):
    """Resolve store:// paths for a platform ("windows"|"linux",
    defaults to the current one). Non-store paths pass through."""
    if not path or not path.startswith(PREFIX):
        return path
    rest = path[len(PREFIX):]
    store_name, _, remainder = rest.partition("/")
    stores = _load_stores()
    roots = stores.get(store_name)
    if not roots:
        return path  # unknown store: leave as-is rather than guessing
    platform = platform or LOCAL_PLATFORM
    root = roots.get(platform) or roots.get("linux") or roots.get("windows")
    if not root:
        return path
    root = root.rstrip("/\\")
    return f"{root}/{remainder}" if remainder else root


def resolve_for(path, machine_os=None):
    """Resolve a store:// path for the OS of the machine that will consume
    it. machine_os is "linux" | "windows" | "macos" (from the machine
    config); defaults to the platform this app is running on."""
    return resolve(path, machine_os or LOCAL_PLATFORM)


def to_alias(path):
    """Reverse of resolve(): turn a concrete platform path back into its
    store:// alias so saved configs stay portable. Unknown paths pass
    through unchanged."""
    if not path or not path.startswith(PREFIX):
        stores = _load_stores()
        for name, roots in stores.items():
            for root in roots.values():
                if not root:
                    continue
                norm_root = root.rstrip("/\\")
                # normalize separators for comparison; case-insensitive only
                # on Windows/case-insensitive mounts (macOS APFS can be
                # case-sensitive: two distinct paths must not merge).
                p_norm = path.replace("\\", "/")
                r_norm = norm_root.replace("\\", "/")
                if IS_WINDOWS:
                    match = p_norm.lower().startswith(r_norm.lower())
                else:
                    match = p_norm.startswith(r_norm)
                if match:
                    remainder = path[len(norm_root):].strip("/\\")
                    return f"{PREFIX}{name}/{remainder}"
    return path