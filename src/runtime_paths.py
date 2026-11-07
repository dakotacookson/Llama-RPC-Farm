"""Writable data vs bundled asset paths for source and frozen builds."""
import os
import sys
from pathlib import Path


def asset_dir() -> str:
    """Read-only bundled resources (ui, examples, bundled agent wheel)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir() -> str:
    """Per-user writable state (config, vault, generated files)."""
    override = (os.environ.get("LLAMA_FARM_DATA_DIR") or "").strip()
    if override:
        path = Path(override).expanduser()
    elif not getattr(sys, "frozen", False):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    elif os.name == "nt":
        path = Path(os.environ.get("APPDATA") or Path.home()) / "LlamaRPCFarm"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "LlamaRPCFarm"
    else:
        path = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "llama-rpc-farm"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# Minimal, schema-valid defaults for a brand-new data dir. The app already
# tolerates missing files; writing these up front just makes packaged
# first-run deterministic (no "file not found" surprises in the UI).
_DEFAULT_CONFIG_FILES = {
    "config/settings.yaml": (
        "model_dir: store://llm-models\n"
        "llama_bin: ~/llama.cpp/build/bin\n"
        "theme_family: nord\n"
        "theme_mode: dark\n"
        "engine: llama_cpp\n"
    ),
    "config/machines.yaml": "machines: []\n",
    "config/groups.yaml": "groups: []\n",
    "config/swap_groups.yaml": "swap_groups: []\n",
    "config/stores.yaml": "stores: {}\n",
    "config/swap.yaml": "{}\n",
}


def ensure_runtime_layout() -> str:
    """Create the writable layout a fresh install needs.

    Called by the app entry points (source and frozen). Best-effort:
    a locked/read-only data dir must still boot (the app already tolerates
    missing config), but on a normal first run this guarantees every
    directory and default config file exists before serving.
    """
    root = Path(data_dir())
    for sub in ("config", "config/models", "config/usage", "generated"):
        try:
            (root / sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    for rel, text in _DEFAULT_CONFIG_FILES.items():
        path = root / rel
        try:
            if not path.exists():
                path.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return str(root)
