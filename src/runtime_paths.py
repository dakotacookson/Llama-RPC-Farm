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
