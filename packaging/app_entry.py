"""Frozen-app entry point: serve the FastAPI app with uvicorn."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = getattr(sys, "_MEIPASS", ROOT)
sys.path.insert(0, os.path.join(ASSET, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from app import app  # noqa: E402
import uvicorn  # noqa: E402
from runtime_paths import ensure_runtime_layout  # noqa: E402


def main():
    ensure_runtime_layout()
    host = (os.environ.get("LLAMA_FARM_HOST") or "0.0.0.0").strip()
    try:
        port = int(os.environ.get("LLAMA_FARM_PORT") or 8199)
    except ValueError:
        port = 8199
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
