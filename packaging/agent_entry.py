"""Frozen entry point for the peer-agent CLI."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

from llama_peer_agent.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
