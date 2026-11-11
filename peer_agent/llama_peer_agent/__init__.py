"""Tiny optional peer agent for Llama RPC Farm."""

__version__ = "2.0.23"
PROTOCOL_VERSION = 1

# Baked at build time by scripts/stamp_build_info.py. The master compares it
# to the release manifest so app-only releases do not mark every agent stale.
try:
    from .build_info import AGENT_COMMIT as BUILD_COMMIT
except Exception:  # source/dev run without a stamped build
    BUILD_COMMIT = ""
