#!/usr/bin/env python3
"""Set the release version in every version-bearing source file.

Used locally and by the rolling release workflow so the release tag, app/agent
artifacts, and checked-out source all report the same version.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
AGENT_PYPROJECT = ROOT / "peer_agent" / "pyproject.toml"
AGENT_INIT = ROOT / "peer_agent" / "llama_peer_agent" / "__init__.py"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stamp_version.py MAJOR.MINOR.PATCH", file=sys.stderr)
        return 2
    version = sys.argv[1].strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print(f"invalid version: {version!r}", file=sys.stderr)
        return 2

    VERSION_FILE.write_text(version + "\n", encoding="utf-8")

    text = AGENT_PYPROJECT.read_text(encoding="utf-8")
    text, n1 = re.subn(r'(?m)^version\s*=\s*"[^"]+"',
                       f'version = "{version}"', text)
    if n1 != 1:
        print(f"could not update {AGENT_PYPROJECT}", file=sys.stderr)
        return 1
    AGENT_PYPROJECT.write_text(text, encoding="utf-8")

    text = AGENT_INIT.read_text(encoding="utf-8")
    text, n2 = re.subn(r'(?m)^__version__\s*=\s*"[^"]+"',
                       f'__version__ = "{version}"', text)
    if n2 != 1:
        print(f"could not update {AGENT_INIT}", file=sys.stderr)
        return 1
    AGENT_INIT.write_text(text, encoding="utf-8")
    print(f"stamped {version}: VERSION, agent pyproject, agent __version__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
