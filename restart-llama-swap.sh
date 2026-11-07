#!/usr/bin/env bash
# Backwards-compatible alias for the full cluster restart.
# Prefer: ./restart.sh --full   (or ./restart.sh for the default full mode).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/restart.sh" --full
