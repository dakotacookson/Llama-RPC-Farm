"""Shared lock for long-running update/install jobs."""
import threading

_LLAMA_UPDATES_LOCK = threading.Lock()
