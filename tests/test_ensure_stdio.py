"""Detached restarts can run the app with no stdout/stderr.

uvicorn's DefaultFormatter calls sys.stdout.isatty() when use_colors is
unset, so a None stream crashed startup ("Unable to configure formatter
'default'"). app._ensure_stdio gives missing streams a sink first.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module


def test_missing_streams_get_a_sink(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(app_module.tempfile, "gettempdir", lambda: str(tmp_path))
    app_module._ensure_stdio()
    try:
        assert sys.stdout is not None and sys.stderr is not None
        from uvicorn.logging import DefaultFormatter
        DefaultFormatter()                      # crashed before the guard
        print("hello", flush=True)
        assert "hello" in (tmp_path / "llama-web.log").read_text(encoding="utf-8")
    finally:
        sys.stdout.close()
        sys.stderr.close()


def test_present_streams_are_left_alone(monkeypatch):
    out, err = sys.stdout, sys.stderr
    app_module._ensure_stdio()
    assert sys.stdout is out and sys.stderr is err
