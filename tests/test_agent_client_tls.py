"""agent_client switches to HTTPS/mTLS when a machine is flagged."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import agent_client


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=None):
        return b'{"ok":true}'


def test_base_uses_https_when_agent_tls(monkeypatch):
    m = {"name": "p", "ip": "10.0.0.2", "agent_port": 8198,
         "agent_tls": True, "agent_tls_port": 8197}
    assert agent_client._base(m) == "https://10.0.0.2:8197"
    m["agent_tls"] = False
    assert agent_client._base(m) == "http://10.0.0.2:8198"


def test_request_passes_tls_context(monkeypatch):
    m = {"name": "p", "ip": "10.0.0.2", "agent_tls": True,
         "agent_tls_port": 8197}
    seen = {}
    ctx = object()

    def fake_ctx():
        return ctx

    def fake_urlopen(req, timeout=None, context=None):
        seen["url"] = req.full_url
        seen["context"] = context
        return _Resp()

    monkeypatch.setattr(agent_client, "_token", lambda m: "tok")
    monkeypatch.setattr(agent_client, "_ssl_context", fake_ctx)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = agent_client._request(m, "GET", "/health", timeout=1)
    assert out == {"ok": True}
    assert seen["url"].startswith("https://10.0.0.2:8197/health")
    assert seen["context"] is ctx
