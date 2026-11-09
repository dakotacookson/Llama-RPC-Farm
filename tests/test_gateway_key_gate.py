"""Gateway (:8090) key gate: allow_for_ip must agree with the :8199 seam.

access.restrict_scope_from_key treats a known key with no model grants as
unrestricted, and the first-run admin key is created exactly that way
(models: [], admin: true, admin_unrestricted: true). The gateway used to
refuse that key like an unknown caller, so a logged-in admin got
"invalid or missing API key" on Load / Embeddings / Chat.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from starlette.requests import Request

import swap_gateway as gw

ADMIN = "a" * 32
SCOPED = "s" * 32


@pytest.fixture
def keys(monkeypatch):
    table = {
        ADMIN: {"name": "admin1", "models": set(), "admin": True},
        SCOPED: {"name": "client", "models": {"m1"}, "admin": False},
    }
    monkeypatch.setattr(gw, "access_keys", lambda: table)
    return table


def _req(token=None, ip="192.168.1.50"):
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    return Request({"type": "http", "method": "POST", "path": "/v1/embeddings",
                    "headers": headers, "client": (ip, 40000)})


def test_known_key_without_grants_is_unrestricted(keys):
    assert gw.allow_for_ip(_req(ADMIN)) is None


def test_scoped_key_keeps_its_grants(keys):
    assert gw.allow_for_ip(_req(SCOPED)) == {"m1"}


def test_unknown_or_missing_key_is_refused(keys):
    assert gw.allow_for_ip(_req("x" * 32)) == set()
    assert gw.allow_for_ip(_req(None)) == set()


def test_keyless_loopback_console_stays_open(keys):
    assert gw.allow_for_ip(_req(None, ip="127.0.0.1")) is None


def test_no_keys_configured_is_open(monkeypatch):
    monkeypatch.setattr(gw, "access_keys", lambda: {})
    assert gw.allow_for_ip(_req(None)) is None
