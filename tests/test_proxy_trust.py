"""Reverse-proxy identity: forwarded headers only from trusted proxies."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from starlette.requests import Request

import access
import swap_gateway as gw


def _req(client, headers=None):
    return Request({"type": "http", "method": "GET", "path": "/",
                    "headers": headers or [],
                    "client": (client, 40000)})


def _reset():
    access.set_view_allow(None)
    access.set_restricted(False)


def test_forwarded_ip_trusted_from_loopback_proxy(monkeypatch):
    monkeypatch.setattr(access, "access_keys",
                        lambda: {"sk": {"name": "x"}})
    scope = {"type": "http", "client": ("127.0.0.1", 1),
             "headers": [(b"x-forwarded-for", b"203.0.113.9")]}
    assert access.peer_ip_scope(scope) == "203.0.113.9"
    access.restrict_scope_from_key(scope, "")
    try:
        assert access.gate_open() is False
    finally:
        _reset()


def test_require_auth_blocks_remote_before_keys(monkeypatch):
    monkeypatch.setattr(access, "access_keys", lambda: {})
    monkeypatch.setattr(access, "_REQUIRE_AUTH", True)
    scope = {"type": "http", "client": ("203.0.113.9", 1),
             "headers": []}
    access.restrict_scope_from_key(scope, "")
    try:
        assert access.gate_open() is False
    finally:
        _reset()
    loop = {"type": "http", "client": ("127.0.0.1", 1), "headers": []}
    access.restrict_scope_from_key(loop, "")
    assert access.gate_open() is True


def test_gateway_require_keys_blocks_remote_before_keys(monkeypatch):
    monkeypatch.setattr(gw, "access_keys", lambda: {})
    monkeypatch.setattr(gw, "_REQUIRE_KEYS", True)
    assert gw.allow_for_ip(_req("203.0.113.9")) == set()
    assert gw.allow_for_ip(_req("127.0.0.1")) is None


def test_spoofed_forwarded_ip_from_remote_is_ignored():
    scope = {"type": "http", "client": ("203.0.113.9", 1),
             "headers": [(b"x-forwarded-for", b"127.0.0.1")]}
    assert access.peer_ip_scope(scope) == "203.0.113.9"


def test_gateway_forwarded_ip_requires_trusted_proxy():
    req = _req("127.0.0.1", [(b"x-forwarded-for", b"203.0.113.9")])
    assert gw.client_ip(req) == "203.0.113.9"
    req2 = _req("203.0.113.9", [(b"x-forwarded-for", b"127.0.0.1")])
    assert gw.client_ip(req2) == "203.0.113.9"


def test_cookie_secure_only_from_trusted_https_proxy():
    req = _req("127.0.0.1", [(b"x-forwarded-proto", b"https")])
    assert access.cookie_secure(req) is True
    req2 = _req("203.0.113.9", [(b"x-forwarded-proto", b"https")])
    assert access.cookie_secure(req2) is False
    req3 = _req("127.0.0.1", [(b"x-forwarded-proto", b"http")])
    assert access.cookie_secure(req3) is False


def test_cookie_secure_env_override(monkeypatch):
    monkeypatch.setenv("FARM_COOKIE_SECURE", "1")
    assert access.cookie_secure(_req("203.0.113.9")) is True
