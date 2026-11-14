"""Per-machine llama.cpp version cache: only the machine's build is
cached; `latest` always reflects the tag cache, and failed probes are
not cached (they used to mislabel machines for 10 minutes)."""
import pytest

import llama_version as lv


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    lv._LLAMA_VER_CACHE.clear()
    monkeypatch.setattr(lv, "_latest_llama_tag", lambda: "b12374")
    monkeypatch.setattr(lv, "_llama_version_on",
                        lambda m: "version: 1.0.0-dev (build 12370, commit x)")


def test_basic_payload():
    p = lv._llama_version_payload({"name": "m"})
    assert p["build"] == 12370 and p["latest_build"] == 12374
    assert p["on_latest"] is False


def test_latest_is_not_frozen_in_machine_cache(monkeypatch):
    p1 = lv._llama_version_payload({"name": "m"})
    assert p1["latest"] == "b12374"
    # a new upstream tag appears while the build probe is still cached
    monkeypatch.setattr(lv, "_latest_llama_tag", lambda: "b12375")
    p2 = lv._llama_version_payload({"name": "m"})
    assert p2["latest"] == "b12375"
    assert p2["build"] == 12370          # build cache still used
    assert p2["on_latest"] is False


def test_failed_probe_is_not_cached(monkeypatch):
    monkeypatch.setattr(lv, "_llama_version_on", lambda m: None)
    p = lv._llama_version_payload({"name": "m"})
    assert p["build"] is None
    # binary reappears (e.g. relink finished) -> next call recovers
    monkeypatch.setattr(
        lv, "_llama_version_on",
        lambda m: "version: 1.0.0-dev (build 12374, commit y)")
    p2 = lv._llama_version_payload({"name": "m"})
    assert p2["build"] == 12374
    assert p2["on_latest"] is True


def test_successful_probe_is_cached(monkeypatch):
    calls = {"n": 0}

    def probe(machine):
        calls["n"] += 1
        return "version: 1.0.0-dev (build 12370, commit x)"

    monkeypatch.setattr(lv, "_llama_version_on", probe)
    lv._llama_version_payload({"name": "m"})
    lv._llama_version_payload({"name": "m"})
    assert calls["n"] == 1
