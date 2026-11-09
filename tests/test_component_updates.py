"""Per-component update decisions: build commits beat release tags."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402


def test_matching_commit_is_on_latest_despite_newer_tag():
    # App-only release: agent_commit did not move, so the agent is current
    # even though its stamped version differs from the newest release tag.
    assert app_module._component_on_latest(
        "a" * 40, "a" * 40, "1.0.31", "1.1.0") is True


def test_different_commit_needs_update():
    assert app_module._component_on_latest(
        "a" * 40, "b" * 40, "1.0.31", "1.0.31") is False


def test_no_fingerprint_falls_back_to_version():
    assert app_module._component_on_latest(
        "", "", "1.0.31", "1.0.31") is True
    assert app_module._component_on_latest(
        "", "", "1.0.31", "1.0.32") is False
    # A previously installed release is accepted even if the agent's own
    # build number lags the tag (legacy standalone behavior).
    assert app_module._component_on_latest(
        "", "", "1.0.31", "1.0.32", applied="1.0.32") is True


def test_manifest_missing_falls_back(monkeypatch):
    monkeypatch.setattr(app_module, "_release_feed_versions", lambda: [])
    assert app_module._release_manifest() is None


def test_manifest_commit_helper_tolerates_no_manifest(monkeypatch):
    monkeypatch.setattr(app_module, "_release_manifest", lambda: None)
    assert app_module._manifest_commit("agent_commit") == ""
    monkeypatch.setattr(app_module, "_release_manifest",
                        lambda: {"agent_commit": " deadbeef "})
    assert app_module._manifest_commit("agent_commit") == "deadbeef"
