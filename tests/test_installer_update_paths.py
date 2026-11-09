"""Installer/update path + version checks.

Regression for the aibox3 bug: a machine can have an installer prebuilt in
<root>/bin while the CUDA source build lives in <root>/build/bin. The probe
must classify the source build (and point at its bin), otherwise the
Update button installs a Vulkan prebuilt and never rebuilds the engine.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import llamacpp_installer  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "peer_agent"))
import llama_peer_agent.build as agent_build  # noqa: E402


def test_version_matches_release_tag():
    line = "version: 0.9.0-dev (build 12224, commit 0123abc)"
    assert llamacpp_installer._version_matches_tag(line, "b12224") is True
    assert llamacpp_installer._version_matches_tag(line, "b12211") is False
    # Unknown ref shape (e.g. origin/master) must not block an update.
    assert llamacpp_installer._version_matches_tag(line, "origin/master") is True


def test_agent_probe_finds_source_build_beside_prebuilt(tmp_path, monkeypatch):
    root = tmp_path / "llama.cpp"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "llama-server").write_text("prebuilt")
    (root / "build" / "bin").mkdir(parents=True)
    (root / "build" / "bin" / "llama-server").write_text("source")
    (root / "build" / "CMakeCache.txt").write_text(
        "GGML_CUDA:BOOL=ON\nGGML_RPC:BOOL=ON\n", encoding="utf-8")

    monkeypatch.setattr(agent_build.files, "_path",
                        lambda config, raw: root / "bin")
    monkeypatch.setattr(agent_build, "_bin_meta",
                        lambda lb: ("version: 0.9.0-dev (build 12224)", True))
    monkeypatch.setattr(agent_build, "_run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=str(root), stderr=""))

    out = agent_build.probe(SimpleNamespace(), "~/llama.cpp/bin")
    assert out["kind"] == "source"
    assert out["build_dir"] == str(root / "build")
    assert out["llama_bin"] == str(root / "build" / "bin")
    assert out["flags"].get("GGML_CUDA") == "ON"


def test_agent_probe_keeps_prebuilt_when_no_source_cache(tmp_path,
                                                         monkeypatch):
    root = tmp_path / "llama.cpp"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "llama-server").write_text("prebuilt")

    monkeypatch.setattr(agent_build.files, "_path",
                        lambda config, raw: root / "bin")
    monkeypatch.setattr(agent_build, "_bin_meta",
                        lambda lb: ("version: 0.9.0-dev (build 12224)", True))

    out = agent_build.probe(SimpleNamespace(), "~/llama.cpp/bin")
    assert out["kind"] == "prebuilt"
    assert out["llama_bin"] == str(root / "bin")
