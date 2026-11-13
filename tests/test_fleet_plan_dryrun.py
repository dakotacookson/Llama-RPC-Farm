"""LOCAL-ONLY dry run: what would the installer pick for each machine, and
does that match the machine that is currently working?

No installs, no config writes. Opt in with:

    FARM_DRYRUN=1 .venv/bin/python -m pytest tests/test_fleet_plan_dryrun.py -s

Skipped everywhere else (CI, fresh clones), so it never needs network or
the live machine config.
"""
import os
import re
import socket
import subprocess

import pytest

LOCAL = os.environ.get("FARM_DRYRUN") == "1"
pytestmark = pytest.mark.skipif(
    not LOCAL,
    reason="local fleet dry-run only — set FARM_DRYRUN=1 to run")

MACHINES = ("aibox1", "aibox3", "Aibox2", "winbox",
            "HP-Pavilion-onvertible")


def _build_no(text):
    m = re.search(r"build\s+(\d+)", str(text or ""))
    return int(m.group(1)) if m else 0


def _tag_no(tag):
    m = re.search(r"(\d+)", str(tag or ""))
    return int(m.group(1)) if m else 0


def _list_devices(machine, probe):
    import local_node
    import agent_client
    bin_dir = (probe.get("llama_bin") or machine.get("llama_bin") or "").rstrip("/\\")
    name = "llama-server.exe" if machine.get("os") == "windows" else "llama-server"
    path = f"{bin_dir}/{name}"
    if local_node.is_local(machine):
        p = subprocess.run([path, "--list-devices"], capture_output=True,
                           text=True, timeout=60)
        return (p.stdout or "") + (p.stderr or "")
    r = agent_client.install_verify(machine, path, ["--list-devices"],
                                    timeout=60)
    return str(r.get("output") or "")


@pytest.mark.parametrize("name", MACHINES)
def test_dry_run_plan_matches_working_machine(name):
    import config_loader as cfg
    import llamacpp_installer as li

    machine = cfg.machines.get(name)
    if machine is None:
        pytest.skip(f"{name}: not in the local machine config")
    try:
        probe = li.probe_setup(machine)
    except Exception as e:
        pytest.skip(f"{name}: probe failed: {e}")
    if probe.get("kind") == "missing":
        pytest.skip(f"{name}: no working llama.cpp to compare against")

    try:
        releases = li._fetch_release_asset_sets()
    except Exception as e:
        pytest.skip(f"GitHub releases unreachable: {e}")
    if not releases:
        pytest.skip("no llama.cpp release assets found")
    latest_tag = releases[0][0]

    dec = li.decide_engine_plan(machine, probe=probe)
    actual_kind = probe.get("kind")
    version = probe.get("version") or ""
    print(f"\n{name}: plan={dec['mode']} backend={li.backend_pref(machine)} "
          f"actual={actual_kind} version={version} latest={latest_tag}")

    # 1. The source-of-truth decision must agree with the working install.
    assert dec["mode"] == actual_kind, (
        f"{name}: plan says {dec['mode']} but working install is "
        f"{actual_kind} ({dec['reason']})")

    # 2. Version: the working install must not be NEWER than the plan.
    # Drift (plan newer) is reported, not fatal — upstream releases daily.
    # FARM_DRYRUN_STRICT=1 requires exact equality.
    planned_tag = latest_tag
    planned_no = _tag_no(planned_tag)
    working_no = _build_no(version)
    strict = os.environ.get("FARM_DRYRUN_STRICT") == "1"
    if strict:
        assert working_no == planned_no, (
            f"{name}: plan would install {planned_tag}, working is {version}")
    else:
        assert working_no <= planned_no, (
            f"{name}: working build is NEWER than the plan? "
            f"{version} vs {planned_tag}")
        if working_no < planned_no:
            print(f"  DRIFT: plan {planned_tag} is newer than working {version}")

    # 3. RPC must exist in any working install.
    assert probe.get("rpc") or (probe.get("flags") or {}).get("GGML_RPC") == "ON", \
        f"{name}: ggml-rpc-server/GGML_RPC missing"

    if dec["mode"] == "prebuilt":
        glibc = li._machine_glibc(machine)
        sel = li._choose_release_for(machine, releases, glibc=glibc)
        assert not sel["missing"], (
            f"{name}: plan asset missing upstream: {sel['missing']}")
        if not strict:
            assert _build_no(version) <= _tag_no(sel["tag"]), (
                f"{name}: working build is newer than the plan asset "
                f"({version} vs {sel['tag']})")
        note = (sel.get("backend") or "").lower()
        expect = ("vulkan" if "vulkan" in note else
                  "cuda" if "cuda" in note else
                  "rocm" if "rocm" in note else "")
        devices = _list_devices(machine, probe)
        if expect:
            assert expect in devices.lower(), (
                f"{name}: plan is {sel['backend']} but devices are: "
                f"{devices.strip()[:300]}")
        return

    # 4. Source plan: flags/arch coverage must match the working build.
    args = li.source_build_args(machine)
    assert "-DGGML_RPC=ON" in args
    caps = set(li.machine_cuda_archs(machine))
    flags = probe.get("flags") or {}
    if caps:
        planned = next((a for a in args
                        if a.startswith("-DCMAKE_CUDA_ARCHITECTURES=")), "")
        planned_archs = set(re.findall(r"\d+", planned))
        assert {str(c) for c in caps} <= planned_archs, (
            f"{name}: plan archs {planned_archs} do not cover {caps}")
        actual_archs = set(re.findall(
            r"\d+", str(flags.get("CMAKE_CUDA_ARCHITECTURES") or "")))
        assert {str(c) for c in caps} <= actual_archs, (
            f"{name}: working build archs {actual_archs} do not cover {caps}")
        assert flags.get("GGML_CUDA") == "ON", f"{name}: CUDA not enabled"
    assert flags.get("GGML_RPC") == "ON" or probe.get("rpc"), \
        f"{name}: GGML_RPC not enabled in the working source build"
