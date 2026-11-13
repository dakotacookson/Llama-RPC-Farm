"""Group version lock: every machine in a group must be able to reach the
same latest llama.cpp tag, otherwise the whole group update is blocked."""
import types

import pytest

import llamacpp_installer as li
import routes.install as inst


def _fake_cfg(machines, groups):
    by = {m: {"name": m} for m in machines}
    return types.SimpleNamespace(
        machines=types.SimpleNamespace(get=lambda n: by.get(n)),
        groups=types.SimpleNamespace(all=lambda: groups))


def _patch(monkeypatch, machines, groups, probe, dec, sel=None, caps=None,
           vendor="nvidia", tool_missing=None, local=True):
    monkeypatch.setattr(inst, "cfg", _fake_cfg(machines, groups))
    monkeypatch.setattr(li, "probe_setup", lambda m, log=None: probe[m["name"]])
    monkeypatch.setattr(li, "decide_engine_plan",
                        lambda m, probe=None: {"mode": dec[m["name"]]})
    monkeypatch.setattr(li, "_machine_glibc", lambda m: (2, 39))
    monkeypatch.setattr(li, "_choose_release_for",
                        lambda m, releases, glibc=None: sel[m["name"]])
    monkeypatch.setattr(li, "machine_cuda_archs",
                        lambda m: caps.get(m["name"], []))
    monkeypatch.setattr(li, "_vendor_of", lambda m: vendor)
    monkeypatch.setattr(li, "source_tool_missing",
                        lambda m: tool_missing.get(m["name"], []))
    monkeypatch.setattr(inst.local_node, "is_local", lambda m: local)


RELEASES = [("b20000", ["llama-b20000-bin-ubuntu-x64.tar.gz"], "base")]


def test_group_all_capable(monkeypatch):
    machines = ["m1", "m2"]
    groups = [{"name": "g", "machines": machines}]
    _patch(monkeypatch, machines, groups,
           probe={"m1": {"kind": "prebuilt"}, "m2": {"kind": "source"}},
           dec={"m1": "prebuilt", "m2": "source"},
           sel={"m1": {"tag": "b20000", "missing": []}},
           caps={"m2": [86]})
    pf = inst._group_preflight(groups[0], RELEASES)
    assert pf["ok"] and pf["target"] == "b20000" and not pf["blockers"]


def test_one_member_asset_missing_blocks_group(monkeypatch):
    machines = ["m1", "m2"]
    groups = [{"name": "g", "machines": machines}]
    _patch(monkeypatch, machines, groups,
           probe={"m1": {"kind": "prebuilt"}, "m2": {"kind": "prebuilt"}},
           dec={"m1": "prebuilt", "m2": "prebuilt"},
           sel={"m1": {"tag": "b20000", "missing": []},
                "m2": {"tag": "b19990", "missing": []}})
    pf = inst._group_preflight(groups[0], RELEASES)
    assert not pf["ok"]
    assert any("m2" in b and "b19990" in b for b in pf["blockers"])


def test_source_member_without_toolchain_blocks_group(monkeypatch):
    machines = ["m1", "m2"]
    groups = [{"name": "g", "machines": machines}]
    _patch(monkeypatch, machines, groups,
           probe={"m1": {"kind": "prebuilt"}, "m2": {"kind": "missing"}},
           dec={"m1": "prebuilt", "m2": "source"},
           sel={"m1": {"tag": "b20000", "missing": []}},
           caps={"m2": [86]},
           tool_missing={"m2": ["nvcc (CUDA toolkit)"]})
    pf = inst._group_preflight(groups[0], RELEASES)
    assert not pf["ok"]
    assert any("nvcc" in b for b in pf["blockers"])


def test_gpu_too_old_blocks_group(monkeypatch):
    machines = ["m1", "m2"]
    groups = [{"name": "g", "machines": machines}]
    _patch(monkeypatch, machines, groups,
           probe={"m1": {"kind": "prebuilt"}, "m2": {"kind": "source"}},
           dec={"m1": "prebuilt", "m2": "source"},
           sel={"m1": {"tag": "b20000", "missing": []}},
           caps={"m2": [52]})
    pf = inst._group_preflight(groups[0], RELEASES)
    assert not pf["ok"]
    assert any("too old" in b for b in pf["blockers"])


def test_groups_for_machine():
    groups = [{"name": "a", "machines": ["m1", "m2"]},
              {"name": "b", "machines": ["m3"]}]
    inst.cfg = _fake_cfg(["m1", "m2", "m3"], groups)
    assert [g["name"] for g in inst._groups_for_machine("m1")] == ["a"]
    assert inst._groups_for_machine("m3")[0]["name"] == "b"
    assert inst._groups_for_machine("nope") == []
