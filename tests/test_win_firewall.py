# tests/test_win_firewall.py — installer-created Windows Firewall rules.
#
# Regression for the 2026-10-26 winbox outage: the llama.cpp upgrade
# moved the Windows binaries from llama.cpp\ root into llama.cpp\bin\,
# the pre-upgrade program-scoped allow rule stopped matching the
# listening exe, and Windows' default inbound policy silently DROPPED
# the RPC port's SYNs (no RST) — every winbox-backed model hung at
# load forever. install_on now re-ensures the rules against the ACTUAL
# installed path after every Windows install/upgrade, and the
# uninstaller removes exactly those rules.

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import llamacpp_installer as lci                  # noqa: E402
from llamacpp_uninstaller import _fw_rules_remove  # noqa: E402


class StubRun:
    """run(cmd, timeout=...) -> (code, combined output); records cmds."""

    def __init__(self, code=0, out=""):
        self.code, self.out = code, out
        self.cmds = []

    def __call__(self, cmd, timeout=60):
        self.cmds.append(cmd)
        return self.code, self.out


# ---------------- PowerShell bodies -------------------------------------

def test_ensure_body_has_no_double_quotes():
    # run_remote wraps the body as powershell -NoProfile -Command "..."
    # with \" escaping for embedded quotes (cmd.exe list2cmdline would
    # mangle them) — the body must be double-quote-free to survive.
    assert '"' not in lci._win_fw_ensure_ps()
    assert '"' not in lci._win_fw_remove_ps()


def test_ensure_body_targets_installed_path():
    body = lci._win_fw_ensure_ps()
    # keyed on the ACTUAL installed exe path (llama.cpp\bin\), the very
    # path the 2026-10 upgrade moved the binaries to
    assert "llama.cpp\\bin\\" in body
    assert "Get-NetFirewallApplicationFilter" in body
    assert "New-NetFirewallRule" in body
    assert f"'{lci.WIN_FW_RULE_PREFIX}'" in body
    for exe in lci.WIN_FW_EXES:
        assert f"'{exe}'" in body


def test_remove_body_only_touches_farm_rules():
    body = lci._win_fw_remove_ps()
    assert "Remove-NetFirewallRule" in body
    # removal keyed on the farm DisplayName prefix, never on Program
    # wildcards — a user's own rules for the same exe survive
    assert "Get-NetFirewallRule -DisplayName" in body
    assert f"'{lci.WIN_FW_RULE_PREFIX}'" in body


# ---------------- _ensure_win_firewall result parsing ------------------

def test_ensure_reports_added_rules():
    run = StubRun(out="ggml-rpc-server.exe:added llama-server.exe:ok")
    msgs = []
    note = lci._ensure_win_firewall(run, msgs.append)
    assert "ggml-rpc-server.exe" in note and "added" in note
    assert any("allow rule added" in m for m in msgs)


def test_ensure_warns_when_admin_required():
    run = StubRun(out="ggml-rpc-server.exe:failed llama-server.exe:failed")
    note = lci._ensure_win_firewall(run, lambda m: None)
    assert "WARNING" in note and "admin" in note
    assert "unreachable" in note


def test_ensure_silent_when_rules_already_present():
    run = StubRun(out="ggml-rpc-server.exe:ok llama-server.exe:ok")
    assert lci._ensure_win_firewall(run, lambda m: None) == ""


def test_ensure_warns_when_probe_gives_no_tokens():
    # PowerShell died (e.g. module missing): no exe:state tokens parsed —
    # a silent "" here is exactly the bug class that hid the outage.
    run = StubRun(code=1, out="PowerShell : command not found")
    note = lci._ensure_win_firewall(run, lambda m: None)
    assert "WARNING" in note and "ggml-rpc-server.exe" in note


def test_ensure_run_failure_never_raises():
    class BoomRun:
        def __call__(self, cmd, timeout=60):
            raise RuntimeError("ssh down")
    assert lci._ensure_win_firewall(BoomRun(), lambda m: None) == ""


# ---------------- uninstaller cleanup -----------------------------------

def test_uninstall_removes_rules_on_windows():
    run = StubRun(out="fw-rules-removed")
    rep = _fw_rules_remove(run, is_posix=False)
    assert not rep.get("failed")
    assert "removed" in rep["detail"]
    # explicit wrapper: run comes from _make_runner's raw SSH, and the
    # peer's default shell is cmd.exe (AGENTS.md OS coverage)
    assert run.cmds[0].startswith('powershell -NoProfile -Command "')
    assert lci._win_fw_remove_ps() in run.cmds[0]


def test_uninstall_rule_removal_is_best_effort():
    # non-admin session: removal fails but the uninstall must not
    run = StubRun(code=1, out="access denied")
    rep = _fw_rules_remove(run, is_posix=False)
    assert not rep.get("failed")
    assert "left in place" in rep["detail"]


def test_uninstall_rules_posix_noop():
    rep = _fw_rules_remove(StubRun(), is_posix=True)
    assert not rep.get("failed")
    assert "Windows" in rep["detail"]