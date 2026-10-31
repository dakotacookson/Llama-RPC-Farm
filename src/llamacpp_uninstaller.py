"""llama.cpp uninstaller: removes ONLY what llamacpp_installer created.

Mirrors the B12 uninstaller contract (vllm_uninstaller.py) — run on ONE
machine (local or over SSH), produce a steps.run_steps report, and never
damage the node (card t_df5839d3's core safety property: "a failed
install must never brick the machine" — and a cleanup must never damage
it either):

- PATH GUARD: the deletion target is the llama.cpp install dir the
  installer writes to — DEST_POSIX (~/llama.cpp) on POSIX or DEST_WIN
  (%USERPROFILE%\\llama.cpp) on Windows, identical constants imported
  from llamacpp_installer so uninstall and install always agree on what
  directory is theirs. A machine record pointing elsewhere is refused
  and NOTHING is deleted; never `rm -rf` beyond the guarded root.
  System python, CUDA, drivers, the llama-swap daemon and model stores
  are never touched (nothing here even invokes them).
- Idempotent: an already-absent install reports the success step
  "nothing to remove" instead of failing.
- Only the bin dir the installer populated is removed
  (~/llama.cpp/bin), never a source build (~/llama.cpp/build — those
  builds were not created by the installer; a source tree is treated as
  a foreign path and the run refuses, same guard semantics).
- The Windows Firewall allow rules install_on creates (DisplayName
  prefix "Llama RPC Farm - ") are removed best-effort on Windows; a
  failure leaves them in place (they name a deleted binary — harmless)
  and never fails the run.

On success the machine's llama_bin pointer is cleared (load-modify-write
via cfg.machines.update, unrelated fields untouched) so no later launch
probe chases a dead binary.

install_on's failure rollback calls uninstall_on(...) so a post-install
verify failure (binary missing / --version crash) leaves the host in its
pre-install state.
"""
import os

import local_node
from steps import run_steps

from llamacpp_installer import DEST_POSIX, DEST_WIN, _win_fw_remove_ps
from vllm_installer import _finish, _make_runner

GUARD_TOKEN = "llama.cpp"   # the installer's directory name


# ---------------- path guard -------------------------------------------------

def _is_posix(machine):
    return machine.get("os", "linux") != "windows"


def _default_root(machine):
    """The installer's own install dir for this machine type."""
    return DEST_WIN if not _is_posix(machine) else DEST_POSIX


def _bin_dir_name(is_posix):
    """Bin subdir inside the guarded root the installer populates."""
    return "bin" if is_posix else "bin"


def _configured_bin_dir(machine):
    """The bin dir derived from the machine's llama_bin pointer, or None
    when the record names no path. Accepts both the dir form the
    installer writes (~/llama.cpp/bin) and a full binary path."""
    raw = (machine.get("llama_bin") or "").strip().rstrip("/\\")
    if not raw:
        return None
    for tail in ("llama-server.exe", "llama-server"):
        if raw.endswith("/" + tail) or raw.endswith("\\" + tail):
            raw = raw[: -len(tail) - 1]
            break
    return raw.rstrip("/\\") or raw


def _is_farm_bin_dir(bin_dir, machine):
    """True only for the bin dir the installer writes: its own DEST
    (POSIX or Windows form), with ~/ / $HOME / %USERPROFILE% expanded.
    The installer never writes anywhere else, so uninstall agrees on
    the deletion target by construction and `rm -rf` of an arbitrary
    directory is structurally impossible. A source build
    (~/llama.cpp/build/bin), an install in a foreign parent
    (~/models/llama.cpp/bin) or the parent root itself is NOT
    farm-managed and is refused."""
    cand = (bin_dir or "").strip().rstrip("/\\")
    if not cand:
        return False
    if _is_posix(machine):
        home = os.path.expanduser("~")
        if cand.startswith("$HOME"):
            cand = home + cand[len("$HOME"):]
        elif cand.startswith("~"):
            cand = home + cand[1:]
        return os.path.normpath(cand) == os.path.normpath(
            os.path.expanduser(DEST_POSIX))
    # windows: case-insensitive compare after %USERPROFILE% collapse
    dest = DEST_WIN.replace("%USERPROFILE%", "UP").replace("\\", "/").lower()
    cc = cand.replace("%USERPROFILE%", "UP").replace("\\", "/").lower()
    return cc.rstrip("/") == dest.rstrip("/")


def _guarded_bin_dir(machine):
    """The bin dir to remove, or None when the machine's llama_bin is
    not the farm-managed install (uninstall step then refuses)."""
    cand = _configured_bin_dir(machine)
    return cand if cand and _is_farm_bin_dir(cand, machine) else None


# ---------------- shell helpers ----------------------------------------------

def _dq(p):
    return '"' + _shellpath(p) + '"'


def _shellpath(p):
    """Settings path safe for a bash -c script: a leading ~ becomes $HOME
    (verbatim from vllm_uninstaller so both sides expand identically)."""
    p = (p or "").strip()
    if p == "~":
        return "$HOME"
    if p.startswith("~/"):
        return "$HOME/" + p[2:]
    return p


def _ps_path(p):
    """Windows-safe path literal for a PowerShell -Command script: the
    %USERPROFILE% form passes through untouched (PowerShell does not
    expand cmd-style env syntax, but the installer's own scripts use the
    literal '$d=...' form below, which resolves at expand time)."""
    return p


# ---------------- steps ------------------------------------------------------

def _bin_probe(run, bin_file, is_posix):
    """Does the llama-server binary still exist on the host? (Idempotency
    probe — pure existence check, nothing removed here.)"""
    if is_posix:
        cmd = f"test -f {_dq(bin_file)} && echo present || echo absent"
    else:
        cmd = (f"if (Test-Path '{bin_file}') {{ 'present' }} "
               f"else {{ 'absent' }}")
    code, out = run(cmd, timeout=30)
    text = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
    if code != 0:
        return {"failed": True,
                "detail": f"existence probe for {bin_file} failed: "
                          f"{text[:200]}"}
    return {"present": "absent" not in text}


def _version_probe(run, bin_file, is_posix):
    """The installed binary must still answer --version before removal —
    this doubles as the INSTALLER's post-install verify (rollback calls
    uninstall only after a failed verify, so never probe a dead binary
    into a rollback loop)."""
    cmd = (f'& "{bin_file}" --version' if not is_posix
           else f"{_dq(bin_file)} --version")
    code, out = run(cmd, timeout=60)
    if code != 0:
        return {"failed": True,
                "detail": f"llama-server --version failed: "
                          f"{(out or '').strip()[-200:]}"}
    text = (out or "").strip()
    return {"detail": f"llama-server responds: "
                      f"{(text.splitlines() or [''])[0][:80]}"}


def _bin_remove(run, bin_dir, is_posix):
    """Delete the installer's bin dir — after an existence probe.
    Idempotent: an already-absent dir reports the success step
    "nothing to remove" and runs NO rm command. The command removes
    exactly the guarded bin dir and nothing else (no wildcards, no
    system paths)."""
    if is_posix:
        probe_cmd = f"test -d {_dq(bin_dir)} && echo present || echo absent"
        rm_cmd = f"rm -rf {_dq(bin_dir)}"
    else:
        probe_cmd = (f"if (Test-Path '{bin_dir}') {{ 'present' }} "
                     f"else {{ 'absent' }}")
        rm_cmd = f"Remove-Item -Recurse -Force '{bin_dir}'"
    code, out = run(probe_cmd, timeout=30)
    text = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
    if code != 0:
        return {"failed": True,
                "detail": f"existence probe for {bin_dir} failed: "
                          f"{text[:200]}"}
    if text != "present":
        return {"detail": "nothing to remove — llama.cpp bin dir already "
                          "absent (idempotent no-op)"}
    code, out = run(rm_cmd, timeout=120)
    if code != 0:
        return {"failed": True,
                "detail": f"removing bin dir {bin_dir} failed: "
                          f"{(out or '').strip()[-300:]}"}
    return {"detail": f"bin dir {bin_dir} removed"}


def _fw_rules_remove(run, is_posix):
    """Remove the Windows Firewall rules install_on created (they are
    installer-created state, so uninstall owns them — DisplayName prefix
    match only, never a user's own rules). Best-effort BY DESIGN: a
    leftover allow rule merely names a now-deleted binary (harmless), so
    a failure here must never fail the uninstall — and never fail the
    install-rollback that calls it."""
    if is_posix:
        return {"detail": "no firewall rules on this OS "
                          "(installer creates them on Windows only)"}
    # run comes from vllm_installer._make_runner (raw SSH): Windows
    # peers' OpenSSH default shell is cmd.exe, so the PowerShell body
    # needs the explicit wrapper — same convention as rpc_manager.
    cmd = 'powershell -NoProfile -Command "' \
          + _win_fw_remove_ps().replace('"', '\\"') + '"'
    code, out = run(cmd, timeout=60)
    if code != 0:
        return {"detail": "firewall rules left in place (removal needs "
                          "an admin session; harmless — they name a "
                          f"deleted binary): {(out or '').strip()[-120:]}"}
    return {"detail": "installer firewall rules removed"}


def _clear_llama_bin(machine):
    """Clear the machine's llama_bin pointer (load-modify-write via the
    config layer) — unrelated fields survive untouched. The binary the
    pointer named is gone, so leaving it would make every later probe
    or launch chase a dead path."""
    import config_loader as cfg
    name = machine.get("name", "")
    try:
        cfg.machines.update(name, {"llama_bin": ""})
    except Exception as e:
        return {"failed": True, "detail": f"clearing llama_bin failed: {e}"}
    return {"detail": f"machine record {name}: llama_bin cleared"}


# ---------------- entry point ------------------------------------------------

def uninstall(machine, log=None):
    """Uninstall llama.cpp from ONE machine — the inverse of install_on.

    Returns a steps report {machine, target, steps[], ok, failed,
    detail}, same shape as the vLLM uninstaller's. The install dir is
    deleted ONLY when the path guard says it is farm-managed; a foreign
    llama_bin fails the run with the guard message and never deletes.
    """
    machine = machine or {}
    report = {"machine": machine.get("name", "?"), "steps": [],
              "failed": False}
    is_posix = _is_posix(machine)
    bin_dir = _guarded_bin_dir(machine)
    if bin_dir is None:
        # PATH GUARD: the machine record names an install dir the
        # installer would never have created — refuse to delete
        # anything at all.
        report["target"] = machine.get("llama_bin") or None
        report["steps"] = [{"step": "guard:path",
                            "failed": True,
                            "detail": (
                                "refusing to remove llama.cpp: path is "
                                "not the farm-managed install (expected "
                                f"{_default_root(machine)}) — nothing "
                                "was deleted")}]
        return _finish_uninstall(report)
    if log:
        log(f"llama.cpp uninstall on {report['machine']} (dir={bin_dir})")
    run = _make_runner(machine)
    bin_file = (bin_dir + "/llama-server") if is_posix \
        else (bin_dir + "\\llama-server.exe")
    report["target"] = bin_file
    steps = [("guard:path", lambda: {"detail": f"bin dir {bin_dir} is "
                                                "farm-managed"}),
             ("verify:version", _version_probe, run, bin_file, is_posix),
             ("bins:remove", _bin_remove, run, bin_dir, is_posix),
             ("firewall:rules", _fw_rules_remove, run, is_posix),
             ("machines:clear", _clear_llama_bin, machine)]
    run_steps(report, steps)
    return _finish_uninstall(report)


def _finish_uninstall(report):
    """Roll the step report up (installer's _finish semantics), with
    uninstall-specific closing detail."""
    out = _finish(report)
    if out.get("ok") and out.get("detail") == "install complete":
        out["detail"] = "llama.cpp uninstall complete"
    return out


# vllm_uninstaller contract name, for call-site symmetry.
uninstall_on = uninstall