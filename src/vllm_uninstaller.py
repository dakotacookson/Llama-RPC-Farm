"""vLLM uninstaller: removes ONLY what vllm_installer created.

Mirrors llamacpp_installer.py's contract — run on ONE machine (local or
over SSH), report a steps.run_steps report — in reverse: the installer
creates a dedicated venv and points settings vllm.venv_bin at it, so the
uninstaller removes that venv (guarded) and clears the settings key
(load-modify-write, unrelated keys preserved byte-for-byte in YAML
round-trip terms).

The core safety property (card t_df5839d3): "a failed install must never
brick the machine" — and a cleanup must never damage it either. So:

- venv delivery: the venv root named by vllm.venv_bin is removed ONLY if
  it is a venv the installer would have created — the derived root must
  equal DEFAULT_VENV_ROOT (the installer's own fallback) or carry the
  installer's `vllm-env` directory name. Anything else is refused and
  nothing at all is deleted; NEVER `rm -rf` beyond the guarded venv
  root. System python, pip, drivers and CUDA are never touched (nothing
  here even invokes them).
- docker delivery: DEAD CODE this wave (USER DIRECTIVE 2026-09-24 —
  venv/PyPI only). Refuses with the installer's verbatim message; never
  runs `docker rmi` anything.
- Idempotent: an already-absent venv reports the success step
  "nothing to remove" instead of failing.

install_on's failure rollback calls uninstall_on(machine, ...) so a
post-install probe failure (import vllm broken) leaves the host in its
pre-install state (minus the pip download cache — acceptable residue,
noted in the rollback step detail).
"""
import os
import shlex

import gpu_vendor
import local_node
from steps import run_steps

from vllm_installer import (
    DEFAULT_VENV_ROOT,
    _DOCKER_UNSUPPORTED,
    _finish,
    _make_runner,
    _vllm_block,
)

VENV_GUARD_TOKEN = "vllm-env"   # the installer's directory name


# ---------------- path guard -------------------------------------------------

def _venv_root(venv_bin):
    """Dedicated-venv root from a venv_bin path — identical derivation to
    vllm_installer._venv_root ('~/x/bin/vllm' -> '~/x'; anything else
    falls back to the farm-managed default) so uninstall and install
    always agree on what directory is theirs."""
    vb = (venv_bin or "").strip()
    if "/bin/" in vb:
        return vb.rsplit("/bin/", 1)[0]
    return DEFAULT_VENV_ROOT


def _is_farm_venv(root):
    """True only for a venv the installer would have created.

    Guard on the ROOT (not venv_bin): the installer derives the venv root
    the same way, so the two must agree on the deletion target. The
    farm-managed default root always qualifies; other roots must carry
    the installer's `vllm-env` directory name somewhere in their path.
    Everything else is refused — this function is the whole reason
    uninstall cannot `rm -rf` an arbitrary directory."""
    root = (root or "").strip()
    if not root:
        return False
    if root.rstrip("/") == DEFAULT_VENV_ROOT.rstrip("/"):
        return True
    return VENV_GUARD_TOKEN in os.path.basename(root.rstrip("/")) or \
        VENV_GUARD_TOKEN in root.split("/")[-1]


def _guarded_venv_root(vllm_cfg):
    """The venv root to remove, or None when the configured venv_bin is
    not a farm-managed path (uninstall step then refuses to delete)."""
    root = _venv_root(vllm_cfg.get("venv_bin"))
    return root if _is_farm_venv(root) else None


def _dq(p):
    return '"' + _shellpath(p) + '"'


def _shellpath(p):
    """Settings path safe for a bash -c script: a leading ~ becomes $HOME
    (verbatim from vllm_installer so both sides expand identically)."""
    p = (p or "").strip()
    if p == "~":
        return "$HOME"
    if p.startswith("~/"):
        return "$HOME/" + p[2:]
    return p


# ---------------- steps ------------------------------------------------------

def _venv_probe(run, root):
    """Does the guarded venv still exist on the host? (Idempotency probe
    — pure existence check, nothing removed here.)"""
    code, out = run(f"test -d {_dq(root)} && echo present || echo absent",
                    timeout=30)
    text = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
    if code != 0:
        return {"failed": True,
                "detail": f"existence probe for {root} failed: {text[:200]}"}
    return {"present": text != "absent"}


def _venv_remove(run, root):
    """Delete the guarded venv directory — after an existence probe.
    Idempotent: an already-absent venv reports the success step
    "nothing to remove" and runs NO rm command. The command removes
    exactly the guarded root and nothing else (no wildcards, no system
    paths)."""
    probe = _venv_probe(run, root)
    if probe.get("failed"):
        return probe
    if not probe.get("present"):
        return {"detail": "nothing to remove — venv already absent "
                          "(idempotent no-op)"}
    code, out = run(f"rm -rf {_dq(root)}", timeout=120)
    if code != 0:
        return {"failed": True,
                "detail": f"removing venv {root} failed: "
                          f"{(out or '').strip()[-300:]}"}
    return {"detail": f"venv {root} removed"}


def _clear_settings_key(settings_path, root):
    """Clear settings vllm.venv_bin (load-modify-write) —
    Comment-preserving round-trip (DEC-14, B9c pattern): the SAME
    persister app.py uses, so unrelated keys, key order, and comments
    survive (a plain yaml.safe_dump here wiped every settings comment
    on each uninstall). The venv the key pointed at is gone, so leaving
    it would make every later engine start probe a dead binary."""
    import engine_registry
    import settings_persister
    path = settings_path or engine_registry.SETTINGS_FILE
    data = settings_persister.load_yaml_preserving(path)
    if not isinstance(data, dict):
        data = {}
    block = data.get("vllm") or {}
    if not isinstance(block, dict) or not block:
        return {"detail": "settings vllm.venv_bin already clear"}
    if "venv_bin" not in block:
        return {"detail": "settings vllm.venv_bin already absent"}
    # the key being cleared must name the venv that was (or would have
    # been) removed — never a foreign path
    pointed_root = _venv_root(block.get("venv_bin"))
    if pointed_root.rstrip("/") != root.rstrip("/"):
        return {"detail": f"settings vllm.venv_bin points elsewhere "
                          f"({block.get('venv_bin')}) — left untouched"}
    block.pop("venv_bin", None)
    data["vllm"] = block
    settings_persister.dump_yaml_preserving(data, path)
    return {"detail": f"settings vllm.venv_bin cleared (was {root}/bin/vllm)"}


# ---------------- docker (DEAD CODE this wave) -------------------------------

def _docker_remove(run, image):
    """Unreachable from uninstall_on (delivery: docker refuses at
    gate:delivery first, exactly like the installer). Kept verbatim so a
    future wave can re-enable docker without archaeology — it is the
    only place a `docker rmi` would ever be issued, guarded by an exact
    image-name match against the settings value."""
    code, out = run(f"docker rmi {shlex.quote(image)}", timeout=120)
    if code != 0:
        text = (out or "").strip()
        if "No such image" in text or "no such image" in text:
            return {"detail": f"docker image {image} already absent"}
        return {"failed": True,
                "detail": f"docker rmi {image} failed: {text[-300:]}"}
    return {"detail": f"docker image {image} removed"}


# ---------------- entry point ------------------------------------------------

def uninstall(machine, log=None, settings_path=None):
    """Uninstall vLLM from ONE machine — the inverse of install_on.

    Returns a steps report {machine, delivery, target, steps[], ok,
    failed, detail}, same shape as the installer's. USER DIRECTIVE
    (2026-09-24): only the PyPI venv delivery exists — delivery: docker
    refuses before any command runs. The venv root is deleted ONLY when
    the path guard says it is farm-managed; a foreign venv_bin fails
    the run with the guard message and never deletes.
    """
    machine = machine or {}
    report = {"machine": machine.get("name", "?"), "steps": [],
              "failed": False}
    try:
        vllm_cfg = _vllm_block(settings_path)
    except ValueError as e:
        report.update(delivery=None, ok=False, detail=str(e),
                      steps=[{"step": "load:settings", "error": str(e)}])
        return report
    delivery = vllm_cfg.get("delivery") or "venv"
    report["delivery"] = delivery
    if delivery == "docker":
        report["target"] = None
        report["steps"] = [{"step": "gate:delivery",
                            "failed": True,
                            "detail": _DOCKER_UNSUPPORTED}]
        return _finish_uninstall(report)
    if log:
        log(f"vLLM uninstall on {report['machine']} (delivery={delivery})")
    run = _make_runner(machine)

    root = _guarded_venv_root(vllm_cfg)
    if root is None:
        # PATH GUARD: settings name a venv the installer would never
        # have created — refuse to delete anything at all.
        report["target"] = vllm_cfg.get("venv_bin")
        report["steps"] = [{"step": "guard:path",
                            "failed": True,
                            "detail": (
                                "refusing to remove venv: path is not a "
                                "farm-managed vLLM venv (expected under "
                                f"{DEFAULT_VENV_ROOT} or a *{VENV_GUARD_TOKEN}* "
                                "directory) — nothing was deleted")}]
        return _finish_uninstall(report)

    report["target"] = f"{root}/bin/vllm"
    steps = [("guard:path", lambda: {"detail": f"venv root {root} is "
                                               "farm-managed"}),
             ("venv:remove", _venv_remove, run, root)]
    if settings_path is not None:
        steps.append(("settings:clear", _clear_settings_key,
                      settings_path, root))
    run_steps(report, steps)
    return _finish_uninstall(report)


def _finish_uninstall(report):
    """Roll the step report up (installer's _finish semantics), with
    uninstall-specific closing detail."""
    out = _finish(report)
    if out.get("ok") and out.get("detail") == "install complete":
        out["detail"] = "vLLM uninstall complete"
    return out


# llamacpp_installer contract name, for call-site symmetry.
uninstall_on = uninstall