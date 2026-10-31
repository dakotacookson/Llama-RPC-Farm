"""vLLM installer: dedicated venv (default) or docker image, with gates.

Mirrors llamacpp_installer.py's contract — install on ONE machine (local
or over SSH), skip when already present, update the farm settings on
success — but the deliverable is a Python ecosystem instead of a prebuilt
binary, so every step is a gate or a package-manager command and the
result is a steps.run_steps report (ARD t_76f3d1c0 §5). Windows/macOS
hosts are refused before any command runs (research: no native Windows
support — WSL2 only; vLLM-Metal is a separate package the farm does not
install).

Delivery comes from settings.yaml `vllm:` (schema owned by config_loader):
- venv (default): create the venv named by vllm.venv_bin (uv when
  available, else python3 -m venv) -> install the pinned vllm wheel
  (ROCm hosts get the ROCm wheel index) -> smoke `vllm --version` ->
  point settings vllm.venv_bin at the installed binary.
- docker: verify docker works -> `docker pull` the configured image.

Gates (research pitfalls -> hard checks, never silent):
- nvidia: nvidia-smi must answer before anything is installed
- amd/ROCm: rocm-smi/amd-smi present AND python3 is 3.12 (ROCm wheels
  are 3.12-only; other pythons silently receive the CUDA wheel), plus a
  post-install probe that imports vllm and asserts torch.version.hip —
  the silent-CUDA-fallback guard.
- intel/cpu hosts: refused (no farm-validated wheel path)

The experimental vllm-gguf-plugin is NOT auto-installed (opt-in only).
"""
import shlex

import gpu_vendor
import local_node
from steps import run_steps
# NOTE: vllm_uninstaller imports helpers back from this module, so it is
# imported lazily inside _rollback_failed_install (a module-level import
# here would be circular and silently resolve to None).

VLLM_VERSION_PIN = "0.30.0"
# USER DIRECTIVE (2026-09-24): vLLM installation supports ONLY the PyPI
# venv path (https://pypi.org/project/vllm/) — pip install into the
# dedicated venv. Docker delivery is dead code this wave: install/plan
# refuse it cleanly with the message below (kept verbatim for test
# parity); the docker plumbing stays but is unreachable from settings
# validation until a future wave flips VALID_VLLM_DELIVERY.
_DOCKER_UNSUPPORTED = ("docker delivery not supported yet — use venv "
                       "(PyPI)")
DEFAULT_VENV_ROOT = "~/vllm-env"
ROCM_INDEX_URL = "https://wheels.vllm.ai/rocm/"
GATE_TIMEOUT = 30          # probes / version smokes
VENV_TIMEOUT = 300         # venv creation
BACKEND_TIMEOUT = 300      # `import vllm` cold start
INSTALL_TIMEOUT = 1800     # pip install / docker pull

# Post-install backend probe (runs inside the fresh venv): imports vllm
# for real and reports which GPU backend the bundled torch was built for
# — torch.version.hip is set exactly on ROCm builds.
_BACKEND_PROBE = ('import vllm, torch; print("vllm", vllm.__version__, '
                  '"backend", "HIP" if getattr(torch.version, "hip", None) '
                  'else "CUDA")')


def _shellpath(p):
    """Settings path safe for a bash -c script: a leading ~ becomes $HOME
    (paths are double-quoted at call sites — single quotes would freeze
    the expansion, and shlex.quote would do exactly that)."""
    p = (p or "").strip()
    if p == "~":
        return "$HOME"
    if p.startswith("~/"):
        return "$HOME/" + p[2:]
    return p


def _dq(p):
    return '"' + _shellpath(p) + '"'


def _vllm_block(settings_path=None):
    """Normalized settings `vllm:` block. Raises ValueError on a bad
    settings file — install() turns that into a failed report."""
    from engine_registry import load_settings
    return load_settings(settings_path).get("vllm") or {}


def _machine_delivery(machine, vllm_cfg):
    """B13 per-machine method matching: the delivery for THIS machine.

    A machine record may pin `vllm_delivery:` (venv|docker) — it wins
    over the settings global (e.g. a host without python3.12 pinned to
    docker once docker ships). Raises ValueError on a bad override value;
    callers turn that into a failed report (never a silent fallback).
    Returns (delivery, reason): reason is user-facing ("will install via
    <method>: <reason>").
    """
    import config_loader
    return config_loader.resolve_vllm_delivery(
        machine, vllm_cfg.get("delivery"))


def _make_runner(machine):
    """run(cmd, timeout) -> (exit_code, combined output): local bash or
    SSH, mirroring llamacpp_installer. vLLM is linux-only (the OS gate
    refuses everything else), so there is no PowerShell branch."""
    is_local = local_node.is_local(machine)

    def run(cmd, timeout=60):
        if is_local:
            import subprocess
            proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                                  text=True, timeout=timeout)
            return proc.returncode, proc.stdout + proc.stderr
        from ssh_manager import SSHManager
        ssh = SSHManager(machine["ip"], machine.get("username", ""),
                         machine.get("password", ""),
                         port=machine.get("port", 22))
        code, out, err = ssh.run(cmd, timeout=timeout)
        return code, out + err

    return run


# ---------------- gates (ARD §5: research pitfalls -> hard checks) ---------

def _gate_os(machine):
    os_type = ((machine or {}).get("os") or "linux").strip().lower()
    if os_type == "windows":
        return {"failed": True, "detail":
                "vLLM cannot run natively on Windows — install WSL2, add "
                "the WSL instance to the farm as a linux machine, and "
                "install vLLM there"}
    if os_type == "macos":
        return {"failed": True, "detail":
                "vLLM on macOS ships as the separate vLLM-Metal package "
                "(MLX-based, feature-limited) which the farm does not "
                "install — install it manually on the host, or run vLLM "
                "on a Linux host"}
    return {"detail": f"os {os_type}: supported"}


def _gate_gpu(machine, run):
    """Driver/tooling gate: nvidia-smi (or the ROCm equivalent) must be
    present and answering BEFORE any install is attempted."""
    vendor = gpu_vendor.normalize_vendor((machine or {}).get("gpu_vendor"))
    if vendor == "nvidia":
        code, out = run("nvidia-smi --query-gpu=driver_version,name "
                        "--format=csv,noheader", timeout=GATE_TIMEOUT)
        first = (out or "").strip().splitlines()
        if code != 0 or not first or not first[0].strip():
            return {"failed": True, "detail":
                    "nvidia-smi is missing or fails — install/verify the "
                    "NVIDIA driver before installing vLLM"}
        return {"detail": f"nvidia driver ok: {first[0].strip()}"}
    if vendor == "amd":
        code, out = run("command -v rocm-smi || command -v amd-smi",
                        timeout=GATE_TIMEOUT)
        if code != 0:
            return {"failed": True, "detail":
                    "rocm-smi/amd-smi not found — install the ROCm driver "
                    "and userspace before installing vLLM"}
        found = (out or "").strip().splitlines()[-1]
        return {"detail": f"ROCm tooling found: {found}"}
    if vendor == "intel":
        return {"failed": True, "detail":
                "Intel XPU vLLM wheels have no farm-validated install "
                "path yet — install vLLM manually or use an NVIDIA/AMD "
                "host"}
    return {"failed": True, "detail":
            f"vLLM needs an NVIDIA or AMD GPU host (gpu_vendor="
            f"{vendor or 'unset'})"}


def _gate_python_312(run):
    """ROCm wheels are Python-3.12-only; any other python silently
    installs the CUDA wheel (research pitfall) — refuse up front."""
    code, out = run("python3 --version 2>&1", timeout=GATE_TIMEOUT)
    ver = (out or "").strip()
    if code != 0 or not ver.startswith("Python 3.12"):
        return {"failed": True, "detail":
                f"ROCm vLLM wheels are Python 3.12-only (host reports "
                f"{ver or 'no python3'}) — install python3.12 and retry"}
    return {"detail": f"python {ver.split()[-1]} (ROCm requires 3.12)"}


def _gate_docker(run):
    code, out = run("docker version --format '{{.Server.Version}}'",
                    timeout=GATE_TIMEOUT)
    if code != 0:
        return {"failed": True, "detail":
                "docker is not usable on this host — install docker "
                "(plus the NVIDIA Container Toolkit for GPU access) "
                "before installing vLLM"}
    return {"detail": f"docker server {(out or '').strip()}"}


# ---------------- skip-if-present -------------------------------------------

def _detect_present(vllm_cfg, run, delivery=None):
    """Skip-if-present probe: {"skipped": True} when a working install is
    already there (venv: `vllm --version` exits 0; docker: the image
    exists locally). Anything else reports not-present and installs.
    `delivery` (B13) is the RESOLVED per-machine method; it defaults to
    the settings global for backward-compatible direct calls.

    The probe exports the venv's own nvjitlink lib dir FIRST (same
    derivation as the cluster start scripts): pip-installed torch needs
    its wheel's nvjitlink to win over a system CUDA, and without it
    `vllm --version` dies at import with an undefined-symbol error —
    a working install would falsely report "missing" (seen live on a
    cu129/AMD64 box with an older driver CUDA). The glob resolves in
    the target shell, so this works identically local and over SSH.
    """
    delivery = delivery or vllm_cfg.get("delivery") or "venv"
    if delivery == "docker":
        # venv-only wave: no docker probe — the settings block is the
        # only place docker can come from (validate_model_config keeps
        # delivery: docker out of settings entirely).
        return {"skipped": False, "detail": _DOCKER_UNSUPPORTED}
    venv_bin = vllm_cfg.get("venv_bin") or DEFAULT_VENV_ROOT + "/bin/vllm"
    root = _venv_root(venv_bin)
    probe = (f'export LD_LIBRARY_PATH="$(echo {root}/lib/python*/'
             f'site-packages/nvidia/nvjitlink/lib | cut -d\' \' -f1)'
             f'${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"; '
             f"{_dq(venv_bin)} --version")
    code, out = run(probe, timeout=GATE_TIMEOUT)
    text = (out or "").strip()
    if code == 0 and text:
        import re as _re
        ver = next((l.strip() for l in text.splitlines()
                    if _re.fullmatch(r"\d+\.\d+(\.\d+)?([.+]\\S*)?",
                                     l.strip())), None)
        return {"skipped": True,
                "detail": f"vLLM already installed: "
                          f"{ver or text.splitlines()[0]}"}
    return {"detail": f"{venv_bin} not usable yet "
                      f"({text[-120:] or 'no output'})"}


# ---------------- venv delivery ----------------------------------------------

def _has_uv(run):
    code, out = run("command -v uv >/dev/null 2>&1 && echo yes || echo no",
                    timeout=GATE_TIMEOUT)
    return (out or "").strip() == "yes"


def _venv_root(venv_bin):
    """Dedicated-venv root from a venv_bin path ('~/x/bin/vllm' -> '~/x').
    Anything else (bare command, odd layout) falls back to the
    farm-managed default so the installer always owns a real venv."""
    vb = (venv_bin or "").strip()
    if "/bin/" in vb:
        return vb.rsplit("/bin/", 1)[0]
    return DEFAULT_VENV_ROOT


def _venv_create(run, root):
    tool = "uv" if _has_uv(run) else "python3 -m venv"
    cmd = (f"uv venv --python 3.12 --seed {_dq(root)}" if tool == "uv"
           else f"python3 -m venv {_dq(root)}")
    code, out = run(cmd, timeout=VENV_TIMEOUT)
    if code != 0:
        return {"failed": True, "tool": tool,
                "detail": f"venv creation failed ({tool}): "
                          f"{(out or '').strip()[-300:]}"}
    return {"tool": tool, "detail": f"venv ready: {root} via {tool}"}


def _venv_install(run, root, vendor, pin):
    use_uv = _has_uv(run)
    spec = f"vllm=={pin}"
    index = f" --extra-index-url {ROCM_INDEX_URL}" if vendor == "amd" else ""
    if use_uv:
        # torch-backend=auto picks the torch wheel matching the NVIDIA
        # driver; on ROCm it must NOT be used (the rocm index decides).
        torch_backend = " --torch-backend=auto" if vendor == "nvidia" else ""
        cmd = (f"uv pip install --python {_dq(root + '/bin/python')} "
               f"{spec}{torch_backend}{index}")
    else:
        cmd = f"{_dq(root + '/bin/python')} -m pip install {spec}{index}"
    code, out = run(cmd, timeout=INSTALL_TIMEOUT)
    if code != 0:
        return {"failed": True,
                "detail": f"vllm=={pin} install failed: "
                          f"{(out or '').strip()[-400:]}"}
    return {"detail": f"installed vllm=={pin} into {root}"}


def _verify_version(run, venv_bin):
    code, out = run(f"{_dq(venv_bin)} --version", timeout=GATE_TIMEOUT)
    text = (out or "").strip()
    if code != 0 or not text:
        return {"failed": True,
                "detail": f"`vllm --version` failed after install: "
                          f"{text[-300:] or 'no output'}"}
    return {"detail": f"vLLM {text.splitlines()[-1]}"}


def _verify_backend(run, root):
    """ROCm-only: prove the ROCm wheel actually landed. `import vllm`
    succeeding is not enough — the CUDA wheel also imports but fails (or
    silently mis-serves) on AMD hardware, so the probe asserts
    torch.version.hip."""
    code, out = run(
        f"{_dq(root + '/bin/python')} -c {shlex.quote(_BACKEND_PROBE)}",
        timeout=BACKEND_TIMEOUT)
    text = (out or "").strip()
    if code != 0:
        return {"failed": True,
                "detail": f"`import vllm` failed after install: "
                          f"{text[-300:] or 'no output'}"}
    if "backend HIP" not in text:
        return {"failed": True,
                "detail": f"silent CUDA fallback detected ({text[-160:]}) — "
                          f"the ROCm wheel did not install (needs python "
                          f"3.12 + --extra-index-url {ROCM_INDEX_URL})"}
    return {"detail": text.splitlines()[-1]}


# ---------------- docker delivery (DEAD CODE this wave) ---------------------
# USER DIRECTIVE (2026-09-24): only the PyPI venv path is supported. These
# helpers are unreachable from install()/plan() (delivery: docker refuses
# at gate:delivery before this branch would ever be chosen) and are kept
# verbatim so a future wave can re-enable docker without archaeology.

def _docker_pull(run, image):
    code, out = run(f"docker pull {shlex.quote(image)}",
                    timeout=INSTALL_TIMEOUT)
    if code != 0:
        return {"failed": True,
                "detail": f"docker pull {image} failed: "
                          f"{(out or '').strip()[-300:]}"}
    return {"detail": f"pulled {image}"}


def _verify_image(run, image):
    code, out = run(f"docker image inspect {shlex.quote(image)}",
                    timeout=GATE_TIMEOUT)
    if code != 0:
        return {"failed": True,
                "detail": f"image {image} missing after pull: "
                          f"{(out or '').strip()[-200:]}"}
    return {"detail": f"docker image {image} present"}


# ---------------- settings ---------------------------------------------------

def _update_settings(settings_path, venv_bin):
    """Point settings vllm.venv_bin at the freshly installed binary.

    Comment-preserving round-trip (DEC-14, B9c pattern): the SAME
    persister app.py uses, so unrelated keys, key order, and comments
    survive (a plain yaml.safe_dump here wiped every settings comment
    on each install). The load_settings validation already ran at
    install() entry — a broken settings file never reaches this step."""
    import os as _os
    import engine_registry
    import settings_persister
    path = settings_path or engine_registry.SETTINGS_FILE
    data = settings_persister.load_yaml_preserving(path)
    block = data.get("vllm") or {}
    if block.get("venv_bin") == venv_bin:
        return {"detail": f"settings vllm.venv_bin already {venv_bin}"}
    block["venv_bin"] = venv_bin
    data["vllm"] = block
    settings_persister.dump_yaml_preserving(data, path)
    return {"detail": f"settings vllm.venv_bin -> {venv_bin}"}


# ---------------- entry points -----------------------------------------------

def _rollback_failed_install(machine, report, settings_path=None):
    """Failure rollback (card t_df5839d3): after venv:install, if a
    verify step (version smoke / ROCm backend probe) FAILS, the fresh
    venv is a broken install — auto-run the uninstall path so the host
    is left in its pre-install state. The uninstaller re-derives the
    venv root from the SAME settings the installer used, and its own
    path guard applies, so rollback can never delete anything the
    installer would not have created.

    Appends to report["steps"]: "rollback:uninstall" (the uninstaller's
    own run_steps detail, including the accepted pip-cache residue) plus
    a "rollback:failed" step when the cleanup itself failed. The report
    stays failed with BOTH the probe error and the rollback outcome.
    """
    # Lazy import: vllm_uninstaller imports helpers back from this
    # module, so a top-level import would be circular.
    import vllm_uninstaller

    steps = report.setdefault("steps", [])
    failed_step = next((s for s in steps
                        if s.get("error") or s.get("failed")), None)
    probe_detail = ((failed_step or {}).get("error")
                    or (failed_step or {}).get("detail") or "")
    rb_report = vllm_uninstaller.uninstall_on(machine,
                                              settings_path=settings_path)
    removed = not rb_report.get("failed")
    note = ("rolled back: venv removed (pip download cache left in place "
            "— acceptable residue)"
            if removed else
            "rollback FAILED — venv may still be present; remove it "
            "manually: " + str(rb_report.get("detail", "")))
    steps.append({"step": "rollback:uninstall",
                  **({} if removed else {"failed": True}),
                  "detail": note})
    if rb_report.get("steps"):
        steps.append({"step": "rollback:steps",
                      "detail": "; ".join(
                          f"{s.get('step')}: {s.get('detail') or s.get('error')}"
                          for s in rb_report["steps"])})
    report["detail"] = (f"{probe_detail} — {note}").strip(" —")
    return report


def install(machine, log=None, settings_path=None, pin=VLLM_VERSION_PIN):
    """Install vLLM on ONE machine (single-host tool — no fleet install,
    mirroring llamacpp_installer/swap_installer). Returns a steps report:
    {machine, delivery, method_reason, target, steps[], skipped?, ok,
    failed, detail}. USER DIRECTIVE (2026-09-24): only the PyPI venv
    delivery is supported — delivery: docker (global or per-machine
    override) refuses before any command runs.

    B13 method matching: the machine's `vllm_delivery` override (when
    set) picks the method; otherwise the settings global. The resolved
    method + reason ride the report as `method_reason` so the caller sees
    WHY this method was chosen."""
    machine = machine or {}
    report = {"machine": machine.get("name", "?"), "steps": [],
              "failed": False}
    try:
        vllm_cfg = _vllm_block(settings_path)
    except ValueError as e:
        report.update(delivery=None, ok=False, detail=str(e),
                      steps=[{"step": "load:settings", "error": str(e)}])
        return report
    try:
        delivery, method_reason = _machine_delivery(machine, vllm_cfg)
    except ValueError as e:
        report.update(delivery=None, method_reason=None, ok=False,
                      detail=str(e),
                      steps=[{"step": "gate:delivery", "failed": True,
                              "detail": str(e)}])
        return _finish(report)
    report["delivery"] = delivery
    report["method_reason"] = method_reason
    if delivery == "docker":
        report["target"] = None
        report["steps"] = [{"step": "gate:delivery",
                            "failed": True,
                            "detail": _DOCKER_UNSUPPORTED}]
        return _finish(report)
    if log:
        log(f"vLLM install on {report['machine']} (delivery={delivery})")
    run = _make_runner(machine)
    vendor = gpu_vendor.normalize_vendor(machine.get("gpu_vendor"))

    # gates first; skip-if-present before any heavy gate or install.
    # _detect_present gets the RESOLVED delivery (per-machine override
    # may pin docker → refusal, never a venv probe against a docker plan)
    run_steps(report, [("gate:os", _gate_os, machine),
                       ("verify:existing", _detect_present, vllm_cfg, run,
                        delivery)])
    existing = next((s for s in report["steps"]
                     if s["step"] == "verify:existing"), {})
    if report["failed"]:
        return _finish(report)
    if existing.get("skipped"):
        report["skipped"] = True
        report["target"] = (vllm_cfg.get("docker_image")
                            if delivery == "docker"
                            else vllm_cfg.get("venv_bin"))
        return _finish(report)

    if delivery == "docker":
        image = vllm_cfg.get("docker_image") or "vllm/vllm-openai:latest"
        report["target"] = image
        steps = [("gate:gpu", _gate_gpu, machine, run),
                 ("gate:docker", _gate_docker, run),
                 ("install:pull", _docker_pull, run, image),
                 ("verify:image", _verify_image, run, image)]
    else:
        venv_bin = vllm_cfg.get("venv_bin") or DEFAULT_VENV_ROOT + "/bin/vllm"
        root = _venv_root(venv_bin)
        target_bin = f"{root}/bin/vllm"
        report["target"] = target_bin
        steps = [("gate:gpu", _gate_gpu, machine, run)]
        if vendor == "amd":
            steps.append(("gate:python", _gate_python_312, run))
        steps += [("venv:create", _venv_create, run, root),
                  ("venv:install", _venv_install, run, root, vendor, pin),
                  ("verify:version", _verify_version, run, target_bin)]
        if vendor == "amd":
            steps.append(("verify:backend", _verify_backend, run, root))
        steps.append(("settings:update", _update_settings,
                      settings_path, target_bin))
    run_steps(report, steps)
    # Failure rollback (t_df5839d3): a failed verify AFTER venv:install
    # means a broken venv was created — auto-uninstall so a failed
    # install never bricks the node. Earlier failures (gates, venv
    # creation, the pip install itself) happen before/without a
    # half-working venv and are reported as-is.
    if report["failed"]:
        installed = next((i for i, s in enumerate(report["steps"])
                          if s["step"] == "venv:install"), None)
        first_fail = next((i for i, s in enumerate(report["steps"])
                           if s.get("error") or s.get("failed")), None)
        if (installed is not None and first_fail is not None
                and first_fail > installed):
            report = _rollback_failed_install(machine, report,
                                              settings_path=settings_path)
    return _finish(report)


def plan(machine, settings_path=None, pin=VLLM_VERSION_PIN):
    """What WOULD be installed for this machine — no commands, no side
    effects (UI preview, mirroring llamacpp_installer.plan). B13: the
    resolved method + why ride the plan (`delivery` + `method_reason` —
    machine override wins over the settings global), so the UI can show
    "will install via <method>: <reason>" BEFORE anything runs. Docker
    delivery refuses cleanly (venv-only wave, USER DIRECTIVE
    2026-09-24)."""
    machine = machine or {}
    try:
        vllm_cfg = _vllm_block(settings_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    try:
        delivery, method_reason = _machine_delivery(machine, vllm_cfg)
    except ValueError as e:
        return {"ok": False, "delivery": None, "method_reason": None,
                "target": None, "error": str(e)}
    if delivery == "docker":
        return {"ok": False, "delivery": "docker", "method_reason":
                method_reason, "target": None,
                "error": _DOCKER_UNSUPPORTED}
    gate = _gate_os(machine)
    out = {"ok": not gate.get("failed"),
           "delivery": delivery,
           "method_reason": method_reason,
           "version": pin,
           "target": ((vllm_cfg.get("docker_image")
                       or "vllm/vllm-openai:latest")
                      if delivery == "docker"
                      else (vllm_cfg.get("venv_bin")
                            or DEFAULT_VENV_ROOT + "/bin/vllm")),
           "gguf_plugin": "not auto-installed (experimental, opt-in via "
                          "vllm-gguf-plugin)"}
    if gate.get("failed"):
        out["error"] = gate["detail"]
    return out


def _finish(report):
    """Roll the step report up into ok/failed/detail (run_steps semantics:
    an 'error' key, a truthy 'failed' key, or a falsy killed/running/valid
    flag marks the failing step)."""
    failed_step = next((s for s in report.get("steps", [])
                        if s.get("error") or s.get("failed")
                        or any(s.get(k) is False
                               for k in ("killed", "running", "valid"))),
                       None)
    report["failed"] = bool(report.get("failed")) or failed_step is not None
    report["ok"] = not report["failed"]
    if failed_step and not report.get("detail"):
        report["detail"] = (failed_step.get("error")
                            or failed_step.get("detail")
                            or f"step {failed_step.get('step')} failed")
    if not report.get("detail"):
        report["detail"] = ("skipped (already present)"
                            if report.get("skipped")
                            else "install complete")
    return report


# llamacpp_installer contract name, for call-site symmetry.
install_on = install