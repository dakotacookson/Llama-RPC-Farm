"""llama-swap installer — mirrors llamacpp_installer.py.

Downloads the right official llama-swap prebuilt for the machine
(os + arch) and installs it (local or over SSH). Local machines (no SSH
needed) use the same shell-wrapper pattern as the llama.cpp installer.

Asset map (llama-swap releases, verified against v256):
  linux  amd64/arm64 -> llama-swap_<tag>_<os>_<arch>.tar.gz (flat tar)
  darwin amd64/arm64 -> llama-swap_<tag>_<os>_<arch>.tar.gz
  windows amd64      -> llama-swap_<tag>_windows_amd64.zip     (flat zip)

Destination: ~/.local/bin/llama-swap (POSIX) or
%USERPROFILE%\\llama-swap\\llama-swap.exe (Windows). On success the
swap settings' binary is updated (POSIX: the full resolved path, since
~/.local/bin may not be on every shell's PATH).
"""
import json
import os
import urllib.request

GITHUB_API = "https://api.github.com/repos/mostlygeek/llama-swap/releases"
DEST_POSIX = "~/.local/bin"
DEST_WIN = "%USERPROFILE%\\llama-swap"


def _fetch_latest_assets(limit=10):
    """(tag, [asset names], download_base) of the newest llama-swap
    release. Per-tag download base (NOT releases/latest/download —
    same rule as the llama.cpp installer)."""
    req = urllib.request.Request(
        f"{GITHUB_API}?per_page={limit}",
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "llama-rpc-farm"})
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.loads(r.read().decode())
    for rel in releases:
        names = [a["name"] for a in rel.get("assets", [])
                 if a["name"].startswith("llama-swap_")
                 and (a["name"].endswith(".tar.gz") or a["name"].endswith(".zip"))]
        if names:
            base = (f"https://github.com/mostlygeek/llama-swap/releases/"
                    f"download/{rel['tag_name']}")
            return rel["tag_name"], names, base
    return None, [], ""


def _pick_asset(machine, tag, assets):
    """(asset_name, dest, version_number) or raises ValueError."""
    os_type = machine.get("os", "linux")
    arch = (machine.get("detected") or {}).get("arch") or "x86_64"
    kind = "arm64" if arch in ("arm64", "aarch64") else "amd64"
    os_map = {"linux": "linux", "macos": "darwin", "windows": "windows"}
    if os_type not in os_map:
        raise ValueError(f"unsupported OS for llama-swap: {os_type}")
    # asset names carry the tag NUMBER only (v256 release -> llama-swap_256_...)
    tag_num = _tag_number(tag)
    name = f"llama-swap_{tag_num}_{os_map[os_type]}_{kind}"
    ext = ".zip" if os_type == "windows" else ".tar.gz"
    name += ext
    if name not in assets:
        raise ValueError(f"llama-swap release asset not found upstream: {name}")
    return name, (DEST_WIN if os_type == "windows" else DEST_POSIX), tag_num


def _tag_number(tag):
    return (tag or "").lstrip("v")


def plan(machine):
    """What WOULD be installed; no side effects (UI pre-check)."""
    try:
        tag, assets, base = _fetch_latest_assets()
    except Exception as e:
        return {"ok": False, "error": f"cannot reach llama-swap releases: {e}"}
    try:
        name, dest, _v = _pick_asset(machine, tag, assets)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "tag": tag, "assets": [name], "dest": dest,
            "missing": []}


def install_on(machine, log=lambda m: None):
    """Install the official llama-swap binary on a machine (local or SSH).
    Returns dict {ok, machine, bin, detail}."""
    import config_loader as cfg
    from ssh_manager import SSHManager, SSHError
    import local_node
    import subprocess   # TimeoutExpired is a SubprocessError, NOT an OSError

    name = machine.get("name", "?")
    is_local = local_node.is_local(machine)
    is_win = machine.get("os", "linux") == "windows"

    def run_local(cmd, timeout=600):
        import subprocess
        # list + shell=False on every platform (cmd.exe list2cmdline trap)
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd] if is_win
            else ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout, shell=False)
        return proc.returncode, proc.stdout + proc.stderr

    def run_remote(cmd, timeout=600):
        # Normalize to the run_local (code, out) shape — ssh.run returns
        # a 3-tuple, which used to crash install_on's unpack on EVERY
        # remote install (fixed 2026-10-31).
        # Remote Windows: the default OpenSSH shell is cmd.exe; PS-syntax
        # bodies must be wrapped in powershell -NoProfile with escaped
        # quotes (same pattern as llamacpp_installer.run_remote).
        ssh = SSHManager(machine.get("ip"), machine.get("username"),
                         machine.get("password"), port=machine.get("port", 22))
        if is_win:
            cmd = ('powershell -NoProfile -Command "'
                   + cmd.replace('"', '\\"') + '"')
        code, out, err = ssh.run(cmd, timeout=timeout)
        return code, (out or "") + (err or "")

    run = run_local if is_local else run_remote
    try:
        # arch probe when not detected yet (same as llama.cpp installer)
        if not ((machine.get("detected") or {}).get("arch")):
            if is_win:
                code, out = run("echo $env:PROCESSOR_ARCHITECTURE")
                a = (out or "").strip().lower()
                machine.setdefault("detected", {})["arch"] = \
                    "arm64" if "arm" in a else "x86_64"
            else:
                code, out = run("uname -m")
                machine.setdefault("detected", {})["arch"] = \
                    (out or "").strip() or "x86_64"
        plan_info = plan(machine)
        if not plan_info.get("ok"):
            return {"ok": False, "machine": name, "detail": plan_info["error"]}
        urls = [f"https://github.com/mostlygeek/llama-swap/releases/"
                f"download/{plan_info['tag']}/{n}"
                for n in plan_info["assets"]]
        if is_win:
            # $d uses $env:USERPROFILE: PS never expands cmd-style %VAR%,
            # and on a LOCAL windows install there is no cmd.exe wrapper
            # to expand it first (fixed 2026-10-31 — same trap as the
            # llama.cpp installer's dest dir).
            ps = ["$ErrorActionPreference='Stop'",
                  '$d="$env:USERPROFILE\\llama-swap"',
                  "New-Item -ItemType Directory -Force -Path $d | Out-Null",
                  f"Invoke-WebRequest -Uri '{urls[0]}' "
                  "-OutFile \"$env:TEMP\\llamaswap.zip\"",
                  "Expand-Archive -Force \"$env:TEMP\\llamaswap.zip\" $d",
                  "try{ Remove-Item \"$env:TEMP\\llamaswap.zip\" }catch{}"]
            code, out = run("; ".join(ps))
        else:
            # tarball is FLAT (llama-swap binary at root, plus docs) —
            # extract only the binary into ~/.local/bin
            cmds = ["mkdir -p \"$HOME/.local/bin\"",
                    f"curl -fSL --retry 3 -o /tmp/llamaswap.tgz '{urls[0]}'",
                    "tar -xzf /tmp/llamaswap.tgz -C /tmp llama-swap",
                    "mv /tmp/llama-swap \"$HOME/.local/bin/llama-swap\"",
                    "rm -f /tmp/llamaswap.tgz",
                    "chmod +x \"$HOME/.local/bin/llama-swap\""]
            code, out = run(" && ".join(cmds))
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"install failed (exit {code}): {out[-500:]}"}
    except (SSHError, OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "machine": name, "detail": f"transfer failed: {e}"}

    # post-install verify: the binary must exist and be executable —
    # partial downloads/extracts used to report success silently.
    if is_win:
        vcmd = 'if(Test-Path "$env:USERPROFILE\\llama-swap\\llama-swap.exe"){echo OK}else{echo MISSING; exit 1}'
    else:
        vcmd = '[ -x "$HOME/.local/bin/llama-swap" ] && echo OK || { echo MISSING; exit 1; }'
    try:
        vcode, vout = run(vcmd, timeout=60)
    except (SSHError, OSError, subprocess.TimeoutExpired) as e:
        vcode, vout = 1, f"verify crashed: {e}"
    if vcode != 0:
        return {"ok": False, "machine": name,
                "detail": f"installed but verify failed "
                          f"(exit {vcode}): {(vout or '')[-300:]}"}

    if is_win:
        bin_path = f"{DEST_WIN}\\llama-swap.exe"
    else:
        bin_path = f"{DEST_POSIX}/llama-swap"
    # persist into swap settings so launch() uses it
    try:
        import swap_manager
        swap_manager.save_settings(binary=bin_path)
    except Exception:
        pass
    return {"ok": True, "machine": name, "bin": bin_path,
            "detail": f"installed llama-swap ({plan_info['tag']}) to {bin_path}"}


def detect_on(machine):
    """True if llama-swap is already present on the machine (local: PATH +
    well-known locations; remote: probe). Returns (found, bin_path)."""
    import local_node
    is_local = local_node.is_local(machine)
    is_win = machine.get("os", "linux") == "windows"
    if is_win:
        # $env:USERPROFILE — PS never expands cmd-style %VAR%, so the old
        # %USERPROFILE% probe always tested a literal path and reported
        # MISSING on every Windows machine (fixed 2026-10-31). Local runs
        # hit PowerShell directly; remote goes through the cmd wrapper,
        # which passes double quotes through with \" escaping.
        ps_cmd = ('$p=@("$env:USERPROFILE\\llama-swap\\llama-swap.exe",'
                  '"$env:USERPROFILE\\llama-swap.exe");'
                  "foreach($c in $p){if(Test-Path $c){echo $c;exit}};echo MISSING")
        probe = ps_cmd
    else:
        probe = ("for c in \"$HOME/.local/bin/llama-swap\" "
                 "/usr/local/bin/llama-swap /usr/bin/llama-swap; do "
                 "[ -x \"$c\" ] && echo \"$c\" && exit 0; done; "
                 "command -v llama-swap || echo MISSING")
    if is_local:
        import shutil
        found = shutil.which("llama-swap")
        if found:
            return True, found
        if is_win:
            import subprocess
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", probe],
                capture_output=True, text=True, timeout=15, shell=False)
            out = (proc.stdout or "").strip()
        else:
            import subprocess
            proc = subprocess.run(
                ["bash", "-c", probe], capture_output=True, text=True,
                timeout=15, shell=False)
            out = (proc.stdout or "").strip()
        if out and out != "MISSING":
            return True, out.splitlines()[0]
        return False, None
    try:
        from ssh_manager import SSHManager
        ssh = SSHManager(machine.get("ip"), machine.get("username"),
                         machine.get("password"), port=machine.get("port", 22))
        if is_win:
            # same cmd.exe->PowerShell wrapper rule as install_on
            probe = ('powershell -NoProfile -Command "'
                     + probe.replace('"', '\\"') + '"')
        code, out, _ = ssh.run(probe, timeout=15)
        out = (out or "").strip()
        if out and "MISSING" not in out:
            return True, out.splitlines()[0]
    except Exception:
        pass
    return False, None
