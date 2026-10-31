"""Starts/stops ggml-rpc-server processes across machines via SSH.

Faithful port of the user's existing .sh scripts on 10.0.0.104:
  - ggml-rpc-server via setsid with -H/-p/-t flags (-c disk cache removed:
    peers stream from NAS, small VM disks must not fill)
  - user-scoped pkill
  - /dev/tcp port wait with log tail on failure
  - trap-based cleanup on exit
Generated scripts mirror start-huihui-35b-rpc.sh / start-rpc-qwen.sh.
This module NEVER touches files outside its own folder.
"""
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

from ssh_manager import SSHManager, SSHError
import config_loader as cfg
import gpu_vendor
import gguf_inspect
import local_node
import path_resolver
import stats_collector

GENERATED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated"
)

DEFAULT_BIN = "~/llama.cpp/build/bin"

# Per-model RPC port allocation.
# ggml-rpc-server is single-client: two co-served llama-server processes
# cannot share one rpc-server. So each (model, peer) gets its OWN
# rpc-server on a distinct port on the peer. Ports are allocated
# deterministically from a hash of (model_id, peer_name) so the same
# model always lands on the same port across regenerations and live
# launches — no persisted allocation state needed.
#
# IMPORTANT: per-model ports use a HIGH base (51000), well above the
# Windows dynamic port-exclusion ranges Hyper-V/WSL reserve
# (typically 50000-50500; check `netsh int ipv4 show excludedportrange
# protocol=tcp` on a peer). Ports in those ranges fail to bind with
# "Failed to create server socket". The machine's rpc_port (default
# 50052) stays as the LEGACY single-server port (base-port fallback),
# which already has a working bind / firewall setup.
RPC_PORT_RANGE = 500   # how many ports to hand out above the base
RPC_PORT_BASE = 51000  # high base, clear of Windows excluded ranges
# remote-local (proxy) llama-server auto-ports: deterministic per
# (model, box), clear of the RPC range. 100s of local variants on one
# box never collide; launch_mode_local_port pins an explicit override.
LOCAL_PORT_BASE = 52000

# Windows Hyper-V/WSL port-exclusion ranges (from `netsh int ipv4 show
# excludedportrange protocol=tcp`). Both the dynamic (non-`*`) rows and
# the `*` "administered" rows block NEW binds; the legacy base 50052 sits
# in an administered range yet binds only because its server started
# BEFORE the exclusion was added (grandfathered). Any NEW port in these
# ranges fails with "Failed to create server socket".
_WIN_EXCLUDED = [(50000, 50059), (50060, 50159), (50160, 50259),
                 (50376, 50475), (49698, 49797), (49798, 49897),
                 (49898, 49997)]

# 50052 is the legacy single-server base — grandfathered (its server was
# bound before the Windows exclusion was applied), so it stays safe even
# though it's nominally inside an excluded range. Never bump it.
_LEGACY_BASE = 50052


def _is_excluded_port(port):
    """True if `port` falls in a Windows-excluded range AND isn't the
    grandfathered legacy base (50052)."""
    if port == _LEGACY_BASE:
        return False
    return any(lo <= port <= hi for lo, hi in _WIN_EXCLUDED)


def _safe_port(port):
    """If `port` is in a Windows-excluded range, bump it to the safe high
    base; otherwise return it unchanged. Used so a user-set machine
    rpc_port in a bad range doesn't produce an unbindable rpc-server."""
    if _is_excluded_port(port):
        return RPC_PORT_BASE + (port % RPC_PORT_RANGE)
    return port


def _model_rpc_port(model_id, peer_name, group="", base=RPC_PORT_BASE):
    """Deterministic per-(model,peer[,group]) RPC port in [base, base+RANGE).
    Same inputs always yield the same port, so config generation and
    live launch agree, and a re-launched model reuses its own port
    (no orphaned rpc-server on an old port). Uses a high base to dodge
    Windows Hyper-V/WSL excluded port ranges. `group` distinguishes the
    per-peer backend groups ("gpu" / "cpu") so a peer running TWO
    servers (accelerators + RAM) gets two stable, distinct ports."""
    import hashlib
    h = int(hashlib.md5(
        f"{model_id}\0{peer_name}\0{group}".encode()).hexdigest(), 16)
    return base + (h % RPC_PORT_RANGE)


def local_port_for(model_id, peer_name):
    """Deterministic, COLLISION-FREE port for a remote-local llama-server
    (launch-mode=local proxy models). Same hash base as RPC ports but
    resolves hash collisions by linear probing over the deployment set —
    the caller passes every other local deployment for that box in
    `taken`, mirroring the rpc allocator's `used` set. 100+ local
    variants on one box each get their own port; the SAME (model, box)
    always lands on the same port across pushes."""
    base = LOCAL_PORT_BASE
    taken = {t.get("_local_port_of") for t in _LOCAL_PORT_REGISTRY}
    h = int(hashlib.md5(f"{model_id}\0{peer_name}".encode()).hexdigest(), 16)
    port = base + (h % RPC_PORT_RANGE)
    while port in taken:
        port = port + 1 if port < base + RPC_PORT_RANGE else base
    return port


def _port_match_re(token):
    """ERE fragment for "the process bound --port <token>" that works
    whether --port sits at the END of the command line (the remote-local
    launcher renders it last) or has arguments after it. Trailing-space
    anchoring alone (`--port N `) NEVER matches `... --port N` at cmdline
    end — observed live: pkill rc=1 (no match) on a healthy server whose
    cmdline ended in the port; VRAM stayed held through every cmdStop.
    The alternation `( |$)` keeps the digit-anchoring property (52342
    never matches 523421) while matching both positions. ERE — safe in
    pkill/pgrep -f on POSIX."""
    return rf"--port {token}( |$)"


# per-generate() registry so launch_plan's sequential _process_one calls
# can linear-probe against ports already handed out in THIS plan
_LOCAL_PORT_REGISTRY: list = []


def _swap_state_file(model_id):
    """Remote filename (under ~/.cache/llama-wrapper/) holding the ACTUAL
    ports a launch-mode=local deployment's llama-server last served. The
    key is NOT hard-coded anywhere: at push time the farm resolves the
    deployment's SERVE GROUP (group members share one file, so kills are
    group-scoped — swapping within 'aibox3-only' tears down the whole
    group's servers) and passes it down; non-grouped deployments fall
    back to their model id. The launcher writes it (all paths), the swap
    cmdStop reads it — so a port that rolled past the config's probe
    point is still killed on unload/swap. Shared by rpc_manager
    (launcher) and swap_manager (cmdStop) — keep the derivation
    identical."""
    safe = "".join(
        c if c.isalnum() or c in "-_." else "-"
        for c in str(model_id)).strip("-") or "model"
    return f"swap.port.{safe}"


def swap_state_killer_snippet(host, state_key, fallback_port,
                              os_type="linux"):
    """cmdStop one-liner: kill every llama-server whose --port is listed
    in the group/machine state file, PLUS the config port. Serve-group
    aware: swap_manager resolves the deployment's group and passes it as
    `state_key` (never a hard-coded name in the generated string).

    llama-swap execs cmdStop itself (no local shell) over sshpass — the
    remote side runs ONE command so a missing state file degrades to the
    bare config-port pkill (pre-fix behavior). Trailing-space anchoring
    (`--port N `) is preserved: 52342 never matches 523421.

    Returns the full `sshpass ... ssh ... '<remote>'` string; $SSH_PASS
    stays BARE for llama-swap's direct exec."""
    user = host.get("username") or "root"
    ip = (host.get("ip") or "").strip()
    if os_type != "windows":
        state_rel = f".cache/llama-wrapper/{_swap_state_file(state_key)}"
        remote = (
            "bash -c 'ports=$(cat $HOME/" + state_rel + " 2>/dev/null); "
            'pkill -u ' + user + ' -f "llama-[s]erver.*'
            + _port_match_re(fallback_port) + '" 2>/dev/null; '
            'for p in $ports; do '
            r'pkill -u ' + user + ' -f "llama-[s]erver.*--port $p( |$)" '
            "2>/dev/null; done; "
            "rm -f $HOME/" + state_rel + "; true'")
        return (
            'sshpass -p "$SSH_PASS" ssh -n '
            '-o StrictHostKeyChecking=no -o ConnectTimeout=5 '
            f'{user}@{ip} {remote}')
    # Windows box: PowerShell reads the state file and kills every
    # llama-server whose command line carries ANY recorded --port (or the
    # config port), then clears the file. ASCII only (PS 5.1 over SSH).
    state_win = f".cache\\llama-wrapper\\{_swap_state_file(state_key)}"
    ps = (
        "$f=\"$env:USERPROFILE\\" + state_win + "\"; "
        "$ports=@('" + str(fallback_port) + "'); "
        "foreach ($p in (Get-Content $f -ErrorAction SilentlyContinue)) "
        "{ $ports += [string]$p }; "
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -like 'llama-server*' -and ("
        "$cl = ' ' + $_.CommandLine + ' '; "
        "($ports | ForEach-Object { "
        "$cl.Contains(' --port ' + $_ + ' ') -or "
        "$cl.EndsWith(' --port ' + $_) }) "
        "-contains $true) } | ForEach-Object { "
        "Stop-Process -Id $_.ProcessId -Force }; "
        "Remove-Item $f -ErrorAction SilentlyContinue")
    return (
        'sshpass -p "$SSH_PASS" ssh -n '
        '-o StrictHostKeyChecking=no -o ConnectTimeout=5 '
        f'{user}@{ip} '
        '"powershell -NoProfile -Command \\"' + ps + '\\""')


def _model_rpc_ports(model_cfg, peers, device_order=None):
    """{(peer_name, group): port} for a model across its peers.

    Each peer gets one port per backend GROUP it needs:
      - "gpu": all selected accelerator devices (one server)
      - "cpu": the CPU/RAM backend ONLY when the peer's RAM chip is
        selected — deliberately SEPARATE from the GPU server so the
        CPU backend never serializes behind GPU work on the single
        RPC client connection.
    Legacy single-port layout (gpu group only, CPU folded in) is what
    peers without a RAM chip get: same hash as before (group="" ->
    "gpu" keeps old ports stable via the group tag below).
    Collision-free within this model: two (peer, group) pairs never
    share a port."""
    base = RPC_PORT_BASE
    mid = model_cfg.get("model_id") or model_cfg.get("_file") or "model"
    used = set()
    out = {}
    for p in peers:
        groups = _peer_server_groups(model_cfg.get("device_order") or "", p)
        for gname, _devs in groups:
            port = _model_rpc_port(mid, p["name"], gname, base)
            while port in used:
                port = port + 1 if port < base + RPC_PORT_RANGE else base + 1
            used.add(port)
            out[(p["name"], gname)] = port
    return out


def _peer_server_groups(device_order, peer):
    """Split a peer's selected backends into rpc-server GROUPS.

    MINIMAL-ENDPOINT POLICY: every selected backend of a peer — ALL its
    accelerators AND its CPU/RAM backend — rides ONE rpc-server on ONE
    port, whenever a single binary can serve them together. One endpoint
    per peer is the fewest LAN connections the client must open and the
    fewest per-token round-trip hops the graph can have: splitting the
    CPU backend onto its own server adds an endpoint without removing
    any serialization (the graph is topologically serial anyway), so it
    can only add trips, never remove them.
    Grouping rule: devices are merged into the single "all" group
    (accelerators in --device order, CPU appended last) — exactly what
    one ggml-rpc-server --device list accepts. The only case that would
    NEED multiple servers is a peer whose selected accelerators span
    binaries that cannot coexist in one process (e.g. separate CUDA-only
    and ROCm-only builds); that is not representable in current configs
    and yields one server per vendor group via the same mechanism.
    A GPU-less peer with its RAM chip selected yields one "cpu" group
    (identical to legacy behavior). Peer with GPUs but no RAM chip
    yields one "gpu" group (identical to legacy, same port).

    Returns an ordered list of (group_name, [device tokens])."""
    gpus, ram = _peer_selected_resources(
        device_order, peer.get("name", ""), peer)
    devices = list(gpus) + (["CPU"] if ram else [])
    if not devices:
        # nothing selected and peer lists no GPUs: default server (all
        # its backends), matching the legacy fallback
        return [("gpu", [])]
    # one vendor-mixed binary can host CUDA + CPU (ggml-cpu is always
    # compiled in) — group by nothing, just merge. (Cross-vendor peers
    # with separate binaries would extend here: group tokens by the
    # binary that can load them, one group per binary.)
    return [("all", devices)]


def _global_bin():
    """Global default from config/settings.yaml — fallback when machine has no override."""
    try:
        import yaml as _yaml
        from pathlib import Path as _P
        p = _P(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if p.exists():
            data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            v = (data.get("llama_bin") or "").strip()
            if v:
                return v
    except Exception:
        pass
    return DEFAULT_BIN


def _effective_bin_dir(machine):
    """Machine override wins, else global settings default, else DEFAULT_BIN."""
    mb = (machine.get("llama_bin") or "").strip()
    if mb:
        return mb
    return _global_bin()


def _win_exe_expr(machine, server="ggml-rpc-server"):
    """PowerShell expression for a Windows .exe honoring machine llama_bin.

    llama_bin is a *directory* (same convention as Linux). Tokens
    %USERPROFILE% / $env:USERPROFILE / ~ / $HOME map to $env:USERPROFILE;
    back/forward slashes are normalized by PowerShell. Empty -> legacy
    default (USERPROFILE\\llama.cpp) to avoid breaking existing hosts.
    server: which binary under the bin dir (ggml-rpc-server.exe |
    llama-server.exe)."""
    exe_name = "ggml-rpc-server.exe" if server == "ggml-rpc-server" \
        else "llama-server.exe"
    raw = ((machine.get("llama_bin") if isinstance(machine, dict) else "") or "").strip().rstrip("/\\")
    if not raw:
        return f"Join-Path $env:USERPROFILE 'llama.cpp\\{exe_name}'"
    norm = raw.replace("/", "\\")
    low = norm.lower()
    rest = None
    for tok in ("%userprofile%\\", "$env:userprofile\\", "~\\", "$home\\"):
        if low.startswith(tok):
            rest = norm[len(tok):]
            break
    if rest is not None:
        rest = rest.replace("'", "''")
        base = f"Join-Path $env:USERPROFILE '{rest}'" if rest else "$env:USERPROFILE"
    elif low.startswith("%userprofile%") or low in ("$env:userprofile", "~", "$home"):
        base = "$env:USERPROFILE"
    elif len(norm) >= 2 and norm[1] == ":" or norm.startswith("\\\\"):
        norm = norm.replace("'", "''")
        base = f"'{norm}'"
    else:
        # bare relative dir (e.g. 'llama.cpp') -> under USERPROFILE
        norm = norm.replace("'", "''")
        base = f"Join-Path $env:USERPROFILE '{norm}'"
    return f"Join-Path ({base}) '{exe_name}'"


class _MacCapturingSSH(SSHManager):
    """SSHManager that remembers the machine it connects to. On the first
    successful command run, grabs the machine's MAC (the identity that
    never changes across DHCP re-leases) and stores it on the machine
    config — later IP drift can then be auto-healed from the ARP table.
    Capture is once-per-machine, off the hot path."""
    def __init__(self, machine, *a, **kw):
        super().__init__(machine["ip"], machine.get("username", ""),
                         machine.get("password", ""), port=machine.get("port", 22),
                         *a, **kw)
        self._machine = machine
        self._mac_done = bool((machine.get("mac") or "").strip())

    def run(self, command, timeout=60):
        code, out, err = super().run(command, timeout)
        if code == 0 and not self._mac_done:
            self._mac_done = True  # once per process, even on failure
            try:
                import discovery
                os_type = self._machine.get("os", "linux")
                mac_ip, mac = discovery.query_machine_mac(self, os_type)
                if mac:
                    self._machine["mac"] = mac
                    cfg.machines.update(
                        self._machine.get("name") or self._machine.get("id"),
                        {"mac": mac})
                    print(f"[mac-capture] {self._machine.get('name')}: {mac} @ {mac_ip}")
            except Exception:
                pass  # best-effort: never break a working command
        return code, out, err


def _ssh_for(machine):
    return _MacCapturingSSH(machine)


def _bin_subdir(machine, binary):
    """Bin path for remote commands. ~/ paths become $HOME-relative
    (like the user's scripts); absolute paths stay as-is. The remote
    shell expands $HOME, so this works for both SSH calls and scripts."""
    bin_dir = _effective_bin_dir(machine)
    if bin_dir.startswith("~/"):
        bin_dir = f'$HOME/{bin_dir[2:]}'
    return f'{bin_dir}/{binary}'


# All llama-related processes to kill, in order. The supervisor MUST die
# first or llama-swap respawns llama-server models right back.
# Patterns use the [b]racket trick so the pattern text doesn't match
# the killing shell's own command line (pkill -f self-match).
LINUX_LLAMA_PATTERN = "llama-[s]wap|llama-[s]erver|ggml-[r]pc-server"
WIN_LLAMA_PROCESSES = ["llama-swap.exe", "llama-server.exe", "ggml-rpc-server.exe"]


# ---------------- live RPC control over SSH ----------------
def _run_local(cmd, is_win, timeout=20):
    import subprocess
    try:
        proc = subprocess.run(
            cmd if is_win else ["bash", "-c", cmd],
            shell=is_win, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def stop_rpc(machine_name, port=None):
    """Stop llama processes on a machine, OS-aware, works locally and over SSH.

    port=None (default): legacy "stop everything" — kills llama-swap,
    llama-server, and ALL ggml-rpc-server instances on the box. Used by
    the global stop / stop-all buttons and by the master's pre-launch clear.

    port=<N>: per-model — kill ONLY the ggml-rpc-server listening on that
    port (matched by ``-p <port>`` in its command line), leaving other
    co-served models' rpc-servers untouched. llama-swap/llama-server are
    not touched in this mode (they belong to whoever started them)."""
    machine = cfg.machines.require(machine_name)
    is_win = machine.get("os", "linux") == "windows"
    is_local = local_node.is_local(machine)
    os_type = "windows" if is_win else "linux"

    def run(cmd, timeout=20):
        if is_local:
            return _run_local(cmd, is_win, timeout)
        try:
            return _ssh_for(machine).run(cmd, timeout=timeout)
        except SSHError as e:
            return 255, "", str(e)

    # ---- per-model port-targeted stop ----
    if port is not None:
        results = []
        if is_win:
            # match the rpc-server whose command line contains "-p <port>"
            ps = ("powershell -NoProfile -Command \""
                  "Get-CimInstance Win32_Process | "
                  f"Where-Object {{ (' ' + $_.CommandLine + ' ') -like '* -p {port} *' -and $_.Name -like 'ggml-rpc-server*' }} | "
                  "ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}\"")
            code, out, err = run(ps)
            results.append((f"ggml-rpc-server:-p{port}", code, out, err))
            # stale RPC tensor cache cleanup (see linux branch)
            run("powershell -NoProfile -Command \"Remove-Item "
                "$env:USERPROFILE/.cache/llama.cpp/rpc -Recurse -Force "
                "-ErrorAction SilentlyContinue\"")
            still = False
        else:
            user = (machine.get("username") or "").strip()
            uspec = f"-u {user} " if user else ""  # local master: no -u
            pat = f"ggml-[r]pc-server.*-p {port} "
            code, out, err = run(f'pkill {uspec}-f "{pat}"')
            results.append((f"ggml-rpc-server:-p{port}", code, out, err))
            # stale RPC tensor cache (-c leftovers from older runs, up to
            # GBs per file): the disk-cache flag is no longer used, but
            # old files must not starve the disk. Safe: purely derived.
            run("rm -rf $HOME/.cache/llama.cpp/rpc 2>/dev/null; true")
            still = True
            for _ in range(5):
                time.sleep(0.3)
                vcode, _, _ = run(f'pgrep {uspec}-f "{pat}"')
                if vcode != 0:
                    still = False
                    break
            code = 0 if not still else code
        return {
            "machine": machine_name, "port": port, "killed": not still,
            "detail": [{"process": p, "exit_code": c, "stdout": o, "stderr": e}
                       for p, c, o, e in results],
        }

    # ---- legacy: stop ALL llama processes ----
    results = []
    if is_win:
        for proc in WIN_LLAMA_PROCESSES:
            code, out, err = run(f"taskkill /F /IM {proc}")
            results.append((proc, code, out, err))
        # verify: any of them still running?
        code, out, _ = run(
            "powershell -NoProfile -Command \"(Get-Process -Name "
            "llama-swap,llama-server,ggml-rpc-server -ErrorAction SilentlyContinue)"
            ".Count\"")
        still = False
        try:
            still = int((out or "0").strip().splitlines()[-1]) > 0
        except (ValueError, IndexError):
            pass
    else:
        # stop+disable the systemd unit FIRST: it has Restart=always, so a
        # bare pkill just gets resurrected. Then pkill the stragglers.
        run("systemctl --user stop llama-swap 2>/dev/null; "
            "systemctl --user disable llama-swap 2>/dev/null; true")
        # one pkill pass over all three, ordered; 1 = nothing matched is fine
        user = (machine.get("username") or "").strip()
        uspec = f"-u {user} " if user else ""  # local master: no -u
        code, out, err = run(f'pkill {uspec}-f "{LINUX_LLAMA_PATTERN}"')
        results.append(("llama-*", code, out, err))
        # verify they're actually gone (pkill is async)
        still = True
        for _ in range(5):
            time.sleep(0.5)
            vcode, vout, _ = run(f'pgrep {uspec}-f "{LINUX_LLAMA_PATTERN}"')
            if vcode != 0:  # pgrep exit 1 = no matches = all dead
                still = False
                break
        code = 0 if not still else code

    return {
        "machine": machine_name,
        "killed": not still,
        "detail": [
            {"process": p, "exit_code": c, "stdout": o, "stderr": e}
            for p, c, o, e in results
        ],
    }


def stop_llama_server_port(machine_name, port):
    """Kill ONLY the llama-server bound to --port <port> on a machine.

    Serve-group-safe pre-launch hygiene for single-box (local) launches:
    unlike the legacy stop_rpc (which clears every llama process on the
    box), this leaves co-served models on other ports untouched. Same
    trailing-space-anchored `--port <N> ` match the swap cmdStop uses, so
    8080 never matches 80808. POSIX (linux/macos) via pkill; Windows via
    Get-CimInstance on llama-server* processes."""
    machine = cfg.machines.require(machine_name)
    is_win = machine.get("os", "linux") == "windows"
    is_local = local_node.is_local(machine)

    def run(cmd, timeout=20):
        if is_local:
            return _run_local(cmd, is_win, timeout)
        try:
            return _ssh_for(machine).run(cmd, timeout=timeout)
        except SSHError as e:
            return 255, "", str(e)

    port = int(port)
    # Windows/POSIX port match at cmdline END or mid-line — the
    # trailing-space-only pattern never matched `... --port N` at cmdline
    # end (the remote-local launcher renders --port last), leaving VRAM
    # held through every cmdStop (observed on aibox3 group swaps).
    if is_win:
        ps = ("powershell -NoProfile -Command \""
              "Get-CimInstance Win32_Process | "
              f"Where-Object {{ $cl = ' ' + $_.CommandLine + ' '; "
              f"(($cl -like '* --port {port} *') -or "
              f"($cl -like '* --port {port}')) -and "
              "($_.Name -like 'llama-server*') }} | "
              "ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}\"")
        code, out, err = run(ps)
        return {"machine": machine_name, "port": port, "killed": code == 0,
                "detail": [{"process": f"llama-server:--port{port}",
                            "exit_code": code, "stdout": out, "stderr": err}]}
    user = (machine.get("username") or "").strip()
    uspec = f"-u {user} " if user else ""
    pat = f"llama-[s]erver.*{_port_match_re(port)}"
    code, out, err = run(f'pkill {uspec}-f "{pat}"')
    still = True
    for _ in range(5):
        time.sleep(0.3)
        vcode, _, _ = run(f'pgrep {uspec}-f "{pat}"')
        if vcode != 0:
            still = False
            break
    code = 0 if not still else code
    return {
        "machine": machine_name, "port": port, "killed": not still,
        "detail": [{"process": f"llama-server:--port{port}", "exit_code": code,
                    "stdout": out, "stderr": err}],
    }


# Per-port RPC state. Each (model,peer) rpc-server records the --device
# set it was started with in rpc.state.{port}, so ensure checks are per
# instance — co-served models on separate ports never clobber each other.
# Older single-server installs wrote a bare rpc.state; we still read it
# as a fallback for the legacy base port for backward compatibility.
RPC_STATE_REL = ".cache/llama-wrapper/rpc.state"
RPC_STATE_TMP_LINUX = "/tmp/llama-wrapper-rpc.state"
RPC_STATE_TMP_WIN = "$env:TEMP/llama-wrapper-rpc.state"


def _rpc_state_path(port, is_win):
    """Per-port state file path (primary, under $HOME / %USERPROFILE%).
    The base port (50052) keeps the legacy bare 'rpc.state' name so an
    existing single-server install still reuses cleanly."""
    if port == 50052:
        return RPC_STATE_REL
    rel = f".cache/llama-wrapper/rpc.state.{port}"
    return rel


def _rpc_state_tmp(port, is_win):
    if port == 50052:
        return RPC_STATE_TMP_WIN if is_win else RPC_STATE_TMP_LINUX
    return (f"$env:TEMP/llama-wrapper-rpc.state.{port}" if is_win
            else f"/tmp/llama-wrapper-rpc.state.{port}")


def _rpc_log_name(port):
    """Per-port rpc-server log filename. Base port keeps the legacy
    'rpc-server.log' name; per-model ports get 'rpc-server-{port}.log'."""
    return "rpc-server.log" if port == 50052 else f"rpc-server-{port}.log"


def _rpc_state_want(machine, expose_override=None, device_list=None):
    """Backend-set string a peer's rpc-server must expose. Mirrors the
    _expose_args mapping so Python and generated scripts agree."""
    if device_list is None:
        expose = (expose_override if expose_override is not None
                  else machine.get("expose") or {})
        if expose.get("mode") == "ram" and not (expose.get("gpus") or []):
            return "CPU"
        gpus = expose.get("gpus") or []
        return ",".join(gpus) if gpus else "default"
    return ",".join(device_list) if device_list else "default"


def _rpc_state_read(machine, is_local, port=50052):
    """Backend set the peer's rpc-server (on `port`) was started with
    ('' when unknown -- treated as mismatch, i.e. restart). Per-port:
    each model's rpc-server has its own state file so co-served models
    on separate ports never clobber each other's recorded backends."""
    is_win = machine.get("os", "linux") == "windows"
    rel = _rpc_state_path(port, is_win)
    tmp = _rpc_state_tmp(port, is_win)
    if is_local:
        try:
            if is_win:
                primary = os.path.join(os.environ.get("USERPROFILE", ""),
                                       *rel.split("/"))
                fallback = os.path.join(os.environ.get("TEMP", ""),
                                        os.path.basename(tmp))
            else:
                primary = os.path.expanduser("~/" + rel)
                fallback = tmp
            for path in (primary, fallback):
                try:
                    with open(path, encoding="utf-8") as f:
                        return f.read().strip()
                except OSError:
                    continue
            return ""
        except OSError:
            return ""
    if is_win:
        cmd = ("powershell -NoProfile -Command "
               f"\"if (Test-Path $env:USERPROFILE/{rel}) "
               f"{{ Get-Content $env:USERPROFILE/{rel} }} "
               f"elseif (Test-Path {tmp}) "
               f"{{ Get-Content {tmp} }}\"")
    else:
        cmd = (f"cat $HOME/{rel} 2>/dev/null || "
               f"cat {tmp} 2>/dev/null || true")
    try:
        _, out, _ = _ssh_for(machine).run(cmd, timeout=10)
    except SSHError:
        return ""
    return (out or "").strip()


def _rpc_state_write(machine, is_local, want, port=50052):
    """Record the backend set a freshly started rpc-server (on `port`)
    exposes. Per-port so each co-served model tracks its own peer."""
    is_win = machine.get("os", "linux") == "windows"
    rel = _rpc_state_path(port, is_win)
    tmp = _rpc_state_tmp(port, is_win)
    if is_local:
        try:
            if is_win:
                primary = os.path.join(os.environ.get("USERPROFILE", ""),
                                       *rel.split("/"))
                fallback = os.path.join(os.environ.get("TEMP", ""),
                                        os.path.basename(tmp))
            else:
                primary = os.path.expanduser("~/" + rel)
                fallback = tmp
            try:
                os.makedirs(os.path.dirname(primary), exist_ok=True)
                with open(primary, "w", encoding="utf-8") as f:
                    f.write(want + "\n")
            except OSError:
                with open(fallback, "w", encoding="utf-8") as f:
                    f.write(want + "\n")
        except OSError:
            pass
        return
    if is_win:
        cmd = ("powershell -NoProfile -Command "
               "\"New-Item -ItemType Directory -Force "
               f"$env:USERPROFILE/.cache/llama-wrapper -ErrorAction SilentlyContinue | Out-Null; "
               f"Set-Content $env:USERPROFILE/{rel} "
               f"-Value \\\"{want}\\\" -ErrorAction SilentlyContinue; "
               f"if (-not (Test-Path $env:USERPROFILE/{rel})) "
               f"{{ Set-Content {tmp} -Value \\\"{want}\\\" }}\"")
    else:
        cmd = (f"mkdir -p $HOME/.cache/llama-wrapper && echo \"{want}\" "
               f"> $HOME/{rel} || echo \"{want}\" "
               f"> {tmp}")
    try:
        _ssh_for(machine).run(cmd, timeout=10)
    except SSHError:
        pass
def _expose_args(expose):
    """ggml-rpc-server flags for what to expose. NOTE: ggml-rpc-server
    has no -g flag (verified against --help: flags are -t/-d/-H/-p/-c,
    where -c is the on-disk tensor cache — deliberately NOT used: peers
    are light VMs that stream from the NAS; -c makes each peer keep a
    full local copy of received tensors and fills small disks, which
    crashed aibox2 at 100% disk mid-warmup). GPU restriction is expressed
    purely via --device with the gpus subset; empty subset = server
    default. mode 'ram' (CPU RAM only, no GPUs picked) maps to an
    explicit CPU device, the backend name ggml uses for host memory."""
    expose = expose or {}
    if expose.get("mode") == "ram" and not (expose.get("gpus") or []):
        return " --device CPU"
    gpus = expose.get("gpus") or []
    args = ""
    if gpus:
        args += f" --device {','.join(gpus)}"
    return args


def start_rpc(machine_name, port=None, expose_override=None, device_list=None,
              ensure=False):
    """Start ggml-rpc-server on a machine on a per-model port, OS-aware.

    Each (model, peer) gets its OWN rpc-server on a distinct port, because
    ggml-rpc-server is single-client — two co-served llama-server processes
    cannot share one. `port` is the per-model port (from
    _model_rpc_port); when None it falls back to the machine's base
    rpc_port (legacy single-server behavior).

    ensure=True: a reachable server on `port` whose recorded backends
    match is reused untouched (per-port rpc.state.{port}). Otherwise the
    instance on `port` is killed (only that port, not other models') and
    restarted, then its backend set is recorded for future ensure checks.
    """
    machine = cfg.machines.require(machine_name)
    port = _safe_port(port or machine.get("rpc_port", 50052))
    is_win = machine.get("os", "linux") == "windows"
    is_local = local_node.is_local(machine)
    want = _rpc_state_want(machine, expose_override, device_list)
    if ensure and _port_open(machine, port, is_local):
        have = _rpc_state_read(machine, is_local, port)
        if have == want:
            return {
                "machine": machine_name, "ip": machine["ip"], "port": port,
                "running": True, "reused": True, "exit_code": 0,
                "stderr": "", "log_tail": "",
            }
    ssh = None if is_local else _ssh_for(machine)

    # kill ONLY this port's instance — other co-served models' rpc-servers
    # on other ports must survive.
    kill_result = stop_rpc(machine_name, port)
    if not kill_result["killed"]:
        return {
            "machine": machine_name, "ip": machine["ip"], "port": port,
            "running": False, "exit_code": 1,
            "stderr": f"failed to clear old rpc-server on port {port}: {kill_result.get('detail', '')}",
            "log_tail": "",
        }
    time.sleep(1)

    # Use model-specific expose config if provided, else fall back to machine config
    expose = expose_override if expose_override is not None else machine.get("expose") or {}
    if device_list is None:
        dev_args = _expose_args(expose)
    else:
        dev_args = f" --device {','.join(device_list)}" if device_list else ""
    log_name = _rpc_log_name(port)

    if not is_win:
        # macOS has no nproc (coreutils); sysctl is the Apple equivalent
        ncpu = ("$(sysctl -n hw.ncpu)" if machine.get("os") == "macos"
                else "$(nproc --all)")
        env_prefix = _rpc_env_exports(machine)
        # setsid detaches (no controlling tty -> survives ssh drop);
        # macOS lacks setsid, so fall back to nohup there.
        detach = ("nohup" if machine.get("os") == "macos" else "setsid")
        # setsid takes COMMAND WORDS, not a shell line: it would exec the
        # bare env token ("failed to execute GGML_SCHED_DEBUG=3"). Wrap in
        # sh -c so the env exports apply and the binary path stays quoted.
        remote = (
            f'{detach} sh -c "{env_prefix}\\"{_bin_subdir(machine, "ggml-rpc-server")}\\" '
            f'-H 0.0.0.0 -p {port} -t \\"{ncpu}\\"{dev_args} '
            f'</dev/null > \\"$HOME/{log_name}\\" 2>&1 &"'
        )
        if is_local:
            import subprocess
            proc = subprocess.run(
                ["bash", "-c", remote], capture_output=True, text=True, timeout=15)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        else:
            code, out, err = ssh.run(remote)
    else:
        # faithful port of push_win_launcher() + start_rpc_win()
        launcher = _win_launcher(machine, port, expose, device_list)
        if is_local:
            import subprocess
            import tempfile
            path = os.path.join(tempfile.gettempdir(), "start-rpc.ps1")
            with open(path, "w", encoding="utf-8") as f:
                f.write(launcher)
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
                capture_output=True, text=True, timeout=30)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        else:
            ssh.push_text(launcher, "start-rpc.ps1", "windows")
            code, out, err = ssh.run(
                "powershell -NoProfile -ExecutionPolicy Bypass -File "
                '"%USERPROFILE%\\start-rpc.ps1"'
            )

    waited = 0
    while waited < 30:
        if _port_open(machine, port, is_local):
            break
        time.sleep(1)
        waited += 1

    ok = _port_open(machine, port, is_local)
    tail = ""
    if not ok:
        tail = _tail_log(machine, is_win, is_local, port)
    else:
        # record backend set so ensure checks (here and in .sh scripts)
        # can reuse this server instead of restarting it
        _rpc_state_write(machine, is_local, want, port)
    return {
        "machine": machine_name,
        "ip": machine["ip"],
        "port": port,
        "running": ok,
        "exit_code": code,
        "stderr": err,
        "log_tail": tail,
    }


def _win_launcher(machine, port, expose=None, device_list=None, parametrized=False,
                  expose_args=None):
    """The .ps1 launcher from the user's 4-node script.
    device_list: explicit backends to expose (wins over expose config).
    parametrized: take an -ExtraArgs parameter instead, so generated
    .sh scripts can pass per-node --device lists at runtime (one
    rpc-server per machine, each exposing exactly its selected GPUs +
    CPU-iff-RAM-selected).
    expose_args: pre-rendered static flags (used by script generation,
    which already computed them); wins over everything when given."""
    if expose_args is None:
        if device_list is None:
            expose_args = _expose_args(expose if expose is not None
                                       else machine.get("expose"))
        else:
            expose_args = f" --device {','.join(device_list)}" if device_list else ""
    log_name = _rpc_log_name(port)
    # Env pairs for the launcher. NOTE: Win32_Process.Create does NOT
    # inherit the PowerShell session's $env: — the child gets wmiprvse's
    # environment instead. So besides the $env: lines (kept: they apply
    # when the .ps1 is run directly), the vars are ALSO chained as
    # `set K=V &&` inside the cmd.exe command line, which provably
    # reaches ggml-rpc-server.exe. Without this, CUDA_DEVICE_ORDER never
    # arrived on Windows peers and --device CUDA1 hit the wrong card.
    win_env_pairs = [e.split("=", 1)
                     for e in _rpc_env_exports(machine).strip().split()
                     if "=" in e]
    win_env_lines = "".join(f'$env:{k}="{v}"\n' for k, v in win_env_pairs)
    cmd_env_chain = "".join(f"set {k}={v} && " for k, v in win_env_pairs)
    if parametrized:
        # Parametrized so a generated .sh with several Windows peers on
        # different per-model ports can pass -Port/-LogName per node at
        # runtime (one launcher template, N ports). Defaults bake in the
        # generated port so a bare invocation still works.
        # NOTE: -f binds tighter than + in PowerShell, so each piece is
        # formatted separately before concatenating; a single trailing
        # -f would only expand the last string and leave literal {0}.
        return f"""param([int]$Port = {port}, [string]$LogName = '{log_name}', [string]$ExtraArgs = "")
$ErrorActionPreference = 'Stop'
$env:GGML_SCHED_DEBUG="3"
$env:GGML_RPC_DEBUG="3"
{win_env_lines}$exe = {_win_exe_expr(machine)}
$log = Join-Path $env:USERPROFILE $LogName
if (-not (Test-Path $exe)) {{ Write-Error "not found: $exe"; exit 1 }}
# NOTE: no self-kill here -- the caller (.sh ensure step / start_rpc)
# stops a mismatched server on THIS port BEFORE invoking this launcher,
# so a co-served model's rpc-server on another port is never touched.
# Format the exe path first (-f binds tighter than +), then splice the
# runtime port + extra args, then the log redirection. $env: lines above
# do NOT reach a CIM-created child (wmiprvse environment instead), so
# the vars ride inline as `set K=V &&` inside the cmd line.
# cmd.exe /c strips the outermost quote pair, so with the `set &&`
# prefix the exe is single-quoted ("{0}"): mid-line `""{0}""` dies with
# `'""...exe""' is not recognized` (seen live on winbox).
$base = ('cmd.exe /c "{cmd_env_chain}\"{{0}}\" -H 0.0.0.0 -p {{1}}{expose_args}' -f $exe, $Port)
$cmd = $base + $ExtraArgs + (' > "{{0}}" 2>&1"' -f $log)
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = $cmd }}
if ($r.ReturnValue -ne 0) {{ Write-Error "Win32_Process.Create failed: $($r.ReturnValue)"; exit 1 }}
Write-Output "started pid $($r.ProcessId)"
"""
    return f"""$ErrorActionPreference = 'Stop'
$env:GGML_SCHED_DEBUG="3"
$env:GGML_RPC_DEBUG="3"
{win_env_lines}$exe = {_win_exe_expr(machine)}
$log = Join-Path $env:USERPROFILE '{log_name}'
if (-not (Test-Path $exe)) {{ Write-Error "not found: $exe"; exit 1 }}
# NOTE: no self-kill here -- start_rpc() stops leftovers on this port first.
# $env: lines do NOT reach a CIM-created child, so vars ride inline below
# (single-quoted exe: cmd.exe /c strips the outer pair, see above).
$cmd = 'cmd.exe /c "{cmd_env_chain}\"{{0}}\" -H 0.0.0.0 -p {port}{expose_args} > "{{1}}" 2>&1"' -f $exe, $log
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = $cmd }}
if ($r.ReturnValue -ne 0) {{ Write-Error "Win32_Process.Create failed: $($r.ReturnValue)"; exit 1 }}
Write-Output "started pid $($r.ProcessId)"
"""


def _port_open(machine, port, is_local):
    # native-CM peer (rdma+rdma_cm): no TCP listener exists — alive means the
    # rpc-server process for this port is running on the peer.
    if machine.get("rdma") and machine.get("rdma_cm") and not is_local:
        try:
            from ssh_manager import SSHManager
            ssh = SSHManager(machine["ip"], machine["username"],
                             machine.get("password", ""),
                             port=machine.get("port", 22))
            code, out, _ = ssh.run(
                f"pgrep -f 'ggml-[r]pc-server.*-p {port} ' >/dev/null && echo up",
                timeout=8)
            return code == 0 and "up" in (out or "")
        except Exception:
            return False
    host = "127.0.0.1" if is_local else machine["ip"]
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _tail_log(machine, is_win, is_local, port=50052):
    """Last 20 lines of the rpc-server log for `port` on the target machine.
    Per-model ports log to rpc-server-{port}.log; the base port keeps the
    legacy rpc-server.log name."""
    log_name = _rpc_log_name(port)
    if is_win:
        cmd = ("powershell -NoProfile -Command "
               f'"Get-Content $env:USERPROFILE\\{log_name} -Tail 20"')
    else:
        cmd = f'tail -20 "$HOME/{log_name}"'
    if is_local:
        import subprocess
        try:
            proc = subprocess.run(
                cmd if is_win else ["bash", "-c", cmd],
                shell=is_win, capture_output=True, text=True, timeout=10)
            return proc.stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""
    try:
        _, tail, _ = _ssh_for(machine).run(cmd, timeout=10)
        return tail
    except SSHError:
        return ""


def _tail_all_rpc_logs(machine, is_win, is_local):
    """Last 20 lines of EVERY rpc-server log on the target machine —
    the legacy rpc-server.log (base port) plus all per-model
    rpc-server-{port}.log files. Used by the Flow Visualizer's sched
    stream so every co-served model's per-port peer log is tailed."""
    if is_win:
        # tail rpc-server.log + rpc-server-*.log under USERPROFILE
        cmd = ("powershell -NoProfile -Command \""
               "Get-ChildItem $env:USERPROFILE -Filter 'rpc-server*.log' | "
               "ForEach-Object { Get-Content $_.FullName -Tail 20 }\"")
    else:
        cmd = 'tail -n 20 "$HOME"/rpc-server.log "$HOME"/rpc-server-*.log 2>/dev/null'
    if is_local:
        import subprocess
        try:
            proc = subprocess.run(
                cmd if is_win else ["bash", "-c", cmd],
                shell=is_win, capture_output=True, text=True, timeout=10)
            return proc.stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""
    try:
        _, tail, _ = _ssh_for(machine).run(cmd, timeout=10)
        return tail
    except SSHError:
        return ""


# ---------------- status ----------------
def _sync_machine_ip(machine):
    """IP drift auto-heal: DHCP re-leases change machine IPs; the MAC is
    the stable identity. If a stored MAC resolves (via ARP) to a different
    IP than configured, fix the config and return the updated machine.
    Only touches machines that HAVE a stored mac. Returns machine
    (possibly the same object) after any fix."""
    import socket
    stored_mac = (machine.get("mac") or "").strip().lower()
    if not stored_mac or local_node.is_local(machine):
        return machine
    import discovery
    current_ip = discovery.resolve_ip_by_mac(stored_mac)
    if not current_ip or current_ip == machine.get("ip"):
        return machine
    # sanity: the new IP must actually speak SSH before we switch to it
    try:
        with socket.create_connection((current_ip, machine.get("port", 22)),
                                      timeout=3):
            pass
    except OSError:
        return machine
    old_ip = machine["ip"]
    machine["ip"] = current_ip
    try:
        cfg.machines.update(machine.get("name") or machine.get("id"),
                            {"ip": current_ip, "mac": stored_mac})
    except Exception:
        pass  # best-effort persist; the in-memory fix already helps
    print(f"[ip-sync] {machine.get('name')}: {old_ip} -> {current_ip} "
          f"(matched stored MAC {stored_mac})")
    return machine


def machine_status(machine_name):
    """Per-machine status. rpc_running is True if ANY ggml-rpc-server is
    running on the box (not just the base port), since per-model ports
    mean the base port may be unused while other models serve.
    On unreachability, tries MAC-based IP healing before reporting down."""
    machine = cfg.machines.require(machine_name)
    is_local = local_node.is_local(machine)
    if not is_local:
        ssh = _ssh_for(machine)
        reachable = ssh.is_reachable()
        if not reachable:
            # IP failed: the stored MAC never changes — resolve it via ARP
            # and retry at the healed IP before reporting the machine down.
            machine = _sync_machine_ip(machine)
            if machine["ip"] != cfg.machines.require(machine_name)["ip"]:
                ssh = _ssh_for(machine)
                reachable = ssh.is_reachable()
    else:
        reachable = True  # we're running on it
        # opportunistically capture the local MAC too (identity for peers)
        if not (machine.get("mac") or "").strip():
            try:
                mac = local_node.primary_mac()
                if mac:
                    machine["mac"] = mac
                    try:
                        cfg.machines.update(machine.get("name") or machine.get("id"),
                                            {"mac": mac})
                    except Exception:
                        pass
            except Exception:
                pass
    if not reachable:
        return {
            "name": machine_name, "ip": machine["ip"], "reachable": False,
            "rpc_running": False, "gpus": [], "is_local": is_local,
            "is_master": machine.get("role") == "master",
        }
    if is_local:
        rpc_running = _any_rpc_running_local(machine)
        gpus = _local_gpu_info(machine)
    else:
        ssh = _ssh_for(machine)
        rpc_running = _any_rpc_running_remote(machine, ssh)
        gpus = ssh.gpu_info(machine.get("gpu_vendor"))
    return {
        "name": machine_name,
        "ip": machine["ip"],
        "reachable": True,
        "rpc_running": rpc_running,
        "gpus": gpus,
        "is_local": is_local,
        "is_master": machine.get("role") == "master",
    }


def _local_port_open(port):
    import socket as _socket
    try:
        with _socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def _roll_port_if_busy(machine, port, is_local, tries=20):
    """Return the first bindable port at or above `port` on the target
    machine. A busy requested port (leftover llama-server, another app)
    must not fail the launch outright — roll forward instead. Also skips
    Windows-excluded ranges (whole ranges at once, they span ~200 ports)
    so a rolled port can actually bind there.
    Returns (port, rolled_bool)."""
    candidate = int(port)
    for _ in range(tries):
        skipped = False
        for lo, hi in _WIN_EXCLUDED:
            if lo <= candidate <= hi and candidate != _LEGACY_BASE:
                candidate = hi + 1
                skipped = True
                break
        if skipped:
            continue
        if _port_open(machine, candidate, is_local):
            candidate += 1
            continue
        return candidate, candidate != int(port)
    return int(port), False  # give up rolling: caller tries the port as-is


def _any_rpc_running_local(machine):
    """True if any ggml-rpc-server process is running locally (any port).
    Falls back to probing the base port if the process listing fails."""
    import subprocess
    if os.name == "nt":
        # no pgrep on Windows: tasklist filter (name match, no shell)
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ggml-rpc-server.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10)
            return "ggml-rpc-server" in (proc.stdout or "")
        except (OSError, subprocess.SubprocessError):
            pass
        return _local_port_open(machine.get("rpc_port", 50052))
    try:
        user = machine.get("username") or os.environ.get("USER", "")
        if user:  # -u "" is an invalid arg: skip the flag when unnamed
            proc = subprocess.run(["pgrep", "-u", user,
                                   "-f", "ggml-[r]pc-server"],
                                  capture_output=True, text=True)
        else:
            proc = subprocess.run(["pgrep", "-f", "ggml-[r]pc-server"],
                                  capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    return _local_port_open(machine.get("rpc_port", 50052))


def _any_rpc_running_remote(machine, ssh):
    """True if any ggml-rpc-server process is running on the remote box."""
    is_win = machine.get("os", "linux") == "windows"
    try:
        if is_win:
            code, out, _ = ssh.run(
                "powershell -NoProfile -Command \"(Get-Process -Name "
                "ggml-rpc-server -ErrorAction SilentlyContinue).Count\"")
            try:
                return int((out or "0").strip().splitlines()[-1]) > 0
            except (ValueError, IndexError):
                return False
        code, _, _ = ssh.run("pgrep -f ggml-[r]pc-server")
        return code == 0
    except SSHError:
        return ssh.check_port(machine.get("rpc_port", 50052))


def _local_gpu_info(machine=None):
    """GPU query rows on THIS machine, vendor-aware (nvidia-smi /
    rocm-smi), normalized to nvidia-style csv rows."""
    import subprocess
    if gpu_vendor.normalize_vendor((machine or {}).get("gpu_vendor")) == "apple":
        return gpu_vendor.macos_gpu_rows()
    vendor = gpu_vendor.normalize_vendor((machine or {}).get("gpu_vendor"))
    queries = ([gpu_vendor.nvidia_query(
        "index,name,memory.total,memory.used,temperature.gpu")]
        if vendor == "nvidia" else
        [gpu_vendor.rocm_query_json()] if vendor == "amd" else
        [gpu_vendor.intel_query_json()] if vendor == "intel" else
        [gpu_vendor.nvidia_query(
            "index,name,memory.total,memory.used,temperature.gpu"),
         gpu_vendor.rocm_query_json(),
         gpu_vendor.intel_query_json()])
    for q in queries:
        try:
            if os.name == "nt":
                # no bash on Windows: run the query binary directly.
                # q is a single command line; split its argv (no shell
                # metachars in the vendor queries).
                import shlex
                proc = subprocess.run(
                    shlex.split(q), capture_output=True, text=True, timeout=15)
            else:
                proc = subprocess.run(
                    ["bash", "-c", q], capture_output=True, text=True, timeout=15)
        except OSError:
            return []
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        out = proc.stdout
        if q.startswith("rocm-smi"):
            rows = gpu_vendor.parse_rocm_json(out)
            if rows:
                return rows
            continue
        if q.startswith("xpu-smi"):
            rows = gpu_vendor.parse_xpu_smi_json(out)
            if rows:
                return rows
            continue
        return [l.strip() for l in out.splitlines() if l.strip()]
    return []


def cluster_status(names=None):
    names = names or [m["name"] for m in cfg.machines.all()]
    with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
        return list(pool.map(machine_status, names))


def group_status(group_name):
    group = cfg.groups.require(group_name)
    return cluster_status(group.get("machines", []))


# ---------------- command generation ----------------
def _split_servers(servers, master_name=None):
    """Returns (master, peers). Master = master_name when given,
    else role:master, else first entry."""
    machines = [cfg.machines.require(n) for n in servers]
    if master_name:
        master = next((m for m in machines if m["name"] == master_name),
                      machines[0])
    else:
        master = next((m for m in machines if m.get("role") == "master"),
                      machines[0])
    peers = [m for m in machines if m is not master]
    return master, peers


def build_llama_command(model_cfg, servers, master_name=None):
    """Build the llama-server command from a model config.

    Flags are stored verbatim (e.g. '-ngl': 99, '--jinja': True) and
    emitted in stored order. --rpc / --device / --tensor-split are
    derived: device_order comes from the model config verbatim when set
    (the user's scripts define it per model), else auto from machine GPUs.
    master_name forces which server is master (used by llama-swap
    generation, where the host always runs llama-server).
    """
    master, peers = _split_servers(servers, master_name)
    flags = model_cfg.get("flags", {})
    # Visualizer ground truth: llama-server only logs the verbose
    # "http: streamed chunk" lines that drive the upstream fallback
    # (single-node/single-GPU models have no cross-device sched events)
    # when -v is on. Always inject unless the model already sets them.
    def _flag_set(*names):
        return any(k.lstrip("-") in names for k in flags)
    injected = {}
    if not _flag_set("v", "verbose"):
        injected["-v"] = True
    if not _flag_set("log-timestamps"):
        injected["log-timestamps"] = True
    if not _flag_set("log-prefix"):
        injected["log-prefix"] = True
    if injected:
        flags = {**flags, **injected}
    # MLX runtime (Apple Silicon): base_path is an HF repo id, not a
    # GGUF path. mlx_lm.server is OpenAI-compatible; no RPC/--device.
    if (model_cfg.get("runtime") or "llama") == "mlx":
        repo = path_resolver.resolve_for(
            model_cfg.get("base_path") or "", master.get("os", "linux"))
        return (
            'python -m mlx_lm server \\\n'
            f'  --model "{repo}" \\\n'
            '  --host 0.0.0.0 \\\n'
            '  --port "$PORT"'
        )
    # paths in the command must match the OS of the machine running llama-server
    target_os = master.get("os", "linux")
    model_path = path_resolver.resolve_for(
        model_cfg.get("base_path") or model_cfg.get("path", ""), target_os)
    # binary: machine override wins, else global settings default (see _effective_bin_dir),
    # else per-model bin. This keeps heterogeneous clusters (linux + windows) correct:
    # set Settings as default, override per-machine when its path differs.
    effective = _effective_bin_dir(master)
    # if effective came from global and model has explicit bin, prefer model when effective is just DEFAULT
    use_effective = bool((master.get("llama_bin") or "").strip() or effective != DEFAULT_BIN or not model_cfg.get("llama_bin"))
    if use_effective:
        bin_dir = effective.rstrip("/\\")
        # windows: exe + backslashes; linux + macos: bare binary + forward slashes
        if target_os == "windows":
            server_bin = f"{bin_dir}\\llama-server.exe"
        else:
            server_bin = f"{bin_dir}/llama-server"
    else:
        if target_os == "windows":
            server_bin = model_cfg.get("llama_bin") or f"{effective}\\llama-server.exe"
        else:
            server_bin = model_cfg.get("llama_bin") or f"{effective}/llama-server"
    parts = [
        server_bin,
        f'-m "{model_path}"',
    ]

    # host/port are always appended below; skip them if already in flags.
    # Empty-valued flags are dropped: an emitted `--flag ` with no value
    # would swallow the NEXT flag as its value and shift every argument
    # after it (this produced bare-token 'invalid argument' crashes).
    # Some flags stored as bool actually REQUIRE a value upstream (e.g.
    # --cors-origins takes the allowed-origin list); emitting them bare
    # would swallow the next flag — emit the sensible default instead.
    flag_value_defaults = {"cors-origins": "*"}
    # In fit mode llama.cpp's own fitter assigns whole layers per device
    # using ACTUAL free memory measured in-process (it also projects
    # compute buffers + KV exactly). A user-set --n-gpu-layers NUMBER
    # makes the fitter abort ("n_gpu_layers already set by user") and
    # --tensor-split disables its auto-distribution entirely — upstream
    # docs are explicit. So fit mode skips both: steering happens via
    # --fit-target reserves only. n-gpu-layers still applies in manual
    # and auto modes.
    split_mode = model_cfg.get("tensor_split_mode", "auto")
    if split_mode == "manual":
        split_mode = "fit"   # legacy manual ratios -> fit (explicit weights
        # live under mode "split" + tensor_split string instead)
    computed_split = None
    if split_mode == "fit":
        # Policy: llama.cpp's --fit fitter has an unsigned-underflow
        # endless loop in its MoE dense-conversion step when the last
        # --device is a RAM/CPU-backed device left with 0 layers
        # (observed: n_layer=4291484623, one layer per ~60s probe, never
        # finishes -> llama-swap health-check timeout after 2h). Fit mode
        # is only safe when everything fits in VRAM. Any RAM device in
        # the order downgrades fit to an explicit computed split here —
        # the command stays correct without the buggy fitter path.
        device_order_probe = model_cfg.get("device_order") or ""
        has_ram = any(t.strip().endswith("-RAM")
                      for t in device_order_probe.split(",") if t.strip())
        if has_ram:
            computed_split = ",".join(
                str(w) for w in _computed_split_weights(model_cfg, master, peers))
            split_mode = "split"
            model_cfg = dict(model_cfg)
            model_cfg["tensor_split"] = computed_split
    skip_keys = ("host", "port")
    if split_mode == "fit":
        skip_keys += ("n-gpu-layers", "n_gpu_layers", "ngl")
    for key, value in flags.items():
        if key in skip_keys:
            continue
        if isinstance(value, str) and value == "":
            continue
        flag = key if key.startswith("-") else f"--{key}"
        if isinstance(value, str):
            value = path_resolver.resolve_for(value, target_os)
        if isinstance(value, bool):
            if value:
                plain = key.lstrip("-")
                if plain in flag_value_defaults:
                    parts.append(f"{flag} '{flag_value_defaults[plain]}'")
                else:
                    parts.append(flag)
        else:
            # shell-quote arbitrary string values: a user-supplied value
            # with ; / ` / && / | must stay one argv token, never code
            import shlex as _shlex
            parts.append(f"{flag} {_shlex.quote(str(value))}")

    if peers:
        # per-(model,peer,GROUP) endpoints: each co-served model gets its
        # own rpc-server per backend group (accelerators on one port, the
        # CPU/RAM backend on its own port), so the CPU backend never
        # serializes behind GPU tensors on a single-client connection.
        # Endpoint order here MUST match _map_devices numbering (both
        # iterate peers, then gpu-group devices, then the cpu device).
        raw = model_cfg.get("device_order") or ""
        port_of = _model_rpc_ports(model_cfg, peers)
        rpc_list = ",".join(
            f"{p['ip']}:{port_of.get((p['name'], gname), p.get('rpc_port', 50052))}"
            for p in peers
            for gname, _devs in _peer_server_groups(raw, p))
        parts.append(f'--rpc "{rpc_list}"')

    device_order = model_cfg.get("device_order") or _auto_device_order(master, peers)
    if device_order:
        # Convert UI chips (e.g. "aibox1-CUDA0", "box2-RAM") into real
        # llama-server --device tokens, like the proven 4-node script:
        #   master CUDAi -> CUDAi, master RAM -> CPU
        #   each selected peer GPU/RAM -> its own RPC<N> device, numbered
        #   sequentially across peers in --rpc order (GPUs by index, RAM last)
        device_order_clean = _device_order_to_llama(
            device_order, master, peers,
            model_cfg.get("device_order_policy", "as-selected"))
        parts.append(f'--device "{device_order_clean}"')

    if split_mode == "fit":
        parts.append("--fit on")
        parts.append(f"--fit-target {_fit_target_list(model_cfg, master, peers)}")
    elif split_mode == "split":
        # forced tensor-split: explicit per-device layer weights (same
        # order as --device), fitter off. Weights are relative ratios
        # (cf. #18273 integers 4,3,3,4,4), NOT gigabytes.
        weights = (model_cfg.get("tensor_split") or "").strip()
        if weights:
            parts.append("--fit off")
            parts.append(f"--tensor-split {weights}")
            # offload-all needs an explicit ngl; inject 99 when the user
            # didn't set one (fit mode deliberately strips it instead)
            norm = {str(k).lstrip("-").replace("_", "-") for k in flags}
            if not ({"n-gpu-layers", "ngl"} & norm):
                parts.append("--ngl 99")
    # auto: omit --tensor-split entirely; llama-server (and its fitter)
    # splits evenly by free device memory

    parts.append('--host 0.0.0.0')
    parts.append('--port "$PORT"')
    return " \\\n  ".join(parts)


def _parse_chip(token):
    """Split a UI chip like 'aibox1-CUDA0' or 'box2-RAM' into
    (machine, resource). Machine names may contain hyphens, so split
    on the LAST hyphen. Bare tokens (no hyphen) return ('', token)."""
    token = (token or "").strip()
    if not token:
        return "", ""
    if "-" not in token:
        return "", token
    machine, resource = token.rsplit("-", 1)
    return machine.strip(), resource.strip()


def _gpu_sort_key(name):
    digits = "".join(ch for ch in (name or "") if ch.isdigit())
    return int(digits) if digits else 0


def _peer_selected_resources(device_order, peer_name, peer):
    """Selected (gpus, ram) for one peer from UI device_order chips.

    SINGLE SOURCE OF TRUTH used both for the llama-server --device RPC
    mapping and for what each rpc-server exposes, so the two can never
    drift apart (a drifted RPC id is exactly the 'invalid device'
    failure). Falls back to all listed peer GPUs (no RAM) when the peer
    has no chips (auto/legacy orders); a GPU-less peer contributes its
    CPU, matching the rpc-server default on CPU-only boxes.
    """
    chips = []
    for tok in (device_order or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        machine, resource = _parse_chip(tok)
        if machine == peer_name:
            chips.append(resource)
    if not chips:
        gpus = [d.strip() for d in (peer.get("gpu", "") or "").split(",")
                if d.strip()]
        gpus = sorted(set(gpus), key=_gpu_sort_key)
        return gpus, (not gpus)
    gpus = sorted({r for r in chips if gpu_vendor.is_gpu_token(r)},
                  key=_gpu_sort_key)
    return gpus, ("RAM" in chips)


def compute_rpc_map(model_cfg, master, peers):
    """Machine-readable RPC id -> {machine, resource} map for a model.

    SINGLE SOURCE OF TRUTH for the visualizer: identical numbering to
    _rpc_legend/_map_devices/_peer_server_groups (peer order = --rpc
    order; per peer: gpu-group devices by index, then the cpu group's
    device iff RAM selected — mirrors --rpc endpoint order).
    Returns {} for single-node models (no RPC ids exist)."""
    rpc_map = {}
    counter = 0
    device_order = model_cfg.get("device_order") or ""
    for peer in peers:
        pname = peer.get("name", "")
        for _gname, devs in _peer_server_groups(device_order, peer):
            for r in devs:
                res = "RAM" if r == "CPU" else r
                rpc_map[f"RPC{counter}"] = {"machine": pname, "resource": res}
                counter += 1
    return rpc_map


def _rpc_legend(device_order, master, peers):
    """Human-readable map of RPC ids to hardware, e.g.
    'RPC0=aibox3-CUDA0, RPC1=aibox3-CUDA1, RPC2=aibox3-RAM'. Display
    only: logged by generated scripts so llama-server's RPC0..N errors
    can be traced back to a box. Never fed to any command."""
    parts = []
    counter = 0
    for peer in peers:
        pname = peer.get("name", "")
        for _gname, devs in _peer_server_groups(device_order, peer):
            for r in devs:
                label = "RAM" if r == "CPU" else r
                parts.append(f"RPC{counter}={pname}-{label}")
                counter += 1
    return ",".join(parts)


def _peer_backend_label(device_order, peer):
    """Display-only hardware description for one peer, e.g.
    'CUDA0 (RTX 2060 SUPER), CUDA1 (GTX 1080), CPU (7.7 GB RAM)'.
    Falls back to bare device names when detection data is missing."""
    gpus, ram = _peer_selected_resources(
        device_order, peer.get("name", ""), peer)
    detected = peer.get("detected") or {}
    gpu_names = detected.get("gpu_names") or []
    chunks = []
    for g in gpus:
        name = ""
        try:
            name = gpu_names[int("".join(
                ch for ch in g if ch.isdigit()))] or ""
        except (ValueError, IndexError):
            name = ""
        name = name.replace("NVIDIA GeForce ", "")
        chunks.append(f"{g} ({name})" if name else g)
    if ram:
        total = detected.get("ram_total_gb")
        chunks.append(f"CPU ({total} GB RAM)" if total else "CPU (RAM)")
    label = ", ".join(chunks) if chunks else "default (all GPUs)"
    return label.replace('"', "'").replace("$", "").replace("`", "")


def _peer_server_devices(device_order, peer):
    """Explicit --device list for starting this peer's rpc-server:
    its selected GPUs (by index) with CPU appended iff its RAM chip is
    selected. Order matches the client-side RPC numbering exactly, and
    mirrors how ggml-rpc-server enumerates an explicit --device list.
    Empty list = let the server use its default (all GPUs / CPU-only).

    NOTE: legacy single-server helper. Grouped deployments (GPU server
    + separate CPU server) use _peer_server_groups + _group_devices."""
    gpus, ram = _peer_selected_resources(
        device_order, peer.get("name", ""), peer)
    return gpus + (["CPU"] if ram else [])


def _peer_endpoint_order(model_cfg, master, peers):
    """Endpoints for --rpc in the order the client enumerates them:
    per peer (in peers order), per group (gpu first, then cpu), one
    endpoint per group. Returns [(peer, group_name, port)] and also
    drives RPC id numbering in _map_devices — the two can never drift
    because both iterate this same order."""
    raw_order = model_cfg.get("device_order") or ""
    endpoints = []
    for p in peers:
        for gname, _devs in _peer_server_groups(raw_order, p):
            endpoints.append((p, gname))
    return endpoints


def _vram_first_key(token, rpc_rev, master, peers):
    """Sort key implementing the vram-first policy: all GPUs before any
    CPU; within each tier the host (master) always leads, then machines
    by total VRAM desc (GPUs) or RAM desc (CPUs). Local PCIe beats LAN
    hops, so host-first is explicit, not incidental. Stable sort
    preserves chip order within ties."""
    peers_idx = {p.get("name", ""): i for i, p in enumerate(peers)}
    master_name = master.get("name", "")

    def _machine(machine):
        if machine == master.get("name", ""):
            return master
        for p in peers:
            if p.get("name", "") == machine:
                return p
        return {}

    def _vram(machine):
        try:
            return float((_machine(machine).get("detected") or {}).get("vram_total_gb") or 0)
        except (ValueError, TypeError):
            return 0.0

    def _ram(machine):
        try:
            return float((_machine(machine).get("detected") or {}).get("ram_total_gb") or 0)
        except (ValueError, TypeError):
            return 0.0

    if gpu_vendor.is_gpu_token(token):
        try:
            gidx = int("".join(ch for ch in token if ch.isdigit()))
        except ValueError:
            gidx = 0
        return (0, 0, -_vram(master_name), -1, gidx, token)
    if token == "CPU":
        return (1, 0, 0.0, -1, 0, token)
    if token.startswith("RPC") and token in rpc_rev:
        pname, resource = rpc_rev[token]
        if resource == "RAM":
            return (1, 1, -_ram(pname), peers_idx.get(pname, 99), 0, token)
        try:
            gidx = int("".join(ch for ch in resource if ch.isdigit()))
        except ValueError:
            gidx = 0
        return (0, 1, -_vram(pname), peers_idx.get(pname, 99), gidx, token)
    return (2, 1, 0.0, 99, 0, token)


def _detected_mem(machine, key):
    """Detected hardware stat (vram_total_gb / ram_total_gb) or 0."""
    try:
        return float((machine.get("detected") or {}).get(key) or 0)
    except (ValueError, TypeError, AttributeError):
        return 0.0


def _computed_split_weights(model_cfg, master, peers):
    """Estimate per-device layer weights for an explicit --tensor-split,
    so fit mode can be downgraded to split when RAM devices are present.

    Weights are PROPORTIONAL LAYER SHARES summed to ~n_layers — NOT
    "huge remainder" ratios. tensor-split in split-mode layer splits the
    layer count by weight ratio, so a 3482673-vs-11 weight gives the GPU
    devices ~0.001 layers (all layers spill to the RAM device; measured
    4 tok/s with every GPU idle). VRAM devices get shares by free VRAM;
    each RAM-backed device gets a small fixed slice for overflow."""
    n_layers = 48  # layer count doesn't affect relative weights; keep simple
    raw = model_cfg.get("device_order") or _auto_device_order(master, peers)
    policy = model_cfg.get("device_order_policy", "as-selected")
    items = _map_devices(raw, master, peers, policy)
    margin = int(model_cfg.get("fit_margin", 1024) or 1024)
    by_name = {master.get("name", ""): master}
    by_name.update({p.get("name", ""): p for p in peers})
    n_ram = sum(1 for _, _, is_ram in items if is_ram)
    ram_share = min(6, n_ram * 3)  # layers reserved for RAM spill total
    # capacity per device in MiB: VRAM free (fit_margin kept), RAM small
    caps = []
    for _tok, machine, is_ram in items:
        if is_ram:
            # RAM devices get a small fixed slice: enough for overflow,
            # never competing with VRAM
            caps.append(2048.0)
            continue
        mm = by_name.get(machine, {})
        total = _detected_mem(mm, "vram_total_gb") * 1024
        caps.append(max(1024.0, total - margin))
    gpu_share = n_layers - ram_share if n_ram else n_layers
    total_cap = sum(caps) or 1.0
    return [max(1, round(gpu_share * c / total_cap)) for c in caps]


def _map_devices(device_order, master, peers, policy="as-selected"):
    """Convert UI device_order chips into (token, machine, is_ram) items.

    Mirrors the proven start-qwen3-8-27b-4node.sh pattern where every
    selected remote GPU (and selected remote RAM, served by that peer's
    CPU backend) shows up as its own RPC<N> device, numbered
    sequentially across peers in --rpc order, while master GPUs stay
    CUDA<N>. Master (host) RAM needs no token: host memory is implicit
    to llama-server, so a master RAM chip emits nothing.
    policy="as-selected" emits in chip order (e.g. small-cards-first);
    policy="vram-first" sorts GPUs by machine VRAM desc, host CPU next,
    then peer CPUs -- closest to a fill-VRAM-then-spill cascade that a
    plain device list can express. RPC numbering never changes: only
    the output order is sorted.
    """
    master_name = master.get("name", "")
    # 1. assign RPC numbers per peer in --rpc (peers) order, using the
    #    same per-peer resource lists the rpc-servers are started with
    #    (_peer_server_groups, in _peer_endpoint_order), so numbering
    #    always matches reality. GPU-group devices enumerate first (in
    #    --device order), then the CPU group's single device — matching
    #    the --rpc endpoint order (gpu server, then cpu server per peer).
    rpc_of = {}  # (peer_name, resource) -> RPC<N>
    counter = 0
    for peer in peers:
        pname = peer.get("name", "")
        for gname, devs in _peer_server_groups(device_order, peer):
            for r in devs:
                res = "RAM" if r == "CPU" else r
                rpc_of[(pname, res)] = f"RPC{counter}"
                counter += 1
    # 2. emit --device tokens in the user's chip order
    out = []  # (token, machine, is_ram)
    seen = set()
    for tok in (device_order or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        machine, resource = _parse_chip(tok)
        if not machine:
            # already a raw token (e.g. from _auto_device_order): bare
            # CUDA is the master's; bare RPC unattributable (VRAM rule)
            if tok not in seen:
                seen.add(tok)
                out.append((tok, master_name if gpu_vendor.is_gpu_token(tok) else "",
                            False))
            continue
        if machine == master_name or (
            machine and not any(p.get("name") == machine for p in peers)
        ):
            # master resources stay local (host RAM is implicit: no token);
            # unknown machines pass through untouched, except a bare RAM
            # token which no backend would accept
            if resource == "RAM":
                continue
            if resource and resource not in seen:
                seen.add(resource)
                out.append((resource, machine, False))
            continue
        key = (machine, resource)
        if key not in rpc_of:
            # chip for a machine with no running rpc-server (e.g. stale
            # chip for a deselected server): skip it instead of inventing
            # an RPC id that llama-server would reject as invalid device
            continue
        if rpc_of[key] not in seen:
            seen.add(rpc_of[key])
            out.append((rpc_of[key], machine, resource == "RAM"))
    if policy == "vram-first":
        rpc_rev = {v: k for k, v in rpc_of.items()}
        out = sorted(out, key=lambda it: _vram_first_key(
            it[0], rpc_rev, master, peers))
    return out


def _device_order_to_llama(device_order, master, peers, policy="as-selected"):
    """--device string for llama-server (see _map_devices)."""
    return ",".join(t for t, _, _ in _map_devices(
        device_order, master, peers, policy))


def _fit_target_list(model_cfg, master, peers):
    """--fit-target reserve list aligned to the --device order: VRAM
    devices keep fit_margin free; RAM-backed devices keep all but
    fit_ram_allow free, so the fitter fills VRAM first and touches RAM
    only for overflow. Unmeasured RAM pools fall back to fit_margin.
    Runtime: uses live free (total−used) when available, so RAM calculations
    stay correct as usage changes (tensor-split and fit both)."""
    raw = model_cfg.get("device_order") or _auto_device_order(master, peers)
    policy = model_cfg.get("device_order_policy", "as-selected")
    items = _map_devices(raw, master, peers, policy)
    margin = int(model_cfg.get("fit_margin", 1024) or 1024)
    allow = int(model_cfg.get("fit_ram_allow", 4096) or 0)
    by_name = {master.get("name", ""): master}
    by_name.update({p.get("name", ""): p for p in peers})
    # live RAM for runtime-aware reserve
    live_by_machine = {}
    for nm, mm in by_name.items():
        try:
            live_by_machine[nm] = stats_collector.machine_stats(mm)
        except Exception:
            live_by_machine[nm] = None
    out = []
    for _tok, machine, is_ram in items:
        if not is_ram:
            out.append(str(margin))
            continue
        live = live_by_machine.get(machine)
        if live and live.get("status") == "online":
            total_mb = live.get("ram_total_gb", 0) * 1024
            used_mb = live.get("ram_used_gb", 0) * 1024
            free_mb = max(0, total_mb - used_mb)
            if total_mb <= 0:
                out.append(str(margin))
            else:
                # keep free−allow reserved, so fitter may use at most allow
                # out of free. Clamp: a target >= free would be a negative
                # fit budget, which trips llama.cpp's uint32 underflow in
                # the MoE dense-conversion step (observed as an
                # n_layer=4291484623 endless 1-layer/pass loop). Cap at
                # half of free so the target always leaves usable slack.
                out.append(str(max(0, min(int(free_mb - allow), int(free_mb // 2)))))
        else:
            total_mb = _detected_mem(by_name.get(machine, {}), "ram_total_gb") * 1024
            if total_mb <= 0:
                out.append(str(margin))
            else:
                out.append(str(max(0, min(int(total_mb - allow), int(total_mb // 2)))))
    return ",".join(out)


def _auto_device_order(master, peers):
    """CUDA devices of the master, then one RPC<N> per remote GPU/RAM
    exposed by each peer (only used when the model config has no
    explicit device_order)."""
    devices = [d.strip() for d in master.get("gpu", "").split(",") if d.strip()]
    n = 0
    for peer in peers:
        gpus = [d.strip() for d in (peer.get("gpu", "") or "").split(",") if d.strip()]
        for _ in gpus or [None]:
            # one RPC device per remote GPU; peers with no listed GPU
            # still contribute a single (CPU/RAM) RPC device
            devices.append(f"RPC{n}")
            n += 1
    return ",".join(devices)


# ---------------- script generation ----------------
def generate_script(model_cfg, servers, script_path=None):
    """Write a standalone .sh mirroring the user's script structure:
    stop_rpc/start_rpc/wait_for_rpc functions, /dev/tcp wait, traps.
    Single-node models get the simple exec form (start-huihui-35b.sh)."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    script_path = script_path or os.path.join(
        GENERATED_DIR, f"start-{model_cfg.get('model_id', 'model')}.sh"
    )
    master, peers = _split_servers(servers)

    if not peers:
        content = _render_single_node(model_cfg, master)
    else:
        content = _render_rpc(model_cfg, master, peers)

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    try:
        # 0700: scripts drive SSH sessions via $SSH_PASS — owner-only even
        # though no secret is baked in anymore (defense in depth)
        os.chmod(script_path, 0o700)
    except OSError:
        pass  # no-op on Windows
    return script_path


def _rpc_env_exports(machine):
    """Per-machine env prefix for ggml-rpc-server launches (both the
    live start_rpc() remote command and rendered launcher scripts).

    Always includes GGML_SCHED_DEBUG=3 / GGML_RPC_DEBUG=3: real
    per-tensor backend assignment + RPC peer logs are the ground truth
    the visualizer and log tails animate — a silent rpc-server is
    undebuggable.

    rdma=False (default): export GGML_RPC_NO_RDMA=1 — pure TCP RPC. A wedged
    RoCE QP (seen under heavy sustained tensor streaming on mlx4) freezes the
    connection with no timeout on either side, so RDMA is strictly opt-in.

    rdma=True: export GGML_RDMA_DEV/GGML_RDMA_GID when the machine's RoCE GID
    index is known, pinning llama.cpp's GID matcher to the right RoCE v2 entry
    instead of relying on auto-detection.

    NVIDIA peers also get CUDA_DEVICE_ORDER=PCI_BUS_ID. Without it the CUDA
    runtime numbers GPUs fastest-first while nvidia-smi (and our detection,
    labels, and --device tokens) number by PCI bus — on a heterogeneous
    peer (e.g. 3050 + 4080) --device CUDA1 then lands on the WRONG card.
    The master already gets this via _device_env_exports; the rpc-server
    side was missing it, which is exactly that class of bug.

    When per-GPU UUIDs were captured at detect time, CUDA_VISIBLE_DEVICES
    (those UUIDs, PCI order) is emitted INSTEAD of the order pin: identity
    beats ordering, and old runtimes that fail CUDA init outright under
    PCI_BUS_ID (seen live: v7-era ggml on winbox) enumerate cleanly under
    a UUID list. No UUIDs -> the order pin (status quo)."""
    pre = "GGML_SCHED_DEBUG=3 GGML_RPC_DEBUG=3 "
    try:
        _vis = gpu_vendor.cuda_visible_devices(machine)
    except Exception:
        _vis = ""
    if _vis:
        pre += f"CUDA_VISIBLE_DEVICES={_vis} "
    else:
        try:
            _vv = gpu_vendor.normalize_vendor((machine or {}).get("gpu_vendor"))
        except Exception:
            _vv = ""
        if _vv not in ("amd", "intel", "apple", "cpu"):
            # Unknown/empty vendor defaults to NVIDIA (same as gpu_prefix):
            # CUDA_DEVICE_ORDER is a no-op on non-NVIDIA runtimes anyway.
            pre += "CUDA_DEVICE_ORDER=PCI_BUS_ID "
    if not machine.get("rdma"):
        return pre + "GGML_RPC_NO_RDMA=1 "
    if _vv not in ("amd", "intel", "apple", "cpu"):
        # Unknown/empty vendor defaults to NVIDIA (same as gpu_prefix):
        # CUDA_DEVICE_ORDER is a no-op on non-NVIDIA runtimes anyway.
        pre += "CUDA_DEVICE_ORDER=PCI_BUS_ID "
    if not machine.get("rdma"):
        return pre + "GGML_RPC_NO_RDMA=1 "
    dev = machine.get("rdma_dev") or ""
    gid = machine.get("rdma_gid")
    if dev:
        pre += f"GGML_RDMA_DEV={dev} "
    if gid is not None and gid != "":
        pre += f"GGML_RDMA_GID={gid} "
    # GGML_RDMA_USE_CM=1 -> rdma_cm QP path resolution (ConnectX-3/mlx4).
    # Unset -> upstream raw ibv_modify_qp path (mlx5/E810). TCP stays available
    # via GGML_RPC_NO_RDMA=1 when rdma=False.
    if machine.get("rdma_cm"):
        pre += "GGML_RDMA_USE_CM=1 "
    return pre


def _parse_env_exports(env_pre):
    """Parse 'export K=V\\nexport K2=V2' lines into [(key, value)] pairs so
    Windows launchers (PowerShell env) can reuse the same export source."""
    out = []
    for line in (env_pre or "").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if not line or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out.append((k.strip(), v.strip()))
    return out


def _device_env_exports(master):
    """Env exports the llama-server command needs, per GPU vendor.
    CUDA_DEVICE_ORDER is NVIDIA-only; ROCm ignores it (harmless but
    noisy), so AMD hosts get P2P only.

    RDMA handling: the master's llama-server is the RPC *client* — it
    auto-negotiates RDMA per connection and there is no client-side
    GGML_RPC_NO_RDMA to force TCP. Gating happens on the SERVER side
    (rpc-server env, see _rpc_env_exports): a peer with rdma=False simply
    never offers RDMA. When the master itself has rdma=True we may pin
    GGML_RDMA_DEV/GGML_RDMA_GID so its GID matcher picks the right RoCE
    v2 entry."""
    v = gpu_vendor.normalize_vendor(master.get("gpu_vendor"))
    # GGML_SCHED_DEBUG=3: real per-tensor backend/GPU assignment + cross-device
    # copies — ground truth the flow visualizer animates. GGML_RPC_DEBUG=3:
    # same for the RPC peer itself (single-RPC models like winbox-only would
    # otherwise be silent — the host has nothing to schedule).
    sched = "export GGML_SCHED_DEBUG=3\nexport GGML_RPC_DEBUG=3"
    rdma = ""
    if master.get("rdma"):
        if master.get("rdma_dev"):
            rdma += f"\nexport GGML_RDMA_DEV={master['rdma_dev']}"
        if master.get("rdma_gid") not in (None, ""):
            rdma += f"\nexport GGML_RDMA_GID={master['rdma_gid']}"
        if master.get("rdma_cm"):
            rdma += "\nexport GGML_RDMA_USE_CM=1"
    if v == "amd":
        return f"export GGML_CUDA_P2P=1\n{sched}{rdma}"
    if v == "intel":
        # SYCL: persist the oneAPI kernel cache so model loads stay fast
        return f"export SYCL_CACHE_PERSISTENT=1\n{sched}{rdma}"
    # UUID identity beats ordering (same heterogeneous-peer argument as
    # _rpc_env_exports): when detect captured GPU UUIDs, pin exact cards
    # via CUDA_VISIBLE_DEVICES and skip the order pin (old runtimes can
    # fail CUDA init under PCI_BUS_ID). No UUIDs -> the order pin.
    try:
        _vis = gpu_vendor.cuda_visible_devices(master)
    except Exception:
        _vis = ""
    if _vis:
        return f"export CUDA_VISIBLE_DEVICES={_vis}\nexport GGML_CUDA_P2P=1\n{sched}{rdma}"
    return f"export CUDA_DEVICE_ORDER=PCI_BUS_ID\nexport GGML_CUDA_P2P=1\n{sched}{rdma}"


def _render_single_node(model_cfg, master):
    """Mirrors start-huihui-35b.sh."""
    cmd = build_llama_command(model_cfg, [master["name"]])
    return f"""#!/bin/bash
{_device_env_exports(master)}
PORT="${{1:-8080}}"
# Preflight: llama-swap owns $PORT for this model. If a previous instance
# is still shutting down, wait briefly; if a FOREIGN process holds it,
# fail fast with a clear message instead of a cryptic bind death.
for _i in $(seq 1 10); do
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break
  sleep 1
done
if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "!!! port $PORT is held by another process (not llama-swap) — refusing to start" >&2
  ss -ltnp 2>/dev/null | grep ":$PORT " >&2 || true
  exit 1
fi
exec {cmd}
"""


def render_remote_local_script(model_cfg, master, remote_port,
                               state_key=None):
    """launcher script for launch-mode=local with a REMOTE llama-server box.

    Written to the swap scripts_dir and executed by llama-swap's cmd on
    model load. Over SSH it: (1) short-circuits when the remote port is
    already serving — llama-swap reloads its config on every push and
    would otherwise restart a healthy server on each request; (2) clears
    stale llama processes on that box (pre-launch hygiene, same rule as
    the live RPC path); (3) starts llama-server ON THAT BOX with ITS OWN
    GPUs (fit mode distributes by actual free VRAM; no --rpc tokens, no
    network tensor hops — the entire point of local mode); (4) returns
    only when the port answers, so llama-swap's proxy comes up against a
    live upstream. Requires $SSH_PASS (sourced from the sidecar) and the
    model file readable on the box (store:// alias resolved for the
    box's OS — the shared CIFS mount covers it).

    ACTUAL-PORT STATE: the launcher may bind a HIGHER port than the
    config's probe point (`_roll_port_if_busy` — a stale co-served
    server or any listener pushes it forward). cmdStop is generated
    with the config port and would then MISS the live server, leaving
    its VRAM held after a swap unload (observed: group swap between
    two DAD variants left --dad holding 7+11+8 GB while --DAD loaded).
    So the script records every port it (re)serves into the per-model
    state file `.cache/llama-wrapper/swap.port.<model_id>` on the box —
    written on the REUSE path too, since a previously-rolled port is
    exactly the one cmdStop must find. swap_manager builds cmdStop as
    a loop over `cat` of that file PLUS the config port fallback.
    """
    model_cfg = dict(model_cfg)
    # single-box rewrite (same as launch_model mode="local"): the box's
    # own accelerators only, fit mode, no RPC split, concrete port (no
    # ${PORT} macro exists on this path — llama-swap's cmd passes none)
    expose = (model_cfg.get("expose") or {}).get(master["name"]) or {}
    raw_gpus = expose.get("gpus") or master.get("gpu", "") or []
    if isinstance(raw_gpus, str):
        raw_gpus = raw_gpus.split(",")
    gpus = [d.strip() for d in raw_gpus if str(d).strip()]
    model_cfg["device_order"] = ",".join(gpus)
    # single-box split policy: keep an explicit split the variant asked
    # for (tensor_split_mode 'split' + non-empty weights — RATIO weights,
    # layer count irrelevant); only fall back to fit mode otherwise. The
    # fitter's by-free-VRAM distribution can thrash when a slow big-VRAM
    # card (P100) fills up and spills layers to CPU (observed: 121 CPU
    # layers on the 27B dad model = half the model in system RAM).
    if not (model_cfg.get("tensor_split_mode") == "split"
            and (model_cfg.get("tensor_split") or "").strip()):
        model_cfg["tensor_split_mode"] = "fit"
        model_cfg["tensor_split"] = ""
    cmd = build_llama_command(model_cfg, [master["name"]])
    # the rendered command is multi-line (" \\\n  " joins) — a remote
    # bash -c '<cmd>' would only take line 1 and the rest would be parsed
    # as bare `export` tokens. Collapse to one shell line first.
    cmd = cmd.replace(" \\\n  ", " ").replace('"$PORT"', str(remote_port))
    env = _device_env_exports(master).replace("\n", "; ")
    if master.get("os", "linux") == "windows":
        # build_llama_command emits POSIX quoting (shlex single-quotes);
        # cmd.exe wants double quotes. Rewrite: 'x y' -> "x y", drop
        # backslash-escapes (cmd.exe has none), strip empty quotes.
        import re as _re
        cmd = _re.sub(r"'([^']*)'", lambda m: '"' + m.group(1) + '"', cmd)
        cmd = cmd.replace('\\"', '"')
        cmd = _re.sub(r'""+', '"', cmd)
        # PowerShell single-quoted string literal: only ' needs doubling
        cmd_var = "'" + cmd.replace("'", "''") + "'"
    user = master.get("username") or "root"
    ip = (master.get("ip") or "").strip()
    # Per-port remote log: co-served models (and successive attempts of
    # the SAME model) must never overwrite each other's server output —
    # a shared llama-server.log destroys every post-mortem.
    remote_log = f"llama-server-{remote_port}.log"
    log = remote_log
    # per-group actual-port state file (see docstring: cmdStop must find
    # a rolled port). `state_key` is the deployment's SERVE GROUP when it
    # belongs to one (group-scoped kills — the swap semantic), else its
    # model id; resolved by swap_manager at push time and passed in —
    # never hard-coded here. Filename derivation is shared
    # (_swap_state_file) with the cmdStop builder so they cannot drift.
    state_name = _swap_state_file(state_key or model_cfg.get("model_id")
                                  or model_cfg.get("_file") or "model")
    state_rel = f".cache/llama-wrapper/{state_name}"
    # Windows path form for the ps1 branch below (backslashes, no $)
    state_rel_windows = f".cache\\llama-wrapper\\{state_name}"
    # pkill/pgrep pattern: cmdline may end right after --port (see
    # _port_match_re — trailing-space-only never matched there)
    port_re = _port_match_re(remote_port)
    if master.get("os", "linux") == "windows":
        # Windows llama-server box: push a .ps1 launcher (same mechanics
        # as _win_launcher: Win32_Process.Create detached start, $exe
        # resolution honoring machine llama_bin) and drive it via
        # sshpass/ssh from the swap host. The .ps1 embeds the env vars,
        # the model command (paths already resolved for Windows by
        # build_llama_command), and its own port-wait — the bash wrapper
        # only uploads, runs, and polls the port over SSH.
        env_ps = "; ".join(
            f'$env:{k}="{v}"'
            for k, v in _parse_env_exports(
                _device_env_exports(master)) if k and v)
        ps1 = f"""$ErrorActionPreference = 'Stop'
{env_ps}
$port = {remote_port}
# reuse guard: a healthy server on this port is left untouched
try {{ $c = New-Object Net.Sockets.TcpClient('127.0.0.1', $port); $c.Close(); Write-Output 'already serving'; exit 0 }} catch {{ }}
$exe = {_win_exe_expr(master, server='llama-server')}
if (-not (Test-Path $exe)) {{ Write-Error "not found: $exe"; exit 1 }}
# record ACTUAL port for cmdStop (see render_remote_local_script docstring)
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\\.cache\\llama-wrapper" | Out-Null
Set-Content -Path "$env:USERPROFILE\\{state_rel_windows}" -Value "{remote_port}"
# Retry-cascade guard (same as the POSIX branch): a previous attempt may
# have started this server seconds ago - never pkill a still-starting
# process, or every retry resets a multi-minute load to zero.
$existing = @(Get-CimInstance Win32_Process | Where-Object {{ $cl = ' ' + $_.CommandLine + ' '; (($cl -like '* --port {remote_port} *') -or ($cl -like '* --port {remote_port}')) -and ($_.Name -like 'llama-server*') }})
if ($existing.Count -gt 0) {{ Write-Output 'already starting - waiting, no kill' }}
else {{
  $existing | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}
  Start-Sleep -Seconds 1
  $log = Join-Path $env:USERPROFILE '{log}'
  $cmd = ('cmd.exe /c "' + {cmd_var} + '" > "{{0}}" 2>&1') -f $log
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = $cmd }}
  if ($r.ReturnValue -ne 0) {{ Write-Error "Win32_Process.Create failed: $($r.ReturnValue)"; exit 1 }}
}}
# ~2h load/readiness budget (2400 x ~3s) - same thin-link rule as the
# POSIX branch: slow loads never time out here, only a dead process
# (fast-fail) or an explicit evict ends the wait. ASCII ONLY in this
# whole ps1: Win PowerShell 5.1 -File reads BOM-less scripts as
# Windows-1252, where an em-dash byte (0x94) decodes as a literal
# double-quote - toggling string state and breaking the parse with
# MissingEndCurlyBrace errors (observed live on winbox).
for ($i = 0; $i -lt 2400; $i++) {{
  try {{ $c = New-Object Net.Sockets.TcpClient('127.0.0.1', $port); $c.Close(); Write-Output 'server up'; exit 0 }} catch {{ Start-Sleep -Seconds 2 }}
  if (($i % 20) -eq 0) {{
    $alive = @(Get-CimInstance Win32_Process | Where-Object {{ $cl = ' ' + $_.CommandLine + ' '; (($cl -like '* --port {remote_port} *') -or ($cl -like '* --port {remote_port}')) -and ($_.Name -like 'llama-server*') }})
    if ($alive.Count -eq 0) {{ Write-Error "server process died before opening port {remote_port} - check ~/{log}"; exit 1 }}
  }}
}}
Write-Error "port {remote_port} did not come up in ~2h - check ~/{log}"
exit 1
"""
        sh = f"""#!/bin/bash
# launch-mode=local remote launcher (windows) — model lives on {master['name']}
set -uo pipefail
: "${{SSH_PASS:?SSH_PASS is not set — sourced from swap-secrets.env}}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
# reuse guard over the network first: skip upload entirely when serving
sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
  "powershell -NoProfile -Command \\"try {{ \\$c = New-Object Net.Sockets.TcpClient('127.0.0.1', {remote_port}); \\$c.Close(); exit 0 }} catch {{ exit 1 }}\\"" 2>/dev/null && {{
  echo ">>> {master['name']}:{remote_port} already serving — reusing"
  # stay alive as llama-swap's child until the port closes (see linux note)
  while sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
      "powershell -NoProfile -Command \\"try {{ \\$c = New-Object Net.Sockets.TcpClient('127.0.0.1', {remote_port}); \\$c.Close(); exit 0 }} catch {{ exit 1 }}\\"" 2>/dev/null; do
    sleep 5
  done
  echo ">>> {master['name']}:{remote_port} closed — exiting"
  exit 0
}}
tmp="$(mktemp /tmp/start-remote-local.XXXXXX.ps1)"
cat > "$tmp" <<'PS1'
{ps1}PS1
sshpass -p "$SSH_PASS" scp $SSH_OPTS -q "$tmp" "{user}@{ip}:start-remote-local.ps1"
rc=$?
rm -f "$tmp"
[ $rc -ne 0 ] && {{ echo ">>> ps1 upload failed" >&2; exit 1; }}
echo ">>> starting llama-server on {master['name']}:{remote_port}"
# -File path in DOUBLE quotes (\\" idiom): remote cmd.exe expands
# %USERPROFILE% and hands child powershell.exe a double-quoted path —
# the proven shape from the RPC launcher (line ~2221). Single quotes
# arrive doubled (''C:\...'') and -File rejects them ("path's format
# is not supported", observed live on winbox).
sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
  "powershell -NoProfile -ExecutionPolicy Bypass -File \\"%USERPROFILE%\\start-remote-local.ps1\\""
# stay alive as llama-swap's child until cmdStop closes the port
while sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "powershell -NoProfile -Command \\"try {{ \\$c = New-Object Net.Sockets.TcpClient('127.0.0.1', {remote_port}); \\$c.Close(); exit 0 }} catch {{ exit 1 }}\\"" 2>/dev/null; do
  sleep 5
done
echo ">>> {master['name']}:{remote_port} closed — exiting"
exit 0
"""
        return sh
    return f"""#!/bin/bash
# launch-mode=local remote launcher (generated) — model lives on {master['name']}
set -uo pipefail
: "${{SSH_PASS:?SSH_PASS is not set — sourced from swap-secrets.env}}"
{env} > /dev/null 2>&1 || true
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
# Actual-port state write: {state_rel} on the box records the port this
# model's server really serves (config port when it bound directly, the
# rolled port otherwise). cmdStop reads it — see the ACTUAL-PORT STATE
# note in this module's render_remote_local_script docstring.
_state_write="bash -c 'mkdir -p \\$HOME/.cache/llama-wrapper && echo {remote_port} > \\$HOME/{state_rel}'"
sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" "$_state_write" 2>/dev/null || true
# Reuse probe, retried: on thin links a single dropped SSH probe must
# never read as "port closed" — that misread would pkill a healthy
# server below and restart a multi-minute load for nothing.
for _i in $(seq 1 5); do
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "bash -c 'exec 3<>/dev/tcp/127.0.0.1/{remote_port}'" 2>/dev/null && {{
    echo ">>> {master['name']}:{remote_port} already serving — reusing"
    # STAY ALIVE as llama-swap's child until the port closes (cmdStop's
    # pkill closes it) — exiting would trip 'upstream exited prematurely'
    while sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
        "bash -c 'echo > /dev/tcp/127.0.0.1/{remote_port}'" 2>/dev/null; do
      sleep 5
    done
    echo ">>> {master['name']}:{remote_port} closed — exiting"
    exit 0
  }}
  sleep 2
done
echo ">>> starting llama-server on {master['name']}:{remote_port}"
# Retry-cascade guard: a previous load attempt may have started this very
# server seconds ago (multi-minute tensor loads). pkill-ing blindly here
# resets that load to zero on EVERY retry — the reset loop. When a
# matching server process is already alive but its port isn't open yet,
# skip the kill AND the start and go straight to waiting on it.
if sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "pgrep -u {user} -f 'llama-[s]erver.*{port_re}' >/dev/null" 2>/dev/null; then
  echo ">>> {master['name']}:{remote_port} already starting (process alive, port closed) — waiting, no kill"
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" "$_state_write" 2>/dev/null || true
else
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "pkill -u {user} -f 'llama-[s]erver.*{port_re}' 2>/dev/null; true"
  sleep 1
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "{env}; nohup bash -c '{cmd}' </dev/null > \\$HOME/{remote_log} 2>&1 &"
fi
# Load/readiness budget: ~2h (2400 polls x ~3s). Thin-link/CIFS farm
# rule — a slow load must NEVER time out on our end; only a dead
# server process (fast-fail below) or an explicit evict ends the wait.
for _i in $(seq 1 2400); do
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
    "bash -c 'echo > /dev/tcp/127.0.0.1/{remote_port}'" 2>/dev/null && {{
    echo ">>> {master['name']}:{remote_port} is up"
    # STAY ALIVE as llama-swap's child: cmd IS the tracked upstream —
    # exiting here would trip 'upstream exited prematurely'. Hold until
    # the remote port CLOSES (cmdStop's SSH pkill closes it), then exit
    # so llama-swap sees a clean stop.
    while sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
        "bash -c 'echo > /dev/tcp/127.0.0.1/{remote_port}'" 2>/dev/null; do
      sleep 5
    done
    echo ">>> {master['name']}:{remote_port} closed — exiting"
    exit 0
  }}
  # fast-fail: server process died without ever opening the port (bad
  # flags, missing GGUF, OOM) — say so now instead of idling the full
  # ~2h and tripping 'upstream exited prematurely' with no context.
  # (A missing process is a real failure, never a timeout.)
  if [ $((_i % 20)) -eq 0 ]; then
    sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "{user}@{ip}" \\
      "pgrep -u {user} -f 'llama-[s]erver.*{port_re}' >/dev/null" 2>/dev/null || {{
      echo "!!! {master['name']}:{remote_port} server process died before opening the port — check ~/{remote_log} there" >&2
      exit 1
    }}
  fi
  sleep 2
done
echo "!!! {master['name']}:{remote_port} did not come up in ~2h — check ~/{remote_log} there" >&2
exit 1
"""


def _render_rpc(model_cfg, master, peers, cmd=None):
    """Generate a per-model start script.

    Per-model RPC ports: ggml-rpc-server is single-client, so each
    (model, peer, GROUP) gets its OWN rpc-server on a distinct port on
    the peer (port = _model_rpc_port(model_id, peer_name, group)).
    Groups: ALL selected accelerators of a peer share one server
    ("gpu" group — every CUDA/ROCm/... token in one endpoint); the
    CPU/RAM backend runs as its own separate server ("cpu" group) so
    host-memory work never serializes behind GPU tensors on the
    single RPC client connection. A peer needs two script entries only
    when its RAM chip is selected; otherwise behavior is identical to
    the legacy per-peer layout. Each port has its own rpc.state.{port}
    (backend-set record) and rpc-server-{port}.log. Kill/stop are
    port-targeted so a co-served model's rpc-server on another port is
    never touched. On EXIT only servers this script itself (re)started
    are stopped. Explicit teardown via stop_rpc(port).
    """
    # macos peers speak POSIX too (pkill/pgrep/$HOME all exist) — they ride
    # the linux path; only setsid/timeout differ (handled at runtime below).
    linux = [p for p in peers if p.get("os", "linux") in ("linux", "macos")]
    win = [p for p in peers if p.get("os", "linux") == "windows"]
    raw_order = model_cfg.get("device_order") or _auto_device_order(master, peers)
    # per-(model,peer,GROUP) ports — deterministic, collision-free within model
    peer_ports = _model_rpc_ports(model_cfg, peers)
    # script "nodes": one entry per (peer, group). The node tag is the IP
    # for single-group peers; "IP#cpu" for the second (CPU) server so the
    # per-IP associative maps never clobber each other. Each tag keeps:
    # ip, group, port, devices, log, label. The ssh target is the IP part.
    nodes = []  # {tag, ip, group, os, port, devs, label}
    for p in peers:
        pgroups = _peer_server_groups(raw_order, p)
        for gname, devs in pgroups:
            multi = len(pgroups) > 1
            tag = f'{p["ip"]}#{gname}' if multi else p["ip"]
            nodes.append({
                "tag": tag, "ip": p["ip"], "group": gname,
                "os": p.get("os", "linux"),
                "port": peer_ports.get((p["name"], gname),
                                       p.get("rpc_port", 50052)),
                "devs": list(devs),
                "label": _peer_backend_label(raw_order, p),
                "machine": p,
            })
    policy = model_cfg.get("device_order_policy", "as-selected")
    device_order = _device_order_to_llama(raw_order, master, peers, policy)
    if cmd is None:
        cmd = build_llama_command(model_cfg, [master["name"]] + [p["name"] for p in peers])

    linux_nodes = " ".join(f'"{n["ip"]}"' for n in nodes if n["os"] in ("linux", "macos"))
    win_nodes = " ".join(f'"{n["ip"]}"' for n in nodes if n["os"] == "windows")
    win_users = {p["username"] for p in win}
    win_user = win_users.pop() if win_users else "zazit"
    # NOTE: the parametrized .ps1 takes per-node --device via -ExtraArgs,
    # so expose_args is intentionally "" (a baked master --device would
    # duplicate the flag on every Windows peer).
    # per-server backends, keyed by node TAG (IP, or IP#cpu for the
    # separate CPU/RAM server); the SAME lists the --device RPC mapping
    # counts, so numbering can never drift from reality
    peer_dev_pairs = []
    for n in nodes:
        if n["devs"]:
            peer_dev_pairs.append(f'["{n["tag"]}"]="{",".join(n["devs"])}"')
    peer_devices_decl = "declare -A PEER_DEVICES" + (
        "\n" + "\n".join(f"PEER_DEVICES{a}" for a in peer_dev_pairs)
        if peer_dev_pairs else "")
    # per-server ports, keyed by node TAG
    peer_port_pairs = [f'["{n["tag"]}"]="{n["port"]}"' for n in nodes]
    peer_ports_decl = "declare -A RPC_PORTS\n" + "\n".join(
        f"RPC_PORTS{a}" for a in peer_port_pairs)
    # per-server log names, keyed by node TAG
    peer_log_pairs = [f'["{n["tag"]}"]="{_rpc_log_name(n["port"])}"' for n in nodes]
    peer_logs_decl = "declare -A RPC_LOGS\n" + "\n".join(
        f"RPC_LOGS{a}" for a in peer_log_pairs)
    # per-server rpc-server binary, keyed by TAG: each peer launches from
    # its OWN bin dir (master layout must never be assumed on a peer —
    # a Windows master + Linux peer would otherwise bake a bad path).
    # (pre-expanded: the f-string below can't nest ${..:-..} braces)
    _master_rpc_bin = _bin_subdir(master, "ggml-rpc-server")
    rpc_bins_bin = "${RPC_BINS[$node]:-" + _master_rpc_bin + "}"
    peer_bin_pairs = [f'["{n["tag"]}"]="{_bin_subdir(n["machine"], "ggml-rpc-server")}"'
                      for n in nodes if n["os"] in ("linux", "macos")]
    peer_bins_decl = "declare -A RPC_BINS\n" + "\n".join(
        f"RPC_BINS{a}" for a in peer_bin_pairs)
    # per-server transport mode: 1 = native rdma_cm (no TCP listener on the
    # peer, so health checks must NOT use /dev/tcp), empty = classic TCP.
    peer_cm_pairs = [f'["{n["tag"]}"]="1"' for n in nodes
                     if n["machine"].get("rdma") and n["machine"].get("rdma_cm")]
    peer_cm_decl = "declare -A RPC_CM\n" + "\n".join(
        f"RPC_CM{a}" for a in peer_cm_pairs)
    # per-server RDMA env for the rpc-server launch, keyed by TAG.
    # rdma=False -> GGML_RPC_NO_RDMA=1 (TCP only, default); rdma=True ->
    # optional GGML_RDMA_DEV/GGML_RDMA_GID pinning. NVIDIA peers also carry
    # CUDA_DEVICE_ORDER=PCI_BUS_ID so --device indexes match nvidia-smi
    # (see _rpc_env_exports); the fallback below mirrors that default.
    peer_env_pairs = [f'["{n["tag"]}"]="{_rpc_env_exports(n["machine"]).strip()}"'
                      for n in nodes]
    peer_envs_decl = "declare -A RPC_ENVS\n" + "\n".join(
        f"RPC_ENVS{a}" for a in peer_env_pairs)
    # display-only labels + RPC legend (never fed to any command)
    label_pairs = [f'["{n["tag"]}"]="{n["label"]}"' for n in nodes]
    peer_labels_decl = ("declare -A PEER_LABELS\n" + "\n".join(
        f"PEER_LABELS{a}" for a in label_pairs))
    rpc_legend = _rpc_legend(raw_order, master, peers)
    # the .ps1 executes ON Windows peers: its exe path must come from a
    # Windows machine's bin dir. Using master (usually Linux) baked a Linux
    # path under USERPROFILE and every win launch failed its own check.
    # Port is passed at runtime via -Port, so one template serves all peers.
    ps1_machine = next((p for p in peers if p.get("os", "linux") == "windows"), master)
    ps1_text = _win_launcher(ps1_machine, 50052, expose_args="",
                             parametrized=True)

    # SSH password is NEVER baked into the script (vault-only). It is
    # supplied at runtime via $SSH_PASS — see scripts/farm-run.sh, which
    # decrypts the vault entry and execs the generated script. This keeps
    # generated/*.sh free of plaintext secrets (they are git-ignored but
    # sit on disk readable).
    return f"""#!/bin/bash
set -uo pipefail
LINUX_USER="{master['username']}"
WIN_USER="{win_user}"
: "${{SSH_PASS:?SSH_PASS is not set — run via scripts/farm-run.sh <machine> <script> (vault decrypts it at runtime, never baked into this file)}}"
# NODES are per-SERVER entries (tag = IP, or IP#cpu for the separate
# CPU/RAM server). SSH targets strip the #group suffix.
LINUX_NODES=({linux_nodes})
WIN_NODES=({win_nodes})
ALL_NODES=("${{LINUX_NODES[@]}}" "${{WIN_NODES[@]}}")
node_ip() {{ local t="$1"; echo "${{t%%#*}}"; }}
PORT="${{1:-8080}}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
WIN_SSH_OPTS="$SSH_OPTS -o BatchMode=yes"
DEVICE_ORDER="{device_order}"
{peer_devices_decl}
{peer_ports_decl}
{peer_logs_decl}
{peer_envs_decl}
{peer_bins_decl}
{peer_cm_decl}
{peer_labels_decl}

    start_rpc_linux() {{
      local node="$1"
      local port="${{RPC_PORTS[$node]:-50052}}"
      local log="${{RPC_LOGS[$node]:-rpc-server.log}}"
      echo ">>> starting rpc-server on $node:$port (linux)"
      local devs="${{PEER_DEVICES[$node]:-}}"
      local extra=""
      [[ -n "$devs" ]] && extra=" --device $devs"
      echo ">>> rpc backends: ${{devs:-default}}  [${{PEER_LABELS[$node]:-}}]"
      # Kill only the rpc-server on THIS port (co-served models on other
      # ports must survive), then start a fresh one on this port.
      sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
        "pkill -u $LINUX_USER -f \\"ggml-[r]pc-server.*-p $port \\"" >/dev/null 2>&1 || true
      sleep 1
      local envs="${{RPC_ENVS[$node]:-GGML_SCHED_DEBUG=3 GGML_RPC_DEBUG=3 CUDA_DEVICE_ORDER=PCI_BUS_ID GGML_RPC_NO_RDMA=1}}"
      # setsid: Linux util-linux. macOS lacks it — fall back to nohup.
      # ($envs already carries GGML_SCHED_DEBUG/RPC_DEBUG + transport flags
      # from the machine config — single source of truth, no hardcoding.)
      sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
         "export $envs; \\
          setsid \\"{rpc_bins_bin}\\" \\
           -H 0.0.0.0 -p $port -t \\"$(nproc --all 2>/dev/null || sysctl -n hw.ncpu)\\"$extra \\
           </dev/null > \\"$HOME/$log\\" 2>&1 & \\
         command -v setsid >/dev/null 2>&1 || \\
         nohup \\"{rpc_bins_bin}\\" \\
           -H 0.0.0.0 -p $port -t \\"$(nproc --all 2>/dev/null || sysctl -n hw.ncpu)\\"$extra \\
           </dev/null >> \\"$HOME/$log\\" 2>&1 &"
    }}
push_win_launcher() {{
  local node="$1" tmp rc
  tmp="$(mktemp /tmp/start-rpc.XXXXXX.ps1)"
  cat > "$tmp" <<'PS1'
{ps1_text}PS1
  scp $WIN_SSH_OPTS -q "$tmp" "$WIN_USER@$(node_ip "$node"):start-rpc.ps1"
  rc=$?
  rm -f "$tmp"
  return $rc
}}
    start_rpc_win() {{
      local node="$1"
      local port="${{RPC_PORTS[$node]:-50052}}"
      local log="${{RPC_LOGS[$node]:-rpc-server.log}}"
      echo ">>> starting rpc-server on $node:$port (windows)"
      local devs="${{PEER_DEVICES[$node]:-}}"
      local args=""
      [[ -n "$devs" ]] && args=" --device $devs"
      echo ">>> rpc backends: ${{devs:-default}}  [${{PEER_LABELS[$node]:-}}]"
      # Kill only the rpc-server on THIS port (CIM filter on command line),
      # leaving co-served models' rpc-servers on other ports running.
      ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" \\
        "powershell -NoProfile -Command \\"Get-CimInstance Win32_Process | Where-Object {{ (' ' + \$_.CommandLine + ' ') -like '* -p $port *' -and \\$_.Name -like 'ggml-rpc-server*' }} | ForEach-Object {{ Stop-Process -Id \\$_.ProcessId -Force }}\\"" >/dev/null 2>&1 || true
      sleep 1
      push_win_launcher "$node" || {{ echo "!!! could not copy launcher to $node" >&2; return 1; }}
      ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" \\
        "powershell -NoProfile -ExecutionPolicy Bypass -File \\"%USERPROFILE%\\start-rpc.ps1\\" -Port $port -LogName $log -ExtraArgs \\"$args\\""
    }}
stop_node_rpc_linux() {{
  local node="$1"
  local port="${{RPC_PORTS[$node]:-50052}}"
  # ONLY the rpc daemon on this port: llama-server / llama-swap on this
  # box may belong to co-served models and must survive our exit.
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
    "pkill -u $LINUX_USER -f \\"ggml-[r]pc-server.*-p $port \\"" >/dev/null 2>&1 || true
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
    "rm -f $HOME/.cache/llama-wrapper/rpc.state.$port /tmp/llama-wrapper-rpc.state.$port" >/dev/null 2>&1 || true
  # stale RPC tensor cache (older -c runs): GBs per file, must not starve
  # small peer disks. Safe: purely derived, rebuilt from NAS on demand.
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
    "rm -rf $HOME/.cache/llama.cpp/rpc" >/dev/null 2>&1 || true
}}
stop_node_rpc_win() {{
  local node="$1"
  local port="${{RPC_PORTS[$node]:-50052}}"
  ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" \\
    "powershell -NoProfile -Command \\"Get-CimInstance Win32_Process | Where-Object {{ (' ' + \$_.CommandLine + ' ') -like '* -p $port *' -and \\$_.Name -like 'ggml-rpc-server*' }} | ForEach-Object {{ Stop-Process -Id \\$_.ProcessId -Force }}\\"" >/dev/null 2>&1 || true
  ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" \\
    "powershell -NoProfile -Command \\"Remove-Item \\$env:USERPROFILE/.cache/llama-wrapper/rpc.state.$port, \\$env:TEMP/llama-wrapper-rpc.state.$port -ErrorAction SilentlyContinue\\"" >/dev/null 2>&1 || true
  # stale RPC tensor cache (older -c runs): must not starve peer disks.
  ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" \\
    "powershell -NoProfile -Command \\"Remove-Item \\$env:USERPROFILE/.cache/llama.cpp/rpc -Recurse -Force -ErrorAction SilentlyContinue\\"" >/dev/null 2>&1 || true
}}
# peers this run actually (re)started -- the ONLY ones we may stop.
# A reused peer belongs to whoever started it (possibly another model).
STARTED_NODES=""
note_started() {{
  local node="$1"
  [[ " $STARTED_NODES " == *" $node "* ]] || STARTED_NODES="$STARTED_NODES $node"
}}
stop_started_rpc() {{
  local node
  for node in $STARTED_NODES; do
    if [[ " ${{WIN_NODES[@]}} " == *" $node "* ]]; then
      stop_node_rpc_win "$node"
    else
      stop_node_rpc_linux "$node"
    fi
  done
}}
rpc_state_linux() {{
  local node="$1"
  local port="${{RPC_PORTS[$node]:-50052}}"
  # $HOME first, /tmp fallback (a full home disk must not silently
  # disable reuse -- it would restart the peer on every load)
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
    "cat $HOME/.cache/llama-wrapper/rpc.state.$port 2>/dev/null || cat /tmp/llama-wrapper-rpc.state.$port 2>/dev/null || true" || true
}}
rpc_state_win() {{
  local node="$1"
  local port="${{RPC_PORTS[$node]:-50052}}"
  ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" \\
    "powershell -NoProfile -Command \\"if (Test-Path \\$env:USERPROFILE/.cache/llama-wrapper/rpc.state.$port) {{ Get-Content \\$env:USERPROFILE/.cache/llama-wrapper/rpc.state.$port }} elseif (Test-Path \\$env:TEMP/llama-wrapper-rpc.state.$port) {{ Get-Content \\$env:TEMP/llama-wrapper-rpc.state.$port }}\\"" || true
}}
set_rpc_state_linux() {{
  local node="$1" want="$2"
  local port="${{RPC_PORTS[$node]:-50052}}"
  sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
    "mkdir -p $HOME/.cache/llama-wrapper && echo \\"$want\\" > $HOME/.cache/llama-wrapper/rpc.state.$port || echo \\"$want\\" > /tmp/llama-wrapper-rpc.state.$port" >/dev/null 2>&1 || true
}}
set_rpc_state_win() {{
  local node="$1" want="$2"
  local port="${{RPC_PORTS[$node]:-50052}}"
  ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" \\
    "powershell -NoProfile -Command \\"New-Item -ItemType Directory -Force \\$env:USERPROFILE/.cache/llama-wrapper -ErrorAction SilentlyContinue | Out-Null; Set-Content \\$env:USERPROFILE/.cache/llama-wrapper/rpc.state.$port -Value \\"$want\\" -ErrorAction SilentlyContinue; if (-not (Test-Path \\$env:USERPROFILE/.cache/llama-wrapper/rpc.state.$port)) {{ Set-Content \\$env:TEMP/llama-wrapper-rpc.state.$port -Value \\"$want\\" }}\\"" >/dev/null 2>&1 || true
}}
port_open() {{
  local node="$1" port="${{2:-}}"
  [[ -z "$port" ]] && port="${{RPC_PORTS[$node]:-50052}}"
  if [[ -n "${{RPC_CM[$node]:-}}" ]]; then
    # native-CM peer: no TCP listener exists — alive == process running
    sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
      "pgrep -f \\"ggml-[r]pc-server.*-p $port \\" >/dev/null" 2>/dev/null
    return $?
  fi
  # GNU timeout (coreutils) missing on macOS — bash /dev/tcp probe is portable
  timeout 2 bash -c "echo > /dev/tcp/$(node_ip "$node")/$port" 2>/dev/null || bash -c "echo > /dev/tcp/$(node_ip "$node")/$port" 2>/dev/null
}}
rpc_log_for_port() {{
  # mirror _rpc_log_name: the base port keeps the legacy bare log name
  if [[ "$1" == "50052" ]]; then echo "rpc-server.log"; else echo "rpc-server-$1.log"; fi
}}
tcp_probe() {{
  # raw TCP open probe (no pgrep shortcut, no CM special-case):
  # is ANYTHING at all listening on node:port?
  local node="$1" port="$2"
  timeout 2 bash -c "echo > /dev/tcp/$(node_ip "$node")/$port" 2>/dev/null || bash -c "echo > /dev/tcp/$(node_ip "$node")/$port" 2>/dev/null
}}
win_port_excluded() {{
  # Windows Hyper-V/WSL excluded ranges where a fresh bind fails with
  # "Failed to create server socket" (50052 is grandfathered, stays legal)
  local p="$1"
  [[ "$p" == "50052" ]] && return 1
  (( p >= 50000 && p <= 50059 )) && return 0
  (( p >= 50060 && p <= 50159 )) && return 0
  (( p >= 50160 && p <= 50259 )) && return 0
  (( p >= 50376 && p <= 50475 )) && return 0
  (( p >= 49698 && p <= 49797 )) && return 0
  (( p >= 49798 && p <= 49897 )) && return 0
  (( p >= 49898 && p <= 49997 )) && return 0
  return 1
}}
roll_rpc_port() {{
  # A foreign process holds this peer's RPC port (our rpc-server could
  # not bind there): roll forward to the next free port and update the
  # live maps, so the peer list, waits, state files and cleanup below
  # all stay consistent on the ports actually serving.
  local node="$1"
  local port="${{RPC_PORTS[$node]:-50052}}"
  local old_port="$port"
  local is_win=0
  [[ " ${{WIN_NODES[@]}} " == *" $node "* ]] && is_win=1
  local tries=0
  while (( tries < 25 )); do
    port=$((port + 1))
    if (( is_win )) && win_port_excluded "$port"; then continue; fi
    if ! tcp_probe "$node" "$port"; then
      echo ">>> $node: RPC port $old_port blocked — rolling to $port"
      RPC_PORTS["$node"]="$port"
      RPC_LOGS["$node"]="$(rpc_log_for_port "$port")"
      return 0
    fi
    tries=$((tries + 1))
  done
  echo "!!! $node: no free RPC port found" >&2
  return 1
}}
ensure_node() {{
  local node="$1"
  if [[ " ${{WIN_NODES[@]}} " == *" $node "* ]]; then
    ensure_rpc_win "$node"
  else
    ensure_rpc_linux "$node"
  fi
}}
ensure_rpc_linux() {{
  # Reuse a running rpc-server on this port whose backends match; restart
  # only on mismatch. Per-port: co-served models on other ports are never
  # touched, so they never need to "agree" on backends anymore.
  local node="$1" want have
  want="${{PEER_DEVICES[$node]:-}}"
  [[ -z "$want" ]] && want="default"
  if port_open "$node"; then
    have="$(rpc_state_linux "$node")"
    if [[ "$have" == "$want" ]]; then
      echo ">>> reusing rpc-server on $node:${{RPC_PORTS[$node]}} (linux) [$want]"
      return 0
    fi
    echo ">>> rpc-server on $node:${{RPC_PORTS[$node]}} backends mismatch (have '${{have:-none}}' want '$want'), restarting"
  fi
  stop_node_rpc_linux "$node"
  sleep 1
  start_rpc_linux "$node" || return 1
  set_rpc_state_linux "$node" "$want"
  note_started "$node"
}}
ensure_rpc_win() {{
  local node="$1" want have
  want="${{PEER_DEVICES[$node]:-}}"
  [[ -z "$want" ]] && want="default"
  if port_open "$node"; then
    have="$(rpc_state_win "$node")"
    if [[ "$have" == "$want" ]]; then
      echo ">>> reusing rpc-server on $node:${{RPC_PORTS[$node]}} (windows) [$want]"
      return 0
    fi
    echo ">>> rpc-server on $node:${{RPC_PORTS[$node]}} backends mismatch (have '${{have:-none}}' want '$want'), restarting"
  fi
  stop_node_rpc_win "$node"
  sleep 1
  start_rpc_win "$node" || return 1
  set_rpc_state_win "$node" "$want"
  note_started "$node"
}}
wait_for_rpc() {{
  local node="$1"
  local rolls=0
  while :; do
  local port="${{RPC_PORTS[$node]:-50052}}"
  local log="${{RPC_LOGS[$node]:-rpc-server.log}}"
  if [[ -n "${{RPC_CM[$node]:-}}" ]]; then
    # native-CM peer: wait for the process + its "listening" line in the log
    for i in $(seq 1 60); do
      if sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" \\
        "pgrep -f \\"ggml-[r]pc-server.*-p $port \\" >/dev/null && grep -a \\"listening on\\" \\"$HOME/$log\\" 2>/dev/null | grep -q ." 2>/dev/null; then
        echo ">>> $node listening on $port (native RDMA)"
        return 0
      fi
      sleep 2
    done
    echo "!!! rpc-server never came up on $node:$port" >&2
    sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" "tail -20 \\"$HOME/$log\\"" >&2 2>/dev/null || true
    return 1
  fi
  for i in $(seq 1 30); do
    if timeout 2 bash -c "echo > /dev/tcp/$(node_ip "$node")/$port" 2>/dev/null || bash -c "echo > /dev/tcp/$(node_ip "$node")/$port" 2>/dev/null; then
      echo ">>> $node listening on $port"
      return 0
    fi
    sleep 1
  done
  # timed out: foreign squatter holding the port, or our server died?
  if tcp_probe "$node" "$port"; then
    echo "!!! $node:$port is held by another process — our rpc-server could not bind there" >&2
    rolls=$((rolls + 1))
    if (( rolls > 10 )); then
      echo "!!! $node: too many blocked ports, giving up" >&2
      return 1
    fi
    roll_rpc_port "$node" || return 1
    ensure_node "$node" || return 1
    continue
  fi
  echo "!!! rpc-server never came up on $node:$port" >&2
  echo "--- last log lines from $node ---" >&2
  if [[ " ${{WIN_NODES[@]}} " == *" $node "* ]]; then
    ssh -n $WIN_SSH_OPTS "$WIN_USER@$(node_ip "$node")" "powershell -NoProfile -Command \\"Get-Content \\$env:USERPROFILE\\$log -Tail 20\\"" >&2 2>/dev/null || true
  else
    sshpass -p "$SSH_PASS" ssh -n $SSH_OPTS "$LINUX_USER@$(node_ip "$node")" "tail -20 \\"$HOME/$log\\"" >&2 2>/dev/null || true
  fi
  return 1
  done
}}

for node in "${{LINUX_NODES[@]}}"; do [[ -n "$node" ]] && ensure_rpc_linux "$node"; done
for node in "${{WIN_NODES[@]}}"; do [[ -n "$node" ]] && {{ ensure_rpc_win "$node" || {{ stop_started_rpc; exit 1; }}; }}; done
for node in "${{ALL_NODES[@]}}"; do
  [[ -n "$node" ]] || continue
  wait_for_rpc "$node" || {{ stop_started_rpc; exit 1; }}
done
trap 'stop_started_rpc' EXIT
RPC_LIST=""
for node in "${{ALL_NODES[@]}}"; do
  [[ -n "$node" ]] || continue
  RPC_LIST="${{RPC_LIST:+$RPC_LIST,}}$node:${{RPC_PORTS[$node]}}"
done
echo ">>> rpc peers:    $RPC_LIST"
echo ">>> device order: $DEVICE_ORDER"
echo ">>> rpc map:      {rpc_legend}"
echo ">>> wrapper-gen:  {model_cfg.get('model_id', 'model')} @{time.strftime('%Y-%m-%dT%H:%M:%S')}"
{_device_env_exports(master)}
{cmd} &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null' TERM INT
wait "$CHILD"
STATUS=$?
wait "$CHILD" 2>/dev/null
exit $STATUS
"""


def _bin_subdir(machine, binary):
    """Bin path for remote commands. ~/ paths become $HOME-relative
    (like the user's scripts); absolute paths stay as-is.
    NOTE: duplicate definition kept in sync — machine override wins,
    else global settings default, else DEFAULT_BIN."""
    bin_dir = _effective_bin_dir(machine)
    if bin_dir.startswith("~/"):
        bin_dir = f'$HOME/{bin_dir[2:]}'
    return f'{bin_dir}/{binary}'

# ---------------- model validation ----------------
CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl",
               "q5_0", "q5_1"}


def validate_model(model_id):
    """Static config validation: catches bad flags and device mappings
    without touching any machine (no side effects). Returns
    {"model_id", "ok", "issues": [{"level": "error"|"warn", "msg"}]}."""
    issues = []

    def err(msg):
        issues.append({"level": "error", "msg": msg})

    def warn(msg):
        issues.append({"level": "warn", "msg": msg})

    model = cfg.get_model(model_id)
    if model is None:
        return {"model_id": model_id, "ok": False,
                "issues": [{"level": "error",
                            "msg": f"model '{model_id}' not found"}]}
    # runtime: custom (external server) — llama.cpp static checks (GGUF
    # presence, flags, device maps) don't apply; liveness is the swap
    # /health probe. A warn-only report keeps the Check button green
    # with context instead of a misleading red.
    if (model.get("runtime") or "") == "custom":
        return {"model_id": model_id, "ok": True,
                "issues": [{"level": "warn",
                            "msg": "custom runtime: served externally — "
                                   "static llama.cpp checks skipped"}]}
    servers = model.get("servers") or []
    if not servers:
        err("no servers configured")
        return {"model_id": model_id, "ok": False, "issues": issues}
    try:
        master, peers = _split_servers(servers)
    except ValueError as e:
        err(str(e))
        return {"model_id": model_id, "ok": False, "issues": issues}
    target_os = master.get("os", "linux")

    # 0. reverse format gate (DEC-07): a model whose downloaded format is
    # outside the ACTIVE engine's downloads vocabulary must not validate.
    # hf (safetensors repo dir) under llama.cpp is a hard error — the
    # llama-server RPC launch cannot serve it; gguf under vLLM only warns
    # (llama.cpp GGUF-on-vLLM stays explicitly experimental, ARD §3.4),
    # so the gate never blocks that combination.
    try:
        import engine_registry
        engine = engine_registry.get_engine()
        fmt = (model.get("format")
               or ("hf" if not str(model.get("base_path") or "")
                   .lower().endswith(".gguf") else "gguf"))
        allowed = engine.ui_manifest().get("downloads", {}) \
            .get("formats", [])
        if fmt == "hf" and "hf" not in allowed:
            err(f"model format 'hf' (safetensors repo) is not servable "
                f"by the {engine.name} engine — llama.cpp serves GGUF "
                f"files; switch the model to a .gguf or the engine to "
                f"vLLM")
        elif fmt == "gguf" and "gguf" not in allowed:
            warn(f"GGUF under the {engine.name} engine is experimental — "
                 f"vLLM's supported format is 'hf' (safetensors)")
    except ValueError:
        raise  # invalid engine setting is a real error, not a gate skip
    except Exception:
        pass  # settings unreadable etc: gate is best-effort, validation continues

    # 1. device mapping: every selected GPU must exist on its machine,
    #    and the mapped --device list must be contiguous RPCs
    raw_order = model.get("device_order") or ""
    policy = model.get("device_order_policy", "as-selected")
    if policy not in ("as-selected", "vram-first"):
        warn(f"unknown device_order_policy '{policy}', using as-selected")
        policy = "as-selected"
    mapped = _device_order_to_llama(raw_order, master, peers, policy)
    if not mapped:
        err("empty --device mapping (no hardware selected?)")
    else:
        rpc_ids = sorted(int(t[3:]) for t in mapped.split(",")
                         if t.startswith("RPC") and t[3:].isdigit())
        if rpc_ids and rpc_ids != list(range(rpc_ids[-1] + 1)):
            err(f"non-contiguous RPC ids in --device '{mapped}' "
                "(a peer in the list will not be started?)")
    for machine in [master] + peers:
        mname = machine.get("name", "")
        have = {d.strip() for d in (machine.get("gpu", "") or "").split(",")
                if d.strip()}
        for tok in raw_order.split(","):
            tok = tok.strip()
            if not tok:
                continue
            mchip, resource = _parse_chip(tok)
            if mchip != mname:
                continue
            if gpu_vendor.is_gpu_token(resource) and resource not in have:
                err(f"{mname}: selects {resource} but machine exposes "
                    f"{{{', '.join(sorted(have)) or 'nothing'}}}")
    if not peers:
        info_mapped = mapped or "(local devices)"
    else:
        info_mapped = mapped

    # 2. flags llama-server actually accepts
    flags = model.get("flags", {}) or {}
    for key in ("cache-type-k", "cache-type-v"):
        val = flags.get(key)
        if val is not None and str(val) not in CACHE_TYPES:
            err(f"--{key} '{val}' invalid: llama-server allows "
                f"{', '.join(sorted(CACHE_TYPES))}")

    # 2c. speculative decoding must match tensors in the file: draft-mtp
    # needs MTP layers either embedded in the model or in an external
    # companion (spec-draft-model). llama.cpp hard-fails without them:
    # "model doesn't contain MTP layers". Header-only scan, no weights.
    def _check_mtp_file(label, raw):
        try:
            resolved = path_resolver.resolve_for(raw, target_os)
        except Exception as e:
            err(f"{label} path does not resolve: {e}")
            return
        local_path = os.path.expanduser(resolved)
        if (local_node.is_local(master) and target_os == "linux"
                and os.path.exists(local_path)):
            try:
                if not gguf_inspect.has_mtp_layers_anywhere(local_path):
                    err(f"{label} has no MTP draft tensors; llama-server "
                        f"will fail with 'failed to create MTP context'")
            except Exception as e:
                warn(f"could not inspect {label} tensors: {e}")
        else:
            warn(f"{label} is not locally reachable, so MTP layers "
                 f"cannot be verified statically (use Tune)")

    if str(flags.get("spec-type", "")).lower() == "draft-mtp":
        draft_raw = (flags.get("spec-draft-model")
                     or flags.get("model-draft") or "")
        if draft_raw:
            _check_mtp_file("spec draft model", draft_raw)
        else:
            _check_mtp_file("model", model.get("base_path")
                            or model.get("path", ""))

    # 3. explicit tensor-split weights must match --device count
    split_mode = model.get("tensor_split_mode", "auto")
    if split_mode == "manual":
        split_mode = "fit"   # legacy manual ratios -> fit
    if split_mode == "split":
        weights = (model.get("tensor_split") or "").strip()
        if not weights:
            err("split mode needs tensor_split weights (e.g. '4,4,4,2,2,2')")
        else:
            n_dev = len((mapped or "").split(",")) if mapped else 0
            n_w = len([w for w in weights.split(",") if w.strip()])
            if n_dev and n_w != n_dev:
                warn(f"tensor_split has {n_w} weights but --device has "
                     f"{n_dev} devices ('{mapped}'); layers will mis-assign")
    # 2b. flags must not collide with backend-derived args (they would
    # be emitted twice: once verbatim, once derived). host/port are
    # already skipped by the command builder, so only these matter.
    derived = {"device", "rpc"}
    if split_mode == "split" and model.get("tensor_split"):
        derived.add("tensor-split")
    for key in flags:
        norm = str(key).lstrip("-")
        if norm in derived:
            warn(f"flag '{key}' collides with a backend-derived argument "
                 f"and will be passed twice; remove it from flags")
    for key, value in flags.items():
        if isinstance(value, str) and value == "":
            warn(f"flag '{key}' has an empty value and will be dropped "
                 f"(an empty value would swallow the next flag)")
    if "n-gpu-layers" in flags and split_mode == "fit":
        warn("n-gpu-layers is user-set: llama.cpp aborts param fitting "
             "('failed to fit params ... abort'). The launcher strips it "
             "from fit-mode commands automatically, but remove it from "
             "flags to avoid the warning")

    # 4. paths resolve (and exist when the master is this machine)
    for label, path in (("model", model.get("base_path") or model.get("path", "")),
                        ("mmproj", flags.get("mmproj", "")),
                        ("draft", flags.get("spec-draft-model")
                         or flags.get("model-draft") or "")):
        if not path:
            if label == "model":
                err("no model base_path set")
            continue
        try:
            resolved = path_resolver.resolve_for(path, target_os)
        except Exception as e:
            err(f"{label} path '{path}' does not resolve: {e}")
            continue
        if local_node.is_local(master) and target_os == "linux":
            import glob as _glob
            import re as _re
            if not _glob.glob(os.path.expanduser(resolved)):
                err(f"{label} file missing on master: {resolved}")
            elif label == "model":
                # split model (-00001-of-00002): every sibling shard must be
                # present or llama-server aborts at load (the 1-of-2 incident)
                m = _re.search(r"-(\d+)-of-(\d+)\.gguf$",
                               os.path.basename(resolved), _re.IGNORECASE)
                if m:
                    width = len(m.group(1))
                    total = int(m.group(2))
                    stem = os.path.basename(resolved)[:m.start()]
                    d = os.path.dirname(os.path.expanduser(resolved))
                    try:
                        have = set(os.listdir(d))
                    except OSError:
                        have = set()
                    missing = [f"{stem}-{i:0{width}d}-of-{total:0{width}d}.gguf"
                               for i in range(1, total + 1)
                               if f"{stem}-{i:0{width}d}-of-{total:0{width}d}.gguf" not in have]
                    if missing:
                        err(f"model is split ({total} shards) but missing on "
                            f"master: {', '.join(missing)} — re-download all "
                            f"shards (HF browse fetches every shard together)")
    fit_targets = ""
    if split_mode == "fit":
        try:
            allow = int(model.get("fit_ram_allow", 4096) or 0)
        except (ValueError, TypeError):
            allow = -1
        if allow < 0:
            err("fit_ram_allow must be >= 0 MiB")
        else:
            fit_targets = _fit_target_list(model, master, peers)
    return {"model_id": model_id, "ok": not any(i["level"] == "error" for i in issues),
            "devices": info_mapped, "fit_targets": fit_targets,
            "issues": issues}


# ---------------- live launch (no scripts needed) ----------------
def _resolve_variant(model, variant_label):
    """Return merged model dict for variant label, or base model if None/empty.
    Does not mutate stored model. Inherit logic mirrors swap_manager."""
    if not variant_label:
        return model
    for var in (model.get("variants") or []):
        if (var.get("label") or "").strip() == variant_label.strip():
            merged = dict(model)
            merged.pop("variants", None)
            for k in ("servers", "group", "device_order", "device_order_policy",
                      "tensor_split_mode", "tensor_split", "fit_margin", "fit_ram_allow",
                      "expose", "pinned", "ttl",
                      "launch_mode", "launch_mode_local_port"):
                if k in var and var[k] is not None:
                    merged[k] = var[k]
            if var.get("llama_bin") is not None and (var.get("llama_bin") or "").strip():
                merged["llama_bin"] = var["llama_bin"]
            if var.get("flags") is not None and isinstance(var["flags"], dict) and var["flags"]:
                base_flags = dict(merged.get("flags") or {})
                base_flags.update(var["flags"])
                merged["flags"] = base_flags
            # vllm (A2): partial override, same semantics as flags — keeps
            # _resolve_variant (live launch) byte-identical with the swap
            # plan's variant merge in swap_manager.
            if var.get("vllm") is not None and isinstance(var["vllm"], dict) and var["vllm"]:
                base_vllm = dict(merged.get("vllm") or {})
                base_vllm.update(var["vllm"])
                merged["vllm"] = base_vllm
            merged["base_path"] = model.get("base_path")
            merged["model_id"] = model.get("model_id")
            merged["name"] = model.get("name") or model.get("model_id")
            merged["_variant_label"] = variant_label.strip()
            return merged
    raise ValueError(f"variant '{variant_label}' not found for model '{model.get('model_id')}'")


def _launch_model_remote_local(model, master, peers, port):
    """launch_model(mode='local') where the launch master is a REMOTE box:
    llama-server runs on that box with ITS OWN GPUs (no RPC peers, no
    network tensor traffic) — the whole point of local mode on aibox3.

    Mechanism: the same SSH detach path _start_llama_server already uses
    for a remote master (env exports + nohup, log to $HOME). The model is
    first REWRITTEN to a single-box deployment of the launch master (its
    expose GPUs, fit mode), so the rendered command has no --rpc/--device
    RPC tokens at all. The model file must be readable on that box (a
    store:// alias resolved for the box's OS — shared CIFS covers it).

    Stopping stays symmetric: serve-group-aware stops clear only this
    model's remote port (stop_llama_server_port); swap-served deployments
    additionally unload via the swap host's unload proxy."""
    # single-box rewrite (same as the is_local branch — shared semantics)
    expose = (model.get("expose") or {}).get(master["name"]) or {}
    raw_gpus = expose.get("gpus") or master.get("gpu", "") or []
    if isinstance(raw_gpus, str):
        raw_gpus = raw_gpus.split(",")
    gpus = [d.strip() for d in raw_gpus if str(d).strip()]
    model = dict(model)
    model["device_order"] = ",".join(gpus)
    # same single-box split policy as render_remote_local_script: keep an
    # explicit variant-provided split, else fit mode
    if not (model.get("tensor_split_mode") == "split"
            and (model.get("tensor_split") or "").strip()):
        model["tensor_split_mode"] = "fit"
        model["tensor_split"] = ""
    from steps import run_steps
    report = {"model_id": model.get("model_id"), "steps": [],
              "mode": "local-remote", "master": master["name"]}
    # Serve-group-safe: clear ONLY this model's remote port — a legacy
    # blanket stop here would kill co-served models on the same box.
    steps = [(f"stop:{master['name']}", stop_llama_server_port,
              master["name"], port),
             (f"serve:{master['name']}", _start_llama_server,
              model, master, [], port)]
    report = run_steps(report, steps, catch=(SSHError, ValueError))
    for st in report.get("steps", []):
        if st.get("step", "").startswith("serve:") and st.get("port"):
            report["port"] = st["port"]
            if st.get("port_rolled"):
                report["port_rolled"] = True
            break
    return report


def launch_model(model_id, servers, port=8080, variant=None, master_name=None,
                 mode="rpc"):
    """Full live orchestration, computed under the hood from model config:
      1. stop llama processes on the master (clears any previous live model)
      2. ensure a ggml-rpc-server on each peer on the model's OWN per-model
         port (single-client, so co-served models get separate ports); a
         compatible running server on that port is reused untouched
      3. launch llama-server on the master as a background process
    Returns a report of each step. This replaces running generated scripts.
    When variant is set, the variant's deployment overrides are merged.

    master_name: force which server runs llama-server (default =
    role:master, else first entry). Lets the Launch page run a model "on
    aibox3" directly.
    mode: "rpc" (default) chains every selected machine via --rpc/--device
    with llama-server on the master. "local" is the single-box path:
    llama-server runs on THIS machine only (peers ignored, no RPC ports,
    no cross-machine tensor traffic) — the launch is refused for a remote
    server since the app has no direct process control over another box."""
    from steps import run_steps
    # variant ids arrive as "base--label" (Launch page cards key on the
    # deployment label, same shape as the llama-swap keys) — split before
    # the config lookup so cfg.get_model finds the base model file.
    base_id, var_label = model_id, None
    if "--" in model_id and variant is None:
        cand_base, cand_label = model_id.split("--", 1)
        if cfg.get_model(cand_base) is not None:
            base_id, var_label = cand_base, cand_label
    model = cfg.get_model(base_id)
    if model is None:
        raise ValueError(f"model '{model_id}' not found")
    if variant or var_label:
        model = _resolve_variant(model, variant or var_label)
    master, peers = _split_servers(servers, master_name)
    if mode == "local" and not local_node.is_local(master):
        # Remote-box local launch: the app drives llama-server over SSH
        # (the same detach/nohup path _start_llama_server already uses for
        # a remote master), then proxies. The model file must be readable
        # ON THAT BOX (store:// alias resolved for the box's OS) — a CIFS
        # mount visible to both covers it; a local-disk model cannot be
        # served this way.
        return _launch_model_remote_local(model, master, peers, port)
    if mode == "local":
        # Single-box: rewrite the deployment to this machine's own
        # accelerators so build_llama_command emits no --rpc and no RPC<N>
        # tokens. The stored expose list (mode vram) names this box's
        # GPUs; host RAM is implicit to llama-server (no device token).
        expose = (model.get("expose") or {}).get(master["name"]) or {}
        raw_gpus = expose.get("gpus") or master.get("gpu", "") or []
        if isinstance(raw_gpus, str):
            raw_gpus = raw_gpus.split(",")
        gpus = [d.strip() for d in raw_gpus if str(d).strip()]
        model = dict(model)
        model["device_order"] = ",".join(gpus)
        # fit mode lets llama.cpp's own fitter distribute layers across
        # this box's GPUs by ACTUAL free memory — better than any static
        # tensor-split computed for a multi-box RPC chain.
        model["tensor_split_mode"] = "fit"
        model["tensor_split"] = ""
        peers = []
    report = {"model_id": model_id, "steps": [], "port": port}
    steps = [(f"stop:{master['name']}", stop_rpc, master["name"])]
    # Per-(model,peer,GROUP) ports — each co-served model gets its own
    # single-client rpc-server per backend GROUP: one server for all
    # selected accelerators, a separate server for the CPU/RAM backend
    # so host-memory work never serializes behind GPU tensors.
    peer_ports = _model_rpc_ports(model, peers)
    # Pass model-specific expose config to each peer's RPC server, plus
    # the explicit backend list per group so each server exposes exactly
    # what the --device RPC mapping counts.
    # ensure=True: never kill a peer's running server on this port when
    # its backends already match -- another instance of the SAME model
    # may be reusing it.
    model_expose = model.get("expose") or {}
    raw_order = model.get("device_order") or _auto_device_order(master, peers)
    rpc_steps = []
    for p in peers:
        for gname, devs in _peer_server_groups(raw_order, p):
            rpc_steps.append((f"rpc:{p['name']}:{gname}", start_rpc,
                              p["name"],
                              peer_ports.get((p["name"], gname)),
                              model_expose.get(p["name"]), devs, True))
    steps += rpc_steps
    steps.append((f"serve:{master['name']}", _start_llama_server,
                  model, master, peers, port))
    report = run_steps(report, steps, catch=(SSHError, ValueError))
    # surface the ACTUAL bound port (it may have rolled past a busy one)
    for st in report.get("steps", []):
        if st.get("step", "").startswith("serve:") and st.get("port"):
            report["port"] = st["port"]
            if st.get("port_rolled"):
                report["port_rolled"] = True
            break
    return report


def _start_llama_server(model, master, peers, port):
    """Launch llama-server on the master machine (local or via SSH) in the
    background, using the computed command from the model config. If the
    requested HTTP port is already taken (leftover server, unrelated app),
    roll forward to the next free port instead of failing the launch."""
    is_local = local_node.is_local(master)
    # swap the ${PORT} placeholder for the concrete port, rolling to a
    # free one when something is already bound there
    port, rolled = _roll_port_if_busy(master, port, is_local)
    cmd = build_llama_command(model, [master["name"]] + [p["name"] for p in peers])
    cmd = cmd.replace('"$PORT"', str(port))
    # machine env (CUDA order, P2P, GGML_SCHED/RPC_DEBUG) — the rendered
    # scripts bake these in; live launches need them too or the server
    # runs undebuggable (no sched logs) and possibly misordered GPUs
    env_pre = _device_env_exports(master)

    if is_local:
        import subprocess
        os_type = master.get("os", "linux")
        log = os.path.join(GENERATED_DIR, "llama-server.log")
        os.makedirs(GENERATED_DIR, exist_ok=True)
        if os_type == "windows":
            # detach: new process group so it survives this app
            proc = subprocess.Popen(
                f"{env_pre}\n{cmd}",
                shell=True,
                stdout=open(log, "w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                             | subprocess.DETACHED_PROCESS,
            )
        else:
            proc = subprocess.Popen(
                ["bash", "-c", f"{env_pre}\n{cmd}"],
                stdout=open(log, "w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid = proc.pid
    else:
        is_win = master.get("os", "linux") == "windows"
        if is_win:
            # Windows master over SSH: cmd.exe is the default shell, so
            # nohup/bash syntax would fail. Detach via PowerShell
            # Start-Process with the command wrapped in cmd /c, env vars
            # set first so the child inherits them.
            remote_log = "%USERPROFILE%\\llama-server.log"
            env_ps = "; ".join(
                f'$env:{k}="{v}"' for k, v in _parse_env_exports(env_pre))
            ps = (
                f'{env_ps}; '
                f'Start-Process -WindowStyle Hidden cmd.exe '
                f'-ArgumentList \'/c {cmd} > "{remote_log}" 2>&1\''
            )
            code, out, err = _ssh_for(master).run(
                f'powershell -NoProfile -Command "{ps}"')
            if code != 0:
                return {"running": False, "stderr": err or out}
            pid = None
        else:
            remote_log = '"$HOME/llama-server.log"'
            # exports must be shell statements (nohup takes a command, not
            # env). Two traps the llama-swap script path (line ~1914) already
            # handles — mirror them EXACTLY here:
            # 1. env lines end with no trailing separator: '{env_sh} nohup'
            #    would glue 'nohup' onto the last export's arguments and the
            #    whole command dies as 'export: ... not a valid identifier'
            #    (seen live on Aibox2 2026-10-03). End env_sh with ';'.
            # 2. build_llama_command joins with " \\\n  " — remote bash only
            #    reads line 1; the rest becomes bare export tokens. Collapse
            #    to one shell line first.
            env_sh = env_pre.replace("\n", "; ").rstrip("; ") + "; "
            cmd_one_line = cmd.replace(" \\\n  ", " ")
            remote = f'{env_sh}nohup {cmd_one_line} > {remote_log} 2>&1 & echo "PID=$!"'
            code, out, err = _ssh_for(master).run(remote)
            if code != 0:
                return {"running": False, "stderr": err or out}
            pid = None
            for line in out.splitlines():
                if line.startswith("PID="):
                    pid = line[4:].strip()

    # wait for the http port to come up
    waited = 0
    up = False
    while waited < 120:
        if _port_open(master, port, is_local):
            up = True
            break
        time.sleep(2)
        waited += 2
    return {"running": up, "pid": pid, "port": port,
            "port_rolled": rolled,
            "command": f"{env_pre}\n{cmd}",
            **({} if up else {"hint": "check llama-server.log"})}


def stop_model(servers, model_id=None, variant=None, master_name=None,
               launch_port=None):
    """Stop a live-launched model. On the master, stop ONLY this model's
    llama-server — port-targeted (stop_llama_server_port over the config
    port plus the rolled-forward candidates _roll_port_if_busy may have
    picked), because the launch page's per-model Stop must never run
    legacy stop_rpc there: no port = stop EVERYTHING, which kills
    llama-swap and every co-served model with it (the stop-all/restart
    buttons' job, not this one). On each peer, stop ONLY this model's
    rpc-server on its per-model port — other co-served models'
    rpc-servers on other ports keep running. When model_id is unknown,
    falls back to legacy stop-all on each server. master_name forces
    which server is treated as the llama-server host (must match the
    launch master — with mode="local" launches the server ran there, not
    on role:master). launch_port: the HTTP port the launch actually
    requested (card's port input) — the anchor for the master-side
    port-targeted kill."""
    out = []
    if model_id:
        try:
            model = cfg.get_model(model_id)
            if variant and model is not None:
                model = _resolve_variant(model, variant)
        except Exception:
            model = None
        if model is not None:
            master, peers = _split_servers(servers, master_name)
            # master: kill only THIS model's llama-server. Config port
            # first (_port_match_re anchors at cmdline end too, so a
            # directly-bound server dies), then the rolled candidates —
            # _roll_port_if_busy binds N+1, N+2... when the config port
            # was busy; a co-served model or stale process holding the
            # base port otherwise leaves VRAM held forever.
            base = int(launch_port) if launch_port else 8080
            for port in (base, base + 1, base + 2):
                out.append(stop_llama_server_port(master["name"], port))
            # peers: stop only this model's group ports on each
            ports = _model_rpc_ports(model, peers)
            for p in peers:
                for (pname, _gname), port in ports.items():
                    if pname == p["name"]:
                        out.append(stop_rpc(p["name"], port))
            return out
    # legacy: no model_id → stop everything on each server
    return [stop_rpc(name) for name in servers]
