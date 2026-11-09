"""Starts/stops ggml-rpc-server processes across machines via the peer agent.

Remote lifecycle uses the agent's typed endpoints (`/rpc/ensure`,
`/rpc/stop`, `/rpc/list`, `/port/open`). Local machines run the same
process logic through `bash`.

Generated scripts preserve the proven behaviors:
  - ggml-rpc-server via the agent with -H/-p/-t flags (-c disk cache
    removed: peers stream from NAS, small VM disks must not fill)
  - per-port start/stop so co-served models never collide
  - agent port-open/list wait with per-port log tail on failure
  - trap-based cleanup on exit
This module NEVER touches files outside its own folder.
"""
import os
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

import config_loader as cfg
import agent_client
import peer_transport
import gpu_vendor
import gguf_inspect
import local_node
import path_resolver
import stats_collector
from runtime_paths import data_dir

GENERATED_DIR = os.path.join(data_dir(), "generated")

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
        p = os.path.join(data_dir(), "config", "settings.yaml")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
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


def _bin_subdir(machine, binary):
    """Bin path for generated scripts. ~/ paths become $HOME-relative
    (like the user's scripts); absolute paths stay as-is. The remote
    shell expands $HOME."""
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


# ---------------- live RPC control over the peer agent ----------------
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


def stop_rpc(machine_name, port=None, use_agent=True):
    """Stop llama processes on a machine, OS-aware, local or via the agent.

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

    # Optional agent: per-port RPC stop, or the explicit stop-all parity
    # endpoint when port is None.
    if use_agent and not is_local and agent_client.should_use_agent(machine):
        if port is None:
            try:
                res = agent_client.stop_all(machine)
                return {
                    "machine": machine_name, "port": None,
                    "killed": bool(res.get("killed")),
                    "detail": [], "remaining": res.get("remaining", {}),
                }
            except Exception as e:
                return {
                    "machine": machine_name, "port": None,
                    "killed": False, "detail": [],
                    "stderr": f"agent error: {e}",
                }
        if port is not None:
            try:
                res = agent_client.rpc_stop(machine, int(port))
                killed = bool(res.get("killed"))
                return {
                    "machine": machine_name, "port": int(port),
                    "killed": killed,
                    "detail": [{"process": f"ggml-rpc-server:-p{port}",
                                "exit_code": 0 if killed else 1,
                                "stdout": "", "stderr": ""}],
                }
            except Exception as e:
                return {
                    "machine": machine_name, "port": int(port),
                    "killed": False,
                    "detail": [{"process": f"ggml-rpc-server:-p{port}",
                                "exit_code": 1, "stdout": "",
                                "stderr": f"agent error: {e}"}],
                }

    if not is_local:
        return {
            "machine": machine_name, "port": port, "killed": False,
            "detail": [], "stderr": "remote stop requires the peer agent",
        }

    def run(cmd, timeout=20):
        return _run_local(cmd, is_win, timeout)

    # ---- per-model port-targeted stop ----
    if port is not None:
        results = []
        if is_win:
            # Pipe-free PowerShell: cmd.exe mangles the pipe form of this
            # command (observed: Stop-Process never ran, but the old code
            # reported success). foreach + explicit count is verifiable.
            ps_kill = (
                "$c=0; foreach ($x in (Get-CimInstance Win32_Process)) { "
                f"if ($x.Name -like 'ggml-rpc-server*' -and "
                f"((' ' + $x.CommandLine + ' ') -like '*-p {port} *')) "
                "{ Stop-Process -Id $x.ProcessId -Force; $c++ } }; $c")
            code, out, err = run('powershell -NoProfile -Command "' + ps_kill + '"')
            results.append((f"ggml-rpc-server:-p{port}", code, out, err))
            # stale RPC tensor cache cleanup (see linux branch)
            run("powershell -NoProfile -Command \"Remove-Item "
                "$env:USERPROFILE/.cache/llama.cpp/rpc -Recurse -Force "
                "-ErrorAction SilentlyContinue\"")
            still = True
            ps_count = (
                "$c=0; foreach ($x in (Get-CimInstance Win32_Process)) { "
                f"if ($x.Name -like 'ggml-rpc-server*' -and "
                f"((' ' + $x.CommandLine + ' ') -like '*-p {port} *')) "
                "{ $c++ } }; $c")
            for _ in range(5):
                time.sleep(0.3)
                _vc, vout, _ = run('powershell -NoProfile -Command "' + ps_count + '"')
                try:
                    count = int((vout or "").strip().splitlines()[-1])
                except (ValueError, IndexError):
                    count = 0
                if count == 0:
                    still = False
                    break
            code = 0 if not still else code
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
        run("systemctl --user stop llama-swap 2>/dev/null; true")
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


def stop_llama_server_port(machine_name, port, use_agent=True):
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

    if use_agent and not is_local and agent_client.should_use_agent(machine):
        try:
            res = agent_client.llama_stop(machine, int(port))
            killed = bool(res.get("killed"))
            return {"machine": machine_name, "port": int(port),
                    "killed": killed,
                    "detail": [{"process": f"llama-server:--port{port}",
                                "exit_code": 0 if killed else 1,
                                "stdout": "", "stderr": ""}]}
        except Exception as e:
            return {"machine": machine_name, "port": int(port),
                    "killed": False,
                    "detail": [{"process": f"llama-server:--port{port}",
                                "exit_code": 1, "stdout": "",
                                "stderr": f"agent error: {e}"}]}

    if not is_local:
        return {"machine": machine_name, "port": int(port), "killed": False,
                "detail": [], "stderr": "remote stop requires the peer agent"}

    def run(cmd, timeout=20):
        return _run_local(cmd, is_win, timeout)

    port = int(port)
    # Windows/POSIX port match at cmdline END or mid-line — the
    # trailing-space-only pattern never matched `... --port N` at cmdline
    # end (the remote-local launcher renders --port last), leaving VRAM
    # held through every cmdStop (observed on aibox3 group swaps).
    if is_win:
        ps_kill = (
            "$c=0; foreach ($x in (Get-CimInstance Win32_Process)) { "
            f"if ($x.Name -like 'llama-server*' -and "
            f"((' ' + $x.CommandLine + ' ') -like '* --port {port} *')) "
            "{ Stop-Process -Id $x.ProcessId -Force; $c++ } }; $c")
        code, out, err = run('powershell -NoProfile -Command "' + ps_kill + '"')
        still = True
        ps_count = (
            "$c=0; foreach ($x in (Get-CimInstance Win32_Process)) { "
            f"if ($x.Name -like 'llama-server*' -and "
            f"((' ' + $x.CommandLine + ' ') -like '* --port {port} *')) "
            "{ $c++ } }; $c")
        for _ in range(5):
            time.sleep(0.3)
            _vc, vout, _ = run('powershell -NoProfile -Command "' + ps_count + '"')
            try:
                count = int((vout or "").strip().splitlines()[-1])
            except (ValueError, IndexError):
                count = 0
            if count == 0:
                still = False
                break
        return {"machine": machine_name, "port": port, "killed": not still,
                "detail": [{"process": f"llama-server:--port{port}",
                            "exit_code": 0 if not still else 1,
                            "stdout": out, "stderr": err}]}
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
    return ""


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
    # Optional agent: the agent owns the start/reuse/stop decision and
    # writes the same per-port state file as the generated scripts.
    if not is_local and agent_client.should_use_agent(machine):
        devices = [] if want in ("", "default") else [
            d.strip() for d in want.split(",") if d.strip()]
        try:
            if not ensure:
                # explicit start (machine Start button) keeps the legacy
                # restart semantics; ensure=True (live launch) is allowed
                # to reuse a matching server.
                try:
                    agent_client.rpc_stop(machine, port)
                except Exception:
                    pass
            res = agent_client.rpc_ensure(
                machine, port, devices, _agent_env_dict(machine))
            return {
                "machine": machine_name, "ip": machine["ip"], "port": port,
                "running": bool(res.get("running")),
                "reused": bool(res.get("reused")),
                "exit_code": 0 if res.get("running") else 1,
                "stderr": res.get("error") or "",
                "log_tail": res.get("log_tail") or "",
            }
        except Exception as e:
            return {
                "machine": machine_name, "ip": machine["ip"], "port": port,
                "running": False, "reused": False, "exit_code": 1,
                "stderr": f"agent error: {e}", "log_tail": "",
            }
    if not is_local:
        return {
            "machine": machine_name, "ip": machine["ip"], "port": port,
            "running": False, "reused": False, "exit_code": 1,
            "stderr": "remote start requires the peer agent", "log_tail": "",
        }
    if ensure and _port_open(machine, port, is_local):
        have = _rpc_state_read(machine, is_local, port)
        if have == want:
            return {
                "machine": machine_name, "ip": machine["ip"], "port": port,
                "running": True, "reused": True, "exit_code": 0,
                "stderr": "", "log_tail": "",
            }
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
        # setsid detaches (no controlling tty -> survives parent exit);
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
        import subprocess
        proc = subprocess.run(
            ["bash", "-c", remote], capture_output=True, text=True, timeout=15)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    else:
        # faithful port of push_win_launcher() + start_rpc_win()
        launcher = _win_launcher(machine, port, expose, device_list)
        import subprocess
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "start-rpc.ps1")
        with open(path, "w", encoding="utf-8") as f:
            f.write(launcher)
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
            capture_output=True, text=True, timeout=30)
        code, out, err = proc.returncode, proc.stdout, proc.stderr

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
        if agent_client.should_use_agent(machine):
            try:
                return any(int(p.get("port") or 0) == int(port)
                           for p in agent_client.rpc_list(machine))
            except Exception:
                return False
    host = "127.0.0.1" if is_local else machine["ip"]
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _tail_log(machine, is_win, is_local, port=50052, use_agent=True):
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
    if use_agent and not is_local:
        try:
            return peer_transport.for_machine(machine).logs_tail(
                machine, _rpc_log_name(port), 20).get("text", "")
        except Exception:
            return ""
    return ""


def _tail_all_rpc_logs(machine, is_win, is_local, use_agent=True):
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
    if use_agent and not is_local and agent_client.should_use_agent(machine):
        try:
            names = [n for n in agent_client.logs_list(machine)
                     if n.startswith("rpc-server")]
            chunks = []
            for name in names:
                text = agent_client.logs_tail(machine, name, 20).get("text", "")
                if text:
                    chunks.append(text)
            return "\n".join(chunks)
        except Exception:
            return ""
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
    # sanity: the new IP must actually answer on the agent port
    try:
        with socket.create_connection(
                (current_ip, agent_client.agent_port(machine)), timeout=3):
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


def has_live_llama(machine_name, use_agent=True):
    """True if a native llama-server is running on this machine (any port).

    The Launch page's live launch runs llama-server directly — it opens no
    RPC port, so an RPC-port probe alone reports the machine idle even
    while a model is serving. Remote peers are asked through the agent.
    """
    machine = cfg.machines.require(machine_name)
    is_local = local_node.is_local(machine)
    if not is_local and use_agent and agent_client.should_use_agent(machine):
        try:
            return bool(agent_client.status(machine).get("llama_running"))
        except Exception:
            return False
    if is_local:
        import subprocess as _sp
        if os.name == "nt":
            try:
                p = _sp.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"],
                            capture_output=True, text=True, timeout=8)
                return "llama-server.exe" in (p.stdout or "").lower()
            except (OSError, _sp.SubprocessError):
                return False
        try:
            if machine.get("os") == "macos":
                p = _sp.run(["ps", "-axc", "-o", "comm"], capture_output=True,
                            text=True, timeout=8)
                return "llama-server" in (p.stdout or "")
            p = _sp.run(["pgrep", "-f", "llama-[s]erver"], capture_output=True,
                        timeout=8)
            return p.returncode == 0
        except (OSError, _sp.SubprocessError):
            return False
    return False


def machine_status(machine_name, use_agent=True):
    """Per-machine status. rpc_running is True if ANY ggml-rpc-server is
    running on the box (not just the base port), since per-model ports
    mean the base port may be unused while other models serve.
    On unreachability, tries MAC-based IP healing before reporting down."""
    machine = cfg.machines.require(machine_name)
    is_local = local_node.is_local(machine)
    if use_agent and not is_local and agent_client.should_use_agent(machine):
        try:
            st = agent_client.status(machine)
            return {
                "name": machine_name, "ip": machine["ip"],
                "reachable": True,
                "rpc_running": bool(st.get("rpc_running")
                                    or st.get("rpc")),
                "gpus": st.get("gpus") or [],
                "is_local": False,
                "is_master": machine.get("role") == "master",
                "agent": True,
                "processes": st,
            }
        except Exception as e:
            # DHCP drift: the agent may now answer at a different IP. MAC
            # is the stable identity — reconcile from UDP discovery + ARP,
            # then retry once before reporting the machine down.
            try:
                import discovery as _disc
                _disc.reconcile_machine_ips()
            except Exception:
                pass
            try:
                machine = _sync_machine_ip(cfg.machines.require(machine_name))
            except Exception:
                pass
            try:
                st = agent_client.status(machine)
                return {
                    "name": machine_name, "ip": machine["ip"],
                    "reachable": True,
                    "rpc_running": bool(st.get("rpc_running")
                                        or st.get("rpc")),
                    "gpus": st.get("gpus") or [],
                    "is_local": False,
                    "is_master": machine.get("role") == "master",
                    "agent": True,
                    "processes": st,
                }
            except Exception as e2:
                return {
                    "name": machine_name, "ip": machine["ip"],
                    "reachable": False, "rpc_running": False,
                    "gpus": [], "is_local": False,
                    "is_master": machine.get("role") == "master",
                    "error": f"agent error: {e2}",
                }
    if not is_local:
        return {
            "name": machine_name, "ip": machine["ip"], "reachable": False,
            "rpc_running": False, "gpus": [], "is_local": False,
            "is_master": machine.get("role") == "master",
            "error": "peer agent required",
        }
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
    rpc_running = _any_rpc_running_local(machine)
    gpus = _local_gpu_info(machine)
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


def _parse_etime(text):
    """'MM:SS' / 'HH:MM:SS' / 'DD-HH:MM:SS' -> seconds (BSD/macOS ps)."""
    text = (text or "").strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    try:
        parts = [int(x) for x in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, sec = 0, 0, parts[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec


def list_rpc_servers(machine_name, use_agent=True):
    """Running ggml-rpc-server processes on a machine.

    Returns [{pid, port, age_s, cmd}]. Used by the orphan reaper; ports are
    parsed from the process command line (-p N). Best-effort: a failed OS
    query returns [] (reaper then does nothing for that box)."""
    machine = cfg.machines.require(machine_name)
    is_win = machine.get("os", "linux") == "windows"
    is_local = local_node.is_local(machine)
    os_type = "windows" if is_win else machine.get("os", "linux")
    if use_agent and not is_local and agent_client.should_use_agent(machine):
        try:
            return agent_client.rpc_list(machine)
        except Exception as e:
            print(f"[agent] {machine_name}: rpc/list failed: {e}")
            return []
    if not is_local:
        return []
    out = ""
    if is_win:
        ps = (
            "Get-CimInstance Win32_Process | Where-Object { "
            "$_.Name -eq 'ggml-rpc-server.exe' } | "
            "ForEach-Object { [pscustomobject]@{ Pid=$_.ProcessId; "
            "Age=[int]((Get-Date)-$_.CreationDate).TotalSeconds; "
            "Cmd=$_.CommandLine } } | ConvertTo-Json -Compress")
        cmd = 'powershell -NoProfile -Command "' + ps + '"'
        try:
            code, out, err = _run_local(cmd, True, 20)
        except Exception:
            return []
        if not (out or "").strip():
            return []
        try:
            import json
            data = json.loads(out)
        except Exception:
            return []
        if isinstance(data, dict):
            data = [data]
        servers = []
        for e in data if isinstance(data, list) else []:
            try:
                c = e.get("Cmd") or ""
                m = re.search(r"(?:^|\s)-p\s+(\d+)", c)
                if not m:
                    continue
                servers.append({
                    "pid": int(e.get("Pid") or 0),
                    "port": int(m.group(1)),
                    "age_s": int(e.get("Age")) if e.get("Age") is not None else None,
                    "cmd": c,
                })
            except Exception:
                continue
        return servers
    if os_type == "macos":
        cmd = "ps -axo pid=,etime=,command= 2>/dev/null | grep '[g]gml-rpc-server' || true"
    else:
        cmd = "ps -eo pid=,etimes=,args= 2>/dev/null | grep '[g]gml-rpc-server' || true"
    try:
        code, out, err = _run_local(cmd, False, 20)
    except Exception:
        return []
    servers = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if os_type == "macos":
            age_s = _parse_etime(parts[1])
        else:
            try:
                age_s = int(parts[1])
            except ValueError:
                age_s = None
        cmdline = parts[2]
        m = re.search(r"(?:^|\s)-p\s+(\d+)", cmdline)
        if not m:
            continue
        servers.append({"pid": pid, "port": int(m.group(1)),
                        "age_s": age_s, "cmd": cmdline})
    return servers


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
    # Bash does not expand `~` inside quotes; emit `$HOME/...` so the
    # quoted model path resolves on POSIX masters (observed: literal
    # `~/GGUF_models/...` caused "No such file or directory").
    if model_path.startswith("~/") and target_os != "windows":
        model_path = "$HOME/" + model_path[2:]
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
    # Normalize legacy/short GPU-layer keys: `ngl` must emit `-ngl`, not
    # the invalid long form `--ngl` (observed: "invalid argument: --ngl").
    flag_aliases = {
        "ngl": "-ngl",
        "n_gpu_layers": "-ngl",
        "n-gpu-layers": "-ngl",
    }
    for key, value in flags.items():
        if key in skip_keys:
            continue
        key = flag_aliases.get(key.lstrip("-"), key)
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
        if device_order_clean:
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
    # Newer llama.cpp builds label the Apple Metal device MTL0 (older
    # builds used plain "Metal"). Normalize the auto order so a macOS
    # master never emits a device token the backend rejects.
    devices = [
        "MTL0" if d.strip() == "Metal" else d.strip()
        for d in master.get("gpu", "").split(",") if d.strip()
    ]
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
        # 0700: launcher scripts are owner-only even
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
        _vv = gpu_vendor.normalize_vendor((machine or {}).get("gpu_vendor"))
    except Exception:
        _vv = ""
    # CUDA_* vars are NVIDIA-only. Unknown/empty vendor defaults to NVIDIA
    # (same as gpu_prefix); AMD/Intel/Apple/CPU never get them.
    if _vv not in ("amd", "intel", "apple", "cpu"):
        try:
            _vis = gpu_vendor.cuda_visible_devices(machine)
        except Exception:
            _vis = ""
        if _vis:
            pre += f"CUDA_VISIBLE_DEVICES={_vis} "
        else:
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


def _cmd_to_args(cmd, os_type="linux"):
    """Split a build_llama_command() string into argv (binary dropped).

    The agent enforces --host/--port itself, so those are stripped here.
    Windows tokens keep their quoting during shlex(posix=False); strip the
    outer quotes before handing the list to subprocess.Popen.
    """
    import shlex
    text = (cmd or "").replace(" \\\n  ", " ")
    tokens = shlex.split(text, posix=(os_type != "windows"))
    if os_type == "windows":
        tokens = [t[1:-1] if (len(t) >= 2 and t[0] == '"' and t[-1] == '"')
                  else t for t in tokens]
    if tokens:
        tokens = tokens[1:]  # drop the llama-server binary
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--host", "--port"):
            i += 2
            continue
        if tok.startswith(("--host=", "--port=")):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _agent_base_url(machine):
    ip = (machine.get("ip") or "").strip()
    return f"http://{ip}:{agent_client.agent_port(machine)}"


def _agent_token_var(machine_name):
    return "AGENT_TOKEN_" + "".join(
        c if c.isalnum() else "_" for c in str(machine_name or "").upper())


def _agent_env_dict(machine):
    """Parse _rpc_env_exports() space-separated assignments into a dict."""
    import shlex
    out = {}
    for tok in shlex.split(_rpc_env_exports(machine or {})):
        if "=" in tok:
            key, value = tok.split("=", 1)
            if key:
                out[key] = value
    return out


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
    if v in ("apple", "cpu"):
        # Metal and CPU need no CUDA/HIP/SYCL device env — just the debug
        # flags (+ RDMA when enabled) that every backend uses.
        return sched + rdma
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
    model_cfg = dict(model_cfg)
    gpus = [d.strip() for d in (master.get("gpu", "") or "").split(",")
            if d.strip()]
    if (master.get("os") == "macos" and len(gpus) == 1
            and gpus[0] in ("Metal", "MTL0")):
        model_cfg["tensor_split_mode"] = "auto"
        model_cfg["tensor_split"] = ""
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


def _remote_local_model(model_cfg, master):
    """Normalize a deployment for single-box native execution on `master`."""
    model_cfg = dict(model_cfg)
    expose = (model_cfg.get("expose") or {}).get(master["name"]) or {}
    raw_gpus = expose.get("gpus") or master.get("gpu", "") or []
    if isinstance(raw_gpus, str):
        raw_gpus = raw_gpus.split(",")
    gpus = [d.strip() for d in raw_gpus if str(d).strip()]
    orig_order = str(model_cfg.get("device_order") or "")
    model_cfg["device_order"] = ",".join(gpus)
    # CPU-only local box: fit mode has no VRAM devices to target and emits an
    # empty --fit-target, which llama-server rejects. Keep the RAM device and
    # use auto mode.
    if not gpus:
        ram = bool(expose.get("ram")) or orig_order.endswith("-RAM")
        model_cfg["device_order"] = (f"{master['name']}-RAM" if ram else "")
        model_cfg["tensor_split_mode"] = "auto"
        model_cfg["tensor_split"] = ""
        # No accelerators: never offload layers. Vulkan/llvmpipe may be
        # auto-detected on a CPU-only box and then hang during upload.
        flags = dict(model_cfg.get("flags") or {})
        for k in list(flags):
            if k.lstrip("-").replace("_", "-") in (
                    "n-gpu-layers", "ngl"):
                flags[k] = 0
        flags["n-gpu-layers"] = 0
        model_cfg["flags"] = flags
        return model_cfg
    # keep an explicit variant split; otherwise fit mode — except a single
    # Metal device, where fit mode is broken and auto mode is required.
    is_metal_single = (master.get("os") == "macos"
                       and len(gpus) == 1
                       and gpus[0] in ("Metal", "MTL0"))
    if is_metal_single:
        model_cfg["tensor_split_mode"] = "auto"
        model_cfg["tensor_split"] = ""
    elif not (model_cfg.get("tensor_split_mode") == "split"
              and (model_cfg.get("tensor_split") or "").strip()):
        model_cfg["tensor_split_mode"] = "fit"
        model_cfg["tensor_split"] = ""
    return model_cfg


def remote_local_agent_payload(model_cfg, master, remote_port, state_key=None):
    """Structured /llama/start payload for a native single-box deployment."""
    normalized = _remote_local_model(model_cfg, master)
    cmd = build_llama_command(normalized, [master["name"]])
    cmd = cmd.replace(" \
  ", " ").replace('"$PORT"', str(remote_port))
    args = _cmd_to_args(cmd, master.get("os", "linux"))
    env = dict(_parse_env_exports(_device_env_exports(master)))
    payload = {"port": int(remote_port), "args": args, "env": env,
               "state_key": state_key}
    # Send the exact binary the app probed. The agent otherwise re-runs
    # discovery, which can prefer a stale legacy build/bin over the path
    # the master's readiness/update checks use.
    lb = (master.get("llama_bin") or "").strip().rstrip("/\\")
    if lb:
        name = ("llama-server.exe" if master.get("os") == "windows"
                else "llama-server")
        if not lb.endswith(name):
            lb = f"{lb}/{name}"
        payload["llama_bin"] = lb
    return payload




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
    # ip, group, port, devices, log, label. The agent target is the IP part.
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
    # Per-server data, keyed by node TAG (IP, or IP#cpu for a peer's
    # separate CPU/RAM server). The SAME device lists the --device RPC
    # mapping counts, so numbering can never drift from reality.
    #
    # Bash 3.2 compatibility: macOS /bin/bash is 3.2 and has NO `declare
    # -A`/associative arrays. Emit case-lookup functions instead, so a
    # macOS master can run the same agent-only script. Values are emitted
    # into double-quoted `echo` lines; escape backslashes/quotes, but keep
    # `${AGENT_TOKEN_*}` intact so it expands at runtime from the sidecar.
    import json as _json
    import shlex as _shlex

    def _case_val(value):
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def _case_fn(name, pairs, default=""):
        lines = [f'{name}() {{ case "$1" in']
        for tag, val in pairs:
            lines.append(f'        "{_case_val(tag)}") '
                         f'echo "{_case_val(val)}" ;;')
        lines.append(f'        *) echo "{_case_val(default)}" ;;')
        lines.append("    esac; }")
        return "\n".join(lines)

    agent_nodes = [n for n in nodes if agent_client.should_use_agent(n["machine"])]
    agent_nodes_decl = " ".join(f'"{n["tag"]}"' for n in agent_nodes)
    peer_dev_pairs = [(n["tag"], ",".join(n["devs"])) for n in nodes]
    peer_port_pairs = [(n["tag"], str(n["port"])) for n in nodes]
    label_pairs = [(n["tag"], n["label"]) for n in nodes]
    agent_base_pairs = [
        (n["tag"], _agent_base_url(n["machine"])) for n in agent_nodes]
    agent_token_pairs = [
        (n["tag"], "${" + _agent_token_var(n["machine"].get("name")) + ":-}")
        for n in agent_nodes]
    agent_payload_pairs = []
    for n in agent_nodes:
        body = _json.dumps({
            "devices": list(n["devs"]),
            "env": _agent_env_dict(n["machine"]),
        }, separators=(",", ":"))
        payload = '{"port":__PORT__,' + body[1:]
        # _case_fn emits `echo "<escaped>"`; pass the raw JSON. Pre-quoting
        # with shlex made the single quotes part of the HTTP body (400).
        agent_payload_pairs.append((n["tag"], payload))
    peer_devices_decl = _case_fn("peer_devices", peer_dev_pairs)
    rpc_ports_decl = _case_fn("rpc_port_of", peer_port_pairs, default="50052")
    rpc_logs_decl = (
        "rpc_log_for_port() {\n"
        '  if [[ "$1" == "50052" ]]; then echo "rpc-server.log"; '
        'else echo "rpc-server-$1.log"; fi\n'
        "}\n"
        'rpc_log_of() { rpc_log_for_port "$(rpc_port_of "$1")"; }'
    )
    peer_labels_decl = _case_fn("peer_label_of", label_pairs)
    agent_bases_decl = _case_fn("agent_base_of", agent_base_pairs)
    agent_tokens_decl = _case_fn("agent_token_of", agent_token_pairs)
    agent_payloads_decl = _case_fn("agent_payload_of", agent_payload_pairs)
    rpc_legend = _rpc_legend(raw_order, master, peers)
    # the .ps1 executes ON Windows peers: its exe path must come from a
    # Windows machine's bin dir. Using master (usually Linux) baked a Linux
    # path under USERPROFILE and every win launch failed its own check.
    # Port is passed at runtime via -Port, so one template serves all peers.
    ps1_machine = next((p for p in peers if p.get("os", "linux") == "windows"), master)
    ps1_text = _win_launcher(ps1_machine, 50052, expose_args="",
                             parametrized=True)

    # Agent-only launch script: one ggml-rpc-server per (peer, group) is
    # started through the peer agent's typed /rpc/ensure, then llama-server
    # runs in the foreground. No passwords are ever embedded.
    return f"""#!/bin/bash
set -uo pipefail
PORT="${{1:-8080}}"
DEVICE_ORDER="{device_order}"
{peer_devices_decl}
{rpc_ports_decl}
{rpc_logs_decl}
{peer_labels_decl}
{agent_bases_decl}
{agent_tokens_decl}
{agent_payloads_decl}
AGENT_NODES=({agent_nodes_decl})
STARTED=()
note_started() {{ STARTED+=("$1"); }}
agent_ensure() {{
  local node="$1"
  local port; port="$(rpc_port_of "$node")"
  local payload; payload="$(agent_payload_of "$node")"
  payload="${{payload//__PORT__/$port}}"
  local base; base="$(agent_base_of "$node")"
  local token; token="$(agent_token_of "$node")"
  local devs; devs="$(peer_devices "$node")"
  local out
  echo ">>> starting rpc-server on $node:$port [${{devs:-default}}] [$(peer_label_of "$node")]"
  out="$(curl -fsS --max-time 90 -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" --data-binary "$payload" \
    "$base/rpc/ensure")" || return 1
  echo "$out"
  [[ "$out" == *'"reused":true'* ]] || note_started "$node"
}}
agent_stop() {{
  local node="$1"
  local port; port="$(rpc_port_of "$node")"
  local base; base="$(agent_base_of "$node")"
  local token; token="$(agent_token_of "$node")"
  curl -fsS --max-time 60 -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" --data-binary '{{"port":'"$port"'}}' \
    "$base/rpc/stop" >/dev/null 2>&1 || true
}}
agent_port_open() {{
  local node="$1"
  local port; port="$(rpc_port_of "$node")"
  local base; base="$(agent_base_of "$node")"
  local token; token="$(agent_token_of "$node")"
  curl -fsS --max-time 5 -H "Authorization: Bearer $token" \
    "$base/port/open?port=$port" 2>/dev/null | grep -q '"open":true'
}}
agent_rpc_listed() {{
  local node="$1"
  local port; port="$(rpc_port_of "$node")"
  local base; base="$(agent_base_of "$node")"
  local token; token="$(agent_token_of "$node")"
  curl -fsS --max-time 8 -H "Authorization: Bearer $token" \
    "$base/rpc/list" 2>/dev/null | grep -q "port.:$port"
}}
stop_started_rpc() {{
  local node
  for node in "${{STARTED[@]:-}}"; do
    [[ -n "$node" ]] && agent_stop "$node"
  done
  STARTED=()
}}
trap stop_started_rpc EXIT
for node in "${{AGENT_NODES[@]}}"; do
  [[ -n "$node" ]] || continue
  agent_ensure "$node" || {{ echo "!!! rpc-server failed on $node" >&2; stop_started_rpc; exit 1; }}
done
for node in "${{AGENT_NODES[@]}}"; do
  [[ -n "$node" ]] || continue
  ok=0
  for i in $(seq 1 60); do
    if agent_port_open "$node" || agent_rpc_listed "$node"; then ok=1; break; fi
    sleep 2
  done
  if [[ "$ok" != "1" ]]; then
    echo "!!! rpc-server never came up on $node:$(rpc_port_of "$node") (log: $(rpc_log_of "$node"))" >&2
    stop_started_rpc
    exit 1
  fi
  echo ">>> $node listening on $(rpc_port_of "$node") (agent)"
done
RPC_LIST=""
for node in "${{AGENT_NODES[@]}}"; do
  [[ -n "$node" ]] || continue
  RPC_LIST="${{RPC_LIST:+$RPC_LIST,}}${{node%%#*}}:$(rpc_port_of "$node")"
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
                "issues": [{"level": "info",
                            "msg": "custom runtime: served externally — "
                                   "static llama.cpp checks skipped"}]}
    # MLX is a separate runtime (mlx_lm.server), not a ggml backend: it has
    # no --rpc/--device and never uses peers. Running the llama.cpp static
    # checks on an HF repo id would produce misleading missing-file errors.
    if (model.get("runtime") or "") == "mlx":
        return {"model_id": model_id, "ok": True,
                "issues": [{"level": "info",
                            "msg": "MLX runtime: single-Mac serving — "
                                   "llama.cpp/RPC checks skipped"}]}
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
    # llama-server RPC launch cannot serve it.
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
                f"files; switch the model to a .gguf")
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
    # A RAM-only / CPU-only box legitimately has no --device, and an empty
    # device_order means "auto at launch". Only a non-empty order that
    # selects accelerator chips but maps to nothing is a real error.
    selected_gpu = any(
        gpu_vendor.is_gpu_token(_parse_chip(t.strip())[1])
        for t in raw_order.split(",") if t.strip())
    if not mapped and selected_gpu:
        err("device_order selects accelerator chips but none map to "
            "--device (check machine names / exposed GPUs)")
    elif mapped:
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
        # Empty weights = llama.cpp's default placement (valid, used by
        # working fleet configs). A shorter weight list is also accepted
        # upstream, so don't flag either shape as an error.
        pass
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

    Mechanism: _start_llama_server sends the peer agent a typed
    `/llama/start` (args + env + state key). The model is first REWRITTEN
    to a single-box deployment of the launch master (its expose GPUs, fit
    mode), so the rendered command has no --rpc/--device RPC tokens at
    all. The model file must be readable on that box (a store:// alias
    resolved for the box's OS — shared CIFS covers it).

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
    # explicit variant-provided split, else fit mode — except a single Metal
    # device, where fit mode is broken and auto mode is required.
    is_metal_single = (master.get("os") == "macos"
                       and len(gpus) == 1
                       and gpus[0] in ("Metal", "MTL0"))
    if is_metal_single:
        model["tensor_split_mode"] = "auto"
        model["tensor_split"] = ""
    elif not (model.get("tensor_split_mode") == "split"
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
    report = run_steps(report, steps, catch=(RuntimeError, ValueError))
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
        # Remote-box local launch: the app starts llama-server on that box
        # through the peer agent's `/llama/start`, then proxies. The model
        # file must be readable ON THAT BOX (store:// alias resolved for
        # the box's OS) — a CIFS mount visible to both covers it; a
        # local-disk model cannot be served this way.
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
        # BUT: fit mode breaks on Metal (single device MTL0) — skip fit
        # when there's only one Metal device, use auto mode with -ngl instead.
        is_metal_single = len(gpus) == 1 and gpus[0] in ("Metal", "MTL0")
        if not is_metal_single:
            model["tensor_split_mode"] = "fit"
            model["tensor_split"] = ""
        else:
            # For Metal single device, ensure auto mode (no --fit)
            model["tensor_split_mode"] = "auto"
            model["tensor_split"] = ""
        peers = []
    report = {"model_id": model_id, "steps": [], "port": port,
              "master": master["name"]}
    # Clear only THIS model's previous llama-server on the requested live
    # port. Never call stop_rpc() with no port here: that is the legacy
    # stop-everything path and kills/disables llama-swap (which then frees
    # :8080 for llama-server and breaks serving/reboot).
    steps = [(f"stop:{master['name']}", stop_llama_server_port,
              master["name"], port)]
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
    report = run_steps(report, steps, catch=(RuntimeError, ValueError))
    # surface the ACTUAL bound port (it may have rolled past a busy one)
    for st in report.get("steps", []):
        if st.get("step", "").startswith("serve:") and st.get("port"):
            report["port"] = st["port"]
            if st.get("port_rolled"):
                report["port_rolled"] = True
            break
    return report


def _start_llama_server(model, master, peers, port):
    """Launch llama-server on the master machine (local or via the peer
    agent) in the background, using the computed command from the model
    config. If the requested HTTP port is already taken (leftover server,
    unrelated app), roll forward to the next free port instead of failing
    the launch."""
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

    # Optional agent: native llama-server start on the peer. Only for
    # non-MLX (the agent launches llama-server, not mlx_lm.server).
    if (not is_local and (model.get("runtime") or "llama") != "mlx"
            and agent_client.should_use_agent(master)):
        try:
            args = _cmd_to_args(cmd, master.get("os", "linux"))
            env = dict(_parse_env_exports(env_pre))
            res = agent_client.llama_start(master, port, args, env)
            if not (res.get("running") or res.get("loading")
                    or res.get("reused")):
                return {"running": False, "port": port,
                        "port_rolled": rolled,
                        "stderr": res.get("error") or "agent llama/start failed"}
            else:
                waited = 0
                up = bool(res.get("running"))
                while not up and waited < 120:
                    time.sleep(2)
                    waited += 2
                    up = _port_open(master, port, is_local=False)
                return {
                    "running": up, "pid": res.get("pid"), "port": port,
                    "port_rolled": rolled, "agent": True,
                    "command": f"{env_pre}\n{cmd}",
                    **({} if up else {"hint": res.get("log_tail")
                                      or "check the agent's llama-server log"}),
                }
        except Exception as e:
            return {"running": False, "port": port,
                    "port_rolled": rolled,
                    "stderr": f"agent error: {e}"}

    if not is_local:
        return {"running": False, "port": port, "port_rolled": rolled,
                "stderr": "remote launch requires the peer agent"}

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
