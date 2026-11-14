"""Machine CRUD, discovery/hardware sync, groups, and serve groups."""
import math
from concurrent.futures import ThreadPoolExecutor

import pydantic
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

import access
import agent_client
import config_loader as cfg
import engine_registry
import gpu_vendor
import local_node
import rpc_manager
import stats_collector
import swap_manager
from machine_conn import _machine_connection_caps
from machine_tokens import _agent_guard, _auto_agent_token, _ensure_agent_ready
from secret_fields import strip_secret_keys
from swap_apply import apply_swap_config
from web_shared import _raw_is_private

router = APIRouter()

_CONN_CACHE = {"ts": 0.0, "data": None}
_CONN_TTL = 5.0


def _board_model_local():
    """Board model from the device tree (Raspberry Pi, other SBCs)."""
    for path in ("/proc/device-tree/model",
                 "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as f:
                return f.read().replace(b"\x00", b"").decode(
                    "utf-8", "replace").strip()
        except OSError:
            continue
    return ""


def _ram_gb(value):
    """Whole-GB RAM, but keep sub-GB boards nonzero (Pi Zero 2 W)."""
    return stats_collector.ram_gb(value)


def _detected_vendor(stats, machine):
    """GPU vendor from a stats payload, correcting a stale/default form value.

    A fresh Raspberry Pi (or any GPU-less Linux box) often starts with the
    form default `nvidia`; if stats report no GPUs the machine is CPU-only,
    and leaving `nvidia` would make the installer pick a CUDA asset that
    cannot run there."""
    gpus = (stats or {}).get("gpus") or []
    detected = gpu_vendor.normalize_vendor((stats or {}).get("gpu_vendor"))
    if detected:
        return detected
    if not gpus:
        return "cpu"
    names = " ".join(str(g.get("name") or "") for g in gpus).lower()
    if any(k in names for k in ("nvidia", "geforce", "rtx", "tesla",
                                "quadro")):
        return "nvidia"
    if any(k in names for k in ("amd", "radeon", "instinct", "firepro")):
        return "amd"
    if "apple" in names:
        return "apple"
    if "intel" in names or "arc" in names:
        return "intel"
    return gpu_vendor.normalize_vendor(machine.get("gpu_vendor")) or "cpu"


def _detect_arch_board(machine):
    """(arch, board) for hardware detection.

    `detected` is replaced wholesale by the detect endpoints, so arch/board
    must be recomputed every time — otherwise a Raspberry Pi would lose its
    arm64 arch and later asset picks would silently fall back to x64.
    """
    import llamacpp_installer
    arch = llamacpp_installer.resolve_arch(machine)
    board = ""
    try:
        if local_node.is_local(machine):
            board = _board_model_local()
        else:
            board = str((agent_client.info(machine) or {}).get("board") or "")
    except Exception:
        board = ""
    return arch, board


# ---------------- machines ----------------
class ExposeIn(pydantic.BaseModel):
    mode: str = "both"      # both | vram | ram
    gpus: list[str] = []    # empty = all GPUs


class MachineIn(pydantic.BaseModel):
    name: str
    ip: str
    username: str = ""
    password: str = ""
    port: int = 22
    mac: str | None = None      # None = preserve/auto-fill; identity for ARP healing
    transport: str = "agent"    # agent-only (kept for config compatibility)
    agent_port: int = 8198
    agent_token: str = ""       # write-only; stored in the vault, never YAML
    role: str = "peer"
    os: str = "linux"
    gpu_vendor: str = "nvidia"   # nvidia | amd | cpu
    rpc_port: int = 50052
    gpu: str = ""
    ram_total_gb: float | None = None
    llama_bin: str = ""
    # llama.cpp update strategy: auto (detect) | prebuilt | source.
    # None = "not supplied" so a machine edit never resets the choice.
    engine_mode: str | None = None
    # explicit prebuilt backend pin (auto|cpu|cuda|rocm|sycl|vulkan|metal)
    engine_backend: str | None = None
    expose: ExposeIn = ExposeIn()
    # RDMA transport for llama.cpp RPC on this machine. When False (default)
    # the rpc-server is started with GGML_RPC_NO_RDMA=1 so a wedged RoCE QP
    # can never stall tensor traffic. When True, rdma_ip must be set — it is
    # the RoCE NIC address that both the RPC client connects to AND the GID
    # matcher uses (the IP must exist in the NIC's GID table for
    # auto-negotiation to engage).
    rdma: bool = False
    rdma_ip: str = ""
    # optional explicit pinning (discovered by machine detect); empty = let
    # llama.cpp auto-match the RoCE v2 GID from the socket's local address
    rdma_dev: str = ""
    rdma_gid: str = ""
    # rdma_cm: use rdma_cm QP path resolution (ConnectX-3/mlx4). Unset = upstream
    # raw ibv_modify_qp path. TCP always available via rdma=False.
    rdma_cm: bool = False
    # DEC-03 (AUDIT-T6 §1.6): per-machine swap-config overrides — empty =
    # inherit the global per-engine path (swap.yaml config_path), non-empty
    # = pin THIS machine's llama-swap config file.
    swap_config_path: str = ""

    @pydantic.field_validator("engine_mode")
    @classmethod
    def _engine_mode_ok(cls, v):
        if v is None:
            return None
        v = str(v).strip().lower()
        if v not in ("auto", "prebuilt", "source"):
            raise ValueError("engine_mode must be auto, prebuilt, or source")
        return v

    @pydantic.field_validator("engine_backend")
    @classmethod
    def _engine_backend_ok(cls, v):
        if v is None:
            return None
        v = str(v).strip().lower()
        if v not in ("auto", "cpu", "cuda", "rocm", "sycl",
                     "vulkan", "metal"):
            raise ValueError("engine_backend must be auto, cpu, cuda, "
                             "rocm, sycl, vulkan, or metal")
        return v


class EngineModeIn(pydantic.BaseModel):
    mode: str = "auto"


@router.post("/api/machines/{name}/engine-mode")
async def set_engine_mode(name: str, p: EngineModeIn):
    mode = (p.mode or "auto").strip().lower()
    if mode not in ("auto", "prebuilt", "source"):
        raise HTTPException(400, "mode must be auto, prebuilt, or source")
    if cfg.machines.get(name) is None:
        raise HTTPException(404, f"machine '{name}' not found")
    cfg.machines.update(name, {"engine_mode": mode})
    return {"ok": True, "machine": name, "engine_mode": mode}


class EngineBackendIn(pydantic.BaseModel):
    backend: str = "auto"


@router.post("/api/machines/{name}/engine-backend")
async def set_engine_backend(name: str, p: EngineBackendIn):
    backend = (p.backend or "auto").strip().lower()
    if backend not in ("auto", "cpu", "cuda", "rocm", "sycl",
                       "vulkan", "metal"):
        raise HTTPException(400, "backend must be auto, cpu, cuda, rocm, "
                                 "sycl, vulkan, or metal")
    if cfg.machines.get(name) is None:
        raise HTTPException(404, f"machine '{name}' not found")
    cfg.machines.update(name, {"engine_backend": backend})
    return {"ok": True, "machine": name, "engine_backend": backend}


@router.get("/api/machines")
async def list_machines():
    """Return the machine list. Records are copied so neither the flags nor
    resolved secrets leak into the machine store/YAML."""
    # never leak resolved passwords to the browser — in-memory only
    out = []
    for m in cfg.machines.all():
        # Copy and strip every secret-named field as a response-boundary
        # guard; the store read path already does this, but an endpoint must
        # never echo a plaintext secret even from a hand-edited config.
        m = strip_secret_keys(dict(m))
        try:
            import secrets_vault as _vault
            m["password_set"] = bool(_vault.machine_password_set(m))
            m["agent_token_set"] = bool(_vault.machine_agent_token_set(m))
        except Exception:
            m["password_set"] = False
            m["agent_token_set"] = False
        out.append(m)
    return out


@router.get("/api/machines/connections")
async def machine_connections():
    import time as _time
    now = _time.time()
    if _CONN_CACHE["data"] is not None and now - _CONN_CACHE["ts"] < _CONN_TTL:
        return _CONN_CACHE["data"]
    machines = cfg.machines.all()
    def one(m):
        return m.get("name"), _machine_connection_caps(m)
    with ThreadPoolExecutor(max_workers=min(32, len(machines) or 1)) as pool:
        result = dict(pool.map(one, machines))
    _CONN_CACHE.update(ts=_time.time(), data=result)
    return result


class AgentCheckIn(pydantic.BaseModel):
    ip: str
    agent_port: int = 8198
    agent_token: str = ""
    name: str = ""


@router.post("/api/machines/agent-check")
async def machine_agent_check(p: AgentCheckIn):
    """Pre-save peer-agent reachability check for the add/edit form."""
    try:
        port = int(p.agent_port or 8198)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="agent port must be an integer")
    if not 1 <= port <= 65535:
        raise HTTPException(status_code=400, detail="agent port must be in 1..65535")
    existing = cfg.machines.get(p.name) if p.name else None
    machine = {"name": p.name or p.ip, "ip": p.ip, "agent_port": port,
               "transport": "agent",
               "id": (existing or {}).get("id")}
    _caps, err = await run_in_threadpool(
        _ensure_agent_ready, machine, p.agent_token or "")
    if err:
        raise HTTPException(status_code=502,
                            detail=f"peer agent auth failed: {err}")
    out = await run_in_threadpool(
        agent_client.health, machine, 4.0, p.agent_token or None)
    return {"ok": True, **out}


# DEC-03 (AUDIT-T6 §1.6): validated per-machine swap-config overrides —
# runs on the machine save path (add/update) so a bad value never lands
# in machines.yaml.
def _validate_machine_swap_config(payload):
    cfg.machine_swap_config_override(payload, cfg.ENGINE_LLAMA_CPP)


def _validate_machine_connection(payload):
    """Agent-only machines: normalize the agent port and clear legacy
    credential transport fields so they never reach machines.yaml."""
    try:
        agent_port = int(payload.get("agent_port") or 8198)
    except (TypeError, ValueError) as e:
        raise ValueError("agent_port must be an integer") from e
    if not 1 <= agent_port <= 65535:
        raise ValueError("agent_port must be in 1..65535")
    payload["transport"] = "agent"
    payload["agent_port"] = agent_port
    payload["password"] = ""

@router.post("/api/machines")
async def add_machine(m: MachineIn):
    try:
        payload = m.model_dump()
        payload["engine_mode"] = payload.get("engine_mode") or "auto"
        payload["engine_backend"] = payload.get("engine_backend") or "auto"
        requested_token = (payload.pop("agent_token", "") or "").strip()
        import uuid
        payload.setdefault("id", uuid.uuid4().hex[:8])
        _validate_machine_swap_config(payload)
        _validate_machine_connection(payload)
        if not local_node.is_local(payload):
            import secrets_vault as _vault
            try:
                tok = _auto_agent_token(payload, requested_token)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            if tok:
                _vault.set_machine_agent_token(payload["id"], tok)
                payload["_agent_token"] = tok
            if not payload.get("mac"):
                # Store the MAC so DHCP re-leases self-heal via ARP later.
                try:
                    payload["mac"] = str(
                        (agent_client.info(payload) or {}).get("mac") or "")
                except Exception:
                    payload["mac"] = ""
        payload["mac"] = payload.get("mac") or ""
        detail = _agent_guard(payload)
        if detail:
            raise HTTPException(status_code=400, detail=detail)
        saved = cfg.machines.add(payload)
        saved = dict(saved)
        saved["agent_token_set"] = bool(
            __import__("secrets_vault").machine_agent_token_set(saved))
        saved["password_set"] = bool(
            __import__("secrets_vault").machine_password_set(saved))
        saved.pop("password", None)
        await apply_swap_config(f"machine add {saved.get('name')}")
        return saved
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/api/machines/{name}")
async def update_machine(name: str, m: MachineIn):
    try:
        payload = m.model_dump()
        if payload.get("engine_mode") is None:
            payload.pop("engine_mode", None)
        if payload.get("engine_backend") is None:
            payload.pop("engine_backend", None)
        if payload.get("mac") is None:
            payload.pop("mac", None)   # omitted = keep the stored identity
        old_name = str(name)
        new_name = str(payload.get("name") or old_name).strip()
        if new_name and new_name != old_name and cfg.machines.get(new_name):
            raise ValueError(f"machine '{new_name}' already exists")
        agent_token = payload.pop("agent_token", "")
        if agent_token:
            import secrets_vault as _vault
            target = cfg.machines.get(name) or {}
            _vault.set_machine_agent_token(target.get("id") or name,
                                           agent_token)
        _validate_machine_swap_config(payload)
        _validate_machine_connection(payload)
        detail = _agent_guard(payload)
        if detail:
            raise HTTPException(status_code=400, detail=detail)
        saved = cfg.machines.update(name, payload)
        if new_name and new_name != old_name:
            # Machine names are the reference key in model configs/groups/
            # layout; cascade the rename so deployments don't orphan.
            try:
                cfg.rename_machine_references(old_name, new_name)
            except Exception as exc:
                print(f"[machines] rename references failed: {exc}")
        saved = dict(saved)
        saved["password_set"] = bool(
            __import__("secrets_vault").machine_password_set(saved))
        saved.pop("password", None)
        saved["agent_token_set"] = bool(agent_token) or bool(
            __import__("secrets_vault").machine_agent_token_set(saved))
        await apply_swap_config(f"machine update {new_name or old_name}")
        return saved
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/api/machines/{name}")
async def delete_machine(name: str):
    try:
        import secrets_vault as _vault
        target = cfg.machines.get(name) or {}
        # Remove every app-side secret keyed to this machine (id, name, and
        # the requested name) so a fresh re-add starts clean instead of
        # inheriting a stale token.
        for ident in {target.get("id"), target.get("name"), name}:
            if not ident:
                continue
            _vault.set_machine_agent_token(ident, "")
            _vault.delete_env_value(_vault.machine_env_key(ident))
    except Exception:
        pass
    remaining = cfg.machines.delete(name)
    await apply_swap_config(f"machine delete {name}")
    return {"remaining": remaining}


@router.post("/api/machines/{name}/detect")
async def detect_hardware(name: str):
    """Discover GPUs/cores/RAM on a machine and persist into its config,
    so expose defaults to everything available with no manual entry."""
    machine = cfg.machines.require(name)
    # A saved machine may pre-date its agent's token: pair/verify first so
    # the stats call itself cannot fail on an unpaired fresh agent.
    if not local_node.is_local(machine):
        detail = _agent_guard(machine)
        if detail:
            raise HTTPException(status_code=503, detail=detail)
    stats = await run_in_threadpool(stats_collector.machine_stats, machine)
    if stats["status"] != "online":
        detail = _agent_guard(machine)
        raise HTTPException(
            status_code=503,
            detail=detail or f"{name} is offline (agent port open but the "
                             f"agent failed to answer)")
    vendor = _detected_vendor(stats, machine)
    m2 = dict(machine, gpu_vendor=vendor)
    gpus = [gpu_vendor.gpu_token(m2, g["index"]) for g in stats["gpus"]]
    detected = dict(machine.get("detected") or {})
    detected.update({
        "gpu_names": [g["name"] for g in stats["gpus"]],
        "compute_cap": [g.get("compute_cap") for g in stats["gpus"]],
        "gpu_uuids": [g.get("uuid") for g in stats["gpus"]],
        "vram_total_gb": round(sum(g["mem_total_gb"] for g in stats["gpus"]), 1),
        "cores": stats["cores"],
        "ram_total_gb": _ram_gb(stats["ram_total_gb"]),
    })
    arch, board = _detect_arch_board(machine)
    if arch:
        detected["arch"] = arch
    if board:
        detected["board"] = board
    updates = {
        "gpu": ",".join(gpus),
        "gpu_vendor": vendor,
        "ram_total_gb": _ram_gb(stats["ram_total_gb"]) if stats["ram_total_gb"] else None,
        "detected": detected,
    }
    # RDMA auto-detect: probe for an ACTIVE RoCE NIC even when the flag is
    # currently off. If one is found, auto-enable RDMA and persist the
    # device/GID/IP (user can turn it back off in the machine editor if
    # they prefer TCP). Detection is Linux-only; Windows/macOS return {}.
    is_local = False
    try:
        is_local = local_node.is_local(machine)
    except Exception:
        pass
    rdma_info = {}
    rdma_info = await run_in_threadpool(_detect_rdma_nic, machine, is_local)
    if rdma_info:
        updates["rdma"] = True
        updates["rdma_dev"] = rdma_info.get("dev", "")
        updates["rdma_gid"] = rdma_info.get("gid", "")
        if rdma_info.get("ip"):
            updates["rdma_ip"] = rdma_info["ip"]
    updates = {k: v for k, v in updates.items() if v is not None}
    result = cfg.machines.update(name, updates)
    result["synced_models"] = cfg.sync_models_to_hardware()
    if rdma_info:
        result["rdma_detected"] = rdma_info
    await apply_swap_config(f"hardware detect {name}")
    return result


def _detect_rdma_nic(machine, is_local):
    """Discover the machine's RoCE NIC: active ibverbs device, its active
    port, the RoCE v2 GID index on that port that maps to a local IP, and
    that IP. Returns {} when no RDMA device is found.
    ONE bundled remote command (single agent round trip) — a shell script
    scans all devices/ports/GIDs locally and emits "dev|port|gid|ip"."""
    remote_script = (
        'for dev in $(ibv_devices 2>/dev/null | awk "NR>2{print \\$1}"); do '
        'for port in 1 2; do '
        'st=$(cat /sys/class/infiniband/$dev/ports/$port/state 2>/dev/null); '
        '[[ "$st" != *ACTIVE* ]] && continue; '
        'gids=/sys/class/infiniband/$dev/ports/$port/gids; '
        'for gf in $(ls $gids 2>/dev/null); do '
        'gid=$(cat $gids/$gf 2>/dev/null); '
        'gt=$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/types/$gf 2>/dev/null); '
        'nd=$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/ndevs/$gf 2>/dev/null); '
        'norm=$(echo "$gid" | tr -d :); '
        '[[ ${#norm} -ne 32 ]] && continue; [[ "$gt" != *v2* ]] && continue; '
        '[[ "${norm:20:4}" != ffff ]] && continue; '
        'ip=$((16#${norm:24:2})).$((16#${norm:26:2})).$((16#${norm:28:2})).$((16#${norm:30:2})); '
        'echo "$dev|$port|$gf|$ip|$nd"; '
        'done; done; done'
    )
    if is_local:
        import subprocess
        try:
            out = subprocess.run(["bash", "-c", remote_script], capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            # no bash on Windows (RDMA tooling is Linux-only anyway)
            return {}
    else:
        try:
            return agent_client.rdma(machine) or {}
        except Exception:
            return {}
    for line in (out or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        dev, port, gid, ip = parts[0], parts[1], parts[2], parts[3]
        if ip.startswith("10.0.0.") or (len(parts) > 4 and parts[4]):
            return {"dev": dev, "port": port, "gid": gid, "ip": ip}
    return {}


class ProbeIn(pydantic.BaseModel):
    ip: str
    os: str = "linux"
    gpu_vendor: str = "nvidia"
    rpc_port: int = 50052
    agent_port: int = 8198
    agent_token: str = ""
    name: str = ""

@router.post("/api/machines/probe")
async def probe_hardware(p: ProbeIn):
    """Detect hardware from raw form inputs -- works for machines that
    aren't saved yet. Returns gpu list + RAM + detected block; persists
    nothing."""
    existing = cfg.machines.get(p.name) if p.name else None
    machine = {"name": p.name or p.ip, "ip": p.ip, "os": p.os,
               "gpu_vendor": p.gpu_vendor, "rpc_port": p.rpc_port,
               "agent_port": p.agent_port, "transport": "agent",
               "id": (existing or {}).get("id")}
    if local_node.is_local(machine):
        machine["gpu_vendor"] = ""  # detect the real local vendor
    if p.agent_token:
        machine["_agent_token"] = p.agent_token
    detail = _agent_guard(machine)
    if detail:
        raise HTTPException(status_code=503, detail=detail)

    def run():
        if not local_node.is_local(machine):
            _caps, err = _ensure_agent_ready(machine)
            if err:
                return {"online": False, "error": err}
        s = stats_collector.machine_stats(machine)
        if s["status"] != "online":
            return {"online": False}
        vendor = _detected_vendor(s, machine)
        machine["gpu_vendor"] = vendor
        detected = dict((existing or {}).get("detected") or {})
        detected.update({
            "gpu_names": [g["name"] for g in s["gpus"]],
            "compute_cap": [g.get("compute_cap") for g in s["gpus"]],
            "gpu_uuids": [g.get("uuid") for g in s["gpus"]],
            "vram_total_gb": round(sum(g["mem_total_gb"] for g in s["gpus"]), 1),
            "cores": s["cores"],
            "ram_total_gb": _ram_gb(s["ram_total_gb"]),
        })
        arch, board = _detect_arch_board(machine)
        if arch:
            detected["arch"] = arch
        if board:
            detected["board"] = board
        return {
            "online": True,
            "gpu_vendor": vendor,
            "gpu": ",".join(gpu_vendor.gpu_token(machine, g["index"]) for g in s["gpus"]),
            "ram_total_gb": float(_ram_gb(s["ram_total_gb"])) if s["ram_total_gb"] else None,
            "cores": s["cores"],
            "detected": detected,
        }
    result = await run_in_threadpool(run)
    if not result["online"]:
        raise HTTPException(
            status_code=503,
            detail=result.get("error") or (
                f"{p.ip} is unreachable over the peer agent "
                f"(port {p.agent_port}) — install/start the agent there "
                f"and check the token"))
    result["detected"] = result.get("detected") or {}
    return result


class DiscoverIn(pydantic.BaseModel):
    include_known: bool = False  # hide already-added machines by default
    subnet: str = ""             # accepted for UI compatibility; ignored
    port: int = 0
    username: str = ""
    password: str = ""
    ping: bool = False
    agent_only: bool = True


@router.get("/api/discover/subnets")
async def discover_subnets():
    """Local /24 networks of this machine — for the discovery dropdown."""
    import discovery
    return {"subnets": discovery.local_subnets(),
            "default": discovery.local_subnet() or ""}


@router.post("/api/discover")
async def discover_hosts(p: DiscoverIn):
    """Scan the LAN for peer agents (UDP discovery). No shell scan: peers
    are agent-installed machines, identified by MAC/hostname."""
    import discovery

    def run():
        discovery.pair_agents(include_known=True)
        return discovery.discover_agents(include_known=p.include_known)

    result = await run_in_threadpool(run)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/api/machines/detect-all")
async def detect_all():
    """Detect hardware on every online machine in parallel."""
    machines = cfg.machines.all()

    def detect(m):
        try:
            stats = stats_collector.machine_stats(m)
        except Exception:
            return None
        if stats["status"] != "online":
            return None
        vendor = _detected_vendor(stats, m)
        m2 = dict(m, gpu_vendor=vendor)
        gpus = [gpu_vendor.gpu_token(m2, g["index"]) for g in stats["gpus"]]
        detected = dict(m.get("detected") or {})
        detected.update({
            "gpu_names": [g["name"] for g in stats["gpus"]],
            "compute_cap": [g.get("compute_cap") for g in stats["gpus"]],
            "gpu_uuids": [g.get("uuid") for g in stats["gpus"]],
            "vram_total_gb": round(sum(g["mem_total_gb"] for g in stats["gpus"]), 1),
            "cores": stats["cores"],
            "ram_total_gb": _ram_gb(stats["ram_total_gb"]),
        })
        arch, board = _detect_arch_board(m2)
        if arch:
            detected["arch"] = arch
        if board:
            detected["board"] = board
        upd = {
            "gpu_vendor": vendor,
            "gpu": ",".join(gpus),
            "detected": detected,
        }
        if stats["ram_total_gb"]:
            upd["ram_total_gb"] = _ram_gb(stats["ram_total_gb"])
        return m["name"], upd

    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        results = list(pool.map(detect, machines))
    updated = []
    for r in results:
        if r:
            cfg.machines.update(r[0], r[1])
            updated.append(r[0])
    synced = cfg.sync_models_to_hardware() if updated else []
    if updated:
        await apply_swap_config("hardware detect-all")
    return {"updated": updated, "synced_models": synced}


# ---------------- groups ----------------
class GroupIn(pydantic.BaseModel):
    name: str
    description: str = ""
    machines: list[str] = []


class SwapGroupIn(pydantic.BaseModel):
    name: str
    description: str = ""
    members: list[str] = []
    swap: bool = False
    exclusive: bool = False
    persistent: bool = False


def _active_engine_name():
    """DEC-17: the engine whose group world the API serves. Resolved once
    per request from the registry (settings.yaml `engine`); ValueError
    from a bad settings value propagates as a 500 — the settings PUT path
    already rejects bad engines, so it never happens in practice."""
    return engine_registry.get_engine().name


def _group_has_private_member(group):
    """access.py: a group whose membership includes a `private: true`
    model disappears for restricted viewers — one of its members reads
    as non-existent (`mid--variant` labels still reference their base
    id), so the whole group reads as never existing. Trusted IPs see
    every group."""
    if access.gate_open():
        return False
    for mem in (group.get("members") or []):
        if _raw_is_private(str(mem).split("--")[0]):
            return True
    return False


def _swap_group_write_guard(data, new_group, existing_name=None):
    """Serve-group writes from a restricted viewer: members must ALL be
    visible to them (no smuggling a foreign model into a group), and
    update/delete must target a group they can already see. Trusted
    users pass untouched. 404-shaped on every refusal (never a hint)."""
    if access.gate_open():
        return
    allow = access.view_allow()
    if not new_group:
        # visibility of the TARGET group under the same filter the
        # listing applies — invisible group = never existed
        for grp in _engine_groups(cfg.swap_groups):
            if grp.get("name") == existing_name:
                break
        else:
            raise HTTPException(404, "not found")
    if data is None:
        return
    members = data.get("members") or []
    for mem in members:
        token = str(mem)
        base_mid = token.split("--")[0]
        if _raw_is_private(base_mid):
            raise HTTPException(404, "not found")
        # EXACT token match — bare mid grants the base only; a variant
        # member requires its own mid--label grant
        if allow is not None and token not in allow:
            raise HTTPException(404, "not found")


def _engine_groups(collection):
    """DEC-17: only the ACTIVE engine's group entries. Missing `engine`
    (legacy YAML) reads as llama_cpp — same default the collection
    stamps on write, so un-migrated files keep rendering under
    llama_cpp. access.py: groups holding a private member additionally
    vanish in a restricted view."""
    engine = _active_engine_name()
    out = []
    for g in [g for g in collection.all()
              if (g.get("engine") or "llama_cpp") == engine]:
        if _group_has_private_member(g):
            continue
        out.append(g)
    return out


def _stamp_group_engine(item: dict) -> dict:
    """Migration (DEC-17): stamp the ACTIVE engine on a group being
    created/updated. New groups are born engine-qualified; an update
    re-stamps only when the stored engine is missing."""
    item.setdefault("engine", _active_engine_name())
    return item


def _reject_cross_engine(collection, name: str):
    """DEC-17 safety net (unreachable from the UI — GET only lists the
    active engine's groups): mutating a group that belongs to the OTHER
    engine fails loudly instead of silently editing it. Raises
    HTTPException 409."""
    g = collection.get(name)
    if g is not None and (g.get("engine") or "llama_cpp") != _active_engine_name():
        raise HTTPException(
            status_code=409,
            detail=f"group '{name}' belongs to the "
                   f"{g.get('engine')} engine (active: "
                   f"{_active_engine_name()})")


@router.get("/api/groups")
async def list_groups():
    return _engine_groups(cfg.groups)


@router.post("/api/groups")
async def add_group(g: GroupIn):
    # DEC-17: new machine groups are stamped with the active engine.
    data = _stamp_group_engine(g.model_dump())
    try:
        saved = cfg.groups.add(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        import config_advice
        warnings = config_advice.machine_group_warnings(saved)
    except Exception:
        warnings = []
    await apply_swap_config(f"group add {saved.get('name')}")
    return {**saved, "warnings": warnings}


@router.put("/api/groups/{name}")
async def update_group(name: str, g: GroupIn):
    _reject_cross_engine(cfg.groups, name)  # DEC-17 safety net
    # re-stamp only fills a MISSING engine (legacy entry) — an update
    # never moves a group between engines.
    data = _stamp_group_engine(g.model_dump())
    try:
        saved = cfg.groups.update(name, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        import config_advice
        warnings = config_advice.machine_group_warnings(saved)
    except Exception:
        warnings = []
    await apply_swap_config(f"group update {saved.get('name')}")
    return {**saved, "warnings": warnings}


@router.delete("/api/groups/{name}")
async def delete_group(name: str):
    _reject_cross_engine(cfg.groups, name)  # DEC-17 safety net
    remaining = cfg.groups.delete(name)
    await apply_swap_config(f"group delete {name}")
    return {"remaining": remaining}


# ---------------- serve groups (llama-swap model co-serving) ----------------
@router.get("/api/swap-groups")
async def list_swap_groups():
    groups = _engine_groups(cfg.swap_groups)
    # access.py: restricted viewers see ONLY their own members — each
    # group's membership is filtered through the same gate the model
    # list uses (private flag + view allowlist), and groups that end up
    # with no visible members vanish entirely (a foreign serve group
    # reads as never existing, matching the model-table semantics).
    if access.gate_open():
        return groups
    allow = access.view_allow()
    visible = []
    for g in groups:
        members = []
        for mem in (g.get("members") or []):
            token = str(mem)
            base_mid = token.split("--")[0]
            if _raw_is_private(base_mid):
                continue
            # EXACT token match (see _swap_group_write_guard note)
            if allow is not None and token not in allow:
                continue
            members.append(mem)
        if not members:
            continue
        visible.append({**g, "members": members})
    return visible


@router.post("/api/swap-groups")
async def add_swap_group(g: SwapGroupIn):
    # DEC-17: new serve groups are stamped with the active engine.
    data = _stamp_group_engine(g.model_dump())
    _swap_group_write_guard(data, new_group=True)
    try:
        saved = cfg.swap_groups.add(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Serve-group truth: a deployment in 2+ swap:true groups only takes
    # effect in the first — warn loudly at save time instead of letting
    # the generator silently drop the loser (one group per model).
    warnings = swap_manager.duplicate_swap_claims()
    try:
        import config_advice
        warnings += config_advice.serve_group_warnings(saved)
    except Exception:
        pass
    await apply_swap_config(f"serve group add {saved.get('name')}")
    return {"group": saved, "warnings": warnings}


@router.put("/api/swap-groups/{name}")
async def update_swap_group(name: str, g: SwapGroupIn):
    _reject_cross_engine(cfg.swap_groups, name)  # DEC-17 safety net
    data = _stamp_group_engine(g.model_dump())
    _swap_group_write_guard(data, new_group=False, existing_name=name)
    try:
        saved = cfg.swap_groups.update(name, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    warnings = swap_manager.duplicate_swap_claims()
    try:
        import config_advice
        warnings += config_advice.serve_group_warnings(saved)
    except Exception:
        pass
    await apply_swap_config(f"serve group update {saved.get('name')}")
    return {"group": saved, "warnings": warnings}


@router.delete("/api/swap-groups/{name}")
async def delete_swap_group(name: str):
    _reject_cross_engine(cfg.swap_groups, name)  # DEC-17 safety net
    if not access.gate_open():
        # restricted deletion must target a group they can fully see —
        # same visibility rule the listing shows them (404 otherwise)
        _swap_group_write_guard(None, new_group=False, existing_name=name)
    remaining = cfg.swap_groups.delete(name)
    await apply_swap_config(f"serve group delete {name}")
    return {"remaining": remaining}


@router.get("/api/groups/{name}/status")
async def group_status(name: str):
    try:
        return await run_in_threadpool(rpc_manager.group_status, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
