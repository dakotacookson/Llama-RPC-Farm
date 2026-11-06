"""Agent configuration (env + CLI)."""
from dataclasses import dataclass, field
import os
from pathlib import Path

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8198
DEFAULT_DISCOVERY_PORT = 8199


def _int_env(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


@dataclass
class AgentConfig:
    token: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    rpc_bin: str = ""
    llama_bin: str = ""
    log_dir: str = ""
    state_dir: str = ""
    model_roots: list[str] = field(default_factory=list)
    file_roots: list[str] = field(default_factory=list)
    discovery: bool = True
    discovery_host: str = DEFAULT_HOST
    discovery_port: int = DEFAULT_DISCOVERY_PORT

    @classmethod
    def from_env(cls, token=None, host=None, port=None, rpc_bin=None,
                 llama_bin=None, log_dir=None, state_dir=None,
                 model_roots=None, discovery=None, discovery_host=None,
                 discovery_port=None, file_roots=None):
        roots = list(model_roots or [])
        if not roots:
            roots = [p for p in (os.environ.get("PEER_AGENT_MODEL_ROOTS") or "")
                     .split(os.pathsep) if p]
        return cls(
            token=(token if token is not None
                   else (os.environ.get("PEER_AGENT_TOKEN")
                         or os.environ.get("AGENT_TOKEN") or "")),
            host=(host if host is not None
                  else os.environ.get("PEER_AGENT_HOST") or DEFAULT_HOST),
            port=(int(port) if port is not None
                  else _int_env("PEER_AGENT_PORT", DEFAULT_PORT)),
            rpc_bin=(rpc_bin if rpc_bin is not None
                     else os.environ.get("PEER_AGENT_RPC_BIN") or ""),
            llama_bin=(llama_bin if llama_bin is not None
                       else os.environ.get("PEER_AGENT_LLAMA_BIN") or ""),
            log_dir=(log_dir if log_dir is not None
                     else os.environ.get("PEER_AGENT_LOG_DIR") or ""),
            state_dir=(state_dir if state_dir is not None
                       else os.environ.get("PEER_AGENT_STATE_DIR") or ""),
            model_roots=roots,
            file_roots=list(file_roots or [
                p for p in (os.environ.get("PEER_AGENT_FILE_ROOTS") or "")
                .split(os.pathsep) if p]),
            discovery=(bool(discovery) if discovery is not None else
                       (os.environ.get("PEER_AGENT_DISCOVERY") or "1").lower()
                       not in ("0", "false", "no", "off")),
            discovery_host=(discovery_host if discovery_host is not None
                            else os.environ.get("PEER_AGENT_DISCOVERY_HOST")
                            or DEFAULT_HOST),
            discovery_port=(int(discovery_port) if discovery_port is not None
                            else _int_env("PEER_AGENT_DISCOVERY_PORT",
                                          DEFAULT_DISCOVERY_PORT)),
        )


def ensure_dirs(config: "AgentConfig") -> None:
    """Create the log/state dirs the agent writes into on first use.

    A freshly installed standalone agent starts with no ~/.cache tree;
    the writers already mkdir lazily, but doing it at startup makes the
    install self-contained and surfaces permission problems early.
    """
    log_dir = Path(config.log_dir).expanduser() if config.log_dir else Path.home()
    state_dir = (Path(config.state_dir).expanduser() if config.state_dir
                 else Path.home() / ".cache" / "llama-wrapper")
    for path in (log_dir, state_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
