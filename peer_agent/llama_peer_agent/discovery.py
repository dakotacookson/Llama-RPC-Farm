"""UDP discovery responder for the peer agent.

Agents answer a tiny, unauthenticated discovery probe with identity-only
metadata (no secrets). Enrollment still requires the bearer token.
"""
import json
import logging
import socket
import threading

from . import PROTOCOL_VERSION, __version__
from . import processes
from .config import AgentConfig

MAGIC = b"LLAMA_FARM_DISCOVER_V1"
DEFAULT_DISCOVERY_PORT = 8199
log = logging.getLogger("llama_peer_agent.discovery")


def payload(config: AgentConfig) -> dict:
    body = processes.info(config)
    body.update({
        "magic": MAGIC.decode("ascii"),
        "protocol_version": PROTOCOL_VERSION,
        "agent_version": __version__,
    })
    return body


def create_socket(config: AgentConfig) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((config.discovery_host, int(config.discovery_port)))
    sock.settimeout(0.5)
    return sock


def serve(config: AgentConfig, sock=None, stop_event=None):
    """Answer discovery probes until stop_event is set."""
    stop_event = stop_event or threading.Event()
    own_socket = sock is None
    if own_socket:
        sock = create_socket(config)
    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.strip() != MAGIC:
                continue
            reply = json.dumps(payload(config), separators=(",", ":")).encode()
            try:
                sock.sendto(reply, addr)
            except OSError:
                pass
    finally:
        if own_socket:
            try:
                sock.close()
            except OSError:
                pass


def start(config: AgentConfig) -> tuple:
    """Start the responder thread. Returns (thread, socket, stop_event)."""
    sock = create_socket(config)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=serve, args=(config, sock, stop_event),
        name="llama-peer-agent-discovery", daemon=True)
    thread.start()
    return thread, sock, stop_event
