"""CLI entry point: python -m llama_peer_agent / llama-peer-agent."""
import argparse
import sys

from . import PROTOCOL_VERSION, __version__
from .config import AgentConfig
from .server import serve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llama-peer-agent",
                                 description="Optional Llama RPC Farm peer agent")
    ap.add_argument("--host", default=None,
                    help="bind address (default: PEER_AGENT_HOST or 0.0.0.0)")
    ap.add_argument("--port", type=int, default=None,
                    help="listen port (default: PEER_AGENT_PORT or 8198)")
    ap.add_argument("--token", default=None,
                    help="bearer token (default: PEER_AGENT_TOKEN)")
    ap.add_argument("--rpc-bin", default=None,
                    help="path to ggml-rpc-server (default: auto-discover)")
    ap.add_argument("--llama-bin", default=None,
                    help="path to llama-server (default: auto-discover)")
    ap.add_argument("--log-dir", default=None,
                    help="directory for rpc-server-<port>.log (default: $HOME)")
    ap.add_argument("--state-dir", default=None,
                    help="directory for rpc.state.<port> (default: ~/.cache/llama-wrapper)")
    ap.add_argument("--model-root", action="append", default=None,
                    help="repeatable model store root reported by /capabilities")
    ap.add_argument("--file-root", action="append", default=None,
                    help="repeatable extra root allowed for file operations")
    ap.add_argument("--no-discovery", action="store_true",
                    help="disable the UDP discovery responder")
    ap.add_argument("--discovery-host", default=None,
                    help="discovery bind address (default: 0.0.0.0)")
    ap.add_argument("--discovery-port", type=int, default=None,
                    help="discovery UDP port (default: 8199)")
    ap.add_argument("--version", action="store_true",
                    help="print version + protocol and exit")
    args = ap.parse_args(argv)
    if args.version:
        print(f"llama-peer-agent {__version__} protocol={PROTOCOL_VERSION}")
        return 0
    try:
        config = AgentConfig.from_env(
            token=args.token, host=args.host, port=args.port,
            rpc_bin=args.rpc_bin, llama_bin=args.llama_bin,
            log_dir=args.log_dir,
            state_dir=args.state_dir, model_roots=args.model_root,
            discovery=(False if args.no_discovery else None),
            discovery_host=args.discovery_host,
            discovery_port=args.discovery_port,
            file_roots=args.file_root)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not config.token:
        print("error: PEER_AGENT_TOKEN is required (or --token)", file=sys.stderr)
        return 2
    try:
        serve(config)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
