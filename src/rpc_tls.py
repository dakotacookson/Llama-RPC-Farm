"""Optional mutual-TLS transport for llama.cpp RPC.

llama.cpp RPC itself has no auth and no TLS. This module wraps the raw TCP
stream without touching llama.cpp:

  peer agent:   TLS listener :53000+  -> 127.0.0.1:<existing rpc port>
  master app:   127.0.0.1:<existing rpc port> -> peer:<tls port>

The existing RPC port numbers, hashing, state files, RPC<N> numbering, and
start/stop scripts stay the source of truth. Only the `--rpc` host changes
from `peer-ip` to `127.0.0.1` when secure mode is on.
"""
from __future__ import annotations

import datetime
import ipaddress
import os
import socket
import ssl
import struct
import threading

from runtime_paths import data_dir

RPC_PORT_BASE = 51000
TUNNEL_PORT = 5399
# Legacy per-port mapping retained for older status displays/tests.
TLS_PORT_BASE = 53000
TLS_PORT_RANGE = 500
TLS_PORT_LEGACY_BASE = 53500
CERT_DIR = os.path.join(data_dir(), "config", "rpc_tls")
CA_CRT = os.path.join(CERT_DIR, "ca.crt")
CA_KEY = os.path.join(CERT_DIR, "ca.key")
MASTER_CRT = os.path.join(CERT_DIR, "master.crt")
MASTER_KEY = os.path.join(CERT_DIR, "master.key")


def tls_port_for(rpc_port):
    """Legacy helper retained for callers/tests; live transport uses the
    single TUNNEL_PORT per peer."""
    p = int(rpc_port)
    if RPC_PORT_BASE <= p < RPC_PORT_BASE + TLS_PORT_RANGE:
        return TLS_PORT_BASE + (p - RPC_PORT_BASE)
    return TLS_PORT_LEGACY_BASE + (p % TLS_PORT_RANGE)


def enabled():
    try:
        import settings_store
        s = settings_store._load_settings_yaml_preserving(
            settings_store.SETTINGS_FILE) or {}
        return str(s.get("rpc_security") or "off").lower() == "mtls"
    except Exception:
        return False


def set_enabled(on):
    import settings_store
    data = settings_store._load_settings_yaml_preserving(
        settings_store.SETTINGS_FILE) or {}
    data["rpc_security"] = "mtls" if on else "off"
    settings_store._dump_settings_yaml_preserving(
        data, settings_store.SETTINGS_FILE)


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------

def _cert_paths(name):
    safe = "".join(c if c.isalnum() or c in "-_." else "-"
                   for c in str(name or "peer")).strip("-") or "peer"
    return (os.path.join(CERT_DIR, f"{safe}.crt"),
            os.path.join(CERT_DIR, f"{safe}.key"))


def _write_private(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _load_or_create_ca():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    if os.path.isfile(CA_CRT) and os.path.isfile(CA_KEY):
        return CA_CRT, CA_KEY
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                         "Llama RPC Farm RPC CA")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                           critical=True)
            .sign(key, hashes.SHA256()))
    _write_private(CA_KEY, key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    _write_private(CA_CRT, cert.public_bytes(serialization.Encoding.PEM))
    return CA_CRT, CA_KEY


def issue_cert(common_name, ips=None, client=False):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    ca_crt, ca_key = _load_or_create_ca()
    with open(ca_key, "rb") as f:
        ca_private = serialization.load_pem_private_key(f.read(), None)
    with open(ca_crt, "rb") as f:
        ca_cert = x509.load_pem_x509_certificate(f.read())
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (x509.CertificateBuilder()
               .subject_name(x509.Name([
                   x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
               .issuer_name(ca_cert.subject)
               .public_key(key.public_key())
               .serial_number(x509.random_serial_number())
               .not_valid_before(now - datetime.timedelta(minutes=5))
               .not_valid_after(now + datetime.timedelta(days=825)))
    san = []
    for ip in ips or []:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(str(ip).strip())))
        except ValueError:
            pass
    if san:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(san), critical=False)
    eku = (ExtendedKeyUsageOID.CLIENT_AUTH if client
           else ExtendedKeyUsageOID.SERVER_AUTH)
    builder = builder.add_extension(x509.ExtendedKeyUsage([eku]),
                                    critical=False)
    cert = builder.sign(ca_private, hashes.SHA256())
    crt, key_path = _cert_paths(common_name)
    _write_private(key_path, key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    _write_private(crt, cert.public_bytes(serialization.Encoding.PEM))
    return crt, key_path


def ensure_ca():
    _load_or_create_ca()
    return CA_CRT


def issue_master_cert():
    if os.path.isfile(MASTER_CRT) and os.path.isfile(MASTER_KEY):
        return MASTER_CRT, MASTER_KEY
    crt, key = issue_cert("llama-rpc-master", client=True)
    os.replace(crt, MASTER_CRT)
    os.replace(key, MASTER_KEY)
    return MASTER_CRT, MASTER_KEY


def issue_peer_cert(machine):
    name = machine.get("name") or "peer"
    ips = [machine.get("ip")] if machine.get("ip") else []
    return issue_cert(f"llama-rpc-peer-{name}", ips=ips, client=False)


def cert_bundle(machine):
    crt, key = issue_peer_cert(machine)
    with open(crt, "rb") as f:
        cert = f.read()
    with open(key, "rb") as f:
        priv = f.read()
    with open(CA_CRT, "rb") as f:
        ca = f.read()
    return {"cert": cert.decode("utf-8", "replace"),
            "key": priv.decode("utf-8", "replace"),
            "ca": ca.decode("utf-8", "replace")}


def server_context(machine_name):
    from cryptography import x509  # noqa: F401  (import check)
    crt, key = _cert_paths(f"llama-rpc-peer-{machine_name}")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(crt, key)
    ctx.load_verify_locations(CA_CRT)
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def client_context():
    issue_master_cert()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(MASTER_CRT, MASTER_KEY)
    ctx.load_verify_locations(CA_CRT)
    ctx.verify_mode = ssl.CERT_REQUIRED
    # Peer IPs can drift; the CA + client cert is the identity gate.
    ctx.check_hostname = False
    return ctx


# ---------------------------------------------------------------------------
# TCP proxy
# ---------------------------------------------------------------------------

def _pump(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


class TLSProxy:
    """One listener; each accepted connection is bridged to a TCP target."""

    def __init__(self, listen_host, listen_port, target_host, target_port,
                 context, server_side, route_port=None):
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.target_host = target_host
        self.target_port = int(target_port) if target_port else None
        self.context = context
        self.server_side = bool(server_side)
        self.route_port = int(route_port) if route_port else None
        self._sock = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._sock is not None:
            return self
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.listen_host, self.listen_port))
        s.listen(64)
        s.settimeout(0.5)
        self._sock = s
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name=f"rpc-tls-{self.listen_port}")
        self._thread.start()
        return self

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn):
        target = None
        try:
            if self.server_side:
                conn = self.context.wrap_socket(conn, server_side=True)
                target_port = self.target_port
                if target_port is None:
                    hdr = conn.recv(2)
                    if len(hdr) != 2:
                        raise OSError("missing tunnel target")
                    target_port = struct.unpack("!H", hdr)[0]
                    if not (target_port == 50052
                            or 51000 <= target_port <= 51499
                            or 52000 <= target_port <= 52499):
                        raise OSError("tunnel target outside RPC range")
                target = socket.create_connection(
                    ("127.0.0.1", target_port), timeout=10)
            else:
                target = socket.create_connection(
                    (self.target_host, self.target_port), timeout=10)
                target = self.context.wrap_socket(
                    target, server_hostname=self.target_host)
                if self.route_port:
                    target.sendall(struct.pack("!H", self.route_port))
            threading.Thread(target=_pump, args=(conn, target),
                             daemon=True).start()
            threading.Thread(target=_pump, args=(target, conn),
                             daemon=True).start()
        except Exception:
            for s in (conn, target):
                try:
                    if s:
                        s.close()
                except OSError:
                    pass

    def stop(self):
        self._stop.set()
        try:
            if self._sock:
                self._sock.close()
        except OSError:
            pass
        self._sock = None


# master-side proxies: (peer_ip, rpc_port) -> TLSProxy
_LOCAL_PROXIES = {}
_LOCAL_LOCK = threading.Lock()


def local_proxy_key(peer_ip, rpc_port):
    return (str(peer_ip), int(rpc_port))


def ensure_local_proxy(peer_ip, rpc_port):
    key = local_proxy_key(peer_ip, rpc_port)
    with _LOCAL_LOCK:
        existing = _LOCAL_PROXIES.get(key)
        if existing is not None:
            return existing
        p = TLSProxy("127.0.0.1", int(rpc_port), str(peer_ip),
                     TUNNEL_PORT, client_context(),
                     server_side=False, route_port=rpc_port)
        p.start()
        _LOCAL_PROXIES[key] = p
        return p


def stop_local_proxy(peer_ip, rpc_port):
    key = local_proxy_key(peer_ip, rpc_port)
    with _LOCAL_LOCK:
        p = _LOCAL_PROXIES.pop(key, None)
    if p:
        p.stop()


def stop_all_local_proxies():
    with _LOCAL_LOCK:
        items = list(_LOCAL_PROXIES.values())
        _LOCAL_PROXIES.clear()
    for p in items:
        p.stop()


def refresh_local_proxies():
    """Start proxies for every configured RPC endpoint, stop stale ones.

    Called at app startup and after config pushes. With security off it
    stops every proxy and returns.
    """
    if not enabled():
        stop_all_local_proxies()
        return {"enabled": False, "proxies": []}
    import config_loader as cfg
    import local_node
    import rpc_manager
    import swap_manager
    host_name = None
    try:
        host_name = swap_manager.load_settings().get("host_name")
    except Exception:
        pass
    if not host_name:
        for m in cfg.machines.all():
            if local_node.is_local(m):
                host_name = m.get("name")
                break
    wanted = {}
    try:
        for model in cfg.list_models():
            deployments = [model]
            for v in (model.get("variants") or []):
                try:
                    deployments.append(rpc_manager._resolve_variant(
                        model, v.get("label")))
                except Exception:
                    pass
            for dep in deployments:
                servers = dep.get("servers") or []
                if not servers:
                    continue
                _master, peers = rpc_manager._split_servers(
                    servers, host_name)
                if not peers:
                    continue
                ports = rpc_manager._model_rpc_ports(dep, peers)
                for p in peers:
                    for (pname, _gname), port in ports.items():
                        if pname == p.get("name") and p.get("ip"):
                            wanted[local_proxy_key(p["ip"], port)] = True
    except Exception as e:
        return {"enabled": True, "error": str(e), "proxies": []}
    with _LOCAL_LOCK:
        stale = [k for k in _LOCAL_PROXIES if k not in wanted]
    for ip, port in stale:
        stop_local_proxy(ip, port)
    started = []
    for (ip, port) in wanted:
        try:
            ensure_local_proxy(ip, port)
            started.append({"ip": ip, "port": port,
                            "tls_port": tls_port_for(port)})
        except Exception as e:
            started.append({"ip": ip, "port": port, "error": str(e)})
    return {"enabled": True, "proxies": started}


def status():
    with _LOCAL_LOCK:
        proxies = [{"ip": k[0], "port": k[1], "tls_port": TUNNEL_PORT}
                   for k in _LOCAL_PROXIES]
    return {"enabled": enabled(), "proxies": proxies,
            "tunnel_port": TUNNEL_PORT,
            "ca": os.path.isfile(CA_CRT),
            "master_cert": os.path.isfile(MASTER_CRT)}
