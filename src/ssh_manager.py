"""SSH layer: runs commands on remote machines.

Always uses paramiko for live SSH (paramiko works identically on
Windows and Linux). sshpass is NOT used for live calls -- the winget
sshpass shim on Windows silently fails; sshpass only appears inside
generated scripts, mirroring the user's existing .sh pattern.

Host-key fingerprinting (first-time SSH logins): Llama RPC Farm keeps its
own known-hosts store at config/ssh_known_hosts.json. A host seen for
the first time is fingerprinted (SHA256, pre-auth — no login needed)
and trusted on first use (TOFU); every later connection pins the
stored key, so a changed key is a hard, explicit error instead of a
silent MITM.
"""
import base64
import hashlib
import json
import os
import socket

import paramiko


class SSHError(Exception):
    pass


class HostKeyMismatch(SSHError):
    """The host presented a different key than the one we stored — the
    classic known_hosts 'WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED'
    case. Raised BEFORE auth, never auto-accepted."""
    pass


def _known_hosts_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "config", "ssh_known_hosts.json")


def load_known_hosts():
    try:
        with open(_known_hosts_path(), "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_known_hosts(store):
    tmp = _known_hosts_path() + ".tmp"
    os.makedirs(os.path.dirname(_known_hosts_path()), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=1)
    os.replace(tmp, _known_hosts_path())


def _host_id(ip, port):
    return f"{ip}:{int(port)}"


def host_fingerprint(ip, port=22, timeout=10):
    """Fetch a host's SSH key fingerprint WITHOUT logging in.

    Returns {"type", "fingerprint", "bits"} — the key exchange happens
    before auth, so this works for first-time contacts with no creds.
    Raises SSHError on unreachable/failed transport."""
    transport = paramiko.Transport((ip, int(port)), )
    transport.start_client(timeout=timeout)
    try:
        key = transport.get_remote_server_key()
        fp = "SHA256:" + base64.b64encode(
            hashlib.sha256(key.get_fingerprint()).digest()).decode().rstrip("=")
        return {"type": key.get_name(),
                "fingerprint": fp,
                "bits": key.get_bits()}
    finally:
        transport.close()


class SSHManager:
    def __init__(self, ip, username, password, port=22, timeout=10):
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.timeout = timeout

    # ---------- public API ----------
    def run(self, command, timeout=60):
        """Run a command on the remote machine. Returns (exit_code, stdout,
        stderr). Raises SSHError on connection/transport failure."""
        return self._run_paramiko(command, timeout)

    def push_file(self, local_path, remote_path):
        """SFTP a file to the remote machine (mirrors scp in the user's
        push_win_launcher pattern). Raises SSHError on failure."""
        client = paramiko.SSHClient()
        try:
            self._connect(client)
            with client.open_sftp() as sftp:
                sftp.put(local_path, remote_path)
        except (paramiko.ssh_exception.SSHException, OSError) as e:
            if isinstance(e, SSHError):
                raise
            if isinstance(e, OSError):
                raise SSHError(self._friendly_conn_error(e)) from e
            raise SSHError(f"SFTP to {self.ip} failed: {e}") from e
        finally:
            client.close()

    def push_text(self, text, remote_path, os_type="linux"):
        """Write a string to a remote file via SFTP (atomic: tmp + rename).

        Atomic so a stale root-owned file in a writable dir gets replaced
        instead of Errno 13. The rename command is chosen by os_type
        ("windows" -> PowerShell Move-Item, else mv -f), with a
        cross-fallback if the first attempt fails.
        """
        import io
        import time
        client = paramiko.SSHClient()
        try:
            self._connect(client)
            tmp = f"{remote_path}.tmp-{int(time.time() * 1000) % 1000000}"
            with client.open_sftp() as sftp:
                with sftp.open(tmp, "w") as f:
                    f.write(text)
            # Rename by OS type so it overwrites a stale root-owned dest
            # when the dir allows. Falls back to the other OS's command.
            def _mv(cmd):
                _, so, se = client.exec_command(cmd, timeout=15)
                o = so.read().decode("utf-8", errors="replace")
                e = se.read().decode("utf-8", errors="replace")
                return so.channel.recv_exit_status(), o, e

            is_win = (os_type or "linux").lower() == "windows"
            win_cmd = (f'powershell -NoProfile -Command '
                       f'"Move-Item -Force \\"{tmp}\\" \\"{remote_path}\\""')
            lin_cmd = f'mv -f "{tmp}" "{remote_path}"'
            first, second = (win_cmd, lin_cmd) if is_win else (lin_cmd, win_cmd)
            code, out, err = _mv(first)
            if code != 0:
                code, out, err = _mv(second)
            if code != 0:
                raise SSHError(
                    f"SFTP move to {self.ip}:{remote_path} failed: {err or out}")
        except (paramiko.ssh_exception.SSHException, OSError) as e:
            if isinstance(e, SSHError):
                raise
            if isinstance(e, OSError):
                raise SSHError(self._friendly_conn_error(e)) from e
            raise SSHError(f"SFTP to {self.ip} failed: {e}") from e
        finally:
            client.close()

    def check_port(self, port, timeout=3):
        """Check if a TCP port is open on the remote machine."""
        try:
            with socket.create_connection((self.ip, port), timeout=timeout):
                return True
        except OSError:
            return False

    def ssh_port_open(self, timeout=4):
        """Check if the machine's SSH port itself is open."""
        return self.check_port(self.port, timeout=timeout)

    def ssh_port_error(self, timeout=4):
        """None if SSH port is open, else a human-readable reason why not.

        Used when adding a never-before-seen machine so the UI can tell
        the user SSH was rejected because the port is closed — instead
        of a cryptic Errno 111 / timeout.
        """
        import errno as _errno
        try:
            with socket.create_connection((self.ip, self.port), timeout=timeout):
                return None
        except ConnectionRefusedError:
            return (f"SSH rejected: port {self.port} not open on {self.ip} "
                    f"(connection refused) — enable the OpenSSH server / "
                    f"open TCP port {self.port} on the target, then retry")
        except socket.timeout:
            return (f"SSH rejected: port {self.port} on {self.ip} timed out — "
                    f"host unreachable or SSH filtered; check IP/firewall")
        except OSError as e:
            if e.errno in (_errno.EHOSTUNREACH, _errno.ENETUNREACH):
                return (f"SSH rejected: {self.ip} unreachable ({e}) — "
                        f"check IP/network route")
            return (f"SSH rejected: cannot reach {self.ip} port {self.port} "
                    f"({e}) — check IP, port, and firewall")

    def is_reachable(self):
        """Quick reachability check: can we SSH and run 'echo ok'."""
        try:
            code, out, _ = self.run("echo ok", timeout=10)
            return code == 0 and "ok" in out
        except (SSHError, OSError):
            return False

    def gpu_info(self, vendor=None):
        """GPU query rows (index,name,total,used,temp — csv), or [].
        vendor: 'nvidia'|'amd'|None (None = try nvidia-smi, fall back to
        rocm-smi/amd-smi and report which one answered)."""
        import gpu_vendor as _gv
        queries = []
        v = _gv.normalize_vendor(vendor)
        if v == "nvidia":
            queries = ["nvidia"]
        elif v == "amd":
            queries = ["rocm"]
        elif v == "intel":
            queries = ["intel"]
        else:
            queries = ["nvidia", "rocm", "intel"]
        for which in queries:
            rows = self._query_gpu(which)
            if rows:
                return rows
        return []

    def _query_gpu(self, which):
        """Run one vendor's GPU query; returns csv rows or []."""
        import gpu_vendor as _gv
        try:
            if which == "nvidia":
                code, out, _ = self.run(_gv.nvidia_query(
                    "index,name,memory.total,memory.used,temperature.gpu"),
                    timeout=15)
                if code == 0 and out.strip():
                    return [l.strip() for l in out.splitlines() if l.strip()]
                return []
            if which == "intel":
                code, out, _ = self.run(_gv.intel_query_json(), timeout=20)
                if code == 0 and out.strip():
                    rows = _gv.parse_xpu_smi_json(out)
                    if rows:
                        return rows
                return []
            # AMD: rocm-smi JSON (ships with every ROCm install)
            code, out, _ = self.run(_gv.rocm_query_json(), timeout=20)
            if code == 0 and out.strip():
                rows = _gv.parse_rocm_json(out)
                if rows:
                    return rows
            return []
        except SSHError:
            return []

    # ---------- transport ----------
    def _connect(self, client):
        """Connect with host-key fingerprint handling:
        - host known: pin the stored key (RejectPolicy) — a changed key
          raises HostKeyMismatch BEFORE auth.
        - host new: AutoAddPolicy (TOFU) and record the fingerprint in
          config/ssh_known_hosts.json after a successful handshake.
        Returns (fingerprint, was_first_use)."""
        store = load_known_hosts()
        hid = _host_id(self.ip, self.port)
        known = store.get(hid)
        if known:
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            # HostKeys needs a PKey object, not the base64 string
            ktype = known.get("type", "")
            keydata = base64.b64decode(known["key"])
            if ktype.startswith("ssh-rsa"):
                pkey = paramiko.RSAKey(data=keydata)
            elif ktype.startswith("ecdsa-"):
                pkey = paramiko.ECDSAKey(data=keydata)
            else:
                pkey = paramiko.Ed25519Key(data=keydata)
            client.get_host_keys().add(self.ip, ktype, pkey)
        else:
            pinned = None
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        fp = was_first = None
        try:
            client.connect(
                self.ip,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=self.timeout,
                allow_agent=False,
                look_for_keys=False,
            )
        finally:
            # The key exchange completes BEFORE auth, so even a failed
            # login has a host key: record TOFU on first sight regardless
            # of the connect() outcome. (Auth failures keep raising; we
            # just don't lose the fingerprint.)
            transport = client.get_transport()
            if transport is not None:
                key = transport.get_remote_server_key()
                fp = "SHA256:" + base64.b64encode(
                    hashlib.sha256(key.get_fingerprint()).digest()
                ).decode().rstrip("=")
                if known:
                    if fp != known.get("fingerprint"):
                        raise HostKeyMismatch(
                            f"Host key changed for {self.ip}:{self.port}!\n"
                            f"  stored: {known.get('fingerprint')}\n"
                            f"  now:    {fp}\n"
                            f"If you reinstalled/re-keyed the host, delete "
                            f"config/ssh_known_hosts.json to re-trust.")
                else:
                    store[hid] = {"type": key.get_name(),
                                  "key": key.get_base64(),
                                  "fingerprint": fp,
                                  "added": __import__("datetime").datetime
                                  .now().isoformat(timespec="seconds")}
                    save_known_hosts(store)
                    was_first = True
        if known and fp != known.get("fingerprint"):
            raise HostKeyMismatch(
                f"Host key changed for {self.ip}:{self.port}!")
        return fp, bool(was_first)

    def _friendly_conn_error(self, e):
        """Map a raw connect failure to an actionable SSHError message."""
        import errno as _errno
        msg = str(e)
        # paramiko wraps refused/timeout connects in NoValidConnectionsError
        # (an OSError). Verify against the live port so auth-era failures
        # aren't misdiagnosed as a closed port.
        if "NoValidConnectionsError" in type(e).__name__ \
                or "Unable to connect to port" in msg:
            live = self.ssh_port_error(timeout=3)
            if live:
                return live
            return f"SSH to {self.ip} failed: {msg}"
        if isinstance(e, ConnectionRefusedError) or e.errno == _errno.ECONNREFUSED \
                or "Connection refused" in msg:
            return (f"SSH to {self.ip} rejected: SSH port {self.port} not open "
                    f"(connection refused) — enable the OpenSSH server / open "
                    f"TCP port {self.port} on the target, then retry")
        if isinstance(e, socket.timeout) or "timed out" in msg:
            return (f"SSH to {self.ip} timed out (port {self.port}) — host "
                    f"unreachable or SSH filtered; check IP/firewall")
        if isinstance(e, OSError) and e.errno in (_errno.EHOSTUNREACH, _errno.ENETUNREACH):
            return f"SSH to {self.ip} failed: host unreachable ({msg})"
        return f"SSH to {self.ip} failed: {msg}"

    def _run_paramiko(self, command, timeout):
        client = paramiko.SSHClient()
        try:
            self._connect(client)
            _, stdout, stderr = client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            code = stdout.channel.recv_exit_status()
            return code, out, err
        except HostKeyMismatch:
            raise
        except paramiko.ssh_exception.SSHException as e:
            # Auth failures / banner issues: surface verbatim, but if the
            # SSH port itself is closed, say so explicitly (fresh hosts).
            detail = self.ssh_port_error(timeout=3)
            if detail:
                raise SSHError(detail) from e
            raise SSHError(f"SSH to {self.ip} failed: {e}") from e
        except (OSError, socket.timeout) as e:
            raise SSHError(self._friendly_conn_error(e)) from e
        finally:
            client.close()