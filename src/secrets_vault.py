"""Secrets vault: single-.env encrypted storage for sensitive values.

Design (per user request):
- ONE git-ignored `.env` at the project root holds everything sensitive:
  FARM_SECRET_KEY, FARM_SALT, HF_TOKEN_ENC, MACHINE_<ID>_PASSWORD_ENC.
- YAML configs keep only non-secret fields. Machine `password` and
  `settings.yaml:hf_token` are migrated to the vault and cleared.
- At rest nothing sensitive is plaintext: values in `.env` are Fernet
  tokens (AES-128-CBC + HMAC-SHA256). They are decrypted on the fly only
  at the moment of use (SSH connect, HF API call, generated script).
- SHA-256 + salt is used two ways:
  1. `hash_password` / `verify_password` for any future app-login secret
     (one-way fingerprint: SHA-256(salt || password), compare_digest).
  2. Fernet itself authenticates with HMAC-SHA256 under the hood.

Security note: the Fernet key lives in the same `.env` (git-ignored, file
mode 0600). Anyone with read access to `.env` can decrypt — this protects
against accidental leaks (git, logs, screenshots, YAML sharing), not
against a full host compromise. A master-password prompt (key derived via
PBKDF2-HMAC-SHA256, never stored) can be added later for stronger at-rest
protection.
"""
import base64
import hashlib
import hmac
import os
import re
import secrets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(ENV_PATH, override=False)
except Exception:
    # lightweight fallback: parse KEY=VALUE lines ourselves
    try:
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:  # pragma: no cover - import error surfaces at ensure time
    Fernet = None
    InvalidToken = Exception


def _safe_id(machine_id):
    return re.sub(r"[^A-Z0-9]+", "_", str(machine_id or "").upper()).strip("_")


def machine_env_key(machine_id):
    return f"MACHINE_{_safe_id(machine_id)}_PASSWORD_ENC"


def _get_salt_hex():
    s = (os.environ.get("FARM_SALT") or "").strip()
    if not s:
        s = secrets.token_hex(16)
        set_env_value("FARM_SALT", s)
    return s


def _get_fernet():
    if Fernet is None:
        raise RuntimeError("cryptography package is required (pip install -r requirements.txt)")
    key = (os.environ.get("FARM_SECRET_KEY") or "").strip()
    if not key:
        key = Fernet.generate_key().decode()
        set_env_value("FARM_SECRET_KEY", key)
    return Fernet(key.encode())


# ---------------- password hashing (verification only) ----------------
# PBKDF2-HMAC-SHA256 (600k iters, OWASP 2023 guidance level) format:
#   pbkdf2$<iter>$<salt_hex>$<hash_hex>
# Legacy format (plain SHA-256 + farm salt: 'salt$hexdigest') verifies
# transparently — existing logins keep working; the NEXT password save
# upgrades the stored form to PBKDF2 automatically.
_PBKDF2_ITERATIONS = 600_000
_PBKDF2_PREFIX = "pbkdf2"


def hash_password(plain, salt_hex=None):
    """Return 'pbkdf2$<iter>$<salt>$<hash>' (or the legacy
    'salt$hexdigest' shape when a salt is passed explicitly — schema
    callers who pre-compute salts keep their format)."""
    import binascii
    salt_hex = salt_hex or secrets.token_hex(16)
    if salt_hex is not None and salt_hex.__class__ is not str:
        salt_hex = str(salt_hex)
    if salt_hex and _PBKDF2_PREFIX in (salt_hex or ""):
        salt_hex = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", (plain or "").encode(),
        bytes.fromhex(salt_hex), _PBKDF2_ITERATIONS).hex()
    return f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}${salt_hex}${digest}"


def _hash_legacy(plain, salt_hex):
    """The pre-PBKDF2 scheme: SHA-256(salt || password)."""
    return hashlib.sha256(
        bytes.fromhex(salt_hex) + (plain or "").encode()).hexdigest()


def verify_password(plain, stored):
    import binascii
    import hmac as _hmac
    try:
        stored = (stored or "")
        if stored.startswith(f"{_PBKDF2_PREFIX}$"):
            _, iters_s, salt_hex, digest = stored.split("$", 3)
            iters = max(1, min(int(iters_s), 5_000_000))
            expect = hashlib.pbkdf2_hmac(
                "sha256", (plain or "").encode(),
                bytes.fromhex(salt_hex), iters).hex()
            return _hmac.compare_digest(expect, digest)
        # legacy: farm-salt single-pass SHA-256
        salt_hex, digest = stored.split("$", 1)
        expect = _hash_legacy(
            plain, salt_hex or _get_salt_hex())
        return _hmac.compare_digest(expect, digest)
    except Exception:
        return False


# ---------------- reversible Fernet (decrypt-on-the-fly use) ----------------
def encrypt_value(plain):
    if not plain:
        return ""
    return _get_fernet().encrypt(str(plain).encode()).decode()


def decrypt_value(token):
    if not token:
        return ""
    token = token.strip()
    if not token:
        return ""
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except Exception:
        # legacy plaintext (pre-migration): return as-is so old .env still works
        return token


# ---------------- .env read/write ----------------
def _read_env_lines():
    if not os.path.exists(ENV_PATH):
        return []
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        return f.readlines()


def set_env_value(key, value):
    """Upsert KEY=VALUE in the single .env (mode 0600). Updates os.environ."""
    os.environ[key] = value
    lines = _read_env_lines()
    prefix = f"{key}="
    out, found = [], False
    for line in lines:
        if line.strip().startswith(prefix):
            out.append(f"{key}={value}\n")
            found = True
        else:
            out.append(line)
    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append(f"{key}={value}\n")
    tmp = ENV_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(out)
    os.replace(tmp, ENV_PATH)
    try:
        os.chmod(ENV_PATH, 0o600)
    except Exception:
        pass


def delete_env_value(key):
    os.environ.pop(key, None)
    if not os.path.exists(ENV_PATH):
        return
    prefix = f"{key}="
    out = [l for l in _read_env_lines() if not l.strip().startswith(prefix)]
    tmp = ENV_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(out)
    os.replace(tmp, ENV_PATH)


# ---------------- machine passwords ----------------
def set_machine_password(machine_id, plain):
    """Encrypt + store a machine SSH password in .env. Empty clears it."""
    key = machine_env_key(machine_id)
    if plain:
        set_env_value(key, encrypt_value(plain))
    else:
        delete_env_value(key)


def get_machine_password(machine):
    """Resolve a machine dict's password: vault first, legacy YAML fallback.

    Returns plaintext ONLY in memory for immediate use (SSH connect, script
    gen). Never writes it back to YAML.
    """
    mid = (machine or {}).get("id") or (machine or {}).get("name") or ""
    enc = os.environ.get(machine_env_key(mid), "")
    if enc:
        return decrypt_value(enc)
    # legacy: plaintext still sitting in machines.yaml pre-migration
    return (machine or {}).get("password") or ""


# ---------------- HF token ----------------
def get_hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        v = (os.environ.get(k) or "").strip()
        if v and not v.startswith("gAAAAA"):
            return v  # operator-provided plaintext env wins
    enc = (os.environ.get("HF_TOKEN_ENC") or "").strip()
    if enc:
        return decrypt_value(enc)
    return ""


def set_hf_token(plain):
    if plain and plain.strip():
        set_env_value("HF_TOKEN_ENC", encrypt_value(plain.strip()))
    else:
        delete_env_value("HF_TOKEN_ENC")


def hf_token_set():
    return bool(get_hf_token().strip())


# ---------------- farm web logins (users -> api key identity) -------------
# LOGIN_<NAME>_PASS     salted SHA-256 hash (hash_password) of the web
#                       login password for farm user <NAME>
# LOGIN_<NAME>_KEY      the Fernet-ENCRYPTED access key token that this
#                       login maps to (the same token clients can also
#                       use directly as Bearer on :8090)
# LOGIN_<NAME>_ADMIN    "1" if an admin identity
# WEB_SESSION_<hex>     encrypted "{user}|{expiry_iso}" session records —
#                       survive daemon restarts (nothing in memory)

def _login_env_key(name):
    n = re.sub(r"[^A-Z0-9]+", "_", str(name or "").upper()).strip("_")
    return f"LOGIN_{n}"

def set_web_login(name, password, access_key_token, admin=False):
    """Create/update a farm web login: salted-hash the password
    (never stored plaintext), encrypt the access-key mapping."""
    k = _login_env_key(name)
    if password:
        set_env_value(f"{k}_PASS", hash_password(password))
    if access_key_token:
        set_env_value(f"{k}_KEY", encrypt_value(access_key_token))
    set_env_value(f"{k}_ADMIN", "1" if admin else "0")

def get_web_login(name):
    """{hash, key, admin} for a login name or None if not set up."""
    reload_env()
    k = _login_env_key(name)
    h = os.environ.get(f"{k}_PASS") or ""
    key = decrypt_value(os.environ.get(f"{k}_KEY") or "")
    admin = (os.environ.get(f"{k}_ADMIN") or "0") == "1"
    if not h:
        return None
    return {"hash": h, "key": key, "admin": admin}

def web_login_exists(name):
    return bool(os.environ.get(f"{_login_env_key(name)}_PASS"))

def list_web_logins():
    out = []
    seen = set()
    for k in os.environ:
        if k.startswith("LOGIN_") and k.endswith("_PASS"):
            out.append(k[len("LOGIN_"):-len("_PASS")].lower())
            seen.add(k)
    return sorted(out)

def set_web_session(sid, user, ttl_hours=24*30):
    """Store an encrypted session record. Long-lived (30d) so browser
    restarts keep their identity; revocable by deleting the record."""
    import datetime as _dt
    blob = f"{user}|{(_dt.datetime.now().isoformat())}"
    enc = encrypt_value(blob)
    set_env_value(f"WEB_SESSION_{sid}", enc)
    _ = ttl_hours  # expiry is enforced by record age via verify below

def reload_env():
    """Re-read .env into os.environ. The gateway is a long-lived process
    while the farm app keeps WRITING the vault (new logins, sessions,
    rotated keys) — env was snapshotted at import, so session/cookie
    resolution silently went stale until restart (401s with valid
    grants). override=False: real env vars (SSH_PASS etc.) still win."""
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(ENV_PATH, override=False)
    except Exception:
        try:
            if os.path.exists(ENV_PATH):
                with open(ENV_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


def get_web_session(sid):
    """Resolve sid -> user name; None for unknown/garbage."""
    import datetime as _dt, json as _json
    reload_env()
    val = decrypt_value(os.environ.get(f"WEB_SESSION_{sid}") or "")
    if not val or "|" not in val:
        return None
    user, created = val.split("|", 1)
    try:
        age = _dt.datetime.now() - _dt.datetime.fromisoformat(created)
        if age.total_seconds() > 30*24*3600:
            delete_env_value(f"WEB_SESSION_{sid}")
            return None
    except Exception:
        pass
    return user.strip()

def delete_web_session(sid):
    delete_env_value(f"WEB_SESSION_{sid}")
