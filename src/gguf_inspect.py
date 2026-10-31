"""Minimal GGUF header reader: tensor names without loading weights.

llama.cpp has no dry-run validator, so static checks like "does this
file contain MTP draft layers" need the header parsed directly. Only
the header (metadata + tensor infos) is read; weight data is never
touched, so even 30GB files answer in milliseconds.
"""
import os
import struct

# GGUF value types (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}
_STR, _ARR = 8, 9
_MAX_ITEMS = 10_000_000
_MAX_HEADER_BYTES = 1 << 30  # 1 GiB sanity cap


def _read(f, n):
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF header")
    return data


def _read_str(f):
    (ln,) = struct.unpack("<Q", _read(f, 8))
    if ln > _MAX_HEADER_BYTES:
        raise ValueError("absurd GGUF string length")
    return _read(f, ln).decode("utf-8", errors="replace")


def _read_value(f, vtype):
    if vtype in _FIXED_SIZES:
        raw = _read(f, _FIXED_SIZES[vtype])
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
        return struct.unpack(fmt, raw)[0]
    if vtype == _STR:
        return _read_str(f)
    if vtype == _ARR:
        (elem,) = struct.unpack("<I", _read(f, 4))
        (ln,) = struct.unpack("<Q", _read(f, 8))
        if ln > _MAX_ITEMS:
            raise ValueError("absurd GGUF array length")
        return [_read_value(f, elem) for _ in range(ln)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def _skip_value(f, vtype):
    _read_value(f, vtype)


# ggml type sizes in bytes/element (quantized include scale overhead)
_GGML_TYPE_SIZES = {
    0: 4.0,    # F32
    1: 2.0,    # F16
    2: 0.5625,  # Q4_0 (18/32)
    3: 0.625,   # Q4_1 (20/32)
    5: 0.5625,  # Q4_0_4_4
    6: 0.5625,  # Q4_0_4_8
    7: 0.625,   # Q4_0_8_8
    8: 1.0625,  # Q8_0 (34/32)
    9: 0.5625,  # Q5_0 (22/32 -> 0.6875, approx)
    10: 0.6875,  # Q5_1
    12: 0.5625,  # Q2_K
    13: 0.5625,  # Q3_K
    15: 0.5625,  # Q4_K
    16: 0.625,   # Q5_K
    17: 0.6875,  # Q6_K
    18: 1.0625,  # Q8_K
    19: 2.0,     # IQ2_XXS
    24: 2.0,     # BF16
}


def metadata(path):
    """Full metadata dict (scalars only; arrays kept as lists)."""
    meta = {}
    with open(path, "rb") as f:
        magic, _, _, n_kv = struct.unpack("<4sIQQ", _read(f, 24))
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", _read(f, 4))
            meta[key] = _read_value(f, vtype)
            if f.tell() > _MAX_HEADER_BYTES:
                raise ValueError("GGUF metadata exceeds sanity cap")
    return meta


def get_architecture(path):
    """Extract the model architecture from GGUF metadata (general.architecture)."""
    try:
        meta = metadata(path)
        arch = meta.get("general.architecture", "")
        return str(arch).lower() if arch else "unknown"
    except Exception:
        return "unknown"


def get_decision_type(path):
    """TypeSafe decision-model type ('laya', 'clef', ...) or ''.

    Mirrors llama.cpp's own gate (common_get_decision_type in
    common/common.cpp): a model is a decision model iff GGUF metadata
    carries <general.architecture>.decision.type. Those models serve
    only /v1/systemone — no chat generation — so flags like jinja,
    sampling, MTP or KV-cache quantization never apply to them.
    """
    try:
        meta = metadata(path)
        arch = str(meta.get("general.architecture", "") or "").strip()
        if not arch:
            return ""
        dtype = meta.get(f"{arch}.decision.type", "")
        return str(dtype).strip().lower() if dtype else ""
    except Exception:
        return ""


def weight_bytes(path):
    """Total weight bytes from tensor infos (dtype x elements)."""
    total = 0.0
    with open(path, "rb") as f:
        magic, _, n_tensors, n_kv = struct.unpack("<4sIQQ", _read(f, 24))
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        for _ in range(n_kv):
            _read_str(f)
            (vtype,) = struct.unpack("<I", _read(f, 4))
            _skip_value(f, vtype)
        for _ in range(n_tensors):
            _read_str(f)  # name
            (ndims,) = struct.unpack("<I", _read(f, 4))
            elems = 1
            for _ in range(ndims):
                (d,) = struct.unpack("<Q", _read(f, 8))
                elems *= d
            (dtype,) = struct.unpack("<I", _read(f, 4))
            _read(f, 8)  # offset
            total += elems * _GGML_TYPE_SIZES.get(dtype, 2.0)
    return int(total)


def tensor_names(path):
    """Return the tensor name list from a GGUF file header."""
    names = []
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack(
            "<4sIQQ", _read(f, 24))
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        if n_tensors > _MAX_ITEMS or n_kv > _MAX_ITEMS:
            raise ValueError("absurd GGUF counts")
        for _ in range(n_kv):
            _read_str(f)  # key
            (vtype,) = struct.unpack("<I", _read(f, 4))
            _skip_value(f, vtype)
        if f.tell() > _MAX_HEADER_BYTES:
            raise ValueError("GGUF metadata exceeds sanity cap")
        for _ in range(n_tensors):
            names.append(_read_str(f))  # tensor name
            (ndims,) = struct.unpack("<I", _read(f, 4))
            if ndims > 8:
                raise ValueError("absurd GGUF tensor rank")
            _read(f, 8 * ndims + 4 + 8)  # dims + dtype + offset
    return names


def has_mtp_layers(path):
    """True if the file carries MTP draft tensors, either embedded
    (blk.N.nextn.*) or as a standalone companion (top-level nextn.*).

    This mirrors llama.cpp's own gate: a draft-mtp context requires
    n_layer_nextn > 0, which only exists when nextn tensors are present.

    Also catches Eagle/Medusa style draft tensors (eagle.*, medusa.*,
    draft.*, decode_*.draft).
    """
    import re
    mtp = re.compile(r"(^|\.)nextn(\.|$)")
    eagle = re.compile(r"(^|\.)eagle(\.|$)")
    medusa = re.compile(r"(^|\.)medusa(\.|$)")
    draft = re.compile(r"(^|\.)draft(\.|$)")
    decode_draft = re.compile(r"(^|\.)decode.*\.draft(\.|$)")
    return any(
        mtp.search(n) or eagle.search(n) or medusa.search(n)
        or draft.search(n) or decode_draft.search(n)
        for n in tensor_names(path)
    )


def shard_paths(first_path):
    """All shard files for a (possibly sharded) model, starting at the
    first shard. Non-sharded input returns [first_path]."""
    import re
    base_dir = os.path.dirname(first_path)
    fname = os.path.basename(first_path)
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", fname, re.IGNORECASE)
    if not m:
        return [first_path]
    stem = fname[: m.start()]
    total = int(m.group(2))
    found = []
    try:
        names = sorted(os.listdir(base_dir))
    except OSError:
        return [first_path]
    for i in range(1, total + 1):
        want = f"{stem}-{i:05d}-of-{total:05d}.gguf"
        if want in names:
            found.append(os.path.join(base_dir, want))
    return found or [first_path]


def has_mtp_layers_anywhere(first_path):
    """MTP check across all shards (draft tensors may live past shard 1)."""
    return any(has_mtp_layers(p) for p in shard_paths(first_path))


def block_attention_kinds(first_path):
    """Classify blocks across all shards: (n_full_attention, n_linear,
    ssm_state_bytes). Full-attention blocks carry attn_k tensors and
    need per-token KV cache; linear/recurrent (ssm/linear) blocks hold
    fixed-size state instead. Counting from tensor names, not metadata.
    """
    import re
    kinds = {}
    for path in shard_paths(first_path):
        for name in tensor_names(path):
            m = re.match(r"blk\.(\d+)\.(.+)", name)
            if m:
                kinds.setdefault(int(m.group(1)), set()).add(m.group(2))
    # Architecture-agnostic linear/recurrent patterns
    _LINEAR_PATTERNS = (
        "ssm", "linear", "mamba", "rnn", "recurrent",
        "gru", "lstm", "rwkv", "linear_attn", "mamba2", "rnn2",
    )

    n_full = n_lin = 0
    for tensors in kinds.values():
        joined = " ".join(tensors)
        if "attn_k" in joined:
            n_full += 1
        elif any(p in joined for p in _LINEAR_PATTERNS):
            n_lin += 1
        else:
            n_full += 1  # unknown layout: assume KV (conservative)
    return n_full, n_lin, kinds


def ssm_state_bytes(first_path, meta):
    """Fixed recurrent-state bytes for linear layers (not ctx-scaled)."""
    def _u(*suffixes):
        for key, val in meta.items():
            kl = str(key).lower()
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)) and any(
                    kl.endswith(s) for s in suffixes):
                return int(val)
        return 0
    state = _u(".ssm.state_size", "state_size")
    groups = _u(".ssm.group_count", ".ssm.num_groups", "group_count") or 1
    inner = _u(".ssm.inner_size", "inner_size")
    if not (state and inner):
        return 0
    n_full, n_lin, _ = block_attention_kinds(first_path)
    return n_lin * groups * state * inner * 4  # f32 running state
