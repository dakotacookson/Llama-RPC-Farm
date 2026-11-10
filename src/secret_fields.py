"""Single source of truth for which config/API fields hold secrets.

Kept dependency-free so the API-only ``farmctl`` client can reuse the same
rule as the app and the machine store without importing the vault or
touching ``config/``. Secret values must never reach machines.yaml, an API
response, or a support snapshot; the *naming* rule for those fields lives
here so the three consumers cannot drift apart again.
"""

_SECRET_NAMES = frozenset({
    "password",
    "secret",
    "api_key",
    "access_key",
    "token",
})


def is_secret_key(key) -> bool:
    """True for a field name that carries a secret value.

    Covers the exact names above plus any ``*_token`` field (e.g.
    ``_agent_token``, ``agent_token``, ``hf_token``). Flag fields such as
    ``agent_token_set``/``password_set`` are deliberately not secret: they
    don't end in ``_token`` and only report whether a vault entry exists.
    """
    low = str(key).lower()
    return low in _SECRET_NAMES or low.endswith("_token")


def strip_secret_keys(record):
    """Return a shallow copy of ``record`` without secret-named fields."""
    return {k: v for k, v in record.items() if not is_secret_key(k)}
