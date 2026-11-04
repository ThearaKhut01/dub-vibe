"""Secure API key storage for online AI providers.

Keys are stored in the Windows Credential Manager (via the keyring package)
under the 'DubVibe Studio' service — never in the frontend, never in plain
files, never returned to the browser. The frontend can only set a key, delete
it, and ask whether a key exists.
"""
from __future__ import annotations

import threading

PROVIDERS = ('gemini', 'openai', 'codecraft', 'custom')
_SERVICE = 'DubVibe Studio'

_LOCK = threading.Lock()


def _account(provider: str) -> str:
    return f'{provider} api key'


def set_key(provider: str, key: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f'Unknown provider: {provider}')
    key = (key or '').strip()
    if not key:
        raise ValueError('API key is empty')
    import keyring

    with _LOCK:
        keyring.set_password(_SERVICE, _account(provider), key)


def delete_key(provider: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f'Unknown provider: {provider}')
    import keyring

    with _LOCK:
        try:
            keyring.delete_password(_SERVICE, _account(provider))
        except Exception:
            pass  # already absent


def has_key(provider: str) -> bool:
    if provider not in PROVIDERS:
        return False
    import keyring

    with _LOCK:
        try:
            return bool(keyring.get_password(_SERVICE, _account(provider)))
        except Exception:
            return False


def get_key(provider: str) -> str | None:
    """Backend-only accessor for provider calls. Never exposed via the API."""
    if provider not in PROVIDERS:
        return None
    import keyring

    with _LOCK:
        try:
            return keyring.get_password(_SERVICE, _account(provider))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Pooled keys: one credential per API-pool configuration
# ---------------------------------------------------------------------------
#
# A provider can have several API-pool configurations, each with its own key.
# They live under the same service with a `pool api key <id>` account, so the
# single-key API above keeps working exactly as before and the pool never
# overwrites the key a user saved in API Settings.

def _pool_account(entry_id: str) -> str:
    return f'pool api key {entry_id}'


def _check_entry_id(entry_id: str) -> str:
    entry_id = str(entry_id or '').strip()
    if not entry_id:
        raise ValueError('Pool configuration id is empty')
    if len(entry_id) > 128:
        raise ValueError('Pool configuration id is too long')
    return entry_id


def set_pool_key(entry_id: str, key: str) -> None:
    """Store the key of one pool configuration (never the shared provider key)."""
    entry_id = _check_entry_id(entry_id)
    key = (key or '').strip()
    if not key:
        raise ValueError('API key is empty')
    import keyring

    with _LOCK:
        keyring.set_password(_SERVICE, _pool_account(entry_id), key)


def get_pool_key(entry_id: str) -> str | None:
    """Backend-only accessor for a pool configuration's key."""
    if not entry_id:
        return None
    import keyring

    with _LOCK:
        try:
            return keyring.get_password(_SERVICE, _pool_account(entry_id))
        except Exception:
            return None


def has_pool_key(entry_id: str) -> bool:
    return bool(get_pool_key(entry_id))


def delete_pool_key(entry_id: str) -> None:
    if not entry_id:
        return
    import keyring

    with _LOCK:
        try:
            keyring.delete_password(_SERVICE, _pool_account(entry_id))
        except Exception:
            pass  # already absent
