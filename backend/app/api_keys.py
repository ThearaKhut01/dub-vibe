"""Secure API key storage for online AI providers.

Keys are stored in the Windows Credential Manager (via the keyring package)
under the 'DubVibe Studio' service — never in the frontend, never in plain
files, never returned to the browser. The frontend can only set a key, delete
it, and ask whether a key exists.
"""
from __future__ import annotations

import threading

PROVIDERS = ('gemini', 'openai')
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
