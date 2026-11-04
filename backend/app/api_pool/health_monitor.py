"""Per-entry health: cooldowns, consecutive failures and the last error.

A parked credential must stay parked across a restart. Without that, restarting
the app inside a 600 s quota cooldown puts the same exhausted key back at the
front of the queue and every subtitle line pays for the discovery again. The
cooldowns therefore live in ``health.json``; everything cheap and short-lived
(consecutive failure counts, latency) stays in memory and is rebuilt as the pool
is used.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from .models import health_path, storage_dir

_LOCK = threading.RLock()
_STATE: dict[str, dict[str, Any]] = {}
_LOADED = False
_PATH = ''

# A credential that fails this many times in a row is shown as degraded even
# when no cooldown is active (failover still uses it: the next model or endpoint
# may answer).
_DEGRADED_AFTER = 2


def _blank() -> dict[str, Any]:
    return {
        'consecutive_failures': 0,
        'cooldown_until': 0.0,
        'cooldown_reason': '',
        'last_error': '',
        'last_error_at': 0.0,
        'last_success_at': 0.0,
        'last_latency_ms': 0,
        'total_requests': 0,
        'total_failures': 0,
    }


def _load() -> None:
    global _LOADED, _PATH
    with _LOCK:
        current = str(health_path())
        if _LOADED and current == _PATH:
            return
        _LOADED = True
        _PATH = current
        _STATE.clear()
        try:
            raw = json.loads(health_path().read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        for entry_id, record in (raw.get('entries') or {}).items():
            if not isinstance(record, dict):
                continue
            state = _blank()
            try:
                state['cooldown_until'] = float(record.get('cooldown_until') or 0.0)
            except (TypeError, ValueError):
                state['cooldown_until'] = 0.0
            state['cooldown_reason'] = str(record.get('cooldown_reason') or '')
            state['last_error'] = str(record.get('last_error') or '')
            _STATE[str(entry_id)] = state


def _persist() -> None:
    """Write the durable part (cooldowns) atomically."""
    with _LOCK:
        payload = {
            'version': 1,
            'entries': {
                entry_id: {
                    'cooldown_until': state['cooldown_until'],
                    'cooldown_reason': state['cooldown_reason'],
                    'last_error': state['last_error'],
                }
                for entry_id, state in _STATE.items()
                if state['cooldown_until'] or state['last_error']
            },
        }
        target = health_path()
        temporary = storage_dir() / 'health.json.tmp'
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            import os

            os.replace(temporary, target)
        except OSError:
            pass  # health is best-effort; a read-only folder must not fail a request


def _state(entry_id: str) -> dict[str, Any]:
    _load()
    with _LOCK:
        return _STATE.setdefault(entry_id, _blank())


def cooldown_remaining(entry_id: str) -> float:
    state = _state(entry_id)
    with _LOCK:
        return max(0.0, float(state['cooldown_until']) - time.time())


def available(entry_id: str) -> bool:
    return cooldown_remaining(entry_id) <= 0.0


def record_success(entry_id: str, latency_ms: int = 0) -> None:
    state = _state(entry_id)
    with _LOCK:
        state['consecutive_failures'] = 0
        state['cooldown_until'] = 0.0
        state['cooldown_reason'] = ''
        state['last_success_at'] = time.time()
        state['last_latency_ms'] = int(latency_ms or 0)
        state['total_requests'] += 1


def record_failure(entry_id: str, *, message: str, cooldown: float = 0.0,
                   reason: str = '') -> None:
    """Note a failed call and park the entry when the failure justifies it."""
    state = _state(entry_id)
    persist = False
    with _LOCK:
        state['consecutive_failures'] += 1
        state['total_requests'] += 1
        state['total_failures'] += 1
        state['last_error'] = str(message or '')[:400]
        state['last_error_at'] = time.time()
        if cooldown > 0:
            until = time.time() + min(float(cooldown), 6 * 3600.0)
            # Never shorten a longer cooldown that is already in place.
            if until > state['cooldown_until']:
                state['cooldown_until'] = until
                state['cooldown_reason'] = reason or str(message or '')[:120]
                persist = True
    if persist:
        _persist()


def set_cooldown(entry_id: str, seconds: float, reason: str = '') -> None:
    state = _state(entry_id)
    with _LOCK:
        state['cooldown_until'] = time.time() + max(0.0, min(float(seconds), 6 * 3600.0))
        state['cooldown_reason'] = reason
    _persist()


def clear(entry_id: str) -> None:
    """Forget a cooldown and the failure streak (after the user fixes a key)."""
    _load()
    with _LOCK:
        _STATE[entry_id] = _blank()
    _persist()


def clear_all() -> None:
    _load()
    with _LOCK:
        for entry_id in list(_STATE):
            _STATE[entry_id] = _blank()
    _persist()


def state_of(entry_id: str) -> str:
    """'cooling', 'degraded', 'healthy' or 'unknown'."""
    state = _state(entry_id)
    with _LOCK:
        if float(state['cooldown_until']) > time.time():
            return 'cooling'
        if state['consecutive_failures'] >= _DEGRADED_AFTER:
            return 'degraded'
        if state['last_success_at']:
            return 'healthy'
    return 'unknown'


def snapshot(entry_id: str) -> dict[str, Any]:
    state = _state(entry_id)
    with _LOCK:
        copy = dict(state)
    remaining = max(0.0, float(copy['cooldown_until']) - time.time())
    return {
        'state': state_of(entry_id),
        'cooldown_remaining': round(remaining, 1),
        'cooldown_reason': copy['cooldown_reason'] if remaining > 0 else '',
        'consecutive_failures': int(copy['consecutive_failures']),
        'last_error': copy['last_error'],
        'last_error_at': copy['last_error_at'] or None,
        'last_success_at': copy['last_success_at'] or None,
        'last_latency_ms': int(copy['last_latency_ms']),
        'total_requests': int(copy['total_requests']),
        'total_failures': int(copy['total_failures']),
    }
