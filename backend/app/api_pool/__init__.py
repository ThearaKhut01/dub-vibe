"""Multi-Provider API Pool Manager.

Public surface used by the rest of the app:

- :func:`has_entries` / :func:`is_active` — does this engine have pool
  configurations (the provider classes route on this)
- :func:`chat` — one completion, scheduled and with failover
- :func:`status` / :func:`upsert_entry` / :func:`delete_entry` /
  :func:`test_entry` / :func:`save_settings` — the API Pool panel
- :func:`cached_translation` / :func:`remember_translation` — translation cache

The pieces are separate modules on purpose: scheduling, failover, quota,
health, the request journal and the cache each answer one question, and the
manager only wires them together.
"""
from __future__ import annotations

from .failover import Failure
from .models import (
    POOL_PROVIDERS,
    PROVIDER_LABELS,
    SCHEDULING_STRATEGIES,
    PoolEntry,
)
from .pool_manager import (
    PoolUnavailable,
    ProviderCallError,
    cache_enabled,
    cache_enabled_for,
    cached_translation,
    chat,
    clear_cache,
    custom_endpoints_are_local,
    delete_entry,
    entry_count,
    entry_view,
    has_entries,
    import_legacy,
    is_active,
    plan,
    remember_translation,
    reset_for_tests,
    reset_health,
    retry_failed,
    save_settings,
    settings,
    startup,
    status,
    test_entry,
    upsert_entry,
    usable_count,
)

__all__ = [
    'POOL_PROVIDERS', 'PROVIDER_LABELS', 'SCHEDULING_STRATEGIES', 'PoolEntry',
    'Failure', 'PoolUnavailable', 'ProviderCallError',
    'cache_enabled', 'cache_enabled_for', 'cached_translation', 'chat',
    'clear_cache', 'custom_endpoints_are_local', 'delete_entry', 'entry_count', 'entry_view', 'has_entries', 'import_legacy',
    'is_active', 'plan', 'remember_translation', 'reset_for_tests',
    'reset_health', 'retry_failed', 'save_settings', 'settings', 'startup', 'status',
    'test_entry', 'upsert_entry', 'usable_count',
]
