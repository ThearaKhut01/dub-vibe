"""Data model and persistence for the Multi-Provider API Pool Manager.

One *entry* is one API configuration: a provider, a credential, a model, and an
optional base-URL override. A pool is simply the set of entries that share a
provider, so "Gemini pool" is every entry whose provider is ``gemini``.

The pool keeps its own state under ``<app data>/api_pool``:

- ``pool.json`` — entries and settings (never a secret: credentials stay in
  Windows Credential Manager, reached through :mod:`backend.app.api_keys`)
- ``health.json`` — cooldowns, so a parked credential stays parked across a
  restart instead of being hammered again on the next launch
- ``pool.sqlite3`` — measured usage, the request journal and the translation
  cache

What this module deliberately does **not** do: invent quota numbers. Providers
do not publish a usable "remaining quota" endpoint, so a limit exists only when
the user types one in; observed usage is measured (token counts the provider
reports, otherwise the characters sent).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# Providers the app already integrates. Kept identical to api_keys.PROVIDERS so
# a pool entry can always fall back to the single stored key of the same name.
POOL_PROVIDERS: tuple[str, ...] = ('gemini', 'openai', 'codecraft', 'custom')

PROVIDER_LABELS: dict[str, str] = {
    'gemini': 'Google Gemini',
    'openai': 'OpenAI',
    'codecraft': 'CodeCraft',
    'custom': 'Custom API',
}

# How a request is scheduled across the entries of one provider.
SCHEDULING_STRATEGIES: tuple[str, ...] = ('round_robin', 'least_used', 'hybrid')
DEFAULT_STRATEGY = 'hybrid'

# Where an entry's credential lives. 'legacy' points at the single key the API
# Settings panel already stores, so importing it never duplicates a secret.
KEY_SOURCES: tuple[str, ...] = ('pool', 'legacy')

DEFAULT_SETTINGS: dict[str, Any] = {
    'strategy': DEFAULT_STRATEGY,
    'cache_enabled': True,
    'cache_ttl_days': 7,
    # Transient retries per entry (5xx/network). A 429/402/401 never retries the
    # same entry: those answers are about the credential, not this request.
    'max_attempts': 3,
    'request_timeout': 90,
    # Ceiling for parallel subtitle lines when several pool entries can serve.
    'max_parallel': 4,
}


def storage_dir() -> Path:
    """Folder holding the pool's own state.

    ``DUBVIBE_API_POOL_DIR`` overrides it — the test suite points this at a
    temporary folder so a developer's live pool never leaks into a test run.
    Resolved once per override value: this is called for every subtitle line,
    and :func:`get_storage_dir` performs a write test on each call.
    """
    return _resolve_storage(os.environ.get('DUBVIBE_API_POOL_DIR') or '')


@lru_cache(maxsize=8)
def _resolve_storage(override: str) -> Path:
    if override:
        path = Path(override).expanduser()
    else:
        from ..providers import get_storage_dir

        path = get_storage_dir('data') / 'api_pool'
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_path() -> Path:
    return storage_dir() / 'pool.json'


def health_path() -> Path:
    return storage_dir() / 'health.json'


def db_path() -> Path:
    return storage_dir() / 'pool.sqlite3'


# ---------------------------------------------------------------------------
# SQLite: one place for the connection rules the other modules share
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    entry_id    TEXT NOT NULL,
    day         TEXT NOT NULL,
    requests    INTEGER NOT NULL DEFAULT 0,
    successes   INTEGER NOT NULL DEFAULT 0,
    failures    INTEGER NOT NULL DEFAULT 0,
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    chars_in    INTEGER NOT NULL DEFAULT 0,
    chars_out   INTEGER NOT NULL DEFAULT 0,
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (entry_id, day)
);
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT NOT NULL,
    entry_id    TEXT,
    model       TEXT,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    payload     TEXT,
    result      TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    latency_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS requests_status ON requests (status, created_at);
CREATE TABLE IF NOT EXISTS cache (
    cache_key    TEXT PRIMARY KEY,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    source_text  TEXT NOT NULL,
    translation  TEXT NOT NULL,
    provider     TEXT,
    model        TEXT,
    created_at   REAL NOT NULL,
    last_hit     REAL,
    hits         INTEGER NOT NULL DEFAULT 0
);
"""

_SCHEMA_LOCK = threading.Lock()
# The database the schema was created in. Compared by path, not with a boolean:
# the test suite points each case at its own temporary folder.
_SCHEMA_PATH: str | None = None


def connect() -> sqlite3.Connection:
    """A configured connection to the pool database.

    A fresh connection per call keeps this safe across the translation thread
    pool without a shared handle; WAL lets readers work while a writer commits.
    """
    global _SCHEMA_PATH
    path = db_path()
    connection = sqlite3.connect(path, timeout=20.0)
    connection.row_factory = sqlite3.Row
    try:
        # Per-connection and cheap. WAL is set once below: it is a persistent
        # property of the file, and re-applying it on every connect asks for a
        # write lock that four translation workers would fight over.
        connection.execute('PRAGMA synchronous=NORMAL')
        connection.execute('PRAGMA busy_timeout=15000')
    except sqlite3.DatabaseError:
        pass  # a read-only or locked file still works for reads
    if _SCHEMA_PATH != str(path):
        with _SCHEMA_LOCK:
            if _SCHEMA_PATH != str(path):
                try:
                    connection.execute('PRAGMA journal_mode=WAL')
                except sqlite3.DatabaseError:
                    pass
                connection.executescript(_SCHEMA)
                connection.commit()
                _SCHEMA_PATH = str(path)
    return connection


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

@dataclass
class PoolEntry:
    """One API configuration inside a provider's pool."""

    id: str
    provider: str
    model: str
    label: str = ''
    base_url: str = ''
    enabled: bool = True
    weight: int = 1
    daily_limit: int | None = None
    monthly_limit: int | None = None
    key_source: str = 'pool'
    notes: str = ''
    created_at: float = field(default_factory=time.time)

    @property
    def display_name(self) -> str:
        base = self.label.strip() or PROVIDER_LABELS.get(self.provider, self.provider)
        return f'{base} · {self.model}' if self.model else base

    def to_record(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'provider': self.provider,
            'model': self.model,
            'label': self.label,
            'base_url': self.base_url,
            'enabled': bool(self.enabled),
            'weight': int(self.weight),
            'daily_limit': self.daily_limit,
            'monthly_limit': self.monthly_limit,
            'key_source': self.key_source,
            'notes': self.notes,
            'created_at': float(self.created_at),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> 'PoolEntry':
        provider = str(record.get('provider') or '').strip().lower()
        if provider not in POOL_PROVIDERS:
            raise ValueError(f'Unknown provider: {provider or "(empty)"}')
        key_source = str(record.get('key_source') or 'pool').strip().lower()
        if key_source not in KEY_SOURCES:
            key_source = 'pool'
        return cls(
            id=str(record.get('id') or ''),
            provider=provider,
            model=str(record.get('model') or '').strip(),
            label=str(record.get('label') or '').strip(),
            base_url=str(record.get('base_url') or '').strip(),
            enabled=bool(record.get('enabled', True)),
            weight=max(0, min(int(record.get('weight') or 1), 10)),
            daily_limit=_optional_count(record.get('daily_limit')),
            monthly_limit=_optional_count(record.get('monthly_limit')),
            key_source=key_source,
            notes=str(record.get('notes') or ''),
            created_at=float(record.get('created_at') or time.time()),
        )

    def to_public(self, *, has_key: bool, key_hint: str = '') -> dict[str, Any]:
        """Browser-safe view: never the credential itself, only whether one exists."""
        return {
            'id': self.id,
            'provider': self.provider,
            'provider_label': PROVIDER_LABELS.get(self.provider, self.provider),
            'model': self.model,
            'label': self.label,
            'display_name': self.display_name,
            'base_url': self.base_url,
            'enabled': self.enabled,
            'weight': self.weight,
            'daily_limit': self.daily_limit,
            'monthly_limit': self.monthly_limit,
            'key_source': self.key_source,
            'has_key': has_key,
            'key_hint': key_hint,
            'notes': self.notes,
            'created_at': self.created_at,
        }


def _optional_count(value: Any) -> int | None:
    """A non-negative integer limit, or None for "no limit configured"."""
    if value is None or value == '':
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def validate_base_url(provider: str, url: str) -> str:
    """Normalise an entry's base URL.

    OpenAI-compatible providers use ``custom_api.normalize_base_url`` (the same
    rule the Custom API panel applies, so a bare host gets ``/v1``). Gemini's
    REST root is already versioned, so it is only trimmed and checked.
    """
    url = str(url or '').strip().rstrip('/')
    if not url:
        return ''
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('Base URL must start with http:// or https:// and include a host.')
    if provider == 'gemini':
        return url
    from ..custom_api import normalize_base_url

    return normalize_base_url(url)


# ---------------------------------------------------------------------------
# State store (entries + settings)
# ---------------------------------------------------------------------------

class PoolStore:
    """Thread-safe access to ``pool.json``.

    Reads are served from memory and revalidated against the file's mtime, so a
    second process (a dev server next to the desktop app) sees the same pool.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: list[PoolEntry] = []
        self._settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
        self._mtime: float | None = None
        self._path = ''
        self._loaded = False

    # -- reading --------------------------------------------------------
    def _load(self, force: bool = False) -> None:
        with self._lock:
            current_path = str(state_path())
            try:
                mtime = state_path().stat().st_mtime
            except OSError:
                mtime = None
            same_file = self._loaded and current_path == self._path
            if same_file and not force and mtime == self._mtime:
                return
            entries: list[PoolEntry] = []
            settings = dict(DEFAULT_SETTINGS)
            raw: dict[str, Any] = {}
            if mtime is not None:
                try:
                    parsed = json.loads(state_path().read_text(encoding='utf-8'))
                    if isinstance(parsed, dict):
                        raw = parsed
                except (OSError, ValueError):
                    raw = {}  # a corrupt file must not take the app down
            for record in raw.get('entries', []) or []:
                if not isinstance(record, dict):
                    continue
                try:
                    entries.append(PoolEntry.from_record(record))
                except ValueError:
                    continue
            settings.update(_clean_settings(raw.get('settings')))
            self._entries = entries
            self._settings = settings
            self._mtime = mtime
            self._path = current_path
            self._loaded = True

    def _save(self) -> None:
        """Write the state atomically, so a crash never leaves half a file."""
        with self._lock:
            payload = {
                'version': 1,
                'entries': [entry.to_record() for entry in self._entries],
                'settings': self._settings,
            }
            target = state_path()
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
            os.replace(temporary, target)
            self._mtime = target.stat().st_mtime
            self._path = str(target)
            self._loaded = True

    def entries(self, provider: str | None = None) -> list[PoolEntry]:
        self._load()
        with self._lock:
            items = list(self._entries)
        if provider is None:
            return items
        return [entry for entry in items if entry.provider == provider]

    def settings(self) -> dict[str, Any]:
        self._load()
        with self._lock:
            return dict(self._settings)

    def get(self, entry_id: str) -> PoolEntry | None:
        for entry in self.entries():
            if entry.id == entry_id:
                return entry
        return None

    # -- writing --------------------------------------------------------
    def upsert(self, entry: PoolEntry) -> PoolEntry:
        with self._lock:
            self._load()
            for index, existing in enumerate(self._entries):
                if existing.id == entry.id:
                    # created_at belongs to the first save, not the latest edit.
                    stored = replace(entry, created_at=existing.created_at)
                    self._entries[index] = stored
                    self._save()
                    return stored
            self._entries.append(entry)
            self._save()
            return entry

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            self._load()
            remaining = [entry for entry in self._entries if entry.id != entry_id]
            if len(remaining) == len(self._entries):
                return False
            self._entries = remaining
            self._save()
            return True

    def save_settings(self, update: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._load()
            self._settings.update(_clean_settings(update))
            self._save()
            return dict(self._settings)

    def reload(self) -> None:
        self._load(force=True)


def _clean_settings(update: Any) -> dict[str, Any]:
    """Keep only known settings with sane values."""
    if not isinstance(update, dict):
        return {}
    clean: dict[str, Any] = {}
    strategy = str(update.get('strategy') or '').strip().lower()
    if strategy in SCHEDULING_STRATEGIES:
        clean['strategy'] = strategy
    if 'cache_enabled' in update:
        clean['cache_enabled'] = bool(update['cache_enabled'])
    if 'cache_ttl_days' in update:
        try:
            clean['cache_ttl_days'] = max(1, min(int(update['cache_ttl_days']), 365))
        except (TypeError, ValueError):
            pass
    if 'max_attempts' in update:
        try:
            clean['max_attempts'] = max(1, min(int(update['max_attempts']), 6))
        except (TypeError, ValueError):
            pass
    if 'request_timeout' in update:
        try:
            clean['request_timeout'] = max(10, min(int(update['request_timeout']), 300))
        except (TypeError, ValueError):
            pass
    if 'max_parallel' in update:
        try:
            clean['max_parallel'] = max(1, min(int(update['max_parallel']), 8))
        except (TypeError, ValueError):
            pass
    return clean


_STORE = PoolStore()


def store() -> PoolStore:
    return _STORE


def new_entry_id() -> str:
    return uuid.uuid4().hex[:12]


def provider_entries(provider: str) -> list[PoolEntry]:
    return store().entries(provider)


def all_entries() -> list[PoolEntry]:
    return store().entries()


def enabled_entries(provider: str) -> list[PoolEntry]:
    return [entry for entry in provider_entries(provider) if entry.enabled]


def iter_providers() -> Iterable[str]:
    return POOL_PROVIDERS
