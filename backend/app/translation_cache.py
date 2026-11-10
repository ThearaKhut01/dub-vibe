"""SQLite cache for validated subtitle translations.

Only **validated** translations are stored: the caller saves a line after the
script and length checks in :mod:`backend.app.providers` accepted it, so a wrong
script or a reasoning dump can never be replayed from here.

The key is built by :mod:`backend.app.translation_memory` (prompt version, model,
language pair, glossary digest and the exact source line), so switching engines
or editing the glossary cannot replay another model's wording.

The file lives next to the project database (``<app data>/translation-cache.sqlite3``);
delete it to forget every remembered line. ``DUBVIBE_TRANSLATION_CACHE=0`` turns
the feature off without touching the file.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .providers import get_storage_dir

# Bump when the translation prompt changes, so old answers are not reused with
# instructions they were never written for.
PROMPT_VERSION = 'v2'

_WRITE_LOCK = threading.Lock()
_READY: set[str] = set()


def _database_path() -> Path:
    override = os.environ.get('DUBVIBE_TRANSLATION_CACHE_DIR')
    folder = Path(override).expanduser() if override else get_storage_dir('data')
    return folder / 'translation-cache.sqlite3'


def connect() -> sqlite3.Connection:
    """One connection to the cache database, schema created on first use."""
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    key = str(path)
    if key not in _READY:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS cache ('
            ' cache_key TEXT PRIMARY KEY,'
            ' source_lang TEXT NOT NULL,'
            ' target_lang TEXT NOT NULL,'
            ' source_text TEXT NOT NULL,'
            ' translation TEXT NOT NULL,'
            ' provider TEXT NOT NULL DEFAULT \'\','
            ' model TEXT NOT NULL DEFAULT \'\','
            ' created_at REAL NOT NULL,'
            ' last_hit REAL,'
            ' hits INTEGER NOT NULL DEFAULT 0)'
        )
        connection.commit()
        _READY.add(key)
    return connection


def _normalise(text: str) -> str:
    return ' '.join(str(text or '').split())


def _language(value: str) -> str:
    try:
        from .providers import _resolve_lang_code

        return _resolve_lang_code(str(value or ''))
    except Exception:
        return str(value or '').strip().lower()


def make_key(text: str, source_lang: str, target_lang: str, style_hint: str = '') -> str:
    """Key for the plain (model-less) cache used by older callers."""
    payload = '\x1f'.join((
        PROMPT_VERSION,
        _language(source_lang),
        _language(target_lang),
        _normalise(style_hint),
        _normalise(text),
    ))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def lookup(cache_key: str, ttl_days: int = 7) -> str | None:
    """The cached translation, or None when absent or older than the TTL."""
    connection = connect()
    try:
        row = connection.execute(
            'SELECT translation, created_at FROM cache WHERE cache_key = ?', (cache_key,)
        ).fetchone()
        if row is None:
            return None
        if ttl_days > 0 and float(row['created_at']) < time.time() - ttl_days * 86400:
            return None
        with _WRITE_LOCK:
            connection.execute(
                'UPDATE cache SET hits = hits + 1, last_hit = ? WHERE cache_key = ?',
                (time.time(), cache_key),
            )
            connection.commit()
        return str(row['translation'])
    finally:
        connection.close()


def store(
    cache_key: str,
    *,
    translation: str,
    source_text: str,
    source_lang: str,
    target_lang: str,
    provider: str = '',
    model: str = '',
) -> None:
    translation = str(translation or '').strip()
    if not cache_key or not translation:
        return
    now = time.time()
    with _WRITE_LOCK:
        connection = connect()
        try:
            connection.execute(
                """
                INSERT INTO cache (cache_key, source_lang, target_lang, source_text,
                                   translation, provider, model, created_at, last_hit, hits)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    translation = excluded.translation,
                    provider    = excluded.provider,
                    model       = excluded.model,
                    created_at  = excluded.created_at
                """,
                (
                    cache_key, _language(source_lang), _language(target_lang),
                    _normalise(source_text), translation, provider, model, now,
                ),
            )
            connection.commit()
        finally:
            connection.close()


def stats() -> dict[str, Any]:
    connection = connect()
    try:
        row = connection.execute(
            'SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits, '
            'MAX(created_at) AS newest FROM cache'
        ).fetchone()
    finally:
        connection.close()
    return {
        'entries': int(row['entries'] or 0) if row else 0,
        'hits': int(row['hits'] or 0) if row else 0,
        'newest_at': float(row['newest']) if row and row['newest'] else None,
    }


def clear() -> int:
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute('DELETE FROM cache')
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()


def prune(ttl_days: int) -> int:
    """Drop entries older than the TTL (called on startup)."""
    if ttl_days <= 0:
        return 0
    cutoff = time.time() - ttl_days * 86400
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute('DELETE FROM cache WHERE created_at < ?', (cutoff,))
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()
