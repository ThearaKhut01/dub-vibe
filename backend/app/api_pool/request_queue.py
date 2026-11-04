"""Persistent request journal and retry queue.

Every call the pool makes is written to SQLite *before* it leaves the machine and
updated when it returns. That single table is what makes the rest honest:

- **Persistent queue** — rows sit in ``pending``/``in_flight`` while a request is
  out. A crash or a killed process leaves them behind instead of pretending the
  line was translated; :func:`recover_stalled` turns them into ``failed`` on the
  next start so the UI can offer a retry instead of silently losing them.
- **Request history** — provider, entry, model, attempts, latency and the error
  for every call, newest first.
- **Retry management** — :func:`retry` and :func:`retry_all_failed` put failed
  rows back in ``pending`` with the original payload, ready to be re-sent.

The payload never contains a credential: the pool resolves the entry's key at
call time through :mod:`backend.app.api_keys`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from .models import connect

_WRITE_LOCK = threading.Lock()

PENDING = 'pending'
IN_FLIGHT = 'in_flight'
DONE = 'done'
FAILED = 'failed'


def _row(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    if record.get('payload'):
        try:
            record['payload'] = json.loads(record['payload'])
        except ValueError:
            record['payload'] = None
    return record


def enqueue(payload: dict[str, Any], provider: str, *, entry_id: str = '', model: str = '') -> int:
    """Journal a request before sending it. Returns the row id."""
    now = time.time()
    serialised = json.dumps(payload, ensure_ascii=False)
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO requests (provider, entry_id, model, status, attempts,
                                      error, payload, result, created_at, updated_at, latency_ms)
                VALUES (?, ?, ?, ?, 0, NULL, ?, NULL, ?, ?, NULL)
                """,
                (provider, entry_id or None, model or None, PENDING, serialised, now, now),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()


def mark_in_flight(request_id: int, *, entry_id: str = '', model: str = '') -> None:
    if not request_id:
        return
    with _WRITE_LOCK:
        connection = connect()
        try:
            connection.execute(
                """
                UPDATE requests SET status = ?, updated_at = ?,
                       attempts = attempts + 1,
                       entry_id = COALESCE(?, entry_id),
                       model = COALESCE(?, model)
                WHERE id = ?
                """,
                (IN_FLIGHT, time.time(), entry_id or None, model or None, request_id),
            )
            connection.commit()
        finally:
            connection.close()


def finish(request_id: int, *, entry_id: str, model: str, result: str, latency_ms: int) -> None:
    if not request_id:
        return
    with _WRITE_LOCK:
        connection = connect()
        try:
            connection.execute(
                """
                UPDATE requests SET status = ?, entry_id = COALESCE(?, entry_id),
                       model = COALESCE(?, model), result = ?, error = NULL,
                       latency_ms = ?, updated_at = ?
                WHERE id = ?
                """,
                (DONE, entry_id or None, model or None, str(result or '')[:4000],
                 int(latency_ms or 0), time.time(), request_id),
            )
            connection.commit()
        finally:
            connection.close()


def fail(request_id: int, *, entry_id: str, model: str, error: str, attempts: int) -> None:
    if not request_id:
        return
    with _WRITE_LOCK:
        connection = connect()
        try:
            connection.execute(
                """
                UPDATE requests SET status = ?, entry_id = COALESCE(?, entry_id),
                       model = COALESCE(?, model), error = ?, attempts = MAX(attempts, ?),
                       updated_at = ?
                WHERE id = ?
                """,
                (FAILED, entry_id or None, model or None, str(error or '')[:800],
                 int(attempts or 0), time.time(), request_id),
            )
            connection.commit()
        finally:
            connection.close()


def recover_stalled() -> int:
    """Mark requests left in flight by a crash as failed (and retryable)."""
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute(
                """
                UPDATE requests SET status = ?, error = ?, updated_at = ?
                WHERE status IN (?, ?)
                """,
                (
                    FAILED, 'interrupted before a response was recorded (app restart)',
                    time.time(), IN_FLIGHT, PENDING,
                ),
            )
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()


def stats() -> dict[str, Any]:
    connection = connect()
    try:
        rows = connection.execute(
            'SELECT status, COUNT(*) AS total FROM requests GROUP BY status'
        ).fetchall()
        oldest = connection.execute(
            'SELECT MIN(created_at) AS oldest FROM requests WHERE status = ?', (FAILED,)
        ).fetchone()
        recent = connection.execute(
            'SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0) AS ok '
            'FROM requests WHERE created_at >= ?',
            (DONE, time.time() - 86400),
        ).fetchone()
    finally:
        connection.close()
    counts = {str(row['status']): int(row['total']) for row in rows}
    return {
        'pending': counts.get(PENDING, 0),
        'in_flight': counts.get(IN_FLIGHT, 0),
        'done': counts.get(DONE, 0),
        'failed': counts.get(FAILED, 0),
        'total': sum(counts.values()),
        'oldest_failed_at': float(oldest['oldest']) if oldest and oldest['oldest'] else None,
        'last_24h': {'total': int(recent['total'] or 0), 'ok': int(recent['ok'] or 0)} if recent else {},
    }


def failed(limit: int = 20) -> list[dict[str, Any]]:
    connection = connect()
    try:
        rows = connection.execute(
            'SELECT * FROM requests WHERE status = ? ORDER BY updated_at DESC LIMIT ?',
            (FAILED, max(1, min(int(limit), 200))),
        ).fetchall()
    finally:
        connection.close()
    return [_row(row) for row in rows]


def history(limit: int = 50) -> list[dict[str, Any]]:
    connection = connect()
    try:
        rows = connection.execute(
            'SELECT id, provider, entry_id, model, status, attempts, error, latency_ms, '
            'created_at, updated_at FROM requests ORDER BY id DESC LIMIT ?',
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    finally:
        connection.close()
    return [_row(row) for row in rows]


def retry(request_id: int) -> bool:
    """Put one failed request back in the queue. Returns False if it was not failed."""
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute(
                'UPDATE requests SET status = ?, error = NULL, updated_at = ? '
                'WHERE id = ? AND status = ?',
                (PENDING, time.time(), request_id, FAILED),
            )
            connection.commit()
            return bool(cursor.rowcount)
        finally:
            connection.close()


def retry_all_failed() -> int:
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute(
                'UPDATE requests SET status = ?, error = NULL, updated_at = ? WHERE status = ?',
                (PENDING, time.time(), FAILED),
            )
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()


def pending(limit: int = 50) -> list[dict[str, Any]]:
    """Queued payloads, oldest first — what a drain would send."""
    connection = connect()
    try:
        rows = connection.execute(
            'SELECT * FROM requests WHERE status = ? ORDER BY id ASC LIMIT ?',
            (PENDING, max(1, min(int(limit), 200))),
        ).fetchall()
    finally:
        connection.close()
    return [_row(row) for row in rows]


def prune(keep: int = 5000) -> int:
    """Keep the journal bounded; the newest ``keep`` rows survive."""
    keep = max(100, int(keep))
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute(
                """
                DELETE FROM requests
                WHERE id <= (SELECT COALESCE(MAX(id), 0) - ? FROM requests)
                """,
                (keep,),
            )
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()
