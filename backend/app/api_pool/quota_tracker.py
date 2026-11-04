"""Measured usage and user-configured quota limits.

The pool can only be honest about quota if it separates two things:

* **Measured usage** — requests sent through the pool, successes/failures, and
  the tokens the provider itself reported (Gemini answers ``usageMetadata``,
  OpenAI-compatible endpoints answer ``usage``). When a provider reports no
  token counts, the characters sent are recorded instead, and the UI says so.
* **Limits** — only what the user typed into an entry (daily / monthly
  requests). Providers do not expose a reliable "remaining quota" endpoint, so
  nothing here is guessed, and an entry without a limit is simply unlimited.

Days are UTC, matching how provider quotas roll over and how the API reports
usage timestamps.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
import threading
import time
from typing import Any, Iterable

from .models import PoolEntry, connect

# Writes are short; serialising them avoids lock contention ("database is
# locked") between four translation workers and the UI polling status.
_WRITE_LOCK = threading.Lock()


def utc_day(moment: float | None = None) -> str:
    return time.strftime('%Y-%m-%d', time.gmtime(moment if moment is not None else time.time()))


def utc_month(moment: float | None = None) -> str:
    return utc_day(moment)[:7]


def record(
    entry_id: str,
    *,
    ok: bool,
    tokens_in: int = 0,
    tokens_out: int = 0,
    chars_in: int = 0,
    chars_out: int = 0,
    latency_ms: int = 0,
) -> None:
    """Add one request to this entry's usage for today.

    Failures still count as requests: the provider received (and billed) a call.
    Token counts are stored only when the provider reported them.
    """
    if not entry_id:
        return
    day = utc_day()
    with _WRITE_LOCK:
        connection = connect()
        try:
            connection.execute(
                """
                INSERT INTO usage (entry_id, day, requests, successes, failures,
                                   tokens_in, tokens_out, chars_in, chars_out, latency_ms)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id, day) DO UPDATE SET
                    requests   = requests + 1,
                    successes  = successes + excluded.successes,
                    failures   = failures + excluded.failures,
                    tokens_in  = tokens_in + excluded.tokens_in,
                    tokens_out = tokens_out + excluded.tokens_out,
                    chars_in   = chars_in + excluded.chars_in,
                    chars_out  = chars_out + excluded.chars_out,
                    latency_ms = latency_ms + excluded.latency_ms
                """,
                (
                    entry_id, day,
                    1 if ok else 0, 0 if ok else 1,
                    max(0, int(tokens_in or 0)), max(0, int(tokens_out or 0)),
                    max(0, int(chars_in or 0)), max(0, int(chars_out or 0)),
                    max(0, int(latency_ms or 0)),
                ),
            )
            connection.commit()
        finally:
            connection.close()


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {
            'requests': 0, 'successes': 0, 'failures': 0,
            'tokens_in': 0, 'tokens_out': 0, 'chars_in': 0, 'chars_out': 0,
            'latency_ms': 0,
        }
    return {
        'requests': int(row['requests'] or 0),
        'successes': int(row['successes'] or 0),
        'failures': int(row['failures'] or 0),
        'tokens_in': int(row['tokens_in'] or 0),
        'tokens_out': int(row['tokens_out'] or 0),
        'chars_in': int(row['chars_in'] or 0),
        'chars_out': int(row['chars_out'] or 0),
        'latency_ms': int(row['latency_ms'] or 0),
    }


def usage_for_day(entry_id: str, day: str | None = None) -> dict[str, Any]:
    day = day or utc_day()
    connection = connect()
    try:
        row = connection.execute(
            'SELECT * FROM usage WHERE entry_id = ? AND day = ?', (entry_id, day)
        ).fetchone()
    finally:
        connection.close()
    return _row_dict(row)


def usage_for_month(entry_id: str, month: str | None = None) -> dict[str, Any]:
    """Totals for a UTC month, summed across its days."""
    month = month or utc_month()
    connection = connect()
    try:
        row = connection.execute(
            """
            SELECT SUM(requests) AS requests, SUM(successes) AS successes,
                   SUM(failures) AS failures, SUM(tokens_in) AS tokens_in,
                   SUM(tokens_out) AS tokens_out, SUM(chars_in) AS chars_in,
                   SUM(chars_out) AS chars_out, SUM(latency_ms) AS latency_ms
            FROM usage WHERE entry_id = ? AND substr(day, 1, 7) = ?
            """,
            (entry_id, month),
        ).fetchone()
    finally:
        connection.close()
    return _row_dict(row)


def usage_map(entry_ids: Iterable[str], *, scope: str = 'day') -> dict[str, dict[str, Any]]:
    """Today's (or this month's) usage for several entries in one query."""
    ids = [entry_id for entry_id in entry_ids if entry_id]
    if not ids:
        return {}
    placeholders = ','.join('?' for _ in ids)
    if scope == 'month':
        column, window = 'substr(day, 1, 7)', utc_month()
    else:
        column, window = 'day', utc_day()
    connection = connect()
    try:
        rows = connection.execute(
            f"""
            SELECT entry_id, SUM(requests) AS requests, SUM(successes) AS successes,
                   SUM(failures) AS failures, SUM(tokens_in) AS tokens_in,
                   SUM(tokens_out) AS tokens_out, SUM(chars_in) AS chars_in,
                   SUM(chars_out) AS chars_out, SUM(latency_ms) AS latency_ms
            FROM usage WHERE entry_id IN ({placeholders}) AND {column} = ?
            GROUP BY entry_id
            """,
            (*ids, window),
        ).fetchall()
    finally:
        connection.close()
    return {str(row['entry_id']): _row_dict(row) for row in rows}


def daily_series(entry_ids: Iterable[str], days: int = 7) -> list[dict[str, Any]]:
    """Per-day totals for the last ``days`` UTC days (oldest first, zero-filled)."""
    ids = [entry_id for entry_id in entry_ids if entry_id]
    days = max(1, min(int(days), 90))
    today = _dt.datetime.now(_dt.timezone.utc).date()
    dates = [(today - _dt.timedelta(days=offset)).isoformat() for offset in range(days - 1, -1, -1)]
    totals = {day: {'day': day, 'requests': 0, 'failures': 0, 'tokens_in': 0, 'tokens_out': 0} for day in dates}
    if ids:
        placeholders = ','.join('?' for _ in ids)
        connection = connect()
        try:
            rows = connection.execute(
                f"""
                SELECT day, SUM(requests) AS requests, SUM(failures) AS failures,
                       SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out
                FROM usage
                WHERE entry_id IN ({placeholders}) AND day >= ?
                GROUP BY day ORDER BY day
                """,
                (*ids, dates[0]),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            day = str(row['day'])
            if day in totals:
                totals[day].update(
                    requests=int(row['requests'] or 0),
                    failures=int(row['failures'] or 0),
                    tokens_in=int(row['tokens_in'] or 0),
                    tokens_out=int(row['tokens_out'] or 0),
                )
    return [totals[day] for day in dates]


def limit_status(
    entry: PoolEntry,
    today: dict[str, Any] | None = None,
    month: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Whether an entry may be used now, given its configured limits.

    ``remaining`` is None when no limit is configured — that is "unknown", not
    "unlimited because the provider said so", and the UI labels it that way.
    """
    today = today if today is not None else usage_for_day(entry.id)
    month = month if month is not None else usage_for_month(entry.id)
    used_today = int(today.get('requests', 0))
    used_month = int(month.get('requests', 0))
    blocked = None
    if entry.daily_limit is not None and used_today >= entry.daily_limit:
        blocked = f'daily limit reached ({used_today}/{entry.daily_limit})'
    elif entry.monthly_limit is not None and used_month >= entry.monthly_limit:
        blocked = f'monthly limit reached ({used_month}/{entry.monthly_limit})'
    return {
        'requests_today': used_today,
        'requests_month': used_month,
        'daily_limit': entry.daily_limit,
        'monthly_limit': entry.monthly_limit,
        'daily_remaining': None if entry.daily_limit is None else max(0, entry.daily_limit - used_today),
        'monthly_remaining': None if entry.monthly_limit is None else max(0, entry.monthly_limit - used_month),
        'allowed': blocked is None,
        'reason': blocked,
    }


def headroom(entry: PoolEntry, today: dict[str, Any] | None = None) -> float:
    """Fraction of the configured daily allowance still available (1.0 = unset).

    Used by the hybrid scheduler; an entry with no limit scores full marks
    because nothing is known to be exhausted.
    """
    if entry.daily_limit is None:
        return 1.0
    used = int((today or usage_for_day(entry.id)).get('requests', 0))
    if entry.daily_limit <= 0:
        return 0.0
    return max(0.0, min(1.0, (entry.daily_limit - used) / entry.daily_limit))


def reset(entry_id: str) -> int:
    """Forget recorded usage for one entry (after the user tops a key up)."""
    with _WRITE_LOCK:
        connection = connect()
        try:
            cursor = connection.execute('DELETE FROM usage WHERE entry_id = ?', (entry_id,))
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()


def totals(entry_ids: Iterable[str]) -> dict[str, Any]:
    """Today's and this month's pooled totals, for the status panel."""
    ids = [entry_id for entry_id in entry_ids if entry_id]
    today_rows = usage_map(ids, scope='day')
    month_rows = usage_map(ids, scope='month')

    def _sum(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
        keys = ('requests', 'successes', 'failures', 'tokens_in', 'tokens_out', 'chars_in', 'chars_out')
        return {key: sum(int(row.get(key, 0)) for row in rows.values()) for key in keys}

    return {'today': _sum(today_rows), 'month': _sum(month_rows)}
