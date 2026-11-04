"""Failure classification, retry policy and the failover loop.

Three questions decide what happens after a failed call:

1. **Is this request worth retrying on the same entry?** A 5xx or a dropped
   connection is; a 429, a 402, a 401 or a retired model name is not — those are
   answers about the credential or the model, and repeating the call wastes the
   user's time (and sometimes their money).
2. **How long should this entry stay parked?** A 429 uses the provider's own
   ``Retry-After`` when it sends one; nothing is guessed when it does not.
3. **Is another entry worth trying?** Always, unless the request itself was
   rejected (400/422): a malformed prompt fails the same way everywhere, so the
   caller gets the error instead of a dozen identical rejections.
"""
from __future__ import annotations

import random
import time
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Statuses that describe the credential or the account, not this request.
AUTH_STATUS = frozenset({401, 403})
QUOTA_STATUS = frozenset({402})
RATE_LIMIT_STATUS = frozenset({429})
TRANSIENT_STATUS = frozenset({408, 425, 500, 502, 503, 504})

DEFAULT_COOLDOWNS: dict[str, float] = {
    'auth': 900.0,      # bad/revoked key: a retry loop must not burn the run
    'quota': 600.0,     # no allowance / billing: it will not fix itself mid-run
    'rate_limit': 60.0,  # only when the provider sends no Retry-After
    'model': 600.0,     # retired model name: do not re-ask for it this session
    'transient': 20.0,
    'network': 60.0,
    'request': 0.0,     # a 400/422 is about the request, not the endpoint
    'unknown': 30.0,
}


@dataclass(frozen=True)
class Failure:
    """What went wrong, and what it means for this entry and this request."""

    kind: str
    message: str
    status: int | None = None
    retry_same_entry: bool = False
    cooldown: float = 0.0
    park_reason: str = ''


def parse_retry_after(headers: Any) -> float | None:
    """Seconds from a ``Retry-After`` header, when the provider sent one."""
    if headers is None:
        return None
    try:
        raw = headers.get('Retry-After') or headers.get('retry-after')
    except AttributeError:
        return None
    if not raw:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, min(seconds, 3600.0))


def retry_delay(attempt: int, *, base: float = 0.7, cap: float = 6.0) -> float:
    """Exponential backoff with jitter (attempt is zero-based)."""
    delay = min(cap, base * (2 ** max(0, attempt)))
    return delay + random.uniform(0.0, 0.25)


def classify(error: BaseException) -> Failure:
    """Turn an exception into a :class:`Failure` decision."""
    status = getattr(error, 'status', None)
    message = str(error) or error.__class__.__name__

    if isinstance(error, urllib.error.HTTPError):
        status = error.code
        headers = error.headers
        detail = ''
        try:
            detail = error.read().decode('utf-8', errors='replace')
        except Exception:
            detail = ''
        message = f'HTTP {error.code}' + (f': {detail[:200]}' if detail.strip() else '')
    else:
        headers = getattr(error, 'headers', None)

    if status is None:
        # Transport-level problems carry no HTTP status.
        if isinstance(error, (TimeoutError, OSError)):
            return Failure(
                kind='network',
                message=message,
                retry_same_entry=True,
                cooldown=DEFAULT_COOLDOWNS['network'],
                park_reason='unreachable',
            )
        kind = str(getattr(error, 'kind', 'unknown') or 'unknown')
        if kind in DEFAULT_COOLDOWNS:
            return Failure(
                kind=kind,
                message=message,
                retry_same_entry=kind in ('transient', 'network'),
                cooldown=DEFAULT_COOLDOWNS[kind],
                park_reason=kind,
            )
        return Failure(kind='unknown', message=message, cooldown=DEFAULT_COOLDOWNS['unknown'])

    if status in AUTH_STATUS:
        return Failure('auth', message, status, False, DEFAULT_COOLDOWNS['auth'], 'credential rejected')
    if status in QUOTA_STATUS:
        return Failure('quota', message, status, False, DEFAULT_COOLDOWNS['quota'], 'no remaining allowance')
    if status in RATE_LIMIT_STATUS:
        retry_after = getattr(error, 'retry_after', None)
        if retry_after is None:
            retry_after = parse_retry_after(headers)
        cooldown = float(retry_after) if retry_after is not None else DEFAULT_COOLDOWNS['rate_limit']
        return Failure('rate_limit', message, status, False, cooldown, 'rate limited')
    if status == 404:
        return Failure('model', message, status, False, DEFAULT_COOLDOWNS['model'], 'model not available')
    if status in TRANSIENT_STATUS:
        return Failure('transient', message, status, True, DEFAULT_COOLDOWNS['transient'], 'provider error')
    if 400 <= status < 500:
        return Failure('request', message, status, False, 0.0, '')
    return Failure('unknown', message, status, False, DEFAULT_COOLDOWNS['unknown'])


@dataclass
class AttemptRecord:
    entry_id: str
    ok: bool
    error: str = ''
    kind: str = ''
    latency_ms: int = 0


@dataclass
class Outcome:
    """Result of walking the candidate list."""

    text: str | None = None
    entry: Any | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    attempts: int = 0
    error: BaseException | None = None
    history: list[AttemptRecord] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.text is not None


def execute(
    candidates: Sequence[Any],
    call: Callable[[Any], tuple[str, dict[str, Any]]],
    *,
    max_attempts: int = 3,
    on_success: Callable[[Any, dict[str, Any], int], None] | None = None,
    on_failure: Callable[[Any, Failure], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Outcome:
    """Try each candidate until one answers.

    ``call`` returns ``(text, usage)`` or raises. ``on_success`` / ``on_failure``
    receive the entry so the caller can record usage and health without this
    module knowing anything about storage. ``sleep`` is injectable so a retry
    path can be tested without waiting.
    """
    sleep = sleep or time.sleep
    history: list[AttemptRecord] = []
    last_error: BaseException | None = None
    max_attempts = max(1, int(max_attempts))

    for entry in candidates:
        entry_id = getattr(entry, 'id', '')
        tries = 0
        while True:
            tries += 1
            started = time.perf_counter()
            try:
                text, usage = call(entry)
            except Exception as error:  # noqa: BLE001 - every failure is classified
                latency_ms = int((time.perf_counter() - started) * 1000)
                failure = classify(error)
                history.append(AttemptRecord(
                    entry_id=entry_id, ok=False, error=failure.message,
                    kind=failure.kind, latency_ms=latency_ms,
                ))
                last_error = error
                if on_failure is not None:
                    on_failure(entry, failure)
                if failure.retry_same_entry and tries < max_attempts:
                    sleep(retry_delay(tries - 1))
                    continue
                break
            latency_ms = int((time.perf_counter() - started) * 1000)
            history.append(AttemptRecord(entry_id=entry_id, ok=True, latency_ms=latency_ms))
            if on_success is not None:
                on_success(entry, usage or {}, latency_ms)
            return Outcome(
                text=text, entry=entry, usage=usage or {}, latency_ms=latency_ms,
                attempts=len(history), error=None, history=history,
            )

    return Outcome(
        text=None, entry=None, usage={}, latency_ms=0, attempts=len(history),
        error=last_error, history=history,
    )
