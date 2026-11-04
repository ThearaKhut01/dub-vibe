"""Scheduling: which pool entry should serve the next request.

Three strategies, all operating on the same candidate facts (quota headroom,
today's load, health) so switching strategy never changes *whether* an entry may
be used — only the order:

``round_robin``
    Strict rotation per provider. Every usable entry gets one request before any
    entry gets two. Best when the configurations are equivalent.

``least_used``
    Lowest request count today first, oldest last-used breaking the tie. Best
    when the configurations have different limits and should be evened out.

``hybrid`` (default)
    Score = 60·headroom + 25·success_rate + 10·weight − 20·load, each term
    normalised to 0..1. It prefers an entry that is healthy, has allowance left
    and has not just been hammered, but never excludes one: a low score only
    moves it later in the list.

An entry that is disabled, cooling down after a failure, or over a limit the
user configured is not a candidate at all, and the reason is reported so the UI
can explain an empty pool instead of silently doing nothing.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .models import DEFAULT_STRATEGY, PoolEntry, SCHEDULING_STRATEGIES


@dataclass
class Candidate:
    """One entry plus the facts that decide whether and when to use it."""

    entry: PoolEntry
    requests_today: int = 0
    requests_month: int = 0
    successes: int = 0
    failures: int = 0
    headroom: float = 1.0
    health_state: str = 'unknown'
    cooldown_remaining: float = 0.0
    quota_reason: str = ''
    last_used_at: float = 0.0

    @property
    def usable(self) -> bool:
        return self.reason == ''

    @property
    def reason(self) -> str:
        if not self.entry.enabled:
            return 'disabled'
        if self.cooldown_remaining > 0:
            return f'cooling down ({int(self.cooldown_remaining)}s)'
        if self.quota_reason:
            return self.quota_reason
        return ''

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        if total <= 0:
            return 1.0
        return max(0.0, min(1.0, self.successes / total))


class Scheduler:
    """Orders candidates; holds only the round-robin cursor."""

    def __init__(self) -> None:
        self._cursor: dict[str, int] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._cursor.clear()

    def _rotate(self, provider: str, items: list[Candidate]) -> list[Candidate]:
        if not items:
            return []
        with self._lock:
            start = self._cursor.get(provider, 0) % len(items)
            self._cursor[provider] = (start + 1) % len(items)
        return items[start:] + items[:start]

    def plan(self, candidates: list[Candidate], strategy: str = DEFAULT_STRATEGY) -> list[Candidate]:
        """Usable candidates in the order they should be tried.

        Unusable candidates are dropped; ``explain`` reports why.
        """
        usable = [candidate for candidate in candidates if candidate.usable]
        if not usable:
            return []
        strategy = strategy if strategy in SCHEDULING_STRATEGIES else DEFAULT_STRATEGY
        provider = usable[0].entry.provider
        rotated = self._rotate(provider, usable)

        if strategy == 'round_robin':
            return rotated
        if strategy == 'least_used':
            return sorted(rotated, key=lambda candidate: (
                candidate.requests_today, candidate.requests_month, -candidate.last_used_at,
            ))
        return sorted(rotated, key=lambda candidate: -self._score(candidate, usable))

    @staticmethod
    def _score(candidate: Candidate, pool: list[Candidate]) -> float:
        max_weight = max((item.entry.weight for item in pool), default=1) or 1
        max_load = max((item.requests_today for item in pool), default=0) or 1
        weight_ratio = max(0, min(candidate.entry.weight, 10)) / max_weight
        load_ratio = candidate.requests_today / max_load
        return (
            60.0 * candidate.headroom
            + 25.0 * candidate.success_rate
            + 10.0 * weight_ratio
            - 20.0 * load_ratio
        )


def explain(candidates: list[Candidate]) -> dict[str, str]:
    """entry id -> why it cannot serve right now (empty for usable entries)."""
    return {candidate.entry.id: candidate.reason for candidate in candidates if candidate.reason}


# A process-wide scheduler, so the round-robin cursor survives between requests.
_SCHEDULER = Scheduler()


def scheduler() -> Scheduler:
    return _SCHEDULER
