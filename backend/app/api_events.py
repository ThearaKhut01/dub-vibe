"""Live view of API activity: node registry, counters and an event bus.

One *node* is one credential that can serve a request — an API-pool entry, or the
single key stored in API Settings ("legacy:<provider>"). Everything here is in
memory and resets on restart: the point is to watch a run happen, not to keep a
ledger (the pool's own usage table is the ledger).

No key material ever enters an event. A node only carries the last four
characters (`••••ZGvo`), and every error string is passed through
:func:`sanitize_error`, which strips URLs, `?key=` query values and anything that
looks like a token.
"""
from __future__ import annotations

import itertools
import re
import threading
import time
from collections import deque
from typing import Any, Callable

MAX_LOG = 50
RATE_WINDOW_SECONDS = 60.0
MAX_RATE_SAMPLES = 4000

_lock = threading.Lock()
_nodes: dict[str, dict[str, Any]] = {}
_recent: deque[dict[str, Any]] = deque(maxlen=MAX_LOG)
_stamps: deque[float] = deque(maxlen=MAX_RATE_SAMPLES)
_ids = itertools.count(1)
_publisher: Callable[[dict[str, Any]], None] | None = None

# Anything that looks like a credential: Google keys, OpenAI-style keys, the
# app's own CodeCraft prefix, or a long unbroken token.
_SECRET_PATTERNS = (
    re.compile(r'(?i)\b(?:key|token|api[_-]?key|authorization)\s*[:=]\s*[^\s&,;"\']+'),
    re.compile(r'[?&](?:key|api_key|token)=[^&\s]+'),
    re.compile(r'\bAIza[0-9A-Za-z_\-]{20,}\b'),
    re.compile(r'\bsk-[A-Za-z0-9_\-]{12,}\b'),
    re.compile(r'\bcc_[A-Za-z0-9_\-]{12,}\b'),
    re.compile(r'\b[A-Za-z0-9_\-]{32,}\b'),
)
_URL = re.compile(r'https?://\S+')


def set_publisher(callback: Callable[[dict[str, Any]], None] | None) -> None:
    """Register the WebSocket broadcaster (called by main on startup)."""
    global _publisher
    _publisher = callback


def hint_for(secret: str | None) -> str:
    """`••••` plus the last four characters, never more."""
    value = str(secret or '').strip()
    return f'••••{value[-4:]}' if len(value) >= 4 else ''


def sanitize_error(message: object, limit: int = 80) -> str:
    """A short, credential-free description of a failure."""
    text = str(message or '')
    text = _URL.sub('<url>', text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub('<redacted>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:limit]


def register_node(
    node_id: str,
    *,
    label: str,
    provider: str,
    model: str = '',
    key_hint: str = '',
    enabled: bool = True,
    cooldown_until: float | None = None,
) -> dict[str, Any]:
    """Create or refresh a node without touching its counters."""
    with _lock:
        return dict(_ensure(
            node_id, label=label, provider=provider, model=model,
            key_hint=key_hint, enabled=enabled, cooldown_until=cooldown_until,
        ))


def prune(valid_ids: set[str]) -> None:
    """Drop nodes whose pool entry (or key) no longer exists."""
    with _lock:
        for node_id in list(_nodes):
            if node_id not in valid_ids:
                _nodes.pop(node_id, None)


def _ensure(node_id: str, *, label: str = '', provider: str = '', model: str = '',
            key_hint: str = '', enabled: bool = True,
            cooldown_until: float | None = None) -> dict[str, Any]:
    """Node record, created on first sight (a run can precede the snapshot)."""
    node = _nodes.get(node_id)
    if node is None:
        node = {
            'id': node_id, 'label': label or node_id, 'provider': provider,
            'model': model, 'key_hint': key_hint, 'enabled': enabled,
            'cooldown_until': cooldown_until, 'requests': 0, 'successes': 0,
            'failures': 0, 'tokens_total': 0, 'last_latency_ms': 0,
            'last_phase': '', 'last_error': '', 'updated_at': 0.0,
        }
        _nodes[node_id] = node
    else:
        if label:
            node['label'] = label
        if provider:
            node['provider'] = provider
        if model:
            node['model'] = model
        if key_hint:
            node['key_hint'] = key_hint
        node['enabled'] = enabled
        if cooldown_until is not None:
            node['cooldown_until'] = cooldown_until
    return node


def _publish(event: dict[str, Any]) -> None:
    publisher = _publisher
    if publisher is None:
        return
    try:
        publisher(event)
    except Exception:
        pass  # a live view must never break a translation


def start(
    node_id: str,
    *,
    provider: str,
    model: str = '',
    lines: int = 0,
    key_hint: str = '',
) -> str:
    """Begin one request; returns its id for the matching success/failure."""
    request_id = f'r{next(_ids)}'
    event = {
        'type': 'api_event', 'phase': 'start', 'request_id': request_id,
        'node_id': node_id, 'provider': provider, 'model': model, 'lines': int(lines),
        'key_hint': key_hint, 'latency_ms': 0,
        'prompt_tokens': 0, 'output_tokens': 0, 'thoughts_tokens': 0,
        'http_status': 0, 'error_short': '',
    }
    with _lock:
        node = _ensure(node_id, provider=provider, model=model, key_hint=key_hint)
        node['requests'] += 1
        node['last_phase'] = 'start'
        node['updated_at'] = time.time()
        _stamps.append(time.time())
    _publish(event)
    return request_id


def finish(
    request_id: str,
    node_id: str,
    *,
    ok: bool,
    provider: str = '',
    model: str = '',
    lines: int = 0,
    latency_ms: int = 0,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    thoughts_tokens: int = 0,
    http_status: int = 0,
    error: object = '',
    key_hint: str = '',
    retry: bool = False,
) -> None:
    """Complete a request: update counters, then publish the event."""
    short = sanitize_error(error)
    total_tokens = int(prompt_tokens) + int(output_tokens) + int(thoughts_tokens)
    event = {
        'type': 'api_event',
        'phase': 'success' if ok else ('retry' if retry else 'fail'),
        'request_id': request_id, 'node_id': node_id, 'provider': provider,
        'model': model, 'lines': int(lines), 'key_hint': key_hint,
        'latency_ms': int(latency_ms),
        'prompt_tokens': int(prompt_tokens), 'output_tokens': int(output_tokens),
        'thoughts_tokens': int(thoughts_tokens),
        'http_status': int(http_status), 'error_short': short,
    }
    with _lock:
        node = _ensure(node_id, provider=provider, model=model, key_hint=key_hint)
        if ok:
            node['successes'] += 1
        else:
            node['failures'] += 1
        node['tokens_total'] += total_tokens
        node['last_latency_ms'] = int(latency_ms)
        node['last_phase'] = event['phase']
        node['last_error'] = short
        node['updated_at'] = time.time()
        _recent.appendleft({
            'at': time.time(), 'request_id': request_id, 'node_id': node_id,
            'label': node.get('label', node_id),
            'provider': provider or node.get('provider', ''),
            'model': model or node.get('model', ''),
            'lines': int(lines), 'latency_ms': int(latency_ms),
            'tokens': total_tokens, 'status': event['phase'],
            'error_short': short, 'http_status': int(http_status),
        })
    _publish(event)


def emit_cache_hit(*, node_id: str = 'hub', lines: int = 0, provider: str = '', key_hint: str = '') -> None:
    """A line served from the translation cache: hub-only ripple, no request."""
    event = {
        'type': 'api_event', 'phase': 'cache_hit', 'request_id': f'c{next(_ids)}',
        'node_id': node_id, 'provider': provider, 'model': '', 'lines': int(lines),
        'key_hint': key_hint, 'latency_ms': 0,
        'prompt_tokens': 0, 'output_tokens': 0, 'thoughts_tokens': 0,
        'http_status': 0, 'error_short': '',
    }
    _publish(event)


def _state_of(node: dict[str, Any], now: float) -> str:
    if not node.get('enabled'):
        return 'disabled'
    until = node.get('cooldown_until')
    if until and float(until) > now:
        return 'cooling'
    if node.get('last_phase') in ('fail', 'retry'):
        return 'failed'
    if node.get('requests', 0) == 0:
        return 'idle'
    return 'ready'


def _req_per_min(now: float) -> float:
    horizon = now - RATE_WINDOW_SECONDS
    while _stamps and _stamps[0] < horizon:
        _stamps.popleft()
    return round(len(_stamps) / (RATE_WINDOW_SECONDS / 60.0), 1)


def snapshot() -> dict[str, Any]:
    """Everything the Live tab needs to draw itself before the first event."""
    now = time.time()
    with _lock:
        nodes = []
        totals = {
            'requests': 0, 'successes': 0, 'failures': 0,
            'tokens_total': 0, 'prompt_tokens': 0, 'output_tokens': 0, 'thoughts_tokens': 0,
        }
        for node in _nodes.values():
            view = dict(node)
            view['state'] = _state_of(node, now)
            view['cooldown_remaining'] = max(0, int((node.get('cooldown_until') or 0) - now))
            nodes.append(view)
            totals['requests'] += node['requests']
            totals['successes'] += node['successes']
            totals['failures'] += node['failures']
            totals['tokens_total'] += node['tokens_total']
        nodes.sort(key=lambda item: (item['provider'], item['label']))
        return {
            'nodes': nodes,
            'totals': totals,
            'req_per_min': _req_per_min(now),
            'recent': list(_recent),
        }


def recent() -> list[dict[str, Any]]:
    with _lock:
        return list(_recent)


def reset_for_tests() -> None:
    """Clear every counter (the test suite only)."""
    global _publisher
    with _lock:
        _nodes.clear()
        _recent.clear()
        _stamps.clear()
    _publisher = None
