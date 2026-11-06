"""The Multi-Provider API Pool Manager.

A pool is a set of API configurations for one provider (Gemini, OpenAI,
CodeCraft, or any OpenAI-compatible endpoint). Requests are scheduled across
them, failures move the request to the next configuration, and every call is
measured so the panel can show what actually happened.

Integration is deliberately narrow. The pool speaks *chat completions* and
nothing else, because that is the only thing the existing translation engines
use it for: :class:`backend.app.providers.OnlineLLMTranslationProvider` asks the
pool for a completion when the engine has pool entries, and keeps its own
credential path when it does not. Subtitle prompts, batching, SRT timestamps and
the model picker are untouched — the pool only decides *which* configuration
answers.

Nothing here invents quota. A limit exists only when the user configures one;
usage is what the provider reported, or the characters sent when it reported
nothing.
"""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
from typing import Any

from . import (
    cache,
    failover,
    health_monitor as health,
    quota_tracker as quota,
    request_queue as queue,
    scheduler as scheduling,
)
from .models import (
    POOL_PROVIDERS,
    PROVIDER_LABELS,
    SCHEDULING_STRATEGIES,
    PoolEntry,
    all_entries,
    new_entry_id,
    provider_entries,
    store,
    validate_base_url,
)

logger = logging.getLogger(__name__)


class PoolUnavailable(RuntimeError):
    """No pool configuration could serve this request.

    Raised when every entry is disabled, cooling down, over a configured limit,
    or when the last attempt failed. The caller may still hold a single stored
    key, which is a legitimate last resort — the provider classes decide that.
    """


class ProviderCallError(RuntimeError):
    """A pool call failed with something the failover layer can classify."""

    def __init__(self, message: str, *, status: int | None = None, kind: str = 'unknown',
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Credentials and endpoints
# ---------------------------------------------------------------------------

def _credential(entry: PoolEntry) -> str:
    """The entry's key: its own pooled secret, or the key from API Settings."""
    from .. import api_keys

    if entry.key_source == 'legacy':
        return api_keys.get_key(entry.provider) or ''
    return api_keys.get_pool_key(entry.id) or ''


def _key_hint(credential: str) -> str:
    if not credential:
        return ''
    return f'••••{credential[-4:]}' if len(credential) > 4 else '••••'


def _default_model(provider: str) -> str:
    from ..online_providers import (
        CODECRAFT_DEFAULT_CHAT_MODEL,
        GEMINI_DEFAULT_CHAT_MODEL,
        OPENAI_DEFAULT_CHAT_MODEL,
    )

    return {
        'gemini': GEMINI_DEFAULT_CHAT_MODEL,
        'openai': OPENAI_DEFAULT_CHAT_MODEL,
        'codecraft': CODECRAFT_DEFAULT_CHAT_MODEL,
        'custom': '',
    }.get(provider, '')


def _base_url(entry: PoolEntry) -> str:
    """Where this entry's requests go."""
    from ..online_providers import CODECRAFT_BASE, GEMINI_BASE, OPENAI_BASE

    if entry.base_url:
        return entry.base_url.rstrip('/')
    if entry.provider == 'gemini':
        return GEMINI_BASE
    if entry.provider == 'openai':
        return OPENAI_BASE
    if entry.provider == 'codecraft':
        return CODECRAFT_BASE
    from .. import custom_api

    configured = custom_api.get_config().get('base_url') or ''
    if not configured:
        raise ProviderCallError(
            'Custom API base URL is not configured for this pool entry '
            '(set it on the entry, or in API Settings).',
            kind='request',
        )
    return str(configured).rstrip('/')


def _request_shape(entry: PoolEntry, messages: list[dict], max_tokens: int | None,
                   generation_config: dict[str, Any] | None = None
                   ) -> tuple[str, dict, bool]:
    """(url, body, needs_auth_header) for this provider's chat shape.

    ``generation_config`` lets a caller keep its own sampling settings (the batch
    translator sends ``temperature`` and, for Gemini, ``thinkingConfig``) without
    giving up scheduler, failover and quota accounting. ``max_tokens`` always wins
    over an ``maxOutputTokens`` in the config, so one source of truth remains.
    """
    if entry.provider == 'gemini':
        system = '\n'.join(
            str(message.get('content') or '') for message in messages
            if message.get('role') == 'system'
        ).strip()
        contents = [
            {
                'role': 'model' if message.get('role') == 'assistant' else 'user',
                'parts': [{'text': str(message.get('content') or '')}],
            }
            for message in messages
            if message.get('role') != 'system'
        ]
        body: dict[str, Any] = {'contents': contents}
        if system:
            body['systemInstruction'] = {'parts': [{'text': system}]}
        config = dict(generation_config or {})
        if max_tokens:
            config['maxOutputTokens'] = int(max_tokens)
        if config:
            body['generationConfig'] = config
        url = f'{_base_url(entry)}/models/{entry.model}:generateContent'
        return url, body, False

    body = {'model': entry.model, 'messages': messages}
    if max_tokens:
        body['max_tokens'] = int(max_tokens)
    temperature = (generation_config or {}).get('temperature')
    if temperature is not None:
        body['temperature'] = float(temperature)
    return f'{_base_url(entry)}/chat/completions', body, True


def _parse_reply(entry: PoolEntry, payload: dict) -> tuple[str, dict[str, Any]]:
    """(text, usage) from a provider's JSON reply."""
    if entry.provider == 'gemini':
        candidates = payload.get('candidates') or []
        parts = (candidates[0].get('content', {}).get('parts') or []) if candidates else []
        text = ''.join(str(part.get('text') or '') for part in parts).strip()
        if not text:
            reason = (
                payload.get('error', {}).get('message')
                or payload.get('promptFeedback', {}).get('blockReason')
                or 'no candidates'
            )
            raise ProviderCallError(f'Gemini returned no completion: {reason}', kind='transient')
        meta = payload.get('usageMetadata') or {}
        usage = {
            'tokens_in': int(meta.get('promptTokenCount') or 0),
            'tokens_out': int(meta.get('candidatesTokenCount') or 0),
            'thoughts': int(meta.get('thoughtsTokenCount') or 0),
        }
    else:
        choices = payload.get('choices') or []
        content = choices[0].get('message', {}).get('content') if choices else None
        if isinstance(content, list):  # some compatible servers answer content parts
            content = ''.join(
                str(part.get('text') or '') for part in content if isinstance(part, dict)
            )
        text = str(content or '').strip()
        if not text:
            raise ProviderCallError(f'{PROVIDER_LABELS.get(entry.provider, entry.provider)} returned no completion', kind='transient')
        meta = payload.get('usage') or {}
        usage = {
            'tokens_in': int(meta.get('prompt_tokens') or 0),
            'tokens_out': int(meta.get('completion_tokens') or 0),
        }
    usage['chars_out'] = len(text)
    return text, usage


def _call_entry(entry: PoolEntry, messages: list[dict], *, max_tokens: int | None,
                timeout: int, generation_config: dict[str, Any] | None = None,
                api_meta: dict[str, Any] | None = None
                ) -> tuple[str, dict[str, Any]]:
    """One HTTP call to one pool entry. Raises ProviderCallError on failure."""
    from ..online_providers import _http_error_detail, _request_json
    from .. import api_events

    credential = _credential(entry)
    label = PROVIDER_LABELS.get(entry.provider, entry.provider)
    meta = api_meta or {}
    hint = api_events.hint_for(credential)
    started = time.monotonic()
    request_id = api_events.start(
        entry.id, provider=entry.provider, model=entry.model,
        lines=int(meta.get('lines') or 0), key_hint=hint,
    )

    def failed(error: ProviderCallError) -> ProviderCallError:
        api_events.finish(
            request_id, entry.id, ok=False, provider=entry.provider, model=entry.model,
            lines=int(meta.get('lines') or 0), latency_ms=int((time.monotonic() - started) * 1000),
            http_status=int(getattr(error, 'status', 0) or 0), error=str(error),
            retry=bool(getattr(error, 'cooldown', 0) or getattr(error, 'kind', '') == 'transient'),
            key_hint=hint,
        )
        return error

    try:
        url, body, needs_auth = _request_shape(entry, messages, max_tokens, generation_config)
    except ProviderCallError as error:
        raise failed(error)
    if entry.provider == 'gemini':
        url = f'{url}?key={urllib.parse.quote(credential, safe="")}'
    try:
        payload = _request_json(url, credential, body, timeout=timeout, force_auth=needs_auth)
    except urllib.error.HTTPError as error:
        # Read the body once, here: the classifier must not re-read a consumed
        # stream. Retry-After is carried over so a 429 is honoured exactly.
        raise failed(ProviderCallError(
            _http_error_detail(error), status=error.code,
            retry_after=failover.parse_retry_after(error.headers),
        )) from error
    except urllib.error.URLError as error:
        raise failed(ProviderCallError(f'{label} is unreachable: {error}', kind='network')) from error
    except (TimeoutError, OSError) as error:
        raise failed(ProviderCallError(f'{label} request failed: {error}', kind='network')) from error
    except RuntimeError as error:
        # _request_json turns a TLS verification failure into an explanation.
        raise failed(ProviderCallError(str(error), kind='network')) from error
    text, usage = _parse_reply(entry, payload)
    api_events.finish(
        request_id, entry.id, ok=True, provider=entry.provider, model=entry.model,
        lines=int(meta.get('lines') or 0), latency_ms=int((time.monotonic() - started) * 1000),
        prompt_tokens=int(usage.get('tokens_in') or 0),
        output_tokens=int(usage.get('tokens_out') or 0),
        thoughts_tokens=int(usage.get('thoughts') or 0),
        key_hint=hint,
    )
    return text, usage


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def candidates(provider: str) -> list[scheduling.Candidate]:
    """Every entry of a provider with the facts the scheduler needs."""
    entries = provider_entries(provider)
    if not entries:
        return []
    ids = [entry.id for entry in entries]
    today = quota.usage_map(ids, scope='day')
    month = quota.usage_map(ids, scope='month')
    result: list[scheduling.Candidate] = []
    for entry in entries:
        today_row = today.get(entry.id, {})
        month_row = month.get(entry.id, {})
        limits = quota.limit_status(entry, today_row, month_row)
        result.append(scheduling.Candidate(
            entry=entry,
            requests_today=limits['requests_today'],
            requests_month=limits['requests_month'],
            successes=int(today_row.get('successes', 0)),
            failures=int(today_row.get('failures', 0)),
            headroom=quota.headroom(entry, today_row),
            health_state=health.state_of(entry.id),
            cooldown_remaining=health.cooldown_remaining(entry.id),
            quota_reason='' if limits['allowed'] else str(limits['reason'] or 'limit reached'),
        ))
    return result


def plan(provider: str, strategy: str | None = None) -> list[scheduling.Candidate]:
    """Usable entries in the order this request should try them."""
    strategy = strategy or store().settings().get('strategy')
    return scheduling.scheduler().plan(candidates(provider), strategy or 'hybrid')


def has_entries(provider: str) -> bool:
    """True when this provider has pool configurations (enabled or not)."""
    return bool(provider_entries(provider))


def is_active(provider: str) -> bool:
    """True when at least one enabled configuration exists for this provider."""
    return any(entry.enabled for entry in provider_entries(provider))


def entry_count(provider: str) -> int:
    """How many configurations exist for a provider (enabled or not)."""
    return len(provider_entries(provider))


def usable_count(provider: str) -> int:
    try:
        return len(plan(provider))
    except Exception:
        return 0


def custom_endpoints_are_local() -> bool:
    """True when every custom configuration that can serve points at this machine/LAN.

    A local server shares the CPU with the UI, so translation stays serial there
    — the same rule the single Custom API endpoint already follows.
    """
    from ..custom_api import is_local_url

    urls: list[str] = []
    for candidate in plan('custom'):
        try:
            urls.append(_base_url(candidate.entry))
        except ProviderCallError:
            return False
    if not urls:
        return False
    return all(is_local_url(url) for url in urls)


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------

def _unavailable_message(provider: str) -> str:
    label = PROVIDER_LABELS.get(provider, provider)
    entries = provider_entries(provider)
    if not entries:
        return f'No {label} API pool configuration is set up.'
    reasons = []
    for candidate in candidates(provider):
        if candidate.reason:
            reasons.append(f'{candidate.entry.display_name}: {candidate.reason}')
    detail = '; '.join(reasons) if reasons else 'no configuration reported available'
    return f'Every {label} pool configuration is unavailable ({detail}).'


def chat(
    provider: str,
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_attempts: int | None = None,
    generation_config: dict[str, Any] | None = None,
    usage_out: dict[str, int] | None = None,
    served_out: dict[str, Any] | None = None,
    api_meta: dict[str, Any] | None = None,
) -> str:
    """One completion from the provider's pool, with scheduling and failover.

    Raises :class:`PoolUnavailable` when nothing could serve the request; the
    per-attempt errors are included so the caller can explain why.

    ``generation_config`` is forwarded to the provider body (temperature, and the
    Gemini thinking switch), and ``usage_out`` receives the token counts the pool
    already parses, so a caller can log usage without a second code path.

    ``max_attempts`` overrides the stored retry policy for this call only — a big
    batch or a connection test wants one try per configuration, not three.
    """
    settings = store().settings()
    ordered = plan(provider)
    if not ordered:
        raise PoolUnavailable(_unavailable_message(provider))

    timeout = max(10, int(timeout or settings.get('request_timeout') or 90))
    chars_in = sum(len(str(message.get('content') or '')) for message in messages)
    request_id = queue.enqueue(
        {
            'messages': messages,
            'max_tokens': max_tokens,
            'timeout': timeout,
            'generation_config': generation_config,
            'provider': provider,
        },
        provider,
    )

    seen: list[str] = []

    def attempt(entry: PoolEntry) -> tuple[str, dict[str, Any]]:
        queue.mark_in_flight(request_id, entry_id=entry.id, model=entry.model)
        return _call_entry(
            entry, messages, max_tokens=max_tokens, timeout=timeout,
            generation_config=generation_config, api_meta=api_meta,
        )

    def on_success(entry: PoolEntry, usage: dict[str, Any], latency_ms: int) -> None:
        health.record_success(entry.id, latency_ms)
        quota.record(
            entry.id, ok=True,
            tokens_in=int(usage.get('tokens_in') or 0),
            tokens_out=int(usage.get('tokens_out') or 0),
            chars_in=chars_in,
            chars_out=int(usage.get('chars_out') or 0),
            latency_ms=latency_ms,
        )
        if usage_out is not None:
            usage_out['prompt_tokens'] = int(usage.get('tokens_in') or 0)
            usage_out['candidates_tokens'] = int(usage.get('tokens_out') or 0)
            usage_out['thoughts_tokens'] = int(usage.get('thoughts') or 0)
        if served_out is not None:
            served_out.update({
                'entry_id': entry.id, 'model': entry.model, 'label': entry.display_name,
            })

    def on_failure(entry: PoolEntry, failure: failover.Failure) -> None:
        seen.append(f'{entry.display_name}: {failure.message}')
        health.record_failure(
            entry.id, message=failure.message,
            cooldown=failure.cooldown, reason=failure.park_reason,
        )
        quota.record(entry.id, ok=False, chars_in=chars_in)

    outcome = failover.execute(
        # The scheduler works in candidates; the failover loop only needs entries.
        [candidate.entry for candidate in ordered], attempt,
        max_attempts=int(max_attempts or settings.get('max_attempts') or 3),
        on_success=on_success,
        on_failure=on_failure,
    )
    if not outcome.ok or outcome.entry is None:
        error = outcome.error
        queue.fail(
            request_id,
            entry_id='', model='',
            error='; '.join(seen) or str(error or 'no pool configuration answered'),
            attempts=outcome.attempts,
        )
        detail = f' Last error: {seen[-1]}.' if seen else ''
        raise PoolUnavailable(
            f'No {PROVIDER_LABELS.get(provider, provider)} pool configuration answered.{detail}'
        ) from error

    queue.finish(
        request_id,
        entry_id=outcome.entry.id,
        model=outcome.entry.model,
        result=outcome.text or '',
        latency_ms=outcome.latency_ms,
    )
    return outcome.text or ''


# ---------------------------------------------------------------------------
# Translation cache (only validated lines are stored — see api_pool.cache)
# ---------------------------------------------------------------------------

def settings() -> dict[str, Any]:
    """Current pool settings (a copy; editing goes through save_settings)."""
    return store().settings()


def cache_enabled() -> bool:
    return bool(store().settings().get('cache_enabled'))


def cache_enabled_for(provider: str) -> bool:
    """Caching follows the pool: it is on when the pool serves this provider."""
    return cache_enabled() and is_active(provider)


def cached_translation(text: str, source_lang: str, target_lang: str,
                       style_hint: str | None = '') -> str | None:
    if not cache_enabled() or not str(text or '').strip():
        return None
    key = cache.make_key(text, source_lang, target_lang, style_hint or '')
    try:
        return cache.lookup(key, int(store().settings().get('cache_ttl_days') or 7))
    except Exception as error:  # a broken cache must never fail a translation
        logger.info('API pool cache lookup failed: %s', error)
        return None


def remember_translation(text: str, translation: str, source_lang: str, target_lang: str,
                         style_hint: str | None = '', provider: str = '',
                         model: str = '') -> None:
    if not cache_enabled() or not str(translation or '').strip():
        return
    key = cache.make_key(text, source_lang, target_lang, style_hint or '')
    try:
        cache.store(
            key, translation=translation, source_text=text,
            source_lang=source_lang, target_lang=target_lang,
            provider=provider, model=model,
        )
    except Exception as error:
        logger.info('API pool cache store failed: %s', error)


# ---------------------------------------------------------------------------
# Configuration management
# ---------------------------------------------------------------------------

def _legacy_available(provider: str) -> bool:
    from .. import api_keys, custom_api

    try:
        if provider == 'custom':
            return bool(custom_api.is_configured())
        return bool(api_keys.has_key(provider))
    except Exception:
        return False


def entry_view(entry: PoolEntry) -> dict[str, Any]:
    credential = _credential(entry)
    candidate = next((item for item in candidates(entry.provider) if item.entry.id == entry.id), None)
    today = quota.usage_for_day(entry.id)
    month = quota.usage_for_month(entry.id)
    return {
        **entry.to_public(has_key=bool(credential), key_hint=_key_hint(credential)),
        'health': health.snapshot(entry.id),
        'usage': quota.limit_status(entry, today, month),
        'usable': bool(candidate and candidate.usable),
        'unavailable_reason': candidate.reason if candidate else '',
    }


def upsert_entry(payload: dict[str, Any]) -> dict[str, Any]:
    """Create or update one pool configuration. A provided key replaces the old one."""
    from .. import api_keys

    if not isinstance(payload, dict):
        raise ValueError('A pool configuration object is required.')
    entry_id = str(payload.get('id') or '').strip()
    existing = store().get(entry_id) if entry_id else None

    provider = str(payload.get('provider') or (existing.provider if existing else '')).strip().lower()
    if provider not in POOL_PROVIDERS:
        raise ValueError(f'Unknown provider: {provider or "(empty)"}.')

    model = str(payload.get('model') or '').strip() or (existing.model if existing else '')
    if not model:
        model = _default_model(provider)
    if not model:
        raise ValueError('Model name is required for this configuration.')

    raw_base = payload.get('base_url')
    base_url = validate_base_url(provider, raw_base if raw_base is not None else (existing.base_url if existing else ''))

    if provider == 'custom' and not base_url:
        from .. import custom_api

        if not custom_api.is_configured():
            raise ValueError(
                'A Custom API configuration needs a base URL (or a Custom API '
                'endpoint saved in API Settings).'
            )

    key_source = str(payload.get('key_source') or (existing.key_source if existing else '')).strip().lower()
    if key_source not in ('pool', 'legacy'):
        key_source = 'pool'

    new_key = str(payload.get('key') or '').strip()
    if new_key:
        entry_id = entry_id or new_entry_id()
        api_keys.set_pool_key(entry_id, new_key)
        key_source = 'pool'
    entry_id = entry_id or new_entry_id()

    if not new_key and existing is None:
        # A brand-new entry with no key of its own: reuse the single key from API
        # Settings when there is one, so importing never copies a secret.
        if provider != 'custom' and api_keys.has_key(provider):
            key_source = 'legacy'
        elif provider != 'custom' and not api_keys.has_key(provider):
            raise ValueError(
                f'{PROVIDER_LABELS[provider]} needs an API key. Paste one here, or save '
                'one in API Settings first.'
            )

    entry = PoolEntry(
        id=entry_id,
        provider=provider,
        model=model,
        label=str(payload.get('label') or (existing.label if existing else '')).strip(),
        base_url=base_url,
        enabled=bool(payload.get('enabled', existing.enabled if existing else True)),
        weight=max(1, min(int(payload.get('weight') or (existing.weight if existing else 1)), 10)),
        daily_limit=_limit(payload, 'daily_limit', existing),
        monthly_limit=_limit(payload, 'monthly_limit', existing),
        key_source=key_source,
        notes=str(payload.get('notes') or (existing.notes if existing else '')),
        created_at=existing.created_at if existing else time.time(),
    )
    stored = store().upsert(entry)
    # Editing a configuration is an attempt to fix it: drop a stale cooldown so
    # the user's next test is not blocked by the previous credential's failure.
    health.clear(stored.id)
    scheduling.scheduler().reset()
    return entry_view(stored)


def _limit(payload: dict[str, Any], field: str, existing: PoolEntry | None) -> int | None:
    if field in payload:
        value = payload.get(field)
        if value is None or value == '':
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f'{field} must be a whole number of requests.')
        if number < 0:
            raise ValueError(f'{field} cannot be negative.')
        return number or None
    return getattr(existing, field) if existing else None


def delete_entry(entry_id: str) -> dict[str, Any]:
    from .. import api_keys

    entry = store().get(entry_id)
    if entry is None:
        raise ValueError('Unknown pool configuration.')
    store().delete(entry_id)
    if entry.key_source == 'pool':
        api_keys.delete_pool_key(entry_id)
    health.clear(entry_id)
    scheduling.scheduler().reset()
    return {'status': 'ok', 'deleted': entry_id}


def test_entry(entry_id: str) -> dict[str, Any]:
    """Prove one configuration works: a single tiny completion, journalled."""
    entry = store().get(entry_id)
    if entry is None:
        raise ValueError('Unknown pool configuration.')
    messages = [{'role': 'user', 'content': 'Reply with the single word OK.'}]
    chars_in = len(messages[0]['content'])
    request_id = queue.enqueue(
        {'messages': messages, 'max_tokens': 8, 'provider': entry.provider, 'context': 'connection test'},
        entry.provider,
    )
    queue.mark_in_flight(request_id, entry_id=entry.id, model=entry.model)
    started = time.monotonic()
    try:
        text, usage = _call_entry(entry, messages, max_tokens=8, timeout=45)
    except Exception as error:
        failure = failover.classify(error)
        latency_ms = int((time.monotonic() - started) * 1000)
        health.record_failure(entry.id, message=failure.message,
                              cooldown=failure.cooldown, reason=failure.park_reason)
        quota.record(entry.id, ok=False, chars_in=chars_in)
        queue.fail(request_id, entry_id=entry.id, model=entry.model,
                   error=failure.message, attempts=1)
        raise RuntimeError(failure.message) from error
    latency_ms = int((time.monotonic() - started) * 1000)
    health.record_success(entry.id, latency_ms)
    quota.record(
        entry.id, ok=True,
        tokens_in=int(usage.get('tokens_in') or 0),
        tokens_out=int(usage.get('tokens_out') or 0),
        chars_in=chars_in, chars_out=int(usage.get('chars_out') or 0),
        latency_ms=latency_ms,
    )
    queue.finish(request_id, entry_id=entry.id, model=entry.model,
                 result=text, latency_ms=latency_ms)
    return {
        'ok': True,
        'entry': entry_view(store().get(entry_id) or entry),
        'model': entry.model,
        'sample': text.strip()[:80],
        'latency_ms': latency_ms,
    }


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(update, dict):
        raise ValueError('A settings object is required.')
    if 'strategy' in update and str(update.get('strategy') or '') not in SCHEDULING_STRATEGIES:
        raise ValueError(f'Unknown strategy: {update.get("strategy")}.')
    settings = store().save_settings(update)
    scheduling.scheduler().reset()
    return settings


# Cooldowns this app imposes on itself. An explicit "retry" may ignore these;
# a provider's Retry-After, an empty allowance or a rejected key may not.
_SOFT_COOLDOWN_REASONS = frozenset({'provider error', 'unknown', 'unreachable'})


def _soften_cooldowns() -> None:
    """Drop the self-imposed pauses so a manual retry really tries again."""
    for entry in all_entries():
        snapshot = health.snapshot(entry.id)
        if snapshot['cooldown_remaining'] > 0 and snapshot['cooldown_reason'] in _SOFT_COOLDOWN_REASONS:
            health.clear(entry.id)


def retry_failed(limit: int = 20) -> dict[str, Any]:
    """Re-send failed requests from the persistent queue.

    The result is stored in the request journal; subtitle translations are
    restored by pressing Translate in the project (finished lines are kept, and
    a pooled line that was already validated comes back from the cache).
    """
    _soften_cooldowns()
    retried = queue.retry_all_failed()
    rows = queue.pending(limit=max(1, min(int(limit), 50)))
    summary = {'requeued': retried, 'sent': 0, 'ok': 0, 'failed': 0, 'errors': []}
    for row in rows:
        payload = row.get('payload') or {}
        messages = payload.get('messages') if isinstance(payload, dict) else None
        if not messages:
            queue.fail(int(row['id']), entry_id='', model='',
                       error='stored payload is missing', attempts=0)
            summary['failed'] += 1
            continue
        summary['sent'] += 1
        try:
            text = chat(
                str(row.get('provider') or ''),
                messages,
                max_tokens=payload.get('max_tokens'),
                timeout=payload.get('timeout'),
                generation_config=payload.get('generation_config') or None,
            )
        except Exception as error:
            # The re-sent request has its own journal row; this one records the
            # outcome of the retry itself, so the queue does not keep a row in
            # 'pending' forever.
            queue.fail(int(row['id']), entry_id='', model='', error=str(error), attempts=0)
            summary['failed'] += 1
            if len(summary['errors']) < 5:
                summary['errors'].append(str(error)[:200])
        else:
            queue.finish(int(row['id']), entry_id='', model='', result=text, latency_ms=0)
            summary['ok'] += 1
    return summary


def import_legacy() -> dict[str, Any]:
    """Create pool entries that reuse the keys already saved in API Settings."""
    from .. import api_keys, custom_api

    created: list[str] = []
    for provider in ('gemini', 'openai', 'codecraft'):
        if provider_entries(provider) or not api_keys.has_key(provider):
            continue
        entry = PoolEntry(
            id=new_entry_id(),
            provider=provider,
            model=_default_model(provider),
            label=f'{PROVIDER_LABELS[provider]} · API Settings key',
            key_source='legacy',
        )
        store().upsert(entry)
        created.append(entry.id)
    if not provider_entries('custom') and custom_api.is_configured():
        config = custom_api.get_config()
        if config.get('base_url') and config.get('primary'):
            entry = PoolEntry(
                id=new_entry_id(),
                provider='custom',
                model=str(config['primary']),
                base_url=str(config['base_url']),
                label='Custom API · API Settings endpoint',
                key_source='legacy',
            )
            store().upsert(entry)
            created.append(entry.id)
    scheduling.scheduler().reset()
    return {'status': 'ok', 'created': created, 'pool': status()}


def reset_health(entry_id: str = '') -> dict[str, Any]:
    if entry_id:
        health.clear(entry_id)
    else:
        health.clear_all()
    return {'status': 'ok'}


def clear_cache() -> dict[str, Any]:
    return {'status': 'ok', 'removed': cache.clear()}


def status() -> dict[str, Any]:
    """Everything the API Pool panel renders."""
    settings = store().settings()
    entries = all_entries()
    views = [entry_view(entry) for entry in entries]
    providers: dict[str, Any] = {}
    order: dict[str, list[str]] = {}
    for provider in POOL_PROVIDERS:
        provider_views = [view for view in views if view['provider'] == provider]
        providers[provider] = {
            'label': PROVIDER_LABELS[provider],
            'entries': len(provider_views),
            'usable': sum(1 for view in provider_views if view['usable']),
            'has_legacy_credential': _legacy_available(provider),
        }
        try:
            order[provider] = [candidate.entry.display_name for candidate in plan(provider)]
        except Exception:
            order[provider] = []
    ids = [entry.id for entry in entries]
    return {
        'settings': {**settings, 'strategies': list(SCHEDULING_STRATEGIES)},
        'providers': providers,
        'entries': views,
        'usage': {
            'totals': quota.totals(ids),
            'daily': quota.daily_series(ids, 7),
        },
        'queue': {**queue.stats(), 'failed_items': queue.failed(10)},
        'cache': cache.stats(),
        'plan': order,
    }


def startup() -> None:
    """Recover a journal left by a crash and keep the on-disk data bounded."""
    try:
        stalled = queue.recover_stalled()
        if stalled:
            logger.info('API pool: marked %s interrupted request(s) as failed (retryable).', stalled)
        queue.prune(5000)
        cache.prune(int(store().settings().get('cache_ttl_days') or 7))
    except Exception as error:  # a broken journal must never stop the app
        logger.warning('API pool startup recovery failed: %s', error)


def reset_for_tests() -> None:
    """Drop in-memory caches (used by the test suite between cases)."""
    store().reload()
    scheduling.scheduler().reset()
    health.clear_all()
