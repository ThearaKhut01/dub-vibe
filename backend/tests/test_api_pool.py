"""Tests for the Multi-Provider API Pool Manager.

What these pin, in order of how badly a regression would hurt:

* a pool that is not configured changes nothing — online translation keeps using
  the single stored key, byte for byte (the whole suite's other tests rely on it)
* scheduling actually rotates / evens out / prefers a healthy configuration
* a failure is classified honestly: 429 parks the entry for the provider's own
  Retry-After, a 5xx retries the same entry, a 401 does not
* usage is measured, limits are only what the user configured, and a key never
  travels back to the browser
* the HTTP surface the API Pool panel uses stays complete
"""
from __future__ import annotations

import email.message
import io
import json
import time
import types
import urllib.error
from types import SimpleNamespace

import pytest

from backend.app import api_keys, api_pool, providers
from backend.app.api_pool import failover, quota_tracker, request_queue


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------

@pytest.fixture
def credentials(monkeypatch):
    """In-memory credentials: pool keys and the single key from API Settings.

    No test touches Windows Credential Manager, and the Custom API entry saved on
    this machine cannot leak into a routing decision.
    """
    pool: dict[str, str] = {}
    legacy: dict[str, str] = {}
    monkeypatch.setattr(api_keys, 'set_pool_key', lambda entry_id, key: pool.__setitem__(entry_id, key.strip()))
    monkeypatch.setattr(api_keys, 'get_pool_key', lambda entry_id: pool.get(entry_id))
    monkeypatch.setattr(api_keys, 'has_pool_key', lambda entry_id: bool(pool.get(entry_id)))
    monkeypatch.setattr(api_keys, 'delete_pool_key', lambda entry_id: pool.pop(entry_id, None))
    monkeypatch.setattr(api_keys, 'has_key', lambda provider: bool(legacy.get(provider)))
    monkeypatch.setattr(api_keys, 'get_key', lambda provider: legacy.get(provider))
    empty_custom = {
        'base_url': '', 'model': '', 'models': [], 'primary': '',
        'configured': False, 'has_key': False,
    }
    monkeypatch.setattr('backend.app.custom_api.is_configured', lambda: False)
    monkeypatch.setattr('backend.app.custom_api.get_config', lambda: dict(empty_custom))
    return SimpleNamespace(pool=pool, legacy=legacy)


def openai_reply(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        'choices': [{'message': {'content': text}}],
        'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens},
    }


def gemini_reply(text: str) -> dict:
    return {
        'candidates': [{'content': {'parts': [{'text': text}]}}],
        'usageMetadata': {'promptTokenCount': 11, 'candidateTokenCount': 7},
    }


def http_error(code: int, message: str = 'denied', retry_after: float | None = None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers['Retry-After'] = str(retry_after)
    body = json.dumps({'error': {'message': message}}).encode('utf-8')
    return urllib.error.HTTPError('https://example.invalid', code, message, headers, io.BytesIO(body))


@pytest.fixture
def http(monkeypatch):
    """Every provider request, answered from a queue of canned replies."""
    calls: list[dict] = []
    replies: list[object] = []

    def fake_request_json(url, key, body=None, timeout=60, force_auth=False):
        calls.append({'url': url, 'key': key, 'body': body, 'timeout': timeout, 'force_auth': force_auth})
        answer = replies.pop(0) if replies else openai_reply('fallback answer')
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr('backend.app.online_providers._request_json', fake_request_json)
    return SimpleNamespace(calls=calls, replies=replies)


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff delays instead of waiting for them."""
    delays: list[float] = []

    class Clock:
        def sleep(self, seconds: float) -> None:
            delays.append(seconds)

        def __getattr__(self, name):  # perf_counter, monotonic...
            return getattr(time, name)

    monkeypatch.setattr(failover, 'time', Clock())
    return delays


def add_entry(**payload) -> dict:
    return api_pool.upsert_entry(payload)


# ---------------------------------------------------------------------------
# A pool that is not configured must change nothing
# ---------------------------------------------------------------------------

def test_pool_is_inert_without_entries(credentials):
    assert api_pool.has_entries('openai') is False
    assert providers.OpenAITranslationProvider.is_available() is False
    assert providers.HybridTranslationProvider(preference='openai').first_engine() != 'openai'


def test_legacy_key_still_enables_the_engine(credentials):
    credentials.legacy['openai'] = 'sk-legacy'
    assert providers.OpenAITranslationProvider.is_available() is True


def test_provider_uses_the_single_key_when_no_pool_exists(credentials, http):
    credentials.legacy['openai'] = 'sk-legacy'
    http.replies.append(openai_reply('សួស្តី'))
    translator = providers.OpenAITranslationProvider()
    assert translator.translate('hello', 'Khmer', 'en') == 'សួស្តី'
    assert http.calls[0]['key'] == 'sk-legacy'
    assert http.calls[0]['url'] == 'https://api.openai.com/v1/chat/completions'
    # No pool means no cache: a second identical line is a second real request.
    http.replies.append(openai_reply('សួស្តី'))
    assert translator.translate('hello', 'Khmer', 'en') == 'សួស្តី'
    assert len(http.calls) == 2


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

def test_entry_is_created_and_the_key_never_leaves_the_backend(credentials):
    entry = add_entry(provider='gemini', label='primary', model='gemini-flash-latest',
                      key='AIza-secret-value-1234')
    assert entry['has_key'] is True
    assert entry['key_hint'] == '••••1234'
    assert entry['usable'] is True
    assert credentials.pool[entry['id']] == 'AIza-secret-value-1234'

    status = api_pool.status()
    dumped = json.dumps(status)
    assert 'AIza-secret-value-1234' not in dumped
    assert 'AIza-secret-value-1234' not in json.dumps(entry)


def test_editing_an_entry_keeps_its_key_when_none_is_sent(credentials):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-first', label='one')
    updated = add_entry(id=entry['id'], provider='openai', model='gpt-4o-mini', label='renamed')
    assert updated['id'] == entry['id']
    assert updated['label'] == 'renamed'
    assert updated['has_key'] is True
    assert credentials.pool[entry['id']] == 'sk-first'


def test_new_entry_can_reuse_the_key_from_api_settings(credentials):
    credentials.legacy['gemini'] = 'AIza-legacy'
    entry = add_entry(provider='gemini', model='gemini-flash-latest')
    assert entry['key_source'] == 'legacy'
    assert entry['has_key'] is True
    assert credentials.pool == {}  # nothing copied, nothing duplicated


def test_entry_without_any_key_is_refused(credentials):
    with pytest.raises(ValueError, match='needs an API key'):
        add_entry(provider='openai', model='gpt-4o-mini')


def test_unknown_provider_and_bad_base_url_are_refused(credentials):
    with pytest.raises(ValueError, match='Unknown provider'):
        add_entry(provider='anthropic', model='x', key='k')
    with pytest.raises(ValueError, match='Base URL'):
        add_entry(provider='custom', model='x', key='', base_url='not-a-url')


def test_deleting_an_entry_removes_its_credential(credentials):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    assert entry['id'] in credentials.pool
    api_pool.delete_entry(entry['id'])
    assert entry['id'] not in credentials.pool
    assert api_pool.status()['entries'] == []
    with pytest.raises(ValueError):
        api_pool.delete_entry(entry['id'])


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def test_round_robin_rotates_between_entries(credentials):
    add_entry(provider='openai', label='a', model='gpt-4o-mini', key='sk-a')
    add_entry(provider='openai', label='b', model='gpt-4o-mini', key='sk-b')
    api_pool.save_settings({'strategy': 'round_robin'})
    first = [candidate.entry.label for candidate in api_pool.plan('openai')]
    second = [candidate.entry.label for candidate in api_pool.plan('openai')]
    assert set(first) == {'a', 'b'}
    assert first != second


def test_least_used_prefers_the_entry_with_fewer_requests(credentials):
    first = add_entry(provider='openai', label='a', model='gpt-4o-mini', key='sk-a')
    second = add_entry(provider='openai', label='b', model='gpt-4o-mini', key='sk-b')
    quota_tracker.record(first['id'], ok=True)
    quota_tracker.record(first['id'], ok=True)
    api_pool.save_settings({'strategy': 'least_used'})
    order = [candidate.entry.id for candidate in api_pool.plan('openai')]
    assert order[0] == second['id']


def test_hybrid_drops_a_cooling_entry_and_explains_why(credentials):
    good = add_entry(provider='openai', label='good', model='gpt-4o-mini', key='sk-good')
    bad = add_entry(provider='openai', label='bad', model='gpt-4o-mini', key='sk-bad')
    api_pool.save_settings({'strategy': 'hybrid'})
    from backend.app.api_pool import health_monitor, pool_manager
    from backend.app.api_pool import scheduler as pool_scheduler

    health_monitor.record_failure(bad['id'], message='HTTP 503', cooldown=120, reason='provider error')
    usable = [candidate.entry.id for candidate in api_pool.plan('openai')]
    assert usable == [good['id']]
    reasons = pool_scheduler.explain(pool_manager.candidates('openai'))
    assert 'cooling down' in reasons[bad['id']]


def test_an_entry_over_its_configured_limit_is_not_a_candidate(credentials):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1', daily_limit=2)
    assert len(api_pool.plan('openai')) == 1
    quota_tracker.record(entry['id'], ok=True)
    quota_tracker.record(entry['id'], ok=True)
    assert api_pool.plan('openai') == []
    view = api_pool.status()['entries'][0]
    assert view['usable'] is False
    assert 'daily limit' in view['unavailable_reason']
    with pytest.raises(api_pool.PoolUnavailable, match='daily limit'):
        api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}])


def test_no_limit_means_no_invented_ceiling(credentials):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    for _ in range(25):
        quota_tracker.record(entry['id'], ok=True)
    view = api_pool.status()['entries'][0]
    assert view['usage']['requests_today'] == 25
    assert view['usage']['daily_limit'] is None
    assert view['usage']['daily_remaining'] is None
    assert view['usable'] is True


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------

def test_rate_limit_moves_to_the_next_entry_and_honours_retry_after(credentials, http):
    add_entry(provider='openai', label='primary', model='gpt-4o-mini', key='sk-1')
    add_entry(provider='openai', label='backup', model='gpt-4o-mini', key='sk-2')
    api_pool.save_settings({'strategy': 'round_robin'})
    http.replies.extend([http_error(429, 'rate limited', retry_after=42), openai_reply('សួស្តី')])

    assert api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}]) == 'សួស្តី'
    assert [call['key'] for call in http.calls] == ['sk-1', 'sk-2']

    views = api_pool.status()['entries']
    cooled = next(view for view in views if view['health']['state'] == 'cooling')
    assert cooled['health']['cooldown_remaining'] > 30  # Retry-After, not a guess
    assert cooled['usable'] is False


def test_transient_error_retries_the_same_entry(credentials, http, no_sleep):
    add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    api_pool.save_settings({'max_attempts': 3})
    http.replies.extend([http_error(503, 'busy'), openai_reply('ok')])

    assert api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}]) == 'ok'
    assert [call['key'] for call in http.calls] == ['sk-1', 'sk-1']
    assert len(no_sleep) == 1 and no_sleep[0] > 0


def test_auth_failure_is_not_retried_and_parks_the_entry(credentials, http, no_sleep):
    credentials.legacy['openai'] = 'sk-legacy'
    add_entry(provider='openai', model='gpt-4o-mini', key='sk-bad')
    http.replies.extend([http_error(401, 'invalid api key')])

    with pytest.raises(api_pool.PoolUnavailable):
        api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}])
    assert len(http.calls) == 1  # a rejected key is not asked twice
    assert no_sleep == []
    view = api_pool.status()['entries'][0]
    assert view['health']['cooldown_remaining'] > 600
    assert 'invalid api key' in view['health']['last_error']


def test_all_entries_failing_reports_every_reason(credentials, http):
    add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    api_pool.save_settings({'max_attempts': 1})
    http.replies.append(http_error(500, 'boom'))
    with pytest.raises(api_pool.PoolUnavailable, match='boom'):
        api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}])


def test_exhausted_pool_falls_back_to_the_single_stored_key(credentials, http):
    credentials.legacy['openai'] = 'sk-legacy'
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-pool', daily_limit=1)
    quota_tracker.record(entry['id'], ok=True)

    http.replies.append(openai_reply('សួស្តី'))
    translator = providers.OpenAITranslationProvider()
    assert translator.translate('hello', 'Khmer', 'en') == 'សួស្តី'
    assert http.calls[0]['key'] == 'sk-legacy'


# ---------------------------------------------------------------------------
# Provider integration
# ---------------------------------------------------------------------------

def test_pool_entry_enables_the_engine_without_a_legacy_key(credentials):
    add_entry(provider='gemini', model='gemini-flash-latest', key='AIza-pool')
    assert providers.GeminiTranslationProvider.is_available() is True
    translator = providers.HybridTranslationProvider(preference='gemini')
    assert translator.first_engine() == 'gemini'
    translator.check_preferred_engine('Khmer', 'en')


def test_gemini_entry_uses_the_query_string_key_and_its_own_shape(credentials, http):
    entry = add_entry(provider='gemini', model='gemini-flash-latest', key='AIza-pool')
    http.replies.append(gemini_reply('សួស្តី'))

    text = api_pool.chat('gemini', [
        {'role': 'system', 'content': 'Be brief.'},
        {'role': 'user', 'content': 'hi'},
    ])
    assert text == 'សួស្តី'
    call = http.calls[0]
    assert call['url'].startswith('https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent')
    assert 'key=AIza-pool' in call['url']
    assert call['force_auth'] is False
    assert call['body']['systemInstruction']['parts'][0]['text'] == 'Be brief.'
    assert quota_tracker.usage_for_day(entry['id'])['tokens_in'] == 11


def test_validated_translation_is_cached_and_reused(credentials, http):
    add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    http.replies.extend([openai_reply('សួស្តី')])

    translator = providers.OpenAITranslationProvider()
    first = translator.translate('hello', 'Khmer', 'en')
    second = translator.translate('hello  ', 'Khmer', 'en')  # whitespace only
    assert first == second == 'សួស្តី'
    assert len(http.calls) == 1
    stats = api_pool.status()['cache']
    assert stats['entries'] == 1
    assert stats['hits'] >= 1


def test_pool_entries_widen_the_parallel_line_budget(credentials):
    add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    assert providers.HybridTranslationProvider(preference='openai').max_parallel_lines() == 4

    for index in range(2, 4):
        add_entry(provider='openai', model='gpt-4o-mini', key=f'sk-{index}')
    # Three usable entries suggest 6 lines, but the default ceiling is 4.
    assert providers.HybridTranslationProvider(preference='openai').max_parallel_lines() == 4
    api_pool.save_settings({'max_parallel': 8})
    assert providers.HybridTranslationProvider(preference='openai').max_parallel_lines() == 6


def test_local_custom_configurations_stay_serial(credentials):
    add_entry(provider='custom', model='stub-a', base_url='http://127.0.0.1:11434/v1')
    add_entry(provider='custom', model='stub-b', base_url='http://localhost:1234/v1')
    api_pool.save_settings({'max_parallel': 8})
    translator = providers.HybridTranslationProvider(preference='custom')
    assert translator.first_engine() == 'custom'
    # A local server shares the CPU with the UI: parallel lines would only fight it.
    assert translator.max_parallel_lines() == 1


# ---------------------------------------------------------------------------
# Usage, journal and queue
# ---------------------------------------------------------------------------

def test_usage_records_provider_reported_tokens(credentials, http):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    http.replies.append(openai_reply('សួស្តី', prompt_tokens=21, completion_tokens=9))

    api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}])
    row = quota_tracker.usage_for_day(entry['id'])
    assert row['requests'] == 1 and row['successes'] == 1
    assert row['tokens_in'] == 21 and row['tokens_out'] == 9
    assert row['chars_in'] == 2

    status = api_pool.status()
    assert status['usage']['totals']['today']['requests'] == 1
    assert len(status['usage']['daily']) == 7


def test_successful_and_failed_requests_are_journalled(credentials, http):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    api_pool.save_settings({'max_attempts': 1})
    http.replies.extend([openai_reply('ok'), http_error(500, 'boom')])

    api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}])
    with pytest.raises(api_pool.PoolUnavailable):
        api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}])

    journal = request_queue.history(10)
    assert [row['status'] for row in journal] == ['failed', 'done']
    assert all(row['entry_id'] == entry['id'] for row in journal)
    stats = api_pool.status()['queue']
    assert stats['done'] == 1 and stats['failed'] == 1
    assert stats['failed_items'][0]['error']


def test_failed_requests_can_be_retried_from_the_queue(credentials, http):
    add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    api_pool.save_settings({'max_attempts': 1})
    http.replies.extend([http_error(500, 'boom')])
    with pytest.raises(api_pool.PoolUnavailable):
        api_pool.chat('openai', [{'role': 'user', 'content': 'hi'}])
    assert api_pool.status()['queue']['failed'] == 1

    http.replies.append(openai_reply('recovered'))
    result = api_pool.retry_failed()
    assert result['ok'] == 1 and result['failed'] == 0
    stats = api_pool.status()['queue']
    assert stats['failed'] == 0
    # The retried row is resolved, not left pending forever.
    assert stats['pending'] == 0
    assert request_queue.history(1)[0]['status'] == 'done'


def test_a_crash_left_request_is_recovered_as_retryable(credentials):
    request_id = request_queue.enqueue({'messages': [{'role': 'user', 'content': 'x'}]}, 'openai')
    request_queue.mark_in_flight(request_id, entry_id='gone', model='m')
    assert api_pool.status()['queue']['in_flight'] == 1

    assert request_queue.recover_stalled() == 1
    stats = api_pool.status()['queue']
    assert stats['in_flight'] == 0 and stats['failed'] == 1
    assert 'interrupted' in stats['failed_items'][0]['error']


# ---------------------------------------------------------------------------
# Settings and connection test
# ---------------------------------------------------------------------------

def test_settings_are_validated_and_persisted(credentials):
    saved = api_pool.save_settings({'strategy': 'least_used', 'max_attempts': 9, 'cache_ttl_days': 0})
    assert saved['strategy'] == 'least_used'
    assert saved['max_attempts'] == 6  # clamped
    assert saved['cache_ttl_days'] == 1
    assert api_pool.status()['settings']['strategy'] == 'least_used'
    with pytest.raises(ValueError, match='Unknown strategy'):
        api_pool.save_settings({'strategy': 'magic'})


def test_connection_test_proves_one_real_request(credentials, http):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    http.replies.append(openai_reply('OK'))

    result = api_pool.test_entry(entry['id'])
    assert result['sample'] == 'OK'
    assert result['model'] == 'gpt-4o-mini'
    assert http.calls[0]['body']['max_tokens'] == 8
    assert quota_tracker.usage_for_day(entry['id'])['requests'] == 1


def test_failed_connection_test_records_health_and_reports_the_reason(credentials, http):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    http.replies.append(http_error(404, 'model not found'))
    with pytest.raises(RuntimeError, match='model not found'):
        api_pool.test_entry(entry['id'])
    view = api_pool.status()['entries'][0]
    assert view['health']['cooldown_remaining'] > 0
    assert 'model not found' in view['health']['last_error']


def test_importing_legacy_credentials_never_copies_a_key(credentials):
    credentials.legacy['gemini'] = 'AIza-legacy'
    result = api_pool.import_legacy()
    assert result['created']
    entries = api_pool.status()['entries']
    assert entries[0]['provider'] == 'gemini'
    assert entries[0]['key_source'] == 'legacy'
    assert entries[0]['has_key'] is True
    assert credentials.pool == {}


# ---------------------------------------------------------------------------
# HTTP surface used by the panel
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.app import main

    return TestClient(main.app)


def test_status_route_reports_the_pool_without_secrets(credentials, client):
    add_entry(provider='openai', model='gpt-4o-mini', key='sk-hidden-value')
    response = client.get('/api/api-pool/status')
    assert response.status_code == 200
    body = response.json()
    assert body['entries'][0]['model'] == 'gpt-4o-mini'
    assert 'sk-hidden-value' not in json.dumps(body)
    assert set(body['providers']) == {'gemini', 'openai', 'codecraft', 'custom'}


def test_provider_keys_route_reports_pool_readiness(credentials, client):
    """The engine picker must offer a provider backed only by a pool."""
    add_entry(provider='gemini', model='gemini-flash-latest', key='AIza-pool-only')
    body = client.get('/api/providers/keys').json()
    assert body['gemini']['has_key'] is False
    assert body['gemini']['pool_entries'] == 1
    assert body['gemini']['pool_usable'] == 1
    assert body['openai']['pool_usable'] == 0
    assert 'AIza-pool-only' not in json.dumps(body)


def test_entry_routes_create_update_and_delete(credentials, client):
    created = client.post('/api/api-pool/entries', json={
        'provider': 'openai', 'label': 'main', 'model': 'gpt-4o-mini', 'key': 'sk-1',
    })
    assert created.status_code == 200
    entry_id = created.json()['entry']['id']

    updated = client.post('/api/api-pool/entries', json={
        'id': entry_id, 'provider': 'openai', 'model': 'gpt-4o-mini', 'daily_limit': 50,
    })
    assert updated.status_code == 200
    assert updated.json()['entry']['usage']['daily_limit'] == 50

    deleted = client.delete(f'/api/api-pool/entries/{entry_id}')
    assert deleted.status_code == 200
    assert deleted.json()['pool']['entries'] == []


def test_entry_route_rejects_a_missing_key(credentials, client):
    response = client.post('/api/api-pool/entries', json={'provider': 'openai', 'model': 'gpt-4o-mini'})
    assert response.status_code == 400
    assert 'API key' in response.json()['detail']


def test_unknown_entry_routes_answer_404(credentials, client):
    assert client.delete('/api/api-pool/entries/nope').status_code == 404
    assert client.post('/api/api-pool/entries/nope/test').status_code == 404


def test_settings_route_validates_the_strategy(credentials, client):
    ok = client.post('/api/api-pool/settings', json={'strategy': 'round_robin'})
    assert ok.status_code == 200 and ok.json()['settings']['strategy'] == 'round_robin'
    bad = client.post('/api/api-pool/settings', json={'strategy': 'nope'})
    assert bad.status_code == 400


def test_cache_and_health_routes_clear_state(credentials, http, client):
    entry = add_entry(provider='openai', model='gpt-4o-mini', key='sk-1')
    http.replies.append(openai_reply('សួស្តី'))
    providers.OpenAITranslationProvider().translate('hello', 'Khmer', 'en')
    assert client.get('/api/api-pool/status').json()['cache']['entries'] == 1

    cleared = client.post('/api/api-pool/cache/clear')
    assert cleared.status_code == 200
    assert cleared.json()['pool']['cache']['entries'] == 0

    from backend.app.api_pool import health_monitor

    health_monitor.record_failure(entry['id'], message='boom', cooldown=60, reason='test')
    reset = client.post('/api/api-pool/health/reset', json={'entry_id': entry['id']})
    assert reset.status_code == 200
    assert reset.json()['pool']['entries'][0]['health']['cooldown_remaining'] == 0
