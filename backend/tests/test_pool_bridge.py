"""The bridge between the API pool, the provider test route and batch translation.

Three things must stay true:
* a batch request uses the pool (rotation, quota, health) when the engine has
  entries, and keeps its own sampling config through the pool;
* the pool reports the token usage it already parses;
* "Test connection" exercises the path that will actually run.
"""
from __future__ import annotations

import pytest

from backend.app.api_pool import pool_manager as pool
from backend.app.providers import GeminiTranslationProvider


def _entry(**overrides) -> pool.PoolEntry:
    values = dict(id='e1', provider='gemini', model='gemini-3.6-flash', label='main')
    values.update(overrides)
    return pool.PoolEntry(**values)


@pytest.fixture(autouse=True)
def _reset_pool_breaker():
    """The 'pool is down' breaker lives at module scope; start each test clean."""
    from backend.app import providers

    providers._POOL_DOWN_UNTIL.clear()
    yield
    providers._POOL_DOWN_UNTIL.clear()


# ---------------------------------------------------------------------------
# Request shape: generation config survives the pool
# ---------------------------------------------------------------------------

def test_gemini_request_shape_forwards_temperature_and_thinking():
    url, body, needs_auth = pool._request_shape(
        _entry(), [{'role': 'user', 'content': 'hi'}], 256,
        {'temperature': 0.2, 'thinkingConfig': {'thinkingBudget': 0}},
    )
    assert url.endswith('/models/gemini-3.6-flash:generateContent')
    assert body['generationConfig']['temperature'] == 0.2
    assert body['generationConfig']['thinkingConfig'] == {'thinkingBudget': 0}
    assert body['generationConfig']['maxOutputTokens'] == 256
    assert needs_auth is False


def test_max_tokens_wins_over_a_conflicting_max_output_tokens():
    _, body, _ = pool._request_shape(
        _entry(), [{'role': 'user', 'content': 'hi'}], 100,
        {'maxOutputTokens': 999, 'temperature': 0.2},
    )
    assert body['generationConfig']['maxOutputTokens'] == 100


def test_openai_shaped_pool_forwards_temperature():
    _, body, needs_auth = pool._request_shape(
        _entry(provider='openai', model='gpt-4o-mini'), [{'role': 'user', 'content': 'hi'}], 64,
        {'temperature': 0.2, 'thinkingConfig': {'thinkingBudget': 0}},
    )
    assert body['temperature'] == 0.2
    assert body['max_tokens'] == 64
    assert 'thinkingConfig' not in body  # Gemini-only switch must not leak
    assert needs_auth is True


def test_pool_chat_forwards_config_and_returns_usage_and_server(monkeypatch):
    entry = _entry()
    monkeypatch.setattr(pool, 'plan', lambda provider: [type('C', (), {'entry': entry})()])
    seen: dict = {}

    def fake_call(entry, messages, *, max_tokens, timeout, generation_config=None, api_meta=None):
        seen['config'] = generation_config
        return '1|ខ្មែរ', {'tokens_in': 11, 'tokens_out': 22, 'thoughts': 0, 'chars_out': 3}

    monkeypatch.setattr(pool, '_call_entry', fake_call)
    monkeypatch.setattr(pool.health, 'record_success', lambda *a, **k: None)
    monkeypatch.setattr(pool.quota, 'record', lambda *a, **k: None)

    usage: dict[str, int] = {}
    served: dict = {}
    text = pool.chat(
        'gemini', [{'role': 'user', 'content': 'x'}], max_tokens=8,
        generation_config={'temperature': 0.2, 'thinkingConfig': {'thinkingBudget': 0}},
        usage_out=usage, served_out=served,
    )
    assert text == '1|ខ្មែរ'
    assert seen['config']['thinkingConfig'] == {'thinkingBudget': 0}
    assert usage == {'prompt_tokens': 11, 'candidates_tokens': 22, 'thoughts_tokens': 0}
    assert served['model'] == 'gemini-3.6-flash'


# ---------------------------------------------------------------------------
# Batch translation goes through the pool
# ---------------------------------------------------------------------------

def test_batch_uses_the_pool_when_the_engine_has_entries(monkeypatch):
    from backend.app import api_pool

    provider = GeminiTranslationProvider()
    calls: list[dict] = []

    monkeypatch.setattr(api_pool, 'has_entries', lambda name: True)

    def fake_chat(name, messages, **kwargs):
        calls.append({'name': name, 'messages': messages, **kwargs})
        kwargs['usage_out'].update({'prompt_tokens': 40, 'candidates_tokens': 12})
        return '1|សួស្តី'

    monkeypatch.setattr(api_pool, 'chat', fake_chat)
    usage: dict[str, int] = {}
    text = provider._batch_complete('SYSTEM', 'TRANSLATE:\n1|hello', 128, usage)

    assert text == '1|សួស្តី'
    assert calls[0]['name'] == 'gemini'
    assert calls[0]['messages'][0] == {'role': 'system', 'content': 'SYSTEM'}
    assert calls[0]['generation_config']['thinkingConfig'] == {'thinkingBudget': 0}
    assert calls[0]['generation_config']['temperature'] == 0.2
    assert usage['prompt_tokens'] == 40


def test_batch_falls_back_to_the_stored_key_when_the_pool_is_unavailable(monkeypatch):
    from backend.app import api_pool

    provider = GeminiTranslationProvider()
    monkeypatch.setattr(api_pool, 'has_entries', lambda name: True)

    def dead_chat(*args, **kwargs):
        raise api_pool.PoolUnavailable('every configuration is cooling down')

    monkeypatch.setattr(api_pool, 'chat', dead_chat)
    monkeypatch.setattr('backend.app.api_keys.has_key', lambda name: True)
    seen: dict = {}

    def fake_gemini_chat(key, messages, **kwargs):
        seen.update(kwargs)
        return '1|សួស្តី', {'usage': {'prompt_tokens': 9, 'candidates_tokens': 2, 'thoughts_tokens': 0},
                            'finish_reason': 'STOP', 'model': 'gemini-3.6-flash'}

    monkeypatch.setattr('backend.app.online_providers.gemini_chat', fake_gemini_chat)
    text = provider._batch_complete('SYSTEM', 'TRANSLATE:\n1|hello', 128, {})

    assert text == '1|សួស្តី'
    assert seen['system_instruction'] == 'SYSTEM'
    assert seen['generation_config']['thinkingConfig'] == {'thinkingBudget': 0}


def test_batch_raises_when_neither_pool_nor_key_can_serve(monkeypatch):
    from backend.app import api_pool

    provider = GeminiTranslationProvider()
    monkeypatch.setattr(api_pool, 'has_entries', lambda name: True)
    monkeypatch.setattr(
        api_pool, 'chat',
        lambda *a, **k: (_ for _ in ()).throw(api_pool.PoolUnavailable('parked')),
    )
    monkeypatch.setattr('backend.app.api_keys.has_key', lambda name: False)
    monkeypatch.setattr('backend.app.api_keys.get_key', lambda name: None)

    with pytest.raises(RuntimeError, match='pool is unavailable'):
        provider._batch_complete('SYSTEM', 'TRANSLATE:\n1|hello', 128, {})


# ---------------------------------------------------------------------------
# "Test connection" follows the runtime path
# ---------------------------------------------------------------------------

def test_provider_test_uses_the_pool_without_any_stored_key(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main.api_pool, 'entry_count', lambda provider: 2)
    monkeypatch.setattr(main.api_keys, 'has_key', lambda provider: False)

    def fake_pool_chat(provider, messages, **kwargs):
        kwargs['served_out'].update({'model': 'gemini-3.6-flash', 'entry_id': 'e1'})
        return 'OK'

    monkeypatch.setattr(main.api_pool, 'chat', fake_pool_chat)

    response = TestClient(main.app).post('/api/providers/gemini/test')
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['connected'] is True
    assert payload['via'] == 'pool'
    assert payload['model'] == 'gemini-3.6-flash'
    assert payload['sample'] == 'OK'


def test_provider_test_falls_back_to_the_key_and_reports_the_pool_error(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main.api_pool, 'entry_count', lambda provider: 1)
    monkeypatch.setattr(
        main.api_pool, 'chat',
        lambda *a, **k: (_ for _ in ()).throw(main.api_pool.PoolUnavailable('HTTP 503')),
    )
    monkeypatch.setattr(main.api_keys, 'has_key', lambda provider: True)
    monkeypatch.setattr(main.api_keys, 'get_key', lambda provider: 'k')
    monkeypatch.setattr(main, '_provider_model_lists', None, raising=False)

    async def fake_lists(provider, key):
        return {'models': ['gemini-tts']}, {'models': ['gemini-text']}

    monkeypatch.setattr(main, '_provider_model_lists', fake_lists)
    monkeypatch.setattr(
        'backend.app.online_providers.gemini_test', lambda key: {'models': ['gemini-tts']},
    )
    monkeypatch.setattr(
        'backend.app.online_providers.gemini_text_test', lambda key: {'models': ['gemini-text']},
    )

    response = TestClient(main.app).post('/api/providers/gemini/test')
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['via'] == 'key'
    assert 'HTTP 503' in payload['pool_error']
    assert payload['translation_models'] == ['gemini-text']
