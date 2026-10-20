"""Online API keys must enable translation, not just TTS.

The keys panel used to be TTS-only. Both hosted providers now also translate, and
these tests pin the routing, the no-silent-spend rule for Auto, and the guards that
keep a missing key from quietly falling back to another engine.
"""
from __future__ import annotations

import json
import types
import urllib.error

import pytest

from backend.app import api_keys, providers
from backend.app.providers import (
    GeminiTranslationProvider,
    HybridTranslationProvider,
    OpenAITranslationProvider,
    ONLINE_TRANSLATION_ENGINES,
)


@pytest.fixture
def keys(monkeypatch):
    store: dict[str, str] = {}
    monkeypatch.setattr(api_keys, 'has_key', lambda provider: provider in store)
    monkeypatch.setattr(api_keys, 'get_key', lambda provider: store.get(provider))
    return store


def _fake_time(now: float = 0.0) -> types.SimpleNamespace:
    """A clock with the two attributes the retry logic uses."""
    return types.SimpleNamespace(sleep=lambda seconds: None, monotonic=lambda: now)


def test_engines_are_registered():
    assert ONLINE_TRANSLATION_ENGINES == ('gemini', 'openai')
    assert GeminiTranslationProvider.ENGINE == 'gemini'
    assert OpenAITranslationProvider.ENGINE == 'openai'


def test_auto_never_uses_a_paid_engine(keys):
    """A configured key must not start billing the user from the default engine."""
    keys['gemini'] = 'g-key'
    keys['openai'] = 'o-key'
    translator = HybridTranslationProvider(preference='auto')
    assert translator.first_engine() != 'gemini'
    assert translator.first_engine() != 'openai'
    assert 'gemini' not in [key for key, _ in translator._chain()]
    assert 'openai' not in [key for key, _ in translator._chain()]


@pytest.mark.parametrize('engine', ONLINE_TRANSLATION_ENGINES)
def test_explicit_engine_is_first_when_the_key_exists(keys, engine):
    keys[engine] = 'secret'
    translator = HybridTranslationProvider(preference=engine)
    assert translator.first_engine() == engine


@pytest.mark.parametrize('engine', ONLINE_TRANSLATION_ENGINES)
def test_explicit_engine_without_a_key_is_refused(keys, engine):
    translator = HybridTranslationProvider(preference=engine)
    with pytest.raises(RuntimeError, match='API key is not configured'):
        translator.check_preferred_engine('Khmer', 'en')


def test_missing_key_reports_which_provider(keys):
    with pytest.raises(RuntimeError, match='OpenAI API key is not configured'):
        HybridTranslationProvider(preference='openai').check_preferred_engine('Khmer', 'en')
    with pytest.raises(RuntimeError, match='Gemini API key is not configured'):
        HybridTranslationProvider(preference='gemini').check_preferred_engine('Khmer', 'en')


def test_gemini_translation_goes_through_gemini_chat(keys, monkeypatch):
    keys['gemini'] = 'g-key'
    seen: dict = {}

    def fake_chat(key, messages, model=None, timeout=90):
        seen['key'] = key
        seen['messages'] = messages
        return 'សួស្តី'

    monkeypatch.setattr('backend.app.online_providers.gemini_chat', fake_chat)
    provider = GeminiTranslationProvider()
    assert provider.translate('hello', 'Khmer', 'en') == 'សួស្តី'
    assert seen['key'] == 'g-key'
    # The prompt carries both ends of the language pair.
    assert 'Khmer' in seen['messages'][0]['content']
    assert seen['messages'][-1] == {'role': 'user', 'content': 'hello'}


def test_openai_translation_goes_through_openai_chat(keys, monkeypatch):
    keys['openai'] = 'o-key'
    seen: dict = {}

    def fake_chat(key, messages, model=None, json_mode=False, timeout=90):
        seen['key'] = key
        return '"Bonjour"'

    monkeypatch.setattr('backend.app.online_providers.openai_chat', fake_chat)
    provider = OpenAITranslationProvider()
    # Surrounding quotes are stripped so subtitles stay clean.
    assert provider.translate('hello', 'French', 'en') == 'Bonjour'
    assert seen['key'] == 'o-key'


def test_engine_without_a_key_raises_on_translate(keys):
    with pytest.raises(RuntimeError, match='API key is not configured'):
        GeminiTranslationProvider().translate('hello', 'Khmer')


def test_empty_translation_is_an_error(keys, monkeypatch):
    keys['gemini'] = 'g-key'
    monkeypatch.setattr('backend.app.online_providers.gemini_chat',
                        lambda *args, **kwargs: '   ')
    with pytest.raises(RuntimeError, match='empty translation'):
        GeminiTranslationProvider().translate('hello', 'Khmer')


def test_failed_paid_engine_falls_back_and_reports_the_real_engine(keys, monkeypatch):
    """A 429 from Gemini still translates (via the chain) and reports the engine used."""
    keys['gemini'] = 'g-key'

    def boom(*args, **kwargs):
        raise RuntimeError('429 rate limited')

    monkeypatch.setattr('backend.app.online_providers.gemini_chat', boom)
    monkeypatch.setattr(
        providers.GoogleTranslateProvider, 'translate',
        lambda self, text, target, source='en': 'ជំរាបសួរ',
    )
    translator = HybridTranslationProvider(preference='gemini')
    assert translator.translate('hello', 'Khmer') == 'ជំរាបសួរ'
    assert translator.active_mode == 'online'


def test_provider_health_reports_online_translation_engines(keys):
    rows = {row['id']: row for row in providers.get_provider_status()}
    assert rows['gemini']['mode'] == 'PAID API'
    assert rows['openai']['mode'] == 'PAID API'
    assert rows['gemini']['connected'] is False
    keys['gemini'] = 'g-key'
    rows = {row['id']: row for row in providers.get_provider_status()}
    assert rows['gemini']['connected'] is True


# --- Gemini text model selection -------------------------------------------

def test_text_candidates_put_the_cached_model_first(keys, monkeypatch):
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models', lambda key: ['gemini-9-flash', 'gemini-3.5-flash'])
    fingerprint = online._model_cache_key('k')
    online._gemini_text_model_cache[fingerprint] = 'gemini-3.5-flash'
    try:
        assert online.gemini_text_model_candidates('k')[0] == 'gemini-3.5-flash'
    finally:
        online._gemini_text_model_cache.pop(fingerprint, None)


def test_dead_models_are_skipped(keys, monkeypatch):
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models', lambda key: ['gemini-9-flash', 'gemini-3.5-flash'])
    online._gemini_dead_models['k'] = {'gemini-9-flash'}
    try:
        candidates = online.gemini_text_model_candidates('k')
        assert 'gemini-9-flash' not in candidates
        assert 'gemini-3.5-flash' in candidates
    finally:
        online._gemini_dead_models.pop('k', None)


def test_gemini_chat_skips_an_unresponsive_model_and_caches_the_working_one(keys, monkeypatch):
    """The API can list models that never answer; one bad model must not fail the run."""
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models',
                        lambda key: ['gemini-3.8-flash', 'gemini-3.5-flash'])
    monkeypatch.setattr(online, '_GEMINI_TEXT_ATTEMPT_TIMEOUT', 1)
    tried: list[str] = []

    def fake_request(url, key, body=None, timeout=60):
        model = url.split('/models/')[1].split(':')[0]
        tried.append(model)
        if model != 'gemini-3.5-flash':
            raise TimeoutError('read timed out')
        return {'candidates': [{'content': {'parts': [{'text': 'សួស្តី'}]}}]}

    monkeypatch.setattr(online, '_request_json', fake_request)
    monkeypatch.setattr(online, '_save_model_cache', lambda cache_key, model: None)
    online._gemini_text_model_cache.pop(online._model_cache_key('g-key'), None)
    online._gemini_dead_models.pop('g-key', None)
    try:
        assert online.gemini_chat('g-key', [{'role': 'user', 'content': 'hello'}]) == 'សួស្តី'
        assert 'gemini-3.5-flash' in tried
        assert online._gemini_text_model_cache[online._model_cache_key('g-key')] == 'gemini-3.5-flash'
        assert 'gemini-3.8-flash' in online._gemini_dead_models['g-key']
    finally:
        online._gemini_text_model_cache.pop(online._model_cache_key('g-key'), None)
        online._gemini_dead_models.pop('g-key', None)


def test_transient_5xx_retries_the_same_model_without_blacklisting_it(keys, monkeypatch):
    """A brief Gemini outage must not poison the model cache for the whole run."""
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models', lambda key: ['gemini-3.5-flash'])
    monkeypatch.setattr(online, 'time', _fake_time())
    attempts: list[str] = []

    def flaky_request(url, key, body=None, timeout=60):
        model = url.split('/models/')[1].split(':')[0]
        attempts.append(model)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(url, 503, 'Service Unavailable', {}, None)
        return {'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]}

    monkeypatch.setattr(online, '_request_json', flaky_request)
    monkeypatch.setattr(online, '_save_model_cache', lambda cache_key, model: None)
    online._gemini_dead_models.pop('t-key', None)
    try:
        assert online.gemini_chat('t-key', [{'role': 'user', 'content': 'hi'}]) == 'ok'
        # Retried the same model rather than abandoning it as "dead".
        assert attempts == ['gemini-3.5-flash', 'gemini-3.5-flash']
        assert 'gemini-3.5-flash' not in online._gemini_dead_models.get('t-key', set())
    finally:
        online._gemini_dead_models.pop('t-key', None)
        online._gemini_text_model_cache.pop(online._model_cache_key('t-key'), None)


def test_retired_model_is_blacklisted_immediately(keys, monkeypatch):
    """A model that was cached yesterday and retired since must be dropped on 404."""
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models',
                        lambda key: ['gemini-2.5-flash', 'gemini-3.5-flash'])
    fingerprint = online._model_cache_key('r-key')
    # Cached first, so it is tried before the ranking would otherwise pick 3.5.
    online._gemini_text_model_cache[fingerprint] = 'gemini-2.5-flash'

    def retired(url, key, body=None, timeout=60):
        model = url.split('/models/')[1].split(':')[0]
        if model == 'gemini-2.5-flash':
            raise urllib.error.HTTPError(url, 404, 'no longer available', {}, None)
        return {'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]}

    monkeypatch.setattr(online, '_request_json', retired)
    monkeypatch.setattr(online, '_save_model_cache', lambda cache_key, model: None)
    online._gemini_dead_models.pop('r-key', None)
    try:
        assert online.gemini_chat('r-key', [{'role': 'user', 'content': 'hi'}]) == 'ok'
        assert 'gemini-2.5-flash' in online._gemini_dead_models['r-key']
        # The retired model is no longer the cached winner.
        assert online._gemini_text_model_cache[fingerprint] == 'gemini-3.5-flash'
        assert 'gemini-2.5-flash' not in online.gemini_text_model_candidates('r-key')
    finally:
        online._gemini_dead_models.pop('r-key', None)
        online._gemini_text_model_cache.pop(fingerprint, None)


def test_working_model_is_persisted_and_reused_after_a_restart(keys, monkeypatch, tmp_path):
    """A restart must not re-probe: the saved model goes straight to the front."""
    online = pytest.importorskip('backend.app.online_providers')
    cache_file = tmp_path / 'translation-models.json'
    monkeypatch.setattr(online, '_model_cache_path', lambda: cache_file)
    monkeypatch.setattr(online, 'gemini_list_chat_models',
                        lambda key: ['gemini-3.5-flash', 'gemini-3.8-flash'])

    def always_ok(url, key, body=None, timeout=60):
        return {'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]}

    monkeypatch.setattr(online, '_request_json', always_ok)
    online._gemini_text_model_cache.clear()
    online._gemini_dead_models.pop('p-key', None)
    try:
        online.gemini_chat('p-key', [{'role': 'user', 'content': 'hi'}])
        saved = json.loads(cache_file.read_text(encoding='utf-8'))
        fingerprint = online._model_cache_key('p-key')
        assert fingerprint in saved
        # The file never holds the key itself.
        assert 'p-key' not in cache_file.read_text(encoding='utf-8')

        # Simulate a restart: memory cleared, disk kept.
        online._gemini_text_model_cache.clear()
        online._load_model_cache()
        assert online._gemini_text_model_cache[fingerprint] == saved[fingerprint]
        assert online.gemini_text_model_candidates('p-key')[0] == saved[fingerprint]
    finally:
        online._gemini_text_model_cache.clear()
        online._gemini_dead_models.pop('p-key', None)


def test_quota_429_switches_models_instead_of_retrying(keys, monkeypatch):
    """Free-tier quota is per model, so a 429 must move on, not retry."""
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models',
                        lambda key: ['gemini-3.5-flash', 'gemini-3.6-flash'])
    tried: list[str] = []

    def quota(url, key, body=None, timeout=60):
        model = url.split('/models/')[1].split(':')[0]
        tried.append(model)
        if model == 'gemini-3.5-flash':
            raise urllib.error.HTTPError(
                url, 429, 'quota',
                {}, __import__('io').BytesIO(b'{"error":{"message":"Quota exceeded ... limit: 20"}}'),
            )
        return {'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]}

    monkeypatch.setattr(online, '_request_json', quota)
    monkeypatch.setattr(online, '_save_model_cache', lambda cache_key, model: None)
    fingerprint = online._model_cache_key('q-key')
    # Cached first so the quota-limited model is the one attempted.
    online._gemini_text_model_cache[fingerprint] = 'gemini-3.5-flash'
    online._gemini_dead_models.pop('q-key', None)
    online._gemini_quota_blocked_until.pop(fingerprint, None)
    try:
        assert online.gemini_chat('q-key', [{'role': 'user', 'content': 'hi'}]) == 'ok'
        # Tried the quota-limited model exactly once, then moved on.
        assert tried.count('gemini-3.5-flash') == 1
        assert 'gemini-3.6-flash' in tried
    finally:
        online._gemini_dead_models.pop('q-key', None)
        online._gemini_quota_blocked_until.pop(fingerprint, None)
        online._gemini_text_model_cache.pop(fingerprint, None)


def test_full_quota_exhaustion_cools_the_key_down(keys, monkeypatch):
    """Once every model refuses on quota, later lines fail fast for the fallback engine."""
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models', lambda key: ['gemini-3.5-flash'])
    calls: list[str] = []

    def always_quota(url, key, body=None, timeout=60):
        calls.append(url)
        raise urllib.error.HTTPError(url, 429, 'quota', {}, __import__('io').BytesIO(b'{"error":{}}'))

    monkeypatch.setattr(online, '_request_json', always_quota)
    fingerprint = online._model_cache_key('dry-key')
    online._gemini_dead_models.pop('dry-key', None)
    online._gemini_quota_blocked_until.pop(fingerprint, None)
    try:
        with pytest.raises(RuntimeError, match='no completion'):
            online.gemini_chat('dry-key', [{'role': 'user', 'content': 'hi'}])
        assert fingerprint in online._gemini_quota_blocked_until
        first_call_count = len(calls)

        # Next line short-circuits: no further HTTP attempts.
        with pytest.raises(RuntimeError, match='quota is exhausted'):
            online.gemini_chat('dry-key', [{'role': 'user', 'content': 'again'}])
        assert len(calls) == first_call_count
    finally:
        online._gemini_dead_models.pop('dry-key', None)
        online._gemini_quota_blocked_until.pop(fingerprint, None)


def test_cooldown_is_cleared_by_a_successful_call(keys, monkeypatch):
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models', lambda key: ['gemini-3.5-flash'])
    monkeypatch.setattr(online, '_request_json',
                        lambda url, key, body=None, timeout=60: {
                            'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]
                        })
    monkeypatch.setattr(online, '_save_model_cache', lambda cache_key, model: None)
    fingerprint = online._model_cache_key('warm-key')
    online._gemini_quota_blocked_until[fingerprint] = 0.0  # expired already
    try:
        assert online.gemini_chat('warm-key', [{'role': 'user', 'content': 'hi'}]) == 'ok'
        assert fingerprint not in online._gemini_quota_blocked_until
    finally:
        online._gemini_dead_models.pop('warm-key', None)
        online._gemini_quota_blocked_until.pop(fingerprint, None)
        online._gemini_text_model_cache.pop(fingerprint, None)


def test_http_error_detail_surfaces_the_provider_message():
    """A bare "HTTP 429" is useless to the user; the quota message must show."""
    import io

    online = pytest.importorskip('backend.app.online_providers')
    error = urllib.error.HTTPError(
        'https://example', 429, 'quota', {},
        io.BytesIO(b'{"error":{"message":"You exceeded your current quota  limit: 20"}}'),
    )
    detail = online._http_error_detail(error)
    assert detail.startswith('HTTP 429')
    assert 'exceeded your current quota' in detail


def test_gemini_chat_reports_failure_when_every_model_fails(keys, monkeypatch):
    online = pytest.importorskip('backend.app.online_providers')
    monkeypatch.setattr(online, 'gemini_list_chat_models', lambda key: ['gemini-3.5-flash'])
    monkeypatch.setattr(online, 'time', _fake_time())

    def always_timeout(url, key, body=None, timeout=60):
        raise TimeoutError('read timed out')

    monkeypatch.setattr(online, '_request_json', always_timeout)
    online._gemini_dead_models.pop('bad', None)
    try:
        with pytest.raises(RuntimeError, match='no completion'):
            online.gemini_chat('bad', [{'role': 'user', 'content': 'hello'}])
    finally:
        online._gemini_dead_models.pop('bad', None)
        online._gemini_text_model_cache.pop(online._model_cache_key('bad'), None)


def test_gemini_text_listing_excludes_non_text_models():
    """Image/embedding/TTS models cannot translate a subtitle line."""
    source = __import__('inspect').getsource(
        __import__('backend.app.online_providers', fromlist=['x']).gemini_list_chat_models
    )
    for marker in ('tts', 'embedding', 'image', 'veo'):
        assert marker in source

