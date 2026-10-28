"""Online API keys must enable translation, not just TTS.

The keys panel used to be TTS-only. Both hosted providers now also translate, and
these tests pin the routing, the no-silent-spend rule for Auto, and the guards that
keep a missing key from quietly falling back to another engine.
"""
from __future__ import annotations

import json
import sqlite3
import types
import urllib.error

import pytest

from backend.app import api_keys, providers
from backend.app.providers import (
    CodeCraftTranslationProvider,
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
    # The model catalog is cached per key in production; tests swap the patched
    # catalogs between cases, so start each one cold.
    from backend.app import online_providers

    monkeypatch.setattr(online_providers, '_gemini_catalog_cache', {})
    return store


def _fake_time(now: float = 0.0) -> types.SimpleNamespace:
    """A clock with the two attributes the retry logic uses."""
    return types.SimpleNamespace(sleep=lambda seconds: None, monotonic=lambda: now)


def test_engines_are_registered():
    assert ONLINE_TRANSLATION_ENGINES == ('gemini', 'openai', 'codecraft')
    assert GeminiTranslationProvider.ENGINE == 'gemini'
    assert OpenAITranslationProvider.ENGINE == 'openai'
    assert CodeCraftTranslationProvider.ENGINE == 'codecraft'


def test_auto_never_uses_a_keyed_engine(keys):
    """A configured key must not start billing or spending quota from Auto."""
    keys['gemini'] = 'g-key'
    keys['openai'] = 'o-key'
    keys['codecraft'] = 'cc-key'
    translator = HybridTranslationProvider(preference='auto')
    for engine in ONLINE_TRANSLATION_ENGINES:
        assert translator.first_engine() != engine
        assert engine not in [key for key, _ in translator._chain()]


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
    with pytest.raises(RuntimeError, match='CodeCraft API key is not configured'):
        HybridTranslationProvider(preference='codecraft').check_preferred_engine('Khmer', 'en')


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


def test_codecraft_translation_goes_through_codecraft_chat(keys, monkeypatch):
    keys['codecraft'] = 'cc-key'
    seen: dict = {}

    def fake_chat(key, messages, model=None, timeout=90):
        seen['key'] = key
        seen['model'] = model
        seen['messages'] = messages
        return 'សួស្តី'

    monkeypatch.setattr('backend.app.online_providers.codecraft_chat', fake_chat)
    provider = CodeCraftTranslationProvider()
    assert provider.translate('hello', 'Khmer', 'en') == 'សួស្តី'
    assert seen['key'] == 'cc-key'
    # The same dubbing prompt as the other hosted engines.
    assert 'Khmer' in seen['messages'][0]['content']
    assert seen['messages'][-1] == {'role': 'user', 'content': 'hello'}


def test_engine_without_a_key_raises_on_translate(keys):
    with pytest.raises(RuntimeError, match='API key is not configured'):
        GeminiTranslationProvider().translate('hello', 'Khmer')
    with pytest.raises(RuntimeError, match='CodeCraft API key is not configured'):
        CodeCraftTranslationProvider().translate('hello', 'Khmer')


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


# --- Wrong-script protection (Khmer must never be dubbed from English text) --

def test_script_validation_rules():
    """The validator accepts real translations and rejects wrong-script answers."""
    assert providers._translation_script_ok('សួស្តី', 'hello', 'en', 'Khmer')
    assert not providers._translation_script_ok('hello', 'hello', 'en', 'Khmer')
    assert not providers._translation_script_ok('Sorry, I cannot help with that.', 'hello', 'en', 'Khmer')
    # A brand name inside a real Khmer translation stays acceptable.
    assert providers._translation_script_ok('សួស្តី OK', 'hello', 'en', 'Khmer')
    # Latin-script targets are never script-checked.
    assert providers._translation_script_ok('Bonjour', 'hello', 'en', 'French')
    # Pure numbers/punctuation have nothing to verify.
    assert providers._translation_script_ok('1234', '1234', 'en', 'Khmer')


def test_wrong_script_answer_costs_exactly_one_call(keys, monkeypatch):
    """Normal cost: one engine call per line — no corrective second request."""
    keys['gemini'] = 'g-key'
    calls: list[int] = []

    def fake_chat(key, messages, model=None, timeout=90):
        calls.append(1)
        return 'Sorry, I cannot help with that.'

    monkeypatch.setattr('backend.app.online_providers.gemini_chat', fake_chat)
    with pytest.raises(RuntimeError, match='did not return Khmer'):
        GeminiTranslationProvider().translate('hello', 'Khmer', 'en')
    assert calls == [1]  # no retry: the chain will try the next engine instead


def test_engine_that_never_returns_khmer_fails_over_to_the_next_engine(keys, monkeypatch):
    """Two wrong-script answers raise, and the chain dubs with the next engine."""
    keys['gemini'] = 'g-key'
    monkeypatch.setattr(
        'backend.app.online_providers.gemini_chat',
        lambda *args, **kwargs: 'This is English, not Khmer.',
    )
    monkeypatch.setattr(
        providers.GoogleTranslateProvider, 'translate',
        lambda self, text, target, source='en': 'ជំរាបសួរ',
    )
    translator = HybridTranslationProvider(preference='gemini')
    assert translator.translate('hello', 'Khmer') == 'ជំរាបសួរ'
    assert translator.active_mode == 'online'


def test_echoed_source_is_rejected_for_khmer(keys, monkeypatch):
    """The English line coming back unchanged is never accepted as a Khmer dub."""
    keys['codecraft'] = 'cc-key'
    monkeypatch.setattr('backend.app.online_providers.codecraft_chat', lambda *a, **k: 'hello')
    with pytest.raises(RuntimeError, match='did not return Khmer'):
        CodeCraftTranslationProvider().translate('hello', 'Khmer', 'en')


# --- Reasoning-dump pollution (analysis must never become the subtitle) -------

_REASONING_DUMP = (
    'The user wants a translation for a fantasy drama subtitle line. '
    'Breakdown of each word, then candidates:\n'
    '*   Option 1 (Formal/Epic): ថាមពលឈាមនៃត្រកូលទាំងមូលក៏នឹងរលត់អស់ដែរ។\n'
    '*   Option 2 (More concise/Spoken): កម្លាំងឈាមនៃត្រកូលទាំងមូលនឹងត្រូវអស់រលីងដែរ។\n'
    'Let me refine for natural dialogue and pick the best balance:\n'
    '*   Selected Translation: ថាមពលឈាមនៃត្រកូលទាំងមូលក៏នឹងរលត់អស់ដែរ។'
)


def test_reasoning_dump_is_salvaged_to_the_final_line(keys, monkeypatch):
    """An analysis essay still contains one clean answer line — use it, no retry."""
    keys['codecraft'] = 'cc-key'
    calls: list[int] = []

    def fake_chat(key, messages, model=None, timeout=90):
        calls.append(1)
        return _REASONING_DUMP

    monkeypatch.setattr('backend.app.online_providers.codecraft_chat', fake_chat)
    provider = CodeCraftTranslationProvider()
    result = provider.translate('其全族血脉之力也将枯竭', 'Khmer', 'zh')
    assert result == 'ថាមពលឈាមនៃត្រកូលទាំងមូលក៏នឹងរលត់អស់ដែរ។'
    assert calls == [1]  # salvaged from the first answer, no retry spent


def test_reasoning_tag_is_stripped_before_use(keys, monkeypatch):
    keys['gemini'] = 'g-key'
    monkeypatch.setattr(
        'backend.app.online_providers.gemini_chat',
        lambda *args, **kwargs: 'Khmer greeting is suosdey.</think>"សួស្តី"',
    )
    assert GeminiTranslationProvider().translate('hello', 'Khmer', 'en') == 'សួស្តី'


def test_doubled_final_line_is_collapsed(keys, monkeypatch):
    """'A។A។' must never reach the dub audio as one doubled sentence."""
    keys['codecraft'] = 'cc-key'
    doubled = 'ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។'
    monkeypatch.setattr('backend.app.online_providers.codecraft_chat', lambda *a, **k: doubled)
    result = CodeCraftTranslationProvider().translate('其全族血脉之力也将枯竭', 'Khmer', 'zh')
    assert result == 'ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។'


def test_pure_analysis_with_no_answer_is_refused(keys, monkeypatch):
    """A dump that contains no actual translation fails over to the next engine."""
    keys['gemini'] = 'g-key'
    monkeypatch.setattr(
        'backend.app.online_providers.gemini_chat',
        lambda *args, **kwargs: (
            'The user wants a translation for a fantasy drama subtitle line.\n'
            'Source text (zh): 其全族血脉之力也将枯竭\n'
            'Meaning: The bloodline power of their entire clan will be exhausted.\n'
            '*   Option 1 (Formal/Epic): [need khmer wording]\n'
            '*   Option 2 (More concise/Spoken): [need khmer wording]\n'
            'Selected Translation: [need khmer wording]'
        ),
    )
    monkeypatch.setattr(
        providers.GoogleTranslateProvider, 'translate',
        lambda self, text, target, source='zh': 'ថាមពលឈាមនៃត្រកូលទាំងមូលនឹងអស់រលីង។',
    )
    translator = HybridTranslationProvider(preference='gemini')
    assert translator.translate('其全族血脉之力也将枯竭', 'Khmer') == 'ថាមពលឈាមនៃត្រកូលទាំងមូលនឹងអស់រលីង។'
    assert translator.active_mode == 'online'


def test_short_cjk_name_romanization_is_accepted():
    """A character name has no Khmer translation; romanizing it must not fail the run."""
    assert providers._translation_script_ok('Xiao Yan', '蕭顏', 'zh', 'Khmer')
    assert providers._translation_script_ok('Xueer?', '薛儿吗', 'zh', 'Khmer')
    # Names kept in Latin next to real Khmer text are normal subtitle style.
    assert providers._translation_script_ok(
        'Jiuwen Thunder Eagle Group បានធំឡើង', '九闻雷鹰组长大了', 'zh', 'Khmer',
    )
    # An English sentence is not a name and is still refused.
    assert not providers._translation_script_ok(
        'Sorry, I cannot help with that.', '你好吗', 'zh', 'Khmer',
    )


def test_subtitle_length_guard_rules():
    source = '其全族血脉之力也将枯竭'
    assert providers._subtitle_length_ok('ថាមពលឈាមនៃត្រកូលទាំងមូលក៏នឹងរលត់អស់ដែរ។', source)
    assert not providers._subtitle_length_ok('word ' * 80, source)


def test_doubled_sentences_collapse_helper():
    assert providers._collapse_duplicate_sentences(
        'ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។'
    ) == 'ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។'
    # A single sentence passes through untouched.
    assert providers._collapse_duplicate_sentences(
        'ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។'
    ) == 'ថាមពលឈាមនៃត្រកូលទាំងមូល ក៏នឹងរលត់អស់ដែរ។'


def test_provider_health_reports_online_translation_engines(keys):
    rows = {row['id']: row for row in providers.get_provider_status()}
    assert rows['gemini']['mode'] == 'PAID API'
    assert rows['openai']['mode'] == 'PAID API'
    assert rows['codecraft']['mode'] == 'FREE ONLINE'
    assert rows['gemini']['connected'] is False
    assert rows['codecraft']['connected'] is False
    keys['gemini'] = 'g-key'
    keys['codecraft'] = 'cc-key'
    rows = {row['id']: row for row in providers.get_provider_status()}
    assert rows['gemini']['connected'] is True
    assert rows['codecraft']['connected'] is True


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
        with pytest.raises(RuntimeError, match='not usable for this key'):
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
        online._gemini_quota_blocked_until.pop(online._model_cache_key('bad'), None)


def test_gemini_catalog_is_listed_once_for_many_lines(keys, monkeypatch):
    """One request per line: the model catalog is cached, not re-fetched per line."""
    online = pytest.importorskip('backend.app.online_providers')
    listings: list[str] = []
    completions: list[str] = []

    def fake_request(url, key, body=None, timeout=60):
        if body is None:
            listings.append(url)
            return {'models': [
                {'name': 'models/gemini-3.5-flash',
                 'supportedGenerationMethods': ['generateContent']},
            ]}
        completions.append(url)
        return {'candidates': [{'content': {'parts': [{'text': 'សួស្តី'}]}}]}

    monkeypatch.setattr(online, '_request_json', fake_request)
    monkeypatch.setattr(online, '_save_model_cache', lambda *args, **kwargs: None)
    fingerprint = online._model_cache_key('catalog-key')
    online._gemini_dead_models.pop('catalog-key', None)
    online._gemini_quota_blocked_until.pop(fingerprint, None)
    online._gemini_text_model_cache.pop(fingerprint, None)
    for _ in range(5):
        assert online.gemini_chat('catalog-key', [{'role': 'user', 'content': 'hi'}]) == 'សួស្តី'
    assert len(completions) == 5
    assert len(listings) == 1, 'the catalog GET must be reused across lines'


def test_failed_model_sweep_parks_the_key_for_the_next_line(keys, monkeypatch):
    """A key whose models all reject the call must not cost a 4-request sweep per line."""
    online = pytest.importorskip('backend.app.online_providers')
    calls: list[str] = []

    def fake_request(url, key, body=None, timeout=60):
        calls.append(url)
        if body is None:
            return {'models': [
                {'name': f'models/gemini-3.{index}-flash',
                 'supportedGenerationMethods': ['generateContent']}
                for index in range(3)
            ]}
        raise urllib.error.HTTPError(url, 400, 'bad model', {}, __import__('io').BytesIO(b'{}'))

    monkeypatch.setattr(online, '_request_json', fake_request)
    monkeypatch.setattr(online, '_save_model_cache', lambda *args, **kwargs: None)
    fingerprint = online._model_cache_key('reject-key')
    online._gemini_dead_models.pop('reject-key', None)
    online._gemini_quota_blocked_until.pop(fingerprint, None)
    online._gemini_text_model_cache.pop(fingerprint, None)
    try:
        with pytest.raises(RuntimeError, match='no completion'):
            online.gemini_chat('reject-key', [{'role': 'user', 'content': 'one'}])
        after_first_line = len(calls)
        # One catalog GET + the three listed models + the default alias = 5.
        assert after_first_line == 5
        assert len(calls) <= 6, 'one bounded sweep, not a sweep per model twice'
        with pytest.raises(RuntimeError, match='not usable for this key'):
            online.gemini_chat('reject-key', [{'role': 'user', 'content': 'two'}])
        assert len(calls) == after_first_line, 'the next line must cost zero requests'
    finally:
        online._gemini_dead_models.pop('reject-key', None)
        online._gemini_quota_blocked_until.pop(fingerprint, None)


def test_identical_lines_share_one_translation_call(tmp_path, monkeypatch):
    """Repeated dialogue is translated once and reused — not paid for line by line."""
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'PROJECT_DB', tmp_path / 'data' / 'projects.sqlite3')
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(tmp_path / 'data' / 'projects.sqlite3') as connection:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL)'
        )
    calls: list[str] = []

    class FakeTranslator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_preferred_engine(self, *args, **kwargs) -> None:
            return None

        def first_engine(self) -> str:
            return 'gemini'

        def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
            calls.append(text)
            return f'{text}!'

    monkeypatch.setattr(main, 'HybridTranslationProvider', FakeTranslator)
    project_id = 'dedupe-translate'
    main.projects[project_id] = {
        'id': project_id, 'name': 'd', 'mode': 'DUB', 'media_path': None,
        'duration': '00:02:00', 'source_language': 'en', 'target_language': 'Khmer',
        'segments': [
            {'id': 'a', 'start': 0.0, 'end': 1.0, 'text': 'Yes.'},
            {'id': 'b', 'start': 1.0, 'end': 2.0, 'text': 'Yes.'},
            {'id': 'c', 'start': 2.0, 'end': 3.0, 'text': 'Come on.'},
        ],
    }
    try:
        response = TestClient(main.app).post(
            '/api/process/translate',
            json={'project_id': project_id, 'target_language': 'Khmer'},
        )
        assert response.status_code == 200, response.text
        chunks = response.json()['translation']
    finally:
        main.projects.pop(project_id, None)

    assert sorted(calls) == ['Come on.', 'Yes.']
    assert [chunk['translation'] for chunk in chunks] == ['Yes.!', 'Yes.!', 'Come on.!']


def test_gemini_text_listing_excludes_non_text_models():
    """Image/embedding/TTS models cannot translate a subtitle line."""
    source = __import__('inspect').getsource(
        __import__('backend.app.online_providers', fromlist=['x']).gemini_list_chat_models
    )
    for marker in ('tts', 'embedding', 'image', 'veo'):
        assert marker in source

