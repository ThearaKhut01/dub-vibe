"""Batch translation: grouping, parsing, retries, cache and the route.

These tests never touch a network: the batch text is produced by a fake
completion, exactly as a real engine's answer would be.
"""
from __future__ import annotations

import sqlite3

import pytest

from backend.app import providers, translation_memory
from backend.app.providers import (
    GeminiTranslationProvider,
    HybridTranslationProvider,
    build_translation_system_prompt,
    build_translation_user_message,
    batch_output_token_budget,
    parse_batch_reply,
    split_translation_batches,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_batches_respect_line_cap():
    texts = [f'line {index}' for index in range(95)]
    batches = split_translation_batches(texts)
    assert [len(batch) for batch in batches] == [40, 40, 15]
    assert [text for batch in batches for text in batch] == texts


def test_batches_respect_character_cap_and_keep_order():
    texts = ['a' * 1200, 'b' * 1200, 'c' * 1200]
    batches = split_translation_batches(texts)
    # 1200 + 8 per line: only two fit under 3000 characters.
    assert [len(batch) for batch in batches] == [2, 1]
    assert batches[0] == texts[:2]


def test_oversized_line_still_gets_its_own_batch():
    batches = split_translation_batches(['x' * 9000, 'short'])
    assert batches == [['x' * 9000], ['short']]


def test_parse_accepts_fences_prose_and_keeps_last_duplicate():
    raw = (
        'Here are the translations:\n'
        '```\n'
        '1|សួស្តី\n'
        '2| លា \n'
        '2|លាហើយ\n'
        '3|\n'
        'not a line\n'
        '```'
    )
    parsed = parse_batch_reply(raw)
    assert parsed == {1: 'សួស្តី', 2: 'លាហើយ'}


def test_parse_filters_ids_outside_the_batch():
    assert parse_batch_reply('1|a\n2|b', {2}) == {2: 'b'}


def test_user_message_has_context_block_and_numbered_lines():
    message = build_translation_user_message(['Hello', 'Bye'], context=['A', 'B', 'C', 'D'])
    assert message.startswith('CONTEXT:\nB\nC\nD\n\nTRANSLATE:')
    assert '1|Hello' in message and '2|Bye' in message


def test_user_message_without_context_says_none():
    assert build_translation_user_message(['Hi']).startswith('CONTEXT:\nnone')


def test_prompt_contains_exact_rules_and_glossary():
    prompt = build_translation_system_prompt('王明=វ៉ាង មីង')
    assert 'Khmer (ភាសាខ្មែរ)' in prompt
    assert 'English or Chinese (auto-detect per line)' in prompt
    assert '王明=វ៉ាង មីង' in prompt
    assert 'Never merge, split, skip, or add lines' in prompt
    # The empty glossary must read as "none", never as a blank section.
    assert 'Glossary (use these exact translations):\nnone' in build_translation_system_prompt('')


def test_output_budget_is_bounded():
    small = batch_output_token_budget(build_translation_user_message(['hi']), 1)
    huge = batch_output_token_budget('x' * 100000, 40)
    assert 512 <= small <= providers.TRANSLATION_OUTPUT_TOKEN_CAP
    assert huge == providers.TRANSLATION_OUTPUT_TOKEN_CAP


def test_gemini_model_comes_from_settings(monkeypatch):
    from backend.app import model_manager

    monkeypatch.setattr(model_manager, 'load_settings', lambda: {'gemini_translation_model': 'gemini-2.5-flash-lite'})
    assert GeminiTranslationProvider.model_name() == 'gemini-2.5-flash-lite'
    monkeypatch.setattr(model_manager, 'load_settings', lambda: {})
    assert GeminiTranslationProvider.model_name() == 'gemini-2.5-flash'


# ---------------------------------------------------------------------------
# translate_batch: retries only the missing ids
# ---------------------------------------------------------------------------

class _FakeLLM(GeminiTranslationProvider):
    """Answers like a model would, and records every request it receives."""

    def __init__(self, replies):  # noqa: D401 - test double
        super().__init__()
        self.replies = list(replies)
        self.requests: list[str] = []
        self.usage_calls: list[dict] = []

    def _key(self) -> str:
        return 'fake-key'

    def _batch_complete(self, system, user, max_tokens, usage=None, meta=None) -> str:
        self.requests.append(user)
        if usage is not None:
            usage['prompt_tokens'] = usage.get('prompt_tokens', 0) + 100
            usage['candidates_tokens'] = usage.get('candidates_tokens', 0) + 40
        return self.replies.pop(0)


def test_batch_retries_only_missing_lines(monkeypatch):
    monkeypatch.setattr(providers.time, 'sleep', lambda *_: None)
    fake = _FakeLLM([
        '1|សួស្តី\n3|លាហើយ',          # 2 is missing
        '2|អរគុណ',                      # retry carries ONLY the missing line
    ])
    usage: dict[str, int] = {}
    result = fake.translate_batch(
        ['Hello', 'Thanks', 'Bye'], context=[], target_language='Khmer',
        source_language='en', usage=usage,
    )
    assert result == ['សួស្តី', 'អរគុណ', 'លាហើយ']
    assert len(fake.requests) == 2
    assert '2|Thanks' in fake.requests[1]
    assert 'Hello' not in fake.requests[1] and 'Bye' not in fake.requests[1]
    assert usage['requests'] == 2 and usage['retries'] == 1
    assert usage['prompt_tokens'] == 200


def test_batch_gives_up_after_two_retries(monkeypatch):
    monkeypatch.setattr(providers.time, 'sleep', lambda *_: None)
    fake = _FakeLLM(['1|សួស្តី', '', ''])
    result = fake.translate_batch(['Hello', 'Thanks'], target_language='Khmer', source_language='en')
    assert result == ['សួស្តី', None]
    assert len(fake.requests) == 3  # first try + two retries, never the whole batch again


def test_truncated_answer_is_retried_in_smaller_groups(monkeypatch):
    """A cut-off answer must not be re-sent at the same size every time."""
    import re

    monkeypatch.setattr(providers.time, 'sleep', lambda *_: None)
    lines = [f'line {index}' for index in range(12)]
    sizes: list[int] = []

    class Fake(_FakeLLM):
        def __init__(self) -> None:
            super().__init__([])

        def _batch_complete(self, system, user, max_tokens, usage=None, meta=None) -> str:
            identifiers = re.findall(r'(?m)^(\d+)\|', user)
            sizes.append(len(identifiers))
            if len(sizes) == 1:
                return '1|ភាសា1\n2|ភាសា2'  # truncated: 2 of 12
            return '\n'.join(f'{identifier}|ភាសា{identifier}' for identifier in identifiers)

    fake = Fake()
    result = fake.translate_batch(lines, target_language='Khmer', source_language='en')
    assert all(value for value in result)
    assert sizes[0] == 12
    assert sizes[1] <= 6, sizes


def test_batch_rejects_the_source_line(monkeypatch):
    monkeypatch.setattr(providers.time, 'sleep', lambda *_: None)
    fake = _FakeLLM(['1|Hello', '1|Hello', '1|Hello'])
    result = fake.translate_batch(['Hello'], target_language='Khmer', source_language='en')
    assert result == [None]


def test_hybrid_skips_engines_without_batch_support(monkeypatch):
    class OnlyGoogle:
        def translate(self, text, target_language, source_language='en'):
            return text

        def supports_batch(self) -> bool:
            return False

    hybrid = HybridTranslationProvider(preference='auto')
    monkeypatch.setattr(hybrid, '_chain', lambda: [('google', OnlyGoogle())])
    assert hybrid.supports_batch() is False


# ---------------------------------------------------------------------------
# Translation memory
# ---------------------------------------------------------------------------

def test_cache_key_changes_with_model_glossary_and_text():
    base = translation_memory.cache_key('Hello', 'en', 'Khmer', 'gemini-2.5-flash', 'A=B')
    assert base != translation_memory.cache_key('Hello', 'en', 'Khmer', 'other-model', 'A=B')
    assert base != translation_memory.cache_key('Hello', 'en', 'Khmer', 'gemini-2.5-flash', 'A=C')
    assert base != translation_memory.cache_key('Hello', 'en', 'Khmer', 'gemini-2.5-flash')
    assert base != translation_memory.cache_key('Hello', 'fr', 'Khmer', 'gemini-2.5-flash', 'A=B')
    assert base == translation_memory.cache_key('  Hello ', 'en', 'Khmer', 'gemini-2.5-flash', 'A=B')


def test_cache_round_trip_without_any_api_pool_entry():
    translation_memory.remember('Hello', 'សួស្តី', 'en', 'Khmer', 'gemini-2.5-flash', '')
    assert translation_memory.lookup('Hello', 'en', 'Khmer', 'gemini-2.5-flash', '') == 'សួស្តី'
    # Another model must not inherit the wording.
    assert translation_memory.lookup('Hello', 'en', 'Khmer', 'mistral-large') is None


# ---------------------------------------------------------------------------
# Route: one request for the whole (deduped) project
# ---------------------------------------------------------------------------

def _prepare_project(monkeypatch, tmp_path, texts: list[str], glossary: str = ''):
    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'PROJECT_DB', tmp_path / 'data' / 'projects.sqlite3')
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(tmp_path / 'data' / 'projects.sqlite3') as connection:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL)'
        )
    project_id = 'batch-translate'
    main.projects[project_id] = {
        'id': project_id, 'name': 'batch', 'mode': 'DUB', 'media_path': None,
        'duration': '00:02:00', 'source_language': 'en', 'target_language': 'Khmer',
        'glossary': glossary,
        'segments': [
            {'id': f's{index}', 'start': float(index), 'end': float(index) + 0.9, 'text': text}
            for index, text in enumerate(texts)
        ],
    }
    return main, project_id


def test_translate_route_batches_and_dedupes(monkeypatch, tmp_path):
    texts = ['Hello', 'Hello', 'Bye', 'See you', 'Bye']
    main, project_id = _prepare_project(monkeypatch, tmp_path, texts, glossary='Bye=លាហើយ')
    calls: list[dict] = []

    class FakeTranslator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_preferred_engine(self, *args, **kwargs) -> None:
            return None

        def first_engine(self) -> str:
            return 'gemini'

        def supports_batch(self) -> bool:
            return True

        def translate_batch(self, lines, context=None, glossary='', *, source_language='auto',
                            target_language='Khmer', tone='natural and consistent', usage=None):
            calls.append({'lines': list(lines), 'context': list(context or []), 'glossary': glossary, 'tone': tone})
            table = {'Hello': 'សួស្តី', 'Bye': 'លាហើយ', 'See you': 'ជួបគ្នាថ្ងៃក្រោយ'}
            if usage is not None:
                usage['requests'] = 1
                usage['prompt_tokens'] = 300
                usage['candidates_tokens'] = 90
                usage['thoughts_tokens'] = 0
            return [table[line] for line in lines]

        def translate(self, text, target_language, source_language='en'):
            raise AssertionError('the fallback path must not run here')

    monkeypatch.setattr(main, 'HybridTranslationProvider', FakeTranslator)
    try:
        from fastapi.testclient import TestClient

        response = TestClient(main.app).post(
            '/api/process/translate',
            json={'project_id': project_id, 'target_language': 'Khmer', 'emotion': 'Happy'},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
    finally:
        main.projects.pop(project_id, None)

    assert len(calls) == 1, 'five segments must cost one request'
    assert calls[0]['lines'] == ['Hello', 'Bye', 'See you']
    assert calls[0]['glossary'] == 'Bye=លាហើយ'
    assert calls[0]['tone'].startswith('happy')
    assert [item['translation'] for item in payload['translation']] == [
        'សួស្តី', 'សួស្តី', 'លាហើយ', 'ជួបគ្នាថ្ងៃក្រោយ', 'លាហើយ',
    ]
    assert payload['usage']['batches'] == 1
    assert payload['usage']['lines'] == 3
    assert payload['usage']['prompt_tokens'] == 300
    assert payload['usage']['candidates_tokens'] == 90


def test_second_run_is_served_from_cache(monkeypatch, tmp_path):
    main, project_id = _prepare_project(monkeypatch, tmp_path, ['Hello', 'Bye'])
    calls: list[list[str]] = []

    class FakeTranslator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_preferred_engine(self, *args, **kwargs) -> None:
            return None

        def first_engine(self) -> str:
            return 'gemini'

        def supports_batch(self) -> bool:
            return True

        def translate_batch(self, lines, context=None, glossary='', *, source_language='auto',
                            target_language='Khmer', tone='natural and consistent', usage=None):
            calls.append(list(lines))
            return ['សួស្តី' if line == 'Hello' else 'លាហើយ' for line in lines]

        def translate(self, text, target_language, source_language='en'):
            raise AssertionError('no single-line call expected')

    monkeypatch.setattr(main, 'HybridTranslationProvider', FakeTranslator)
    try:
        from fastapi.testclient import TestClient

        client = TestClient(main.app)
        first = client.post('/api/process/translate', json={'project_id': project_id, 'target_language': 'Khmer'})
        assert first.status_code == 200, first.text
        # Clear the translations, keep the cache: a re-run must not call the model.
        for segment in main.projects[project_id]['segments']:
            segment['translation'] = ''
        second = client.post('/api/process/translate', json={'project_id': project_id, 'target_language': 'Khmer'})
        assert second.status_code == 200, second.text
        payload = second.json()
    finally:
        main.projects.pop(project_id, None)

    assert len(calls) == 1, 'the second run must read the cache, not the API'
    assert payload['usage']['cache_hits'] == 2
    assert [item['translation'] for item in payload['translation']] == ['សួស្តី', 'លាហើយ']


def test_glossary_route_round_trip(monkeypatch, tmp_path):
    main, project_id = _prepare_project(monkeypatch, tmp_path, ['Hello'])
    try:
        from fastapi.testclient import TestClient

        client = TestClient(main.app)
        saved = client.post(f'/api/projects/{project_id}/glossary', json={'glossary': 'Hello=សួស្តី'})
        assert saved.status_code == 200, saved.text
        fetched = client.get(f'/api/projects/{project_id}/glossary')
        assert fetched.json()['glossary'] == 'Hello=សួស្តី'
    finally:
        main.projects.pop(project_id, None)


def test_gemini_inlines_the_system_prompt_when_a_model_rejects_it(monkeypatch):
    """Some served models answer 400 to systemInstruction; retry with it inline."""
    import io
    import urllib.error

    from backend.app import online_providers

    calls: list[dict] = []

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        calls.append(body)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                url, 400, 'Bad Request', {},
                io.BytesIO(b'{"error":{"message":"Developer instruction is not enabled for this model"}}'),
            )
        return {
            'candidates': [{'content': {'parts': [{'text': 'សួស្តី'}]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 3},
        }

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    monkeypatch.setattr(online_providers, 'gemini_text_model_catalog', lambda key: ['gemini-2.5-flash'])
    monkeypatch.setattr(online_providers, '_gemini_dead_models', {})
    monkeypatch.setattr(online_providers, '_gemini_quota_blocked_until', {})

    text, meta = online_providers.gemini_chat(
        'k', [{'role': 'user', 'content': '1|hello'}], model='gemini-2.5-flash',
        generation_config={'temperature': 0.2}, system_instruction='SYS',
        want_usage=True, prefer_requested=True,
    )
    assert text == 'សួស្តី'
    assert 'systemInstruction' in calls[0]
    assert 'systemInstruction' not in calls[1]
    assert calls[1]['contents'][0]['parts'][0]['text'].startswith('SYS')
    assert meta['usage']['prompt_tokens'] == 12


def test_gemini_drops_thinkingconfig_when_the_model_rejects_it(monkeypatch):
    """400 "Thinking is not enabled" -> retry the same model without thinkingConfig."""
    import io
    import urllib.error

    from backend.app import online_providers

    calls: list[dict] = []

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        calls.append(body)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                url, 400, 'Bad Request', {},
                io.BytesIO(b'{"error":{"message":"Thinking is not enabled for this model"}}'),
            )
        return {
            'candidates': [{'content': {'parts': [{'text': 'សួស្តី'}]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 3},
        }

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    monkeypatch.setattr(online_providers, 'gemini_text_model_catalog', lambda key: ['gemini-2.5-flash'])
    monkeypatch.setattr(online_providers, '_gemini_dead_models', {})
    monkeypatch.setattr(online_providers, '_gemini_quota_blocked_until', {})

    text, meta = online_providers.gemini_chat(
        'k', [{'role': 'user', 'content': '1|hello'}], model='gemini-2.5-flash',
        generation_config={'temperature': 0.2, 'thinkingConfig': {'thinkingBudget': 0}},
        system_instruction='SYS', want_usage=True, prefer_requested=True,
    )
    assert text == 'សួស្តី'
    assert calls[0]['generationConfig']['thinkingConfig'] == {'thinkingBudget': 0}
    assert 'thinkingConfig' not in calls[1]['generationConfig']
    assert calls[1]['generationConfig']['temperature'] == 0.2
    assert 'systemInstruction' in calls[1]


def test_explicit_gemini_keeps_custom_api_as_fallback(monkeypatch):
    """Choosing a paid engine must still fall back to the user's own endpoint."""
    from backend.app import api_keys

    monkeypatch.setattr(api_keys, 'has_key', lambda provider: provider in {'gemini', 'custom'})
    monkeypatch.setattr(api_keys, 'get_key', lambda provider: f'{provider}-key')
    monkeypatch.setattr(providers, '_argos_available', lambda: False)
    monkeypatch.setattr(providers, 'qwen_local', None, raising=False)
    monkeypatch.setattr(providers.CustomAPITranslationProvider, 'is_available', staticmethod(lambda: True))
    monkeypatch.setattr(providers.GeminiTranslationProvider, 'is_available', staticmethod(lambda: True))

    hybrid = HybridTranslationProvider(preference='gemini')
    keys = [key for key, _ in hybrid._chain()]
    assert keys[:2] == ['gemini', 'custom'], keys


def test_route_falls_back_to_one_line_for_an_unfilled_batch(monkeypatch, tmp_path):
    main, project_id = _prepare_project(monkeypatch, tmp_path, ['Hello', 'Bye'])
    single: list[str] = []

    class FakeTranslator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_preferred_engine(self, *args, **kwargs) -> None:
            return None

        def first_engine(self) -> str:
            return 'gemini'

        def supports_batch(self) -> bool:
            return True

        def translate_batch(self, lines, context=None, glossary='', *, source_language='auto',
                            target_language='Khmer', tone='natural and consistent', usage=None):
            return ['សួស្តី', None]

        def translate(self, text, target_language, source_language='en'):
            single.append(text)
            return 'លាហើយ'

    monkeypatch.setattr(main, 'HybridTranslationProvider', FakeTranslator)
    try:
        from fastapi.testclient import TestClient

        response = TestClient(main.app).post(
            '/api/process/translate', json={'project_id': project_id, 'target_language': 'Khmer'},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
    finally:
        main.projects.pop(project_id, None)

    assert single == ['Bye'], 'only the missing line goes through the single-line path'
    assert [item['translation'] for item in payload['translation']] == ['សួស្តី', 'លាហើយ']
