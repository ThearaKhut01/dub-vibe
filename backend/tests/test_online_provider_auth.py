"""Online providers must authenticate with each API's own mechanism.

OpenAI rejects every request without an ``Authorization: Bearer <key>`` header,
while Gemini takes its key in the query string. Both paths are exercised with a
fake transport so a regression fails here instead of at the provider.
"""
from __future__ import annotations

import json
import urllib.request

from backend.app import online_providers


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> '_FakeResponse':
        return self

    def __exit__(self, *args) -> bool:
        return False


def _capture(monkeypatch, payload: bytes) -> dict:
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen['request'] = request
        seen['timeout'] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    return seen


def test_openai_list_models_sends_bearer_token(monkeypatch):
    seen = _capture(monkeypatch, json.dumps({'data': [{'id': 'tts-1'}]}).encode('utf-8'))
    assert online_providers.openai_list_models('key-123') == ['tts-1']
    assert seen['request'].headers.get('Authorization') == 'Bearer key-123'


def test_openai_chat_sends_bearer_token(monkeypatch):
    payload = {'choices': [{'message': {'content': 'hello'}}]}
    seen = _capture(monkeypatch, json.dumps(payload).encode('utf-8'))
    assert online_providers.openai_chat('key-123', [{'role': 'user', 'content': 'hi'}]) == 'hello'
    assert seen['request'].headers.get('Authorization') == 'Bearer key-123'


def test_openai_speech_sends_bearer_token(monkeypatch, tmp_path):
    seen = _capture(monkeypatch, b'ID3fake-mp3')
    out = tmp_path / 'line.mp3'
    online_providers.openai_synthesize('key-123', 'hello', str(out))
    assert seen['request'].headers.get('Authorization') == 'Bearer key-123'
    assert out.read_bytes() == b'ID3fake-mp3'


def test_gemini_keeps_its_key_in_the_query_string(monkeypatch):
    seen = _capture(monkeypatch, json.dumps({'models': [{'name': 'models/gemini-2.5-flash-preview-tts'}]}).encode('utf-8'))
    assert online_providers.gemini_list_models('key-123') == ['gemini-2.5-flash-preview-tts']
    assert 'key=key-123' in seen['request'].full_url
    assert seen['request'].headers.get('Authorization') is None


def test_codecraft_chat_sends_bearer_token_to_the_openai_compatible_endpoint(monkeypatch):
    payload = {'choices': [{'message': {'content': 'hello'}}]}
    seen = _capture(monkeypatch, json.dumps(payload).encode('utf-8'))
    assert online_providers.codecraft_chat('cc-123', [{'role': 'user', 'content': 'hi'}]) == 'hello'
    assert seen['request'].headers.get('Authorization') == 'Bearer cc-123'
    assert seen['request'].full_url == 'https://codecraftapi.com/v1/chat/completions'


def test_codecraft_list_models_only_returns_chat_models(monkeypatch):
    payload = {
        'data': [
            {'id': 'chat-model', 'type': 'chat'},
            {'id': 'embed-model', 'type': 'embedding'},
            {'id': 'another-chat', 'type': 'chat'},
        ],
    }
    _capture(monkeypatch, json.dumps(payload).encode('utf-8'))
    assert online_providers.codecraft_list_models('cc-123') == ['another-chat', 'chat-model']


def test_codecraft_retries_with_a_listed_model_when_the_default_is_retired(monkeypatch):
    """A retired default model must not fail the run; use the cheapest listed chat model."""
    import io
    import urllib.error

    calls: list[tuple[str, dict]] = []

    def fake_request(url, key, body=None, timeout=60):
        calls.append((url, dict(body or {})))
        if url.endswith('/chat/completions') and body and body['model'] == online_providers.CODECRAFT_DEFAULT_CHAT_MODEL:
            raise urllib.error.HTTPError(url, 404, 'model not found', {},
                                         io.BytesIO(b'{"error":{"message":"model not found"}}'))
        if url.endswith('/models'):
            return {'data': [
                {'id': 'pricier-chat', 'type': 'chat', 'pricing': {'input_per_1k': 0.002}},
                {'id': 'cheapest-chat', 'type': 'chat', 'pricing': {'input_per_1k': 0.0001}},
            ]}
        return {'choices': [{'message': {'content': 'Bonjour'}}]}

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    assert online_providers.codecraft_chat('cc-123', [{'role': 'user', 'content': 'hi'}]) == 'Bonjour'
    assert calls[-1][1]['model'] == 'cheapest-chat'
