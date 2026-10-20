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
