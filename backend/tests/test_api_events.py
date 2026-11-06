"""Live API events: no key may ever appear in an event or a snapshot."""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from backend.app import api_events


@pytest.fixture(autouse=True)
def _clean_bus():
    api_events.reset_for_tests()
    yield
    api_events.reset_for_tests()


# ---------------------------------------------------------------------------
# Sanitising
# ---------------------------------------------------------------------------

def test_hint_is_only_the_last_four_characters():
    assert api_events.hint_for('AIzaSyABCDEFGHIJKLMNOP1234') == '••••1234'
    assert api_events.hint_for('abc') == ''
    assert api_events.hint_for('') == ''


@pytest.mark.parametrize('message', [
    'HTTP 400: bad key AIzaSyABCDEFGHIJKLMNOPQRSTUV0123456789',
    'GET https://generativelanguage.googleapis.com/v1beta/models/x?key=AIzaSyABCDEFGHIJKLMNOPQRSTUV0123456789',
    'invalid sk-live-abcdefghijklmnopqrstuvwx',
    'rejected cc_abcdefghijklmnopqrstuvwx',
    'Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789',
])
def test_sanitize_error_removes_credentials(message):
    cleaned = api_events.sanitize_error(message)
    assert len(cleaned) <= 80
    for marker in ('AIzaSy', 'sk-live', 'cc_', 'abcdefghijklmnopqrstuvwxyz'):
        assert marker not in cleaned


def test_sanitize_error_keeps_a_useful_reason():
    cleaned = api_events.sanitize_error('HTTP 503: This model is currently experiencing high demand')
    assert '503' in cleaned and 'high demand' in cleaned


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def test_event_payload_carries_no_key_and_counts_tokens():
    captured: list[dict] = []
    api_events.set_publisher(captured.append)

    request_id = api_events.start('legacy:gemini', provider='gemini', model='gemini-3.6-flash',
                                  lines=4, key_hint=api_events.hint_for('SECRETKEY0000001234'))
    api_events.finish(request_id, 'legacy:gemini', ok=True, provider='gemini',
                      model='gemini-3.6-flash', lines=4, latency_ms=120,
                      prompt_tokens=100, output_tokens=20, thoughts_tokens=0)

    assert [event['phase'] for event in captured] == ['start', 'success']
    blob = json.dumps(captured, ensure_ascii=False)
    assert 'SECRETKEY' not in blob
    assert '••••1234' in blob
    assert captured[1]['prompt_tokens'] == 100
    assert captured[1]['output_tokens'] == 20


def test_failure_event_shortens_and_sanitises_the_error():
    captured: list[dict] = []
    api_events.set_publisher(captured.append)
    request_id = api_events.start('entry-1', provider='gemini', model='m', lines=2)
    api_events.finish(
        request_id, 'entry-1', ok=False, provider='gemini', model='m', lines=2,
        http_status=429, error='HTTP 429: quota for AIzaSyABCDEFGHIJKLMNOPQRSTUV0123456789 exceeded',
        retry=True,
    )
    event = captured[-1]
    assert event['phase'] == 'retry'
    assert event['http_status'] == 429
    assert 'AIzaSy' not in event['error_short']
    assert len(event['error_short']) <= 80


def test_snapshot_counts_and_states():
    captured: list[dict] = []
    api_events.set_publisher(captured.append)
    api_events.register_node('entry-1', label='main', provider='gemini', model='gemini-3.6-flash',
                             key_hint='••••1234')
    api_events.register_node('entry-2', label='spare', provider='gemini', model='gemini-3.6-flash',
                             enabled=False, key_hint='••••5678')

    first = api_events.start('entry-1', provider='gemini', model='gemini-3.6-flash', lines=5)
    api_events.finish(first, 'entry-1', ok=True, provider='gemini', lines=5,
                      prompt_tokens=50, output_tokens=10, latency_ms=90)
    second = api_events.start('entry-1', provider='gemini', model='gemini-3.6-flash', lines=5)
    api_events.finish(second, 'entry-1', ok=False, provider='gemini', lines=5, http_status=500,
                      error='boom')

    snapshot = api_events.snapshot()
    nodes = {node['id']: node for node in snapshot['nodes']}
    assert nodes['entry-1']['requests'] == 2
    assert nodes['entry-1']['successes'] == 1
    assert nodes['entry-1']['failures'] == 1
    assert nodes['entry-1']['tokens_total'] == 60
    assert nodes['entry-1']['state'] == 'failed'
    assert nodes['entry-2']['state'] == 'disabled'
    assert snapshot['totals']['requests'] == 2
    assert snapshot['totals']['tokens_total'] == 60
    assert snapshot['recent'] and snapshot['recent'][0]['status'] == 'fail'


def test_cache_hit_does_not_touch_counters():
    captured: list[dict] = []
    api_events.set_publisher(captured.append)
    api_events.register_node('legacy:gemini', label='saved key', provider='gemini')
    api_events.emit_cache_hit(lines=3, provider='gemini')
    assert [event['phase'] for event in captured] == ['cache_hit']
    assert api_events.snapshot()['totals']['requests'] == 0


def test_prune_drops_removed_entries():
    api_events.register_node('entry-1', label='a', provider='gemini')
    api_events.register_node('entry-2', label='b', provider='gemini')
    api_events.prune({'entry-1'})
    assert [node['id'] for node in api_events.snapshot()['nodes']] == ['entry-1']


# ---------------------------------------------------------------------------
# The real Gemini path must not leak the key into an event
# ---------------------------------------------------------------------------

def test_gemini_failure_event_never_contains_the_key(monkeypatch):
    from backend.app import online_providers as online

    captured: list[dict] = []
    api_events.set_publisher(captured.append)
    key = 'AIzaSyFAKEfakeFAKEfake1234567890abcdef'

    def fake_request(url, key_arg, body=None, timeout=60, force_auth=False):
        raise urllib.error.HTTPError(
            url, 400, 'Bad Request', {},
            io.BytesIO(f'{{"error":{{"message":"bad key {key}"}}}}'.encode()),
        )

    monkeypatch.setattr(online, '_request_json', fake_request)
    monkeypatch.setattr(online, 'gemini_text_model_catalog', lambda key_arg: ['gemini-3.6-flash'])
    monkeypatch.setattr(online, '_gemini_dead_models', {})
    monkeypatch.setattr(online, '_gemini_quota_blocked_until', {})

    with pytest.raises(RuntimeError):
        online.gemini_chat(
            key, [{'role': 'user', 'content': 'hi'}], model='gemini-3.6-flash',
            api_meta={'node_id': 'legacy:gemini', 'lines': 3, 'key_hint': api_events.hint_for(key)},
        )

    blob = json.dumps(captured, ensure_ascii=False)
    assert key not in blob
    assert 'AIzaSy' not in blob
    assert captured and captured[0]['phase'] == 'start'
    assert any(event['phase'] in ('fail', 'retry') for event in captured)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def test_snapshot_route_lists_legacy_keys_without_exposing_them(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    secret = 'AIzaSyLEGACYfakeKEY1234567890abcdefgh'
    monkeypatch.setattr(main.api_pool, 'status', lambda: {'entries': []})
    monkeypatch.setattr(main.api_keys, 'has_key', lambda provider: provider == 'gemini')
    monkeypatch.setattr(main.api_keys, 'get_key', lambda provider: secret)

    response = TestClient(main.app).get('/api/api-live/snapshot')
    assert response.status_code == 200, response.text
    body = response.json()
    assert [node['id'] for node in body['nodes']] == ['legacy:gemini']
    assert body['nodes'][0]['key_hint'] == '••••efgh'
    assert secret not in response.text


def test_test_route_runs_a_three_line_batch(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    seen: dict = {}

    class FakeTranslator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def translate_batch(self, lines, **kwargs):
            seen['lines'] = list(lines)
            return ['ក', 'ខ', 'គ']

    monkeypatch.setattr(main, 'HybridTranslationProvider', FakeTranslator)
    monkeypatch.setattr(main.api_keys, 'has_key', lambda provider: True)

    response = TestClient(main.app).post('/api/api-live/test', json={'provider': 'gemini'})
    assert response.status_code == 200, response.text
    assert len(seen['lines']) == 3
    assert response.json()['translations'] == ['ក', 'ខ', 'គ']


def test_reduced_motion_is_handled_in_css():
    """A guard for the one requirement that is easy to delete by accident."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / 'frontend' / 'src' / 'components' / 'ApiLiveView.tsx'
    text = source.read_text(encoding='utf-8')
    assert 'prefers-reduced-motion' in text
    assert 'requestAnimationFrame' in text
