"""One line with nothing speakable in it must not kill a dub or blacklist the engine.

Production failure (Battle Through the Heavens EP 218, Khmer dub, 188 lines): one
leftover untranslated line ("千年前"). Edge's Khmer voices answer NoAudioReceived
for text with nothing they can pronounce; the provider burned 3 retries plus the
alternate voice, raised SpeechUnavailableError, and the service then blacklisted
Edge+Khmer for 600 s — the dub aborted at line 2 and every retry re-hit the wall.
The guard below refuses such text before the network call, and a text-specific
failure never poisons the engine for the other 187 lines.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from backend.app import tts_service
from backend.app.providers import (
    EdgeTTSProvider,
    Pyttsx3Provider,
    UnspeakableTextError,
    _text_speakable,
)


@pytest.fixture(autouse=True)
def clear_short_circuit():
    tts_service.reset_unavailable()
    yield
    tts_service.reset_unavailable()


def test_speakability_rules_match_the_live_engine():
    """Rules verified live against km-KH-PisethNeural / en-US-GuyNeural (2026-11)."""
    # Deterministically fatal for a Khmer voice:
    assert not _text_speakable('千年前', 'Khmer')        # pure leftover source line
    assert not _text_speakable('...', 'Khmer')          # punctuation only
    assert not _text_speakable('។', 'Khmer')            # script punctuation is not a word
    assert not _text_speakable('  ', 'Khmer')
    # Speakable for a Khmer voice:
    assert _text_speakable('សួស្តី', 'Khmer')
    assert _text_speakable('1234', 'Khmer')             # digits are read out
    assert _text_speakable('១២៣៤។', 'Khmer')            # native digits + sentence ender
    assert _text_speakable('សមរភូមិ 千年前', 'Khmer')  # speaks the Khmer part
    assert _text_speakable('Xiao Yan', 'Khmer')         # romanized names stay fine
    # Latin targets: only Latin letters/digits are safe to assume.
    assert _text_speakable('hello', 'English')
    assert _text_speakable('1234', 'English')
    assert not _text_speakable('សួស្តី', 'English')    # a Khmer word on an en voice fails
    assert not _text_speakable('...', 'English')
    # Right script, right language:
    assert _text_speakable('千年前', 'Chinese')
    # An explicit neural id speaks its own language regardless of the dub language:
    assert _text_speakable('សួស្តី', 'English', 'km-KH-PisethNeural')
    assert not _text_speakable('千年前', 'Khmer', 'en-US-GuyNeural')


def test_edge_provider_flags_unspeakable_text_before_calling_the_service(monkeypatch, tmp_path):
    """The guard must fire before any network call — never reach edge_tts."""
    module = types.ModuleType('edge_tts')

    class Communicate:
        def __init__(self, *args, **kwargs):
            raise AssertionError('edge_tts must not be reached for unspeakable text')

    module.Communicate = Communicate
    monkeypatch.setitem(sys.modules, 'edge_tts', module)

    with pytest.raises(UnspeakableTextError) as info:
        asyncio.run(EdgeTTSProvider().synthesize(
            '千年前', 'Male Voice A', str(tmp_path / 'a.mp3'), 'Khmer',
        ))
    message = str(info.value)
    assert 'Khmer' in message
    assert '千年前' in message
    assert not (tmp_path / 'a.mp3').exists()


def test_edge_provider_still_synthesizes_speakable_text(monkeypatch, tmp_path):
    """Negative control: real Khmer text is not blocked by the guard."""
    from backend.app import providers

    class FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            self.voice = voice

        async def save(self, path):
            Path(path).write_bytes(b'ID3' + b'\x00' * 512)

    monkeypatch.setitem(sys.modules, 'edge_tts', types.SimpleNamespace(Communicate=FakeCommunicate))
    monkeypatch.setattr(providers, '_audio_has_sound', lambda _path: True)

    result = asyncio.run(EdgeTTSProvider().synthesize(
        'សួស្តី', 'Male Voice A', str(tmp_path / 'b.mp3'), 'Khmer',
    ))
    assert result['status'] == 'ok'


def test_unspeakable_line_raises_immediately_and_never_blacklists(monkeypatch, tmp_path):
    """A text-specific failure must not poison (engine, language) for the other lines."""
    edge_calls: list[str] = []
    sapi_calls: list[str] = []

    monkeypatch.setattr(tts_service, 'candidate_providers',
                        lambda pref, lang, profile: ['edge', 'sapi5'])

    def unspeakable(self, text, voice, output_path, language):
        edge_calls.append(text)
        raise UnspeakableTextError('nothing speakable')

    def never(self, text, voice, output_path, language):
        sapi_calls.append(text)
        raise AssertionError('sapi5 must not be tried for unspeakable text')

    monkeypatch.setattr(EdgeTTSProvider, 'synthesize', unspeakable)
    monkeypatch.setattr(Pyttsx3Provider, 'synthesize', never)

    for tag in ('a', 'b'):
        with pytest.raises(UnspeakableTextError):
            tts_service.synthesize_line(
                '千年前', str(tmp_path / f'{tag}.mp3'), provider_pref='auto', language_name='Khmer',
                lang_code='km', voice_pref=None, model_pref=None, speed=1.0, emotion='Neutral',
                intensity=70, character_profile=None, analysis={},
            )
    assert len(edge_calls) == 2  # re-probed per line: a text problem, not an engine outage
    assert sapi_calls == []
    assert tts_service._unavailable_until == {}


def test_speech_unavailable_still_blacklists(monkeypatch, tmp_path):
    """The old rule stays: an engine-level outage IS short-circuited for later lines."""
    calls: list[str] = []
    monkeypatch.setattr(tts_service, 'candidate_providers', lambda pref, lang, profile: ['edge'])

    def unavailable(self, text, voice, output_path, language):
        calls.append(text)
        raise tts_service.SpeechUnavailableError(f'no audio for {language}')

    monkeypatch.setattr(EdgeTTSProvider, 'synthesize', unavailable)

    for tag in ('a', 'b'):
        with pytest.raises(tts_service.SpeechUnavailableError):
            tts_service.synthesize_line(
                'សួស្តី', str(tmp_path / f'{tag}.mp3'), provider_pref='auto', language_name='Khmer',
                lang_code='km', voice_pref=None, model_pref=None, speed=1.0, emotion='Neutral',
                intensity=70, character_profile=None, analysis={},
            )
    assert len(calls) == 1  # second line short-circuited
    assert tts_service._unavailable_until


def _install_project(pid: str, translations: list[str]) -> None:
    from backend.app import main

    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'target_language': 'Khmer',
        'segments': [
            {'id': f's{i + 1}', 'start': float(i * 2), 'end': float(i * 2 + 2),
             'speaker': 'Speaker 1', 'text': f'line {i + 1}', 'translation': translation}
            for i, translation in enumerate(translations)
        ],
        'audio_segments': [],
    }


def test_dub_run_skips_an_unspeakable_line_and_finishes_the_rest(monkeypatch, tmp_path):
    """One leftover untranslated line must not abort the dub at that line."""
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')

    def fake_synthesize_line(text, out_path, **kwargs):
        if not _text_speakable(text, kwargs['language_name']):
            raise UnspeakableTextError(f'no characters a {kwargs["language_name"]} voice can pronounce: {text}')
        Path(out_path).write_bytes(b'mp3')
        return {'provider': 'edge', 'voice': kwargs['voice_pref'], 'path': out_path}

    monkeypatch.setattr(main.tts_service, 'synthesize_line', fake_synthesize_line)
    pid = 'unspeakable-dub'
    _install_project(pid, ['សួស្តី', '千年前', 'អរគុណ'])
    try:
        response = TestClient(main.app).post(
            '/api/process/tts',
            json={'project_id': pid, 'target_language': 'Khmer', 'tts_provider': 'edge', 'acting': False},
        )
        assert response.status_code == 422, response.text
        detail = response.json()['detail']
        assert 'line 2' in detail
        assert '千年前' in detail
        assert '2 finished line(s) are kept' in detail
        # The good lines were synthesized and persisted; the bad one was skipped.
        assert [clip['id'] for clip in main.projects[pid]['audio_segments']] == ['s1', 's3']
    finally:
        main.projects.pop(pid, None)


def test_dub_run_still_completes_without_unspeakable_lines(monkeypatch, tmp_path):
    """The guard must not disturb a clean dub: no unspeakable line, status ok."""
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'find_media_tool', lambda name: None)

    def fake_synthesize_line(text, out_path, **kwargs):
        Path(out_path).write_bytes(b'mp3')
        return {'provider': 'edge', 'voice': kwargs['voice_pref'], 'path': out_path}

    monkeypatch.setattr(main.tts_service, 'synthesize_line', fake_synthesize_line)
    pid = 'clean-dub'
    _install_project(pid, ['សួស្តី', 'អរគុណ'])
    try:
        response = TestClient(main.app).post(
            '/api/process/tts',
            json={'project_id': pid, 'target_language': 'Khmer', 'tts_provider': 'edge', 'acting': False},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body['result']['segments']) == 2
        assert [clip['id'] for clip in main.projects[pid]['audio_segments']] == ['s1', 's2']
    finally:
        main.projects.pop(pid, None)
