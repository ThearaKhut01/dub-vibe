"""Edge's sibling-voice fallback (one voice silent, the other fine)."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from backend.app import providers


def _fake_edge(monkeypatch: pytest.MonkeyPatch, silent: set[str]) -> list[str]:
    calls: list[str] = []

    class FakeCommunicate:
        def __init__(self, _text: str, voice: str) -> None:
            self.voice = voice
            calls.append(voice)

        async def save(self, path: str) -> None:
            if self.voice in silent:
                raise RuntimeError('No audio was received. Please verify that your parameters are correct.')
            Path(path).write_bytes(b'ID3' + b'\x00' * 512)

    monkeypatch.setitem(sys.modules, 'edge_tts', types.SimpleNamespace(Communicate=FakeCommunicate))
    monkeypatch.setattr(providers, '_audio_has_sound', lambda _path: True)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(providers.asyncio, 'sleep', no_sleep)
    return calls


def test_a_silent_voice_falls_back_to_the_other_khmer_voice(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: Path) -> None:
    calls = _fake_edge(monkeypatch, silent={'km-KH-PisethNeural'})

    result = asyncio.run(providers.EdgeTTSProvider().synthesize(
        'សួស្តី', 'Male Voice A', str(tmp_path / 'a.mp3'), 'Khmer',
    ))

    assert result['voice'] == 'km-KH-SreymomNeural'
    assert calls[-1] == 'km-KH-SreymomNeural'
    assert calls.count('km-KH-PisethNeural') == 3, 'the chosen voice still gets its three attempts'


def test_both_voices_silent_reports_the_language_unavailable(monkeypatch: pytest.MonkeyPatch,
                                                            tmp_path: Path) -> None:
    _fake_edge(monkeypatch, silent={'km-KH-PisethNeural', 'km-KH-SreymomNeural'})

    with pytest.raises(providers.SpeechUnavailableError) as caught:
        asyncio.run(providers.EdgeTTSProvider().synthesize(
            'សួស្តី', 'Male Voice A', str(tmp_path / 'a.mp3'), 'Khmer',
        ))

    message = str(caught.value)
    assert 'km-KH-PisethNeural' in message and 'km-KH-SreymomNeural' in message
    assert 'Press Dub again' in message


def test_an_english_voice_is_not_used_for_khmer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _fake_edge(monkeypatch, silent={'km-KH-PisethNeural', 'km-KH-SreymomNeural'})

    with pytest.raises(providers.SpeechUnavailableError):
        asyncio.run(providers.EdgeTTSProvider().synthesize(
            'សួស្តី', 'nonsense-voice', str(tmp_path / 'a.mp3'), 'Khmer',
        ))

    assert all(voice.startswith('km-KH-') for voice in calls)
