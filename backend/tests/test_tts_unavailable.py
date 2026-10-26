"""A dub must never silently come out with no voice.

Two separate failures were found in production:
1. `edge-tts` 7.2.7 was blocked by Microsoft (403 handshake) — the packaged version
   must stay above that floor.
2. While Edge's Khmer voices returned "No audio was received", the SAPI5 fallback
   "succeeded" by writing a 226-byte header-only MP3 with no audio frames. The dub
   then exported with no voice and no error.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from backend.app import tts_service
from backend.app.providers import EdgeTTSProvider, SpeechUnavailableError, _audio_has_sound


@pytest.fixture(autouse=True)
def clear_short_circuit():
    tts_service.reset_unavailable()
    yield
    tts_service.reset_unavailable()


def _fake_edge_module(error: Exception | None = None, writes_audio: bool = False):
    """Install a fake edge_tts whose save() fails or writes an empty file."""
    module = types.ModuleType('edge_tts')

    class Communicate:
        def __init__(self, text, voice, **kwargs):
            self.voice = voice

        async def save(self, path):
            if error is not None:
                raise error
            if writes_audio:
                Path(path).write_bytes(b'ID3' + b'\0' * 2000)
            else:
                Path(path).write_bytes(b'')

    module.Communicate = Communicate
    return module


def test_no_audio_from_edge_is_reported_as_unavailable_not_retried_forever(monkeypatch, tmp_path):
    """'No audio was received' is a language-level outage: retrying cannot help."""
    import asyncio

    monkeypatch.setitem(sys.modules, 'edge_tts', _fake_edge_module(RuntimeError('No audio was received. Please verify')))
    monkeypatch.setattr(asyncio, 'sleep', lambda seconds: _noop())

    with pytest.raises(SpeechUnavailableError) as info:
        asyncio.run(EdgeTTSProvider().synthesize('សួស្តី', 'Male Voice A', str(tmp_path / 'a.mp3'), 'Khmer'))
    message = str(info.value)
    assert 'no audio for Khmer' in message
    # The message tells the user it is a service-side problem, not their setup.
    assert 'not serving this language' in message


async def _noop():
    return None


def test_edge_empty_file_is_rejected(monkeypatch, tmp_path):
    """A success that leaves no audible audio must not be accepted."""
    import asyncio

    monkeypatch.setitem(sys.modules, 'edge_tts', _fake_edge_module(writes_audio=False))
    monkeypatch.setattr(asyncio, 'sleep', lambda seconds: _noop())

    with pytest.raises(SpeechUnavailableError, match='no audible audio'):
        asyncio.run(EdgeTTSProvider().synthesize('hi', 'Male Voice A', str(tmp_path / 'b.mp3'), 'English'))


def test_header_only_clip_has_no_sound(tmp_path):
    """The exact SAPI5 output that caused the silent dub: 226 bytes, no frames."""
    clip = tmp_path / 'empty.mp3'
    clip.write_bytes(b'\xff\xfb\x90\x64' + b'\0' * 222)
    assert _audio_has_sound(clip) is False


def test_real_audio_has_sound(tmp_path):
    """Negative control: a real WAV (silence but valid frames) is readable."""
    import wave

    target = tmp_path / 'ok.wav'
    with wave.open(str(target), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b'\0\0' * 800)
    assert _audio_has_sound(target) is True


def test_missing_file_has_no_sound(tmp_path):
    assert _audio_has_sound(tmp_path / 'nope.mp3') is False


# --- No ffprobe: the CI/clean-machine path -----------------------------------
# The guard must not depend on ffmpeg being installed, or a header-only clip is
# accepted and the dub exports silently with no voice.

@pytest.fixture
def no_ffprobe(monkeypatch):
    """Simulate a machine without ffmpeg (exactly the GitHub runner)."""
    import shutil

    monkeypatch.setattr(shutil, 'which', lambda name: None)
    monkeypatch.setattr('backend.app.main.find_media_tool', lambda name: None)


def test_header_only_clip_is_rejected_without_ffprobe(tmp_path, no_ffprobe):
    clip = tmp_path / 'empty.mp3'
    clip.write_bytes(b'\xff\xfb\x90\x64' + b'\0' * 222)
    assert _audio_has_sound(clip) is False


def test_header_only_clip_behind_an_id3_tag_is_rejected_without_ffprobe(tmp_path, no_ffprobe):
    """edge-tts/SAPI5 files start with an ID3v2 tag; its size must be skipped."""
    clip = tmp_path / 'tagged.mp3'
    id3 = b'ID3\x04\x00\x00' + bytes([0, 0, 0, 20]) + b'\0' * 20  # 20-byte tag body
    clip.write_bytes(id3 + b'\xff\xfb\x90\x64' + b'\0' * 222)
    assert _audio_has_sound(clip) is False


def test_mp3_with_one_complete_frame_has_sound_without_ffprobe(tmp_path, no_ffprobe):
    """The 417-byte frame implied by the header makes the clip audible."""
    clip = tmp_path / 'frame.mp3'
    clip.write_bytes(b'\xff\xfb\x90\x64' + b'\0' * 413)
    assert _audio_has_sound(clip) is True


def test_real_audio_still_has_sound_without_ffprobe(tmp_path, no_ffprobe):
    import wave

    target = tmp_path / 'ok.wav'
    with wave.open(str(target), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b'\0\0' * 800)
    assert _audio_has_sound(target) is True


def test_empty_wav_is_rejected_without_ffprobe(tmp_path, no_ffprobe):
    import wave

    target = tmp_path / 'silent.wav'
    with wave.open(str(target), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b'')
    assert _audio_has_sound(target) is False


def test_unverifiable_container_is_accepted_without_ffprobe(tmp_path, no_ffprobe):
    """Never block a dub over a format the frame check cannot read."""
    clip = tmp_path / 'voice.ogg'
    clip.write_bytes(b'OggS' + b'\0' * 400)
    assert _audio_has_sound(clip) is True


def test_synthesize_line_falls_through_when_a_provider_makes_no_audio(monkeypatch, tmp_path):
    """A provider that writes an empty file is skipped, and the next one is used."""
    import backend.app.providers as providers

    monkeypatch.setattr(tts_service, 'candidate_providers', lambda pref, lang, profile: ['sapi5', 'edge'])

    def silent_sapi(self, text, voice, output_path, language):
        Path(output_path).write_bytes(b'')
        return {'provider': 'sapi5', 'voice': voice, 'output': output_path, 'status': 'ok'}

    async def good_edge(self, text, voice, output_path, language):
        Path(output_path).write_bytes(b'ID3' + b'\0' * 4000)
        return {'provider': 'Edge TTS', 'voice': voice, 'output': output_path, 'status': 'ok'}

    monkeypatch.setattr(providers.Pyttsx3Provider, 'synthesize', silent_sapi)
    monkeypatch.setattr(providers.EdgeTTSProvider, 'synthesize', good_edge)
    monkeypatch.setattr(tts_service, '_has_audio', lambda path: Path(path).stat().st_size > 2000)
    monkeypatch.setattr(tts_service, 'strip_audio_silence', lambda path: None)

    out = tmp_path / 'line.mp3'
    result = tts_service.synthesize_line(
        'hello', str(out), provider_pref='auto', language_name='English', lang_code='en',
        voice_pref=None, model_pref=None, speed=1.0, emotion='Neutral', intensity=70,
        character_profile=None, analysis={},
    )
    assert result['provider'] == 'edge'
    assert any('produced no audible audio' in attempt for attempt in result['attempts'])


def test_unavailable_engine_is_skipped_on_later_lines(monkeypatch, tmp_path):
    """A broken voice must not be retried 3x for every line of a long project."""
    calls: list[str] = []

    monkeypatch.setattr(tts_service, 'candidate_providers', lambda pref, lang, profile: ['edge'])

    from backend.app.providers import EdgeTTSProvider as RealEdge

    def always_unavailable(self, text, voice, output_path, language):
        calls.append(language)
        raise SpeechUnavailableError(f'Edge TTS returned no audio for {language}')

    monkeypatch.setattr(RealEdge, 'synthesize', always_unavailable)

    def run(tag: str):
        out = tmp_path / f'{tag}.mp3'
        with pytest.raises(SpeechUnavailableError):
            tts_service.synthesize_line(
                'សួស្តី', str(out), provider_pref='auto', language_name='Khmer', lang_code='km',
                voice_pref=None, model_pref=None, speed=1.0, emotion='Neutral', intensity=70,
                character_profile=None, analysis={},
            )

    run('first')
    assert calls == ['Khmer']  # attempted once
    run('second')
    assert calls == ['Khmer']  # short-circuited, not attempted again


def test_reset_unavailable_lets_the_user_retry(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(tts_service, 'candidate_providers', lambda pref, lang, profile: ['edge'])

    from backend.app.providers import EdgeTTSProvider as RealEdge

    def unavailable(self, text, voice, output_path, language):
        calls.append(language)
        raise SpeechUnavailableError('no audio')

    monkeypatch.setattr(RealEdge, 'synthesize', unavailable)

    for tag in ('a', 'b'):
        with pytest.raises(SpeechUnavailableError):
            tts_service.synthesize_line(
                'x', str(tmp_path / f'{tag}.mp3'), provider_pref='auto', language_name='Khmer',
                lang_code='km', voice_pref=None, model_pref=None, speed=1.0, emotion='Neutral',
                intensity=70, character_profile=None, analysis={},
            )
    assert calls == ['Khmer']

    # An explicit retry (Dub again / Generate) must probe the engine again.
    tts_service.reset_unavailable()
    with pytest.raises(SpeechUnavailableError):
        tts_service.synthesize_line(
            'x', str(tmp_path / 'c.mp3'), provider_pref='auto', language_name='Khmer',
            lang_code='km', voice_pref=None, model_pref=None, speed=1.0, emotion='Neutral',
            intensity=70, character_profile=None, analysis={},
        )
    assert calls == ['Khmer', 'Khmer']


def test_all_engines_unavailable_gives_one_clear_sentence(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_service, 'candidate_providers', lambda pref, lang, profile: ['edge', 'sapi5'])

    from backend.app.providers import EdgeTTSProvider as RealEdge, Pyttsx3Provider as RealSapi

    def unavailable(self, text, voice, output_path, language):
        raise SpeechUnavailableError(f'engine cannot speak {language}')

    monkeypatch.setattr(RealEdge, 'synthesize', unavailable)
    monkeypatch.setattr(RealSapi, 'synthesize', unavailable)

    with pytest.raises(SpeechUnavailableError) as info:
        tts_service.synthesize_line(
            'x', str(tmp_path / 'd.mp3'), provider_pref='auto', language_name='Khmer', lang_code='km',
            voice_pref=None, model_pref=None, speed=1.0, emotion='Neutral', intensity=70,
            character_profile=None, analysis={},
        )
    assert str(info.value).startswith('No voice engine can produce Khmer audio right now.')


def test_edge_tts_floor_version_is_pinned():
    """7.2.7 is blocked by Microsoft (403 handshake); the requirement must be above it."""
    import re

    text = (Path(__file__).resolve().parents[2] / 'backend' / 'requirements.txt').read_text(encoding='utf-8')
    match = re.search(r'^edge-tts\s*([<>=!]+)\s*([0-9.]+)', text, re.MULTILINE)
    assert match, 'edge-tts must be pinned in requirements.txt'
    operator, version = match.group(1), match.group(2)
    assert operator in ('==', '>=', '~=', '>'), (operator, version)
    parts = tuple(int(part) for part in version.split('.')[:2])
    assert parts > (7, 2), f'edge-tts {version} is the version Microsoft blocks'
