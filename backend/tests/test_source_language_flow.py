"""The transcript's language is the film's language, and a saved translation must be one.

Two production failures on a 3h+ Chinese film (Battle Through the Heavens, Khmer dub):
1. Whisper re-detects the language per chunk; the song-heavy ending chunks flipped
   to English ("I'm nothing I...", "myriad soul...", "shut them now") and the
   transcript changed script mid-film. The majority chunk detection is the film's
   language, so the divergent chunks are re-decoded with it forced.
2. Lines saved by older versions kept the untranslated source as the "translation"
   (an English line left as-is on a Khmer dub) and resume trusted them forever; a
   saved line now must pass the same script check a fresh answer must.
Also: the Gemini Web prompt names the source language ("from Chinese"), not the
bare ISO code the project stores ("from zh").
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.app import gemini_web, main


# --- majority language across transcription chunks ---------------------------

def test_majority_language_picks_the_dominant_detection():
    assert main._majority_language(['zh', 'zh', 'zh', 'en']) == 'zh'
    assert main._majority_language(['zh']) == 'zh'
    assert main._majority_language(['zh', 'zh', 'en', 'en', 'en']) == 'en'  # exactly the 0.6 share


def test_majority_language_refuses_an_even_split():
    """A genuinely two-language detection is kept per chunk, not forced."""
    assert main._majority_language(['zh', 'en', 'zh', 'en']) is None
    assert main._majority_language([]) is None


def test_majority_language_ties_keep_the_earliest():
    assert main._majority_language(['zh', 'en', 'en']) == 'en'


# --- chunked transcription forces the film's language ------------------------

AUTO_DETECT = {'c1': 'zh', 'c2': 'zh', 'song': 'en', 'c4': 'zh', 'song2': 'en'}
# Forced decoding always answers in the forced language, like Whisper.


class _FakeWhisper:
    calls: list[tuple[str, str | None]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def transcribe(self, file_path, progress_callback=None, should_cancel=None,
                   language=None, beam_size=5, word_timestamps=False):
        _FakeWhisper.calls.append((Path(file_path).stem, language))
        chunk_id = Path(file_path).stem
        detected = language or AUTO_DETECT.get(chunk_id, 'zh')
        text = {'zh': '对话台词', 'en': "I'm nothing I..."}[detected]
        return {
            'provider': 'Faster-Whisper', 'model': 'fake', 'language': detected,
            'segments': [{'id': 'seg_1', 'start': 0.0, 'end': 2.0,
                          'speaker': 'Speaker 1', 'text': text}],
            'source_file': file_path, 'cancelled': False, 'status': 'ok',
        }


@pytest.fixture
def transcribe_env(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'FasterWhisperProvider', _FakeWhisper)
    monkeypatch.setattr(main.model_manager, 'selected_model_id', lambda: 'fake-model')
    monkeypatch.setattr(main.model_manager, 'ensure_selected_ready', lambda cb: ('fake-model', 'cpu'))
    media = tmp_path / 'movie.mp4'
    media.write_bytes(b'fake')
    return media


def _chunk_project(pid: str, media: Path, chunk_ids: list[str]) -> None:
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': str(media),
        'duration': '00:04:00', 'target_language': 'Khmer',
        'chunks': [
            {'start': float(i * 60.0), 'duration': 60.0, 'path': str(media.parent / f'{cid}.bin')}
            for i, cid in enumerate(chunk_ids)
        ],
        'segments': [], 'audio_segments': [],
    }
    for cid in chunk_ids:
        (media.parent / f'{cid}.bin').write_bytes(b'audio')


def test_transcription_redecodes_song_chunks_in_the_film_language(transcribe_env):
    from fastapi.testclient import TestClient

    pid = 'majority-zh'
    _chunk_project(pid, transcribe_env, ['c1', 'c2', 'song'])
    _FakeWhisper.calls = []
    try:
        response = TestClient(main.app).post(
            '/api/process/transcribe', json={'project_id': pid},
        )
        assert response.status_code == 200, response.text
        body = response.json()['result']
        # The song chunk was detected as English, then re-decoded forced to zh.
        assert _FakeWhisper.calls == [('c1', None), ('c2', None), ('song', None), ('song', 'zh')]
        assert body['language'] == 'zh'
        assert [segment['text'] for segment in body['segments']] == ['对话台词', '对话台词', '对话台词']
        assert main.projects[pid]['source_language'] == 'zh'
    finally:
        main.projects.pop(pid, None)


def test_explicit_language_needs_no_majority_pass(transcribe_env):
    from fastapi.testclient import TestClient

    pid = 'forced-zh'
    _chunk_project(pid, transcribe_env, ['c1', 'song'])
    _FakeWhisper.calls = []
    try:
        response = TestClient(main.app).post(
            '/api/process/transcribe', json={'project_id': pid, 'language': 'Chinese'},
        )
        assert response.status_code == 200, response.text
        # The user's choice is already forced onto every chunk: one pass, no re-decode.
        assert _FakeWhisper.calls == [('c1', 'zh'), ('song', 'zh')]
        assert response.json()['result']['language'] == 'zh'
    finally:
        main.projects.pop(pid, None)


def test_an_even_split_is_not_forced(transcribe_env):
    from fastapi.testclient import TestClient

    pid = 'even-split'
    _chunk_project(pid, transcribe_env, ['c1', 'song', 'c4', 'song2'])  # 2 zh / 2 en
    _FakeWhisper.calls = []
    try:
        response = TestClient(main.app).post(
            '/api/process/transcribe', json={'project_id': pid},
        )
        assert response.status_code == 200, response.text
        assert len(_FakeWhisper.calls) == 4  # one pass, no re-decode
        assert response.json()['result']['language'] == 'zh'  # first chunk's detection
    finally:
        main.projects.pop(pid, None)


# --- saved translations must be real translations ----------------------------

def test_translation_ok_rules():
    assert main._translation_ok('你好', 'សួស្តី', 'zh', 'Khmer')
    assert main._translation_ok('蕭顏', 'Xiao Yan', 'zh', 'Khmer')  # romanized names stay valid
    assert main._translation_ok('1234', '1234', 'zh', 'Khmer')
    # The untranslated source echoed back, or English kept on a Khmer dub:
    assert not main._translation_ok("I'm nothing I...", "I'm nothing I...", 'zh', 'Khmer')
    assert not main._translation_ok('你好', "I'm nothing I...", 'zh', 'Khmer')
    assert not main._translation_ok('你好', '', 'zh', 'Khmer')
    assert not main._translation_ok('你好', '   ', 'zh', 'Khmer')


class _FakeTranslator:
    requests: list[str] = []

    def __init__(self, preference=None) -> None:
        self._preference = preference

    def check_preferred_engine(self, target_language, source_language='en'):
        return None

    def first_engine(self):
        return 'fake'

    def supports_batch(self):
        return False

    def translate(self, text, target_language, source_language='en'):
        _FakeTranslator.requests.append(text)
        return f'ការបកប្រែ {text}'


def test_resume_retranslates_a_saved_source_echo(tmp_path, monkeypatch):
    """A stale 'translation' that is really the source line is redone, not kept."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv('DUBVIBE_TRANSLATION_CACHE', '0')
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'HybridTranslationProvider', _FakeTranslator)

    pid = 'stale-echo'
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'target_language': 'Khmer', 'source_language': 'zh',
        'segments': [
            {'id': 's1', 'start': 0.0, 'end': 2.0, 'text': '你好', 'translation': 'សួស្តី'},
            {'id': 's2', 'start': 2.0, 'end': 4.0, 'text': "I'm nothing I...", 'translation': "I'm nothing I..."},
            {'id': 's3', 'start': 4.0, 'end': 6.0, 'text': '再见', 'translation': ''},
        ],
        'audio_segments': [],
    }
    _FakeTranslator.requests = []
    try:
        response = TestClient(main.app).post(
            '/api/process/translate',
            json={'project_id': pid, 'target_language': 'Khmer', 'translation_provider': 'gemini_web'},
        )
        assert response.status_code == 200, response.text
        segments = main.projects[pid]['segments']
        assert segments[0]['translation'] == 'សួស្តី'          # valid saved line: kept
        assert segments[1]['translation'].startswith('ការបកប្រែ')  # stale echo: re-translated
        assert segments[2]['translation'].startswith('ការបកប្រែ')  # empty: translated
        assert _FakeTranslator.requests == ["I'm nothing I...", '再见']
    finally:
        main.projects.pop(pid, None)


# --- the Gemini Web prompt names the source language -------------------------

def test_gemini_web_prompt_names_the_source_language():
    prompt = gemini_web.build_batch_prompt(['你好'], 'Khmer', 'zh')
    assert 'from Chinese' in prompt
    assert 'from zh' not in prompt
    single = gemini_web.build_translation_prompt('你好', 'Khmer', 'zh')
    assert 'from Chinese' in single
    assert 'from zh' not in single
    # A language name passed by a caller stays a name:
    assert 'from Chinese' in gemini_web.build_batch_prompt(['你好'], 'Khmer', 'Chinese')


def test_gemini_web_prompt_keeps_the_mixed_source_guard():
    prompt = gemini_web.build_batch_prompt(['你好', "I'm nothing"], 'Khmer', 'zh')
    assert 'even when a source line is already in another language' in prompt
    assert 'Never copy a source line unchanged' in prompt
