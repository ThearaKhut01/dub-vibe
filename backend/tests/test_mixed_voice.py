"""'Mixed — male & female' must assign voices by the speakers' real gender.

The old mixed mode alternated male/female on a >1 s pause — a pure timing guess
that landed ~40% on real dialogue. Now the line's saved gender, the character
profile's gender (Speaker Detection's ECAPA classifier), or — for lines nobody
classified — the line's own audio through the same classifier decides; unsure
lines keep the previous voice, and only a project with no gender signal at all
falls back to the pause heuristic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import main


def _segment(index: int, start: float, end: float, speaker: str = 'Speaker 1',
             gender: str | None = None) -> dict:
    segment = {'id': f's{index}', 'start': start, 'end': end, 'speaker': speaker,
               'text': 'x', 'translation': 'x'}
    if gender:
        segment['gender'] = gender
    return segment


@pytest.fixture
def no_classifier(monkeypatch):
    def never(media_path, start, end, ffmpeg):
        raise AssertionError('classification must not run when genders are already known')

    monkeypatch.setattr(main.speaker_diarization, 'classify_segment_gender', never)


def test_profile_genders_decide_without_classification(no_classifier):
    segments = [_segment(1, 0.0, 1.0, 'Speaker 1'), _segment(2, 1.0, 2.0, 'Speaker 2')]
    characters = {'Speaker 1': {'gender': 'Male'}, 'Speaker 2': {'gender': 'Female'}}
    labels = main.mixed_voice_labels(segments, characters, '', None)
    assert labels == ['Male Voice A', 'Female Voice B']
    assert all('gender' not in segment for segment in segments)


def test_saved_line_gender_wins_over_the_profile(no_classifier):
    segments = [_segment(1, 0.0, 1.0, 'Speaker 1', gender='Female'),
                _segment(2, 1.0, 2.0, 'Speaker 1')]
    characters = {'Speaker 1': {'gender': 'Male'}}
    labels = main.mixed_voice_labels(segments, characters, '', None)
    # Line 1 carries the editor's explicit pick; line 2 falls back to the profile.
    assert labels == ['Female Voice B', 'Male Voice A']


def test_classifier_fills_unknown_lines_and_persists(monkeypatch):
    """Every line is heard on its own — never branded by another line's result."""
    calls: list[float] = []

    def classify(media_path, start, end, ffmpeg):
        calls.append(start)
        return 'Female' if start == 0.0 else None  # lines 2 and 3 are too noisy

    monkeypatch.setattr(main.speaker_diarization, 'classify_segment_gender', classify)
    segments = [
        _segment(1, 0.0, 1.0, 'Speaker 1'),
        _segment(2, 1.0, 2.0, 'Speaker 1'),  # same label, still heard separately
        _segment(3, 2.0, 3.0, 'Speaker 2'),
    ]
    characters: dict = {}
    labels = main.mixed_voice_labels(segments, characters, 'media.mp4', 'ffmpeg')
    assert calls == [0.0, 1.0, 2.0]
    assert labels == ['Female Voice B', 'Female Voice B', 'Female Voice B']  # unsure lines inherit
    assert segments[0]['gender'] == 'Female'
    assert 'gender' not in segments[1] and 'gender' not in segments[2]  # unsure: nothing saved
    assert characters == {}  # speaker profiles belong to Speaker Detection, not one line


def test_no_gender_signal_falls_back_to_the_pause_heuristic(monkeypatch):
    monkeypatch.setattr(main.speaker_diarization, 'classify_segment_gender',
                        lambda *args: None)
    segments = [
        _segment(1, 0.0, 1.0),                      # Male (start)
        _segment(2, 4.0, 5.0),                      # 3 s pause -> flip to Female
        _segment(3, 5.0, 6.0),                      # continuous -> stays Female
    ]
    labels = main.mixed_voice_labels(segments, {}, '', None)
    assert labels == ['Male Voice A', 'Female Voice B', 'Female Voice B']


def test_dub_run_uses_detected_genders_for_mixed(tmp_path, monkeypatch):
    """End to end: a mixed dub classifies each line and synthesizes with that voice."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'find_media_tool', lambda name: None)
    monkeypatch.setattr(
        main.speaker_diarization, 'classify_segment_gender',
        lambda media_path, start, end, ffmpeg: 'Female' if start == 0.0 else 'Male',
    )

    used_voices: list[str] = []

    def fake_synthesize_line(text, out_path, **kwargs):
        used_voices.append(str(kwargs['voice_pref']))
        Path(out_path).write_bytes(b'mp3')
        return {'provider': 'edge', 'voice': kwargs['voice_pref'], 'path': out_path}

    monkeypatch.setattr(main.tts_service, 'synthesize_line', fake_synthesize_line)
    pid = 'mixed-gender-dub'
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': 'movie.mp4',
        'duration': '00:00:10', 'target_language': 'Khmer',
        'segments': [
            {'id': 's1', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1',
             'text': 'x', 'translation': 'សួស្តី'},
            {'id': 's2', 'start': 2.0, 'end': 4.0, 'speaker': 'Speaker 2',
             'text': 'x', 'translation': 'អរគុណ'},
        ],
        'audio_segments': [],
    }
    try:
        response = TestClient(main.app).post(
            '/api/process/tts',
            json={'project_id': pid, 'target_language': 'Khmer', 'tts_provider': 'edge',
                  'voice': 'mixed', 'acting': False},
        )
        assert response.status_code == 200, response.text
        assert used_voices == ['Female Voice B', 'Male Voice A']
        # The response carries the per-line genders so the editor auto-selects them.
        assert [row.get('gender') for row in response.json()['result']['segments']] == ['Female', 'Male']
        # The detections were saved on the lines for the editor and the next run.
        assert [segment['gender'] for segment in main.projects[pid]['segments']] == ['Female', 'Male']
    finally:
        main.projects.pop(pid, None)


def test_detect_genders_endpoint_classifies_and_persists(tmp_path, monkeypatch):
    """Auto-detect all: every line's gender is heard, saved, and auto-selected."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    calls: list[float] = []

    def classify(media_path, start, end, ffmpeg):
        calls.append(start)
        return 'Female' if start == 0.0 else 'Male'

    monkeypatch.setattr(main.speaker_diarization, 'classify_segment_gender', classify)
    media = tmp_path / 'movie.mp4'
    media.write_bytes(b'video')
    pid = 'detect-all'
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': str(media),
        'duration': '00:00:10', 'target_language': 'Khmer',
        'segments': [
            {'id': 'a', 'start': 0.0, 'end': 1.0, 'speaker': 'Speaker 1', 'text': 'x', 'translation': 'x'},
            {'id': 'b', 'start': 1.0, 'end': 2.0, 'speaker': 'Speaker 2', 'text': 'x', 'translation': 'x'},
            {'id': 'c', 'start': 2.0, 'end': 3.0, 'speaker': 'Speaker 1', 'text': 'x', 'translation': 'x'},
        ],
        'audio_segments': [],
    }
    try:
        response = TestClient(main.app).post(f'/api/projects/{pid}/segments/detect-genders')
        assert response.status_code == 200, response.text
        body = response.json()
        # Every line is re-listened from scratch — no per-speaker branding.
        assert body['detected'] == {'a': 'Female', 'b': 'Male', 'c': 'Male'}
        assert calls == [0.0, 1.0, 2.0]
        assert body['message'].startswith('Detected 2 male and 1 female line(s).')
        segments = {segment['id']: segment for segment in main.projects[pid]['segments']}
        assert segments['a']['gender'] == 'Female' and segments['b']['gender'] == 'Male'
    finally:
        main.projects.pop(pid, None)


def test_detect_genders_keeps_saved_picks_and_reports_unsure(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    calls: list[float] = []

    def classify(media_path, start, end, ffmpeg):
        calls.append(start)
        return None  # this recording is too noisy to hear

    monkeypatch.setattr(main.speaker_diarization, 'classify_segment_gender', classify)
    media = tmp_path / 'movie.mp4'
    media.write_bytes(b'video')
    pid = 'detect-all-unsure'
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': str(media),
        'duration': '00:00:10', 'target_language': 'Khmer',
        'segments': [
            {'id': 'a', 'start': 0.0, 'end': 1.0, 'speaker': 'Speaker 1', 'text': 'x',
             'translation': 'x', 'gender': 'Male'},  # the editor's saved pick: never re-classified
            {'id': 'b', 'start': 1.0, 'end': 2.0, 'speaker': 'Speaker 1', 'text': 'x', 'translation': 'x'},
        ],
        'audio_segments': [],
    }
    try:
        response = TestClient(main.app).post(f'/api/projects/{pid}/segments/detect-genders')
        assert response.status_code == 200, response.text
        body = response.json()
        # Auto-detect all re-listens to every line; the saved pick is kept only
        # because this recording could not be heard.
        assert body['detected'] == {'a': 'Male'}
        assert body['unsure_lines'] == [2]
        assert calls == [0.0, 1.0]
        assert 'pick a voice manually' in body['message']
    finally:
        main.projects.pop(pid, None)


def test_per_line_generate_auto_detects_saves_and_uses_the_gender(tmp_path, monkeypatch):
    """POST segments/{id}/tts with gender 'auto': listen, save, speak with the match."""
    from fastapi.testclient import TestClient

    from backend.app import main, segment_tts

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    classify_calls: list[float] = []

    def classify(media_path, start, end, ffmpeg):
        classify_calls.append(start)
        return 'Female'

    monkeypatch.setattr(main.speaker_diarization, 'classify_segment_gender', classify)

    def fake_synthesize_segment(**kwargs):
        segment = kwargs['segment']
        directory = segment_tts.segment_audio_dir(kwargs['project_id'], kwargs['data_root'])
        final = directory / f"{segment['id']}.mp3"
        final.write_bytes(b'mp3')
        return {
            'id': segment['id'], 'start': segment['start'], 'end': segment['end'],
            'source_start': float(segment['start']), 'text': segment['translation'],
            'character': segment.get('speaker') or 'Speaker 1',
            'voice': f"edge:{kwargs['voice']}",
            'gender': 'Female' if kwargs['voice'] == segment_tts.GENDER_VOICES['female'] else 'Male',
            'path': str(final),
        }

    monkeypatch.setattr(segment_tts, 'synthesize_segment', fake_synthesize_segment)
    pid = 'auto-tts'
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': 'movie.mp4',
        'duration': '00:00:10', 'target_language': 'Khmer',
        'segments': [{'id': 'a', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1',
                      'text': 'x', 'translation': 'សួស្តី'}],
        'audio_segments': [],
    }
    try:
        client = TestClient(main.app)
        response = client.post(f'/api/projects/{pid}/segments/a/tts', json={'gender': 'auto'})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['audio']['gender'] == 'Female'
        assert body['audio']['voice'] == 'edge:Female Voice B'
        assert main.projects[pid]['segments'][0]['gender'] == 'Female'
        assert classify_calls == [0.0]

        # Auto is the "hear it again" action: it re-classifies even a saved line.
        response = client.post(f'/api/projects/{pid}/segments/a/tts', json={'gender': 'auto'})
        assert response.status_code == 200, response.text
        assert classify_calls == [0.0, 0.0]
    finally:
        main.projects.pop(pid, None)


def test_per_line_auto_reports_when_it_cannot_hear_the_line(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main.speaker_diarization, 'classify_segment_gender', lambda *args: None)
    pid = 'auto-tts-unsure'
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': 'movie.mp4',
        'duration': '00:00:10', 'target_language': 'Khmer',
        'segments': [{'id': 'a', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1',
                      'text': 'x', 'translation': 'សួស្តី'}],
        'audio_segments': [],
    }
    try:
        response = TestClient(main.app).post(f'/api/projects/{pid}/segments/a/tts', json={'gender': 'auto'})
        assert response.status_code == 503, response.text
        assert 'pick Male voice / Female voice' in response.json()['detail']
        # Nothing was guessed and nothing was saved.
        assert main.projects[pid]['segments'][0].get('gender') is None
    finally:
        main.projects.pop(pid, None)
