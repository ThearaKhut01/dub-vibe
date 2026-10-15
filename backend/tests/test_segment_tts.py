"""Dubbing Track editor: per-segment TTS, keep-on-failure, structural edits, and timing warnings.

The synthesis engine is faked, so these run without network access or model weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import segment_tts


def _segment(sid='1', start=0.0, end=2.0, text='Hello there.', speaker='Speaker 1'):
    return {'id': sid, 'start': start, 'end': end, 'speaker': speaker, 'translation': text}


def _fake_engine(monkeypatch, *, fail=False, write_bytes=b'mp3-bytes'):
    calls: list[dict] = []

    def fake_synthesize_line(text, out_path, **kwargs):
        calls.append({'text': text, 'out_path': out_path, **kwargs})
        if fail:
            raise RuntimeError('engine down')
        Path(out_path).write_bytes(write_bytes)
        return {'provider': 'edge', 'voice': kwargs['voice_pref'], 'path': out_path}

    monkeypatch.setattr(segment_tts.tts_service, 'synthesize_line', fake_synthesize_line)
    return calls


def test_gender_maps_to_app_voice_labels():
    assert segment_tts.voice_for_gender('Female') == 'Female Voice B'
    assert segment_tts.voice_for_gender('male') == 'Male Voice A'
    assert segment_tts.voice_for_gender(None) == 'Male Voice A'


def test_generate_writes_new_audio_and_reports_voice(tmp_path, monkeypatch):
    calls = _fake_engine(monkeypatch)
    entry = segment_tts.synthesize_segment(
        project_id='p1', data_root=tmp_path, segment=_segment(text='Hi there.'),
        previous=None, voice='Female Voice B', provider='edge', language_name='Khmer', lang_code='km',
    )
    assert Path(entry['path']).read_bytes() == b'mp3-bytes'
    assert entry['text'] == 'Hi there.'
    assert entry['gender'] == 'Female'
    assert entry['voice'] == 'edge:Female Voice B'
    assert calls[0]['voice_pref'] == 'Female Voice B'
    assert calls[0]['model_pref'] is None


def test_failed_regeneration_keeps_previous_audio(tmp_path, monkeypatch):
    """The spec: keep the old audio until the replacement is written successfully."""
    old_dir = segment_tts.segment_audio_dir('p1', tmp_path)
    old_file = old_dir / '1.mp3'
    old_file.write_bytes(b'old-audio')
    previous = {'id': '1', 'path': str(old_file), 'text': 'Old text.'}
    _fake_engine(monkeypatch, fail=True)
    with pytest.raises(RuntimeError):
        segment_tts.synthesize_segment(
            project_id='p1', data_root=tmp_path, segment=_segment(text='New text.'),
            previous=previous, voice='Male Voice A', provider='edge', language_name='Khmer', lang_code='km',
        )
    assert old_file.read_bytes() == b'old-audio'
    assert not (old_dir / '1.tmp.mp3').exists()


def test_empty_engine_output_is_treated_as_failure(tmp_path, monkeypatch):
    _fake_engine(monkeypatch, write_bytes=b'')
    with pytest.raises(RuntimeError, match='no audio'):
        segment_tts.synthesize_segment(
            project_id='p1', data_root=tmp_path, segment=_segment(),
            previous=None, voice='Male Voice A', provider='edge', language_name='Khmer', lang_code='km',
        )


def test_blank_text_is_refused_before_synthesis(tmp_path, monkeypatch):
    calls = _fake_engine(monkeypatch)
    with pytest.raises(ValueError, match='dubbed text'):
        segment_tts.synthesize_segment(
            project_id='p1', data_root=tmp_path, segment=_segment(text='   '),
            previous=None, voice='Male Voice A', provider='edge', language_name='Khmer', lang_code='km',
        )
    assert calls == []


def test_structural_edit_keeps_clips_whose_text_is_unchanged():
    audio = [
        {'id': '1', 'text': 'Keep me.', 'start': 0, 'end': 2, 'path': 'a.mp3'},
        {'id': '2', 'text': 'Old text.', 'start': 2, 'end': 4, 'path': 'b.mp3'},
        {'id': '3', 'text': 'Deleted.', 'start': 4, 'end': 6, 'path': 'c.mp3'},
    ]
    segments = [
        {'id': '1', 'start': 0.0, 'end': 2.0, 'translation': 'Keep me.'},
        {'id': '2', 'start': 2.0, 'end': 4.0, 'translation': 'Changed text.'},
    ]
    kept = segment_tts.keep_audio_by_id(audio, segments)
    assert [item['id'] for item in kept] == ['1']


def test_structural_edit_refreshes_timing_of_kept_clips():
    audio = [{'id': '1', 'text': 'Same.', 'start': 0, 'end': 2, 'path': 'a.mp3'}]
    segments = [{'id': '1', 'start': 1.5, 'end': 3.5, 'translation': 'Same.'}]
    kept = segment_tts.keep_audio_by_id(audio, segments)
    assert (kept[0]['start'], kept[0]['end']) == (1.5, 3.5)


def test_timing_flags_overlap_invalid_and_beyond_video():
    segments = [
        {'id': 'a', 'start': 0.0, 'end': 3.0, 'translation': 'x'},
        {'id': 'b', 'start': 2.5, 'end': 4.0, 'translation': 'y'},
        {'id': 'c', 'start': 5.0, 'end': 5.0, 'translation': 'z'},
        {'id': 'd', 'start': 6.0, 'end': 20.0, 'translation': 'w'},
    ]
    problems = segment_tts.timing_problems(segments, duration=10.0)
    kinds = {(item['segment'], item['kind']) for item in problems}
    assert ('a', 'overlap') in kinds
    assert ('c', 'invalid') in kinds
    assert ('d', 'beyond_video') in kinds


def test_clean_timeline_has_no_warnings():
    segments = [
        {'id': 'a', 'start': 0.0, 'end': 2.0, 'translation': 'x'},
        {'id': 'b', 'start': 2.0, 'end': 4.0, 'translation': 'y'},
    ]
    assert segment_tts.timing_problems(segments, duration=10.0) == []


def test_add_segment_route_inserts_sorted_segment(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    pid = 'add-seg'
    main.projects[pid] = {
        'id': pid, 'name': 'add', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'segments': [{'id': 'late', 'start': 5.0, 'end': 6.0, 'speaker': 'Speaker 1', 'translation': 'x', 'text': 'x'}],
        'audio_segments': [],
    }
    response = TestClient(main.app).post(f'/api/projects/{pid}/segments', json={'start': 1.0, 'end': 2.5})
    assert response.status_code == 200, response.text
    starts = [item['start'] for item in main.projects[pid]['segments']]
    assert starts == [1.0, 5.0]
    main.projects.pop(pid, None)


def test_add_segment_rejects_too_short(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    pid = 'add-short'
    main.projects[pid] = {'id': pid, 'name': 'short', 'mode': 'DUB', 'media_path': None,
                          'duration': '00:00:10', 'segments': [], 'audio_segments': []}
    response = TestClient(main.app).post(f'/api/projects/{pid}/segments', json={'start': 1.0, 'end': 1.05})
    assert response.status_code == 400
    main.projects.pop(pid, None)


def test_segment_audio_route_serves_clip_by_id(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    clip = tmp_path / 'clip.mp3'
    clip.write_bytes(b'ID3-clip')
    pid = 'by-id'
    main.projects[pid] = {
        'id': pid, 'name': 'id', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'segments': [{'id': 'seg-a', 'start': 0, 'end': 2, 'speaker': 'Speaker 1', 'translation': 'x'}],
        'audio_segments': [{'id': 'seg-a', 'start': 0, 'end': 2, 'text': 'x', 'path': str(clip)}],
    }
    client = TestClient(main.app)
    ok = client.get(f'/api/projects/{pid}/segments/seg-a/audio')
    assert ok.status_code == 200 and ok.content == b'ID3-clip'
    missing = client.get(f'/api/projects/{pid}/segments/seg-z/audio')
    assert missing.status_code == 404
    main.projects.pop(pid, None)


def test_regeneration_leaves_one_clip_per_segment_for_export(tmp_path, monkeypatch):
    """Export mixes project['audio_segments'] by id. Regenerating a line must replace its clip,
    not add a second one, or the export would play the line twice."""
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    def fake_synthesize_line(text, out_path, **kwargs):
        Path(out_path).write_bytes(b'mp3')
        return {'provider': 'edge', 'voice': kwargs['voice_pref'], 'path': out_path}

    monkeypatch.setattr(segment_tts.tts_service, 'synthesize_line', fake_synthesize_line)
    pid = 'regen-once'
    main.projects[pid] = {
        'id': pid, 'name': 'regen', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'target_language': 'Khmer',
        'segments': [
            {'id': 'a', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'First.'},
            {'id': 'b', 'start': 3.0, 'end': 5.0, 'speaker': 'Speaker 2', 'translation': 'Second.'},
        ],
        'audio_segments': [],
    }
    client = TestClient(main.app)
    for _ in range(2):
        response = client.post(f'/api/projects/{pid}/segments/a/tts', json={'gender': 'Female'})
        assert response.status_code == 200, response.text
    clips = main.projects[pid]['audio_segments']
    assert [clip['id'] for clip in clips] == ['a']
    assert clips[0]['gender'] == 'Female'
    main.projects.pop(pid, None)


def test_chosen_gender_is_saved_on_the_segment_before_audio_exists(tmp_path, monkeypatch):
    """A Female choice must survive a reload even when the line has no audio yet."""
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')

    def failing_synthesize_line(text, out_path, **kwargs):
        raise RuntimeError('engine down')

    monkeypatch.setattr(segment_tts.tts_service, 'synthesize_line', failing_synthesize_line)
    pid = 'gender-saved'
    main.projects[pid] = {
        'id': pid, 'name': 'gender', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'target_language': 'Khmer',
        'segments': [{'id': 'g1', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Hi.'}],
        'audio_segments': [],
    }
    response = TestClient(main.app).post(f'/api/projects/{pid}/segments/g1/tts', json={'gender': 'Female'})
    assert response.status_code == 502
    assert main.projects[pid]['segments'][0]['gender'] == 'Female'
    assert main.projects[pid]['audio_segments'] == []
    main.projects.pop(pid, None)


def test_fit_report_flags_overrun_without_truncating():
    report = segment_tts.fit_report(3.4, 2.0)
    assert report['fits'] is False
    assert report['overrun_seconds'] == 1.4


def test_fit_report_accepts_clip_within_slot():
    assert segment_tts.fit_report(1.8, 2.0)['fits'] is True


def test_fit_report_without_duration_is_not_a_warning():
    report = segment_tts.fit_report(None, 2.0)
    assert report['fits'] is True and report['clip_seconds'] is None


def test_generate_route_stores_fit_on_the_clip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'find_media_tool', lambda name: 'ffprobe' if name == 'ffprobe' else None)
    monkeypatch.setattr(main, '_probe_duration', lambda ffprobe, path: 3.5)

    def fake_synthesize_line(text, out_path, **kwargs):
        Path(out_path).write_bytes(b'mp3')
        return {'provider': 'edge', 'voice': kwargs['voice_pref']}

    monkeypatch.setattr(segment_tts.tts_service, 'synthesize_line', fake_synthesize_line)
    pid = 'fit-route'
    main.projects[pid] = {
        'id': pid, 'name': 'fit', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'target_language': 'Khmer',
        'segments': [{'id': 'f1', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Long line.'}],
        'audio_segments': [],
    }
    response = TestClient(main.app).post(f'/api/projects/{pid}/segments/f1/tts', json={'gender': 'Male'})
    assert response.status_code == 200, response.text
    clip = main.projects[pid]['audio_segments'][0]
    assert clip['fit']['fits'] is False
    assert clip['fit']['overrun_seconds'] == 1.5
    assert clip['fit']['clip_seconds'] == 3.5
    main.projects.pop(pid, None)


def test_clip_records_the_engine_that_actually_produced_it(tmp_path, monkeypatch):
    """If the requested engine falls back, the clip must say which engine made it."""
    from backend.app import tts_service

    def fallback_synthesize_line(text, out_path, **kwargs):
        Path(out_path).write_bytes(b'mp3')
        return {'provider': 'sapi5', 'voice': 'Female Voice B', 'path': out_path}

    monkeypatch.setattr(tts_service, 'synthesize_line', fallback_synthesize_line)
    entry = segment_tts.synthesize_segment(
        project_id='p-engine', data_root=tmp_path,
        segment={'id': 'e1', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'សួស្តី'},
        previous=None, voice='Female Voice B', provider='edge', language_name='Khmer', lang_code='km',
    )
    assert entry['voice'] == 'sapi5:Female Voice B'
    assert not entry['voice'].startswith('edge:')


def test_engine_of_clip_reads_the_engine_that_made_it():
    assert segment_tts.engine_of_clip({'voice': 'edge:Female Voice B'}) == 'edge'
    assert segment_tts.engine_of_clip({'voice': 'sapi5:Female Voice B'}) == 'sapi5'


def test_engine_of_clip_falls_back_when_unrecorded():
    assert segment_tts.engine_of_clip({}) == 'auto'
    assert segment_tts.engine_of_clip(None, fallback='edge') == 'edge'
    assert segment_tts.engine_of_clip({'voice': 'no-colon'}) == 'auto'


def test_overlap_fix_resynthesizes_with_the_clip_engine_not_auto(tmp_path, monkeypatch):
    """Regression: an overflowing clip must be re-made with its own engine, not 'auto'."""
    from fastapi.testclient import TestClient

    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'find_media_tool', lambda name: 'ffprobe')
    monkeypatch.setattr(main, '_probe_audio_duration', lambda ffprobe, path: 1.8 if 'fix_temp' in str(path) else 2.4)
    seen: dict[str, object] = {}

    def record_synthesis(text, out_path, **kwargs):
        seen['provider_pref'] = kwargs['provider_pref']
        Path(out_path).write_bytes(b'mp3')
        return {'provider': kwargs['provider_pref'], 'voice': 'v', 'path': out_path}

    monkeypatch.setattr(main.tts_service, 'synthesize_line', record_synthesis)
    clip = tmp_path / 'line.mp3'
    clip.write_bytes(b'mp3')
    pid = 'fix-engine'
    main.projects[pid] = {
        'id': pid, 'name': 'fix', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'target_language': 'Khmer',
        'segments': [{'id': 'o1', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Long line.'}, {'id': 'o2', 'start': 1.0, 'end': 3.0, 'speaker': 'Speaker 1', 'translation': 'Next.'}],
        'audio_segments': [{'id': 'o1', 'start': 0.0, 'end': 2.0, 'text': 'Long line.', 'path': str(clip), 'voice': 'edge:Female Voice B'}, {'id': 'o2', 'start': 1.0, 'end': 3.0, 'text': 'Next.', 'path': str(clip), 'voice': 'edge:Male Voice A'}],
    }
    response = TestClient(main.app).post(f'/api/projects/{pid}/fix-overlaps')
    assert response.status_code == 200, response.text
    assert seen.get('provider_pref') == 'edge'
    main.projects.pop(pid, None)
