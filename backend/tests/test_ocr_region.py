"""The OCR box: one rectangle the user drags, read from a single frame.

The feature is a sibling of the blur box — same normalised 0..1 rectangle, same
drag/resize UI — so these tests pin the shared pixel conversion and the preview
endpoint that turns the box into text.
"""
from __future__ import annotations

import pytest

from backend.app import export_design, main, ocr_subs


# --- the shared rectangle -> pixels conversion ------------------------------

def test_box_to_pixels_matches_the_blur_box_geometry():
    """The blur export must keep using exactly the numbers it used before."""
    x, y, w, h = export_design.box_to_pixels(0.05, 0.78, 0.9, 0.14, 1080, 1920)
    assert (x, y, w, h) == (54, 1498, 972, 269)


def test_blur_filter_still_crops_the_same_rectangle():
    design = export_design.normalize_design({
        'platform': 'original',
        'blur_box': {'enabled': True, 'x': 0.05, 'y': 0.78, 'w': 0.9, 'h': 0.14, 'strength': 20},
    })
    graph, _label = export_design.frame_filter_graph(design, 1080, 1920)
    assert 'crop=972:269:54:1498' in graph


def test_box_to_pixels_keeps_the_rectangle_inside_the_frame():
    x, y, w, h = export_design.box_to_pixels(0.99, 0.99, 1.0, 1.0, 1280, 536)
    assert 0 <= x < 1280 and 0 <= y < 536
    assert x + w <= 1280 and y + h <= 536


# --- request validation -----------------------------------------------------

def test_normalize_box_clamps_and_fills_defaults():
    assert ocr_subs.normalize_box(None) is None
    assert ocr_subs.normalize_box('nope') is None
    assert ocr_subs.normalize_box({}) == {'x': 0.05, 'y': 0.7, 'w': 0.9, 'h': 0.22}
    box = ocr_subs.normalize_box({'x': -3, 'y': 9, 'w': 99, 'h': 0})
    assert box == {'x': 0.0, 'y': 0.98, 'w': 1.0, 'h': 0.02}
    # A box that starts near the edge is shrunk to fit, never pushed outside.
    edge = ocr_subs.normalize_box({'x': 0.9, 'y': 0.9, 'w': 1, 'h': 1})
    assert edge['x'] + edge['w'] <= 1.0 + 1e-9
    assert edge['y'] + edge['h'] <= 1.0 + 1e-9


def test_ocr_language_follows_the_source_language():
    assert ocr_subs.ocr_language('Chinese') == 'ch'
    assert ocr_subs.ocr_language('zh') == 'ch'
    assert ocr_subs.ocr_language('Thai') == 'th'
    assert ocr_subs.ocr_language('Japanese') == 'japan'
    assert ocr_subs.ocr_language('Korean') == 'korean'
    assert ocr_subs.ocr_language('Arabic') == 'ar'
    assert ocr_subs.ocr_language('Russian') == 'ru'
    assert ocr_subs.ocr_language('Hindi') == 'hi'
    assert ocr_subs.ocr_language('English') == 'en'
    assert ocr_subs.ocr_language('Khmer') == 'en'   # no Paddle model: English fallback
    assert ocr_subs.ocr_language('') == 'en'


# --- the preview endpoint ---------------------------------------------------

def _install_project(pid: str, media: str, source_language: str = 'Chinese') -> None:
    main.projects[pid] = {
        'id': pid, 'name': pid, 'mode': 'DUB', 'media_path': media,
        'source_language': source_language, 'segments': [], 'audio_segments': [],
    }


def _post(pid: str, body: dict):
    from fastapi.testclient import TestClient

    return TestClient(main.app).post(f'/api/projects/{pid}/ocr/test', json=body)


def test_ocr_test_needs_a_project_a_video_and_a_box(tmp_path):
    assert _post('missing', {'time': 0, 'box': {'x': 0.1}}).status_code == 404

    pid = 'no-media'
    _install_project(pid, '')
    assert _post(pid, {'time': 0, 'box': {'x': 0.1}}).status_code == 409

    pid = 'no-box'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'not really a video')
    _install_project(pid, str(video))
    assert _post(pid, {'time': 0}).status_code == 409


def test_ocr_test_reports_the_missing_engine_instead_of_crashing(tmp_path, monkeypatch):
    pid = 'no-engine'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'not really a video')
    _install_project(pid, str(video))
    monkeypatch.setattr(main.ocr_subs, 'ocr_available', lambda: False)

    response = _post(pid, {'time': 1.5, 'box': {'x': 0.05, 'y': 0.72, 'w': 0.9, 'h': 0.18}})
    assert response.status_code == 503
    assert 'paddleocr' in response.json()['detail'].lower()


def test_ocr_test_reads_the_frame_and_remembers_the_box(tmp_path, monkeypatch):
    pid = 'ocr-ok'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'not really a video')
    _install_project(pid, str(video))

    seen: dict = {}

    def fake_read_frame(media_path, time, box, source_language='', min_score=0.5):
        seen.update(media_path=media_path, time=time, box=box, source_language=source_language)
        return {
            'text': 'សួស្តី', 'score': 0.94,
            'lines': [{'text': 'សួស្តី', 'score': 0.94}],
            'crop': {'x': 64, 'y': 385, 'w': 1152, 'h': 96,
                     'frame_width': 1280, 'frame_height': 536},
            'language': 'latin',
        }

    monkeypatch.setattr(main.ocr_subs, 'ocr_available', lambda: True)
    monkeypatch.setattr(main.ocr_subs, 'read_frame', fake_read_frame)

    response = _post(pid, {
        'time': 12.5,
        'box': {'x': 0.05, 'y': 0.72, 'w': 0.9, 'h': 0.18},
        'source_language': 'Khmer',
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['text'] == 'សួស្តី'
    assert body['crop']['w'] == 1152
    assert seen['time'] == 12.5
    assert seen['source_language'] == 'Khmer'
    # The box is stored on the project, clamped, ready for the full-video pass.
    assert main.projects[pid]['ocr_box'] == {'x': 0.05, 'y': 0.72, 'w': 0.9, 'h': 0.18}


# --- the full-video extraction ----------------------------------------------

def _post_extract(pid: str, body: dict):
    from fastapi.testclient import TestClient

    return TestClient(main.app).post(f'/api/projects/{pid}/ocr/extract', json=body)


def test_ocr_extract_needs_a_project_a_video_and_a_box(tmp_path, monkeypatch):
    monkeypatch.setattr(main.ocr_subs, 'ocr_available', lambda: True)
    assert _post_extract('missing', {}).status_code == 404

    pid = 'extract-no-media'
    _install_project(pid, '')
    assert _post_extract(pid, {'box': {'x': 0.1}}).status_code == 409

    pid = 'extract-no-box'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'not really a video')
    _install_project(pid, str(video))
    assert _post_extract(pid, {}).status_code == 409


def test_ocr_extract_reports_the_missing_engine(tmp_path, monkeypatch):
    pid = 'extract-no-engine'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'not really a video')
    _install_project(pid, str(video))
    monkeypatch.setattr(main.ocr_subs, 'ocr_available', lambda: False)

    response = _post_extract(pid, {'box': {'x': 0.05, 'y': 0.7, 'w': 0.9, 'h': 0.22}})
    assert response.status_code == 503
    assert 'paddleocr' in response.json()['detail'].lower()


def test_ocr_extract_replaces_the_transcript_and_clears_audio(tmp_path, monkeypatch):
    pid = 'extract-ok'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'not really a video')
    _install_project(pid, str(video))
    main.projects[pid]['segments'] = [
        {'id': 'old', 'start': 0, 'end': 1, 'speaker': 'Speaker 1', 'text': 'old', 'translation': ''},
    ]
    main.projects[pid]['audio_segments'] = [{'id': 'old', 'path': 'old.mp3'}]

    seen: dict = {}

    def fake_extract(media_path, box, source_language='', **options):
        seen.update(media_path=media_path, box=box, source_language=source_language, **options)
        return [
            {'start': 0.0, 'end': 2.5, 'text': 'Hello'},
            {'start': 2.5, 'end': 5.0, 'text': 'World'},
        ]

    monkeypatch.setattr(main.ocr_subs, 'ocr_available', lambda: True)
    monkeypatch.setattr(main.ocr_subs, 'extract_cues', fake_extract)

    response = _post_extract(pid, {
        'box': {'x': 0.05, 'y': 0.7, 'w': 0.9, 'h': 0.22},
        'source_language': 'English',
        'fps': 2,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['count'] == 2
    segments = main.projects[pid]['segments']
    assert [segment['text'] for segment in segments] == ['Hello', 'World']
    assert all(segment['id'].startswith('ocr-') for segment in segments)
    assert all(segment['speaker'] == 'Speaker 1' for segment in segments)
    assert segments[0]['start'] == 0.0 and segments[1]['end'] == 5.0
    # Re-dub needed: old clips belonged to the old transcript.
    assert main.projects[pid]['audio_segments'] == []
    assert seen['fps'] == 2.0
    assert seen['box'] == {'x': 0.05, 'y': 0.7, 'w': 0.9, 'h': 0.22}


def test_ocr_extract_reuses_the_saved_box_and_keeps_the_transcript_when_empty(tmp_path, monkeypatch):
    pid = 'extract-empty'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'not really a video')
    _install_project(pid, str(video))
    main.projects[pid]['ocr_box'] = {'x': 0.1, 'y': 0.6, 'w': 0.8, 'h': 0.3}
    main.projects[pid]['segments'] = [
        {'id': 'keep', 'start': 0, 'end': 1, 'speaker': 'Speaker 1', 'text': 'keep', 'translation': ''},
    ]

    monkeypatch.setattr(main.ocr_subs, 'ocr_available', lambda: True)
    monkeypatch.setattr(main.ocr_subs, 'extract_cues', lambda *args, **kwargs: [])

    response = _post_extract(pid, {})
    assert response.status_code == 200, response.text
    assert response.json()['count'] == 0
    # No text found: the transcript the user already had is not wiped.
    assert main.projects[pid]['segments'][0]['id'] == 'keep'
