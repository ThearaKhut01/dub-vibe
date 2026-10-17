"""A speed-adjust fix must never be proposed from a re-synthesis with no readable audio.

A header-only result reads as zero length, so it would pass the fit test and then overwrite a
good clip when applied. The route must skip that option instead.

The probe measures the original clip as too long (so the speed-adjust branch runs) and the
re-synthesis as fitting. Without the readable check the header-only file would be proposed.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import main


def _run(tmp_path, monkeypatch, pid: str, readable: bool) -> list[dict]:
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'find_media_tool', lambda name: 'ffprobe')

    def probe(ffprobe, path):
        # The original clip overflows its 2s slot by 20%; the re-synthesis fits comfortably.
        return 1.8 if 'fix_temp' in str(path) else 2.4

    monkeypatch.setattr(main, '_probe_audio_duration', probe)
    monkeypatch.setattr(main, '_is_readable_audio', lambda ffprobe, path: readable)

    def header_only_synthesis(text, out_path, **kwargs):
        Path(out_path).write_bytes(b'ID3-header-only')
        return {'provider': 'edge', 'voice': 'v', 'path': out_path}

    monkeypatch.setattr(main.tts_service, 'synthesize_line', header_only_synthesis)
    project_dir = tmp_path / 'data' / pid
    project_dir.mkdir(parents=True)
    clip = project_dir / 'line.mp3'
    clip.write_bytes(b'good-clip')
    main.projects[pid] = {
        'id': pid, 'name': 'r', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'target_language': 'English',
        'segments': [
            {'id': 'a', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Long.'},
            {'id': 'b', 'start': 3.0, 'end': 5.0, 'speaker': 'Speaker 1', 'translation': 'Next.'},
        ],
        'audio_segments': [
            {'id': 'a', 'start': 0.0, 'end': 2.0, 'text': 'Long.', 'path': str(clip), 'voice': 'edge:v'},
            {'id': 'b', 'start': 3.0, 'end': 5.0, 'text': 'Next.', 'path': str(clip), 'voice': 'edge:v'},
        ],
    }
    response = TestClient(main.app).post(f'/api/projects/{pid}/fix-overlaps')
    assert response.status_code == 200, response.text
    main.projects.pop(pid, None)
    return response.json().get('proposals', [])


def test_header_only_resynthesis_is_not_proposed_as_speed_adjust(tmp_path, monkeypatch):
    proposals = _run(tmp_path, monkeypatch, 'fix-readable-a', readable=False)
    speed = [p for p in proposals if p.get('type') == 'speed_adjust']
    assert speed == [], 'an unreadable re-synthesis must not become a speed-adjust proposal'


def test_readable_resynthesis_is_still_proposed(tmp_path, monkeypatch):
    """Positive control: when the re-synthesis is readable, the speed-adjust fix is offered."""
    proposals = _run(tmp_path, monkeypatch, 'fix-readable-b', readable=True)
    assert any(p.get('type') == 'speed_adjust' for p in proposals)
