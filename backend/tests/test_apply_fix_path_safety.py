"""Fix Overlaps apply must never move files that live outside the project folder.

The proposal's file paths come from the client. Without the containment check, a request could
point a clip at any file on disk and overwrite it.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import main


def _setup(tmp_path, monkeypatch, pid: str):
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    project_dir = tmp_path / 'data' / pid
    project_dir.mkdir(parents=True)
    clip = project_dir / 'line.mp3'
    clip.write_bytes(b'original-clip')
    main.projects[pid] = {
        'id': pid, 'name': 'safe', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'segments': [], 'audio_segments': [
            {'id': 's1', 'start': 0.0, 'end': 2.0, 'text': 'x', 'path': str(clip), 'voice': 'edge:v'},
        ],
    }
    return project_dir, clip


def test_inside_project_dir_accepts_only_project_paths(tmp_path, monkeypatch):
    project_dir, clip = _setup(tmp_path, monkeypatch, 'safe-a')
    assert main._inside_project_dir('safe-a', str(clip)) is True
    assert main._inside_project_dir('safe-a', str(tmp_path / 'elsewhere.mp3')) is False
    assert main._inside_project_dir('safe-a', None) is False
    main.projects.pop('safe-a', None)


def test_apply_refuses_to_overwrite_a_file_outside_the_project(tmp_path, monkeypatch):
    pid = 'safe-b'
    project_dir, clip = _setup(tmp_path, monkeypatch, pid)
    victim = tmp_path / 'victim.txt'
    victim.write_bytes(b'do-not-touch')
    evil_source = tmp_path / 'evil.mp3'
    evil_source.write_bytes(b'overwrite')

    response = TestClient(main.app).post(
        f'/api/projects/{pid}/fix-overlaps/apply',
        json={'proposals': [{'segment_id': 's1', 'type': 'speed_adjust',
                             'new_audio_path': str(evil_source), 'new_speed': 1.1}]},
    )
    assert response.status_code == 200, response.text
    assert victim.read_bytes() == b'do-not-touch'
    assert clip.read_bytes() == b'original-clip'
    main.projects.pop(pid, None)


def test_apply_still_fixes_a_clip_inside_the_project(tmp_path, monkeypatch):
    pid = 'safe-c'
    project_dir, clip = _setup(tmp_path, monkeypatch, pid)
    fixed = project_dir / 'fix_temp_s1.mp3'
    fixed.write_bytes(b'fixed-audio')

    response = TestClient(main.app).post(
        f'/api/projects/{pid}/fix-overlaps/apply',
        json={'proposals': [{'segment_id': 's1', 'type': 'speed_adjust',
                             'new_audio_path': str(fixed), 'new_speed': 1.1}]},
    )
    assert response.status_code == 200, response.text
    assert clip.read_bytes() == b'fixed-audio'
    main.projects.pop(pid, None)
