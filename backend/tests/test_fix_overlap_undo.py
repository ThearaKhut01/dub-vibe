"""Auto-Fix Overlaps must be reversible.

Applying a speed fix moves the re-synthesized clip over the original file, so the
previous audio is unrecoverable unless the apply step saves it first. These tests
drive the real endpoints and check both the timing and the audio bytes.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import main


def _project(tmp_path, monkeypatch, project_id: str, name: str = 'clip.mp3', body: bytes = b'original-bytes') -> tuple[str, str]:
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    data_root = tmp_path / 'data'
    project_dir = data_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    clip = project_dir / name
    clip.write_bytes(body)
    main.projects[project_id] = {
        'id': project_id, 'name': 'p', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'segments': [{'id': 's1', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'text': 'hi', 'translation': 'hi'}],
        'audio_segments': [{'id': 's1', 'start': 0.0, 'end': 2.0, 'path': str(clip), 'voice': 'edge:km-KH-PisethNeural'}],
    }
    return project_id, str(clip)


def test_apply_snapshots_the_clip_and_undo_restores_it(tmp_path, monkeypatch):
    project_id, original = _project(tmp_path, monkeypatch, 'undo-ok')
    client = TestClient(main.app)
    # A re-synthesized clip sits next to the original until it is applied.
    replacement = tmp_path / 'data' / project_id / 'fix_temp_s1.mp3'
    replacement.write_bytes(b'new-clip-bytes')

    applied = client.post(f'/api/projects/{project_id}/fix-overlaps/apply', json={'proposals': [
        {'segment_id': 's1', 'type': 'speed_adjust', 'new_audio_path': str(replacement),
         'new_start': 0.0, 'new_end': 2.0, 'new_speed': 1.2},
    ]})
    assert applied.status_code == 200, applied.text
    body = applied.json()
    assert body['applied'] == 1
    assert body['can_undo'] is True
    assert (tmp_path / 'data' / project_id / 'clip.mp3').read_bytes() == b'new-clip-bytes'

    undone = client.post(f'/api/projects/{project_id}/fix-overlaps/undo')
    assert undone.status_code == 200, undone.text
    assert undone.json()['audio_segments'][0]['start'] == 0.0
    # The original audio comes back, not just the metadata.
    assert (tmp_path / 'data' / project_id / 'clip.mp3').read_bytes() == b'original-bytes'
    # The backup is consumed: a second undo has nothing to restore.
    assert client.post(f'/api/projects/{project_id}/fix-overlaps/undo').status_code == 409
    main.projects.pop(project_id, None)


def test_undo_restores_the_previous_timing(tmp_path, monkeypatch):
    project_id, _clip = _project(tmp_path, monkeypatch, 'undo-timing', name='b.mp3')
    client = TestClient(main.app)

    applied = client.post(f'/api/projects/{project_id}/fix-overlaps/apply', json={'proposals': [
        {'segment_id': 's1', 'type': 'silence_absorb', 'new_start': 0.5},
    ]})
    assert applied.status_code == 200, applied.text
    assert main.projects[project_id]['audio_segments'][0]['start'] == 0.5

    undone = client.post(f'/api/projects/{project_id}/fix-overlaps/undo')
    assert undone.status_code == 200, undone.text
    assert main.projects[project_id]['audio_segments'][0]['start'] == 0.0
    main.projects.pop(project_id, None)


def test_undo_without_a_snapshot_is_refused(tmp_path, monkeypatch):
    project_id, _clip = _project(tmp_path, monkeypatch, 'undo-empty', name='c.mp3')
    response = TestClient(main.app).post(f'/api/projects/{project_id}/fix-overlaps/undo')
    assert response.status_code == 409
    main.projects.pop(project_id, None)


def test_workspace_reports_whether_undo_is_available(tmp_path, monkeypatch):
    project_id, _clip = _project(tmp_path, monkeypatch, 'undo-workspace', name='d.mp3')
    client = TestClient(main.app)
    assert client.get(f'/api/projects/{project_id}/workspace').json()['can_undo_overlaps'] is False

    client.post(f'/api/projects/{project_id}/fix-overlaps/apply', json={'proposals': [
        {'segment_id': 's1', 'type': 'silence_absorb', 'new_start': 0.4},
    ]})
    assert client.get(f'/api/projects/{project_id}/workspace').json()['can_undo_overlaps'] is True

    client.post(f'/api/projects/{project_id}/fix-overlaps/undo')
    assert client.get(f'/api/projects/{project_id}/workspace').json()['can_undo_overlaps'] is False
    main.projects.pop(project_id, None)
