"""The Male/Female voice choice must survive a reload, even before any audio exists."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import main


def test_patch_saves_the_voice_choice_on_the_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    pid = 'gender-patch'
    main.projects[pid] = {
        'id': pid, 'name': 'g', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'segments': [{'id': 'g1', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Hi.'}],
        'audio_segments': [],
    }
    client = TestClient(main.app)
    response = client.patch(f'/api/projects/{pid}/segments/g1', json={'gender': 'Female'})
    assert response.status_code == 200, response.text
    assert main.projects[pid]['segments'][0]['gender'] == 'Female'
    assert main.projects[pid]['audio_segments'] == []
    main.projects.pop(pid, None)


def test_patch_rejects_an_unknown_gender(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    pid = 'gender-bad'
    main.projects[pid] = {
        'id': pid, 'name': 'g', 'mode': 'DUB', 'media_path': None, 'duration': '00:00:10',
        'segments': [{'id': 'g2', 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Hi.'}],
        'audio_segments': [],
    }
    response = TestClient(main.app).patch(f'/api/projects/{pid}/segments/g2', json={'gender': 'Robot'})
    assert response.status_code == 422
    main.projects.pop(pid, None)
