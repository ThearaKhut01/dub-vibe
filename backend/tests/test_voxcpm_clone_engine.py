"""VoxCPM clone engine for the DUB pipeline: per-line synthesis and the route gate.

The model is faked, so these run without VoxCPM2 weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import voxcpm_dub, voxcpm_local

# CI installs only the API test packages (see .github/workflows/ci.yml). These guards
# skip the tests that need the audio stack or the HTTP test client, so CI stays green.
# Both still run locally, where the full requirements are installed.
AUDIO_TESTS = pytest.mark.skipif(
    __import__('importlib').util.find_spec('numpy') is None or __import__('importlib').util.find_spec('soundfile') is None,
    reason='needs numpy and soundfile (audio stack)')
HTTP_TESTS = pytest.mark.skipif(
    __import__('importlib').util.find_spec('httpx') is None,
    reason='needs httpx for the FastAPI test client')


class _FakeTtsModel:
    sample_rate = 48000


class _FakeModel:
    tts_model = _FakeTtsModel()

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        import numpy as np

        self.calls.append(kwargs)
        return np.full(4800, 0.2, dtype=np.float32)


@pytest.fixture()
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(voxcpm_local, 'is_installed', lambda: True)
    monkeypatch.setattr(voxcpm_local, '_load', lambda: model)
    return model


@AUDIO_TESTS
def test_clone_line_writes_wav_with_reference(tmp_path, fake_model):
    reference = tmp_path / 'ref.wav'
    reference.write_bytes(b'w')
    out = voxcpm_dub.synthesize_clone_line('Hello there.', reference, tmp_path / 'out' / '0001.wav')
    assert out.is_file() and out.stat().st_size > 0
    assert fake_model.calls == [{
        'text': 'Hello there.',
        'reference_wav_path': str(reference),
        'cfg_value': 2.0,
        'inference_timesteps': 10,
    }]


@AUDIO_TESTS
def test_clone_line_rejects_empty_text(tmp_path, fake_model):
    with pytest.raises(ValueError, match='no translated text'):
        voxcpm_dub.synthesize_clone_line('   ', tmp_path / 'ref.wav', tmp_path / 'o.wav')
    assert fake_model.calls == []


@AUDIO_TESTS
def test_clone_line_errors_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(voxcpm_local, 'is_installed', lambda: False)
    with pytest.raises(RuntimeError, match='not installed'):
        voxcpm_dub.synthesize_clone_line('Hi', tmp_path / 'ref.wav', tmp_path / 'o.wav')


def test_clone_engine_is_not_in_fallback_chain():
    """A clone line must never silently fall back to a generic voice."""
    from backend.app import tts_service

    assert 'voxcpm_clone' not in tts_service.QUALITY_ORDER
    assert 'voxcpm_clone' not in tts_service.OFFLINE_MULTILINGUAL_PROVIDERS


def test_clone_engine_gated_to_verified_languages():
    assert voxcpm_local.supports_language('en')
    assert not voxcpm_local.supports_language('km')


@HTTP_TESTS
def test_process_tts_rejects_clone_for_unverified_language(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main

    media = tmp_path / 'v.mp4'
    media.write_bytes(b'x')
    pid = 'clone-gate'
    main.projects[pid] = {
        'id': pid, 'name': 'gate', 'mode': 'DUB', 'media_path': str(media), 'target_language': 'Khmer',
        'duration': '00:00:10', 'segments': [
            {'id': 1, 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'សួស្តី'},
        ], 'audio_segments': [],
    }
    response = TestClient(main.app).post('/api/process/tts', json={
        'project_id': pid, 'workflow': 'tts', 'target_language': 'Khmer', 'tts_provider': 'voxcpm_clone',
    })
    assert response.status_code == 409
    assert 'verified languages' in response.json()['detail']
    main.projects.pop(pid, None)


@HTTP_TESTS
@AUDIO_TESTS
def test_process_tts_uses_clone_engine_per_line(tmp_path, monkeypatch, fake_model):
    from fastapi.testclient import TestClient

    from backend.app import main

    media = tmp_path / 'v.mp4'
    media.write_bytes(b'x')
    pid = 'clone-run'
    main.projects[pid] = {
        'id': pid, 'name': 'run', 'mode': 'DUB', 'media_path': str(media), 'target_language': 'English',
        'duration': '00:00:12', 'segments': [
            {'id': 1, 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Hello there.'},
            {'id': 2, 'start': 3.0, 'end': 9.0, 'speaker': 'Speaker 2', 'translation': 'A longer line.'},
        ], 'audio_segments': [],
    }
    monkeypatch.setattr(voxcpm_dub, 'dub_dir', lambda _pid: tmp_path / 'work')
    monkeypatch.setattr(voxcpm_dub, '_cut_reference',
                        lambda ffmpeg, source, segment, target: (target.write_bytes(b'w') or True))
    monkeypatch.setattr(main, 'find_media_tool', lambda name: 'ffmpeg')
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')

    response = TestClient(main.app).post('/api/process/tts', json={
        'project_id': pid, 'workflow': 'tts', 'target_language': 'English', 'tts_provider': 'voxcpm_clone',
    })
    assert response.status_code == 200, response.text
    segments = response.json()['result']['segments']
    assert [s['voice'] for s in segments] == ['voxcpm_clone:Speaker 1', 'voxcpm_clone:Speaker 2']
    assert [Path(call['reference_wav_path']).name for call in fake_model.calls] == [
        'ref-Speaker_1.wav', 'ref-Speaker_2.wav',
    ]
    assert all(Path(s['path']).is_file() for s in segments)
    main.projects.pop(pid, None)

@HTTP_TESTS
def test_explicit_voxcpm2_refuses_khmer_instead_of_falling_back_to_edge(tmp_path, monkeypatch):
    """Picking VoxCPM2 for Khmer must fail loudly, not silently dub with Edge."""
    from fastapi.testclient import TestClient

    from backend.app import main

    media = tmp_path / 'v.mp4'
    media.write_bytes(b'x')
    pid = 'voxcpm2-khmer-gate'
    main.projects[pid] = {
        'id': pid, 'name': 'gate2', 'mode': 'DUB', 'media_path': str(media), 'target_language': 'Khmer',
        'duration': '00:00:10', 'segments': [
            {'id': 1, 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'សួស្តី'},
        ], 'audio_segments': [],
    }
    edge_calls: list[str] = []
    from backend.app import tts_service
    monkeypatch.setattr(tts_service, 'synthesize_line', lambda *a, **k: edge_calls.append('called') or {})
    response = TestClient(main.app).post('/api/process/tts', json={
        'project_id': pid, 'workflow': 'tts', 'target_language': 'Khmer', 'tts_provider': 'voxcpm2',
    })
    assert response.status_code == 409
    assert 'only speaks English' in response.json()['detail']
    assert edge_calls == []
    main.projects.pop(pid, None)

@HTTP_TESTS
@AUDIO_TESTS
def test_clone_rerun_reuses_finished_lines(tmp_path, monkeypatch, fake_model):
    """A second Dub run must reuse finished clone lines instead of synthesizing them again."""
    from fastapi.testclient import TestClient

    from backend.app import main

    media = tmp_path / 'v.mp4'
    media.write_bytes(b'x')
    pid = 'clone-resume'
    main.projects[pid] = {
        'id': pid, 'name': 'resume', 'mode': 'DUB', 'media_path': str(media), 'target_language': 'English',
        'duration': '00:00:12', 'segments': [
            {'id': 1, 'start': 0.0, 'end': 2.0, 'speaker': 'Speaker 1', 'translation': 'Hello there.'},
            {'id': 2, 'start': 3.0, 'end': 9.0, 'speaker': 'Speaker 2', 'translation': 'A longer line.'},
        ], 'audio_segments': [],
    }
    monkeypatch.setattr(voxcpm_dub, 'dub_dir', lambda _pid: tmp_path / 'work')
    monkeypatch.setattr(voxcpm_dub, '_cut_reference',
                        lambda ffmpeg, source, segment, target: (target.write_bytes(b'w') or True))
    monkeypatch.setattr(main, 'find_media_tool', lambda name: 'ffmpeg')
    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    client = TestClient(main.app)
    body = {'project_id': pid, 'workflow': 'tts', 'target_language': 'English', 'tts_provider': 'voxcpm_clone'}

    assert client.post('/api/process/tts', json=body).status_code == 200
    assert len(fake_model.calls) == 2

    response = client.post('/api/process/tts', json=body)
    assert response.status_code == 200, response.text
    assert len(fake_model.calls) == 2, 'second run re-synthesized finished lines'
    assert len(response.json()['result']['segments']) == 2
    main.projects.pop(pid, None)