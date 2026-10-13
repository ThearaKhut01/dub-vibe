"""VoxCPM Mode workspace: request validation, voice-design text form, and model verification.

These run without the model: they cover the rules the UI and API depend on. End-to-end
generation was verified separately against the real VoxCPM2 weights.
"""
from __future__ import annotations

import pytest

from backend.app import voxcpm_local, voxcpm_workspace as workspace


@pytest.fixture()
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('DUBVIBE_MODELS_DIR', str(tmp_path))
    return tmp_path


def test_plain_tts_request_gets_documented_defaults(models_dir):
    request = workspace.validate_request({'mode': 'tts', 'text': 'Hello there.'})
    assert request['cfg_value'] == 2.0
    assert request['inference_timesteps'] == 10
    assert request['speed'] == 1.0
    assert workspace.compose_text(request) == 'Hello there.'


def test_voice_design_wraps_description_in_parentheses(models_dir):
    request = workspace.validate_request(
        {'mode': 'design', 'text': 'Good morning.', 'description': 'a warm calm narrator'}
    )
    assert workspace.compose_text(request) == '(a warm calm narrator)Good morning.'


@pytest.mark.parametrize('payload, message', [
    ({'mode': 'nope', 'text': 'x'}, 'Unknown mode'),
    ({'mode': 'tts', 'text': '   '}, 'Enter the text'),
    ({'mode': 'tts', 'text': 'x' * (workspace.MAX_TEXT_CHARS + 1)}, 'too long'),
    ({'mode': 'design', 'text': 'x', 'description': ''}, 'Describe the voice'),
    ({'mode': 'design', 'text': 'x', 'description': 'calm (soft)'}, 'parentheses'),
    ({'mode': 'tts', 'text': 'x', 'cfg_value': 9}, 'cfg_value must be between'),
    ({'mode': 'tts', 'text': 'x', 'inference_timesteps': 'abc'}, 'must be a number'),
])
def test_invalid_requests_are_refused_with_a_clear_message(models_dir, payload, message):
    with pytest.raises(ValueError, match=message):
        workspace.validate_request(payload)


def test_clone_requires_an_uploaded_reference(models_dir):
    with pytest.raises(ValueError, match='Upload a reference'):
        workspace.validate_request({'mode': 'clone', 'text': 'x'})


def test_ultimate_cloning_requires_the_reference_transcript(models_dir):
    reference = workspace.save_reference(b'RIFFdata', 'voice.wav')
    with pytest.raises(ValueError, match='exact transcript'):
        workspace.validate_request({'mode': 'ultimate', 'text': 'x', 'reference_id': reference})


def test_ultimate_cloning_accepts_reference_with_transcript(models_dir):
    reference = workspace.save_reference(b'RIFFdata', 'voice.wav')
    request = workspace.validate_request(
        {'mode': 'ultimate', 'text': 'x', 'reference_id': reference, 'prompt_text': 'Hi there.'}
    )
    assert request['prompt_text'] == 'Hi there.'
    assert request['reference_path'].endswith('.wav')


@pytest.mark.parametrize('name, size, message', [
    ('virus.exe', 1, 'WAV, MP3, FLAC, OGG, or M4A'),
    ('voice.wav', 0, 'empty'),
    ('voice.wav', workspace.MAX_REFERENCE_BYTES + 1, '25 MB'),
])
def test_reference_upload_rules(models_dir, name, size, message):
    # Build the payload here so pytest's failure output never echoes megabytes of data.
    with pytest.raises(ValueError, match=message):
        workspace.save_reference(b'x' * size, name)


def test_verification_rejects_a_truncated_weights_file(models_dir):
    target = voxcpm_local.model_dir()
    (target / 'model.safetensors').write_bytes(b'x' * 1024)
    ok, reason = voxcpm_local.verify_files()
    assert ok is False
    assert 'incomplete' in reason


def test_verification_reports_missing_files(models_dir):
    ok, reason = voxcpm_local.verify_files()
    assert ok is False
    assert 'missing' in reason


def test_verification_rejects_wrong_architecture(models_dir):
    target = voxcpm_local.model_dir()
    # Size-only placeholders with the official byte counts, so verification reaches the config check.
    for name, size in voxcpm_local.OFFICIAL_FILE_BYTES.items():
        with open(target / name, 'wb') as handle:
            handle.truncate(size)
    (target / 'config.json').write_text('{"architecture": "voxcpm"}', encoding='utf-8')
    ok, reason = voxcpm_local.verify_files()
    assert ok is False
    assert 'expected "voxcpm2"' in reason


def test_is_installed_is_false_without_verified_files(models_dir):
    assert voxcpm_local.is_installed() is False


def test_start_job_is_refused_while_another_job_runs(models_dir, monkeypatch):
    monkeypatch.setitem(workspace._JOBS, 'busy', {'status': 'running'})
    with pytest.raises(ValueError, match='already running'):
        workspace.start_job({'mode': 'tts', 'text': 'Hello.'})
    workspace._JOBS.pop('busy', None)


def test_cancel_marks_a_running_job_and_leaves_finished_jobs_alone(models_dir, monkeypatch):
    monkeypatch.setitem(workspace._JOBS, 'run1', {'id': 'run1', 'status': 'running', 'cancel': False, 'message': ''})
    monkeypatch.setitem(workspace._JOBS, 'done1', {'id': 'done1', 'status': 'done', 'cancel': False, 'message': ''})
    assert workspace.cancel_job('run1')['status'] == 'running'
    assert workspace._JOBS['run1']['cancel'] is True
    workspace.cancel_job('done1')
    assert workspace._JOBS['done1']['cancel'] is False
    workspace._JOBS.pop('run1', None)
    workspace._JOBS.pop('done1', None)
