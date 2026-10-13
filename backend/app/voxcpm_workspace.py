"""VoxCPM Mode: standalone AI voice generation (no video required).

Four generation modes, matching the official VoxCPM2 API:
  tts       plain text to speech
  design    voice design: a natural-language description prepended as "(description)text"
  clone     controllable cloning from one authorized reference clip (reference_wav_path)
  ultimate  ultimate cloning: reference clip plus its exact transcript (prompt_wav_path + prompt_text)

Jobs run on a background thread. Cancellation is checked between streamed chunks, so
a cancelled job stops early and writes no file. Output is 16-bit WAV at the model's
native sample rate (48 kHz for VoxCPM2). The model is loaded on first use, shared by
jobs, and released after an idle period or on request, so it does not stay in memory.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .providers import models_root

MODES = ('tts', 'design', 'clone', 'ultimate')
MAX_TEXT_CHARS = 1000
MAX_REFERENCE_BYTES = 25 * 1024 * 1024
IDLE_UNLOAD_SECONDS = 600

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_GENERATE_LOCK = threading.Lock()
_last_used = 0.0
_idle_timer: threading.Timer | None = None


def outputs_dir() -> Path:
    directory = models_root().parent / 'voxcpm-outputs'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def references_dir() -> Path:
    directory = models_root().parent / 'voxcpm-references'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized request or raise ValueError with a user-facing message."""
    mode = str(payload.get('mode') or '').strip().lower()
    if mode not in MODES:
        raise ValueError(f'Unknown mode "{mode}". Use one of: {", ".join(MODES)}.')

    text = str(payload.get('text') or '').strip()
    if not text:
        raise ValueError('Enter the text to speak.')
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f'Text is too long ({len(text)} characters). The limit is {MAX_TEXT_CHARS}.')

    description = str(payload.get('description') or '').strip()
    if mode == 'design':
        if not description:
            raise ValueError('Describe the voice you want (for example: "a warm, calm female narrator").')
        if '(' in description or ')' in description:
            raise ValueError('Leave out parentheses in the voice description; they are added automatically.')

    reference_id = str(payload.get('reference_id') or '').strip()
    reference_path = None
    if mode in ('clone', 'ultimate'):
        if not reference_id:
            raise ValueError('Upload a reference voice recording for this mode.')
        reference_path = references_dir() / reference_id
        if not reference_path.is_file():
            raise ValueError('The reference recording was not found. Upload it again.')

    prompt_text = str(payload.get('prompt_text') or '').strip()
    if mode == 'ultimate' and not prompt_text:
        raise ValueError('Ultimate cloning needs the exact transcript of the reference recording.')

    def _number(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(payload.get(name, default))
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} must be a number.') from error
        if not low <= value <= high:
            raise ValueError(f'{name} must be between {low:g} and {high:g}.')
        return value

    return {
        'mode': mode,
        'text': text,
        'description': description,
        'reference_path': str(reference_path) if reference_path else None,
        'prompt_text': prompt_text if mode == 'ultimate' else None,
        'cfg_value': _number('cfg_value', 2.0, 1.0, 3.0),
        'inference_timesteps': int(_number('inference_timesteps', 10, 4, 30)),
        'speed': _number('speed', 1.0, 0.5, 1.5),
    }


def compose_text(request: dict[str, Any]) -> str:
    """The text as the model expects it. Voice design uses the official "(description)text" form."""
    if request['mode'] == 'design':
        return f"({request['description']}){request['text']}"
    return request['text']


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()
    _schedule_idle_unload()


def _schedule_idle_unload() -> None:
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
    _idle_timer = threading.Timer(IDLE_UNLOAD_SECONDS, _unload_if_idle)
    _idle_timer.daemon = True
    _idle_timer.start()


def _unload_if_idle() -> None:
    with _GENERATE_LOCK:
        if time.monotonic() - _last_used >= IDLE_UNLOAD_SECONDS and not _any_running_locked():
            unload_model()


def _any_running() -> bool:
    """Caller must hold _JOBS_LOCK, or use _any_running_locked() instead."""
    return any(job['status'] == 'running' for job in _JOBS.values())


def _any_running_locked() -> bool:
    with _JOBS_LOCK:
        return _any_running()


def unload_model() -> None:
    from . import voxcpm_local

    voxcpm_local.unload()


def _run_job(job_id: str, request: dict[str, Any]) -> None:
    """Run one generation in a child process; map its JSON-line messages onto the job.

    The child exits after each job, so the OS reclaims the model's memory. Cancel
    terminates the child, which is the only way to stop a generation mid-run.
    """
    global _last_used
    import json
    import subprocess
    import sys

    job = _JOBS[job_id]
    output_path = outputs_dir() / f'{job_id}.wav'
    request_path = outputs_dir() / f'{job_id}.request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    command = [sys.executable, '-m', 'backend.app.voxcpm_worker', str(request_path), str(output_path)]
    finished = False
    try:
        with _GENERATE_LOCK:
            job.update(message='Starting generation process...')
            child = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                encoding='utf-8', cwd=str(Path(__file__).resolve().parents[2]),
            )
            try:
                for line in child.stdout:
                    if job.get('cancel'):
                        child.terminate()
                        break
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue
                    kind = message.get('kind')
                    if kind == 'progress':
                        job.update(progress=int(message['percent']), message=str(message['text']))
                    elif kind == 'done':
                        job.update(status='done', progress=100, message='Done',
                                   output=message['output'], sample_rate=message['sample_rate'],
                                   duration=message['duration'])
                        finished = True
                    elif kind == 'error':
                        job.update(status='error', message=str(message['text']),
                                   error_code=str(message['code']))
                        finished = True
            finally:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=15)
            if job.get('cancel') and not finished:
                job.update(status='cancelled', message='Cancelled.')
            elif not finished:
                job.update(status='error', message='The generation process ended unexpectedly.',
                           error_code='failed')
    except Exception as error:  # noqa: BLE001 - surface the real reason to the UI
        job.update(status='error', message=f'Generation failed: {error}', error_code='failed')
    finally:
        request_path.unlink(missing_ok=True)
        _last_used = time.monotonic()
        _schedule_idle_unload()

def _change_speed(waveform, sample_rate: int, speed: float):
    """Resample to change pace; speed > 1 is faster and shorter."""
    import numpy as np

    new_length = max(1, int(round(waveform.size / speed)))
    old_positions = np.linspace(0, waveform.size - 1, num=waveform.size)
    new_positions = np.linspace(0, waveform.size - 1, num=new_length)
    return np.interp(new_positions, old_positions, waveform).astype(np.float32)


def start_job(payload: dict[str, Any]) -> dict[str, Any]:
    request = validate_request(payload)
    job_id = uuid.uuid4().hex[:12]
    job = {
        'id': job_id,
        'mode': request['mode'],
        'status': 'running',
        'progress': 0,
        'message': 'Queued...',
        'cancel': False,
        'output': None,
        'sample_rate': None,
        'duration': None,
        'error_code': None,
    }
    with _JOBS_LOCK:
        if _any_running():
            raise ValueError('A VoxCPM generation is already running. Wait for it or cancel it first.')
        _JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job_id, request), name=f'voxcpm-{job_id}', daemon=True).start()
    return public_job(job_id)


def public_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return {key: value for key, value in job.items() if key != 'cancel'}


def cancel_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job['status'] == 'running':
            job['cancel'] = True
            job['message'] = 'Cancelling...'
    return public_job(job_id)


def save_reference(data: bytes, filename: str) -> str:
    """Store an uploaded reference recording and return its id. Accepts WAV/MP3/FLAC/OGG."""
    suffix = Path(filename or '').suffix.lower()
    if suffix not in ('.wav', '.mp3', '.flac', '.ogg', '.m4a'):
        raise ValueError('Reference audio must be a WAV, MP3, FLAC, OGG, or M4A file.')
    if not data:
        raise ValueError('The uploaded reference file is empty.')
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError('The reference file is larger than 25 MB. Use a shorter clip.')
    reference_id = f'{uuid.uuid4().hex[:12]}{suffix}'
    (references_dir() / reference_id).write_bytes(data)
    return reference_id
