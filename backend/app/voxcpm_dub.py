"""VoxCPM Dub: clone each speaker's voice for the DUB pipeline.

Each translated line is synthesized with VoxCPM2 in clone mode. The reference clip is cut
from the project's source audio at the speaker's own speech, so the output keeps that voice.
The result uses the same `audio_segments` shape as the Edge/MMS dub path, so overlap fitting
and export work unchanged. The engine is selected in DUB mode as `voxcpm_clone`.

VoxCPM2 is only verified for English in this app (see voxcpm_local.VERIFIED_LANGUAGE_CODES),
so other target languages are rejected up front rather than producing garbled audio.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .providers import get_storage_dir

REFERENCE_MIN_SECONDS = 4.0
REFERENCE_MAX_SECONDS = 12.0

def dub_dir(project_id: str) -> Path:
    # Project work files live with the rest of the project data (same root as
    # main.DATA_ROOT), so deleting a project removes them and DUBVIBE_DATA_DIR applies.
    directory = get_storage_dir('data') / project_id / 'voxcpm-dub'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def check_project(project: dict[str, Any], target_language: str) -> None:
    """Raise ValueError with a user-facing message when the project can't be dubbed with VoxCPM."""
    from . import voxcpm_local

    if not voxcpm_local.supports_language(_lang_code(target_language)):
        raise ValueError(
            f'VoxCPM dubbing supports only the verified languages ({", ".join(sorted(voxcpm_local.VERIFIED_LANGUAGE_CODES))}). '
            f'"{target_language}" is not verified. Use DUB MODE for this language.'
        )
    if not project.get('media_path') or not Path(str(project['media_path'])).is_file():
        raise ValueError('Import a video into the project first.')
    segments = project.get('segments') or []
    if not segments:
        raise ValueError('Transcribe the video first.')
    if any(not segment.get('translation') for segment in segments):
        raise ValueError('Translate the transcript before VoxCPM dubbing.')


def _lang_code(language: str) -> str:
    from .providers import _resolve_lang_code

    return _resolve_lang_code(language)


def _speaker_key(segment: dict[str, Any]) -> str:
    return str(segment.get('speaker') or 'Speaker 1')


def _reference_name(key: str) -> str:
    return 'ref-' + ''.join(ch if ch.isalnum() else '_' for ch in key) + '.wav'


def _cut_reference(ffmpeg: str, source: str, segment: dict[str, Any], target: Path) -> bool:
    """Cut a clean reference clip for one line from the source audio. True when the clip exists."""
    start = float(segment.get('start') or 0)
    end = float(segment.get('end') or start)
    length = min(REFERENCE_MAX_SECONDS, max(REFERENCE_MIN_SECONDS, end - start))
    command = [
        ffmpeg, '-y', '-ss', f'{start:.3f}', '-t', f'{length:.3f}', '-i', source,
        '-vn', '-ac', '1', '-ar', '48000', '-acodec', 'pcm_s16le', str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and target.is_file() and target.stat().st_size > 0


def build_references(ffmpeg: str, project: dict[str, Any], work: Path) -> dict[str, Path]:
    """One reference clip per speaker, taken from that speaker's longest line.

    Longer lines give VoxCPM more of the voice to copy, so the longest clean line is used.
    """
    work.mkdir(parents=True, exist_ok=True)
    source = str(project['media_path'])
    best: dict[str, dict[str, Any]] = {}
    for segment in project.get('segments') or []:
        key = _speaker_key(segment)
        length = float(segment.get('end') or 0) - float(segment.get('start') or 0)
        if key not in best or length > float(best[key].get('end') or 0) - float(best[key].get('start') or 0):
            best[key] = segment
    references: dict[str, Path] = {}
    for key, segment in best.items():
        target = work / _reference_name(key)
        if _cut_reference(ffmpeg, source, segment, target):
            references[key] = target
    return references


CLONE_ENGINE = 'voxcpm_clone'


def synthesize_clone_line(text: str, reference: Path, out_path: Path) -> Path:
    """Speak one translated line in the reference speaker's voice. Raises on failure.

    Used by the DUB pipeline when the voice engine is VoxCPM clone. It never falls back
    to another engine, so a line can't silently lose the cloned voice.
    """
    import numpy as np
    import soundfile as sf

    from . import voxcpm_local

    if not voxcpm_local.is_installed():
        raise RuntimeError('VoxCPM2 is not installed. Download it from AI Models first.')
    text = text.strip()
    if not text:
        raise ValueError('Line has no translated text to speak.')
    model = voxcpm_local._load()
    sample_rate = int(getattr(model.tts_model, 'sample_rate', 48000))
    with voxcpm_local._INFERENCE_LOCK:
        waveform = model.generate(
            text=text, reference_wav_path=str(reference), cfg_value=2.0, inference_timesteps=10,
        )
    array = np.asarray(waveform, dtype=np.float32).squeeze()
    if array.size == 0:
        raise RuntimeError('VoxCPM2 produced empty audio for a line.')
    peak = float(np.max(np.abs(array)))
    if peak > 1.0:
        array = array / peak
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), array, sample_rate, subtype='PCM_16')
    return out_path
