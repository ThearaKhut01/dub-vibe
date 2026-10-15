"""Per-segment dubbing: synthesize one subtitle segment on demand.

The Dubbing Track editor regenerates a single line without re-running the whole dub.
This module reuses the same synthesis path as the full dub (tts_service.synthesize_line,
with its provider fallback, cache, and acting), so there is no second TTS engine.

Failure rule: the stored audio for a segment is replaced only after new audio is written
successfully. A failed regeneration leaves the previous audio in place.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import tts_service

GENDER_VOICES = {'male': 'Male Voice A', 'female': 'Female Voice B'}


def voice_for_gender(gender: str | None) -> str:
    """Map the editor's Male/Female choice to the app's Edge voice label."""
    return GENDER_VOICES.get(str(gender or '').strip().lower(), GENDER_VOICES['male'])


def segment_audio_dir(project_id: str, data_root: Path) -> Path:
    directory = data_root / project_id / 'segment-audio'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def synthesize_segment(
    *,
    project_id: str,
    data_root: Path,
    segment: dict[str, Any],
    previous: dict[str, Any] | None,
    voice: str,
    provider: str,
    language_name: str,
    lang_code: str,
    speed: float = 1.0,
) -> dict[str, Any]:
    """Synthesize `segment['translation']` and return the new audio entry.

    Writes to a temporary file first and only then replaces the segment's audio file, so a
    failed run never leaves a half-written or missing clip. Raises RuntimeError on failure.
    """
    text = str(segment.get('translation') or '').strip()
    if not text:
        raise ValueError('Write the dubbed text before generating audio.')

    directory = segment_audio_dir(project_id, data_root)
    segment_id = str(segment['id'])
    final_path = directory / f'{segment_id}.mp3'
    temp_path = directory / f'{segment_id}.tmp.mp3'

    try:
        result = tts_service.synthesize_line(
            text, str(temp_path), provider_pref=provider, language_name=language_name,
            lang_code=lang_code, voice_pref=voice, model_pref=None, speed=speed,
            emotion='Auto', intensity=70, character_profile=None,
            analysis={'emotion': 'Neutral', 'intensity': 70, 'style': 'Natural',
                      'pacing': 'Natural', 'pause_before': False, 'analyzer': 'none'},
        )
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise RuntimeError('The voice engine returned no audio.')
        shutil.move(str(temp_path), str(final_path))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        'id': segment['id'],
        'start': segment['start'],
        'end': segment['end'],
        'source_start': float(segment['start']),
        'text': text,
        'character': segment.get('speaker') or 'Speaker 1',
        'voice': f"{result['provider']}:{result['voice']}",
        'gender': 'Female' if voice == GENDER_VOICES['female'] else 'Male',
        'path': str(final_path),
        'generated_text': text,
        'previous_path': previous.get('path') if previous else None,
    }


def keep_audio_by_id(
    old_audio: list[dict[str, Any]], segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep audio for segments that still exist and whose text did not change.

    Used after structural edits (split, merge, delete). The old full-project reset forced a
    complete re-dub; keeping per-id audio lets the editor regenerate only what changed.
    A clip is kept only when its text still matches the segment's current translation.
    """
    current = {str(segment['id']): segment for segment in segments}
    kept: list[dict[str, Any]] = []
    for audio in old_audio:
        segment = current.get(str(audio.get('id')))
        if segment is None:
            continue
        if str(audio.get('text') or '') != str(segment.get('translation') or ''):
            continue
        kept.append({**audio, 'start': segment['start'], 'end': segment['end']})
    return kept


def timing_problems(segments: list[dict[str, Any]], duration: float | None) -> list[dict[str, Any]]:
    """Find invalid timecodes, overlaps, and out-of-range segments for the editor to warn about."""
    problems: list[dict[str, Any]] = []
    ordered = sorted(segments, key=lambda item: float(item.get('start') or 0))
    for index, segment in enumerate(ordered):
        sid = str(segment.get('id'))
        start = float(segment.get('start') or 0)
        end = float(segment.get('end') or 0)
        if end <= start:
            problems.append({'segment': sid, 'kind': 'invalid', 'message': 'End is not after start.'})
        if duration is not None and end > duration + 0.05:
            problems.append({'segment': sid, 'kind': 'beyond_video', 'message': 'Segment ends after the video.'})
        if index + 1 < len(ordered):
            nxt = ordered[index + 1]
            if end > float(nxt.get('start') or 0) + 1e-6:
                problems.append({
                    'segment': sid, 'kind': 'overlap',
                    'message': f'Overlaps the next segment by {end - float(nxt.get("start") or 0):.2f}s.',
                })
    return problems


def fit_report(clip_seconds: float | None, slot_seconds: float) -> dict[str, Any]:
    """Compare a generated clip with its time slot. Speech is never cut to fit: an overrun is
    reported so the user can shorten the text or retime the line."""
    slot = max(0.0, float(slot_seconds))
    if clip_seconds is None:
        return {'clip_seconds': None, 'slot_seconds': round(slot, 3), 'overrun_seconds': 0.0, 'fits': True}
    clip = round(float(clip_seconds), 3)
    overrun = round(max(0.0, clip - slot), 3)
    return {
        'clip_seconds': clip,
        'slot_seconds': round(slot, 3),
        'overrun_seconds': overrun,
        'fits': overrun <= 0.05,
    }


def engine_of_clip(clip: dict[str, Any] | None, fallback: str = 'auto') -> str:
    """The TTS engine that produced a clip, read from its recorded voice ('engine:voice').

    Used so re-synthesis (overlap fixes) keeps the same engine instead of re-routing through
    'auto', which can pick a different engine (for example a generic fallback voice).
    """
    voice = str((clip or {}).get('voice') or '')
    engine = voice.split(':', 1)[0].strip() if ':' in voice else ''
    return engine or fallback
