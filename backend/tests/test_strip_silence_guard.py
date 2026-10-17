"""Synthesis-time silence trim must never replace a clip with an unreadable file.

strip_audio_silence runs on every generated line. A fully silent line trims to a header-only
MP3 (not zero bytes), which the old size-only check accepted, and it then broke the export.
These tests use the real ffmpeg/ffprobe and skip cleanly when they are not installed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from backend.app import tts_service

FFMPEG = shutil.which('ffmpeg')
FFPROBE = shutil.which('ffprobe')

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason='needs ffmpeg and ffprobe on PATH (media tools)',
)


def _silent_mp3(path: Path) -> None:
    subprocess.run(
        [FFMPEG, '-y', '-f', 'lavfi', '-i', 'anullsrc=r=22050:cl=mono', '-t', '1.5',
         '-c:a', 'libmp3lame', str(path)],
        check=True, capture_output=True,
    )


def _speech_mp3(path: Path) -> None:
    subprocess.run(
        [FFMPEG, '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=22050:duration=1',
         '-c:a', 'libmp3lame', str(path)],
        check=True, capture_output=True,
    )


def test_silent_clip_keeps_its_original_file(tmp_path, monkeypatch):
    """The bug: a silent clip was overwritten with a header-only file."""
    monkeypatch.setattr(tts_service, 'find_media_tool', lambda name: FFMPEG, raising=False)
    clip = tmp_path / 'silent.mp3'
    _silent_mp3(clip)
    original = clip.read_bytes()
    tts_service.strip_audio_silence(str(clip))
    assert clip.read_bytes() == original, 'an unreadable trim must not replace the original clip'


def test_speech_clip_is_still_trimmed_and_readable(tmp_path):
    clip = tmp_path / 'speech.mp3'
    _speech_mp3(clip)
    tts_service.strip_audio_silence(str(clip))
    assert tts_service._has_audio(clip) is True


def test_has_audio_is_false_for_a_header_only_file(tmp_path):
    empty = tmp_path / 'empty.mp3'
    _silent_mp3(empty)
    trimmed = tmp_path / 'trimmed.mp3'
    subprocess.run(
        [FFMPEG, '-y', '-i', str(empty),
         '-af', 'silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse,'
                'silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse',
         '-c:a', 'libmp3lame', '-q:a', '4', str(trimmed)],
        check=True, capture_output=True,
    )
    assert trimmed.stat().st_size > 0
    assert tts_service._has_audio(trimmed) is False
