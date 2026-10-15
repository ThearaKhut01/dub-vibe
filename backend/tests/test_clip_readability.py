"""Silent or header-only dub clips must never reach the export mix.

A silence trim on a quiet clip writes a file of about 226 bytes with no audio frames. The
old size-only check kept that file, and the export then failed on it. These tests use the
real ffmpeg/ffprobe when available and skip cleanly when they are not installed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from backend.app import main

FFMPEG = shutil.which('ffmpeg') or next(iter(Path('.').glob('**/ffmpeg.exe')), None)
FFPROBE = shutil.which('ffprobe')

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason='needs ffmpeg and ffprobe on PATH (media tools)',
)


def _make_silent_mp3(path: Path, seconds: float = 1.5) -> None:
    subprocess.run(
        [str(FFMPEG), '-y', '-f', 'lavfi', '-i', 'anullsrc=r=22050:cl=mono', '-t', str(seconds),
         '-c:a', 'libmp3lame', str(path)],
        check=True, capture_output=True,
    )


def _make_speech_mp3(path: Path, seconds: float = 1.0) -> None:
    subprocess.run(
        [str(FFMPEG), '-y', '-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=22050:duration={seconds}',
         '-c:a', 'libmp3lame', str(path)],
        check=True, capture_output=True,
    )


def test_speech_clip_is_readable(tmp_path):
    clip = tmp_path / 'speech.mp3'
    _make_speech_mp3(clip)
    assert main._is_readable_audio(str(FFPROBE), str(clip)) is True


def test_silent_trim_output_is_detected_as_unreadable(tmp_path):
    """The bug: the trim of a silent clip is a non-zero but header-only file."""
    clip = tmp_path / 'silent.mp3'
    _make_silent_mp3(clip)
    trimmed = tmp_path / 'trimmed.mp3'
    subprocess.run(
        [str(FFMPEG), '-y', '-i', str(clip),
         '-af', 'silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse,'
                'silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse',
         '-c:a', 'libmp3lame', '-q:a', '4', str(trimmed)],
        check=True, capture_output=True,
    )
    assert trimmed.stat().st_size > 0, 'the old size-only check would have accepted this file'
    assert main._is_readable_audio(str(FFPROBE), str(trimmed)) is False


def test_strip_keeps_original_when_trim_is_unreadable(tmp_path):
    clip = tmp_path / 'dub' / 'silent.mp3'
    clip.parent.mkdir()
    _make_silent_mp3(clip)
    result = main._strip_clip_silence(str(FFMPEG), str(FFPROBE), str(clip))
    assert result == str(clip), 'an unreadable trim must fall back to the original clip'
    assert not (clip.parent / 'clean' / clip.name).exists()


def test_missing_file_is_not_readable(tmp_path):
    assert main._is_readable_audio(str(FFPROBE), str(tmp_path / 'nope.mp3')) is False
