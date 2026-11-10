"""The fitter really shortens an overrunning clip (pitch preserved)."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from backend.app import segment_tts


def _ffmpeg() -> str:
    return shutil.which('ffmpeg') or shutil.which('ffmpeg.exe') or ''


def _ffprobe() -> str:
    return shutil.which('ffprobe') or shutil.which('ffprobe.exe') or ''


@pytest.mark.skipif(not _ffmpeg() or not _ffprobe(), reason='ffmpeg/ffprobe not on PATH')
def test_an_overrunning_clip_is_time_compressed(tmp_path) -> None:
    clip = tmp_path / 'line.wav'
    subprocess.run(
        [_ffmpeg(), '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
         '-c:a', 'pcm_s16le', str(clip)],
        check=True, capture_output=True,
    )

    report = segment_tts.fit_clip_to_slot(str(clip), 1.5, ffmpeg=_ffmpeg(), ffprobe=_ffprobe())

    assert report['tempo'] > 1.0
    assert report['clip_seconds'] < 3.0
    assert report['slot_seconds'] == 1.5
    assert clip.is_file()


@pytest.mark.skipif(not _ffmpeg() or not _ffprobe(), reason='ffmpeg/ffprobe not on PATH')
def test_a_clip_that_fits_is_left_alone(tmp_path) -> None:
    clip = tmp_path / 'short.wav'
    subprocess.run(
        [_ffmpeg(), '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
         '-c:a', 'pcm_s16le', str(clip)],
        check=True, capture_output=True,
    )

    report = segment_tts.fit_clip_to_slot(str(clip), 2.0, ffmpeg=_ffmpeg(), ffprobe=_ffprobe())

    assert report['tempo'] == 1.0
    assert report['fits'] is True
    assert round(report['clip_seconds']) == 1


def test_a_missing_file_is_reported_not_raised(tmp_path) -> None:
    report = segment_tts.fit_clip_to_slot(str(tmp_path / 'nope.mp3'), 2.0, ffmpeg='ffmpeg', ffprobe='ffprobe')

    assert report['fits'] is True
    assert report['clip_seconds'] is None
