"""The TTS response must report the engine that really produced the audio.

It used to be hard-coded to 'Edge TTS', so a run that fell back to another engine still
claimed Edge was used.
"""
from __future__ import annotations

from backend.app import main


def test_single_engine_is_reported_by_name():
    clips = [{'voice': 'voxcpm2:auto'}, {'voice': 'voxcpm2:auto'}]
    assert main._engines_used(clips) == 'voxcpm2'


def test_mixed_engines_are_listed():
    clips = [{'voice': 'edge:Male Voice A'}, {'voice': 'voxcpm2:auto'}]
    assert main._engines_used(clips) == 'mixed: edge, voxcpm2'


def test_no_clips_reports_none_not_edge():
    assert main._engines_used([]) == 'none'


def test_fallback_is_never_reported_as_edge():
    clips = [{'voice': 'sapi5:Female Voice B'}]
    assert main._engines_used(clips) != 'Edge TTS'
    assert main._engines_used(clips) == 'sapi5'
