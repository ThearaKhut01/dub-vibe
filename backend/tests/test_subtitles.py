"""Subtitle cues: at most five lines each, with timing kept continuous."""
from __future__ import annotations

from backend.app.chunking import (
    SUBTITLE_MAX_LINES,
    SmartChunkEngine,
    split_subtitle_cues,
    wrap_subtitle_lines,
)


def _words(count: int) -> str:
    return ' '.join(f'word{index:03d}' for index in range(count))


def test_short_text_stays_one_cue_with_original_timing():
    segment = {'start': 1.0, 'end': 3.0, 'text': 'Hello there, how are you?'}
    cues = split_subtitle_cues(segment)
    assert len(cues) == 1
    assert cues[0]['start'] == 1.0 and cues[0]['end'] == 3.0


def test_long_text_splits_into_cues_of_at_most_five_lines():
    cues = split_subtitle_cues({'start': 10.0, 'end': 40.0, 'text': _words(120)})
    assert len(cues) > 1
    for cue in cues:
        assert len(cue['text'].split('\n')) <= SUBTITLE_MAX_LINES


def test_split_timing_is_continuous_and_covers_the_segment():
    cues = split_subtitle_cues({'start': 10.0, 'end': 40.0, 'text': _words(120)})
    assert cues[0]['start'] == 10.0
    assert abs(cues[-1]['end'] - 40.0) < 1e-6
    for previous, following in zip(cues, cues[1:]):
        assert abs(previous['end'] - following['start']) < 1e-3


def test_no_words_lost_or_reordered():
    text = _words(120)
    cues = split_subtitle_cues({'start': 0.0, 'end': 30.0, 'text': text})
    rebuilt = ' '.join(line for cue in cues for line in cue['text'].split('\n')).split()
    assert rebuilt == text.split()


def test_wrapped_lines_respect_the_width():
    for line in wrap_subtitle_lines(_words(200)):
        assert len(line) <= 42 or ' ' not in line


def test_vtt_output_uses_the_split():
    vtt = SmartChunkEngine().build_vtt([{'start': 0.0, 'end': 30.0, 'text': _words(120)}])
    cue_count = vtt.count(' --> ')
    assert cue_count > 1
    assert vtt.startswith('WEBVTT')
