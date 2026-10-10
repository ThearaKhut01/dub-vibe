from __future__ import annotations

import re
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp (HH:MM:SS.mmm)."""
    total_ms = max(0.0, seconds) * 1000
    hours = int(total_ms // 3600000)
    minutes = int(total_ms // 60000 % 60)
    secs = int(total_ms // 1000 % 60)
    milliseconds = int(total_ms % 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}'


SUBTITLE_LINE_CHARS = 42
SUBTITLE_MAX_LINES = 5


def wrap_subtitle_lines(text: str, line_chars: int = SUBTITLE_LINE_CHARS) -> list[str]:
    """Wrap text into display lines of at most line_chars (a single long word stays whole)."""
    words = re.sub(r'\s+', ' ', text or '').strip().split()
    lines: list[str] = []
    current = ''
    for word in words:
        candidate = f'{current} {word}' if current else word
        if current and len(candidate) > line_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def split_subtitle_cues(segment: dict[str, Any], max_lines: int = SUBTITLE_MAX_LINES) -> list[dict[str, Any]]:
    """Return one or more cues for a segment, each holding at most max_lines lines.

    Cues that fit are returned unchanged. Longer text is split at word boundaries,
    and the segment's time window is divided in proportion to each part's length.
    """
    text = str(segment.get('text', '')).strip()
    lines = wrap_subtitle_lines(text)
    if len(lines) <= max_lines:
        return [{**segment, 'text': '\n'.join(lines)}]

    start = float(segment.get('start', 0.0))
    end = float(segment.get('end', 0.0))
    total_chars = sum(len(line) for line in lines)
    cues: list[dict[str, Any]] = []
    cursor = start
    for offset in range(0, len(lines), max_lines):
        part = lines[offset:offset + max_lines]
        part_chars = sum(len(line) for line in part)
        part_end = end if offset + max_lines >= len(lines) else cursor + (end - start) * part_chars / total_chars
        cues.append({**segment, 'start': round(cursor, 3), 'end': round(part_end, 3), 'text': '\n'.join(part)})
        cursor = part_end
    return cues


class SmartChunkEngine:
    """Split long speech into stable, export-friendly chunks for dubbing workflows."""

    def chunk_text(self, text: str, max_chars: int = 180, min_chars: int = 60) -> list[str]:
        normalized = re.sub(r"\s+", ' ', text or '').strip()
        if not normalized:
            return []

        tokens = normalized.split()
        if len(normalized) <= max_chars:
            return [normalized]

        chunks: list[str] = []
        current = ''

        for token in tokens:
            candidate = f'{current} {token}'.strip() if current else token
            if current and len(candidate) > max_chars:
                if len(current) < min_chars and len(token) <= max_chars:
                    current = candidate
                    continue
                chunks.append(current.strip())
                current = token
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        if not chunks:
            chunks = [normalized[:max_chars]]

        return [chunk for chunk in chunks if chunk]

    def build_segments(self, text: str, chunk_duration: float = 3.0) -> list[dict[str, Any]]:
        chunks = self.chunk_text(text)
        cursor = 0.0
        segments: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            start = cursor
            end = start + max(chunk_duration, len(chunk) / 18.0)
            segments.append(
                {
                    'id': f'chunk_{index + 1}',
                    'start': round(start, 2),
                    'end': round(end, 2),
                    'speaker': f'Speaker {index % 2 + 1}',
                    'text': chunk,
                    'duration': round(end - start, 2),
                }
            )
            cursor = end + 0.25

        return segments

    def build_vtt(self, segments: list[dict[str, Any]]) -> str:
        lines = ['WEBVTT', '']
        cues = [cue for segment in segments for cue in split_subtitle_cues(segment)]
        for index, cue in enumerate(cues):
            start = format_timestamp(float(cue.get('start', 0.0)))
            end = format_timestamp(float(cue.get('end', 0.0)))
            lines.append(f'{index + 1}')
            lines.append(f'{start} --> {end}')
            lines.append(str(cue.get('text', '')).strip())
            lines.append('')
        return '\n'.join(lines).strip() + '\n'

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        return format_timestamp(seconds)
