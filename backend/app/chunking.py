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
        for index, segment in enumerate(segments):
            start = format_timestamp(float(segment.get('start', 0.0)))
            end = format_timestamp(float(segment.get('end', 0.0)))
            lines.append(f'{index + 1}')
            lines.append(f'{start} --> {end}')
            lines.append(str(segment.get('text', '')).strip())
            lines.append('')
        return '\n'.join(lines).strip() + '\n'

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        return format_timestamp(seconds)
