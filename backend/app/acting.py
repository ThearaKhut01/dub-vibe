"""AI acting: turn scene context into concrete acting directions for TTS.

Two analysis backends:
- OpenAI (optional, when a key + toggle are set): a dubbing-director prompt
  that returns structured JSON (emotion, intensity, style, pacing, pauses).
- Local heuristic fallback: punctuation/caps rules — always available offline.

The output is converted into provider-specific TTS instructions:
- Gemini TTS: a natural-language style sentence spoken before the line.
- OpenAI gpt-4o-mini-tts: the `instructions` field.
- Edge/SAPI5: no real acting control — the instruction is intentionally None
  (the capability table keeps this honest instead of faking pitch tweaks).
"""
from __future__ import annotations

import json

from . import api_keys
from .online_providers import EMOTIONS


def heuristic_analysis(text: str, previous: str | None = None, following: str | None = None,
                       character: str | None = None) -> dict:
    """Offline, rule-based scene reading. Not an actor — but never wrong about
    the obvious cases (shouting, trailing silence, excitement)."""
    clean = (text or '').strip()
    emotion, intensity, style, pacing = 'Neutral', 60, 'Natural', 'Natural'
    pause_before = False
    if clean.endswith('!'):
        emotion = 'Excited' if len(clean) <= 40 else 'Angry'
        intensity = 85
        style = 'Energetic'
        pacing = 'Fast'
    if clean.isupper() and len(clean) > 3:
        emotion = 'Shout'
        intensity = 95
        style = 'Forceful'
        pacing = 'Fast'
    if clean.endswith(('...', '…')) or '...' in clean:
        emotion = 'Sad'
        intensity = 70
        style = 'Weak, trailing off'
        pacing = 'Slow'
        pause_before = True
    if clean.endswith('?'):
        emotion = 'Nervous' if len(clean) <= 30 else 'Neutral'
        intensity = 65
        style = 'Questioning'
    if previous and previous.rstrip().endswith(('?',)):
        pause_before = True
    return {
        'emotion': emotion,
        'intensity': intensity,
        'style': style,
        'pacing': pacing,
        'pause_before': pause_before,
        'analyzer': 'heuristic',
    }


def openai_analysis(text: str, previous: str | None, following: str | None,
                    character: str | None, target_language: str) -> dict:
    """Dubbing-director analysis via the OpenAI API. Raises on failure so the
    caller can fall back to the heuristic without breaking the dub."""
    from .online_providers import openai_chat

    key = api_keys.get_key('openai')
    if not key:
        raise RuntimeError('OpenAI API key is not configured')
    context_lines = [
        f'Character: {character or "Unknown"}',
        f'Previous line: {previous or "(none)"}',
        f'Current line: {text}',
        f'Next line: {following or "(none)"}',
        f'Dub target language: {target_language}',
    ]
    content = openai_chat(
        key,
        [
            {'role': 'system', 'content': (
                'You are a dubbing acting director. Read the scene and decide how the '
                'current line should be performed. Respond ONLY with JSON: '
                '{"emotion": one of ' + json.dumps(EMOTIONS[1:]) + ', "intensity": 0-100, '
                '"style": short speaking-style description, "pacing": "Slow"|"Natural"|"Fast", '
                '"pause_before": true|false}. Choose acting that sounds like a real actor.'
            )},
            {'role': 'user', 'content': '\n'.join(context_lines)},
        ],
        json_mode=True,
    )
    analysis = json.loads(content)
    if analysis.get('emotion') not in EMOTIONS:
        analysis['emotion'] = 'Neutral'
    analysis['intensity'] = max(0, min(100, int(analysis.get('intensity', 60))))
    analysis['analyzer'] = 'openai'
    return analysis


def analyze(text: str, previous: str | None, following: str | None, character: str | None,
            target_language: str, use_openai: bool) -> dict:
    if use_openai:
        try:
            return openai_analysis(text, previous, following, character, target_language)
        except Exception:
            pass  # network/quota errors fall back to the offline heuristic
    return heuristic_analysis(text, previous, following, character)


def style_instruction(provider: str, analysis: dict) -> str | None:
    """Provider-specific acting direction. None = provider has no acting control."""
    emotion = str(analysis.get('emotion') or 'Neutral')
    intensity = int(analysis.get('intensity') or 60)
    style = str(analysis.get('style') or 'Natural')
    pacing = str(analysis.get('pacing') or 'Natural')

    if provider == 'gemini':
        depth = 'deeply' if intensity >= 70 else 'subtly' if intensity >= 35 else 'barely'
        direction = f'Say this {depth} {emotion.lower()}'
        if style and style.lower() != 'natural':
            direction += f', with a {style.lower()} delivery'
        if pacing.lower() != 'natural':
            direction += f', speaking at a {pacing.lower()} pace'
        if analysis.get('pause_before'):
            direction += ', starting after a short pause'
        return direction + '.'
    if provider == 'openai':
        depth = 'deeply' if intensity >= 70 else 'subtly' if intensity >= 35 else 'lightly'
        parts = [f'Perform this line like a voice actor: {emotion} tone, {depth} felt']
        if style and style.lower() != 'natural':
            parts.append(f'{style} delivery')
        if pacing.lower() != 'natural':
            parts.append(f'{pacing} pacing')
        if analysis.get('pause_before'):
            parts.append('leave a brief pause before speaking')
        return ', '.join(parts) + '.'
    # Edge TTS / SAPI5 have no real acting control — do not fake it with pitch.
    return None
