"""TTS service: provider selection, capability gating, caching, fallback.

One service for every interface. Resolution order:
  1. Character profile override (provider + voice live on the character).
  2. Requested provider preference.
  3. Quality-ordered fallback: gemini -> openai -> edge -> voxcpm2 -> mms_khm -> mms_khm_base -> sapi5.
Language capability is checked per provider (Khmer is NOT claimed where the
provider does not officially support it — it falls back to Edge TTS, which
does support Khmer). voxcpm2 is an offline model limited to the languages it
was verified for (English); it is used when Edge TTS is unavailable. mms_khm and mms_khm_base are offline
Khmer-only models used when Edge TTS is unavailable. Paid API results are cached under
%LOCALAPPDATA%-style model root: cache/tts/<provider>/ so identical requests
(text + voice + model + acting style + speed) never cost twice.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from . import acting, api_keys
from .online_providers import (
    GEMINI_DEFAULT_MODEL,
    GEMINI_VOICES,
    OPENAI_DEFAULT_MODEL,
    OPENAI_VOICES,
    gemini_capabilities,
    gemini_synthesize,
    openai_capabilities,
    openai_synthesize,
)

QUALITY_ORDER = ['gemini', 'openai', 'edge', 'voxcpm2', 'mms_khm', 'mms_khm_base', 'sapi5']
OFFLINE_KHMER_PROVIDERS = ('mms_khm', 'mms_khm_base')
OFFLINE_MULTILINGUAL_PROVIDERS = ('voxcpm2',)


def tts_cache_dir(provider: str) -> Path:
    from .providers import models_root

    directory = models_root().parent / 'cache' / 'tts' / provider
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_key(provider: str, model: str, voice: str, text: str, style: str | None, speed: float) -> str:
    material = '|'.join((provider, model, voice, text, style or '', f'{speed:.2f}'))
    return hashlib.sha256(material.encode('utf-8')).hexdigest()


def _find_cached(provider: str, key: str) -> Path | None:
    matches = list(tts_cache_dir(provider).glob(f'{key}.*'))
    return matches[0] if matches else None


def _store_cache(provider: str, key: str, path: Path) -> None:
    shutil.copy2(path, tts_cache_dir(provider) / f'{key}{path.suffix.lower()}')


def language_supported(provider: str, lang_code: str) -> bool:
    lang = (lang_code or 'en').lower()
    if provider == 'gemini':
        return lang in gemini_capabilities(GEMINI_DEFAULT_MODEL)['languages']
    if provider == 'openai':
        return lang in openai_capabilities(OPENAI_DEFAULT_MODEL)['languages']
    if provider in OFFLINE_KHMER_PROVIDERS:
        return lang == 'km'
    if provider in OFFLINE_MULTILINGUAL_PROVIDERS:
        from . import voxcpm_local

        return voxcpm_local.supports_language(lang)
    return True  # Edge TTS (incl. Khmer) and SAPI5 carry their own voices


def _default_voice(provider: str, requested: str | None) -> str:
    if provider == 'gemini':
        return requested if requested in GEMINI_VOICES else 'Kore'
    if provider == 'openai':
        return requested if requested in OPENAI_VOICES else 'alloy'
    if provider in OFFLINE_KHMER_PROVIDERS:
        return 'khm-single'
    if provider in OFFLINE_MULTILINGUAL_PROVIDERS:
        return 'auto'
    return requested or 'auto'


def candidate_providers(pref: str, lang_code: str, profile: dict | None) -> list[str]:
    """Ordered provider candidates honoring character override + capability."""
    override = (profile or {}).get('provider')
    head: list[str] = []
    if override and override in QUALITY_ORDER:
        head.append(override)
    elif pref in QUALITY_ORDER and pref != 'auto':
        head.append(pref)
    ordered = head + [p for p in QUALITY_ORDER if p not in head]
    # Providers that cannot speak the target language are skipped entirely.
    return [provider for provider in ordered if language_supported(provider, lang_code)] or ['edge', 'sapi5']


def synthesize_line(text: str, out_path: str, *, provider_pref: str, language_name: str,
                    lang_code: str, voice_pref: str | None, model_pref: str | None,
                    speed: float, emotion: str, intensity: int, character_profile: dict | None,
                    analysis: dict) -> dict:
    """Synthesize one line with acting directions, cache, and fallback.

    Returns {provider, voice, model, path, cached, attempts}.
    """
    profile = character_profile or {}
    candidates = candidate_providers(provider_pref, lang_code, profile)
    attempts: list[str] = []

    for provider in candidates:
        model = model_pref or (GEMINI_DEFAULT_MODEL if provider == 'gemini' else OPENAI_DEFAULT_MODEL) or model_pref
        voice = _default_voice(provider, profile.get('voice') or voice_pref)
        style = acting.style_instruction(provider, analysis)
        key = cache_key(provider, model, voice, text, style, speed)

        cached = _find_cached(provider, key)
        if cached:
            shutil.copy2(cached, out_path)
            return {'provider': provider, 'voice': voice, 'model': model, 'path': out_path, 'cached': True, 'attempts': attempts}

        try:
            if provider == 'gemini':
                key = api_keys.get_key('gemini')
                if not key:
                    attempts.append('gemini: no API key')
                    continue
                gemini_synthesize(key, text, out_path, model=model, voice=voice, style_instruction=style)
            elif provider == 'openai':
                key = api_keys.get_key('openai')
                if not key:
                    attempts.append('openai: no API key')
                    continue
                openai_synthesize(key, text, out_path, model=model, voice=voice,
                                  instructions=style, speed=speed)
            elif provider == 'edge':
                import asyncio
                from .providers import EdgeTTSProvider

                asyncio.run(EdgeTTSProvider().synthesize(text, voice, out_path, language_name))
            elif provider in OFFLINE_MULTILINGUAL_PROVIDERS:
                from .voxcpm_local import VoxCPMProvider

                VoxCPMProvider().synthesize(text, voice, out_path, language_name, speed=speed)
            elif provider in OFFLINE_KHMER_PROVIDERS:
                from .mms_tts_local import MmsKhmerProvider

                MmsKhmerProvider(provider).synthesize(text, voice, out_path, language_name, speed=speed)
            elif provider == 'sapi5':
                from .providers import Pyttsx3Provider

                Pyttsx3Provider().synthesize(text, voice, out_path, language_name)
            else:
                attempts.append(f'{provider}: unknown provider')
                continue

            strip_audio_silence(out_path)
            _store_cache(provider, key, Path(out_path)) if provider in ('gemini', 'openai') else None
            return {
                'provider': provider,
                'voice': voice,
                'model': model,
                'path': out_path,
                'cached': False,
                'attempts': attempts,
            }
        except Exception as error:  # noqa: BLE001 - try the next provider
            attempts.append(f'{provider}: {error}')
            continue

    raise RuntimeError('All TTS providers failed: ' + '; '.join(attempts))


def strip_audio_silence(path: str) -> None:
    """Trim dead silence padding from synthesized TTS audio while preserving ~40ms natural breath margin."""
    try:
        import subprocess
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            try:
                from .main import find_media_tool
                ffmpeg = find_media_tool('ffmpeg')
            except Exception:
                pass
        if not ffmpeg or not Path(path).is_file():
            return
        temp_out = Path(path).with_name(f'clean_{Path(path).name}')
        cmd = [
            ffmpeg, '-y', '-i', str(path),
            '-af', 'silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse,silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse',
            '-c:a', 'libmp3lame', '-q:a', '4', str(temp_out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and temp_out.is_file() and temp_out.stat().st_size > 0:
            temp_out.replace(path)
        else:
            temp_out.unlink(missing_ok=True)
    except Exception:
        pass


def preview_line(text: str, *, provider: str, language_name: str, lang_code: str,
                 voice: str | None, model: str | None, emotion: str, intensity: int,
                 character_profile: dict | None, use_openai_acting: bool, speed: float = 1.0) -> str:
    """Generate one preview line and return it as a base64 data URL for <audio>."""
    import base64

    if provider in ('gemini', 'openai') and not api_keys.has_key(provider):
        raise RuntimeError(f'{provider.capitalize()} API key is not configured (set it in API Settings).')

    character = character_profile or {}
    analysis = acting.analyze(
        text,
        character.get('previous'),
        character.get('next'),
        character.get('name'),
        language_name,
        use_openai=use_openai_acting and provider not in ('sapi5', *OFFLINE_KHMER_PROVIDERS, *OFFLINE_MULTILINGUAL_PROVIDERS),
    )
    if emotion and emotion != 'Auto':
        analysis = {**analysis, 'emotion': emotion, 'intensity': intensity}

    key = cache_key(provider, model or '', voice or '', text,
                    acting.style_instruction(provider, analysis), speed)
    cached = _find_cached(provider, key)
    if not cached:
        from .providers import models_root

        temp_dir = models_root().parent / 'cache' / 'tts' / 'preview'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f'{key}.preview'
        synthesize_line(
            text, str(temp_path), provider_pref=provider, language_name=language_name,
            lang_code=lang_code, voice_pref=voice, model_pref=model, speed=speed,
            emotion=emotion, intensity=intensity, character_profile=character_profile,
            analysis=analysis,
        )
        cached = temp_path
    audio = Path(cached).read_bytes()
    mime = 'audio/mpeg' if Path(cached).suffix.lower() in ('.mp3', '.preview') else 'audio/wav'
    return f'data:{mime};base64,{base64.b64encode(audio).decode("ascii")}'
