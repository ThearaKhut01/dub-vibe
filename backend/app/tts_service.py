"""TTS service: provider selection, capability gating, caching, fallback.

One service for every interface. Resolution order:
  1. Character profile override (provider + voice live on the character).
  2. Requested provider preference.
  3. Quality-ordered fallback: gemini -> openai -> edge -> voxcpm2 -> sapi5.
Language capability is checked per provider (Khmer is NOT claimed where the
provider does not officially support it — it falls back to Edge TTS, which
does support Khmer). voxcpm2 is an offline model limited to the languages it
was verified for (English); it is used when Edge TTS is unavailable.
Khmer-only models used when Edge TTS is unavailable. Paid API results are cached under
%LOCALAPPDATA%-style model root: cache/tts/<provider>/ so identical requests
(text + voice + model + acting style + speed) never cost twice.
"""
from __future__ import annotations

import subprocess
import hashlib
import shutil
import time
from pathlib import Path

from . import acting, api_keys
from .providers import SpeechUnavailableError
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

QUALITY_ORDER = ['gemini', 'openai', 'edge', 'voxcpm2', 'sapi5']
OFFLINE_MULTILINGUAL_PROVIDERS = ('voxcpm2',)
# After an engine reports "no audio" for a language, skip it for that language for
# this long. Without this a 300-line project retries a dead voice 3x per line.
_UNAVAILABLE_SECONDS = 600.0
_unavailable_until: dict[tuple[str, str], float] = {}


def reset_unavailable() -> None:
    """Clear the 'this engine has no audio for this language' short-circuit.

    Called when the user explicitly asks for audio again (a full Dub run or a
    single-line regenerate), so a recovered engine is retried immediately.
    """
    _unavailable_until.clear()


def tts_cache_dir(provider: str) -> Path:
    from .providers import models_root

    directory = models_root().parent / 'cache' / 'tts' / provider
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_key(provider: str, model: str, voice: str, text: str, style: str | None, speed: float) -> str:
    material = '|'.join((provider, model or '', voice, text, style or '', f'{speed:.2f}'))
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
    if provider in OFFLINE_MULTILINGUAL_PROVIDERS:
        from . import voxcpm_local

        return voxcpm_local.supports_language(lang)
    return True  # Edge TTS (incl. Khmer) and SAPI5 carry their own voices


def _default_voice(provider: str, requested: str | None) -> str:
    if provider == 'gemini':
        return requested if requested in GEMINI_VOICES else 'Kore'
    if provider == 'openai':
        return requested if requested in OPENAI_VOICES else 'alloy'
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


def _model_for(provider: str, model_pref: str | None) -> str | None:
    """The model a provider should use. Only Gemini and OpenAI take a model name, so a model
    chosen for one engine is never sent to another (for example a Gemini model to Edge)."""
    if provider == 'gemini':
        return model_pref or GEMINI_DEFAULT_MODEL
    if provider == 'openai':
        return model_pref or OPENAI_DEFAULT_MODEL
    return None

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
    unavailable: list[str] = []

    for provider in candidates:
        # Skip an engine already known to have no audio for this language, so a
        # long project does not retry a broken voice 3x on every single line.
        blocked_until = _unavailable_until.get((provider, lang_code), 0.0)
        if time.monotonic() < blocked_until:
            attempts.append(f'{provider}: skipped (no audio for {language_name} just now)')
            unavailable.append(provider)
            continue

        model = _model_for(provider, model_pref)
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
            elif provider == 'sapi5':
                from .providers import Pyttsx3Provider

                Pyttsx3Provider().synthesize(text, voice, out_path, language_name)
            else:
                attempts.append(f'{provider}: unknown provider')
                continue

            # A provider can "succeed" yet produce no audible audio — SAPI5 given
            # Khmer text on an English-only Windows voice writes a header-only MP3.
            # Accepting that silently is what made exports come out with no voice.
            if not _has_audio(Path(out_path)):
                attempts.append(f'{provider}: produced no audible audio')
                Path(out_path).unlink(missing_ok=True)
                continue

            strip_audio_silence(out_path)
            _unavailable_until.pop((provider, lang_code), None)
            _store_cache(provider, key, Path(out_path)) if provider in ('gemini', 'openai') else None
            return {
                'provider': provider,
                'voice': voice,
                'model': model,
                'path': out_path,
                'cached': False,
                'attempts': attempts,
            }
        except SpeechUnavailableError as error:
            attempts.append(f'{provider}: {error}')
            unavailable.append(provider)
            _unavailable_until[(provider, lang_code)] = time.monotonic() + _UNAVAILABLE_SECONDS
            continue
        except Exception as error:  # noqa: BLE001 - try the next provider
            attempts.append(f'{provider}: {error}')
            continue

    if unavailable and len(unavailable) == len(candidates):
        # Every engine that could run had no audio for this language: say so
        # plainly instead of burying it in the provider log.
        raise SpeechUnavailableError(
            f'No voice engine can produce {language_name} audio right now. '
            + ' '.join(attempts)
        )
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
        # A fully silent clip trims to a header-only MP3 that is not zero bytes. Accept the
        # trim only when ffprobe reads audio in it; otherwise keep the untrimmed clip.
        if res.returncode == 0 and temp_out.is_file() and temp_out.stat().st_size > 0 and _has_audio(path=temp_out):
            temp_out.replace(path)
        else:
            temp_out.unlink(missing_ok=True)
    except Exception:
        pass


def _has_audio(path: Path) -> bool:
    """True when ffprobe reads a positive duration from the file."""
    try:
        from .main import find_media_tool
        ffprobe = find_media_tool('ffprobe')
    except Exception:
        ffprobe = None
    if not ffprobe:
        return True  # cannot verify without ffprobe; keep the existing behaviour
    try:
        probe = subprocess.run(
            [ffprobe, '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(probe.stdout.strip() or '0') > 0.0
    except (subprocess.CalledProcessError, ValueError, OSError):
        return False


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
                    use_openai=use_openai_acting and provider not in ('sapi5', *OFFLINE_MULTILINGUAL_PROVIDERS),
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
