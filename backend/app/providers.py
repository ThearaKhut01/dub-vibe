from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base providers
# ---------------------------------------------------------------------------

class SpeechToTextProvider(ABC):
    @abstractmethod
    def transcribe(self, file_path: str) -> dict:
        """Return transcript data, including timestamps if available."""


class TranslationProvider(ABC):
    @abstractmethod
    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        """Translate a source string into the target language."""


class TextToSpeechProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str, output_path: str, language: str) -> dict:
        """Generate a TTS payload for the given segment."""


class SpeakerDetectionProvider(ABC):
    @abstractmethod
    def detect(self, segments: list[dict]) -> list[dict]:
        """Attach speaker labels to transcript segments."""


# ---------------------------------------------------------------------------
# App paths: Store models & tools in app's own folder (CapCut / portable style)
# ---------------------------------------------------------------------------

def get_app_root() -> Path:
    """Return the application installation root directory."""
    if getattr(sys, 'frozen', False):
        exe_path = Path(sys.executable).resolve()
        if exe_path.parent.name.lower() in ('binaries', 'resources', 'tools', 'backend', 'dubvibe-backend-complete'):
            return exe_path.parent.parent
        return exe_path.parent
    return Path(__file__).resolve().parent.parent.parent


def models_root() -> Path:
    """Root folder for all downloaded AI models.

    Always %LOCALAPPDATA%\\DubVibe Studio\\models (resolved from the Windows
    LOCALAPPDATA environment variable, never hard-coded, never Program Files)
    so the local web UI and the packaged desktop app share one model store.
    DUBVIBE_MODELS_DIR overrides for advanced setups.
    """
    override = os.environ.get('DUBVIBE_MODELS_DIR')
    if override:
        root = Path(override).expanduser()
    else:
        local_app_data = os.environ.get('LOCALAPPDATA')
        base = Path(local_app_data) if local_app_data else Path.home() / 'AppData' / 'Local'
        root = base / 'DubVibe Studio' / 'models'
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_storage_dir(category: str) -> Path:
    """Return an app-local storage directory for models, data, or cache.

    Models always live in the shared AppData model root (see models_root) so
    both interfaces use the same store. Other categories prefer
    `<AppRoot>/<category>` and fall back to `%LOCALAPPDATA%\\DubVibe\\<category>`
    when the install folder is read-only (e.g. C:\\Program Files).
    """
    if category == 'models':
        return models_root()

    env_override = os.environ.get(f'DUBVIBE_{category.upper()}_DIR')
    if env_override:
        path = Path(env_override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    app_root = get_app_root()
    app_target = app_root / category
    try:
        app_target.mkdir(parents=True, exist_ok=True)
        test_file = app_target / '.write_test'
        test_file.touch()
        test_file.unlink()
        return app_target
    except (PermissionError, OSError):
        local_app_data = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        fallback = local_app_data / 'DubVibe' / category
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# ---------------------------------------------------------------------------
# Speech-to-text: Faster-Whisper (local, offline after first model download)
# ---------------------------------------------------------------------------

class FasterWhisperProvider(SpeechToTextProvider):
    """Transcription through the shared model manager.

    The active model (tiny .. large-v3-turbo), device, and compute type come
    from the model manager, so the local web UI and the Windows desktop app
    always transcribe with the same engine and the same model files.
    """

    def transcribe(self, file_path: str, progress_callback=None, should_cancel=None,
                   language: str | None = None, beam_size: int = 5,
                   word_timestamps: bool = False) -> dict:
        from . import model_manager

        model = model_manager.get_whisper_model()
        # Cancel checks around the two uninterruptible phases — model load and
        # the upfront decode+VAD inside transcribe() can take tens of seconds
        # on long videos, and the flag was previously only seen between
        # segments, so a cancel during that window appeared dead.
        if should_cancel is not None and should_cancel():
            return {'provider': 'Faster-Whisper', 'model': model_manager.selected_model_id(),
                    'language': None, 'segments': [], 'source_file': file_path,
                    'cancelled': True, 'status': 'ok'}
        segments, info = model.transcribe(
            file_path,
            vad_filter=True,
            beam_size=beam_size,
            language=language,
            word_timestamps=word_timestamps,
        )
        if should_cancel is not None and should_cancel():
            return {'provider': 'Faster-Whisper', 'model': model_manager.selected_model_id(),
                    'language': info.language, 'segments': [], 'source_file': file_path,
                    'cancelled': True, 'status': 'ok'}
        results = []
        cancelled = False
        for segment in segments:
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            text = segment.text.strip()
            if not text:
                continue
            entry = {
                'id': f'seg_{len(results) + 1}',
                'start': round(segment.start, 3),
                'end': round(segment.end, 3),
                'speaker': 'Speaker 1',
                'text': text,
            }
            if word_timestamps and segment.words:
                entry['words'] = [
                    {
                        'start': round(word.start, 3),
                        'end': round(word.end, 3),
                        'word': word.word,
                        'probability': round(word.probability, 2),
                    }
                    for word in segment.words
                ]
            results.append(entry)
            # Throttled: one frame per 10 segments keeps progress useful without
            # flooding the WebSocket and the UI log on long videos.
            if progress_callback is not None and len(results) % 10 == 0:
                progress_callback(len(results), float(segment.end))
        return {
            'provider': 'Faster-Whisper',
            'model': model_manager.selected_model_id(),
            'language': info.language,
            'segments': results,
            'source_file': file_path,
            'cancelled': cancelled,
            'status': 'ok',
        }


# ---------------------------------------------------------------------------
# Translation: Argos Translate (local offline) with Google Translate fallback
# ---------------------------------------------------------------------------

# Standard language name -> ISO code mapping shared by translation providers.
LANGUAGE_CODES: dict[str, str] = {
    'English': 'en',
    'French': 'fr',
    'Khmer': 'km',
    'Spanish': 'es',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Thai': 'th',
    'Vietnamese': 'vi',
    'German': 'de',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Indonesian': 'id',
}


def _resolve_lang_code(language: str) -> str:
    """Resolve a human-readable language name or code to an ISO 639-1 code."""
    return LANGUAGE_CODES.get(language, language.lower()[:2])


def _argos_available() -> bool:
    """Return True if argostranslate is installed and importable."""
    try:
        import importlib.util
        return importlib.util.find_spec('argostranslate') is not None
    except Exception:
        return False


class ArgosTranslationProvider(TranslationProvider):
    """Lightweight offline translation using Argos Translate.

    Language packages are downloaded on first use (~50-100 MB per pair) and
    stored under %LOCALAPPDATA%/DubVibe/argos.  After the initial download
    translations run entirely offline.
    """

    @staticmethod
    def _data_dir() -> Path:
        return get_storage_dir('models') / 'argos'

    @staticmethod
    def _ensure_package(source: str, target: str) -> bool:
        """Download the language package if it is not already installed.

        Returns True when the requested pair is available, False otherwise.
        """
        try:
            import argostranslate.settings
            from argostranslate import package as argos_package, translate as argos_translate

            data_dir = ArgosTranslationProvider._data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            packages_dir = data_dir / 'packages'
            packages_dir.mkdir(parents=True, exist_ok=True)

            argostranslate.settings.data_dir = data_dir
            argostranslate.settings.package_dirs = [packages_dir]
            argostranslate.settings.package_data_dir = packages_dir

            # Check if the pair is already installed.
            installed = argos_translate.get_installed_languages()
            if ArgosTranslationProvider._find_installed_translation(installed, source, target) is not None:
                return True

            # Try to download the package.
            argos_package.update_package_index()
            available = argos_package.get_available_packages()
            pkg = next(
                (p for p in available if p.from_code == source and p.to_code == target),
                None,
            )
            if pkg is None:
                # No direct package for this pair; the caller may pivot via English.
                return False
            download_path = pkg.download()
            argos_package.install_from_path(download_path)
            installed = argos_translate.get_installed_languages()
            return ArgosTranslationProvider._find_installed_translation(installed, source, target) is not None
        except Exception as error:
            logger.warning('Argos package install failed for %s->%s: %s', source, target, error)
            return False

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        from argostranslate import translate as argos_translate

        source = _resolve_lang_code(source_language)
        target = _resolve_lang_code(target_language)
        if source == target:
            return text

        # Ensure the language package is installed.
        if self._ensure_package(source, target):
            return self._translate_with_pair(text, source, target)

        # Argos has no direct package: pivot through English (e.g. km->en->fr).
        if source != 'en' and target != 'en':
            if self._ensure_package(source, 'en') and self._ensure_package('en', target):
                intermediate = self._translate_with_pair(text, source, 'en')
                return self._translate_with_pair(intermediate, 'en', target)

        raise RuntimeError(
            f'Argos Translate language pair {source}->{target} is not available. '
            'Download it first or fall back to online translation.'
        )

    @staticmethod
    def _find_installed_translation(installed: list, source: str, target: str):
        """Return an installed translation for source->target, or None.

        Note: ``Language.get_translation`` takes a single Language object and
        returns one translation or None — it must not be given the list.
        """
        source_lang = next((lang for lang in installed if lang.code == source), None)
        target_lang = next((lang for lang in installed if lang.code == target), None)
        if source_lang is None or target_lang is None:
            return None
        return source_lang.get_translation(target_lang)

    @staticmethod
    def _translate_with_pair(text: str, source: str, target: str) -> str:
        from argostranslate import translate as argos_translate

        installed = argos_translate.get_installed_languages()
        translation_obj = ArgosTranslationProvider._find_installed_translation(installed, source, target)
        if translation_obj is None:
            raise RuntimeError(f'Translation pair {source}->{target} not found in Argos Translate')

        result = translation_obj.translate(text)
        if not result or not result.strip():
            raise RuntimeError('Argos Translate returned an empty result')
        return result


class GoogleTranslateProvider(TranslationProvider):
    """Free online translation via the Google Translate API (requires internet)."""

    max_attempts = 4

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        target = _resolve_lang_code(target_language)
        source = _resolve_lang_code(source_language)
        if source == target:
            return text
        query = urlencode({'client': 'gtx', 'sl': source, 'tl': target, 'dt': 't', 'q': text})
        url = f'https://translate.googleapis.com/translate_a/single?{query}'
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                # A browser-like User-Agent avoids the aggressive throttling that
                # the endpoint applies to default urllib clients on long runs.
                request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                translated = ''.join(part[0] for part in payload[0] if part and part[0])
                if not translated:
                    raise RuntimeError('Google Translate returned an empty result')
                return translated
            except Exception as error:
                last_error = error
                if attempt < self.max_attempts - 1:
                    # Exponential backoff with jitter rides out rate limits and
                    # transient network drops instead of failing the whole run.
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0.0, 0.5))
        raise RuntimeError(
            f'Google Translate failed after {self.max_attempts} attempts: {last_error}'
        ) from last_error


class HybridTranslationProvider(TranslationProvider):
    """Translation chain with a user-selectable first choice.

    Default order: Qwen3 4B (local LLM) -> Argos (offline) -> Google (online).
    ``preference`` moves one engine to the front; the rest keep the order as
    fallbacks so a failed request never leaves a line untranslated.
    """

    def __init__(self, style_hint: str | None = None, preference: str = 'auto') -> None:
        self._qwen: TranslationProvider | None = None
        self._offline: TranslationProvider | None = None
        self._online = GoogleTranslateProvider()
        self._mode: str = 'online'
        self._preference = preference if preference in ('qwen', 'argos', 'google') else 'auto'

        try:
            from . import qwen_local

            if qwen_local.is_installed():
                self._qwen = qwen_local.QwenTranslationProvider(style_hint)
                self._mode = 'qwen'
        except Exception as error:
            logger.info('Qwen local translation unavailable: %s', error)

        if _argos_available():
            try:
                self._offline = ArgosTranslationProvider()
                self._mode = 'offline'
            except Exception:
                logger.info('Argos Translate is installed but failed to initialise; using online fallback.')

    @property
    def active_mode(self) -> str:
        return self._mode

    def _chain(self) -> list[tuple[str, TranslationProvider]]:
        """Available providers, preference first, each entry at most once."""
        order = ([self._preference] if self._preference != 'auto' else []) + ['qwen', 'argos', 'google']
        available = {'qwen': self._qwen, 'argos': self._offline, 'google': self._online}
        chain: list[tuple[str, TranslationProvider]] = []
        for key in order:
            provider = available.get(key)
            if provider is not None and all(provider is not seen for _, seen in chain):
                chain.append((key, provider))
        return chain

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        modes = {'qwen': 'qwen', 'argos': 'offline', 'google': 'online'}
        last_error: Exception | None = None
        for key, provider in self._chain():
            try:
                result = provider.translate(text, target_language, source_language)
                self._mode = modes[key]
                return result
            except Exception as error:
                logger.info('%s translation failed (%s), trying the next provider.', key, error)
                last_error = error
        if last_error is not None:
            raise last_error
        raise RuntimeError('No translation provider is available')


# ---------------------------------------------------------------------------
# TTS: pyttsx3 (local offline) with Edge TTS fallback
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _pyttsx3_available() -> bool:
    """Return True if pyttsx3 is installed and can initialise an engine.

    Cached because ``pyttsx3.init()`` creates a SAPI5/COM engine, which is
    expensive and not something to repeat on every provider status poll.
    """
    try:
        import importlib.util
        if importlib.util.find_spec('pyttsx3') is None:
            return False
        import pyttsx3
        engine = pyttsx3.init()
        engine.stop()
        return True
    except Exception:
        return False


class Pyttsx3Provider(TextToSpeechProvider):
    """Offline TTS using the Windows SAPI5 voices bundled with the OS.

    Produces WAV files (zero external downloads required).

    This is a blocking implementation (COM + ``runAndWait``); callers must not
    run it directly on an asyncio event loop.
    """

    def synthesize(self, text: str, voice: str, output_path: str, language: str) -> dict:
        # SAPI5 requires COM on the calling thread, and synthesis may run inside
        # a worker thread (asyncio.to_thread) where COM is not initialised yet.
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            comtypes = None
        try:
            return self._synthesize(text, voice, output_path, language)
        finally:
            if comtypes is not None:
                try:
                    comtypes.CoUninitialize()
                except Exception:
                    pass

    def _synthesize(self, text: str, voice: str, output_path: str, language: str) -> dict:
        import pyttsx3

        # Map DubVibe voice labels to pyttsx3 voice selection logic.
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        # Try to find a matching voice by language / gender heuristic.
        lang_code = _resolve_lang_code(language)
        is_female = 'female' in voice.lower()

        selected_voice_id = None
        # First pass: match language and gender.
        for v in voices:
            v_name = (v.name or '').lower()
            v_langs = [lid.lower() for lid in (v.languages or [])] if v.languages else []
            lang_match = any(lang_code in lid for lid in v_langs) or lang_code in v_name
            gender_match = ('female' in v_name or 'zira' in v_name or 'hazel' in v_name) == is_female
            if lang_match and gender_match:
                selected_voice_id = v.id
                break
        # Second pass: match language only.
        if selected_voice_id is None:
            for v in voices:
                v_name = (v.name or '').lower()
                v_langs = [lid.lower() for lid in (v.languages or [])] if v.languages else []
                if any(lang_code in lid for lid in v_langs) or lang_code in v_name:
                    selected_voice_id = v.id
                    break
        # Fallback: use the first available voice.
        if selected_voice_id is None and voices:
            selected_voice_id = voices[0].id

        if selected_voice_id:
            engine.setProperty('voice', selected_voice_id)
        engine.setProperty('rate', 160)

        # pyttsx3 saves to WAV; convert the output path to .wav.
        wav_path = str(Path(output_path).with_suffix('.wav'))
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        engine.stop()

        # Convert WAV to MP3 with ffmpeg if available, otherwise keep WAV.
        final_path = output_path
        try:
            import shutil
            ffmpeg = shutil.which('ffmpeg')
            if ffmpeg and Path(wav_path).is_file():
                mp3_path = str(Path(output_path).with_suffix('.mp3'))
                subprocess.run(
                    [ffmpeg, '-y', '-i', wav_path, '-codec:a', 'libmp3lame', '-q:a', '4', mp3_path],
                    check=True,
                    capture_output=True,
                )
                Path(wav_path).unlink(missing_ok=True)
                final_path = mp3_path
            else:
                final_path = wav_path
        except Exception:
            final_path = wav_path

        voice_name = selected_voice_id.split('\\')[-1] if selected_voice_id and '\\' in selected_voice_id else (selected_voice_id or voice)
        return {
            'provider': 'pyttsx3 (Offline)',
            'voice': voice_name,
            'text': text,
            'output': final_path,
            'status': 'ok',
        }


# (language, UI label) -> Edge neural voice id. The labels are what the
# Characters UI and default voice options use; the ids are what edge-tts
# speaks with. Used by EdgeTTSProvider and by gender-based voice assignment.
EDGE_LANGUAGE_VOICES: dict[tuple[str, str], str] = {
    ('Khmer', 'Male Voice A'): 'km-KH-PisethNeural',
    ('Khmer', 'Female Voice B'): 'km-KH-SreymomNeural',
    ('French', 'Male Voice A'): 'fr-FR-HenriNeural',
    ('French', 'Female Voice B'): 'fr-FR-DeniseNeural',
    ('English', 'Male Voice A'): 'en-US-GuyNeural',
    ('English', 'Female Voice B'): 'en-US-JennyNeural',
    ('English', 'Narrator Voice'): 'en-US-ChristopherNeural',
    ('Spanish', 'Male Voice A'): 'es-ES-AlvaroNeural',
    ('Spanish', 'Female Voice B'): 'es-ES-ElviraNeural',
    ('Chinese', 'Male Voice A'): 'zh-CN-YunxiNeural',
    ('Chinese', 'Female Voice B'): 'zh-CN-XiaoxiaoNeural',
    ('Japanese', 'Male Voice A'): 'ja-JP-KeitaNeural',
    ('Japanese', 'Female Voice B'): 'ja-JP-NanamiNeural',
    ('Korean', 'Male Voice A'): 'ko-KR-InJoonNeural',
    ('Korean', 'Female Voice B'): 'ko-KR-SunHiNeural',
    ('Thai', 'Male Voice A'): 'th-TH-NiwatNeural',
    ('Thai', 'Female Voice B'): 'th-TH-PremwadeeNeural',
    ('Vietnamese', 'Male Voice A'): 'vi-VN-NamMinhNeural',
    ('Vietnamese', 'Female Voice B'): 'vi-VN-HoaiMyNeural',
    ('German', 'Male Voice A'): 'de-DE-ConradNeural',
    ('German', 'Female Voice B'): 'de-DE-KatjaNeural',
    ('Portuguese', 'Male Voice A'): 'pt-BR-AntonioNeural',
    ('Portuguese', 'Female Voice B'): 'pt-BR-FranciscaNeural',
    ('Russian', 'Male Voice A'): 'ru-RU-DmitryNeural',
    ('Russian', 'Female Voice B'): 'ru-RU-SvetlanaNeural',
    ('Arabic', 'Male Voice A'): 'ar-SA-HamedNeural',
    ('Arabic', 'Female Voice B'): 'ar-SA-ZariyahNeural',
    ('Hindi', 'Male Voice A'): 'hi-IN-MadhurNeural',
    ('Hindi', 'Female Voice B'): 'hi-IN-SwaraNeural',
    ('Indonesian', 'Male Voice A'): 'id-ID-ArdiNeural',
    ('Indonesian', 'Female Voice B'): 'id-ID-GadisNeural',
}

_NEURAL_ID_RE = re.compile(r'^[a-z]{2}-[A-Z]{2}-\w+Neural$')


def edge_gender_voice(language: str, gender: str) -> str | None:
    """Edge neural voice id for a detected speaker gender in the target language."""
    label = 'Female Voice B' if gender == 'Female' else 'Male Voice A'
    return EDGE_LANGUAGE_VOICES.get((language, label))


class EdgeTTSProvider(TextToSpeechProvider):
    async def synthesize(self, text: str, voice: str, output_path: str, language: str) -> dict:
        import edge_tts

        if _NEURAL_ID_RE.match(voice or ''):
            # An explicit neural voice id (gender auto-assign or a manual
            # character pick) is used as-is — the caller chose it on purpose.
            selected_voice = voice
        else:
            selected_voice = EDGE_LANGUAGE_VOICES.get((language, voice))
            if selected_voice is None and voice == 'Narrator Voice':
                # Narrator falls back to the language's male voice.
                selected_voice = EDGE_LANGUAGE_VOICES.get((language, 'Male Voice A'))
            if selected_voice is None:
                # Never synthesize with a mismatched language (e.g. Khmer text with
                # an English voice): fall back to any voice for the target language.
                selected_voice = next(
                    (neural for (lang, _), neural in EDGE_LANGUAGE_VOICES.items() if lang == language),
                    'en-US-DavisNeural',
                )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                output.unlink(missing_ok=True)
                await edge_tts.Communicate(text, selected_voice).save(str(output))
                break
            except Exception as error:
                last_error = error
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f'Edge TTS failed after 3 attempts: {last_error}') from last_error
        return {
            'provider': 'Edge TTS',
            'voice': selected_voice,
            'text': text,
            'output': output_path,
            'status': 'ok',
        }


class HybridTTSProvider(TextToSpeechProvider):
    """Try Edge TTS (online, higher quality) first, fall back to pyttsx3 (offline)."""

    def __init__(self) -> None:
        self._online = EdgeTTSProvider()
        self._offline: TextToSpeechProvider | None = None
        self._mode: str = 'online'

        if _pyttsx3_available():
            self._offline = Pyttsx3Provider()

    @property
    def active_mode(self) -> str:
        return self._mode

    async def synthesize(self, text: str, voice: str, output_path: str, language: str) -> dict:
        # Try Edge TTS (online) first for better quality.
        online_error: Exception | None = None
        try:
            result = await self._online.synthesize(text, voice, output_path, language)
            self._mode = 'online'
            return result
        except Exception as error:
            online_error = error
            logger.info('Edge TTS failed (%s), trying offline fallback.', error)

        # Fall back to pyttsx3 (offline). It is fully blocking, so keep it off
        # the event loop inside a worker thread.
        if self._offline is not None:
            try:
                result = await asyncio.to_thread(
                    self._offline.synthesize, text, voice, output_path, language
                )
                self._mode = 'offline'
                return result
            except Exception as offline_error:
                logger.warning('Offline TTS also failed: %s', offline_error)
                raise RuntimeError(
                    f'Both online (Edge TTS) and offline (pyttsx3) TTS failed. '
                    f'Online: {online_error}  Offline: {offline_error}'
                ) from offline_error

        raise RuntimeError(f'Edge TTS failed and no offline TTS is available: {online_error}') from online_error


# ---------------------------------------------------------------------------
# Speaker detection (stub)
# ---------------------------------------------------------------------------

class SpeakerDetector(SpeakerDetectionProvider):
    def detect(self, segments: list[dict]) -> list[dict]:
        for index, segment in enumerate(segments):
            segment['speaker'] = segment.get('speaker') or f'Speaker {index + 1}'
        return segments


# ---------------------------------------------------------------------------
# Provider availability helpers (used by the /api/providers endpoint)
# ---------------------------------------------------------------------------

def get_provider_status() -> list[dict]:
    """Return a summary of all available providers and their modes."""
    import importlib.util

    whisper_ok = importlib.util.find_spec('faster_whisper') is not None
    argos_ok = _argos_available()
    pyttsx3_ok = _pyttsx3_available()
    edge_ok = importlib.util.find_spec('edge_tts') is not None
    try:
        from . import qwen_local

        qwen_ok = qwen_local.is_installed()
    except Exception:
        qwen_ok = False
    try:
        from . import mms_tts_local

        mms_ok = mms_tts_local.is_installed('mms_khm')
        mms_base_ok = mms_tts_local.is_installed('mms_khm_base')
    except Exception:
        mms_ok = False
        mms_base_ok = False

    return [
        {
            'id': 'whisper',
            'name': 'Faster-Whisper',
            'mode': 'LOCAL AI',
            'connected': whisper_ok,
            'primary': True,
            'fallback': False,
        },
        {
            'id': 'qwen',
            'name': 'Qwen3 4B (translation)',
            'mode': 'LOCAL AI',
            'connected': qwen_ok,
            'primary': qwen_ok,
            'fallback': False,
        },
        {
            'id': 'argos_translate',
            'name': 'Argos Translate',
            'mode': 'LOCAL AI',
            'connected': argos_ok,
            'primary': argos_ok,
            'fallback': False,
        },
        {
            'id': 'google_translate',
            'name': 'Google Translate',
            'mode': 'FREE ONLINE',
            'connected': True,
            'primary': not argos_ok,
            'fallback': argos_ok,
        },
        {
            'id': 'speaker-diarization',
            'name': 'Speaker Detection',
            'mode': 'LOCAL AI',
            'connected': importlib.util.find_spec('resemblyzer') is not None,
            'primary': False,
            'fallback': False,
        },
        {
            'id': 'pyttsx3',
            'name': 'System TTS (SAPI5)',
            'mode': 'LOCAL AI',
            'connected': pyttsx3_ok,
            'primary': False,
            'fallback': pyttsx3_ok,
        },
        {
            'id': 'edge_tts',
            'name': 'Edge TTS',
            'mode': 'FREE ONLINE',
            'connected': edge_ok,
            'primary': True,
            'fallback': False,
        },
        {
            'id': 'mms_khm',
            'name': 'Khmer MMS-VITS (offline, non-commercial)',
            'mode': 'LOCAL AI',
            'connected': mms_ok,
            'primary': False,
            'fallback': mms_ok,
        },
        {
            'id': 'mms_khm_base',
            'name': 'Khmer MMS-VITS base (offline, non-commercial)',
            'mode': 'LOCAL AI',
            'connected': mms_base_ok,
            'primary': False,
            'fallback': mms_base_ok,
        },
    ]
