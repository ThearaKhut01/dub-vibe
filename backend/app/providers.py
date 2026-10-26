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
import wave
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base providers
# ---------------------------------------------------------------------------

class SpeechUnavailableError(RuntimeError):
    """The voice engine answered but produced no audio for this voice/language.

    Distinct from a generic failure: it usually means the engine is having a
    server-side problem with that language (Edge TTS returning no audio) or has no
    installed voice for it (SAPI5). Callers use it to stop retrying the same engine
    for every line and to explain the problem to the user.
    """


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


# ---------------------------------------------------------------------------
# Target-script validation for translated lines
# ---------------------------------------------------------------------------

# Unicode script ranges for the non-Latin writing systems the app can target,
# keyed by ISO 639-1 code. Latin-script targets (fr, es, de, vi...) have no
# script guarantee to verify, so they are absent on purpose.
NON_LATIN_SCRIPT_RANGES: dict[str, str] = {
    'km': '\u1780-\u17FF',               # Khmer: consonants, vowels, signs, digits
    'th': '\u0E00-\u0E7F',               # Thai (also lo partly)
    'zh': '\u4E00-\u9FFF',               # CJK unified ideographs
    'ja': '\u3040-\u30FF\u4E00-\u9FFF',  # kana + kanji
    'ko': '\uAC00-\uD7AF\u1100-\u11FF',  # Hangul syllables + jamo
    'ru': '\u0400-\u04FF',               # Cyrillic (also uk, bg, sr...)
    'ar': '\u0600-\u06FF',               # Arabic (also fa, ur...)
    'hi': '\u0900-\u097F',               # Devanagari (also mr, ne...)
}


def target_uses_non_latin_script(target_language: str) -> bool:
    """Return True when the target language is written in a non-Latin script."""
    return _resolve_lang_code(target_language) in NON_LATIN_SCRIPT_RANGES


def _translation_script_ok(
    text: str, source_text: str, source_language: str, target_language: str,
) -> bool:
    """Return True when `text` plausibly IS a translation into the target language.

    The most damaging online-engine failure for dubbing is an answer in the wrong
    script — an English refusal, a transliteration, or the source line echoed
    back — which would be dubbed as English-sounding audio over the video. For a
    non-Latin target the result must actually be written in that script, and
    Latin letters must not dominate it (a brand name inside a real translation
    stays acceptable). Latin-script targets and number-only lines carry nothing
    to verify and always pass.
    """
    script = NON_LATIN_SCRIPT_RANGES.get(_resolve_lang_code(target_language))
    if script is None:
        return True

    source = str(source_text or '')
    if not re.search(r'[^\W\d_]', source):
        return True  # pure numbers/punctuation: no script to check against

    body = str(text or '').strip()
    if not body:
        return False
    if re.sub(r'\s+', ' ', body).casefold() == re.sub(r'\s+', ' ', source).strip().casefold():
        return False  # the source came back unchanged

    script_chars = len(re.findall(f'[{script}]', body))
    if script_chars == 0:
        return False
    latin_chars = len(re.findall('[A-Za-z]', body))
    return latin_chars <= script_chars


# Reasoning-style models sometimes answer with their entire chain of thought
# ("Source text... Meaning... Option 1... Selected Translation...") instead of
# the line alone. These helpers strip, validate, and salvage that pollution so
# a subtitle never ends up as an analysis essay or a doubled sentence.
_SENTENCE_ENDERS = '។।。！!？?；;…'
_META_LINE = re.compile(
    r'^\s*(?:[*>-]|\d+[.)])?\s*'
    r'(?:source text|meaning|context|tone|breakdown|option|alternative|candidate'
    r'|selected translation|final translation|translation selected|note|notes'
    r'|pinyin|romanization|check|checking|refine|refining|script analysis|intent)\b',
    re.IGNORECASE,
)


class _WrongScriptError(RuntimeError):
    """An engine answered in the wrong script; re-requesting cannot fix it."""


def _strip_reasoning(raw: str) -> str:
    """Remove <think>-style reasoning blocks and stray think tags from an answer."""
    text = str(raw or '')
    # Paired blocks go away entirely.
    if re.search(r'<think\b[^>]*>.*?</think\s*>', text, re.DOTALL | re.IGNORECASE):
        text = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # An answer emitted behind an unterminated block sits after the LAST
    # closing tag; anything above it is reasoning, never the translation.
    opens = re.findall(r'<think\b', text, re.IGNORECASE)
    closes = re.findall(r'</think\s*>', text, re.IGNORECASE)
    if closes:
        text = re.split(r'</think\s*>', text, flags=re.IGNORECASE)[-1]
        text = re.sub(r'^[\s>]*', '', text)
    elif opens:
        text = ''  # unclosed think block: no usable answer followed it
    # Stray tags left around non-think markup.
    text = re.sub(r'</?(?:think|reasoning)\b[^>]*>', '', text, flags=re.IGNORECASE)
    return text.strip()


def _subtitle_length_ok(text: str, source_text: str) -> bool:
    """Reject sprawling answers: a subtitle line is about the size of its source.

    A real translation grows at most a few times over the source (Khmer writes
    Chinese/English dialogue with roughly 3-6x more characters). A dumped
    reasoning chain runs dozens of times longer — that is never a line.
    """
    body = re.sub(r'\s+', '', str(text or ''))
    source = re.sub(r'\s+', '', str(source_text or ''))
    if not body or not source or not re.search(r'[^\W\d_]', source):
        return True
    return len(body) <= max(len(source) * 6, len(source) + 30)


def _collapse_duplicate_sentences(text: str) -> str:
    """Remove an immediately repeated sentence ('A។A។' -> 'A។').

    Some reasoning answers end with the chosen line twice back to back; the
    second copy must never reach the dub audio.
    """
    parts = re.split(rf'([{_SENTENCE_ENDERS}])', str(text or ''))
    rebuilt = ''
    previous: str | None = None
    index = 0
    while index < len(parts):
        unit = parts[index]
        ender = parts[index + 1] if index + 1 < len(parts) else ''
        key = re.sub(r'\s+', '', unit)
        if key and key == previous:
            index += 2  # same sentence again: drop it, keep the first copy
            continue
        if key:
            previous = key
        rebuilt += unit + ender
        index += 2
    return rebuilt.strip() or str(text or '').strip()


def _extract_subtitle_answer(
    raw: str, source_text: str, source_language: str, target_language: str,
) -> str | None:
    """Salvage the real translated line out of a verbose model answer.

    Iterates the lines of the answer bottom-up (reasoning models put the final
    choice last), keeps every translation-labelled line and every candidate
    that reads like the target language at subtitle length, and returns the
    most recent one. Returns None when nothing in the answer looks like the
    translation — the caller then retries or fails over.
    """
    lines = [line.strip() for line in _strip_reasoning(raw).splitlines() if line.strip()]
    extracted = ''
    for line in reversed(lines):
        label = re.match(r'^\s*[-*>]?\s*(?:selected translation|final translation|best translation)\s*[:：]\s*(.+)$', line, re.IGNORECASE)
        if label:
            line = label.group(1).strip().strip('"“”').strip()
        elif _META_LINE.match(line):
            continue
        line = line.strip('`').strip()
        if not line:
            continue
        line = _collapse_duplicate_sentences(line)
        if not re.sub(r'\s+', '', line).strip(_SENTENCE_ENDERS):
            continue
        if not _translation_script_ok(line, source_text, source_language, target_language):
            continue
        if not _subtitle_length_ok(line, source_text):
            continue
        extracted = line
        break
    return extracted or None


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
    def _configure_settings() -> None:
        """Point Argos Translate at the app's model folder.

        Argos resolves installed packages through ``package_dirs`` and extracts
        downloads into ``package_data_dir``, so all three settings must move
        together: setting only ``data_dir`` leaves the default package folder in
        place, which hides packages that are actually installed (and breaks the
        offline engine check). The language list is lru_cached, so it is cleared
        when the paths change.
        """
        import argostranslate.settings as argos_settings
        from argostranslate import translate as argos_translate

        data_dir = ArgosTranslationProvider._data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        packages_dir = data_dir / 'packages'
        packages_dir.mkdir(parents=True, exist_ok=True)
        unchanged = (
            Path(argos_settings.data_dir) == data_dir
            and [Path(entry) for entry in argos_settings.package_dirs] == [packages_dir]
            and Path(argos_settings.package_data_dir) == packages_dir
        )
        if unchanged:
            return
        argos_settings.data_dir = data_dir
        argos_settings.package_dirs = [packages_dir]
        argos_settings.package_data_dir = packages_dir
        argos_translate.get_installed_languages.cache_clear()

    @staticmethod
    def _ensure_package(source: str, target: str) -> bool:
        """Download the language package if it is not already installed.

        Returns True when the requested pair is available, False otherwise.
        """
        try:
            from argostranslate import package as argos_package, translate as argos_translate

            ArgosTranslationProvider._configure_settings()

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
    def supports_pair(source_language: str, target_language: str) -> bool:
        """Return True when Argos can translate source->target, directly or via English.

        Mirrors translate(): a direct package, or a pivot through English. Answers
        from the installed packages and the online catalogue without translating.
        """
        source = _resolve_lang_code(source_language)
        target = _resolve_lang_code(target_language)
        if source == target:
            return True
        try:
            from argostranslate import package as argos_package, translate as argos_translate

            ArgosTranslationProvider._configure_settings()

            installed = argos_translate.get_installed_languages()
            if ArgosTranslationProvider._find_installed_translation(installed, source, target) is not None:
                return True
            argos_package.update_package_index()
            catalogue = {(p.from_code, p.to_code) for p in argos_package.get_available_packages()}
            if (source, target) in catalogue:
                return True
            if source != 'en' and target != 'en':
                return (source, 'en') in catalogue and ('en', target) in catalogue
            return False
        except Exception as error:
            logger.warning('Argos pair check failed for %s->%s: %s', source, target, error)
            return False

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

        ArgosTranslationProvider._configure_settings()
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
        # Same trust settings as the keyed providers, so a corporate CA bundle
        # (DUBVIBE_CA_BUNDLE) fixes every online call in the app, not just some.
        from backend.app.online_providers import _ssl_context

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                # A browser-like User-Agent avoids the aggressive throttling that
                # the endpoint applies to default urllib clients on long runs.
                request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urlopen(request, timeout=30, context=_ssl_context()) as response:
                    payload = json.load(response)
                translated = ''.join(part[0] for part in payload[0] if part and part[0])
                if not translated:
                    raise RuntimeError('Google Translate returned an empty result')
                # Guard the last-resort engine too, but never re-request: a
                # wrong-script answer will not change on a second identical
                # call, so fail over immediately instead of sleeping/retrying.
                if not _translation_script_ok(translated, text, source_language, target_language):
                    raise _WrongScriptError(
                        f'Google Translate returned text that is not in {target_language}'
                    )
                return translated
            except _WrongScriptError:
                raise
            except Exception as error:
                last_error = error
                if attempt < self.max_attempts - 1:
                    # Exponential backoff with jitter rides out rate limits and
                    # transient network drops instead of failing the whole run.
                    # A 429 means the endpoint wants a longer pause, so it waits longer.
                    rate_limited = isinstance(error, HTTPError) and error.code == 429
                    base = 15.0 if rate_limited else 0.5 * (2 ** attempt)
                    time.sleep(min(60.0, base * (2 ** attempt if rate_limited else 1)) + random.uniform(0.0, 0.5))
        raise RuntimeError(
            f'Google Translate failed after {self.max_attempts} attempts: {last_error}'
        ) from last_error


class OnlineLLMTranslationProvider(TranslationProvider):
    """Base for the paid online engines: one line in, one translation out.

    Both hosted LLMs support every language this app offers (Khmer included), which
    is why they are not gated on the TTS language tables. The prompt mirrors the
    local Qwen engine so switching engines keeps the same dubbing style.
    """

    ENGINE = ''
    DEFAULT_MODEL = ''

    def __init__(self, style_hint: str | None = None) -> None:
        self.style_hint = style_hint

    def _provider_label(self) -> str:
        return self.ENGINE.capitalize()

    def _key(self) -> str:
        from . import api_keys

        key = api_keys.get_key(self.ENGINE)
        if not key:
            raise RuntimeError(
                f'{self._provider_label()} API key is not configured '
                '(add it in API Settings to translate with this engine).'
            )
        return key

    def _messages(self, text: str, target_language: str, source_language: str) -> list[dict]:
        system = (
            'You are a professional subtitle translator for movie dubbing. '
            f"Translate the user's line from {source_language} to {target_language}. "
            'Keep it natural, spoken, and concise — like real movie dialogue. '
            'Preserve tone and emotion. '
            'Answer with ONLY the translated line itself — one line of plain text. '
            'Never show reasoning, analysis, alternatives, labels, quotes, or the '
            'source line, and never repeat the translation.'
        )
        if target_uses_non_latin_script(target_language):
            system += (
                f' Write the line in {target_language} native script only — never '
                f'transliterate, never repeat the {source_language} text, never explain.'
            )
        if self.style_hint:
            system += f' Character/style direction: {self.style_hint}'
        return [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text},
        ]

    def _complete(self, messages: list[dict], key: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def _clean_result(result: object) -> str:
        return str(result or '').strip().strip('"“”').strip()

    def _accepts(self, candidate: str, text: str, source_language: str, target_language: str) -> bool:
        """A clean result that is in the target script and subtitle-sized."""
        return bool(candidate) and _translation_script_ok(
            candidate, text, source_language, target_language,
        ) and _subtitle_length_ok(candidate, text)

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        if not str(text or '').strip():
            return ''
        # Exactly ONE engine call per line (the normal cost): cleanup and
        # salvage below are local string work and never spend tokens or time.
        raw = self._complete(self._messages(text, target_language, source_language), self._key())
        cleaned = self._clean_result(_strip_reasoning(raw))
        if not cleaned:
            raise RuntimeError(f'{self._provider_label()} returned an empty translation')
        if self._accepts(cleaned, text, source_language, target_language):
            return cleaned
        # Reasoning-style engines sometimes answer with their whole analysis
        # ("Source text... Meaning... Option 1... Selected Translation...") or
        # the line doubled. Salvage the last line that actually reads like the
        # translation — free, same single answer, no second request.
        salvaged = _extract_subtitle_answer(raw, text, source_language, target_language)
        if salvaged:
            return salvaged
        # Nothing usable in the answer: fail over immediately (as any provider
        # error always has), never by paying for another completion.
        raise RuntimeError(
            f'{self._provider_label()} did not return {target_language} text'
            + (f' (got: {cleaned[:60]})' if cleaned else '')
        )


class GeminiTranslationProvider(OnlineLLMTranslationProvider):
    """Google Gemini text translation through the stored Gemini API key."""

    ENGINE = 'gemini'

    def _complete(self, messages: list[dict], key: str) -> str:
        from .online_providers import GEMINI_DEFAULT_CHAT_MODEL, gemini_chat

        return gemini_chat(key, messages, model=GEMINI_DEFAULT_CHAT_MODEL)

    @staticmethod
    def is_available() -> bool:
        from . import api_keys

        return api_keys.has_key('gemini')


class OpenAITranslationProvider(OnlineLLMTranslationProvider):
    """OpenAI text translation through the stored OpenAI API key."""

    ENGINE = 'openai'

    def _complete(self, messages: list[dict], key: str) -> str:
        from .online_providers import OPENAI_DEFAULT_CHAT_MODEL, openai_chat

        return openai_chat(key, messages, model=OPENAI_DEFAULT_CHAT_MODEL)

    @staticmethod
    def is_available() -> bool:
        from . import api_keys

        return api_keys.has_key('openai')


class CodeCraftTranslationProvider(OnlineLLMTranslationProvider):
    """CodeCraft API text translation (OpenAI-compatible, free monthly tier).

    Translation only: the CodeCraft API exposes no speech endpoint, so this
    engine never appears in the TTS picker. Keys start with ``cc_``.
    """

    ENGINE = 'codecraft'

    def _provider_label(self) -> str:
        return 'CodeCraft'

    def _complete(self, messages: list[dict], key: str) -> str:
        from .online_providers import CODECRAFT_DEFAULT_CHAT_MODEL, codecraft_chat

        return codecraft_chat(key, messages, model=CODECRAFT_DEFAULT_CHAT_MODEL)

    @staticmethod
    def is_available() -> bool:
        from . import api_keys

        return api_keys.has_key('codecraft')


# Every language the app offers is supported by the hosted LLMs (unlike their TTS
# voices, which do not officially list Khmer).
ONLINE_TRANSLATION_ENGINES = ('gemini', 'openai', 'codecraft')


class HybridTranslationProvider(TranslationProvider):
    """Translation chain with a user-selectable first choice.

    Default (``auto``) order stays local/free: Qwen3 4B -> Argos -> Google, plus
    Gemini -> OpenAI -> CodeCraft when their keys exist but only if explicitly
    chosen, so configuring a key never silently spends the user's quota or money.
    ``preference`` moves one engine to the front; the rest stay as fallbacks.
    """

    def __init__(self, style_hint: str | None = None, preference: str = 'auto') -> None:
        self._qwen: TranslationProvider | None = None
        self._nllb: TranslationProvider | None = None
        self._offline: TranslationProvider | None = None
        self._online = GoogleTranslateProvider()
        # Paid engines are instantiated only when a key exists; the key is read
        # again per call, so adding a key takes effect without a restart.
        self._gemini: TranslationProvider | None = None
        self._openai: TranslationProvider | None = None
        self._codecraft: TranslationProvider | None = None
        self._mode: str = 'online'
        self._preference = preference if preference in (
            'qwen', 'nllb', 'argos', 'google', *ONLINE_TRANSLATION_ENGINES,
        ) else 'auto'

        try:
            if GeminiTranslationProvider.is_available():
                self._gemini = GeminiTranslationProvider(style_hint)
        except Exception as error:
            logger.info('Gemini translation unavailable: %s', error)

        try:
            if OpenAITranslationProvider.is_available():
                self._openai = OpenAITranslationProvider(style_hint)
        except Exception as error:
            logger.info('OpenAI translation unavailable: %s', error)

        try:
            if CodeCraftTranslationProvider.is_available():
                self._codecraft = CodeCraftTranslationProvider(style_hint)
        except Exception as error:
            logger.info('CodeCraft translation unavailable: %s', error)

        try:
            from . import qwen_local

            if qwen_local.is_installed():
                self._qwen = qwen_local.QwenTranslationProvider(style_hint)
                self._mode = 'qwen'
        except Exception as error:
            logger.info('Qwen local translation unavailable: %s', error)

        try:
            from . import nllb_local

            if nllb_local.is_installed():
                self._nllb = nllb_local.NllbTranslationProvider()
                self._mode = 'nllb'
        except Exception as error:
            logger.info('NLLB-200 translation unavailable: %s', error)

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
        # Auto stays local/free: a configured key is used only when the user picks
        # that engine, so translation never costs money by surprise. NLLB is opt-in
        # for the same reason (it is not in the Auto order).
        order = ([self._preference] if self._preference != 'auto' else []) + ['qwen', 'argos', 'google']
        available = {
            'qwen': self._qwen,
            'nllb': self._nllb,
            'argos': self._offline,
            'google': self._online,
            'gemini': self._gemini,
            'openai': self._openai,
            'codecraft': self._codecraft,
        }
        chain: list[tuple[str, TranslationProvider]] = []
        for key in order:
            provider = available.get(key)
            if provider is not None and all(provider is not seen for _, seen in chain):
                chain.append((key, provider))
        return chain

    def first_engine(self) -> str | None:
        """Key of the first engine tried ('qwen', 'nllb', 'argos', 'google', 'gemini', 'openai', 'codecraft')."""
        chain = self._chain()
        return chain[0][0] if chain else None

    def check_preferred_engine(self, target_language: str, source_language: str = 'en') -> None:
        """Raise before any translation if the engine the user chose cannot handle the language.

        Only an explicit choice is checked. 'auto' keeps its fallback chain, and
        Google is never checked here because it is the online last resort.
        """
        if self._preference == 'auto' or self._preference == 'google':
            return
        if self._preference in ONLINE_TRANSLATION_ENGINES:
            online = {
                'gemini': (self._gemini, 'Gemini'),
                'openai': (self._openai, 'OpenAI'),
                'codecraft': (self._codecraft, 'CodeCraft'),
            }
            provider, label = online[self._preference]
            if provider is None:
                raise RuntimeError(
                    f'{label} API key is not configured. Add it in API Settings or choose another engine.'
                )
            return
        if self._preference == 'argos':
            if self._offline is None:
                raise RuntimeError('Argos Translate is not installed. Choose another translation engine.')
            if not ArgosTranslationProvider.supports_pair(source_language, target_language):
                raise RuntimeError(
                    f'Argos Translate has no {source_language} -> {target_language} language package, '
                    'so it cannot translate this project. Choose Google Translate or another engine.'
                )
        if self._preference == 'qwen' and self._qwen is None:
            raise RuntimeError('Qwen3 4B is not downloaded. Install it in AI Models or choose another engine.')
        if self._preference == 'nllb':
            if self._nllb is None:
                raise RuntimeError('NLLB-200 is not downloaded. Install it in AI Models or choose another engine.')
            from . import nllb_local

            for language in (source_language, target_language):
                try:
                    nllb_local.nllb_code(language)
                except ValueError as error:
                    raise RuntimeError(f'{error}. Choose Google Translate or another engine.') from error

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        modes = {
            'qwen': 'qwen', 'nllb': 'nllb', 'argos': 'offline', 'google': 'online',
            'gemini': 'gemini', 'openai': 'openai', 'codecraft': 'codecraft',
        }
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

        # pyttsx3 can only save WAV, so render there first, then place the audio at
        # the caller's exact out_path. Writing to a different extension (the old
        # behaviour) left the caller pointing at a file that did not exist.
        wav_path = str(Path(output_path).with_suffix('.wav'))
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        engine.stop()

        final_path = output_path
        rendered = Path(wav_path)
        if not rendered.is_file():
            raise RuntimeError('The Windows voice produced no audio file.')
        try:
            import shutil
            ffmpeg = shutil.which('ffmpeg')
            if ffmpeg and Path(output_path).suffix.lower() == '.mp3':
                subprocess.run(
                    [ffmpeg, '-y', '-i', wav_path, '-codec:a', 'libmp3lame', '-q:a', '4', output_path],
                    check=True,
                    capture_output=True,
                )
                rendered.unlink(missing_ok=True)
            elif str(rendered) != str(output_path):
                shutil.move(str(rendered), output_path)
        except Exception:
            # Keep the WAV, but still place it where the caller expects it.
            import shutil
            if str(rendered) != str(output_path) and rendered.is_file():
                shutil.move(str(rendered), output_path)

        # A Windows voice with no matching language still "saves" — as a header-only
        # file with no audio frames. Report that as a failure so the chain can move on
        # instead of shipping a silent dub.
        if not _audio_has_sound(final_path):
            Path(final_path).unlink(missing_ok=True)
            raise SpeechUnavailableError(
                f'The Windows voice cannot speak {language} '
                '(no matching SAPI5 voice is installed).'
            )

        voice_name = selected_voice_id.split('\\')[-1] if selected_voice_id and '\\' in selected_voice_id else (selected_voice_id or voice)
        return {
            'provider': 'pyttsx3 (Offline)',
            'voice': voice_name,
            'text': text,
            'output': final_path,
            'status': 'ok',
        }


_MPEG1_KBPS: dict[int, tuple[int, ...]] = {
    3: (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    2: (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    1: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
}
_MPEG2_KBPS: dict[int, tuple[int, ...]] = {
    3: (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    1: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_MPEG1_RATES = (44100, 48000, 32000)
_MPEG2_RATES = (22050, 24000, 16000)
_MPEG25_RATES = (11025, 12000, 8000)


def _mp3_frame_length(header: bytes) -> int | None:
    """Byte length of the MPEG audio frame whose 4-byte header is `header`.

    None when the bytes are not a usable frame header (reserved version/layer,
    free or invalid bitrate, invalid sample rate).
    """
    if len(header) < 4:
        return None
    version = (header[1] >> 3) & 0x03  # 3=MPEG1, 2=MPEG2, 0=MPEG2.5, 1=reserved
    layer = (header[1] >> 1) & 0x03  # 3=Layer I, 2=Layer II, 1=Layer III, 0=reserved
    if version == 1 or layer == 0:
        return None
    bitrate_index = (header[2] >> 4) & 0x0F
    rate_index = (header[2] >> 2) & 0x03
    padding = (header[2] >> 1) & 0x01
    if bitrate_index in (0, 15) or rate_index == 3:
        return None
    mpeg1 = version == 3
    table = _MPEG1_KBPS if mpeg1 else _MPEG2_KBPS
    bitrate = table[layer][bitrate_index] * 1000
    if mpeg1:
        sample_rate = _MPEG1_RATES[rate_index]
    elif version == 2:
        sample_rate = _MPEG2_RATES[rate_index]
    else:
        sample_rate = _MPEG25_RATES[rate_index]
    if layer == 3:  # Layer I: 4-byte slots
        return (12 * bitrate // sample_rate + padding) * 4
    if layer == 1 and not mpeg1:  # Layer III, MPEG2/2.5 halves the samples
        return 72 * bitrate // sample_rate + padding
    return 144 * bitrate // sample_rate + padding


def _frames_without_ffprobe(target: Path) -> bool | None:
    """Decoder-free frame check: False only when the container provably has none.

    Returns None when the format is not recognized, so callers keep the
    permissive "accept a non-empty file" behavior for anything unverifiable.
    """
    try:
        data = target.read_bytes()
    except OSError:
        return False
    if not data:
        return False

    if data[:4] in (b'RIFF', b'RIFX', b'RF64'):
        try:
            with wave.open(str(target), 'rb') as handle:
                return handle.getnframes() > 0
        except (wave.Error, EOFError, OSError):
            return None

    start = 0
    is_mp3 = False
    if data[:3] == b'ID3' and len(data) >= 10:
        is_mp3 = True
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) \
            | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        start = 10 + size + (10 if data[5] & 0x10 else 0)  # optional footer
    elif len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        is_mp3 = True
    if not is_mp3:
        return None

    for index in range(start, len(data) - 3):
        if data[index] != 0xFF or (data[index + 1] & 0xE0) != 0xE0:
            continue
        length = _mp3_frame_length(data[index:index + 4])
        # A frame only counts when it actually fits: the header-only clip has a
        # valid sync word but stops before its first frame ends.
        if length and index + length <= len(data):
            return True
    return False


def _audio_has_sound(path: str | Path) -> bool:
    """True only when the file really contains audio frames.

    Used to reject a header-only clip (a few hundred bytes, no audio frames) that a
    TTS engine can produce without raising — for example a Windows voice asked for a
    language it cannot speak. ffprobe is authoritative when present; without it a
    decoder-free frame check still rejects a header-only MP3 or an empty WAV, and
    anything unrecognizable is accepted rather than blocking the dub.
    """
    target = Path(path)
    try:
        if not target.is_file() or target.stat().st_size == 0:
            return False
    except OSError:
        return False
    try:
        import shutil

        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            from .main import find_media_tool

            ffprobe = find_media_tool('ffprobe')
    except Exception:
        ffprobe = None
    if not ffprobe:
        # Without ffprobe the file used to be accepted unconditionally, which is
        # how a 226-byte header-only MP3 reached the exported dub.
        measured = _frames_without_ffprobe(target)
        return True if measured is None else measured
    try:
        result = subprocess.run(
            [ffprobe, '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(target)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip() or '0') > 0.0
    except (subprocess.CalledProcessError, ValueError, OSError):
        return False


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


def edge_voice_label(neural: str, gender: str) -> str:
    """Display name of an Edge neural voice: km-KH-PisethNeural -> 'Piseth (Male)'.

    The UI used to show the abstract slot ("Male Voice A"), which hid the fact
    that Edge offers Piseth and Sreymom for Khmer. Naming the real voice keeps
    the list self-explanatory.
    """
    name = neural.rsplit('-', 1)[-1].removesuffix('Neural')
    return f'{name} (Narrator)' if gender == 'Any' else f'{name} ({gender})'


def edge_voice_options(language: str) -> list[dict[str, str]]:
    """Every Edge voice that really speaks `language`, in display order.

    The list is derived from EDGE_LANGUAGE_VOICES (the table the synthesizer
    resolves against), so the UI can only offer voices that will actually speak
    the target language — Khmer has a male and a female voice, never "Narrator".
    """
    options: list[dict[str, str]] = []
    for (lang, label), neural in EDGE_LANGUAGE_VOICES.items():
        if lang != language:
            continue
        lowered = label.lower()
        gender = 'Female' if 'female' in lowered else 'Male' if 'male' in lowered else 'Any'
        options.append({
            'id': label,
            'label': edge_voice_label(neural, gender),
            'voice': neural,
            'gender': gender,
        })
    return options


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
            # "No audio was received" is what Edge returns when its backend has no
            # audio for that voice (seen for whole languages during an outage) —
            # retrying the same voice will not help, so report it as unavailable.
            if last_error is not None and 'No audio' in str(last_error):
                raise SpeechUnavailableError(
                    f'Edge TTS returned no audio for {language} ({selected_voice}). '
                    'Microsoft\'s Edge voice service is currently not serving this '
                    'language — other languages still work. Retry later or pick another '
                    'voice engine.'
                ) from last_error
            raise RuntimeError(f'Edge TTS failed after 3 attempts: {last_error}') from last_error
        # Some failures skip the loop's break but still leave an empty file.
        if not _audio_has_sound(output):
            output.unlink(missing_ok=True)
            raise SpeechUnavailableError(
                f'Edge TTS produced no audible audio for {language} ({selected_voice}).'
            )
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
        from . import nllb_local

        nllb_ok = nllb_local.is_installed()
    except Exception:
        nllb_ok = False
    try:
        from . import voxcpm_local

        voxcpm_ok = voxcpm_local.is_installed()
    except Exception:
        voxcpm_ok = False
    try:
        from . import demucs_local

        # Separation needs both the model and the package, so report ready only with both.
        demucs_ok = demucs_local.is_installed() and importlib.util.find_spec('demucs') is not None
    except Exception:
        demucs_ok = False

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
            'id': 'nllb',
            'name': 'NLLB-200 600M (translation)',
            'mode': 'LOCAL AI',
            'connected': nllb_ok,
            'primary': False,
            'fallback': nllb_ok,
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
            'id': 'gemini',
            'name': 'Google Gemini (translation, TTS)',
            'mode': 'PAID API',
            'connected': GeminiTranslationProvider.is_available(),
            'primary': False,
            'fallback': GeminiTranslationProvider.is_available(),
        },
        {
            'id': 'openai',
            'name': 'OpenAI (translation, TTS)',
            'mode': 'PAID API',
            'connected': OpenAITranslationProvider.is_available(),
            'primary': False,
            'fallback': OpenAITranslationProvider.is_available(),
        },
        {
            'id': 'codecraft',
            'name': 'CodeCraft (translation, free tier)',
            'mode': 'FREE ONLINE',
            'connected': CodeCraftTranslationProvider.is_available(),
            'primary': False,
            'fallback': CodeCraftTranslationProvider.is_available(),
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
            'id': 'demucs',
            'name': 'Demucs 6-stem (audio separation)',
            'mode': 'LOCAL AI',
            'connected': demucs_ok,
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
            'id': 'voxcpm2',
            'name': 'VoxCPM2 (offline, English, Apache-2.0)',
            'mode': 'LOCAL AI',
            'connected': voxcpm_ok,
            'primary': False,
            'fallback': voxcpm_ok,
        },
    ]
