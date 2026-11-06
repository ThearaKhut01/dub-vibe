from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
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


_CJK_RANGES = '\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF'


def _short_cjk_source(source_text: str) -> bool:
    """True for a short Chinese/Japanese/Korean line (usually a name or term)."""
    body = re.sub(r'\s+', '', str(source_text or ''))
    return 1 <= len(body) <= 12 and bool(re.search(f'[{_CJK_RANGES}]', body))


def _looks_romanized_name(text: str) -> bool:
    """A name-sized Latin answer: one to three capitalized words, no placeholders."""
    body = str(text or '')
    if re.search(r'[\[\]{}<>]', body):
        return False  # "[need khmer wording]" and other placeholders are not names
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", body)
    if not 1 <= len(words) <= 3:
        return False
    return all(word[:1].isupper() for word in words)


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

    Short CJK lines are the exception: a character name has no translation and
    every engine romanizes it ("蕭顏" -> "Xiao Yan"). Rejecting that used to fail
    an entire run on a handful of names, so a name-sized romanization is accepted.
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
    latin_chars = len(re.findall('[A-Za-z]', body))
    name_like = _short_cjk_source(source)
    if script_chars == 0:
        # Romanized name/term: "Xiao Yan", "Xueer?" — not an English refusal.
        return name_like and _looks_romanized_name(body)
    if latin_chars <= script_chars:
        return True
    # Mostly-Latin answer for a short CJK line: proper names kept in Latin next
    # to real Khmer text ("... បានធំឡើង" with three names in front).
    return name_like and not re.search(r'[\[\]{}<>]', body)


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


# Argos keeps an online package index. Refreshing it costs a network request,
# and it used to be refreshed once per line whenever a pair was missing — a
# Khmer video with no zh->km package made one request per subtitle line before
# falling through to Google. Cache the catalogue and remember pairs that are
# not available so the chain fails over immediately instead of re-probing.
_ARGOS_INDEX_TTL = 600.0
_ARGOS_MISSING_TTL = 600.0
_ARGOS_PACKAGES: list | None = None
_ARGOS_INDEX_AT = 0.0
_ARGOS_MISSING_PAIRS: dict[tuple[str, str], float] = {}
_ARGOS_LOCK = threading.Lock()


# Google's free endpoint answers 429 when a machine sends too many lines in a
# short time. Retrying the same line four times costs ~100 s and still fails, so
# a 429 parks the engine briefly: the rest of the run stops with one clear
# message instead of sending hundreds of doomed requests.
_GOOGLE_COOLDOWN_SECONDS = 60.0
_GOOGLE_COOLDOWN_UNTIL = 0.0
_GOOGLE_LOCK = threading.Lock()


def reset_argos_cache() -> None:
    """Forget the cached package index and missing pairs (after installing one)."""
    global _ARGOS_PACKAGES, _ARGOS_INDEX_AT
    with _ARGOS_LOCK:
        _ARGOS_PACKAGES = None
        _ARGOS_INDEX_AT = 0.0
        _ARGOS_MISSING_PAIRS.clear()


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
    def _available_packages(force_refresh: bool = False) -> list:
        """The Argos package catalogue, refreshed at most once every ten minutes."""
        global _ARGOS_PACKAGES, _ARGOS_INDEX_AT
        from argostranslate import package as argos_package

        with _ARGOS_LOCK:
            stale = (
                _ARGOS_PACKAGES is None
                or force_refresh
                or time.monotonic() - _ARGOS_INDEX_AT > _ARGOS_INDEX_TTL
            )
            if stale:
                argos_package.update_package_index()
                _ARGOS_PACKAGES = list(argos_package.get_available_packages())
                _ARGOS_INDEX_AT = time.monotonic()
            return list(_ARGOS_PACKAGES or [])

    @staticmethod
    def _pair_recently_missing(source: str, target: str) -> bool:
        with _ARGOS_LOCK:
            marked = _ARGOS_MISSING_PAIRS.get((source, target))
            if marked is None:
                return False
            if time.monotonic() - marked > _ARGOS_MISSING_TTL:
                _ARGOS_MISSING_PAIRS.pop((source, target), None)
                return False
            return True

    @staticmethod
    def _mark_pair_missing(source: str, target: str) -> None:
        with _ARGOS_LOCK:
            _ARGOS_MISSING_PAIRS[(source, target)] = time.monotonic()

    @staticmethod
    def _ensure_package(source: str, target: str) -> bool:
        """Download the language package if it is not already installed.

        Returns True when the requested pair is available, False otherwise.
        A missing pair is remembered for ten minutes so a whole run does not
        refresh the catalogue (a network request) once per subtitle line.
        """
        try:
            from argostranslate import package as argos_package, translate as argos_translate

            ArgosTranslationProvider._configure_settings()

            # Check if the pair is already installed.
            installed = argos_translate.get_installed_languages()
            if ArgosTranslationProvider._find_installed_translation(installed, source, target) is not None:
                return True
            if ArgosTranslationProvider._pair_recently_missing(source, target):
                return False

            # Try to download the package.
            pkg = next(
                (
                    candidate
                    for candidate in ArgosTranslationProvider._available_packages()
                    if candidate.from_code == source and candidate.to_code == target
                ),
                None,
            )
            if pkg is None:
                # No direct package for this pair; the caller may pivot via English.
                ArgosTranslationProvider._mark_pair_missing(source, target)
                return False
            download_path = pkg.download()
            argos_package.install_from_path(download_path)
            installed = argos_translate.get_installed_languages()
            found = ArgosTranslationProvider._find_installed_translation(installed, source, target) is not None
            reset_argos_cache()
            return found
        except Exception as error:
            logger.warning('Argos package install failed for %s->%s: %s', source, target, error)
            ArgosTranslationProvider._mark_pair_missing(source, target)
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
            from argostranslate import translate as argos_translate

            ArgosTranslationProvider._configure_settings()

            installed = argos_translate.get_installed_languages()
            if ArgosTranslationProvider._find_installed_translation(installed, source, target) is not None:
                return True
            if ArgosTranslationProvider._pair_recently_missing(source, target):
                return False
            packages = ArgosTranslationProvider._available_packages()
            catalogue = {(p.from_code, p.to_code) for p in packages}
            if (source, target) in catalogue:
                return True
            if source != 'en' and target != 'en':
                if (source, 'en') in catalogue and ('en', target) in catalogue:
                    return True
            ArgosTranslationProvider._mark_pair_missing(source, target)
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

    @staticmethod
    def _cooldown_remaining() -> float:
        with _GOOGLE_LOCK:
            return max(0.0, _GOOGLE_COOLDOWN_UNTIL - time.monotonic())

    @staticmethod
    def _park() -> None:
        """Stop sending lines to Google after it says 429 (rate limited)."""
        global _GOOGLE_COOLDOWN_UNTIL
        with _GOOGLE_LOCK:
            _GOOGLE_COOLDOWN_UNTIL = time.monotonic() + _GOOGLE_COOLDOWN_SECONDS

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        target = _resolve_lang_code(target_language)
        source = _resolve_lang_code(source_language)
        if source == target:
            return text
        remaining = self._cooldown_remaining()
        if remaining > 0:
            raise RuntimeError(
                'Google Translate is rate-limiting this machine. '
                f'Wait about {max(1, int(remaining))} seconds, then press Translate again — '
                'finished lines are already saved.'
            )
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
                # A 429 is about the whole endpoint, not this line: retrying it
                # four times costs ~100 seconds per subtitle line and still
                # fails. Park the engine so the run stops with one clear
                # message instead of hammering Google for the rest of the video.
                if isinstance(error, HTTPError) and error.code == 429:
                    self._park()
                    raise RuntimeError(
                        'Google Translate rate-limited this machine (HTTP 429). '
                        f'Wait about {int(_GOOGLE_COOLDOWN_SECONDS)} seconds, then press '
                        'Translate again — finished lines are already saved.'
                    ) from error
                if attempt < self.max_attempts - 1:
                    # Transient network drops: exponential backoff with jitter.
                    time.sleep(0.5 * (2 ** attempt) + random.uniform(0.0, 0.5))
        raise RuntimeError(
            f'Google Translate failed after {self.max_attempts} attempts: {last_error}'
        ) from last_error


# ---------------------------------------------------------------------------
# Batch translation: one request for a group of subtitle lines
# ---------------------------------------------------------------------------

# One subtitle line is a tiny request, but repeating the full instruction block
# for every line is what made the old translate step expensive: on a 300-line
# video the system prompt alone was sent 300 times. Lines are now grouped into
# requests that are bounded by BOTH a line count and a character budget, so a
# batch never turns into a huge single completion either.
TRANSLATION_BATCH_MAX_LINES = 40
TRANSLATION_BATCH_MAX_CHARS = 3000
TRANSLATION_CONTEXT_LINES = 3
TRANSLATION_BATCH_RETRIES = 2
TRANSLATION_OUTPUT_TOKEN_CAP = 8192

# "1|text" (the id is batch-local, 1-based) — tolerant of the spacing models add.
_BATCH_LINE = re.compile(r'^\s*(\d+)\s*\|\s?(.*)$')

_TRANSLATION_SYSTEM_TEMPLATE = (
    'You are a professional dubbing translator. Translate subtitle lines into '
    '{TARGET} for voice-over. The source language is {SOURCE} (auto-detect per line).\n'
    '\n'
    'Rules:\n'
    '1. Input: one line per item as "ID|text". Output the same IDs in the same order as "ID|translation".\n'
    '2. Output exactly one line per input line. Never merge, split, skip, or add lines.\n'
    '3. Output ONLY the {TARGET_NAME} translations. No explanations, notes, quotes, pinyin, or markdown.\n'
    '4. Translate for speech: natural spoken {TARGET_NAME}, easy to read aloud. Not word by word. '
    'Keep it as short as possible so it fits the original timing.\n'
    '5. Chinese: translate the meaning, not the characters. Convert idioms, slang, and internet '
    'phrases into natural {TARGET_NAME}. Resolve omitted subjects using CONTEXT.\n'
    '6. Names: keep brand/product/technical terms in original form. Chinese person names in '
    '{TARGET_NAME} transliteration, spelled the same way every time.\n'
    '7. Numbers as digits.\n'
    '8. Use CONTEXT only to understand meaning. Never output it.\n'
    '9. Lines that are only a sound or symbol ([Music], ♪, (笑)) are returned unchanged.\n'
    '10. Keep the tone consistent: {TONE}.\n'
    '\n'
    'Glossary (use these exact translations):\n'
    '{GLOSSARY}'
)

# The prompt above is written for the product's default target; other targets
# substitute their own name so the same rules still read correctly.
_TARGET_LABELS = {
    'km': ('Khmer', 'ភាសាខ្មែរ'),
    'khmer': ('Khmer', 'ភាសាខ្មែរ'),
}


def translation_target_label(target_language: str) -> tuple[str, str]:
    """(English name, native name) for the system prompt."""
    code = _resolve_lang_code(target_language or 'Khmer')
    if code in _TARGET_LABELS:
        return _TARGET_LABELS[code]
    name = str(target_language or 'Khmer').strip() or 'Khmer'
    return name, name


def build_translation_system_prompt(
    glossary: str = '',
    tone: str = 'natural and consistent',
    target_language: str = 'Khmer',
) -> str:
    """The batch instruction block, sent once per request (never per line).

    ``glossary`` is the project's small term list ("王明=វ៉ាង មីង" lines); an
    empty glossary becomes the literal ``none`` so the model never gets a blank
    section it might fill with invented terms.
    """
    target, native = translation_target_label(target_language)
    return _TRANSLATION_SYSTEM_TEMPLATE.format(
        TARGET=f'{target} ({native})',
        TARGET_NAME=target,
        SOURCE='English or Chinese',
        TONE=str(tone or 'natural and consistent').strip(),
        GLOSSARY=str(glossary or '').strip() or 'none',
    )


def build_translation_user_message(lines: list[str], context: list[str] | None = None,
                                  ids: list[int] | None = None) -> str:
    """``CONTEXT:`` (previous lines, never translated) + ``TRANSLATE:`` with IDs.

    ``ids`` keeps a line's id stable when only the failed lines are re-sent; by
    default the lines are numbered 1..n.
    """
    context_block = '\n'.join(str(line) for line in (context or [])[-TRANSLATION_CONTEXT_LINES:] if str(line).strip())
    identifiers = list(ids) if ids is not None else list(range(1, len(lines) + 1))
    numbered = '\n'.join(f'{identifier}|{line}' for identifier, line in zip(identifiers, lines))
    return f'CONTEXT:\n{context_block or "none"}\n\nTRANSLATE:\n{numbered}'


def estimate_tokens(text: str) -> int:
    """Rough token count for a mixed English/CJK/Khmer body.

    Deliberately conservative: Khmer costs more tokens per character than Latin
    text, and under-estimating would truncate the batch.
    """
    return max(1, int(len(str(text or '')) / 2.2) + 1)


def batch_output_token_budget(user_message: str, line_count: int) -> int:
    """About 3x the input tokens, bounded by a sane floor and ceiling.

    The floor is per line, not a constant: Khmer costs far more tokens per
    character in some tokenizers (Mistral emitted ~64 tokens for one short line),
    so a 25-line batch needs real headroom or the answer is cut mid-line.
    """
    budget = 3 * estimate_tokens(user_message) + 64
    floor = 512 + 80 * max(0, line_count - 1)
    return int(min(TRANSLATION_OUTPUT_TOKEN_CAP, max(budget, floor)))


def split_translation_batches(
    texts: list[str],
    max_lines: int = TRANSLATION_BATCH_MAX_LINES,
    max_chars: int = TRANSLATION_BATCH_MAX_CHARS,
) -> list[list[str]]:
    """Order-preserving groups bounded by line count and characters.

    A single line longer than the character budget still gets its own batch:
    dropping it would silently leave that subtitle untranslated.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    for text in texts:
        cost = len(str(text or '')) + 8  # the "N|" prefix and the newline
        if current and (len(current) >= max_lines or used + cost > max_chars):
            batches.append(current)
            current, used = [], 0
        current.append(text)
        used += cost
    if current:
        batches.append(current)
    return batches


def parse_batch_reply(raw: str, expected_ids: list[int] | None = None) -> dict[int, str]:
    """``ID|translation`` lines from a model answer, keyed by ID.

    Tolerant of markdown fences, blank lines and prose the model adds around the
    answer: only lines that start with an id and a pipe are kept. Duplicate ids
    keep the last answer (reasoning models sometimes restate a line).
    """
    text = _strip_reasoning(raw)
    text = re.sub(r'```[a-zA-Z]*', '', text)
    parsed: dict[int, str] = {}
    for line in text.splitlines():
        match = _BATCH_LINE.match(line.strip())
        if not match:
            continue
        identifier = int(match.group(1))
        if expected_ids is not None and identifier not in expected_ids:
            continue
        value = match.group(2).strip()
        if value:
            parsed[identifier] = value
    return parsed


def _clean_batch_value(raw: str, source: str, source_language: str, target_language: str) -> str | None:
    """One accepted translation (or None) using the existing validators."""
    cleaned = str(raw or '').strip().strip('"“”').strip()
    if not cleaned:
        return None
    if _translation_script_ok(cleaned, source, source_language, target_language) and _subtitle_length_ok(cleaned, source):
        return cleaned
    salvaged = _extract_subtitle_answer(cleaned, source, source_language, target_language)
    if salvaged and _translation_script_ok(salvaged, source, source_language, target_language) and _subtitle_length_ok(salvaged, source):
        return salvaged
    return None


# Engines that answered "slow down" are skipped for a while instead of being
# asked again for every line of the video (the same policy the Google engine
# already applies to its own 429s).
_PARKED_ENGINES: dict[str, float] = {}
_PARK_SECONDS = 60.0


def _park_engine(key: str, seconds: float = _PARK_SECONDS) -> None:
    _PARKED_ENGINES[key] = time.monotonic() + seconds


def _engine_parked(key: str) -> bool:
    until = _PARKED_ENGINES.get(key)
    if until is None:
        return False
    if time.monotonic() >= until:
        _PARKED_ENGINES.pop(key, None)
        return False
    return True


# A pool whose entries are all down (an unavailable model, a revoked key) would
# otherwise cost every batch a round of failing attempts before the stored key
# takes over. One failure marks the pool down briefly; the fallback key serves
# the run in the meantime.
_POOL_DOWN_UNTIL: dict[str, float] = {}
_POOL_DOWN_SECONDS = 300.0

# A batch is large and few: one try per configuration and a bounded wait. A
# healthy pool answers in seconds; a broken one must not hold up the run, and the
# breaker above keeps the next batches off it for a while.
BATCH_POOL_TIMEOUT = 30


def _pool_usable(engine: str) -> bool:
    return time.monotonic() >= _POOL_DOWN_UNTIL.get(engine, 0.0)


def _mark_pool_down(engine: str, seconds: float = _POOL_DOWN_SECONDS) -> None:
    _POOL_DOWN_UNTIL[engine] = time.monotonic() + seconds


def clear_pool_breaker() -> None:
    """Forget "this pool is down" (the panel's Clear-cooldowns action)."""
    _POOL_DOWN_UNTIL.clear()


def _looks_rate_limited(error: Exception) -> bool:
    message = str(error).lower()
    return '429' in message or 'rate limit' in message or 'quota' in message or 'resource_exhausted' in message


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
            'source line, and never repeat the translation. '
            'Write personal names phonetically in the target language\'s own script.'
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

    def _stored_key_missing(self) -> bool:
        """True when no single stored credential exists either."""
        try:
            return not self._key()
        except RuntimeError:
            return True

    def _invoke(self, messages: list[dict]) -> str:
        """One completion: the API pool when this engine has pool entries.

        Without pool configurations the call goes through the single stored key,
        exactly as before. When the pool exists but every configuration is parked
        (quota, cooldown, a rejected key), a stored single key is still tried as
        the last resort instead of failing the whole run.
        """
        from . import api_pool

        if api_pool.has_entries(self.ENGINE):
            try:
                return api_pool.chat(self.ENGINE, messages)
            except api_pool.PoolUnavailable as pool_error:
                if self._stored_key_missing():
                    raise RuntimeError(
                        f'{self._provider_label()} pool is unavailable: {pool_error}'
                    ) from pool_error
                logger.info(
                    'API pool unavailable for %s (%s); using the single stored key.',
                    self.ENGINE, pool_error,
                )
        return self._complete(messages, self._key())

    @staticmethod
    def _clean_result(result: object) -> str:
        return str(result or '').strip().strip('"“”').strip()

    def _accepts(self, candidate: str, text: str, source_language: str, target_language: str) -> bool:
        """A clean result that is in the target script and subtitle-sized."""
        return bool(candidate) and _translation_script_ok(
            candidate, text, source_language, target_language,
        ) and _subtitle_length_ok(candidate, text)

    # Batched requests want a low temperature (subtitle lines must stay literal);
    # engines with extra switches return them from _batch_generation_config.
    BATCH_TEMPERATURE = 0.2

    def _batch_generation_config(self) -> dict:
        return {'temperature': self.BATCH_TEMPERATURE}

    def _batch_messages(self, system: str, user: str) -> list[dict]:
        return [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]

    def _batch_complete(self, system: str, user: str, max_tokens: int,
                        usage: dict[str, int] | None = None,
                        meta: dict[str, Any] | None = None) -> str:
        """One batched completion: the API pool when it has entries, else the key.

        The pool is what actually serves this engine at runtime (rotation, quota,
        health), so a batch must not bypass it — otherwise two configured keys turn
        into one, and the pool's usage numbers miss every subtitle line.
        """
        from . import api_events, api_pool

        meta = dict(meta or {})
        meta.setdefault('node_id', f'legacy:{self.ENGINE}')
        lines = int(meta.get('lines') or 0)
        messages = self._batch_messages(system, user)
        if api_pool.has_entries(self.ENGINE) and _pool_usable(self.ENGINE):
            try:
                return api_pool.chat(
                    self.ENGINE, messages, max_tokens=max_tokens, max_attempts=1,
                    timeout=BATCH_POOL_TIMEOUT,
                    generation_config=self._batch_generation_config(), usage_out=usage,
                    api_meta=meta,
                )
            except api_pool.PoolUnavailable as error:
                _mark_pool_down(self.ENGINE)
                if self._stored_key_missing():
                    raise RuntimeError(
                        f'{self._provider_label()} pool is unavailable: {error}'
                    ) from error
                logger.info(
                    '%s pool unavailable for a batch (%s); using the stored key.',
                    self.ENGINE, error,
                )
        key = self._key()
        meta['key_hint'] = api_events.hint_for(key)
        request_id = api_events.start(
            meta['node_id'], provider=self.ENGINE, model=self.DEFAULT_MODEL,
            lines=lines, key_hint=meta['key_hint'],
        )
        started = time.monotonic()
        try:
            text = self._complete(messages, key)
        except Exception as error:
            api_events.finish(
                request_id, meta['node_id'], ok=False, provider=self.ENGINE,
                lines=lines, latency_ms=int((time.monotonic() - started) * 1000),
                error=error, key_hint=meta['key_hint'],
            )
            raise
        api_events.finish(
            request_id, meta['node_id'], ok=True, provider=self.ENGINE, lines=lines,
            latency_ms=int((time.monotonic() - started) * 1000), key_hint=meta['key_hint'],
        )
        return text

    def supports_batch(self) -> bool:
        return True

    def translate_batch(
        self,
        lines: list[str],
        context: list[str] | None = None,
        glossary: str = '',
        *,
        source_language: str = 'auto',
        target_language: str = 'Khmer',
        tone: str = 'natural and consistent',
        usage: dict[str, int] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> list[str | None]:
        """Translate ``lines`` in as few requests as possible.

        Returns one entry per input line: the translation, or ``None`` when the
        model never produced a usable line for it (the caller then falls back to
        the single-line path for just that line). Missing ids are re-requested on
        their own — never the whole batch.
        """
        results: list[str | None] = [None] * len(lines)
        if not lines:
            return results
        system = build_translation_system_prompt(
            glossary=glossary, tone=tone, target_language=target_language,
        )
        pending: dict[int, str] = {index + 1: text for index, text in enumerate(lines)}
        # A truncated answer (the model stops mid-line at its token ceiling) is
        # retried in smaller groups instead of the same oversized request.
        group_size = len(lines)
        for round_index in range(TRANSLATION_BATCH_RETRIES + 1):
            if not pending:
                break
            if round_index:
                # Backoff with jitter, only for the lines that came back missing.
                time.sleep(0.6 * (2 ** (round_index - 1)) + random.uniform(0.0, 0.4))
                if usage is not None:
                    usage['retries'] = int(usage.get('retries', 0)) + 1
            pending_ids = sorted(pending)
            groups = [pending_ids[start:start + group_size] for start in range(0, len(pending_ids), group_size)]
            for ids in groups:
                subset = [pending[identifier] for identifier in ids]
                user = build_translation_user_message(subset, context=context, ids=ids)
                before = dict(usage) if usage is not None else {}
                raw = self._batch_complete(
                    system, user, batch_output_token_budget(user, len(subset)), usage,
                    {'lines': len(subset), **(meta or {})},
                )
                if usage is not None:
                    usage['requests'] = int(usage.get('requests', 0)) + 1
                    # One line per request: what was sent, what it cost, which retry
                    # round it belonged to. This is the audit trail the old per-line
                    # path never had.
                    logger.info(
                        'Translation batch request: lines=%d retry=%d prompt_tokens=%d '
                        'candidates_tokens=%d thoughts_tokens=%d',
                        len(subset), round_index,
                        int(usage.get('prompt_tokens', 0)) - int(before.get('prompt_tokens', 0)),
                        int(usage.get('candidates_tokens', 0)) - int(before.get('candidates_tokens', 0)),
                        int(usage.get('thoughts_tokens', 0)) - int(before.get('thoughts_tokens', 0)),
                    )
                parsed = parse_batch_reply(raw, set(ids))
                if not parsed and str(raw or '').strip():
                    # The model answered, but not in "ID|translation" form: keep the
                    # evidence instead of silently reporting an empty result.
                    logger.warning(
                        'Unparseable batch answer from %s (lines=%d): %s',
                        type(self).__name__, len(subset), str(raw)[:300],
                    )
                for identifier in ids:
                    candidate = parsed.get(identifier)
                    if candidate is None:
                        continue
                    cleaned = _clean_batch_value(
                        candidate, pending[identifier], source_language, target_language,
                    )
                    if cleaned is None:
                        continue
                    results[identifier - 1] = cleaned
                    pending.pop(identifier, None)
            if pending:
                group_size = max(4, group_size // 2)
        return results

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        if not str(text or '').strip():
            return ''
        from . import api_pool

        # A line with this exact source text keeps the wording that already passed
        # validation while the pool serves this engine: repeated dialogue stays
        # consistent, and it is never paid for twice.
        use_cache = api_pool.cache_enabled_for(self.ENGINE)
        if use_cache:
            cached = api_pool.cached_translation(text, source_language, target_language, self.style_hint)
            if cached:
                return cached
        # Exactly ONE engine call per line (the normal cost): cleanup and
        # salvage below are local string work and never spend tokens or time.
        raw = self._invoke(self._messages(text, target_language, source_language))
        cleaned = self._clean_result(_strip_reasoning(raw))
        if not cleaned:
            raise RuntimeError(f'{self._provider_label()} returned an empty translation')
        if self._accepts(cleaned, text, source_language, target_language):
            result = cleaned
        else:
            # Reasoning-style engines sometimes answer with their whole analysis
            # ("Source text... Meaning... Option 1... Selected Translation...") or
            # the line doubled. Salvage the last line that actually reads like the
            # translation — free, same single answer, no second request.
            result = _extract_subtitle_answer(raw, text, source_language, target_language) or ''
            if not result:
                # Nothing usable in the answer: fail over immediately (as any
                # provider error always has), never by paying for another
                # completion.
                raise RuntimeError(
                    f'{self._provider_label()} did not return {target_language} text'
                    + (f' (got: {cleaned[:60]})' if cleaned else '')
                )
        if use_cache:
            api_pool.remember_translation(
                text, result, source_language, target_language,
                self.style_hint, provider=self.ENGINE,
            )
        return result


class GeminiTranslationProvider(OnlineLLMTranslationProvider):
    """Google Gemini text translation through the stored Gemini API key."""

    ENGINE = 'gemini'
    # app-settings.json key: the model this engine uses for translation.
    MODEL_SETTING = 'gemini_translation_model'
    FALLBACK_MODEL = 'gemini-2.5-flash'

    @classmethod
    def model_name(cls) -> str:
        """Configured text model (``gemini-2.5-flash`` unless the user changes it)."""
        from . import model_manager

        try:
            configured = str(model_manager.load_settings().get(cls.MODEL_SETTING) or '').strip()
        except Exception:
            configured = ''
        return configured or cls.FALLBACK_MODEL

    def _complete(self, messages: list[dict], key: str) -> str:
        from .online_providers import gemini_chat

        return gemini_chat(key, messages, model=self.model_name())

    def _batch_generation_config(self) -> dict:
        # 2.5-style models think by default and a subtitle batch never needs it;
        # the pool forwards this verbatim, so the saving holds on both paths.
        return {'temperature': self.BATCH_TEMPERATURE, 'thinkingConfig': {'thinkingBudget': 0}}

    def _batch_complete(self, system: str, user: str, max_tokens: int,
                        usage: dict[str, int] | None = None,
                        meta: dict[str, Any] | None = None) -> str:
        """Batched request: instruction once, low temperature, no thinking.

        The pool serves this engine at runtime, so the batch goes through it first
        (both keys, quota, health); the stored key is the fallback.
        """
        from . import api_events, api_keys, api_pool
        from .online_providers import gemini_chat

        meta = dict(meta or {})
        meta.setdefault('node_id', 'legacy:gemini')
        if api_pool.has_entries('gemini') and _pool_usable('gemini'):
            try:
                return api_pool.chat(
                    'gemini', self._batch_messages(system, user), max_tokens=max_tokens,
                    max_attempts=1, timeout=BATCH_POOL_TIMEOUT,
                    generation_config=self._batch_generation_config(), usage_out=usage,
                    api_meta=meta,
                )
            except api_pool.PoolUnavailable as error:
                _mark_pool_down('gemini')
                if not api_keys.has_key('gemini'):
                    raise RuntimeError(f'Gemini pool is unavailable: {error}') from error
                logger.info('Gemini pool unavailable for a batch (%s); using the stored key.', error)

        key = self._key()
        meta['key_hint'] = api_events.hint_for(key)
        text, result_meta = gemini_chat(
            key,
            [{'role': 'user', 'content': user}],
            model=self.model_name(),
            generation_config={
                'temperature': self.BATCH_TEMPERATURE,
                'maxOutputTokens': int(max_tokens),
                'thinkingConfig': {'thinkingBudget': 0},
            },
            system_instruction=system,
            want_usage=True,
            prefer_requested=True,
            api_meta=meta,
        )
        if usage is not None:
            totals = result_meta.get('usage') or {}
            for field in ('prompt_tokens', 'candidates_tokens', 'thoughts_tokens'):
                usage[field] = int(usage.get(field, 0)) + int(totals.get(field, 0))
            logger.info(
                'Gemini translation: model=%s prompt=%s candidates=%s thoughts=%s finish=%s',
                result_meta.get('model'), totals.get('prompt_tokens'), totals.get('candidates_tokens'),
                totals.get('thoughts_tokens'), result_meta.get('finish_reason'),
            )
        return text

    @staticmethod
    def is_available() -> bool:
        from . import api_keys, api_pool

        return api_keys.has_key('gemini') or api_pool.has_entries('gemini')


class OpenAITranslationProvider(OnlineLLMTranslationProvider):
    """OpenAI text translation through the stored OpenAI API key."""

    ENGINE = 'openai'

    def _complete(self, messages: list[dict], key: str) -> str:
        from .online_providers import OPENAI_DEFAULT_CHAT_MODEL, openai_chat

        return openai_chat(key, messages, model=OPENAI_DEFAULT_CHAT_MODEL)

    @staticmethod
    def is_available() -> bool:
        from . import api_keys, api_pool

        return api_keys.has_key('openai') or api_pool.has_entries('openai')


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
        from . import api_keys, api_pool

        return api_keys.has_key('codecraft') or api_pool.has_entries('codecraft')


class CustomAPITranslationProvider(OnlineLLMTranslationProvider):
    """Any OpenAI-compatible endpoint the user configures (base URL + model).

    Translation only: the app cannot know whether a custom endpoint has a speech
    API, so it never appears in the TTS picker. The key is optional because local
    servers (LM Studio, vLLM, Ollama) usually accept requests without one.
    """

    ENGINE = 'custom'

    def _provider_label(self) -> str:
        return 'Custom API'

    def _key(self) -> str:
        from . import api_keys

        return api_keys.get_key('custom') or ''

    def _complete(self, messages: list[dict], key: str) -> str:
        from . import custom_api

        return custom_api.chat(key, messages)

    def _batch_complete(self, system: str, user: str, max_tokens: int,
                        usage: dict[str, int] | None = None,
                        meta: dict[str, Any] | None = None) -> str:
        """Batched request with an explicit output ceiling and token accounting."""
        from . import api_events, api_pool, custom_api

        meta = dict(meta or {})
        meta.setdefault('node_id', 'legacy:custom')
        messages = self._batch_messages(system, user)
        if api_pool.has_entries('custom') and _pool_usable('custom'):
            try:
                return api_pool.chat(
                    'custom', messages, max_tokens=max_tokens, max_attempts=1,
                    timeout=BATCH_POOL_TIMEOUT,
                    generation_config=self._batch_generation_config(), usage_out=usage,
                    api_meta=meta,
                )
            except api_pool.PoolUnavailable as error:
                _mark_pool_down('custom')
                if self._stored_key_missing():
                    raise RuntimeError(f'Custom API pool is unavailable: {error}') from error
                logger.info('Custom API pool unavailable for a batch (%s); using the endpoint settings.', error)

        key = self._key()
        meta['key_hint'] = api_events.hint_for(key)
        tokens: dict[str, int] = {}
        text = custom_api.chat(
            key,
            messages,
            max_tokens=int(max_tokens),
            usage_out=tokens,
            api_meta=meta,
        )
        if usage is not None:
            for field in ('prompt_tokens', 'candidates_tokens', 'thoughts_tokens'):
                usage[field] = int(usage.get(field, 0)) + int(tokens.get(field, 0))
        return text

    @staticmethod
    def is_available() -> bool:
        from . import api_pool, custom_api

        return custom_api.is_configured() or api_pool.has_entries('custom')

    def _stored_key_missing(self) -> bool:
        """A local custom endpoint is configured even when it needs no key."""
        from . import custom_api

        return not custom_api.is_configured()


# Every language the app offers is supported by the hosted LLMs (unlike their TTS
# voices, which do not officially list Khmer).
ONLINE_TRANSLATION_ENGINES = ('gemini', 'openai', 'codecraft', 'custom')


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
        self._custom: TranslationProvider | None = None
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
            if CustomAPITranslationProvider.is_available():
                self._custom = CustomAPITranslationProvider(style_hint)
        except Exception as error:
            logger.info('Custom API translation unavailable: %s', error)

        try:
            from . import qwen_local

            # A Qwen that cannot load is worse than no Qwen: the chain would
            # retry it on every line and then translate the video with the
            # fallback engine while the UI still says "Qwen". Skip it here and
            # let first_engine() report the engine that will really run.
            if qwen_local.is_installed() and qwen_local.runtime_problem() is None:
                self._qwen = qwen_local.QwenTranslationProvider(style_hint)
                self._mode = 'qwen'
            elif qwen_local.is_installed():
                logger.warning(
                    'Qwen3 4B is installed but llama.cpp cannot start (%s); '
                    'translation will use the next engine.', qwen_local.runtime_problem(),
                )
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
        # An explicit LOCAL choice stays local: choosing Qwen/NLLB/Argos must never
        # send the video to Google behind the user's back.
        local_engines = ('qwen', 'nllb', 'argos')
        if self._preference in local_engines:
            order = [self._preference] + [key for key in local_engines if key != self._preference]
        else:
            order = ([self._preference] if self._preference != 'auto' else []) + ['qwen', 'argos', 'google']
            # Choosing a paid engine means "use this endpoint"; when it fails the
            # user's own configured endpoint is the fallback they expect, before
            # the free/local engines take over.
            if (
                self._preference in ONLINE_TRANSLATION_ENGINES
                and self._preference != 'custom'
                and self._custom is not None
            ):
                order.insert(1, 'custom')
        available = {
            'qwen': self._qwen,
            'nllb': self._nllb,
            'argos': self._offline,
            'google': self._online,
            'gemini': self._gemini,
            'openai': self._openai,
            'codecraft': self._codecraft,
            'custom': self._custom,
        }
        chain: list[tuple[str, TranslationProvider]] = []
        for key in order:
            provider = available.get(key)
            if provider is not None and all(provider is not seen for _, seen in chain):
                chain.append((key, provider))
        return chain

    def first_engine(self) -> str | None:
        """Key of the first engine tried ('qwen', 'nllb', 'argos', 'google', 'gemini', 'openai', 'codecraft', 'custom')."""
        chain = self._chain()
        return chain[0][0] if chain else None

    def max_parallel_lines(self) -> int:
        """How many lines may translate at once.

        Only a paid keyed engine benefits from a pool. Google's free endpoint
        throttles parallel callers (parallel requests are what earns a 429), and
        local engines run one generation at a time, so they stay serial — Auto
        included, whatever its first engine happens to be. A custom endpoint on
        this machine/LAN stays serial too: four workers would fight the UI.
        """
        first = self.first_engine()
        from . import api_pool

        local_custom = False
        if first == 'custom':
            from . import custom_api

            local_custom = (
                api_pool.custom_endpoints_are_local()
                if api_pool.has_entries('custom')
                else custom_api.is_local()
            )
            limit = 1 if local_custom else 4
        else:
            limit = 4 if first in ONLINE_TRANSLATION_ENGINES else 1
        if first in ONLINE_TRANSLATION_ENGINES and not local_custom:
            # Several pool configurations are independent allowances, so the run
            # may go wider than one key allows — never past the ceiling the user
            # chose in the API Pool panel.
            entries = api_pool.usable_count(first)
            if entries >= 2:
                ceiling = int(api_pool.settings().get('max_parallel') or 4)
                limit = min(ceiling, max(limit, entries * 2))
        return limit

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
                'custom': (self._custom, 'Custom API'),
            }
            provider, label = online[self._preference]
            if provider is None:
                if self._preference == 'custom':
                    raise RuntimeError(
                        'Custom API is not configured. Set its base URL and model in '
                        'API Settings, or choose another engine.'
                    )
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
        if self._preference == 'qwen':
            from . import qwen_local

            if not qwen_local.is_installed():
                raise RuntimeError(
                    'Qwen3 4B is not downloaded. Install it in AI Models or choose another engine.'
                )
            problem = qwen_local.runtime_problem()
            if problem:
                raise RuntimeError(
                    f'Qwen3 4B cannot start on this machine: {problem}. '
                    'Reinstall the CPU llama.cpp build (or install the NVIDIA CUDA '
                    'runtime), or choose another engine. Nothing was translated, so '
                    'no online requests were made.'
                )
            if self._qwen is None:
                raise RuntimeError('Qwen3 4B could not be initialised. Choose another engine.')
        if self._preference == 'nllb':
            if self._nllb is None:
                raise RuntimeError('NLLB-200 is not downloaded. Install it in AI Models or choose another engine.')
            from . import nllb_local

            for language in (source_language, target_language):
                try:
                    nllb_local.nllb_code(language)
                except ValueError as error:
                    raise RuntimeError(f'{error}. Choose Google Translate or another engine.') from error

    def supports_batch(self) -> bool:
        """True when the engine that would run can translate a whole group at once."""
        for key, provider in self._chain():
            if _engine_parked(key):
                continue
            return bool(getattr(provider, 'supports_batch', lambda: False)())
        return False

    def translate_batch(
        self,
        lines: list[str],
        context: list[str] | None = None,
        glossary: str = '',
        *,
        source_language: str = 'auto',
        target_language: str = 'Khmer',
        tone: str = 'natural and consistent',
        usage: dict[str, int] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> list[str | None]:
        """Batch translation through the chain, skipping parked engines.

        Each engine gets the whole batch; an engine that raises (quota, network)
        parks itself when the message looks like rate limiting and the next one
        retries the same lines. A batch where every line came back empty counts
        as a failure so the fallback engine still gets its turn.
        """
        modes = {
            'qwen': 'qwen', 'nllb': 'nllb', 'argos': 'offline', 'google': 'online',
            'gemini': 'gemini', 'openai': 'openai', 'codecraft': 'codecraft',
            'custom': 'custom',
        }
        last_error: Exception | None = None
        for key, provider in self._chain():
            if _engine_parked(key):
                continue
            if not getattr(provider, 'supports_batch', lambda: False)():
                continue
            try:
                result = provider.translate_batch(
                    lines, context=context, glossary=glossary,
                    source_language=source_language, target_language=target_language,
                    tone=tone, usage=usage,
                    meta=meta,
                )
            except Exception as error:
                logger.info('%s batch translation failed (%s), trying the next provider.', key, error)
                if _looks_rate_limited(error):
                    _park_engine(key)
                last_error = error
                continue
            self._mode = modes[key]
            if any(item is not None for item in result):
                return result
            last_error = RuntimeError(f'{key} returned no usable translations')
        if last_error is not None:
            raise last_error
        raise RuntimeError('No translation provider is available for batches')

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        modes = {
            'qwen': 'qwen', 'nllb': 'nllb', 'argos': 'offline', 'google': 'online',
            'gemini': 'gemini', 'openai': 'openai', 'codecraft': 'codecraft',
            'custom': 'custom',
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
            'id': 'custom',
            'name': 'Custom API (translation)',
            'mode': 'CUSTOM API',
            'connected': CustomAPITranslationProvider.is_available(),
            'primary': False,
            'fallback': CustomAPITranslationProvider.is_available(),
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
