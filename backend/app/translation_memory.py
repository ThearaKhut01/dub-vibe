"""Translation memory for subtitle lines that already passed validation.

The cache itself lives in :mod:`backend.app.translation_cache` (one SQLite file
next to the project database, same TTL and pruning rules, "validated
translations only"). This module owns the key:

    sha256(PROMPT_VERSION, model, src, tgt, glossary_hash, text)

Model in the key means switching engines cannot replay another model's wording;
glossary in the key means editing the glossary invalidates the lines it affects.
Turning it off is one env var (``DUBVIBE_TRANSLATION_CACHE=0``) for anyone who
wants every run to hit the API.
"""
from __future__ import annotations

import hashlib
import os

from . import translation_cache as cache

DEFAULT_TTL_DAYS = 7


def enabled() -> bool:
    """Translation memory is on unless explicitly switched off."""
    return str(os.environ.get('DUBVIBE_TRANSLATION_CACHE', '1')).strip().lower() not in {'0', 'false', 'off', 'no'}


def _normalise(text: str) -> str:
    return ' '.join(str(text or '').split())


def glossary_hash(glossary: str) -> str:
    """Short, stable digest of the glossary text (order-insensitive)."""
    lines = sorted(
        _normalise(line).lower()
        for line in str(glossary or '').splitlines()
        if _normalise(line)
    )
    return hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()[:16]


def cache_key(
    text: str,
    source_lang: str = '',
    target_lang: str = '',
    model: str = '',
    glossary: str = '',
) -> str:
    """The key described in the module docstring."""
    payload = '\x1f'.join((
        cache.PROMPT_VERSION,
        _normalise(model).lower() or 'any',
        cache._language(source_lang),
        cache._language(target_lang),
        glossary_hash(glossary),
        _normalise(text),
    ))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def stats() -> dict:
    """Entries/hits of the cache file (the panel shows them)."""
    try:
        return cache.stats()
    except Exception:
        return {'entries': 0, 'hits': 0, 'newest_at': None}


def clear() -> int:
    """Forget every remembered line; returns how many were removed."""
    try:
        return cache.clear()
    except Exception:
        return 0


def lookup(
    text: str,
    source_lang: str = '',
    target_lang: str = '',
    model: str = '',
    glossary: str = '',
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> str | None:
    """A previously validated translation for this exact text, or None."""
    if not enabled() or not _normalise(text):
        return None
    try:
        return cache.lookup(cache_key(text, source_lang, target_lang, model, glossary), ttl_days)
    except Exception:  # a broken cache must never fail a translation run
        return None


def remember(
    text: str,
    translation: str,
    source_lang: str = '',
    target_lang: str = '',
    model: str = '',
    glossary: str = '',
    provider: str = '',
) -> None:
    """Store a translation that already passed the script/length validators."""
    if not enabled() or not _normalise(translation):
        return
    try:
        cache.store(
            cache_key(text, source_lang, target_lang, model, glossary),
            translation=translation,
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            provider=provider,
            model=model,
        )
    except Exception:
        pass
