"""Keep on-disk state out of the developer's real data folder.

The translation cache is a SQLite file next to the project database. Without
this fixture a line cached by one test would answer another test's request (and
the developer's own `data` folder would collect test lines), so every test gets
its own throwaway cache folder.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_translation_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('DUBVIBE_TRANSLATION_CACHE_DIR', str(tmp_path / 'translation-cache'))
    yield
