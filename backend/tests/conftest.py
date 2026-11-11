"""Keep on-disk state out of the developer's real data folder.

The translation cache and the project database are SQLite files in `data/`.
Without these fixtures a line cached by one test would answer another test's
request, and every test project would be saved into the developer's own
`projects.sqlite3`, so each test gets throwaway files.
"""
from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture(autouse=True)
def _isolated_translation_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('DUBVIBE_TRANSLATION_CACHE_DIR', str(tmp_path / 'translation-cache'))
    yield


@pytest.fixture(autouse=True)
def _isolated_project_database(tmp_path, monkeypatch):
    """Never write test projects into the developer's real projects.sqlite3."""
    from backend.app import main

    db_path = tmp_path / 'projects.sqlite3'
    monkeypatch.setattr(main, 'PROJECT_DB', db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL)'
        )
    main.projects.clear()
    yield
    main.projects.clear()

