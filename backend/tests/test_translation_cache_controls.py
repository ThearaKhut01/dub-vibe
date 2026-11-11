"""Clear-cache endpoint and the 'translate again from scratch' flag."""
from __future__ import annotations

import json
import sqlite3

import pytest


def _project(tmp_path, monkeypatch, texts, translations):
    from backend.app import main

    monkeypatch.setattr(main, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(main, 'PROJECT_DB', tmp_path / 'data' / 'projects.sqlite3')
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(tmp_path / 'data' / 'projects.sqlite3') as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
    project_id = 'cache-control'
    main.projects[project_id] = {
        'id': project_id, 'name': 'cache', 'mode': 'DUB', 'media_path': None,
        'duration': '00:00:10', 'source_language': 'en', 'target_language': 'Khmer', 'glossary': '',
        'segments': [
            {'id': f's{i}', 'start': float(i), 'end': float(i) + 0.9, 'text': text,
             'translation': translations[i] if i < len(translations) else ''}
            for i, text in enumerate(texts)
        ],
    }
    return main, project_id


def _fake_translator(monkeypatch, calls, main):
    class FakeTranslator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_preferred_engine(self, *args, **kwargs) -> None:
            return None

        def first_engine(self) -> str:
            return 'gemini_web'

        def supports_batch(self) -> bool:
            return True

        def translate_batch(self, lines, **kwargs):
            calls.append(list(lines))
            return [f'NEW-{text}' for text in lines]

        def translate(self, text, target_language, source_language='en'):
            calls.append([text])
            return f'NEW-{text}'

    monkeypatch.setattr(main, 'HybridTranslationProvider', FakeTranslator)


def test_saved_translations_are_kept_without_the_flag(monkeypatch, tmp_path) -> None:
    # Valid Khmer translations: resume must reuse them without any engine call.
    main, project_id = _project(tmp_path, monkeypatch, ['Yes.', 'No.'], ['បាទ', 'ទេ'])
    calls: list[list[str]] = []
    _fake_translator(monkeypatch, calls, main)
    try:
        from fastapi.testclient import TestClient

        response = TestClient(main.app).post('/api/process/translate', json={'project_id': project_id})
        assert response.status_code == 200, response.text
    finally:
        main.projects.pop(project_id, None)

    assert calls == [], 'nothing to translate: the saved lines are reused'


def test_the_retranslate_flag_sends_every_line_again(monkeypatch, tmp_path) -> None:
    main, project_id = _project(tmp_path, monkeypatch, ['Yes.', 'No.'], ['បាទ', 'ទេ'])
    calls: list[list[str]] = []
    _fake_translator(monkeypatch, calls, main)
    try:
        from fastapi.testclient import TestClient

        response = TestClient(main.app).post(
            '/api/process/translate',
            json={'project_id': project_id, 'retranslate': True},
        )
        assert response.status_code == 200, response.text
        segments = main.projects[project_id]['segments']
    finally:
        main.projects.pop(project_id, None)

    assert calls == [['Yes.', 'No.']], 'both lines go to the engine again'
    assert [segment['translation'] for segment in segments] == ['NEW-Yes.', 'NEW-No.']


def test_the_cache_is_not_consulted_while_retranslating(monkeypatch, tmp_path) -> None:
    main, project_id = _project(tmp_path, monkeypatch, ['Yes.'], [''])
    calls: list[list[str]] = []
    _fake_translator(monkeypatch, calls, main)
    from backend.app import translation_memory

    looked_up: list[str] = []
    monkeypatch.setattr(translation_memory, 'lookup', lambda *args, **kwargs: looked_up.append('hit') or 'FROM-CACHE')
    try:
        from fastapi.testclient import TestClient

        response = TestClient(main.app).post(
            '/api/process/translate',
            json={'project_id': project_id, 'retranslate': True},
        )
        assert response.status_code == 200, response.text
        assert main.projects[project_id]['segments'][0]['translation'] == 'NEW-Yes.'
    finally:
        main.projects.pop(project_id, None)

    assert looked_up == [], 'the memory must not answer a fresh run'
    assert calls == [['Yes.']]


def test_the_cache_status_route_reports_the_memory(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from backend.app import main, translation_memory

    monkeypatch.setattr(translation_memory, 'stats', lambda: {'entries': 12, 'hits': 3, 'newest_at': 1.0})
    monkeypatch.setattr(translation_memory, 'enabled', lambda: True)

    body = TestClient(main.app).get('/api/translation-cache').json()

    assert body == {'entries': 12, 'hits': 3, 'newest_at': 1.0, 'enabled': True}


def test_clearing_the_cache_reports_what_it_removed(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from backend.app import main, translation_memory

    monkeypatch.setattr(translation_memory, 'clear', lambda: 7)
    monkeypatch.setattr(translation_memory, 'stats', lambda: {'entries': 0, 'hits': 0, 'newest_at': None})

    body = TestClient(main.app).post('/api/translation-cache/clear').json()

    assert body['status'] == 'ok'
    assert body['removed'] == 7
    assert body['entries'] == 0
