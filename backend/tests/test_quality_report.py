"""The pre-export quality report."""
from __future__ import annotations

import pytest


def _project(monkeypatch, **overrides) -> tuple:
    from backend.app import main

    project = {
        'id': 'qc', 'name': 'qc', 'mode': 'DUB', 'media_path': None,
        'duration': '00:00:10', 'target_language': 'Khmer',
        'segments': [
            {'id': 's1', 'start': 0.0, 'end': 1.5, 'text': 'One', 'translation': 'មួយ'},
            {'id': 's2', 'start': 1.5, 'end': 3.0, 'text': 'Two', 'translation': ''},
        ],
        'audio_segments': [],
    }
    project.update(overrides)
    main.projects[project['id']] = project
    return main, project['id']


def test_missing_translation_and_audio_are_reported(monkeypatch) -> None:
    main, project_id = _project(monkeypatch)
    try:
        report = main._quality_report(main.projects[project_id])
    finally:
        main.projects.pop(project_id, None)

    assert report['summary']['missing_translation'] == 1
    assert report['summary']['missing_audio'] == 1, 'the first line has no clip yet'
    assert report['problems']['missing_translation'][0]['id'] == 's2'


def test_a_stored_fit_overrun_is_reported(monkeypatch) -> None:
    audio = [{
        'id': 's1', 'start': 0.0, 'end': 1.5, 'path': '',
        'fit': {'clip_seconds': 2.0, 'slot_seconds': 1.5, 'overrun_seconds': 0.5, 'fits': False},
    }]
    main, project_id = _project(monkeypatch, segments=[
        {'id': 's1', 'start': 0.0, 'end': 1.5, 'text': 'One', 'translation': 'មួយ'},
    ], audio_segments=audio)
    try:
        report = main._quality_report(main.projects[project_id])
    finally:
        main.projects.pop(project_id, None)

    assert report['summary']['overruns'] == 1
    assert report['problems']['overruns'][0]['overrun'] == 0.5


def test_a_dense_line_is_flagged_as_too_long(monkeypatch) -> None:
    main, project_id = _project(monkeypatch, segments=[
        {'id': 's1', 'start': 0.0, 'end': 1.0, 'text': 'One',
         'translation': 'នេះជាប្រយោគវែងខ្លាំងណាស់សម្រាប់មួយវិនាទី'},
    ], audio_segments=[])
    try:
        report = main._quality_report(main.projects[project_id])
    finally:
        main.projects.pop(project_id, None)

    assert report['summary']['long_lines'] == 1
    assert report['problems']['long_lines'][0]['per_second'] > 14


def test_the_route_answers_for_a_project(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    main, project_id = _project(monkeypatch)
    try:
        response = TestClient(main.app).get(f'/api/projects/{project_id}/quality')
    finally:
        main.projects.pop(project_id, None)

    assert response.status_code == 200, response.text
    assert response.json()['summary']['segments'] == 2
