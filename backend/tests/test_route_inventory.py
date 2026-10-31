"""Guard against accidental route removal.

Earlier, a cleanup that deleted the Khmer MMS routes also removed the VoxCPM model routes,
because they sat in the same block. The AI Models panel then polled a 404. These tests pin the
routes the frontend depends on, so a removal fails the suite instead of breaking the UI.
"""
from __future__ import annotations

from backend.app import main


def _routes() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for route in main.app.routes:
        path = getattr(route, 'path', None)
        for method in getattr(route, 'methods', set()) or set():
            found.add((method, path))
    return found


REQUIRED = {
    ('GET', '/api/models/voxcpm'),
    ('POST', '/api/models/voxcpm/download'),
    ('DELETE', '/api/models/voxcpm'),
    ('GET', '/api/models/faster-whisper'),
    ('GET', '/api/models/nllb'),
    ('GET', '/api/models/demucs'),
    ('GET', '/api/models/qwen'),
    ('GET', '/api/models/cuda-pack'),
    ('GET', '/api/models/storage'),
    ('GET', '/api/providers/custom/config'),
    ('POST', '/api/providers/custom/config'),
    ('GET', '/api/tts/capabilities'),
    ('POST', '/api/tts/preview'),
    ('POST', '/api/process/tts'),
    ('POST', '/api/process/export'),
    ('POST', '/api/projects/{project_id}/segments/{segment_id}/tts'),
    ('POST', '/api/projects/{project_id}/segments'),
    ('GET', '/api/projects/{project_id}/segments/{segment_id}/audio'),
    ('GET', '/api/projects/{project_id}/segment-timing'),
    ('POST', '/api/projects/{project_id}/fix-overlaps'),
    ('POST', '/api/projects/{project_id}/fix-overlaps/apply'),
    ('POST', '/api/projects/{project_id}/fix-overlaps/undo'),
}


def test_frontend_dependent_routes_are_registered():
    missing = sorted(REQUIRED - _routes())
    assert not missing, f'routes the UI depends on are missing: {missing}'


def test_removed_khmer_mms_routes_stay_removed():
    removed = {
        ('GET', '/api/models/mms-khm'),
        ('GET', '/api/models/mms/{engine_id}'),
        ('POST', '/api/models/mms/{engine_id}/download'),
        ('DELETE', '/api/models/mms/{engine_id}'),
    }
    still_there = sorted(removed & _routes())
    assert not still_there, f'the Khmer MMS routes were meant to be removed: {still_there}'
