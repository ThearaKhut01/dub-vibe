"""The route reports an interrupted run as such, not as an empty message."""
from __future__ import annotations

from concurrent.futures import CancelledError


def test_cancelled_runs_are_explained() -> None:
    from backend.app import main

    text = main._error_text(CancelledError())

    assert 'interrupted' in text
    assert 'Press' in text or 'press' in text
    assert text.strip() != ''


def test_a_silent_exception_reports_its_type() -> None:
    from backend.app import main

    assert main._error_text(Exception()) == 'Exception (no message)'


def test_a_normal_message_is_untouched() -> None:
    from backend.app import main

    assert main._error_text(RuntimeError('boom')) == 'boom'
