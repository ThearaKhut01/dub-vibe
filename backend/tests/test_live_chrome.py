"""The Live tab follows the Chrome profile, not API keys."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app import api_events, chrome_profiles, gemini_web


class _FakeManager:
    def __init__(self, profile_id: str) -> None:
        self.config = SimpleNamespace(read=lambda: {'selected_profile_id': profile_id})

    def get_selected_profile(self):  # pragma: no cover - only used by snapshot tests
        return None

    def launch_selected_profile(self, headless=None):
        raise chrome_profiles.ChromeProfileError('Chrome is using this profile.', 'locked')


@pytest.fixture(autouse=True)
def _clean_events():
    api_events.reset_for_tests()
    yield
    api_events.reset_for_tests()


def _profile() -> dict:
    return {
        'id': 'abc123',
        'profile_folder': 'Profile 5',
        'display_name': 'Theara',
        'emails': ['someone@example.com'],
        'is_locked': False,
    }


def test_live_snapshot_registers_only_the_selected_chrome_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import main

    monkeypatch.setattr(main.chrome_profiles, 'manager', lambda: SimpleNamespace(get_selected_profile=_profile))

    snapshot = main._live_snapshot()

    assert [node['id'] for node in snapshot['nodes']] == ['chrome:abc123']
    node = snapshot['nodes'][0]
    assert node['provider'] == 'chrome'
    assert node['label'] == 'Chrome · Profile 5'
    assert node['model'] == gemini_web.MODEL_LABEL
    # The Chrome profile's name is a description, never a credential.
    assert node['key_hint'] == 'Theara'
    assert node['requests'] == 0


def test_live_snapshot_is_empty_without_a_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import main

    monkeypatch.setattr(main.chrome_profiles, 'manager', lambda: SimpleNamespace(get_selected_profile=lambda: None))

    assert main._live_snapshot()['nodes'] == []


def test_live_snapshot_drops_a_previously_registered_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import main

    api_events.register_node('chrome:old', label='Chrome · Profile 1', provider='chrome')
    monkeypatch.setattr(main.chrome_profiles, 'manager', lambda: SimpleNamespace(get_selected_profile=_profile))

    nodes = main._live_snapshot()['nodes']

    assert [node['id'] for node in nodes] == ['chrome:abc123']


def test_a_failed_prompt_is_recorded_on_the_profile_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)

    automator = gemini_web.GeminiWebAutomator(_FakeManager('abc123'))
    result = automator.ask('hello')

    assert result['success'] is False
    assert result['status'] == 'locked'

    nodes = api_events.snapshot()['nodes']
    assert [node['id'] for node in nodes] == ['chrome:abc123']
    node = nodes[0]
    assert node['provider'] == 'chrome'
    assert node['requests'] == 1
    assert node['failures'] == 1
    assert 'using this profile' in node['last_error']


def test_an_empty_prompt_never_registers_a_node() -> None:
    with pytest.raises(gemini_web.GeminiWebError):
        gemini_web.GeminiWebAutomator(_FakeManager('abc123')).ask('   ')
    assert api_events.snapshot()['nodes'] == []
