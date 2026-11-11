"""Chrome profile detection: metadata-only scanning, lock handling, selection.

The fixtures are fake Chrome user-data roots, so these tests never touch the
real ``%LOCALAPPDATA%\\Google\\Chrome`` folder and never need a browser.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from backend.app import chrome_profiles as cp


@pytest.fixture(autouse=True)
def _no_real_chrome_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scans must not depend on the machine's process table (psutil is slow there).

    Tests that simulate Chrome override this inside the test itself.
    """
    monkeypatch.setattr(cp, 'chrome_processes', lambda: [])


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture()
def chrome_root(tmp_path: Path) -> Path:
    """A root shaped like Chrome's: Local State + Default / Profile 1 / Profile 2."""
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {
        'profile': {
            'last_used': 'Profile 1',
            'info_cache': {
                'Default': {'name': 'Personal', 'user_name': 'person@example.com', 'gaia_id': '111'},
                'Profile 1': {'name': 'Work', 'user_name': 'worker@corp.example.com', 'gaia_id': '222'},
                'Profile 2': {'name': 'Side', 'user_name': '', 'gaia_id': ''},
            },
        },
    })
    _write_json(root / 'Default' / 'Preferences', {
        'account_info': [{'email': 'person@example.com', 'full_name': 'Person'}],
    })
    # Current Chrome writes account_info: null and keeps the e-mail in Local State.
    _write_json(root / 'Profile 1' / 'Preferences', {'account_info': None})
    _write_json(root / 'Profile 2' / 'Preferences', {'account_info': []})
    # Chrome's own internal folder: never a user profile.
    (root / 'System Profile').mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chrome_root: Path) -> cp.ChromeProfileManager:
    """A manager pointed at the fake root, with a temporary config file."""
    monkeypatch.setattr(cp, 'scan_profiles', lambda errors=None: cp.scan_root(chrome_root, errors))
    monkeypatch.setattr(cp, 'chrome_processes', lambda: [])
    return cp.ChromeProfileManager(config_path=tmp_path / 'chrome_profile_config.json',
                                   sessions_dir=tmp_path / 'sessions')


def test_scan_lists_only_real_profile_folders(chrome_root: Path) -> None:
    profiles = cp.scan_profiles([chrome_root])
    assert [profile['profile_folder'] for profile in profiles] == ['Default', 'Profile 1', 'Profile 2']


def test_scan_reads_names_and_emails(chrome_root: Path) -> None:
    by_folder = {profile['profile_folder']: profile for profile in cp.scan_profiles([chrome_root])}

    assert by_folder['Default']['display_name'] == 'Personal'
    assert by_folder['Default']['emails'] == ['person@example.com']
    assert by_folder['Default']['has_google_account'] is True

    # Preferences carries no account_info: the e-mail comes from Local State.
    assert by_folder['Profile 1']['emails'] == ['worker@corp.example.com']
    assert by_folder['Profile 1']['has_google_account'] is True

    assert by_folder['Profile 2']['emails'] == []
    assert by_folder['Profile 2']['has_google_account'] is False


def test_scan_reports_the_required_shape(chrome_root: Path) -> None:
    profile = cp.scan_profiles([chrome_root])[0]
    for key in ('id', 'root_path', 'profile_folder', 'display_name', 'emails', 'path',
                'has_google_account', 'is_locked'):
        assert key in profile
    assert profile['path'] == str(chrome_root / profile['profile_folder'])
    assert Path(profile['path']).name == profile['profile_folder']


def test_profile_id_is_stable_and_unique(chrome_root: Path) -> None:
    first = {profile['profile_folder']: profile['id'] for profile in cp.scan_profiles([chrome_root])}
    second = {profile['profile_folder']: profile['id'] for profile in cp.scan_profiles([chrome_root])}

    assert first == second
    assert len(set(first.values())) == len(first)
    # Windows paths are case-insensitive: the same profile keeps its id.
    assert cp.profile_id(str(chrome_root).upper(), 'Default') == cp.profile_id(str(chrome_root), 'Default')


def test_broken_preferences_is_reported_not_raised(tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Kept'}}}})
    (root / 'Default').mkdir(parents=True)
    (root / 'Default' / 'Preferences').write_text('{ this is not json', encoding='utf-8')

    errors: list[str] = []
    profiles = cp.scan_profiles([root], errors)

    assert [profile['profile_folder'] for profile in profiles] == ['Default']
    assert any('Preferences' in message for message in errors)


def test_missing_local_state_still_lists_profiles(tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Default' / 'Preferences', {'account_info': [{'email': 'only@prefs.example'}]})

    errors: list[str] = []
    profiles = cp.scan_profiles([root], errors)

    assert [profile['profile_folder'] for profile in profiles] == ['Default']
    assert profiles[0]['emails'] == ['only@prefs.example']
    assert any('Local State' in message for message in errors)


def test_empty_folder_without_metadata_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {}}})
    (root / 'Profile 3').mkdir(parents=True)
    assert cp.scan_profiles([root]) == []


def test_lock_state_follows_the_running_profile(monkeypatch: pytest.MonkeyPatch, chrome_root: Path) -> None:
    # A process with an explicit --user-data-dir pointing at our root owns Profile 1.
    monkeypatch.setattr(cp, 'chrome_processes', lambda: [
        (4242, f'"chrome.exe" --user-data-dir="{chrome_root}" --profile-directory=Profile 1'),
    ])
    by_folder = {profile['profile_folder']: profile for profile in cp.scan_profiles([chrome_root])}

    assert by_folder['Profile 1']['is_locked'] is True
    assert by_folder['Profile 1']['locked_reason'] == 'in_use'
    assert by_folder['Profile 2']['is_locked'] is False


def test_lock_state_ignores_other_user_data_roots(monkeypatch: pytest.MonkeyPatch,
                                                  chrome_root: Path, tmp_path: Path) -> None:
    other = tmp_path / 'Some Automation Profile'
    other.mkdir()
    monkeypatch.setattr(cp, 'chrome_processes', lambda: [
        (7, f'chrome.exe --user-data-dir="{other}" --profile-directory=Default'),
    ])
    profiles = cp.scan_profiles([chrome_root])
    assert all(profile['is_locked'] is False for profile in profiles)


def test_lock_state_is_conservative_without_command_lines(monkeypatch: pytest.MonkeyPatch,
                                                          chrome_root: Path) -> None:
    # tasklist gives PIDs only: Chrome is running, the profile cannot be told.
    monkeypatch.setattr(cp, 'chrome_processes', lambda: [(1, '')])
    profiles = cp.scan_profiles([chrome_root])

    assert all(profile['is_locked'] for profile in profiles)
    assert all(profile['locked_reason'] == 'chrome_running' for profile in profiles)


def test_flag_value_handles_every_chrome_command_line_shape() -> None:
    root = r'C:\Users\me\AppData\Local\Google\Chrome\User Data'
    # Chrome quotes the whole --flag=value token when the value has spaces.
    assert cp._flag_value(f'"chrome.exe" "--user-data-dir={root}" /prefetch:1', '--user-data-dir') == root
    # ... and quotes only the value in other builds.
    assert cp._flag_value(f'chrome.exe --user-data-dir="{root}"', '--user-data-dir') == root
    # Two-token form.
    assert cp._flag_value(f'chrome.exe --profile-directory "Profile 2"', '--profile-directory') == 'Profile 2'
    # Chrome's child processes leave the value bare even with a space in it.
    assert cp._flag_value(
        r'chrome.exe --type=renderer --user-data-dir=C:\a\User Data --lang=en-US',
        '--user-data-dir',
    ) == r'C:\a\User Data'
    assert cp._flag_value(
        r'chrome.exe --profile-directory=Default /prefetch:1',
        '--profile-directory',
    ) == 'Default'
    # Plain form (a bare two-word value still cannot be told from two arguments).
    assert cp._flag_value('chrome.exe', '--profile-directory') == ''
    # ... while the profile-folder reader still sees the whole folder name.
    assert cp._profile_folder('chrome.exe --profile-directory=Profile 1') == 'Profile 1'
    assert cp._profile_folder('chrome.exe --profile-directory="Profile 2"') == 'Profile 2'
    assert cp._profile_folder('chrome.exe --profile-directory=Default /prefetch:1') == 'Default'
    assert cp._profile_folder('chrome.exe') == ''


def test_config_stores_only_the_selection(tmp_path: Path) -> None:
    path = tmp_path / 'chrome_profile_config.json'
    config = cp.ChromeProfileConfig(path)
    config.write(
        selected_profile_id='abc123',
        last_selected_profile_folder='Profile 1',
        last_selected_root_path=r'C:\Users\me\AppData\Local\Google\Chrome\User Data',
        headless=True,
        # Anything else — a password, a cookie, a token — must not be written.
        password='hunter2',
    )

    stored = json.loads(path.read_text(encoding='utf-8'))
    assert stored == {
        'selected_profile_id': 'abc123',
        'last_selected_profile_folder': 'Profile 1',
        'last_selected_root_path': r'C:\Users\me\AppData\Local\Google\Chrome\User Data',
        'headless': True,
        'shared_window': False,
    }
    assert config.read()['selected_profile_id'] == 'abc123'


def test_select_then_read_selection(manager: cp.ChromeProfileManager) -> None:
    profile = manager.list_profiles()[1]
    manager.select_profile(profile['id'])

    selected = manager.get_selected_profile()
    assert selected is not None
    assert selected['id'] == profile['id']
    assert manager.config.read()['last_selected_profile_folder'] == profile['profile_folder']


def test_select_unknown_profile_is_an_error(manager: cp.ChromeProfileManager) -> None:
    with pytest.raises(cp.ChromeProfileError):
        manager.select_profile('not-a-profile')


def test_validate_unknown_profile_reports_error(manager: cp.ChromeProfileManager) -> None:
    result = manager.validate_profile('missing')
    assert result['success'] is False
    assert result['status'] == 'error'
    assert 'no longer available' in result['message']


def test_chrome_using_the_source_profile_does_not_block(monkeypatch: pytest.MonkeyPatch,
                                                        chrome_root: Path, tmp_path: Path) -> None:
    """The tool works on its own copy, so an open Chrome window is not an error."""
    monkeypatch.setattr(cp, 'scan_profiles', lambda errors=None: cp.scan_root(chrome_root, errors))
    monkeypatch.setattr(cp, 'chrome_processes', lambda: [
        (1, f'chrome.exe --user-data-dir="{chrome_root}" --profile-directory=Default'),
    ])
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = profile_manager.list_profiles()[0]
    assert profile['is_locked'] is True

    profile_manager._ensure_usable(profile)  # must not raise


def test_the_tool_window_makes_the_profile_busy(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Copy'}}}})
    _write_json(root / 'Default' / 'Preferences', {'account_info': []})
    monkeypatch.setattr(cp, 'scan_profiles', lambda errors=None: cp.scan_root(root, errors))
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = profile_manager.list_profiles()[0]
    user_data = profile_manager.session_profile(profile)
    monkeypatch.setattr(cp, 'chrome_processes', lambda: [
        (99, f'chrome.exe "--user-data-dir={user_data}" --profile-directory=Default'),
    ])

    result = profile_manager.validate_profile(profile['id'])

    assert result['status'] == 'locked'
    assert "tool's Chrome window" in result['message']


def test_validate_without_playwright_explains_what_to_install(manager: cp.ChromeProfileManager,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp, 'playwright_available', lambda: False)
    profile = manager.list_profiles()[0]

    result = manager.validate_profile(profile['id'])

    assert result['status'] == 'error'
    assert 'Playwright' in result['message']


def test_launch_without_a_selection_is_an_error(manager: cp.ChromeProfileManager,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp, 'playwright_available', lambda: True)
    with pytest.raises(cp.ChromeProfileError):
        manager.launch_selected_profile()


def test_open_account_chooser_uses_googles_picker(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Copy'}}}})
    _write_json(root / 'Default' / 'Preferences', {'account_info': []})
    monkeypatch.setattr(cp, 'scan_profiles', lambda errors=None: cp.scan_root(root, errors))
    fake_chrome = tmp_path / 'chrome.exe'
    fake_chrome.write_text('', encoding='utf-8')
    monkeypatch.setattr(cp, 'chrome_executable', lambda: fake_chrome)
    started: list[list[str]] = []

    def fake_popen(command, **_kwargs):
        started.append(command)
        return object()

    monkeypatch.setattr(cp.subprocess, 'Popen', fake_popen)
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = profile_manager.list_profiles()[0]

    result = profile_manager.open_account_chooser(profile['id'])

    assert result['success'] is True
    assert result['status'] == 'account_chooser'
    assert 'Use another account' in result['message']
    assert started[0][-1] == cp.ACCOUNT_CHOOSER_URL
    assert 'accounts.google.com/AccountChooser' in cp.ACCOUNT_CHOOSER_URL


def test_open_manual_reports_a_missing_chrome(manager: cp.ChromeProfileManager,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp, 'chrome_executable', lambda: None)
    profile = manager.list_profiles()[0]

    result = manager.open_manual(profile['id'])

    assert result['success'] is False
    assert 'Chrome was not found' in result['message']


def test_open_manual_builds_a_profile_scoped_command(manager: cp.ChromeProfileManager,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    fake_chrome = tmp_path / 'chrome.exe'
    fake_chrome.write_text('', encoding='utf-8')
    monkeypatch.setattr(cp, 'chrome_executable', lambda: fake_chrome)
    started: list[list[str]] = []

    class FakeProcess:
        pass

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        started.append(command)
        return FakeProcess()

    monkeypatch.setattr(cp.subprocess, 'Popen', fake_popen)
    profile = manager.list_profiles()[1]

    result = manager.open_manual(profile['id'])

    assert result['success'] is True
    assert started, 'Chrome was never started'
    command = started[0]
    assert command[0] == str(fake_chrome)
    assert f'--profile-directory={profile["profile_folder"]}' in command
    assert cp.GEMINI_URL in command


def test_the_shared_window_flag_is_stored(tmp_path: Path) -> None:
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    assert profile_manager.shared_window_enabled() is False

    profile_manager.set_shared_window(True)

    assert profile_manager.shared_window_enabled() is True
    assert json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))['shared_window'] is True


def test_ensure_window_launches_chrome_with_a_debug_port(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Copy'}}}})
    _write_json(root / 'Default' / 'Preferences', {'account_info': []})
    monkeypatch.setattr(cp, 'scan_profiles', lambda errors=None: cp.scan_root(root, errors))
    fake_chrome = tmp_path / 'chrome.exe'
    fake_chrome.write_text('', encoding='utf-8')
    monkeypatch.setattr(cp, 'chrome_executable', lambda: fake_chrome)
    monkeypatch.setattr(cp, '_port_free', lambda port: port == 9335)
    monkeypatch.setattr(cp, '_wait_for_devtools', lambda port, timeout=None: True)
    started: list[list[str]] = []

    def fake_popen(command, **_kwargs):
        started.append(command)
        return object()

    monkeypatch.setattr(cp.subprocess, 'Popen', fake_popen)
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = profile_manager.list_profiles()[0]

    port = profile_manager.ensure_window(profile)

    assert port == 9335
    assert '--remote-debugging-port=9335' in started[0]
    assert started[0][-1] == 'about:blank'
    marker = json.loads((tmp_path / 'sessions' / profile['id'] / 'window.json').read_text(encoding='utf-8'))
    assert marker['port'] == 9335


def test_shared_mode_attaches_to_the_open_window(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Copy'}}}})
    _write_json(root / 'Default' / 'Preferences', {'account_info': []})
    monkeypatch.setattr(cp, 'scan_profiles', lambda errors=None: cp.scan_root(root, errors))
    monkeypatch.setattr(cp, 'playwright_available', lambda: True)
    (tmp_path / 'chrome.exe').write_text('', encoding='utf-8')
    monkeypatch.setattr(cp, 'chrome_executable', lambda: tmp_path / 'chrome.exe')
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile_manager.set_shared_window(True)
    profile = profile_manager.list_profiles()[0]
    marker = tmp_path / 'sessions' / profile['id'] / 'window.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'port': 9333}), encoding='utf-8')

    attached: list[int] = []

    class FakeSession:
        page = object()

        def close(self) -> None:
            return None

    monkeypatch.setattr(cp, 'chrome_processes', lambda: [
        (7, f'chrome.exe "--user-data-dir={profile_manager.session_dir(profile)}"'),
    ])
    monkeypatch.setattr(cp, '_devtools_ready', lambda port: port == 9333)

    def fake_attach(target_profile, port):
        attached.append(port)
        return FakeSession()

    monkeypatch.setattr(profile_manager, '_attach_shared', fake_attach)

    session = profile_manager.launch_profile(profile['id'])

    assert attached == [9333], 'the open window must be attached to, not relaunched'
    assert session.page is not None


def test_session_profile_is_seeded_once_and_then_reused(tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Copy'}}}})
    profile_dir = root / 'Default'
    _write_json(profile_dir / 'Preferences', {'account_info': []})
    (profile_dir / 'Cookies').write_text('chrome cookies live here', encoding='utf-8')
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = {'id': 'fake', 'root_path': str(root), 'profile_folder': 'Default', 'path': str(profile_dir)}

    first = profile_manager.session_profile(profile)
    assert profile_manager.has_session_profile(profile) is True
    # A sign-in written into the tool's profile must survive the next run.
    (first / 'Default' / 'signed-in.txt').write_text('session', encoding='utf-8')

    second = profile_manager.session_profile(profile)

    assert second == first
    assert (second / 'Default' / 'signed-in.txt').is_file()


def test_the_tool_profile_does_not_offer_the_other_chrome_profiles(tmp_path: Path) -> None:
    """Chrome's profile menu comes from info_cache: only the copied folder may stay."""
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {
        'profile': {
            'last_used': 'Default',
            'info_cache': {
                'Default': {'name': 'Main', 'user_name': 'main@example.com'},
                'Profile 1': {'name': 'Work', 'user_name': 'work@example.com'},
            },
        },
    })
    _write_json(root / 'Profile 1' / 'Preferences', {'account_info': []})
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = {'id': 'fake', 'root_path': str(root), 'profile_folder': 'Profile 1',
               'path': str(root / 'Profile 1')}

    user_data = profile_manager.session_profile(profile)

    import json as _json
    state = _json.loads((user_data / 'Local State').read_text(encoding='utf-8'))
    assert list(state['profile']['info_cache']) == ['Profile 1']
    assert state['profile']['last_used'] == 'Profile 1'
    assert state['profile']['info_cache']['Profile 1']['name'] == 'Work'


def test_resetting_the_session_profile_removes_it(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Copy'}}}})
    _write_json(root / 'Default' / 'Preferences', {'account_info': []})
    monkeypatch.setattr(cp, 'scan_profiles', lambda errors=None: cp.scan_root(root, errors))
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = profile_manager.list_profiles()[0]
    profile_manager.session_profile(profile)
    assert profile_manager.has_session_profile(profile) is True

    profile_manager.reset_session_profile(profile['id'])

    assert profile_manager.has_session_profile(profile) is False


@pytest.mark.skipif(sys.platform != 'win32', reason='the selector loop only exists on Windows')
def test_playwright_gets_the_proactor_policy_and_the_previous_one_is_restored() -> None:
    """Playwright cannot spawn its driver on a Windows selector loop."""
    import asyncio

    previous = asyncio.get_event_loop_policy()
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        with cp._playwright_loop_policy():
            inside = asyncio.get_event_loop_policy()
        after = asyncio.get_event_loop_policy()
    finally:
        asyncio.set_event_loop_policy(previous)

    assert isinstance(inside, asyncio.WindowsProactorEventLoopPolicy)
    assert isinstance(after, asyncio.WindowsSelectorEventLoopPolicy)


def test_attach_shared_names_the_event_loop_when_the_driver_cannot_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A bare NotImplementedError from Playwright must not reach the UI as-is."""
    import sys
    import types

    fake = types.ModuleType('playwright.sync_api')

    def start() -> None:
        raise NotImplementedError

    fake.sync_playwright = start
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', fake)
    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')

    with pytest.raises(cp.ChromeProfileError) as info:
        profile_manager._attach_shared({'id': 'abc', 'profile_folder': 'Default'}, 9333)

    assert 'event loop' in str(info.value)


class _Node:
    def __init__(self, box: dict | None = None) -> None:
        self._box = box

    def bounding_box(self) -> dict | None:
        return self._box

    def is_visible(self) -> bool:
        return self._box is not None


class _Nodes:
    def __init__(self, boxes: list[dict | None]) -> None:
        self._boxes = boxes

    def count(self) -> int:
        return len(self._boxes)

    def nth(self, index: int) -> _Node:
        return _Node(self._boxes[index])

    @property
    def first(self) -> _Node:
        return _Node(self._boxes[0])


class _FakePage:
    """Just enough page for gemini_state: url, locator counts, button boxes."""

    def __init__(self, url: str = cp.GEMINI_URL, markers: tuple[str, ...] = (),
                 sign_in_boxes: list[dict | None] | None = None) -> None:
        self.url = url
        self._markers = set(markers)
        self._sign_in_boxes = sign_in_boxes or []

    def goto(self, url: str, **_kwargs: object) -> None:
        # A real navigation would replace the page; the fake keeps its fixtures.
        self.navigated_to = url

    def wait_for_timeout(self, _milliseconds: int) -> None:
        return None

    def locator(self, selector: str) -> _Nodes:
        present = selector in self._markers
        return _Nodes([{'x': 0, 'y': 0, 'width': 10, 'height': 10}] if present else [])

    def get_by_role(self, _role: str, name: object = None) -> _Nodes:
        return _Nodes(self._sign_in_boxes)


HEADER_SIGN_IN = {'x': 1199, 'y': 14, 'width': 68, 'height': 36}
FOOTER_SIGN_IN = {'x': 16, 'y': 808, 'width': 258, 'height': 36}


def test_gemini_reports_ready_when_the_account_menu_is_present() -> None:
    # A session needs a visible account menu *and* a rendered composer: an
    # account node on a half-loaded shell is not a signed-in profile.
    page = _FakePage(markers=('a[href*="SignOutOptions"]', 'rich-textarea'))
    state, message, url = cp.gemini_state(page)
    assert state == 'ready'
    assert url == cp.GEMINI_URL
    assert 'signed in' in message


def test_gemini_reports_login_required_for_the_signed_out_header() -> None:
    page = _FakePage(sign_in_boxes=[FOOTER_SIGN_IN, HEADER_SIGN_IN])
    state, message, _url = cp.gemini_state(page)
    assert state == 'login_required'
    assert 'Open manual login' in message


def test_gemini_does_not_treat_a_footer_link_as_signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp, 'GEMINI_SETTLE_SECONDS', 0.3)
    page = _FakePage(sign_in_boxes=[FOOTER_SIGN_IN])
    state, _message, _url = cp.gemini_state(page)
    assert state == 'error'


def test_gemini_returns_error_quickly_when_the_page_is_unrecognised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp, 'GEMINI_SETTLE_SECONDS', 0.3)
    state, message, _url = cp.gemini_state(_FakePage())
    assert state == 'error'
    assert 'may have changed' in message


def test_gemini_reports_login_required_on_the_google_sign_in_page() -> None:
    page = _FakePage(url='https://accounts.google.com/v3/signin/identifier?continue=...')
    state, _message, url = cp.gemini_state(page)
    assert state == 'login_required'
    assert 'accounts.google.com' in url


def test_session_profile_contains_metadata_but_never_login_data(tmp_path: Path) -> None:
    root = tmp_path / 'User Data'
    _write_json(root / 'Local State', {'profile': {'info_cache': {'Default': {'name': 'Copy'}}}})
    profile_dir = root / 'Default'
    _write_json(profile_dir / 'Preferences', {'account_info': []})
    (profile_dir / 'Cookies').write_text('chrome cookies live here', encoding='utf-8')
    (profile_dir / 'Login Data').write_text('never copied', encoding='utf-8')

    profile_manager = cp.ChromeProfileManager(config_path=tmp_path / 'config.json',
                                              sessions_dir=tmp_path / 'sessions')
    profile = {
        'id': 'fake',
        'root_path': str(root),
        'profile_folder': 'Default',
        'path': str(profile_dir),
    }

    copy_root = profile_manager.session_profile(profile)

    assert (copy_root / 'Local State').is_file()
    assert (copy_root / 'Default' / 'Preferences').is_file()
    assert (copy_root / 'Default' / 'Cookies').is_file()
    assert not (copy_root / 'Default' / 'Login Data').exists()
