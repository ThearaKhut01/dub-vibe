"""Detect installed Chrome profiles and reuse one already-signed-in session.

What this module does
---------------------
* Reads Chrome's own *metadata* files only: ``Local State`` (profile names and
  the account each profile is signed in with) and each profile's
  ``Preferences`` (``account_info``). Nothing else is opened.
* Reports a stable id, display name, masked-able e-mail list, and whether the
  profile is currently held open by Chrome.
* Opens the chosen profile with Playwright's persistent context, so the browser
  starts with the cookies the user already has.

What it deliberately does NOT do
--------------------------------
* No password, cookie, token or ``Login Data`` is read, decrypted or uploaded.
* No login is automated, no account is created and no CAPTCHA is touched. When
  Gemini asks for a sign-in, the answer is "sign in yourself".

Chrome 136 and later refuse ``--remote-debugging-pipe`` on the *default* data
directory (https://developer.chrome.com/blog/remote-debugging-port). For that
case (and only that case) the module makes a private working copy of the
profile's metadata + cookie store under the app's data folder, launches Chrome
there, and deletes the copy when the session closes. Chrome decrypts its own
cookies; this code never sees them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from backend.app.providers import get_storage_dir

GEMINI_URL = 'https://gemini.google.com/app'
CONFIG_FILENAME = 'chrome_profile_config.json'
SESSION_DIRNAME = 'chrome-profile-sessions'

# Chrome's own folder naming for profiles ("Default", "Profile 1", ...).
PROFILE_FOLDER_PATTERN = re.compile(r'^(?:Default|Profile \d+)$')

# The only config fields this feature stores. No password, cookie or token.
CONFIG_DEFAULTS: dict[str, Any] = {
    'selected_profile_id': '',
    'last_selected_profile_folder': '',
    'last_selected_root_path': '',
    'headless': False,
}

CHROME_EXECUTABLE_NAMES = ('chrome.exe', 'chrome-beta.exe', 'chrome-sxs.exe')
CHROME_PROCESS_NAMES = {'chrome.exe', 'chrome-beta.exe', 'chrome-sxs.exe', 'chrome'}

# Chrome refuses remote debugging on its default data directory from v136 on.
CHROME_REMOTE_DEBUG_MIN_BLOCK = 136

# Metadata + the cookie store Chrome itself reads. Never "Login Data",
# never "Web Data", never a password database.
SHADOW_PROFILE_FILES = (
    'Preferences',
    'Secure Preferences',
    'Cookies',
    'Network/Cookies',
)

PLAYWRIGHT_TIMEOUT_MS = 45_000
GEMINI_SETTLE_SECONDS = 20.0
LOCKED_MESSAGE = 'Chrome is currently using this profile. Close that Chrome window and try again.'


def locked_message(folder: str = '') -> str:
    """Only the window that runs *this* profile has to be closed.

    Chrome keeps one browser process (and usually one window) per profile, and a
    profile is locked only while that process holds it. Other profiles can stay
    open: the session uses either the profile directory itself or a private copy
    of it, never the whole Chrome install.
    """
    target = f' {folder}' if folder else ' this profile'
    return (
        f'Chrome is using{target} right now. Close the Chrome window that runs it '
        '(other profiles can stay open), then try again.'
    )
MISSING_PLAYWRIGHT_MESSAGE = (
    'Playwright is not installed in this Python environment. '
    'Run "pip install playwright" (no browser download is needed — the app drives your installed Chrome).'
)
MISSING_CHROME_MESSAGE = 'Google Chrome was not found. Install Chrome, or set DUBVIBE_CHROME_PATH to chrome.exe.'


class ChromeProfileError(RuntimeError):
    """A profile operation failed in a way the UI should show verbatim."""

    def __init__(self, message: str, status: str = 'error') -> None:
        super().__init__(message)
        self.status = status


# --- local paths -------------------------------------------------------------


def _local_app_data() -> Path:
    value = os.environ.get('LOCALAPPDATA')
    return Path(value) if value else Path.home() / 'AppData' / 'Local'


def chrome_root_candidates() -> list[tuple[str, Path]]:
    """Chrome user-data roots that may exist, as (channel, path)."""
    base = _local_app_data() / 'Google'
    return [
        ('stable', base / 'Chrome' / 'User Data'),
        ('beta', base / 'Chrome Beta' / 'User Data'),
        ('canary', base / 'Chrome SxS' / 'User Data'),
    ]


def default_chrome_root() -> Path:
    return chrome_root_candidates()[0][1]


def chrome_roots(extra: Iterable[Path | str] = ()) -> list[Path]:
    """Every Chrome user-data root that exists on this machine, de-duplicated."""
    found: list[Path] = []
    seen: set[str] = set()
    for _channel, path in chrome_root_candidates():
        if path.is_dir() and _key(path) not in seen:
            seen.add(_key(path))
            found.append(path)
    for raw in extra:
        path = Path(raw)
        if path.is_dir() and _key(path) not in seen:
            seen.add(_key(path))
            found.append(path)
    return found


def chrome_channel(root: Path) -> str:
    for channel, path in chrome_root_candidates():
        if _key(path) == _key(root):
            return channel
    return 'custom'


def chrome_executable() -> Path | None:
    """chrome.exe for the stable build, or DUBVIBE_CHROME_PATH when set."""
    override = os.environ.get('DUBVIBE_CHROME_PATH')
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    roots = [
        Path(os.environ.get('PROGRAMFILES', r'C:\Program Files')) / 'Google' / 'Chrome' / 'Application',
        Path(os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')) / 'Google' / 'Chrome' / 'Application',
        _local_app_data() / 'Google' / 'Chrome' / 'Application',
    ]
    for folder in roots:
        candidate = folder / 'chrome.exe'
        if candidate.is_file():
            return candidate
    return None


def chrome_major_version(root: Path) -> int | None:
    """Chrome's own "Last Version" file, e.g. 155.0.8059.84 -> 155."""
    try:
        text = (root / 'Last Version').read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return None
    head = text.split('.', 1)[0]
    return int(head) if head.isdigit() else None


# --- JSON helpers ------------------------------------------------------------


def _key(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(os.path.realpath(str(path))))


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read one Chrome metadata file. A broken file is reported, never raised."""
    try:
        with path.open('r', encoding='utf-8', errors='replace') as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None, f'{path.name} is missing'
    except (OSError, ValueError) as error:
        return None, f'{path.name} could not be read ({error.__class__.__name__})'
    if not isinstance(data, dict):
        return None, f'{path.name} does not contain an object'
    return data, None


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ''


def _clean_email(value: Any) -> str:
    text = _clean_text(value)
    # Cheap sanity check: an e-mail has one "@" and no spaces. Chrome stores the
    # signed-in account there; anything else is metadata noise we drop.
    if not text or ' ' in text or text.count('@') != 1:
        return ''
    return text


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        marker = value.casefold()
        if marker and marker not in seen:
            seen.add(marker)
            out.append(value)
    return out


# --- profile metadata --------------------------------------------------------


def profile_id(root: Path | str, folder: str) -> str:
    """Stable id from the root path + profile folder (case-insensitive on Windows)."""
    material = f'{_key(root)}\0{folder.strip()}'
    return hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]


def preferences_emails(preferences: dict[str, Any] | None) -> list[str]:
    """E-mails from ``Preferences -> account_info`` (a list or a mapping)."""
    if not isinstance(preferences, dict):
        return []
    account_info = preferences.get('account_info')
    entries: list[Any]
    if isinstance(account_info, list):
        entries = account_info
    elif isinstance(account_info, dict):
        entries = list(account_info.values())
    else:
        entries = []
    emails: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            emails.append(_clean_email(entry.get('email')))
    return _dedupe(email for email in emails if email)


def _profile_dirs(root: Path) -> list[Path]:
    try:
        entries = [entry for entry in root.iterdir() if entry.is_dir()]
    except OSError:
        return []
    found = [entry for entry in entries if PROFILE_FOLDER_PATTERN.match(entry.name)]
    return sorted(found, key=lambda entry: (entry.name != 'Default', _folder_number(entry.name), entry.name))


def _folder_number(name: str) -> int:
    digits = re.sub(r'\D', '', name)
    return int(digits) if digits else 0


def scan_root(root: Path, errors: list[str] | None = None) -> list[dict[str, Any]]:
    """Every Chrome profile inside one user-data root.

    ``errors`` collects human-readable warnings (missing/broken metadata) so the
    caller can show them without losing the profiles that did parse.
    """
    warnings_sink = errors if errors is not None else []
    local_state, local_error = _read_json(root / 'Local State')
    if local_error:
        warnings_sink.append(f'{root}: {local_error}')

    cache = {}
    last_used = ''
    if isinstance(local_state, dict):
        profile_section = local_state.get('profile')
        if isinstance(profile_section, dict):
            cache = profile_section.get('info_cache') if isinstance(profile_section.get('info_cache'), dict) else {}
            last_used = _clean_text(profile_section.get('last_used'))

    running = running_profile_folders(root)
    profiles: list[dict[str, Any]] = []
    for folder_dir in _profile_dirs(root):
        folder = folder_dir.name
        preferences, pref_error = _read_json(folder_dir / 'Preferences')
        cache_entry = cache.get(folder) if isinstance(cache.get(folder), dict) else {}
        if preferences is None and not cache_entry:
            # A folder Chrome never finished creating: nothing to show.
            if pref_error:
                warnings_sink.append(f'{folder}: {pref_error} (profile skipped)')
            continue
        if pref_error:
            warnings_sink.append(f'{folder}: {pref_error}')

        emails = _dedupe(
            list(preferences_emails(preferences))
            + [_clean_email(cache_entry.get('user_name'))]
        )
        gaia_id = _clean_text(cache_entry.get('gaia_id'))
        display_name = _clean_text(cache_entry.get('name'))
        if not display_name and isinstance(preferences, dict):
            profile_section = preferences.get('profile')
            if isinstance(profile_section, dict):
                display_name = _clean_text(profile_section.get('name'))
        locked, reason = _lock_state(folder, running)

        profiles.append({
            'id': profile_id(root, folder),
            'root_path': str(root),
            'root_channel': chrome_channel(root),
            'profile_folder': folder,
            'display_name': display_name or folder,
            'emails': emails,
            'path': str(folder_dir),
            # A Gaia id or a signed-in e-mail is what "has a Google account"
            # means here; both come from Chrome's own metadata.
            'has_google_account': bool(emails) or bool(gaia_id),
            'is_locked': locked,
            'locked_reason': reason,
            'is_last_used': bool(last_used) and folder == last_used,
            'warnings': [pref_error] if pref_error else [],
        })
    return profiles


def scan_profiles(roots: Iterable[Path | str] | None = None, errors: list[str] | None = None) -> list[dict[str, Any]]:
    """Scan every Chrome user-data root (or the ones passed in)."""
    if roots is None:
        resolved = chrome_roots()
        if not resolved:
            if errors is not None:
                errors.append('No Chrome user-data folder was found on this PC.')
            return []
    else:
        resolved = [Path(root) for root in roots]
    profiles: list[dict[str, Any]] = []
    for root in resolved:
        profiles.extend(scan_root(root, errors))
    return profiles


# --- Chrome process / lock detection ----------------------------------------


def _split_command_line(command_line: str) -> list[str]:
    """Split a Windows command line into arguments, honouring quotes."""
    tokens: list[str] = []
    current: list[str] = []
    quote = ''
    for character in command_line:
        if quote:
            if character == quote:
                quote = ''
            else:
                current.append(character)
        elif character in ('"', "'"):
            quote = character
        elif character.isspace():
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(character)
    if current:
        tokens.append(''.join(current))
    return tokens


def _flag_value(command_line: str, flag: str) -> str:
    """Value of ``--flag=value``, ``--flag value`` or ``"--flag=value with spaces"``.

    Chrome is inconsistent: the browser process quotes a path with spaces, child
    processes leave it bare (``--user-data-dir=<root> --type=renderer``),
    so the value runs until the next option rather than the next space.
    """
    pattern = re.compile(
        rf'{re.escape(flag)}(?:=|\s+)(?:"([^"]*)"|(.*?))(?=\s+[-/][-\w]|\s*$)',
    )
    match = pattern.search(command_line)
    if not match:
        return ''
    return (match.group(1) or match.group(2) or '').strip().strip('"')


# Chrome writes "Profile 1" either quoted or bare, so the folder name cannot
# always be recovered with a plain token split.
_PROFILE_FLAG = re.compile(
    r'--profile-directory=(?:"([^"]*)"|((?:Profile\s+\d+|Default)\b|\S+))',
    re.IGNORECASE,
)


def _profile_folder(command_line: str) -> str:
    match = _PROFILE_FLAG.search(command_line)
    if match:
        value = (match.group(1) or match.group(2) or '').strip()
        if value:
            return value
    return _flag_value(command_line, '--profile-directory')


def _tasklist_chrome() -> list[tuple[int, str]]:
    """Fallback probe: PIDs only (Windows ``tasklist``), command lines unknown."""
    try:
        completed = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq chrome.exe', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    found: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip('"') for part in line.split('","')]
        if len(parts) >= 2 and parts[0].lower().endswith('chrome.exe') and parts[1].isdigit():
            found.append((int(parts[1]), ''))
    return found


def chrome_processes() -> list[tuple[int, str]]:
    """(pid, command line) for every running Chrome process, command line possibly empty."""
    try:
        import psutil  # type: ignore
    except Exception:
        return _tasklist_chrome()
    found: list[tuple[int, str]] = []
    try:
        iterator = psutil.process_iter(['pid', 'name', 'cmdline'])
    except Exception:
        return _tasklist_chrome()
    for process in iterator:
        try:
            name = str((process.info or {}).get('name') or '').lower()
            if name not in CHROME_PROCESS_NAMES:
                continue
            command_line = ' '.join((process.info or {}).get('cmdline') or [])
            found.append((int(process.info.get('pid') or 0), command_line))
        except Exception:
            continue
    return found


def chrome_running() -> bool:
    return bool(chrome_processes())


def running_profile_folders(root: Path) -> set[str] | None:
    """Profile folders Chrome currently has open for ``root``.

    ``None`` means "Chrome is running but this probe cannot say which profile"
    (no psutil, only PIDs available) — callers treat that as "may be locked".
    """
    processes = chrome_processes()
    if not processes:
        return set()
    if not any(command_line for _pid, command_line in processes):
        return None
    root_key = _key(root)
    default_key = _key(default_chrome_root())
    folders: set[str] = set()
    for _pid, command_line in processes:
        user_data_dir = _flag_value(command_line, '--user-data-dir')
        # Chrome started without the flag uses the default root; a process with
        # an explicit (different) root — including our own working copies — must
        # not make real profiles look busy.
        if user_data_dir:
            if _key(user_data_dir) != root_key:
                continue
        elif root_key != default_key:
            continue
        folder = _profile_folder(command_line) or 'Default'
        folders.add(folder)
    return folders


def _lock_state(folder: str, running: set[str] | None) -> tuple[bool, str]:
    if running is None:
        return True, 'chrome_running'
    if folder in running:
        return True, 'in_use'
    return False, ''


def profile_locked(root: Path, folder: str) -> bool:
    return _lock_state(folder, running_profile_folders(root))[0]


# --- local config ------------------------------------------------------------


class ChromeProfileConfig:
    """Only the four non-secret fields the selection needs."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else get_storage_dir('data') / CONFIG_FILENAME
        self._lock = threading.RLock()

    def read(self) -> dict[str, Any]:
        state = dict(CONFIG_DEFAULTS)
        data, _error = _read_json(self.path)
        if isinstance(data, dict):
            for key, default in CONFIG_DEFAULTS.items():
                value = data.get(key)
                if isinstance(default, bool):
                    state[key] = bool(value) if isinstance(value, bool) else default
                elif isinstance(value, str):
                    state[key] = value
        return state

    def write(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            state = self.read()
            for key in CONFIG_DEFAULTS:
                if key in changes:
                    value = changes[key]
                    state[key] = bool(value) if isinstance(CONFIG_DEFAULTS[key], bool) else str(value or '')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + '.tmp')
            temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
            os.replace(temporary, self.path)
            return state


# --- Playwright --------------------------------------------------------------


def playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except Exception:
        return False
    return True


class ProfileSession:
    """One Playwright persistent context.

    Playwright's sync API is bound to the thread that opened it, so a session is
    opened, used and closed inside one call (see :meth:`ChromeProfileManager.validate_profile`).
    """

    def __init__(self, profile: dict[str, Any], context: Any, playwright: Any) -> None:
        self.profile = profile
        self.context = context
        self.playwright = playwright
        # The session profile is the tool's own, so a sign-in inside it survives.
        self.via = 'session_profile'

    @property
    def page(self) -> Any:
        pages = self.context.pages
        return pages[0] if pages else self.context.new_page()

    def close(self) -> None:
        try:
            self.context.close()
        except Exception:
            pass
        try:
            self.playwright.stop()
        except Exception:
            pass
        # The profile is intentionally kept: it holds the sign-in.

    def __enter__(self) -> 'ProfileSession':
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ChromeProfileManager:
    """Detection, selection and launching of Chrome profiles."""

    def __init__(self, config_path: Path | str | None = None,
                 sessions_dir: Path | str | None = None) -> None:
        self.config = ChromeProfileConfig(config_path)
        self.sessions_dir = Path(sessions_dir) if sessions_dir else get_storage_dir('data') / SESSION_DIRNAME
        self._lock = threading.RLock()
        self._session: ProfileSession | None = None
        self.last_scan_errors: list[str] = []

    # -- scanning -------------------------------------------------------------

    def refresh_profiles(self) -> list[dict[str, Any]]:
        errors: list[str] = []
        profiles = scan_profiles(errors=errors)
        self.last_scan_errors = errors
        return profiles

    def list_profiles(self) -> list[dict[str, Any]]:
        # Always rescan: "is Chrome using this profile" changes with every window
        # the user opens, and it is only a few small JSON reads.
        return self.refresh_profiles()

    def get_profile(self, profile_id_value: str) -> dict[str, Any] | None:
        wanted = (profile_id_value or '').strip()
        if not wanted:
            return None
        for profile in self.list_profiles():
            if profile['id'] == wanted:
                return profile
        return None

    # -- selection ------------------------------------------------------------

    def select_profile(self, profile_id_value: str) -> dict[str, Any]:
        profile = self.get_profile(profile_id_value)
        if profile is None:
            raise ChromeProfileError('That Chrome profile is no longer available. Refresh the list.', 'error')
        self.config.write(
            selected_profile_id=profile['id'],
            last_selected_profile_folder=profile['profile_folder'],
            last_selected_root_path=profile['root_path'],
        )
        return profile

    def get_selected_profile(self) -> dict[str, Any] | None:
        state = self.config.read()
        selected_id = state.get('selected_profile_id', '')
        if not selected_id:
            return None
        for profile in self.list_profiles():
            if profile['id'] == selected_id:
                return profile
        # The profile moved (reinstalled Chrome, renamed folder): report the
        # stale selection instead of silently picking another account.
        return None

    # -- launching ------------------------------------------------------------

    def launch_profile(self, profile_id_value: str, headless: bool | None = None) -> ProfileSession:
        """Open one profile. Any session still open is closed first."""
        profile = self.get_profile(profile_id_value)
        if profile is None:
            raise ChromeProfileError('That Chrome profile is no longer available. Refresh the list.', 'error')
        self._ensure_not_busy(profile)
        if not playwright_available():
            raise ChromeProfileError(MISSING_PLAYWRIGHT_MESSAGE, 'error')
        executable = chrome_executable()
        if executable is None:
            raise ChromeProfileError(MISSING_CHROME_MESSAGE, 'error')
        with self._lock:
            self.close()
            session = self._open(profile, executable, headless)
            self._session = session
            return session

    def launch_selected_profile(self, headless: bool | None = None) -> ProfileSession:
        state = self.config.read()
        selected_id = state.get('selected_profile_id', '')
        if not selected_id:
            raise ChromeProfileError('No Chrome profile is selected yet.', 'error')
        if headless is None:
            headless = bool(state.get('headless', False))
        return self.launch_profile(selected_id, headless=headless)

    def close(self) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session is not None:
            session.close()

    def _open(self, profile: dict[str, Any], executable: Path, headless: bool | None) -> ProfileSession:
        folder = profile['profile_folder']
        # The tool's own profile: a stable data directory Chrome accepts for
        # remote debugging, and the place a sign-in survives.
        user_data_dir = self.session_profile(profile)

        from playwright.sync_api import sync_playwright  # imported lazily: optional dependency

        playwright = sync_playwright().start()
        try:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                executable_path=str(executable),
                headless=bool(headless),
                args=[
                    # Chrome keeps profiles inside one User Data root: the root is
                    # the user-data dir, the folder is selected by this flag.
                    f'--profile-directory={folder}',
                    '--no-first-run',
                    '--no-default-browser-check',
                    '--hide-crash-restore-bubble',
                ],
                # Do not advertise automation: Chrome shows a bar that changes
                # the Gemini layout, and the user is signing in to a real account.
                ignore_default_args=['--enable-automation'],
            )
        except Exception as error:
            try:
                playwright.stop()
            except Exception:
                pass
            raise ChromeProfileError(_launch_error_message(error), _launch_error_status(error)) from error
        return ProfileSession(profile, context, playwright)

    def session_dir(self, profile: dict[str, Any]) -> Path:
        """The tool's own Chrome profile for this Chrome profile (kept between runs)."""
        return self.sessions_dir / profile['id'] / 'User Data'

    def has_session_profile(self, profile: dict[str, Any]) -> bool:
        marker = self.sessions_dir / profile['id'] / 'session.json'
        user_data = self.session_dir(profile)
        return marker.is_file() and (user_data / profile['profile_folder']).is_dir()

    def session_profile(self, profile: dict[str, Any]) -> Path:
        """Seed the tool's profile from Chrome once, then always reuse it.

        Chrome 136+ refuses remote debugging on the real default data directory,
        and a fresh copy of a profile is a *new device* for Google — so the copy
        is created once and then kept: signing in inside it makes the session
        stick, which is what Validate and the Gemini prompt use afterwards.
        """
        base = self.sessions_dir / profile['id']
        user_data = base / 'User Data'
        if self.has_session_profile(profile):
            return user_data

        shutil.rmtree(base, ignore_errors=True)
        folder_dir = user_data / profile['profile_folder']
        folder_dir.mkdir(parents=True, exist_ok=True)
        source_root = Path(profile['root_path'])
        source_profile = Path(profile['path'])
        local_state = source_root / 'Local State'
        if local_state.is_file():
            shutil.copy2(local_state, user_data / 'Local State')
            self._trim_local_state(user_data / 'Local State', profile['profile_folder'])
        for relative in SHADOW_PROFILE_FILES:
            source = source_profile / relative
            if source.is_file():
                destination = folder_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        # Local Storage carries the Gemini shell's own state; without it the app
        # reloads as a guest even when the cookies survive.
        local_storage = source_profile / 'Local Storage'
        if local_storage.is_dir():
            try:
                shutil.copytree(local_storage, folder_dir / 'Local Storage', dirs_exist_ok=True)
            except OSError:
                pass
        (base / 'session.json').write_text(
            json.dumps({'source_profile_id': profile['id'], 'profile_folder': profile['profile_folder']}),
            encoding='utf-8',
        )
        return user_data

    @staticmethod
    def _trim_local_state(path: Path, folder: str) -> None:
        """Keep only the copied profile in ``info_cache``.

        Chrome's profile menu is built from that cache; leaving the other entries
        in would offer profiles whose folders do not exist in the tool's profile,
        and switching to one would open an empty browser.
        """
        try:
            data = json.loads(path.read_text(encoding='utf-8', errors='replace'))
        except (OSError, ValueError):
            return
        section = data.get('profile') if isinstance(data, dict) else None
        if not isinstance(section, dict):
            return
        cache = section.get('info_cache')
        entry = cache.get(folder) if isinstance(cache, dict) else None
        section['info_cache'] = {folder: entry if isinstance(entry, dict) else {'name': folder}}
        section['last_used'] = folder
        try:
            path.write_text(json.dumps(data), encoding='utf-8')
        except OSError:
            pass

    def reset_session_profile(self, profile_id_value: str) -> dict[str, Any]:
        """Throw the tool's profile away (forces a fresh seed and a new sign-in)."""
        profile = self.get_profile(profile_id_value)
        if profile is None:
            raise ChromeProfileError('That Chrome profile is no longer available.', 'error')
        shutil.rmtree(self.sessions_dir / profile['id'], ignore_errors=True)
        return {'status': 'ok', 'profile_id': profile['id'], 'message': 'The tool\'s Chrome profile was reset.'}

    def session_in_use(self, profile: dict[str, Any]) -> bool:
        """True while the tool's own Chrome window holds this profile open."""
        target = _key(self.session_dir(profile))
        for _pid, command_line in chrome_processes():
            user_data_dir = _flag_value(command_line, '--user-data-dir')
            if user_data_dir and _key(user_data_dir) == target:
                return True
        return False

    def _ensure_not_busy(self, profile: dict[str, Any]) -> None:
        folder = profile['profile_folder']
        if self.session_in_use(profile):
            raise ChromeProfileError(
                f"The tool's Chrome window for {folder} is open. Close it, then try again.", 'locked',
            )
        if profile['is_locked'] and not self.has_session_profile(profile):
            # Seeding needs a quiet source profile: Chrome may be mid-write.
            raise ChromeProfileError(locked_message(folder), 'locked')

    # -- validation -----------------------------------------------------------

    def validate_profile(self, profile_id_value: str) -> dict[str, Any]:
        """Open the profile, load Gemini, report ready / login_required / locked / error."""
        profile = self.get_profile(profile_id_value)
        if profile is None:
            return _validation(profile_id_value, 'error', 'That Chrome profile is no longer available. Refresh the list.')
        try:
            self._ensure_not_busy(profile)
        except ChromeProfileError as error:
            return _validation(profile['id'], error.status, str(error))
        if not playwright_available():
            return _validation(profile['id'], 'error', MISSING_PLAYWRIGHT_MESSAGE)
        if chrome_executable() is None:
            return _validation(profile['id'], 'error', MISSING_CHROME_MESSAGE)
        try:
            with self.launch_profile(profile['id'], headless=bool(self.config.read().get('headless', False))) as session:
                state, message, url = gemini_state(session.page)
                result = _validation(profile['id'], state, message)
                result['url'] = url
                result['via'] = session.via
                return result
        except ChromeProfileError as error:
            return _validation(profile['id'], error.status, str(error))
        except Exception as error:  # one broken profile must never take the server down
            return _validation(profile['id'], 'error', f'{error.__class__.__name__}: {error}')

    # -- manual login ---------------------------------------------------------

    def open_manual(self, profile_id_value: str) -> dict[str, Any]:
        """Open the tool's own profile in Chrome, on Gemini, so you can sign in.

        Signing in here (not in your normal Chrome window) is what makes the
        session usable: Validate and the Gemini prompt then open this very
        profile, and Google sees the same browser it saw at sign-in.
        """
        profile = self.get_profile(profile_id_value)
        if profile is None:
            return {'success': False, 'status': 'error',
                    'message': 'That Chrome profile is no longer available. Refresh the list.'}
        executable = chrome_executable()
        if executable is None:
            return {'success': False, 'status': 'error', 'message': MISSING_CHROME_MESSAGE}
        user_data_dir = self.session_profile(profile)
        command = [
            str(executable),
            f'--user-data-dir={user_data_dir}',
            f'--profile-directory={profile["profile_folder"]}',
            GEMINI_URL,
        ]
        flags = 0
        if sys.platform == 'win32':
            flags = getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        try:
            subprocess.Popen(command, close_fds=True, creationflags=flags)
        except OSError as error:
            return {'success': False, 'status': 'error', 'message': f'Chrome could not be started: {error}'}
        return {
            'success': True,
            'status': 'manual_login',
            'profile_id': profile['id'],
            'message': (
                "The tool's own Chrome window opened on Gemini. Sign in there if you have not yet, browse its "
                'chat list to see the runs it saved, then close the window (Validate and Translate need it closed).'
            ),
        }


def _validation(profile_id_value: str, status: str, message: str) -> dict[str, Any]:
    return {
        'success': status == 'ready',
        'profile_id': profile_id_value,
        'status': status,
        'message': message,
    }


def _launch_error_status(error: Exception) -> str:
    text = str(error).lower()
    if 'already in use' in text or 'processsingleton' in text or 'singletonlock' in text:
        return 'locked'
    return 'error'


def _launch_error_message(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if 'already in use' in lowered or 'processsingleton' in lowered or 'singletonlock' in lowered:
        return LOCKED_MESSAGE
    if 'executable' in lowered and 'does not exist' in lowered:
        return MISSING_CHROME_MESSAGE
    if 'target page, context or browser has been closed' in lowered:
        return 'Chrome closed before the profile could be opened. Close Chrome completely and try again.'
    return f'Chrome could not open this profile: {text.splitlines()[0][:400]}'


# --- Gemini page state -------------------------------------------------------


def gemini_state(page: Any) -> tuple[str, str, str]:
    """(status, message, url) for the Gemini tab, using only the visible UI.

    Gemini always renders a composer, so "signed in" is read as the *absence* of a
    sign-in call to action next to it: the account chip is drawn by Google's
    ``ogs.google.com`` iframe and is not reliably visible from this frame. Two
    consecutive readings are required, so a half-loaded page cannot pass as a
    session.
    """
    try:
        page.goto(GEMINI_URL, wait_until='domcontentloaded', timeout=PLAYWRIGHT_TIMEOUT_MS)
    except Exception as error:
        return 'error', f'Gemini did not load: {str(error).splitlines()[0][:300]}', _safe_url(page)

    deadline = time.monotonic() + GEMINI_SETTLE_SECONDS
    settled = 0
    while time.monotonic() < deadline:
        url = _safe_url(page)
        if 'accounts.google.com' in url:
            return ('login_required',
                    'Gemini redirected to the Google sign-in page. Use "Open manual login" and sign in yourself.',
                    url)
        if _shows_sign_in(page):
            return ('login_required',
                    'Gemini is signed out in the tool\'s own browser for this profile. Use "Open manual login" '
                    'and sign in to Gemini in the window it opens (not your normal Chrome), then validate again.',
                    url)
        if _shows_account_menu(page) or _has_prompt_box(page):
            settled += 1
            if settled >= 2:
                return 'ready', 'Gemini is signed in for this profile.', url
        else:
            settled = 0
        try:
            page.wait_for_timeout(500)
        except Exception:
            break
    return ('error',
            'Gemini never showed a usable page (no composer, no sign-in button) within '
            f'{int(GEMINI_SETTLE_SECONDS)} s. The page may have changed.',
            _safe_url(page))



def _safe_url(page: Any) -> str:
    try:
        return page.url
    except Exception:
        return ''


# The account menu only exists for a signed-in profile; the labels below are the
# stable part of Google's header across Gemini versions.
ACCOUNT_MENU_SELECTORS = (
    'a[href*="SignOutOptions"]',
    '[aria-label*="Google Account" i]',
)


def _shows_account_menu(page: Any) -> bool:
    """A *visible* account menu — a hidden or half-loaded node is not a session."""
    for selector in ACCOUNT_MENU_SELECTORS:
        try:
            locator = page.locator(selector)
            if locator.count() > 0 and locator.first.is_visible():
                return True
        except Exception:
            continue
    return False


def _has_prompt_box(page: Any) -> bool:
    for selector in ('rich-textarea', 'div.ql-editor[contenteditable="true"]', '[role="textbox"]'):
        try:
            locator = page.locator(selector)
            if locator.count() > 0 and locator.first.is_visible():
                return True
        except Exception:
            continue
    return False


def _shows_sign_in(page: Any) -> bool:
    """A visible sign-in call to action.

    Gemini renders its header as a plain ``Sign in`` link to
    ``accounts.google.com/ServiceLogin`` (the button role is not guaranteed
    across versions), so the link is checked first and the role second.
    """
    for selector in ('a[href*="ServiceLogin"]',):
        try:
            locator = page.locator(selector)
            if locator.count() > 0 and locator.first.is_visible():
                return True
        except Exception:
            continue
    try:
        buttons = page.get_by_role('button', name=re.compile(r'^\s*sign in\s*$', re.IGNORECASE))
        total = buttons.count()
    except Exception:
        return False
    for index in range(min(total, 4)):
        try:
            node = buttons.nth(index)
            box = node.bounding_box()
        except Exception:
            continue
        if box and box.get('width') and box.get('y', 1000) < 120:
            return True
    return False


# --- app-wide instance -------------------------------------------------------


_manager: ChromeProfileManager | None = None
_manager_lock = threading.Lock()


def manager() -> ChromeProfileManager:
    """Process-wide manager (the selection lives in one JSON file)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ChromeProfileManager()
        return _manager


def reset_manager() -> None:
    """Test hook: drop the cached manager (closing any open browser)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            try:
                _manager.close()
            except Exception:
                pass
        _manager = None


def snapshot() -> dict[str, Any]:
    """Everything the Chrome Profiles panel needs, in one round trip."""
    profile_manager = manager()
    profiles = profile_manager.refresh_profiles()
    for profile in profiles:
        # The panel explains the first run: sign in once inside the tool's window.
        profile['session_ready'] = profile_manager.has_session_profile(profile)
    state = profile_manager.config.read()
    selected_id = state.get('selected_profile_id', '')
    selected = next((profile for profile in profiles if profile['id'] == selected_id), None)
    executable = chrome_executable()
    return {
        'profiles': profiles,
        'selected': selected,
        'selected_profile_id': selected_id,
        'selected_missing': bool(selected_id) and selected is None,
        'last_selected_profile_folder': state.get('last_selected_profile_folder', ''),
        'last_selected_root_path': state.get('last_selected_root_path', ''),
        'headless': bool(state.get('headless', False)),
        'roots': [str(root) for root in chrome_roots()],
        'chrome_path': str(executable) if executable else '',
        'playwright_ready': playwright_available(),
        'chrome_running': chrome_running(),
        'errors': list(profile_manager.last_scan_errors),
    }
