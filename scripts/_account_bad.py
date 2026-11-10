"""Show which Google account the tool's own profile is signed into."""
from pathlib import Path


def patch(path: str, edits: list[tuple[str, str]]) -> None:
    file = Path(path)
    raw = file.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    text = raw.decode('utf-8').replace('\r\n', '\n')
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f'{path}: expected 1 match, found {count}: {old[:70]!r}')
        text = text.replace(old, new)
    file.write_bytes(text.replace('\n', newline).encode('utf-8'))
    print('patched', path)


patch('backend/app/chrome_profiles.py', [
    (
        '''    profile_manager = manager()
    profiles = profile_manager.refresh_profiles()''',
        '''    profile_manager = manager()
    profiles = profile_manager.refresh_profiles()
    # Which Google account the tool's own profile carries: the fastest way to see
    # why a chat shows in one window and not the other.
    session_account = ''
    session_state = profile_manager.sessions_dir and (profile_manager.sessions_dir)
    if session_state:
        for profile in profiles:
            if profile['id'] == selected_profile_id_hint(profile_manager):
                session_account = _session_profile_account(profile_manager, profile)
                break''',
    ),
    (
        """        'roots': [str(root) for root in chrome_roots()],""",
        """        'roots': [str(root) for root in chrome_roots()],
        'session_account': session_account,""",
    ),
])
