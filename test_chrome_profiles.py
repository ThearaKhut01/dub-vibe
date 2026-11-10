"""List the Chrome profiles on this PC and check the saved one.

    python test_chrome_profiles.py
    python test_chrome_profiles.py --validate
    python test_chrome_profiles.py --select "Profile 1"
    python test_chrome_profiles.py --json

Run it with the app's interpreter (``.venv\\Scripts\\python.exe``) so the backend
package — and Playwright, when you validate — resolves.

Scanning reads Chrome's own metadata only: ``Local State`` and each profile's
``Preferences``. No cookie, password, token or ``Login Data`` file is opened, and
nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys

from backend.app import chrome_profiles

__test__ = False  # a manual script, not a pytest module


def mask(email: str) -> str:
    at = email.find('@')
    return email if at < 1 else f'{email[: min(2, at)]}****{email[at:]}'


def _find(profiles: list[dict], needle: str) -> dict | None:
    for profile in profiles:
        if profile['id'] == needle or profile['profile_folder'].casefold() == needle.casefold():
            return profile
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Detect Chrome profiles (metadata only).')
    parser.add_argument('--validate', action='store_true',
                        help='open the saved profile on gemini.google.com and report the login state')
    parser.add_argument('--profile', metavar='FOLDER', help='profile folder or id (defaults to the saved one)')
    parser.add_argument('--select', metavar='FOLDER', help='save this profile folder (or id) as the selection')
    parser.add_argument('--json', action='store_true', help='print the raw result instead of a table')
    args = parser.parse_args(argv)

    manager = chrome_profiles.ChromeProfileManager()
    profiles = manager.refresh_profiles()
    selected = manager.get_selected_profile()

    if args.json:
        print(json.dumps({
            'chrome_path': str(chrome_profiles.chrome_executable() or ''),
            'playwright_ready': chrome_profiles.playwright_available(),
            'roots': [str(root) for root in chrome_profiles.chrome_roots()],
            'profiles': profiles,
            'selected': selected,
            'config': manager.config.read(),
            'errors': manager.last_scan_errors,
        }, indent=2))
        return 0

    print(f'Chrome    : {chrome_profiles.chrome_executable() or "not found"}')
    print(f'User data : {", ".join(str(root) for root in chrome_profiles.chrome_roots()) or "none"}')
    print(f'Playwright: {"available" if chrome_profiles.playwright_available() else "missing (pip install playwright)"}')
    print()

    if not profiles:
        print('No Chrome profiles with Google account metadata found. Open Chrome and sign in manually first.')
        return 1

    print(f'{"  ":2}{"folder":<10} {"name":<16} {"email":<30} {"google":<7} {"state":<7} id')
    print('-' * 92)
    for profile in profiles:
        marker = '*' if selected and selected['id'] == profile['id'] else ' '
        email = mask(profile['emails'][0]) if profile['emails'] else '-'
        print(
            f'{marker:2}{profile["profile_folder"]:<10} {profile["display_name"][:16]:<16} '
            f'{email:<30} {("yes" if profile["has_google_account"] else "no"):<7} '
            f'{("locked" if profile["is_locked"] else "open"):<7} {profile["id"]}'
        )
    print()
    print(f'Selected  : {selected["profile_folder"] + " / " + mask(selected["emails"][0]) if selected and selected["emails"] else (selected["profile_folder"] if selected else "none")}')

    for message in manager.last_scan_errors:
        print(f'warning   : {message}')

    if args.select:
        target = _find(profiles, args.select)
        if target is None:
            print(f'\nNo profile matches "{args.select}".')
            return 2
        manager.select_profile(target['id'])
        print(f'\nSaved selection: {target["profile_folder"]} ({target["id"]})')

    if args.validate:
        target = _find(profiles, args.profile) if args.profile else (manager.get_selected_profile() or None)
        if target is None:
            print('\nNo profile to validate: pass --profile "Profile 1" or select one first.')
            return 2
        print(f'\nValidating {target["profile_folder"]} on {chrome_profiles.GEMINI_URL} ...')
        result = manager.validate_profile(target['id'])
        print(f'status    : {result["status"]}')
        print(f'message   : {result["message"]}')
        if result.get('url'):
            print(f'url       : {result["url"]}')
        if result.get('via'):
            print(f'opened    : {result["via"]}')
        return 0 if result['status'] == 'ready' else 2

    return 0


if __name__ == '__main__':
    sys.exit(main())
