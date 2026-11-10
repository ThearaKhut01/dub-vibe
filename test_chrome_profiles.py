"""List the Chrome profiles on this PC, check the saved one, and drive Gemini.

    python test_chrome_profiles.py
    python test_chrome_profiles.py --select "Profile 5"
    python test_chrome_profiles.py --open                 # sign in manually
    python test_chrome_profiles.py --validate             # signed in?
    python test_chrome_profiles.py --ask "Where were you last night?"
    python test_chrome_profiles.py --json

Run it with the app's interpreter (``.venv\\Scripts\\python.exe``) so the backend
package — and Playwright, when a browser is opened — resolves.

Scanning reads Chrome's own metadata only: ``Local State`` and each profile's
``Preferences``. No cookie, password, token or ``Login Data`` file is opened, and
nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from backend.app import chrome_profiles, gemini_web

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
    parser = argparse.ArgumentParser(description='Detect Chrome profiles (metadata only) and drive Gemini web.')
    parser.add_argument('--validate', action='store_true',
                        help='open the profile on gemini.google.com and report the login state')
    parser.add_argument('--profile', metavar='FOLDER', help='profile folder or id (defaults to the saved one)')
    parser.add_argument('--select', metavar='FOLDER', help='save this profile folder (or id) as the selection')
    parser.add_argument('--open', action='store_true',
                        help="open the tool's own Chrome profile on Gemini, so you can sign in once")
    parser.add_argument('--signin', action='store_true',
                        help='--open, wait until you close that window, then validate')
    parser.add_argument('--history', action='store_true',
                        help="open the tool's own Chrome profile on Gemini, to browse the chats it saved")
    parser.add_argument('--chats', type=int, nargs='?', const=15, metavar='N',
                        help='list the last N chats Gemini has for this profile (no window stays open)')
    parser.add_argument('--reset-session', action='store_true',
                        help="delete the tool's own Chrome profile (forces a new sign-in)")
    parser.add_argument('--ask', metavar='LINE', help='send one line to Gemini web and print the answer')
    parser.add_argument('--translate', metavar='LINE',
                        help='translate one subtitle line through Gemini web (see --to)')
    parser.add_argument('--to', default='Khmer', metavar='LANGUAGE',
                        help='target language for --translate (default: Khmer)')
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

    def target_for(needle: str | None, action: str) -> dict | None:
        profile = _find(profiles, needle) if needle else (manager.get_selected_profile() or None)
        if profile is None:
            print(f'\nNo profile to {action}: pass --profile "Profile 1" or select one first.')
        return profile

    if args.select:
        target = _find(profiles, args.select)
        if target is None:
            print(f'\nNo profile matches "{args.select}".')
            return 2
        manager.select_profile(target['id'])
        selected = target
        print(f'\nSaved selection: {target["profile_folder"]} ({target["id"]})')

    if args.open or args.signin or args.history:
        if args.history and not (args.open or args.signin):
            print("\nOpening the tool's own Chrome profile on Gemini, on the page that lists its chats.")
            print('The tool\'s runs are the entries Gemini titled from the prompt ("Subtitle sentence translation…").')
        target = target_for(args.profile, 'open')
        if target is None:
            return 2
        result = manager.open_manual(target['id'])
        print(f'\n{result["message"]}')
        if not result['success'] or not args.signin:
            return 0 if result['success'] else 2

        print('  1. click "Verify it\'s you" if Chrome asks (that is Chrome, not Gemini)')
        print('  2. sign in to Gemini in that window (top-right "Sign in")')
        print('  3. close that window when the Gemini header shows your avatar')
        print('\nWaiting for the window to close (up to 10 minutes) ...', end='', flush=True)
        deadline = time.monotonic() + 600
        while manager.session_in_use(target) and time.monotonic() < deadline:
            time.sleep(3)
            print('.', end='', flush=True)
        still_open = manager.session_in_use(target)
        print()

        print(f'Validating {target["profile_folder"]} on {chrome_profiles.GEMINI_URL} ...')
        check = manager.validate_profile(target['id'])
        print(f'status    : {check["status"]}')
        print(f'message   : {check["message"]}')
        if check['status'] == 'ready':
            print('\nReady. Try one subtitle line:')
            print(f'  .venv\\Scripts\\python.exe test_chrome_profiles.py --ask "Where were you last night?"')
            return 0
        if still_open:
            print('\nThat window is still open — close it, then run --validate again.')
        return 2

    if args.chats:
        target = target_for(args.profile, 'list chats for')
        if target is None:
            return 2
        if not selected or selected['id'] != target['id']:
            manager.select_profile(target['id'])
            print(f'\nSaved selection: {target["profile_folder"]}')
        print(f'\nChats Gemini has for {target["profile_folder"]} (newest first):')
        try:
            titles = gemini_web.recent_chats(limit=max(1, args.chats))
        except gemini_web.GeminiWebError as error:
            print(f'error     : {error}')
            return 2
        if not titles:
            print('  (none)')
            return 0
        for index, title in enumerate(titles, start=1):
            print(f'  {index:2}. {title}')
        return 0

    if args.reset_session:
        target = target_for(args.profile, 'reset')
        if target is None:
            return 2
        result = manager.reset_session_profile(target['id'])
        print(f"\n{result['message']}")
        return 0

    if args.validate:
        target = target_for(args.profile, 'validate')
        if target is None:
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

    if args.translate:
        target = target_for(args.profile, 'translate')
        if target is None:
            return 2
        if not selected or selected['id'] != target['id']:
            manager.select_profile(target['id'])
            print(f'\nSaved selection: {target["profile_folder"]}')
        prompt = gemini_web.build_translation_prompt(args.translate, args.to)
        print(f'\nTranslating into {args.to} in {target["profile_folder"]}: {args.translate!r}')
        try:
            result = gemini_web.GeminiWebAutomator(manager).ask(prompt)
        except gemini_web.GeminiWebError as error:
            print(f'error     : {error}')
            return 2
        print(f'status    : {result.get("status")}')
        if result.get('success'):
            print('line      :')
            print(f'\n{result.get("reply", "")}\n')
            return 0
        print(f'message   : {result.get("message", "")}')
        return 2

    if args.ask:
        target = target_for(args.profile, 'ask')
        if target is None:
            return 2
        if not selected or selected['id'] != target['id']:
            # The automator always uses the saved selection, so save it first.
            manager.select_profile(target['id'])
            print(f'\nSaved selection: {target["profile_folder"]}')
        print(f'\nAsking Gemini in {target["profile_folder"]}: {args.ask!r}')
        try:
            result = gemini_web.GeminiWebAutomator(manager).ask(args.ask)
        except gemini_web.GeminiWebError as error:
            print(f'error     : {error}')
            return 2
        print(f'status    : {result.get("status")}')
        if result.get('success'):
            print('answer    :')
            print(f'\n{result.get("reply", "")}\n')
            if result.get('via'):
                print(f'opened    : {result["via"]}')
            return 0
        print(f'message   : {result.get("message", "")}')
        if result.get('status') == 'login_required':
            print('\nSign in in the tool\'s window first:  python test_chrome_profiles.py --open')
        return 2

    return 0


if __name__ == '__main__':
    sys.exit(main())
