"""Talk to Gemini's web app through the Chrome profile the user selected.

This is the browser path, not the API path: the app's translation/TTS calls keep
using ``online_providers`` (an API key needs no browser and no sign-in). This
module exists for the case where the user wants to reuse a subscription they are
already signed in to, in a profile that is already logged in.

It only reads what is on screen. It never types credentials, never creates an
account and never tries to pass a CAPTCHA: if Gemini asks for a sign-in, the
answer is a message telling the user to sign in themselves.

Every prompt is recorded as one node event (``chrome:<profile id>``), which is
what the Live tab draws.
"""
from __future__ import annotations

import re
import time
from typing import Any

from backend.app import api_events, chrome_profiles

PROMPT_SELECTORS = (
    'rich-textarea',
    'div.ql-editor[contenteditable="true"]',
    '[contenteditable="true"]',
)
RESPONSE_SELECTORS = (
    'model-response',
    'message-content',
    '.model-response-text',
)
POLL_INTERVAL_MS = 700
MODEL_LABEL = 'gemini.google.com'


class GeminiWebError(RuntimeError):
    """The browser path could not run; the message is meant for the UI."""

    def __init__(self, message: str, status: str = 'error') -> None:
        super().__init__(message)
        self.status = status


class GeminiWebAutomator:
    """One-shot prompts in the selected profile.

    Each call opens the profile, sends the prompt, reads the answer and closes
    the browser, because Playwright's sync API is tied to the calling thread.
    """

    def __init__(self, manager: chrome_profiles.ChromeProfileManager | None = None) -> None:
        self.manager = manager or chrome_profiles.manager()

    def status(self) -> dict[str, Any]:
        """Selected profile + whether the browser path can run at all."""
        selected = self.manager.get_selected_profile()
        return {
            'selected': selected,
            'playwright_ready': chrome_profiles.playwright_available(),
            'chrome_path': str(chrome_profiles.chrome_executable() or ''),
            'ready': bool(selected) and chrome_profiles.playwright_available(),
        }

    def ask(self, prompt: str, timeout_s: float = 180.0) -> dict[str, Any]:
        """Send one prompt and return Gemini's answer as plain text."""
        text = (prompt or '').strip()
        if not text:
            raise GeminiWebError('An empty prompt has nothing to send.', 'error')
        if len(text) > 20_000:
            raise GeminiWebError('That prompt is longer than 20,000 characters.', 'error')
        profile_id = str(self.manager.config.read().get('selected_profile_id') or '')
        if not profile_id:
            raise GeminiWebError('Select a Chrome profile first.', 'error')
        if not chrome_profiles.playwright_available():
            raise GeminiWebError(chrome_profiles.MISSING_PLAYWRIGHT_MESSAGE, 'error')

        # One node per profile, so the Live tab can show this request travelling.
        node_id = f'chrome:{profile_id}'
        request_id = api_events.start(node_id, provider='chrome', model=MODEL_LABEL, lines=1)
        started = time.monotonic()
        try:
            result = self._ask_in_profile(text, timeout_s)
        except Exception as error:
            self._finish(request_id, node_id, started, ok=False, error=error)
            raise
        self._finish(
            request_id, node_id, started, ok=bool(result.get('success')),
            error='' if result.get('success') else str(result.get('message') or ''),
        )
        return result

    def _ask_in_profile(self, text: str, timeout_s: float) -> dict[str, Any]:
        try:
            with self.manager.launch_selected_profile() as session:
                page = session.page
                state, message, url = chrome_profiles.gemini_state(page)
                if state != 'ready':
                    return {
                        'success': False,
                        'status': state,
                        'message': message,
                        'url': url,
                        'profile_id': session.profile['id'],
                    }
                before = _response_count(page)
                _submit(page, text)
                try:
                    reply = _read_answer(page, before, timeout_s)
                except Exception as error:
                    return {
                        'success': False,
                        'status': 'error',
                        'message': f'Reading Gemini\'s answer failed: {str(error).splitlines()[0][:300]}',
                        'url': _url(page),
                        'profile_id': session.profile['id'],
                    }
                if not reply:
                    return {
                        'success': False,
                        'status': 'error',
                        'message': (
                            'Gemini accepted the prompt but no answer appeared in time. '
                            'The page layout may have changed, or the account hit its limit.'
                        ),
                        'url': _url(page),
                        'profile_id': session.profile['id'],
                    }
                return {
                    'success': True,
                    'status': 'ready',
                    'reply': reply,
                    'url': _url(page),
                    'profile_id': session.profile['id'],
                    'via': session.via,
                }
        except chrome_profiles.ChromeProfileError as error:
            return {'success': False, 'status': error.status, 'message': str(error)}

    @staticmethod
    def _finish(request_id: str, node_id: str, started: float, *, ok: bool, error: object = '') -> None:
        api_events.finish(
            request_id, node_id, ok=ok, provider='chrome', model=MODEL_LABEL, lines=1,
            latency_ms=int((time.monotonic() - started) * 1000), error=error,
        )


def _first_visible(page: Any, selectors: tuple[str, ...]) -> Any:
    """First locator that exists for any of the known selectors."""
    for selector in selectors:
        try:
            locator = page.locator(selector)
            if locator.count() > 0:
                return locator.first
        except Exception:
            continue
    return None


def _url(page: Any) -> str:
    try:
        return page.url
    except Exception:
        return ''


def _response_count(page: Any) -> int:
    best = 0
    for selector in RESPONSE_SELECTORS:
        try:
            best = max(best, page.locator(selector).count())
        except Exception:
            continue
    return best


def _submit(page: Any, text: str) -> None:
    box = _first_visible(page, PROMPT_SELECTORS)
    if box is None:
        raise GeminiWebError('The Gemini prompt box was not found; the page layout may have changed.', 'error')
    try:
        box.click()
        box.fill(text)
        page.keyboard.press('Enter')
    except Exception as error:
        raise GeminiWebError(f'The prompt could not be typed into Gemini: {str(error).splitlines()[0][:300]}', 'error') from error


def _read_answer(page: Any, before: int, timeout_s: float) -> str:
    """Wait until a new answer stops growing, then return its text."""
    deadline = time.monotonic() + max(15.0, timeout_s)
    last_text = ''
    stable_reads = 0
    while time.monotonic() < deadline:
        page.wait_for_timeout(POLL_INTERVAL_MS)
        if _response_count(page) <= before:
            continue
        current = _last_answer_text(page)
        if current and current == last_text:
            stable_reads += 1
            if stable_reads >= 3:
                return current
        else:
            stable_reads = 0
            last_text = current
    return last_text


def _last_answer_text(page: Any) -> str:
    for selector in RESPONSE_SELECTORS:
        try:
            locator = page.locator(selector)
            if locator.count() > 0:
                text = locator.last.inner_text().strip()
                if text:
                    return re.sub(r'\n{3,}', '\n\n', text)
        except Exception:
            continue
    return ''
