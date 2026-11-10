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
    'rich-textarea div.ql-editor[contenteditable="true"]',
    'div.ql-editor[contenteditable="true"]',
    '[role="textbox"][contenteditable="true"]',
    'rich-textarea',
    'textarea',
)
PROMPT_WAIT_SECONDS = 15.0
RESPONSE_SELECTORS = (
    'model-response',
    'message-content',
    '.model-response-text',
)
POLL_INTERVAL_MS = 700
# Never give up on an answer sooner than this, even when a caller asks for less.
MIN_ANSWER_SECONDS = 15.0
MODEL_LABEL = 'gemini.google.com'
# Gemini web answers a numbered list reliably up to about this many lines; more
# than that and it starts merging or skipping entries.
BATCH_SIZE = 20
MIN_BATCH_SECONDS = 120.0


def build_batch_prompt(lines: list[str], target_language: str = 'Khmer', source_language: str = '',
                       glossary: str = '') -> str:
    """One message for a whole subtitle batch, numbered so the answer is parseable."""
    source = f' from {source_language}' if source_language else ''
    numbered = '\n'.join(f'{index + 1}. {line}' for index, line in enumerate(lines))
    terms = str(glossary or '').strip()
    glossary_part = f'\nUse these term translations exactly: {terms}' if terms else ''
    return (
        f'Translate each numbered subtitle line{source} into {target_language}. '
        'Keep the same numbering, one translation per number, and reply with the list only — '
        f'no quotes, no transliteration, no notes, no explanations.{glossary_part}\n'
        f'{numbered}'
    )


_NUMBERED = re.compile(r'^\s*(\d{1,3})\s*[.)\]:-]\s*(.+?)\s*$', re.MULTILINE)


def parse_numbered_answer(reply: str, count: int) -> list[str | None]:
    """Map a numbered answer back to the lines it belongs to (None when missing)."""
    found: dict[int, str] = {}
    for match in _NUMBERED.finditer(str(reply or '')):
        index = int(match.group(1))
        text = match.group(2).strip().strip('"“”').strip()
        if 1 <= index <= count and text and index not in found:
            found[index] = text
    return [found.get(index + 1) for index in range(count)]


def translate_lines(
    lines: list[str],
    target_language: str = 'Khmer',
    source_language: str = '',
    timeout_s: float = MIN_BATCH_SECONDS,
    glossary: str = '',
) -> list[str | None]:
    """Translate subtitle lines through the selected profile, one message per batch.

    A line the answer skipped is retried once on its own; if it is still missing
    the entry stays ``None`` so the caller can fall back for that line only.
    """
    automator = GeminiWebAutomator()
    results: list[str | None] = []
    for start in range(0, len(lines), BATCH_SIZE):
        batch = lines[start:start + BATCH_SIZE]
        prompt = build_batch_prompt(batch, target_language, source_language, glossary)
        answer = automator.ask(prompt, timeout_s=timeout_s, lines=len(batch))
        if not answer.get('success'):
            raise GeminiWebError(str(answer.get('message') or 'Gemini web did not answer.'), answer.get('status', 'error'))
        parsed = parse_numbered_answer(answer.get('reply', ''), len(batch))
        for offset, item in enumerate(parsed):
            if item is not None:
                continue
            single = automator.ask(
                build_translation_prompt(batch[offset], target_language, source_language),
                timeout_s=timeout_s,
            )
            if single.get('success'):
                parsed[offset] = str(single.get('reply') or '').strip().strip('"“”') or None
        results.extend(parsed)
    return results


def build_translation_prompt(text: str, target_language: str = 'Khmer', source_language: str = '') -> str:
    """A subtitle-ready prompt: one line in, one translation out.

    Subtitles need the line itself, not a conversation about it, so the
    instruction asks for the translation only and forbids commentary.
    """
    source = f' from {source_language}' if source_language else ''
    return (
        f'Translate this subtitle line{source} into {target_language}. '
        'Reply with the translation only — no quotes, no transliteration, no notes, '
        'no explanation, nothing else.\n'
        f'Line: {text}'
    )


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

    def ask(self, prompt: str, timeout_s: float = 180.0, lines: int = 1) -> dict[str, Any]:
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
        request_id = api_events.start(node_id, provider='chrome', model=MODEL_LABEL, lines=max(1, int(lines)))
        started = time.monotonic()
        try:
            result = self._ask_in_profile(text, timeout_s)
        except Exception as error:
            self._finish(request_id, node_id, started, ok=False, error=error)
            raise
        self._finish(
            request_id, node_id, started, ok=bool(result.get('success')), lines=max(1, int(lines)),
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
                try:
                    _submit(page, text)
                    reply = _read_answer(page, before, timeout_s)
                except GeminiWebError as error:
                    # A page we cannot drive is a result for the panel, not a crash.
                    return {
                        'success': False,
                        'status': error.status,
                        'message': str(error),
                        'url': _url(page),
                        'profile_id': session.profile['id'],
                    }
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
    def _finish(request_id: str, node_id: str, started: float, *, ok: bool, lines: int = 1,
                error: object = '') -> None:
        api_events.finish(
            request_id, node_id, ok=ok, provider='chrome', model=MODEL_LABEL, lines=lines,
            latency_ms=int((time.monotonic() - started) * 1000), error=error,
        )


def _wait_for_prompt_box(page: Any, timeout_s: float | None = None) -> Any:
    # Read the module constant at call time so tests (and callers) can shorten it.
    timeout = PROMPT_WAIT_SECONDS if timeout_s is None else timeout_s
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        box = _first_visible(page, PROMPT_SELECTORS)
        if box is not None:
            return box
        try:
            page.wait_for_timeout(300)
        except Exception:
            break
    return None


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
    # The editor hydrates after the shell: wait for it instead of failing on the
    # first paint.
    box = _wait_for_prompt_box(page)
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
    deadline = time.monotonic() + max(MIN_ANSWER_SECONDS, timeout_s)
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
                    # Gemini prefixes its answers with a "Gemini said" label.
                    text = re.sub(r'^Gemini\s+said\s*\n+', '', text, flags=re.IGNORECASE)
                    return re.sub(r'\n{3,}', '\n\n', text).strip()
        except Exception:
            continue
    return ''
