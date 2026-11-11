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
import threading
import time
from typing import Any

from backend.app import api_events, chrome_profiles
# The same target-script check the keyed engines use. A browser answer can drift
# into the source language (a mostly-Chinese batch makes Gemini answer English
# lines in Chinese) or echo the source; such a line is a missing line, not a
# translation, and must be re-asked before it reaches the dub.
from backend.app.providers import _resolve_lang_code, _translation_script_ok

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
# Gemini assigns the conversation id and saves the chat right after an answer.
# Closing the tab the instant the text stops growing loses that round trip (the
# chat then never shows up in Gemini's history), so give it a moment first.
CONVERSATION_SAVE_SECONDS = 2.5
# One browser, one tab, one conversation at a time. Two prompts at once would
# make the second launch close the first one's page, so every request in this
# module waits for the lock — a batch that overlaps another would otherwise
# fail and fall back to one-line chats.
_ASK_LOCK = threading.Lock()
CONVERSATION_ID = re.compile(r'/app/[0-9a-z]{6,}', re.IGNORECASE)
# Gemini inserts zero-width spaces between Khmer syllables; subtitles should not
# carry them (they break wrapping and look like typos in an editor).
_ZERO_WIDTH = re.compile('[\u200b\u200c\u200d\ufeff]')
# Never give up on an answer sooner than this, even when a caller asks for less.
MIN_ANSWER_SECONDS = 15.0
MODEL_LABEL = 'gemini.google.com'
# Gemini web answers a numbered list reliably up to about this many lines; more
# than that and it starts merging or skipping entries.
BATCH_SIZE = 20
# When a batch answer skips lines (or drifts to Thai/Chinese), retrying the same
# large group can skip them again. Small groups are far more reliable.
RETRY_BATCH_SIZE = 5
MIN_BATCH_SECONDS = 120.0


# Sidebar entries that are navigation, not conversations.
_NAV_LABELS = {
    'gemini', 'new chat', 'library', 'images', 'scheduled', 'students', 'customise',
    'customize', 'new notebook', 'notebooks', 'all', 'settings', 'help', 'sign in',
}
# Gemini renders each chat as a gem-nav-list-item[data-test-id=conversation];
# navigation links, the account chip and the upgrade banner are not matched.
_CONVERSATION_JS = (
    "() => [...document.querySelectorAll('[data-test-id=\"conversation\"]')]"
    ".map(el => (el.innerText || '').trim())"
    ".filter(text => text && text.length < 160)"
)


def clean_chat_titles(raw: list[str], limit: int = 20) -> list[str]:
    """Drop navigation labels and duplicates; keep the order Gemini shows."""
    seen: set[str] = set()
    titles: list[str] = []
    for value in raw or []:
        title = ' '.join(str(value or '').split())
        key = title.casefold()
        if not title or key in _NAV_LABELS or key in seen:
            continue
        seen.add(key)
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def recent_chats(limit: int = 20) -> list[str]:
    """The chats this profile can see in Gemini's sidebar (the tool's runs included)."""
    # Never open a second session while a prompt is being answered.
    with _ASK_LOCK:
        return _recent_chats(limit)


def _recent_chats(limit: int) -> list[str]:
    manager = chrome_profiles.manager()
    with manager.launch_selected_profile() as session:
        page = session.page
        state, message, _url = chrome_profiles.gemini_state(page)
        if state != 'ready':
            raise GeminiWebError(message, state)
        # The conversation list is rendered lazily, after the composer appears.
        deadline = time.monotonic() + 8.0
        raw: list[str] = []
        while time.monotonic() < deadline:
            raw = page.evaluate(_CONVERSATION_JS) or []
            if raw:
                break
            try:
                page.wait_for_timeout(500)
            except Exception:
                break
    return clean_chat_titles([str(item) for item in raw], limit)


def srt_time(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS,mmm`` (the shape SRT uses)."""
    total = max(0.0, float(seconds))
    hours, rest = divmod(int(total), 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((total - int(total)) * 1000))
    if millis == 1000:
        secs, millis = secs + 1, 0
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def build_batch_prompt(lines: list[str], target_language: str = 'Khmer', source_language: str = '',
                       glossary: str = '', cues: list[tuple[float, float]] | None = None,
                       context: list[str] | None = None, strict: bool = False) -> str:
    """One message for a whole subtitle batch.

    With ``cues`` the batch is sent as SRT blocks — number, timecode, text — so the
    model can see how long each line has to be spoken and keep the translation
    short enough. Without cues it falls back to a plain numbered list.

    ``strict`` is used for the retry of lines whose first answer was in the wrong
    language: it repeats that every block must be rewritten in the target.
    """
    source = f' from {source_language}' if source_language else ''
    terms = str(glossary or '').strip()
    glossary_part = f' Use these term translations exactly: {terms}.' if terms else ''
    earlier = [line for line in (context or []) if str(line or '').strip()][-3:]
    context_part = (
        ' Previous lines, for context only — do not translate them: '
        + ' | '.join(earlier) + '.'
    ) if earlier else ''
    # The source may mix languages (Chinese dialogue with English names, or a
    # subtitle file with both). Saying so up front stops the model from carrying
    # the source language into the answer, which is the most common failure on a
    # long film.
    script_part = (
        f'Every answer must be written in {target_language}, even when a source line is already in '
        'another language. Never copy a source line unchanged, never answer in a different language, '
        'and never reply with an apology, a note, or an explanation.'
    )
    guard = _script_guard(target_language)
    if guard:
        script_part += f' {guard}'
    if strict:
        script_part += (
            f' The previous attempt for some blocks was not in {target_language}: '
            f'translate every block again, each answer in {target_language} only.'
        )

    if cues and len(cues) == len(lines):
        blocks = '\n\n'.join(
            f'{index + 1}\n{srt_time(start)} --> {srt_time(end)}\n{line}'
            for index, (line, (start, end)) in enumerate(zip(lines, cues))
        )
        return (
            f'Translate the text of each subtitle block{source} into {target_language}. '
            'Keep each block number and its timecode exactly as they are and replace only the text. '
            'The timecode tells you how long the line is spoken: keep the translation short enough to be said in '
            'that time, without adding words the speaker did not say. '
            f'Reply with the blocks only — no notes, no quotes, no explanations. '
            f'{script_part}{glossary_part}{context_part}\n\n'
            f'{blocks}'
        )

    numbered = '\n'.join(f'{index + 1}. {line}' for index, line in enumerate(lines))
    return (
        f'Translate each numbered subtitle line{source} into {target_language}. '
        'Reply with one line per item, each starting with its number and a period — '
        'for example "1. <translation>" — in the same order, never merged, never skipped. '
        f'No quotes, no transliteration, no notes, no explanations. '
        f'{script_part}{glossary_part}{context_part}\n'
        f'{numbered}'
    )


_NUMBERED = re.compile(r'^\s*(\d{1,3})\s*[.)\]:-]\s*(.+?)\s*$', re.MULTILINE)

# Gemini confuses Khmer with Thai on Chinese sources (both scripts are
# Indic-derived), and a Thai answer is worthless for a Khmer dub. Naming the
# wrong script and showing a Khmer sample anchors it; verified live.
_SCRIPT_GUARDS = {
    'km': (
        'Use Khmer (Cambodian) Unicode letters — for example សួស្តី — '
        'never Thai letters (ก ข ค ง), Chinese characters, or English words.'
    ),
}


def _script_guard(target_language: str) -> str:
    return _SCRIPT_GUARDS.get(_resolve_lang_code(target_language), '')


def parse_numbered_answer(reply: str, count: int) -> list[str | None]:
    """Map an answer back to the lines it belongs to (None when missing).

    Gemini usually keeps the numbering, but it regularly answers with the
    translations alone (blank-line separated). Both shapes are accepted; a shape
    whose item count does not match is treated as unusable rather than guessed.
    """
    raw = str(reply or '')
    found: dict[int, str] = {}
    # SRT blocks first: "3\n00:00:04,000 --> 00:00:05,200\ntext".
    for block in re.split(r'\n\s*\n', raw):
        rows = [row for row in block.strip().splitlines() if row.strip()]
        if len(rows) < 3 or not rows[0].strip().isdigit() or '-->' not in rows[1]:
            continue
        index = int(rows[0].strip())
        text = _ZERO_WIDTH.sub('', ' '.join(rows[2:])).strip().strip('"“”').strip()
        if 1 <= index <= count and text and index not in found:
            found[index] = text
    if found:
        return [found.get(index + 1) for index in range(count)]

    for match in _NUMBERED.finditer(raw):
        index = int(match.group(1))
        text = match.group(2).strip().strip('"“”').strip()
        if 1 <= index <= count and text and index not in found:
            found[index] = text
    if found:
        return [found.get(index + 1) for index in range(count)]

    def clean(value: str) -> str:
        return _ZERO_WIDTH.sub('', value).strip().strip('"“”').strip()

    blocks = [clean(block) for block in re.split(r'\n\s*\n', raw) if clean(block)]
    if len(blocks) == count:
        return blocks
    lines = [clean(line) for line in raw.splitlines() if clean(line)]
    if len(lines) == count:
        return lines
    return [None] * count


def _answer_is_translation(value: str | None, source: str, target_language: str) -> bool:
    """True when the answer is a real translation, not the source or a refusal.

    Gemini drifts on mixed-language batches: a mostly-Chinese film makes it answer
    the English lines in Chinese, echo them unchanged, or apologise. Those lines
    must go back into the retry batch instead of reaching the dub.
    """
    if value is None or not str(value).strip():
        return False
    return _translation_script_ok(str(value), source, '', target_language)


def _retry_indices(
    automator: Any,
    parsed: list[str | None],
    batch: list[str],
    indices: list[int],
    target_language: str,
    source_language: str,
    glossary: str,
    batch_cues: list[tuple[float, float]] | None,
    timeout_s: float,
) -> None:
    """Re-ask for `indices` in one message and keep only valid translations."""
    if not indices:
        return
    retry_cues = [batch_cues[index] for index in indices] if batch_cues else None
    try:
        retry = automator.ask(
            build_batch_prompt(
                [batch[index] for index in indices], target_language, source_language,
                glossary, cues=retry_cues, strict=True,
            ),
            timeout_s=timeout_s,
            lines=len(indices),
        )
    except Exception:
        return
    if not retry.get('success'):
        return
    again = parse_numbered_answer(retry.get('reply', ''), len(indices))
    for slot, index in enumerate(indices):
        if _answer_is_translation(again[slot], batch[index], target_language):
            parsed[index] = again[slot]


def translate_lines(
    lines: list[str],
    target_language: str = 'Khmer',
    source_language: str = '',
    timeout_s: float = MIN_BATCH_SECONDS,
    glossary: str = '',
    cues: list[tuple[float, float]] | None = None,
    context: list[str] | None = None,
) -> list[str | None]:
    """Translate subtitle lines through the selected profile, one message per batch.

    A line the answer skipped, or answered in the wrong language, is retried in
    one message with the rest of the bad lines (split into small groups when the
    group is large); if it is still missing the entry stays ``None`` so the caller
    can report it instead of dubbing it.
    """
    automator = GeminiWebAutomator()
    results: list[str | None] = []
    for start in range(0, len(lines), BATCH_SIZE):
        batch = lines[start:start + BATCH_SIZE]
        batch_cues = cues[start:start + BATCH_SIZE] if cues else None
        prompt = build_batch_prompt(batch, target_language, source_language, glossary,
                                    cues=batch_cues, context=context)
        answer = automator.ask(prompt, timeout_s=timeout_s, lines=len(batch))
        if not answer.get('success'):
            raise GeminiWebError(str(answer.get('message') or 'Gemini web did not answer.'), answer.get('status', 'error'))
        parsed = parse_numbered_answer(answer.get('reply', ''), len(batch))
        invalid = [
            index for index, item in enumerate(parsed)
            if not _answer_is_translation(item, batch[index], target_language)
        ]
        # A wrong-language answer is not a translation: drop it before the retry,
        # otherwise it would survive whenever the retry also fails.
        for index in invalid:
            parsed[index] = None
        if invalid:
            # Re-ask for whatever the answer skipped or wrote in the wrong
            # language. One message for a handful of lines; a large group is
            # split into small groups, because the same size that failed once
            # tends to fail again. Still a few chats, never one per line.
            if len(invalid) <= RETRY_BATCH_SIZE * 2:
                groups = [invalid]
            else:
                groups = [
                    invalid[position:position + RETRY_BATCH_SIZE]
                    for position in range(0, len(invalid), RETRY_BATCH_SIZE)
                ]
            for group in groups:
                _retry_indices(
                    automator, parsed, batch, group, target_language, source_language,
                    glossary, batch_cues, timeout_s,
                )
        # Last resort, capped: a single line at a time for at most three lines.
        leftovers = [
            index for index, item in enumerate(parsed)
            if not _answer_is_translation(item, batch[index], target_language)
        ][:3]
        for index in leftovers:
            single = automator.ask(
                build_translation_prompt(batch[index], target_language, source_language),
                timeout_s=timeout_s,
            )
            if single.get('success'):
                candidate = str(single.get('reply') or '').strip().strip('"“”') or None
                if _answer_is_translation(candidate, batch[index], target_language):
                    parsed[index] = candidate
        results.extend(parsed)
    return results


def build_translation_prompt(text: str, target_language: str = 'Khmer', source_language: str = '') -> str:
    """A subtitle-ready prompt: one line in, one translation out.

    Subtitles need the line itself, not a conversation about it, so the
    instruction asks for the translation only and forbids commentary.
    """
    source = f' from {source_language}' if source_language else ''
    guard = _script_guard(target_language)
    guard_part = f' {guard}' if guard else ''
    return (
        f'Translate this subtitle line{source} into {target_language}, even if the line is already in '
        'another language. '
        f'Reply with the translation only — written in {target_language}, no quotes, no transliteration, '
        f'no notes, no explanation, nothing else.{guard_part}\n'
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
        # One browser session at a time: launching another closes this one's page.
        with _ASK_LOCK:
            return self._ask_locked(text, timeout_s)

    def _ask_locked(self, text: str, timeout_s: float) -> dict[str, Any]:
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
                # Keep the tab alive until Gemini has committed the chat.
                _wait_for_conversation(page)
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


def _wait_for_conversation(page: Any, timeout_s: float | None = None) -> str:
    """Wait for ``/app/<id>`` (and a short grace period) so history keeps the chat."""
    timeout = CONVERSATION_SAVE_SECONDS if timeout_s is None else timeout_s
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if CONVERSATION_ID.search(_url(page)):
            try:
                page.wait_for_timeout(1200)
            except Exception:
                pass
            return _url(page)
        try:
            page.wait_for_timeout(300)
        except Exception:
            break
    return _url(page)


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
                text = _clean_answer_text(locator.last.inner_text())
                if text:
                    return text
        except Exception:
            continue
    return ''


# While Gemini works it renders its activity line ("Translating English to Khmer
# subtitles") and the "Gemini said" label before the answer itself. That header
# is stable for a few polls, so without this guard it was read as the answer on
# long batches — and every line in the batch then looked untranslated.
_STATUS_LINE = re.compile(
    r'^(?:gemini\s+said|translating\b.*|thinking\b.*|analysing\b.*|analyzing\b.*'
    r'|generating\b.*|searching\b.*|show thinking|stop response|working\b.*)$',
    re.IGNORECASE,
)


def _clean_answer_text(raw: str) -> str:
    """Drop Gemini's status header and keep the answer body."""
    text = _ZERO_WIDTH.sub('', str(raw or ''))
    lines = text.splitlines()
    while lines and (not lines[0].strip() or _STATUS_LINE.match(lines[0].strip())):
        lines.pop(0)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
