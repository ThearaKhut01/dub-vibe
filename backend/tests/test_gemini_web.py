"""The prompt→answer mechanics, against a stand-in for Gemini's DOM.

The real end-to-end test needs a profile that is signed in to Gemini; these
tests cover everything around that: the line is typed into the prompt box, Enter
is pressed, the new answer is read once it stops growing, and the Live node
records one success.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from backend.app import api_events, chrome_profiles, gemini_web

SUBTITLE = "Where were you last night?"
ANSWER = "តើអ្នកនៅឯណាពេលយប់មិញ?"


class _Box:
    def __init__(self, page) -> None:
        self.page = page

    def click(self) -> None:
        self.page.clicked = True

    def fill(self, text: str) -> None:
        self.page.typed = text
        self.page.entered = True
        # Gemini moves the tab to the conversation id once the chat exists.
        self.page.url = 'https://gemini.google.com/app/abc123def456'

    def inner_text(self) -> str:
        return self.page.answer


class _Nodes:
    def __init__(self, page, count: int) -> None:
        self.page = page
        self._count = count

    def count(self) -> int:
        return self._count

    @property
    def first(self) -> _Box:
        return _Box(self.page)

    @property
    def last(self) -> _Box:
        return _Box(self.page)


class _FakePage:
    """What the automator uses: goto, wait, locator, keyboard, url."""

    def __init__(self, answer: str = ANSWER) -> None:
        self.url = 'https://gemini.google.com/app'
        self.answer = answer
        self.typed = ''
        self.clicked = False
        self.entered = False
        self.keyboard = SimpleNamespace(press=lambda key: setattr(self, 'key', key))

    def goto(self, url: str, **_kwargs) -> None:
        self.url = url

    def wait_for_timeout(self, _ms: int) -> None:
        return None

    def locator(self, selector: str) -> _Nodes:
        if selector == 'rich-textarea':
            return _Nodes(self, 1)
        if selector in gemini_web.RESPONSE_SELECTORS:
            return _Nodes(self, 1 if self.entered else 0)
        return _Nodes(self, 0)

    def get_by_role(self, _role: str, name=None) -> _Nodes:
        return _Nodes(self, 0)


class _Session:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.profile = {'id': 'abc123'}
        self.via = 'working_copy'

    def __enter__(self) -> '_Session':
        return self

    def __exit__(self, *_exc) -> None:
        return None


class _Manager:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.config = SimpleNamespace(read=lambda: {'selected_profile_id': 'abc123'})

    def launch_selected_profile(self, headless=None) -> _Session:
        return _Session(self.page)


@pytest.fixture(autouse=True)
def _clean_events():
    api_events.reset_for_tests()
    yield
    api_events.reset_for_tests()


def test_a_subtitle_line_is_typed_and_the_answer_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))
    page = _FakePage()

    result = gemini_web.GeminiWebAutomator(_Manager(page)).ask(SUBTITLE)

    assert result['success'] is True
    assert result['reply'] == ANSWER
    assert page.typed == SUBTITLE, 'the subtitle must go into the prompt box verbatim'
    assert page.clicked and page.key == 'Enter'


def test_the_success_is_recorded_on_the_profile_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))

    gemini_web.GeminiWebAutomator(_Manager(_FakePage())).ask(SUBTITLE)

    node = api_events.snapshot()['nodes'][0]
    assert node['id'] == 'chrome:abc123'
    assert node['requests'] == 1
    assert node['successes'] == 1
    assert node['failures'] == 0


def test_a_missing_prompt_box_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))

    monkeypatch.setattr(gemini_web, 'PROMPT_WAIT_SECONDS', 0.1)

    class EmptyPage(_FakePage):
        def locator(self, selector: str) -> _Nodes:
            return _Nodes(self, 0)

    result = gemini_web.GeminiWebAutomator(_Manager(EmptyPage())).ask(SUBTITLE)

    assert result['success'] is False
    assert result['status'] == 'error'
    assert 'prompt box was not found' in result['message']


def test_an_empty_answer_is_not_reported_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))
    monkeypatch.setattr(gemini_web, 'MIN_ANSWER_SECONDS', 0.2)
    page = _FakePage(answer='')

    result = gemini_web.GeminiWebAutomator(_Manager(page)).ask(SUBTITLE, timeout_s=0.2)

    assert result['success'] is False
    assert 'no answer appeared' in result['message']


def test_concurrent_prompts_wait_for_the_single_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second prompt must not launch a session that closes the first one's page."""
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))

    state = {'active': 0, 'peak': 0}
    guard = threading.Lock()

    class GuardedSession(_Session):
        def __enter__(self) -> 'GuardedSession':
            with guard:
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
            time.sleep(0.05)
            return self

        def __exit__(self, *_exc) -> None:
            with guard:
                state['active'] -= 1
            return None

    class GuardedManager(_Manager):
        def launch_selected_profile(self, headless=None) -> _Session:
            return GuardedSession(self.page)

    results: list[dict] = []

    def run() -> None:
        results.append(gemini_web.GeminiWebAutomator(GuardedManager(_FakePage())).ask(SUBTITLE))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state['peak'] == 1, 'two browser sessions were open at the same time'
    assert [result['success'] for result in results] == [True, True]


def test_translation_prompt_asks_for_the_line_only() -> None:
    prompt = gemini_web.build_translation_prompt('Hello', 'Khmer')

    assert 'Translate this subtitle line' in prompt
    assert 'into Khmer' in prompt
    assert 'translation only' in prompt
    assert prompt.rstrip().endswith('Hello')


def test_the_gemini_said_label_is_not_part_of_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))
    page = _FakePage(answer='Gemini said\n\nយប់មិញអ្នកនៅឯណា?')

    result = gemini_web.GeminiWebAutomator(_Manager(page)).ask(SUBTITLE)

    assert result['reply'] == 'យប់មិញអ្នកនៅឯណា?'


def test_a_status_header_is_not_read_as_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini's "Translating..." line is stable before the answer exists."""
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))
    monkeypatch.setattr(gemini_web, 'MIN_ANSWER_SECONDS', 5.0)

    class HeaderPage(_FakePage):
        def __init__(self) -> None:
            super().__init__(answer='')
            self.reads = 0

        @property
        def answer(self) -> str:
            self.reads += 1
            if self.reads <= 3:
                return 'Translating English to Khmer subtitles\nGemini said'
            return ANSWER

        @answer.setter
        def answer(self, value: str) -> None:
            self._answer = value

    result = gemini_web.GeminiWebAutomator(_Manager(HeaderPage())).ask(SUBTITLE)

    assert result['success'] is True
    assert result['reply'] == ANSWER


def test_a_batch_prompt_numbers_every_line() -> None:
    prompt = gemini_web.build_batch_prompt(['One', 'Two', 'Three'], 'Khmer')

    assert '1. One' in prompt
    assert '2. Two' in prompt
    assert '3. Three' in prompt
    assert 'starting with its number' in prompt


def test_a_saved_conversation_is_waited_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(chrome_profiles, 'gemini_state', lambda page: ('ready', 'signed in', page.url))
    page = _FakePage()
    seen: dict = {}
    original = gemini_web._wait_for_conversation

    def spy(target_page, timeout_s=None):
        seen['url'] = target_page.url
        return original(target_page, 0)

    monkeypatch.setattr(gemini_web, '_wait_for_conversation', spy)

    gemini_web.GeminiWebAutomator(_Manager(page)).ask(SUBTITLE)

    assert '/app/abc123def456' in seen['url'], 'the chat id must exist before the tab closes'


def test_the_glossary_reaches_the_prompt() -> None:
    prompt = gemini_web.build_batch_prompt(['One'], 'Khmer', glossary='Bye = លាហើយ')

    assert 'Use these term translations exactly: Bye = លាហើយ' in prompt


def test_numbered_answers_are_mapped_back_to_their_lines() -> None:
    reply = '1. មួយ\n2) ពីរ\n4: បួន'

    assert gemini_web.parse_numbered_answer(reply, 4) == ['មួយ', 'ពីរ', None, 'បួន']


def test_zero_width_spaces_are_removed_from_answers() -> None:
    reply = 'កុំ\u200bឱ្យ\u200bគាត់\u200bចេញពីបន្ទប់នេះ។'

    assert gemini_web.parse_numbered_answer(reply, 1) == ['កុំឱ្យគាត់ចេញពីបន្ទប់នេះ។']


def test_an_unnumbered_answer_is_mapped_by_position() -> None:
    reply = 'ថ្ងៃម្សិលមិញអ្នកទៅណា?\n\nខ្ញុំនៅការិយាល័យ។'

    assert gemini_web.parse_numbered_answer(reply, 2) == ['ថ្ងៃម្សិលមិញអ្នកទៅណា?', 'ខ្ញុំនៅការិយាល័យ។']


def test_a_wrong_count_is_not_guessed() -> None:
    assert gemini_web.parse_numbered_answer('one\n\ntwo\n\nthree', 2) == [None, None]


def test_the_skipped_lines_are_retried_in_one_message(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeAutomator:
        def ask(self, prompt, timeout_s=180.0, lines=1):
            calls.append(prompt)
            if '2. I was at the office.' in prompt:
                # First answer: numbered but drops line 2.
                return {'success': True, 'reply': '1. យប់មិញអ្នកនៅឯណា?'}
            if prompt.startswith('Translate each numbered subtitle line') and prompt.rstrip().endswith('I was at the office.'):
                # Retry message contains only the missing line.
                return {'success': True, 'reply': '1. ខ្ញុំនៅការិយាល័យ។'}
            return {'success': True, 'reply': 'x'}

    monkeypatch.setattr(gemini_web, 'GeminiWebAutomator', FakeAutomator)

    results = gemini_web.translate_lines(['Where were you last night?', 'I was at the office.'], 'Khmer')

    assert results == ['យប់មិញអ្នកនៅឯណា?', 'ខ្ញុំនៅការិយាល័យ។']
    assert len(calls) == 2, 'the batch, then one retry message — not one per line'


def test_the_prompt_forbids_carrying_the_source_language_over() -> None:
    prompt = gemini_web.build_batch_prompt(['Well played boy'], 'Khmer', 'Chinese')

    assert 'even when a source line is already in another language' in prompt
    assert 'Never copy a source line unchanged' in prompt


def test_a_wrong_language_answer_is_retried_in_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Chinese/English answer must not be accepted as Khmer — ask again once."""
    calls: list[str] = []

    class FakeAutomator:
        def ask(self, prompt, timeout_s=180.0, lines=1):
            calls.append(prompt)
            if 'previous attempt' in prompt:
                return {'success': True, 'reply': '1. លេងបានល្អណាស់ កូនក្មេង។'}
            # First answer: line 1 came back in Chinese, line 2 in Khmer.
            return {'success': True, 'reply': '1. 做得好，小子\n2. មកដល់ហើយ។'}

    monkeypatch.setattr(gemini_web, 'GeminiWebAutomator', FakeAutomator)

    results = gemini_web.translate_lines(['Well played boy', 'We are here.'], 'Khmer')

    assert results == ['លេងបានល្អណាស់ កូនក្មេង។', 'មកដល់ហើយ។']
    assert len(calls) == 2, 'the wrong line is retried inside one batch message'


def test_a_large_skipped_group_is_retried_in_small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 20-line answer that skips most lines must not be re-asked as 20 again."""
    calls: list[int] = []

    class FakeAutomator:
        def ask(self, prompt, timeout_s=180.0, lines=1):
            calls.append(lines)
            if lines > gemini_web.RETRY_BATCH_SIZE * 2:
                return {'success': True, 'reply': '1. បន្ទាត់ទី១'}
            body = '\n'.join(f'{index}. បន្ទាត់ទី{index}' for index in range(1, lines + 1))
            return {'success': True, 'reply': body}

    monkeypatch.setattr(gemini_web, 'GeminiWebAutomator', FakeAutomator)

    results = gemini_web.translate_lines([f'line {index}' for index in range(20)], 'Khmer')

    assert all(results), 'every line must be filled by the small retry groups'
    assert calls[0] == 20
    assert all(size <= gemini_web.RETRY_BATCH_SIZE for size in calls[1:]), calls


def test_an_echoed_or_refused_line_is_not_returned_as_a_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source copied back and an apology are missing lines, never subtitles."""

    class FakeAutomator:
        def ask(self, prompt, timeout_s=180.0, lines=1):
            if prompt.startswith('Translate this subtitle line'):
                return {'success': True, 'reply': "I'm having a hard time fulfilling your request."}
            return {'success': True, 'reply': '1. Now it is time to die'}

    monkeypatch.setattr(gemini_web, 'GeminiWebAutomator', FakeAutomator)

    results = gemini_web.translate_lines(["Now it's time to die"], 'Khmer')

    assert results == [None], 'a copied English line would be dubbed as English'


def test_the_browser_engine_has_its_own_cache_namespace() -> None:
    from backend.app import main

    assert main._translation_model_hint('gemini_web') == gemini_web.MODEL_LABEL
    # Free engines share the "no model choice" key; the browser engine must not.
    assert main._translation_model_hint('google') == ''
    assert main._translation_model_hint('gemini_web') != main._translation_model_hint('google')


def test_chat_titles_drop_navigation_and_duplicates() -> None:
    raw = [
        'Gemini', 'New chat', 'Library', 'Customise',
        'ការបកប្រែប្រយោគអត្ថបទរងជាភាសាខ្មែរ',
        'ការបកប្រែប្រយោគអត្ថបទរងជាភាសាខ្មែរ',
        '  Where   is the station?  ',
        'New Chat',
    ]

    titles = gemini_web.clean_chat_titles(raw, limit=10)

    assert titles == ['ការបកប្រែប្រយោគអត្ថបទរងជាភាសាខ្មែរ', 'Where is the station?']


def test_chat_titles_respect_the_limit() -> None:
    assert gemini_web.clean_chat_titles([f'chat {i}' for i in range(30)], limit=5) == [f'chat {i}' for i in range(5)]


def test_the_engine_needs_a_selected_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import providers

    monkeypatch.setattr(chrome_profiles, 'playwright_available', lambda: True)
    monkeypatch.setattr(
        chrome_profiles, 'manager',
        lambda: SimpleNamespace(config=SimpleNamespace(read=lambda: {'selected_profile_id': ''})),
    )
    assert providers.GeminiWebTranslationProvider.is_available() is False

    monkeypatch.setattr(
        chrome_profiles, 'manager',
        lambda: SimpleNamespace(config=SimpleNamespace(read=lambda: {'selected_profile_id': 'abc'})),
    )
    assert providers.GeminiWebTranslationProvider.is_available() is True


def test_the_browser_engine_runs_one_batch_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import providers

    monkeypatch.setattr(providers.GeminiWebTranslationProvider, 'is_available', staticmethod(lambda: True))
    translator = providers.HybridTranslationProvider(preference='gemini_web')

    assert translator.first_engine() == 'gemini_web'
    assert translator.max_parallel_lines() == 1


def test_auto_never_picks_the_browser_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import providers

    monkeypatch.setattr(providers.GeminiWebTranslationProvider, 'is_available', staticmethod(lambda: True))
    translator = providers.HybridTranslationProvider(preference='auto')

    chain = [key for key, _provider in translator._chain()]
    assert 'gemini_web' not in chain


def test_a_batch_with_cues_is_sent_as_srt_blocks() -> None:
    prompt = gemini_web.build_batch_prompt(
        ['Where were you?', 'At the office.'],
        'Khmer', 'Chinese',
        cues=[(1.0, 2.4), (2.4, 3.2)],
    )

    assert '1\n00:00:01,000 --> 00:00:02,400\nWhere were you?' in prompt
    assert '2\n00:00:02,400 --> 00:00:03,200\nAt the office.' in prompt
    assert 'Keep each block number and its timecode' in prompt


def test_an_srt_answer_is_parsed_back() -> None:
    reply = (
        '1\n00:00:01,000 --> 00:00:02,400\nយប់មិញអ្នកនៅឯណា?\n\n'
        '2\n00:00:02,400 --> 00:00:03,200\nខ្ញុំនៅការិយាល័យ។'
    )

    assert gemini_web.parse_numbered_answer(reply, 2) == ['យប់មិញអ្នកនៅឯណា?', 'ខ្ញុំនៅការិយាល័យ។']


def test_cues_and_context_reach_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    class FakeAutomator:
        def ask(self, prompt, timeout_s=180.0, lines=1):
            prompts.append(prompt)
            return {'success': True, 'reply': '1\n00:00:01,000 --> 00:00:02,400\nមួយ'}

    monkeypatch.setattr(gemini_web, 'GeminiWebAutomator', FakeAutomator)

    gemini_web.translate_lines(
        ['Where were you?'], 'Khmer', 'Chinese',
        cues=[(1.0, 2.4)], context=['Earlier line.'],
    )

    assert '00:00:01,000 --> 00:00:02,400' in prompts[0]
    assert 'Earlier line.' in prompts[0]


