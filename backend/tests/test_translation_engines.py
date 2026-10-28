"""Translation engine choice: the chain order and the refusal before any work starts.

These guard the bug where choosing Argos for Khmer silently sent work to Google.
"""
from __future__ import annotations

from unittest import mock

import pytest

from backend.app import providers
from backend.app.providers import ArgosTranslationProvider, HybridTranslationProvider


def _chain_keys(preference: str) -> list[str]:
    return [key for key, _ in HybridTranslationProvider(preference=preference)._chain()]


def test_auto_chain_excludes_nllb_and_always_ends_with_google():
    # NLLB is opt-in; Auto must behave as it did before NLLB existed.
    keys = _chain_keys('auto')
    assert 'nllb' not in keys
    assert keys[-1] == 'google'


def test_auto_chain_puts_argos_before_google_when_argos_is_installed():
    # Argos is optional: it is only in the chain when the package is present.
    with mock.patch.object(providers, '_argos_available', return_value=True), \
            mock.patch.object(providers.ArgosTranslationProvider, '__init__', return_value=None):
        provider = HybridTranslationProvider(preference='auto')
        provider._offline = object()
        keys = [key for key, _ in provider._chain()]
    assert keys.index('argos') < keys.index('google')


def test_explicit_nllb_choice_runs_nllb_first_when_installed():
    provider = HybridTranslationProvider(preference='nllb')
    provider._nllb = object()  # the chain only checks presence here
    assert [key for key, _ in provider._chain()][0] == 'nllb'


def test_google_choice_never_blocked_by_the_gate():
    HybridTranslationProvider(preference='google').check_preferred_engine('Khmer', 'en')


def test_argos_choice_refused_when_pair_unavailable_and_google_untouched():
    provider = HybridTranslationProvider(preference='argos')
    provider._offline = object()
    with mock.patch.object(ArgosTranslationProvider, 'supports_pair', return_value=False), \
            mock.patch.object(providers.GoogleTranslateProvider, 'translate',
                              side_effect=AssertionError('Google must not be called')) as google:
        with pytest.raises(RuntimeError, match='no en -> Khmer language package'):
            provider.check_preferred_engine('Khmer', 'en')
    assert not google.called


def test_argos_choice_allowed_when_pair_available():
    provider = HybridTranslationProvider(preference='argos')
    provider._offline = object()
    with mock.patch.object(ArgosTranslationProvider, 'supports_pair', return_value=True):
        provider.check_preferred_engine('French', 'en')


def test_nllb_choice_refuses_unknown_language_with_clear_message():
    provider = HybridTranslationProvider(preference='nllb')
    provider._nllb = object()
    with pytest.raises(RuntimeError, match='NLLB has no code for language: Klingon'):
        provider.check_preferred_engine('Klingon', 'en')


def test_first_engine_reports_the_engine_that_will_run():
    assert HybridTranslationProvider(preference='google').first_engine() == 'google'


@pytest.mark.parametrize('preference', ['qwen', 'nllb', 'argos'])
def test_local_choice_never_falls_back_to_google(preference):
    """Picking Qwen/NLLB/Argos is a promise: no line goes silently online."""
    provider = HybridTranslationProvider(preference=preference)
    provider._qwen = provider._qwen or object()
    assert 'google' not in [key for key, _ in provider._chain()]


def test_failed_local_engine_is_not_replaced_by_google():
    provider = HybridTranslationProvider(preference='qwen')

    class Broken:
        def translate(self, text, target_language, source_language='en'):
            raise RuntimeError('qwen exploded')

    provider._qwen = Broken()
    provider._offline = None
    provider._nllb = None
    with mock.patch.object(providers.GoogleTranslateProvider, 'translate',
                           side_effect=AssertionError('Google must not be called')) as google:
        with pytest.raises(RuntimeError, match='qwen exploded'):
            provider.translate('hello', 'Khmer', 'en')
    assert not google.called


def test_qwen_choice_refuses_when_llama_cpp_cannot_start(monkeypatch):
    from backend.app import qwen_local

    monkeypatch.setattr(qwen_local, 'is_installed', lambda: True)
    monkeypatch.setattr(qwen_local, 'runtime_problem',
                        lambda: 'this llama.cpp build needs the NVIDIA CUDA runtime')
    provider = HybridTranslationProvider(preference='qwen')
    with pytest.raises(RuntimeError, match='cannot start on this machine'):
        provider.check_preferred_engine('Khmer', 'zh')


def test_auto_skips_a_qwen_that_cannot_load_instead_of_retrying_every_line(monkeypatch):
    """A broken local engine must not be reported as the running engine."""
    from backend.app import qwen_local

    monkeypatch.setattr(qwen_local, 'is_installed', lambda: True)
    monkeypatch.setattr(qwen_local, 'runtime_problem', lambda: 'llama.dll failed to load')
    provider = HybridTranslationProvider(preference='auto')
    assert provider._qwen is None
    assert provider.first_engine() != 'qwen'
