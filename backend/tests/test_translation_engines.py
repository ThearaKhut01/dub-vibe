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
