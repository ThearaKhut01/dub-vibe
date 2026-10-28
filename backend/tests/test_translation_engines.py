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


def _fake_argos(monkeypatch, installed=(), catalogue=()):
    """Point Argos at a fake catalogue and count index refreshes."""
    from argostranslate import package as argos_package, translate as argos_translate

    calls = {'index': 0}

    def update_index():
        calls['index'] += 1

    monkeypatch.setattr(argos_package, 'update_package_index', update_index)
    monkeypatch.setattr(argos_package, 'get_available_packages', lambda: list(catalogue))
    monkeypatch.setattr(argos_translate, 'get_installed_languages', lambda: list(installed))
    monkeypatch.setattr(ArgosTranslationProvider, '_configure_settings', staticmethod(lambda: None))
    providers.reset_argos_cache()
    return calls


def test_argos_missing_pair_is_not_rechecked_for_every_line(monkeypatch):
    """A Khmer run without the zh->km package must not hit the network per line."""
    calls = _fake_argos(monkeypatch)
    try:
        for _ in range(25):
            assert ArgosTranslationProvider._ensure_package('zh', 'km') is False
        assert calls['index'] == 1
    finally:
        providers.reset_argos_cache()


def test_argos_missing_pair_is_retried_after_the_cache_expires(monkeypatch):
    import time

    calls = _fake_argos(monkeypatch)
    try:
        assert ArgosTranslationProvider._ensure_package('zh', 'km') is False
        assert calls['index'] == 1
        expired = time.monotonic() - providers._ARGOS_INDEX_TTL - 1
        providers._ARGOS_INDEX_AT = expired
        providers._ARGOS_MISSING_PAIRS[('zh', 'km')] = expired
        assert ArgosTranslationProvider._ensure_package('zh', 'km') is False
        assert calls['index'] == 2
    finally:
        providers.reset_argos_cache()


def test_argos_installed_pair_needs_no_catalogue_refresh(monkeypatch):
    calls = _fake_argos(monkeypatch, installed=[object()])
    monkeypatch.setattr(ArgosTranslationProvider, '_find_installed_translation',
                        staticmethod(lambda installed, source, target: object()))
    try:
        assert ArgosTranslationProvider._ensure_package('en', 'km') is True
        assert calls['index'] == 0
    finally:
        providers.reset_argos_cache()


def test_nllb_choice_refuses_unknown_language_with_clear_message():
    provider = HybridTranslationProvider(preference='nllb')
    provider._nllb = object()
    with pytest.raises(RuntimeError, match='NLLB has no code for language: Klingon'):
        provider.check_preferred_engine('Klingon', 'en')


def test_first_engine_reports_the_engine_that_will_run():
    assert HybridTranslationProvider(preference='google').first_engine() == 'google'


def test_lines_are_serial_when_google_is_reachable(monkeypatch):
    """Parallel requests are what earns Google's 429, so free engines run one at a time."""
    assert HybridTranslationProvider(preference='google').max_parallel_lines() == 1
    with mock.patch.object(providers, '_argos_available', return_value=True):
        assert HybridTranslationProvider(preference='auto').max_parallel_lines() == 1


def test_paid_keyed_engine_still_uses_a_small_pool(monkeypatch):
    with mock.patch.object(providers.CodeCraftTranslationProvider, 'is_available', return_value=True):
        provider = HybridTranslationProvider(preference='codecraft')
    assert provider.max_parallel_lines() == 4


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
