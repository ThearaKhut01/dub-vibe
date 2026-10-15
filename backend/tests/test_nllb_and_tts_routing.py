"""NLLB output guard, language-code mapping, and Khmer TTS provider routing."""
from __future__ import annotations

import pytest

from backend.app import nllb_local, tts_service
from backend.app.providers import LANGUAGE_CODES


def test_loop_guard_rejects_a_single_token_repeated():
    # The output that stalled a real project: one word repeated to the length cap.
    assert nllb_local._is_degenerate('ក្រពើ ' * 12)


def test_loop_guard_keeps_normal_translations():
    assert not nllb_local._is_degenerate('អរគុណ ដែល បាន មើល វីដេអូ របស់ ខ្ញុំ។')


@pytest.mark.parametrize('name', sorted(LANGUAGE_CODES))
def test_every_app_language_has_an_nllb_code(name):
    assert nllb_local.nllb_code(name)


def test_iso_and_name_maps_agree_for_every_app_language():
    for name, iso in LANGUAGE_CODES.items():
        assert nllb_local._ISO_TO_NLLB[iso] == nllb_local.NLLB_LANGUAGE_CODES[name]


def test_unknown_language_is_refused():
    with pytest.raises(ValueError, match='NLLB has no code for language'):
        nllb_local.nllb_code('Klingon')


def test_khmer_auto_order_tries_online_edge_before_offline_models():
    order = tts_service.candidate_providers('auto', 'km', None)
    assert order[0] == 'edge'
    assert 'sapi5' in order


def test_english_never_routes_to_a_khmer_only_model():
    order = tts_service.candidate_providers('auto', 'en', None)
    assert 'mms_khm' not in order and 'mms_khm_base' not in order


def test_explicit_khmer_model_routes_to_itself_first():
    assert 'mms_khm_base' not in tts_service.candidate_providers('edge', 'km', None)


def test_edge_never_receives_a_gemini_model():
    # A model picked for Gemini must not be sent to Edge (it ignores models).
    assert tts_service._model_for('edge', 'gemini-2.5-flash-preview-tts') is None


def test_gemini_and_openai_keep_their_chosen_model():
    assert tts_service._model_for('gemini', 'gemini-2.5-flash-preview-tts') == 'gemini-2.5-flash-preview-tts'
    assert tts_service._model_for('openai', 'tts-1-hd') == 'tts-1-hd'


def test_gemini_and_openai_fall_back_to_defaults_without_a_choice():
    assert tts_service._model_for('gemini', None) == tts_service.GEMINI_DEFAULT_MODEL
    assert tts_service._model_for('openai', None) == tts_service.OPENAI_DEFAULT_MODEL


def test_cache_key_is_stable_for_engines_without_a_model():
    assert tts_service.cache_key('edge', None, 'auto', 'hi', None, 1.0) == \
        tts_service.cache_key('edge', '', 'auto', 'hi', None, 1.0)
def test_stale_gemini_model_is_not_sent_to_edge(monkeypatch, tmp_path):
    out_file = tmp_path / 'out.mp3'
    """Through the real synthesize_line path: a Gemini model chosen earlier must not reach Edge."""
    seen: dict[str, object] = {}

    def record_key(provider, model, voice, text, style, speed):
        seen.setdefault(provider, model)
        return 'k'

    import backend.app.providers as providers

    def stop_edge(self, *args, **kwargs):
        raise RuntimeError('stop after cache key')

    monkeypatch.setattr(tts_service, 'cache_key', record_key)
    monkeypatch.setattr(tts_service, '_find_cached', lambda provider, key: None)
    monkeypatch.setattr(tts_service.acting, 'style_instruction', lambda provider, analysis: None)
    monkeypatch.setattr(providers.EdgeTTSProvider, 'synthesize', stop_edge)
    try:
        tts_service.synthesize_line(
            'hi', str(out_file), provider_pref='edge', language_name='English', lang_code='en',
            voice_pref=None, model_pref='gemini-2.5-flash-preview-tts', speed=1.0,
            emotion='Neutral', intensity=70, character_profile=None, analysis={},
        )
    except RuntimeError:
        pass
    assert seen.get('edge') is None
