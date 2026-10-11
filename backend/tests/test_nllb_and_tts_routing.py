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
    assert order.index('mms_khm') < order.index('sapi5')


def test_english_never_routes_to_a_khmer_only_model():
    order = tts_service.candidate_providers('auto', 'en', None)
    assert 'mms_khm' not in order and 'mms_khm_base' not in order


def test_explicit_khmer_model_routes_to_itself_first():
    assert tts_service.candidate_providers('mms_khm_base', 'km', None)[0] == 'mms_khm_base'
