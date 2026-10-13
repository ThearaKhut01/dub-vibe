"""VoxCPM2 offline multilingual TTS: routing, language gating, and model store."""
from __future__ import annotations

import pytest

from backend.app import tts_service, voxcpm_local


def test_voxcpm_sits_after_edge_and_before_khmer_only_models():
    order = tts_service.QUALITY_ORDER
    assert order.index('edge') < order.index('voxcpm2') < order.index('mms_khm')


def test_auto_routing_for_english_prefers_edge_over_voxcpm():
    order = tts_service.candidate_providers('auto', 'en', None)
    assert order.index('edge') < order.index('voxcpm2')


def test_explicit_voxcpm_preference_routes_to_itself_first():
    assert tts_service.candidate_providers('voxcpm2', 'en', None)[0] == 'voxcpm2'


def test_voxcpm_is_offered_for_english():
    assert 'voxcpm2' in tts_service.candidate_providers('auto', 'en', None)


@pytest.mark.parametrize('lang', ['km', 'th', 'vi', 'zh'])
def test_voxcpm_is_skipped_for_languages_it_was_not_verified_for(lang):
    # Khmer and Thai produced garbled audio in verification, so they must not route here.
    assert 'voxcpm2' not in tts_service.candidate_providers('auto', lang, None)


def test_voxcpm_language_gate_matches_verified_set():
    assert voxcpm_local.supports_language('en')
    assert not voxcpm_local.supports_language('km')


def test_voxcpm_default_voice_is_auto_regardless_of_request():
    assert tts_service._default_voice('voxcpm2', 'Male Voice A') == 'auto'
    assert tts_service._default_voice('voxcpm2', None) == 'auto'


def test_voxcpm_model_dir_lives_in_the_shared_model_store(tmp_path, monkeypatch):
    monkeypatch.setenv('DUBVIBE_MODELS_DIR', str(tmp_path))
    assert voxcpm_local.model_dir() == tmp_path / 'voxcpm' / 'VoxCPM2'


def test_voxcpm_is_not_installed_without_weights(tmp_path, monkeypatch):
    monkeypatch.setenv('DUBVIBE_MODELS_DIR', str(tmp_path))
    assert voxcpm_local.is_installed() is False
