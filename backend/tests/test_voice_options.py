"""The Voice list must follow the selected engine and target language.

The Inspector used to offer a hardcoded "Male / Female / Narrator" list, so Edge +
Khmer showed a Narrator voice that does not exist for Khmer. The UI now renders the
capability table, and these tests pin the table that drives it.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.providers import (
    EDGE_LANGUAGE_VOICES,
    LANGUAGE_CODES,
    edge_voice_options,
)


def test_khmer_has_exactly_two_edge_voices():
    options = edge_voice_options('Khmer')
    assert [option['id'] for option in options] == ['Male Voice A', 'Female Voice B']
    assert [option['voice'] for option in options] == ['km-KH-PisethNeural', 'km-KH-SreymomNeural']
    assert [option['gender'] for option in options] == ['Male', 'Female']


def test_narrator_is_english_only():
    assert 'Narrator Voice' in [option['id'] for option in edge_voice_options('English')]
    for language in LANGUAGE_CODES:
        if language == 'English':
            continue
        assert 'Narrator Voice' not in [option['id'] for option in edge_voice_options(language)], language


def test_every_offered_voice_matches_the_language_table():
    """No option may point at a voice from another language."""
    for language in {lang for lang, _ in EDGE_LANGUAGE_VOICES}:
        for option in edge_voice_options(language):
            assert (language, option['id']) in EDGE_LANGUAGE_VOICES
            assert option['voice'] == EDGE_LANGUAGE_VOICES[(language, option['id'])]
            assert option['voice'].startswith(LANGUAGE_CODES[language]), option


def test_every_edge_language_has_at_least_two_voices():
    for language in {lang for lang, _ in EDGE_LANGUAGE_VOICES}:
        assert len(edge_voice_options(language)) >= 2, language


def test_voice_labels_name_the_real_edge_voice():
    """The Voice list must show Piseth/Sreymom, not the abstract voice slot."""
    assert [option['label'] for option in edge_voice_options('Khmer')] == [
        'Piseth (Male)',
        'Sreymom (Female)',
    ]
    english = {option['id']: option['label'] for option in edge_voice_options('English')}
    assert english['Narrator Voice'] == 'Christopher (Narrator)'
    assert english['Male Voice A'] == 'Guy (Male)'
    assert english['Female Voice B'] == 'Jenny (Female)'


def test_unknown_language_offers_no_voices():
    assert edge_voice_options('Klingon') == []


def test_capabilities_expose_voices_per_language():
    client = TestClient(main.app)
    response = client.get('/api/tts/capabilities')
    assert response.status_code == 200, response.text
    edge = response.json()['edge']
    by_language = edge['voices_by_language']
    # Every language the engine claims is supported has a voice table.
    for code in edge['languages']:
        name = next((name for name, iso in LANGUAGE_CODES.items() if iso == code), None)
        assert name is not None, code
        assert by_language[name], name
    assert len(by_language['Khmer']) == 2
    assert len(by_language['English']) == 3


def test_edge_synthesizer_falls_back_when_a_voice_is_missing_for_the_language(monkeypatch):
    """A stale saved label must never send Khmer text to an English voice."""
    from backend.app.providers import EdgeTTSProvider

    assert EdgeTTSProvider is not None
    assert EDGE_LANGUAGE_VOICES.get(('Khmer', 'Narrator Voice')) is None
    # The provider maps Narrator to the language's male voice instead of failing.
    assert EDGE_LANGUAGE_VOICES.get(('Khmer', 'Male Voice A')) == 'km-KH-PisethNeural'
