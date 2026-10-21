"""The gender classifier (JaesungHuh/voice-gender-classifier) and its wiring.

The ECAPA model replaces median-pitch F0 as the primary gender signal for
speaker clusters; these tests pin the voting, the confidence floor, the
pitch fallback, and — with the vendored architecture — that the published
forward pass runs on the installed torch/torchaudio stack. No network.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from backend.app import gender_classifier
from backend.app import speaker_diarization


@pytest.fixture
def no_weights(monkeypatch, tmp_path):
    """Point the weights path at an empty dir and record any download."""
    monkeypatch.setattr(gender_classifier, 'weights_path', lambda: tmp_path / 'model.safetensors')
    downloads: list[tuple[str, str]] = []

    def fake_download(repo_id, filename, **kwargs):
        downloads.append((repo_id, filename))
        path = gender_classifier.weights_path()
        path.write_bytes(b'weights')

    monkeypatch.setattr('backend.app.hf_progress.download_file', fake_download)
    return downloads


# --- Classifier voting ------------------------------------------------------

def test_classify_speaker_windows_votes_over_a_batch(monkeypatch):
    """Windows are batched once; female probability means 'Female' over 0.7."""
    batches: list = []

    class FakeModel:
        def to(self, device):
            return self

        def __call__(self, batch):
            batches.append(batch.shape[0])
            return batch.new_tensor([[-5.0, 5.0], [-5.0, 4.0]])  # both female

    monkeypatch.setattr(gender_classifier, '_model', None)
    monkeypatch.setattr(gender_classifier, '_get_model', lambda: FakeModel())
    windows = [np.zeros(32000, dtype=np.float32) for _ in range(3)]  # 2 s at 16 kHz
    result = gender_classifier.classify_speaker_windows(windows)
    assert result == {'gender': 'Female', 'confidence': 1.0}
    assert batches == [3]  # one batch, every window


def test_classify_returns_male_and_low_confidence_fallback(monkeypatch):
    class MaleModel(types.SimpleNamespace):
        def to(self): ...

    class FakeModel:
        def to(self, device):
            return self

        def __call__(self, batch):
            # logits nearly equal -> confidence below the floor
            return batch.new_tensor([[0.05, -0.05]])

    monkeypatch.setattr(gender_classifier, '_model', None)
    monkeypatch.setattr(gender_classifier, '_get_model', lambda: FakeModel())
    result = gender_classifier.classify_speaker_windows([np.zeros(16000, dtype=np.float32)])
    assert result is None  # unsure -> caller keeps its fallback


def test_classifies_male_when_the_first_logit_wins(monkeypatch):
    class FakeModel:
        def to(self, device):
            return self

        def __call__(self, batch):
            return batch.new_tensor([[5.0, -5.0]])

    monkeypatch.setattr(gender_classifier, '_model', None)
    monkeypatch.setattr(gender_classifier, '_get_model', lambda: FakeModel())
    result = gender_classifier.classify_speaker_windows([np.zeros(16000, dtype=np.float32)])
    assert result['gender'] == 'Male'


def test_short_and_missing_windows_are_not_a_signal(monkeypatch):
    assert gender_classifier.classify_speaker_windows([]) is None
    # A 0.25 s blip is below the minimum voiced length.
    assert gender_classifier.classify_speaker_windows([np.zeros(4000, dtype=np.float32)]) is None


# --- Download ---------------------------------------------------------------

def test_download_happens_once_into_the_model_store(no_weights):
    assert gender_classifier.is_downloaded() is False
    assert gender_classifier.ensure_downloaded() is True
    assert gender_classifier.is_downloaded() is True
    assert no_weights == [
        ('JaesungHuh/voice-gender-classifier', 'model.safetensors'),
    ]
    # Second call: the file is already there, no network again.
    assert gender_classifier.ensure_downloaded() is True
    assert len(no_weights) == 1


# --- Diarization fallback logic ---------------------------------------------

def test_classifier_beats_the_pitch_threshold(monkeypatch):
    """A deep female voice reads 'Male' in F0 — the model corrects it."""
    monkeypatch.setattr(
        gender_classifier, 'classify_speaker_windows',
        lambda windows: {'gender': 'Female', 'confidence': 0.93},
    )
    pitches = [90.0] * 50  # below the pitch threshold: heuristic says Male
    result = speaker_diarization._classify_gender(pitches, [np.zeros(1)])
    assert result['gender'] == 'Female'
    assert result['gender_source'] == 'ecapa'
    assert result['gender_confidence'] == 0.93
    assert result['median_f0_hz'] == 90.0  # kept for diagnostics


def test_pitch_fallback_when_the_classifier_abstains():
    pitches = [90.0] * 50
    result = speaker_diarization._classify_gender(pitches, None)
    assert result == {'gender': 'Male', 'median_f0_hz': 90.0}


def test_no_pitch_and_no_classifier_is_unknown():
    result = speaker_diarization._classify_gender([], None)
    assert result == {'gender': None, 'median_f0_hz': None}


# --- Vendored architecture ---------------------------------------------------

def test_ecapa_forward_runs_on_the_installed_stack():
    """Random weights are fine: only the wiring is verified, not the accuracy."""
    torch = pytest.importorskip('torch')
    torchaudio = pytest.importorskip('torchaudio')  # noqa: F841 - the model imports it
    from backend.app.ecapa_gender import ECAPA_gender

    model = ECAPA_gender(C=1024)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 16000))
    assert output.shape == (1, 2)
    probabilities = torch.softmax(output, dim=1)
    assert pytest.approx(1.0, abs=1e-5) == float(probabilities.sum())
