"""Voice gender classification (male/female) from the original speaker audio.

Model: JaesungHuh/voice-gender-classifier — an ECAPA-TDNN speaker encoder
finetuned on VoxCeleb2 as a two-class gender classifier (98.7% accuracy on
the VoxCeleb1 test split, MIT license). It is the primary gender signal for
speaker clusters: the median-pitch F0 heuristic fails for low-pitched
females and high-pitched males, while this model hears the whole voice. F0
stays as the fallback whenever the model is absent or uncertain, so behavior
degrades to exactly the old heuristic instead of getting worse.

The weights (~62 MB) are downloaded once into the shared model store and every
run after that is fully offline. The architecture lives in `ecapa_gender.py`.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .providers import get_storage_dir

logger = logging.getLogger(__name__)

_REPO_ID = 'JaesungHuh/voice-gender-classifier'
# Public alias: the model manager panel shows it as the provenance of the download.
REPO_ID = _REPO_ID
_MODEL_FILE = 'model.safetensors'
# Pinned revision: a re-trained upstream model can never silently change
# dub-voice decisions on existing installs until the pin is updated here.
_REVISION = 'db1222153bd60337e900be22add7af180452adc0'

# The model's vote across a speaker's windows must be at least this sure
# before it overrides the pitch heuristic; anything lower is not a signal.
_MIN_CONFIDENCE = 0.70
# A short window (half a second of speech) is already enough; more is not
# needed — the vote over a handful is effectively the speaker's real voice.
_MAX_WINDOWS_PER_SPEAKER = 6

_lock = threading.Lock()
_model: Any = None  # ECAPA_gender, lazy-loaded once per process


def weights_path() -> Path:
    # Keeps the Hub's own filename: `download_file` streams into
    # <local_dir>/<filename>, so the two must agree for is_downloaded().
    return get_storage_dir('models') / 'gender' / 'model.safetensors'


def is_available() -> bool:
    """True when torch and torchaudio — the model's only runtime deps — import."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec('torch') is not None
            and importlib.util.find_spec('torchaudio') is not None
        )
    except Exception:
        return False


def is_downloaded() -> bool:
    return weights_path().is_file()


def ensure_downloaded(progress_cb=None) -> bool:
    """Download the weights once; True when the classifier is ready to run.

    Mirrors Faster-Whisper: the model arrives on first use, then runs fully
    offline. A failed download is not fatal — callers fall back to pitch.
    """
    if not is_available() or is_downloaded():
        return is_available() and is_downloaded()
    from . import hf_progress

    try:
        hf_progress.download_file(
            _REPO_ID, _MODEL_FILE,
            local_dir=str(weights_path().parent),
            revision=_REVISION,
            progress_cb=progress_cb,
        )
    except Exception as error:
        logger.warning('Gender classifier download failed: %s', error)
    return is_downloaded()


def _get_model() -> Any:
    global _model
    with _lock:
        if _model is None:
            from safetensors.torch import load_file

            from .ecapa_gender import ECAPA_gender

            model = ECAPA_gender(C=1024)
            model.load_state_dict(load_file(str(weights_path())))
            model.eval()
            _model = model
        return _model


def classify_speaker_windows(
    windows: list[Any], sample_rate: int = 16000,
) -> dict[str, Any] | None:
    """Vote over one speaker's voiced windows.

    Returns {'gender': 'Male' | 'Female', 'confidence': float} or None when
    the vote is inconclusive (too little speech, low confidence, no model) —
    the caller then uses its fallback. All windows are padded/trimmed to one
    length so the whole vote runs as a single batch.
    """
    import torch

    prepared = [
        np.asarray(window, dtype=np.float32)
        for window in (windows or [])[:_MAX_WINDOWS_PER_SPEAKER]
        if len(np.asarray(window)) >= int(0.5 * sample_rate)
    ]
    if not prepared:
        return None
    try:
        model = _get_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        shortest = min(len(window) for window in prepared)
        batch = torch.stack([
            torch.from_numpy(window[:shortest]) for window in prepared
        ]).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(batch), dim=1)[:, 1]
        female_probability = float(np.mean(probabilities.cpu().numpy()))
        confidence = max(female_probability, 1.0 - female_probability)
        if confidence < _MIN_CONFIDENCE:
            return None
        return {
            'gender': 'Female' if female_probability >= 0.5 else 'Male',
            'confidence': round(confidence, 3),
        }
    except Exception as error:
        logger.warning('Gender classification failed: %s', error)
        return None
