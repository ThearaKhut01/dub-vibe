"""Khmer offline TTS: MMS-VITS models run locally with Hugging Face Transformers.

Two Khmer models share one implementation:
  - mms_khm       khmerttsopensource/khmer-tts (fine-tuned from facebook/mms-tts-khm)
  - mms_khm_base  facebook/mms-tts-khm (the pretrained base model)
Both are cc-by-nc-4.0 (non-commercial). Weights are downloaded through the model
manager into %LOCALAPPDATA%\\DubVibe Studio\\models\\mms-tts\\<folder>\\ and run
fully offline on CPU (or CUDA when available).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .providers import TextToSpeechProvider, models_root

_MODEL_FILES = ['model.safetensors', '*.json']


@dataclass(frozen=True)
class MmsSpec:
    engine_id: str
    folder: str
    repo: str
    size_mb: int
    name: str


MMS_MODELS: dict[str, MmsSpec] = {
    'mms_khm': MmsSpec(
        engine_id='mms_khm',
        folder='mms-tts-khm',
        repo='khmerttsopensource/khmer-tts',
        size_mb=320,
        name='Khmer MMS-VITS (fine-tuned, offline, non-commercial)',
    ),
    'mms_khm_base': MmsSpec(
        engine_id='mms_khm_base',
        folder='mms-tts-khm-base',
        repo='facebook/mms-tts-khm',
        size_mb=150,
        name='Khmer MMS-VITS (base model, offline, non-commercial)',
    ),
}

# Per-model runtime state: engine_id -> (model, tokenizer).
_LOADED: dict[str, tuple] = {}
_LOAD_LOCK = threading.Lock()
# VitsModel is not documented as thread-safe for concurrent forward passes.
_INFERENCE_LOCK = threading.Lock()


def spec_for(engine_id: str) -> MmsSpec:
    if engine_id not in MMS_MODELS:
        raise KeyError(f'Unknown Khmer MMS model: {engine_id}')
    return MMS_MODELS[engine_id]


def mms_model_dir(engine_id: str = 'mms_khm') -> Path:
    directory = models_root() / 'mms-tts' / spec_for(engine_id).folder
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_installed(engine_id: str = 'mms_khm') -> bool:
    weights = mms_model_dir(engine_id) / 'model.safetensors'
    return weights.is_file() and weights.stat().st_size > 1024 * 1024 * 10


def download(engine_id: str = 'mms_khm', progress_cb=None) -> None:
    """Blocking download of one Khmer VITS model into the shared model store.

    progress_cb receives a 0-100 percent, estimated from the bytes on disk
    against the model's size (snapshot_download does not expose byte progress).
    """
    from huggingface_hub import snapshot_download

    spec = spec_for(engine_id)
    target = mms_model_dir(engine_id)
    done = threading.Event()

    def watch() -> None:
        expected = spec.size_mb * 1024 * 1024
        while not done.wait(1.0):
            written = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
            if progress_cb:
                progress_cb(min(99, int(written * 100 / expected)))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot_download(repo_id=spec.repo, local_dir=str(target), allow_patterns=_MODEL_FILES)
    finally:
        done.set()
        watcher.join(timeout=2)
    if progress_cb:
        progress_cb(100)


def _load(engine_id: str):
    with _LOAD_LOCK:
        if engine_id not in _LOADED:
            from transformers import AutoTokenizer, VitsModel

            directory = str(mms_model_dir(engine_id))
            tokenizer = AutoTokenizer.from_pretrained(directory)
            model = VitsModel.from_pretrained(directory)
            model.eval()
            _LOADED[engine_id] = (model, tokenizer)
        return _LOADED[engine_id]


def unload(engine_id: str | None = None) -> None:
    with _LOAD_LOCK:
        if engine_id is None:
            _LOADED.clear()
        else:
            _LOADED.pop(engine_id, None)


class MmsKhmerProvider(TextToSpeechProvider):
    """Offline Khmer TTS for one engine id. Writes an MP3 or 16-bit WAV file."""

    def __init__(self, engine_id: str = 'mms_khm') -> None:
        self.engine_id = engine_id

    def synthesize(self, text: str, voice: str, output_path: str, language: str, speed: float = 1.0) -> dict:
        import numpy as np
        import soundfile as sf
        import torch

        name = spec_for(self.engine_id).name
        if not is_installed(self.engine_id):
            raise RuntimeError(f'{name} is not installed. Download it from AI Models first.')
        if not text or not text.strip():
            raise RuntimeError(f'{name} needs non-empty text.')

        model, tokenizer = _load(self.engine_id)
        inputs = tokenizer(text, return_tensors='pt')
        if inputs['input_ids'].shape[-1] == 0:
            raise RuntimeError(f'{name} produced no tokens for this text.')
        # VITS scales predicted durations by 1 / speaking_rate, so <1 is slower.
        rate = min(max(float(speed or 1.0), 0.5), 1.0)
        with _INFERENCE_LOCK, torch.no_grad():
            model.speaking_rate = rate
            waveform = model(**inputs).waveform.squeeze().cpu().numpy()

        if waveform.size == 0:
            raise RuntimeError(f'{name} produced empty audio.')

        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == '.mp3':
            sf.write(str(target), waveform, model.config.sampling_rate, format='MP3')
        else:
            sf.write(str(target), waveform, model.config.sampling_rate, subtype='PCM_16')
        return {
            'provider': f'{self.engine_id} (Offline)',
            'voice': 'khm-single',
            'path': str(target),
        }
