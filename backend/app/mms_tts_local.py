"""Khmer offline TTS: fine-tuned MMS-VITS (Hugging Face Transformers, local).

Model: khmerttsopensource/khmer-tts (fine-tuned from facebook/mms-tts-khm).
License: cc-by-nc-4.0 (non-commercial). The weights are downloaded through the
model manager into %LOCALAPPDATA%\\DubVibe Studio\\models\\mms-tts-khm\\ and
run fully offline on CPU (or CUDA when available).
"""
from __future__ import annotations

import threading
from pathlib import Path

from .providers import TextToSpeechProvider, models_root

MMS_KHM_MODEL_ID = 'mms-tts-khm'
MMS_KHM_REPO = 'khmerttsopensource/khmer-tts'
MMS_KHM_SIZE_MB = 320
_MODEL_FILES = ['model.safetensors', '*.json']

_MODEL = None
_TOKENIZER = None
_LOAD_LOCK = threading.Lock()
# VitsModel is not documented as thread-safe for concurrent forward passes.
_INFERENCE_LOCK = threading.Lock()


def mms_model_dir() -> Path:
    directory = models_root() / 'mms-tts' / MMS_KHM_MODEL_ID
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_installed() -> bool:
    weights = mms_model_dir() / 'model.safetensors'
    return weights.is_file() and weights.stat().st_size > 1024 * 1024 * 10


def download(progress_cb=None) -> None:
    """Blocking download of the Khmer VITS model into the shared model store.

    progress_cb receives a 0-100 percent, estimated from the bytes on disk
    against MMS_KHM_SIZE_MB (snapshot_download does not expose byte progress).
    """
    import threading
    from huggingface_hub import snapshot_download

    target = mms_model_dir()
    done = threading.Event()

    def watch() -> None:
        expected = MMS_KHM_SIZE_MB * 1024 * 1024
        while not done.wait(1.0):
            written = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
            if progress_cb:
                progress_cb(min(99, int(written * 100 / expected)))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot_download(repo_id=MMS_KHM_REPO, local_dir=str(target), allow_patterns=_MODEL_FILES)
    finally:
        done.set()
        watcher.join(timeout=2)
    if progress_cb:
        progress_cb(100)


def _load():
    global _MODEL, _TOKENIZER
    with _LOAD_LOCK:
        if _MODEL is None or _TOKENIZER is None:
            from transformers import AutoTokenizer, VitsModel

            directory = str(mms_model_dir())
            _TOKENIZER = AutoTokenizer.from_pretrained(directory)
            _MODEL = VitsModel.from_pretrained(directory)
            _MODEL.eval()
    return _MODEL, _TOKENIZER


def unload() -> None:
    global _MODEL, _TOKENIZER
    with _LOAD_LOCK:
        _MODEL = None
        _TOKENIZER = None


class MmsKhmerProvider(TextToSpeechProvider):
    """Offline Khmer TTS. Writes a 16-bit WAV file to output_path."""

    def synthesize(self, text: str, voice: str, output_path: str, language: str, speed: float = 1.0) -> dict:
        import numpy as np
        import soundfile as sf
        import torch

        if not is_installed():
            raise RuntimeError('Khmer MMS-VITS model is not installed. Call download() first.')
        if not text or not text.strip():
            raise RuntimeError('Khmer MMS-VITS needs non-empty text.')

        model, tokenizer = _load()
        inputs = tokenizer(text, return_tensors='pt')
        if inputs['input_ids'].shape[-1] == 0:
            raise RuntimeError('Khmer MMS-VITS produced no tokens for this text.')
        # VITS scales predicted durations by 1 / speaking_rate, so <1 is slower.
        rate = min(max(float(speed or 1.0), 0.5), 1.0)
        with _INFERENCE_LOCK, torch.no_grad():
            model.speaking_rate = rate
            waveform = model(**inputs).waveform.squeeze().cpu().numpy()

        if waveform.size == 0:
            raise RuntimeError('Khmer MMS-VITS produced empty audio.')

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
            'provider': 'mms-tts-khm (Offline)',
            'voice': 'khm-single',
            'path': str(target),
        }
