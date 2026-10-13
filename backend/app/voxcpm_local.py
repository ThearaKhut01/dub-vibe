"""Offline TTS: VoxCPM2 (OpenBMB, 2B parameters, Apache-2.0).

Weights are downloaded through the model manager into
%LOCALAPPDATA%\\DubVibe Studio\\models\\voxcpm\\VoxCPM2 and run fully offline.
The model advertises 30 languages, but only English was verified to produce
intelligible speech in this app (Khmer and Thai came out garbled on CPU), so
routing is limited to the verified languages in VERIFIED_LANGUAGE_CODES.
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

from .providers import TextToSpeechProvider, models_root

ENGINE_ID = 'voxcpm2'
REPO = 'openbmb/VoxCPM2'
NAME = 'VoxCPM2 (English, offline, 48 kHz)'
SIZE_MB = 4730
# Only languages verified to produce intelligible audio (ISO codes). The model
# documents 30 languages, but Khmer and Thai failed the transcription check.
VERIFIED_LANGUAGE_CODES = frozenset({'en'})
_MODEL_FILES = ['*.safetensors', '*.pth', '*.json', '*.py', 'tokenizer*']

# Single loaded model: VoxCPM is not documented as safe for concurrent generation.
_LOADED: dict[str, object] = {}
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def model_dir() -> Path:
    directory = models_root() / 'voxcpm' / 'VoxCPM2'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Exact byte sizes of the official openbmb/VoxCPM2 files. A partial or corrupt
# download fails verification and is treated as not installed.
OFFICIAL_FILE_BYTES = {
    'model.safetensors': 4580080592,
    'audiovae.pth': 376951122,
}


def verify_files() -> tuple[bool, str | None]:
    """(ok, reason). Checks the official file sizes and that config.json declares VoxCPM2."""
    import json

    directory = model_dir()
    for name, expected in OFFICIAL_FILE_BYTES.items():
        path = directory / name
        if not path.is_file():
            return False, f'{name} is missing.'
        actual = path.stat().st_size
        if actual != expected:
            return False, f'{name} is incomplete ({actual} of {expected} bytes). Download it again.'
    config = directory / 'config.json'
    if not config.is_file():
        return False, 'config.json is missing.'
    try:
        architecture = json.loads(config.read_text(encoding='utf-8')).get('architecture')
    except (OSError, ValueError) as error:
        return False, f'config.json is unreadable: {error}'
    if architecture != 'voxcpm2':
        return False, f'config.json declares architecture "{architecture}", expected "voxcpm2".'
    return True, None


def is_installed() -> bool:
    """True only when the verified weights and the voxcpm Python package are usable."""
    if importlib.util.find_spec('voxcpm') is None:
        return False
    ok, _ = verify_files()
    return ok


def supports_language(lang_code: str) -> bool:
    return (lang_code or 'en').lower() in VERIFIED_LANGUAGE_CODES


def download(progress_cb=None) -> None:
    """Blocking download of the VoxCPM2 snapshot into the shared model store.

    progress_cb receives a 0-100 percent estimated from bytes on disk, because
    snapshot_download does not expose byte-level progress.
    """
    from huggingface_hub import snapshot_download

    target = model_dir()
    done = threading.Event()
    expected = SIZE_MB * 1024 * 1024

    def watch() -> None:
        while not done.wait(1.0):
            written = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
            if progress_cb:
                progress_cb(min(99, int(written * 100 / expected)))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot_download(repo_id=REPO, local_dir=str(target), allow_patterns=_MODEL_FILES)
    finally:
        done.set()
        watcher.join(timeout=2)
    if progress_cb:
        progress_cb(100)


# Rough peak-memory need to load and run VoxCPM2 on CPU: ~4.4 GB weights plus the
# 360 MB audio VAE plus activations. Checked against available RAM before a load.
LOAD_RAM_GB = 7.0


def hardware_report() -> dict:
    """Device, memory, and whether a load is expected to fit. Never raises."""
    from . import model_manager

    total_gb, available_gb = model_manager._memory_status_gb()
    device = resolve_device()
    gpu = None
    vram_gb = None
    try:
        import torch

        if device == 'cuda':
            gpu = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    if device == 'cuda' and vram_gb is not None:
        fits = vram_gb >= 6.0
        reason = None if fits else f'Only {vram_gb} GB VRAM; VoxCPM2 needs about 6 GB.'
    else:
        fits = available_gb <= 0 or available_gb >= LOAD_RAM_GB
        reason = None if fits else (
            f'Only {available_gb} GB RAM free; VoxCPM2 needs about {LOAD_RAM_GB:.0f} GB. '
            'Close other apps or use a machine with more memory.'
        )
    return {
        'device': device,
        'gpu': gpu,
        'vram_gb': vram_gb,
        'ram_total_gb': total_gb,
        'ram_available_gb': available_gb,
        'fits': fits,
        'reason': reason,
        'loaded': ENGINE_ID in _LOADED,
    }


def resolve_device() -> str:
    """'cuda' only when torch itself can use CUDA; otherwise 'cpu'.

    The GPU Acceleration Pack ships cuBLAS/cuDNN for ctranslate2, not torch
    kernels, so torch.cuda is the only signal that matters for this engine.
    """
    try:
        import torch

        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:  # noqa: BLE001 - torch missing or broken: CPU path
        return 'cpu'


def _load():
    with _LOAD_LOCK:
        if ENGINE_ID not in _LOADED:
            from voxcpm import VoxCPM

            report = hardware_report()
            if not report['fits']:
                raise MemoryError(report['reason'])
            device = report['device']
            # No denoiser (it would fetch a second ModelScope model) and no
            # torch.compile (CUDA-only and slow to warm up on first use).
            _LOADED[ENGINE_ID] = VoxCPM.from_pretrained(
                str(model_dir()),
                load_denoiser=False,
                optimize=False,
                local_files_only=True,
                device=device,
            )
        return _LOADED[ENGINE_ID]


def unload() -> None:
    with _LOAD_LOCK:
        _LOADED.pop(ENGINE_ID, None)


def loaded() -> bool:
    return ENGINE_ID in _LOADED


class VoxCPMProvider(TextToSpeechProvider):
    """Offline multilingual TTS. Writes an MP3 or 16-bit WAV file."""

    def synthesize(self, text: str, voice: str, output_path: str, language: str, speed: float = 1.0) -> dict:
        import numpy as np
        import soundfile as sf

        if not is_installed():
            raise RuntimeError(f'{NAME} is not installed. Download it from AI Models first.')
        if not text or not text.strip():
            raise RuntimeError(f'{NAME} needs non-empty text.')

        model = _load()
        # VoxCPM2 detects the language from the text itself, so no language tag is sent.
        with _INFERENCE_LOCK:
            waveform = model.generate(text=text.strip(), cfg_value=2.0, inference_timesteps=10)
        waveform = np.asarray(waveform, dtype=np.float32).squeeze()
        if waveform.size == 0:
            raise RuntimeError(f'{NAME} produced empty audio.')

        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak

        sample_rate = int(getattr(model.tts_model, 'sample_rate', 48000))
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == '.mp3':
            sf.write(str(target), waveform, sample_rate, format='MP3')
        else:
            sf.write(str(target), waveform, sample_rate, subtype='PCM_16')
        return {
            'provider': f'{ENGINE_ID} (Offline)',
            'voice': voice or 'auto',
            'path': str(target),
        }
