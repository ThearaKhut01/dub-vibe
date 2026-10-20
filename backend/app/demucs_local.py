"""Demucs 6-stem separation model (htdemucs_6s), managed like the other models.

The weights come from the HuggingFace hub (repo `adefossez/HTDemucs-6s`) and are
cached inside the shared model store — see main.py which points HF_HUB_CACHE at
`%LOCALAPPDATA%\\DubVibe Studio\\models\\huggingface` before huggingface_hub is
imported. Separation requires this model; download it in the AI Model Manager.
"""
from __future__ import annotations

from pathlib import Path

from .providers import models_root

DEMUCS_MODEL_NAME = 'htdemucs_6s'
DEMUCS_REPO_ID = 'adefossez/HTDemucs-6s'
DEMUCS_SIZE_MB = 53  # safetensors checkpoint size (yaml is negligible)


def hf_cache_dir() -> Path:
    """Where the HuggingFace hub cache lives (must match HF_HUB_CACHE set in main.py)."""
    directory = models_root() / 'huggingface'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _snapshot_dir() -> Path:
    """Legacy Hub cache location, still checked so existing installs keep working."""
    return hf_cache_dir() / f'models--{DEMUCS_REPO_ID.replace("/", "--")}' / 'snapshots'


def demucs_dir() -> Path:
    """Where the checkpoint files are stored (shared model store, not the Hub cache)."""
    directory = models_root() / 'demucs' / DEMUCS_MODEL_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def checkpoint_files() -> list[Path]:
    """Downloaded weight/config files, empty when the model is not installed."""
    files = sorted(demucs_dir().glob('*.safetensors')) + sorted(demucs_dir().glob('*.yaml'))
    found = [path for path in files if path.is_file()]
    if found:
        return found
    # Older builds stored the checkpoint in the HuggingFace cache.
    snapshots = _snapshot_dir()
    if not snapshots.is_dir():
        return []
    legacy = sorted(snapshots.glob('*/*.safetensors')) + sorted(snapshots.glob('*/*.yaml'))
    return [path for path in legacy if path.is_file()]


def is_installed() -> bool:
    files = checkpoint_files()
    return any(path.suffix == '.safetensors' for path in files) and any(path.suffix == '.yaml' for path in files)


def installed_size_mb() -> float | None:
    files = checkpoint_files()
    weights = [path for path in files if path.suffix == '.safetensors']
    if not weights:
        return None
    return round(weights[0].stat().st_size / (1024 * 1024), 1)


def download_with_progress(progress_cb=None) -> None:
    """Blocking download of the htdemucs_6s weights into the shared store."""
    from .hf_progress import download_file

    target = demucs_dir()
    # The yaml is a few hundred bytes, so progress tracks the weights only.
    yaml_path = Path(download_file(DEMUCS_REPO_ID, f'{DEMUCS_MODEL_NAME}.yaml', local_dir=target))
    import yaml as yaml_lib

    with open(yaml_path, encoding='utf-8') as handle:
        bag = yaml_lib.safe_load(handle)
    signatures = bag.get('models') or ['5c90dfd2']
    for signature in signatures:
        download_file(DEMUCS_REPO_ID, f'{signature}.safetensors', local_dir=target,
                      progress_cb=progress_cb)
    if progress_cb:
        progress_cb(100)
