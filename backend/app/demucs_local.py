"""Demucs 6-stem separation model (htdemucs_6s), managed like the other models.

The weights come from the HuggingFace hub (repo `adefossez/HTDemucs-6s`) and are
cached inside the shared model store — see main.py which points HF_HUB_CACHE at
`%LOCALAPPDATA%\\DubVibe Studio\\models\\huggingface` before huggingface_hub is
imported. Separation requires this model; download it in the AI Model Manager.
"""
from __future__ import annotations

import threading
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
    return hf_cache_dir() / f'models--{DEMUCS_REPO_ID.replace("/", "--")}' / 'snapshots'


def checkpoint_files() -> list[Path]:
    """Downloaded weight/config files, empty when the model is not installed."""
    snapshots = _snapshot_dir()
    if not snapshots.is_dir():
        return []
    found = sorted(snapshots.glob('*/*.safetensors')) + sorted(snapshots.glob('*/*.yaml'))
    return [path for path in found if path.is_file()]


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
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm as base_tqdm

    done = threading.Event()
    bars: dict[int, dict] = {}
    lock = threading.Lock()

    class ReportingTqdm(base_tqdm):
        def update(self, n=1):
            result = super().update(n)
            try:
                with lock:
                    bars[id(self)] = {'n': float(self.n or 0), 'total': float(self.total or 0)}
                    done_sum = sum(entry['n'] for entry in bars.values())
                    total = sum(entry['total'] for entry in bars.values())
                if total > 0:
                    if progress_cb:
                        progress_cb(min(99, int(100.0 * done_sum / total)))
            except Exception:
                pass
            return result

        def close(self):
            try:
                with lock:
                    bars[id(self)] = {'n': float(self.total or 0), 'total': float(self.total or 0)}
                    done_sum = sum(entry['n'] for entry in bars.values())
                    total = sum(entry['total'] for entry in bars.values())
                if total > 0 and progress_cb:
                    progress_cb(min(99, int(100.0 * done_sum / total)))
            except Exception:
                pass
            super().close()

    try:
        yaml_path = Path(hf_hub_download(repo_id=DEMUCS_REPO_ID, filename=f'{DEMUCS_MODEL_NAME}.yaml', tqdm_class=ReportingTqdm))
        import yaml as yaml_lib

        with open(yaml_path, encoding='utf-8') as handle:
            bag = yaml_lib.safe_load(handle)
        signatures = bag.get('models') or ['5c90dfd2']
        for signature in signatures:
            hf_hub_download(repo_id=DEMUCS_REPO_ID, filename=f'{signature}.safetensors', tqdm_class=ReportingTqdm)
    finally:
        done.set()
    if progress_cb:
        progress_cb(100)
