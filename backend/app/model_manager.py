"""Shared AI model manager: discovery, download, validation, loading.

ONE manager, ONE storage root, used by every interface (local web UI and the
packaged Windows desktop app) through the backend API. Nothing here is
UI-specific and nothing about the filesystem is exposed to the browser.

Storage layout (under %LOCALAPPDATA%\\DubVibe Studio\\models):

    faster-whisper/<model-id>/     one folder per Faster-Whisper model
    argos/                         Argos Translate language packages
    huggingface/                   HuggingFace hub cache (Demucs separation model)
    qwen/<model-id>/               local translation engine (llama.cpp GGUF)

App settings (selected model, device, compute) live in the project data
folder so both interfaces read and write the same values.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import threading
from pathlib import Path

from . import cuda_pack
from .providers import get_app_root, get_storage_dir, models_root

# Faster-Whisper catalog: model id -> display name, HuggingFace repo, approx
# download size in MB (used for progress estimates and disk-space checks).
WHISPER_CATALOG: dict[str, dict] = {
    'tiny': {'name': 'Faster-Whisper Tiny', 'repo': 'Systran/faster-whisper-tiny', 'size_mb': 80},
    'base': {'name': 'Faster-Whisper Base', 'repo': 'Systran/faster-whisper-base', 'size_mb': 150},
    'small': {'name': 'Faster-Whisper Small', 'repo': 'Systran/faster-whisper-small', 'size_mb': 490},
    'medium': {'name': 'Faster-Whisper Medium', 'repo': 'Systran/faster-whisper-medium', 'size_mb': 1550},
    'large-v3': {'name': 'Faster-Whisper Large-v3', 'repo': 'Systran/faster-whisper-large-v3', 'size_mb': 3100},
    'large-v3-turbo': {'name': 'Faster-Whisper Large-v3-Turbo', 'repo': 'deepdml/faster-whisper-large-v3-turbo-ct2', 'size_mb': 1650},
}

_REQUIRED_FILES = ('model.bin', 'config.json')

_STATE_LOCK = threading.RLock()
_DOWNLOADS: dict[str, dict] = {}   # model_id -> live download/validate job
_LOADED: dict | None = None        # {'id': model_id, 'model': WhisperModel}

_DEFAULT_SETTINGS = {
    'selected_whisper_model': os.environ.get('DUBVIBE_WHISPER_MODEL', 'base'),
    'device': 'auto',     # auto | cpu | cuda
    'compute': 'auto',    # auto | int8 | float16 | float32
}
_SETTINGS_CACHE: dict | None = None


# ---------------------------------------------------------------------------
# Settings (shared by web + desktop through the API)
# ---------------------------------------------------------------------------

def _settings_path() -> Path:
    return get_storage_dir('data') / 'app-settings.json'


def load_settings() -> dict:
    global _SETTINGS_CACHE
    with _STATE_LOCK:
        if _SETTINGS_CACHE is None:
            settings = dict(_DEFAULT_SETTINGS)
            try:
                raw = json.loads(_settings_path().read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    settings.update(raw)
            except (OSError, json.JSONDecodeError):
                pass
            if settings.get('selected_whisper_model') not in WHISPER_CATALOG:
                settings['selected_whisper_model'] = 'base'
            _SETTINGS_CACHE = settings
        return dict(_SETTINGS_CACHE)


def save_settings(update: dict) -> dict:
    global _SETTINGS_CACHE
    with _STATE_LOCK:
        settings = load_settings()
        for key in ('selected_whisper_model', 'device', 'compute'):
            if key in update and update[key]:
                settings[key] = update[key]
        if 'custom_api' in update:
            custom = update['custom_api']
            if isinstance(custom, dict):
                settings['custom_api'] = {
                    'base_url': str(custom.get('base_url') or ''),
                    'model': str(custom.get('model') or ''),
                }
            else:
                settings.pop('custom_api', None)
        if settings.get('selected_whisper_model') not in WHISPER_CATALOG:
            settings['selected_whisper_model'] = 'base'
        try:
            path = _settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError:
            pass  # settings are a convenience; never fail a request over them
        _SETTINGS_CACHE = settings
        return dict(settings)


def selected_model_id() -> str:
    return load_settings()['selected_whisper_model']


# ---------------------------------------------------------------------------
# Disk locations and installation discovery
# ---------------------------------------------------------------------------

def whisper_model_dir(model_id: str) -> Path:
    return models_root() / 'faster-whisper' / model_id


def _dir_size_mb(path: Path) -> float:
    total = 0
    for file in path.rglob('*'):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def _model_bin_candidates(model_id: str) -> list[Path]:
    """Every model.bin for this model, newest first.

    Searches the AppData install first, then any copy bundled inside the
    desktop app (read-only is fine for loading). Handles both layouts:
    flat (model.bin directly inside) and HuggingFace cache snapshots
    (models--*/snapshots/<rev>/model.bin).
    """
    search_dirs = [whisper_model_dir(model_id)]
    if getattr(sys, 'frozen', False):
        search_dirs.append(Path(sys.executable).resolve().parent / 'models' / 'faster-whisper' / model_id)
    found: list[Path] = []
    for directory in search_dirs:
        if directory.is_dir():
            found.extend(
                sorted(directory.rglob('model.bin'), key=lambda p: p.stat().st_mtime, reverse=True)
            )
    return found


def is_installed(model_id: str) -> bool:
    if model_id not in WHISPER_CATALOG:
        return False
    return bool(_model_bin_candidates(model_id))


def model_path(model_id: str) -> str:
    """Directory that can be handed to WhisperModel for offline loading."""
    candidates = _model_bin_candidates(model_id)
    if candidates:
        return str(candidates[0].parent)
    return str(whisper_model_dir(model_id))


# ---------------------------------------------------------------------------
# Hardware detection (no new dependencies; CUDA via torch when available)
# ---------------------------------------------------------------------------

def _memory_status_gb() -> tuple[float, float]:
    """(total_gb, available_gb) of physical RAM. Windows-only fast path; 0s elsewhere."""
    if os.name == 'nt':
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return (round(status.ullTotalPhys / 2**30, 1), round(status.ullAvailPhys / 2**30, 1))
        except Exception:
            pass
    return (_system_ram_gb(), 0.0)


def _system_ram_gb() -> float:
    if os.name == 'nt':
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 2**30, 1)
        except Exception:
            pass
    try:
        return round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def hardware_info() -> dict:
    import platform

    settings = load_settings()
    info = {
        'cpu': platform.processor() or platform.machine(),
        'cpu_cores': os.cpu_count() or 0,
        'ram_gb': _system_ram_gb(),
        'cuda_available': False,
        'gpu': None,
        'vram_gb': None,
        'device': settings.get('device', 'auto'),
        'compute': settings.get('compute', 'auto'),
    }
    gpu = cuda_pack.nvidia_gpu_info()
    if gpu:
        # Detected via the NVIDIA driver (nvidia-smi) — works with the CPU
        # torch build bundled in the app.
        info['cuda_available'] = True
        info['gpu'] = gpu['name']
        info['vram_gb'] = gpu['vram_gb']
        info['gpu_pack_installed'] = cuda_pack.is_pack_installed()
    info['resolved_device'], info['resolved_compute'] = resolve_device_compute()
    return info


def resolve_device_compute() -> tuple[str, str]:
    settings = load_settings()
    device = settings.get('device', 'auto')
    compute = settings.get('compute', 'auto')
    # CUDA here means: an NVIDIA GPU is present AND the GPU Acceleration Pack
    # (cuBLAS/cuDNN runtime DLLs) has been downloaded. The bundled torch is a
    # CPU build, so torch.cuda can never be the signal — the pack is.
    cuda_usable = cuda_pack.is_pack_installed() and bool(cuda_pack.nvidia_gpu_info())
    if device == 'auto':
        device = 'cuda' if cuda_usable else 'cpu'
    if device == 'cuda' and not cuda_usable:
        device = 'cpu'  # safe fallback when CUDA is unavailable
    if compute == 'auto':
        compute = 'float16' if device == 'cuda' else 'int8'
    return device, compute


# ---------------------------------------------------------------------------
# Status snapshot for the API
# ---------------------------------------------------------------------------

def qwen_status() -> dict:
    from . import qwen_local

    job = _DOWNLOADS.get('qwen3-4b')
    installed = qwen_local.is_installed()
    if job and job.get('status') == 'downloading':
        status = 'downloading'
    elif job and job.get('status') == 'error':
        status = 'error'
    elif installed:
        status = 'installed'
    else:
        status = 'not_installed'
    size_mb = round(qwen_local.qwen_model_file().stat().st_size / (1024 * 1024), 1) if installed else None
    try:
        from . import llama_backend

        runtime = llama_backend.status()
    except Exception as error:  # noqa: BLE001
        runtime = {'backend': 'unknown', 'label': 'unknown', 'reason': str(error), 'gpu': False}
    return {
        'id': 'qwen3-4b',
        'name': 'Qwen3 4B Instruct (local translation)',
        'repo': qwen_local.QWEN_REPO,
        'installed': installed,
        'loaded': False,
        'selected': False,
        'status': status,
        'progress': job.get('progress', 0) if job else 0,
        'downloaded_mb': round(job.get('downloaded', 0), 1) if job else 0,
        'total_mb': qwen_local.QWEN_SIZE_MB,
        'size_mb': size_mb,
        'path': str(qwen_local.qwen_model_file()) if installed else None,
        'error': job.get('error') if job else None,
        'backend': runtime.get('backend'),
        'backend_label': runtime.get('label'),
        'backend_reason': runtime.get('reason'),
        'cuda_backend_installed': runtime.get('cuda_installed'),
        'runtime_problem': qwen_local.runtime_problem() if installed else None,
    }


def start_download_qwen() -> dict:
    from . import qwen_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get('qwen3-4b')
        if job and job.get('status') == 'downloading':
            return job
        if qwen_installed():
            return {'status': 'installed', 'progress': 100}
        free_bytes = shutil.disk_usage(models_root()).free
        needed = qwen_local.QWEN_SIZE_MB * 1024 * 1024 * 1.25
        if free_bytes < needed:
            raise ValueError(
                f'Not enough disk space: {needed / 2**30:.1f} GB needed, '
                f'{free_bytes / 2**30:.1f} GB free on {models_root()}'
            )
        job = {'status': 'downloading', 'progress': 0, 'downloaded': 0.0, 'error': None}
        _DOWNLOADS['qwen3-4b'] = job

    def worker() -> None:
        try:
            def job_progress(percent: int) -> None:
                job.update(progress=percent, downloaded=qwen_local.QWEN_SIZE_MB * percent / 100)

            qwen_local.download_with_progress(job_progress)
            job.update(status='installed', progress=100)
        except Exception as error:  # noqa: BLE001
            job.update(status='error', error=str(error))

    threading.Thread(target=worker, name='download-qwen3-4b', daemon=True).start()
    return job


def qwen_installed() -> bool:
    from . import qwen_local

    return qwen_local.is_installed()


def delete_qwen() -> None:
    from . import qwen_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get('qwen3-4b')
        if job and job.get('status') == 'downloading':
            raise ValueError('Cannot delete a model while it is downloading.')
        qwen_local.unload()
    shutil.rmtree(qwen_local.qwen_model_dir(), ignore_errors=True)


# ---------------------------------------------------------------------------
# Voice gender classifier (JaesungHuh/voice-gender-classifier, MIT). Small
# (~60 MB). Auto-downloads on the first Detect Speakers run, but it is also
# registered here so the AI Models panel can install and delete it manually.
# ---------------------------------------------------------------------------

GENDER_JOB_ID = 'ecapa-gender'


def gender_status() -> dict:
    from . import gender_classifier

    job = _DOWNLOADS.get(GENDER_JOB_ID)
    installed = gender_classifier.is_downloaded()
    if job and job.get('status') == 'downloading':
        status = 'downloading'
    elif job and job.get('status') == 'error':
        status = 'error'
    elif installed:
        status = 'installed'
    else:
        status = 'not_installed'
    weights = gender_classifier.weights_path()
    size_mb = round(weights.stat().st_size / (1024 * 1024), 1) if installed else None
    return {
        'id': GENDER_JOB_ID,
        'name': 'Voice Gender Classifier (speaker detection)',
        'repo': gender_classifier.REPO_ID,
        'installed': installed,
        'loaded': False,
        'selected': False,
        'status': status,
        'progress': job.get('progress', 0) if job else 0,
        'downloaded_mb': round(job.get('downloaded', 0), 1) if job else 0,
        'total_mb': 60,
        'size_mb': size_mb,
        'path': str(weights) if installed else None,
        'error': job.get('error') if job else None,
    }


def start_download_gender() -> dict:
    from . import gender_classifier

    with _STATE_LOCK:
        job = _DOWNLOADS.get(GENDER_JOB_ID)
        if job and job.get('status') == 'downloading':
            return job
        if gender_classifier.is_downloaded():
            return {'status': 'installed', 'progress': 100}
        free_bytes = shutil.disk_usage(models_root()).free
        needed = 60 * 1024 * 1024 * 1.25
        if free_bytes < needed:
            raise ValueError(
                f'Not enough disk space: {needed / 2**30:.1f} GB needed, '
                f'{free_bytes / 2**30:.1f} GB free on {models_root()}'
            )
        job = {'status': 'downloading', 'progress': 0, 'downloaded': 0.0, 'error': None}
        _DOWNLOADS[GENDER_JOB_ID] = job

    def worker() -> None:
        try:
            def job_progress(percent: int) -> None:
                job.update(progress=percent, downloaded=60.0 * percent / 100)

            # _get_model cache is reset on delete below; an older cached model
            # inside this process stays valid for an identical revision.
            gender_classifier.ensure_downloaded(job_progress)
            job.update(status='installed', progress=100)
        except Exception as error:  # noqa: BLE001
            job.update(status='error', error=str(error))

    threading.Thread(target=worker, name='download-ecapa-gender', daemon=True).start()
    return job


def delete_gender() -> None:
    from . import gender_classifier

    with _STATE_LOCK:
        job = _DOWNLOADS.get(GENDER_JOB_ID)
        if job and job.get('status') == 'downloading':
            raise ValueError('Cannot delete a model while it is downloading.')
    shutil.rmtree(gender_classifier.weights_path().parent, ignore_errors=True)



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# VoxCPM2 multilingual offline TTS (OpenBMB, Apache-2.0). Large (~4.7 GB).
# Downloaded only when the user clicks Download; never fetched automatically.
# ---------------------------------------------------------------------------

VOXCPM_JOB_ID = 'voxcpm:voxcpm2'


def voxcpm_status() -> dict:
    from . import voxcpm_local

    job = _DOWNLOADS.get(VOXCPM_JOB_ID)
    installed = voxcpm_local.is_installed()
    if job and job.get('status') == 'downloading':
        status = 'downloading'
    elif job and job.get('status') == 'error':
        status = 'error'
    elif installed:
        status = 'installed'
    else:
        status = 'not_installed'
    size_mb = round(_dir_size_mb(voxcpm_local.model_dir()), 1) if installed else None
    _, verify_reason = voxcpm_local.verify_files()
    return {
        'id': voxcpm_local.ENGINE_ID,
        'name': voxcpm_local.NAME,
        'repo': voxcpm_local.REPO,
        'installed': installed,
        'verify_error': None if installed else verify_reason,
        'loaded': voxcpm_local.loaded(),
        'selected': False,
        'status': status,
        'progress': job.get('progress', 0) if job else 0,
        'downloaded_mb': round(job.get('downloaded', 0), 1) if job else 0,
        'total_mb': voxcpm_local.SIZE_MB,
        'size_mb': size_mb,
        'path': str(voxcpm_local.model_dir()) if installed else None,
        'error': job.get('error') if job else None,
    }


def start_download_voxcpm() -> dict:
    from . import voxcpm_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get(VOXCPM_JOB_ID)
        if job and job.get('status') == 'downloading':
            return job
        if voxcpm_local.is_installed():
            return {'status': 'installed', 'progress': 100}
        free_bytes = shutil.disk_usage(models_root()).free
        needed = voxcpm_local.SIZE_MB * 1024 * 1024 * 1.25
        if free_bytes < needed:
            raise ValueError(
                f'Not enough disk space: {needed / 2**30:.1f} GB needed, '
                f'{free_bytes / 2**30:.1f} GB free on {models_root()}'
            )
        job = {'status': 'downloading', 'progress': 0, 'downloaded': 0.0, 'error': None}
        _DOWNLOADS[VOXCPM_JOB_ID] = job

    def worker() -> None:
        try:
            def job_progress(percent: int) -> None:
                job.update(progress=percent, downloaded=voxcpm_local.SIZE_MB * percent / 100)

            voxcpm_local.download(job_progress)
            job.update(status='installed', progress=100)
        except Exception as error:  # noqa: BLE001
            job.update(status='error', error=str(error))

    threading.Thread(target=worker, name='download-voxcpm2', daemon=True).start()
    return job


def delete_voxcpm() -> None:
    from . import voxcpm_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get(VOXCPM_JOB_ID)
        if job and job.get('status') == 'downloading':
            raise ValueError('Cannot delete a model while it is downloading.')
        voxcpm_local.unload()
        _DOWNLOADS.pop(VOXCPM_JOB_ID, None)
    shutil.rmtree(voxcpm_local.model_dir(), ignore_errors=True)


# ---------------------------------------------------------------------------
# NLLB-200 offline translation (Meta). Non-commercial research license (cc-by-nc-4.0).
# Downloaded only when the user clicks Download; never fetched automatically.
# ---------------------------------------------------------------------------

NLLB_JOB_ID = 'nllb-200'


def nllb_status() -> dict:
    from . import nllb_local

    job = _DOWNLOADS.get(NLLB_JOB_ID)
    installed = nllb_local.is_installed()
    if job and job.get('status') == 'downloading':
        status = 'downloading'
    elif job and job.get('status') == 'error':
        status = 'error'
    elif installed:
        status = 'installed'
    else:
        status = 'not_installed'
    size_mb = round(_dir_size_mb(nllb_local.nllb_model_dir()), 1) if installed else None
    return {
        'id': nllb_local.NLLB_MODEL_ID,
        'name': 'NLLB-200 600M (offline translation, non-commercial)',
        'repo': nllb_local.NLLB_REPO,
        'installed': installed,
        'loaded': nllb_local._MODEL is not None,
        'selected': False,
        'status': status,
        'progress': job.get('progress', 0) if job else 0,
        'downloaded_mb': round(job.get('downloaded', 0), 1) if job else 0,
        'total_mb': nllb_local.NLLB_SIZE_MB,
        'size_mb': size_mb,
        'path': str(nllb_local.nllb_model_dir()) if installed else None,
        'error': job.get('error') if job else None,
    }


def start_download_nllb() -> dict:
    from . import nllb_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get(NLLB_JOB_ID)
        if job and job.get('status') == 'downloading':
            return job
        if nllb_local.is_installed():
            return {'status': 'installed', 'progress': 100}
        free_bytes = shutil.disk_usage(models_root()).free
        needed = nllb_local.NLLB_SIZE_MB * 1024 * 1024 * 1.25
        if free_bytes < needed:
            raise ValueError(
                f'Not enough disk space: {needed / 2**30:.1f} GB needed, '
                f'{free_bytes / 2**30:.1f} GB free on {models_root()}'
            )
        job = {'status': 'downloading', 'progress': 0, 'downloaded': 0.0, 'error': None}
        _DOWNLOADS[NLLB_JOB_ID] = job

    def worker() -> None:
        try:
            def job_progress(percent: int) -> None:
                job.update(progress=percent, downloaded=nllb_local.NLLB_SIZE_MB * percent / 100)

            nllb_local.download(job_progress)
            job.update(status='installed', progress=100)
        except Exception as error:  # noqa: BLE001
            job.update(status='error', error=str(error))

    threading.Thread(target=worker, name='download-nllb-200', daemon=True).start()
    return job


def delete_nllb() -> None:
    from . import nllb_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get(NLLB_JOB_ID)
        if job and job.get('status') == 'downloading':
            raise ValueError('Cannot delete a model while it is downloading.')
        nllb_local.unload()
        _DOWNLOADS.pop(NLLB_JOB_ID, None)
    shutil.rmtree(nllb_local.nllb_model_dir(), ignore_errors=True)


# ---------------------------------------------------------------------------
# Demucs 6-stem separation model (audio separation feature)
# ---------------------------------------------------------------------------

def demucs_status() -> dict:
    from . import demucs_local

    job = _DOWNLOADS.get('demucs-htdemucs-6s')
    installed = demucs_local.is_installed()
    if job and job.get('status') == 'downloading':
        status = 'downloading'
    elif job and job.get('status') == 'error':
        status = 'error'
    elif installed:
        status = 'installed'
    else:
        status = 'not_installed'
    return {
        'id': 'demucs-htdemucs-6s',
        'name': 'Demucs 6-stem (audio separation)',
        'repo': demucs_local.DEMUCS_REPO_ID,
        'installed': installed,
        'loaded': False,
        'selected': False,
        'status': status,
        'progress': job.get('progress', 0) if job else 0,
        'downloaded_mb': round(job.get('downloaded', 0), 1) if job else 0,
        'total_mb': demucs_local.DEMUCS_SIZE_MB,
        'size_mb': demucs_local.installed_size_mb(),
        'path': str(demucs_local.demucs_dir()) if installed else None,
        'error': job.get('error') if job else None,
    }


def start_download_demucs() -> dict:
    from . import demucs_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get('demucs-htdemucs-6s')
        if job and job.get('status') == 'downloading':
            return job
        if demucs_local.is_installed():
            return {'status': 'installed', 'progress': 100}
        free_bytes = shutil.disk_usage(models_root()).free
        needed = demucs_local.DEMUCS_SIZE_MB * 1024 * 1024 * 1.25
        if free_bytes < needed:
            raise ValueError(
                f'Not enough disk space: {needed / 2**20:.0f} MB needed, '
                f'{free_bytes / 2**30:.1f} GB free on {models_root()}'
            )
        job = {'status': 'downloading', 'progress': 0, 'downloaded': 0.0, 'error': None}
        _DOWNLOADS['demucs-htdemucs-6s'] = job

    def worker() -> None:
        try:
            def job_progress(percent: int) -> None:
                job.update(progress=percent, downloaded=demucs_local.DEMUCS_SIZE_MB * percent / 100)

            demucs_local.download_with_progress(job_progress)
            job.update(status='installed', progress=100)
        except Exception as error:  # noqa: BLE001
            job.update(status='error', error=str(error))

    threading.Thread(target=worker, name='download-demucs', daemon=True).start()
    return job


def demucs_installed() -> bool:
    from . import demucs_local

    return demucs_local.is_installed()


def delete_demucs() -> None:
    from . import demucs_local

    with _STATE_LOCK:
        job = _DOWNLOADS.get('demucs-htdemucs-6s')
        if job and job.get('status') == 'downloading':
            raise ValueError('Cannot delete a model while it is downloading.')
    shutil.rmtree(demucs_local.demucs_dir(), ignore_errors=True)
    # Also remove a copy installed by an older version into the Hub cache.
    shutil.rmtree(demucs_local.hf_cache_dir() / f'models--{demucs_local.DEMUCS_REPO_ID.replace("/", "--")}',
                  ignore_errors=True)


def whisper_models_status() -> list[dict]:
    with _STATE_LOCK:
        selected = selected_model_id()
        models = []
        for model_id, meta in WHISPER_CATALOG.items():
            job = _DOWNLOADS.get(model_id)
            installed = is_installed(model_id)
            loaded = bool(_LOADED and _LOADED.get('id') == model_id)
            if job and job.get('status') == 'downloading':
                status = 'downloading'
            elif job and job.get('status') == 'validating':
                status = 'validating'
            elif job and job.get('status') == 'error':
                status = 'error'
            elif loaded:
                status = 'loaded'
            elif installed:
                status = 'installed'
            else:
                status = 'not_installed'
            models.append({
                'id': model_id,
                'name': meta['name'],
                'repo': meta['repo'],
                'installed': installed,
                'loaded': loaded,
                'selected': selected == model_id,
                'status': status,
                'progress': job.get('progress', 0) if job else 0,
                'downloaded_mb': round(job.get('downloaded', 0), 1) if job else 0,
                'total_mb': meta['size_mb'],
                'size_mb': round(_dir_size_mb(whisper_model_dir(model_id)), 1) if installed else None,
                'path': model_path(model_id) if installed else None,
                'error': job.get('error') if job else None,
            })
        return models


def model_status(model_id: str) -> dict | None:
    for entry in whisper_models_status():
        if entry['id'] == model_id:
            return entry
    return None


def storage_info() -> dict:
    root = models_root()
    usage = shutil.disk_usage(root)
    per_model = {
        model_id: round(_dir_size_mb(whisper_model_dir(model_id)), 1)
        for model_id in WHISPER_CATALOG
        if is_installed(model_id)
    }
    return {
        'root': str(root),
        'free_gb': round(usage.free / 2**30, 1),
        'total_gb': round(usage.total / 2**30, 1),
        'installed_mb': round(sum(per_model.values()), 1),
        'models': per_model,
    }


# ---------------------------------------------------------------------------
# Download + validation
# ---------------------------------------------------------------------------

def _download_sync(model_id: str, progress_cb=None) -> None:
    """Blocking download of one model into its own folder. Raises on failure."""
    from .hf_progress import download_snapshot

    meta = WHISPER_CATALOG[model_id]
    target = whisper_model_dir(model_id)
    target.mkdir(parents=True, exist_ok=True)

    def on_fraction(percent: int) -> None:
        if progress_cb:
            progress_cb(int(percent))

    download_snapshot(
        meta['repo'],
        local_dir=target,
        progress_cb=on_fraction,
    )
    if progress_cb:
        progress_cb(100)


def _validate_model(model_id: str) -> None:
    """Verify required files exist AND the model can actually be loaded."""
    directory = whisper_model_dir(model_id)
    missing = [name for name in _REQUIRED_FILES if not any(directory.rglob(name))]
    if missing:
        raise RuntimeError(f'Model download is incomplete (missing: {", ".join(missing)}). Retry the download.')
    from faster_whisper import WhisperModel

    model = WhisperModel(str(directory), device='cpu', compute_type='int8')
    del model
    gc.collect()


def start_download(model_id: str) -> dict:
    if model_id not in WHISPER_CATALOG:
        raise ValueError(f'Unknown model: {model_id}')
    with _STATE_LOCK:
        job = _DOWNLOADS.get(model_id)
        if job and job.get('status') in ('downloading', 'validating'):
            return job  # already running: never download twice
        if is_installed(model_id):
            return {'status': 'installed', 'progress': 100}
        free_bytes = shutil.disk_usage(models_root()).free
        needed = WHISPER_CATALOG[model_id]['size_mb'] * 1024 * 1024 * 1.25
        if free_bytes < needed:
            raise ValueError(
                f'Not enough disk space: {needed / 2**30:.1f} GB needed, '
                f'{free_bytes / 2**30:.1f} GB free on {models_root()}'
            )
        job = {'status': 'downloading', 'progress': 0, 'downloaded': 0.0, 'error': None}
        _DOWNLOADS[model_id] = job

    def worker() -> None:
        try:
            def job_progress(percent: int) -> None:
                job.update(progress=percent, downloaded=WHISPER_CATALOG[model_id]['size_mb'] * percent / 100)

            _download_sync(model_id, job_progress)
            job.update(status='validating', progress=100)
            _validate_model(model_id)
            job.update(status='installed', progress=100)
        except Exception as error:  # noqa: BLE001 - surfaced to the UI
            job.update(status='error', error=str(error))

    threading.Thread(target=worker, name=f'download-{model_id}', daemon=True).start()
    return job


def ensure_installed(model_id: str, progress_cb=None) -> None:
    """Blocking: download the model if it is missing (used before transcription)."""
    if is_installed(model_id):
        return
    _download_sync(model_id, progress_cb)
    _validate_model(model_id)


def delete_model(model_id: str) -> None:
    if model_id not in WHISPER_CATALOG:
        raise ValueError(f'Unknown model: {model_id}')
    with _STATE_LOCK:
        job = _DOWNLOADS.get(model_id)
        if job and job.get('status') in ('downloading', 'validating'):
            raise ValueError('Cannot delete a model while it is downloading.')
        if _LOADED and _LOADED.get('id') == model_id:
            _unload_locked()
        shutil.rmtree(whisper_model_dir(model_id), ignore_errors=True)


# ---------------------------------------------------------------------------
# Loading / unloading / transcription access
# ---------------------------------------------------------------------------

def _unload_locked() -> None:
    global _LOADED
    _LOADED = None
    gc.collect()


def unload() -> None:
    with _STATE_LOCK:
        _unload_locked()


def load_model(model_id: str) -> dict:
    global _LOADED
    if model_id not in WHISPER_CATALOG:
        raise ValueError(f'Unknown model: {model_id}')
    if not is_installed(model_id):
        raise ValueError(f'{WHISPER_CATALOG[model_id]["name"]} is not installed. Download it first.')
    with _STATE_LOCK:
        if _LOADED and _LOADED.get('id') == model_id:
            save_settings({'selected_whisper_model': model_id})
            return {'status': 'loaded', 'id': model_id}
        _unload_locked()
        from faster_whisper import WhisperModel

        device, compute = resolve_device_compute()
        try:
            model = WhisperModel(model_path(model_id), device=device, compute_type=compute)
        except Exception as error:
            if device == 'cuda':
                # Fall back safely to CPU when GPU acceleration fails.
                model = WhisperModel(model_path(model_id), device='cpu', compute_type='int8')
                device, compute = 'cpu', 'int8'
            else:
                raise RuntimeError(f'Model failed to load: {error}') from error
        _LOADED = {'id': model_id, 'model': model}
        save_settings({'selected_whisper_model': model_id})
        return {'status': 'loaded', 'id': model_id, 'device': device, 'compute': compute}


def get_whisper_model():
    """Return the loaded WhisperModel for the selected model, loading on demand.

    Raises a clear error when the selected model is missing so callers can
    surface a useful message instead of a crash.
    """
    with _STATE_LOCK:
        if _LOADED:
            return _LOADED['model']
    model_id = selected_model_id()
    if not is_installed(model_id):
        raise RuntimeError(
            f'Transcription model "{model_id}" is not installed. Open the AI Model Manager and download it first.'
        )
    load_model(model_id)
    with _STATE_LOCK:
        assert _LOADED is not None
        return _LOADED['model']


def ensure_selected_ready(progress_cb=None) -> tuple[str, str]:
    """Blocking: make sure the selected model is downloaded (and report progress)."""
    model_id = selected_model_id()
    if not is_installed(model_id):
        _download_sync(model_id, progress_cb)
        _validate_model(model_id)
    device, compute = resolve_device_compute()
    return model_id, f'{device}/{compute}'


# ---------------------------------------------------------------------------
# One-time migration from the old per-install model locations
# ---------------------------------------------------------------------------

def migrate_legacy_models() -> None:
    """Move models from the legacy locations into the shared AppData root.

    Legacy locations: <AppRoot>/models (dev repo, packaged install) and
    %LOCALAPPDATA%/DubVibe/models (old read-only-install fallback).
    Best effort: anything that cannot be moved simply stays put.
    """
    root = models_root()
    legacy_roots = [get_app_root() / 'models']
    local = os.environ.get('LOCALAPPDATA')
    if local:
        legacy_roots.append(Path(local) / 'DubVibe' / 'models')

    for legacy in legacy_roots:
        if not legacy.is_dir() or legacy.resolve() == root.resolve():
            continue
        whisper_legacy = legacy / 'whisper'
        if whisper_legacy.is_dir():
            for cache_dir in whisper_legacy.glob('models--*'):
                # models--Systran--faster-whisper-base -> base
                marker = 'faster-whisper-'
                if marker in cache_dir.name:
                    model_id = cache_dir.name.split(marker, 1)[1]
                    if model_id in WHISPER_CATALOG:
                        target = root / 'faster-whisper' / model_id / cache_dir.name
                        if not target.exists() and not is_installed(model_id):
                            target.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                shutil.move(str(cache_dir), str(target))
                            except OSError:
                                continue
        for name in ('argos', 'torch'):
            source = legacy / name
            if source.is_dir():
                target = root / name
                if not target.exists():
                    try:
                        shutil.move(str(source), str(target))
                    except OSError:
                        continue
