"""NVIDIA GPU acceleration pack for Faster-Whisper (ctranslate2).

The desktop/web bundle ships CPU-only inference. On machines with an NVIDIA
GPU, the user can download the GPU Acceleration Pack: the cuBLAS + cuDNN
runtime DLLs (CUDA 12) that ctranslate2 needs to run Whisper on the GPU.
The pack is stored under the shared model root:

    %LOCALAPPDATA%\\DubVibe Studio\\models\\cuda\\bin\\

and is activated by registering that folder on the DLL search path before
ctranslate2 loads. Everything degrades gracefully: without the pack (or
without an NVIDIA GPU) transcription simply runs on CPU.

The DLLs are fetched from the official NVIDIA pip wheels on PyPI and
extracted — versions are resolved at download time so they track CUDA 12
updates instead of being hard-coded.
"""
from __future__ import annotations

import json
import shutil
import ssl
import threading
import urllib.request
import zipfile
from pathlib import Path

from .providers import models_root

# ctranslate2 4.x requires cuDNN 9 + CUDA 12 (cublas). These PyPI packages are
# NVIDIA's own redistributions of those runtimes for Windows.
_PACK_WHEELS = {
    'cublas': ('nvidia-cublas-cu12', 'cublas64_12.dll', 'nvidia/cublas/bin'),
    'cudnn': ('nvidia-cudnn-cu12', 'cudnn64_9.dll', 'nvidia/cudnn/bin'),
}

_DOWNLOADS: dict[str, dict] = {}
_LOCK = threading.Lock()


def cuda_bin_dir() -> Path:
    directory = models_root() / 'cuda' / 'bin'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_pack_installed() -> bool:
    bin_dir = cuda_bin_dir()
    return all((bin_dir / dll).is_file() for _, dll, _ in _PACK_WHEELS.values())


def nvidia_gpu_info() -> dict | None:
    """Detect an NVIDIA GPU via the driver's nvidia-smi (works without torch)."""
    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        first = result.stdout.strip().splitlines()[0]
        name, mb = [part.strip() for part in first.split(',')]
        return {'name': name, 'vram_gb': round(float(mb) / 1024, 1)}
    except Exception:
        return None


def pack_status() -> dict:
    gpu = nvidia_gpu_info()
    with _LOCK:
        job = _DOWNLOADS.get('cuda-gpu-pack')
    installed = is_pack_installed()
    if job and job.get('status') == 'downloading':
        status = 'downloading'
    elif job and job.get('status') == 'error':
        status = 'error'
    elif installed:
        status = 'installed'
    elif gpu:
        status = 'available'   # NVIDIA present: the pack can be used
    else:
        status = 'unavailable' # no NVIDIA GPU: pack would do nothing
    bin_dir = cuda_bin_dir()
    size_mb = sum(f.stat().st_size for f in bin_dir.rglob('*') if f.is_file()) / 2**20 if installed else None
    return {
        'id': 'cuda-gpu-pack',
        'name': 'GPU Acceleration Pack (NVIDIA CUDA 12)',
        'installed': installed,
        'status': status,
        'progress': job.get('progress', 0) if job else 0,
        'downloaded_mb': round(job.get('downloaded_mb', 0), 1) if job else 0,
        'total_mb': round(job.get('total_mb', 0), 1) if job else 1600,
        'error': job.get('error') if job else None,
        'size_mb': round(size_mb, 1) if installed else None,
        'path': str(bin_dir) if installed else None,
        'gpu': gpu,
    }


def _version_key(version: str):
    try:
        from packaging.version import parse

        return parse(version)
    except Exception:
        return tuple(int(part) if part.isdigit() else 0 for part in version.split('.'))


def _wheel_download_url(package: str) -> tuple[str, int]:
    """Highest-version win_amd64 wheel URL + size for an NVIDIA runtime package.

    PyPI's releases mapping is unordered, so "first match" can return an
    ancient wheel with a different DLL layout — always pick the newest.
    """
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/json', timeout=60) as response:
        payload = json.loads(response.read().decode('utf-8'))
    best = None
    for version, release_files in payload.get('releases', {}).items():
        for entry in release_files:
            if not entry.get('filename', '').endswith('win_amd64.whl'):
                continue
            key = _version_key(version)
            if best is None or key > best[0]:
                best = (key, entry['url'], int(entry.get('size') or 0))
    if best is None:
        raise RuntimeError(f'No Windows wheel found for {package}')
    return best[1], best[2]


def _download_and_extract(url: str, dest_dir: Path, wanted_subdir: str, wanted_dll: str,
                          base_fraction: float, span: float, progress_cb=None) -> None:
    ssl_context = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=ssl_context) as response:
        total = int(response.headers.get('Content-Length') or 0)
        wheel_path = dest_dir / 'pack-download.whl'
        downloaded = 0
        with open(wheel_path, 'wb') as out:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if progress_cb and total:
                    progress_cb(base_fraction + span * min(1.0, downloaded / total))
    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.namelist():
            if member.startswith(wanted_subdir) and member.endswith('.dll'):
                target = dest_dir / Path(member).name
                with archive.open(member) as source, open(target, 'wb') as out:
                    shutil.copyfileobj(source, out)
    wheel_path.unlink(missing_ok=True)
    # Sanity check: the pack is only complete when the anchor DLL is present.
    if not (dest_dir / wanted_dll).is_file():
        raise RuntimeError(f'{wanted_dll} was not found in the downloaded package')


def start_download() -> dict:
    with _LOCK:
        job = _DOWNLOADS.get('cuda-gpu-pack')
        if job and job.get('status') == 'downloading':
            return job
        if is_pack_installed():
            return {'status': 'installed', 'progress': 100}
        job = {'status': 'downloading', 'progress': 0, 'downloaded_mb': 0.0, 'total_mb': 1600.0, 'error': None}
        _DOWNLOADS['cuda-gpu-pack'] = job

    def worker() -> None:
        try:
            bin_dir = cuda_bin_dir()
            spans = [(0.0, 50.0), (50.0, 100.0)]  # cublas first, then cudnn
            for (name, (package, anchor_dll, wheel_subdir)), (base_fraction, span) in zip(_PACK_WHEELS.items(), spans):
                url, size_bytes = _wheel_download_url(package)
                _download_and_extract(url, bin_dir, wheel_subdir, anchor_dll,
                                      progress_cb=lambda f, size_bytes=size_bytes: job.update(
                                          progress=min(99, int(f)),
                                          downloaded_mb=round(size_bytes / 2**20 * min(100.0, f) / 100, 1)),
                                      base_fraction=base_fraction, span=span)
            if not is_pack_installed():
                raise RuntimeError('Pack extraction did not produce the required DLLs')
            job.update(status='installed', progress=100)
            activate_dll_search_path()  # usable immediately, no restart needed
        except Exception as error:  # noqa: BLE001
            job.update(status='error', error=str(error))

    threading.Thread(target=worker, name='download-cuda-pack', daemon=True).start()
    return job


def delete_pack() -> None:
    with _LOCK:
        job = _DOWNLOADS.get('cuda-gpu-pack')
        if job and job.get('status') == 'downloading':
            raise ValueError('Cannot delete the pack while it is downloading.')
    bin_dir = cuda_bin_dir()
    if bin_dir.exists():
        shutil.rmtree(bin_dir.parent, ignore_errors=True)


def activate_dll_search_path() -> None:
    """Register the pack's DLL folder for this process (call once at startup)."""
    if is_pack_installed():
        import os

        bin_dir = str(cuda_bin_dir())
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(bin_dir)
        except Exception:
            pass
