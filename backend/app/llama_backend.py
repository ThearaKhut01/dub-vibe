"""Pick the llama.cpp runtime: CPU everywhere, NVIDIA CUDA when it can work.

The app ships a **CPU-only** llama.cpp build, so Qwen3 4B starts on every
laptop (that is the only build that can load without an NVIDIA driver). On a
machine with an NVIDIA GPU — and the CUDA runtime DLLs, either from the GPU
Acceleration Pack or a CUDA install — the official CUDA build of the *same*
llama-cpp-python version is downloaded into the model store and selected before
``llama_cpp`` is imported. `llama_cpp` honours ``LLAMA_CPP_LIB_PATH`` for
exactly this, so the Python side stays identical and there is still only one app.

Nothing here is fatal: when CUDA cannot load, the CPU build is used and the
reason is kept so the UI/run start can say so instead of pretending.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import ssl
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from .providers import models_root

# The official Windows CUDA wheels are tagged py3-none (llama.cpp is loaded
# through ctypes), so one wheel serves every Python version of the app.
_WHEEL_TAG = 'cu124'
_CHOICE: dict | None = None
_FORCE_CPU = False
_LOCK = threading.RLock()


def llama_version() -> str | None:
    """Installed llama-cpp-python version (the CUDA build must match it)."""
    try:
        from importlib.metadata import version

        return version('llama-cpp-python')
    except Exception:
        return None


def cpu_lib_dir() -> Path:
    """The DLLs bundled with the installed llama-cpp-python package."""
    from importlib.util import find_spec

    spec = find_spec('llama_cpp')
    if spec is None or not spec.submodule_search_locations:
        return Path('llama_cpp') / 'lib'
    return Path(list(spec.submodule_search_locations)[0]) / 'lib'


def cuda_backend_dir(version: str | None = None) -> Path:
    """Where the downloaded CUDA DLLs live (per llama-cpp-python version)."""
    name = version or llama_version() or 'unknown'
    return models_root() / 'llama-cuda' / name / 'lib'


def cuda_backend_installed() -> bool:
    directory = cuda_backend_dir()
    return (directory / 'llama.dll').is_file() and (directory / 'ggml-cuda.dll').is_file()


def cuda_backend_size_mb() -> float | None:
    directory = cuda_backend_dir()
    if not cuda_backend_installed():
        return None
    total = sum(path.stat().st_size for path in directory.glob('*') if path.is_file())
    return round(total / 2**20, 1)


def cuda_wheel_url(version: str | None = None) -> str:
    version = version or llama_version()
    if not version:
        raise RuntimeError('llama-cpp-python is not installed; cannot pick a CUDA build.')
    return (
        'https://github.com/abetlen/llama-cpp-python/releases/download/'
        f'v{version}-{_WHEEL_TAG}/llama_cpp_python-{version}-py3-none-win_amd64.whl'
    )


_GPU_CHECK: tuple[float, bool] | None = None
_GPU_TTL = 60.0


def has_nvidia_gpu(force: bool = False) -> bool:
    """True when an NVIDIA driver is present (nvcuda.dll) or nvidia-smi answers."""
    global _GPU_CHECK
    if os.name != 'nt':
        return False
    if not force and _GPU_CHECK is not None and time.monotonic() - _GPU_CHECK[0] < _GPU_TTL:
        return _GPU_CHECK[1]
    detected = False
    try:
        ctypes.WinDLL('nvcuda.dll')
        detected = True
    except OSError:
        try:
            from . import cuda_pack

            detected = cuda_pack.nvidia_gpu_info() is not None
        except Exception:
            detected = False
    _GPU_CHECK = (time.monotonic(), detected)
    return detected


def _runtime_dirs() -> list[Path]:
    """Folders that hold cudart/cublas for the CUDA build.

    The GPU Acceleration Pack, the NVIDIA pip packages (bundled or installed by
    torch/ctranslate2) and a CUDA toolkit install are all valid sources; adding
    every one that exists is harmless when CUDA is never used.
    """
    bases = [Path(sys.prefix) / 'Lib' / 'site-packages']
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        bases.append(Path(meipass))
    directories: list[Path] = []
    for base in bases:
        directories.extend(path for path in base.glob('nvidia/*/bin') if path.is_dir())
        torch_lib = base / 'torch' / 'lib'
        if torch_lib.is_dir():
            directories.append(torch_lib)
    pack = models_root() / 'cuda' / 'bin'
    if pack.is_dir():
        directories.append(pack)
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        directories.append(Path(cuda_path) / 'bin')
    return directories


def _register_runtime_dirs() -> None:
    for directory in _runtime_dirs():
        try:
            os.add_dll_directory(str(directory))
        except Exception:
            pass
        os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')


def _runtime_ready() -> tuple[bool, list[str]]:
    """Can the CUDA llama.dll's dependencies load (driver + CUDA 12 runtime)?"""
    missing: list[str] = []
    for name in ('nvcuda.dll', 'cudart64_12.dll', 'cublas64_12.dll'):
        try:
            ctypes.WinDLL(name)
        except OSError:
            missing.append(name)
    return not missing, missing


def prepare() -> dict:
    """Choose the backend once per process, before llama_cpp is imported.

    Returns {'backend': 'cpu'|'cuda'|'none', 'lib_dir': ..., 'reason': ...}.
    'none' means no llama.cpp build is usable at all (not installed).
    """
    global _CHOICE
    with _LOCK:
        if _CHOICE is not None:
            return dict(_CHOICE)
        _register_runtime_dirs()
        choice = {
            'backend': 'cpu',
            'lib_dir': str(cpu_lib_dir()),
            'reason': None,
            'gpu': has_nvidia_gpu(),
        }
        if not (cpu_lib_dir() / 'llama.dll').is_file():
            choice['backend'] = 'none'
            choice['reason'] = 'llama-cpp-python is not installed'
            _CHOICE = choice
            return dict(choice)
        if _FORCE_CPU:
            choice['reason'] = 'CUDA backend failed earlier in this session; using CPU'
            _CHOICE = choice
            return dict(choice)
        if not cuda_backend_installed():
            if choice['gpu']:
                choice['reason'] = (
                    'NVIDIA GPU detected, but the CUDA translation backend is not '
                    'downloaded; using CPU (install the GPU Acceleration Pack to add it)'
                )
            _CHOICE = choice
            return dict(choice)
        if not choice['gpu']:
            choice['reason'] = 'CUDA backend installed, but no NVIDIA GPU/driver: using CPU'
            _CHOICE = choice
            return dict(choice)

        ready, missing = _runtime_ready()
        if ready:
            cuda_dir = cuda_backend_dir()
            os.environ['LLAMA_CPP_LIB_PATH'] = str(cuda_dir)
            choice.update(backend='cuda', lib_dir=str(cuda_dir))
        else:
            os.environ.pop('LLAMA_CPP_LIB_PATH', None)
            choice['reason'] = (
                'the CUDA translation backend needs '
                f'{", ".join(missing)} (install the GPU Acceleration Pack or the '
                'NVIDIA CUDA 12 runtime); using CPU'
            )
        _CHOICE = choice
        return dict(choice)


def force_cpu() -> None:
    """Remember that CUDA failed at import time so the next import uses CPU."""
    global _CHOICE, _FORCE_CPU
    with _LOCK:
        _FORCE_CPU = True
        _CHOICE = None
        os.environ.pop('LLAMA_CPP_LIB_PATH', None)


def reset() -> None:
    """Forget the choice (after installing/removing the CUDA backend)."""
    global _CHOICE, _FORCE_CPU
    with _LOCK:
        _CHOICE = None
        _FORCE_CPU = False
        os.environ.pop('LLAMA_CPP_LIB_PATH', None)


def backend_label() -> str:
    backend = prepare()['backend']
    return {'cuda': 'NVIDIA GPU', 'cpu': 'CPU', 'none': 'not installed'}.get(backend, backend)


def status() -> dict:
    choice = prepare()
    return {
        'backend': choice['backend'],
        'label': backend_label(),
        'reason': choice['reason'],
        'lib_dir': choice['lib_dir'],
        'version': llama_version(),
        'gpu': has_nvidia_gpu(),
        'cuda_installed': cuda_backend_installed(),
        'cuda_size_mb': cuda_backend_size_mb(),
    }


def download_cuda_backend(progress_cb=None) -> Path:
    """Download the official CUDA llama.cpp DLLs for the installed version.

    Only the ``llama_cpp/lib`` DLLs are kept (~950 MB of the wheel is the CUDA
    backend); the wheel is deleted afterwards, leaving no second Python package.
    """
    version = llama_version()
    if not version:
        raise RuntimeError('Install llama-cpp-python first (it ships with the app).')
    target = cuda_backend_dir(version)
    target.mkdir(parents=True, exist_ok=True)
    url = cuda_wheel_url(version)
    context = ssl.create_default_context()

    with tempfile.TemporaryDirectory(prefix='dubvibe-llama-cuda-') as work:
        wheel = Path(work) / 'cuda-backend.whl'
        with urllib.request.urlopen(url, timeout=120, context=context) as response:
            total = int(response.headers.get('Content-Length') or 0)
            done = 0
            with open(wheel, 'wb') as out:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress_cb and total:
                        progress_cb(min(99.0, 100.0 * done / total))
        _extract_lib_dlls(wheel, target)

    if not cuda_backend_installed():
        raise RuntimeError('The downloaded CUDA backend is incomplete (llama.dll/ggml-cuda.dll missing).')
    if progress_cb:
        progress_cb(100.0)
    reset()
    return target


def _extract_lib_dlls(wheel: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.startswith('llama_cpp/lib/') and member.lower().endswith('.dll'):
                name = Path(member).name
                with archive.open(member) as source, open(target / name, 'wb') as out:
                    shutil.copyfileobj(source, out)
