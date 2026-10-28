"""CPU/NVIDIA llama.cpp selection: the same app must run Qwen on both.

The bundle ships a CPU build (works on every laptop). An NVIDIA machine may
download the CUDA build, which is selected before llama_cpp is imported.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from backend.app import llama_backend


@pytest.fixture(autouse=True)
def _reset_choice():
    llama_backend.reset()
    yield
    llama_backend.reset()


def _fake_cpu_lib(tmp_path: Path) -> Path:
    lib = tmp_path / 'cpu-lib'
    lib.mkdir()
    (lib / 'llama.dll').write_bytes(b'fake llama')
    return lib


def test_cuda_wheel_url_matches_the_installed_version(monkeypatch):
    monkeypatch.setattr(llama_backend, 'llama_version', lambda: '0.3.32')
    assert llama_backend.cuda_wheel_url() == (
        'https://github.com/abetlen/llama-cpp-python/releases/download/'
        'v0.3.32-cu124/llama_cpp_python-0.3.32-py3-none-win_amd64.whl'
    )


def test_cpu_is_used_on_a_laptop_without_nvidia(monkeypatch, tmp_path):
    lib = _fake_cpu_lib(tmp_path)
    monkeypatch.setattr(llama_backend, 'cpu_lib_dir', lambda: lib)
    monkeypatch.setattr(llama_backend, 'has_nvidia_gpu', lambda: False)
    monkeypatch.setattr(llama_backend, 'cuda_backend_installed', lambda: False)
    choice = llama_backend.prepare()
    assert choice['backend'] == 'cpu'
    assert choice['lib_dir'] == str(lib)
    assert 'LLAMA_CPP_LIB_PATH' not in __import__('os').environ


def test_cuda_is_used_when_gpu_and_backend_are_present(monkeypatch, tmp_path):
    lib = _fake_cpu_lib(tmp_path)
    cuda = tmp_path / 'cuda-lib'
    cuda.mkdir()
    monkeypatch.setattr(llama_backend, 'cpu_lib_dir', lambda: lib)
    monkeypatch.setattr(llama_backend, 'cuda_backend_dir', lambda version=None: cuda)
    monkeypatch.setattr(llama_backend, 'cuda_backend_installed', lambda: True)
    monkeypatch.setattr(llama_backend, 'has_nvidia_gpu', lambda: True)
    monkeypatch.setattr(llama_backend, '_runtime_ready', lambda: (True, []))
    choice = llama_backend.prepare()
    assert choice['backend'] == 'cuda'
    assert choice['lib_dir'] == str(cuda)
    assert __import__('os').environ['LLAMA_CPP_LIB_PATH'] == str(cuda)


def test_broken_cuda_backend_falls_back_to_cpu_with_the_reason(monkeypatch, tmp_path):
    lib = _fake_cpu_lib(tmp_path)
    cuda = tmp_path / 'cuda-lib'
    cuda.mkdir()
    monkeypatch.setattr(llama_backend, 'cpu_lib_dir', lambda: lib)
    monkeypatch.setattr(llama_backend, 'cuda_backend_dir', lambda version=None: cuda)
    monkeypatch.setattr(llama_backend, 'cuda_backend_installed', lambda: True)
    monkeypatch.setattr(llama_backend, 'has_nvidia_gpu', lambda: True)
    monkeypatch.setattr(
        llama_backend, '_runtime_ready',
        lambda: (False, ['nvcuda.dll']),
    )
    choice = llama_backend.prepare()
    assert choice['backend'] == 'cpu'
    assert 'nvcuda.dll' in (choice['reason'] or '')
    assert 'LLAMA_CPP_LIB_PATH' not in __import__('os').environ


def test_force_cpu_wins_after_a_failed_cuda_import(monkeypatch, tmp_path):
    lib = _fake_cpu_lib(tmp_path)
    cuda = tmp_path / 'cuda-lib'
    cuda.mkdir()
    monkeypatch.setattr(llama_backend, 'cpu_lib_dir', lambda: lib)
    monkeypatch.setattr(llama_backend, 'cuda_backend_dir', lambda version=None: cuda)
    monkeypatch.setattr(llama_backend, 'cuda_backend_installed', lambda: True)
    monkeypatch.setattr(llama_backend, 'has_nvidia_gpu', lambda: True)
    monkeypatch.setattr(llama_backend, '_runtime_ready', lambda: (True, []))
    assert llama_backend.prepare()['backend'] == 'cuda'
    llama_backend.force_cpu()
    choice = llama_backend.prepare()
    assert choice['backend'] == 'cpu'
    assert 'LLAMA_CPP_LIB_PATH' not in __import__('os').environ
    assert 'failed earlier' in (choice['reason'] or '')


def test_cuda_download_keeps_only_the_backend_dlls(tmp_path):
    wheel = tmp_path / 'build.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('llama_cpp/lib/llama.dll', b'llama')
        archive.writestr('llama_cpp/lib/ggml-cuda.dll', b'cuda')
        archive.writestr('llama_cpp/lib/README.md', b'notes')
        archive.writestr('llama_cpp/llama.py', b'python')
    target = tmp_path / 'out'
    llama_backend._extract_lib_dlls(wheel, target)
    assert sorted(path.name for path in target.iterdir()) == ['ggml-cuda.dll', 'llama.dll']


def test_gpu_pack_also_installs_the_cuda_runtime_llama_needs():
    from backend.app import cuda_pack

    assert 'cudart' in cuda_pack._PACK_WHEELS
    _, anchor, subdir = cuda_pack._PACK_WHEELS['cudart']
    assert anchor == 'cudart64_12.dll'
    assert subdir == 'nvidia/cuda_runtime/bin'
