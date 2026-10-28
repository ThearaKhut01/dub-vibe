"""Qwen3 4B local translation engine (llama.cpp, fully offline).

The model is downloaded through the model manager into
%LOCALAPPDATA%\\DubVibe Studio\\models\\qwen\\ and runs with llama-cpp-python
on the CPU. Translation quality beats dictionary-based Argos for natural
phrasing, at the cost of speed on CPU.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from .providers import TranslationProvider, models_root

QWEN_MODEL_ID = 'qwen3-4b'
QWEN_REPO = 'unsloth/Qwen3-4B-Instruct-2507-GGUF'
QWEN_FILENAME = 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
QWEN_SIZE_MB = 2430

_LLM = None
_LLM_LOCK = threading.Lock()
# A Llama instance is NOT thread-safe: create_chat_completion mutates the
# shared context/KV buffers, and concurrent calls segfault inside ggml-cpu.dll
# (observed as 0xc0000005 crashes of the packaged sidecar mid-translation).
_INFERENCE_LOCK = threading.Lock()


def qwen_model_dir() -> Path:
    directory = models_root() / 'qwen' / QWEN_MODEL_ID
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def qwen_model_file() -> Path:
    return qwen_model_dir() / QWEN_FILENAME


def is_installed() -> bool:
    return qwen_model_file().is_file() and qwen_model_file().stat().st_size > 1024 * 1024 * 100


_RUNTIME_PROBLEM: str | None | bool = False
_RUNTIME_LOCK = threading.Lock()


def _missing_cuda_runtime() -> list[str]:
    """CUDA DLLs a CUDA-linked llama.cpp needs but this machine cannot load."""
    import ctypes
    import importlib.util

    try:
        spec = importlib.util.find_spec('llama_cpp')
    except Exception:
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    lib_dir = Path(list(spec.submodule_search_locations)[0]) / 'lib'
    if not (lib_dir / 'ggml-cuda.dll').is_file():
        return []  # CPU-only build: nothing CUDA to miss
    missing: list[str] = []
    for name in ('cudart64_12.dll', 'cublas64_12.dll', 'nvcuda.dll'):
        try:
            local = lib_dir / name
            ctypes.WinDLL(str(local) if local.is_file() else name)
        except OSError:
            missing.append(name)
    return missing


def runtime_problem() -> str | None:
    """Why llama.cpp cannot start on this machine, or None when it can.

    Importing llama_cpp loads llama.dll. A CUDA-enabled build fails at that
    import on a machine without the CUDA runtime/driver, and the translation
    chain would then quietly send every line to the fallback engine — so this
    check runs before a run starts instead of after 200 silent Google calls.
    """
    global _RUNTIME_PROBLEM
    with _RUNTIME_LOCK:
        if _RUNTIME_PROBLEM is not False:
            return _RUNTIME_PROBLEM
        problem: str | None = None
        try:
            import llama_cpp  # noqa: F401  (import loads llama.dll)
        except Exception as error:  # OSError from ctypes, ImportError, ...
            detail = str(error)
            lowered = detail.lower()
            missing = _missing_cuda_runtime()
            if missing:
                detail = (
                    'this llama.cpp build links the NVIDIA CUDA backend, but '
                    f'{", ".join(missing)} cannot be loaded (no NVIDIA driver/runtime)'
                )
            elif 'cudart' in lowered or 'cublas' in lowered or 'nvcuda' in lowered:
                detail = (
                    'this llama.cpp build needs the NVIDIA CUDA runtime, which is '
                    'not installed on this machine'
                )
            elif 'llama.dll' in lowered or 'shared library' in lowered:
                detail = 'llama.cpp could not load its libraries (llama.dll)'
            problem = detail
        _RUNTIME_PROBLEM = problem
        return problem


def clear_runtime_problem() -> None:
    """Forget a cached failure (after installing/replacing llama.cpp)."""
    global _RUNTIME_PROBLEM
    with _RUNTIME_LOCK:
        _RUNTIME_PROBLEM = False


def download_with_progress(progress_cb=None) -> None:
    """Blocking download of the Qwen3 4B GGUF file."""
    from .hf_progress import download_file

    target = qwen_model_dir()
    target.mkdir(parents=True, exist_ok=True)

    def on_progress(percent: int) -> None:
        if progress_cb:
            progress_cb(min(99, percent))

    download_file(
        QWEN_REPO,
        QWEN_FILENAME,
        local_dir=target,
        progress_cb=on_progress,
    )
    if progress_cb:
        progress_cb(100)


def _get_llm():
    """Load (once) the Qwen model for translation. Thread-safe.

    Honors the Model Manager device setting: 'cpu' forces CPU inference,
    'cuda'/'auto' offload every layer to the GPU when one is available.
    """
    global _LLM
    with _LLM_LOCK:
        if _LLM is None:
            if not is_installed():
                raise RuntimeError('Qwen3 4B is not installed. Download it in the AI Model Manager first.')
            from llama_cpp import Llama

            try:
                from .model_manager import resolve_device_compute

                device, _ = resolve_device_compute()
            except Exception:
                device = 'cpu'
            _LLM = Llama(
                model_path=str(qwen_model_file()),
                n_ctx=4096,
                n_threads=max(2, os.cpu_count() or 2),
                n_gpu_layers=-1 if device == 'cuda' else 0,
                verbose=False,
            )
        return _LLM


def unload() -> None:
    global _LLM
    with _LLM_LOCK:
        _LLM = None


class QwenTranslationProvider(TranslationProvider):
    """Natural, context-aware translation with the local Qwen3 4B model."""

    def __init__(self, style_hint: str | None = None) -> None:
        self.style_hint = style_hint

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        llm = _get_llm()
        system = (
            'You are a professional subtitle translator for movie dubbing. '
            f'Translate the user\'s line from {source_language} to {target_language}. '
            'Keep it natural, spoken, and concise — like real movie dialogue. '
            'Preserve tone and emotion. Respond ONLY with the translation, no notes. '
            'Write personal names phonetically in the target language\'s own script.'
        )
        if self.style_hint:
            system += f' Character/style direction: {self.style_hint}'
        with _INFERENCE_LOCK:
            completion = llm.create_chat_completion(
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': text},
                ],
                max_tokens=512,
                temperature=0.2,
            )
        translated = str(completion['choices'][0]['message']['content']).strip()
        # Qwen3 is a thinking-capable model: its reasoning or an over-eager
        # answer can come back as analysis instead of the line. The validators
        # live in providers (shared with the online engines). Local inference is
        # free, so the cleanup is always zero-cost — no second generation.
        from .providers import _extract_subtitle_answer, _strip_reasoning, _subtitle_length_ok, _translation_script_ok

        translated = _strip_reasoning(translated).strip()
        if not translated:
            raise RuntimeError('Qwen returned an empty translation')
        if _translation_script_ok(translated, text, source_language, target_language) and _subtitle_length_ok(
                translated, text):
            return translated
        salvage = _extract_subtitle_answer(translated, text, source_language, target_language)
        if salvage:
            return salvage
        # Nothing usable: return what was generated (normal behavior) rather
        # than triggering a paid fallback for a local-only run.
        return translated
