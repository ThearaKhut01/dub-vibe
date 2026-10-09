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


def download_with_progress(progress_cb=None) -> None:
    """Blocking download of the Qwen3 4B GGUF file."""
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm as base_tqdm

    qwen_model_dir().mkdir(parents=True, exist_ok=True)
    done = threading.Event()

    def on_fraction(fraction: float) -> None:
        if progress_cb:
            progress_cb(min(99, int(fraction)))

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
                    on_fraction(100.0 * done_sum / total)
            except Exception:
                pass
            return result

        def close(self):
            try:
                with lock:
                    entry = bars.get(id(self))
                    if entry is not None:
                        entry['n'] = entry['total']
                        done_sum = sum(entry_['n'] for entry_ in bars.values())
                        total = sum(entry_['total'] for entry_ in bars.values())
                if total > 0:
                    on_fraction(100.0 * done_sum / total)
            except Exception:
                pass
            super().close()

    def monitor() -> None:
        while not done.is_set():
            if progress_cb and qwen_model_file().exists():
                progress_cb(99)
            import time
            time.sleep(1.0)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        hf_hub_download(repo_id=QWEN_REPO, filename=QWEN_FILENAME, local_dir=str(qwen_model_dir()), tqdm_class=ReportingTqdm)
    finally:
        done.set()
        monitor_thread.join(timeout=2)
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
            'Preserve tone and emotion. Respond ONLY with the translation, no notes.'
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
        if not translated:
            raise RuntimeError('Qwen returned an empty translation')
        return translated
