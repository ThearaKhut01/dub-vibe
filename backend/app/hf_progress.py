"""HuggingFace downloads with real, byte-accurate progress.

Why the Hub's own progress is not enough here:
- `hf_hub_download` has no `tqdm_class` parameter, so passing it raises
  `TypeError: hf_hub_download() got an unexpected keyword argument 'tqdm_class'`.
- When the optional `hf_xet` backend is installed, Hub downloads go through Xet,
  which reported **0 bytes** of progress for the whole transfer (measured) and only
  writes the file at the end.

So the primary path streams the resolved file URL with `requests` and reports the
bytes actually received. That is exact, resumable, and independent of the Hub's
transport choice. If that path fails, the wrappers fall back to the Hub functions
with whatever progress hook the installed version supports.
"""
from __future__ import annotations

import fnmatch
import inspect
import threading
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[int], None]

_CHUNK_BYTES = 64 * 1024
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 90


def _supports_parameter(function: Callable[..., Any], parameter: str) -> bool:
    """True when `function` accepts `parameter` (version check, never assumes)."""
    try:
        return parameter in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


class _Progress:
    """Percent reporter that never floods the UI with duplicate values.

    Reports 0-99; the caller reports 100 once the file is in place. Values are
    clamped and monotonic, so a retried transfer cannot make the bar jump back.
    """

    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._last = -1
        self._lock = threading.Lock()

    def bytes(self, done: int, total: int | None) -> None:
        if self._callback is None or not total or total <= 0:
            return
        percent = max(0, min(99, int(done * 100 / total)))
        with self._lock:
            if percent <= self._last:
                return
            self._last = percent
        try:
            self._callback(percent)
        except Exception:
            pass  # progress reporting must never break a download


def _hub_token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _stream_to_path(url: str, destination: Path, *, total_bytes: int | None,
                    progress: _Progress, token: str | None = None) -> None:
    """Stream `url` into `destination`, resuming a partial `.part` file if present."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f'{destination.name}.part')
    resume_from = part.stat().st_size if part.is_file() else 0

    headers = {'Accept-Encoding': 'identity'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    if resume_from:
        headers['Range'] = f'bytes={resume_from}-'

    with requests.get(url, headers=headers, stream=True,
                      timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)) as response:
        if response.status_code == 416 and resume_from:
            # The partial file already holds the whole download.
            part.replace(destination)
            progress.bytes(resume_from, total_bytes or resume_from)
            return
        response.raise_for_status()

        # The server may ignore the Range request and send the file from byte 0.
        if resume_from and response.status_code != 206:
            resume_from = 0

        try:
            length = int(response.headers.get('Content-Length', ''))
        except ValueError:
            length = None
        total = (resume_from + length) if length is not None else total_bytes

        done = resume_from
        with part.open('ab' if resume_from else 'wb') as handle:
            for chunk in response.iter_content(_CHUNK_BYTES):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                progress.bytes(done, total)

    if total and part.stat().st_size != total:
        raise IOError(f'incomplete download: {part.stat().st_size} of {total} bytes')
    part.replace(destination)


# --- Progress hooks for the Hub's own downloaders (fallback path) ----------

def make_reporting_tqdm(progress_cb: ProgressCallback | None) -> type | None:
    """tqdm subclass reporting aggregate progress of every active bar."""
    if progress_cb is None:
        return None
    try:
        from tqdm import tqdm as base_tqdm
    except Exception:
        return None

    bars: dict[int, dict[str, float]] = {}
    lock = threading.Lock()
    reporter = _Progress(progress_cb)

    def report() -> None:
        with lock:
            done = sum(entry['n'] for entry in bars.values())
            total = sum(entry['total'] for entry in bars.values())
        reporter.bytes(int(done), int(total) or None)

    class ReportingTqdm(base_tqdm):  # type: ignore[misc, valid-type]
        def update(self, n=1):
            result = super().update(n)
            with lock:
                bars[id(self)] = {'n': float(self.n or 0), 'total': float(self.total or 0)}
            report()
            return result

        def close(self):
            with lock:
                entry = bars.get(id(self))
                if entry is not None:
                    entry['n'] = entry['total']
            report()
            super().close()

    return ReportingTqdm


_TQDM_PATCH_LOCK = threading.Lock()
_TQDM_PATCH_SAVED: list[tuple[Any, Any]] = []


def _hub_progress_modules() -> list[Any]:
    """Modules that build Hub progress bars from their own module-level `tqdm`.

    The bar is created by `_get_progress_bar_context`, which resolves `tqdm` from
    the `huggingface_hub.utils.tqdm` module globals, so that module is patched
    (the package attribute of the same name is the class, not the module, which is
    why `sys.modules` is used).
    """
    import importlib
    import sys

    importlib.import_module('huggingface_hub.file_download')

    modules: list[Any] = []
    for name in ('huggingface_hub.utils.tqdm', 'huggingface_hub.file_download'):
        module = sys.modules.get(name)
        if module is not None:
            modules.append(module)
    return modules


def _patch_hub_tqdm(tqdm_class: type) -> bool:
    """Route HuggingFace's own progress bar to `tqdm_class` for one call."""
    if not _TQDM_PATCH_LOCK.acquire(blocking=False):
        return False
    saved: list[tuple[Any, Any]] = []
    try:
        for module in _hub_progress_modules():
            original = getattr(module, 'tqdm', None)
            if original is None or original is tqdm_class:
                continue
            setattr(module, 'tqdm', tqdm_class)
            saved.append((module, original))
    except Exception:
        for module, original in saved:
            setattr(module, 'tqdm', original)
        _TQDM_PATCH_LOCK.release()
        return False
    if not saved:
        _TQDM_PATCH_LOCK.release()
        return False
    _TQDM_PATCH_SAVED.extend(saved)
    return True


def _restore_hub_tqdm() -> None:
    """Undo `_patch_hub_tqdm` and release the patch lock."""
    try:
        while _TQDM_PATCH_SAVED:
            module, original = _TQDM_PATCH_SAVED.pop()
            setattr(module, 'tqdm', original)
    except Exception:
        pass
    finally:
        _TQDM_PATCH_LOCK.release()


def _call_hub(function: Callable[..., Any], kwargs: dict[str, Any],
              progress_cb: ProgressCallback | None) -> Any:
    """Call a Hub download function with the progress mechanism it supports."""
    tqdm_class = make_reporting_tqdm(progress_cb)
    if tqdm_class is not None and _supports_parameter(function, 'tqdm_class'):
        return function(**kwargs, tqdm_class=tqdm_class)
    if tqdm_class is not None and _patch_hub_tqdm(tqdm_class):
        try:
            return function(**kwargs)
        finally:
            _restore_hub_tqdm()
    return function(**kwargs)


# --- Public API -------------------------------------------------------------

def download_file(repo_id: str, filename: str, *, local_dir: str | Path | None = None,
                  repo_type: str | None = None, revision: str | None = None,
                  progress_cb: ProgressCallback | None = None) -> str:
    """Download one file with real progress; returns its local path.

    With `local_dir` the file is streamed there (resumable, exact byte progress).
    Without it the Hub owns the cache location, so the Hub function is used with
    whatever progress hook the installed version supports.
    """
    from huggingface_hub import hf_hub_download, hf_hub_url

    kwargs: dict[str, Any] = {'repo_id': repo_id, 'filename': filename}
    if local_dir is not None:
        kwargs['local_dir'] = str(local_dir)
    if repo_type is not None:
        kwargs['repo_type'] = repo_type
    if revision is not None:
        kwargs['revision'] = revision

    if local_dir is None:
        return str(_call_hub(hf_hub_download, kwargs, progress_cb))

    destination = Path(local_dir) / filename
    url = hf_hub_url(repo_id, filename, repo_type=repo_type, revision=revision)
    try:
        _stream_to_path(url, destination, total_bytes=None, progress=_Progress(progress_cb),
                        token=_hub_token())
        return str(destination)
    except Exception:
        # Any partial file stays for the Hub's own resume logic.
        return str(_call_hub(hf_hub_download, kwargs, progress_cb))


def _repo_files(repo_id: str, allow_patterns: list[str] | None) -> list[tuple[str, int]]:
    """(filename, size) pairs for the repo, filtered by `allow_patterns`."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True, token=_hub_token())
    files: list[tuple[str, int]] = []
    for sibling in info.siblings or []:
        name = getattr(sibling, 'rfilename', None)
        if not name:
            continue
        if allow_patterns and not any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
            continue
        files.append((name, int(getattr(sibling, 'size', 0) or 0)))
    return files


def download_snapshot(repo_id: str, *, local_dir: str | Path,
                      allow_patterns: list[str] | None = None,
                      progress_cb: ProgressCallback | None = None) -> str:
    """Download a repo snapshot with real progress aggregated across its files."""
    from huggingface_hub import hf_hub_url, snapshot_download

    target = Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    reporter = _Progress(progress_cb)
    try:
        files = _repo_files(repo_id, allow_patterns)
        if not files:
            raise RuntimeError('the repository listing is empty')
        total = sum(size for _name, size in files)
        done_before = 0
        token = _hub_token()
        for name, size in files:
            existing = target / name
            if size and existing.is_file() and existing.stat().st_size == size:
                done_before += size
                reporter.bytes(done_before, total or None)
                continue

            def file_progress(percent: int, base: int = done_before, size: int = size) -> None:
                reporter.bytes(base + int(size * percent / 100), total or None)

            _stream_to_path(
                hf_hub_url(repo_id, name), existing,
                total_bytes=size or None,
                progress=_Progress(file_progress if progress_cb is not None else None),
                token=token,
            )
            done_before += size
            reporter.bytes(done_before, total or None)
        return str(target)
    except Exception:
        kwargs: dict[str, Any] = {'repo_id': repo_id, 'local_dir': str(target)}
        if allow_patterns:
            kwargs['allow_patterns'] = list(allow_patterns)
        return str(_call_hub(snapshot_download, kwargs, progress_cb))
