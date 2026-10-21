"""Model downloads must report real bytes and survive Hub API changes.

Two independent regressions are pinned here:
1. `hf_hub_download` has no `tqdm_class` parameter — passing it raised
   "hf_hub_download() got an unexpected keyword argument 'tqdm_class'" and aborted
   the Qwen3 / Demucs downloads before anything was written.
2. The Hub's own progress is unusable once `hf_xet` is installed: it reported 0
   bytes for a whole 52 MB transfer. The primary path streams the file and reports
   the bytes actually received.
"""
from __future__ import annotations

import http.server
import sys
import threading
import types
from pathlib import Path

import pytest

from backend.app import hf_progress


_LOOPBACK_OK: list[bool] = []


def _loopback_http_works() -> bool:
    """True when this machine can talk HTTP to 127.0.0.1.

    Some sandboxes/CI runners block loopback HTTP (the server accepts the
    connection but the body never arrives). Probing once, with a short timeout,
    keeps the suite honest: the download logic is still covered when loopback
    works, and these tests skip instead of each burning a 90 s timeout when it
    does not.
    """
    if _LOOPBACK_OK:
        return _LOOPBACK_OK[0]
    import http.client

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _PingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=2)
        connection.request('GET', '/ping')
        works = connection.getresponse().read() == b'ok'
    except Exception:
        works = False
    finally:
        server.shutdown()
        server.server_close()
    _LOOPBACK_OK.append(works)
    return works


class _PingHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):  # noqa: N802 - stdlib naming
        body = b'ok'
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


class _FileServer:
    """Local HTTP server serving one payload, with optional Range support."""

    def __init__(self, payload: bytes, send_length: bool = True) -> None:
        self.payload = payload
        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_GET(self):  # noqa: N802 - stdlib naming
                start = 0
                range_header = self.headers.get('Range')
                if range_header and range_header.startswith('bytes='):
                    start = int(range_header[6:].split('-')[0])
                body = outer.payload[start:]
                outer.requests.append({'start': start, 'range': range_header})
                self.send_response(206 if start else 200)
                self.send_header('Content-Type', 'application/octet-stream')
                if send_length:
                    self.send_header('Content-Length', str(len(body)))
                else:
                    # No length: the client must rely on the expected size.
                    self.send_header('Connection', 'close')
                    self.close_connection = True
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return

        self._server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self._server.server_address[1]}/file.bin'
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> '_FileServer':
        if not _loopback_http_works():
            pytest.skip('loopback HTTP is unavailable in this environment')
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        self._server.shutdown()
        self._server.server_close()
        return False


def _fake_hub(monkeypatch, *, tqdm_class: bool, url: str | None = None,
              files: list[tuple[str, int]] | None = None) -> dict:
    """Install a fake huggingface_hub module and record the calls it receives."""
    calls: dict = {}
    module = types.ModuleType('huggingface_hub')

    def write(filename: str, local_dir=None) -> str:
        target = Path(local_dir or '.') / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'x' * 1024)
        return str(target)

    if tqdm_class:
        def hf_hub_download(repo_id, filename, local_dir=None, tqdm_class=None):
            calls['tqdm_class'] = tqdm_class
            return write(filename, local_dir)

        def snapshot_download(repo_id, local_dir=None, allow_patterns=None, tqdm_class=None):
            calls['tqdm_class'] = tqdm_class
            return str(local_dir)
    else:
        # The real 0.36.x signature: no tqdm_class on hf_hub_download.
        def hf_hub_download(repo_id, filename, local_dir=None):
            calls['filename'] = filename
            return write(filename, local_dir)

        def snapshot_download(repo_id, local_dir=None, allow_patterns=None):
            return str(local_dir)

    siblings = [types.SimpleNamespace(rfilename=name, size=size) for name, size in (files or [])]
    module.hf_hub_download = hf_hub_download
    module.snapshot_download = snapshot_download
    module.get_token = lambda: None
    module.hf_hub_url = lambda repo_id, filename, repo_type=None, revision=None: url or 'http://127.0.0.1:1/x'
    module.HfApi = lambda: types.SimpleNamespace(
        model_info=lambda repo_id, files_metadata=False, token=None: types.SimpleNamespace(siblings=siblings)
    )
    monkeypatch.setitem(sys.modules, 'huggingface_hub', module)
    return calls


# --- The streaming path: real, byte-accurate progress -----------------------

def test_stream_reports_every_percent_from_real_bytes(tmp_path):
    payload = b'z' * (8 * 1024 * 1024 + 12345)
    seen: list[int] = []
    with _FileServer(payload) as server:
        hf_progress._stream_to_path(server.url, tmp_path / 'file.bin', total_bytes=None,
                                    progress=hf_progress._Progress(seen.append))
    assert (tmp_path / 'file.bin').read_bytes() == payload
    assert seen == list(range(100)), 'progress must advance one percent at a time'


def test_stream_resumes_a_partial_file(tmp_path):
    payload = b'a' * (1024 * 1024)
    destination = tmp_path / 'file.bin'
    part = destination.with_name('file.bin.part')
    part.write_bytes(payload[:400_000])

    seen: list[int] = []
    with _FileServer(payload) as server:
        hf_progress._stream_to_path(server.url, destination, total_bytes=len(payload),
                                    progress=hf_progress._Progress(seen.append))
        assert server.requests[0]['range'] == 'bytes=400000-'
    assert destination.read_bytes() == payload
    assert not part.exists()
    # It resumed instead of restarting from zero.
    assert seen[0] >= 38


def test_download_file_streams_into_local_dir(monkeypatch, tmp_path):
    payload = b'q' * 300_000
    seen: list[int] = []
    with _FileServer(payload) as server:
        _fake_hub(monkeypatch, tqdm_class=False, url=server.url)
        path = hf_progress.download_file('owner/repo', 'weights.bin', local_dir=tmp_path,
                                         progress_cb=seen.append)
    assert Path(path).read_bytes() == payload
    assert seen[-1] == 99


def test_incomplete_transfer_is_rejected(tmp_path):
    """A truncated body must fail loudly instead of leaving a corrupt model."""
    payload = b'k' * 500_000
    with _FileServer(payload, send_length=False) as server:
        with pytest.raises(IOError, match='incomplete download'):
            hf_progress._stream_to_path(server.url, tmp_path / 'x.bin',
                                        total_bytes=len(payload) * 2,
                                        progress=hf_progress._Progress(None))


# --- The fallback path: Hub API compatibility -------------------------------

def test_hf_hub_download_fallback_without_tqdm_class(monkeypatch, tmp_path):
    """The exact failure the user saw, on the Hub fallback code path."""
    calls = _fake_hub(monkeypatch, tqdm_class=False)
    path = hf_progress.download_file('owner/repo', 'weights.bin', local_dir=tmp_path)
    assert Path(path).is_file()
    assert calls['filename'] == 'weights.bin'


def test_hf_hub_download_uses_the_tqdm_hook_when_available(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tqdm_class=False)
    hf_progress.download_file('owner/repo', 'weights.bin', local_dir=tmp_path)
    assert 'tqdm_class' not in calls

    calls = _fake_hub(monkeypatch, tqdm_class=True)
    hf_progress.download_file('owner/repo', 'weights.bin', local_dir=tmp_path,
                              progress_cb=lambda percent: None)
    assert calls['tqdm_class'] is not None


def test_snapshot_falls_back_when_the_listing_is_empty(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tqdm_class=False)
    result = hf_progress.download_snapshot('owner/repo', local_dir=tmp_path)
    assert result == str(tmp_path)


def test_snapshot_reports_aggregate_bytes_across_files(monkeypatch, tmp_path):
    """Two files: progress reflects total bytes, not per-file restarts."""
    first, second = b'1' * 400_000, b'2' * 600_000
    with _FileServer(first + second) as server:
        # One URL per filename would need two servers; both resolve to the same
        # host here, and the size comes from the (fake) listing.
        _fake_hub(monkeypatch, tqdm_class=False, url=server.url,
                  files=[('a.bin', len(first)), ('b.bin', len(second))])
        seen: list[int] = []
        hf_progress.download_snapshot('owner/repo', local_dir=tmp_path, progress_cb=seen.append)
    assert (tmp_path / 'a.bin').is_file() and (tmp_path / 'b.bin').is_file()
    assert seen == sorted(seen), 'aggregate progress must not go backwards'
    assert seen[-1] == 99


def test_real_hub_signature_has_no_tqdm_class_for_hf_hub_download():
    """Guard: the original bug was assuming a kwarg that does not exist."""
    from huggingface_hub import hf_hub_download

    assert not hf_progress._supports_parameter(hf_hub_download, 'tqdm_class')


def test_hub_tqdm_patch_is_applied_and_restored():
    import importlib

    importlib.import_module('huggingface_hub.file_download')
    bar_module = sys.modules['huggingface_hub.utils.tqdm']
    original = bar_module.tqdm
    reporting = hf_progress.make_reporting_tqdm(lambda percent: None)
    assert reporting is not None

    assert hf_progress._patch_hub_tqdm(reporting) is True
    assert bar_module.tqdm is reporting
    hf_progress._restore_hub_tqdm()
    assert bar_module.tqdm is original
    # The lock was released, so a second patch succeeds.
    assert hf_progress._patch_hub_tqdm(reporting) is True
    hf_progress._restore_hub_tqdm()


# --- Progress reporting rules ----------------------------------------------

def test_progress_is_monotonic_and_clamped():
    seen: list[int] = []
    progress = hf_progress._Progress(seen.append)
    progress.bytes(50, 100)
    progress.bytes(10, 100)   # a retry must not move the bar backwards
    progress.bytes(99, 100)
    progress.bytes(100, 100)  # reporting 100 is the caller's job
    assert seen == [50, 99]


def test_progress_survives_a_failing_callback():
    def boom(percent: int) -> None:
        raise RuntimeError('ui is gone')

    progress = hf_progress._Progress(boom)
    progress.bytes(10, 100)  # must not raise
