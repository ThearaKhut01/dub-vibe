import asyncio
import faulthandler
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow `python backend/run_server.py`: the script directory would otherwise be
# sys.path[0] and `backend.app.main` would not be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from backend.app.main import app

_crash_log = None


def _redirect_output_to_log() -> None:
    """Send stdout/stderr and native crash dumps to a log file.

    The Tauri sidecar has no console, so without this a segfault (e.g. inside
    ggml-cpu.dll) dies silently and only Windows Event Log knows about it.
    """
    global _crash_log
    try:
        from backend.app.providers import models_root

        log_dir = models_root().parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'backend.log'
        try:
            if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
                log_path.unlink()
        except OSError:
            pass
        _crash_log = open(log_path, 'a', buffering=1, encoding='utf-8', errors='replace')
        faulthandler.enable(file=_crash_log)
        sys.stdout = _crash_log
        sys.stderr = _crash_log
        print(f'=== backend start {datetime.now().isoformat()} pid={os.getpid()} ===', flush=True)
    except Exception:
        pass


if __name__ == '__main__':
    _redirect_output_to_log()
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        app,
        host='127.0.0.1',
        port=int(os.environ.get('DUBVIBE_PORT', '8001')),
        log_level='info',
    )
