"""One-off: copy a project's exported parts into its default Videos folder."""
import json
import re
import shutil
import sqlite3
from pathlib import Path

DATA_ROOT = Path(r'C:\dub-vibe\data')
DB = DATA_ROOT / 'projects.sqlite3'


def safe(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', '_', (name or '').strip()).strip(' .')[:60]
    return cleaned or 'project'


with sqlite3.connect(DB) as conn:
    rows = list(conn.execute('SELECT id, data FROM projects'))

for pid, payload in rows:
    data = json.loads(payload)
    name = data.get('name') or 'project'
    if 'smoke' in name:
        conn.execute('DELETE FROM projects WHERE id=?', (pid,))
        shutil.rmtree(DATA_ROOT / pid, ignore_errors=True)
        print('deleted leaked test project:', name)
        continue
    parts_dir = data.get('parts_dir')
    if parts_dir and Path(parts_dir).is_dir():
        dest = Path.home() / 'Videos' / 'DubVibe' / safe(name)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for part in sorted(Path(parts_dir).glob('part_*.mp4')):
            shutil.copy2(part, dest / part.name)
            copied += 1
        print(f'copied {copied} parts to {dest}')
    else:
        print('no exported parts for:', name[:50])
