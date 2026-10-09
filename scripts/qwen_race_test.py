"""Reproduce the ggml-cpu segfault: concurrent create_chat_completion on one Llama.

Usage:
  python scripts/qwen_race_test.py          # 4 threads, no lock  -> expected to crash
  python scripts/qwen_race_test.py --lock   # 4 threads, serialized -> expected to pass
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import qwen_local

USE_LOCK = '--lock' in sys.argv
llm = qwen_local._get_llm()
infer_lock = threading.Lock()
errors: list[str] = []


def worker(n: int) -> None:
    try:
        for i in range(2):
            messages = [{'role': 'user', 'content': f'Reply with exactly: {n}{i}'}]
            if USE_LOCK:
                with infer_lock:
                    out = llm.create_chat_completion(messages=messages, max_tokens=8, temperature=0)
            else:
                out = llm.create_chat_completion(messages=messages, max_tokens=8, temperature=0)
            print(f'worker {n} iter {i} ok', flush=True)
    except Exception as error:  # noqa: BLE001
        errors.append(f'{n}: {error}')


threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
start = time.time()
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(f'DONE in {time.time() - start:.1f}s errors={errors}', flush=True)
