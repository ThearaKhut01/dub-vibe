"""Child-process entry point for one VoxCPM generation.

A generation runs in its own process so the OS reclaims the model's memory when the
job ends. The parent (voxcpm_workspace) starts this module with
`python -m backend.app.voxcpm_worker <request.json> <output.wav>`, reads one JSON
message per line from stdout, and terminates the process to cancel.

The child never re-imports the server's entry point, so it works under uvicorn, the
Tauri sidecar, and tests alike.

Messages (one JSON object per line):
  {"kind": "progress", "percent": int, "text": str}
  {"kind": "done", "output": str, "sample_rate": int, "duration": float}
  {"kind": "error", "code": str, "text": str}
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _emit(kind: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({'kind': kind, **fields}, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def run(request: dict[str, Any], output_path: str) -> None:
    import numpy as np
    import soundfile as sf

    from . import voxcpm_local
    from .voxcpm_workspace import _change_speed, compose_text

    def progress(percent: int, text: str) -> None:
        _emit('progress', percent=percent, text=text)

    try:
        if not voxcpm_local.is_installed():
            _emit('error', code='not_installed',
                  text='VoxCPM2 is not installed. Download it from AI Models first.')
            return

        progress(5, 'Checking hardware and loading the model...')
        report = voxcpm_local.hardware_report()
        if not report['fits']:
            _emit('error', code='insufficient_memory',
                  text=report['reason'] or 'Not enough memory to load VoxCPM2.')
            return

        model = voxcpm_local._load()

        progress(15, 'Generating speech...')
        kwargs: dict[str, Any] = {
            'text': compose_text(request),
            'cfg_value': request['cfg_value'],
            'inference_timesteps': request['inference_timesteps'],
        }
        if request.get('reference_path'):
            kwargs['reference_wav_path'] = request['reference_path']
        if request['mode'] == 'ultimate':
            kwargs['prompt_wav_path'] = request['reference_path']
            kwargs['prompt_text'] = request['prompt_text']

        sample_rate = int(getattr(model.tts_model, 'sample_rate', 48000))
        chunks = []
        seconds = 0.0
        for chunk in model.generate_streaming(**kwargs):
            array = np.asarray(chunk, dtype=np.float32).squeeze()
            if array.size:
                chunks.append(array)
                seconds += array.size / sample_rate
                progress(min(90, 15 + len(chunks) * 3), f'Generated {seconds:.1f} s of audio...')

        if not chunks:
            _emit('error', code='failed', text='VoxCPM2 produced no audio for this input.')
            return

        waveform = np.concatenate(chunks)
        if request['speed'] != 1.0:
            waveform = _change_speed(waveform, sample_rate, request['speed'])
        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak

        progress(95, 'Saving WAV...')
        sf.write(output_path, waveform, sample_rate, subtype='PCM_16')
        _emit('done', output=output_path, sample_rate=sample_rate,
              duration=round(waveform.size / sample_rate, 2))
    except MemoryError as error:
        _emit('error', code='insufficient_memory', text=str(error))
    except Exception as error:  # noqa: BLE001 - report the real reason to the UI
        _emit('error', code='failed', text=f'Generation failed: {error}')


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        _emit('error', code='failed', text='voxcpm_worker expects a request file and an output path.')
        return 2
    with open(argv[1], encoding='utf-8') as handle:
        request = json.load(handle)
    run(request, argv[2])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))