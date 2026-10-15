from __future__ import annotations

import asyncio
import math
import re
import threading
import uuid
import json
import os
import sqlite3
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Keep the HuggingFace hub cache (Demucs separation weights) inside the shared
# model store instead of ~/.cache — must happen before huggingface_hub imports.
from backend.app.providers import models_root as _models_root

os.environ.setdefault('HF_HUB_CACHE', str(_models_root() / 'huggingface'))


def _register_nvidia_dll_dirs() -> None:
    """Make cuBLAS/cuDNN/cudart DLLs visible to CTranslate2 and llama.cpp.

    Without this, faster-whisper falls back to CPU even on machines with an
    NVIDIA GPU, and the CUDA llama.dll fails to load, because Windows cannot
    locate the CUDA runtime DLLs next to those packages.
    """
    bases = [Path(sys.prefix) / 'Lib' / 'site-packages']
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        bases.append(Path(meipass))

    dll_dirs: list[Path] = []
    for base in bases:
        for dll_dir in base.glob('nvidia/*/bin'):
            if dll_dir.is_dir():
                dll_dirs.append(dll_dir)
        torch_lib = base / 'torch' / 'lib'  # torch ships cudart/cublas for llama.dll
        if torch_lib.is_dir():
            dll_dirs.append(torch_lib)

    for dll_dir in dll_dirs:
        os.add_dll_directory(str(dll_dir))
        os.environ['PATH'] = str(dll_dir) + os.pathsep + os.environ.get('PATH', '')


_register_nvidia_dll_dirs()

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from urllib.parse import quote

from backend.app.chunking import SmartChunkEngine, format_timestamp, split_subtitle_cues
from backend.app import acting, api_keys, cuda_pack, model_manager, online_providers, speaker_diarization, tts_service, voxcpm_local
from backend.app import segment_tts
from backend.app.models import ProjectRequest, ProjectSummary, ProviderSettings, ProcessingRequest, SegmentSplit, SegmentUpdate, WorkflowProgress
from backend.app.providers import (
    FasterWhisperProvider,
    HybridTTSProvider,
    HybridTranslationProvider,
    SpeakerDetector,
    _resolve_lang_code,
    edge_gender_voice,
    get_app_root,
    get_provider_status,
    get_storage_dir,
)

app = FastAPI(title='DubVibe Studio API', version='0.1.0')

# The API is unauthenticated and bound to localhost, so only accept the origins
# the app itself runs from (Vite dev server and the Tauri webview).
ALLOWED_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://tauri.localhost',
    'https://tauri.localhost',
    'tauri://localhost',
]
extra_origins = os.environ.get('DUBVIBE_ALLOWED_ORIGINS')
if extra_origins:
    ALLOWED_ORIGINS.extend(origin.strip() for origin in extra_origins.split(',') if origin.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)

DATA_ROOT = get_storage_dir('data')
PROJECT_DB = DATA_ROOT / 'projects.sqlite3'
projects: dict[str, dict[str, Any]] = {}
active_connections: list[WebSocket] = []
_transcribe_cancelled: set[str] = set()
chunk_engine = SmartChunkEngine()


def persist_project(project_id: str) -> None:
    project = projects.get(project_id)
    if project is None:
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PROJECT_DB) as connection:
        connection.execute(
            'INSERT INTO projects (id, data) VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET data = excluded.data',
            (project_id, json.dumps(project, ensure_ascii=False)),
        )


def load_projects() -> None:
    if not PROJECT_DB.is_file():
        return
    with sqlite3.connect(PROJECT_DB) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        for project_id, payload in connection.execute('SELECT id, data FROM projects'):
            try:
                project = json.loads(payload)
                media_path = project.get('media_path')
                project['has_media'] = bool(media_path and Path(media_path).is_file())
                projects[project_id] = project
            except (TypeError, json.JSONDecodeError):
                continue


DATA_ROOT.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(PROJECT_DB) as connection:
    connection.execute('CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
load_projects()
# One-time: move models from the old per-install locations into the shared
# %LOCALAPPDATA%\DubVibe Studio\models store used by web + desktop.
model_manager.migrate_legacy_models()
# GPU Acceleration Pack (if downloaded): register its DLL folder so
# ctranslate2 can load cuBLAS/cuDNN when running Whisper on the GPU.
cuda_pack.activate_dll_search_path()


def find_media_tool(name: str) -> str | None:
    """Find media tool (ffmpeg, ffprobe) in app bundle, app directory, or system PATH."""
    executable = f'{name}.exe' if os.name == 'nt' else name

    # 1. PyInstaller onefile temporary extraction directory
    bundle_root = getattr(sys, '_MEIPASS', None)
    if bundle_root:
        bundled_path = Path(bundle_root) / executable
        if bundled_path.is_file():
            return str(bundled_path)

    # 2. Check next to running executable (sidecar binary directory)
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).resolve().parent
        candidate = exe_dir / executable
        if candidate.is_file():
            return str(candidate)

    # 3. Check within application installation directory structure
    app_root = get_app_root()
    search_dirs = [
        app_root / 'tools' / 'ffmpeg' / 'bin',
        app_root / 'tools' / 'ffmpeg',
        app_root / 'tools',
        app_root / 'resources' / 'ffmpeg' / 'bin',
        app_root / 'resources' / 'ffmpeg',
        app_root / 'resources' / 'binaries',
        app_root / 'resources',
        app_root / 'binaries',
        app_root,
    ]
    for directory in search_dirs:
        candidate = directory / executable
        if candidate.is_file():
            return str(candidate)

    # 4. Fallback to system PATH
    return shutil.which(name)


def build_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    cues = [cue for segment in segments for cue in split_subtitle_cues(segment)]
    for index, cue in enumerate(cues, start=1):
        start = format_timestamp(float(cue.get('start', 0))).replace('.', ',')
        end = format_timestamp(float(cue.get('end', 0))).replace('.', ',')
        text = str(cue.get('text', '')).replace('\r', ' ').strip()
        lines.extend((str(index), f'{start} --> {end}', text, ''))
    return '\n'.join(lines)


def escape_filter_path(path: Path) -> str:
    return str(path.resolve()).replace('\\', '/').replace(':', r'\:').replace("'", r"\'")


def update_project(project_id: str, **updates: Any) -> ProjectSummary | None:
    project = projects.get(project_id)
    if project is None:
        return None

    project.update(updates)
    persist_project(project_id)
    return ProjectSummary(**project)


async def emit_progress(project_id: str, stage: str, progress: int, message: str, engine: str | None = None) -> None:
    payload = WorkflowProgress(project_id=project_id, stage=stage, progress=progress, message=message, engine=engine).model_dump()
    for connection in list(active_connections):
        try:
            await connection.send_json(payload)
        except Exception:
            if connection in active_connections:
                active_connections.remove(connection)


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'dubvibe-api'}


@app.get('/api/providers')
async def list_providers() -> list[ProviderSettings]:
    # Provider probes (e.g. pyttsx3 COM init) can block; keep them off the event loop.
    status = await run_in_threadpool(get_provider_status)
    return [ProviderSettings(**provider) for provider in status]


@app.post('/api/projects/{project_id}/cut')
async def cut_project_video(project_id: str, payload: ProcessingRequest) -> dict[str, Any]:
    """Cut a time range out of the video, or keep only the range.

    mode='remove': delete [start, end] from the movie; everything after shifts
    back. mode='keep': trim the movie to just [start, end]. Transcript
    segments and dub audio follow: overlapping lines are removed, lines after
    the cut shift back. Chunks and separation are invalidated (re-chunk /
    re-separate after cutting). The dub mix must be regenerated.
    """
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    media_path = project.get('media_path')
    if not media_path or not Path(media_path).is_file():
        raise HTTPException(status_code=409, detail='Import a video before cutting')
    ffmpeg = find_media_tool('ffmpeg')
    ffprobe = find_media_tool('ffprobe')
    if not ffmpeg or not ffprobe:
        raise HTTPException(status_code=503, detail='FFmpeg and ffprobe must be installed and available on PATH')

    start = max(0.0, float(payload.start or 0))
    end = float(payload.end or 0)
    cut_mode = payload.cut_mode or 'remove'
    duration = _probe_duration(ffprobe, str(media_path))
    end = min(end, duration)
    if end - start < 0.2:
        raise HTTPException(status_code=400, detail='The selected range must be at least 0.2 seconds')
    if cut_mode == 'remove' and (end - start) >= duration - 0.2:
        raise HTTPException(status_code=400, detail='Cannot remove the entire video')

    cut_len = end - start
    cut_video = DATA_ROOT / project_id / 'cut-video.mp4'

    await emit_progress(project_id, 'Cutting video', 20, 'Cutting the selected range from the video...')
    if cut_mode == 'keep':
        command = [ffmpeg, '-y', '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(media_path)]
    else:
        command = [ffmpeg, '-y', '-i', str(media_path),
                   '-vf', f"select='not(between(t,{start},{end}))',setpts=N/FRAME_RATE/TB",
                   '-af', f"aselect='not(between(t,{start},{end}))',asetpts=N/SR/TB"]
    command.extend(['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-c:a', 'aac', str(cut_video)])
    try:
        await run_in_threadpool(subprocess.run, command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        cut_video.unlink(missing_ok=True)
        update_project(project_id, status='Ready')
        detail = error.stderr[-500:] if error.stderr else str(error)
        raise HTTPException(status_code=500, detail=f'Video cutting failed: {detail}') from error

    # Shift/trim transcript segments to match the new timeline.
    removed = 0
    shifted = 0
    kept_segments = []
    for segment in project.get('segments', []):
        s = float(segment.get('start') or 0)
        e = float(segment.get('end') or 0)
        if cut_mode == 'keep':
            if s >= start - 0.05 and e <= end + 0.05:
                segment['start'] = round(max(0.0, s - start), 3)
                segment['end'] = round(min(e, end) - start, 3)
                kept_segments.append(segment)
            else:
                removed += 1
        else:  # remove
            if e <= start + 0.05:  # fully before the cut: unchanged
                kept_segments.append(segment)
            elif s >= end - 0.05:  # fully after the cut: shift back
                segment['start'] = round(s - cut_len, 3)
                segment['end'] = round(e - cut_len, 3)
                shifted += 1
                kept_segments.append(segment)
            else:
                removed += 1  # overlaps the cut range: dropped
    project['segments'] = kept_segments

    # Dub audio follows the same rules.
    kept_audio = []
    for audio in project.get('audio_segments', []):
        s = float(audio.get('start') or 0)
        e = float(audio.get('end') or 0)
        if cut_mode == 'keep':
            if s >= start - 0.05 and e <= end + 0.05:
                audio['start'] = round(max(0.0, s - start), 3)
                audio['end'] = round(min(e, end) - start, 3)
                kept_audio.append(audio)
        else:
            if e <= start + 0.05:
                kept_audio.append(audio)
            elif s >= end - 0.05:
                audio['start'] = round(s - cut_len, 3)
                audio['end'] = round(e - cut_len, 3)
                kept_audio.append(audio)
    project['audio_segments'] = kept_audio

    # Chunks and separation reference the old timeline: invalidated.
    project['chunks'] = []
    project['chunk_seconds'] = None
    project['separated'] = False
    shutil.rmtree(DATA_ROOT / project_id / 'stems', ignore_errors=True)
    shutil.rmtree(DATA_ROOT / project_id / 'chunks', ignore_errors=True)

    # Swap in the cut video as the working media.
    old_media = Path(str(media_path))
    cut_video.replace(old_media)

    new_duration = _probe_duration(ffprobe, str(old_media))
    hours = int(new_duration // 3600)
    minutes = int(new_duration // 60 % 60)
    seconds = int(new_duration % 60)
    project['duration'] = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    project['video_path'] = None
    persist_project(project_id)

    final_message = (f'Cut {cut_len:.1f}s from the video · {removed} lines removed · {shifted} shifted. '
                     'Re-dub to regenerate audio.')
    await emit_progress(project_id, 'Cutting video', 100, final_message)
    update_project(project_id, status='Ready', progress=0)
    return {'project_id': project_id, 'status': 'ok', 'mode': cut_mode, 'duration': project['duration'], 'message': final_message}


@app.get('/api/system/info')
async def system_info() -> dict[str, Any]:
    app_root = get_app_root()
    models_dir = get_storage_dir('models')
    data_dir = get_storage_dir('data')
    ffmpeg_bin = find_media_tool('ffmpeg')
    ffprobe_bin = find_media_tool('ffprobe')
    return {
        'app_root': str(app_root),
        'models_dir': str(models_dir),
        'data_dir': str(data_dir),
        'ffmpeg_available': ffmpeg_bin is not None,
        'ffmpeg_path': ffmpeg_bin,
        'ffprobe_available': ffprobe_bin is not None,
        'is_bundled': getattr(sys, 'frozen', False),
    }


@app.get('/api/models')
async def list_all_models() -> dict[str, Any]:
    """Shared model manager state: one source of truth for web + desktop."""
    return {
        'engines': [
            {
                'id': 'faster-whisper',
                'name': 'Faster-Whisper (transcription)',
                'models': model_manager.whisper_models_status(),
            },
            {
                'id': 'qwen',
                'name': 'Qwen3 4B (local translation)',
                'models': [model_manager.qwen_status()],
            },
            {
                'id': 'demucs',
                'name': 'Demucs 6-stem (audio separation)',
                'models': [model_manager.demucs_status()],
            },
            {
                'id': 'nllb',
                'name': 'NLLB-200 (offline translation)',
                'models': [model_manager.nllb_status()],
            },
        ],
        'settings': model_manager.load_settings(),
    }


@app.get('/api/system/hardware')
async def system_hardware() -> dict[str, Any]:
    return await run_in_threadpool(model_manager.hardware_info)


@app.get('/api/models/storage')
async def models_storage() -> dict[str, Any]:
    return model_manager.storage_info()


@app.get('/api/models/faster-whisper')
async def list_whisper_models() -> list[dict[str, Any]]:
    return model_manager.whisper_models_status()


@app.get('/api/models/faster-whisper/{model_id}/status')
async def whisper_model_status(model_id: str) -> dict[str, Any]:
    status = model_manager.model_status(model_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f'Unknown model: {model_id}')
    return status


@app.post('/api/models/faster-whisper/{model_id}/download')
async def download_whisper_model(model_id: str) -> dict[str, Any]:
    try:
        job = model_manager.start_download(model_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'model': model_id, 'job': job}


@app.delete('/api/models/faster-whisper/{model_id}')
async def delete_whisper_model(model_id: str) -> dict[str, str]:
    try:
        model_manager.delete_model(model_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'deleted': model_id}


@app.post('/api/models/faster-whisper/{model_id}/load')
async def load_whisper_model(model_id: str) -> dict[str, Any]:
    try:
        result = await run_in_threadpool(model_manager.load_model, model_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {'status': 'ok', 'model': model_id, **result}


@app.post('/api/models/faster-whisper/{model_id}/unload')
async def unload_whisper_model(model_id: str) -> dict[str, str]:
    model_manager.unload()
    return {'status': 'ok', 'unloaded': model_id}


@app.post('/api/models/settings')
async def save_model_settings(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {key: payload[key] for key in ('selected_whisper_model', 'device', 'compute') if key in payload}
    settings = model_manager.save_settings(allowed)
    return {'status': 'ok', 'settings': settings}


@app.post('/api/models/demucs/download')
async def download_demucs_model() -> dict[str, Any]:
    try:
        job = model_manager.start_download_demucs()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'model': 'demucs-htdemucs-6s', 'job': job}


@app.delete('/api/models/demucs')
async def delete_demucs_model() -> dict[str, str]:
    try:
        model_manager.delete_demucs()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'deleted': 'demucs-htdemucs-6s'}


@app.get('/api/models/demucs')
async def demucs_model_status() -> dict[str, Any]:
    return model_manager.demucs_status()


@app.post('/api/models/qwen/download')
async def download_qwen_model() -> dict[str, Any]:
    try:
        job = model_manager.start_download_qwen()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'model': 'qwen3-4b', 'job': job}


@app.delete('/api/models/qwen')
async def delete_qwen_model() -> dict[str, str]:
    try:
        model_manager.delete_qwen()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'deleted': 'qwen3-4b'}


@app.get('/api/models/cuda-pack')
async def cuda_pack_status() -> dict[str, Any]:
    return await run_in_threadpool(cuda_pack.pack_status)


@app.post('/api/models/cuda-pack/download')
async def download_cuda_pack() -> dict[str, Any]:
    try:
        job = await run_in_threadpool(cuda_pack.start_download)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'model': 'cuda-gpu-pack', 'job': job}


@app.delete('/api/models/cuda-pack')
async def delete_cuda_pack() -> dict[str, str]:
    try:
        cuda_pack.delete_pack()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'deleted': 'cuda-gpu-pack'}


@app.get('/api/models/qwen')
async def qwen_model_status() -> dict[str, Any]:
    return model_manager.qwen_status()


@app.get('/api/models/nllb')
async def nllb_model_status() -> dict[str, Any]:
    return model_manager.nllb_status()


@app.post('/api/models/nllb/download')
async def download_nllb_model() -> dict[str, Any]:
    try:
        job = model_manager.start_download_nllb()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'model': model_manager.NLLB_JOB_ID, 'job': job}


@app.delete('/api/models/nllb')
async def delete_nllb_model() -> dict[str, str]:
    try:
        model_manager.delete_nllb()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'deleted': model_manager.NLLB_JOB_ID}


# --- Online provider keys (stored in Windows Credential Manager, never sent
# back to the browser) --------------------------------------------------------

@app.get('/api/providers/keys')
async def provider_keys_status() -> dict[str, Any]:
    return {
        provider: {'has_key': api_keys.has_key(provider)}
        for provider in api_keys.PROVIDERS
    }


@app.post('/api/providers/{provider}/key')
async def set_provider_key(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        api_keys.set_key(provider, str(payload.get('key') or ''))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'provider': provider, 'key_stored': True}


@app.delete('/api/providers/{provider}/key')
async def remove_provider_key(provider: str) -> dict[str, Any]:
    api_keys.delete_key(provider)
    return {'status': 'ok', 'provider': provider, 'key_stored': False}


@app.post('/api/providers/{provider}/test')
async def test_provider(provider: str) -> dict[str, Any]:
    """Live connection test: lists the provider's real TTS models."""
    if provider not in api_keys.PROVIDERS:
        raise HTTPException(status_code=404, detail=f'Unknown provider: {provider}')
    key = api_keys.get_key(provider)
    if not key:
        raise HTTPException(status_code=400, detail=f'No API key stored for {provider}. Save a key first.')
    try:
        if provider == 'gemini':
            from backend.app.online_providers import gemini_test

            result = await run_in_threadpool(gemini_test, key)
        else:
            from backend.app.online_providers import openai_test

            result = await run_in_threadpool(openai_test, key)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f'{provider} connection failed: {error}') from error
    return {'status': 'ok', 'connected': True, 'provider': provider, 'tts_models': result.get('models', [])}


@app.get('/api/tts/capabilities')
async def tts_capabilities(model: str | None = None) -> dict[str, Any]:
    """Honest per-provider capability tables — the UI renders exactly this."""
    gemini_model = model or 'gemini-2.5-flash-preview-tts'
    openai_model = model or 'gpt-4o-mini-tts'
    return {
        'gemini': {
            **online_providers.gemini_capabilities(gemini_model),
            'default_model': online_providers.GEMINI_DEFAULT_MODEL,
            'key_configured': api_keys.has_key('gemini'),
            'label': 'Google Gemini TTS',
        },
        'openai': {
            **online_providers.openai_capabilities(openai_model),
            'default_model': online_providers.OPENAI_DEFAULT_MODEL,
            'key_configured': api_keys.has_key('openai'),
            'label': 'OpenAI TTS',
        },
        'edge': {
            'voices': [],  # served by Edge TTS' own language voice table
            'emotions': [],  # no real acting control
            'style_control': False,
            'languages': ['en', 'km', 'fr', 'es', 'zh', 'ja', 'ko', 'th', 'vi', 'de', 'pt', 'ru', 'ar', 'hi', 'id'],
            'khmer_supported': True,
            'key_configured': True,
            'label': 'Edge TTS (free)',
        },
        'sapi5': {
            'voices': [],
            'emotions': [],
            'style_control': False,
            'languages': ['en', 'km'],
            'khmer_supported': True,
            'key_configured': True,
            'label': 'System TTS (offline)',
        },
        'voxcpm2': {
            'voices': [],
            'emotions': [],
            'style_control': False,
            'languages': sorted(voxcpm_local.VERIFIED_LANGUAGE_CODES),
            'khmer_supported': True,
            'key_configured': True,
            'installed': model_manager.voxcpm_status()['installed'],
            'label': voxcpm_local.NAME,
        },
        'voxcpm_clone': {
            'voices': [],
            'emotions': [],
            'style_control': False,
            'languages': sorted(voxcpm_local.VERIFIED_LANGUAGE_CODES),
            'khmer_supported': False,
            'key_configured': True,
            'installed': model_manager.voxcpm_status()['installed'],
            'label': 'VoxCPM2 Voice Clone (English, own voice)',
        },
        'emotions': online_providers.EMOTIONS,
    }


@app.get('/api/tts/gemini/models')
async def gemini_tts_models() -> dict[str, Any]:
    key = api_keys.get_key('gemini')
    if not key:
        raise HTTPException(status_code=400, detail='No Gemini API key stored. Save a key in API Settings first.')
    try:
        from backend.app.online_providers import gemini_list_models

        models = await run_in_threadpool(gemini_list_models, key)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f'Could not list Gemini models: {error}') from error
    return {'status': 'ok', 'models': models}


@app.post('/api/tts/preview')
async def tts_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate (and cache) one line so the user can hear it before a full dub."""
    text = str(payload.get('text') or '').strip()
    if not text:
        raise HTTPException(status_code=400, detail='Enter a line to preview')
    provider = str(payload.get('provider') or 'edge')
    language_name = str(payload.get('language') or 'English')
    lang_code = _resolve_lang_code(language_name)
    try:
        audio_url = await run_in_threadpool(
            tts_service.preview_line,
            text,
            provider=provider,
            language_name=language_name,
            lang_code=lang_code,
            voice=payload.get('voice'),
            model=payload.get('model'),
            emotion=str(payload.get('emotion') or 'Auto'),
            intensity=int(payload.get('intensity') or 70),
            character_profile=payload.get('character_profile'),
            use_openai_acting=bool(payload.get('openai_acting')) and api_keys.has_key('openai'),
            speed=float(payload.get('speed') or 1.0),
        )
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {'status': 'ok', 'audio': audio_url}


@app.get('/api/projects/{project_id}/characters')
async def get_characters(project_id: str) -> dict[str, Any]:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    speakers = sorted({str(segment.get('speaker') or 'Speaker 1') for segment in project.get('segments', [])})
    profiles = project.get('characters') or {}
    return {
        'speakers': speakers,
        'characters': {
            name: profile for name, profile in profiles.items()
        },
    }


@app.post('/api/projects/{project_id}/characters')
async def save_characters(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    characters = payload.get('characters')
    if not isinstance(characters, dict):
        raise HTTPException(status_code=400, detail='Body must contain a characters object')
    project['characters'] = characters
    persist_project(project_id)
    return {'status': 'ok', 'characters': characters}

def _segment_index(project: dict[str, Any], segment_id: str) -> int:
    for index, segment in enumerate(project.get('segments', [])):
        if str(segment.get('id')) == str(segment_id):
            return index
    return -1


def _drop_stale_dub_audio(project: dict[str, Any], segment_id: str | None = None) -> None:
    """Remove dub-audio entries for one segment (text/translation changed), or
    all of them after a structural edit (indexes shift -> old files unusable)."""
    if segment_id is None:
        project['audio_segments'] = []
        return
    project['audio_segments'] = [
        audio for audio in project.get('audio_segments', [])
        if str(audio.get('id')) != str(segment_id)
    ]


@app.patch('/api/projects/{project_id}/segments/{segment_id}')
async def update_segment(project_id: str, segment_id: str, payload: SegmentUpdate) -> dict[str, Any]:
    """Edit a segment's text, translation, or timing.

    Text/translation edits drop that line's dub audio so it is re-synthesized;
    timing edits keep the audio and move where it is placed in the mix.
    """
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    index = _segment_index(project, segment_id)
    if index < 0:
        raise HTTPException(status_code=404, detail='Segment not found')
    segment = project['segments'][index]

    audio_stale = False
    if payload.text is not None:
        segment['text'] = payload.text.strip()
        audio_stale = True
    if payload.translation is not None:
        segment['translation'] = payload.translation.strip()
        audio_stale = True
    new_start = float(payload.start) if payload.start is not None else None
    new_end = float(payload.end) if payload.end is not None else None
    if new_start is not None:
        segment['start'] = round(max(0.0, new_start), 3)
    if new_end is not None:
        segment['end'] = round(max(segment['start'] + 0.1, new_end), 3)

    if audio_stale:
        _drop_stale_dub_audio(project, segment_id)
    else:
        # Timing-only edit: keep the audio, move where it is placed.
        for audio in project.get('audio_segments', []):
            if str(audio.get('id')) == str(segment_id):
                audio['start'] = segment['start']
                audio['end'] = segment['end']
    persist_project(project_id)
    return {'status': 'ok', 'segment': segment}


@app.post('/api/projects/{project_id}/segments/{segment_id}/split')
async def split_segment(project_id: str, segment_id: str, payload: SegmentSplit) -> dict[str, Any]:
    """Split one segment into two at time `at`.

    The caller supplies the texts for both halves (split at a word boundary in
    the UI). Structural edit: the dub mix is rebuilt on the next Dub run.
    """
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    index = _segment_index(project, segment_id)
    if index < 0:
        raise HTTPException(status_code=404, detail='Segment not found')
    segment = project['segments'][index]
    start = float(segment.get('start') or 0)
    end = float(segment.get('end') or 0)
    at = float(payload.at)
    if not (start + 0.2 <= at <= end - 0.2):
        raise HTTPException(status_code=400, detail=f'Split point must be between {start:.2f}s and {end:.2f}s')

    text1 = payload.text1.strip()
    text2 = payload.text2.strip()
    if not (text1 or text2):
        raise HTTPException(status_code=400, detail='Split needs text on at least one side')
    translation1 = payload.translation1.strip()
    translation2 = payload.translation2.strip()

    segment_a = {**segment, 'id': f'{segment_id}a', 'end': round(at, 3),
                 'text': text1, 'translation': translation1}
    segment_b = {**segment, 'id': f'{segment_id}b', 'start': round(at, 3),
                 'text': text2, 'translation': translation2}
    project['segments'][index:index + 1] = [segment_a, segment_b]
    project['audio_segments'] = segment_tts.keep_audio_by_id(project.get('audio_segments', []), project['segments'])  # structural edit: keep unchanged clips
    persist_project(project_id)
    return {'status': 'ok', 'segments': [segment_a, segment_b]}


@app.post('/api/projects/{project_id}/segments/{segment_id}/merge-next')
async def merge_segment_next(project_id: str, segment_id: str) -> dict[str, Any]:
    """Merge a segment with the one that follows it."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    index = _segment_index(project, segment_id)
    if index < 0:
        raise HTTPException(status_code=404, detail='Segment not found')
    segments = project['segments']
    if index + 1 >= len(segments):
        raise HTTPException(status_code=400, detail='No next segment to merge with')
    first, second = segments[index], segments[index + 1]
    merged = {
        **first,
        'end': second.get('end', first.get('end')),
        'text': (str(first.get('text') or '') + ' ' + str(second.get('text') or '')).strip(),
        'translation': (str(first.get('translation') or '') + ' ' + str(second.get('translation') or '')).strip(),
    }
    segments[index] = merged
    del segments[index + 1]
    project['audio_segments'] = segment_tts.keep_audio_by_id(project.get('audio_segments', []), project['segments'])  # structural edit: keep unchanged clips
    persist_project(project_id)
    return {'status': 'ok', 'segment': merged}


@app.delete('/api/projects/{project_id}/segments/{segment_id}')
async def delete_segment(project_id: str, segment_id: str) -> dict[str, Any]:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    index = _segment_index(project, segment_id)
    if index < 0:
        raise HTTPException(status_code=404, detail='Segment not found')
    project['segments'].pop(index)
    project['audio_segments'] = segment_tts.keep_audio_by_id(project.get('audio_segments', []), project['segments'])  # structural edit: keep unchanged clips
    persist_project(project_id)
    return {'status': 'ok', 'deleted': segment_id}

@app.post('/api/projects/{project_id}/segments/{segment_id}/tts')
async def generate_segment_audio(project_id: str, segment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Generate audio for one subtitle segment from its current dubbed text.

    Uses the same synthesis path as the full dub. The segment's previous audio is kept
    until the new clip is written successfully.
    """
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    index = _segment_index(project, segment_id)
    if index < 0:
        raise HTTPException(status_code=404, detail='Segment not found')
    segment = project['segments'][index]
    gender = 'Female' if str(payload.get('gender') or '').strip().lower() == 'female' else 'Male'
    voice = segment_tts.voice_for_gender(gender)
    # Remember the speaker's voice on the line itself, so it survives a reload before audio exists.
    segment['gender'] = gender
    target_language = str(payload.get('target_language') or project.get('target_language') or 'Khmer')
    provider = str(payload.get('provider') or 'edge')
    speed = float(payload.get('speed') or 1.0)
    previous = next((item for item in project.get('audio_segments', []) if str(item.get('id')) == str(segment_id)), None)
    try:
        entry = await run_in_threadpool(
            segment_tts.synthesize_segment,
            project_id=project_id,
            data_root=DATA_ROOT,
            segment=segment,
            previous=previous,
            voice=voice,
            provider=provider,
            language_name=target_language,
            lang_code=_resolve_lang_code(target_language),
            speed=speed,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=f'Audio not replaced: {error}') from error

    clip_seconds = None
    ffprobe = find_media_tool('ffprobe')
    if ffprobe:
        try:
            clip_seconds = await run_in_threadpool(_probe_duration, ffprobe, entry['path'])
        except (subprocess.CalledProcessError, ValueError, KeyError, OSError):
            clip_seconds = None
    slot_seconds = float(segment['end']) - float(segment['start'])
    entry['fit'] = segment_tts.fit_report(clip_seconds, slot_seconds)
    audio = [item for item in project.get('audio_segments', []) if str(item.get('id')) != str(segment_id)]
    audio.append({key: value for key, value in entry.items() if key != 'previous_path'})
    audio.sort(key=lambda item: float(item.get('start') or 0))
    project['audio_segments'] = audio
    persist_project(project_id)
    previous_path = entry.get('previous_path')
    if previous_path and previous_path != entry['path'] and Path(str(previous_path)).is_file():
        Path(str(previous_path)).unlink(missing_ok=True)
    return {'status': 'ok', 'audio': entry}


@app.post('/api/projects/{project_id}/segments')
async def add_segment(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Insert a new empty subtitle segment between the given times."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    start = round(max(0.0, float(payload.get('start') or 0)), 3)
    end = round(float(payload.get('end') or start + 1.0), 3)
    if end - start < 0.2:
        raise HTTPException(status_code=400, detail='A new segment must be at least 0.2s long')
    segment = {
        'id': f'seg-{uuid.uuid4().hex[:10]}',
        'start': start,
        'end': end,
        'speaker': str(payload.get('speaker') or 'Speaker 1'),
        'text': '',
        'translation': '',
    }
    segments = project.setdefault('segments', [])
    segments.append(segment)
    segments.sort(key=lambda item: float(item.get('start') or 0))
    persist_project(project_id)
    return {'status': 'ok', 'segment': segment}


@app.get('/api/projects/{project_id}/segments/{segment_id}/audio')
async def get_segment_audio(project_id: str, segment_id: str, request: Request):
    """Serve one segment's dubbed clip by id (stable across split, merge, and reorder)."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    clip = next((item for item in project.get('audio_segments', []) if str(item.get('id')) == str(segment_id)), None)
    if clip is None or not clip.get('path') or not Path(str(clip['path'])).is_file():
        raise HTTPException(status_code=404, detail='This line has no generated audio yet')
    media_type = 'audio/wav' if str(clip['path']).lower().endswith('.wav') else 'audio/mpeg'
    return _file_response_with_range(request, Path(str(clip['path'])), media_type)

@app.get('/api/projects/{project_id}/segment-timing')
async def get_segment_timing(project_id: str) -> dict[str, Any]:
    """Overlaps, invalid timecodes, and segments that run past the video."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    problems = segment_tts.timing_problems(project.get('segments', []), _media_duration_seconds(project))
    return {'status': 'ok', 'problems': problems}


@app.get('/api/projects')
async def list_projects() -> list[ProjectSummary]:
    return [ProjectSummary(**project) for project in projects.values()]


@app.get('/api/projects/{project_id}')
async def get_project(project_id: str) -> ProjectSummary:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f'Project {project_id} not found')
    return ProjectSummary(**project)


@app.delete('/api/projects/{project_id}')
async def delete_project(project_id: str) -> dict[str, str]:
    if project_id not in projects:
        raise HTTPException(status_code=404, detail=f'Project {project_id} not found')
    projects.pop(project_id)
    with sqlite3.connect(PROJECT_DB) as connection:
        connection.execute('DELETE FROM projects WHERE id = ?', (project_id,))
    shutil.rmtree(DATA_ROOT / project_id, ignore_errors=True)
    return {'status': 'ok', 'deleted': project_id}


@app.delete('/api/projects')
async def delete_all_projects() -> dict[str, Any]:
    """Clear the whole project history: rows, in-memory state, and data folders."""
    count = len(projects)
    for project_id in list(projects.keys()):
        projects.pop(project_id)
        shutil.rmtree(DATA_ROOT / project_id, ignore_errors=True)
    with sqlite3.connect(PROJECT_DB) as connection:
        connection.execute('DELETE FROM projects')
    return {'status': 'ok', 'deleted': count}


@app.get('/api/projects/{project_id}/workspace')
async def get_project_workspace(project_id: str) -> dict[str, Any]:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f'Project {project_id} not found')
    segments = project.get('segments', [])
    return {
        'project': ProjectSummary(**project),
        'segments': segments,
        'audio_segments': project.get('audio_segments', []),
        'chunks': project.get('chunks', []),
        'chunk_seconds': project.get('chunk_seconds'),
        'parts_count': (
            len(list(Path(str(project['parts_dir'])).glob('part_*.mp4')))
            if project.get('parts_dir') and Path(str(project['parts_dir'])).is_dir() else 0
        ),
        'export_folder': project.get('export_folder'),
        'separated': bool(
            (project.get('stems') or {}).get('sfx')
            and Path(str((project.get('stems') or {})['sfx'])).is_file()
        ),
        'workflow': {
            'imported': bool(project.get('media_path') and Path(project['media_path']).is_file()),
            'chunked': bool(project.get('chunks')),
            'transcribed': bool(segments),
            'translated': bool(segments) and all(segment.get('translation') for segment in segments),
            'dubbed': bool(project.get('audio_segments')),
            'subtitles_exported': bool(project.get('subtitle_path') and Path(project['subtitle_path']).is_file()),
            'video_ready': bool(project.get('video_path') and Path(project['video_path']).is_file()),
        },
    }


def _file_response_with_range(request: Request, path: Path, media_type: str, filename: str | None = None):
    """Serve a file with HTTP Range support.

    Starlette's FileResponse ignores Range headers, and Chromium then reports
    `seekable = [0, 0]` for <video>/<audio> backed by such a response — every
    skip/seek clamps back to zero. Handle byte ranges here instead.
    """
    file_size = path.stat().st_size
    headers: dict[str, str] = {'accept-ranges': 'bytes'}
    if filename:
        headers['content-disposition'] = f"attachment; filename*=utf-8''{quote(filename)}"
    start, end, status = 0, file_size - 1, 200
    range_header = request.headers.get('range')
    if range_header:
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header.strip())
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
            else:  # suffix form: last N bytes
                start = max(0, file_size - int(match.group(2)))
                end = file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                return StreamingResponse(
                    iter(()),
                    status_code=416,
                    headers={'content-range': f'bytes */{file_size}'},
                )
            status = 206
            headers['content-range'] = f'bytes {start}-{end}/{file_size}'

    def byte_stream():
        with path.open('rb') as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1024 * 256, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(byte_stream(), status_code=status, media_type=media_type, headers=headers)


@app.get('/api/projects/{project_id}/media')
async def get_project_media(project_id: str, request: Request):
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    media_path = project.get('media_path')
    if not media_path or not Path(media_path).is_file():
        raise HTTPException(status_code=404, detail='Imported video is not available')
    media_types = {
        '.mp4': 'video/mp4', '.m4v': 'video/mp4', '.webm': 'video/webm',
        '.mkv': 'video/x-matroska', '.mov': 'video/quicktime', '.avi': 'video/x-msvideo',
    }
    media_type = media_types.get(Path(media_path).suffix.lower(), 'video/mp4')
    return _file_response_with_range(request, Path(media_path), media_type, project.get('media_name', 'source-video'))


@app.post('/api/projects')
async def create_project(payload: ProjectRequest) -> ProjectSummary:
    project_id = str(uuid.uuid4())
    project = ProjectSummary(
        id=project_id,
        name=payload.name,
        mode=payload.mode,
        status='Queued',
        progress=5,
        duration='00:00:00',
    )
    projects[project_id] = {**project.model_dump(), 'media_path': None, 'segments': [], 'audio_segments': []}
    persist_project(project_id)
    return project


@app.post('/api/projects/{project_id}/media')
async def upload_project_media(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')

    original_name = Path(file.filename or 'video').name
    extension = Path(original_name).suffix.lower()
    if extension not in {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}:
        raise HTTPException(status_code=415, detail='Choose an MP4, MOV, MKV, WebM, AVI, or M4V video')

    media_dir = DATA_ROOT / project_id
    media_dir.mkdir(parents=True, exist_ok=True)
    destination = media_dir / f'original{extension}'
    try:
        with destination.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    project['media_path'] = str(destination)
    project['media_name'] = original_name
    project['has_media'] = True
    project['segments'] = []
    project['audio_segments'] = []
    shutil.rmtree(DATA_ROOT / project_id / 'chunks', ignore_errors=True)
    project['chunks'] = []
    project['chunk_seconds'] = None
    shutil.rmtree(DATA_ROOT / project_id / 'stems', ignore_errors=True)
    project['stems'] = None
    project['status'] = 'Ready'
    project['progress'] = 5
    metadata: dict[str, Any] = {'duration': '00:00:00', 'resolution': 'Unknown'}
    ffprobe = find_media_tool('ffprobe')
    if ffprobe:
        try:
            probe = subprocess.run(
                [ffprobe, '-v', 'error', '-show_entries', 'format=duration:stream=width,height', '-of', 'json', str(destination)],
                check=True,
                capture_output=True,
                text=True,
            )
            probe_data = json.loads(probe.stdout)
            duration = int(float(probe_data.get('format', {}).get('duration', 0)))
            metadata['duration'] = f'{duration // 3600:02}:{duration // 60 % 60:02}:{duration % 60:02}'
            video_stream = next((stream for stream in probe_data.get('streams', []) if stream.get('width') and stream.get('height')), None)
            if video_stream:
                metadata['resolution'] = f"{video_stream['width']}x{video_stream['height']}"
            project['duration'] = metadata['duration']
            project['media_resolution'] = metadata['resolution']
            project['media_size'] = destination.stat().st_size
        except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
            destination.unlink(missing_ok=True)
            project['media_path'] = None
            project['has_media'] = False
            persist_project(project_id)
            raise HTTPException(status_code=415, detail=f'Could not read the uploaded video: {error}') from error
    persist_project(project_id)
    return {
        'project_id': project_id,
        'filename': original_name,
        'size': destination.stat().st_size,
        **metadata,
    }


def _duck_ratio(level: float) -> float:
    """Sidechain compression ratio that dips full-scale audio to ~level while
    dubbed speech plays (detected at ~0 dBFS against a -30.5 dB threshold).
    The achievable dip is capped near 30 dB, so deeper requests clamp there."""
    target_db = min(-20.0 * math.log10(max(level, 0.003)), 28.0)
    return min(50.0, max(1.5, 30.5 / max(30.5 - target_db, 0.5)))


def _probe_duration(ffprobe: str, media_path: str) -> float:
    probe = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(media_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(probe.stdout)['format']['duration'])


@app.post('/api/process/chunk')
async def process_chunk(payload: ProcessingRequest) -> dict[str, Any]:
    """Split the imported video into fixed-length parts (stream copy, fast).

    Chunks are a processing aid: transcription offsets each part's timestamps
    back onto the global timeline, so translation, dubbing, and export are
    unaffected. Sending chunk_seconds=None removes the split.
    """
    project = projects.get(payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    media_path = project.get('media_path')
    if not media_path or not Path(media_path).is_file():
        raise HTTPException(status_code=409, detail='Import a video before splitting it')
    ffmpeg = find_media_tool('ffmpeg')
    ffprobe = find_media_tool('ffprobe')
    if not ffmpeg or not ffprobe:
        raise HTTPException(status_code=503, detail='FFmpeg and ffprobe must be installed and available on PATH')

    chunk_dir = DATA_ROOT / payload.project_id / 'chunks'

    def clear_chunks(message: str) -> dict[str, Any]:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        project['chunks'] = []
        project['chunk_seconds'] = None
        persist_project(payload.project_id)
        update_project(payload.project_id, status='Ready')
        return {
            'project_id': payload.project_id,
            'status': 'ok',
            'chunks': [],
            'chunk_seconds': None,
            'count': 0,
            'message': message,
        }

    if not payload.chunk_seconds:
        return clear_chunks('Video chunking removed. Dubbing will use the whole video.')

    chunk_seconds = float(payload.chunk_seconds)
    try:
        duration = _probe_duration(ffprobe, str(media_path))
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f'Could not read the video duration: {error}') from error

    expected = math.ceil(duration / chunk_seconds - 0.001)
    if expected <= 1:
        return clear_chunks('The video is shorter than the chosen part length. No split needed.')

    await emit_progress(payload.project_id, 'Chunking video', 20, f'Splitting the video into {expected} parts...')
    update_project(payload.project_id, status='Processing')

    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob('chunk_*.mkv'):
        old.unlink()

    # Matroska + stream copy: near-instant and works with any source codec.
    command = [
        ffmpeg, '-y', '-i', str(media_path), '-map', '0', '-c', 'copy',
        '-f', 'segment', '-segment_time', f'{chunk_seconds:.3f}', '-reset_timestamps', '1',
        str(chunk_dir / 'chunk_%04d.mkv'),
    ]
    try:
        await run_in_threadpool(subprocess.run, command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        update_project(payload.project_id, status='Ready')
        detail = error.stderr[-1500:] if error.stderr else str(error)
        raise HTTPException(status_code=500, detail=f'Video splitting failed: {detail}') from error

    files = sorted(chunk_dir.glob('chunk_*.mkv'))
    if not files:
        update_project(payload.project_id, status='Ready')
        raise HTTPException(status_code=500, detail='Video splitting produced no output files')

    chunks: list[dict[str, Any]] = []
    cursor = 0.0
    try:
        for index, path in enumerate(files):
            chunk_duration = _probe_duration(ffprobe, str(path))
            chunks.append({
                'index': index + 1,
                'path': str(path),
                'start': round(cursor, 3),
                'duration': round(chunk_duration, 3),
            })
            cursor += chunk_duration
            await emit_progress(
                payload.project_id, 'Chunking video',
                40 + int(55 * (index + 1) / len(files)),
                f'Indexed part {index + 1} of {len(files)}...',
            )
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        project['chunks'] = []
        project['chunk_seconds'] = None
        persist_project(payload.project_id)
        update_project(payload.project_id, status='Ready')
        raise HTTPException(status_code=500, detail=f'Could not index the split parts: {error}') from error

    project['chunks'] = chunks
    project['chunk_seconds'] = chunk_seconds
    persist_project(payload.project_id)
    await emit_progress(payload.project_id, 'Chunking video', 100, f'Video split into {len(chunks)} parts.')
    update_project(payload.project_id, status='Ready')
    return {
        'project_id': payload.project_id,
        'status': 'ok',
        'chunk_seconds': chunk_seconds,
        'count': len(chunks),
        'chunks': chunks,
        'message': f'Video split into {len(chunks)} parts of about {chunk_seconds / 60:g} minutes.',
    }


@app.post('/api/process/separate')
async def process_separate(payload: ProcessingRequest) -> dict[str, Any]:
    """Split the original audio into a voice stem and a sound-effects/music bed.

    After separation, exports use the bed as the original-audio source: the dub
    replaces the original voices while effects and music stay intact (and the
    auto-duck dips the bed under dub lines instead of the full soundtrack).
    """
    project = projects.get(payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    media_path = project.get('media_path')
    if not media_path or not Path(media_path).is_file():
        raise HTTPException(status_code=409, detail='Import a video before separating its audio')

    import importlib.util
    if importlib.util.find_spec('demucs') is None:
        raise HTTPException(
            status_code=503,
            detail='Audio separation needs the Demucs package (pip install demucs), which is not installed in this environment.',
        )
    if not model_manager.demucs_installed():
        raise HTTPException(
            status_code=503,
            detail='The Demucs separation model is not downloaded yet. Open the AI Model Manager and download "Demucs 6-stem (audio separation)" first.',
        )
    ffmpeg = find_media_tool('ffmpeg')
    if not ffmpeg:
        raise HTTPException(status_code=503, detail='FFmpeg must be installed and available on PATH to separate audio')

    await emit_progress(payload.project_id, 'Separating audio', 5,
                        'Extracting voice and sound effects from the original audio...',
                        engine='demucs')
    update_project(payload.project_id, status='Processing')

    loop = asyncio.get_running_loop()
    stems_dir = DATA_ROOT / payload.project_id / 'stems'
    stems_dir.mkdir(parents=True, exist_ok=True)
    work_dir = stems_dir / 'separate'
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Keep the Demucs model cache inside the app's models folder, and make our
    # bundled ffmpeg visible to Demucs (it shells out to ffmpeg for decoding).
    os.environ['TORCH_HOME'] = str(get_storage_dir('models') / 'torch')
    previous_path = os.environ.get('PATH')
    if ffmpeg:
        os.environ['PATH'] = str(Path(ffmpeg).parent) + os.pathsep + (previous_path or '')

    class _ProgressTap:
        """stderr stand-in that scans Demucs' tqdm output for percentages."""

        def __init__(self, fallback: object) -> None:
            self._fallback = fallback
            self._last = 0
            self._buffer = ''

        def write(self, text: str) -> int:
            try:
                self._buffer = (self._buffer + text)[-2000:]
                for match in re.finditer(r'(\d+)%', self._buffer):
                    percent = min(100, int(match.group(1)))
                    if percent >= self._last + 5:
                        self._last = percent
                        asyncio.run_coroutine_threadsafe(
                            emit_progress(
                                payload.project_id, 'Separating audio',
                                5 + int(90 * percent / 100),
                                f'Separating voice from sound effects: {percent}%...',
                                engine='demucs',
                            ),
                            loop,
                        )
            except Exception:
                pass
            try:
                self._fallback.write(text)  # type: ignore[attr-defined]
            except Exception:
                pass
            return len(text) if isinstance(text, str) else 0

        def flush(self) -> None:
            try:
                self._fallback.flush()  # type: ignore[attr-defined]
            except Exception:
                pass

        @property
        def encoding(self) -> str:
            return str(getattr(self._fallback, 'encoding', 'utf-8'))

    def run_separation() -> None:
        # Run Demucs in-process: this works identically in development and in
        # the packaged desktop app (a frozen exe cannot run "python -m demucs").
        from demucs.separate import main as demucs_main

        device, _ = model_manager.resolve_device_compute()
        tap = _ProgressTap(sys.stderr)
        old_stderr = sys.stderr
        sys.stderr = tap  # type: ignore[assignment]
        try:
            demucs_main(['-n', 'htdemucs_6s', '--device', device, '--out', str(work_dir), str(media_path)])
        except SystemExit as error:  # argparse / internal exits must surface as errors
            raise RuntimeError(f'Demucs separation exited with code {error.code}') from None
        finally:
            sys.stderr = old_stderr  # type: ignore[assignment]
            try:
                tap.flush()
            except Exception:
                pass
            if previous_path is not None:
                os.environ['PATH'] = previous_path

    try:
        await run_in_threadpool(run_separation)
    except Exception as error:
        update_project(payload.project_id, status='Ready')
        raise HTTPException(status_code=500, detail=f'Audio separation failed: {error}') from error

    produced = list(work_dir.glob('**/vocals.wav'))
    if not produced:
        update_project(payload.project_id, status='Ready')
        raise HTTPException(status_code=500, detail='Audio separation produced no stems')
    stems_out = produced[0].parent
    voices_path = stems_dir / 'original-voices.wav'
    music_path = stems_dir / 'original-music.wav'
    sfx_path = stems_dir / 'original-sfx.wav'
    shutil.copy2(stems_out / 'vocals.wav', voices_path)
    # htdemucs_6s stems: vocals + drums/bass/piano/guitar (music) + other (SFX/ambience).
    music_sources = [stems_out / name for name in ('drums.wav', 'bass.wav', 'piano.wav', 'guitar.wav')]
    music_sources = [path for path in music_sources if path.is_file()]
    if len(music_sources) >= 2:
        mix_command = [ffmpeg, '-y']
        for source in music_sources:
            mix_command.extend(['-i', str(source)])
        mix_command.extend([
            '-filter_complex',
            f"{''.join(f'[{index}:a]' for index in range(len(music_sources)))}"
            'amix=inputs=' + str(len(music_sources)) + ':duration=longest:normalize=0[m]',
            '-map', '[m]', '-c:a', 'pcm_s16le', str(music_path),
        ])
        subprocess.run(mix_command, check=True, capture_output=True, text=True)
    elif music_sources:
        shutil.copy2(music_sources[0], music_path)
    else:
        raise HTTPException(status_code=500, detail='Audio separation produced no music stems')
    shutil.copy2(stems_out / 'other.wav', sfx_path)
    shutil.rmtree(work_dir, ignore_errors=True)

    project['stems'] = {'voices': str(voices_path), 'music': str(music_path), 'sfx': str(sfx_path)}
    persist_project(payload.project_id)
    final_message = ('Voice, music, and sound effects separated. Exports now replace the original '
                     'dialogue with the dub while SFX/music stay on and duck under it.')
    await emit_progress(payload.project_id, 'Separating audio', 100, final_message, engine='demucs')
    update_project(payload.project_id, status='Ready')
    return {'project_id': payload.project_id, 'status': 'ok', 'stems': project['stems'], 'message': final_message}


@app.post('/api/process/transcribe')
async def process_transcribe(payload: ProcessingRequest) -> dict[str, Any]:
    project = projects.get(payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    media_path = project.get('media_path')
    if not media_path or not Path(media_path).is_file():
        raise HTTPException(status_code=409, detail='Import a video before transcribing')

    _transcribe_cancelled.discard(payload.project_id)
    should_cancel = lambda: payload.project_id in _transcribe_cancelled  # noqa: E731

    model_id = model_manager.selected_model_id()
    await emit_progress(payload.project_id, 'Transcribing', 8,
                        f'Preparing transcription model: {model_id} (downloaded on first use)...')
    update_project(payload.project_id, status='Processing', progress=8)

    stt = FasterWhisperProvider()
    loop = asyncio.get_running_loop()

    try:
        model_id, device_label = await run_in_threadpool(
            model_manager.ensure_selected_ready,
            lambda percent: asyncio.run_coroutine_threadsafe(
                emit_progress(
                    payload.project_id,
                    'Transcribing',
                    8 + int(8 * percent / 100),
                    f'Downloading transcription model {model_id}: {percent}%...',
                ),
                loop,
            ),
        )
    except Exception as error:
        update_project(payload.project_id, status='Ready', progress=5)
        raise HTTPException(
            status_code=500,
            detail=f'Transcription model "{model_id}" failed to prepare: {error}',
        ) from error

    await emit_progress(payload.project_id, 'Transcribing', 18,
                        f'Extracting audio and detecting speech · model {model_id} · {device_label}...')
    update_project(payload.project_id, status='Processing', progress=18)

    language = _resolve_lang_code(payload.language) if payload.language else None

    def make_reporter(chunk_index: int, total: int, chunk_duration: float):
        def report(count: int, audio_end: float) -> None:
            # Runs on the Whisper worker thread; hop back to the event loop to emit.
            intra = min(1.0, audio_end / chunk_duration) if chunk_duration > 0 else 1.0
            percent = 18 + int(38 * ((chunk_index + intra) / total))
            asyncio.run_coroutine_threadsafe(
                emit_progress(
                    payload.project_id,
                    'Transcribing',
                    percent,
                    f'Part {chunk_index + 1} of {total}: detected {count} speech segments...',
                ),
                loop,
            )
        return report

    chunks = [chunk for chunk in project.get('chunks', []) if Path(str(chunk.get('path') or '')).is_file()]
    if len(chunks) != len(project.get('chunks', [])):
        project['chunks'] = chunks
        persist_project(payload.project_id)

    try:
        if chunks:
            merged_segments: list[dict[str, Any]] = []
            detected_language: str | None = None
            total = len(chunks)
            cancelled = False
            for index, chunk in enumerate(chunks):
                result = await run_in_threadpool(
                    stt.transcribe,
                    str(chunk['path']),
                    make_reporter(index, total, float(chunk.get('duration') or 0)),
                    should_cancel,
                    language,
                    payload.beam_size,
                    payload.word_timestamps,
                )
                if detected_language is None:
                    detected_language = result['language']
                offset = float(chunk.get('start') or 0)
                for segment in result['segments']:
                    merged_segments.append({
                        'id': f'seg_{len(merged_segments) + 1}',
                        'start': round(float(segment['start']) + offset, 3),
                        'end': round(float(segment['end']) + offset, 3),
                        'speaker': segment.get('speaker', 'Speaker 1'),
                        'text': segment['text'],
                    })
                if result.get('cancelled'):
                    cancelled = True
                    break
            result = {
                'provider': 'Faster-Whisper',
                'model': model_id,
                'language': detected_language or 'en',
                'segments': merged_segments,
                'source_file': media_path,
                'chunks': total,
                'cancelled': cancelled,
                'status': 'ok',
            }
        else:
            duration_seconds = 0.0
            try:
                hours, minutes, seconds = (int(part) for part in str(project.get('duration') or '').split(':'))
                duration_seconds = hours * 3600 + minutes * 60 + seconds
            except ValueError:
                duration_seconds = 0.0

            def report_transcribe_progress(count: int, audio_end: float) -> None:
                if duration_seconds > 0:
                    percent = 18 + int(38 * min(1.0, audio_end / duration_seconds))
                else:
                    percent = min(55, 18 + count)
                asyncio.run_coroutine_threadsafe(
                    emit_progress(
                        payload.project_id,
                        'Transcribing',
                        percent,
                        f'Detected {count} speech segments...',
                    ),
                    loop,
                )

            result = await run_in_threadpool(
                stt.transcribe,
                media_path,
                report_transcribe_progress,
                should_cancel,
                language,
                payload.beam_size,
                payload.word_timestamps,
            )
    except Exception as error:
        update_project(payload.project_id, status='Ready', progress=5)
        raise HTTPException(status_code=500, detail=f'Transcription failed: {error}') from error
    finally:
        _transcribe_cancelled.discard(payload.project_id)

    # A cancel that lands before the first segment yields returns an empty
    # list — keep an existing transcript instead of wiping it.
    cancelled = bool(result.get('cancelled'))
    if not (cancelled and not result['segments']):
        project['segments'] = result['segments']
        project['source_language'] = result['language']
        project['audio_segments'] = []
    if cancelled:
        final_message = f'Transcription cancelled — {len(result["segments"])} segments kept.'
    else:
        final_message = f"Found {len(result['segments'])} speech segments."
    await emit_progress(payload.project_id, 'Transcribing', 58, final_message)
    update_project(payload.project_id, status='Ready', progress=58)
    return {'project_id': payload.project_id, 'status': 'ok', 'cancelled': cancelled, 'result': result}


@app.post('/api/process/transcribe/cancel')
async def cancel_transcribe(payload: ProcessingRequest) -> dict[str, str]:
    """Ask the running transcription to stop; partial segments are kept."""
    _transcribe_cancelled.add(payload.project_id)
    return {'status': 'cancelling', 'project_id': payload.project_id}


@app.post('/api/process/translate')
async def process_translate(payload: ProcessingRequest) -> dict[str, Any]:
    project = projects.get(payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    segments = project.get('segments', [])
    if not segments:
        raise HTTPException(status_code=409, detail='Transcribe the imported video before translating')

    await emit_progress(payload.project_id, 'Translating', 52, 'Interpreting with global terminology context...')
    update_project(payload.project_id, status='Processing', progress=52)

    translator = HybridTranslationProvider(preference=payload.translation_provider)
    source_language = project.get('source_language', 'en')
    try:
        translator.check_preferred_engine(payload.target_language, source_language)
    except RuntimeError as error:
        update_project(payload.project_id, status='Ready', progress=52)
        raise HTTPException(status_code=409, detail=str(error)) from error
    active_engine = translator.first_engine()
    await emit_progress(
        payload.project_id, 'Translating', 52,
        f'Translating with {active_engine or "no engine"}.', engine=active_engine,
    )
    total = len(segments)
    # Resume support: segments that already carry a translation are kept, so
    # re-running after a failure continues instead of starting over.
    pending = [segment for segment in segments if not segment.get('translation')]
    already_done = total - len(pending)
    if already_done:
        await emit_progress(
            payload.project_id, 'Translating', 52,
            f'Resuming: {already_done} of {total} segments are already translated.',
        )

    loop = asyncio.get_running_loop()
    translated_count = 0
    first_error: Exception | None = None

    def translate_pending() -> None:
        nonlocal translated_count, first_error
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def work(segment: dict[str, Any]) -> str:
            return translator.translate(segment['text'], payload.target_language, source_language)

        # A small bounded pool keeps long runs several times faster while the
        # retry/backoff inside the provider absorbs rate limiting. Google's free
        # endpoint throttles parallel callers, and a single local model (Qwen, NLLB)
        # runs one generation at a time behind its lock, so both run one line at a time.
        serial_engines = ('google', 'nllb', 'qwen')
        workers = 1 if translator.first_engine() in serial_engines else min(4, max(1, len(pending)))
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as pool:
            futures = {pool.submit(work, segment): segment for segment in pending}
            for future in as_completed(futures):
                segment = futures[future]
                try:
                    segment['translation'] = future.result()
                except Exception as error:
                    # Keep every finished translation; remember the first failure.
                    if first_error is None:
                        first_error = error
                    continue
                translated_count += 1
                if translated_count % 5 == 0 or translated_count == len(pending):
                    done = already_done + translated_count
                    asyncio.run_coroutine_threadsafe(
                        emit_progress(
                            payload.project_id, 'Translating',
                            52 + int(28 * done / total),
                            f'Translating segment {done} of {total}...',
                        ),
                        loop,
                    )

    if pending:
        try:
            await run_in_threadpool(translate_pending)
        except Exception as error:
            persist_project(payload.project_id)
            update_project(payload.project_id, status='Ready', progress=52)
            raise HTTPException(
                status_code=502,
                detail=f'Translation failed: {error}. Press Translate again to resume where it stopped.',
            ) from error

    persist_project(payload.project_id)
    done = already_done + translated_count
    if first_error is not None or done < total:
        update_project(payload.project_id, status='Ready', progress=52 + int(28 * done / total))
        raise HTTPException(
            status_code=502,
            detail=(
                f'Translation failed on some segments: {first_error}. '
                f'{done} of {total} are saved. Press Translate again to resume the remaining {total - done}.'
            ),
        )

    if payload.openai_review and api_keys.has_key('openai'):
        # Optional quality pass: OpenAI reviews the local translations line by
        # line. Failures keep the local translation — never the reverse.
        await emit_progress(payload.project_id, 'Translating', 82, 'Reviewing translations with OpenAI...')
        from backend.app.online_providers import openai_chat

        key = api_keys.get_key('openai')
        reviewed = 0
        for segment in segments:
            translation = segment.get('translation')
            if not translation:
                continue
            try:
                improved = (await run_in_threadpool(
                    openai_chat,
                    key,
                    [
                        {'role': 'system', 'content': (
                            f'You review movie-dub subtitles translated to {payload.target_language}. '
                            'Improve naturalness, character tone, and grammar while keeping the meaning '
                            'and a subtitle-friendly length. Respond ONLY with the improved translation.'
                        )},
                        {'role': 'user', 'content': f'Original: {segment["text"]}' + chr(10) + f'Translation: {translation}'},
                    ],
                )).strip()
                if improved and improved != translation:
                    segment['translation'] = improved
                    reviewed += 1
            except Exception:
                continue
        persist_project(payload.project_id)
        await emit_progress(payload.project_id, 'Translating', 82, f'OpenAI review improved {reviewed} lines.')

    translated_chunks = [
        {'id': segment['id'], 'source': segment['text'], 'translation': segment['translation']}
        for segment in segments
        if segment.get('translation')
    ]
    await emit_progress(payload.project_id, 'Translating', 82, 'Translation complete and aligned with timeline.')
    update_project(payload.project_id, status='Ready', progress=82)
    return {'project_id': payload.project_id, 'status': 'ok', 'translation': translated_chunks, 'chunks': translated_chunks}


@app.post('/api/process/tts')
async def process_tts(payload: ProcessingRequest) -> dict[str, Any]:
    project = projects.get(payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    segments = project.get('segments', [])
    if not segments:
        raise HTTPException(status_code=409, detail='Transcribe and translate the video before generating voice')
    if any(not segment.get('translation') for segment in segments):
        raise HTTPException(status_code=409, detail='Translate the transcript before generating voice')
    # An explicit VoxCPM engine must not silently fall back to Edge for a language it can't speak.
    if payload.tts_provider == 'voxcpm2':
        from . import voxcpm_local
        if not voxcpm_local.supports_language(_resolve_lang_code(payload.target_language)):
            raise HTTPException(status_code=409, detail=f'VoxCPM2 only speaks English here, so it cannot dub {payload.target_language}. Choose another voice engine or an English target.')

    await emit_progress(payload.project_id, 'Generating Voice', 70, 'Assigning voices and synthesizing dubbed audio...')
    update_project(payload.project_id, status='Processing', progress=70)

    audio_dir = DATA_ROOT / payload.project_id / 'dub'
    audio_dir.mkdir(parents=True, exist_ok=True)
    total_segments = len(segments)
    characters = project.get('characters') or {}
    lang_code = _resolve_lang_code(payload.target_language)
    use_openai_acting = payload.openai_acting and api_keys.has_key('openai')
    project_chunks = [
        (float(chunk.get('start') or 0), float(chunk.get('start') or 0) + float(chunk.get('duration') or 0))
        for chunk in project.get('chunks', [])
    ]

    def part_label_for(start: float) -> str:
        for part_index, (part_start, part_end) in enumerate(project_chunks):
            if part_start - 1e-6 <= start < part_end:
                return f' (part {part_index + 1} of {len(project_chunks)})'
        return ''

    def mixed_voice_labels(segments_list: list[dict[str, Any]]) -> list[str]:
        """Voice label per line for payload.voice == 'mixed'.

        Without speaker detection every line would get the same fallback
        voice, so alternate male/female on conversation breaks — a pause
        longer than ~1 s suggests the other person started talking. Lines
        within one continuous run keep the same voice.
        """
        labels: list[str] = []
        current = 'Male Voice A'
        previous_end: float | None = None
        for segment in segments_list:
            start = float(segment.get('start') or 0)
            end = float(segment.get('end') or start)
            if previous_end is not None and start - previous_end > 1.0:
                current = 'Female Voice B' if current == 'Male Voice A' else 'Male Voice A'
            labels.append(current)
            previous_end = end
        return labels

    mixed_labels = mixed_voice_labels(segments) if payload.voice == 'mixed' else None
    clone_references: dict[str, Path] | None = None
    if payload.tts_provider == 'voxcpm_clone':
        from . import voxcpm_dub

        try:
            voxcpm_dub.check_project({**project, 'id': payload.project_id}, payload.target_language)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        ffmpeg = find_media_tool('ffmpeg')
        if not ffmpeg:
            raise HTTPException(status_code=409, detail='FFmpeg was not found. Install FFmpeg to build voice references.')
        clone_references = await run_in_threadpool(
            voxcpm_dub.build_references, ffmpeg, {**project, 'id': payload.project_id},
            voxcpm_dub.dub_dir(payload.project_id),
        )
        if not clone_references:
            raise HTTPException(status_code=409, detail='Could not cut a voice reference from the source audio.')
    # Resume support: audio files generated by an earlier (failed or partial)
    # run are reused, so re-running Dub continues instead of starting over.
    previous_by_id = {str(item.get('id')): item for item in project.get('audio_segments', [])}
    audio_segments: list[dict[str, Any]] = []
    reused = 0
    try:
        for index, segment in enumerate(segments):
            previous = previous_by_id.get(str(segment['id']))
            output_path = audio_dir / f'{index + 1:04d}.mp3'
            # The voice this line would get in the current run. Reused audio is
            # only valid when it was synthesized with the SAME voice — changing
            # the voice (e.g. switching to 'mixed' or a new character voice)
            # must re-synthesize, not silently keep the old recording.
            char_name = str(segment.get('speaker') or 'Speaker 1')
            profile = characters.get(char_name) or {}
            expected_voice = profile.get('voice') or (mixed_labels[index] if mixed_labels is not None else payload.voice)
            stored_voice = str(previous.get('voice') or '') if previous else ''
            # Clone lines are stored as 'voxcpm_clone:<speaker>', so they match only the same speaker.
            clone_voice = f"{voxcpm_dub.CLONE_ENGINE}:{char_name}" if clone_references is not None else None
            voice_matches = (stored_voice == clone_voice) if clone_references is not None else (
                stored_voice == expected_voice or stored_voice.endswith(f":{expected_voice}"))
            if previous and Path(str(previous.get('path') or '')).is_file() and voice_matches:
                audio_segments.append({
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'source_start': float(segment['start']),
                    'text': segment['translation'],
                    'voice': previous.get('voice', payload.voice),
                    'path': previous['path'],
                })
                reused += 1
            elif clone_references is not None:
                # VoxCPM clone engine: speak the line in the speaker's own voice. Never
                # falls back to another engine, so the cloned voice can't silently change.
                from . import voxcpm_dub

                reference = clone_references.get(char_name) or next(iter(clone_references.values()))
                clone_out = audio_dir / f'{index + 1:04d}.wav'
                await run_in_threadpool(
                    voxcpm_dub.synthesize_clone_line, segment['translation'], reference, clone_out,
                )
                audio_segments.append({
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'source_start': float(segment['start']),
                    'text': segment['translation'],
                    'character': char_name,
                    'voice': f"{voxcpm_dub.CLONE_ENGINE}:{char_name}",
                    'path': str(clone_out),
                })
            else:
                previous_text = str(segments[index - 1].get('translation') or '') if index > 0 else None
                following_text = str(segments[index + 1].get('translation') or '') if index + 1 < total_segments else None
                if payload.acting:
                    analysis = acting.analyze(
                        segment['translation'], previous_text, following_text,
                        char_name, payload.target_language, use_openai=use_openai_acting,
                    )
                else:
                    analysis = {'emotion': 'Neutral', 'intensity': payload.intensity,
                                'style': 'Natural', 'pacing': 'Natural', 'pause_before': False,
                                'analyzer': 'none'}
                emotion = (analysis.get('emotion') or 'Neutral') if payload.emotion == 'Auto' else payload.emotion
                intensity = analysis.get('intensity', payload.intensity) if payload.emotion == 'Auto' else payload.intensity
                # Character voice wins; 'mixed' alternates genders per line.
                line_voice = payload.voice
                if mixed_labels is not None and not profile.get('voice'):
                    line_voice = mixed_labels[index]
                line_result = await run_in_threadpool(
                    tts_service.synthesize_line,
                    segment['translation'],
                    str(output_path),
                    provider_pref=payload.tts_provider,
                    language_name=payload.target_language,
                    lang_code=lang_code,
                    voice_pref=profile.get('voice') or line_voice,
                    model_pref=payload.tts_model,
                    speed=payload.speed,
                    emotion=emotion,
                    intensity=int(intensity),
                    character_profile=profile,
                    analysis=analysis,
                )
                audio_segments.append({
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'source_start': float(segment['start']),
                    'text': segment['translation'],
                    'character': char_name,
                    'voice': f"{line_result['provider']}:{line_result['voice']}",
                    'path': line_result['path'],
                })
            # Persist as we go so a failed long run keeps its finished work.
            project['audio_segments'] = list(audio_segments)
            if (index + 1) % 10 == 0 or index + 1 == total_segments:
                persist_project(payload.project_id)
            if (index + 1) % 5 == 0 or index + 1 == total_segments:
                await emit_progress(
                    payload.project_id,
                    'Generating Voice',
                    70 + int(24 * (index + 1) / total_segments),
                    f'Synthesizing segment {index + 1} of {total_segments}'
                    f'{part_label_for(float(segment.get("start") or 0))}...',
                )
    except Exception as error:
        persist_project(payload.project_id)
        update_project(payload.project_id, status='Ready', progress=70)
        raise HTTPException(
            status_code=502,
            detail=f'Voice generation failed: {error}. Press Dub again to resume where it stopped.',
        ) from error

    await emit_progress(
        payload.project_id, 'Generating Voice', 95,
        'Fitting each dubbed line into the timeline...',
    )
    audio_segments = await run_in_threadpool(
        _resolve_speech_overlaps, audio_segments, _media_duration_seconds(project),
    )
    project['audio_segments'] = audio_segments
    persist_project(payload.project_id)
    if reused:
        await emit_progress(
            payload.project_id, 'Generating Voice', 96,
            f'Dub audio ready ({reused} parts reused from the previous run).',
        )
    vtt_segments = [
        {**segment, 'text': segment['translation']}
        for segment in segments
    ]
    result = {
        'provider': 'Edge TTS',
        'voice': audio_segments[0]['voice'] if audio_segments else payload.voice,
        'segments': audio_segments,
        'subtitle_vtt': chunk_engine.build_vtt(vtt_segments),
    }

    await emit_progress(payload.project_id, 'Generating Voice', 96, 'Dubbed waveform synced with the original timeline.')
    update_project(payload.project_id, status='Ready', progress=96)
    return {'project_id': payload.project_id, 'status': 'ok', 'result': result}


def _apply_platform_style(ffmpeg: str, source: str, dest: str, target_w: int, target_h: int, fit: str) -> None:
    """Convert a finished video to a platform aspect ratio.

    'blur' fills the target frame with a blurred, scaled copy of the video and
    centers the original on top (the TikTok/Shorts look). 'crop' center-crops
    to fill the frame. Video is re-encoded; audio is copied untouched.
    """
    if fit == 'crop':
        vf = f'scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},setsar=1'
        command = [ffmpeg, '-y', '-i', source, '-vf', vf, '-map', '0:v:0']
    else:
        command = [ffmpeg, '-y', '-i', source, '-filter_complex',
                   f'[0:v]split=2[bg][fg];'
                   f'[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,'
                   f'crop={target_w}:{target_h},boxblur=luma_radius=16:luma_power=2[bgd];'
                   f'[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fgs];'
                   f'[bgd][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v]',
                   '-map', '[v]']
    command.extend(['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                    '-map', '0:a?', '-c:a', 'copy', dest])
    subprocess.run(command, check=True, capture_output=True, text=True)


@app.post('/api/process/export')
async def process_export(payload: ProcessingRequest) -> dict[str, Any]:
    project = projects.get(payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    segments = project.get('segments', [])
    if not segments:
        raise HTTPException(status_code=409, detail='Transcribe the imported video before exporting')
    if payload.burn_subtitles and not payload.include_subtitles:
        raise HTTPException(status_code=422, detail='Enable Subtitle sync before burning captions into the video')

    export_message = 'Preparing subtitles and dubbed video...' if payload.include_subtitles else 'Preparing dubbed video without subtitles...'
    await emit_progress(payload.project_id, 'Exporting', 88, export_message)
    update_project(payload.project_id, status='Exporting', progress=88)

    export_dir = DATA_ROOT / payload.project_id / 'export'
    export_dir.mkdir(parents=True, exist_ok=True)
    stems = project.get('stems') or {}
    separated = any(Path(str(path)).is_file() for path in stems.values())
    audio_segments = project.get('audio_segments', [])
    if audio_segments:
        await emit_progress(payload.project_id, 'Exporting', 88, 'Fitting each dubbed line into the movie...')
        audio_segments = await run_in_threadpool(
            _resolve_speech_overlaps, audio_segments, _media_duration_seconds(project),
        )
        project['audio_segments'] = audio_segments
        persist_project(payload.project_id)
    # Skip any clip that cannot be read (for example a header-only file), so one bad line
    # does not fail the whole export. The skipped lines are reported, not silently dropped.
    ffprobe_for_export = find_media_tool('ffprobe')
    if ffprobe_for_export and audio_segments:
        readable = [item for item in audio_segments if item.get('path') and Path(str(item['path'])).is_file() and _is_readable_audio(ffprobe_for_export, str(item['path']))]
        skipped = len(audio_segments) - len(readable)
        if skipped:
            await emit_progress(payload.project_id, 'Exporting', 88, f'Skipped {skipped} unreadable dubbed line(s); exporting the rest.')
        audio_segments = readable
    dub_by_id = {str(item.get('id')): item for item in audio_segments}

    export_segments = []
    vtt = None
    subtitle_url = None
    subtitle_path = None
    if payload.include_subtitles:
        export_segments = []
        for segment in segments:
            item = {**segment, 'text': segment.get('translation') or segment['text']}
            dub = dub_by_id.get(str(segment.get('id')))
            if dub:
                item['start'] = dub['start']
                item['end'] = dub['end']
            export_segments.append(item)
        vtt = chunk_engine.build_vtt(export_segments)
        subtitle_path = export_dir / 'dubbed-subtitles.vtt'
        subtitle_path.write_text(vtt, encoding='utf-8')
        project['subtitle_path'] = str(subtitle_path)
        subtitle_url = f'/api/projects/{payload.project_id}/export/subtitle'
        if payload.burn_subtitles:
            (export_dir / 'dubbed-subtitles.srt').write_text(build_srt(export_segments), encoding='utf-8')
    else:
        project['subtitle_path'] = None

    video_path = None
    parts_list: list[dict[str, Any]] = []
    export_destination_path: Path | None = None
    exported_files: list[str] = []
    media_path = project.get('media_path')
    if audio_segments and media_path:
        ffmpeg = find_media_tool('ffmpeg')
        ffprobe = find_media_tool('ffprobe')
        if not ffmpeg or not ffprobe:
            raise HTTPException(status_code=503, detail='FFmpeg and ffprobe must be installed and available on PATH')
        try:
            probe = subprocess.run(
                [ffprobe, '-v', 'error', '-show_entries', 'format=duration:stream=codec_type,width,height', '-of', 'json', media_path],
                check=True,
                capture_output=True,
                text=True,
            )
            probe_data = json.loads(probe.stdout)
            duration = float(probe_data['format']['duration'])
            has_original_audio = any(stream.get('codec_type') == 'audio' for stream in probe_data.get('streams', []))
            video_stream = next((stream for stream in probe_data.get('streams', []) if stream.get('width')), None)
            source_width = int(video_stream['width']) if video_stream else 0
            source_height = int(video_stream['height']) if video_stream else 0
            # Platform style: target frame size for the export (9:16 vertical,
            # 16:9 standard, 1:1 square). Skipped when the source already has
            # the target aspect (within 3%).
            platform_target = {'vertical': (1080, 1920), 'standard': (1920, 1080), 'square': (1080, 1080)}.get(payload.platform_style)
            platform_skip = (
                not platform_target
                or not source_width or not source_height
                or abs(source_width / source_height - platform_target[0] / platform_target[1]) / (platform_target[0] / platform_target[1]) < 0.03
            )
            mode = payload.export_mode or 'single'
            chunks = [chunk for chunk in (project.get('chunks') or []) if Path(str(chunk.get('path') or '')).is_file()]
            want_combined = mode in ('single', 'both') or not chunks
            want_parts = mode in ('parts', 'both') and bool(chunks)
            def mix_group(group: list[dict[str, Any]], group_index: int) -> Path:
                """Pre-mix a bounded batch of dubbed segments into one track.

                A long video can have hundreds of segments; putting every input
                and every adelay into a single ffmpeg command overflows the
                Windows ~32k command-line limit (WinError 206). Staged mixing
                keeps each command short and scales to any number of segments.
                """
                group_path = mix_dir / f'group_{group_index:03d}.wav'
                group_filters = []
                labels = []
                for pos, audio in enumerate(group):
                    delay_ms = max(0, round(float(audio['start']) * 1000))
                    label = f'v{pos}'
                    group_filters.append(f'[{pos}:a]adelay={delay_ms}|{delay_ms}[{label}]')
                    labels.append(f'[{label}]')
                group_filters.append(
                    f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[out]"
                )
                group_command = [ffmpeg, '-y']
                for audio in group:
                    group_command.extend(['-i', audio['path']])
                group_command.extend([
                    '-filter_complex', ';'.join(group_filters),
                    '-map', '[out]', '-c:a', 'pcm_s16le', str(group_path),
                ])
                subprocess.run(group_command, check=True, capture_output=True, text=True)
                return group_path

            if want_combined:
                mix_dir = export_dir / 'mixgroups'
                mix_dir.mkdir(parents=True, exist_ok=True)
                for old in mix_dir.glob('group_*.wav'):
                    old.unlink()
                group_size = 80
                groups = [audio_segments[i:i + group_size] for i in range(0, len(audio_segments), group_size)]
                group_paths = []
                for group_index, group in enumerate(groups):
                    await emit_progress(
                        payload.project_id, 'Exporting',
                        88 + int(6 * group_index / len(groups)),
                        f'Mixing dubbed audio {group_index + 1} of {len(groups)}...',
                    )
                    group_paths.append(mix_group(group, group_index))

                filter_parts = []
                command = [ffmpeg, '-y', '-i', media_path]
                # When the audio was separated, the original dialogue layer is
                # replaced by the dub and the "original audio" source is the
                # SFX/music bed with per-layer toggles. The 0% rule must not
                # drop the bed: the slider is the duck depth during dialogue.
                bed_layers: list[str] = []
                if payload.preserve_original_audio:
                    music_stem = str(stems.get('music') or '')
                    if music_stem and Path(music_stem).is_file() and payload.keep_music:
                        bed_layers.append(music_stem)
                    sfx_stem = str(stems.get('sfx') or stems.get('bed') or '')
                    if sfx_stem and Path(sfx_stem).is_file() and payload.keep_sfx:
                        bed_layers.append(sfx_stem)
                bed_active = len(bed_layers) > 0
                # Once separated, the raw media audio must never leak back into
                # the mix: the only original layers are the separated ones.
                original_present = bed_active if separated else has_original_audio
                use_original = (
                    payload.preserve_original_audio and original_present
                    and (bed_active or payload.original_audio_volume > 0.001)
                )
                original_input = '[0:a:0]'
                group_offset = 1
                if bed_active and use_original:
                    for layer in bed_layers:
                        command.extend(['-i', layer])
                    if len(bed_layers) == 1:
                        filter_parts.append('[1:a]anull[bedbus]')
                    else:
                        filter_parts.append('[1:a][2:a]amix=inputs=2:duration=longest:normalize=0[bedbus]')
                    original_input = '[bedbus]'
                    group_offset = 1 + len(bed_layers)
                group_labels = []
                for group_index, group_path in enumerate(group_paths):
                    # Group tracks already carry absolute timing from their internal
                    # adelay chains, so they mix in at t=0 without extra delay.
                    command.extend(['-i', str(group_path)])
                    group_labels.append(f'[{group_offset + group_index}:a]')
                filter_parts.append(
                    f"{''.join(group_labels)}amix=inputs={len(group_labels)}:duration=longest:normalize=0[voicesmix]"
                )
                # 0% original audio = dub voice only: drop the original track
                # from the mix entirely instead of "ducking it to zero".
                use_ducking = (
                    use_original and payload.auto_duck
                    and payload.original_audio_volume < 0.999
                )
                if use_ducking:
                    # Movie feel: original audio at full volume in gaps, dipped to
                    # the chosen level while dubbed speech plays (smooth attack).
                    # The voice bus is padded to the full duration first (so the
                    # sidechain never ends before the video does), then split:
                    # one copy drives the sidechain, the other is mixed back in
                    # after the duck.
                    filter_parts.append(f'[voicesmix]apad=whole_dur={duration:.3f}[voicespad]')
                    if bed_active and payload.original_audio_volume <= 0.02:
                        # One compressor pass bottoms out near 3%; chain a second
                        # pass so a 0% duck goes to inaudible silence.
                        filter_parts.append('[voicespad]asplit=3[vsidechain1][vsidechain2][vmix]')
                        ratio = _duck_ratio(0.02)
                        filter_parts.append(
                            f'{original_input}[vsidechain1]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=8:release=350[ducked1]'
                        )
                        filter_parts.append(
                            f'[ducked1][vsidechain2]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=8:release=350[ducked]'
                        )
                    else:
                        filter_parts.append('[voicespad]asplit=2[vsidechain][vmix]')
                        ratio = _duck_ratio(payload.original_audio_volume)
                        filter_parts.append(
                            f'{original_input}[vsidechain]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=8:release=350[ducked]'
                        )
                    mix_labels = ['[ducked]', '[vmix]']
                else:
                    if use_original:
                        filter_parts.append(f'{original_input}volume={payload.original_audio_volume:.3f}[source]')
                        mix_labels = ['[source]', '[voicesmix]']
                    else:
                        mix_labels = ['[voicesmix]']
                max_audio_end = max((float(seg.get('end') or 0) for seg in audio_segments), default=duration)
                export_duration = max(duration, max_audio_end)
                pad_seconds = max(0.0, max_audio_end - duration)

                filter_parts.append(
                    f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=longest:normalize=0,"
                    f'apad=whole_dur={export_duration:.3f}[dubbed]'
                )
                video_path = export_dir / 'dubbed-video.mp4'
                command.extend([
                    '-filter_complex', ';'.join(filter_parts),
                    '-map', '0:v:0', '-map', '[dubbed]',
                ])
                if payload.burn_subtitles:
                    srt_path = escape_filter_path(export_dir / 'dubbed-subtitles.srt')
                    vf_parts = [f"subtitles='{srt_path}'"]
                    if pad_seconds > 0.05:
                        vf_parts.append(f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}")
                    command.extend(['-vf', ','.join(vf_parts), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20'])
                elif pad_seconds > 0.05:
                    # Dub audio extends past original video: freeze final frame so video never ends before sound
                    command.extend(['-vf', f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}", '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20'])
                else:
                    command.extend(['-c:v', 'copy'])
                command.extend([
                    '-c:a', 'aac', '-b:a', '192k',
                    '-t', f'{export_duration:.3f}', '-movflags', '+faststart', str(video_path),
                ])
                subprocess.run(command, check=True, capture_output=True, text=True)
                project['video_path'] = str(video_path)
                if not platform_skip:
                    styled = export_dir / 'styled-video.mp4'
                    _apply_platform_style(ffmpeg, str(video_path), str(styled), platform_target[0], platform_target[1], payload.platform_fit)
                    video_path.unlink(missing_ok=True)
                    styled.replace(video_path)
                    await emit_progress(payload.project_id, 'Exporting', 94,
                                        f'Converted to {platform_target[0]}×{platform_target[1]} for the selected platform.')

            if want_parts:
                # One dubbed MP4 per chunk, reusing the keyframe-aligned chunk
                # files as video input (stream copy) and offsetting each part's
                # segments by the chunk start. Fast, and each part is standalone.
                parts_dir = export_dir / 'parts'
                parts_dir.mkdir(parents=True, exist_ok=True)
                for old in parts_dir.glob('part_*.mp4'):
                    old.unlink()
                total_parts = len(chunks)
                for chunk_index, chunk in enumerate(chunks):
                    chunk_start = float(chunk.get('start') or 0)
                    chunk_duration = float(chunk.get('duration') or 0)
                    await emit_progress(
                        payload.project_id, 'Exporting',
                        88 + int(6 * chunk_index / total_parts),
                        f'Exporting part {chunk_index + 1} of {total_parts}...',
                    )
                    part_segments = [
                        audio for audio in audio_segments
                        if chunk_start - 1e-6 <= float(audio.get('start') or 0) < chunk_start + chunk_duration
                    ]
                    part_path = parts_dir / f'part_{chunk_index + 1:03d}.mp4'
                    # Layered bed: SFX/ambience and music are independent toggles
                    # and are never dropped by the 0% rule — the slider is the
                    # duck depth during dialogue instead.
                    part_bed_layers: list[str] = []
                    if payload.preserve_original_audio:
                        music_stem = str(stems.get('music') or '')
                        if music_stem and Path(music_stem).is_file() and payload.keep_music:
                            part_bed_layers.append(music_stem)
                        sfx_stem = str(stems.get('sfx') or stems.get('bed') or '')
                        if sfx_stem and Path(sfx_stem).is_file() and payload.keep_sfx:
                            part_bed_layers.append(sfx_stem)
                    part_bed_active = len(part_bed_layers) > 0
                    part_use_original = (
                        payload.preserve_original_audio
                        and (part_bed_active if separated else has_original_audio)
                        and (part_bed_active or payload.original_audio_volume > 0.001)
                    )
                    part_use_ducking = (
                        part_use_original and payload.auto_duck
                        and payload.original_audio_volume < 0.999
                    )

                    if not part_segments:
                        # No dubbed speech in this part: still strip the original
                        # dialogue when the audio was separated — ship the chunk
                        # video with the bed layers (at the chosen level) or
                        # silent in dub-voice-only mode.
                        if part_bed_active:
                            part_command = [ffmpeg, '-y', '-i', str(chunk['path'])]
                            for layer_index, layer in enumerate(part_bed_layers):
                                part_command.extend([
                                    '-ss', f'{chunk_start:.3f}', '-t', f'{chunk_duration:.3f}', '-i', layer,
                                ])
                            if part_use_original:
                                bed_gain = (
                                    '' if payload.original_audio_volume > 0.999
                                    else f'volume={payload.original_audio_volume:.3f},'
                                )
                                bed_sources = ''.join(
                                    f'[{layer_index + 1}:a]' for layer_index in range(len(part_bed_layers))
                                )
                                bed_graph = (
                                    f'{bed_sources}amix=inputs={len(part_bed_layers)}:duration=longest:normalize=0,{bed_gain}anull[bed]'
                                    if len(part_bed_layers) > 1
                                    else f'[{len(part_bed_layers) + 1}:a]{bed_gain}anull[bed]'
                                )
                                part_command.extend([
                                    '-filter_complex', bed_graph,
                                    '-map', '0:v:0', '-map', '[bed]',
                                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                                    '-t', f'{chunk_duration:.3f}', str(part_path),
                                ])
                            else:
                                # Dub-voice-only mode: this part has no dub lines,
                                # so it ships with a silent audio track.
                                part_command.extend([
                                    '-map', '0:v:0', '-c:v', 'copy', '-an',
                                    '-t', f'{chunk_duration:.3f}', str(part_path),
                                ])
                        elif separated:
                            # Separated but both layers are off: strip the voice,
                            # ship the video with a silent audio track.
                            part_command = [
                                ffmpeg, '-y', '-i', str(chunk['path']), '-map', '0:v:0',
                                '-c:v', 'copy', '-an', '-t', f'{chunk_duration:.3f}', str(part_path),
                            ]
                        else:
                            part_command = [ffmpeg, '-y', '-i', str(chunk['path']), '-c', 'copy', str(part_path)]
                        subprocess.run(part_command, check=True, capture_output=True, text=True)
                    else:
                        part_filters = []
                        part_command = [ffmpeg, '-y', '-i', str(chunk['path'])]
                        voice_labels = []
                        for pos, audio in enumerate(part_segments):
                            delay_ms = max(0, round((float(audio.get('start') or 0) - chunk_start) * 1000))
                            part_command.extend(['-i', audio['path']])
                            part_filters.append(f'[{pos + 1}:a]adelay={delay_ms}|{delay_ms}[v{pos}]')
                            voice_labels.append(f'[v{pos}]')
                        part_original_input = '[0:a:0]'
                        if part_bed_active and part_use_original:
                            for layer_index, layer in enumerate(part_bed_layers):
                                part_command.extend([
                                    '-ss', f'{chunk_start:.3f}', '-t', f'{chunk_duration:.3f}', '-i', layer,
                                ])
                            first_bed = len(part_segments) + 1
                            if len(part_bed_layers) == 1:
                                part_filters.append(f'[{first_bed}:a]anull[pbedbus]')
                            else:
                                part_filters.append(
                                    f'[{first_bed}:a][{first_bed + 1}:a]amix=inputs=2:duration=longest:normalize=0[pbedbus]'
                                )
                            part_original_input = '[pbedbus]'
                        if part_use_ducking:
                            # Same movie-style ducking as the combined export: only
                            # the moments with dub voice dip the bed layers.
                            part_filters.append(
                                f"{''.join(voice_labels)}amix=inputs={len(voice_labels)}:duration=longest:normalize=0[pvoicesmix]"
                            )
                            part_filters.append(f'[pvoicesmix]apad=whole_dur={chunk_duration:.3f}[pvoicespad]')
                            if part_bed_active and payload.original_audio_volume <= 0.02:
                                # Deepest duck: chain a second compressor pass.
                                part_filters.append('[pvoicespad]asplit=3[ps1][ps2][pvmix]')
                                ratio = _duck_ratio(0.02)
                                part_filters.append(
                                    f'{part_original_input}[ps1]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=8:release=350[pducked1]'
                                )
                                part_filters.append(
                                    f'[pducked1][ps2]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=8:release=350[pducked]'
                                )
                            else:
                                part_filters.append('[pvoicespad]asplit=2[psidechain][pvmix]')
                                ratio = _duck_ratio(payload.original_audio_volume)
                                part_filters.append(
                                    f'{part_original_input}[psidechain]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=8:release=350[pducked]'
                                )
                            mix_labels = ['[pducked]', '[pvmix]']
                        else:
                            mix_labels = []
                            if part_use_original:
                                part_filters.append(f'{part_original_input}volume={payload.original_audio_volume:.3f}[osrc]')
                                mix_labels.append('[osrc]')
                            mix_labels.extend(voice_labels)
                        part_filters.append(
                            f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=longest:normalize=0,"
                            f'apad=whole_dur={chunk_duration:.3f}[dubbed]'
                        )
                        part_command.extend([
                            '-filter_complex', ';'.join(part_filters),
                            '-map', '0:v:0', '-map', '[dubbed]',
                            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                            '-t', f'{chunk_duration:.3f}', str(part_path),
                        ])
                        subprocess.run(part_command, check=True, capture_output=True, text=True)
                    if not platform_skip and part_path.is_file():
                        styled_part = part_path.with_name(f'styled-{part_path.name}')
                        _apply_platform_style(ffmpeg, str(part_path), str(styled_part), platform_target[0], platform_target[1], payload.platform_fit)
                        styled_part.replace(part_path)
                    parts_list.append({'index': chunk_index + 1, 'name': part_path.name})
                project['parts_dir'] = str(parts_dir)
                persist_project(payload.project_id)

            # Write the finished files straight into a folder — the user's choice
            # or the Videos/DubVibe default. Never skip this: an export without
            # a visible destination is the same as a lost one. (The one exception:
            # the browser delivers the files itself when the user picked a folder
            # through the directory picker.)
            if payload.export_folder or payload.deliver_to_folder:
                if payload.export_folder:
                    destination = Path(str(payload.export_folder)).expanduser()
                else:
                    destination = _default_export_folder(project)
                destination.mkdir(parents=True, exist_ok=True)
                copies: list[tuple[Path, Path]] = []
                if video_path is not None:
                    copies.append((video_path, destination / 'dubbed-video.mp4'))
                if want_parts:
                    for part_file in sorted((export_dir / 'parts').glob('part_*.mp4')):
                        copies.append((part_file, destination / part_file.name))
                if payload.include_subtitles and project.get('subtitle_path'):
                    subtitle_file = Path(str(project['subtitle_path']))
                    if subtitle_file.is_file():
                        copies.append((subtitle_file, destination / 'dubbed-subtitles.vtt'))
                burned_srt = export_dir / 'dubbed-subtitles.srt'
                if payload.burn_subtitles and burned_srt.is_file():
                    copies.append((burned_srt, destination / 'dubbed-subtitles.srt'))
                for source_file, target_file in copies:
                    shutil.copy2(source_file, target_file)
                    exported_files.append(target_file.name)
                export_destination_path = destination
                project['export_folder'] = str(destination)
                persist_project(payload.project_id)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            update_project(payload.project_id, status='Ready', progress=96)
            detail = error.stderr[-1500:] if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
            raise HTTPException(status_code=500, detail=f'Video export failed: {detail}') from error

    final_message = 'Export package ready for review.'
    if parts_list and video_path is not None:
        final_message = f'Export ready: combined video plus {len(parts_list)} parts.'
    elif parts_list:
        final_message = f'Export ready: {len(parts_list)} parts.'
    if export_destination_path is not None:
        final_message += f' Saved to {export_destination_path}.'
    await emit_progress(payload.project_id, 'Exporting', 100, final_message)
    update_project(payload.project_id, status='Ready', progress=100)
    response_dict: dict[str, Any] = {
        'project_id': payload.project_id,
        'status': 'ok',
        'audio_segments': project.get('audio_segments', []),
        'export': {
            'format': 'vtt' if payload.include_subtitles else None,
            'payload': vtt,
            'segments': export_segments,
            'subtitle_url': subtitle_url,
            'video_url': f'/api/projects/{payload.project_id}/export/video' if video_path else None,
            'parts': parts_list,
            'folder': str(export_destination_path) if export_destination_path else None,
            'files': exported_files,
            'message': final_message,
        },
    }
    # Check for remaining overlaps
    overlap_warnings = []
    sorted_segs = sorted(project.get('audio_segments', []), key=lambda s: s['start'])
    for i in range(len(sorted_segs) - 1):
        a, b = sorted_segs[i], sorted_segs[i + 1]
        if b['start'] < a['end']:
            overlap_warnings.append(a['id'])
    if overlap_warnings:
        response_dict['overlap_warnings'] = overlap_warnings
    return response_dict


def _safe_folder_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', '_', (name or '').strip()).strip(' .')[:60]
    return cleaned or 'project'


def _default_export_folder(project: dict[str, Any]) -> Path:
    """Default export destination: Videos/DubVibe/<project name>."""
    videos_dir = Path.home() / 'Videos'
    base = videos_dir if videos_dir.is_dir() else Path.home()
    return base / 'DubVibe' / _safe_folder_name(str(project.get('name') or 'project'))


@app.get('/api/export/destination')
async def get_export_destination(project_id: str) -> dict[str, str]:
    """Default folder exports are written to: Videos/DubVibe/<project name>."""
    project = projects.get(project_id) or {}
    return {'folder': str(_default_export_folder(project))}


@app.get('/api/projects/{project_id}/export/parts/{index}')
async def download_export_part(project_id: str, index: int, request: Request):
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    parts_dir = project.get('parts_dir')
    part_path = Path(str(parts_dir)) / f'part_{index:03d}.mp4' if parts_dir else None
    if not part_path or not part_path.is_file():
        raise HTTPException(status_code=404, detail='Export part has not been generated')
    return _file_response_with_range(request, part_path, 'video/mp4', part_path.name)


@app.get('/api/projects/{project_id}/export/{kind}')
async def download_export(project_id: str, kind: str, request: Request):
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    if kind == 'subtitle':
        path = project.get('subtitle_path')
        media_type = 'text/vtt'
        filename = 'dubbed-subtitles.vtt'
    elif kind == 'video':
        path = project.get('video_path')
        media_type = 'video/mp4'
        filename = 'dubbed-video.mp4'
    else:
        raise HTTPException(status_code=404, detail='Export type not found')
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail='Export file has not been generated')
    return _file_response_with_range(request, Path(path), media_type, filename)


@app.get('/api/projects/{project_id}/dub-audio/{index}')
async def get_project_dub_audio(project_id: str, index: int, request: Request):
    """Serve one synthesized dub segment (1-based index) for preview playback."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    audio_segments = project.get('audio_segments', [])
    if index < 1 or index > len(audio_segments):
        raise HTTPException(status_code=404, detail='Dub audio segment not found')
    audio_path = audio_segments[index - 1].get('path')
    if not audio_path or not Path(audio_path).is_file():
        raise HTTPException(status_code=404, detail='Dub audio has not been generated')
    media_type = 'audio/wav' if str(audio_path).lower().endswith('.wav') else 'audio/mpeg'
    return _file_response_with_range(request, Path(audio_path), media_type)


@app.post('/api/process/speaker')
async def process_speaker(payload: ProcessingRequest) -> dict[str, Any]:
    """Speaker diarization: label every segment with its speaker's voice."""
    project = projects.get(payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    segments = project.get('segments', [])
    if not segments:
        raise HTTPException(status_code=409, detail='Transcribe the imported video before detecting speakers')
    if not speaker_diarization.is_available():
        raise HTTPException(
            status_code=503,
            detail='Speaker detection needs the Resemblyzer package (pip install resemblyzer), which is not installed in this environment.',
        )
    media_path = project.get('media_path')
    if not media_path or not Path(media_path).is_file():
        raise HTTPException(status_code=409, detail='Import a video before detecting speakers')
    ffmpeg = find_media_tool('ffmpeg')
    if not ffmpeg:
        raise HTTPException(status_code=503, detail='FFmpeg must be installed and available on PATH')

    await emit_progress(payload.project_id, 'Detecting speakers', 10, 'Analyzing voices to identify speakers...')
    update_project(payload.project_id, status='Processing')
    loop = asyncio.get_running_loop()

    def run_diarization() -> tuple[list[str], dict[str, dict[str, Any]]]:
        return speaker_diarization.assign_speakers(
            str(media_path), segments, ffmpeg,
            progress_cb=lambda percent: asyncio.run_coroutine_threadsafe(
                emit_progress(payload.project_id, 'Detecting speakers',
                              10 + int(80 * percent / 100),
                              f'Analyzing voices: {percent}%...'),
                loop,
            ),
        )

    try:
        speakers, genders = await run_in_threadpool(run_diarization)
    except Exception as error:
        update_project(payload.project_id, status='Ready')
        raise HTTPException(status_code=500, detail=f'Speaker detection failed: {error}') from error

    for segment, speaker in zip(segments, speakers):
        segment['speaker'] = speaker
    # Auto-create character profiles so every speaker can keep a persistent
    # voice, and assign a gender-appropriate Edge voice for the target
    # language. A voice the user picked manually is never overwritten.
    characters = project.get('characters') or {}
    target_language = str(payload.target_language or 'Khmer')
    assigned: list[str] = []
    for speaker in speakers:
        profile = characters.setdefault(speaker, {})
        info = genders.get(speaker) or {}
        if info.get('gender'):
            profile['gender'] = info['gender']
        current_voice = str(profile.get('voice') or '').strip()
        if not current_voice or current_voice == 'Default voice':
            neural = edge_gender_voice(target_language, info.get('gender') or '')
            if neural:
                profile['voice'] = neural
                if profile.get('provider') in (None, '', 'auto'):
                    profile['provider'] = 'edge'
                assigned.append(f'{speaker} → {neural}')
    project['characters'] = characters
    persist_project(payload.project_id)

    found = sorted(set(speakers))
    gender_bits = [f"{name} ({genders[name]['gender'].lower()})" if genders.get(name, {}).get('gender') else name
                   for name in found]
    final_message = f"Detected {len(found)} speakers: {', '.join(gender_bits)}."
    if assigned:
        final_message += f" Voices assigned: {', '.join(assigned)}."
    await emit_progress(payload.project_id, 'Detecting speakers', 100, final_message)
    update_project(payload.project_id, status='Ready')
    return {'project_id': payload.project_id, 'status': 'ok', 'speakers': found,
            'genders': {name: (genders.get(name) or {}).get('gender') for name in found},
            'message': final_message}


DUB_SPEECH_GAP = 0.08
# Bounded speed-up applied to a dub line that is longer than the room before
# the next cue. Speech is never cut or trimmed — a line that cannot fit even
# at this tempo pushes the next line (and the last line is clamped to the
# movie's end) instead of overlapping.
_MAX_DUB_TEMPO = 1.35


def _is_readable_audio(ffprobe: str, path: str) -> bool:
    """True only when ffprobe reads a clip that has audio in it.

    A silence trim can leave a header-only MP3 (about 226 bytes, no audio frames) that is
    not zero bytes, so a size check alone misses it and the export later fails on that file.
    """
    try:
        return _probe_audio_duration(ffprobe, path) > 0.0
    except (subprocess.CalledProcessError, ValueError, OSError):
        return False



def _probe_audio_duration(ffprobe: str, path: str) -> float:
    result = subprocess.run(
        [ffprobe, '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip() or '0')


def _fit_audio_speed(path: str, speed: float) -> str | None:
    """Speed a dub clip up with ffmpeg atempo; returns the fitted file path.

    Idempotent per (source clip, speed) — repeated exports reuse the file.
    """
    try:
        ffmpeg = find_media_tool('ffmpeg')
        if not ffmpeg or speed <= 1.0 or speed > 2.0:
            return None
        source = Path(path)
        fitted_dir = source.parent / 'fitted'
        fitted_dir.mkdir(parents=True, exist_ok=True)
        target = fitted_dir / f'{source.stem}_{int(round(speed * 100)):03d}{source.suffix}'
        if not (target.is_file() and target.stat().st_size > 0):
            subprocess.run(
                [ffmpeg, '-y', '-i', path, '-filter:a', f'atempo={speed:.4f}',
                 '-c:a', 'libmp3lame', '-q:a', '4', str(target)],
                check=True, capture_output=True, text=True,
            )
        return str(target)
    except Exception:  # noqa: BLE001 - an unfittable clip just plays at its own pace
        return None


def _media_duration_seconds(project: dict[str, Any]) -> float | None:
    """Parse the project's media duration ('HH:MM:SS') into seconds."""
    try:
        hours, minutes, seconds = (int(part) for part in str(project.get('duration') or '').split(':'))
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None


def _preferred_start(seg: dict[str, Any]) -> float:
    if seg.get('source_start') is not None:
        return float(seg['source_start'])
    return float(seg.get('start') or 0)


def _strip_clip_silence(ffmpeg: str, ffprobe: str, path: str) -> str:
    """Trim leading and trailing dead silence from dub speech with subtle ~40ms margin.

    Returns the path to the trimmed audio file (idempotent, stored in parent/'clean').
    """
    try:
        source = Path(path)
        if not source.is_file():
            return path
        clean_dir = source.parent / 'clean'
        clean_dir.mkdir(parents=True, exist_ok=True)
        target = clean_dir / source.name
        if not (target.is_file() and target.stat().st_size > 0):
            cmd = [
                ffmpeg, '-y', '-i', str(source),
                '-af', 'silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse,silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.04,areverse',
                '-c:a', 'libmp3lame', '-q:a', '4', str(target),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
                return path
            # A fully-silent clip trims to a header-only file that is not zero bytes; keep the original.
            if not _is_readable_audio(ffprobe, str(target)):
                target.unlink(missing_ok=True)
                return path
        return str(target)
    except Exception:
        return path


def _resolve_speech_overlaps(audio_segments: list[dict[str, Any]],
                             video_duration: float | None = None) -> list[dict[str, Any]]:
    """Fit every dubbed line into the movie: in full, in sync, not overlapping.

    Trims dead TTS silence padding, then fits each line into the room before
    the next cue (or video duration) using dynamic tempo (up to _MAX_DUB_TEMPO)
    so dub audio stays aligned and never outgrows the video duration.
    """
    ffprobe = find_media_tool('ffprobe')
    ffmpeg = find_media_tool('ffmpeg')
    if not ffprobe or not audio_segments:
        return audio_segments

    # First pass: strip dead silence padding from each clip if ffmpeg is available
    if ffmpeg:
        for seg in audio_segments:
            path = str(seg.get('path') or '')
            if path and Path(path).is_file():
                cleaned = _strip_clip_silence(ffmpeg, ffprobe, path)
                if cleaned != path:
                    if not seg.get('source_path'):
                        seg['source_path'] = path
                    seg['path'] = cleaned

    indexed = list(enumerate(audio_segments))
    indexed.sort(key=lambda item: (_preferred_start(item[1]), item[0]))
    cues = [_preferred_start(seg) for _i, seg in indexed]
    next_bounds = cues[1:] + [video_duration if video_duration else float('inf')]

    # Pre-probe all cleaned clip durations
    durations: list[float] = []
    for _original_index, seg in indexed:
        path = str(seg.get('path') or '')
        try:
            dur = _probe_audio_duration(ffprobe, path) if path and Path(path).is_file() else 0.0
        except Exception:
            dur = max(0.0, float(seg.get('end') or 0) - float(seg.get('start') or 0))
        durations.append(dur)

    cursor = 0.0
    for position, (_original_index, seg) in enumerate(indexed):
        path = str(seg.get('path') or '')
        if not path or not Path(path).is_file():
            continue
        preferred = _preferred_start(seg)
        seg['source_start'] = preferred
        start = max(preferred, cursor)
        duration = durations[position]

        slot = next_bounds[position] - start - DUB_SPEECH_GAP
        speed = 1.0
        if slot > 0 and duration > slot:
            speed = min(duration / slot, _MAX_DUB_TEMPO)
        elif slot <= 0:
            speed = _MAX_DUB_TEMPO

        # If video_duration is set, ensure remaining lines do not overflow the video duration
        if video_duration and video_duration > start:
            rem_time = max(0.1, video_duration - start)
            rem_needed = sum(durations[position:]) + (len(indexed) - 1 - position) * DUB_SPEECH_GAP
            if rem_needed > rem_time:
                speed = max(speed, min(rem_needed / rem_time, _MAX_DUB_TEMPO))

        if speed > 1.02:
            fitted = _fit_audio_speed(path, speed)
            if fitted:
                if not seg.get('source_path'):
                    seg['source_path'] = path
                seg['path'] = fitted
                seg['speed'] = round(speed, 3)
                duration = duration / speed

        start = max(preferred, cursor)
        if position == len(indexed) - 1 and video_duration:
            # The final line ends the soundtrack: keep it inside the movie
            # unless that would collide with the line before it.
            latest_fit = max(0.0, video_duration - duration)
            if start > latest_fit:
                start = max(cursor, latest_fit)

        seg['start'] = round(start, 3)
        seg['end'] = round(start + max(duration, 0.05), 3)
        cursor = float(seg['end']) + DUB_SPEECH_GAP

    return audio_segments


@app.post('/api/projects/{project_id}/fix-overlaps')
async def fix_overlaps(project_id: str) -> dict[str, Any]:
    """Detect TTS speech overlaps and compute proposals to fix them."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    ffprobe = find_media_tool('ffprobe')
    if not ffprobe:
        raise HTTPException(status_code=503, detail='ffprobe must be installed and on PATH')

    audio_segs = sorted(project.get('audio_segments', []), key=lambda s: float(s.get('start') or 0))
    proposals: list[dict[str, Any]] = []

    for i, seg in enumerate(audio_segs):
        seg_start = float(seg.get('start') or 0)
        seg_end   = float(seg.get('end')   or 0)
        slot_dur  = seg_end - seg_start
        if slot_dur <= 0:
            continue

        audio_path = seg.get('path') or ''
        if not audio_path or not Path(audio_path).is_file():
            continue
        try:
            actual_dur = await run_in_threadpool(_probe_audio_duration, ffprobe, audio_path)
        except Exception:
            continue

        next_seg = audio_segs[i + 1] if i + 1 < len(audio_segs) else None
        overlap_with_next = next_seg is not None and float(next_seg.get('start') or 0) < seg_end
        audio_overflows   = actual_dur > slot_dur + 0.05

        if not overlap_with_next and not audio_overflows:
            continue

        # a) Speed adjust: audio overflow <= 30 % of slot
        if actual_dur > slot_dur and actual_dur <= slot_dur * 1.30:
            new_speed = round(min(1.5, max(0.75, actual_dur / slot_dur)), 4)
            temp_path = str(Path(audio_path).parent / f'fix_temp_{seg["id"]}.mp3')
            segments_list = project.get('segments', [])
            transcript_seg = next(
                (s for s in segments_list if str(s.get('id')) == str(seg.get('id'))), None
            )
            text = seg.get('text') or (transcript_seg.get('translation') if transcript_seg else '') or ''
            char_name = str(seg.get('character') or seg.get('voice') or 'Speaker 1')
            project_characters = project.get('characters') or {}
            profile = project_characters.get(char_name) or {}
            try:
                await run_in_threadpool(
                    tts_service.synthesize_line,
                    text, temp_path,
                    provider_pref=segment_tts.engine_of_clip(seg),
                    language_name=project.get('target_language', 'Khmer'),
                    lang_code=_resolve_lang_code(project.get('target_language', 'Khmer')),
                    voice_pref=profile.get('voice') or seg.get('voice'),
                    model_pref=None,
                    speed=new_speed,
                    emotion='Neutral',
                    intensity=70,
                    character_profile=profile,
                    analysis={'emotion': 'Neutral', 'intensity': 70,
                              'style': 'Natural', 'pacing': 'Natural',
                              'pause_before': False, 'analyzer': 'none'},
                )
                new_dur = await run_in_threadpool(_probe_audio_duration, ffprobe, temp_path)
                if new_dur <= slot_dur + 0.05:
                    proposals.append({
                        'segment_id': seg['id'],
                        'type': 'speed_adjust',
                        'old_start': seg_start,
                        'old_end': seg_end,
                        'old_speed': 1.0,
                        'new_start': seg_start,
                        'new_end': seg_end,
                        'new_speed': new_speed,
                        'new_audio_path': temp_path,
                        'description': f'Re-synthesize at {new_speed}x speed to fit {slot_dur:.2f}s slot',
                    })
                    continue
            except Exception:
                pass  # fall through to silence_absorb

        # b) Silence absorption: next segment has a gap >= 0.3 s
        if next_seg is not None:
            next_start = float(next_seg.get('start') or 0)
            gap = next_start - seg_end
            if gap >= 0.3:
                shift = min(gap, actual_dur - slot_dur)
                shift = max(0.0, shift)
                proposals.append({
                    'segment_id': next_seg['id'],
                    'type': 'silence_absorb',
                    'old_start': next_start,
                    'old_end': float(next_seg.get('end') or 0),
                    'old_speed': 1.0,
                    'new_start': round(next_start - shift, 3),
                    'new_end': float(next_seg.get('end') or 0),
                    'new_speed': 1.0,
                    'new_audio_path': None,
                    'description': f'Shift segment {next_seg["id"]} earlier by {shift:.2f}s to absorb overflow',
                })
                continue

        # c) Last resort: flag for re-translation
        target_chars = int(len(str(seg.get('text') or '')) * (slot_dur / max(actual_dur, 0.1)))
        proposals.append({
            'segment_id': seg['id'],
            'type': 'needs_retranslation',
            'old_start': seg_start,
            'old_end': seg_end,
            'old_speed': 1.0,
            'new_start': seg_start,
            'new_end': seg_end,
            'new_speed': 1.0,
            'new_audio_path': None,
            'description': f'Audio ({actual_dur:.2f}s) too long for slot ({slot_dur:.2f}s); shorten translation',
        })

    return {'overlaps_found': len(proposals), 'proposals': proposals}


@app.post('/api/projects/{project_id}/fix-overlaps/apply')
async def apply_fix_overlaps(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Apply approved overlap fix proposals to the project's audio segments."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail='Project not found')
    ffprobe = find_media_tool('ffprobe')
    proposals: list[dict[str, Any]] = payload.get('proposals') or []
    audio_segs = project.get('audio_segments', [])
    seg_by_id = {str(s.get('id')): s for s in audio_segs}

    for proposal in proposals:
        p_type = proposal.get('type')
        seg_id = str(proposal.get('segment_id') or '')
        seg = seg_by_id.get(seg_id)
        if seg is None:
            continue

        if p_type == 'speed_adjust':
            new_path = proposal.get('new_audio_path', '')
            real_path = seg.get('path', '')
            if new_path and Path(new_path).is_file() and real_path:
                shutil.move(new_path, real_path)
                if ffprobe:
                    try:
                        new_dur = await run_in_threadpool(_probe_audio_duration, ffprobe, real_path)
                        seg['end'] = round(float(seg.get('start') or 0) + new_dur, 3)
                    except Exception:
                        pass

        elif p_type == 'silence_absorb':
            new_start = proposal.get('new_start')
            if new_start is not None:
                seg['start'] = round(float(new_start), 3)

        # 'needs_retranslation': skip — no automatic action.

    persist_project(project_id)
    return {'status': 'ok', 'audio_segments': project.get('audio_segments', [])}


@app.websocket('/ws')
async def websocket_status(websocket: WebSocket) -> None:
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
