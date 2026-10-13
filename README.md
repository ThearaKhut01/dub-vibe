# DubVibe Studio

DubVibe is a local-first AI video dubbing studio for Windows. It combines a dark timeline video editor UI, a FastAPI backend for orchestration, and a modular AI provider layer for transcription, translation, and speech synthesis — built around smart chunk processing so long movies dub fast, reliably, and with the original sound intact.

## Architecture

- Frontend: React + TypeScript + Vite + Tailwind CSS (Zustand state)
- Backend: FastAPI with WebSocket live progress and SQLite project storage
- Media engine: FFmpeg/FFprobe (staged audio mixing, per-part renders, sidechain ducking)
- Desktop shell: Tauri (NSIS installer, bundled Python backend sidecar + FFmpeg)
- AI layer: interchangeable providers with automatic fallback —
  Faster-Whisper (local) → Qwen3 4B (local translation) → Argos Translate (offline) → Google Translate (online, retried) → Gemini TTS / OpenAI TTS (online, keyed) → Edge TTS (online, retried) → VoxCPM2 (offline, English, optional) → Khmer MMS-VITS (offline, Khmer only) → SAPI5 (offline).
  API keys live in Windows Credential Manager (never in the browser); AI acting analysis reads the scene and steers supported TTS engines; paid TTS results are cached.

## Quick start (development)

Requirements: Node.js, Python 3.10, and FFmpeg/FFprobe on `PATH`. Translation and Edge TTS need internet; Faster-Whisper downloads its model on first transcription.

1. Install dependencies:
   `npm install` and `npm --prefix frontend install`
2. Create DubVibe's isolated Python environment:
   `npm run setup:backend`
3. Run the dev stack:
   `npm run dev`
4. Open the studio at http://localhost:5173 (API docs at http://localhost:8001/docs).

## Building the Windows app

```bat
cd C:\dub-vibe
npm run build:sidecar
npm run tauri:build
```

`build:sidecar` packages the whole Python backend plus FFmpeg into a single exe; `tauri:build` compiles the frontend and shell and produces the installer at `src-tauri\target\release\bundle\nsis\DubVibe Studio_0.1.0_x64-setup.exe`. The installed app needs no Python, Node, or separately installed FFmpeg.

## Dubbing workflow

- **Import** uploads the video into the local project and reads its duration/resolution with FFprobe.
- **Chunk** (optional) splits long videos into fixed-length parts (1–15 minutes) with a fast FFmpeg stream copy. Pick the part length in the Inspector, or just press START DUBBING and it splits automatically. Part chips show which parts are already dubbed, and part markers appear on the timeline. Transcription runs part by part while every timestamp stays on the original timeline.
- **Transcribe** runs Faster-Whisper locally (`base` model by default) and stores speech segments with timestamps.
- **Translate** uses offline Argos Translate when the language pair is available, otherwise the Google Translate endpoint — each segment retried with backoff. Finished translations are saved as they complete, so a failed run resumes where it stopped instead of starting over.
- **Dub** synthesizes each translated segment with Edge TTS (offline SAPI5 fallback), also resumable, with progress reported per segment and per part.
- **Audio separation** (optional, CapCut-style): splits the original audio into dialogue, music, and SFX/ambience stems (Demucs, runs locally). After separation, exports replace the original voices with the dub while the effects and music bed stays intact — and the auto-duck dips the bed under dub lines. Bundled into the desktop build.
- **Export** renders the final deliverables:
  - *One combined video*, *separate parts* (Part 1…N, each a standalone dubbed MP4), or both.
  - Files are written **directly into a selected folder** — default `Videos\DubVibe\<project name>` — no zip, no browser downloads. A folder picker button makes choosing effortless.
  - **Auto-duck (movie feel):** the original audio plays at full volume everywhere except while a dubbed line is speaking, where it dips to the slider level and fades back — action scenes keep their punch, dialogue stays clear. Turn it off for the classic constant-level mix.
  - Optional WebVTT subtitles, optionally burned into the MP4.

While a workflow runs, progress streams live over WebSocket into the workflow panel and the processing log (timestamps, per-part/per-segment messages, errors). The preview player plays the dubbed segments in sync with the video, and once exported, the dubbed MP4 plays in the Dubbed preview card.

## Projects

Projects (media, transcripts, translations, dub audio, chunk and export state) persist in a local SQLite database and restore when the app reopens. The sidebar lists every project — click to switch, trash to delete (media and exports included). Each export's destination folder is remembered.

## VoxCPM Mode

The third header tab, **VOXCPM MODE**, is a standalone voice workspace for VoxCPM2. It needs no video and never changes a Dubbing project's timeline. It offers four generation modes:

- **Text to Speech:** speaks the text in the default VoxCPM2 voice.
- **Voice Design:** creates a voice from a plain-English description (the app adds the `(description)text` form the model expects).
- **Voice Cloning:** speaks the text in the voice of one reference recording. Use only a voice you are authorized to use.
- **Ultimate Cloning:** continues from a reference recording and requires its exact transcript.

Each job reports progress and can be cancelled. Finished audio plays in the workspace, exports as a 16-bit WAV at 48 kHz, or can be sent to a Dubbing project as a standalone clip in its `voxcpm-clips` folder. Each generation runs in its own process, so the model's memory returns to the OS when the job ends. Generation is refused with a clear message when the machine lacks the free memory it needs (about 7 GB of RAM on CPU, or about 6 GB of VRAM on CUDA).

Generation is verified for English. Speed on CPU is roughly 40–80 seconds per short line; an NVIDIA GPU with a CUDA build of torch is much faster.

**GPU Acceleration Pack (NVIDIA CUDA 12):** the pack speeds up Faster-Whisper transcription only. VoxCPM uses the GPU when a CUDA build of torch is installed. MMS, NLLB, and Demucs run on CPU.

## AI Model Manager

All downloaded AI models live in one shared store — `%LOCALAPPDATA%\DubVibe Studio\models` — used identically by the local web UI and the packaged Windows app. The backend is the single source of truth: it discovers, downloads (with live progress), validates, loads, unloads, and deletes models; the browser never touches the filesystem.

Open **AI Models** in the header to manage Faster-Whisper transcription models — Tiny, Base, Small, **Medium**, Large-v3, and Large-v3-Turbo. Each model shows its status (Not installed / Downloading / Installed / Loaded / Error) with controls matching the state, and a radio selects the model used for transcription. The panel also shows detected hardware (CPU/RAM/GPU with CUDA), disk usage, and transcription settings (language, beam size). Selected model, device, and compute persist in the app settings and are used by both interfaces. Transcriptions can be cancelled mid-run, keeping the segments found so far. After the first download, transcription runs fully offline.

**VoxCPM2 offline voice (optional, English):** the `openbmb/VoxCPM2` model (2B parameters, Apache-2.0, about 4.7 GB) is stored under `models\voxcpm\VoxCPM2`. It is chosen explicitly in the TTS provider list and runs fully offline. Only English has been verified to produce intelligible speech here; Khmer and Thai came out garbled in testing, so VoxCPM2 is not offered for those languages. It runs on CPU but is slow (about 40 s per short line on a CPU-only machine); an NVIDIA GPU is recommended. Install its runtime with `npm run setup:backend`, which installs `voxcpm` without upgrading the pinned stack.

**Khmer offline voice (optional):** the fine-tuned `khmerttsopensource/khmer-tts` model (MMS-VITS, about 320 MB) is stored under `models\mms-tts\mms-tts-khm`. It is used only for Khmer, only after Edge TTS fails, and runs fully offline. Its license is **cc-by-nc-4.0 (non-commercial)**, so do not use it in commercial deployments.

## Settings

| Variable | Purpose |
| --- | --- |
| `DUBVIBE_MODELS_DIR` | Where Whisper/Argos models are stored (default `<AppRoot>\models`, falling back to `%LOCALAPPDATA%\DubVibe\models`) |
| `DUBVIBE_WHISPER_MODEL` | Whisper model size (`base` default) |
| `DUBVIBE_ALLOWED_ORIGINS` | Extra browser origins allowed to call the local API |

## Project goals

- Local-first media processing — nothing leaves the machine except translation/TTS calls
- Timeline-centric AI dubbing workflow
- Smart chunk processing for long videos ✅
- Resilient pipelines: retries, resume, and never-lose-work persistence ✅
- Movie-style audio: per-part exports, folder delivery, auto-ducking ✅
- Windows packaging via Tauri shell ✅

## Important note

DubVibe is designed as an extensible foundation for a professional desktop editor. The core architecture, UI shell, backend orchestration, and provider abstractions are in place; natural next steps include speaker diarization with per-actor voices, audio source separation for full movie-mode dubbing, and lip-sync alignment.
