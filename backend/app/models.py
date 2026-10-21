from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderMode = Literal['LOCAL AI', 'FREE ONLINE', 'PAID API']


class ProviderSettings(BaseModel):
    id: str
    name: str
    mode: ProviderMode
    connected: bool = True
    primary: bool = False
    fallback: bool = False


class ProjectRequest(BaseModel):
    name: str = Field(..., min_length=1)
    mode: Literal['EDIT', 'DUB'] = 'DUB'
    target_language: str = 'Khmer'


class ProjectSummary(BaseModel):
    id: str
    name: str
    mode: str
    status: str = 'Ready'
    progress: int = 0
    duration: str = '00:00:00'
    has_media: bool = False
    media_name: str | None = None
    media_resolution: str | None = None
    media_size: int | None = None


class WorkflowProgress(BaseModel):
    project_id: str
    stage: str
    progress: int
    message: str
    engine: str | None = None


class ProcessingRequest(BaseModel):
    project_id: str
    workflow: str = 'transcribe'
    target_language: str = 'Khmer'
    voice: str = 'Male Voice A'
    include_subtitles: bool = True
    burn_subtitles: bool = False
    preserve_original_audio: bool = True
    original_audio_volume: float = Field(default=0.25, ge=0.0, le=1.0)
    # Movie-style ducking: original audio stays at full volume in gaps and dips
    # to original_audio_volume only while dubbed speech is playing.
    auto_duck: bool = True
    # Separated audio layers kept in the mix (the original dialogue layer is
    # always replaced by the dub when separation is available).
    keep_music: bool = True
    keep_sfx: bool = True
    # Video chunking: length of each part in seconds (10 s – 2 h). None clears chunking.
    chunk_seconds: float | None = Field(default=None, ge=10, le=7200)
    # Export shape: one combined MP4, one MP4 per chunked part, or both.
    export_mode: Literal['single', 'parts', 'both'] = 'single'
    # Destination folder written to directly (no zip/downloads). None = default.
    export_folder: str | None = None
    # False when the browser will deliver the files itself (user picked a folder
    # via the directory picker), so the backend skips its own folder copy.
    deliver_to_folder: bool = True
    # Transcription quality options (movie dialogue).
    language: str | None = None          # None = auto-detect; else ISO code
    beam_size: int = Field(default=5, ge=1, le=10)
    word_timestamps: bool = False
    # Translation engine: which provider goes first (the chain still falls back).
    translation_provider: str = 'auto'   # auto | qwen | nllb | argos | google | gemini | openai | codecraft
    # Platform style: convert the export to a platform aspect ratio.
    # vertical = 9:16 (TikTok/Reels/Shorts), standard = 16:9 (YouTube/FB),
    # square = 1:1 (Instagram/FB feed). 'blur' fills the frame with a blurred
    # copy of the video and centers the original; 'crop' center-crops to fill.
    platform_style: Literal['original', 'vertical', 'standard', 'square'] = 'original'
    platform_fit: Literal['blur', 'crop'] = 'blur'
    # Voice engine + acting (movie dubbing).
    tts_provider: str = 'auto'           # auto | gemini | openai | edge | voxcpm2 | voxcpm_clone | sapi5
    tts_model: str | None = None         # provider-specific TTS model
    emotion: str = 'Auto'                # Auto = analyzed per line
    intensity: int = Field(default=70, ge=0, le=100)
    acting: bool = True                  # analyze context for acting directions
    openai_acting: bool = False          # use the OpenAI API for acting analysis
    openai_review: bool = False          # OpenAI quality pass over translations
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    # Timeline editing: update a segment's text/translation/timing, or split it.
    # Video cutting: cut_mode 'remove' deletes [start, end] from the video,
    # 'keep' keeps only [start, end]. Segments/dubs shift to match.
    cut_mode: Literal['remove', 'keep'] = 'remove'
    text: str | None = None
    translation: str | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    at: float = Field(default=0, ge=0)
    text1: str = ''
    text2: str = ''
    translation1: str = ''
    translation2: str = ''


class SegmentUpdate(BaseModel):
    """PATCH /api/projects/{id}/segments/{sid} — all fields optional."""
    text: str | None = None
    translation: str | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    gender: Literal['Male', 'Female'] | None = None


class SegmentSplit(BaseModel):
    """POST /api/projects/{id}/segments/{sid}/split — time + both texts."""
    at: float = Field(..., ge=0)
    text1: str = ''
    text2: str = ''
    translation1: str = ''
    translation2: str = ''
