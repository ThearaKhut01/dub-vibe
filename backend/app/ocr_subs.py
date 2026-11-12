"""Read burned-in subtitles from one frame inside a user-drawn rectangle.

The preview lets the user drag an OCR box over the video (the same normalised
0..1 space as the blur box, see ``frontend/src/store/useDesignStore.ts``). This
module turns that rectangle into a real frame crop with FFmpeg and reads it with
PaddleOCR, so the rectangle in the preview is exactly the rectangle the engine
sees — no guessing at a subtitle band.

PaddleOCR is optional. The app must still start, and the blur box must still
work, when it is not installed, so every entry point reports that as
``OCRUnavailableError`` instead of failing at import time.
"""
from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .export_design import box_to_pixels
from .providers import _resolve_lang_code

INSTALL_HINT = (
    'OCR needs PaddleOCR, which is not installed yet. Run '
    '".venv\\Scripts\\python.exe -m pip install paddlepaddle paddleocr", '
    'then restart the backend.'
)

# PaddleOCR 3.x names the engine by a language code, not by a script: it picks
# the model itself (PP-OCRv6 for Chinese/English/Japanese and the Latin
# languages, PP-OCRv5 for Thai/Korean/Arabic/Cyrillic/Devanagari). Passing a
# script name such as "latin" raises "No models are available for lang='latin'",
# so these values must stay real language codes.
OCR_LANGUAGE_MODELS: dict[str, str] = {
    'zh': 'ch',
    'en': 'en',
    'fr': 'fr',
    'es': 'es',
    'de': 'de',
    'pt': 'pt',
    'vi': 'vi',
    'id': 'id',
    'ja': 'japan',
    'ko': 'korean',
    'th': 'th',
    'ar': 'ar',
    'ru': 'ru',
    'hi': 'hi',
}

# PaddleOCR answers with a confidence per line. Below this the read is usually a
# texture or a logo rather than a subtitle, and a wrong line is worse than none.
DEFAULT_MIN_SCORE = 0.5


class OCRUnavailableError(RuntimeError):
    """The OCR engine is not installed (or cannot run) on this machine."""


class OCRFailedError(RuntimeError):
    """The engine ran but could not read the frame."""


def ocr_available() -> bool:
    """True when the PaddleOCR package is importable."""
    return importlib.util.find_spec('paddleocr') is not None


def ocr_language(source_language: str) -> str:
    """PaddleOCR ``lang`` value for the project's source language.

    The app stores human names ("Chinese", "Khmer") in most places and ISO
    codes in others, so resolve through the same table the translation
    providers use. Unknown or empty languages fall back to English, which
    PP-OCRv6 still reads as ASCII.
    """
    code = _resolve_lang_code(str(source_language or '').strip())
    # English is the safe fallback: PP-OCRv6 still reads ASCII, and a language
    # PaddleOCR has no model for (Khmer, for example) must not crash the call.
    return OCR_LANGUAGE_MODELS.get(code, 'en')


def normalize_box(box: Any) -> dict[str, float] | None:
    """Clamp the browser's rectangle to 0..1 and keep it inside the frame.

    The preview already clamps while dragging, but the request is the trust
    boundary: an out-of-range box must never become a negative crop.
    """
    if not isinstance(box, dict):
        return None

    def number(key: str, default: float) -> float:
        try:
            value = float(box.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    x = min(max(number('x', 0.05), 0.0), 0.98)
    y = min(max(number('y', 0.7), 0.0), 0.98)
    w = min(max(number('w', 0.9), 0.02), 1.0 - x)
    h = min(max(number('h', 0.22), 0.02), 1.0 - y)
    return {'x': round(x, 4), 'y': round(y, 4), 'w': round(w, 4), 'h': round(h, 4)}


def _tool(name: str) -> str:
    """Locate ffmpeg/ffprobe on PATH, else through the app's own finder."""
    found = shutil.which(name)
    if found:
        return found
    from .main import find_media_tool

    found = find_media_tool(name)
    if not found:
        raise OCRFailedError(f'{name} was not found. Install FFmpeg to read frames.')
    return found


def video_size(media_path: str) -> tuple[int, int]:
    """Pixel size of the video's first video stream."""
    result = subprocess.run(
        [_tool('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(media_path)],
        capture_output=True, text=True, check=True,
    )
    try:
        width, height = (int(part) for part in result.stdout.strip().split(',')[:2])
    except (ValueError, TypeError) as error:
        raise OCRFailedError(f'Could not read the video size: {result.stdout!r}') from error
    return width, height


def grab_frame(media_path: str, time: float, box: dict[str, float],
               out_path: str | Path) -> dict[str, Any]:
    """Write one cropped frame to `out_path`; returns the pixel rectangle used."""
    width, height = video_size(str(media_path))
    x, y, w, h = box_to_pixels(
        float(box.get('x', 0.05)), float(box.get('y', 0.7)),
        float(box.get('w', 0.9)), float(box.get('h', 0.22)),
        width, height,
    )
    subprocess.run(
        [_tool('ffmpeg'), '-v', 'error', '-y',
         '-ss', f'{max(0.0, float(time)):.3f}', '-i', str(media_path),
         '-frames:v', '1', '-vf', f'crop={w}:{h}:{x}:{y}', str(out_path)],
        capture_output=True, check=True,
    )
    if not Path(out_path).is_file() or Path(out_path).stat().st_size == 0:
        raise OCRFailedError('FFmpeg produced no frame — is the time inside the video?')
    return {'x': x, 'y': y, 'w': w, 'h': h, 'frame_width': width, 'frame_height': height}


_ENGINES: dict[str, Any] = {}


def _engine(model_language: str):
    """One PaddleOCR instance per script, built on first use (import is slow)."""
    engine = _ENGINES.get(model_language)
    if engine is not None:
        return engine
    try:
        from paddleocr import PaddleOCR
    except Exception as error:  # noqa: BLE001 - any import failure means "not installed"
        raise OCRUnavailableError(INSTALL_HINT) from error
    try:
        engine = PaddleOCR(
            lang=model_language,
            device='cpu',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # PaddlePaddle 3.x on Windows crashes inside its oneDNN/PIR path
            # ("ConvertPirAttribute2RuntimeAttribute not support"), so run the
            # plain CPU kernels. Slower per frame, but it actually returns text.
            enable_mkldnn=False,
        )
    except TypeError:
        # PaddleOCR 2.x has none of the 3.x keyword arguments.
        engine = PaddleOCR(
            lang=model_language, use_angle_cls=False, show_log=False, enable_mkldnn=False,
        )
    _ENGINES[model_language] = engine
    return engine


def _page_fields(page: Any) -> dict[str, Any]:
    """Normalise one PaddleOCR result page (3.x dict-like, 2.x nested lists)."""
    for candidate in (page, getattr(page, 'json', None)):
        if isinstance(candidate, dict):
            data = candidate.get('res', candidate)
            if isinstance(data, dict) and 'rec_texts' in data:
                return data
    try:
        return {'rec_texts': list(page['rec_texts']), 'rec_scores': list(page['rec_scores'])}
    except Exception:  # noqa: BLE001 - fall through to the 2.x shape
        pass
    texts: list[str] = []
    scores: list[float] = []
    for line in page or []:
        try:
            text, score = line[1][0], line[1][1]
        except (IndexError, TypeError):
            continue
        texts.append(text)
        scores.append(score)
    return {'rec_texts': texts, 'rec_scores': scores}


def _predict(engine: Any, image: Any) -> Any:
    """Run one image (path or BGR ndarray) through PaddleOCR 3.x or 2.x."""
    try:
        return engine.predict(image) if hasattr(engine, 'predict') else engine.ocr(image, cls=False)
    except Exception as error:  # noqa: BLE001 - engine/runtime failure, not a code bug
        raise OCRFailedError(f'PaddleOCR could not read the frame: {error}') from error


def _lines_from_pages(pages: Any, min_score: float) -> list[dict[str, Any]]:
    """Flatten PaddleOCR result pages into text lines above the score floor."""
    lines: list[dict[str, Any]] = []
    for page in pages or []:
        fields = _page_fields(page)
        for text, score in zip(fields.get('rec_texts') or [], fields.get('rec_scores') or []):
            body = str(text or '').strip()
            if not body:
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence >= min_score:
                lines.append({'text': body, 'score': round(confidence, 4)})
    return lines


def recognize_image(image_path: str | Path, model_language: str,
                    min_score: float = DEFAULT_MIN_SCORE) -> list[dict[str, Any]]:
    """Read every text line in an image, dropping low-confidence noise."""
    engine = _engine(model_language)
    return _lines_from_pages(_predict(engine, str(image_path)), min_score)


def read_frame(media_path: str, time: float, box: dict[str, float],
               source_language: str = '', min_score: float = DEFAULT_MIN_SCORE) -> dict[str, Any]:
    """Crop one frame with the user's box and read it. The OCR preview call."""
    if not ocr_available():
        raise OCRUnavailableError(INSTALL_HINT)
    model_language = ocr_language(source_language)
    with tempfile.TemporaryDirectory(prefix='dubvibe-ocr-') as folder:
        image = Path(folder) / 'frame.png'
        crop = grab_frame(media_path, time, box, image)
        lines = recognize_image(image, model_language, min_score)
    return {
        'text': '\n'.join(line['text'] for line in lines),
        'score': max((line['score'] for line in lines), default=0.0),
        'lines': lines,
        'crop': crop,
        'language': model_language,
    }


def video_duration(media_path: str) -> float:
    """Seconds of the container, which is the range the cue times live in."""
    result = subprocess.run(
        [_tool('ffprobe'), '-v', 'error',
         '-show_entries', 'format=duration', '-of', 'csv=p=0', str(media_path)],
        capture_output=True, text=True, check=True,
    )
    try:
        return max(0.0, float(result.stdout.strip()))
    except (ValueError, TypeError) as error:
        raise OCRFailedError(f'Could not read the video duration: {result.stdout!r}') from error


def _clean_text(lines: list[dict[str, Any]]) -> str:
    """One subtitle line: join the frame's boxes, collapse spacing.

    PaddleOCR returns one entry per text box. A subtitle that wraps onto two
    rows arrives as two entries, so they are joined with a space rather than a
    newline — the transcript is spoken, not typeset.
    """
    body = ' '.join(str(line['text']).strip() for line in lines)
    return ' '.join(body.split())


def extract_cues(media_path: str, box: dict[str, float], source_language: str = '',
                 *, start: float = 0.0, end: float | None = None, fps: float = 1.0,
                 min_score: float = DEFAULT_MIN_SCORE, min_duration: float = 0.3,
                 progress: Any = None) -> list[dict[str, Any]]:
    """Read the burned-in subtitle line across the video and return transcript cues.

    Frames are streamed from FFmpeg as raw BGR, cropped with the user's box, and
    read one at a time. Consecutive frames with the same text become one cue, so
    a line that stays on screen for three seconds is one segment, not three.

    ``progress(done, total)`` is called every few frames so the caller can put a
    percentage on the workflow socket.
    """
    if not ocr_available():
        raise OCRUnavailableError(INSTALL_HINT)

    import numpy as np  # only needed by this path; keep the import cheap

    width, height = video_size(str(media_path))
    box_x, box_y, box_w, box_h = box_to_pixels(
        float(box.get('x', 0.05)), float(box.get('y', 0.7)),
        float(box.get('w', 0.9)), float(box.get('h', 0.22)),
        width, height,
    )
    duration = video_duration(str(media_path))
    begin = max(0.0, min(float(start or 0.0), duration))
    finish = duration if end is None else max(begin, min(float(end), duration))
    if finish - begin < 0.2:
        return []
    rate = min(max(float(fps or 1.0), 0.2), 5.0)
    total = max(1, int(math.ceil((finish - begin) * rate)))
    frame_bytes = box_w * box_h * 3

    engine = _engine(ocr_language(source_language))
    command = [
        _tool('ffmpeg'), '-v', 'error',
        '-ss', f'{begin:.3f}', '-t', f'{finish - begin:.3f}',
        '-i', str(media_path),
        '-vf', f'crop={box_w}:{box_h}:{box_x}:{box_y},fps={rate:.6f}',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-',
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    cues: list[dict[str, Any]] = []
    active_text = ''
    active_start = begin
    frame_index = 0

    def close(at: float) -> None:
        nonlocal active_text
        body = active_text
        active_text = ''
        if not body:
            return
        # A line of pure punctuation is detection noise, not a subtitle.
        if not any(character.isalnum() for character in body):
            return
        cues.append({
            'start': round(active_start, 3),
            'end': round(max(at, active_start + 0.05), 3),
            'text': body,
        })

    try:
        while True:
            raw = process.stdout.read(frame_bytes) if process.stdout else b''
            if len(raw) < frame_bytes:
                break
            time = begin + frame_index / rate
            frame_index += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(box_h, box_w, 3).copy()
            text = _clean_text(_lines_from_pages(_predict(engine, frame), min_score))
            if text != active_text:
                close(time)
                if text:
                    active_text = text
                    active_start = time
            if progress is not None and frame_index % 10 == 0:
                progress(frame_index, total)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        process.wait()

    close(min(finish, begin + frame_index / rate))

    # Drop flashes and OCR noise, then re-join a line split by one bad frame.
    kept: list[dict[str, Any]] = []
    for cue in cues:
        if cue['end'] - cue['start'] < min_duration:
            continue
        previous = kept[-1] if kept else None
        if previous and previous['text'] == cue['text'] and cue['start'] - previous['end'] < 0.25:
            previous['end'] = cue['end']
        else:
            kept.append(cue)
    return kept
