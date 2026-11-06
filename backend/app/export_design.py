"""Platform frame design: the numbers the preview uses, applied by FFmpeg.

The preview and the export must agree pixel for pixel, so both sides work from
the same normalised values (see ``frontend/src/store/useDesignStore.ts``):

- ``video.x`` / ``video.y``  centre offset, in frame widths/heights (-1..1)
- ``video.size``             video width as a fraction of the frame width
- ``video.rotation``         degrees
- ``subtitle.x`` / ``y``     0..1 inside the frame
- ``subtitle.size``          font size as a fraction of the frame height
- ``background.blur``        0..50 at a 720 px-wide frame; scaled by width/720

Everything here is a pure function so the filter graph can be tested without
running FFmpeg.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

# Target frames, identical to the preview's 9:16 / 16:9 / 1:1 boxes.
PLATFORM_FRAMES: dict[str, tuple[int, int]] = {
    'vertical': (1080, 1920),
    'standard': (1920, 1080),
    'square': (1080, 1080),
}

BACKGROUND_MODES = ('blur', 'color', 'gradient')
_HEX = re.compile(r'^#?[0-9a-fA-F]{6}$')


@dataclass(frozen=True)
class Design:
    platform: str = 'original'
    background_mode: str = 'blur'
    background_blur: float = 0.0
    background_color: str = '#0b1220'
    background_from: str = '#1e293b'
    background_to: str = '#0ea5e9'
    x: float = 0.0
    y: float = 0.0
    size: float = 1.0
    rotation: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    caption: bool = True
    caption_x: float = 0.5
    caption_y: float = 0.88
    caption_size: float = 0.06
    # 0 = the whole line in one cue; 1–12 = advance through the line in cues of
    # at most that many words.
    caption_words: int = 0
    # A blurred rectangle over the frame, hiding subtitles or logos that are
    # burned into the source video.
    blur_enabled: bool = False
    blur_x: float = 0.05
    blur_y: float = 0.78
    blur_w: float = 0.9
    blur_h: float = 0.14
    blur_strength: float = 20.0

    @property
    def has_transform(self) -> bool:
        """True when the video itself is moved, scaled, rotated or flipped."""
        return (
            abs(self.size - 1.0) > 0.001
            or abs(self.x) > 0.001
            or abs(self.y) > 0.001
            or abs(self.rotation) > 0.01
            or self.flip_h
            or self.flip_v
        )


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _colour(value: Any, default: str) -> str:
    text = str(value or '').strip()
    return text if _HEX.match(text) else default


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def normalize_design(payload: Any) -> Design | None:
    """Validate the browser's design payload; None when there is nothing to do."""
    if not isinstance(payload, dict):
        return None
    platform = str(payload.get('platform') or 'original').strip().lower()
    if platform not in ('original', *PLATFORM_FRAMES):
        platform = 'original'

    background = payload.get('background') if isinstance(payload.get('background'), dict) else {}
    mode = str(background.get('mode') or 'blur').strip().lower()
    if mode not in BACKGROUND_MODES:
        mode = 'blur'

    video = payload.get('video') if isinstance(payload.get('video'), dict) else {}
    subtitle = payload.get('subtitle') if isinstance(payload.get('subtitle'), dict) else {}
    blur = payload.get('blur_box') if isinstance(payload.get('blur_box'), dict) else {}

    return Design(
        platform=platform,
        background_mode=mode,
        background_blur=_clamp(_number(background.get('blur'), 0.0), 0.0, 50.0),
        background_color=_colour(background.get('color'), '#0b1220'),
        background_from=_colour(background.get('from'), '#1e293b'),
        background_to=_colour(background.get('to'), '#0ea5e9'),
        x=_clamp(_number(video.get('x'), 0.0), -1.0, 1.0),
        y=_clamp(_number(video.get('y'), 0.0), -1.0, 1.0),
        size=_clamp(_number(video.get('size'), 1.0), 0.05, 1.0),
        rotation=_clamp(_number(video.get('rotation'), 0.0), -180.0, 180.0),
        flip_h=_flag(video.get('flipH')),
        flip_v=_flag(video.get('flipV')),
        caption=_flag(subtitle.get('enabled', True)),
        caption_x=_clamp(_number(subtitle.get('x'), 0.5), 0.0, 1.0),
        caption_y=_clamp(_number(subtitle.get('y'), 0.88), 0.0, 1.0),
        caption_size=_clamp(_number(subtitle.get('size'), 0.06), 0.02, 0.2),
        caption_words=int(_clamp(_number(subtitle.get('wordsPerLine'), 0), 0, 12)),
        blur_enabled=_flag(blur.get('enabled')),
        blur_x=_clamp(_number(blur.get('x'), 0.05), 0.0, 0.98),
        blur_y=_clamp(_number(blur.get('y'), 0.78), 0.0, 0.98),
        blur_w=_clamp(_number(blur.get('w'), 0.9), 0.02, 1.0),
        blur_h=_clamp(_number(blur.get('h'), 0.14), 0.02, 1.0),
        blur_strength=_clamp(_number(blur.get('strength'), 20.0), 0.0, 50.0),
    )


def _even(value: float) -> int:
    """Round to the nearest even pixel; libx264 rejects odd dimensions."""
    number = max(2, int(round(value)))
    return number if number % 2 == 0 else number - 1


def frame_filter_graph(design: Design, width: int, height: int, *,
                       caption_file: str | None = None) -> tuple[str, str]:
    """Filter graph turning ``[0:v]`` into the designed frame at width×height.

    Returns (filter_complex, output_label). The graph ends with the caption burn
    when ``caption_file`` is given, so subtitles are placed on the final frame in
    its own coordinates instead of being baked into the source first.
    """
    foreground_width = _even(design.size * width)
    chains: list[str] = []

    if design.background_mode == 'blur':
        # The bars are the clip itself, blurred and cropped to fill.
        radius = int(round(design.background_blur * width / 720.0))
        blur = f',boxblur=luma_radius={radius}:luma_power=2' if radius > 0 else ''
        chains.append('[0:v]split=2[bg][fg]')
        chains.append(
            f'[bg]scale={width}:{height}:force_original_aspect_ratio=increase,'
            f'crop={width}:{height}{blur}[bgd]'
        )
    elif design.background_mode == 'color':
        chains.append('[0:v]null[fg]')
        chains.append(f'color=c=0x{design.background_color.lstrip("#")}:s={width}x{height}[bgd]')
    else:
        chains.append('[0:v]null[fg]')
        chains.append(
            f'gradients=s={width}x{height}:c0=0x{design.background_from.lstrip("#")}:'
            f'c1=0x{design.background_to.lstrip("#")}:x0=0:y0=0:x1={width}:y1={height}[bgd]'
        )

    foreground = [f'scale={foreground_width}:-2']
    if design.flip_h:
        foreground.append('hflip')
    if design.flip_v:
        foreground.append('vflip')
    if abs(design.rotation) > 0.01:
        radians = math.radians(design.rotation)
        foreground.append(
            f'rotate={radians:.6f}:c=none:ow=rotw({radians:.6f}):oh=roth({radians:.6f})'
        )
    chains.append(f'[fg]{",".join(foreground)}[fgs]')

    # Same mapping as the preview: half a frame width per unit of x.
    offset_x = int(round(design.x * width * 0.5))
    offset_y = int(round(design.y * height * 0.5))
    chains.append(
        f'[bgd][fgs]overlay=x=(W-w)/2+{offset_x}:y=(H-h)/2+{offset_y}:shortest=1,'
        f'setsar=1[vframe]'
    )

    final = '[vframe]'
    # Blur box: crop the region, blur it and lay it back over the same spot. It
    # goes after the video is placed (so it covers the source's own subtitles)
    # and before the caption (so our text stays sharp).
    if design.blur_enabled and design.blur_strength > 0:
        box_x = int(round(_clamp(design.blur_x, 0.0, 1.0) * width))
        box_y = int(round(_clamp(design.blur_y, 0.0, 1.0) * height))
        box_w = max(2, int(round(_clamp(design.blur_w, 0.0, 1.0) * width)))
        box_h = max(2, int(round(_clamp(design.blur_h, 0.0, 1.0) * height)))
        box_w = min(box_w, width - box_x) if box_x < width else width
        box_h = min(box_h, height - box_y) if box_y < height else height
        radius = int(round(design.blur_strength * width / 720.0))
        if radius > 0 and box_w >= 2 and box_h >= 2:
            chains.append(f'{final}split=2[bmain][bpatch]')
            chains.append(
                f'[bpatch]crop={box_w}:{box_h}:{box_x}:{box_y},'
                f'boxblur=luma_radius={radius}:luma_power=2[bblur]'
            )
            chains.append(f'[bmain][bblur]overlay={box_x}:{box_y}[vbox]')
            final = '[vbox]'

    chains.append(
        f'{final}format=yuv420p'
        + (f",subtitles='{caption_file}'" if caption_file else '')
        + '[v]'
    )
    return ';'.join(chains), '[v]'


def _ass_time(seconds: float) -> str:
    total = max(0.0, float(seconds or 0.0))
    hours = int(total // 3600)
    minutes = int((total - hours * 3600) // 60)
    secs = total - hours * 3600 - minutes * 60
    return f'{hours}:{minutes:02d}:{secs:05.2f}'


def _ass_text(value: Any) -> str:
    text = str(value or '').strip()
    text = text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
    return text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ') or ' '


def _split_cue_words(text: str, words: int) -> list[str]:
    """The cue's text cut into pieces of at most `words` words (0 = one piece)."""
    body = str(text or '').strip()
    if not body:
        return []
    if words <= 0:
        return [body]
    parts = body.split()
    return [' '.join(parts[index:index + words]) for index in range(0, len(parts), words)]


def caption_ass(segments: Iterable[dict[str, Any]], width: int, height: int,
                design: Design) -> str:
    """An ASS subtitle file placed exactly where the preview draws the caption.

    PlayRes is the final frame, so ``\\pos`` and the font size are the same
    normalised numbers the preview uses — no per-platform recolouring.
    """
    font_size = max(8, int(round(design.caption_size * height)))
    pos_x = int(round(design.caption_x * width))
    pos_y = int(round(design.caption_y * height))
    outline = max(1, int(round(height / 360)))

    lines = [
        '[Script Info]',
        'ScriptType: v4.00+',
        f'PlayResX: {width}',
        f'PlayResY: {height}',
        'WrapStyle: 2',
        'ScaledBorderAndShadow: yes',
        '',
        '[V4+ Styles]',
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, '
        'BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, '
        'BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
        f'Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,'
        f'1,0,0,0,100,100,0,0,1,{outline},0,5,0,0,0,1',
        '',
        '[Events]',
        'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    ]
    for segment in segments:
        text = segment.get('text') or segment.get('translation') or ''
        if not str(text).strip():
            continue
        start = float(segment.get('start') or 0)
        end = float(segment.get('end') or 0)
        chunks = _split_cue_words(text, design.caption_words)
        if not chunks:
            continue
        # One dialogue per word chunk, spread evenly over the cue's time, so the
        # caption advances through the line while it plays — the same timing the
        # preview uses (chunk index = elapsed / duration * chunk count).
        span = max(0.01, end - start) / len(chunks)
        for index, chunk in enumerate(chunks):
            lines.append(
                f'Dialogue: 0,{_ass_time(start + index * span)},{_ass_time(start + (index + 1) * span)},'
                f'Default,,0,0,0,,{{\\pos({pos_x},{pos_y})}}{_ass_text(chunk)}'
            )
    return '\n'.join(lines) + '\n'
