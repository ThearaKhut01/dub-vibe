"""The export half of the canvas editor.

The preview and FFmpeg consume the same normalised numbers, so these tests pin
the translation: what the graph looks like, where the caption lands, and — in
the last test — that the graph actually runs in FFmpeg at the platform size.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.app import export_design, main
from backend.app.export_design import Design, caption_ass, frame_filter_graph, normalize_design


def design(*, platform: str = 'vertical', background: dict | None = None,
           video: dict | None = None, subtitle: dict | None = None,
           blur_box: dict | None = None) -> Design:
    payload = {
        'platform': platform,
        'background': {
            'mode': 'blur', 'blur': 20, 'color': '#112233', 'from': '#000000', 'to': '#ffffff',
            **(background or {}),
        },
        'video': {'x': 0, 'y': 0, 'size': 1, 'rotation': 0, 'flipH': False, 'flipV': False, **(video or {})},
        'subtitle': {'enabled': True, 'x': 0.5, 'y': 0.88, 'size': 0.06, **(subtitle or {})},
        'blur_box': {
            'enabled': False, 'x': 0.05, 'y': 0.78, 'w': 0.9, 'h': 0.14, 'strength': 20,
            **(blur_box or {}),
        },
    }
    parsed = normalize_design(payload)
    assert parsed is not None
    return parsed


def test_normalize_design_clamps_and_defaults():
    parsed = normalize_design({'platform': 'nope', 'video': {'size': 9, 'x': -4}, 'subtitle': {'y': 2}})
    assert parsed is not None
    assert parsed.platform == 'original'
    assert parsed.size == 1.0          # clamped from 9
    assert parsed.x == -1.0            # clamped from -4
    assert parsed.caption_y == 1.0     # clamped from 2
    assert normalize_design(None) is None
    assert normalize_design('nonsense') is None


def test_blur_frame_splits_scales_and_blurs_by_width():
    graph, label = frame_filter_graph(design(video={'size': 0.8}), 1080, 1920)
    assert label == '[v]'
    assert graph.startswith('[0:v]split=2[bg][fg]')
    # 20 at a 1080-wide frame is 30 px of luma radius (the slider is calibrated at 720).
    assert 'boxblur=luma_radius=30:luma_power=2' in graph
    assert 'scale=864:-2' in graph      # 0.8 * 1080
    assert 'overlay=x=(W-w)/2+0:y=(H-h)/2+0' in graph
    assert 'setsar=1[vframe]' in graph
    assert graph.endswith('[vframe]format=yuv420p[v]')


def test_offset_is_half_a_frame_per_unit():
    graph, _ = frame_filter_graph(design(video={'x': 0.5, 'y': -0.5}), 1080, 1920)
    assert 'x=(W-w)/2+270' in graph     # 0.5 * 1080 * 0.5
    assert 'y=(H-h)/2+-480' in graph    # -0.5 * 1920 * 0.5


def test_colour_and_gradient_backgrounds_replace_the_blurred_copy():
    colour, _ = frame_filter_graph(design(background={'mode': 'color', 'color': '#112233'}), 1080, 1920)
    assert 'color=c=0x112233:s=1080x1920[bgd]' in colour
    assert 'split=2' not in colour

    gradient, _ = frame_filter_graph(
        design(background={'mode': 'gradient', 'from': '#010203', 'to': '#aabbcc'}), 1080, 1920,
    )
    assert 'gradients=s=1080x1920:c0=0x010203:c1=0xaabbcc' in gradient


def test_flip_happens_before_rotation():
    graph, _ = frame_filter_graph(design(video={'flipH': True, 'rotation': 30, 'size': 1}), 1080, 1920)
    chain = graph.split('[fg]', 1)[1].split('[fgs]', 1)[0]
    assert chain.index('hflip') < chain.index('rotate=')


def test_blur_box_crops_blurs_and_lays_the_patch_back():
    graph, _ = frame_filter_graph(
        design(blur_box={'enabled': True, 'x': 0.1, 'y': 0.7, 'w': 0.8, 'h': 0.2, 'strength': 20}),
        1080, 1920,
    )
    # 0.1*1080, 0.7*1920, 0.8*1080, 0.2*1920 and a 30 px radius at this width.
    assert '[bpatch]crop=864:384:108:1344,boxblur=luma_radius=30:luma_power=2[bblur]' in graph
    assert '[bmain][bblur]overlay=108:1344[vbox]' in graph
    # The patch goes on after the frame is composed, before the caption.
    assert graph.index('[vbox]format=yuv420p') > graph.index('overlay=108:1344')


def test_blur_box_is_skipped_when_disabled_or_strength_is_zero():
    off = frame_filter_graph(design(blur_box={'enabled': False}), 1080, 1920)[0]
    assert 'bpatch' not in off
    zero = frame_filter_graph(design(blur_box={'enabled': True, 'strength': 0}), 1080, 1920)[0]
    assert 'bpatch' not in zero


def test_caption_is_burned_last_so_it_stays_on_the_final_frame():
    graph, _ = frame_filter_graph(design(), 1080, 1920, caption_file='C\\:/tmp/a.ass')
    assert graph.endswith("format=yuv420p,subtitles='C\\:/tmp/a.ass'[v]")


def test_caption_ass_uses_the_same_normalised_position():
    ass = caption_ass(
        [{'start': 1.0, 'end': 2.5, 'text': 'Hello {world}'}],
        1080, 1920, design(subtitle={'x': 0.5, 'y': 0.25, 'size': 0.1}),
    )
    assert 'PlayResX: 1080' in ass and 'PlayResY: 1920' in ass
    assert 'Style: Default,Arial,192,' in ass            # 0.1 * 1920
    assert '{\\pos(540,480)}Hello \\{world\\}' in ass     # 0.5 * 1080, 0.25 * 1920
    assert 'Dialogue: 0,0:00:01.00,0:00:02.50,Default' in ass


def test_caption_splits_a_line_into_timed_cues():
    ass = caption_ass(
        [{'start': 0, 'end': 3, 'text': 'one two three four five six'}],
        1080, 1920, design(subtitle={'wordsPerLine': 3}),
    )
    dialogues = [line for line in ass.splitlines() if line.startswith('Dialogue:')]
    assert len(dialogues) == 2
    assert dialogues[0].startswith('Dialogue: 0,0:00:00.00,0:00:01.50,') and 'one two three' in dialogues[0]
    assert dialogues[1].startswith('Dialogue: 0,0:00:01.50,0:00:03.00,') and 'four five six' in dialogues[1]


def test_caption_stays_one_cue_when_the_split_is_off():
    ass = caption_ass(
        [{'start': 0, 'end': 3, 'text': 'one two three four'}],
        1080, 1920, design(),
    )
    dialogues = [line for line in ass.splitlines() if line.startswith('Dialogue:')]
    assert len(dialogues) == 1
    assert 'one two three four' in dialogues[0]
    assert '\\N' not in dialogues[0]


def test_request_model_accepts_a_design():
    from backend.app.models import ProcessingRequest

    payload = ProcessingRequest(project_id='p', design={'platform': 'vertical', 'video': {'size': 0.5}})
    assert payload.design is not None
    assert ProcessingRequest(project_id='p').design is None


FFMPEG = shutil.which('ffmpeg')


@pytest.mark.skipif(FFMPEG is None, reason='ffmpeg is not installed')
def test_frame_graph_runs_in_ffmpeg(tmp_path: Path):
    """The graph must actually parse and produce the platform frame."""
    source = tmp_path / 'src.mp4'
    subprocess.run(
        [FFMPEG, '-y', '-f', 'lavfi', '-i', 'testsrc=size=640x360:rate=15:duration=1',
         '-pix_fmt', 'yuv420p', str(source)],
        check=True, capture_output=True,
    )
    caption = tmp_path / 'c.ass'
    caption.write_text(
        caption_ass([{'start': 0, 'end': 1, 'text': 'ok'}], 1080, 1920,
                    design(subtitle={'x': 0.5, 'y': 0.8, 'size': 0.08})),
        encoding='utf-8',
    )
    out = tmp_path / 'out.mp4'
    main._apply_platform_style(
        FFMPEG, str(source), str(out), 1080, 1920, 'blur',
        design=design(
            video={'size': 0.7, 'rotation': 12, 'flipH': True, 'x': 0.1, 'y': -0.05},
            blur_box={'enabled': True, 'x': 0.1, 'y': 0.6, 'w': 0.8, 'h': 0.2, 'strength': 15},
        ),
        caption_file=caption,
    )
    assert out.is_file() and out.stat().st_size > 1000
    probe = subprocess.run(
        [shutil.which('ffprobe') or 'ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'json', str(out)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)['streams'][0]
    assert (stream['width'], stream['height']) == (1080, 1920)
