"""Compare translation models on real lines from one project.

Prints a plain table — source line, the same batch translated by each model, and
the prompt/output token counts — and scores nothing. Khmer quality is a human
call, this only makes the comparison cheap.

Usage:
    python scripts/compare_models.py --project <project-id>
    python scripts/compare_models.py --project <id> --lines 20 \
        --models gemini-2.5-flash-lite,gemini-2.5-flash
    python scripts/compare_models.py --text-file sample.txt

Every model receives the same system prompt, the same CONTEXT block and the same
"ID|text" body that the app sends, so the outputs are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import api_keys, custom_api, model_manager  # noqa: E402
from backend.app.online_providers import gemini_chat  # noqa: E402
from backend.app.providers import (  # noqa: E402
    _resolve_lang_code,
    batch_output_token_budget,
    build_translation_system_prompt,
    build_translation_user_message,
    parse_batch_reply,
)

DEFAULT_MODELS = 'gemini-2.5-flash-lite,gemini-2.5-flash'


def projects_db() -> Path:
    from backend.app import providers

    return providers.get_storage_dir('data') / 'projects.sqlite3'


def list_projects() -> list[tuple[str, str, int]]:
    path = projects_db()
    if not path.is_file():
        return []
    with sqlite3.connect(path) as connection:
        rows = connection.execute('SELECT id, data FROM projects').fetchall()
    out: list[tuple[str, str, int]] = []
    for project_id, raw in rows:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out.append((project_id, str(data.get('name') or ''), len(data.get('segments') or [])))
    return out


def load_lines(project_id: str, limit: int) -> tuple[list[str], str, str]:
    """Up to ``limit`` source lines, CJK/idiom lines first, order preserved."""
    path = projects_db()
    if not path.is_file():
        raise SystemExit(f'No project database at {path}')
    with sqlite3.connect(path) as connection:
        row = connection.execute('SELECT data FROM projects WHERE id = ?', (project_id,)).fetchone()
    if row is None:
        raise SystemExit(f'Project {project_id} not found')
    data = json.loads(row[0])
    segments = [segment for segment in (data.get('segments') or []) if str(segment.get('text') or '').strip()]
    if not segments:
        raise SystemExit('That project has no transcribed lines yet')

    def is_cjk(text: str) -> bool:
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    chosen = [segment for segment in segments if is_cjk(segment['text'])][:limit]
    for segment in segments:
        if len(chosen) >= limit:
            break
        if segment not in chosen:
            chosen.append(segment)
    chosen.sort(key=lambda item: float(item.get('start') or 0))
    return (
        [str(segment['text']) for segment in chosen],
        str(data.get('source_language') or 'en'),
        str(data.get('glossary') or ''),
    )


def gemini_translate(model: str, lines: list[str], context: list[str], glossary: str,
                     target: str, source: str) -> tuple[str, dict]:
    key = api_keys.get_key('gemini')
    if not key:
        raise RuntimeError('Gemini API key is not configured')
    system = build_translation_system_prompt(glossary, target_language=target)
    user = build_translation_user_message(lines, context)
    text, meta = gemini_chat(
        key, [{'role': 'user', 'content': user}], model=model,
        generation_config={
            'temperature': 0.2,
            'maxOutputTokens': batch_output_token_budget(user, len(lines)),
            'thinkingConfig': {'thinkingBudget': 0},
        },
        system_instruction=system, want_usage=True, prefer_requested=True,
    )
    return text, {'prompt': meta['usage']['prompt_tokens'], 'output': meta['usage']['candidates_tokens'],
                  'thoughts': meta['usage']['thoughts_tokens'], 'model': meta.get('model') or model}


def custom_translate(model: str | None, lines: list[str], context: list[str], glossary: str,
                     target: str, source: str) -> tuple[str, dict]:
    config = custom_api.get_config()
    if not config.get('base_url'):
        raise RuntimeError('Custom API base URL is not configured')
    system = build_translation_system_prompt(glossary, target_language=target)
    user = build_translation_user_message(lines, context)
    usage: dict[str, int] = {}
    text = custom_api.chat(
        api_keys.get_key('custom') or '',
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        model=model,
        max_tokens=batch_output_token_budget(user, len(lines)),
        usage_out=usage,
    )
    return text, {'prompt': usage.get('prompt_tokens', 0), 'output': usage.get('candidates_tokens', 0),
                  'thoughts': usage.get('thoughts_tokens', 0), 'model': model or config.get('primary') or 'custom'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', help='project id (see --list)')
    parser.add_argument('--list', action='store_true', help='list projects and exit')
    parser.add_argument('--text-file', help='use one line per line of this UTF-8 file instead')
    parser.add_argument('--lines', type=int, default=20)
    parser.add_argument('--models', default=DEFAULT_MODELS, help='comma separated Gemini models')
    parser.add_argument('--include-custom', action='store_true',
                        help='also translate with the configured Custom API model (Mistral)')
    parser.add_argument('--custom-model', default=None)
    parser.add_argument('--target', default='Khmer')
    args = parser.parse_args()

    if args.list:
        for project_id, name, count in list_projects():
            print(f'{project_id}  {count:>5} lines  {name}')
        return 0

    if args.text_file:
        lines = [line.strip() for line in Path(args.text_file).read_text(encoding='utf-8').splitlines() if line.strip()]
        source, glossary = 'en', ''
    elif args.project:
        lines, source, glossary = load_lines(args.project, args.lines)
    else:
        parser.error('give --project, --text-file or --list')

    lines = lines[: max(1, args.lines)]
    context: list[str] = []

    models = [name.strip() for name in args.models.split(',') if name.strip()]
    runs: list[tuple[str, callable]] = [(name, lambda *a, m=name: gemini_translate(m, *a)) for name in models]
    if args.include_custom:
        label = f'custom:{args.custom_model or custom_api.get_config().get("primary") or "model"}'
        runs.append((label, lambda *a: custom_translate(args.custom_model, *a)))

    print(f'{len(lines)} lines · source={source} · target={args.target}')
    print(f'glossary: {glossary or "none"}')
    print()

    results: dict[str, tuple[str, dict]] = {}
    for label, runner in runs:
        try:
            raw, usage = runner(lines, context, glossary, args.target, source)
        except Exception as error:  # a missing key must not hide the other models
            results[label] = (f'ERROR: {error}', {})
            continue
        results[label] = (raw, usage)
        print(f'{label}: 1 request · prompt={usage.get("prompt")} output={usage.get("output")} '
              f'thoughts={usage.get("thoughts")} (served by {usage.get("model")})')

    print()
    parsed = {label: parse_batch_reply(raw) for label, (raw, _) in results.items()}
    for index, line in enumerate(lines, start=1):
        print(f'{index:>2}. {line}')
        for label, _ in runs:
            value = parsed.get(label, {}).get(index, '—')
            print(f'    {label:<32} {value}')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
