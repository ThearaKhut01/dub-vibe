"""Offline translation with NLLB-200 (Meta), run locally with Hugging Face Transformers.

Model: facebook/nllb-200-distilled-600M (cc-by-nc-4.0, non-commercial, research model).
Weights are downloaded through the model manager into
%LOCALAPPDATA%\\DubVibe Studio\\models\\nllb\\nllb-200-distilled-600m\\ and run offline on CPU.
The checkpoint is loaded with weights_only=True, so it cannot run code while loading.
"""
from __future__ import annotations

import threading
from pathlib import Path

from .providers import TranslationProvider, models_root

NLLB_MODEL_ID = 'nllb-200-distilled-600m'
NLLB_REPO = 'facebook/nllb-200-distilled-600M'
NLLB_SIZE_MB = 2460
_MODEL_FILES = ['*.json', '*.model', 'pytorch_model.bin']

# App language names -> NLLB language codes (FLORES-200 "xxx_Script").
NLLB_LANGUAGE_CODES: dict[str, str] = {
    'English': 'eng_Latn',
    'Khmer': 'khm_Khmr',
    'French': 'fra_Latn',
    'Spanish': 'spa_Latn',
    'Chinese': 'zho_Hans',
    'Japanese': 'jpn_Jpan',
    'Korean': 'kor_Hang',
    'Thai': 'tha_Thai',
    'Vietnamese': 'vie_Latn',
    'German': 'deu_Latn',
    'Portuguese': 'por_Latn',
    'Russian': 'rus_Cyrl',
    'Arabic': 'arb_Arab',
    'Hindi': 'hin_Deva',
    'Indonesian': 'ind_Latn',
}

_ISO_TO_NLLB: dict[str, str] = {
    'en': 'eng_Latn', 'km': 'khm_Khmr', 'fr': 'fra_Latn', 'es': 'spa_Latn',
    'zh': 'zho_Hans', 'ja': 'jpn_Jpan', 'ko': 'kor_Hang', 'th': 'tha_Thai',
    'vi': 'vie_Latn', 'de': 'deu_Latn', 'pt': 'por_Latn', 'ru': 'rus_Cyrl',
    'ar': 'arb_Arab', 'hi': 'hin_Deva', 'id': 'ind_Latn',
}

_MODEL = None
_TOKENIZER = None
_LOAD_LOCK = threading.Lock()
# A generate() call on one model instance is not safe to run concurrently.
_INFERENCE_LOCK = threading.Lock()


def nllb_model_dir() -> Path:
    directory = models_root() / 'nllb' / NLLB_MODEL_ID
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_installed() -> bool:
    weights = nllb_model_dir() / 'pytorch_model.bin'
    return weights.is_file() and weights.stat().st_size > 1024 * 1024 * 1024


def download(progress_cb=None) -> None:
    """Blocking download of the NLLB model into the shared model store.

    progress_cb receives a 0-100 percent estimated from the bytes on disk, since
    snapshot_download does not expose byte progress.
    """
    import threading as _threading
    from huggingface_hub import snapshot_download

    target = nllb_model_dir()
    done = _threading.Event()

    def watch() -> None:
        expected = NLLB_SIZE_MB * 1024 * 1024
        while not done.wait(1.0):
            written = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
            if progress_cb:
                progress_cb(min(99, int(written * 100 / expected)))

    watcher = _threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot_download(repo_id=NLLB_REPO, local_dir=str(target), allow_patterns=_MODEL_FILES)
    finally:
        done.set()
        watcher.join(timeout=2)
    if progress_cb:
        progress_cb(100)


def _load():
    global _MODEL, _TOKENIZER
    with _LOAD_LOCK:
        if _MODEL is None or _TOKENIZER is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            directory = str(nllb_model_dir())
            _TOKENIZER = AutoTokenizer.from_pretrained(directory)
            # transformers >= 4.36 loads .bin files with weights_only=True (no pickled code).
            _MODEL = AutoModelForSeq2SeqLM.from_pretrained(directory, dtype=torch.float32)
            _MODEL.eval()
    return _MODEL, _TOKENIZER


def unload() -> None:
    global _MODEL, _TOKENIZER
    with _LOAD_LOCK:
        _MODEL = None
        _TOKENIZER = None


def nllb_code(language: str) -> str:
    """Resolve an app language name ('Khmer'), ISO code ('km'), or NLLB code ('khm_Khmr')."""
    if language in NLLB_LANGUAGE_CODES:
        return NLLB_LANGUAGE_CODES[language]
    if language in NLLB_LANGUAGE_CODES.values():
        return language
    iso = _ISO_TO_NLLB.get(language.lower())
    if iso:
        return iso
    raise ValueError(f'NLLB has no code for language: {language}')


class NllbTranslationProvider(TranslationProvider):
    """Local translation through NLLB-200. Fully offline once the model is installed."""

    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        import torch

        if not is_installed():
            raise RuntimeError('NLLB-200 is not installed. Download it from AI Models first.')
        source = nllb_code(source_language)
        target = nllb_code(target_language)
        if source == target:
            return text

        model, tokenizer = _load()
        tokenizer.src_lang = source
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with _INFERENCE_LOCK, torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target),
                max_new_tokens=256,
                num_beams=4,
                # Beam search can get stuck repeating one token until the length cap.
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )
        translated = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        if not translated:
            raise RuntimeError('NLLB returned an empty translation')
        if _is_degenerate(translated):
            raise RuntimeError('NLLB produced a repeated, unusable translation for this line')
        return translated


def _is_degenerate(text: str) -> bool:
    """True when the output is a single token repeated (a generation loop)."""
    words = text.split()
    if len(words) >= 4 and len(set(words)) <= 2:
        return True
    return len(text) >= 40 and len(set(text.replace(' ', ''))) <= 8
