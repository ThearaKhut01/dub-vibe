"""Online AI provider clients: Google Gemini TTS and the OpenAI API.

Both clients are backend-only: API keys come from the secure key manager and
are never sent to the browser. All calls use the standard library (urllib) so
no extra HTTP dependency is added.

Provider capability tables live here — the UI renders what the tables say and
never pretends an unsupported control works:
- Gemini TTS: natural-language style control ✓, fixed prebuilt voices,
  Khmer is NOT in the officially supported language list.
- OpenAI TTS (gpt-4o-mini-tts): steerable `instructions` ✓; the legacy
  tts-1/tts-1-hd models have no emotion control.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
import wave

GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta'
OPENAI_BASE = 'https://api.openai.com/v1'

GEMINI_DEFAULT_MODEL = 'gemini-2.5-flash-preview-tts'
OPENAI_DEFAULT_MODEL = 'gpt-4o-mini-tts'
# Text model used for translation. `gemini-flash-latest` is Google's rolling alias
# for the current Flash model, so it does not age out the way a pinned name does
# (gemini-2.5-flash already answers 404 "no longer available to new users").
# Not every listed model answers reliably, so gemini_chat tries a short chain and
# caches the first one that works.
GEMINI_DEFAULT_CHAT_MODEL = 'gemini-flash-latest'
OPENAI_DEFAULT_CHAT_MODEL = 'gpt-4o-mini'
# One subtitle line does not need more than this; a slow model is skipped instead
# of stalling the whole translation run.
_GEMINI_TEXT_ATTEMPT_TIMEOUT = 20
# Free-tier quota is per model (20 requests/day each), so a wider spread of models
# buys more usable capacity; the cap keeps a fully-exhausted key from spinning.
_GEMINI_TEXT_MAX_ATTEMPTS = 6
# Retries for a transient 429/5xx on the SAME model (a brief Gemini outage should
# not blacklist a healthy model or fail the line).
_GEMINI_TEXT_TRANSIENT_RETRIES = 2
# After a full sweep hits quota on every model, skip Gemini for this long so the
# rest of the run falls back immediately instead of re-probing every line.
_GEMINI_QUOTA_COOLDOWN_SECONDS = 600.0
# key -> time.monotonic() before which Gemini text is not worth trying.
_gemini_quota_blocked_until: dict[str, float] = {}
# key -> model that answered successfully, so later lines skip the probing.
_gemini_text_model_cache: dict[str, str] = {}
# key -> models that timed out or 404'd, skipped once measured.
_gemini_dead_models: dict[str, set[str]] = {}


def _model_cache_path():
    """Where the working translation model is remembered between app starts.

    Probing costs ~40 s on a key whose newest models hang, so the winner is saved
    and reloaded: the first translated line after a restart goes straight to it.
    Failures to read or write are ignored — this is a speed-up, not a requirement.
    """
    from pathlib import Path

    from .providers import get_storage_dir

    return Path(get_storage_dir('data')) / 'translation-models.json'


def _model_cache_key(key: str) -> str:
    """Fingerprint of the API key (never the key itself) for the on-disk cache."""
    import hashlib

    return hashlib.sha256(str(key or '').encode('utf-8')).hexdigest()[:16]


def _load_model_cache() -> None:
    import json

    try:
        raw = json.loads(_model_cache_path().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    for cache_key, model in raw.items():
        if isinstance(cache_key, str) and isinstance(model, str) and model:
            _gemini_text_model_cache.setdefault(cache_key, model)


def _save_model_cache(cache_key: str, model: str) -> None:
    """Remember a working model for this key; best-effort only."""
    import json

    try:
        path = _model_cache_path()
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(raw, dict):
                raw = {}
        except (OSError, ValueError):
            raw = {}
        raw[cache_key] = model
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=2), encoding='utf-8')
    except OSError:
        pass


_load_model_cache()


def _model_rank(name: str) -> tuple[int, int, int]:
    """Sort key for model names: newest version first, stable builds preferred."""
    match = re.search(r'(\d+)\.(\d+)', name)
    major, minor = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    stable = 0 if any(tag in name.lower() for tag in ('preview', 'exp', 'lite', 'latest')) else 1
    return (major, minor, stable)


def _newest(names: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in names:
        return preferred
    if not names:
        return None
    return sorted(names, key=_model_rank, reverse=True)[0]


def gemini_text_model_candidates(key: str, preferred: str = GEMINI_DEFAULT_CHAT_MODEL) -> list[str]:
    """Models to try for translation, best guess first.

    Order: the model that already worked for this key, then the newest concrete
    models, then the requested (possibly alias) name. Attempts are capped so a run
    degrades gracefully instead of hanging on one unlucky model.
    """
    candidates: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in candidates:
            candidates.append(name)

    add(_gemini_text_model_cache.get(_model_cache_key(key)))
    try:
        listed = gemini_list_chat_models(key)
    except Exception:
        listed = []
    dead = _gemini_dead_models.get(key, set())
    concrete = [name for name in listed if 'latest' not in name.lower() and name not in dead]
    for name in sorted(concrete, key=_model_rank, reverse=True):
        add(name)
    add(preferred)
    for name in listed:
        if name not in dead:
            add(name)
    return candidates[:_GEMINI_TEXT_MAX_ATTEMPTS]

# Prebuilt Gemini TTS voices (name -> short style hint).
GEMINI_VOICES: dict[str, str] = {
    'Zephyr': 'Bright', 'Puck': 'Upbeat', 'Charon': 'Informative', 'Kore': 'Firm',
    'Fenrir': 'Excitable', 'Leda': 'Youthful', 'Orus': 'Firm', 'Aoede': 'Breezy',
    'Callirrhoe': 'Easy-going', 'Autonoe': 'Bright', 'Enceladus': 'Breathy',
    'Iapetus': 'Clear', 'Umbriel': 'Easy-going', 'Algieba': 'Smooth',
    'Despina': 'Smooth', 'Erinome': 'Clear', 'Algenib': 'Gravelly',
    'Rasalgethi': 'Informative', 'Laomedeia': 'Upbeat', 'Achernar': 'Soft',
    'Alnilam': 'Firm', 'Schedar': 'Even', 'Gacrux': 'Mature',
    'Pulcherrima': 'Forward', 'Achird': 'Friendly', 'Zubenelgenubi': 'Casual',
    'Vindemiatrix': 'Gentle', 'Sadachbia': 'Lively', 'Sadaltager': 'Knowledgeable',
    'Sulafat': 'Warm',
}

OPENAI_VOICES: dict[str, str] = {
    'alloy': 'Neutral', 'ash': 'Neutral', 'ballad': 'Warm', 'coral': 'Warm',
    'echo': 'Calm', 'fable': 'Expressive', 'onyx': 'Deep', 'nova': 'Bright',
    'sage': 'Balanced', 'shimmer': 'Soft',
}

# Officially supported TTS languages (ISO 639-1). Khmer (km) is not in the
# official Gemini TTS language list and OpenAI does not guarantee it either —
# the capability table keeps this honest instead of guessing.
GEMINI_TTS_LANGUAGES = {'en', 'es', 'fr', 'de', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'ru', 'ar', 'th', 'vi'}
OPENAI_TTS_LANGUAGES = {'en', 'es', 'fr', 'de', 'hi', 'id', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'nl', 'th', 'vi', 'zh'}

EMOTIONS = [
    'Auto', 'Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Excited', 'Surprised',
    'Disgusted', 'Romantic', 'Confident', 'Nervous', 'Calm', 'Serious', 'Sarcastic',
    'Whisper', 'Shout', 'Crying', 'Laughing',
]


def _auth_headers(url: str, key: str) -> dict[str, str]:
    """Request headers with the provider's auth in the place it expects it.

    Gemini takes its key in the query string (the callers append it), while every
    OpenAI endpoint requires an `Authorization: Bearer <key>` header. Without it
    every OpenAI call fails with 401.
    """
    headers = {'Content-Type': 'application/json'}
    if url.startswith(OPENAI_BASE):
        headers['Authorization'] = f'Bearer {key}'
    return headers


def _request_json(url: str, key: str, body: dict | None = None, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode('utf-8') if body is not None else None,
        headers=_auth_headers(url, key),
        method='POST' if body is not None else 'GET',
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def gemini_list_models(key: str) -> list[str]:
    """Live TTS-capable model names from the Gemini API (never hard-coded)."""
    payload = _request_json(f'{GEMINI_BASE}/models?key={key}&pageSize=200', key)
    names = []
    for model in payload.get('models', []):
        name = str(model.get('name', '')).split('/')[-1]
        if 'tts' in name.lower():
            names.append(name)
    return sorted(names)


def gemini_capabilities(model: str) -> dict:
    return {
        'voices': [{'id': voice, 'style': style} for voice, style in GEMINI_VOICES.items()],
        'emotions': EMOTIONS[1:],  # Gemini follows natural-language style directions
        'style_control': True,
        'languages': sorted(GEMINI_TTS_LANGUAGES),
        'khmer_supported': False,  # not in the official language list
        'note': 'Style is directed with natural-language instructions; voices are fixed prebuilt names.',
    }


def gemini_synthesize(key: str, text: str, out_path: str, model: str = GEMINI_DEFAULT_MODEL,
                      voice: str = 'Kore', style_instruction: str | None = None) -> str:
    """Generate speech via Gemini TTS and write a 24 kHz WAV to out_path.

    A pinned model name can be retired by Google (a 404 "no longer available"), so
    that case retries once with the newest TTS model this key can actually use.
    """
    spoken = f'{style_instruction}: {text}' if style_instruction else text
    body = {
        'contents': [{'parts': [{'text': spoken}]}],
        'generationConfig': {
            'responseModalities': ['AUDIO'],
            'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': voice}}},
        },
    }

    def request(model_name: str) -> dict:
        return _request_json(f'{GEMINI_BASE}/models/{model_name}:generateContent?key={key}',
                             key, body, timeout=180)

    try:
        payload = request(model)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        fallback = _newest(gemini_list_models(key), model)
        if not fallback or fallback == model:
            raise
        model = fallback
        payload = request(model)

    parts = payload.get('candidates', [{}])[0].get('content', {}).get('parts', [])
    audio_b64 = next((part.get('inlineData', {}).get('data') for part in parts if part.get('inlineData')), None)
    if not audio_b64:
        error = payload.get('error', {}).get('message') or payload.get('promptFeedback', {}).get('blockReason')
        raise RuntimeError(f'Gemini TTS returned no audio: {error or "unknown response"}')
    pcm = base64.b64decode(audio_b64)
    with wave.open(out_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return out_path


def gemini_test(key: str) -> dict:
    models = gemini_list_models(key)
    if not models:
        raise RuntimeError('No TTS-capable Gemini models are available for this key')
    return {'ok': True, 'models': models}


# --- Gemini text (translation / review) ------------------------------------

def gemini_list_chat_models(key: str) -> list[str]:
    """Text-capable Gemini models (the ones that can translate), never hard-coded."""
    payload = _request_json(f'{GEMINI_BASE}/models?key={key}&pageSize=200', key)
    names: list[str] = []
    for model in payload.get('models', []):
        name = str(model.get('name', '')).split('/')[-1]
        lowered = name.lower()
        # Image/audio/embedding models cannot translate a subtitle line.
        if any(marker in lowered for marker in ('tts', 'embedding', 'image', 'imagen', 'veo', 'omni')):
            continue
        methods = model.get('supportedGenerationMethods') or []
        if methods and 'generateContent' not in methods:
            continue
        names.append(name)
    return sorted(names)


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    """Short, human-readable reason from a Gemini/OpenAI error response."""
    try:
        raw = error.read().decode('utf-8', errors='replace')
    except Exception:
        return f'HTTP {error.code}'
    try:
        payload = json.loads(raw)
        message = str(payload.get('error', {}).get('message') or '').strip()
    except ValueError:
        message = raw.strip()
    message = ' '.join(message.split())
    return f'HTTP {error.code}: {message[:200]}' if message else f'HTTP {error.code}'


def gemini_chat(key: str, messages: list[dict], model: str = GEMINI_DEFAULT_CHAT_MODEL,
                timeout: int = 90) -> str:
    """One-shot text completion via generateContent.

    Used for translation (and the optional translation review); the TTS path stays
    on `gemini_synthesize`, which asks for audio instead of text. A retired model
    name (404) retries once with the newest text model the key can use.
    """
    system = '\n'.join(str(message.get('content') or '') for message in messages
                       if message.get('role') == 'system').strip()
    contents = [
        {
            'role': 'model' if message.get('role') == 'assistant' else 'user',
            'parts': [{'text': str(message.get('content') or '')}],
        }
        for message in messages
        if message.get('role') != 'system'
    ]
    body: dict = {'contents': contents}
    if system:
        body['systemInstruction'] = {'parts': [{'text': system}]}

    def request(model_name: str) -> dict:
        # A short per-attempt timeout keeps one slow model from stalling the run.
        return _request_json(f'{GEMINI_BASE}/models/{model_name}:generateContent?key={key}',
                             key, body, timeout=min(timeout, _GEMINI_TEXT_ATTEMPT_TIMEOUT))

    last_error: Exception | None = None
    payload: dict | None = None
    cache_key = _model_cache_key(key)
    dead = _gemini_dead_models.setdefault(key, set())

    blocked_until = _gemini_quota_blocked_until.get(cache_key, 0.0)
    if time.monotonic() < blocked_until:
        # The last full sweep hit quota on every model; fail fast so the caller's
        # fallback engine handles this line instead of probing again.
        raise RuntimeError(
            'Gemini quota is exhausted for this key right now '
            '(free-tier limits are per model and reset daily). Using the fallback engine.'
        )

    quota_hits = 0
    attempted = 0
    for model_name in gemini_text_model_candidates(key, model):
        attempted += 1
        for attempt in range(_GEMINI_TEXT_TRANSIENT_RETRIES + 1):
            try:
                payload = request(model_name)
            except urllib.error.HTTPError as error:
                last_error = RuntimeError(_http_error_detail(error))
                if error.code == 404:
                    # Retired model: never try it again for this key.
                    dead.add(model_name)
                    _gemini_text_model_cache.pop(cache_key, None)
                    break
                if error.code == 429:
                    # Free-tier quota is per model ("limit: 20 ... retry in 21h").
                    # Retrying would waste the run, so switch to another model now.
                    quota_hits += 1
                    dead.add(model_name)
                    _gemini_text_model_cache.pop(cache_key, None)
                    break
                if 500 <= error.code < 600:
                    # Transient server error: retry this model, and do NOT
                    # blacklist it (a short outage must not poison the cache).
                    if attempt < _GEMINI_TEXT_TRANSIENT_RETRIES:
                        time.sleep(1.0 + attempt)
                        continue
                    break
                # Any other client error: try the next model rather than fail.
                break
            except (TimeoutError, OSError) as error:
                # Unresponsive model: skip it for this process, try the next one.
                last_error = error
                dead.add(model_name)
                _gemini_text_model_cache.pop(cache_key, None)
                break
            else:
                _gemini_text_model_cache[cache_key] = model_name
                dead.discard(model_name)
                _gemini_quota_blocked_until.pop(cache_key, None)
                _save_model_cache(cache_key, model_name)
                break
        if payload is not None:
            break
    if payload is None:
        if attempted and quota_hits == attempted:
            # Every model refused on quota: cool the key down.
            _gemini_quota_blocked_until[cache_key] = (
                time.monotonic() + _GEMINI_QUOTA_COOLDOWN_SECONDS
            )
        raise RuntimeError(f'Gemini returned no completion: {last_error or "no usable model"}')

    candidates = payload.get('candidates') or []
    if not candidates:
        error = payload.get('error', {}).get('message') or payload.get('promptFeedback', {}).get('blockReason')
        raise RuntimeError(f'Gemini returned no completion: {error or "unknown response"}')
    parts = candidates[0].get('content', {}).get('parts') or []
    text = ''.join(str(part.get('text') or '') for part in parts).strip()
    if not text:
        raise RuntimeError('Gemini returned an empty completion')
    return text


def gemini_text_test(key: str) -> dict:
    models = gemini_list_chat_models(key)
    if not models:
        raise RuntimeError('No text-capable Gemini models are available for this key')
    return {'ok': True, 'models': models}


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def openai_list_models(key: str) -> list[str]:
    payload = _request_json(f'{OPENAI_BASE}/models', key)
    names = sorted(
        str(entry.get('id'))
        for entry in payload.get('data', [])
        if str(entry.get('id', '')).startswith('tts') or 'tts' in str(entry.get('id', ''))
    )
    return names


def openai_capabilities(model: str) -> dict:
    steerable = 'gpt-4o' in model  # gpt-4o-mini-tts follows instructions; tts-1 does not
    return {
        'voices': [{'id': voice, 'style': style} for voice, style in OPENAI_VOICES.items()],
        'emotions': EMOTIONS[1:] if steerable else ['Neutral'],
        'style_control': steerable,
        'languages': sorted(OPENAI_TTS_LANGUAGES),
        'khmer_supported': False,  # not officially documented
        'note': 'Emotion works through `instructions` on gpt-4o models; tts-1/tts-1-hd are fixed-voice.',
    }


def openai_synthesize(key: str, text: str, out_path: str, model: str = OPENAI_DEFAULT_MODEL,
                      voice: str = 'alloy', instructions: str | None = None,
                      speed: float = 1.0) -> str:
    body: dict = {'model': model, 'voice': voice, 'input': text, 'response_format': 'mp3', 'speed': speed}
    if instructions and 'gpt-4o' in model:
        body['instructions'] = instructions
    speech_url = f'{OPENAI_BASE}/audio/speech'
    request = urllib.request.Request(
        speech_url,
        data=json.dumps(body).encode('utf-8'),
        headers=_auth_headers(speech_url, key),
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            audio = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')[:300]
        raise RuntimeError(f'OpenAI TTS failed ({error.code}): {detail}') from error
    with open(out_path, 'wb') as audio_file:
        audio_file.write(audio)
    return out_path


def openai_chat(key: str, messages: list[dict], model: str = 'gpt-4o-mini', json_mode: bool = False,
                timeout: int = 90) -> str:
    body: dict = {'model': model, 'messages': messages}
    if json_mode:
        body['response_format'] = {'type': 'json_object'}
    payload = _request_json(f'{OPENAI_BASE}/chat/completions', key, body, timeout=timeout)
    try:
        return payload['choices'][0]['message']['content']
    except (KeyError, IndexError) as error:
        raise RuntimeError(f'OpenAI returned no completion: {json.dumps(payload)[:200]}') from error


def openai_test(key: str) -> dict:
    models = openai_list_models(key)
    if not models:
        raise RuntimeError('No TTS models are available for this OpenAI key')
    return {'ok': True, 'models': models}


def openai_text_test(key: str) -> dict:
    """Text (translation) is available with every OpenAI key; list chat models."""
    payload = _request_json(f'{OPENAI_BASE}/models', key)
    names = sorted(
        str(entry.get('id'))
        for entry in payload.get('data', [])
        if not str(entry.get('id', '')).startswith(('tts', 'whisper', 'dall-e', 'text-embedding'))
    )
    return {'ok': True, 'models': names or [OPENAI_DEFAULT_CHAT_MODEL]}
