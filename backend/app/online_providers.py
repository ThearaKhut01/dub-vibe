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
import urllib.error
import urllib.request
import wave

GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta'
OPENAI_BASE = 'https://api.openai.com/v1'

GEMINI_DEFAULT_MODEL = 'gemini-2.5-flash-preview-tts'
OPENAI_DEFAULT_MODEL = 'gpt-4o-mini-tts'

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
    """Generate speech via Gemini TTS and write a 24 kHz WAV to out_path."""
    spoken = f'{style_instruction}: {text}' if style_instruction else text
    body = {
        'contents': [{'parts': [{'text': spoken}]}],
        'generationConfig': {
            'responseModalities': ['AUDIO'],
            'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': voice}}},
        },
    }
    payload = _request_json(f'{GEMINI_BASE}/models/{model}:generateContent?key={key}', key, body, timeout=180)
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
