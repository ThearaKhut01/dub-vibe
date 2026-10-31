"""Custom OpenAI-compatible translation API.

Some users already have an OpenAI-compatible endpoint: OpenRouter, DeepSeek,
Groq, Together, a company gateway, or a local server (LM Studio, vLLM, Ollama's
``/v1``). This module lets them point DubVibe Studio at it without changing the
app.

- The **base URL** and **model** are ordinary settings in ``app-settings.json``.
- The **API key** (optional — local servers often need none) lives in Windows
  Credential Manager through :mod:`api_keys`, never in the settings file and
  never in the browser.
- The endpoint must speak the OpenAI chat-completions shape:
  ``POST {base_url}/chat/completions`` with ``model`` + ``messages``.
"""
from __future__ import annotations

import json
import re
import urllib.error
from urllib.parse import urlparse

DEFAULT_TIMEOUT = 90

# Statuses worth trying the next model for. A free/shared tier answers 429 or
# 5xx when one model is congested, but a 401 or 404 would fail for every model
# on the same endpoint, so those are reported immediately.
FALLBACK_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_MODEL_SEPARATORS = re.compile(r'[,\n;]+')

# Last model that answered, per base URL. A project keeps using the model that
# is actually serving instead of retrying a congested primary on every line.
_STICKY: dict[str, str] = {}

# Hosts that usually run on the same machine or LAN. Translation stays serial on
# them (four parallel requests would fight the UI for CPU), while a hosted
# endpoint keeps the small paid-engine pool.
_LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1', '0.0.0.0', 'host.docker.internal'}


def _settings() -> dict:
    from .model_manager import load_settings

    config = load_settings().get('custom_api')
    return config if isinstance(config, dict) else {}


def parse_models(value: str | list[str] | None) -> list[str]:
    """Model IDs from a comma/newline separated list, in the order given.

    The field holds a fallback chain: ``model-a, model-b`` tries ``model-a``
    first and moves on only when that model is congested or unavailable.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        parts = _MODEL_SEPARATORS.split(str(value or ''))
    models: list[str] = []
    for part in parts:
        name = part.strip()
        if name and name not in models:
            models.append(name)
    return models


def get_config() -> dict:
    """Current configuration, safe to return to the browser (no key)."""
    from . import api_keys

    config = _settings()
    base_url = str(config.get('base_url') or '').strip()
    models = parse_models(config.get('model'))
    return {
        'base_url': base_url,
        # The raw chain, so the settings field round-trips exactly as typed.
        'model': ', '.join(models),
        'models': models,
        'primary': models[0] if models else '',
        'configured': bool(base_url and models),
        'has_key': api_keys.has_key('custom'),
    }


def normalize_base_url(url: str) -> str:
    """Validate and tidy a base URL.

    A bare host (``http://localhost:11434``) gets ``/v1`` appended, because that
    is the OpenAI-compatible prefix every local server uses. A URL that already
    carries a path (``https://openrouter.ai/api/v1``) is kept exactly as entered.
    """
    url = str(url or '').strip().rstrip('/')
    if not url:
        return ''
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('Base URL must start with http:// or https:// and include a host.')
    if not parsed.path or parsed.path == '/':
        url += '/v1'
    return url


def save_config(update: dict) -> dict:
    """Persist the base URL and model chain. An empty pair clears the engine."""
    from .model_manager import save_settings

    base_url = normalize_base_url(update.get('base_url', ''))
    models = parse_models(update.get('model'))
    if base_url and not models:
        raise ValueError('Model name is required when a base URL is set.')
    if models and not base_url:
        raise ValueError('Base URL is required when a model name is set.')
    save_settings({'custom_api': {'base_url': base_url, 'model': ', '.join(models)}})
    # A different chain must not inherit the model remembered from the old one.
    _STICKY.pop(base_url, None)
    return get_config()


def is_configured() -> bool:
    return bool(get_config()['configured'])


def is_local() -> bool:
    """True for an endpoint on this machine or the local network."""
    host = (urlparse(get_config()['base_url']).hostname or '').lower()
    if host in _LOCAL_HOSTS or host.endswith('.local'):
        return True
    if host.startswith(('10.', '192.168.')):
        return True
    if host.startswith('172.'):
        try:
            second = int(host.split('.')[1])
        except (IndexError, ValueError):
            return False
        return 16 <= second <= 31
    return False


class _NoCompletion(RuntimeError):
    """The endpoint answered, but without a usable choice."""


def _ordered_models(base_url: str, models: list[str]) -> list[str]:
    """The chain, with the model that last answered moved to the front."""
    preferred = _STICKY.get(base_url)
    if preferred and preferred in models and models[0] != preferred:
        return [preferred] + [name for name in models if name != preferred]
    return list(models)


def _failure_message(failures: list[str]) -> str:
    if len(failures) == 1:
        return f'Custom API request failed ({failures[0]})'
    return (
        'Custom API request failed — every model in the list was unavailable '
        f'({"; ".join(failures)})'
    )


def _attempt(key: str, base_url: str, model: str, messages: list[dict],
             timeout: int, max_tokens: int | None) -> str:
    """One request for one model. Raises the raw HTTP/transport error."""
    from .online_providers import _request_json

    body: dict = {'model': model, 'messages': messages}
    if max_tokens is not None:
        body['max_tokens'] = max_tokens
    payload = _request_json(
        f'{base_url}/chat/completions', key, body, timeout=timeout, force_auth=True,
    )
    try:
        content = payload['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as error:
        raise _NoCompletion(
            f'returned no completion: {json.dumps(payload)[:160]}'
        ) from error
    return str(content or '')


def chat(key: str, messages: list[dict], model: str | None = None,
         timeout: int = DEFAULT_TIMEOUT, max_tokens: int | None = None) -> str:
    """One chat completion, walking the configured model list on congestion.

    ``model`` overrides the configured chain for a single call. The model that
    answers is remembered, so a busy primary is not retried line after line.
    """
    from .online_providers import _http_error_detail

    config = get_config()
    base_url = config['base_url']
    if not base_url:
        raise RuntimeError('Custom API base URL is not configured (add it in API Settings).')
    candidates = parse_models(model) if model else list(config['models'])
    if not candidates:
        raise RuntimeError('Custom API model is not configured (add it in API Settings).')

    failures: list[str] = []
    for candidate in _ordered_models(base_url, candidates):
        try:
            content = _attempt(key, base_url, candidate, messages, timeout, max_tokens)
        except urllib.error.HTTPError as error:
            detail = _http_error_detail(error)
            if error.code in FALLBACK_STATUS:
                failures.append(f'{candidate}: {detail}')
                continue
            raise RuntimeError(f'Custom API request failed ({detail})') from error
        except _NoCompletion as error:
            failures.append(f'{candidate}: {error}')
            continue
        except (TimeoutError, OSError) as error:
            raise RuntimeError(f'Custom API is unreachable at {base_url}: {error}') from error
        _STICKY[base_url] = candidate
        return content

    raise RuntimeError(_failure_message(failures))


def list_models(key: str, timeout: int = 30) -> list[str]:
    """Model ids the endpoint advertises (``GET {base_url}/models``)."""
    from .online_providers import _request_json

    base_url = get_config()['base_url']
    if not base_url:
        raise RuntimeError('Custom API base URL is not configured (add it in API Settings).')
    payload = _request_json(f'{base_url}/models', key, timeout=timeout, force_auth=True)
    entries = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        entries = payload.get('models') if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    names = []
    for entry in entries:
        if isinstance(entry, dict):
            name = str(entry.get('id') or entry.get('name') or '').strip()
        else:
            name = str(entry or '').strip()
        if name:
            names.append(name)
    return sorted(set(names))


def test(key: str) -> dict:
    """Prove the endpoint works: list models, then answer one tiny prompt.

    Listing alone is not enough — a proxy can answer ``/models`` while the
    configured model name is wrong. The second call is deliberately minimal
    (a few tokens) so a connection test costs almost nothing.
    """
    config = get_config()
    if not config['configured']:
        raise RuntimeError('Set the base URL and model first.')
    models: list[str] = []
    try:
        models = list_models(key)
    except urllib.error.HTTPError as error:
        # Some compatible servers do not implement /models; the chat call below
        # is the real test.
        if error.code not in (404, 405):
            from .online_providers import _http_error_detail

            raise RuntimeError(
                f'Custom API rejected the request ({_http_error_detail(error)})'
            ) from error
    except (TimeoutError, OSError) as error:
        raise RuntimeError(f'Custom API is unreachable at {config["base_url"]}: {error}') from error

    sample = chat(
        key,
        [{'role': 'user', 'content': 'Reply with the single word OK.'}],
        timeout=45,
        max_tokens=8,
    )
    # chat() remembers which model answered; report it so the UI can show that
    # the primary was busy and a fallback took over.
    used = _STICKY.get(config['base_url']) or config['primary']
    return {
        'ok': True,
        'models': models or list(config['models']),
        'model': used,
        'sample': sample.strip()[:80],
    }
