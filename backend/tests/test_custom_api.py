"""Custom OpenAI-compatible API: configuration, auth, and translation routing."""
from __future__ import annotations

import json
import urllib.error

import pytest

from backend.app import custom_api, online_providers, providers
from backend.app.providers import CustomAPITranslationProvider, HybridTranslationProvider


def _configure(monkeypatch, base_url='https://api.example.com/v1', model='my-model',
               key='secret'):
    monkeypatch.setattr(
        'backend.app.model_manager.load_settings',
        lambda: {'custom_api': {'base_url': base_url, 'model': model}},
    )
    monkeypatch.setattr('backend.app.api_keys.has_key', lambda provider: bool(key))
    monkeypatch.setattr('backend.app.api_keys.get_key', lambda provider: key or None)


def test_bare_host_gets_the_openai_v1_prefix():
    assert custom_api.normalize_base_url('http://localhost:11434') == 'http://localhost:11434/v1'
    assert custom_api.normalize_base_url('http://localhost:11434/') == 'http://localhost:11434/v1'


def test_a_url_with_a_path_is_kept_exactly():
    assert custom_api.normalize_base_url('https://openrouter.ai/api/v1/') == 'https://openrouter.ai/api/v1'


@pytest.mark.parametrize('url', ['localhost:11434', 'ftp://example.com', 'https://'])
def test_invalid_base_urls_are_refused(url):
    with pytest.raises(ValueError):
        custom_api.normalize_base_url(url)


def test_chat_posts_to_the_configured_endpoint_with_bearer_auth(monkeypatch):
    _configure(monkeypatch)
    seen: dict = {}

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        seen.update(url=url, key=key, body=body, force_auth=force_auth)
        return {'choices': [{'message': {'content': 'សួស្តី'}}]}

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    assert custom_api.chat('secret', [{'role': 'user', 'content': 'hello'}]) == 'សួស្តី'
    assert seen['url'] == 'https://api.example.com/v1/chat/completions'
    assert seen['body']['model'] == 'my-model'
    assert seen['force_auth'] is True


def test_a_local_endpoint_needs_no_key(monkeypatch):
    _configure(monkeypatch, base_url='http://127.0.0.1:1234/v1', key='')
    seen: dict = {}

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        seen.update(key=key, force_auth=force_auth)
        return {'choices': [{'message': {'content': 'ok'}}]}

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    assert custom_api.chat('', [{'role': 'user', 'content': 'hi'}]) == 'ok'
    assert seen['key'] == ''
    assert seen['force_auth'] is True


def test_no_authorization_header_is_sent_for_an_empty_key():
    headers = online_providers._auth_headers('https://api.example.com/v1/models', '', force_auth=True)
    assert 'Authorization' not in headers


def test_save_config_persists_a_normalized_url_and_model(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(
        'backend.app.model_manager.save_settings',
        lambda update: saved.update(update) or {},
    )
    monkeypatch.setattr('backend.app.model_manager.load_settings', lambda: dict(saved))
    monkeypatch.setattr('backend.app.api_keys.has_key', lambda provider: False)

    config = custom_api.save_config({'base_url': 'http://localhost:11434/', 'model': 'qwen2.5'})
    assert config['base_url'] == 'http://localhost:11434/v1'
    assert config['model'] == 'qwen2.5'
    assert config['configured'] is True
    assert saved['custom_api'] == {'base_url': 'http://localhost:11434/v1', 'model': 'qwen2.5'}


@pytest.mark.parametrize('payload', [
    {'base_url': 'https://api.example.com/v1', 'model': ''},
    {'base_url': '', 'model': 'my-model'},
])
def test_save_config_requires_both_parts(monkeypatch, payload):
    monkeypatch.setattr(
        'backend.app.model_manager.save_settings',
        lambda update: pytest.fail('invalid config must not be saved'),
    )
    with pytest.raises(ValueError):
        custom_api.save_config(payload)


def test_config_round_trips_through_app_settings(monkeypatch, tmp_path):
    """The real settings writer must keep custom_api (it is not a secret)."""
    from backend.app import model_manager

    monkeypatch.setattr(model_manager, '_settings_path', lambda: tmp_path / 'app-settings.json')
    monkeypatch.setattr(model_manager, '_SETTINGS_CACHE', None)
    monkeypatch.setattr('backend.app.api_keys.has_key', lambda provider: False)

    custom_api.save_config({'base_url': 'https://api.example.com', 'model': 'my-model'})

    config = custom_api.get_config()
    assert config['base_url'] == 'https://api.example.com/v1'
    assert config['model'] == 'my-model'
    assert config['configured'] is True
    stored = json.loads((tmp_path / 'app-settings.json').read_text(encoding='utf-8'))
    assert stored['custom_api'] == {'base_url': 'https://api.example.com/v1', 'model': 'my-model'}
    # The settings file never holds an API key.
    assert 'secret' not in json.dumps(stored)


def test_custom_engine_is_only_used_when_it_is_chosen(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr('backend.app.qwen_local.is_installed', lambda: False)
    monkeypatch.setattr(providers, '_argos_available', lambda: False)

    auto = HybridTranslationProvider(preference='auto')
    assert 'custom' not in [key for key, _ in auto._chain()]

    chosen = HybridTranslationProvider(preference='custom')
    assert chosen.first_engine() == 'custom'
    assert [key for key, _ in chosen._chain()][0] == 'custom'


def test_custom_choice_without_configuration_explains_what_is_missing(monkeypatch):
    monkeypatch.setattr('backend.app.custom_api.is_configured', lambda: False)
    provider = HybridTranslationProvider(preference='custom')
    with pytest.raises(RuntimeError, match='base URL and model'):
        provider.check_preferred_engine('Khmer', 'en')


def test_custom_provider_translates_with_the_configured_model(monkeypatch):
    _configure(monkeypatch)
    provider = CustomAPITranslationProvider()
    monkeypatch.setattr(
        provider, '_complete',
        lambda messages, key: 'សួស្តី' if key == 'secret' else 'wrong key',
    )
    assert provider.translate('hello', 'Khmer', 'en') == 'សួស្តី'


def test_custom_endpoint_on_this_machine_stays_serial(monkeypatch):
    _configure(monkeypatch, base_url='http://localhost:11434/v1')
    monkeypatch.setattr('backend.app.qwen_local.is_installed', lambda: False)
    monkeypatch.setattr(providers, '_argos_available', lambda: False)
    assert HybridTranslationProvider(preference='custom').max_parallel_lines() == 1


def test_hosted_custom_endpoint_uses_the_small_pool(monkeypatch):
    _configure(monkeypatch, base_url='https://api.example.com/v1')
    monkeypatch.setattr('backend.app.qwen_local.is_installed', lambda: False)
    monkeypatch.setattr(providers, '_argos_available', lambda: False)
    assert HybridTranslationProvider(preference='custom').max_parallel_lines() == 4


@pytest.fixture(autouse=True)
def _clear_sticky_model():
    """The remembered model is process-wide; never leak it between tests."""
    custom_api._STICKY.clear()
    yield
    custom_api._STICKY.clear()


def test_model_chain_keeps_order_and_drops_duplicates():
    assert custom_api.parse_models('a, b\nc; a') == ['a', 'b', 'c']
    assert custom_api.parse_models('  ') == []


def test_save_config_persists_a_model_chain(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(
        'backend.app.model_manager.save_settings',
        lambda update: saved.update(update) or {},
    )
    monkeypatch.setattr('backend.app.model_manager.load_settings', lambda: dict(saved))
    monkeypatch.setattr('backend.app.api_keys.has_key', lambda provider: False)

    config = custom_api.save_config({
        'base_url': 'https://api.xkiro.com',
        'model': 'mistralai/mistral-large-2512, qwen/qwen3.8-omni-flash:free , mistralai/mistral-large-2512',
    })
    assert config['models'] == ['mistralai/mistral-large-2512', 'qwen/qwen3.8-omni-flash:free']
    assert config['primary'] == 'mistralai/mistral-large-2512'
    assert saved['custom_api']['model'] == 'mistralai/mistral-large-2512, qwen/qwen3.8-omni-flash:free'


def test_a_busy_model_falls_back_to_the_next(monkeypatch):
    _configure(monkeypatch, model='busy-model, backup-model')
    calls: list[str] = []

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        calls.append(body['model'])
        if body['model'] == 'busy-model':
            raise urllib.error.HTTPError(url, 503, 'Service Unavailable', {}, None)
        return {'choices': [{'message': {'content': 'ok'}}]}

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    assert custom_api.chat('secret', [{'role': 'user', 'content': 'hi'}]) == 'ok'
    assert calls == ['busy-model', 'backup-model']


def test_the_model_that_answered_is_reused_for_later_lines(monkeypatch):
    _configure(monkeypatch, model='busy-model, backup-model')
    calls: list[str] = []

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        calls.append(body['model'])
        if body['model'] == 'busy-model':
            raise urllib.error.HTTPError(url, 429, 'Too Many Requests', {}, None)
        return {'choices': [{'message': {'content': 'ok'}}]}

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    custom_api.chat('secret', [{'role': 'user', 'content': 'one'}])
    custom_api.chat('secret', [{'role': 'user', 'content': 'two'}])
    assert calls == ['busy-model', 'backup-model', 'backup-model']


def test_a_rejected_key_does_not_try_the_other_models(monkeypatch):
    _configure(monkeypatch, model='a, b')
    calls: list[str] = []

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        calls.append(body['model'])
        raise urllib.error.HTTPError(url, 401, 'Unauthorized', {}, None)

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    with pytest.raises(RuntimeError, match='401'):
        custom_api.chat('bad-key', [{'role': 'user', 'content': 'hi'}])
    assert calls == ['a']


def test_every_model_failing_reports_the_whole_chain(monkeypatch):
    _configure(monkeypatch, model='a, b')

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        raise urllib.error.HTTPError(url, 503, 'Service Unavailable', {}, None)

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    with pytest.raises(RuntimeError, match='every model'):
        custom_api.chat('secret', [{'role': 'user', 'content': 'hi'}])


def test_test_connection_reports_the_model_that_answered(monkeypatch):
    _configure(monkeypatch, model='busy-model, backup-model')
    monkeypatch.setattr(custom_api, 'list_models', lambda key, timeout=30: ['busy-model', 'backup-model'])

    def fake_request(url, key, body=None, timeout=60, force_auth=False):
        if body['model'] == 'busy-model':
            raise urllib.error.HTTPError(url, 503, 'Service Unavailable', {}, None)
        return {'choices': [{'message': {'content': 'OK'}}]}

    monkeypatch.setattr(online_providers, '_request_json', fake_request)
    result = custom_api.test('secret')
    assert result['ok'] is True
    assert result['model'] == 'backup-model'
    assert result['sample'] == 'OK'


def test_test_connection_falls_back_to_chat_when_models_is_missing(monkeypatch):
    _configure(monkeypatch, base_url='http://localhost:11434/v1')

    def fake_models(key, timeout=30):
        raise urllib.error.HTTPError('http://localhost:11434/v1/models', 404, 'Not Found', {}, None)

    monkeypatch.setattr(custom_api, 'list_models', fake_models)
    monkeypatch.setattr(custom_api, 'chat', lambda key, messages, **kwargs: 'OK')
    result = custom_api.test('')
    assert result['ok'] is True
    assert result['models'] == ['my-model']
