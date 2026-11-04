"""Test-suite isolation for the API pool's on-disk state.

The pool writes a request journal, a translation cache and credential cooldowns
to disk, and it stores keys in Windows Credential Manager. Tests must never see a
developer's live pool (a real cooldown or a real cache entry would change their
outcome), so:

* the module-level assignment below points the pool at a throwaway folder before
  any test module imports :mod:`backend.app.main` (which runs ``api_pool.startup``)
* the autouse fixture gives every test its own empty folder
"""
from __future__ import annotations

import os
import tempfile

# Before any import of backend.app.main: importing it calls api_pool.startup(),
# which must not touch the real journal.
os.environ['DUBVIBE_API_POOL_DIR'] = tempfile.mkdtemp(prefix='dubvibe-pool-tests-')

import pytest  # noqa: E402  (the env var above must be set first)


@pytest.fixture(autouse=True)
def _isolated_api_pool(tmp_path, monkeypatch):
    """A fresh pool folder per test, with no cooldowns carried over."""
    monkeypatch.setenv('DUBVIBE_API_POOL_DIR', str(tmp_path / 'api-pool'))
    from backend.app import api_pool
    from backend.app.api_pool import models

    models.store().reload()
    api_pool.reset_for_tests()
    yield
