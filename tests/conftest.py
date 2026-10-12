import pytest

import taxodist.fetch as fetch


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    """Keep retry logic active but skip the real backoff pauses in tests."""
    monkeypatch.setattr(fetch, "_sleep", lambda seconds: None)
