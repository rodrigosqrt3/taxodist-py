import os

# Use a non-interactive backend so plotting tests never open windows (and do
# not depend on Tk being available).
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import taxodist.fetch as fetch


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    """Keep retry logic active but skip the real backoff pauses in tests."""
    monkeypatch.setattr(fetch, "_sleep", lambda seconds: None)


@pytest.fixture(autouse=True)
def no_request_throttle(monkeypatch):
    """Disable the delay between requests (taxodist.fetch.MIN_INTERVAL)."""
    monkeypatch.setattr(fetch, "MIN_INTERVAL", 0)
