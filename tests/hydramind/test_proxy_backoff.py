"""ProxyCompletion retry/backoff — a rate-limited (429) or transient (5xx / conn
blip) proxy must not kill a live run mid-flight. Offline: urlopen is faked and
sleep is a spy, so nothing sleeps or hits the network."""
from __future__ import annotations

import email.message
import urllib.error
import urllib.request

import pytest

from hydramind import transport
from hydramind.transport import ProxyCompletion


class _FakeResp:
    """Minimal urlopen context-manager stand-in."""
    def __init__(self, body: str, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body.encode("utf-8")


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x", code, f"err{code}", hdrs, None)


def _proxy(monkeypatch, script, sleeps):
    """Patch urlopen to walk `script` (each entry: an Exception to raise or a
    _FakeResp to return); record sleep durations in `sleeps`."""
    it = iter(script)

    def fake_urlopen(req, timeout=None):
        step = next(it)
        if isinstance(step, Exception):
            raise step
        return step

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return ProxyCompletion(max_retries=3, backoff_base=0.5,
                           sleep_fn=lambda d: sleeps.append(d))


_REQ = urllib.request.Request("http://x", data=b"{}", method="POST")


# ── _retry_delay ──────────────────────────────────────────────────────────────

def test_retry_delay_prefers_retry_after():
    p = ProxyCompletion(backoff_base=0.5, backoff_cap=30.0)
    assert p._retry_delay(_http_error(429, retry_after="7"), attempt=0) == 7.0


def test_retry_delay_caps_retry_after():
    p = ProxyCompletion(backoff_cap=30.0)
    assert p._retry_delay(_http_error(429, retry_after="999"), attempt=0) == 30.0


def test_retry_delay_falls_back_to_exponential():
    p = ProxyCompletion(backoff_base=0.5, backoff_cap=30.0)
    assert p._retry_delay(_http_error(429), attempt=0) == 0.5
    assert p._retry_delay(_http_error(429), attempt=2) == 2.0        # 0.5 * 2**2
    # garbage Retry-After (HTTP-date form) → backoff, not a crash
    assert p._retry_delay(_http_error(429, retry_after="Wed, 21 Oct"), attempt=1) == 1.0


# ── _post retry behaviour ─────────────────────────────────────────────────────

def test_post_retries_429_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    p = _proxy(monkeypatch, [_http_error(429), _http_error(429), _FakeResp("ok")], sleeps)
    body, _ = p._post(_REQ)
    assert body == "ok"
    assert len(sleeps) == 2           # two retries before the success


def test_post_honors_retry_after_header(monkeypatch):
    sleeps: list[float] = []
    p = _proxy(monkeypatch, [_http_error(429, retry_after="3"), _FakeResp("ok")], sleeps)
    p._post(_REQ)
    assert sleeps == [3.0]


def test_post_gives_up_after_max_retries(monkeypatch):
    sleeps: list[float] = []
    p = _proxy(monkeypatch, [_http_error(429)] * 4, sleeps)   # max_retries=3 → 4 attempts
    with pytest.raises(urllib.error.HTTPError):
        p._post(_REQ)
    assert len(sleeps) == 3


def test_post_does_not_retry_non_retryable_4xx(monkeypatch):
    sleeps: list[float] = []
    p = _proxy(monkeypatch, [_http_error(400), _FakeResp("unreached")], sleeps)
    with pytest.raises(urllib.error.HTTPError):
        p._post(_REQ)
    assert sleeps == []               # 400 fails fast, no backoff


def test_post_retries_transient_5xx(monkeypatch):
    sleeps: list[float] = []
    p = _proxy(monkeypatch, [_http_error(503), _FakeResp("ok")], sleeps)
    assert p._post(_REQ)[0] == "ok"
    assert len(sleeps) == 1


def test_post_retries_connection_blip(monkeypatch):
    sleeps: list[float] = []
    p = _proxy(monkeypatch, [urllib.error.URLError("conn refused"), _FakeResp("ok")], sleeps)
    assert p._post(_REQ)[0] == "ok"
    assert len(sleeps) == 1
