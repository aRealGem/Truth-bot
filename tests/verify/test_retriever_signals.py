"""P120 PR-2: retriever rate-limit / Max-quota signals feeding the pool governor.
Retrievers stay fail-soft (return []) but ALSO fire on_rate_limit so the governor
can pare/drop a lane. Offline: subprocess and HTTP are stubbed.
"""
from __future__ import annotations

import urllib.error

import pytest

from truthbot.verify import retrievers


# ── R1 (claude CLI) Max usage-limit sniffing ──────────────────────────────────

class _Proc:
    def __init__(self, rc, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


def test_r1_signals_on_usage_limit(monkeypatch):
    fired = []
    r = retrievers.ClaudeWorkerRetriever(on_rate_limit=lambda: fired.append(1))
    monkeypatch.setattr(retrievers.subprocess, "run",
                        lambda *a, **k: _Proc(1, stderr="Claude usage limit reached; try later"))
    assert r.shortlist("claim") == []          # still fail-soft
    assert fired == [1]                         # and signaled the governor


def test_r1_no_signal_on_ordinary_error(monkeypatch):
    fired = []
    r = retrievers.ClaudeWorkerRetriever(on_rate_limit=lambda: fired.append(1))
    monkeypatch.setattr(retrievers.subprocess, "run",
                        lambda *a, **k: _Proc(1, stderr="ENOENT: transcript not found"))
    assert r.shortlist("claim") == []
    assert fired == []                          # not a rate-limit → no signal


def test_r1_no_callback_wired_is_silent(monkeypatch):
    # Default (no governor): a rate-limit exit behaves exactly as before — [] and no crash.
    r = retrievers.ClaudeWorkerRetriever()
    monkeypatch.setattr(retrievers.subprocess, "run",
                        lambda *a, **k: _Proc(1, stderr="rate limit exceeded"))
    assert r.shortlist("claim") == []


# ── R2 / R3 provider 429 ──────────────────────────────────────────────────────

def _http_429():
    return urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)


def test_r2_signals_on_429(monkeypatch):
    fired = []
    r = retrievers.OpenAIBrowsingRetriever(on_rate_limit=lambda: fired.append(1))
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(r, "_post", lambda model, prompt: (_ for _ in ()).throw(_http_429()))
    assert r.shortlist("claim") == []
    assert fired                                # fired (once per model in the fallback chain)


def test_r2_no_signal_on_non_429(monkeypatch):
    fired = []
    r = retrievers.OpenAIBrowsingRetriever(on_rate_limit=lambda: fired.append(1))
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(r, "_post",
                        lambda model, prompt: (_ for _ in ()).throw(RuntimeError("500")))
    assert r.shortlist("claim") == []
    assert fired == []


def test_r3_signals_on_429(monkeypatch):
    fired = []
    r = retrievers.GrokSearchRetriever(on_rate_limit=lambda: fired.append(1))
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setattr(r, "_post",
                        lambda model, prompt, tool: (_ for _ in ()).throw(_http_429()))
    assert r.shortlist("claim") == []
    assert fired == [1]
