"""Regression tests for ``AnthropicAdapter._call_with_fallback`` warning logic.

The prior implementation compared the iteration model against ``self.model_id``
(the immutable class default) instead of the actually-attempted model, so once
``_active_model`` drifted to a fallback, every subsequent claim logged a
spurious ``"model X not available, trying X"`` pair.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

# anthropic package is a project dependency, import at top level
import anthropic


@pytest.fixture
def env_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 100)


def _make_client_sequence(*responses):
    """Return a MagicMock whose messages.create returns (or raises) per call."""
    client = MagicMock()

    def side_effect(*_, **__):
        item = responses_iter.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    responses_iter = list(responses)
    client.messages.create.side_effect = side_effect
    return client


def _fake_not_found(msg: str = "model not found") -> anthropic.NotFoundError:
    """Instantiate a NotFoundError without making a real request."""
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    return anthropic.NotFoundError(
        message=msg,
        response=mock_response,
        body={"error": {"message": msg}},
    )


class TestAnthropicFallbackWarning:
    def test_primary_succeeds_no_warning(self, env_with_key, caplog):
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        ok_response = MagicMock()
        client = _make_client_sequence(ok_response)

        adapter = AnthropicAdapter()
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.anthropic"):
            resp = adapter._call_with_fallback(client, "user text")

        assert resp is ok_response
        fallback_msgs = [
            r.getMessage()
            for r in caplog.records
            if "falling back" in r.getMessage() or "not available" in r.getMessage()
        ]
        assert fallback_msgs == [], (
            f"No fallback warning should fire when primary succeeds; got: {fallback_msgs}"
        )

    def test_primary_fails_then_fallback_succeeds_logs_once_with_real_models(
        self, env_with_key, caplog
    ):
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        err = _fake_not_found("model claude-opus-4-7 not found")
        ok_response = MagicMock()
        client = _make_client_sequence(err, ok_response)

        adapter = AnthropicAdapter()
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.anthropic"):
            resp = adapter._call_with_fallback(client, "user text")

        assert resp is ok_response
        fallback_msgs = [
            r.getMessage()
            for r in caplog.records
            if "falling back" in r.getMessage()
        ]
        assert len(fallback_msgs) == 1, (
            f"Exactly one fallback warning expected, got {len(fallback_msgs)}: {fallback_msgs}"
        )
        msg = fallback_msgs[0]
        assert "claude-opus-4-7" in msg
        assert "claude-opus-4-5" in msg
        assert "claude-opus-4-7" != "claude-opus-4-5"

    def test_repeat_call_does_not_log_spurious_same_model_fallback(
        self, env_with_key, caplog
    ):
        """After a previous call drifted ``_active_model`` to a fallback, a
        subsequent successful primary call must NOT log 'model X ... trying X'.
        """
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()
        # Simulate prior drift
        adapter._active_model = "claude-opus-4-5"

        ok_response = MagicMock()
        client = _make_client_sequence(ok_response)

        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.anthropic"):
            resp = adapter._call_with_fallback(client, "user text")

        assert resp is ok_response
        same_model_msgs = [
            r.getMessage()
            for r in caplog.records
            if "falling back" in r.getMessage() or "not available" in r.getMessage()
        ]
        assert same_model_msgs == [], (
            "No warning should fire when the very first iteration succeeds, "
            "regardless of what _active_model was set to previously. "
            f"Got: {same_model_msgs}"
        )
