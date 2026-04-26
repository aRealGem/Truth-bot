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

    def test_fallback_chain_starts_with_subclass_model_id(self, env_with_key):
        """Regression for the Phase 3a calibration finding: 9/10 Anthropic
        triage calls silently used ``claude-opus-4-7`` because
        ``_call_with_fallback`` iterated the hard-coded ``_FALLBACK_MODELS``
        list (Opus-first) instead of honoring the ``TriageAnthropic`` subclass
        override of ``model_id``. Cost impact was ~25× per triage call.

        The fallback chain must start with ``self.model_id`` so that
        triage subclasses (and env-var overrides) actually exercise their
        configured cheap-tier model on the very first attempt.
        """
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        class TriageStub(AnthropicAdapter):
            model_id = "claude-haiku-4-5"

        ok_response = MagicMock()
        client = _make_client_sequence(ok_response)

        adapter = TriageStub()
        resp = adapter._call_with_fallback(client, "user text")
        assert resp is ok_response

        first_call_model = client.messages.create.call_args_list[0].kwargs.get("model")
        assert first_call_model == "claude-haiku-4-5", (
            f"Triage subclass override must drive the first request; "
            f"got {first_call_model!r}. If this fails with 'claude-opus-4-7' "
            f"the original Phase 3a triage cost bug has regressed."
        )
        assert adapter._active_model == "claude-haiku-4-5"

    def test_fallback_chain_does_not_duplicate_model_id(self, env_with_key):
        """When ``self.model_id`` is already in ``_FALLBACK_MODELS`` (the
        default base-class case), the chain must not retry it twice.
        """
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        err = _fake_not_found("primary unavailable")
        ok_response = MagicMock()
        client = _make_client_sequence(err, ok_response)

        adapter = AnthropicAdapter()
        resp = adapter._call_with_fallback(client, "user text")
        assert resp is ok_response

        attempted_models = [
            c.kwargs.get("model")
            for c in client.messages.create.call_args_list
        ]
        assert attempted_models[0] == "claude-opus-4-7"
        assert attempted_models[1] == "claude-opus-4-5"
        assert len(set(attempted_models)) == len(attempted_models), (
            f"Models attempted in chain must be unique; got {attempted_models}"
        )

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
