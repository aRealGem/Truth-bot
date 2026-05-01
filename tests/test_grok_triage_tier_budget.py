"""Unit coverage for tier-specific Grok ``max_tool_calls`` budgets."""

from __future__ import annotations

from truthbot.verify.adapters.grok import _max_tool_calls_for_tier


def test_max_tool_calls_for_tier_defaults(monkeypatch) -> None:
    monkeypatch.delenv("TRUTHBOT_GROK_MAX_TOOL_CALLS", raising=False)
    monkeypatch.delenv("TRUTHBOT_GROK_TRIAGE_MAX_TOOL_CALLS", raising=False)
    assert _max_tool_calls_for_tier("frontier") == 8
    assert _max_tool_calls_for_tier("triage") == 3
    assert _max_tool_calls_for_tier("anything_else") == 8


def test_max_tool_calls_for_triage_env(monkeypatch) -> None:
    monkeypatch.delenv("TRUTHBOT_GROK_MAX_TOOL_CALLS", raising=False)
    monkeypatch.setenv("TRUTHBOT_GROK_TRIAGE_MAX_TOOL_CALLS", "5")
    assert _max_tool_calls_for_tier("triage") == 5
