"""Regression tests for ``build_triage_adapters`` model defaults.

The 2026-04-23 SOTU run exposed two stale defaults that silently broke triage:

* ``claude-3-5-haiku-20241022`` — retired; Anthropic's fallback logic walked
  the triage class straight into ``claude-opus-4-7``, negating the entire
  cost benefit of a triage tier.
* ``grok-3-mini`` — does not accept server-side tools. xAI returns
  "only the grok-4 family of models are supported" on any triage call that
  tries to web-search.

These tests pin the current defaults AND prove env-var overrides still win.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def env_all_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 100)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 40)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "x" * 36)
    monkeypatch.setenv("XAI_API_KEY", "xai-" + "x" * 60)
    # Make sure no override from the real shell leaks in
    for var in (
        "TRUTHBOT_TRIAGE_ANTHROPIC_MODEL",
        "TRUTHBOT_TRIAGE_OPENAI_MODEL",
        "TRUTHBOT_TRIAGE_GEMINI_MODEL",
        "TRUTHBOT_TRIAGE_GROK_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


class TestTriageDefaults:
    def test_default_models_are_current(self, env_all_keys):
        from truthbot.verify.triage import build_triage_adapters

        adapters = build_triage_adapters()
        by_name = {a.adapter_name: a for a in adapters}
        # All four providers should instantiate (keys are present)
        assert set(by_name) == {"anthropic", "openai", "gemini", "xai"}

        assert by_name["anthropic"].model_id == "claude-haiku-4-5", (
            "Anthropic triage must use the current cheap tier; "
            "claude-3-5-haiku-20241022 has been retired"
        )
        assert by_name["openai"].model_id == "gpt-4o-mini"
        assert by_name["gemini"].model_id == "gemini-2.5-flash"
        assert by_name["xai"].model_id == "grok-4-fast", (
            "xAI triage must use a grok-4 family model — grok-3-mini no longer "
            "supports server-side tools"
        )

    @pytest.mark.parametrize(
        "env_var,expected_model,provider",
        [
            ("TRUTHBOT_TRIAGE_ANTHROPIC_MODEL", "claude-haiku-5-x", "anthropic"),
            ("TRUTHBOT_TRIAGE_OPENAI_MODEL", "gpt-5-nano", "openai"),
            ("TRUTHBOT_TRIAGE_GEMINI_MODEL", "gemini-3.0-flash", "gemini"),
            ("TRUTHBOT_TRIAGE_GROK_MODEL", "grok-5-mini", "xai"),
        ],
    )
    def test_env_var_override_wins(
        self, env_all_keys, monkeypatch, env_var, expected_model, provider
    ):
        monkeypatch.setenv(env_var, expected_model)

        from truthbot.verify.triage import build_triage_adapters

        adapters = build_triage_adapters()
        by_name = {a.adapter_name: a for a in adapters}
        assert by_name[provider].model_id == expected_model
