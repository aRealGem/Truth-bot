"""CLI preflight: API-key sanity check before any spend.

The 2026-04-22 SOTU batch burn root-caused to three ``.env`` keys truncated
at 80 cols by terminal paste. ``validate_api_key`` catches every one of
those shape failures, but we never wired it into the pipeline — so the
first live request ate the error instead of the CLI.
"""

from __future__ import annotations

import pytest


class TestKeyPreflight:
    def test_passes_when_all_keys_look_valid(self, monkeypatch):
        from truthbot.pipeline import _preflight_key_sanity

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 101)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-" + "b" * 150)
        monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "c" * 36)
        monkeypatch.setenv("XAI_API_KEY", "xai-" + "d" * 80)

        _preflight_key_sanity()

    def test_passes_when_keys_missing(self, monkeypatch):
        from truthbot.pipeline import _preflight_key_sanity

        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "XAI_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

        _preflight_key_sanity()

    def test_aborts_on_truncated_anthropic_key(self, monkeypatch):
        from truthbot.pipeline import _preflight_key_sanity

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-tooshort")
        with pytest.raises(SystemExit) as excinfo:
            _preflight_key_sanity()
        assert "ANTHROPIC_API_KEY" in str(excinfo.value)

    def test_aborts_on_trailing_redirect_char(self, monkeypatch):
        """The exact failure mode that burned the 2026-04-22 SOTU run."""
        from truthbot.pipeline import _preflight_key_sanity

        monkeypatch.setenv(
            "ANTHROPIC_API_KEY", "sk-ant-" + "a" * 100 + ">"
        )
        with pytest.raises(SystemExit) as excinfo:
            _preflight_key_sanity()
        assert "ANTHROPIC_API_KEY" in str(excinfo.value)
        assert ">" in str(excinfo.value) or "truncated" in str(excinfo.value).lower()

    def test_aborts_on_wrong_prefix(self, monkeypatch):
        from truthbot.pipeline import _preflight_key_sanity

        monkeypatch.setenv("OPENAI_API_KEY", "AIza" + "x" * 50)
        with pytest.raises(SystemExit) as excinfo:
            _preflight_key_sanity()
        assert "OPENAI_API_KEY" in str(excinfo.value)

    def test_accepts_new_gemini_aq_prefix(self, monkeypatch):
        """Regression for 2026-04-23: the repo's real ``GEMINI_API_KEY``
        starts with ``AQ.`` (Google's 2025+ AI Studio key format). A preflight
        that only knows the legacy ``AIza`` prefix would abort on a working
        key — which is exactly what would have happened on the SOTU rerun
        before this fix landed."""
        from truthbot.pipeline import _preflight_key_sanity

        monkeypatch.setenv(
            "GEMINI_API_KEY",
            "AQ.Ab" + ("8RN6KMzZpsoEN29lQLpzsmKdZ4BOaBRahLBX6aPKQhi0QlRg"),
        )
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        _preflight_key_sanity()  # must not raise

    def test_reports_all_failing_keys_in_one_message(self, monkeypatch):
        from truthbot.pipeline import _preflight_key_sanity

        monkeypatch.setenv("ANTHROPIC_API_KEY", "bogus")
        monkeypatch.setenv("XAI_API_KEY", "also-bogus")
        for var in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(SystemExit) as excinfo:
            _preflight_key_sanity()
        msg = str(excinfo.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "XAI_API_KEY" in msg
