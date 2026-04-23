"""Regression tests for GeminiAdapter CachedContent + tools interaction.

Google's genai API rejects a ``generate_content`` call that sets
``cached_content`` alongside ``system_instruction`` or ``tools`` with:

    CachedContent can not be used with GenerateContent request setting
    system_instruction, tools or tool_config.

The fix pattern is to bind BOTH ``system_instruction`` and ``tools`` into the
``CachedContent`` at creation time, and pass neither on the per-claim
``generate_content`` request when a cache is in play.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from truthbot.models import Claim


@pytest.fixture
def env_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "x" * 36)


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    from truthbot.verify.adapters.gemini import GeminiAdapter
    monkeypatch.setattr(GeminiAdapter, "_cached_content_name", None)


def _fake_types_module():
    """Mock google.genai.types with the attributes the adapter uses."""
    types_mod = MagicMock()

    # types.Tool(google_search=...) returns a sentinel we can assert on.
    def tool_ctor(*args, **kwargs):
        t = MagicMock(name="Tool")
        t.google_search = kwargs.get("google_search")
        return t
    types_mod.Tool.side_effect = tool_ctor

    def google_search_ctor(*args, **kwargs):
        return MagicMock(name="GoogleSearch")
    types_mod.GoogleSearch.side_effect = google_search_ctor

    # CreateCachedContentConfig(**kwargs) — record kwargs on the instance.
    def create_cached_ctor(**kwargs):
        inst = MagicMock(name="CreateCachedContentConfig")
        for k, v in kwargs.items():
            setattr(inst, k, v)
        inst._kwargs = kwargs
        return inst
    types_mod.CreateCachedContentConfig.side_effect = create_cached_ctor

    # GenerateContentConfig(**kwargs) — same pattern.
    def gen_config_ctor(**kwargs):
        inst = MagicMock(name="GenerateContentConfig")
        for k, v in kwargs.items():
            setattr(inst, k, v)
        inst._kwargs = kwargs
        return inst
    types_mod.GenerateContentConfig.side_effect = gen_config_ctor

    return types_mod


class TestGeminiCachedContentConfig:
    def test_cache_create_config_includes_system_instruction_and_tools(
        self, env_with_key
    ):
        from truthbot.verify.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        fake_client = MagicMock()
        cached = MagicMock()
        cached.name = "caches/truthbot-rubric-xyz"
        fake_client.caches.create.return_value = cached
        types_mod = _fake_types_module()

        name = adapter._get_or_create_cached_content(fake_client, types_mod)

        assert name == "caches/truthbot-rubric-xyz"
        assert fake_client.caches.create.called
        cache_config = fake_client.caches.create.call_args.kwargs["config"]
        assert "system_instruction" in cache_config._kwargs, (
            "CreateCachedContentConfig must carry system_instruction "
            "(otherwise the rubric is not cached)"
        )
        assert "tools" in cache_config._kwargs, (
            "CreateCachedContentConfig must carry tools — Google rejects a "
            "GenerateContent call that passes tools alongside cached_content"
        )
        assert len(cache_config._kwargs["tools"]) >= 1

    def test_cache_create_uses_cached_name_on_second_call(self, env_with_key):
        from truthbot.verify.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        fake_client = MagicMock()
        cached = MagicMock()
        cached.name = "caches/truthbot-rubric-xyz"
        fake_client.caches.create.return_value = cached
        types_mod = _fake_types_module()

        first = adapter._get_or_create_cached_content(fake_client, types_mod)
        second = adapter._get_or_create_cached_content(fake_client, types_mod)

        assert first == second == "caches/truthbot-rubric-xyz"
        assert fake_client.caches.create.call_count == 1, (
            "CachedContent is a process-wide singleton — no second create call"
        )

    def test_cache_creation_failure_returns_none(self, env_with_key, caplog):
        from truthbot.verify.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        fake_client = MagicMock()
        fake_client.caches.create.side_effect = RuntimeError("quota exceeded")
        types_mod = _fake_types_module()

        name = adapter._get_or_create_cached_content(fake_client, types_mod)
        assert name is None

    def test_call_with_cached_content_omits_system_instruction_and_tools(
        self, env_with_key
    ):
        """Given a cache is active, the per-claim generate_content config must
        NOT set ``system_instruction`` or ``tools``."""
        from truthbot.verify.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()

        # Pre-seed the process-wide cache handle so the call() path skips
        # the create branch and exercises the cached_content-only config.
        GeminiAdapter._cached_content_name = "caches/truthbot-rubric-xyz"

        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.text = (
            '{"label": "True", "confidence": "High", "explanation": "x", "web_sources": []}'
        )
        fake_response.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=400,
        )

        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response
        types_mod = _fake_types_module()

        fake_genai = MagicMock()
        fake_genai.Client.return_value = fake_client
        fake_genai.types = types_mod

        fake_google = MagicMock()
        fake_google.genai = fake_genai

        claim = Claim(
            transcript_id="t1",
            text="The Sun is a star.",
            speaker="Tester",
        )

        with patch.dict(
            "sys.modules",
            {
                "google": fake_google,
                "google.genai": fake_genai,
                "google.genai.types": types_mod,
            },
        ):
            adapter.call(claim, evidence=[], inject_evidence=False)

        assert fake_client.models.generate_content.called
        gen_config = fake_client.models.generate_content.call_args.kwargs["config"]
        assert "cached_content" in gen_config._kwargs, (
            "When a cache exists the generate_content config must reference it"
        )
        assert "system_instruction" not in gen_config._kwargs, (
            "system_instruction on the generate_content config is a hard "
            "API error when cached_content is set"
        )
        assert "tools" not in gen_config._kwargs, (
            "tools on the generate_content config is a hard API error when "
            "cached_content is set"
        )
