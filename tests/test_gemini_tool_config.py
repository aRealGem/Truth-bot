"""Pin force-tool-use (``function_calling_config.mode='ANY'``) on every Gemini call site.

2026-05-23 substance-track companion to the OpenAI ``tool_choice='required'``
fix. Same failure mode (temporal-dismissal: post-cutoff events returned as
Unverifiable without firing a search query), same shape (nudge the model
to invoke the grounding tool on every call).

The Gemini knob lives in ``types.ToolConfig(function_calling_config=
types.FunctionCallingConfig(mode='ANY'))``. Google's API rejects this knob
on a per-request ``GenerateContentConfig`` when ``cached_content`` is set
(see test_gemini_cache.py), so on the cached-content path the knob must
live on the ``CachedContent`` itself. These tests pin both placements.

Caveat documented elsewhere: ANY mode is documented for function-call
tools, not for the built-in ``google_search`` grounding tool. If Gemini
ignores it for grounding, the prompt-side temporal-anchoring revision in
SYNTHESIS_SYSTEM is the bigger lever. Either way, plumbing the knob is
harmless and keeps the surface symmetric with OpenAI's.
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

    monkeypatch.setattr(GeminiAdapter, "_cached_content_names", {})


def _fake_types_module():
    """Mirror of tests/test_gemini_cache.py's _fake_types_module."""
    types_mod = MagicMock()

    def tool_ctor(*args, **kwargs):
        t = MagicMock(name="Tool")
        t.google_search = kwargs.get("google_search")
        return t

    types_mod.Tool.side_effect = tool_ctor

    def google_search_ctor(*args, **kwargs):
        return MagicMock(name="GoogleSearch")

    types_mod.GoogleSearch.side_effect = google_search_ctor

    def function_calling_config_ctor(**kwargs):
        inst = MagicMock(name="FunctionCallingConfig")
        for k, v in kwargs.items():
            setattr(inst, k, v)
        inst._kwargs = kwargs
        return inst

    types_mod.FunctionCallingConfig.side_effect = function_calling_config_ctor

    def tool_config_ctor(**kwargs):
        inst = MagicMock(name="ToolConfig")
        for k, v in kwargs.items():
            setattr(inst, k, v)
        inst._kwargs = kwargs
        return inst

    types_mod.ToolConfig.side_effect = tool_config_ctor

    def create_cached_ctor(**kwargs):
        inst = MagicMock(name="CreateCachedContentConfig")
        for k, v in kwargs.items():
            setattr(inst, k, v)
        inst._kwargs = kwargs
        return inst

    types_mod.CreateCachedContentConfig.side_effect = create_cached_ctor

    def gen_config_ctor(**kwargs):
        inst = MagicMock(name="GenerateContentConfig")
        for k, v in kwargs.items():
            setattr(inst, k, v)
        inst._kwargs = kwargs
        return inst

    types_mod.GenerateContentConfig.side_effect = gen_config_ctor

    return types_mod


def test_batch_payload_carries_tool_config_any(env_with_key) -> None:
    """``build_batch_payload`` is a dict-shaped Vertex/Gemini batch row.

    The native batch transport is currently sidelined (``supports_batch=False``
    for now), but pin the contract so it's wired correctly when batch lands.
    """
    from truthbot.verify.adapters.gemini import GeminiAdapter

    adapter = GeminiAdapter()
    claim = Claim(transcript_id="t1", text="test", speaker="Test")
    payload = adapter.build_batch_payload(claim, evidence=[])

    assert payload["tool_config"] == {
        "function_calling_config": {"mode": "ANY"}
    }, "Batch payload must force tool use to match the OpenAI tool_choice=required surface."


def test_cache_create_config_carries_tool_config_any(env_with_key) -> None:
    """``CreateCachedContentConfig`` must carry ``tool_config`` so cache-hit
    paths inherit the force-tool-use nudge.

    Per Google's API contract, ``tool_config`` on a per-request
    ``GenerateContentConfig`` with ``cached_content`` set is rejected with::

        CachedContent can not be used with GenerateContent request setting
        system_instruction, tools or tool_config.

    So the only valid spot for ``tool_config`` on a cached-content workflow
    is here.
    """
    from truthbot.verify.adapters.gemini import GeminiAdapter

    adapter = GeminiAdapter()
    fake_client = MagicMock()
    cached = MagicMock()
    cached.name = "caches/truthbot-rubric-xyz"
    fake_client.caches.create.return_value = cached
    types_mod = _fake_types_module()

    adapter._get_or_create_cached_content(fake_client, types_mod)

    cache_config = fake_client.caches.create.call_args.kwargs["config"]
    assert "tool_config" in cache_config._kwargs, (
        "CreateCachedContentConfig must carry tool_config — "
        "without it, the force-tool-use nudge is lost on every cache hit"
    )
    tool_config = cache_config._kwargs["tool_config"]
    fcc = tool_config.function_calling_config
    assert fcc.mode == "ANY"


def test_non_cached_call_path_sets_tool_config_any(env_with_key) -> None:
    """When the CachedContent handle is unavailable, the inline
    ``GenerateContentConfig`` must carry ``tool_config`` directly.

    This is the resilience path (e.g. cache creation failed due to quota
    or the tier doesn't support caching): single-claim ``call()`` must
    still force the model to invoke ``google_search``.
    """
    from truthbot.verify.adapters.gemini import GeminiAdapter

    adapter = GeminiAdapter()

    fake_response = MagicMock()
    fake_response.candidates = []
    fake_response.text = (
        '{"label": "True", "confidence": "High", "explanation": "x", "web_sources": []}'
    )
    fake_response.usage_metadata = MagicMock(
        prompt_token_count=100,
        candidates_token_count=50,
        cached_content_token_count=0,
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    fake_client.caches.create.side_effect = RuntimeError("no cache for you")

    types_mod = _fake_types_module()
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_genai.types = types_mod
    fake_google = MagicMock()
    fake_google.genai = fake_genai

    claim = Claim(transcript_id="t1", text="The Sun is a star.", speaker="Tester")

    with patch.dict(
        "sys.modules",
        {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": types_mod,
        },
    ):
        adapter.call(claim, evidence=[], inject_evidence=False)

    gen_config = fake_client.models.generate_content.call_args.kwargs["config"]
    assert "tool_config" in gen_config._kwargs, (
        "Non-cached fallback path must set tool_config inline on the "
        "GenerateContentConfig — otherwise the force-tool-use nudge "
        "silently disappears whenever cache creation fails"
    )
    tool_config = gen_config._kwargs["tool_config"]
    assert tool_config.function_calling_config.mode == "ANY"


def test_cached_call_path_does_not_set_tool_config(env_with_key) -> None:
    """Regression guard against a hard Google API error.

    With ``cached_content`` set, ``tool_config`` on the per-request
    ``GenerateContentConfig`` triggers::

        CachedContent can not be used with GenerateContent request setting
        system_instruction, tools or tool_config.

    The nudge must live on the CachedContent (pinned above), NOT here.
    """
    from truthbot.verify.adapters.gemini import GeminiAdapter

    adapter = GeminiAdapter()
    GeminiAdapter._cached_content_names = {
        adapter._active_model: "caches/truthbot-rubric-xyz",
    }

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

    claim = Claim(transcript_id="t1", text="The Sun is a star.", speaker="Tester")

    with patch.dict(
        "sys.modules",
        {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": types_mod,
        },
    ):
        adapter.call(claim, evidence=[], inject_evidence=False)

    gen_config = fake_client.models.generate_content.call_args.kwargs["config"]
    assert "cached_content" in gen_config._kwargs
    assert "tool_config" not in gen_config._kwargs, (
        "tool_config on a per-request GenerateContentConfig is a hard API "
        "error when cached_content is set — it must live on the CachedContent"
    )


def test_non_cached_call_multi_sets_tool_config_any(env_with_key) -> None:
    """Same nudge on the multi-claim non-cached fallback path."""
    from truthbot.verify.adapters.gemini import GeminiAdapter

    adapter = GeminiAdapter()

    fake_response = MagicMock()
    fake_response.candidates = []
    fake_response.text = (
        '{"claims":[{"id":"c1","label":"True","confidence":"High",'
        '"explanation":"x","web_sources":[]}]}'
    )
    fake_response.usage_metadata = MagicMock(
        prompt_token_count=100,
        candidates_token_count=50,
        cached_content_token_count=0,
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    fake_client.caches.create.side_effect = RuntimeError("no cache")

    types_mod = _fake_types_module()
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_genai.types = types_mod
    fake_google = MagicMock()
    fake_google.genai = fake_genai

    claims = [Claim(transcript_id="t1", text="x", speaker="t", id="c1")]

    with patch.dict(
        "sys.modules",
        {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": types_mod,
        },
    ):
        adapter.call_multi(claims, evidence_by_claim={}, inject_evidence=False)

    gen_config = fake_client.models.generate_content.call_args.kwargs["config"]
    assert "tool_config" in gen_config._kwargs
    assert gen_config._kwargs["tool_config"].function_calling_config.mode == "ANY"
