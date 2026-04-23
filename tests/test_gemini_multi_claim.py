"""Red-tests for ``GeminiAdapter.call_multi`` (live claim-batching, Phase E slice).

Auto-skipped until ``GeminiAdapter`` raises ``max_claims_per_request`` past 1.
The tests exercise the live (non-batch-API) path because Gemini's native
batch API forbids GoogleSearch grounding — running multi-claim via the live
API with CachedContent reuse is the only way to amortize the rubric tokens
without losing grounding (see ``PROJECT_BOARD.md`` → "Gemini native batch
transport (blocked on Google)").
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthbot.models import Claim, VerdictLabel
from truthbot.verify.adapters.gemini import GeminiAdapter


pytestmark = pytest.mark.skipif(
    GeminiAdapter.max_claims_per_request < 2,
    reason="pending Phase E Gemini live multi-claim override",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def env_with_key(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "x" * 36)


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch) -> None:
    """Clear the process-wide cache-name singleton between tests."""
    monkeypatch.setattr(GeminiAdapter, "_cached_content_name", None)


def _claim(text: str) -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


def _fake_types_module() -> MagicMock:
    """Mock google.genai.types with the attributes the adapter uses."""
    types_mod = MagicMock()

    def tool_ctor(*args, **kwargs):
        t = MagicMock(name="Tool")
        t.google_search = kwargs.get("google_search")
        return t

    types_mod.Tool.side_effect = tool_ctor
    types_mod.GoogleSearch.side_effect = lambda *a, **kw: MagicMock(name="GoogleSearch")

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


def _fake_generate_response(
    verdicts_json: str,
    *,
    urls: list[str] | None = None,
    prompt_tokens: int = 1200,
    candidates_tokens: int = 500,
    cached_tokens: int = 0,
) -> MagicMock:
    """Build a fake GenerateContentResponse with grounding metadata + usage."""
    response = MagicMock()
    response.text = verdicts_json

    # Grounding metadata → grounding_chunks[].web.uri
    candidate = MagicMock()
    gm = MagicMock()
    gm.web_search_queries = []
    chunks = []
    for u in urls or []:
        chunk = MagicMock()
        chunk.web = MagicMock(uri=u)
        chunks.append(chunk)
    gm.grounding_chunks = chunks
    candidate.grounding_metadata = gm
    candidate.content = MagicMock(parts=[MagicMock(text=verdicts_json)])
    response.candidates = [candidate]

    response.usage_metadata = MagicMock(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidates_tokens,
        cached_content_token_count=cached_tokens,
    )
    return response


def _install_fake_genai(
    response: MagicMock,
) -> tuple[MagicMock, MagicMock, Any]:
    """Patch ``sys.modules`` so ``from google import genai`` yields our mocks."""
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = response
    cached = MagicMock()
    cached.name = "caches/truthbot-rubric-multi"
    fake_client.caches.create.return_value = cached

    types_mod = _fake_types_module()
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_genai.types = types_mod

    fake_google = MagicMock()
    fake_google.genai = fake_genai

    patcher = patch.dict(
        "sys.modules",
        {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": types_mod,
        },
    )
    return fake_client, types_mod, patcher


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_gemini_call_multi_single_generate_content_call(env_with_key) -> None:
    """One multi-claim API call → one ``generate_content`` invocation for N claims."""
    claims = [_claim("A"), _claim("B"), _claim("C")]
    text = json.dumps(
        [
            {"claim_id": c.id, "label": "True", "confidence": "High", "explanation": "x"}
            for c in claims
        ]
    )
    response = _fake_generate_response(text)
    fake_client, _types, patcher = _install_fake_genai(response)

    with patcher:
        adapter = GeminiAdapter()
        verdicts = adapter.call_multi(
            claims, {c.id: [] for c in claims}, inject_evidence=False
        )

    assert fake_client.models.generate_content.call_count == 1, (
        "call_multi must issue exactly one generate_content call for N claims"
    )
    assert [v.claim_id for v in verdicts] == [c.id for c in claims]
    assert all(v.label == VerdictLabel.TRUE for v in verdicts)
    assert all(v.adapter_name == "gemini" for v in verdicts)


def test_gemini_call_multi_reuses_cached_content_on_second_call(env_with_key) -> None:
    """A second multi-claim call in the same process must hit the same cache.

    Regression guard: the existing ``_get_or_create_cached_content`` singleton
    stores the cache name on the class; ``call_multi`` must respect it.
    """
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": c.id, "label": "True", "confidence": "High", "explanation": "x"}
            for c in claims
        ]
    )
    response = _fake_generate_response(text)
    fake_client, _types, patcher = _install_fake_genai(response)

    with patcher:
        adapter = GeminiAdapter()
        adapter.call_multi(claims, {c.id: [] for c in claims}, inject_evidence=False)
        adapter.call_multi(claims, {c.id: [] for c in claims}, inject_evidence=False)

    assert fake_client.caches.create.call_count == 1, (
        "Cache must be a process-wide singleton; second call_multi should reuse it"
    )
    assert fake_client.models.generate_content.call_count == 2


def test_gemini_call_multi_omits_system_instruction_when_cache_is_set(
    env_with_key,
) -> None:
    """Regression against 'CachedContent can not be used with system_instruction'.

    When the cache-name singleton is pre-seeded, the per-call
    ``GenerateContentConfig`` must reference the cache and NOT pass
    ``system_instruction`` or ``tools`` — Google rejects that combination.
    """
    GeminiAdapter._cached_content_name = "caches/truthbot-rubric-pre-seeded"
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": c.id, "label": "True", "confidence": "High"}
            for c in claims
        ]
    )
    response = _fake_generate_response(text)
    fake_client, _types, patcher = _install_fake_genai(response)

    with patcher:
        adapter = GeminiAdapter()
        adapter.call_multi(
            claims, {c.id: [] for c in claims}, inject_evidence=False
        )

    gen_config = fake_client.models.generate_content.call_args.kwargs["config"]
    assert "cached_content" in gen_config._kwargs, (
        "Cache-aware call_multi must reference the cached_content handle"
    )
    assert "system_instruction" not in gen_config._kwargs, (
        "system_instruction on the generate_content config is a hard API "
        "error when cached_content is set"
    )
    assert "tools" not in gen_config._kwargs, (
        "tools on the generate_content config is a hard API error when "
        "cached_content is set"
    )


def test_gemini_call_multi_cached_tokens_attributed_to_index_zero(
    env_with_key,
) -> None:
    """cached_content_token_count is per-response; route to index-0 only."""
    claims = [_claim("A"), _claim("B"), _claim("C")]
    text = json.dumps(
        [
            {"claim_id": c.id, "label": "True", "confidence": "High"}
            for c in claims
        ]
    )
    response = _fake_generate_response(
        text, prompt_tokens=2000, candidates_tokens=800, cached_tokens=1500
    )
    _client, _types, patcher = _install_fake_genai(response)

    with patcher:
        adapter = GeminiAdapter()
        verdicts = adapter.call_multi(
            claims, {c.id: [] for c in claims}, inject_evidence=False
        )

    assert verdicts[0].cached_input_tokens == 1500
    assert verdicts[0].input_tokens == 2000
    assert verdicts[0].output_tokens == 800
    for sibling in verdicts[1:]:
        assert sibling.cached_input_tokens == 0
        assert sibling.input_tokens == 0
        assert sibling.output_tokens == 0


def test_gemini_call_multi_backfills_grounding_urls_on_index_zero(
    env_with_key,
) -> None:
    """Harvested grounding_chunks[].web.uri land on index-0's web_sources."""
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": c.id, "label": "True", "confidence": "High"}
            for c in claims
        ]
    )
    response = _fake_generate_response(
        text, urls=["https://bls.gov/a", "https://cbo.gov/b"]
    )
    _client, _types, patcher = _install_fake_genai(response)

    with patcher:
        adapter = GeminiAdapter()
        verdicts = adapter.call_multi(
            claims, {c.id: [] for c in claims}, inject_evidence=False
        )

    assert verdicts[0].web_sources == ["https://bls.gov/a", "https://cbo.gov/b"]
    assert verdicts[1].web_sources == []


def test_gemini_call_multi_malformed_marks_all_no_response(env_with_key) -> None:
    """Garbage response → all claims get UNVERIFIABLE no_response=True."""
    claims = [_claim("A"), _claim("B")]
    response = _fake_generate_response("not-json-at-all")
    _client, _types, patcher = _install_fake_genai(response)

    with patcher:
        adapter = GeminiAdapter()
        verdicts = adapter.call_multi(
            claims, {c.id: [] for c in claims}, inject_evidence=False
        )

    assert len(verdicts) == 2
    assert all(v.no_response for v in verdicts)
    assert all(v.label == VerdictLabel.UNVERIFIABLE for v in verdicts)


def test_gemini_max_claims_per_request_raised_to_four() -> None:
    """Gemini cap documents the conservative per-call chunk size for grounding."""
    assert GeminiAdapter.max_claims_per_request >= 4, (
        "Gemini live multi-claim requires max_claims_per_request >= 4; "
        f"got {GeminiAdapter.max_claims_per_request}"
    )
