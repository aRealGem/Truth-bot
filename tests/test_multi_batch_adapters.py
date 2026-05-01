"""Multi-claim batch-response parsing for the AnthropicAdapter and OpenAIAdapter.

Exercises only the ``build_multi_batch_payload`` / ``parse_multi_batch_response``
pair on each adapter with hand-built response envelopes that mimic what the
real SDKs return. No live calls, no mocked SDKs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from truthbot.models import Claim, VerdictLabel
from truthbot.verify.adapters.anthropic import AnthropicAdapter
from truthbot.verify.adapters.openai import OpenAIAdapter


@pytest.fixture(autouse=True)
def _set_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _claim(text: str) -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


# ── Anthropic multi-claim envelope (mimics messages.batches.results message) ──


def _anthropic_message(
    verdicts_json: str,
    *,
    urls: list[str] | None = None,
    cached: int = 0,
    tool_calls: int = 0,
) -> dict:
    content = []
    for _ in range(tool_calls):
        content.append({"type": "server_tool_use", "name": "web_search"})
    if urls:
        content.append(
            {
                "type": "web_search_tool_result",
                "content": [{"url": u} for u in urls],
            }
        )
    content.append({"type": "text", "text": verdicts_json})
    return {
        "model": "claude-opus-4-7",
        "content": content,
        "usage": {"input_tokens": 500, "output_tokens": 200, "cache_read_input_tokens": cached},
    }


def test_anthropic_build_multi_payload_scales_max_tokens_with_n() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim(f"Claim {i}") for i in range(5)]
    payload = adapter.build_multi_batch_payload(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )
    assert payload["model"] == "claude-opus-4-7"
    assert payload["max_tokens"] >= 1024 * 5
    assert payload["messages"][0]["role"] == "user"
    assert claims[0].id in payload["messages"][0]["content"]
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_parse_multi_all_succeed() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B"), _claim("C")]
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
            {"claim_id": claims[2].id, "label": "Misleading", "confidence": "Medium", "explanation": "c"},
        ]
    )
    raw = _anthropic_message(
        verdicts_json, urls=["https://example.gov/a"], cached=42
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims, batch_call_id="anthropic::multi::xyz")

    assert [v.label for v in verdicts] == [
        VerdictLabel.TRUE,
        VerdictLabel.FALSE,
        VerdictLabel.MISLEADING,
    ]
    assert verdicts[0].batch_call_index == 0
    assert verdicts[1].batch_call_index == 1
    assert verdicts[2].batch_call_index == 2
    assert all(v.batch_call_id == "anthropic::multi::xyz" for v in verdicts)
    # Call-level usage (500 in / 200 out / 42 cached) is attributed to the
    # index-0 verdict only; siblings carry zeros so downstream cost
    # aggregation counts a single batched API call, not N.
    assert verdicts[0].input_tokens == 500
    assert verdicts[0].output_tokens == 200
    assert verdicts[0].cached_input_tokens == 42
    assert verdicts[1].input_tokens == 0
    assert verdicts[1].output_tokens == 0
    assert verdicts[1].cached_input_tokens == 0
    assert verdicts[2].input_tokens == 0
    assert verdicts[2].output_tokens == 0
    assert verdicts[2].cached_input_tokens == 0


def test_anthropic_parse_multi_partial_fills_missing_with_no_response() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B"), _claim("C")]
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[2].id, "label": "False", "confidence": "High", "explanation": "c"},
        ]
    )
    raw = _anthropic_message(verdicts_json)
    verdicts = adapter.parse_multi_batch_response(raw, claims)

    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.UNVERIFIABLE
    assert verdicts[1].no_response is True
    assert "partial" in verdicts[1].explanation.lower()
    assert verdicts[2].label == VerdictLabel.FALSE


def test_anthropic_parse_multi_malformed_json_marks_all_as_no_response() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B")]
    raw = _anthropic_message("this is not json")
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert all(v.no_response for v in verdicts)
    assert all(v.label == VerdictLabel.UNVERIFIABLE for v in verdicts)


def test_anthropic_parse_multi_reordered_ids_still_key_correctly() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B"), _claim("C")]
    # Model returned them in reverse; IDs still line up.
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[2].id, "label": "Misleading", "confidence": "Medium"},
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    raw = _anthropic_message(verdicts_json)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.FALSE
    assert verdicts[2].label == VerdictLabel.MISLEADING


def test_anthropic_parse_multi_backfills_web_sources_on_first_verdict() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B")]
    # Model omitted per-verdict web_sources; envelope has harvested URLs.
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    raw = _anthropic_message(
        verdicts_json, urls=["https://example.gov/x", "https://example.gov/y"]
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].web_sources == ["https://example.gov/x", "https://example.gov/y"]
    assert verdicts[1].web_sources == []


# ── OpenAI multi-claim envelope (mimics Responses API body) ───────────────────


def _openai_body(
    text: str,
    *,
    cached: int = 0,
    input_tokens: int = 1200,
    output_tokens: int = 450,
    tool_calls: int = 0,
    open_page_urls: list[str] | None = None,
    action_sources: list[list[str]] | None = None,
    annotations: list[dict] | None = None,
) -> SimpleNamespace:
    """Build a fake Responses API body.

    Args:
        tool_calls: number of plain ``search``-action ``web_search_call`` items
            (no URLs surfaced — matches the bulk of the real ed7be4ad-… run).
        open_page_urls: each URL spawns one extra ``web_search_call`` with
            ``action.type == 'open_page'`` and ``action.url`` set. Mirrors the
            documented GA web_search "the model directly fetched this URL"
            shape that the legacy parser ignored.
        action_sources: each inner list spawns one ``web_search_call`` whose
            ``action.sources[]`` carries those URLs (defensive coverage for the
            documented but unobserved-in-this-run SERP-source shape).
        annotations: list of ``{url, type?}`` dicts attached to the single
            ``output_text`` block. Covers ``url_citation`` typed annotations
            and the legacy bare ``{url: ...}`` shape.
    """
    output: list[SimpleNamespace] = []
    for i in range(tool_calls):
        output.append(
            SimpleNamespace(
                type="web_search_call",
                id=f"ws_search_{i}",
                action=SimpleNamespace(type="search", queries=["q"], query="q"),
            )
        )
    for i, url in enumerate(open_page_urls or []):
        output.append(
            SimpleNamespace(
                type="web_search_call",
                id=f"ws_open_{i}",
                action=SimpleNamespace(type="open_page", url=url),
            )
        )
    for i, sources in enumerate(action_sources or []):
        output.append(
            SimpleNamespace(
                type="web_search_call",
                id=f"ws_sources_{i}",
                action=SimpleNamespace(
                    type="search",
                    queries=["q"],
                    query="q",
                    sources=[SimpleNamespace(url=u) for u in sources],
                ),
            )
        )
    ann_blocks = [
        SimpleNamespace(
            url=ann["url"],
            type=ann.get("type", "url_citation"),
        )
        for ann in (annotations or [])
    ]
    output.append(
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(type="output_text", text=text, annotations=ann_blocks)
            ],
        )
    )
    return SimpleNamespace(
        model="gpt-5.4",
        output=output,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


def test_openai_build_multi_payload_scales_tool_budget_with_n() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim(f"Claim {i}") for i in range(4)]
    payload = adapter.build_multi_batch_payload(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )
    assert payload["model"] == "gpt-5.4"
    assert payload["max_tool_calls"] == 2 * 4
    assert payload["max_output_tokens"] >= 1024 * 4
    # Prompt-cache parity: the OpenAI system text must match OPENAI_SYNTHESIS_SYSTEM exactly.
    from truthbot.verify.adapters.base import OPENAI_SYNTHESIS_SYSTEM

    assert payload["input"][0]["content"][0]["text"] == OPENAI_SYNTHESIS_SYSTEM
    # Phase 2.5a sentinel: must use GA ``web_search`` tool, not legacy ``web_search_preview``.
    assert payload["tools"] == [{"type": "web_search"}]


def test_openai_build_single_batch_payload_uses_ga_web_search_tool() -> None:
    """Phase 2.5a sentinel for the single-claim batch path."""
    adapter = OpenAIAdapter()
    claim = _claim("A single claim.")
    payload = adapter.build_batch_payload(claim, [], inject_evidence=False)
    assert payload["tools"] == [{"type": "web_search"}]


def test_openai_parse_multi_all_succeed() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
        ]
    )
    raw = _openai_body(text, cached=321, input_tokens=1500, output_tokens=600)
    verdicts = adapter.parse_multi_batch_response(
        raw, claims, batch_call_id="openai::multi::9"
    )
    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.FALSE
    # Call-level usage lands on index-0 only; sibling carries zero so the
    # single batched API call is billed once, not N times.
    assert verdicts[0].input_tokens == 1500
    assert verdicts[0].output_tokens == 600
    assert verdicts[0].cached_input_tokens == 321
    assert verdicts[1].input_tokens == 0
    assert verdicts[1].output_tokens == 0
    assert verdicts[1].cached_input_tokens == 0
    assert all(v.batch_call_id == "openai::multi::9" for v in verdicts)


def test_openai_parse_multi_malformed_marks_all_no_response() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    raw = _openai_body("garbage not json")
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert all(v.no_response for v in verdicts)


def test_openai_parse_multi_duplicated_claim_id_uses_first() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[0].id, "label": "False", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "Mostly True", "confidence": "Medium"},
        ]
    )
    raw = _openai_body(text)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    # First occurrence wins for claims[0]; claims[1] still gets its own verdict.
    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.MOSTLY_TRUE


def test_anthropic_parse_multi_counts_server_tool_use_blocks() -> None:
    """Fix for C6 — Anthropic multi-claim batch must count each
    ``server_tool_use`` content block and attribute the total to the
    index-0 verdict so ``tool_call_count`` telemetry is non-zero on
    successful batched web_search runs."""
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B"), _claim("C")]
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
            {"claim_id": claims[2].id, "label": "Misleading", "confidence": "Medium"},
        ]
    )
    raw = _anthropic_message(verdicts_json, tool_calls=4)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].tool_call_count == 4
    assert verdicts[1].tool_call_count == 0
    assert verdicts[2].tool_call_count == 0


def test_anthropic_parse_multi_zero_tool_calls_stays_zero() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B")]
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    raw = _anthropic_message(verdicts_json)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert all(v.tool_call_count == 0 for v in verdicts)


def test_openai_parse_multi_counts_web_search_call_items() -> None:
    """Fix for C6 — OpenAI multi-claim batch must count every
    ``web_search_call`` output item and attribute the total to the
    index-0 verdict."""
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    raw = _openai_body(text, tool_calls=3)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].tool_call_count == 3
    assert verdicts[1].tool_call_count == 0


def test_openai_parse_multi_zero_tool_calls_stays_zero() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    raw = _openai_body(text)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert all(v.tool_call_count == 0 for v in verdicts)


def test_adapters_expose_multi_claim_caps() -> None:
    assert AnthropicAdapter.max_claims_per_request >= 2
    assert OpenAIAdapter.max_claims_per_request >= 2
    assert AnthropicAdapter.supports_batch is True
    assert OpenAIAdapter.supports_batch is True


def test_live_multi_claim_adapters_expose_caps() -> None:
    """Phase E — Grok/Gemini live claim-batching raises caps past 1.

    Unlike Anthropic/OpenAI (native batch API), these two run live with
    ``supports_batch=False`` but opt into live multi-claim via
    ``call_multi`` + elevated ``max_claims_per_request``.
    """
    from truthbot.verify.adapters.gemini import GeminiAdapter
    from truthbot.verify.adapters.grok import GrokAdapter

    assert GrokAdapter.max_claims_per_request >= 2
    assert GeminiAdapter.max_claims_per_request >= 2
    assert GrokAdapter.supports_batch is False
    assert GeminiAdapter.supports_batch is False


# ── OpenAI tool-URL extraction (regression for the ed7be4ad-… 100% gap) ──────


def test_openai_parse_multi_collects_open_page_url_from_action() -> None:
    """Real ed7be4ad-… batch shape — the only URLs surfaced live on
    ``web_search_call.action.url`` for ``open_page``-action items. The
    legacy parser missed this surface entirely so every model-cited URL
    got stripped (100% fabrication-rate readout). Locks the fix in place.
    """
    adapter = OpenAIAdapter()
    claims = [_claim("A")]
    text = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "False",
                "confidence": "High",
                "explanation": "x",
                "web_sources": [
                    "https://www.bls.gov/news.release/archives/cpi_01132026.htm",
                    "https://www.bls.gov/fabricated.htm",
                ],
            },
        ]
    )
    raw = _openai_body(
        text,
        tool_calls=2,
        open_page_urls=[
            "https://www.bls.gov/news.release/archives/cpi_01132026.htm"
        ],
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].web_sources == [
        "https://www.bls.gov/news.release/archives/cpi_01132026.htm"
    ]
    assert verdicts[0].model_reported_sources == [
        "https://www.bls.gov/news.release/archives/cpi_01132026.htm",
        "https://www.bls.gov/fabricated.htm",
    ]
    assert verdicts[0].stripped_source_count == 1
    assert verdicts[0].tool_call_count == 3  # 2 search + 1 open_page


def test_openai_parse_multi_collects_action_sources_urls() -> None:
    """Defensive coverage — Responses API documents a SERP-style
    ``action.sources[].url`` shape that did not appear in the observed
    batch but must keep grounding correctly if the API surfaces it.
    """
    adapter = OpenAIAdapter()
    claims = [_claim("A")]
    text = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "True",
                "confidence": "High",
                "explanation": "x",
                "web_sources": [
                    "https://example.com/serp-1",
                    "https://example.com/serp-2",
                    "https://example.com/fabricated",
                ],
            },
        ]
    )
    raw = _openai_body(
        text,
        action_sources=[
            ["https://example.com/serp-1", "https://example.com/serp-2"],
        ],
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].web_sources == [
        "https://example.com/serp-1",
        "https://example.com/serp-2",
    ]
    assert verdicts[0].stripped_source_count == 1


def test_openai_parse_multi_collects_url_citation_annotations() -> None:
    """Live Responses API still emits citations on ``output_text``
    annotations; the type is typically ``url_citation`` post-GA. The
    helper must keep collecting those alongside the new action surfaces.
    """
    adapter = OpenAIAdapter()
    claims = [_claim("A")]
    text = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "True",
                "confidence": "High",
                "explanation": "x",
                "web_sources": [
                    "https://example.com/citation-1",
                    "https://example.com/citation-2",
                    "https://example.com/fabricated",
                ],
            },
        ]
    )
    raw = _openai_body(
        text,
        annotations=[
            {"url": "https://example.com/citation-1", "type": "url_citation"},
            {"url": "https://example.com/citation-2", "type": "url_citation"},
        ],
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].web_sources == [
        "https://example.com/citation-1",
        "https://example.com/citation-2",
    ]
    assert verdicts[0].stripped_source_count == 1


def test_openai_parse_multi_unions_all_url_surfaces() -> None:
    """All three URL surfaces on the same body are unioned, deduped, and
    used as the ground-truth set for the intersection."""
    adapter = OpenAIAdapter()
    claims = [_claim("A")]
    text = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "Mostly True",
                "confidence": "High",
                "explanation": "x",
                "web_sources": [
                    "https://example.com/open-page",
                    "https://example.com/serp",
                    "https://example.com/citation",
                    "https://example.com/never-grounded",
                ],
            },
        ]
    )
    raw = _openai_body(
        text,
        open_page_urls=["https://example.com/open-page"],
        action_sources=[["https://example.com/serp"]],
        annotations=[{"url": "https://example.com/citation"}],
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].web_sources == [
        "https://example.com/open-page",
        "https://example.com/serp",
        "https://example.com/citation",
    ]
    assert verdicts[0].stripped_source_count == 1


def test_openai_parse_multi_trusts_model_when_search_fired_but_extraction_empty() -> None:
    """Trust-when-fired (2026-05-01): when ``web_search_call`` tools fired
    (``tool_call_count > 0``) but every action was a pure ``search`` (no
    URLs anywhere), the model's emitted ``web_sources`` are kept
    rather than stripped. The strip-100% behavior was a harness artefact
    of OpenAI's Responses API JSON-output mode (no inline url_citation
    annotations), not genuine fabrication. xAI / Anthropic confirm 0%
    strip is correct when extraction works. The reader-side
    "Model-cited (unverified)" tier (publish/site.py) caveats anything
    that would slip through. See
    metrics/adapter_interpretability/strip_audit_2026-05.md."""
    adapter = OpenAIAdapter()
    claims = [_claim("A")]
    text = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "True",
                "confidence": "High",
                "explanation": "x",
                "web_sources": [
                    "https://www.bls.gov/news.release/cpi.htm",
                ],
            },
        ]
    )
    raw = _openai_body(text, tool_calls=3)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    # Trust-when-fired: tool ran (3 calls) but no URLs surfaced through
    # the API surface → model's URL is kept, not stripped.
    assert verdicts[0].web_sources == [
        "https://www.bls.gov/news.release/cpi.htm"
    ]
    assert verdicts[0].model_reported_sources == verdicts[0].web_sources
    assert verdicts[0].stripped_source_count == 0


def test_openai_parse_multi_strips_when_tool_did_not_fire() -> None:
    """Counter-test to trust-when-fired: when ``tool_call_count == 0`` —
    e.g., the model declined to invoke web_search — strict intersection
    against the empty tool set still strips. The fallback ONLY relaxes
    intersection when tools actually ran; it does NOT weaken
    anti-fabrication for runs where the tool never fired."""
    adapter = OpenAIAdapter()
    claims = [_claim("A")]
    text = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "True",
                "confidence": "High",
                "explanation": "x",
                "web_sources": [
                    "https://example.com/cited-without-grounding",
                ],
            },
        ]
    )
    raw = _openai_body(text, tool_calls=0)  # tool never invoked
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    # Strict strip stands when no tools fired.
    assert verdicts[0].web_sources == []
    assert verdicts[0].model_reported_sources == [
        "https://example.com/cited-without-grounding"
    ]
    assert verdicts[0].stripped_source_count == 1


def test_openai_parse_single_batch_response_collects_open_page_url() -> None:
    """Symmetric coverage for the single-claim ``parse_batch_response``
    entry point — same helper, same surfaces."""
    adapter = OpenAIAdapter()
    claim = _claim("A")
    text = json.dumps(
        {
            "label": "False",
            "confidence": "High",
            "explanation": "x",
            "web_sources": [
                "https://example.com/grounded",
                "https://example.com/fabricated",
            ],
        }
    )
    raw = _openai_body(
        text,
        open_page_urls=["https://example.com/grounded"],
    )
    verdict = adapter.parse_batch_response(raw, claim)
    assert verdict.web_sources == ["https://example.com/grounded"]
    assert verdict.stripped_source_count == 1


def test_openai_parse_multi_uses_real_open_page_fixture() -> None:
    """Loads the sanitized real-batch fixture (excerpt of the ed7be4ad-…
    body) and asserts the observed URL flows through grounding non-empty.
    Guards against any future regression that re-strips this surface."""
    from pathlib import Path

    fixture = (
        Path(__file__).parent / "fixtures" / "openai_batch_response_with_open_page.json"
    )
    body = json.loads(fixture.read_text())
    adapter = OpenAIAdapter()
    claim = Claim(
        id="a11e4bdb-acdd-43a4-b762-5998d92431de",
        transcript_id="t1",
        text="A",
        speaker="Test",
    )
    verdicts = adapter.parse_multi_batch_response(body, [claim])
    assert len(verdicts) == 1
    assert (
        "https://www.bls.gov/news.release/archives/cpi_01132026.htm"
        in verdicts[0].web_sources
    )
    assert verdicts[0].stripped_source_count >= 1
    assert verdicts[0].tool_call_count == 3
