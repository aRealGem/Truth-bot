"""Tests for ``OpenAIAdapter.call_multi`` (Phase 3a — promotion from Batch API).

The OpenAI Batch API was running with a 3–24h SLA in the
``ed7be4ad-…`` SOTU run, which is unworkable for iteration. Phase 3a
introduces the ``TRUTHBOT_OPENAI_LIVE`` toggle that flips OpenAI from
batch to the sidecar live path, where it joins Gemini + Grok behind the
shared multi-claim helpers (``build_multi_user_message`` →
``parse_multi_claim_json`` → ``build_multi_verdicts``). These tests pin
the contract for that live path:

  * one API call carries N claims (amortizes ``OPENAI_SYNTHESIS_SYSTEM``)
  * usage lands on index-0 only (matches the cost-attribution invariant)
  * tool-retrieved URLs from ``web_search_call.action.url`` flow through
    Layer 1d grounding (the bug Phase 1c just fixed for the batch path
    must NOT regress at the live path)
  * malformed JSON marks every claim ``no_response=True`` — no silent
    per-claim retries inside ``call_multi``
  * the request body scales ``max_output_tokens`` and ``max_tool_calls``
    with N (otherwise N=10 chunks truncate or starve the search budget)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from truthbot.models import Claim, VerdictLabel
from truthbot.verify.adapters.openai import OpenAIAdapter


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _claim(text: str) -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


def _fake_response(
    text: str,
    *,
    open_page_urls: list[str] | None = None,
    annotations: list[dict] | None = None,
    input_tokens: int = 1234,
    output_tokens: int = 321,
    cached: int = 0,
    plain_search_calls: int = 0,
    status: str = "completed",
) -> SimpleNamespace:
    """Mimic a Responses API envelope including the post-fix URL surfaces.

    ``open_page_urls`` produces ``web_search_call`` items with
    ``action.type=='open_page'`` and ``action.url`` set — the surface
    that was missing in the legacy batch parser and the one the live
    path must also harvest (otherwise Phase 1c regresses for every
    OpenAI verdict produced live).
    """
    output: list[Any] = []
    for i in range(plain_search_calls):
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
    ann_blocks = [
        SimpleNamespace(url=ann["url"], type=ann.get("type", "url_citation"))
        for ann in (annotations or [])
    ]
    output.append(
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text", text=text, annotations=ann_blocks
                )
            ],
        )
    )
    return SimpleNamespace(
        status=status,
        output=output,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


class _FakeResponses:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.responses = _FakeResponses(response)


def _patch_openai(monkeypatch, response: Any) -> _FakeClient:
    import openai

    client = _FakeClient(response)
    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: client)
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_openai_call_multi_returns_n_verdicts_for_n_claims(monkeypatch) -> None:
    """One multi-claim API call → N ordered verdicts keyed by claim_id."""
    claims = [_claim("A"), _claim("B"), _claim("C")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
            {"claim_id": claims[2].id, "label": "Misleading", "confidence": "Medium", "explanation": "c"},
        ]
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = OpenAIAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert client.responses.calls == 1, (
        "call_multi must issue exactly one API call for N claims — "
        "the whole point of Phase 3a is to amortize OPENAI_SYNTHESIS_SYSTEM "
        "across the chunk just like Grok/Gemini."
    )
    assert [v.claim_id for v in verdicts] == [c.id for c in claims]
    assert [v.label for v in verdicts] == [
        VerdictLabel.TRUE,
        VerdictLabel.FALSE,
        VerdictLabel.MISLEADING,
    ]
    assert all(v.adapter_name == "openai" for v in verdicts)


def test_openai_call_multi_usage_attributed_to_index_zero(monkeypatch) -> None:
    """Single API call → all usage on index-0; siblings carry zeros.

    Same telemetry contract Grok and Gemini honor:
    ``costs.estimate_cost`` must bill once per call, not N times.
    """
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
        ]
    )
    _patch_openai(
        monkeypatch,
        _fake_response(text, input_tokens=1500, output_tokens=600, cached=400),
    )
    adapter = OpenAIAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert verdicts[0].input_tokens == 1500
    assert verdicts[0].output_tokens == 600
    assert verdicts[0].cached_input_tokens == 400
    assert verdicts[0].batch_call_index == 0
    assert verdicts[1].input_tokens == 0
    assert verdicts[1].output_tokens == 0
    assert verdicts[1].batch_call_index == 1


def test_openai_call_multi_grounds_open_page_urls(monkeypatch) -> None:
    """Tool-retrieved URLs from ``open_page`` actions must flow through Layer 1d.

    Regression guard for Phase 1c: the same ``_walk_output_for_urls``
    helper that powers the batch parser drives the live multi-claim
    parser. If the live path sees ``tool_retrieved=[]`` we'd regress to
    the 100% fabrication rate readout that motivated Phase 3a.
    """
    claims = [_claim("CPI claim"), _claim("Other claim")]
    grounded = "https://www.bls.gov/news.release/cpi.htm"
    fabricated = "https://example.com/never-visited"
    text = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "True",
                "confidence": "High",
                "web_sources": [grounded, fabricated],
            },
            {
                "claim_id": claims[1].id,
                "label": "False",
                "confidence": "High",
                "web_sources": [],
            },
        ]
    )
    _patch_openai(
        monkeypatch,
        _fake_response(text, open_page_urls=[grounded]),
    )
    adapter = OpenAIAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert grounded in verdicts[0].web_sources
    assert fabricated not in verdicts[0].web_sources, (
        "URLs the model cited but the tool never opened MUST be stripped "
        "by apply_url_grounding — this is Layer 1d's whole job."
    )
    assert verdicts[0].stripped_source_count >= 1


def test_openai_call_multi_malformed_json_marks_all_no_response(monkeypatch) -> None:
    """Garbage model output → every claim UNVERIFIABLE no_response=True.

    Matches Grok/Gemini: ``call_multi`` must NOT silently per-claim retry,
    so the multi- vs single-claim comparison stays clean. The sidecar
    loop in ``BatchDispatcher`` is the layer that does the per-claim
    fallback.
    """
    claims = [_claim("A"), _claim("B")]
    _patch_openai(monkeypatch, _fake_response("this is not json at all"))
    adapter = OpenAIAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert len(verdicts) == 2
    assert all(v.no_response for v in verdicts)
    assert all(v.label == VerdictLabel.UNVERIFIABLE for v in verdicts)


def test_openai_call_multi_scales_budgets_with_n(monkeypatch) -> None:
    """``max_output_tokens`` and ``max_tool_calls`` scale linearly with N.

    Without scaling, N=10 either truncates (8192 tokens / 10 = 819 per
    claim) or starves the search budget (2 calls total across 10
    claims). Both regressions silently degrade verdict quality.
    """
    claims = [_claim(f"claim {i}") for i in range(5)]
    text = json.dumps(
        [
            {"claim_id": c.id, "label": "True", "confidence": "High", "explanation": ""}
            for c in claims
        ]
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = OpenAIAdapter()
    adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    kwargs = client.responses.last_kwargs
    assert kwargs is not None
    n = len(claims)
    assert kwargs["max_output_tokens"] >= 1024 * n, (
        f"max_output_tokens={kwargs['max_output_tokens']} must scale "
        f"linearly with N={n}; the current cap risks truncation on the "
        "tail of the response array."
    )
    assert kwargs["max_tool_calls"] >= 2 * n, (
        f"max_tool_calls={kwargs['max_tool_calls']} must scale with N; "
        "otherwise N=10 chunks share the same 2-call budget the single-"
        "claim path uses, starving every claim of grounding."
    )


def test_openai_call_multi_user_message_enumerates_all_claim_ids(monkeypatch) -> None:
    """The single API call's user message lists every claim id verbatim."""
    claims = [_claim("Alpha"), _claim("Bravo")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = OpenAIAdapter()
    adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    kwargs = client.responses.last_kwargs
    assert kwargs is not None
    user_blocks = [b for b in kwargs["input"] if b["role"] == "user"]
    user_text = " ".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for b in user_blocks
        for part in (b["content"] if isinstance(b["content"], list) else [b["content"]])
    )
    assert claims[0].id in user_text
    assert claims[1].id in user_text


def test_openai_call_multi_marks_synthesis_mode_live(monkeypatch) -> None:
    """Live verdicts must declare ``synthesis_mode='live'`` for sidecar reconcile.

    The sidecar reconcile in ``BatchDispatcher`` overwrites this back to
    ``"live"`` defensively (see ``_stamp_and_append``), but the adapter
    itself emits the right value so the verdicts are coherent even when
    consumed outside the batch dispatcher (e.g. from the engine path).
    """
    claims = [_claim("A")]
    text = json.dumps(
        [{"claim_id": claims[0].id, "label": "True", "confidence": "High"}]
    )
    _patch_openai(monkeypatch, _fake_response(text))
    adapter = OpenAIAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert verdicts[0].synthesis_mode == "live"
    assert verdicts[0].tier == "frontier"
