"""Red-tests for ``GrokAdapter.call_multi`` (live claim-batching, Phase E slice).

The whole module is auto-skipped until ``GrokAdapter`` actually raises
``max_claims_per_request`` past 1, so these tests are inert pre-implementation
and flip green on the first passing override.

xAI has no batch API as of 2026-04-22 (see ``PROJECT_BOARD.md``) and isn't
getting one soon. Claim-batching at the live layer is where the Grok cost
savings live: one ``client.responses.create`` carrying N claims instead of
N calls each re-sending the ~5.5 KB ``SYNTHESIS_SYSTEM`` rubric.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from truthbot.models import Claim, VerdictLabel
from truthbot.verify.adapters.grok import GrokAdapter


pytestmark = pytest.mark.skipif(
    GrokAdapter.max_claims_per_request < 2,
    reason="pending Phase E Grok live multi-claim override",
)


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")


def _claim(text: str) -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


def _fake_response(
    text: str,
    *,
    urls: list[str] | None = None,
    input_tokens: int = 1234,
    output_tokens: int = 321,
    tool_calls: int = 0,
) -> SimpleNamespace:
    """Mimic xAI ``responses.create`` output envelope."""
    annotations = [SimpleNamespace(url=u) for u in (urls or [])]
    output: list[Any] = [
        SimpleNamespace(type="web_search_call") for _ in range(tool_calls)
    ]
    output.append(
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(type="output_text", text=text, annotations=annotations)
            ],
        )
    )
    return SimpleNamespace(
        output=output,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        citations=[],
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
    """Patch ``openai.OpenAI(...)`` to return a deterministic fake client."""
    import openai

    client = _FakeClient(response)
    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: client)
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_grok_call_multi_returns_n_verdicts_for_n_claims(monkeypatch) -> None:
    """One multi-claim API call → N ordered ModelVerdicts keyed by claim_id."""
    claims = [_claim("A"), _claim("B"), _claim("C")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
            {"claim_id": claims[2].id, "label": "Misleading", "confidence": "Medium", "explanation": "c"},
        ]
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = GrokAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert client.responses.calls == 1, (
        "call_multi must issue exactly one API call for N claims "
        "(that's the whole point — amortize SYNTHESIS_SYSTEM over N claims)"
    )
    assert [v.claim_id for v in verdicts] == [c.id for c in claims]
    assert [v.label for v in verdicts] == [
        VerdictLabel.TRUE,
        VerdictLabel.FALSE,
        VerdictLabel.MISLEADING,
    ]
    assert all(v.adapter_name == "xai" for v in verdicts)


def test_grok_call_multi_usage_attributed_to_index_zero_only(monkeypatch) -> None:
    """Single API call → full usage on index-0; siblings carry zeros.

    Matches the telemetry contract enforced by ``build_multi_verdicts``:
    downstream ``costs.estimate_cost`` must bill once per call, not N times.
    """
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
        ]
    )
    _patch_openai(
        monkeypatch, _fake_response(text, input_tokens=1500, output_tokens=600)
    )
    adapter = GrokAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert verdicts[0].input_tokens == 1500
    assert verdicts[0].output_tokens == 600
    assert verdicts[0].batch_call_index == 0
    assert verdicts[1].input_tokens == 0
    assert verdicts[1].output_tokens == 0
    assert verdicts[1].batch_call_index == 1


def test_grok_call_multi_malformed_json_marks_all_no_response(monkeypatch) -> None:
    """If the model returns garbage, every claim gets UNVERIFIABLE no_response=True.

    The adapter MUST NOT silently per-claim retry inside ``call_multi`` —
    that muddies the single- vs multi-claim comparison. Callers decide.
    """
    claims = [_claim("A"), _claim("B")]
    _patch_openai(monkeypatch, _fake_response("this is not json at all"))
    adapter = GrokAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert len(verdicts) == 2
    assert all(v.no_response for v in verdicts)
    assert all(v.label == VerdictLabel.UNVERIFIABLE for v in verdicts)


def test_grok_call_multi_backfills_urls_on_index_zero(monkeypatch) -> None:
    """When the model omits per-verdict ``web_sources``, harvested URLs land on index-0."""
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    _patch_openai(
        monkeypatch,
        _fake_response(
            text, urls=["https://example.gov/a", "https://example.gov/b"]
        ),
    )
    adapter = GrokAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert verdicts[0].web_sources == [
        "https://example.gov/a",
        "https://example.gov/b",
    ]
    assert verdicts[1].web_sources == []


def test_grok_call_multi_backfills_mrs_on_all_indices(monkeypatch) -> None:
    """Defensive backfill: every chunk index gets ``model_reported_sources``.

    xAI multi-claim routinely emits ``web_sources: []`` per-claim despite
    the search tool firing 6-27 times per chunk. Without this backfill,
    ~22/29 published claims look ungrounded for Grok in a SOTU-sized run.
    Audit trail / cross-claim consensus needs the URLs even when visible
    grounding is reserved for index-0.
    """
    claims = [_claim("A"), _claim("B"), _claim("C")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High",
             "web_sources": []},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High",
             "web_sources": []},
            {"claim_id": claims[2].id, "label": "Misleading",
             "confidence": "Medium", "web_sources": []},
        ]
    )
    tool_urls = ["https://bls.gov/x", "https://bea.gov/y"]
    _patch_openai(
        monkeypatch, _fake_response(text, urls=tool_urls, tool_calls=2)
    )
    adapter = GrokAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert verdicts[0].model_reported_sources == tool_urls
    assert verdicts[1].model_reported_sources == tool_urls
    assert verdicts[2].model_reported_sources == tool_urls
    assert verdicts[0].web_sources == tool_urls, (
        "index-0 keeps the visible-grounding fallback so the publish "
        "layer shows at least one cited source per chunk"
    )
    assert verdicts[1].web_sources == [], (
        "siblings keep web_sources empty to avoid claiming each claim "
        "was independently grounded by the same URL set"
    )
    assert verdicts[2].web_sources == []


def test_grok_max_claims_per_request_raised_to_six() -> None:
    """Grok's class-level cap documents the conservative per-call chunk size."""
    assert GrokAdapter.max_claims_per_request >= 6, (
        "Grok live multi-claim requires max_claims_per_request >= 6; "
        f"got {GrokAdapter.max_claims_per_request}"
    )


def test_grok_call_multi_sends_single_request_with_multi_user_message(monkeypatch) -> None:
    """The single API call's user message must enumerate all claim_ids.

    Pinned by the shared ``build_multi_user_message`` helper in base.py —
    claim IDs appear verbatim in the user prompt so the model knows how to
    key the response JSON array.
    """
    claims = [_claim("Alpha claim"), _claim("Bravo claim")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = GrokAdapter()

    adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    kwargs = client.responses.last_kwargs
    assert kwargs is not None
    input_blocks = kwargs["input"]
    user_blocks = [b for b in input_blocks if b["role"] == "user"]
    user_text = " ".join(
        part if isinstance(part, str) else part.get("text", "")
        for b in user_blocks
        for part in ([b["content"]] if isinstance(b["content"], str) else b["content"])
    )
    assert claims[0].id in user_text
    assert claims[1].id in user_text


# ── Tool-call cap (Grok unbounded-budget fix) ─────────────────────────────────


def test_grok_call_multi_passes_default_max_tool_calls(monkeypatch) -> None:
    """Default cap is 8 tool-calls per claim; multi-claim chunks scale to 8*N.

    Pins the protection against Grok's unbounded ``responses.create`` tool
    budget, which spent $2.92 / 70% of the 2026-04-25 10-claim rerun cost.
    """
    monkeypatch.delenv("TRUTHBOT_GROK_MAX_TOOL_CALLS", raising=False)
    claims = [_claim("A"), _claim("B"), _claim("C")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
            {"claim_id": claims[2].id, "label": "True", "confidence": "High"},
        ]
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = GrokAdapter()

    adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    kwargs = client.responses.last_kwargs
    assert kwargs is not None
    assert kwargs.get("max_tool_calls") == 8 * len(claims), (
        f"max_tool_calls={kwargs.get('max_tool_calls')} must scale "
        f"with N (default 8/claim); got chunk size {len(claims)}."
    )


def test_grok_call_multi_honors_env_override_for_max_tool_calls(monkeypatch) -> None:
    """``TRUTHBOT_GROK_MAX_TOOL_CALLS=4`` should produce ``max_tool_calls=4*N``."""
    monkeypatch.setenv("TRUTHBOT_GROK_MAX_TOOL_CALLS", "4")
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = GrokAdapter()

    adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    kwargs = client.responses.last_kwargs
    assert kwargs is not None
    assert kwargs.get("max_tool_calls") == 4 * len(claims)


def test_grok_call_multi_falls_back_when_xai_rejects_max_tool_calls(monkeypatch) -> None:
    """If xAI's server rejects ``max_tool_calls``, retry once without the kwarg.

    xAI doesn't document this parameter; we pass it defensively. When the
    server returns an error mentioning the unknown param, the adapter MUST
    retry without it rather than failing the entire chunk.
    """
    monkeypatch.delenv("TRUTHBOT_GROK_MAX_TOOL_CALLS", raising=False)
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    response = _fake_response(text)

    call_log: list[dict] = []

    class _RejectingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            call_log.append(dict(kwargs))
            if self.calls == 1 and "max_tool_calls" in kwargs:
                raise ValueError(
                    "Unknown parameter: 'max_tool_calls' is not supported"
                )
            return response

    class _RejectingClient:
        def __init__(self) -> None:
            self.responses = _RejectingResponses()

    import openai

    client = _RejectingClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: client)
    adapter = GrokAdapter()

    verdicts = adapter.call_multi(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )

    assert len(verdicts) == 2
    assert client.responses.calls == 2, "must retry once after rejection"
    assert "max_tool_calls" in call_log[0]
    assert "max_tool_calls" not in call_log[1]


def test_grok_call_default_per_claim_max_tool_calls(monkeypatch) -> None:
    """Single-claim ``call()`` path also passes the per-claim cap."""
    from truthbot.models import Claim

    monkeypatch.delenv("TRUTHBOT_GROK_MAX_TOOL_CALLS", raising=False)
    claim = Claim(transcript_id="t1", text="ping", speaker="Test")
    text = json.dumps(
        {"label": "True", "confidence": "High", "explanation": "ok"}
    )
    client = _patch_openai(monkeypatch, _fake_response(text))
    adapter = GrokAdapter()

    adapter.call(claim, [], inject_evidence=False)

    kwargs = client.responses.last_kwargs
    assert kwargs is not None
    assert kwargs.get("max_tool_calls") == 8
