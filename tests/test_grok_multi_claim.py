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
