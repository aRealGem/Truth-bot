"""Red-tests for the ``BatchDispatcher`` sidecar multi-claim refactor.

Today the sidecar loop in ``BatchDispatcher.submit`` iterates
``(adapter, claim, evidence)`` pairs and calls ``adapter.call(...)`` once
per pair (see ``batch.py`` lines 537-567 pre-refactor). After Phase E Grok
+ Gemini slice, the loop chunks claims per-adapter (clamped by
``max_claims_per_request``) and calls ``adapter.call_multi(...)`` once per
chunk, with per-claim ``adapter.call`` fallback if the multi-claim call
raises.

These tests auto-skip until ``BatchDispatcher`` exposes the
``SIDECAR_SUPPORTS_CALL_MULTI = True`` sentinel that the refactor lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from truthbot.models import (
    Claim,
    Confidence,
    Evidence,
    ModelVerdict,
    VerdictLabel,
)
from truthbot.verify.batch import BatchDispatcher, load_sidecar, sidecar_path


pytestmark = pytest.mark.skipif(
    not getattr(BatchDispatcher, "SIDECAR_SUPPORTS_CALL_MULTI", False),
    reason="pending Phase E sidecar refactor (call_multi chunking)",
)


# ── Fake adapter that records call vs call_multi usage ───────────────────────


class _RecordingSidecarAdapter:
    """Non-batch-API adapter (Grok-shaped) that records which entry points were used."""

    adapter_name = "fake-grok"
    model_id = "fake-grok-1"
    required_env_key = "FAKE_GROK_API_KEY"
    supports_batch = False
    max_claims_per_request = 4

    def __init__(self, *, raise_on_multi: bool = False) -> None:
        self.multi_calls: list[tuple[list[str], dict[str, int]]] = []
        self.single_calls: list[str] = []
        self._raise_on_multi = raise_on_multi

    def call_multi(
        self,
        claims: list[Claim],
        evidence_by_claim: dict[str, list[Evidence]],
        *,
        inject_evidence: bool = True,
        max_evidence_per_claim: int = 5,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> list[ModelVerdict]:
        if self._raise_on_multi:
            raise RuntimeError("simulated multi-claim API failure")

        self.multi_calls.append(
            (
                [c.id for c in claims],
                {cid: len(ev) for cid, ev in evidence_by_claim.items()},
            )
        )
        out: list[ModelVerdict] = []
        for idx, claim in enumerate(claims):
            out.append(
                ModelVerdict(
                    adapter_name=self.adapter_name,
                    model_id=self.model_id,
                    claim_id=claim.id,
                    label=VerdictLabel.TRUE,
                    confidence=Confidence.HIGH,
                    explanation=f"multi:{idx}",
                    input_tokens=1000 if idx == 0 else 0,
                    output_tokens=200 if idx == 0 else 0,
                    batch_call_index=idx,
                    batch_call_id=f"multi-call-{len(self.multi_calls)}",
                )
            )
        return out

    def call(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> ModelVerdict:
        self.single_calls.append(claim.id)
        return ModelVerdict(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            claim_id=claim.id,
            label=VerdictLabel.FALSE,
            confidence=Confidence.HIGH,
            explanation="single-fallback",
        )


def _claim(text: str) -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


def _claims_with_evidence(n: int) -> list[tuple[Claim, list[Evidence]]]:
    return [(_claim(f"Claim {i}"), []) for i in range(n)]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_sidecar_uses_call_multi_for_adapters_with_cap_gt_one(
    tmp_path: Path,
) -> None:
    """Three claims through an adapter with cap=4 → exactly ONE ``call_multi``."""
    adapter = _RecordingSidecarAdapter()
    dispatcher = BatchDispatcher(tmp_path / "metrics")

    dispatcher.submit(
        "run-sidecar-multi-1",
        adapters=[],  # no batch-API adapters
        claims_with_evidence=_claims_with_evidence(3),
        transcript_meta={"speaker": "X", "date": "2026-04-23"},
        inject_evidence=False,
        sidecar_live_adapters=[adapter],
        claims_per_request=4,
    )

    assert len(adapter.multi_calls) == 1, (
        f"expected a single multi-call for 3 claims at cap=4, "
        f"got {len(adapter.multi_calls)}"
    )
    assert len(adapter.multi_calls[0][0]) == 3
    assert adapter.single_calls == [], (
        "call() should not fire when call_multi succeeded"
    )


def test_sidecar_chunks_claims_when_total_exceeds_cap(tmp_path: Path) -> None:
    """7 claims at cap=4 → two chunks (4+3), two ``call_multi`` invocations."""
    adapter = _RecordingSidecarAdapter()
    dispatcher = BatchDispatcher(tmp_path / "metrics")

    dispatcher.submit(
        "run-sidecar-multi-2",
        adapters=[],
        claims_with_evidence=_claims_with_evidence(7),
        transcript_meta={"speaker": "X", "date": "2026-04-23"},
        inject_evidence=False,
        sidecar_live_adapters=[adapter],
        claims_per_request=4,
    )

    assert len(adapter.multi_calls) == 2
    assert [len(call[0]) for call in adapter.multi_calls] == [4, 3]


def test_sidecar_falls_back_to_per_claim_when_call_multi_raises(
    tmp_path: Path,
) -> None:
    """If ``call_multi`` raises, the chunk's claims each fall back to ``call``."""
    adapter = _RecordingSidecarAdapter(raise_on_multi=True)
    dispatcher = BatchDispatcher(tmp_path / "metrics")
    metrics_dir = tmp_path / "metrics"

    dispatcher.submit(
        "run-sidecar-fallback",
        adapters=[],
        claims_with_evidence=_claims_with_evidence(3),
        transcript_meta={"speaker": "X", "date": "2026-04-23"},
        inject_evidence=False,
        sidecar_live_adapters=[adapter],
        claims_per_request=4,
    )

    assert adapter.multi_calls == [], (
        "multi_calls should be empty when call_multi raises — fallback only"
    )
    assert len(adapter.single_calls) == 3, (
        "each claim in the failed chunk must be retried via adapter.call"
    )

    # Every fallback verdict must be durably persisted.
    loaded = load_sidecar(sidecar_path(metrics_dir, "run-sidecar-fallback"))
    assert len(loaded) == 3
    assert all(v.label == VerdictLabel.FALSE for v in loaded)


def test_sidecar_round_trips_batch_call_metadata(tmp_path: Path) -> None:
    """Sidecar JSONL preserves ``batch_call_index`` + ``batch_call_id`` fields.

    Required so the reconcile-time telemetry can stamp ``claim_count`` on
    exactly one row per multi-claim API call (index-0), matching the
    ``build_multi_verdicts`` cost-billing convention.
    """
    adapter = _RecordingSidecarAdapter()
    dispatcher = BatchDispatcher(tmp_path / "metrics")
    metrics_dir = tmp_path / "metrics"

    dispatcher.submit(
        "run-sidecar-roundtrip",
        adapters=[],
        claims_with_evidence=_claims_with_evidence(2),
        transcript_meta={"speaker": "X", "date": "2026-04-23"},
        inject_evidence=False,
        sidecar_live_adapters=[adapter],
        claims_per_request=4,
    )

    loaded = load_sidecar(sidecar_path(metrics_dir, "run-sidecar-roundtrip"))
    assert len(loaded) == 2
    loaded_sorted = sorted(loaded, key=lambda v: v.batch_call_index)
    assert loaded_sorted[0].batch_call_index == 0
    assert loaded_sorted[1].batch_call_index == 1
    assert loaded_sorted[0].input_tokens == 1000
    assert loaded_sorted[1].input_tokens == 0
    assert loaded_sorted[0].batch_call_id == loaded_sorted[1].batch_call_id
    assert loaded_sorted[0].batch_call_id.startswith("multi-call-")


def test_sidecar_single_claim_adapter_still_uses_call(tmp_path: Path) -> None:
    """Adapter with ``max_claims_per_request == 1`` retains legacy per-claim loop."""

    class _LegacyAdapter(_RecordingSidecarAdapter):
        adapter_name = "legacy-sidecar"
        max_claims_per_request = 1

    adapter: Any = _LegacyAdapter()
    dispatcher = BatchDispatcher(tmp_path / "metrics")

    dispatcher.submit(
        "run-sidecar-legacy",
        adapters=[],
        claims_with_evidence=_claims_with_evidence(3),
        transcript_meta={"speaker": "X", "date": "2026-04-23"},
        inject_evidence=False,
        sidecar_live_adapters=[adapter],
        claims_per_request=4,
    )

    # With a cap of 1, chunking collapses to 1-per-chunk — implementations
    # may route these through call_multi (which loops call internally) OR
    # through call directly. The contract is only that every claim gets
    # exactly one verdict spooled.
    total_verdicts = len(adapter.multi_calls) + len(adapter.single_calls)
    assert total_verdicts >= 3, (
        "every claim should produce at least one dispatch path invocation"
    )
