"""Green tests for ``VerificationEngine.verify_bundles_batch`` (Phase E slice).

Covers the multi-claim live fan-out shape: per-adapter workers concurrently
issue chunked ``call_multi`` invocations, results are regrouped per claim_id,
and each claim gets a ``VerdictBundle`` via ``finalize_bundle``.
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
    VerdictBundle,
    VerdictLabel,
)
from truthbot.verify.engine import VerificationEngine


def _claim(text: str) -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


class _MultiAdapter:
    """Multi-claim-capable adapter; records every call for assertions."""

    adapter_name = "multi"
    model_id = "multi-1"
    required_env_key = "FAKE_MULTI_KEY"
    supports_batch = False
    max_claims_per_request = 4

    def __init__(self) -> None:
        self.multi_call_count = 0
        self.multi_chunk_sizes: list[int] = []
        self.single_call_count = 0

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
        self.multi_call_count += 1
        self.multi_chunk_sizes.append(len(claims))
        return [
            ModelVerdict(
                adapter_name=self.adapter_name,
                model_id=self.model_id,
                claim_id=claim.id,
                label=VerdictLabel.TRUE,
                confidence=Confidence.HIGH,
                explanation="multi",
                batch_call_index=idx,
            )
            for idx, claim in enumerate(claims)
        ]

    def call(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> ModelVerdict:
        self.single_call_count += 1
        return ModelVerdict(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            claim_id=claim.id,
            label=VerdictLabel.FALSE,
            confidence=Confidence.HIGH,
            explanation="single-fallback",
        )


class _SingleAdapter:
    """Legacy adapter — no ``call_multi`` override. Uses default loop-of-call."""

    adapter_name = "single"
    model_id = "single-1"
    required_env_key = "FAKE_SINGLE_KEY"
    supports_batch = False
    max_claims_per_request = 1

    def __init__(self) -> None:
        self.call_count = 0

    def call(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> ModelVerdict:
        self.call_count += 1
        return ModelVerdict(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            claim_id=claim.id,
            label=VerdictLabel.TRUE,
            confidence=Confidence.HIGH,
            explanation="single",
        )

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
        # Inline copy of the default ``LLMAdapter.call_multi`` — exercises the
        # "adapter doesn't override" path without importing LLMAdapter.
        return [
            self.call(
                c,
                evidence_by_claim.get(c.id, []),
                inject_evidence=inject_evidence,
                telemetry_tier=telemetry_tier,
                run_id=run_id,
            )
            for c in claims
        ]


def _make_engine(tmp_path: Path, adapters: list[Any]) -> VerificationEngine:
    """Construct an engine with an isolated cache dir and inject fake adapters."""
    engine = VerificationEngine(connectors=[], cache_dir=tmp_path / "cache")
    engine._adapters = adapters  # bypass _build_adapters for testing
    return engine


class TestVerifyBundlesBatch:
    def test_fan_out_shape_mixed_adapters(self, tmp_path: Path) -> None:
        """3 claims across multi-capable + single adapter → bundle per claim.

        - Multi adapter: ONE ``call_multi`` with all 3 claims (cap=4).
        - Single adapter: 3 ``call`` invocations (default loop).
        - Each claim bundle has 2 model verdicts (one per adapter).
        """
        multi = _MultiAdapter()
        single = _SingleAdapter()
        engine = _make_engine(tmp_path, [multi, single])

        claims = [_claim(f"Claim {i}") for i in range(3)]
        bundles = engine.verify_bundles_batch(claims)

        assert len(bundles) == 3
        assert multi.multi_call_count == 1
        assert multi.multi_chunk_sizes == [3]
        assert single.call_count == 3

        # Output ordering preserves input claim order.
        assert [b.claim.id for b in bundles] == [c.id for c in claims]
        for b in bundles:
            adapter_names = {v.adapter_name for v in b.model_verdicts}
            assert adapter_names == {"multi", "single"}
            assert isinstance(b, VerdictBundle)

    def test_chunks_when_claims_exceed_adapter_cap(self, tmp_path: Path) -> None:
        """7 claims × multi adapter with cap=4 → 2 chunks of 4 + 3."""
        multi = _MultiAdapter()
        engine = _make_engine(tmp_path, [multi])

        claims = [_claim(f"Claim {i}") for i in range(7)]
        bundles = engine.verify_bundles_batch(claims)

        assert len(bundles) == 7
        assert multi.multi_call_count == 2
        assert sorted(multi.multi_chunk_sizes) == [3, 4]

    def test_per_claim_fallback_when_call_multi_raises(
        self, tmp_path: Path
    ) -> None:
        """call_multi raising → chunk falls back to per-claim ``call``."""

        class _FlakyAdapter(_MultiAdapter):
            def call_multi(self, claims, *a, **kw):
                raise RuntimeError("simulated API blow-up")

        adapter = _FlakyAdapter()
        engine = _make_engine(tmp_path, [adapter])

        claims = [_claim(f"Claim {i}") for i in range(3)]
        bundles = engine.verify_bundles_batch(claims)

        assert len(bundles) == 3
        assert adapter.single_call_count == 3
        # Each claim got exactly one verdict (from the fallback loop).
        for b in bundles:
            fallback = [v for v in b.model_verdicts if v.explanation == "single-fallback"]
            assert len(fallback) == 1

    def test_empty_claim_list_returns_empty(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, [_MultiAdapter()])
        assert engine.verify_bundles_batch([]) == []

    def test_no_adapters_still_builds_bundles(self, tmp_path: Path) -> None:
        """No active adapters → empty-verdict bundles; consensus is UNVERIFIABLE."""
        engine = _make_engine(tmp_path, [])
        claims = [_claim("A"), _claim("B")]
        bundles = engine.verify_bundles_batch(claims)
        assert len(bundles) == 2
        for b in bundles:
            assert b.model_verdicts == []
            assert b.consensus.consensus_label == VerdictLabel.UNVERIFIABLE
