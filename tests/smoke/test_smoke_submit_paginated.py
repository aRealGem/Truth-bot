"""
Paginated variant of the 2-claim submit smoke.

Uses 5 trivial claims with ``claims_per_request=2`` so the batch providers
produce **3 chunks (2 + 2 + 1)**. This exercises the multi-chunk path in
``BatchDispatcher.submit`` and ``reconcile_run`` — neither of which the
``test_smoke_submit.py`` 2-claim smoke ever triggers.

Manifest keys are suffixed ``_pg`` (anthropic_pg / openai_pg / xai_pg /
gemini_pg) so this suite can coexist with the 2-claim smoke on disk.

Run with:

    pytest tests/smoke/test_smoke_submit_paginated.py -m live -v

Expected wall-clock: ~3-4 min (xAI + Gemini dominate the live path).
Expected cost: ~$0.20-0.30 per full run.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.smoke.conftest import (
    append_smoke_summary,
    chunk_claims,
    is_false_label,
    is_true_label,
    require_key,
    update_manifest,
)
from tests.smoke.test_smoke_submit import _submit_batch

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Batch providers: 5 claims, chunk_size=2 → 3 chunks (2+2+1)
# ---------------------------------------------------------------------------


class TestSubmitAnthropicBatchPaginated:
    """Anthropic Messages Batches API, 5 claims in 3 chunks."""

    @classmethod
    def setup_class(cls):
        require_key("ANTHROPIC_API_KEY")

    def test_submit(self, five_claims, smoke_metrics_dir, manifest_path):
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()
        entry = _submit_batch(
            "anthropic",
            adapter,
            five_claims,
            metrics_dir=smoke_metrics_dir,
            claims_per_request=2,
            run_id_prefix="smoke-pg",
        )
        assert entry["chunk_size"] == 2, (
            f"expected chunk_size=2, got {entry['chunk_size']}"
        )
        assert entry["request_count"] == 3, (
            f"expected 3 requests for 5 claims in chunks of 2, "
            f"got {entry['request_count']}"
        )
        update_manifest(manifest_path, "anthropic_pg", entry)


class TestSubmitOpenAIBatchPaginated:
    """OpenAI Batch API, 5 claims in 3 chunks."""

    @classmethod
    def setup_class(cls):
        require_key("OPENAI_API_KEY")

    def test_submit(self, five_claims, smoke_metrics_dir, manifest_path):
        from truthbot.verify.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter()
        entry = _submit_batch(
            "openai",
            adapter,
            five_claims,
            metrics_dir=smoke_metrics_dir,
            claims_per_request=2,
            run_id_prefix="smoke-pg",
        )
        assert entry["chunk_size"] == 2
        assert entry["request_count"] == 3
        update_manifest(manifest_path, "openai_pg", entry)


# ---------------------------------------------------------------------------
# Live providers: 5 sequential claims, one telemetry_run_context for all
# ---------------------------------------------------------------------------


def _check_label_matches(
    label, expected_true: bool
) -> bool:
    """``True`` iff the label polarity matches the expected truth value."""
    if expected_true:
        return is_true_label(label)
    return is_false_label(label)


class TestXAILivePaginated:
    """xAI Grok via /v1/responses; 5 sequential live calls."""

    @classmethod
    def setup_class(cls):
        require_key("XAI_API_KEY")

    def test_five_claims(
        self,
        five_claims,
        five_claims_truth_pattern,
        manifest_path,
        smoke_metrics_dir,
    ):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.grok import GrokAdapter

        run_id = f"smoke-pg-xai-{int(time.time())}"
        adapter = GrokAdapter()
        verdicts = []
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            for claim in five_claims:
                verdict = adapter.call(claim, evidence=[], inject_evidence=False)
                verdicts.append(verdict)
        wall_s = time.monotonic() - t0

        assert len(verdicts) == 5

        mismatches = [
            (i, claims_text, v.label, expected)
            for i, (claims_text, v, expected) in enumerate(
                zip(
                    [c.text for c in five_claims],
                    verdicts,
                    five_claims_truth_pattern,
                )
            )
            if not _check_label_matches(v.label, expected)
        ]
        assert not mismatches, (
            f"xAI: expected all 5 labels to match truth pattern; "
            f"mismatches: {mismatches}"
        )

        verdict_rows = [
            {"claim": c.text, "label": v.label.value, "expected_true": exp}
            for c, v, exp in zip(five_claims, verdicts, five_claims_truth_pattern)
        ]
        update_manifest(
            manifest_path,
            "xai_pg",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "verdicts": verdict_rows,
                "notes": "grok-4 live, 5 claims sequential",
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            "xai_pg",
            "live-sidecar",
            wall_clock_s=wall_s,
            claim_count=5,
            request_count=5,
            verdicts=verdict_rows,
            notes="grok-4 live, 5 claims",
        )


class TestGeminiLivePaginated:
    """Gemini via generate_content; 5 sequential live calls."""

    @classmethod
    def setup_class(cls):
        require_key("GEMINI_API_KEY")

    def test_five_claims(
        self,
        five_claims,
        five_claims_truth_pattern,
        manifest_path,
        smoke_metrics_dir,
    ):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.gemini import GeminiAdapter

        GeminiAdapter._cached_content_names = {}

        run_id = f"smoke-pg-gemini-{int(time.time())}"
        adapter = GeminiAdapter()
        verdicts = []
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            for claim in five_claims:
                verdict = adapter.call(claim, evidence=[], inject_evidence=False)
                verdicts.append(verdict)
        wall_s = time.monotonic() - t0

        assert len(verdicts) == 5

        for v in verdicts:
            exp = v.explanation or ""
            assert "CachedContent can not be used" not in exp, (
                f"Gemini cache regression: {exp}"
            )

        # Gemini 2.5 Pro occasionally returns non-JSON prose on some calls,
        # which surfaces as UNVERIFIABLE. Tolerate up to one such flake in
        # a 5-claim run rather than making the test brittle to a known
        # upstream flakiness. If >1 fails, something is actually wrong.
        correct_count = sum(
            1
            for v, expected in zip(verdicts, five_claims_truth_pattern)
            if _check_label_matches(v.label, expected)
        )
        assert correct_count >= 4, (
            f"Gemini: expected at least 4/5 labels to match truth pattern; "
            f"got {correct_count}/5. Labels: "
            f"{[v.label.value for v in verdicts]}"
        )

        verdict_rows = [
            {"claim": c.text, "label": v.label.value, "expected_true": exp}
            for c, v, exp in zip(five_claims, verdicts, five_claims_truth_pattern)
        ]
        update_manifest(
            manifest_path,
            "gemini_pg",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "verdicts": verdict_rows,
                "cached_content_name": next(iter(GeminiAdapter._cached_content_names.values()), ""),
                "correct_count": correct_count,
                "notes": "gemini-2.5-pro with GoogleSearch + CachedContent, 5 claims",
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            "gemini_pg",
            "live-sidecar",
            wall_clock_s=wall_s,
            claim_count=5,
            request_count=5,
            verdicts=verdict_rows,
            notes="gemini live, 5 claims with GoogleSearch grounding",
        )


# ---------------------------------------------------------------------------
# Live multi-claim (call_multi) variants — cost-reduction validation
#
# These tests drive ``adapter.call_multi`` N times manually, chunking the 5
# trivial claims by ``chunk_size``. The goal is to measure per-claim cost
# savings vs the single-claim per-claim baseline (TestXAILivePaginated /
# TestGeminiLivePaginated above) which re-sends SYNTHESIS_SYSTEM on every
# call.
#
# Parametrized chunk shapes:
#   - ``chunk_size=2`` (3 API calls, 2+2+1): apples-to-apples comparison
#     with tonight's paginated baseline (same chunking shape, same
#     manifest style).
#   - Cap-wide (``chunk_size=6`` for xAI → 1 call; ``chunk_size=4`` for
#     Gemini → 2 calls, 4+1): maximum achievable savings under the
#     adapter's own ``max_claims_per_request`` cap.
# ---------------------------------------------------------------------------


class TestXAILivePaginatedMultiClaim:
    """xAI Grok ``call_multi`` over chunked 5-claim payloads."""

    @classmethod
    def setup_class(cls):
        require_key("XAI_API_KEY")

    @pytest.mark.parametrize(
        "chunk_size, expected_request_count",
        [(2, 3), (6, 1)],
        ids=["c2", "c6"],
    )
    def test_five_claims_multi(
        self,
        five_claims,
        five_claims_truth_pattern,
        manifest_path,
        smoke_metrics_dir,
        chunk_size,
        expected_request_count,
    ):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.grok import GrokAdapter

        run_id = f"smoke-pg-multi-xai-c{chunk_size}-{int(time.time())}"
        adapter = GrokAdapter()
        chunks = chunk_claims(five_claims, five_claims_truth_pattern, chunk_size)
        assert len(chunks) == expected_request_count, (
            f"chunking sanity: {len(chunks)} chunks != expected "
            f"{expected_request_count} for chunk_size={chunk_size}"
        )

        all_verdicts = []
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            for chunk_claims_list, _ in chunks:
                verdicts = adapter.call_multi(
                    chunk_claims_list,
                    {c.id: [] for c in chunk_claims_list},
                    inject_evidence=False,
                    run_id=run_id,
                )
                all_verdicts.extend(verdicts)
        wall_s = time.monotonic() - t0

        assert len(all_verdicts) == 5

        by_claim = {v.claim_id: v for v in all_verdicts}
        mismatches = [
            (c.text, by_claim[c.id].label.value, expected)
            for c, expected in zip(five_claims, five_claims_truth_pattern)
            if not (
                is_true_label(by_claim[c.id].label)
                if expected
                else is_false_label(by_claim[c.id].label)
            )
        ]
        assert not mismatches, (
            f"xAI chunk_size={chunk_size}: expected all 5 labels to match "
            f"truth pattern; mismatches: {mismatches}"
        )

        batch_call_ids = {v.batch_call_id for v in all_verdicts if v.batch_call_id}
        assert len(batch_call_ids) == expected_request_count, (
            f"expected {expected_request_count} distinct batch_call_ids "
            f"(one per call_multi), got {len(batch_call_ids)}: {batch_call_ids}"
        )

        verdict_rows = [
            {"claim": c.text, "label": by_claim[c.id].label.value, "expected_true": exp}
            for c, exp in zip(five_claims, five_claims_truth_pattern)
        ]
        update_manifest(
            manifest_path,
            f"xai_pg_multi_c{chunk_size}",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "chunk_size": chunk_size,
                "request_count": expected_request_count,
                "verdicts": verdict_rows,
                "notes": (
                    f"grok-4 call_multi, 5 claims in {expected_request_count} "
                    f"chunks of up to {chunk_size}"
                ),
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            f"xai_pg_multi_c{chunk_size}",
            "live-multi-claim",
            wall_clock_s=wall_s,
            claim_count=5,
            request_count=expected_request_count,
            verdicts=verdict_rows,
            notes=(
                f"grok-4 call_multi (chunk_size={chunk_size}, "
                f"{expected_request_count} API call(s) / 5 claims)"
            ),
        )


class TestGeminiLivePaginatedMultiClaim:
    """Gemini ``call_multi`` over chunked 5-claim payloads (with CachedContent)."""

    @classmethod
    def setup_class(cls):
        require_key("GEMINI_API_KEY")

    @pytest.mark.parametrize(
        "chunk_size, expected_request_count",
        [(2, 3), (4, 2)],
        ids=["c2", "c4"],
    )
    def test_five_claims_multi(
        self,
        five_claims,
        five_claims_truth_pattern,
        manifest_path,
        smoke_metrics_dir,
        chunk_size,
        expected_request_count,
    ):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.gemini import GeminiAdapter

        # Reset cache once at the start so the first chunk pays creation
        # and remaining chunks benefit from a warm CachedContent (the
        # point of this test is to measure that warm-path efficiency).
        GeminiAdapter._cached_content_names = {}

        run_id = f"smoke-pg-multi-gemini-c{chunk_size}-{int(time.time())}"
        adapter = GeminiAdapter()
        chunks = chunk_claims(five_claims, five_claims_truth_pattern, chunk_size)
        assert len(chunks) == expected_request_count

        all_verdicts = []
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            for chunk_claims_list, _ in chunks:
                verdicts = adapter.call_multi(
                    chunk_claims_list,
                    {c.id: [] for c in chunk_claims_list},
                    inject_evidence=False,
                    run_id=run_id,
                )
                all_verdicts.extend(verdicts)
        wall_s = time.monotonic() - t0

        assert len(all_verdicts) == 5

        # Hard regression guard for the Gemini
        # ``cached_content + system_instruction`` API error.
        for v in all_verdicts:
            exp = v.explanation or ""
            assert "CachedContent can not be used" not in exp, (
                f"Gemini cache regression in multi-claim path: {exp}"
            )

        by_claim = {v.claim_id: v for v in all_verdicts}
        # Tolerate one UNVERIFIABLE parse flake (Gemini 2.5 Pro occasionally
        # returns non-JSON prose on one claim — known upstream behaviour).
        correct_count = sum(
            1
            for c, expected in zip(five_claims, five_claims_truth_pattern)
            if (
                is_true_label(by_claim[c.id].label)
                if expected
                else is_false_label(by_claim[c.id].label)
            )
        )
        assert correct_count >= 4, (
            f"Gemini chunk_size={chunk_size}: expected at least 4/5 correct; "
            f"got {correct_count}/5. Labels: "
            f"{[by_claim[c.id].label.value for c in five_claims]}"
        )

        batch_call_ids = {v.batch_call_id for v in all_verdicts if v.batch_call_id}
        assert len(batch_call_ids) == expected_request_count, (
            f"expected {expected_request_count} distinct batch_call_ids "
            f"(one per call_multi), got {len(batch_call_ids)}: {batch_call_ids}"
        )

        verdict_rows = [
            {"claim": c.text, "label": by_claim[c.id].label.value, "expected_true": exp}
            for c, exp in zip(five_claims, five_claims_truth_pattern)
        ]
        update_manifest(
            manifest_path,
            f"gemini_pg_multi_c{chunk_size}",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "chunk_size": chunk_size,
                "request_count": expected_request_count,
                "cached_content_name": next(iter(GeminiAdapter._cached_content_names.values()), ""),
                "correct_count": correct_count,
                "verdicts": verdict_rows,
                "notes": (
                    f"gemini-2.5-pro call_multi + CachedContent, 5 claims in "
                    f"{expected_request_count} chunks of up to {chunk_size}"
                ),
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            f"gemini_pg_multi_c{chunk_size}",
            "live-multi-claim",
            wall_clock_s=wall_s,
            claim_count=5,
            request_count=expected_request_count,
            verdicts=verdict_rows,
            notes=(
                f"gemini call_multi (chunk_size={chunk_size}, "
                f"{expected_request_count} API call(s) / 5 claims)"
            ),
        )
