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

        GeminiAdapter._cached_content_name = None

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
                "cached_content_name": GeminiAdapter._cached_content_name or "",
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
