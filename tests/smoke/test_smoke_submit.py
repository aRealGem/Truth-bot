"""
Phase A of the 2-claim live smoke test: submit.

Each test in this file exercises one provider's *fast* path:

- Anthropic + OpenAI:  BatchDispatcher.submit with claims_per_request=2.
  The test returns as soon as the batch_id is stamped; it does NOT wait
  for the batch to complete. The run_id + batch_id are written to the
  manifest so ``test_smoke_reconcile.py`` can pick them up later.

- xAI + Gemini:        live call via the adapter; 2 claims sequentially.
  These complete within the test (~30-60s each), so the test asserts
  verdicts against ground truth directly.

Every test is marked ``@pytest.mark.live``. Default pytest invocations
skip this whole suite (see ``[tool.pytest.ini_options].addopts``).

Run with:

    pytest tests/smoke/test_smoke_submit.py -m live -v

Expected wall-clock: <3 min total for all four providers.
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


pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helper: submit a 2-claim batch for one provider
# ---------------------------------------------------------------------------


def _submit_batch(
    provider_name: str,
    adapter,
    claims,
    *,
    metrics_dir: Path,
    claims_per_request: int = 2,
    run_id_prefix: str = "smoke",
) -> dict:
    """
    Run BatchDispatcher.submit for one adapter.

    Returns a dict {run_id, batch_id, chunk_size, request_count, submitted_at}
    suitable for writing into the manifest.

    The default ``claims_per_request=2`` + 2-claim smoke collapses to a
    single chunk. Pass a different ``claims_per_request`` (and a pool of
    claims > that size) to exercise the multi-chunk path — see
    ``test_smoke_submit_paginated.py``.
    """
    from truthbot.verify.batch import BatchDispatcher, read_batch_job

    dispatcher = BatchDispatcher(metrics_dir)
    claims_with_evidence = [(claim, []) for claim in claims]
    transcript_meta = {
        "speaker": "Smoke Test",
        "date": "2026-04-23",
        "venue": "",
    }
    run_id = f"{run_id_prefix}-{provider_name}-{int(time.time())}"

    submitted_at = time.time()
    dispatcher.submit(
        run_id,
        adapters=[adapter],
        claims_with_evidence=claims_with_evidence,
        transcript_meta=transcript_meta,
        inject_evidence=False,
        claims_per_request=claims_per_request,
    )

    descriptor = read_batch_job(metrics_dir, run_id)
    assert descriptor is not None, f"{provider_name}: descriptor missing after submit"
    entry = (descriptor.get("provider_jobs") or {}).get(provider_name)
    assert entry is not None, (
        f"{provider_name}: no provider_jobs entry in descriptor "
        f"(keys: {list((descriptor.get('provider_jobs') or {}).keys())})"
    )
    batch_id = entry.get("batch_id")
    assert batch_id, f"{provider_name}: batch_id not stamped; entry={entry}"

    return {
        "run_id": run_id,
        "batch_id": batch_id,
        "chunk_size": entry.get("chunk_size", 1),
        "request_count": entry.get("request_count", 0),
        "submitted_at": submitted_at,
        "status": "submitted",
    }


# ---------------------------------------------------------------------------
# Batch providers: submit-only (reconcile runs in Phase B)
# ---------------------------------------------------------------------------


class TestSubmitAnthropicBatch:
    """Anthropic Message Batches API, claims_per_request=2."""

    @classmethod
    def setup_class(cls):
        require_key("ANTHROPIC_API_KEY")

    def test_submit(self, two_claims, smoke_metrics_dir, manifest_path):
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()
        entry = _submit_batch(
            "anthropic", adapter, two_claims, metrics_dir=smoke_metrics_dir
        )
        # claims_per_request=2 with 2 claims must collapse into 1 request.
        assert entry["chunk_size"] == 2, (
            f"expected chunk_size=2 for multi-claim bundling, got {entry['chunk_size']}"
        )
        assert entry["request_count"] == 1, (
            f"expected 1 request (2 claims bundled), got {entry['request_count']}"
        )
        update_manifest(manifest_path, "anthropic", entry)


class TestSubmitOpenAIBatch:
    """OpenAI Batch API, claims_per_request=2."""

    @classmethod
    def setup_class(cls):
        require_key("OPENAI_API_KEY")

    def test_submit(self, two_claims, smoke_metrics_dir, manifest_path):
        from truthbot.verify.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter()
        entry = _submit_batch(
            "openai", adapter, two_claims, metrics_dir=smoke_metrics_dir
        )
        assert entry["chunk_size"] == 2
        assert entry["request_count"] == 1
        update_manifest(manifest_path, "openai", entry)


# ---------------------------------------------------------------------------
# Live providers: run to completion here (no reconcile phase needed)
# ---------------------------------------------------------------------------


class TestXAILive:
    """xAI Grok via /v1/responses; no batch API so we run live."""

    @classmethod
    def setup_class(cls):
        require_key("XAI_API_KEY")

    def test_two_claims(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.grok import GrokAdapter

        run_id = f"smoke-xai-{int(time.time())}"
        adapter = GrokAdapter()
        verdicts = []
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            for claim in two_claims:
                verdict = adapter.call(claim, evidence=[], inject_evidence=False)
                verdicts.append(verdict)
        wall_s = time.monotonic() - t0

        assert len(verdicts) == 2
        assert is_true_label(verdicts[0].label), (
            f"Moon claim should be True-ish; got {verdicts[0].label} "
            f"({(verdicts[0].explanation or '')[:100]})"
        )
        assert is_false_label(verdicts[1].label), (
            f"Eiffel claim should be False-ish; got {verdicts[1].label} "
            f"({(verdicts[1].explanation or '')[:100]})"
        )

        verdict_rows = [
            {"claim": c.text, "label": v.label.value}
            for c, v in zip(two_claims, verdicts)
        ]
        update_manifest(
            manifest_path,
            "xai",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "verdicts": verdict_rows,
                "notes": "grok-4 (frontier) live via Responses API",
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            "xai",
            "live-sidecar",
            wall_clock_s=wall_s,
            claim_count=2,
            request_count=2,
            verdicts=verdict_rows,
            notes="grok-4 live",
        )


class TestGeminiLive:
    """Gemini via generate_content; validates the Phase 1 CachedContent fix."""

    @classmethod
    def setup_class(cls):
        require_key("GEMINI_API_KEY")

    def test_two_claims(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.gemini import GeminiAdapter

        # Reset the process-wide cache handle so the first call in this
        # live run actually exercises CreateCachedContentConfig with tools,
        # rather than reusing a stale cache from a prior test.
        GeminiAdapter._cached_content_names = {}

        run_id = f"smoke-gemini-{int(time.time())}"
        adapter = GeminiAdapter()
        verdicts = []
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            for claim in two_claims:
                verdict = adapter.call(claim, evidence=[], inject_evidence=False)
                verdicts.append(verdict)
        wall_s = time.monotonic() - t0

        assert len(verdicts) == 2

        # The pre-Phase-1 failure mode had Gemini return UNVERIFIABLE with
        # "CachedContent can not be used with GenerateContent request
        # setting system_instruction, tools or tool_config". Fail loudly
        # on that specific string so regression is obvious.
        for v in verdicts:
            exp = v.explanation or ""
            assert "CachedContent can not be used" not in exp, (
                f"Gemini cache regression: {exp}"
            )

        assert is_true_label(verdicts[0].label), (
            f"Moon claim should be True-ish; got {verdicts[0].label} "
            f"({(verdicts[0].explanation or '')[:100]})"
        )
        assert is_false_label(verdicts[1].label), (
            f"Eiffel claim should be False-ish; got {verdicts[1].label} "
            f"({(verdicts[1].explanation or '')[:100]})"
        )

        verdict_rows = [
            {"claim": c.text, "label": v.label.value}
            for c, v in zip(two_claims, verdicts)
        ]
        update_manifest(
            manifest_path,
            "gemini",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "verdicts": verdict_rows,
                "cached_content_name": next(iter(GeminiAdapter._cached_content_names.values()), ""),
                "notes": "gemini-2.5-pro with GoogleSearch + CachedContent",
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            "gemini",
            "live-sidecar",
            wall_clock_s=wall_s,
            claim_count=2,
            request_count=2,
            verdicts=verdict_rows,
            notes="gemini live with GoogleSearch grounding",
        )


# ---------------------------------------------------------------------------
# Phase E — live multi-claim (claim-batching) variants
#
# Each of these issues ONE API call with 2 claims (vs. the 2 sequential
# single-claim calls above), verifies both ground-truth labels, and records
# a ``request_count=1`` smoke summary row so the post-run cost diff is
# obvious against the 2-request baseline.
# ---------------------------------------------------------------------------


class TestXAILiveMulti:
    """xAI Grok live multi-claim: one ``call_multi`` for both claims."""

    @classmethod
    def setup_class(cls):
        require_key("XAI_API_KEY")

    def test_two_claims_multi(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.grok import GrokAdapter

        run_id = f"smoke-xai-multi-{int(time.time())}"
        adapter = GrokAdapter()
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            verdicts = adapter.call_multi(
                list(two_claims),
                {c.id: [] for c in two_claims},
                inject_evidence=False,
                run_id=run_id,
            )
        wall_s = time.monotonic() - t0

        assert len(verdicts) == 2
        by_claim = {v.claim_id: v for v in verdicts}
        assert is_true_label(by_claim[two_claims[0].id].label), (
            f"Moon claim should be True-ish; got {by_claim[two_claims[0].id].label}"
        )
        assert is_false_label(by_claim[two_claims[1].id].label), (
            f"Eiffel claim should be False-ish; got {by_claim[two_claims[1].id].label}"
        )

        # Index-0 carries the entire call's usage; siblings carry zero so
        # downstream cost estimation bills once per call, not twice.
        idx0 = next(v for v in verdicts if v.batch_call_index == 0)
        siblings = [v for v in verdicts if v.batch_call_index != 0]
        assert idx0.input_tokens > 0, "index-0 should carry the call's input tokens"
        for sib in siblings:
            assert sib.input_tokens == 0, (
                f"sibling input_tokens must be 0 for cost parity; got {sib.input_tokens}"
            )

        verdict_rows = [
            {"claim": c.text, "label": by_claim[c.id].label.value}
            for c in two_claims
        ]
        update_manifest(
            manifest_path,
            "xai-multi",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "verdicts": verdict_rows,
                "notes": "grok-4 live multi-claim (single API call for 2 claims)",
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            "xai-multi",
            "live-multi-claim",
            wall_clock_s=wall_s,
            claim_count=2,
            request_count=1,
            verdicts=verdict_rows,
            notes="grok-4 call_multi (1 API call / 2 claims)",
        )


class TestGeminiLiveMulti:
    """Gemini live multi-claim: one ``generate_content`` for both claims."""

    @classmethod
    def setup_class(cls):
        require_key("GEMINI_API_KEY")

    def test_two_claims_multi(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.metrics.telemetry import telemetry_run_context
        from truthbot.verify.adapters.gemini import GeminiAdapter

        # Fresh cache so we see a cache-creation usage signal on call 1 and
        # a cache-hit signal on call 2 (which happens to be this one if a
        # prior Gemini test ran in this process).
        GeminiAdapter._cached_content_names = {}

        run_id = f"smoke-gemini-multi-{int(time.time())}"
        adapter = GeminiAdapter()
        t0 = time.monotonic()
        with telemetry_run_context(
            run_id=run_id, evidence_injected=False, synthesis_mode="live"
        ):
            verdicts = adapter.call_multi(
                list(two_claims),
                {c.id: [] for c in two_claims},
                inject_evidence=False,
                run_id=run_id,
            )
        wall_s = time.monotonic() - t0

        assert len(verdicts) == 2
        by_claim = {v.claim_id: v for v in verdicts}

        for v in verdicts:
            assert "CachedContent can not be used" not in (v.explanation or ""), (
                f"Gemini cache regression in multi-claim path: {v.explanation}"
            )

        assert is_true_label(by_claim[two_claims[0].id].label), (
            f"Moon claim should be True-ish; got {by_claim[two_claims[0].id].label}"
        )
        assert is_false_label(by_claim[two_claims[1].id].label), (
            f"Eiffel claim should be False-ish; got {by_claim[two_claims[1].id].label}"
        )

        idx0 = next(v for v in verdicts if v.batch_call_index == 0)
        siblings = [v for v in verdicts if v.batch_call_index != 0]
        assert idx0.input_tokens > 0, "index-0 should carry the call's input tokens"
        for sib in siblings:
            assert sib.input_tokens == 0

        verdict_rows = [
            {"claim": c.text, "label": by_claim[c.id].label.value}
            for c in two_claims
        ]
        update_manifest(
            manifest_path,
            "gemini-multi",
            {
                "run_id": run_id,
                "status": "complete",
                "wall_clock_s": round(wall_s, 2),
                "verdicts": verdict_rows,
                "cached_content_name": next(iter(GeminiAdapter._cached_content_names.values()), ""),
                "notes": "gemini-2.5-pro call_multi (1 API call for 2 claims) + CachedContent",
            },
        )
        append_smoke_summary(
            smoke_metrics_dir,
            "gemini-multi",
            "live-multi-claim",
            wall_clock_s=wall_s,
            claim_count=2,
            request_count=1,
            verdicts=verdict_rows,
            notes="gemini call_multi (1 API call / 2 claims)",
        )
