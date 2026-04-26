"""One-off: re-parse a completed OpenAI batch through the fixed Layer 1d
URL-grounding logic and rewrite the affected rows in
``metrics/adapter_calls.jsonl`` so ``finalize_run`` reflects the corrected
fabrication-rate readout.

Why:
    The legacy ``parse_multi_batch_response`` only collected URLs from
    ``message.content[].annotations[].url``. Real GA web_search batch bodies
    surface retrieved URLs on ``web_search_call.action.url`` (the
    ``open_page`` action variant). With the empty annotations the parser
    saw, every model-cited URL was stripped → 100% fabrication-rate readout
    for the OpenAI rows in run ``ed7be4ad-…``.

What this does:
    1. Downloads the saved batch result via ``_openai_results``.
    2. Reconstructs the multi-claim chunks from the run descriptor.
    3. Re-parses each chunk through the fixed adapter.
    4. Locates the *latest* batch-mode OpenAI rows for the run in
       ``metrics/adapter_calls.jsonl`` (the ones that already carry
       ``model_reported_source_count`` / ``stripped_source_count``) and
       rewrites those fields in place.
    5. Calls ``finalize_run`` to write a corrected
       ``metrics/run_summaries/<run_id>.json``.

Idempotent: re-running with no parser change should produce no diff.

Usage:
    python scripts/backfill_openai_batch_grounding.py <run_id>
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill_openai_batch")

from truthbot.config import settings
from truthbot.metrics.telemetry import finalize_run
from truthbot.models import Claim
from truthbot.verify.adapters.openai import OpenAIAdapter
from truthbot.verify.batch import _openai_results


def _build_claim_lookup(descriptor: dict) -> dict[str, Claim]:
    """Build ``{claim_id: Claim}`` from the saved batch descriptor.

    Triaged claims appear under ``transcript_meta.triaged_claims``; verified
    claims must be reconstituted from the run's extractions JSONL since the
    descriptor only records ``claims_batched`` count.
    """
    lookup: dict[str, Claim] = {}
    meta = descriptor.get("transcript_meta", {})
    for c in meta.get("triaged_claims", []) or []:
        cid = c["id"]
        lookup[cid] = Claim(
            id=cid,
            transcript_id=c["transcript_id"],
            text=c["text"],
            speaker=c["speaker"],
            context=c.get("context", ""),
            category=c.get("category"),
            is_checkable=c.get("is_checkable", True),
        )

    extractions_path = (
        settings.metrics_dir / "extractions" / f"{descriptor['run_id']}.jsonl"
    )
    if extractions_path.exists():
        with extractions_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cid = row["id"]
                if cid in lookup:
                    continue
                lookup[cid] = Claim(
                    id=cid,
                    transcript_id=row["transcript_id"],
                    text=row["text"],
                    speaker=row["speaker"],
                    context=row.get("context", ""),
                    category=row.get("category"),
                    is_checkable=row.get("is_checkable", True),
                )
    return lookup


def _reparse_openai_batch(run_id: str) -> dict[str, dict[str, int]]:
    """Re-download + re-parse → ``{claim_id: {"reported": N, "stripped": M}}``."""
    descriptor_path = (
        settings.metrics_dir / "batch_jobs" / f"{run_id}.json"
    )
    descriptor = json.loads(descriptor_path.read_text())
    openai_job = (descriptor.get("provider_jobs") or {}).get("openai")
    if not openai_job:
        raise SystemExit(f"No OpenAI provider_job in {descriptor_path}")
    batch_id = openai_job["batch_id"]
    custom_id_to_claims: dict[str, list[str]] = openai_job.get(
        "custom_id_to_claims"
    ) or openai_job.get("custom_id_to_claim") or {}

    claim_lookup = _build_claim_lookup(descriptor)
    logger.info("Loaded %d claims from descriptor + extractions", len(claim_lookup))

    rows = _openai_results(batch_id)
    succeeded = [r for r in rows if r.get("status") == "succeeded"]
    logger.info(
        "Downloaded batch %s: %d rows, %d succeeded", batch_id, len(rows), len(succeeded)
    )

    adapter = OpenAIAdapter()
    by_claim: dict[str, dict[str, int]] = {}
    for row in succeeded:
        custom_id = row.get("custom_id", "")
        body = row.get("body") or {}
        claim_ids = custom_id_to_claims.get(custom_id, [])
        claims = [claim_lookup[cid] for cid in claim_ids if cid in claim_lookup]
        if not claims:
            logger.warning("custom_id %s has no resolvable claims; skipping", custom_id)
            continue
        verdicts = adapter.parse_multi_batch_response(
            body, claims, batch_call_id=custom_id
        )
        for v in verdicts:
            by_claim[v.claim_id] = {
                "reported": len(v.model_reported_sources or []),
                "stripped": int(v.stripped_source_count or 0),
                "kept": len(v.web_sources or []),
            }
    return by_claim


def _rewrite_telemetry(run_id: str, by_claim: dict[str, dict[str, int]]) -> tuple[int, int]:
    """Rewrite ``adapter_calls.jsonl`` in place; return ``(updated, total)``."""
    path = settings.metrics_dir / "adapter_calls.jsonl"
    if not path.exists():
        raise SystemExit(f"No telemetry log at {path}")

    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    candidate_indices: dict[str, int] = {}
    for i, rec in enumerate(rows):
        if rec.get("run_id") != run_id:
            continue
        if rec.get("adapter_name") != "openai":
            continue
        if rec.get("mode") != "batch":
            continue
        if rec.get("model_reported_source_count") is None:
            continue
        cid = rec.get("claim_id")
        if cid in by_claim:
            candidate_indices[cid] = i

    updated = 0
    for cid, idx in candidate_indices.items():
        rec = rows[idx]
        new_reported = by_claim[cid]["reported"]
        new_stripped = by_claim[cid]["stripped"]
        new_kept = by_claim[cid]["kept"]
        if (
            rec.get("model_reported_source_count") == new_reported
            and rec.get("stripped_source_count") == new_stripped
            and rec.get("retrieved_url_count") == new_kept
        ):
            continue
        rec["model_reported_source_count"] = new_reported
        rec["stripped_source_count"] = new_stripped
        rec["retrieved_url_count"] = new_kept
        updated += 1

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    tmp_path.replace(path)

    return updated, len(candidate_indices)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    run_id = sys.argv[1]

    by_claim = _reparse_openai_batch(run_id)
    logger.info("Re-parsed %d claim verdicts", len(by_claim))
    for cid, c in by_claim.items():
        logger.info(
            "  claim=%s reported=%d stripped=%d kept=%d",
            cid[:8],
            c["reported"],
            c["stripped"],
            c["kept"],
        )

    updated, total = _rewrite_telemetry(run_id, by_claim)
    logger.info("Updated %d/%d OpenAI batch telemetry rows", updated, total)

    summary = finalize_run(run_id)
    fab = summary.get("fabrication", {})
    logger.info(
        "New summary: total_calls=%d fabrication_rate=%.3f",
        summary.get("total_calls", 0),
        fab.get("fabrication_rate", 0.0),
    )
    for ad, ad_fab in (fab.get("by_adapter") or {}).items():
        logger.info(
            "  %-10s reported=%d stripped=%d rate=%.3f",
            ad,
            ad_fab.get("reported", 0),
            ad_fab.get("stripped", 0),
            ad_fab.get("rate", 0.0),
        )


if __name__ == "__main__":
    main()
