"""
Async batch job orchestration (Anthropic Message Batches, OpenAI Batch, Gemini).

Grok / xAI: no official batch API in-repo yet — in ``--mode batch`` publish runs,
Grok verdicts are still obtained via **live** calls (sidecar). Extend here when xAI
ships batch support.

This module currently persists job descriptors under ``metrics/batch_jobs/`` and
exposes helpers for ``truthbot batch poll``. Full provider batch submission is
incremental (SDK surface varies); poll merges completed rows when implemented.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def batch_job_path(metrics_dir: Path, run_id: str) -> Path:
    return metrics_dir / "batch_jobs" / f"{run_id}.json"


def write_batch_job(
    metrics_dir: Path,
    run_id: str,
    payload: dict[str, Any],
) -> Path:
    path = batch_job_path(metrics_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote batch job descriptor %s", path)
    return path


def read_batch_job(metrics_dir: Path, run_id: str) -> Optional[dict[str, Any]]:
    path = batch_job_path(metrics_dir, run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class BatchDispatcher:
    """Thin facade around batch job files (extend with provider SDK calls)."""

    def __init__(self, metrics_dir: Path) -> None:
        self._metrics_dir = metrics_dir

    def record_job(
        self,
        run_id: str,
        *,
        transcript_meta: dict[str, Any],
        work_units: list[dict[str, Any]],
        provider_hints: Optional[dict[str, Any]] = None,
    ) -> Path:
        payload = {
            "run_id": run_id,
            "status": "pending",
            "transcript_meta": transcript_meta,
            "work_units": work_units,
            "provider_jobs": provider_hints or {},
            "note": "Provider batch submit/reconcile is incremental — see batch.py.",
        }
        return write_batch_job(self._metrics_dir, run_id, payload)

    def poll(self, run_id: str) -> str:
        """
        Poll job status. Returns a short status string.

        TODO: call Anthropic ``batches.retrieve``, OpenAI ``batches.retrieve``,
        Gemini batch APIs, merge results into VerdictBundle cache keys.
        """
        job = read_batch_job(self._metrics_dir, run_id)
        if not job:
            return "missing"
        logger.info("Batch poll stub for run_id=%s (job present, no remote poll yet)", run_id)
        return job.get("status", "pending")
