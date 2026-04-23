"""
Async batch-job orchestration for LLM verdict synthesis.

In ``--mode batch`` the ``_run_publish`` pipeline:

  1. Builds a per-claim ``build_batch_payload`` for every batch-capable adapter.
  2. Submits one batch per provider via that provider's native batch API
     (Anthropic Message Batches, OpenAI Batch against ``/v1/responses``,
     Gemini batches when available).
  3. Runs any non-batch adapters (currently xAI / Grok) live during submit
     and spools their ``ModelVerdict`` rows into
     ``metrics/batch_sidecar/<run_id>.jsonl`` so they can be merged at poll time.
  4. Writes a descriptor file ``metrics/batch_jobs/<run_id>.json`` containing
     ``provider_jobs`` (batch IDs), ``work_units`` (per-claim manifest), and
     ``sidecar_path``. Submit exits after this — no live frontier calls.

``truthbot batch poll <run_id>`` later re-reads that descriptor, retrieves each
provider's results, parses them through ``adapter.parse_batch_response``,
merges Grok sidecar rows, builds consensus ``VerdictBundle`` s, writes them
into the on-disk bundle cache, publishes the site, and emits one telemetry
row per verdict with ``mode='batch'`` + the real ``batch_job_id`` (so the 50%
batch discount in ``costs.estimate_cost`` only lands on actually-batched calls).

Failure modes are explicit: missing descriptor, pending provider job, or a
provider error short-circuit before the site is regenerated.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from truthbot.models import Claim, Evidence, ModelVerdict
from truthbot.verify.adapters.base import LLMAdapter

logger = logging.getLogger(__name__)


# ── Descriptor I/O ────────────────────────────────────────────────────────────


def batch_job_path(metrics_dir: Path, run_id: str) -> Path:
    return metrics_dir / "batch_jobs" / f"{run_id}.json"


def sidecar_path(metrics_dir: Path, run_id: str) -> Path:
    return metrics_dir / "batch_sidecar" / f"{run_id}.jsonl"


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


# ── Per-provider batch transports ─────────────────────────────────────────────


@dataclass
class BatchSubmission:
    """One provider's submitted batch: its ID + the custom_id → claim_id map."""
    provider: str
    batch_id: str
    custom_id_to_claim: dict[str, str]
    model_id: str


def _custom_id(claim_id: str, adapter_name: str) -> str:
    # Anthropic custom_ids must be ≤64 chars; keep prefix short + deterministic.
    short = claim_id[:40]
    return f"{adapter_name}::{short}"


def _multi_custom_id(adapter_name: str) -> str:
    """Opaque custom_id for a multi-claim request (≤64 chars, Anthropic safe)."""
    return f"{adapter_name}::multi::{uuid.uuid4().hex[:16]}"


def chunk_claims_with_evidence(
    items: list[tuple[Claim, list[Evidence]]],
    chunk_size: int,
) -> list[list[tuple[Claim, list[Evidence]]]]:
    """
    Split ``items`` into contiguous sub-lists of length ``chunk_size``.

    ``chunk_size`` values less than 1 fall back to 1 (every claim in its own
    request). The last sub-list may be shorter.
    """
    if chunk_size < 1:
        chunk_size = 1
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _anthropic_submit(
    adapter: LLMAdapter,
    requests: list[dict],
    custom_id_to_claim: dict[str, str],
) -> BatchSubmission:
    """Submit a Message Batches job; returns BatchSubmission."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.create(requests=requests)
    logger.info(
        "Anthropic batch submitted: id=%s, %d requests", batch.id, len(requests)
    )
    return BatchSubmission(
        provider="anthropic",
        batch_id=batch.id,
        custom_id_to_claim=custom_id_to_claim,
        model_id=adapter.model_id,
    )


def _anthropic_status(batch_id: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.retrieve(batch_id)
    status = getattr(batch, "processing_status", "in_progress")
    counts = getattr(batch, "request_counts", None)
    done = 0
    total = 0
    if counts is not None:
        done = (
            (getattr(counts, "succeeded", 0) or 0)
            + (getattr(counts, "errored", 0) or 0)
            + (getattr(counts, "canceled", 0) or 0)
            + (getattr(counts, "expired", 0) or 0)
        )
        total = done + (getattr(counts, "processing", 0) or 0)
    normalized = "complete" if status == "ended" else "pending"
    return {"status": normalized, "raw_status": status, "done": done, "total": total}


def _anthropic_results(batch_id: str) -> list[dict]:
    """Return a list of ``{custom_id, message_or_error}`` rows."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out: list[dict] = []
    for entry in client.messages.batches.results(batch_id):
        custom_id = getattr(entry, "custom_id", None)
        result = getattr(entry, "result", None)
        rtype = getattr(result, "type", "errored") if result else "errored"
        if rtype == "succeeded":
            msg = getattr(result, "message", None)
            out.append({"custom_id": custom_id, "status": "succeeded", "message": msg})
        else:
            err = getattr(result, "error", None)
            out.append(
                {
                    "custom_id": custom_id,
                    "status": rtype,
                    "error": str(err) if err else f"batch entry {rtype}",
                }
            )
    return out


def _openai_submit(
    adapter: LLMAdapter,
    requests: list[dict],
    custom_id_to_claim: dict[str, str],
    *,
    metrics_dir: Path,
    run_id: str,
) -> BatchSubmission:
    """Write a JSONL file, upload via /files, then create a Batch."""
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tmpdir = metrics_dir / "batch_inputs"
    tmpdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = tmpdir / f"openai-{run_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    logger.info(
        "OpenAI batch submitted: id=%s, input_file=%s, %d requests",
        batch.id,
        uploaded.id,
        len(requests),
    )
    return BatchSubmission(
        provider="openai",
        batch_id=batch.id,
        custom_id_to_claim=custom_id_to_claim,
        model_id=adapter.model_id,
    )


def _openai_status(batch_id: str) -> dict:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(batch_id)
    raw = getattr(batch, "status", "validating")
    counts = getattr(batch, "request_counts", None)
    total = getattr(counts, "total", 0) or 0 if counts else 0
    done = (getattr(counts, "completed", 0) or 0) if counts else 0
    normalized = "complete" if raw == "completed" else (
        "failed" if raw in {"failed", "expired", "cancelled"} else "pending"
    )
    return {"status": normalized, "raw_status": raw, "done": done, "total": total}


def _openai_results(batch_id: str) -> list[dict]:
    """Download the output file and parse JSONL into rows."""
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(batch_id)
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        return []
    content = client.files.content(output_file_id)
    # .content returns a binary response; normalize to text
    text = getattr(content, "text", None)
    if text is None:
        text = content.read().decode("utf-8") if hasattr(content, "read") else str(content)

    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id")
        err = row.get("error")
        resp = row.get("response") or {}
        status_code = resp.get("status_code", 200)
        body = resp.get("body")
        if err or status_code >= 400 or body is None:
            out.append(
                {
                    "custom_id": custom_id,
                    "status": "errored",
                    "error": err or f"HTTP {status_code}",
                }
            )
        else:
            out.append({"custom_id": custom_id, "status": "succeeded", "body": body})
    return out


# ── Sidecar I/O (for non-batch providers run live at submit time) ─────────────


def _append_sidecar(path: Path, verdict: ModelVerdict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(verdict.model_dump_json() + "\n")


def load_sidecar(path: Path) -> list[ModelVerdict]:
    if not path.exists():
        return []
    out: list[ModelVerdict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(ModelVerdict.model_validate_json(line))
        except Exception as exc:
            logger.warning("skipping unreadable sidecar row: %s", exc)
    return out


# ── Orchestrator ──────────────────────────────────────────────────────────────


class BatchDispatcher:
    """
    Orchestrates per-provider batch submit + poll + reconcile for a run.

    A ``run_id`` has:
      - one descriptor file: ``metrics/batch_jobs/<run_id>.json``
      - one optional sidecar:  ``metrics/batch_sidecar/<run_id>.jsonl`` (Grok live rows)
    """

    def __init__(self, metrics_dir: Path) -> None:
        self._metrics_dir = metrics_dir

    # -- Legacy surface retained for callers/tests -----------------------------

    def record_job(
        self,
        run_id: str,
        *,
        transcript_meta: dict[str, Any],
        work_units: list[dict[str, Any]],
        provider_hints: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Write a minimal (pending, no-live-submit) descriptor — used by tests."""
        payload = {
            "run_id": run_id,
            "status": "pending",
            "transcript_meta": transcript_meta,
            "work_units": work_units,
            "provider_jobs": provider_hints or {},
            "sidecar_path": str(sidecar_path(self._metrics_dir, run_id)),
            "note": "record_job descriptor; use BatchDispatcher.submit for live runs.",
        }
        return write_batch_job(self._metrics_dir, run_id, payload)

    def poll(self, run_id: str) -> str:
        """
        Return a short status string: ``missing`` | ``pending`` | ``complete`` | ``failed``.

        Does not reconcile results — use ``reconcile()`` for the full merge+publish path.
        """
        job = read_batch_job(self._metrics_dir, run_id)
        if not job:
            return "missing"
        provider_jobs = job.get("provider_jobs") or {}
        if not provider_jobs:
            return job.get("status", "pending")
        statuses = []
        for provider, entry in provider_jobs.items():
            batch_id = entry.get("batch_id")
            if not batch_id:
                statuses.append("pending")
                continue
            try:
                st = self._provider_status(provider, batch_id)
                statuses.append(st["status"])
                entry["last_status"] = st
            except Exception as exc:
                logger.error("poll: %s status error: %s", provider, exc)
                statuses.append("pending")
        if any(s == "failed" for s in statuses):
            return "failed"
        if all(s == "complete" for s in statuses):
            return "complete"
        return "pending"

    # -- New submit/reconcile path --------------------------------------------

    def submit(
        self,
        run_id: str,
        *,
        adapters: list[LLMAdapter],
        claims_with_evidence: list[tuple[Claim, list[Evidence]]],
        transcript_meta: dict[str, Any],
        inject_evidence: bool,
        sidecar_live_adapters: Optional[list[LLMAdapter]] = None,
        claims_per_request: int = 1,
        max_evidence_per_claim_in_batch: int = 5,
    ) -> Path:
        """
        Submit one batch per ``supports_batch=True`` adapter; run sidecar adapters live.

        When ``claims_per_request > 1`` the dispatcher collapses that many atomic
        claims into a single provider request (per adapter, clamped by each
        adapter's ``max_claims_per_request``). The descriptor records either
        ``custom_id_to_claim`` (single-claim, legacy) or
        ``custom_id_to_claims`` (multi-claim list) per provider.

        Writes the descriptor and returns its path. Does not publish a site.
        """
        provider_jobs: dict[str, dict] = {}
        work_units: list[dict] = [
            {
                "claim_id": claim.id,
                "claim": claim.model_dump(mode="json"),
                "evidence_count": len(ev),
            }
            for claim, ev in claims_with_evidence
        ]

        requested = max(1, int(claims_per_request))

        for adapter in adapters:
            if not getattr(adapter, "supports_batch", False):
                continue

            adapter_cap = max(1, int(getattr(adapter, "max_claims_per_request", 1)))
            chunk_size = min(requested, adapter_cap)
            is_multi = chunk_size > 1 and len(claims_with_evidence) > 1

            requests: list[dict] = []
            custom_id_to_claim: dict[str, str] = {}
            custom_id_to_claims: dict[str, list[str]] = {}

            if is_multi:
                chunks = chunk_claims_with_evidence(claims_with_evidence, chunk_size)
                for chunk in chunks:
                    claims = [c for c, _ in chunk]
                    evidence_by_claim = {c.id: ev for c, ev in chunk}
                    try:
                        payload = adapter.build_multi_batch_payload(
                            claims,
                            evidence_by_claim,
                            inject_evidence=inject_evidence,
                            max_evidence_per_claim=max_evidence_per_claim_in_batch,
                        )
                    except NotImplementedError:
                        logger.warning(
                            "submit: %s lacks build_multi_batch_payload; falling back to single-claim",
                            adapter.adapter_name,
                        )
                        is_multi = False
                        requests = []
                        custom_id_to_claim = {}
                        custom_id_to_claims = {}
                        break

                    cid = _multi_custom_id(adapter.adapter_name)
                    if adapter.adapter_name == "anthropic":
                        requests.append({"custom_id": cid, "params": payload})
                    elif adapter.adapter_name == "openai":
                        requests.append(
                            {
                                "custom_id": cid,
                                "method": "POST",
                                "url": "/v1/responses",
                                "body": payload,
                            }
                        )
                    else:
                        logger.warning(
                            "submit: no multi-claim transport for %s; skipping chunk",
                            adapter.adapter_name,
                        )
                        continue
                    custom_id_to_claims[cid] = [c.id for c in claims]

            if not is_multi:
                for claim, evidence in claims_with_evidence:
                    cid = _custom_id(claim.id, adapter.adapter_name)
                    payload = adapter.build_batch_payload(
                        claim, evidence, inject_evidence=inject_evidence
                    )
                    if adapter.adapter_name == "anthropic":
                        requests.append({"custom_id": cid, "params": payload})
                    elif adapter.adapter_name == "openai":
                        requests.append(
                            {
                                "custom_id": cid,
                                "method": "POST",
                                "url": "/v1/responses",
                                "body": payload,
                            }
                        )
                    else:
                        logger.warning(
                            "submit: no batch transport for %s; skipping",
                            adapter.adapter_name,
                        )
                        continue
                    custom_id_to_claim[cid] = claim.id

            if not requests:
                continue

            try:
                if adapter.adapter_name == "anthropic":
                    sub = _anthropic_submit(
                        adapter, requests, custom_id_to_claim or custom_id_to_claims
                    )
                elif adapter.adapter_name == "openai":
                    sub = _openai_submit(
                        adapter,
                        requests,
                        custom_id_to_claim or custom_id_to_claims,
                        metrics_dir=self._metrics_dir,
                        run_id=run_id,
                    )
                else:
                    continue
            except Exception as exc:
                logger.error(
                    "submit: %s batch submission failed: %s", adapter.adapter_name, exc
                )
                provider_jobs[adapter.adapter_name] = {
                    "batch_id": None,
                    "error": str(exc),
                    "status": "failed",
                    "model_id": adapter.model_id,
                }
                continue

            entry: dict[str, Any] = {
                "batch_id": sub.batch_id,
                "model_id": sub.model_id,
                "request_count": len(requests),
                "status": "pending",
                "chunk_size": chunk_size if is_multi else 1,
            }
            if is_multi:
                entry["custom_id_to_claims"] = custom_id_to_claims
            else:
                entry["custom_id_to_claim"] = custom_id_to_claim
            provider_jobs[adapter.adapter_name] = entry

        # Sidecar: run non-batch adapters live, one worker per adapter × 4 claims.
        sidecar = sidecar_path(self._metrics_dir, run_id)
        if sidecar_live_adapters:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            if sidecar.exists():
                sidecar.unlink()  # start fresh

            def _one(adapter, claim, evidence) -> Optional[ModelVerdict]:
                try:
                    verdict = adapter.call(
                        claim,
                        evidence,
                        inject_evidence=inject_evidence,
                        run_id=run_id,
                    )
                    verdict.tier = "frontier"
                    verdict.synthesis_mode = "live"
                    return verdict
                except Exception as exc:
                    logger.error(
                        "sidecar %s failed on claim %s: %s",
                        adapter.adapter_name,
                        claim.id,
                        exc,
                    )
                    return None

            tasks = [
                (adapter, claim, ev)
                for adapter in sidecar_live_adapters
                for claim, ev in claims_with_evidence
            ]
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(tasks)))) as pool:
                futures = {pool.submit(_one, *t): t for t in tasks}
                for fut in as_completed(futures):
                    v = fut.result()
                    if v is not None:
                        _append_sidecar(sidecar, v)

        payload = {
            "run_id": run_id,
            "status": "submitted",
            "transcript_meta": transcript_meta,
            "work_units": work_units,
            "provider_jobs": provider_jobs,
            "sidecar_path": str(sidecar),
            "inject_evidence": inject_evidence,
            "claims_per_request_requested": requested,
            "max_evidence_per_claim_in_batch": max_evidence_per_claim_in_batch,
        }
        return write_batch_job(self._metrics_dir, run_id, payload)

    def fetch_results(self, run_id: str) -> dict[str, list[dict]]:
        """Fetch raw result rows from each provider. Assumes all are complete."""
        job = read_batch_job(self._metrics_dir, run_id)
        if not job:
            return {}
        results: dict[str, list[dict]] = {}
        for provider, entry in (job.get("provider_jobs") or {}).items():
            batch_id = entry.get("batch_id")
            if not batch_id:
                continue
            try:
                results[provider] = self._provider_results(provider, batch_id)
            except Exception as exc:
                logger.error("fetch_results: %s failed: %s", provider, exc)
                results[provider] = []
        return results

    # -- Internal provider dispatch --------------------------------------------

    def _provider_status(self, provider: str, batch_id: str) -> dict:
        if provider == "anthropic":
            return _anthropic_status(batch_id)
        if provider == "openai":
            return _openai_status(batch_id)
        raise ValueError(f"unknown batch provider: {provider}")

    def _provider_results(self, provider: str, batch_id: str) -> list[dict]:
        if provider == "anthropic":
            return _anthropic_results(batch_id)
        if provider == "openai":
            return _openai_results(batch_id)
        raise ValueError(f"unknown batch provider: {provider}")


# ── Result parsing helper ─────────────────────────────────────────────────────


def reconcile_run(
    metrics_dir: Path,
    run_id: str,
    *,
    adapters_by_name: dict[str, LLMAdapter],
    engine,
) -> dict[str, Any]:
    """
    Reconcile a submitted run: poll → parse → merge sidecar → build+cache bundles.

    Returns a dict with:
      - ``status``: ``missing`` | ``pending`` | ``complete`` | ``failed``
      - ``pending_providers``: list of (provider, raw_status) when pending
      - ``bundles``: list of ``VerdictBundle`` (when complete)
      - ``triaged_bundles``: list of ``VerdictBundle`` loaded from cache (complete)
      - ``transcript_meta``: dict from the descriptor
      - ``descriptor``: the full descriptor dict
    """
    from truthbot.metrics.telemetry import get_telemetry, telemetry_run_context
    from truthbot.models import Claim, VerdictBundle

    descriptor = read_batch_job(metrics_dir, run_id)
    if not descriptor:
        return {"status": "missing", "descriptor": None}

    provider_jobs = descriptor.get("provider_jobs") or {}
    transcript_meta = descriptor.get("transcript_meta", {})
    work_units = descriptor.get("work_units") or []

    # Check readiness
    dispatcher = BatchDispatcher(metrics_dir)
    pending: list[tuple[str, dict]] = []
    for provider, entry in provider_jobs.items():
        batch_id = entry.get("batch_id")
        if not batch_id:
            # submission failed during submit; treat as failed provider
            pending.append((provider, {"status": "failed", "error": entry.get("error")}))
            continue
        try:
            st = dispatcher._provider_status(provider, batch_id)
        except Exception as exc:
            pending.append((provider, {"status": "error", "error": str(exc)}))
            continue
        entry["last_status"] = st
        if st["status"] != "complete":
            pending.append((provider, st))

    if pending:
        # Persist last_status so the user can see progress on re-poll.
        write_batch_job(metrics_dir, run_id, descriptor)
        return {
            "status": "pending",
            "pending_providers": pending,
            "descriptor": descriptor,
            "transcript_meta": transcript_meta,
        }

    # Rebuild Claim objects from work_units
    claim_by_id: dict[str, Claim] = {}
    evidence_count_by_id: dict[str, int] = {}
    for unit in work_units:
        claim_json = unit.get("claim")
        if not claim_json:
            continue
        try:
            claim = Claim.model_validate(claim_json)
        except Exception as exc:
            logger.error("reconcile: could not rehydrate claim: %s", exc)
            continue
        claim_by_id[claim.id] = claim
        evidence_count_by_id[claim.id] = int(unit.get("evidence_count", 0))

    # Collect verdicts per provider
    verdicts_by_claim: dict[str, list] = {cid: [] for cid in claim_by_id}
    speaker = transcript_meta.get("speaker", "")
    date_str = transcript_meta.get("date", "")

    for provider, entry in provider_jobs.items():
        batch_id = entry.get("batch_id")
        if not batch_id:
            continue
        adapter = adapters_by_name.get(provider)
        if adapter is None:
            logger.warning("reconcile: no live adapter for %s", provider)
            continue
        single_map = entry.get("custom_id_to_claim") or {}
        multi_map = entry.get("custom_id_to_claims") or {}
        try:
            rows = dispatcher._provider_results(provider, batch_id)
        except Exception as exc:
            logger.error("reconcile: %s result fetch failed: %s", provider, exc)
            continue

        parsed = parse_provider_results(
            provider,
            rows,
            adapter,
            claim_by_id,
            custom_id_to_claim=single_map,
            custom_id_to_claims=multi_map,
        )

        # Chunk size per call — used to stamp `claim_count` on telemetry rows
        # so downstream aggregation (and spot-checks) can see which calls were
        # multi-claim without re-reading descriptors.
        chunk_size_by_call: dict[str, int] = {
            cid: len(cids) for cid, cids in multi_map.items()
        }

        # Telemetry: emit one row per verdict. Only the index-0 verdict in a
        # multi-claim call carries usage; siblings log zero usage with
        # claim_count > 1 so costs.estimate_cost doesn't N-count a single
        # batched call. The real batch_job_id unlocks the 50% batch discount.
        log = get_telemetry()
        with telemetry_run_context(
            run_id=run_id,
            evidence_injected=bool(descriptor.get("inject_evidence", False)),
            synthesis_mode="batch",
        ):
            for mv in parsed:
                with log.measure(
                    adapter.adapter_name,
                    mv.model_id,
                    mv.claim_id,
                    tier="frontier",
                    mode="batch",
                    batch_job_id=batch_id,
                ) as td:
                    td["input_tokens"] = 0
                    td["output_tokens"] = 0
                    td["tool_call_count"] = 0
                    td["retrieved_url_count"] = len(mv.web_sources or [])
                    td["status"] = "ok" if not mv.no_response else "api_error"
                    td["cache_read_input_tokens"] = int(mv.cached_input_tokens or 0)
                    td["batch_call_index"] = int(mv.batch_call_index)
                    td["batch_call_id"] = mv.batch_call_id or ""
                    td["claim_count"] = chunk_size_by_call.get(
                        mv.batch_call_id, 1
                    )
                verdicts_by_claim.setdefault(mv.claim_id, []).append(mv)

    # Merge sidecar (Grok) rows
    sidecar = sidecar_path(metrics_dir, run_id)
    for mv in load_sidecar(sidecar):
        verdicts_by_claim.setdefault(mv.claim_id, []).append(mv)

    # Build bundles
    bundles = []
    for cid, claim in claim_by_id.items():
        mvs = verdicts_by_claim.get(cid, [])
        bundle = engine.finalize_bundle(
            claim,
            speaker=speaker,
            date_str=date_str,
            model_verdicts=mvs,
            evidence_count=evidence_count_by_id.get(cid, 0),
        )
        bundles.append(bundle)

    # Pull any triaged / previously-cached bundles (the submit step cached them).
    triaged: list = []
    triaged_ids = transcript_meta.get("triaged_claim_ids") or []
    triaged_claims = transcript_meta.get("triaged_claims") or []
    for claim_json in triaged_claims:
        try:
            claim = Claim.model_validate(claim_json)
        except Exception:
            continue
        # Reuse engine cache lookup via maybe_resolve_early — will hit cache.
        bundle, _ = engine.maybe_resolve_early(claim, speaker=speaker, date_str=date_str)
        if bundle is not None:
            triaged.append(bundle)

    descriptor["status"] = "complete"
    write_batch_job(metrics_dir, run_id, descriptor)

    return {
        "status": "complete",
        "descriptor": descriptor,
        "transcript_meta": transcript_meta,
        "bundles": bundles,
        "triaged_bundles": triaged,
        "triaged_claim_ids": triaged_ids,
    }


def parse_provider_results(
    provider: str,
    rows: Iterable[dict],
    adapter: LLMAdapter,
    claim_by_id: dict[str, Claim],
    custom_id_to_claim: Optional[dict[str, str]] = None,
    custom_id_to_claims: Optional[dict[str, list[str]]] = None,
) -> list[ModelVerdict]:
    """
    Parse raw provider rows into ``ModelVerdict`` s.

    Accepts either a single-claim mapping (``custom_id_to_claim``) or a
    multi-claim mapping (``custom_id_to_claims``) per custom_id — or both, for
    descriptors that mix single and multi requests in one provider job.
    """
    from truthbot.models import Confidence, ModelVerdict, VerdictLabel

    single_map = custom_id_to_claim or {}
    multi_map = custom_id_to_claims or {}
    verdicts: list[ModelVerdict] = []

    for row in rows:
        custom_id = row.get("custom_id")

        if custom_id in multi_map:
            claim_ids = multi_map[custom_id]
            claims = [claim_by_id[cid] for cid in claim_ids if cid in claim_by_id]
            missing = [cid for cid in claim_ids if cid not in claim_by_id]
            for cid in missing:
                logger.warning(
                    "parse_provider_results[%s]: multi custom_id %s references unknown claim %s",
                    provider,
                    custom_id,
                    cid,
                )
            if not claims:
                continue

            if row.get("status") != "succeeded":
                err = row.get("error", "unknown")
                for idx, claim in enumerate(claims):
                    verdicts.append(
                        ModelVerdict(
                            adapter_name=adapter.adapter_name,
                            model_id=adapter.model_id,
                            claim_id=claim.id,
                            label=VerdictLabel.UNVERIFIABLE,
                            confidence=Confidence.LOW,
                            explanation=f"Batch error: {err}",
                            tier="frontier",
                            synthesis_mode="batch",
                            no_response=True,
                            batch_call_index=idx,
                            batch_call_id=custom_id or "",
                        )
                    )
                continue

            raw = row.get("message") if provider == "anthropic" else row.get("body")
            try:
                verdicts.extend(
                    adapter.parse_multi_batch_response(
                        raw, claims, batch_call_id=custom_id or ""
                    )
                )
            except NotImplementedError:
                logger.error(
                    "parse_provider_results[%s]: %s adapter has no parse_multi_batch_response",
                    provider,
                    adapter.adapter_name,
                )
            except Exception as exc:
                logger.error(
                    "parse_provider_results[%s]: multi parse failed (call=%s): %s",
                    provider,
                    custom_id,
                    exc,
                )
                for idx, claim in enumerate(claims):
                    verdicts.append(
                        ModelVerdict(
                            adapter_name=adapter.adapter_name,
                            model_id=adapter.model_id,
                            claim_id=claim.id,
                            label=VerdictLabel.UNVERIFIABLE,
                            confidence=Confidence.LOW,
                            explanation=f"Multi-claim parse error: {exc}",
                            tier="frontier",
                            synthesis_mode="batch",
                            no_response=True,
                            batch_call_index=idx,
                            batch_call_id=custom_id or "",
                        )
                    )
            continue

        # Legacy single-claim path
        claim_id = single_map.get(custom_id)
        if not claim_id:
            logger.warning(
                "parse_provider_results[%s]: unknown custom_id %s", provider, custom_id
            )
            continue
        claim = claim_by_id.get(claim_id)
        if claim is None:
            logger.warning(
                "parse_provider_results[%s]: claim_id %s not in manifest",
                provider,
                claim_id,
            )
            continue
        if row.get("status") != "succeeded":
            verdicts.append(
                ModelVerdict(
                    adapter_name=adapter.adapter_name,
                    model_id=adapter.model_id,
                    claim_id=claim.id,
                    label=VerdictLabel.UNVERIFIABLE,
                    confidence=Confidence.LOW,
                    explanation=f"Batch error: {row.get('error', 'unknown')}",
                    tier="frontier",
                    synthesis_mode="batch",
                    no_response=True,
                )
            )
            continue
        raw = row.get("message") if provider == "anthropic" else row.get("body")
        try:
            verdicts.append(adapter.parse_batch_response(raw, claim))
        except Exception as exc:
            logger.error(
                "parse_provider_results[%s]: parse failed for claim %s: %s",
                provider,
                claim.id,
                exc,
            )
    return verdicts
