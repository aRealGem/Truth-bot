"""
JSONL telemetry logger for LLM adapter calls.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from truthbot.metrics.costs import estimate_cost

logger = logging.getLogger(__name__)

_run_id_var: ContextVar[Optional[str]] = ContextVar("tb_run_id", default=None)
_evidence_injected_var: ContextVar[bool] = ContextVar("tb_evidence_injected", default=True)
_synthesis_mode_var: ContextVar[str] = ContextVar("tb_synthesis_mode", default="live")


def get_run_id() -> Optional[str]:
    return _run_id_var.get()


def get_evidence_injected() -> bool:
    return _evidence_injected_var.get()


def get_synthesis_mode() -> str:
    return _synthesis_mode_var.get()


@contextmanager
def telemetry_run_context(
    *,
    run_id: Optional[str] = None,
    evidence_injected: bool = True,
    synthesis_mode: str = "live",
) -> Generator[None, None, None]:
    """Bind per-run telemetry fields for nested adapter calls."""
    rid_tok = _run_id_var.set(run_id)
    ev_tok = _evidence_injected_var.set(evidence_injected)
    sm_tok = _synthesis_mode_var.set(synthesis_mode)
    try:
        yield
    finally:
        _run_id_var.reset(rid_tok)
        _evidence_injected_var.reset(ev_tok)
        _synthesis_mode_var.reset(sm_tok)


@dataclass
class CallRecord:
    """A single adapter call record written to the JSONL log."""
    timestamp: str
    adapter_name: str
    model_id: str
    claim_id: str
    wall_clock_ms: int
    input_tokens: int
    output_tokens: int
    tool_call_count: int
    retrieved_url_count: int
    estimated_cost_usd: float
    status: str
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    openai_cached_prompt_tokens: int = 0
    gemini_cached_content_tokens: int = 0
    run_id: Optional[str] = None
    tier: str = "frontier"
    mode: str = "live"
    evidence_injected: bool = True
    batch_job_id: Optional[str] = None
    # Multi-claim batching: index within a multi-claim API call (0 = primary).
    # Usage is attributed to index 0 only so costs are not N-counted.
    batch_call_index: int = 0
    batch_call_id: str = ""
    claim_count: int = 1
    # Layer 5 — anti-hallucination fabrication-rate telemetry.
    # ``model_reported_source_count`` is the size of the model's raw
    # ``web_sources`` array before the Layer 1d ground-truth intersection.
    # ``stripped_source_count`` is how many of those URLs were stripped
    # because they did not appear in the search tool's retrieved-URL set
    # (i.e. were fabricated). The per-call fabrication rate is the ratio
    # of the two; ``finalize_run`` aggregates per-adapter and overall.
    model_reported_source_count: int = 0
    stripped_source_count: int = 0


class TelemetryLogger:
    """JSONL-based telemetry logger for adapter calls."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path

    def log(self, record: CallRecord) -> None:
        """Append a CallRecord to the JSONL log file."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as exc:
            logger.error("Failed to write telemetry record: %s", exc)

    @contextmanager
    def measure(
        self,
        adapter_name: str,
        model_id: str,
        claim_id: str,
        *,
        run_id: Optional[str] = None,
        tier: str = "frontier",
        mode: str = "live",
        evidence_injected: Optional[bool] = None,
        batch_job_id: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Context manager that times an adapter call and writes a telemetry record.

        The caller fills in the yielded dict with:
          - input_tokens, output_tokens
          - cache_read_input_tokens, cache_creation_input_tokens (Anthropic)
          - openai_cached_prompt_tokens (OpenAI Responses cached prefix)
          - gemini_cached_content_tokens (Gemini context cache)
          - tool_call_count, retrieved_url_count, status
        """
        start = time.monotonic()
        eff_mode = mode if mode != "live" else get_synthesis_mode()
        data: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_call_count": 0,
            "retrieved_url_count": 0,
            "status": "ok",
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "openai_cached_prompt_tokens": 0,
            "gemini_cached_content_tokens": 0,
            "mode": eff_mode,
        }
        eff_run = run_id if run_id is not None else get_run_id()
        eff_ev = evidence_injected if evidence_injected is not None else get_evidence_injected()
        try:
            yield data
        finally:
            wall_clock_ms = int((time.monotonic() - start) * 1000)
            eff_mode = str(data.get("mode") or get_synthesis_mode())
            eff_batch_id = data.get("batch_job_id") or batch_job_id
            estimated_cost = estimate_cost(
                adapter_name,
                model_id,
                data["input_tokens"],
                data["output_tokens"],
                cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0)),
                cache_creation_input_tokens=int(data.get("cache_creation_input_tokens", 0)),
                openai_cached_prompt_tokens=int(data.get("openai_cached_prompt_tokens", 0)),
                gemini_cached_content_tokens=int(data.get("gemini_cached_content_tokens", 0)),
                mode=eff_mode,
                batch_job_id=eff_batch_id,
            )
            record = CallRecord(
                timestamp=datetime.utcnow().isoformat(),
                adapter_name=adapter_name,
                model_id=model_id,
                claim_id=claim_id,
                wall_clock_ms=wall_clock_ms,
                input_tokens=int(data["input_tokens"]),
                output_tokens=int(data["output_tokens"]),
                tool_call_count=int(data["tool_call_count"]),
                retrieved_url_count=int(data["retrieved_url_count"]),
                estimated_cost_usd=estimated_cost,
                status=str(data.get("status", "ok")),
                cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0)),
                cache_creation_input_tokens=int(data.get("cache_creation_input_tokens", 0)),
                openai_cached_prompt_tokens=int(data.get("openai_cached_prompt_tokens", 0)),
                gemini_cached_content_tokens=int(data.get("gemini_cached_content_tokens", 0)),
                run_id=eff_run,
                tier=tier,
                mode=eff_mode,
                evidence_injected=eff_ev,
                batch_job_id=eff_batch_id,
                batch_call_index=int(data.get("batch_call_index", 0) or 0),
                batch_call_id=str(data.get("batch_call_id", "") or ""),
                claim_count=int(data.get("claim_count", 1) or 1),
                model_reported_source_count=int(
                    data.get("model_reported_source_count", 0) or 0
                ),
                stripped_source_count=int(
                    data.get("stripped_source_count", 0) or 0
                ),
            )
            self.log(record)


_telemetry_instance: Optional[TelemetryLogger] = None


def get_telemetry() -> TelemetryLogger:
    """Return the module-level singleton TelemetryLogger."""
    global _telemetry_instance
    if _telemetry_instance is None:
        from truthbot.config import settings
        log_path = settings.metrics_dir / "adapter_calls.jsonl"
        _telemetry_instance = TelemetryLogger(log_path)
    return _telemetry_instance


def finalize_run(run_id: str, *, jsonl_path: Optional[Path] = None) -> dict:
    """
    Aggregate adapter_calls.jsonl rows for ``run_id`` and write:

    - ``metrics/run_summaries/<run_id>.json``
    - append one row to ``metrics/triage_roi.csv``
    """
    from truthbot.config import settings

    path = jsonl_path or (settings.metrics_dir / "adapter_calls.jsonl")
    summary: dict = {
        "run_id": run_id,
        "total_calls": 0,
        "total_cost_usd": 0.0,
        "by_adapter": {},
        "by_tier": {},
        "by_mode": {},
        # Layer 5 — fabrication-rate aggregation.
        # ``fabrication`` totals are summed across calls; the rate is
        # computed once at the end so divide-by-zero is bounded.
        "fabrication": {
            "model_reported_sources_total": 0,
            "stripped_sources_total": 0,
            "fabrication_rate": 0.0,
            "by_adapter": {},
        },
    }
    if not path.exists():
        logger.warning("finalize_run: no telemetry file at %s", path)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("run_id") != run_id:
                    continue
                summary["total_calls"] += 1
                cost = float(rec.get("estimated_cost_usd", 0.0))
                summary["total_cost_usd"] += cost
                ad = rec.get("adapter_name", "unknown")
                summary["by_adapter"][ad] = summary["by_adapter"].get(ad, 0.0) + cost
                tier = rec.get("tier", "frontier")
                summary["by_tier"][tier] = summary["by_tier"].get(tier, 0.0) + cost
                m = rec.get("mode", "live")
                summary["by_mode"][m] = summary["by_mode"].get(m, 0.0) + cost

                reported = int(rec.get("model_reported_source_count", 0) or 0)
                stripped = int(rec.get("stripped_source_count", 0) or 0)
                fab = summary["fabrication"]
                fab["model_reported_sources_total"] += reported
                fab["stripped_sources_total"] += stripped
                ad_fab = fab["by_adapter"].setdefault(
                    ad, {"reported": 0, "stripped": 0, "rate": 0.0}
                )
                ad_fab["reported"] += reported
                ad_fab["stripped"] += stripped

    fab = summary["fabrication"]
    if fab["model_reported_sources_total"] > 0:
        fab["fabrication_rate"] = (
            fab["stripped_sources_total"] / fab["model_reported_sources_total"]
        )
    for ad, ad_fab in fab["by_adapter"].items():
        if ad_fab["reported"] > 0:
            ad_fab["rate"] = ad_fab["stripped"] / ad_fab["reported"]

    out_dir = settings.metrics_dir / "run_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{run_id}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    roi_path = settings.metrics_dir / "triage_roi.csv"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    triage_cost = summary["by_tier"].get("triage", 0.0)
    frontier_cost = summary["by_tier"].get("frontier", 0.0) + summary["by_tier"].get(
        "frontier_shadow", 0.0
    )
    row = {
        "run_id": run_id,
        "total_calls": summary["total_calls"],
        "total_cost_usd": f"{summary['total_cost_usd']:.6f}",
        "triage_cost_usd": f"{triage_cost:.6f}",
        "frontier_cost_usd": f"{frontier_cost:.6f}",
        "net_savings_placeholder": "",
    }
    fieldnames = list(row.keys())
    write_header = not roi_path.exists()
    with roi_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)

    return summary
