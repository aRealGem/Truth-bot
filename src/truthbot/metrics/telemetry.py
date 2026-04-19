"""
JSONL telemetry logger for LLM adapter calls.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from truthbot.metrics.costs import estimate_cost

logger = logging.getLogger(__name__)


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
    ) -> Generator[dict, None, None]:
        """
        Context manager that times an adapter call and writes a telemetry record.

        The caller fills in the yielded dict with:
          - input_tokens (int)
          - output_tokens (int)
          - tool_call_count (int)
          - retrieved_url_count (int)
          - status (str, default "ok")
        """
        start = time.monotonic()
        data: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_call_count": 0,
            "retrieved_url_count": 0,
            "status": "ok",
        }
        try:
            yield data
        finally:
            wall_clock_ms = int((time.monotonic() - start) * 1000)
            estimated_cost = estimate_cost(
                adapter_name,
                model_id,
                data["input_tokens"],
                data["output_tokens"],
            )
            record = CallRecord(
                timestamp=datetime.utcnow().isoformat(),
                adapter_name=adapter_name,
                model_id=model_id,
                claim_id=claim_id,
                wall_clock_ms=wall_clock_ms,
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                tool_call_count=data["tool_call_count"],
                retrieved_url_count=data["retrieved_url_count"],
                estimated_cost_usd=estimated_cost,
                status=data.get("status", "ok"),
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
