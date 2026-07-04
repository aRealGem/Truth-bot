"""
Run manifest (design §3.2, §2 spend truth).

Every run emits a manifest: resolved spec snapshot, per-call cost records, and
dataset hashes — reproducible and diffable. Lane costs are appended to the spend
log via the existing `add_spend` path (P80); that write is represented here as a
SpendSink seam. In C1 this session the sink is Null (no kanban/wiki writes, and
no live costs), but the records are manifest-complete so wiring add_spend later
is a one-liner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Protocol

from .types import CallResult, Spec


@dataclass
class CostRecord:
    call_key: str
    role: str
    item_id: str
    lane: str
    provider: str
    model: str
    returned_model: str
    prompt_version: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


@dataclass
class RunManifest:
    strategy: str
    task: str
    resolved_spec: dict
    dataset_hash: str
    cost_records: list[CostRecord] = field(default_factory=list)
    lane_tally: dict = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    n_items: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None
    project: str = "hydramind"

    @classmethod
    def start(cls, strategy: str, spec: Spec, task: str, dataset_hash: str,
              n_items: int, project: str = "hydramind") -> "RunManifest":
        return cls(strategy=strategy, task=task, resolved_spec=spec.snapshot(),
                   dataset_hash=dataset_hash, n_items=n_items, project=project)

    def record(self, results: list[CallResult]) -> None:
        from .transport import call_key
        for r in results:
            self.cost_records.append(CostRecord(
                call_key=call_key(r.call),
                role=r.call.role, item_id=r.call.item_id, lane=r.lane.value,
                provider=r.call.binding.provider, model=r.call.binding.model,
                returned_model=r.returned_model,
                prompt_version=r.call.prompt.version,
                tokens_in=r.tokens_in, tokens_out=r.tokens_out, cost_usd=r.cost_usd,
            ))
            self.lane_tally[r.lane.value] = self.lane_tally.get(r.lane.value, 0) + 1
            self.total_cost_usd += r.cost_usd
            self.total_tokens_in += r.tokens_in
            self.total_tokens_out += r.tokens_out

    def model_mismatches(self) -> list[dict]:
        """Cost records where the returned model isn't the requested family —
        silent-fallback / unregistered-model detection (fail G5/equivalence)."""
        from .models import returned_ok
        out = []
        for c in self.cost_records:
            if not returned_ok(c.model, c.returned_model):
                out.append({"item_id": c.item_id, "role": c.role,
                            "requested": c.model, "returned": c.returned_model})
        return out

    def to_spend_records(self) -> list[dict]:
        """Lane-level spend rows for the P80 add_spend path (one per lane)."""
        by_lane: dict[str, dict] = {}
        for c in self.cost_records:
            row = by_lane.setdefault(c.lane, {
                "project": self.project, "lane": c.lane,
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0})
            row["cost_usd"] += c.cost_usd
            row["tokens_in"] += c.tokens_in
            row["tokens_out"] += c.tokens_out
            row["calls"] += 1
        return list(by_lane.values())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


class SpendSink(Protocol):
    def push(self, records: list[dict]) -> None: ...


class NullSpendSink:
    """Records spend rows in-memory but performs NO external write. This is the
    seam where the P80 `add_spend` MCP call is wired for live runs; deliberately
    inert under the no-writes constraint."""
    def __init__(self) -> None:
        self.pushed: list[dict] = []

    def push(self, records: list[dict]) -> None:
        self.pushed.extend(records)
