"""Per-seat panel insights from published provenance (P67.10 / T4.1).

The v1 model-insights page summarized a single reconciled pseudo-model
(0% dissent by construction — audit F4). This module computes the REAL
per-seat story from ``claims.json`` provenance (``panel_by_role``,
``panel_escalated``, ``crm114_*``), per report:

* per-seat verdict distributions and False-rates,
* escalation rate (proposer/critic disagreement share),
* arbiter side-taking (sided with proposer / critic / neither),
* Severity-Classifier override direction counts.

The acceptance fixture is the audit's F10/F11 numbers as REPRODUCED
2026-07-21 (the audit's own override counts were exactly 2x): Trump —
critic False-rate 50.0%, escalation 53.4%, arbiter sided proposer 61/95,
overrides 7 MISLEADING→FALSE / 4 FALSE→MISLEADING / 5 DISAGREEMENT→
MISLEADING; Biden — 13.5% / 21.6% / 14/24 / 1 MISLEADING→FALSE. Pinned by
tests against the committed site data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeatStats:
    role: str
    label_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.label_counts.values())

    def rate(self, label: str) -> float:
        return self.label_counts.get(label, 0) / self.total if self.total else 0.0


@dataclass
class ReportSeatInsights:
    report_id: str
    n_claims: int = 0
    seats: dict[str, SeatStats] = field(default_factory=dict)
    escalated: int = 0
    arbiter_sided: dict[str, int] = field(default_factory=dict)  # proposer/critic/neither
    overrides: dict[str, int] = field(default_factory=dict)      # "STAGE1→FINAL" -> n

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.n_claims if self.n_claims else 0.0


def _first(labels) -> str:
    if isinstance(labels, list):
        return str(labels[0]) if labels else ""
    return str(labels or "")


def _norm(label: str) -> str:
    """Normalize a seat/crm label to display casing ('False', 'Disagreement')."""
    return str(label or "").strip().capitalize() if label else ""


def compute_seat_insights(claims: list[dict]) -> dict[str, ReportSeatInsights]:
    """Per-report seat insights keyed by report_id. Pure function of the
    claims index — every rendered figure derives from data (T0.8)."""
    out: dict[str, ReportSeatInsights] = {}
    for c in claims:
        rid = c.get("report_id") or ""
        prov = c.get("provenance") or {}
        ins = out.setdefault(rid, ReportSeatInsights(report_id=rid))
        ins.n_claims += 1

        by_role = prov.get("panel_by_role") or {}
        for role, labels in by_role.items():
            label = _norm(_first(labels))
            if not label:
                continue
            seat = ins.seats.setdefault(role, SeatStats(role=role))
            seat.label_counts[label] = seat.label_counts.get(label, 0) + 1

        if prov.get("panel_escalated"):
            ins.escalated += 1
            arb = _norm(_first(by_role.get("arbiter")))
            prop = _norm(_first(by_role.get("proposer")))
            crit = _norm(_first(by_role.get("critic")))
            if arb and arb == prop:
                side = "proposer"
            elif arb and arb == crit:
                side = "critic"
            else:
                side = "neither"
            ins.arbiter_sided[side] = ins.arbiter_sided.get(side, 0) + 1

        stage1 = _norm(prov.get("crm114_stage1"))
        final = _norm(prov.get("crm114_final"))
        if final and stage1 and final != stage1:
            key = f"{stage1}→{final}"
            ins.overrides[key] = ins.overrides.get(key, 0) + 1
    return out
