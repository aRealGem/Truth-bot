"""
`pca` strategy — proposer → critic(s) → (split gate) → arbiter (design §3.3, V4 build 7).

Single-claim state machine P→C→A. Seats are bound either from a named ROSTER
(spec.raw["roster_resolved"], seat→[aliases]) or, absent a roster, from the YAML
provider pools (back-compat). A critic may be a panel (list of aliases).

Split detection (logged): a P/C **material disagreement** = proposer verdict ≠ any
critic verdict, OR |Δconfidence| ≥ gate_threshold for any critic. Split items
escalate to the arbiter (escalation stub; frontier threshold is a placeholder).

reduce: plurality across all seats; a material tie with no plurality ⇒
disagreement_flagged (I2). Winning verdict's citations checked against the pack (I4).
Per-seat model/lane/cost/returned_model land in the run manifest's cost_records.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from ..types import (
    Wave, Call, TaskBundle, RunState, Spec, PromptRef, Cap,
    StrategyResult, ItemResult, StrategyResultKind, CallResult,
)
from .base import resolve_binding
from ..models import binding_from_alias
from .. import invariants as inv

_DEFAULT_TMPL = "{input}"


def _label(cr: CallResult) -> Optional[str]:
    return cr.output.get("verdict") if cr else None


def _conf(cr: CallResult) -> Optional[float]:
    v = cr.output.get("confidence") if cr else None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class PcaStrategy:
    name = "pca"
    caps = frozenset({Cap.BATCH, Cap.MULTI_ROUND})

    def _prompt(self, spec: Spec, role: str) -> PromptRef:
        tmpl = (spec.raw.get("prompts", {}) or {}).get(role, _DEFAULT_TMPL)
        return PromptRef.of(f"{spec.name}:{role}", tmpl)

    def _seat_bindings(self, spec: Spec, role: str, rotation_index: int = 0):
        """Return list of ModelBindings for a seat: from roster if present, else pool.

        rotation_index rotates a round_robin pool role across its providers — the
        arbiter uses this so gated items spread across frontier providers instead of
        every item landing on providers[0]. Ignored for a non-rotating pool (single
        default) and for roster seats (which rotate across the panel by list index)."""
        roster = spec.raw.get("roster_resolved")
        if roster:
            return [binding_from_alias(a) for a in roster.get(role, [])]
        return [resolve_binding(spec.roles[role], rotation_index=rotation_index)]

    def first(self, task: TaskBundle, spec: Spec) -> Wave:
        calls = []
        for item in task.items:
            for role in spec.flow["wave1"]:            # [proposer, critic]
                for b in self._seat_bindings(spec, role):
                    calls.append(Call(role=role, item_id=item.item_id,
                                      prompt=self._prompt(spec, role),
                                      binding=b, inputs=item.payload))
        return Wave(calls=calls, batchable=True, tag="wave1")

    def next(self, st: RunState) -> Optional[Wave]:
        if st.scratch.get("phase") == "wave2":
            return None
        threshold = st.spec.gate_threshold
        gate_mode = st.spec.flow.get("gate", "material_disagreement")
        # NAMED escalation criterion (P96.2.1); default preserves the legacy rule.
        esc_cfg = st.spec.raw.get("escalation", {}) or {}
        criterion = esc_cfg.get("criterion", "material_disagreement")
        st.scratch["split_criterion"] = criterion
        by_item = st.by_item()
        gated, split_items = [], []
        for item_id, results in by_item.items():
            prop = next((r for r in results if r.call.role == "proposer"), None)
            crits = [r for r in results if r.call.role == "critic"]
            split = any(inv.is_escalation_split(
                criterion, _label(prop), _label(c), _conf(prop), _conf(c), threshold)
                for c in crits)
            if split:
                split_items.append(item_id)
            if gate_mode == "always" or split:
                gated.append(item_id)
        st.scratch["gated"] = gated
        st.scratch["split_items"] = split_items
        st.scratch["phase"] = "wave2"

        # split ⇒ escalate to arbiter, per the NAMED criterion above. frontier
        # threshold stays a placeholder for a future stronger-frontier rung.
        monitor = esc_cfg.get("monitor", {}) or {}
        st.scratch["escalation"] = {
            "trigger": esc_cfg.get("trigger", "on_split"),
            "criterion": criterion,
            "frontier_confidence_threshold": esc_cfg.get("frontier_confidence_threshold", None),
            "rate_watermark": monitor.get("rate_watermark", 0.50),
            "escalated_items": list(gated),
        }
        if not gated:
            return None
        payloads = {i.item_id: i.payload for i in st.task.items}
        calls = []
        for idx, item_id in enumerate(gated):
            # round_robin pool → rotate providers per gated item (resolve_binding);
            # roster panel → rotate across the panel aliases by list index.
            arb_bindings = self._seat_bindings(st.spec, "arbiter", rotation_index=idx)
            b = arb_bindings[idx % len(arb_bindings)]
            calls.append(Call(role="arbiter", item_id=item_id,
                              prompt=self._prompt(st.spec, "arbiter"),
                              binding=b, inputs=payloads[item_id]))
        return Wave(calls=calls, batchable=True, tag="wave2")

    def reduce(self, st: RunState) -> StrategyResult:
        payloads = {i.item_id: i.payload for i in st.task.items}
        out_items = []
        for item_id, results in st.by_item().items():
            votes = [(_label(r), _conf(r), r) for r in results if _label(r) is not None]
            counts = Counter(v[0] for v in votes)
            # Per-seat attribution (P67 Phase 3): role → [labels] (a critic may be a
            # panel, hence lists). Kills the 2-1 tally ambiguity — the arbiter's own
            # label is readable from the artifact instead of provable-by-theorem.
            by_role: dict[str, list[str]] = {}
            # Per-seat RATIONALE TEXT, recorded beside the per-seat labels (R-3,
            # 2026-08-10). ``by_role`` says WHAT each seat concluded; this says
            # WHY, verbatim as that seat wrote it. Without it a TIE carries no
            # rationale anywhere on disk — which is how the stage-2 CRM-114
            # discriminator came to publish verdicts with an empty ``reasoning``,
            # and why a models-split could only ever render as "Panel split".
            # Nothing is synthesized here: it is the seat's own text or "".
            seat_rationales: list[dict] = []
            for label, conf, cr in votes:
                by_role.setdefault(cr.call.role, []).append(label)
                seat_rationales.append({
                    "role": str(cr.call.role),
                    "verdict": label,
                    "confidence": conf,
                    "reasoning": str(cr.output.get("reasoning") or "").strip(),
                    "citations": list(cr.output.get("citations") or []),
                })
            if not counts:
                out_items.append(ItemResult(item_id, StrategyResultKind.DISAGREEMENT_FLAGGED,
                                            {"reason": "no_labels"},
                                            {"votes": {}, "by_role": {},
                                             "seat_rationales": []}))
                continue
            top = counts.most_common()
            winner, wcount = top[0]
            tie = len(top) > 1 and top[1][1] == wcount
            agreement = {"votes": dict(counts), "n": len(votes), "by_role": by_role,
                         "seat_rationales": seat_rationales,
                         "split": item_id in st.scratch.get("split_items", []),
                         "escalated": item_id in st.scratch.get("gated", [])}
            if tie:
                out_items.append(ItemResult(item_id, StrategyResultKind.DISAGREEMENT_FLAGGED,
                                            {"labels": dict(counts)}, agreement))
                continue
            winning_cr = next(
                (v[2] for v in votes if v[2].call.role == "arbiter" and v[0] == winner),
                next(v[2] for v in votes if v[0] == winner))
            citations = list(winning_cr.output.get("citations", []))
            inv.check_i4_citations(citations, payloads.get(item_id, {}).get("evidence_pack_ids", []))
            confs = [v[1] for v in votes if v[0] == winner and v[1] is not None]
            out_items.append(ItemResult(item_id, StrategyResultKind.RESOLVED,
                {"verdict": winner, "citations": citations,
                 "confidence": (sum(confs) / len(confs)) if confs else None,
                 "reasoning": winning_cr.output.get("reasoning", "")}, agreement))

        n = len(out_items)
        notes = {
            "split_criterion": st.scratch.get("split_criterion"),
            "split_rate": len(st.scratch.get("split_items", [])) / n if n else 0.0,
            "escalation_rate": len(st.scratch.get("gated", [])) / n if n else 0.0,
            "escalation": st.scratch.get("escalation"),
            "flagged": sum(1 for r in out_items
                           if r.kind == StrategyResultKind.DISAGREEMENT_FLAGGED),
        }
        return StrategyResult(items=out_items, notes=notes)
