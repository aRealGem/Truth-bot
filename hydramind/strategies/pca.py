"""
`pca` strategy — proposer → critic → (gate) → arbiter (design §3.3).

- wave1: per item, proposer + critic in parallel.
- gate: material_disagreement (label mismatch OR |Δconf| ≥ gate_threshold).
- wave2: rotated arbiter on the gated subset only.
- reduce: plurality vote; a material tie with no plurality ⇒ disagreement_flagged
  (I2 — never a silent tie-break). Winning verdict's citations are checked
  against the evidence pack (I4).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from ..types import (
    Call, Wave, TaskBundle, RunState, Spec, PromptRef, Cap,
    StrategyResult, ItemResult, StrategyResultKind, CallResult,
)
from .base import resolve_binding
from .. import invariants as inv

_PROPOSER_TMPL = "{input}"   # real templates injected via spec.raw["prompts"]


def _label(cr: CallResult) -> Optional[str]:
    return cr.output.get("verdict") if cr else None


def _conf(cr: CallResult) -> Optional[float]:
    v = cr.output.get("confidence") if cr else None
    return float(v) if v is not None else None


class PcaStrategy:
    name = "pca"
    caps = frozenset({Cap.BATCH, Cap.MULTI_ROUND})

    def _prompt(self, spec: Spec, role: str) -> PromptRef:
        tmpl = (spec.raw.get("prompts", {}) or {}).get(role, _PROPOSER_TMPL)
        return PromptRef.of(f"{spec.name}:{role}", tmpl)

    def first(self, task: TaskBundle, spec: Spec) -> Wave:
        calls = []
        for item in task.items:
            for role in spec.flow["wave1"]:
                calls.append(Call(
                    role=role,
                    item_id=item.item_id,
                    prompt=self._prompt(spec, role),
                    binding=resolve_binding(spec.roles[role]),
                    inputs=item.payload,
                ))
        return Wave(calls=calls, batchable=True, tag="wave1")

    def next(self, st: RunState) -> Optional[Wave]:
        if st.scratch.get("phase") == "wave2":
            return None  # arbiter wave already absorbed
        # We've absorbed wave1: compute the gate.
        threshold = st.spec.gate_threshold
        gate_mode = st.spec.flow.get("gate", "material_disagreement")
        by_item = st.by_item()
        gated: list[str] = []
        for item_id, results in by_item.items():
            if gate_mode == "always":               # forced arbitration (audit runs)
                gated.append(item_id)
                continue
            prop = next((r for r in results if r.call.role == "proposer"), None)
            crit = next((r for r in results if r.call.role == "critic"), None)
            if inv.is_material_disagreement(
                _label(prop), _label(crit), _conf(prop), _conf(crit), threshold
            ):
                gated.append(item_id)
        st.scratch["gated"] = gated
        st.scratch["phase"] = "wave2"
        if not gated:
            return None
        arb_spec = st.spec.roles["arbiter"]
        payloads = {i.item_id: i.payload for i in st.task.items}
        calls = [
            Call(
                role="arbiter",
                item_id=item_id,
                prompt=self._prompt(st.spec, "arbiter"),
                binding=resolve_binding(arb_spec, rotation_index=idx),
                inputs=payloads[item_id],
            )
            for idx, item_id in enumerate(gated)
        ]
        return Wave(calls=calls, batchable=True, tag="wave2")

    def reduce(self, st: RunState) -> StrategyResult:
        payloads = {i.item_id: i.payload for i in st.task.items}
        out_items = []
        for item_id, results in st.by_item().items():
            votes = [(_label(r), _conf(r), r) for r in results if _label(r) is not None]
            labels = [v[0] for v in votes]
            counts = Counter(labels)

            if not counts:
                out_items.append(ItemResult(
                    item_id, StrategyResultKind.DISAGREEMENT_FLAGGED,
                    {"reason": "no_labels"}, {"votes": {}}))
                continue

            top = counts.most_common()
            winner, wcount = top[0]
            tie = len(top) > 1 and top[1][1] == wcount

            agreement = {"votes": dict(counts), "n": len(labels)}

            if tie:
                # I2: material tie, no plurality ⇒ flag, never silent break.
                out_items.append(ItemResult(
                    item_id, StrategyResultKind.DISAGREEMENT_FLAGGED,
                    {"labels": dict(counts)}, agreement))
                continue

            # Winning verdict: prefer arbiter's, else the winning-label voter.
            winning_cr = next(
                (v[2] for v in votes if v[2].call.role == "arbiter" and v[0] == winner),
                next(v[2] for v in votes if v[0] == winner),
            )
            citations = list(winning_cr.output.get("citations", []))
            pack_ids = payloads.get(item_id, {}).get("evidence_pack_ids", [])
            # I4: hard fail if a citation is outside the evidence pack.
            inv.check_i4_citations(citations, pack_ids)

            confs = [v[1] for v in votes if v[0] == winner and v[1] is not None]
            out_items.append(ItemResult(
                item_id, StrategyResultKind.RESOLVED,
                {"verdict": winner,
                 "citations": citations,
                 "confidence": (sum(confs) / len(confs)) if confs else None},
                agreement))
        return StrategyResult(items=out_items)
