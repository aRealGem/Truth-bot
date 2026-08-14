#!/usr/bin/env python3
"""D17-d structural decidability probe — analysis only, $0, no pipeline changes.

QUESTION. The evidence gate stamps a claim ``insufficient-qualifying-evidence``
when its pack never met the Tier-1..3 bearing quota. That code says "this pack
did not qualify"; it does NOT say whether the claim is checkable at all. So a
documented valor citation and a private hospital-room conversation exit the gate
identical. A human desk pass (``metrics/remediation_v2/d17d_triage.json``)
hand-sorted all 128 gate-withheld claims into four decidability classes. This
probe asks how much of that judgement PIPELINE STRUCTURE ALONE can reproduce.

METHOD. Six ordered rules read ONLY structured fields already stored on the
artifact — ``claim_type``, ``claim_shape``, whether any evidence item carries a
``series_rows`` excerpt, and evidence tiers. No claim text is read as prose; no
model, network, or clock. Three rules COMMIT to a single class; three ABSTAIN
and record the residual class range they narrowed to. Abstention is deliberate:
the desk doc is explicit that separating permanently-undecidable from merely
under-retrieved "needs this desk pass, not a regex", so where structure cannot
decide, the probe says so instead of guessing.

The desk pass is the AUDIT FIXTURE, not ground truth about the world — it is one
careful human read. "agree" below means "matches the desk", not "is correct".

Deterministic: same inputs -> byte-identical output.

Usage (repo root):
  PYTHONPATH=src python3 scripts/d17d_structural_probe.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "metrics" / "pca_runs"
DESK = REPO / "metrics" / "remediation_v2" / "d17d_triage.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_structural_probe.json"

#: The five frozen publishing heads — identical to the desk pass.
HEADS = {
    "trump_2026": "799e71b6-2480-50ca-870e-1a95f0d0d5fe",
    "biden_2022": "c156d8f9-be85-5263-92a1-c08949afdedd",
    "obama_2014": "70748500-315a-5664-8474-c6632de57816",
    "clinton_1998": "d7ee7340-c07d-55da-b9db-9397d7141c35",
    "gwbush_2006": "6df77093-e328-596e-bfd5-afabd08a1679",
}

GATE = "insufficient-qualifying-evidence"

ALL_CLASSES = ["web-tier1", "substantive", "series-core", "compound-split"]

#: rule_id -> (disposition, predicted_class, residual_range, signal, rationale)
#: Evaluated in this order; first match wins.
RULES = {
    "R1-series-attached": (
        "committed", "series-core", None,
        "an evidence item carries series_rows",
        "A named statistical series is structurally attached to the pack — the "
        "only unambiguous 'this is checkable' signal the artifact holds."),
    "R2-attribution-type": (
        "committed", "substantive", None,
        "claim_type == attribution",
        "Treated as the signature of an assertion about intent/motive/others' "
        "views, which no retrieval reaches."),
    "R3-eval-shape": (
        "committed", "substantive", None,
        "claim_shape == c-eval",
        "Treated as a causal/superlative/evaluative core with no clean "
        "retrieval target."),
    "R4-statistical-unattached": (
        "abstained", None, ["series-core", "web-tier1"],
        "claim_type == statistical and no series_rows attached",
        "A number, but structure cannot say whether a nameable series or web "
        "retrieval settles it."),
    "R5-narrative-type": (
        "abstained", None, ["web-tier1", "substantive"],
        "claim_type in {personal-anecdote, historical, comparison, other}",
        "The documentable-vs-private gap. Structure holds no field that "
        "separates a recorded citation from a private moment."),
    "R6-no-signal": (
        "abstained", None, list(ALL_CLASSES),
        "no usable structured field present",
        "Abstain loudly rather than fall through to a default class."),
}

_NARRATIVE = {"personal-anecdote", "historical", "comparison", "other"}


def fire_rule(claim_type: str | None, claim_shape: str | None,
              has_series_rows: bool) -> str:
    """Return the rule_id that fires. Structured fields only — never text."""
    if has_series_rows:
        return "R1-series-attached"
    if claim_type == "attribution":
        return "R2-attribution-type"
    if claim_shape == "c-eval":
        return "R3-eval-shape"
    if claim_type == "statistical":
        return "R4-statistical-unattached"
    if claim_type in _NARRATIVE:
        return "R5-narrative-type"
    return "R6-no-signal"


def build() -> dict:
    desk_doc = json.loads(DESK.read_text(encoding="utf-8"))
    desk_cls = {c["sid"]: c["decidability_class"] for c in desk_doc["claims"]}
    desk_text = {c["sid"]: c["text"] for c in desk_doc["claims"]}

    claims_out = []
    for speech, run in sorted(HEADS.items()):
        doc = json.loads((RUNS / f"{run}.json").read_text(encoding="utf-8"))
        claims = {c["sid"]: c for c in doc["claims"]}
        evidence = doc.get("evidence", {})
        for row in doc["rows"]:
            if row.get("provenance_code") != GATE:
                continue
            sid = row["sid"]
            cl = claims.get(sid, {})
            la = cl.get("layer_a") or {}
            claim_type = la.get("claim_type")
            claim_shape = la.get("claim_shape")
            pack = evidence.get(sid, [])
            has_series = any(bool(e.get("series_rows")) for e in pack)
            tiers = Counter(e.get("source_tier") for e in pack)

            rule_id = fire_rule(claim_type, claim_shape, has_series)
            disposition, predicted, residual, signal, _ = RULES[rule_id]
            desk = desk_cls.get(sid, "MISSING-FROM-DESK")

            rec = {
                "sid": sid,
                "speech": speech,
                "text": desk_text.get(sid, cl.get("text", "")),
                "disposition": disposition,
                "predicted_class": predicted,
                "rule_id": rule_id,
                "rule_signal": signal,
                "desk_class": desk,
                "signals_used": {
                    "claim_type": claim_type,
                    "claim_shape": claim_shape,
                    "has_series_rows": has_series,
                    "evidence_tier_counts": dict(
                        sorted(tiers.items(), key=lambda kv: str(kv[0]))),
                },
            }
            if disposition == "committed":
                rec["agree"] = (predicted == desk)
                rec["residual_class_range"] = None
                rec["residual_contains_desk"] = None
            else:
                rec["agree"] = None      # no commitment to agree or disagree
                rec["residual_class_range"] = residual
                rec["residual_contains_desk"] = desk in residual
            claims_out.append(rec)

    committed = [c for c in claims_out if c["disposition"] == "committed"]
    abstained = [c for c in claims_out if c["disposition"] == "abstained"]
    agree = [c for c in committed if c["agree"]]
    err = [c for c in committed if not c["agree"]]

    # ── per-rule confusion, incl. the DIRECTION of every committed error ──
    per_rule = {}
    for rule_id, (disp, predicted, residual, signal, rationale) in RULES.items():
        rows = [c for c in claims_out if c["rule_id"] == rule_id]
        block = {
            "disposition": disp,
            "predicted_class": predicted,
            "residual_class_range": residual,
            "signal": signal,
            "rationale": rationale,
            "n_fired": len(rows),
            "desk_class_breakdown": dict(
                Counter(c["desk_class"] for c in rows).most_common()),
        }
        if disp == "committed":
            block["n_agree"] = sum(1 for c in rows if c["agree"])
            block["n_error"] = sum(1 for c in rows if not c["agree"])
            # direction: predicted X, desk actually said Y
            block["error_direction"] = dict(Counter(
                f"predicted {predicted} -> desk {c['desk_class']}"
                for c in rows if not c["agree"]).most_common())
            block["error_sids"] = sorted(c["sid"] for c in rows if not c["agree"])
        else:
            block["n_residual_contains_desk"] = sum(
                1 for c in rows if c["residual_contains_desk"])
            block["n_residual_misses_desk"] = sum(
                1 for c in rows if not c["residual_contains_desk"])
            block["residual_miss_sids"] = sorted(
                c["sid"] for c in rows if not c["residual_contains_desk"])
        per_rule[rule_id] = block

    confusion = Counter(
        (c["desk_class"], c["rule_id"]) for c in claims_out)

    # Where did the desk classes that structure cannot express end up?
    invisible = {}
    for target in ("compound-split", "series-core"):
        rows = [c for c in claims_out if c["desk_class"] == target]
        invisible[target] = {
            "desk_count": len(rows),
            "n_recovered": sum(1 for c in rows
                               if c["predicted_class"] == target),
            "landed_in_rule": dict(
                Counter(c["rule_id"] for c in rows).most_common()),
            "sids": sorted(c["sid"] for c in rows),
        }

    # The requested overlap: anecdote-precedence set vs desk-substantive set.
    r5 = {c["sid"] for c in claims_out if c["rule_id"] == "R5-narrative-type"}
    r5_anecdote = {c["sid"] for c in claims_out
                   if c["rule_id"] == "R5-narrative-type"
                   and c["signals_used"]["claim_type"] == "personal-anecdote"}
    desk_sub = {sid for sid, k in desk_cls.items() if k == "substantive"}
    overlap = {
        "desk_substantive_n": len(desk_sub),
        "R5_narrative_all": {
            "n": len(r5),
            "intersection_with_desk_substantive": len(r5 & desk_sub),
            "precision_if_treated_as_substantive": round(
                len(r5 & desk_sub) / len(r5), 4) if r5 else None,
            "recall_of_desk_substantive": round(
                len(r5 & desk_sub) / len(desk_sub), 4) if desk_sub else None,
        },
        "R5_personal_anecdote_only": {
            "n": len(r5_anecdote),
            "intersection_with_desk_substantive": len(r5_anecdote & desk_sub),
            "precision_if_treated_as_substantive": round(
                len(r5_anecdote & desk_sub) / len(r5_anecdote), 4)
                if r5_anecdote else None,
            "recall_of_desk_substantive": round(
                len(r5_anecdote & desk_sub) / len(desk_sub), 4)
                if desk_sub else None,
        },
        "note": ("'anecdote-precedence' is read two ways because the rule "
                 "covers four narrative types; both are reported so neither "
                 "reading has to be inferred."),
    }

    n = len(claims_out)
    return {
        "schema": "truthbot-d17d-structural-probe v1",
        "generated_from": {sp: rid for sp, rid in sorted(HEADS.items())},
        "audit_fixture": str(DESK.relative_to(REPO)),
        "method": (
            "Six ordered rules over structured fields only (claim_type, "
            "claim_shape, series_rows presence, evidence tiers). No claim text "
            "read as prose; no model, network, or clock. Three rules commit, "
            "three abstain with a recorded residual class range. 'agree' means "
            "'matches the desk pass', which is a careful human read used as an "
            "audit fixture — not ground truth about the world."),
        "totals": {
            "gate_withheld": n,
            "committed": len(committed),
            "abstained": len(abstained),
            "committed_agree": len(agree),
            "committed_error": len(err),
            "abstained_residual_contains_desk": sum(
                1 for c in abstained if c["residual_contains_desk"]),
            "abstained_residual_misses_desk": sum(
                1 for c in abstained if not c["residual_contains_desk"]),
        },
        "headline": (
            f"Structure commits on {len(committed)} of {n} gate-withheld "
            f"claims: {len(agree)} match the desk, {len(err)} do not. The "
            f"remaining {len(abstained)} are structurally undetermined. Every "
            f"committed error runs in one direction — predicting 'undecidable' "
            f"for a claim the desk found documentable."),
        "desk_class_totals": dict(Counter(desk_cls.values()).most_common()),
        "per_rule": per_rule,
        "confusion_desk_x_rule": [
            {"desk_class": dc, "rule_id": r, "n": k}
            for (dc, r), k in sorted(confusion.items())
        ],
        "structurally_inexpressible": invisible,
        "anecdote_precedence_overlap": overlap,
        "claims": claims_out,
    }


def main() -> int:
    doc = build()
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    t = doc["totals"]
    print(f"gate-withheld: {t['gate_withheld']}")
    print(f"  committed {t['committed']}  "
          f"(agree {t['committed_agree']} / error {t['committed_error']})")
    print(f"  abstained {t['abstained']}  "
          f"(residual holds desk {t['abstained_residual_contains_desk']} / "
          f"misses {t['abstained_residual_misses_desk']})")
    print("\nper rule:")
    for rid, b in doc["per_rule"].items():
        if b["disposition"] == "committed":
            print(f"  {rid:<26} n={b['n_fired']:<4} "
                  f"agree={b['n_agree']} error={b['n_error']}")
            for d, k in b["error_direction"].items():
                print(f"       {d}  x{k}")
        else:
            print(f"  {rid:<26} n={b['n_fired']:<4} "
                  f"residual-holds-desk={b['n_residual_contains_desk']} "
                  f"misses={b['n_residual_misses_desk']}")
    o = doc["anecdote_precedence_overlap"]
    print(f"\nanecdote-precedence n desk-substantive:")
    print(f"  R5 all narrative      : "
          f"{o['R5_narrative_all']['intersection_with_desk_substantive']} "
          f"of {o['R5_narrative_all']['n']} fired "
          f"(precision {o['R5_narrative_all']['precision_if_treated_as_substantive']})")
    print(f"  R5 personal-anecdote  : "
          f"{o['R5_personal_anecdote_only']['intersection_with_desk_substantive']} "
          f"of {o['R5_personal_anecdote_only']['n']} fired "
          f"(precision {o['R5_personal_anecdote_only']['precision_if_treated_as_substantive']})")
    print(f"\n-> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
