#!/usr/bin/env python3
"""Localize FALSE→MISLEADING severity-softening in the PCA pipeline (P67.2 Phase 1a).

Offline — reads only committed artifacts; no LLM calls, no spend.

Inputs:
  * eval/benchmarks/claim-set/verdict_gold.train.jsonl      — gold labels + claim text
  * eval/benchmarks/examples/layerb-vs-gold-verdicts*.json  — eval-run rows (one file
    per config; rows carry votes/split/escalated + optional crm114 override)
  * metrics/pca_runs/*.json                                 — full-publish rows; the
    gold subset is joined by NORMALIZED CLAIM TEXT, not sid (F6: gold sids can drift
    vs a full run's extraction — biden_2022:0342 is absent from the backfill artifact)

Every gold-decidable row whose outcome is MILDER than gold (severity FALSE >
MISLEADING > TRUE; abstentions handled separately) is assigned to exactly one bucket
naming the pipeline stage that produced the softening:

  UNANIMOUS_SOFT   non-escalated, single-label vote tally — proposer+critic agreed on
                   the milder label, the arbiter never ran; only CRM-114 can touch it.
  ARBITER_SOFT     escalated 2-1 — under pca.yaml's label_mismatch criterion with a
                   1-critic roster the plurality winner is PROVABLY the arbiter's own
                   label (F2), so a milder 2-1 outcome is an arbiter hedge.
  TIE_ABSTAIN      DISAGREEMENT_FLAGGED with a FALSE vote present — pca.reduce found
                   no plurality and adjudicator's stage 2 skips non-resolved rows
                   (F1), so a seat's correct FALSE vote dies in the tie.
  CRM114_SOFT      the panel produced gold-or-harsher but the CRM-114 override
                   flipped it milder (discriminator-inflicted).

Non-milder context buckets (reported, not softening):
  ARBITER_RESCUE   escalated 2-1 landing exactly on gold — the arbiter fixed a seat.
  CRM114_RESCUE    CRM-114 override landing exactly on gold.
  ABSTAIN          UNVERIFIABLE / non-resolved without a FALSE vote (coverage, not
                   severity, problem).

The escalation-criterion assertion guards the F2 theorem: if pca.yaml ever moves off
`criterion: label_mismatch` (e.g. to a confidence rule), 2-1 tallies stop identifying
the arbiter and ARBITER_SOFT/RESCUE become unsound — fail loudly rather than misbucket.

Usage: python3 eval/benchmarks/diagnose_softening.py [--json out.json]
(stdlib only — runs on system python3; .venv not required)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]
GOLD = HERE / "claim-set" / "verdict_gold.train.jsonl"
EXAMPLES = HERE / "examples"
PCA_RUNS = REPO / "metrics" / "pca_runs"
PCA_SPEC = REPO / "hydramind" / "specs" / "pca.yaml"

SEVERITY = {"FALSE": 2, "MISLEADING": 1, "TRUE": 0}   # UNVERIFIABLE = abstention
SOFT_BUCKETS = ("UNANIMOUS_SOFT", "ARBITER_SOFT", "TIE_ABSTAIN", "CRM114_SOFT")
CTX_BUCKETS = ("ARBITER_RESCUE", "CRM114_RESCUE", "ABSTAIN", "HARSHER", "CORRECT")


def _assert_label_mismatch_criterion() -> None:
    spec = PCA_SPEC.read_text()
    m = re.search(r"^\s*criterion:\s*(\S+)", spec, re.MULTILINE)
    assert m and m.group(1) == "label_mismatch", (
        f"pca.yaml escalation criterion is {m.group(1) if m else 'MISSING'!r}, not "
        "label_mismatch — the 2-1-tally-identifies-the-arbiter theorem (F2) no longer "
        "holds; re-derive the ARBITER_* buckets before trusting this diagnostic.")


def _norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()


def classify(row: dict, gold_label: str) -> str:
    """One bucket per gold-decidable row (gold ∈ SEVERITY)."""
    status = row.get("status")
    votes = row.get("votes") or {}
    if status != "resolved":
        return "TIE_ABSTAIN" if "FALSE" in votes else "ABSTAIN"
    final = row["verdict"]
    if final == "UNVERIFIABLE":
        return "ABSTAIN"
    crm = row.get("crm114")   # {"stage1": ..., "final": ...} when the override fired
    panel = crm["stage1"] if crm else final
    gsev, fsev = SEVERITY[gold_label], SEVERITY[final]
    if fsev == gsev:
        if crm and SEVERITY.get(panel, gsev) != gsev:
            return "CRM114_RESCUE"
        if row.get("escalated") and sorted(votes.values(), reverse=True)[:2] == [2, 1]:
            return "ARBITER_RESCUE"
        return "CORRECT"
    if fsev > gsev:
        return "HARSHER"
    # milder than gold — name the stage
    if crm and SEVERITY.get(panel, -1) >= gsev:
        return "CRM114_SOFT"                       # panel had it; the override softened
    if not row.get("escalated") and len(votes) == 1:
        return "UNANIMOUS_SOFT"
    if row.get("escalated") and sorted(votes.values(), reverse=True)[:2] == [2, 1]:
        return "ARBITER_SOFT"                      # winner == arbiter's label (F2)
    return "UNANIMOUS_SOFT"                        # e.g. escalated→{label:3}: all seats soft


def load_gold() -> dict[str, dict]:
    rows = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    return {r["sid"]: r for r in rows}


def eval_configs() -> dict[str, dict[str, dict]]:
    """config name → {sid: row} from every committed layerb-vs-gold artifact."""
    out = {}
    for p in sorted(EXAMPLES.glob("layerb-vs-gold-verdicts*.json")):
        name = p.stem.replace("layerb-vs-gold-verdicts", "").lstrip("-") or "closedbook"
        out[name] = {r["sid"]: r for r in json.loads(p.read_text())}
    return out


def fullrun_configs(gold: dict[str, dict]) -> tuple[dict[str, dict[str, dict]], dict]:
    """Full-publish runs joined to gold BY TEXT (F6). Returns (configs, join_report)."""
    by_text = {_norm_text(g.get("claim") or g.get("text", "")): sid
               for sid, g in gold.items()}
    configs, report = {}, {}
    for p in sorted(PCA_RUNS.glob("*.json")):
        d = json.loads(p.read_text())
        speech = d["meta"]["speech_id"]
        name = f"fullrun-{speech}"
        rows_by_sid = {r["sid"]: r for r in d["rows"]}
        matched, joined = [], {}
        for c in d["claims"]:
            gsid = by_text.get(_norm_text(c["text"]))
            if gsid is not None:
                joined[gsid] = rows_by_sid[c["sid"]]   # keyed by GOLD sid
                matched.append((gsid, c["sid"]))
        configs[name] = joined
        expected = [s for s in gold if s.startswith(speech)]
        report[name] = {"gold_rows_for_speech": len(expected),
                        "text_matched": len(matched),
                        "unmatched_gold_sids": sorted(set(expected) - {g for g, _ in matched}),
                        "sid_drift": sorted(f"{g}→{r}" for g, r in matched if g != r)}
    return configs, report


def main() -> None:
    _assert_label_mismatch_criterion()
    gold = load_gold()
    decidable = {s: g["gold_verdict"] for s, g in gold.items()
                 if g["gold_verdict"] in SEVERITY}
    gold_false = sorted(s for s, v in decidable.items() if v == "FALSE")

    configs = eval_configs()
    fulls, join_report = fullrun_configs(gold)
    configs.update(fulls)

    print("# diagnose_softening — pca.yaml criterion=label_mismatch asserted (F2 holds)")
    print(f"# gold: {len(gold)} rows, {len(decidable)} decidable, "
          f"FALSE={gold_false}\n")
    for name, rep in join_report.items():
        print(f"# {name}: text-join matched {rep['text_matched']}/"
              f"{rep['gold_rows_for_speech']} gold rows"
              + (f"; UNMATCHED {rep['unmatched_gold_sids']}" if rep["unmatched_gold_sids"] else "")
              + (f"; sid drift {rep['sid_drift']}" if rep["sid_drift"] else ""))
    print()

    table: dict[str, Counter] = {}
    assignments: dict[str, dict[str, str]] = {}      # config → gold sid → bucket
    details: dict[str, dict[str, dict]] = {}
    for name, rows in configs.items():
        buckets = Counter()
        assignments[name], details[name] = {}, {}
        for sid, glabel in decidable.items():
            row = rows.get(sid)
            if row is None:
                continue
            b = classify(row, glabel)
            buckets[b] += 1
            if glabel in ("FALSE", "MISLEADING") and b in SOFT_BUCKETS or glabel == "FALSE":
                assignments[name][sid] = b
                details[name][sid] = {
                    "gold": glabel, "bucket": b, "status": row.get("status"),
                    "verdict": row.get("verdict"), "votes": row.get("votes"),
                    "escalated": row.get("escalated"), "crm114": row.get("crm114")}
        table[name] = buckets

    cols = list(SOFT_BUCKETS) + list(CTX_BUCKETS)
    w = max(len(c) for c in configs) + 2
    print("## per-config × per-bucket (gold-decidable rows present in each config)")
    print(" " * w + "  ".join(f"{c:>15}" for c in cols))
    for name in configs:
        print(f"{name:<{w}}" + "  ".join(f"{table[name].get(c, 0):>15}" for c in cols))

    print("\n## per-gold-FALSE assignments (the 4-row accuracy ceiling)")
    for sid in gold_false:
        print(f"\n### {sid}  (gold FALSE)  {_norm_text(gold[sid].get('claim',''))[:80]}")
        for name in configs:
            d = details[name].get(sid)
            if d is None:
                print(f"  {name:<{w}} — not in this run")
                continue
            crm = f"  crm114={d['crm114']}" if d["crm114"] else ""
            print(f"  {name:<{w}} {d['bucket']:<15} status={d['status']} "
                  f"verdict={d['verdict']} votes={d['votes']} esc={d['escalated']}{crm}")

    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps({"table": {k: dict(v) for k, v in table.items()},
                                   "gold_false": gold_false,
                                   "assignments": assignments,
                                   "details": details,
                                   "join_report": join_report}, indent=2))
        print(f"\n# json → {out}")


if __name__ == "__main__":
    main()
