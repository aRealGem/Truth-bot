#!/usr/bin/env python3
"""
Verdict scorer for truth-bot v2 Layer B (closed-book).

Scores predicted verdicts against an evidence-backed verdict-gold overlay
(claim-set/verdict_gold.train.jsonl; see verdict_gold.SCHEMA.md), with scoring
semantics tuned for a closed-book ABSTAINING system:

  * A model UNVERIFIABLE — or an unresolved / disagreement-flagged item — is an
    ABSTENTION, not a wrong answer. Against a decidable gold (TRUE/FALSE/MISLEADING)
    it is a COVERAGE GAP (the kind Layer C evidence is meant to close); against a
    gold of UNVERIFIABLE it is an APPROPRIATE abstention.
  * decided-accuracy = among items the model actually committed on
    (verdict in TRUE/FALSE/MISLEADING), the fraction matching gold — the real
    quality signal for a closed-book panel.

TRAIN only — heldout gold is a separate, guarded pass (I6).

CLI:
  score_verdict.py <preds.json|preds.jsonl> [gold.jsonl]
    preds.json  = the dev-lot artifact (list of VerdictRows: {sid,status,verdict,...})
    preds.jsonl = one {"sid","status","verdict"} per line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "claim-set" / "verdict_gold.train.jsonl"

LABELS = ["TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"]
_COMMITTED = {"TRUE", "FALSE", "MISLEADING"}     # UNVERIFIABLE is an abstention


def _pred_verdict(row: dict):
    """A substantive committed verdict, else None (abstain). Only a RESOLVED item
    with a TRUE/FALSE/MISLEADING verdict counts as committed; UNVERIFIABLE,
    disagreement and no_label all abstain."""
    if row.get("status") == "resolved" and row.get("verdict") in _COMMITTED:
        return row["verdict"]
    return None


def score_verdicts(gold: dict, preds: dict) -> dict:
    """gold: sid -> gold_verdict (one of LABELS). preds: sid -> pred row dict.
    Returns coverage / decided-accuracy / abstention breakdown + a
    gold x {TRUE,FALSE,MISLEADING,ABSTAIN} confusion matrix."""
    cols = ["TRUE", "FALSE", "MISLEADING", "ABSTAIN"]
    conf = {g: {c: 0 for c in cols} for g in LABELS}
    hit = miss = abstain_ok = abstain_gap = n = 0
    rows = []
    for sid, gold_v in gold.items():
        if sid not in preds:
            continue
        n += 1
        pv = _pred_verdict(preds[sid])
        committed = pv is not None
        if gold_v in conf:
            conf[gold_v][pv if committed else "ABSTAIN"] += 1
        if committed:
            if pv == gold_v:
                hit += 1; cat = "hit"
            else:
                miss += 1; cat = "miss"
        elif gold_v == "UNVERIFIABLE":
            abstain_ok += 1; cat = "abstain_ok"
        else:
            abstain_gap += 1; cat = "abstain_gap"
        rows.append({"sid": sid, "gold": gold_v, "pred": pv or "ABSTAIN", "category": cat})
    decided = hit + miss
    return {
        "n": n,
        "decided": decided,
        "coverage": (decided / n) if n else None,
        "decided_accuracy": (hit / decided) if decided else None,
        "hit": hit, "miss": miss,
        "abstain_gap": abstain_gap,      # decidable gold, model abstained (→ Layer C)
        "abstain_ok": abstain_ok,        # gold UNVERIFIABLE, model rightly abstained
        "confusion": conf,
        "rows": rows,
    }


def load_gold(path: Path = GOLD) -> dict:
    out = {}
    for l in path.read_text().splitlines():
        if l.strip():
            o = json.loads(l); out[o["sid"]] = o["gold_verdict"]
    return out


def load_preds(path: Path) -> dict:
    txt = path.read_text()
    try:
        data = json.loads(txt)
        if isinstance(data, list):                     # dev-lot artifact (list of rows)
            return {r["sid"]: r for r in data}
    except json.JSONDecodeError:
        pass
    out = {}
    for l in txt.splitlines():                         # jsonl fallback
        if l.strip():
            o = json.loads(l); out[o["sid"]] = o
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    preds = load_preds(Path(sys.argv[1]))
    gold = load_gold(Path(sys.argv[2]) if len(sys.argv) > 2 else GOLD)
    rep = score_verdicts(gold, preds)
    print(f"# verdict scoring vs gold (n={rep['n']} gold rows with a prediction)")
    print(f"  coverage         = {rep['coverage']}")
    print(f"  decided_accuracy = {rep['decided_accuracy']}  ({rep['hit']}/{rep['decided']})")
    print(f"  abstain_gap      = {rep['abstain_gap']}  (decidable gold, model abstained → Layer C)")
    print(f"  abstain_ok       = {rep['abstain_ok']}  (gold UNVERIFIABLE, model rightly abstained)")
    print("  confusion (gold row x pred col):")
    for g in LABELS:
        print(f"    {g:12} {rep['confusion'][g]}")


if __name__ == "__main__":
    main()
