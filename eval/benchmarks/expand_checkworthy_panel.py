#!/usr/bin/env python3
"""Expand the sonnet+mistral panel from the POC 53 toward ~150 rows.

Deterministically draws a stratified sample of sids NOT already in panel_labels.json,
rebalanced toward the trump speech (the corpus is trump-heavy but the POC gold was
biden-heavy) and spanning all three heuristic bands so the boundary is well covered.
Runs the SAME neutral panel (claude-sonnet + mistral) as checkworthy_gold_panel.py and
APPENDS to panel_labels.json, preserving existing entries verbatim.

No RNG: within each (speech, old_label) cell the sids are sorted and evenly spaced, so
the sample is reproducible and spread across each speech's timeline.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); import proxy_client
from checkworthy_gold_panel import PANEL, label_one  # reuse the exact rubric + caller

# new-sample targets per (speech, heuristic old_label); see module docstring / session notes
TARGETS = {
    ("biden_2022", "check-worthy"): 7,  ("trump_2026", "check-worthy"): 22,
    ("biden_2022", "opinion"):      8,  ("trump_2026", "opinion"):      30,
    ("biden_2022", "unimportant"):  4,  ("trump_2026", "unimportant"):  26,
}


def even_spaced(items, n):
    """Pick n items evenly spaced across the sorted list (deterministic)."""
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def build_sample(rows, have):
    picked = []
    for (speech, lab), n in sorted(TARGETS.items()):
        cell = sorted(sid for sid, r in rows.items()
                      if sid not in have and r["speech"] == speech and r["label"] == lab)
        take = even_spaced(cell, n)
        if len(take) < n:
            print(f"  WARN cell {(speech,lab)} short: wanted {n}, had {len(cell)}")
        picked += take
    return picked


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return
    key = os.environ[proxy_client.resolve_key_env()]; base = proxy_client.base_url()
    rows = {json.loads(l)["sid"]: json.loads(l) for l in
            (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()}
    panel = json.loads((HERE / "panel_labels.json").read_text())
    have = set(panel)
    sample = build_sample(rows, have)
    print(f"existing panel: {len(have)} | new sample: {len(sample)} | "
          f"projected total: {len(have)+len(sample)}")
    for i, sid in enumerate(sample, 1):
        text = rows[sid]["text"]
        votes = {m: label_one(m, key, base, text) for m in PANEL}
        panel[sid] = {"text": text, "old_label": rows[sid]["label"], **votes}
        print(f"  [{i:3}/{len(sample)}] {sid}: {votes}")
    (HERE / "panel_labels.json").write_text(json.dumps(panel, indent=2, ensure_ascii=False))
    print(f"-> panel_labels.json now {len(panel)} rows")


if __name__ == "__main__":
    main()
