#!/usr/bin/env python3
"""Re-consolidate journaled packs under the S5 saturation cap — $0, offline.

PR-A2.2 / T2.2 (Spend Gate A measurement): loads a Phase R packs journal
(``{"sid", "gate_code", "evidence": [...], "pool": [...]?}`` per line),
rebuilds each claim's retriever shortlists from the stored items, re-runs the
deterministic consolidator (which now applies ``MAX_S5``), and reports
per-claim gate flips plus S5-saturation aggregates.

HONESTY NOTE (why the headline number can be structurally zero): journals
written before PR-A2.2 store only the POST-cap pack — there is no ``pool``
field, so re-consolidation can drop S5 items but has no discarded T1–3
candidates to backfill with, and since S5 never credits the quota the
gate-flip count from capped data alone is provably 0. The script still
reports the saturation stats and counterfactual freed slots (what the cap
WOULD have made room for), and uses ``pool`` when present (all journals
written after PR-A2.2). Deciding a gate-flipped claim additionally needs a
panel re-run, which is LLM spend — this script never spends.

Usage (repo root):
  PYTHONPATH=.:src .venv/bin/python scripts/reconsolidate_s5_cap.py \
      metrics/journals/obama_2014_packs.jsonl \
      --speech-id obama_2014 --utterance 2014-01-28 [--era-mode both]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import speech_context
from truthbot.verdict.consolidator import GATE_INSUFFICIENT, MAX_S5, consolidate
from truthbot.verdict.evidence_pack import window_for


def _shortlists(evs: list[Evidence]) -> list[tuple[str, list[Evidence]]]:
    """Regroup stored items by retriever (source_name), preserving stored
    order — the journal order IS the original consolidated order, so the
    round-robin re-merge reproduces the original ranking modulo the new
    quota rules."""
    by_src: dict[str, list[Evidence]] = {}
    for ev in evs:
        by_src.setdefault(ev.source_name or "R?", []).append(ev)
    return list(by_src.items())


def run(journal: Path, speech_id: str, utterance: date, era_mode: str) -> dict:
    speech_context.register_speech_date(speech_id, utterance)
    n = n_gated = n_flip = n_pool = 0
    s5_before: Counter = Counter()
    dropped_s5_total = 0
    freed_slots_total = 0
    flips: list[str] = []
    print(f"── era_mode={era_mode} "
          f"(MAX_S5={MAX_S5}) ──────────────────────────────────")
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sid = rec["sid"]
        old_gate = rec.get("gate_code") or ""
        source = rec.get("pool") or rec.get("evidence") or []
        has_pool = bool(rec.get("pool"))
        evs = [Evidence.model_validate(d) for d in source]
        n += 1
        n_pool += has_pool
        n_s5 = sum(1 for ev in evs if ev.source_tier == SourceTier.POLITICAL)
        s5_before[n_s5] += 1
        res = consolidate(sid, _shortlists(evs), utterance=utterance,
                          window=window_for(sid), era_mode=era_mode)
        new_gate = res.gate_code
        dropped_s5 = res.dropped.get("s5-quota", 0)
        dropped_s5_total += dropped_s5
        if old_gate == GATE_INSUFFICIENT:
            n_gated += 1
            # Counterfactual: slots the cap frees for a future re-retrieval
            # to fill with quota-crediting items.
            freed_slots_total += dropped_s5
            if new_gate != GATE_INSUFFICIENT:
                n_flip += 1
                flips.append(sid)
                print(f"  FLIP {sid}: gate cleared "
                      f"(s5 dropped={dropped_s5}, items={len(res.items)})")
    print(f"claims={n} (pre-cap pool stored for {n_pool}) | "
          f"originally gated={n_gated} | gate flips={n_flip}")
    print(f"S5-per-pack distribution: "
          f"{dict(sorted(s5_before.items()))}")
    print(f"S5 items dropped by the cap: {dropped_s5_total} | "
          f"slots freed on gated claims: {freed_slots_total}")
    if not n_pool:
        print("NOTE: no pre-cap pools in this journal (pre-A2.2 run) — flips "
              "from capped data alone are structurally 0; freed slots show "
              "what a re-retrieval could now use.")
    return {"era_mode": era_mode, "claims": n, "gated": n_gated,
            "flips": n_flip, "flip_sids": flips,
            "s5_dropped": dropped_s5_total, "freed_slots": freed_slots_total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("journal", type=Path)
    ap.add_argument("--speech-id", required=True)
    ap.add_argument("--utterance", required=True,
                    help="speech date YYYY-MM-DD (era mode is not journaled)")
    ap.add_argument("--era-mode", choices=("strict", "lenient", "both"),
                    default="both")
    args = ap.parse_args()
    utt = date.fromisoformat(args.utterance)
    modes = ("strict", "lenient") if args.era_mode == "both" else (args.era_mode,)
    results = [run(args.journal, args.speech_id, utt, m) for m in modes]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
