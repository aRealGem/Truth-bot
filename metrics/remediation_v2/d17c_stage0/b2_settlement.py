"""D17-c Stage 0 item 8 (R3) — the B2 settlement artifact.

``b2_subset.json`` is a truthful PRE-RUN ESTIMATE and stays exactly as it is.
It was written on 2026-08-08, before the ``haiku-score-2026-08-09``
calibration existed, and it says $0.2299. B2 actually cost $0.5405. Both
numbers are correct about different things, and the D17-c record needs to
cite both: the subset for the design, this artifact for the measurement.

Everything here is MEASURED from the five ``rescored_b2_*.json`` run
artifacts -- their recorded ``spend_usd`` is the settle-ledger figure at
source -- and cross-checked against B2_FINDINGS' independently published
17-of-227 result. The one input that cannot be measured from those artifacts
is ``prompt_chars``: they store replies, not prompts. It is taken from the
superseded worktree estimate (S-9, retained, author unknown) and labelled as
such rather than silently adopted.

$0, offline, no model calls.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from truthbot.costs import (CALIBRATION_ID, CHARS_PER_TOKEN,  # noqa: E402
                            FREETEXT_CHARS_PER_ITEM, estimate_scoring_cost)

#: From the superseded worktree b2_subset.json (_quarantine/pre-d17c-2026-08-12).
#: NOT measurable from the reply artifacts; see module docstring.
PROMPT_CHARS = 412_532
PROMPT_CHARS_SOURCE = ("superseded worktree b2_subset.json, retained under S-9 at "
                       "_quarantine/pre-d17c-2026-08-12/b2_subset.json.worktree; "
                       "the CITED b2_subset.json carries no prompt_chars field")

#: B2_FINDINGS.md section 4 -- published independently of this derivation.
FINDINGS_FLIPS, FINDINGS_TRIGGER = 17, 227
FINDINGS_TOTAL_SPEND = 0.5404          # section 5 table, per-speech rounded
CITED_ESTIMATE = 0.2299                # b2_subset.json as restored

per_speech, items, freetext, calls = {}, 0, 0, 0
for path in sorted(glob.glob(str(HERE.parents[0] / "rescored_b2_*.json"))):
    with open(path) as fh:
        doc = json.load(fh)
    sids = doc["sids"]
    n = sum(len(v) for v in sids.values())
    f = sum(len(i.get("one_line_why") or "") for v in sids.values() for i in v)
    per_speech[doc["speech_id"]] = {
        "spend_usd": doc["spend_usd"], "calls": len(sids), "items": n,
        "freetext_chars": f, "source_run": doc.get("source_run"),
    }
    items += n
    freetext += f
    calls += len(sids)

measured = sum(v["spend_usd"] for v in per_speech.values())
rederived = estimate_scoring_cost(prompt_chars=PROMPT_CHARS, items=items,
                                  freetext_chars=freetext)
gap = abs(rederived["cost_usd_est"] - measured) / measured
factor = measured / CITED_ESTIMATE

print("=== B2 SETTLEMENT ===")
for sid, v in per_speech.items():
    print(f"  {sid:<14} ${v['spend_usd']:.6f}  calls={v['calls']:>3} "
          f"items={v['items']:>4} freetext={v['freetext_chars']:>7,}")
print(f"  {'MEASURED':<14} ${measured:.4f}  calls={calls} items={items} "
      f"freetext={freetext:,}")

print("\n=== cross-checks ===")
checks = {
    "measured total vs B2_FINDINGS table":
        abs(measured - FINDINGS_TOTAL_SPEND) < 0.0005,
    "items == b2_subset estimate items (1028)": items == 1028,
    "calls == b2_subset estimate calls (115)": calls == 115,
    # IDENTITY BY CONSTRUCTION, not independent corroboration: costs.py defines
    # FREETEXT_CHARS_PER_ITEM as the mean one_line_why length over these exact
    # 1,028 B2 replies, so recomputing it from the same population cannot fail.
    # Retained only because it would catch a corpus swap under the constant.
    "measured freetext/item == calibration constant (identity by construction)":
        abs(freetext / items - FREETEXT_CHARS_PER_ITEM) < 0.05,
}
for label, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")

print(f"\n=== re-derivation under {CALIBRATION_ID} ===")
print(f"  prompt_chars {PROMPT_CHARS:,} (tok_in {rederived['tokens_in_est']:,})")
print(f"  re-derived ${rederived['cost_usd_est']:.4f}   measured ${measured:.4f}"
      f"   gap {gap * 100:.2f}%")
within = gap < 0.01
print(f"  within rounding of 0.5389/0.5404? {within}")

print(f"\n=== estimate-vs-actual factor ===")
print(f"  measured {measured:.4f} / cited pre-run estimate {CITED_ESTIMATE} "
      f"= {factor:.2f}x")

# SCOPE CORRECTION (Fable's D17-c ruling). The figures previously stressed here
# -- $0.2063 measured / $0.2992 modelled against the $0.75 whole-programme bound
# -- were the 84-item wave-1 projection. Stage A-FRED is a whole-pack rescore of
# 7 claims / 67 pack items against a $0.15 ceiling; select_rows.py computes it
# under the frequency-aware windows. Stressing the old number against the new
# ceiling would compare two different runs.
STAGE_A_PROJECTION = 0.0511
STAGE_A_CEILING = 0.15
STAGE_A_ACTUAL = 0.053984          # ledger truth, treatment arm
STAGE_A_CONTROL_ACTUAL = 0.026926  # ledger truth, control arm

# 2.351x IS RETIRED FOR MEASURED-BYTE PROJECTIONS (Fable, post-Stage-A).
# It was derived from b2_subset.json, a PRE-RUN ESTIMATE that under-counted
# free-text volume — so it prices the error in an estimate, not the error in a
# measurement. Stage A projected from measured excerpt bytes and realized
# 1.056x. Estimate-based projections KEEP 2.351x; measured-byte projections use
# 1.25x until three realized factors are in hand (this is the first).
MEASURED_BYTE_FACTOR = 1.25
MEASURED_BYTE_REALIZED = [STAGE_A_ACTUAL / STAGE_A_PROJECTION]

print(f"  Stage A-FRED projection ${STAGE_A_PROJECTION:.4f} "
      f"(7 claims / 67 pack items, frequency-aware excerpts)")
print(f"  ACTUAL (ledger) treatment ${STAGE_A_ACTUAL:.6f} + control "
      f"${STAGE_A_CONTROL_ACTUAL:.6f} = "
      f"${STAGE_A_ACTUAL + STAGE_A_CONTROL_ACTUAL:.6f}")
print(f"  realized factor on the treatment arm: "
      f"{MEASURED_BYTE_REALIZED[0]:.3f}x")
print(f"  vs B2's estimate-derived {factor:.3f}x -> RETIRED for measured-byte "
      f"projections")
print(f"  planning factor for measured-byte work: {MEASURED_BYTE_FACTOR:.2f}x "
      f"({len(MEASURED_BYTE_REALIZED)} of 3 realized factors banked)")
print(f"  ceiling ${STAGE_A_CEILING:.2f} -> cumulative actual is "
      f"{(STAGE_A_ACTUAL + STAGE_A_CONTROL_ACTUAL) / STAGE_A_CEILING * 100:.0f}"
      f"% of it")

out = HERE / "b2_settlement.json"
out.write_text(json.dumps({
    "schema": "truthbot-b2-settlement v1",
    "purpose": ("the MEASURED counterpart to b2_subset.json's pre-run estimate; "
                "D17-c Evidence cites the subset for design and this for cost"),
    "calibration_id": CALIBRATION_ID,
    "chars_per_token": CHARS_PER_TOKEN,
    "measured_cost_usd": round(measured, 6),
    "measured_source": "spend_usd recorded in the five rescored_b2_*.json run artifacts",
    "per_speech": per_speech,
    "calls": calls,
    "items": items,
    "freetext_chars_measured": freetext,
    "freetext_chars_per_item_measured": round(freetext / items, 2),
    "prompt_chars": PROMPT_CHARS,
    "prompt_chars_source": PROMPT_CHARS_SOURCE,
    "rederived_estimate": rederived,
    "rederivation_gap_pct": round(gap * 100, 3),
    "within_rounding": within,
    "cited_pre_run_estimate_usd": CITED_ESTIMATE,
    "estimate_vs_actual_factor": round(factor, 3),
    "b2_findings_crosscheck": {"flips": FINDINGS_FLIPS, "trigger_items": FINDINGS_TRIGGER,
                               "published_total_spend_usd": FINDINGS_TOTAL_SPEND},
    "crosschecks": {k: bool(v) for k, v in checks.items()},
}, indent=2, sort_keys=True) + "\n")
print(f"\nsettlement -> {out.name}")

if not within or not all(checks.values()):
    raise SystemExit("HALT: settlement did not reconcile")
