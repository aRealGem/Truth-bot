#!/usr/bin/env python3
"""D17-d web-tier1 lane — a $0 estimate, and an honest statement of its limits.

The largest triage class (81 of 128 withheld claims) has NO measured cost
constant, and S-12 forbids borrowing one: a per-claim number measured on a
different payload is what ran the d17c-wave2 escalation 8.2x over. So this
prices what CAN be measured and refuses to invent the rest.

WHAT IS MEASURED (real bytes, from the committed artifacts):
  the existing pack payload per claim — what a panel call costs today, before
  any new retrieval is added.

WHAT IS ASSUMED (stated as assumptions, never as measurements):
  how much a Tier-1..3 retrieval pass would ADD. Nothing is fetched here, so
  the added volume is a projection from the shape of the existing packs. Every
  assumption is named with its basis so a reviewer can reject it individually
  rather than having to reject the whole number.

WHY THAT DISTINCTION IS THE POINT. The 8.2x miss did not come from arithmetic;
it came from a number whose provenance had been forgotten by the time it was
used. This artifact keeps measured and assumed in separate fields so they
cannot merge into "the estimate".

THE PROBE. A three-claim calibration probe is designed here and NOT run: it is
metered and waits for an owner click. Its own cost band is UNMEASURED, which is
the whole reason it exists.

Usage (repo root):
  PYTHONPATH=src .venv/bin/python scripts/build_d17d_webtier1_estimate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "metrics" / "pca_runs"
TRIAGE = REPO / "metrics" / "remediation_v2" / "d17d_triage.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_webtier1_estimate.json"

HEADS = {
    "trump_2026": "799e71b6-2480-50ca-870e-1a95f0d0d5fe",
    "biden_2022": "c156d8f9-be85-5263-92a1-c08949afdedd",
    "obama_2014": "70748500-315a-5664-8474-c6632de57816",
    "clinton_1998": "d7ee7340-c07d-55da-b9db-9397d7141c35",
    "gwbush_2006": "6df77093-e328-596e-bfd5-afabd08a1679",
}

#: The one measured rate we have, from d17c-wave2. It prices PAYLOAD BYTES on a
#: panel call, which is the part of a web-tier1 claim that resembles what was
#: measured. It does NOT price retrieval itself — see UNPRICED_COMPONENTS.
MEASURED_USD_PER_KCHAR = 0.003124

ASSUMPTIONS = [
    {"id": "A1",
     "assumption": "A Tier-1..3 retrieval pass adds 6 new items to a pack.",
     "basis": ("The existing packs cap at 10 items and the gate wants >=2 "
               "bearing Tier-1..3. Six is the midpoint between 'enough to "
               "clear the quota' and 'fills the cap'."),
     "sensitivity": "Linear. Halving it halves the added payload."},
    {"id": "A2",
     "assumption": "A retrieved item's snippet resembles the ones already "
                   "stored — same connector, same truncation.",
     "basis": "Measured: the mean stored snippet length across these packs.",
     "sensitivity": "Low. Snippet length is tightly clustered in the corpus."},
    {"id": "A3",
     "assumption": "No series_rows are attached on this lane.",
     "basis": ("web-tier1 is defined as claims no series settles, so the "
               "D17-c excerpt path does not apply. This is what makes the "
               "lane cheaper per claim than the series lane."),
     "sensitivity": "High if violated — series rows were 91.2% of the "
                    "payload on the claims that carried them."},
]

UNPRICED_COMPONENTS = [
    {"component": "retrieval itself (search API / connector calls)",
     "why": ("Off-proxy and not on the LiteLLM ledger, so no measured "
             "constant covers it. The d17c lanes were all no-retrieval "
             "re-gates, which is exactly why their constants cannot be "
             "borrowed here.")},
    {"component": "re-retrieval on gate failure",
     "why": ("T2.4 allows ONE targeted re-retrieval before forcing "
             "Unverifiable. How often that fires on this class is unknown "
             "and could add a whole second pass to an unknown fraction.")},
    {"component": "panel escalation on split",
     "why": ("trump_2026:0462 needed three panel calls to break a "
             "persistent split. The rate is unmeasured on this class.")},
]

PROBE = {
    "design": "n=3, chosen to span the class rather than to be cheap",
    "why_three": ("One claim cannot separate a class-wide rate from a "
                  "claim-specific one, and the three sub-shapes below "
                  "plausibly cost different amounts."),
    "claims": [
        {"sid": "trump_2026:0659", "shape": "valor citation",
         "why": ("Documented in a formal award citation. Tests the case "
                 "where the right source is a single authoritative record.")},
        {"sid": "trump_2026:0090", "shape": "institutional fact",
         "why": ("Olympic host-city decision — an institution's own public "
                 "record. Tests the easiest retrieval shape.")},
        {"sid": "trump_2026:0405", "shape": "press-documented individual",
         "why": ("A scholarship reported in press coverage. Tests the "
                 "hardest shape: no single authoritative record, so the "
                 "gate depends on assembling independent reporting.")},
    ],
    "cost_band_usd": None,
    "cost_band_status": "UNMEASURED",
    "cost_band_note": (
        "Deliberately not quoted. Quoting a band for the probe would recreate "
        "the exact error the probe exists to correct — the probe IS the "
        "measurement. Expect it to be small (3 claims), but 'small' is not a "
        "number and must not be recorded as one."),
    "authorization": ("METERED. Waits for an owner click at D17-d scope. Not "
                      "run here."),
}


def main() -> int:
    triage = json.loads(TRIAGE.read_text(encoding="utf-8"))
    web = [c for c in triage["claims"]
           if c["decidability_class"] == "web-tier1"]
    by_speech: dict[str, list] = {}
    for c in web:
        by_speech.setdefault(c["speech"], []).append(c["sid"])

    # MEASURED: today's pack payload for each web-tier1 claim.
    sys.path.insert(0, str(REPO / "src"))
    from truthbot.verdict.publish_pipeline import packs_from_evidence_dict

    measured, snippet_lens = {}, []
    for speech, run in HEADS.items():
        doc = json.loads((RUNS / f"{run}.json").read_text(encoding="utf-8"))
        wanted = set(by_speech.get(speech, []))
        if not wanted:
            continue
        packs = packs_from_evidence_dict(
            {s: doc["evidence"].get(s) or [] for s in wanted})
        for sid, pack in packs.items():
            measured[sid] = len(json.dumps(pack.to_payload()))
            snippet_lens += [len(i.snippet or "") for i in pack.items]

    n = len(measured)
    total_now = sum(measured.values())
    mean_snip = (sum(snippet_lens) / len(snippet_lens)) if snippet_lens else 0
    # ASSUMED addition: A1 items x (A2 snippet + per-item wrapper overhead).
    per_item_overhead = 120          # {id, source, tier, url} JSON scaffolding
    added_per_claim = 6 * (mean_snip + per_item_overhead)
    projected_total = total_now + added_per_claim * n

    doc = {
        "schema": "truthbot-d17d-webtier1-estimate v1",
        "lane": "web-tier1",
        "claims": n,
        "per_speech": {k: len(v) for k, v in sorted(by_speech.items())},
        "measured": {
            "pack_payload_chars_total": total_now,
            "pack_payload_chars_mean": round(total_now / n, 1) if n else 0,
            "mean_stored_snippet_chars": round(mean_snip, 1),
            "basis": "real bytes from the committed publishing heads",
        },
        "assumed": {
            "added_chars_per_claim": round(added_per_claim, 1),
            "projected_payload_chars_total": round(projected_total, 1),
            "assumptions": ASSUMPTIONS,
            "warning": ("These are ASSUMPTIONS. They are recorded separately "
                        "from the measured figures so the two cannot merge "
                        "into 'the estimate'."),
        },
        "indicative_panel_cost": {
            "usd_per_kchar": MEASURED_USD_PER_KCHAR,
            "projected_usd": round(projected_total / 1000
                                   * MEASURED_USD_PER_KCHAR, 4),
            "status": "PARTIAL — panel payload only",
            "note": ("This prices the panel call on a projected payload. It "
                     "does NOT price retrieval, re-retrieval or escalation "
                     "(below), so it is a FLOOR and must never be quoted as "
                     "the lane cost."),
        },
        "unpriced_components": UNPRICED_COMPONENTS,
        "calibration_probe": PROBE,
        "bottom_line": (
            "The web-tier1 lane cannot be honestly costed from what we have. "
            "The panel-payload floor is computable and is stated; the "
            "retrieval components that likely dominate it are not. Run the "
            "3-claim probe and measure."),
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    m, a, i = doc["measured"], doc["assumed"], doc["indicative_panel_cost"]
    print(f"web-tier1 claims: {n}")
    for k, v in doc["per_speech"].items():
        print(f"  {k:<14}{v:>4}")
    print(f"\nMEASURED pack payload : {m['pack_payload_chars_total']:,} chars "
          f"(mean {m['pack_payload_chars_mean']:,}/claim)")
    print(f"ASSUMED addition      : {a['added_chars_per_claim']:,}/claim "
          f"-> {a['projected_payload_chars_total']:,} total")
    print(f"panel-payload FLOOR   : ${i['projected_usd']}  ({i['status']})")
    print(f"unpriced components   : {len(UNPRICED_COMPONENTS)} — retrieval, "
          f"re-retrieval, escalation")
    print(f"probe                 : {len(PROBE['claims'])} claims, cost band "
          f"{PROBE['cost_band_status']}")
    print(f"\n-> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
