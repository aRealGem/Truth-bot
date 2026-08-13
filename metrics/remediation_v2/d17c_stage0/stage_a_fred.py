#!/usr/bin/env python3
"""D17-c Stage A-FRED — does putting the rows in front of the scorer move it?

Whole-pack rescore of the 7 claims carrying the 8 excerptable FRED items, with
each excerpt APPENDED to its item's stored snippet at payload assembly. Append,
never replace: the census measures the MARGINAL effect of the rows, so the rest
of the item has to stay exactly as the shipped run had it.

WHAT THIS DOES NOT DO. It publishes nothing, renders nothing, and writes nothing
back to any stored pack — augmentation happens on in-memory copies at payload
assembly only, and the census lands in a sidecar. Gate implications are COMPUTED
and reported, never APPLIED. Nothing stance-bearing outside the 7 claims is
touched.

THE TRUNCATION CONTRACT. Excerpts run to ~22,000 characters against a 400-char
default cap, so the run sets a cap that provably cannot bite and then ASSERTS
per-item ``chars_truncated == 0`` rather than trusting it. A silent clip here
would produce a complete-looking census measuring 400-character stubs, which is
the specific failure this whole path was built to avoid.

Every per-item payload is sha256'd so the run is byte-reproducible.

Usage (repo root):
  # $0 — plan, price and prove the augmentation without calling anything:
  .venv/bin/python metrics/remediation_v2/d17c_stage0/stage_a_fred.py
  # SPENDS MONEY, capped:
  set -a; . ./.env; . ~/.env; set +a
  .venv/bin/python metrics/remediation_v2/d17c_stage0/stage_a_fred.py \\
      --go --budget 0.15
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "pca_runs"
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

import select_rows as S  # noqa: E402

#: Provably cannot bite: the largest excerpt is ~22k and the largest stored
#: snippet in the corpus is 207. Asserted per item anyway.
STAGE_A_CAP = None

#: Fable's ceiling. Ledger truth, checked before and after every call.
CEILING = 0.15

#: R2: recorded, non-actionable for any Stage B consideration.
PERIOD_MISMATCH = {("obama_2014:0189", "E4")}

SEP = "\n---\n"


def augmented_packs() -> tuple[dict, dict, dict]:
    """(sid -> [Evidence] with excerpts appended, sid -> claim text, aug index).

    Works on deep copies of the stored evidence. The artifacts on disk are the
    record and are not touched.
    """
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict

    goldens = {(g["claim_sid"], g["evidence_id"]): g
               for g in S.build() if g["role"] == "wave1"}
    sids = sorted({sid for sid, _ in goldens})

    packs: dict[str, list] = {}
    texts: dict[str, str] = {}
    aug: dict[tuple[str, str], dict] = {}

    for speech in sorted({s.split(":")[0] for s in sids}):
        doc = json.loads((RUNS / f"{S.HEADS[speech]}.json").read_text())
        texts.update({c["sid"]: c["text"] for c in doc["claims"]})
        wanted = {s for s in sids if s.startswith(f"{speech}:")}
        ev = evidence_from_artifact_dict(
            {s: copy.deepcopy(doc["evidence"][s]) for s in wanted})
        packs.update(ev)

    for (sid, eid), g in goldens.items():
        idx = int(eid[1:]) - 1
        item = packs[sid][idx]
        assert item.source_url == g["full_table"] or "fred.stlouisfed.org" \
            in item.source_url, f"{sid} {eid}: excerpt joined to the wrong item"
        stored = item.snippet or ""
        item.snippet = stored + SEP + S.render(g)
        aug[(sid, eid)] = {"stored_chars": len(stored),
                           "excerpt_chars": len(S.render(g)),
                           "rows_shown": g["rows_shown"],
                           "series_id": g["series_id"]}
    return packs, texts, aug


def item_payload_shas(texts: dict, packs: dict) -> tuple[dict, list]:
    """Per-item {chars_sent, chars_truncated, payload_sha} + any clip violations."""
    from truthbot.verify.relevance import score_payload_ex

    per_item, violations = {}, []
    for sid, evs in sorted(packs.items()):
        payload, meta = score_payload_ex(texts[sid], evs, STAGE_A_CAP)
        items = json.loads(payload)["items"]
        for m, it in zip(meta, items):
            eid = f"E{m['i']}"
            blob = json.dumps(it, sort_keys=True).encode()
            per_item[(sid, eid)] = {
                "chars_sent": m["chars_sent"],
                "chars_truncated": m["chars_truncated"],
                "payload_sha256": hashlib.sha256(blob).hexdigest()}
            if m["chars_truncated"]:
                violations.append((sid, eid, m["chars_truncated"]))
    return per_item, violations


def plan(packs, texts, aug, per_item) -> None:
    from truthbot.costs import CHARS_PER_TOKEN, rates
    r_in, _ = rates("claude-haiku")
    total_items = sum(len(v) for v in packs.values())
    sent = sum(v["chars_sent"] for v in per_item.values())
    exc = sum(a["excerpt_chars"] for a in aug.values())
    print(f"claims {len(packs)}   pack items {total_items}   "
          f"excerpted items {len(aug)}")
    print(f"excerpt chars appended {exc:,}   total payload chars {sent:,}")
    print(f"input tokens {sent / CHARS_PER_TOKEN:,.0f}   "
          f"input cost ${sent / CHARS_PER_TOKEN * r_in / 1e6:.4f}")
    print("\naugmented items:")
    for (sid, eid), a in sorted(aug.items()):
        pi = per_item[(sid, eid)]
        flag = "  MISMATCH(non-actionable)" if (sid, eid) in PERIOD_MISMATCH else ""
        print(f"  {sid:<18}{eid:<4}{a['series_id']:<16}"
              f"{a['rows_shown']:>5} rows  stored {a['stored_chars']:>4}c "
              f"+ excerpt {a['excerpt_chars']:>6}c  sent {pi['chars_sent']:>6}c "
              f"trunc {pi['chars_truncated']}{flag}")


def gate_implications(rows: list[dict]) -> dict:
    """What the T2.4 bearing quota WOULD do. Computed, never applied.

    ``consolidator._bearing()`` needs a True/False stance, so a null cannot
    credit ``MIN_BEARING_T13``: a pack full of good government evidence can be
    forced Unverifiable purely because nothing was scored. That is the defect
    D17-c exists to test, so the census has to say whether the rows moved any
    claim across the line — without moving it.
    """
    from truthbot.models import SourceTier
    from truthbot.verdict.consolidator import MIN_BEARING_T13

    t13 = {t.value for t in (SourceTier.GOVERNMENT, SourceTier.WIRE,
                             SourceTier.ESTABLISHED)}
    tiers = {}
    for speech, run in S.HEADS.items():
        path = RUNS / f"{run}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        for sid, items in doc["evidence"].items():
            for i, e in enumerate(items, start=1):
                tiers[(sid, f"E{i}")] = e.get("source_tier")

    out = {"rule": "MIN_BEARING_T13", "threshold": MIN_BEARING_T13,
           "applied": False, "note": "COMPUTED NOT APPLIED — no verdict "
                                     "rewritten, nothing published",
           "claims": {}}
    for sid in sorted({r["claim_sid"] for r in rows}):
        rs = [r for r in rows if r["claim_sid"] == sid]
        count = lambda k: sum(  # noqa: E731
            1 for r in rs
            if r[k] is not None and tiers.get((sid, r["evidence_id"])) in t13)
        b, a = count("stance_before"), count("stance_after")
        out["claims"][sid] = {
            "bearing_t13_before": b, "bearing_t13_after": a,
            "gate_before": "pass" if b >= MIN_BEARING_T13 else "forced_unverifiable",
            "gate_after": "pass" if a >= MIN_BEARING_T13 else "forced_unverifiable",
            "gate_outcome_changes": (b >= MIN_BEARING_T13) != (a >= MIN_BEARING_T13)}
    out["claims_clearing_before"] = sum(
        1 for v in out["claims"].values() if v["gate_before"] == "pass")
    out["claims_clearing_after"] = sum(
        1 for v in out["claims"].values() if v["gate_after"] == "pass")
    return out


def census(packs, texts, aug, per_item, before, spend) -> dict:
    """Item stance deltas vs the shipped verdicts. Computed, never applied."""
    rows = []
    for sid, evs in sorted(packs.items()):
        for i, ev in enumerate(evs, start=1):
            eid = f"E{i}"
            b_rel, b_sup = before[(sid, eid)]
            pi = per_item[(sid, eid)]
            rows.append({
                "claim_sid": sid, "evidence_id": eid,
                "excerpted": (sid, eid) in aug,
                "series_id": aug.get((sid, eid), {}).get("series_id"),
                "stance_before": b_sup, "stance_after": ev.supports_claim,
                "stance_flipped": b_sup != ev.supports_claim,
                "relevance_before": b_rel, "relevance_after": ev.relevance_score,
                "one_line_why": ev.one_line_why,
                "arithmetic_hinge": getattr(ev, "arithmetic_hinge", None),
                "chars_sent": pi["chars_sent"],
                "chars_truncated": pi["chars_truncated"],
                "payload_sha256": pi["payload_sha256"],
                "window_period_mismatch": (sid, eid) in PERIOD_MISMATCH,
                "stage_b_actionable": (sid, eid) not in PERIOD_MISMATCH,
            })
    flips = [r for r in rows if r["stance_flipped"]]
    return {
        "schema": "truthbot-d17c-stage-a-census v1",
        "scope": {"claims": len(packs), "pack_items": len(rows),
                  "excerpted_items": len(aug)},
        "selector_run_sha256": hashlib.sha256(json.dumps(
            S.build(), sort_keys=True, indent=2).encode()).hexdigest(),
        "cap": STAGE_A_CAP,
        "chars_truncated_total": sum(r["chars_truncated"] for r in rows),
        "spend_usd": round(spend, 6),
        "flips": {"total": len(flips),
                  "on_excerpted_items": sum(1 for r in flips if r["excerpted"]),
                  "on_other_items": sum(1 for r in flips if not r["excerpted"])},
        "gate_implications": gate_implications(rows),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually spend")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--model", default="claude-haiku")
    ap.add_argument("--recompute-gate", action="store_true",
                    help="$0: recompute the gate block from the saved census "
                         "and rewrite it, without re-scoring anything")
    args = ap.parse_args()

    if args.recompute_gate:
        out = HERE / "stage_a_census.json"
        doc = json.loads(out.read_text())
        doc["gate_implications"] = gate_implications(doc["rows"])
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        g = doc["gate_implications"]
        for sid, v in sorted(g["claims"].items()):
            mark = "  <-- CHANGES" if v["gate_outcome_changes"] else ""
            print(f"  {sid:<18}{v['bearing_t13_before']:>3} -> "
                  f"{v['bearing_t13_after']:<3}  {v['gate_before']} -> "
                  f"{v['gate_after']}{mark}")
        print(f"\nclaims clearing the quota: {g['claims_clearing_before']} -> "
              f"{g['claims_clearing_after']} of {len(g['claims'])}")
        print(f"{g['note']}\ncensus -> {out.name}")
        return 0

    packs, texts, aug = augmented_packs()
    per_item, violations = item_payload_shas(texts, packs)

    print("=== STAGE A-FRED: plan ===")
    plan(packs, texts, aug, per_item)

    if violations:
        print("\nHALT: payload was truncated — the census would measure stubs")
        for sid, eid, n in violations:
            print(f"  {sid} {eid}: {n} chars clipped")
        return 1
    print(f"\ntruncation assert: chars_truncated == 0 on all "
          f"{len(per_item)} items -> PASS")

    if not args.go:
        print("\n$0 plan only. Re-run with --go --budget 0.15 to spend.")
        return 0
    if args.budget is None or args.budget > CEILING:
        print(f"\n--budget is required and must not exceed ${CEILING:.2f}")
        return 1

    from truthbot.verdict import proxy_lane
    from truthbot.verify.relevance import build_proxy_llm, score_evidence

    if not proxy_lane.key_present():
        print(proxy_lane.BLOCKED_MSG)
        return 1

    before = {(sid, f"E{i}"): (ev.relevance_score, ev.supports_claim)
              for sid, evs in packs.items() for i, ev in enumerate(evs, start=1)}

    llm = build_proxy_llm(args.model)
    start = proxy_lane.proxy_key_spend()
    print(f"\n=== funded run (ledger start ${start:.6f}) ===")

    for n, (sid, evs) in enumerate(sorted(packs.items()), start=1):
        spent = proxy_lane.proxy_key_spend() - start
        if spent >= args.budget:
            print(f"HALT: ${spent:.6f} >= budget ${args.budget:.2f} "
                  f"before {sid}")
            return 1
        score_evidence(llm, texts[sid], evs, STAGE_A_CAP)
        print(f"  [{n}/{len(packs)}] {sid}  spent ${spent:.6f}", flush=True)

    spend = proxy_lane.proxy_key_spend() - start
    print(f"\nledger spend ${spend:.6f}   ceiling ${CEILING:.2f}   "
          f"{'OK' if spend <= CEILING else 'BREACH'}")

    doc = census(packs, texts, aug, per_item, before, spend)
    out = HERE / "stage_a_census.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\nflips {doc['flips']['total']} "
          f"(excerpted {doc['flips']['on_excerpted_items']}, "
          f"other {doc['flips']['on_other_items']})")
    print(f"census -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
