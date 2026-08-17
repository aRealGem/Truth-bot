"""D17-d STEP 3 fixture: the 62 abstentions, laid out for a human read ($0).

    scripts/d17d_step3_read_fixture.py [--json PATH]

WHY THIS EXISTS
---------------
Step 2 established that the null-stance gap is an ABSTENTION set: every one of
the 62 creditable items on the 23 stance-limited claims was inside the scorer's
scope, was scored, and came back with no stance. Whether a re-score could move
them is a question about the TEXT, and the cheapest way to answer it is for a
competent reader to look at the claim beside the snippet and judge whether a
stance is derivable at all. That read predicts yield BEFORE any money is spent.

This builds the fixture that read runs on. It sends nothing and fetches nothing;
every field comes from artifacts already on disk.

THE SNIPPET IS VERBATIM AND UNTRUNCATED, deliberately. The whole hypothesis
under test is that the abstentions are a snippet-GRANULARITY limit, so trimming
the snippet would destroy the evidence the reader needs. If a snippet is short,
that is a finding, not a formatting problem.

TWO RELEVANCE FIELDS, both disclosed. ``relevance_score_pack`` is what the
stored pack (and therefore the gate and the renderer) carries;
``relevance_score_sidecar`` is what the B1a/B2 re-score sidecar recorded. They
should agree because the sidecar stances were propagated into the head, but a
fixture that silently picked one would hide a disagreement, so both ship.

ROLLUPS (M-6 bookkeeping): per-speech and per-speaker counts are emitted so the
read's coverage is auditable and no speech's share is left implicit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

STEP2 = REPO / "metrics" / "remediation_v2" / "d17d_step2_null_scope.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_step3_read_fixture.json"


def run(out_path: Path = OUT) -> dict:
    import importlib.util

    import phase3_rebuild as p3
    import regate_from_rescore as rg
    import rescore_stored_packs as rsp
    from reshape_rerun_0031 import shipping_artifact

    spec = importlib.util.spec_from_file_location(
        "cs", str(REPO / "scripts" / "d17d_credit_supply.py"))
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    fair_game = cs._fair_game_days()

    step2 = json.loads(STEP2.read_text(encoding="utf-8"))
    gap: dict[str, list] = {}
    for r in step2["claims"]:
        gap.setdefault(r["speech"], []).append(r["sid"])

    rows: list[dict] = []
    speakers: dict[str, str] = {}
    for speech in sorted(gap):
        _path, art = shipping_artifact(speech)
        meta = art.get("meta") or {}
        speaker = meta.get("speaker") or "(unrecorded)"
        speakers[speech] = speaker
        utterance = p3.SPEECHES[speech]["date"]
        claims = {c.get("sid"): c for c in (art.get("claims") or [])}
        evidence = art.get("evidence") or {}

        b1a = rsp.load_sidecar(rsp.sidecar_path(speech), speech, "")
        b2_path = rsp.b2_sidecar_path(speech)
        b2 = rsp.load_sidecar(b2_path, speech, "") if b2_path.exists() else None
        scored = (rg.merge_sidecars(b1a, b2).get("sids") or {})

        for sid in sorted(gap[speech]):
            claim_text = (claims.get(sid, {}).get("text") or "").strip()
            sidecar_rows = {rg.join_key(r.get("source_url") or ""): r
                            for r in (scored.get(sid) or [])}
            for ev in evidence.get(sid) or []:
                # Only the items that COULD credit on tier+date but carry no
                # stance -- the exact set step 2 counted.
                if cs.classify_item(ev, utterance, fair_game) != "creditable":
                    continue
                if ev.get("supports_claim") is not None:
                    continue
                srow = sidecar_rows.get(rg.join_key(ev.get("source_url") or ""))
                rows.append({
                    "sid": sid,
                    "speech": speech,
                    "speaker": speaker,
                    "utterance_date": utterance.isoformat(),
                    "claim_text": claim_text,
                    "item_id": ev.get("id"),
                    "source_url": ev.get("source_url"),
                    "source_name": ev.get("source_name"),
                    "source_tier": ev.get("source_tier"),
                    "published_at": ev.get("published_at"),
                    "relevance_score_pack": ev.get("relevance_score"),
                    "relevance_score_sidecar": (srow or {}).get("relevance_score"),
                    # VERBATIM, untruncated -- see module docstring.
                    "snippet": ev.get("snippet"),
                    "snippet_len": len(ev.get("snippet") or ""),
                })

    by_speech: dict[str, dict] = {}
    by_speaker: dict[str, dict] = {}
    by_tier: dict[str, int] = {}
    n = len(rows) or 1
    for r in rows:
        s = by_speech.setdefault(r["speech"], {"items": 0, "claims": set(),
                                               "speaker": r["speaker"]})
        s["items"] += 1
        s["claims"].add(r["sid"])
        k = by_speaker.setdefault(r["speaker"], {"items": 0, "speeches": set()})
        k["items"] += 1
        k["speeches"].add(r["speech"])
        by_tier[r["source_tier"]] = by_tier.get(r["source_tier"], 0) + 1
    for s in by_speech.values():
        s["claims"] = len(s["claims"])
        s["share_of_total"] = f"{s['items']}/{len(rows)}"
        s["share_pct"] = round(100 * s["items"] / n, 1)
    for k in by_speaker.values():
        k["speeches"] = sorted(k["speeches"])
        k["share_of_total"] = f"{k['items']}/{len(rows)}"
        k["share_pct"] = round(100 * k["items"] / n, 1)

    lens = sorted(r["snippet_len"] for r in rows)
    report = {
        "schema": "truthbot-d17d-step3-read-fixture v1",
        "generated": _now(),
        "purpose": ("Human resolvability read over the null-stance abstentions: "
                    "for each item, can a competent reader derive a True/False "
                    "stance on the claim from this snippet alone? The answer "
                    "predicts re-score yield before any spend."),
        "method": ("$0, stored data only -- shipping-head packs joined to the "
                   "merged B1a+B2 sidecars. No fetches, no model calls."),
        "source_step2": step2.get("generated"),
        "n_items": len(rows),
        "n_claims": len({r["sid"] for r in rows}),
        "snippet_verbatim": True,
        "snippet_len_min": lens[0] if lens else None,
        "snippet_len_median": lens[len(lens) // 2] if lens else None,
        "snippet_len_max": lens[-1] if lens else None,
        "rollup_by_speech": by_speech,
        "rollup_by_speaker": by_speaker,
        "rollup_by_source_tier": by_tier,
        "items": rows,
        "read_protocol_note": (
            "Suggested per-item verdict for the reader: RESOLVABLE (a stance "
            "follows from this snippet), NOT-RESOLVABLE-BUT-RELEVANT (on topic, "
            "does not bear on the specific assertion -- the abstention was "
            "CORRECT), or NEEDS-FULL-TEXT (the source plausibly settles it but "
            "this excerpt does not). The third bucket is the one that justifies "
            "a fuller-text re-score; the second is the one that says the lane "
            "is not recoverable at any price."),
        "spend_note": ("Nothing here authorizes spend. The read is free; a "
                       "re-score is not, and needs a separate owner cap."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def print_report(rep: dict) -> None:
    print(f"\nD17-d step 3 read fixture ($0) -- {rep['n_items']} abstention "
          f"items across {rep['n_claims']} claims")
    print(f"  snippet length min/median/max: {rep['snippet_len_min']}/"
          f"{rep['snippet_len_median']}/{rep['snippet_len_max']} (verbatim)\n")
    print("  by speech:")
    for sp, d in sorted(rep["rollup_by_speech"].items(),
                        key=lambda kv: -kv[1]["items"]):
        print(f"    {sp:<13} {d['share_of_total']:>6} ({d['share_pct']:>4}%)  "
              f"claims={d['claims']:<3} speaker={d['speaker']}")
    print("  by speaker:")
    for k, d in sorted(rep["rollup_by_speaker"].items(),
                       key=lambda kv: -kv[1]["items"]):
        print(f"    {k:<16} {d['share_of_total']:>6} ({d['share_pct']:>4}%)")
    print(f"  by source tier: {rep['rollup_by_source_tier']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=str(OUT))
    args = ap.parse_args(argv)
    rep = run(Path(args.json))
    print_report(rep)
    print(f"\nfixture -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
