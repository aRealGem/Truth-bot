#!/usr/bin/env python3
"""R-1 — re-shape trump_2026:0031 to c-count and re-adjudicate that ONE claim.

THE SHAPE CORRECTION (the part that costs nothing)
--------------------------------------------------
``trump_2026:0031`` — *"And in the last three months of 2025, it was down to 1.7
percent."* — carries ``claim_shape=c-eval``, assigned by the Layer-A backfill.
Judged from the sentence alone, which is what the classifier's own instruction
requires, that is wrong. The sentence has:

  * no superlative,
  * no causal attribution,
  * no comparison to another entity or era.

Every c-eval trigger nearby belongs to the PRECEDING sentence — :0030, *"my
administration has driven core inflation down to the lowest level in more than
five years"*, which is superlative + causal and is correctly c-eval. :0031
states a bare quantity measured against a published series. That is **c-count**.

**This is a SHAPE CORRECTION, not outcome-shopping, and it is recorded as one.**
It moves the gate's quota branch, and the corrections ledger says so explicitly
rather than letting the shape change ride in silently:

  * c-count is a MINISTERIAL shape, so ``evidential_role`` routes SELF sources
    to PRIMARY_RECORD and PARTICIPANT sources to CORROBORANT, where c-eval ×
    SELF is ATTRIBUTION_ONLY and carries weight 0;
  * c-count is not in ``computed_exhibit.INADMISSIBLE_SHAPES``, so the ratified
    exhibit becomes admissible. Under c-eval it was refused, and correctly:
    arithmetic cannot settle an evaluative claim.

The correction is applied to the ARTIFACT and recorded in its own file. The
shape sidecar is NOT edited: it is the record of what the classifier produced,
and overwriting it would erase the evidence that a correction happened at all.

THE DIRECTIONAL ELEMENT
-----------------------
"Down TO 1.7 percent" asserts a level AND a direction, and one window's rate
cannot establish a direction. Left alone, "down" would rest on the panel's own
arithmetic. So the exhibit carries a SECOND computed row — the same series
(CPILFESL), the same pinned vintage (2026-02-24), the same annualization
formula, over the immediately preceding three months: 3.412% (Jul→Sep) against
1.701% (Oct→Dec), a fall of 1.71 percentage points. Same evidence class as the
first row, so it belongs in the exhibit rather than in the rationale.

trump_2026:0030 is NOT touched: it stays c-eval, gets no exhibit, and publishes
as already adjudicated.

SPEND
-----
One claim, one panel call, on the STORED pack — no retrieval. ``--budget`` is
required with ``--go`` and is a hard halt checked before the call is made.

Usage (repo root)::

    PYTHONPATH=.:src .venv/bin/python scripts/reshape_rerun_0031.py --dry-run
    PYTHONPATH=.:src .venv/bin/python scripts/reshape_rerun_0031.py --go --budget 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_wave_rulings import write_artifact  # noqa: E402
from phase3_rebuild import (BudgetHalt, ChunkFailed,  # noqa: E402
                            _adjudicate_chunk, build_verdict_diff,
                            print_diff, update_manifest)
from wave_adjudicate import (build_wave_packs, exhibit_context,  # noqa: E402
                             load_exhibit, merge_wave_evidence,
                             merge_wave_rows)

OUT_DIR = REPO / "metrics" / "remediation_v2"
RUNS_DIR = REPO / "metrics" / "pca_runs"

SPEECH = "trump_2026"
SID = "trump_2026:0031"
SIBLING = "trump_2026:0030"

RULING_DATE = "2026-08-10"
RULING_TAG = "R-1 shape correction + single-claim re-run"
PIPELINE_GENERATION = "v2.3-role-axis-s5cap"

OLD_SHAPE = "c-eval"
NEW_SHAPE = "c-count"

#: The justification, ON CLAIM TEXT. Stored with the correction so the reason a
#: shape moved travels with the shape and is never re-derived from the outcome.
SHAPE_JUSTIFICATION = (
    "Justified on the claim text alone, which is the basis the Layer-A "
    "classifier is instructed to use: \"And in the last three months of 2025, "
    "it was down to 1.7 percent\" contains no superlative, no causal "
    "attribution and no comparison — the three c-eval triggers. All of them "
    "belong to the PRECEDING sentence (trump_2026:0030, \"my administration "
    "has driven core inflation down to the lowest level in more than five "
    "years\"), which is correctly c-eval and is not changed. What :0031 states "
    "is a bare quantity measured against a published data series, which is "
    "c-count. The original c-eval was a Layer-A backfill artifact of reading "
    "the sentence in its neighbour's context.")

#: What the correction MOVES — stated, because a shape change that quietly
#: alters the gate is the thing the ruling forbids.
SHAPE_EFFECTS = [
    ("gate quota branch",
     "c-count is a MINISTERIAL shape (verdict.evidential_role."
     "MINISTERIAL_SHAPES), so a SELF source becomes PRIMARY_RECORD and a "
     "PARTICIPANT source becomes CORROBORANT. Under c-eval, a SELF source was "
     "ATTRIBUTION_ONLY and carried quota weight 0."),
    ("computed-exhibit admissibility",
     "c-eval is the one shape in publish.computed_exhibit."
     "INADMISSIBLE_SHAPES, so the ratified exhibit was REFUSED for this claim. "
     "c-count is not, so it is now admissible and is attached."),
]

#: Seconds to wait before re-reading the proxy key's spend counter. It is
#: written asynchronously, so an immediate read can report $0 for a call that
#: cost money.
SPEND_SETTLE_S = 20

CORRECTION_PATH = OUT_DIR / "shape_correction_trump_2026_0031.json"
REPORT_PATH = OUT_DIR / "r1_reshape_rerun_report.json"
JOURNAL = REPO / "metrics" / "journals" / "r1_0031_reshape.jsonl"


# ── $0 ───────────────────────────────────────────────────────────────────────

def shipping_artifact(speech: str = SPEECH,
                      runs_dir: Path = RUNS_DIR) -> tuple[Path, dict]:
    """The newest artifact for ``speech`` — the one a publish would render.

    Found by lineage rather than by a pinned id: this run has to sit on top of
    the 2026-08-10 rulings pass, and a pinned id would silently re-run against
    a superseded generation and drop the 27 withholdings it applied.

    The run directory holds several unrelated experiment heads for the same
    speech, so "the head" alone is ambiguous. The head that PUBLISHES is the
    one whose lineage passes through the rulings pass, and that is what this
    selects — refusing outright if it is not unique, because guessing which
    artifact ships is exactly the mistake worth failing on."""
    docs = {}
    children = set()
    for path in sorted(runs_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (doc.get("meta") or {}).get("speech_id") != speech:
            continue
        docs[doc.get("run_id")] = (path, doc)
        parent = (doc.get("meta") or {}).get("rebuild_of")
        if parent:
            children.add(parent)

    def _descends_from_rulings(doc: dict) -> bool:
        seen: set[str] = set()
        cur: Optional[dict] = doc
        while cur is not None:
            meta = cur.get("meta") or {}
            if meta.get("rulings"):
                return True
            parent_id = str(meta.get("rebuild_of") or "")
            if not parent_id or parent_id in seen:
                return False
            seen.add(parent_id)
            entry = docs.get(parent_id)
            cur = entry[1] if entry else None
        return False

    heads = [v for k, v in docs.items() if k not in children]
    published = [v for v in heads if _descends_from_rulings(v[1])]
    if not published:
        raise SystemExit(
            f"no head artifact for {speech} descends from the rulings pass — "
            "run scripts/apply_wave_rulings.py --apply first")
    if len(published) > 1:
        raise SystemExit(
            f"{speech} has {len(published)} publishing heads "
            f"({', '.join(sorted(str(d.get('run_id'))[:8] for _p, d in published))}) "
            "— ambiguous lineage; refusing to guess which one publishes")
    return published[0]


def shape_correction(artifact: dict) -> dict:
    """The correction record: what moved, on what basis, and what it changes."""
    claim = next((c for c in artifact.get("claims") or []
                  if c.get("sid") == SID), {})
    return {
        "schema": "truthbot-shape-correction v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "ruling": "R-1",
        "ruling_date": RULING_DATE,
        "sid": SID,
        "claim_text": (claim.get("text") or "").strip(),
        "old_shape": OLD_SHAPE,
        "new_shape": NEW_SHAPE,
        "kind": "shape correction",
        "justification": SHAPE_JUSTIFICATION,
        "moves": [{"what": what, "how": how} for what, how in SHAPE_EFFECTS],
        "not_changed": {
            "sid": SIBLING,
            "shape": OLD_SHAPE,
            "why": ("superlative plus causal attribution — correctly shaped, "
                    "no exhibit, publishes as already adjudicated"),
        },
        "sidecar_untouched": (
            "metrics/remediation_v2/shapes_backfill_trump_2026.json still "
            "records c-eval. It is the record of what the classifier produced; "
            "overwriting it would erase the evidence that a correction "
            "happened."),
    }


def apply_shape(claims: list[dict]) -> list[dict]:
    """The corrected shape, written onto the claim record so every downstream
    consumer (gate, exhibit admissibility, published provenance) reads the same
    value. Non-destructive: the prior shape is kept beside it."""
    out = []
    for claim in claims:
        if claim.get("sid") != SID:
            out.append(claim)
            continue
        layer_a = dict(claim.get("layer_a") or {})
        layer_a["claim_shape"] = NEW_SHAPE
        layer_a["claim_shape_corrected_from"] = OLD_SHAPE
        layer_a["claim_shape_correction"] = f"R-1 ({RULING_DATE})"
        out.append(dict(claim, layer_a=layer_a))
    return out


def preflight(artifact: dict) -> dict:
    """Everything checkable before any money moves: admissibility under both
    shapes, the exhibit's own arithmetic, and the pack the panel would see."""
    from truthbot.publish import computed_exhibit as ce

    exhibit = load_exhibit()
    comp = exhibit.get("comparison") or {}
    return {
        "exhibit_well_formed": ce.is_well_formed(exhibit),
        "admissible_under_old_shape": ce.is_admissible(
            exhibit, claim_shape=OLD_SHAPE),
        "admissible_under_new_shape": ce.is_admissible(
            exhibit, claim_shape=NEW_SHAPE),
        "inadmissible_shapes": sorted(ce.INADMISSIBLE_SHAPES),
        "exhibit_result_pct": round(float(exhibit["result"]) * 100, 3),
        "comparison_present": bool(comp.get("formula")),
        "comparison_result_pct": (round(float(comp["result"]) * 100, 3)
                                  if comp.get("result") is not None else None),
        "comparison_delta_pp": comp.get("delta_pp"),
    }


def build(artifact: dict) -> tuple[dict, dict, list[dict], dict]:
    """(packs, pack telemetry, claims-for-the-panel, preflight) — all $0."""
    from rescore_stored_packs import b2_sidecar_path, sidecar_path
    from regate_from_rescore import load_rescore_sidecar, merge_sidecars

    parts = [p for p in (sidecar_path(SPEECH), b2_sidecar_path(SPEECH))
             if p.exists()]
    sidecar = merge_sidecars(*(load_rescore_sidecar(p, SPEECH, "")
                               for p in parts))
    packs, telemetry = build_wave_packs(
        SPEECH, artifact, sidecar, [SID],
        shapes_override={SID: NEW_SHAPE})

    checks = preflight(artifact)
    src = next(c for c in artifact["claims"] if c.get("sid") == SID)
    context = (src.get("context") or "")
    if checks["admissible_under_new_shape"]:
        context = context + exhibit_context(load_exhibit())
    claims = [{"sid": SID, "text": src["text"], "context": context}]
    return packs, telemetry, claims, checks


# ── the metered path ─────────────────────────────────────────────────────────

def go_refusal(budget: Optional[float]) -> Optional[str]:
    if budget is None or budget <= 0:
        return ("REFUSING to spend: --budget USD is REQUIRED with --go. It is "
                "the halt cap checked before the panel call. No spend "
                "attempted.")
    return None


def run(args) -> int:
    from hydramind.rosters import get_roster
    from truthbot.verdict import adjudicator, proxy_lane, publish_pipeline

    path, artifact = shipping_artifact()
    correction = shape_correction(artifact)
    packs, telemetry, claims, checks = build(artifact)

    print(f"R-1 shape correction — {SID}")
    print(f"  source artifact : {artifact['run_id'][:8]} ({path.name})")
    print(f"  shape           : {OLD_SHAPE} → {NEW_SHAPE}")
    print(f"  gate            : pack_items={telemetry[0]['pack_items']} "
          f"quota_met={telemetry[0]['quota_met']} "
          f"gate_code={telemetry[0]['gate_code']!r}")
    print(f"  credit          : {json.dumps(telemetry[0]['credit'])}")
    print(f"  exhibit         : admissible under {OLD_SHAPE}="
          f"{checks['admissible_under_old_shape']}, under {NEW_SHAPE}="
          f"{checks['admissible_under_new_shape']} "
          f"(INADMISSIBLE_SHAPES={checks['inadmissible_shapes']})")
    print(f"  exhibit result  : {checks['exhibit_result_pct']}%")
    print(f"  directional row : {checks['comparison_result_pct']}% → "
          f"{checks['exhibit_result_pct']}% "
          f"({checks['comparison_delta_pp']:+.2f} pp)")

    report = {
        "schema": "truthbot-r1-reshape-rerun v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "ruling": "R-1", "ruling_date": RULING_DATE,
        "sid": SID, "source_run": artifact.get("run_id"),
        "correction": correction, "preflight": checks,
        "pack_telemetry": telemetry, "spend_usd": 0.0,
    }

    if not args.go:
        print("\nDRY RUN — no panel call, $0 spent.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CORRECTION_PATH.write_text(
            json.dumps(correction, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"wrote {CORRECTION_PATH}")
        return 0

    refusal = go_refusal(args.budget)
    if refusal:
        sys.exit(refusal)
    if not checks["admissible_under_new_shape"]:
        sys.exit("REFUSING to spend: the exhibit is not admissible under the "
                 "corrected shape, so the re-run would not be the ruled one.")
    if not proxy_lane.key_present():
        sys.exit(proxy_lane.BLOCKED_MSG)

    hm = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    roster_note = {"name": "prod", "seats": dict(get_roster("prod").seats)}
    start_spend = proxy_lane.proxy_key_spend()
    print(f"\nproxy key spend at start: ${start_spend:.4f} "
          f"(HARD cap ${args.budget:.2f})", flush=True)

    done_rows, _, banked, _ = publish_pipeline.load_chunk_journal(JOURNAL)
    if done_rows:
        print(f"resume: {len(done_rows)} row(s) already banked (${banked:.4f})")
        rows = done_rows
    else:
        def pack_builder(sid: str, text: str, context: str):
            spent = proxy_lane.proxy_key_spend() - start_spend
            if spent >= args.budget:
                raise BudgetHalt(f"${spent:.4f} >= cap ${args.budget:.2f} "
                                 f"(before the panel call for {sid})")
            return packs[sid]

        t0 = time.time()
        try:
            rows, _manifest, notes = _adjudicate_chunk(
                adjudicator, hm, claims, pack_builder, 1)
        except (BudgetHalt, ChunkFailed) as exc:
            sys.exit(f"HALTED: {exc}")
        publish_pipeline.append_chunk_journal(
            JOURNAL, 1, rows, notes.get("packs") or {},
            proxy_lane.proxy_key_spend() - start_spend, roster=roster_note)
        print(f"panel call: {time.time() - t0:.0f}s", flush=True)

    # The proxy key's spend counter is written ASYNCHRONOUSLY, so reading it
    # the instant the call returns can report $0 for a call that cost real
    # money — which is what happened on the 2026-08-10 run (banked 0.0, true
    # cost $0.0036). Settle, then re-read, and keep the LARGER of the two: a
    # cost report may never round down.
    settled = proxy_lane.proxy_key_spend() - start_spend
    time.sleep(SPEND_SETTLE_S)
    spend = max(settled, proxy_lane.proxy_key_spend() - start_spend) + banked
    row = next(r for r in rows if r.get("sid") == SID)
    row["computed_exhibit"] = dict(load_exhibit())

    merged = merge_wave_rows(artifact, [row])
    claims_out = apply_shape(artifact["claims"])
    payload_src = dict(artifact, claims=claims_out)
    payload_src["evidence"] = merge_wave_evidence(artifact, packs)
    out_path, payload = write_artifact(
        payload_src, merged, speech_id=SPEECH,
        applied={"date": RULING_DATE, "ruling": "R-1",
                 "shape_correction": correction,
                 "reshaped_sids": [SID], "exhibit_attached": [SID],
                 "source": "scripts/reshape_rerun_0031.py"},
        out_dir=RUNS_DIR)
    payload["meta"]["cost_usd"] = round(spend, 4)
    payload["meta"]["remediation"] = RULING_TAG
    out_path.write_text(json.dumps(payload, default=str, ensure_ascii=False),
                        encoding="utf-8")
    update_manifest(payload["run_id"], SPEECH)

    old_rows = {r.get("sid"): r for r in artifact["rows"]}
    diff = build_verdict_diff([old_rows[SID]], [row], artifact["claims"])
    print(f"\nartifact {out_path.name} (rebuild_of {artifact['run_id'][:8]})")
    print_diff(diff)
    diff_out = {"speech_id": SPEECH, "rebuild_of": artifact.get("run_id", ""),
                "new_run_id": payload["run_id"], "r1": True,
                "ruling_sids": [SID],
                "shape": {"from": OLD_SHAPE, "to": NEW_SHAPE},
                **diff}
    (OUT_DIR / f"r1_{SPEECH}_verdict_diff.json").write_text(
        json.dumps(diff_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    report.update(spend_usd=round(spend, 4), cap_usd=args.budget,
                  new_run_id=payload["run_id"], artifact=str(out_path),
                  verdict=row.get("verdict"),
                  reasoning=row.get("reasoning") or "",
                  exhibit_attached=True, counts=diff["counts"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTION_PATH.write_text(
        json.dumps(correction, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nverdict: {row.get('verdict')}")
    print(f"rationale: {row.get('reasoning')}")
    print(f"\nSPEND: ${spend:.4f} of cap ${args.budget:.2f}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="preflight only: gate, admissibility, exhibit. $0")
    group.add_argument("--go", action="store_true",
                       help="run the single panel call (REQUIRES --budget)")
    ap.add_argument("--budget", type=float, default=None, metavar="USD",
                    help="hard halt cap, checked before the call")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
