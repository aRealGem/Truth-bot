#!/usr/bin/env python3
"""Propagate the B1a+B2 re-scores into NEW publishing-head artifacts — $0.

This script makes NO model or API calls. It moves numbers that were already
bought, from the sidecars they were parked in onto the artifacts a publish
would actually render.

WHY IT EXISTS
-------------
B1a and B2 re-scored all 4,344 stored evidence items (stance-null 26.2% ->
~15.6%, relevance 0% -> ~97%). Those scores live ONLY in
``metrics/remediation_v2/rescored_<speech>.json`` and ``rescored_b2_<speech>.json``
because the re-score deliberately left the artifacts unmutated. The adjudication
wave OVERLAID the sidecars at runtime, so every verdict it reached WAS reached on
scored evidence — but the artifacts that would be PUBLISHED still carry the
original unscored evidence. That is two defects, not one:

  1. the Phase-A fitness gate (``publish.consistency.is_fit_to_gate``) correctly
     refuses to publish a run whose stance-null rate exceeds 15%, so the publish
     can never proceed; and
  2. were it forced through, the site would display the panel's OWN evidence as
     unscored — published provenance that does not match what was adjudicated.

(2) is why the fix is a merge and not a threshold change. The gate is right.

WHAT IT DOES
------------
For each speech, take the current PUBLISHING HEAD — selected by
``rerender_pca_site.publishing_heads``, i.e. the same artifact the publish
renders, imported rather than re-derived — and write a NEW artifact whose
evidence carries the sidecars' ``relevance_score`` / ``supports_claim`` (plus
the B2 contract fields ``one_line_why`` / ``arithmetic_hinge``), joined on
``source_url`` within each sid.

The merge is not reimplemented either. ``regate_from_rescore.merge_sidecars``
(B1a first, B2 overriding per SID) and ``regate_from_rescore.overlay_rescores``
(the source_url join) are imported, so the evidence written here is the evidence
the panel saw — if the two ever diverged, a published pack would disagree with
the verdict rendered beside it.

NOTHING IS DROPPED SILENTLY, IN EITHER DIRECTION
------------------------------------------------
A sidecar row with no home in the head (a sid the head no longer carries, or a
URL that is not in that sid's pack) and a head item the sidecars never covered
(evidence the wave / escape runs added afterwards, which keeps its existing
values) are both COUNTED and REPORTED. A merge that quietly lost a quarter of
its rows would look exactly like a merge that worked.

WHAT IT MUST NOT DO
-------------------
Move a verdict. ``rows``, ``claims``, ``characterization`` and ``roster`` are
carried across verbatim; only ``evidence`` changes. The driver asserts the
verdict map is identical before and after and refuses to write if it is not.

Prior artifacts are never touched: each speech gets a NEW run id whose
``rebuild_of`` points at the head it derives from, at the SAME generation, and
the manifest gains a row (``published: false``) without any existing row being
edited.

Usage (repo root, always $0):
  PYTHONPATH=.:src .venv/bin/python scripts/propagate_rescores.py            # dry run
  PYTHONPATH=.:src .venv/bin/python scripts/propagate_rescores.py --apply
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# scripts/ is not a package, so these are path-based imports — the same way the
# wave and re-gate load each other. Everything reused here is $0.
from phase3_rebuild import update_manifest  # noqa: E402
from regate_from_rescore import (load_rescore_sidecar,  # noqa: E402
                                 merge_sidecars, overlay_rescores)
from truthbot.publish.heads import publishing_heads  # noqa: E402
from rescore_stored_packs import (REBUILT_RUNS, b2_sidecar_path,  # noqa: E402
                                  load_artifact, sidecar_path)

RUNS_DIR = REPO / "metrics" / "pca_runs"
OUT_DIR = REPO / "metrics" / "remediation_v2"
REPORT_STEM = "score_propagation"

#: Generation is UNCHANGED. This merge does not change how a pack was built,
#: retrieved or gated — it attaches scores that were bought for these very
#: items. A new generation would falsely claim the method moved.
PIPELINE_GENERATION = "v2.3-role-axis-s5cap"
PROPAGATION_TAG = "score propagation (B1a+B2 sidecars -> published evidence)"
PROPAGATION_DATE = "2026-08-10"


# ── the dict/attribute adapter ───────────────────────────────────────────────

class EvidenceView:
    """Attribute view over a stored artifact's evidence DICT.

    ``overlay_rescores`` was written against ``Evidence`` objects (it reads
    ``ev.source_url`` and assigns ``ev.relevance_score`` / ``ev.supports_claim``
    / ``ev.one_line_why`` / ``ev.arithmetic_hinge``). The artifact stores plain
    dicts. Rebuilding real ``Evidence`` models to satisfy the signature and
    re-dumping them would rewrite every stored field — datetime formats,
    defaults the dump never carried — turning a score merge into a wholesale
    re-serialization. This view lets the SAME merge function run, writing
    straight into the stored dict and touching nothing it was not given.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict) -> None:
        object.__setattr__(self, "_d", d)

    @property
    def source_url(self) -> str:
        return self._d.get("source_url") or ""

    def __setattr__(self, name: str, value) -> None:
        self._d[name] = value


# ── merge ────────────────────────────────────────────────────────────────────

def merged_sidecar(speech: str, *, use_b2: bool = True) -> dict:
    """B1a + B2 for one speech, merged in the order the wave merged them.

    Both are validated against ``REBUILT_RUNS[speech]`` — the artifact revision
    they were SCORED against — by ``load_rescore_sidecar``. The head they are
    merged INTO is a descendant of that revision, which is exactly why the join
    is by ``source_url`` and why non-coverage is reported rather than assumed
    away.
    """
    b1a = load_rescore_sidecar(sidecar_path(speech), speech, REBUILT_RUNS[speech])
    b1a["pass_label"] = "b1a"
    b2 = None
    p2 = b2_sidecar_path(speech)
    if use_b2 and p2.exists():
        b2 = load_rescore_sidecar(p2, speech, REBUILT_RUNS[speech])
        b2["pass_label"] = "b2"
    return merge_sidecars(b1a, b2)


def sidecar_vintages(speech: str, *, use_b2: bool = True) -> list[dict]:
    """The provenance block written into the new artifact's meta: which file,
    which pass, which model, when, scored against which run, what it cost."""
    out = []
    for label, path in (("b1a", sidecar_path(speech)),
                        ("b2", b2_sidecar_path(speech))):
        if label == "b2" and not (use_b2 and path.exists()):
            continue
        doc = load_rescore_sidecar(path, speech, REBUILT_RUNS[speech])
        out.append({
            "pass": label,
            "path": str(path.relative_to(REPO)),
            "model": doc.get("model") or "",
            "generated": doc.get("generated") or "",
            "scored_against_run": doc.get("source_run") or "",
            "sids": len(doc.get("sids") or {}),
            "spend_usd": round(float(doc.get("spend_usd") or 0.0), 6),
        })
    return out


def propagate_evidence(evidence: dict, sidecar_sids: dict) -> tuple[dict, dict]:
    """Overlay the merged sidecar onto a COPY of the head's evidence map.

    Returns ``(new_evidence, telemetry)``. Telemetry names every miss in both
    directions: sidecar sids the head does not carry, sidecar rows whose URL is
    not in the matching pack, and head items no sidecar row covers (those keep
    the values they already had — the wave and escape runs added evidence after
    the re-score, and inventing scores for it would be worse than counting it).
    """
    new_evidence = copy.deepcopy(evidence or {})
    matched = 0
    items = 0
    sidecar_unmatched: list[dict] = []
    artifact_unscored: list[dict] = []
    sids_missing_from_head: list[str] = []
    sids_missing_from_sidecar: list[str] = []

    for sid in (sidecar_sids or {}):
        if sid not in new_evidence:
            sids_missing_from_head.append(sid)
            sidecar_unmatched.extend(
                {"sid": sid, "source_url": r.get("source_url") or "",
                 "why": "sid not in head"}
                for r in (sidecar_sids.get(sid) or []))

    for sid, evs in new_evidence.items():
        rows = (sidecar_sids or {}).get(sid)
        views = [EvidenceView(ev) for ev in (evs or [])]
        items += len(views)
        if rows is None:
            sids_missing_from_sidecar.append(sid)
            artifact_unscored.extend(
                {"sid": sid, "source_url": v.source_url,
                 "why": "sid never re-scored"} for v in views)
            continue
        tel = overlay_rescores(views, rows)
        matched += tel["matched"]
        sidecar_unmatched.extend(
            {"sid": sid, "source_url": u, "why": "url not in pack"}
            for u in tel["sidecar_unmatched"])
        artifact_unscored.extend(
            {"sid": sid, "source_url": u, "why": "item not in sidecar"}
            for u in tel["artifact_unscored"])

    telemetry = {
        "packs": len(new_evidence),
        "items": items,
        "matched": matched,
        "sidecar_unmatched": sidecar_unmatched,
        "artifact_unscored": artifact_unscored,
        "sids_missing_from_head": sorted(sids_missing_from_head),
        "sids_missing_from_sidecar": sorted(sids_missing_from_sidecar),
    }
    return new_evidence, telemetry


# ── verdict invariance ───────────────────────────────────────────────────────

def verdict_map(doc: dict) -> dict:
    """``sid -> verdict`` over an artifact's rows. The thing this merge is
    forbidden to move: it changes evidence PROVENANCE, never an outcome."""
    return {r.get("sid"): r.get("verdict") for r in (doc.get("rows") or [])}


# ── artifact writing ─────────────────────────────────────────────────────────

def build_artifact(head: dict, new_evidence: dict, *, speech: str,
                   sidecars: list[dict], telemetry: dict,
                   run_id: Optional[str] = None) -> dict:
    """The new head. Everything but ``evidence`` is carried across verbatim."""
    meta = dict(head.get("meta") or {})
    meta.update({
        "speech_id": speech,
        "cost_usd": 0.0,
        "rebuild_of": head.get("run_id", ""),
        "pipeline_generation": PIPELINE_GENERATION,
        "remediation": PROPAGATION_TAG,
        "score_propagation": {
            "date": PROPAGATION_DATE,
            "source": "scripts/propagate_rescores.py ($0, no model calls)",
            "what": (
                "relevance_score / supports_claim (and the B2 contract fields "
                "one_line_why / arithmetic_hinge) merged from the B1a+B2 "
                "re-score sidecars onto this speech's stored evidence. The "
                "wave overlaid these same scores at runtime, so this moves the "
                "PUBLISHED evidence into agreement with what was adjudicated."),
            "merge_order": ("B1a first, B2 overriding per SID — "
                            "regate_from_rescore.merge_sidecars"),
            "join": "source_url within sid (regate_from_rescore.overlay_rescores)",
            "retrieval": "none — stored packs re-scored, never re-retrieved",
            "verdicts": "unchanged — rows carried across verbatim",
            "sidecars": sidecars,
            "coverage": {
                "packs": telemetry["packs"],
                "items": telemetry["items"],
                "matched": telemetry["matched"],
                "sidecar_rows_unmatched": len(telemetry["sidecar_unmatched"]),
                "items_not_covered": len(telemetry["artifact_unscored"]),
            },
        },
    })
    payload = {
        "run_id": run_id or str(uuid.uuid4()),
        "meta": meta,
        "claims": list(head.get("claims") or []),
        "rows": list(head.get("rows") or []),
        "characterization": list(head.get("characterization") or []),
        "roster": head.get("roster") or {},
        "evidence": new_evidence,
    }
    try:
        from truthbot.verdict.composition_telemetry import composition_report
        payload["composition"] = composition_report(payload["rows"],
                                                    payload["evidence"])
    except Exception:
        pass
    return payload


def write_artifact(payload: dict, out_dir: Path = RUNS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload, default=str, ensure_ascii=False),
                    encoding="utf-8")
    return path


# ── driver ───────────────────────────────────────────────────────────────────

def fitness(doc: dict) -> dict:
    from truthbot.publish.consistency import is_fit_to_gate
    from truthbot.verdict.consolidator import scoring_telemetry_from_artifact

    tel = scoring_telemetry_from_artifact(doc.get("evidence") or {})
    fit, reason = is_fit_to_gate(doc)
    n = max(tel["items"], 1)
    return {"items": tel["items"],
            "relevance_scored": tel["relevance_scored"],
            "relevance_rate": tel["relevance_scored"] / n,
            "stance_null": tel["stance_null"],
            "stance_null_rate": tel["stance_null_rate"],
            "fit_to_gate": fit, "reason": reason}


def propagate_speech(speech: str, head_path: Path, *,
                     use_b2: bool = True) -> dict:
    """One speech, end to end, WITHOUT writing anything."""
    head = load_artifact(head_path)
    sidecar = merged_sidecar(speech, use_b2=use_b2)
    before = fitness(head)
    new_evidence, telemetry = propagate_evidence(head.get("evidence") or {},
                                                 sidecar.get("sids") or {})
    payload = build_artifact(head, new_evidence, speech=speech,
                             sidecars=sidecar_vintages(speech, use_b2=use_b2),
                             telemetry=telemetry)
    after = fitness(payload)

    # A merge that moved a verdict is not a merge. Refuse, loudly, before the
    # artifact can reach disk.
    if verdict_map(head) != verdict_map(payload):
        raise SystemExit(f"{speech}: VERDICTS MOVED — refusing to write. This "
                         "merge may only change evidence provenance.")
    return {
        "speech": speech,
        "head": head.get("run_id", ""),
        "head_path": str(head_path.relative_to(REPO)),
        "run_id": payload["run_id"],
        "before": before,
        "after": after,
        "telemetry": telemetry,
        "sidecar_spend_usd": round(float(sidecar.get("spend_usd") or 0.0), 6),
        "sids_by_pass": sidecar.get("sids_by_pass") or {},
        "payload": payload,
    }


def render_report(results: list[dict], applied: bool) -> dict:
    return {
        "schema": "truthbot-score-propagation v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "applied": applied,
        "generation": PIPELINE_GENERATION,
        "spend_usd": 0.0,
        "speeches": [{k: v for k, v in r.items() if k != "payload"}
                     for r in results],
        "totals": {
            "items": sum(r["telemetry"]["items"] for r in results),
            "matched": sum(r["telemetry"]["matched"] for r in results),
            "sidecar_rows_unmatched": sum(
                len(r["telemetry"]["sidecar_unmatched"]) for r in results),
            "items_not_covered": sum(
                len(r["telemetry"]["artifact_unscored"]) for r in results),
            "fit_after": sum(1 for r in results if r["after"]["fit_to_gate"]),
            "speeches": len(results),
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--speech", action="append", choices=sorted(REBUILT_RUNS),
                    help="limit to this speech (repeatable); default all five")
    ap.add_argument("--apply", action="store_true",
                    help="write the new artifacts and update the manifest "
                         "(default is a dry run that writes nothing)")
    ap.add_argument("--no-b2", action="store_true",
                    help="ignore the B2 sidecars — the B1a-only merge")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    heads = publishing_heads()
    speeches = args.speech or list(REBUILT_RUNS)
    results = []
    for speech in speeches:
        head_path = heads.get(speech)
        if head_path is None:
            print(f"SKIP {speech}: no evidence-bearing artifact")
            continue
        results.append(propagate_speech(speech, head_path,
                                        use_b2=not args.no_b2))

    print(f"\n{'speech':<14}{'items':>7}{'  relevance-scored':>20}"
          f"{'  stance-null':>26}{'  fit':>8}")
    for r in results:
        b, a = r["before"], r["after"]
        print(f"{r['speech']:<14}{b['items']:>7}"
              f"   {b['relevance_rate']:>6.1%} -> {a['relevance_rate']:>6.1%}"
              f"      {b['stance_null_rate']:>6.1%} -> {a['stance_null_rate']:>6.1%}"
              f"   {str(b['fit_to_gate']):>5} -> {str(a['fit_to_gate'])}")
    for r in results:
        t = r["telemetry"]
        print(f"\n{r['speech']}: {t['matched']}/{t['items']} items matched "
              f"({r['head'][:8]} -> {r['run_id'][:8]})")
        if t["sidecar_unmatched"]:
            print(f"  sidecar rows with no home: {len(t['sidecar_unmatched'])}")
            for row in t["sidecar_unmatched"][:10]:
                print(f"    {row['sid']:<20} {row['why']:<20} {row['source_url']}")
            if len(t["sidecar_unmatched"]) > 10:
                print(f"    … {len(t['sidecar_unmatched']) - 10} more")
        if t["artifact_unscored"]:
            print(f"  items no sidecar covers (values kept): "
                  f"{len(t['artifact_unscored'])}")
            for row in t["artifact_unscored"][:10]:
                print(f"    {row['sid']:<20} {row['why']:<20} {row['source_url']}")
            if len(t["artifact_unscored"]) > 10:
                print(f"    … {len(t['artifact_unscored']) - 10} more")
        if not r["after"]["fit_to_gate"]:
            print(f"  STILL UNFIT: {r['after']['reason']}")

    if args.apply:
        for r in results:
            path = write_artifact(r["payload"])
            update_manifest(r["payload"]["run_id"], r["speech"])
            print(f"wrote {path}")
    else:
        print("\nDRY RUN — nothing written. Re-run with --apply.")

    report = render_report(results, applied=bool(args.apply))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{REPORT_STEM}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {out / (REPORT_STEM + '.json')}")

    unfit = [r["speech"] for r in results if not r["after"]["fit_to_gate"]]
    if unfit:
        print("\nNOT FIT TO GATE after an honest merge: " + ", ".join(unfit))
        print("That is a FINDING. The 15% ceiling is not to be moved to "
              "accommodate it.")
    print("\nno model calls were made; $0 spent.")
    return 0


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
