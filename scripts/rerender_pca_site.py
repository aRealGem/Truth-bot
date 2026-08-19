#!/usr/bin/env python3
"""Re-render the PCA site OFFLINE from metrics/pca_runs replay artifacts — $0.

This is the consumer `_persist_pca_run` always promised: the artifact holds
{meta, claims, rows, characterization, roster, evidence}, which is everything
the bridge + publisher need. A live PCA run is ~30-60 min of proxy spend; this
script re-renders both SOTU reports in seconds with no LLM calls — so render
fixes (provenance display, source collapse, copy changes) ship without a re-run.

Evidence packs are reconstructed from the persisted per-sid Evidence dumps in
ORDER, so pack ids (E1..En) — and therefore the rows' citation references —
resolve exactly as they did in the live run.

Usage (repo root):
  PYTHONPATH=. .venv/bin/python scripts/rerender_pca_site.py \
      --site-root /tmp/site-out [--role President] [artifact.json ...]

With no artifact paths, renders every artifact under metrics/pca_runs/ that
carries an `evidence` key (i.e. post-2026-07-19 runs), oldest first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from truthbot.models import SourceTier
from truthbot.publish.corrections import (apply_to_artifact, load_corrections,
                                          load_notes, load_resolution_changes)
from truthbot.publish.site import SitePublisher, SiteReport
from truthbot.verdict import bridge as bridge_mod
from truthbot.verdict.evidence_pack import EvidencePack, PackItem, _sha256


PCA_RUNS_DIR = REPO / "metrics" / "pca_runs"


# F4: head resolution is single-sourced in ``truthbot.publish.heads`` so the
# renderer, the score-propagation merge, the DC-6 packager, the audits and the
# tests cannot disagree about which run is the head — and it is deterministic on
# a fresh clone (the rebuild_of DAG leaf, not the newest mtime). Re-exported
# here for back-compat with ``from rerender_pca_site import publishing_heads``.
from truthbot.publish.heads import publishing_heads  # noqa: E402,F401


def pack_from_evidence(sid: str, evs: list[dict]) -> EvidencePack:
    """Rebuild an EvidencePack from the artifact's per-sid Evidence dumps.

    Order is the original pack order (bridge serialized it via _pack_to_evidence),
    so enumerated E<n> ids reproduce the live pack exactly."""
    items = []
    for i, ev in enumerate(evs, start=1):
        url = (ev.get("source_url") or "").strip()
        snippet = ev.get("snippet") or ""
        try:
            tier = SourceTier(ev.get("source_tier"))
        except ValueError:
            tier = SourceTier.OTHER
        items.append(PackItem(
            pack_id=f"E{i}",
            source_name=ev.get("source_name") or "Unknown",
            source_url=url,
            tier=tier,
            snippet=snippet,
            retrieved_at=str(ev.get("retrieved_at") or ""),
            sha256=_sha256(url, snippet),
            # Round B.5 stance signals; older artifacts predate them → None.
            supports_claim=ev.get("supports_claim"),
            relevance_score=ev.get("relevance_score"),
            # P67.5: publication date round-trips through artifacts now;
            # pre-fix artifacts carry null here (date lives in the snippet).
            published_at=(str(ev.get("published_at"))[:10]
                          if ev.get("published_at") else None),
            # D17-c: the series excerpt round-trips through the artifact so the
            # rendered page can show the observations the panel reasoned over.
            # Dropped here, the rows reach the stored evidence and then vanish
            # at render — which is exactly what they did until this line.
            series_rows=ev.get("series_rows"),
        ))
    return EvidencePack(sid=sid, window=None, items=items)


def stance_coverage(evidence: dict) -> dict:
    """Per-speech stance-scored coverage for the D-B disclosure block — how much
    of the evidence the scorer took a stance on, plus the tier breakdown of what
    it did not. $0, read from the artifact's own scored evidence.

    stance-null = an item the B1a/B2 scorer returned with no support/refute
    signal. Those items cannot credit the evidence quota, so a high null rate is
    disclosed rather than hidden: the block states it against the 15% ceiling and
    (for the one speech over it) decomposes the nulls by source tier."""
    from collections import Counter

    from truthbot.verify.statistical_agency import is_statistical_agency

    tiers: Counter = Counter()
    items = null = packs_with_null = total_packs = 0
    # The null population, decomposed by what would actually move it:
    #   series  — statistical series the pipeline retrieved but scored without
    #             fetching the data table (the known retrieval gap; fixable)
    #   record  — Government-tier records (statutes, transcripts, budget docs)
    #             that state facts without taking a side (stance-free by nature)
    #   other   — everything else, largely claims no evidence can settle
    series = record = other = 0
    for _sid, pack in sorted((evidence or {}).items()):  # A2: stable key order
        total_packs += 1
        has_null = False
        for e in pack or []:
            items += 1
            if e.get("supports_claim") is None:
                null += 1
                has_null = True
                tier = str(e.get("source_tier"))
                tiers[tier] += 1
                try:
                    is_series = bool(is_statistical_agency(e.get("source_url") or ""))
                except Exception:
                    is_series = False
                if is_series:
                    series += 1
                elif tier == "Government":
                    record += 1
                else:
                    other += 1
        if has_null:
            packs_with_null += 1
    rate = (null / items) if items else 0.0
    # The floor the retrieval fix could reach: even converting EVERY series item
    # leaves this rate. Rendered so the disclosure cannot promise more than the
    # measurement supports.
    best_case = ((null - series) / items) if items else 0.0
    return {
        "stance_null": null, "items": items,
        "rate_pct": round(rate * 100, 1), "ceiling_pct": 15.0,
        "over_ceiling": rate * 100 > 15.0,
        "tier_breakdown": dict(tiers.most_common()),
        "packs_with_null": packs_with_null, "total_packs": total_packs,
        "null_series": series, "null_record": record, "null_other": other,
        "best_case_pct": round(best_case * 100, 1),
        "best_case_clears": best_case * 100 <= 15.0,
    }


def render_artifact(path: Path, publisher: SitePublisher, role: str,
                    corrections: list[dict] | None = None,
                    require_fit: bool = True, mode: str = "skip",
                    resolution: list[dict] | None = None) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    meta = d["meta"]
    label = str(meta.get("speech_id") or path.stem)
    # Phase A (A1) HARD publish gate: a run whose evidence was never scored must
    # not be published. Its Unverifiables come from the T2.4 quota gate finding no
    # stance-bearing Tier-1..3 item — retrieval silence, not evidence. F13: a
    # speech with an owner-ratified, registry-keyed stance-null exception (D-B)
    # publishes anyway, with the exception disclosed; every other unfit speech
    # still refuses on a real publish. ``require_fit=False`` (--allow-unfit-gate)
    # is a STAGED REVIEW escape only and waives nothing on a publish.
    from truthbot.publish.consistency import (check_publish_gate,
                                              publish_gate_notice)
    gate = check_publish_gate(d, label=label)
    if gate:
        if require_fit:
            raise SystemExit("PUBLISH GATE FAILED — " + "; ".join(gate))
        print("  ! staged review render of an UNFIT-TO-GATE run: "
              + "; ".join(gate))
    notice = publish_gate_notice(d, label=label)
    if notice:
        print("  * " + notice)
    if corrections and mode == "apply":
        # Historical replay against PRE-ruling artifacts: rewrite the verdict and
        # stamp the note. Fails closed on an old-verdict mismatch.
        n = apply_to_artifact(d, corrections)
        if n:
            print(f"{meta.get('speech_id')}: applied {n} correction(s)")
    elif corrections and mode == "skip":
        # F12: the staged head already carries the verdict, so annotate the strip
        # (old→new + date) without rewriting — the note the corrections page
        # promises, joined by sid. Includes the F9 resolution-state changes.
        from truthbot.publish.corrections import annotate_to_artifact
        n = annotate_to_artifact(d, corrections, resolution)
        if n:
            print(f"{meta.get('speech_id')}: annotated {n} correction note(s)")
    rows, claims = d["rows"], d["claims"]
    packs = {sid: pack_from_evidence(sid, evs)
             for sid, evs in sorted((d.get("evidence") or {}).items())}  # A2: stable key order

    out = bridge_mod.bridge(rows, claims, packs)

    date_val = None
    if meta.get("date"):
        try:
            date_val = datetime.strptime(meta["date"], "%Y-%m-%d")
        except ValueError:
            pass
    # A2 (Wave A): content-derived report_id so re-renders of the same head land
    # on the same id (was uuid4() -> rotated every render, churning reports.json /
    # claims.json). sha256 over the canonical render inputs, formatted to the
    # existing UUID-string shape. speech_id keys the slug already; this stabilizes
    # the id fields the slug does not cover.
    _canonical = json.dumps({
        "speech_id": str(meta.get("speech_id") or ""),
        "speaker": meta.get("speaker", ""),
        "role": role,
        "date": meta.get("date", ""),
        "venue": meta.get("venue", ""),
    }, sort_keys=True).encode("utf-8")
    _report_id = str(uuid.UUID(hex=hashlib.sha256(_canonical).hexdigest()[:32]))
    site_report = SiteReport(
        report_id=_report_id,
        speaker=meta.get("speaker", ""),
        role=role,
        date=date_val,
        venue=meta.get("venue", ""),
        transcript_source_url="",
        bundles=out.bundles,
        characterization=list(d.get("characterization") or []),
        panel_roster=dict(d.get("roster") or {}),
        speech_id=str(meta.get("speech_id") or ""),
        stance_coverage=stance_coverage(d.get("evidence") or {}),
    )
    report_path = publisher.publish(site_report)
    print(f"{meta.get('speech_id')}: {len(out.bundles)} bundles → {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", nargs="*",
                    help="pca_runs artifact paths (default: latest evidence-bearing artifact per speech)")
    ap.add_argument("--site-root", required=True)
    ap.add_argument("--role", default="President")
    ap.add_argument("--corrections", choices=("apply", "skip"), default="skip",
                    help=(
                        "'skip' (default, F12): the staged heads already carry "
                        "their re-adjudicated verdicts, so the verdict is NOT "
                        "rewritten; instead each corrected claim's provenance "
                        "strip is ANNOTATED with the old->new + date note by "
                        "joining data/corrections.json (and the resolution-state "
                        "changes) by sid. 'apply' is reserved for HISTORICAL "
                        "REPLAY against PRE-ruling artifacts: it rewrites the "
                        "verdict and fails closed on an old_verdict mismatch."))
    ap.add_argument("--allow-unfit-gate", action="store_true",
                    help=(
                        "render runs that fail the Phase-A fitness gate. REVIEW "
                        "ONLY — it waives NOTHING on a publish: an owner-ratified "
                        "stance-null exception (D-B) is data in the gate, not "
                        "this flag, so a real publish never needs it and output "
                        "produced with it must not be published."))
    args = ap.parse_args()

    paths = [Path(p) for p in args.artifacts]
    if not paths:
        # LATEST artifact per speech_id. Pass paths explicitly to render an
        # older run.
        paths = [p for _sid, p in sorted(publishing_heads().items())]  # A2: stable speech order
        if not paths:
            sys.exit("no artifacts with persisted evidence found under metrics/pca_runs/")
        print(f"rendering {len(paths)} artifact(s): {', '.join(p.stem[:8] for p in paths)}")

    corrections = load_corrections(REPO / "data" / "corrections.json")
    resolution = load_resolution_changes(REPO / "data" / "corrections.json")
    if corrections:
        print(f"corrections on file: {len(corrections)} entries + "
              f"{len(resolution)} resolution-state changes"
              + (" (skip: annotate strips, do not rewrite verdicts)"
                 if args.corrections == "skip" else " (apply: historical replay)"))

    # F6: corpus-wide corrections-ledger completeness — publish-blocking. On a
    # full head render (no explicit artifact paths, i.e. a publish) the DC-6' net
    # ledger must account for every changed verdict and be exactly the set the
    # changelog publishes, or this is not a publish. A silent drop here would
    # understate what changed, which is the one thing a corrections record may
    # not do.
    if not args.artifacts:
        from truthbot.publish.consistency import check_ledger_completeness
        net_path = REPO / "metrics" / "remediation_v2" / "dc6_net_ledger.json"
        if net_path.exists():
            net = json.loads(net_path.read_text(encoding="utf-8"))
            gaps = check_ledger_completeness(net, corrections, resolution)
            if gaps:
                raise SystemExit("LEDGER COMPLETENESS GATE FAILED — "
                                 + "; ".join(gaps))
            print(f"ledger completeness: {net['ledger_eligible']} entries, "
                  f"all {net['changed_total']} changed sids accounted for")
        else:
            raise SystemExit(
                "LEDGER COMPLETENESS GATE FAILED — no dc6_net_ledger.json; run "
                "scripts/build_net_ledger.py --write before publishing")
    # The corrections PAGE always renders the full ledger + the resolution-state
    # changes (F9). --corrections governs only whether each claim's strip is
    # annotated in place (skip, default) or the verdict is rewritten (apply,
    # historical replay). Both modes feed render_artifact the ledger; the mode
    # selects the behaviour.
    # A3 (Wave A): the reason-coded render set, built from the fail-closed
    # registries (data/decidability.json + data/reason_codes.json). Keys come
    # only from the recorded axis; empty registries -> empty map -> the
    # species does not render.
    from truthbot.publish.site import build_reason_pills
    reason_pills = build_reason_pills(REPO)
    if reason_pills:
        print(f"reason-coded render set: {len(reason_pills)} claim(s)")
    publisher = SitePublisher(
        site_root=args.site_root, corrections=corrections,
        correction_notes=load_notes(REPO / "data" / "corrections.json"),
        resolution_changes=resolution, reason_pills=reason_pills)
    for p in paths:
        render_artifact(p, publisher, args.role, corrections=corrections,
                        require_fit=not args.allow_unfit_gate,
                        mode=args.corrections, resolution=resolution)
    stats = publisher.summary()
    print(f"site: {stats['root']} — {stats['reports']} report(s), "
          f"{stats['claims']} claim(s), {stats['total_kb']} KB")

    # Build-time figure verification (remediation T0.8): every quantitative
    # figure in site copy must derive from data/*.json. A violation fails
    # the render — hand-typed numbers don't ship.
    from truthbot.publish.consistency import check_site
    violations = check_site(Path(args.site_root))
    if violations:
        print(f"\nCONSISTENCY CHECK FAILED — {len(violations)} violation(s):")
        for v in violations:
            print(f"  · {v}")
        sys.exit(1)
    print("consistency check: all rendered figures derive from data/*.json")


if __name__ == "__main__":
    main()
