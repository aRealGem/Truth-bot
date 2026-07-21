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
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from truthbot.models import SourceTier
from truthbot.publish.site import SitePublisher, SiteReport
from truthbot.verdict import bridge as bridge_mod
from truthbot.verdict.evidence_pack import EvidencePack, PackItem, _sha256


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
        ))
    return EvidencePack(sid=sid, window=None, items=items)


def render_artifact(path: Path, publisher: SitePublisher, role: str) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    meta = d["meta"]
    rows, claims = d["rows"], d["claims"]
    packs = {sid: pack_from_evidence(sid, evs)
             for sid, evs in (d.get("evidence") or {}).items()}

    out = bridge_mod.bridge(rows, claims, packs)

    date_val = None
    if meta.get("date"):
        try:
            date_val = datetime.strptime(meta["date"], "%Y-%m-%d")
        except ValueError:
            pass
    site_report = SiteReport(
        report_id=str(uuid.uuid4()),
        speaker=meta.get("speaker", ""),
        role=role,
        date=date_val,
        venue=meta.get("venue", ""),
        transcript_source_url="",
        bundles=out.bundles,
        characterization=list(d.get("characterization") or []),
        panel_roster=dict(d.get("roster") or {}),
    )
    report_path = publisher.publish(site_report)
    print(f"{meta.get('speech_id')}: {len(out.bundles)} bundles → {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", nargs="*",
                    help="pca_runs artifact paths (default: latest evidence-bearing artifact per speech)")
    ap.add_argument("--site-root", required=True)
    ap.add_argument("--role", default="President")
    args = ap.parse_args()

    paths = [Path(p) for p in args.artifacts]
    if not paths:
        candidates = sorted((REPO / "metrics" / "pca_runs").glob("*.json"),
                            key=lambda p: p.stat().st_mtime)
        # LATEST artifact per speech_id — a superseded run of the same speech
        # must not resurrect its stale report alongside the current one (this
        # bit as soon as a re-publish left two evidence-bearing artifacts per
        # speech, 2026-07-20). Pass paths explicitly to render an older run.
        latest: dict[str, Path] = {}
        for p in candidates:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "evidence" not in d:
                continue
            sid = (d.get("meta") or {}).get("speech_id") or p.stem
            latest[sid] = p          # candidates are mtime-ascending; last wins
        paths = list(latest.values())
        if not paths:
            sys.exit("no artifacts with persisted evidence found under metrics/pca_runs/")
        print(f"rendering {len(paths)} artifact(s): {', '.join(p.stem[:8] for p in paths)}")

    publisher = SitePublisher(site_root=args.site_root)
    for p in paths:
        render_artifact(p, publisher, args.role)
    stats = publisher.summary()
    print(f"site: {stats['root']} — {stats['reports']} report(s), "
          f"{stats['claims']} claim(s), {stats['total_kb']} KB")


if __name__ == "__main__":
    main()
