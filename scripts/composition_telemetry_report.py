#!/usr/bin/env python3
"""Print composition-bias telemetry for stored PCA run artifacts ($0, offline).

Live runs record this block into their own artifact automatically (see
``pipeline._persist_pca_artifact``). This CLI is for reading runs that predate
the telemetry — or for asking the retrospective question:

  --reclassify   re-derive tiers from each URL under *current* rules, instead of
                 the tiers stored at run time. Required to read pre-PR-A
                 artifacts, whose stored tiers predate the S5 tier entirely.

Usage:
  python scripts/composition_telemetry_report.py [artifact-id|path ...] [--reclassify]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))          # hydramind lives at the repo root

from truthbot.verdict.composition_telemetry import composition_report, format_report

RUNS = REPO / "metrics" / "pca_runs"
DEFAULTS = ("23939712-59ea-449d-93f7-a0a0b449efd8", "7208bbbb-c802-4155-932f-d0cc66803b24")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", nargs="*", default=list(DEFAULTS))
    ap.add_argument("--reclassify", action="store_true",
                    help="re-derive tiers from URLs under current rules")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    tier_fn = None
    if args.reclassify:
        from truthbot.verify.source_tiers import classify_tier
        tier_fn = classify_tier

    out = {}
    for a in args.artifacts:
        p = Path(a) if Path(a).exists() else RUNS / f"{a}.json"
        if not p.exists():
            print(f"! missing artifact: {p}", file=sys.stderr)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rep = composition_report(d.get("rows") or [], d.get("evidence") or {},
                                 tier_fn=tier_fn)
        name = (d.get("meta") or {}).get("speech_id") or p.stem
        out[name] = rep
        mode = "current rules (reclassified)" if args.reclassify else "tiers as stored at run time"
        print(f"\n### {name}  [{mode}]")
        print(format_report(rep))

    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
