#!/usr/bin/env python3
"""Era-lint report over persisted pca_runs artifacts — $0, read-only.

Runs the P67.5 era gates (pack-date lint + rationale lint) over historical
run artifacts and prints the re-run routing list: claims whose packs carry
items observed after the speaker's fair-game window (utterance + 7 days),
or whose shipped rationale cites post-window dates as world-state.

Historical artifacts predate the gate, so this is ADVISORY by default —
the report feeds the Phase 1 re-run queue. ``--strict`` exits 1 on any
finding (the mode CI uses once the corrected artifacts land).

Usage (repo root):
  PYTHONPATH=. .venv/bin/python scripts/era_lint_report.py [--strict] [artifact.json ...]

With no paths, lints the latest evidence-bearing artifact per speech.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from truthbot.verdict.era_lint import lint_artifact


def latest_artifacts() -> list[Path]:
    candidates = sorted((REPO / "metrics" / "pca_runs").glob("*.json"),
                        key=lambda p: p.stat().st_mtime)
    latest: dict[str, Path] = {}
    for p in candidates:
        d = json.loads(p.read_text(encoding="utf-8"))
        if "evidence" not in d:
            continue
        sid = (d.get("meta") or {}).get("speech_id") or p.stem
        latest[sid] = p
    return list(latest.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", nargs="*")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any violation/flag (CI mode)")
    ap.add_argument("--max-detail", type=int, default=12,
                    help="per-section detail lines to print")
    args = ap.parse_args()

    paths = [Path(p) for p in args.artifacts] or latest_artifacts()
    if not paths:
        sys.exit("no evidence-bearing artifacts found")

    any_findings = False
    for p in paths:
        artifact = json.loads(p.read_text(encoding="utf-8"))
        report = lint_artifact(artifact)
        print(f"\n=== {report.speech_id or p.stem} (utterance {report.utterance}) — {p.name}")
        print(f"pack items: {report.dated_items} dated / {report.undated_items} undated")
        print(f"pack violations: {len(report.pack_violations)} "
              f"(claims: {len({v.sid for v in report.pack_violations})})")
        for v in report.pack_violations[:args.max_detail]:
            print(f"  · {v.sid} {v.pack_id}: {v.message}")
        if len(report.pack_violations) > args.max_detail:
            print(f"  … +{len(report.pack_violations) - args.max_detail} more")
        print(f"rationale flags: {len(report.rationale_flags)} "
              f"(claims: {len({f.sid for f in report.rationale_flags})})")
        for f in report.rationale_flags[:args.max_detail]:
            print(f"  · {f.sid}: {f.message}")
            print(f"      …{f.excerpt}…")
        if len(report.rationale_flags) > args.max_detail:
            print(f"  … +{len(report.rationale_flags) - args.max_detail} more")
        print(f"RE-RUN QUEUE ({len(report.rerun_sids)} claims): "
              + ", ".join(report.rerun_sids))
        any_findings = any_findings or bool(report.rerun_sids)

    if args.strict and any_findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
