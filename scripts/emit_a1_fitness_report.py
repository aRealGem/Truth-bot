"""A1 fitness report: is any stored run's evidence scored well enough to GATE?

Emits ``metrics/remediation_v2/a1_fitness_report.json`` ($0, read-only — it
recomputes scoring telemetry from artifacts already on disk and calls no model).

The artifact used to be assembled by hand, which is exactly how a headline
number loses its denominator: the finding says "every stored run is unfit to
gate" over SEVENTEEN artifacts, on a site that publishes FIVE reports. This
script is the file's owner, so the finding text
(``consistency.A1_FINDING``), the cohort split (``consistency.run_cohort``) and
the per-run rows (``consistency.run_fitness_report``) all come from the module
that computes them, and the composition travels with the number.

Usage (repo root)::

    .venv/bin/python scripts/emit_a1_fitness_report.py [--out PATH] [--check]

``--check`` re-emits into memory and diffs against the committed file without
writing, so CI or a reviewer can prove the artifact still matches the code.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))          # hydramind lives at the repo root

from truthbot.publish.consistency import (  # noqa: E402
    RUN_COHORT_ORDER, UNFIT_STANCE_NULL_RATE, fitness_finding,
    run_fitness_report)

OUT = ROOT / "metrics" / "remediation_v2" / "a1_fitness_report.json"

GENERATOR = (
    "scripts/emit_a1_fitness_report.py — truthbot.publish.consistency."
    "run_fitness_report(repo_root) recomputes truthbot.verdict.consolidator."
    "scoring_telemetry_from_artifact over each artifact's stored evidence "
    "dict, then is_fit_to_gate(); cohorts via consistency.run_cohort, finding "
    "text from consistency.A1_FINDING")

THRESHOLD_NOTE = (
    "A run is unfit-to-gate when (a) NO item carries a real relevance score "
    "(all still on the 0.5 pydantic default) or (b) the stance-null rate "
    "exceeds the ceiling. Condition (a) alone condemns every run below.")


def build_report(repo_root=ROOT, generated: str | None = None) -> dict:
    # Manifest order is arbitrary; group by cohort (then speech, then run id)
    # so the artifact reads as the composition it now states.
    rows = sorted(run_fitness_report(repo_root),
                  key=lambda r: (RUN_COHORT_ORDER.index(r["cohort"]),
                                 r["speech_id"], r["run_id"]))
    cohorts: dict[str, dict] = {}
    for name in RUN_COHORT_ORDER:
        members = [r for r in rows if r["cohort"] == name]
        if not members:
            continue
        cohorts[name] = {
            "runs": len(members),
            "items": sum(r["items"] for r in members),
            "relevance_scored": sum(r["relevance_scored"] for r in members),
            "stance_null": sum(r["stance_null"] for r in members),
            "fit_to_gate": sum(1 for r in members if r["fit_to_gate"]),
        }
    return {
        "schema": "truthbot-a1-fitness-report v1",
        "generated": generated or date.today().isoformat(),
        "generator": GENERATOR,
        "threshold": {
            "unfit_stance_null_rate": UNFIT_STANCE_NULL_RATE,
            "note": THRESHOLD_NOTE,
        },
        "finding": fitness_finding(rows),
        "cohorts": cohorts,
        "runs": rows,
    }


def _serialize(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--generated", default=None,
                    help="pin the generated date (default: today) so a "
                         "re-emit for a text-only change is a clean diff")
    ap.add_argument("--check", action="store_true",
                    help="compare against the file on disk; do not write")
    args = ap.parse_args()

    out = Path(args.out)
    generated = args.generated
    if generated is None and out.exists():
        # Preserve the recorded date unless explicitly re-dated: the report's
        # subject is the stored corpus, not the day someone re-ran the emitter.
        generated = json.loads(out.read_text(encoding="utf-8")).get("generated")
    doc = build_report(generated=generated)
    text = _serialize(doc)

    if args.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print(f"{out}: {'MATCHES the generator' if same else 'DIFFERS — re-emit'}")
        raise SystemExit(0 if same else 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n_unfit = sum(1 for r in doc["runs"] if not r["fit_to_gate"])
    print(f"A1 fitness report → {out}")
    print(f"  {n_unfit}/{len(doc['runs'])} unfit to gate · "
          + " + ".join(f"{v['runs']} {k}" for k, v in doc["cohorts"].items()))


if __name__ == "__main__":
    main()
