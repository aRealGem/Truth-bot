"""Reference-set regression scorecard for a published truth-bot run.

Joins:
  - metrics/extractions/<run_id>.jsonl    (extracted claim list)
  - metrics/batch_sidecar/<run_id>.jsonl  (per-model verdicts: openai, gemini, xai)
  - site-test/data/claims.json filtered by --report-id (consensus verdicts)

then calls eval.evolver.fitness.FitnessScorer to score against
eval/sotu-2026/reference.json (the GPT 5.4 Pro 29-claim reference set).

Caveat: explanation_quality and source_citation_quality are computed from
sidecar adapter explanations only (3 of 4 voters). Anthropic frontier
explanations live in the rendered claim HTML pages and are not currently
persisted as a single canonical JSON. claim_recall and verdict_agreement
are unaffected (consensus_verdict reflects all four adapters).

Usage:
    python eval/sotu-2026/score_run.py \
        --run-id 258b5758-8e25-4bf0-8f34-63778d2f976e \
        --report-id e81546a0-6371-4e96-9e94-3d6213864d5a
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.evolver.fitness import FitnessScorer  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_explanation_index(sidecar_rows: list[dict]) -> dict[str, str]:
    """Concat sidecar explanations per claim_id (one string per claim).

    Multiple model verdicts per claim get joined with ' | ' so the FitnessScorer
    keyword-counting heuristics see all available evidence.
    """
    by_claim: dict[str, list[str]] = defaultdict(list)
    for row in sidecar_rows:
        cid = row.get("claim_id")
        expl = (row.get("explanation") or "").strip()
        if cid and expl:
            by_claim[cid].append(expl)
    return {cid: " | ".join(parts) for cid, parts in by_claim.items()}


def _filter_claims_by_report(claims_all: list[dict], report_id: str) -> list[dict]:
    return [c for c in claims_all if c.get("report_id") == report_id]


def _token_count_from_sidecar(sidecar_rows: list[dict]) -> int:
    total = 0
    for row in sidecar_rows:
        total += int(row.get("input_tokens") or 0)
        total += int(row.get("output_tokens") or 0)
    return total


def _format_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run UUID")
    parser.add_argument("--report-id", required=True, help="Published report UUID (in site-data claims.json)")
    parser.add_argument(
        "--metrics-dir",
        default=str(_REPO_ROOT / "metrics"),
        help="Path to the metrics directory (default: %(default)s)",
    )
    parser.add_argument(
        "--site-data-dir",
        default=str(_REPO_ROOT / "site-test" / "data"),
        help="Path to the published site data dir containing claims.json (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-fitness",
        type=float,
        default=0.679,
        help="Reference fitness baseline for delta reporting (default: %(default)s, claude-opus-4-7 2026-04-19)",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    site_data_dir = Path(args.site_data_dir)

    extractions_path = metrics_dir / "extractions" / f"{args.run_id}.jsonl"
    sidecar_path = metrics_dir / "batch_sidecar" / f"{args.run_id}.jsonl"
    run_summary_path = metrics_dir / "run_summaries" / f"{args.run_id}.json"
    claims_path = site_data_dir / "claims.json"
    reports_path = site_data_dir / "reports.json"

    for p in (extractions_path, sidecar_path, claims_path, reports_path):
        if not p.exists():
            print(f"ERROR: required input not found: {p}", file=sys.stderr)
            return 2

    extractions = _load_jsonl(extractions_path)
    sidecar = _load_jsonl(sidecar_path)
    all_claims = _load_json(claims_path)
    all_reports = _load_json(reports_path)
    run_summary = _load_json(run_summary_path) if run_summary_path.exists() else {}

    report_meta = next((r for r in all_reports if r.get("id") == args.report_id), None)
    if report_meta is None:
        print(f"ERROR: report-id {args.report_id} not found in {reports_path}", file=sys.stderr)
        return 2

    report_claims = _filter_claims_by_report(all_claims, args.report_id)
    if not report_claims:
        print(f"ERROR: no claims found for report-id {args.report_id}", file=sys.stderr)
        return 2

    expl_by_claim = _build_explanation_index(sidecar)

    verdicts: list[dict] = []
    sidecar_covered = 0
    for c in report_claims:
        cid = c.get("id", "")
        explanation = expl_by_claim.get(cid, "")
        if explanation:
            sidecar_covered += 1
        verdicts.append(
            {
                "claim_text": c.get("claim_text", ""),
                "label": c.get("consensus_verdict", ""),
                "explanation": explanation,
            }
        )

    extracted_for_recall = [
        {"text": e.get("text", ""), "is_checkable": e.get("is_checkable", True)}
        for e in extractions
    ]

    token_count = _token_count_from_sidecar(sidecar)

    scorer = FitnessScorer()
    scores = scorer.score(
        extracted_claims=extracted_for_recall,
        verdicts=verdicts,
        token_count=token_count,
    )

    cost_usd = float(run_summary.get("total_cost_usd", 0.0))

    fitness = scores["fitness"]
    delta = fitness - args.baseline_fitness

    print()
    print(f"Reference-set regression scorecard for run {args.run_id}")
    print("=" * 72)
    print(f"Report:           {report_meta.get('date')} {report_meta.get('speaker')} — {report_meta.get('venue')}")
    print(f"Report ID:        {args.report_id}")
    print(f"Reference set:    eval/sotu-2026/reference.json (29 claims, GPT 5.4 Pro)")
    print()
    print("Inputs:")
    print(f"  Extracted claims:       {len(extracted_for_recall)} (all checkable: {sum(1 for e in extracted_for_recall if e['is_checkable'])})")
    print(f"  Published verdicts:     {len(verdicts)} (consensus from claims.json)")
    print(f"  Sidecar entries:        {len(sidecar)} (OpenAI/Gemini/xAI)")
    print(f"  Sidecar coverage:       {sidecar_covered}/{len(verdicts)} claims have ≥1 sidecar explanation")
    print(f"  Anthropic explanations: excluded (lives in claim HTML — see TODO)")
    print(f"  Token count (sidecar):  {token_count:,}")
    print()
    print("Scores (FitnessScorer, 5-dimension):")
    print(f"  Claim recall:           {_format_pct(scores['claim_recall'])}   weight 0.25  ({scores['matched_count']}/29 reference claims matched)")
    print(f"  Verdict agreement:      {_format_pct(scores['verdict_agreement'])}   weight 0.30")
    print(f"  Explanation quality:    {_format_pct(scores['explanation_quality'])}   weight 0.20  (sidecar-only; Anthropic excluded)")
    print(f"  Source citation:        {_format_pct(scores['source_citation_quality'])}   weight 0.15  (sidecar-only; Anthropic excluded)")
    parsimony_note = (
        "  (target_max=30k calibrated for single-model standalone; not meaningful"
        " for 4-adapter consensus runs — see TODO)"
    )
    print(f"  Parsimony:              {_format_pct(scores['parsimony'])}   weight 0.10  ({token_count:,} tokens, sidecar-only)")
    print(parsimony_note)
    print( "  ──────────────────────────────────────")
    print(f"  Fitness:                {fitness:.4f}")
    print()
    print(f"Vs baseline (best known: {args.baseline_fitness:.3f}, claude-opus-4-7 standalone, 2026-04-19):")
    sign = "+" if delta >= 0 else ""
    print(f"  Fitness delta:          {sign}{delta:.4f}")
    print(f"  Cost:                   ${cost_usd:.2f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
