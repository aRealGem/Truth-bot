#!/usr/bin/env python3
"""
Track B — PCA frontier eval on roster.frontier (P=claude-sonnet, C=claude-sonnet,
A=claude-opus).

Runs the verdict-gold claims (eval/benchmarks/claim-set/verdict_gold.train.jsonl —
all check-worthy, evidence-backed) through the pca state machine CLOSED-BOOK
(empty evidence pack ⇒ citations must be [], I4), then dumps per-item verdicts in
a shape score_verdict.py can consume directly.

Purpose: quantify the decided-accuracy LIFT of an All-Anthropic frontier panel over
roster.dev (mistral/dsv4-flash/haiku) on the same closed-book conditions, before
committing to any full open-book frontier re-publish.

SPENDS REAL PROXY $ — 2x sonnet + 1x opus per claim. Use --limit for a dry run and
read the real cost off `/key/info` (pipeline self-report undercounts ~7-8x).

Env: source the repo .env (LITELLM_PCA_KEY / truth-bot key, LITELLM_BASE_URL).
Run with PYTHONPATH=. (hydramind is a top-level dir, not installed).

Usage:
  PYTHONPATH=. python3 eval/benchmarks/run_pca_frontier.py [--limit N] [--roster NAME] [--out PATH]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))       # sibling helpers
from hydramind import HydraMind
from hydramind.transport import Transport, ProxyCompletion
from hydramind.registry import load_registry
from hydramind.manifest import NullSpendSink
import proxy_client
import ledger as run_ledger

HERE = Path(__file__).parent
GOLD = HERE / "claim-set" / "verdict_gold.train.jsonl"
LEDGER = Path(__file__).resolve().parents[2] / "metrics" / "spend_ledger" / "truthbot.jsonl"

_VERDICTS = "TRUE | FALSE | MISLEADING | UNVERIFIABLE"
_CONTRACT = ('Return JSON only: {"verdict": "%s", "confidence": 0.0-1.0, '
             '"citations": [], "reasoning": "one clause"}. Closed-book: no '
             'external evidence is provided, so cite nothing (citations must be []). '
             'If the claim cannot be adjudicated from general knowledge, verdict=UNVERIFIABLE.'
             % _VERDICTS)
PROMPTS = {
    "proposer": "You are the PROPOSER. Assess the factual claim and draft a verdict. " + _CONTRACT,
    "critic":   "You are the CRITIC. Independently and skeptically assess the same claim; "
                "try to find why a naive verdict could be wrong. " + _CONTRACT,
    "arbiter":  "You are the ARBITER. Adjudicate the claim decisively. " + _CONTRACT,
}


def parse_verdict(raw: dict) -> dict:
    v = (raw.get("verdict") or "").strip().upper()
    if v not in {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}:
        v = "UNVERIFIABLE"
    c = raw.get("confidence")
    try:
        c = float(c)
    except (TypeError, ValueError):
        c = None
    return {"verdict": v, "confidence": c, "citations": list(raw.get("citations", []))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N gold claims (dry-run cost calibration)")
    ap.add_argument("--roster", default="frontier")
    ap.add_argument("--out", default=None,
                    help="verdict artifact path (default examples/pca-frontier-verdicts.json)")
    args = ap.parse_args()

    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return

    rows = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    # verdict-gold rows are all check-worthy by construction; text lives in `claim`.
    items = [{"item_id": r["sid"],
              "payload": {"claim": r["claim"], "context": r.get("context", ""),
                          "evidence_pack_ids": []}} for r in rows]

    print(f"# PCA frontier eval — roster={args.roster}, n={len(items)} verdict-gold claims (closed-book)")

    hm = HydraMind(load_registry(), Transport(
        completion_fn=ProxyCompletion(key_env=proxy_client.resolve_key_env(),
                                      base_url=proxy_client.base_url(),
                                      response_parser=parse_verdict)),
        spend_sink=NullSpendSink(), project=proxy_client.CLIENT)
    result, manifest = hm.run("verdict", items, "pca", roster=args.roster,
                              tune={"prompts": PROMPTS})

    mism = manifest.model_mismatches()
    n = len(result.items)
    resolved = sum(1 for r in result.items if r.kind.value == "resolved")
    print(f"\n# results n={n}")
    print(f"  split_rate      = {result.notes['split_rate']:.3f}")
    print(f"  escalation_rate = {result.notes['escalation_rate']:.3f}  "
          f"(criterion: {result.notes['split_criterion']})")
    print(f"  resolved={resolved}  disagreement_flagged={result.notes['flagged']}")
    print(f"  lane_tally      = {manifest.lane_tally}")
    print(f"  model_mismatches= {mism if mism else '[]'}")
    print(f"  tokens in/out   = {manifest.total_tokens_in}/{manifest.total_tokens_out}")
    seats = {}
    for c in manifest.cost_records:
        seats.setdefault(c.role, (c.model, c.returned_model))
    print(f"  seats           = {seats}")

    # per-item verdicts — `status` (NOT `kind`) so score_verdict.py reads it.
    out = Path(args.out) if args.out else HERE / "examples" / "pca-frontier-verdicts.json"
    out.write_text(json.dumps(
        [{"sid": r.item_id, "status": r.kind.value, **r.value, "agreement": r.agreement}
         for r in result.items], indent=2))
    print(f"# verdicts → {out}")

    rec = run_ledger.append_run(
        LEDGER, manifest, notes=result.notes,
        config={"key_label": proxy_client.KEY_LABEL, "base_url": proxy_client.base_url(),
                "roster": args.roster, "eval": "track-b-frontier"})
    print(f"# ledger ← run {rec['run_id']} (client={rec['client']}, "
          f"${rec['cost']['total_cost_usd']:.4f} SELF-REPORTED [undercounts ~7-8x], "
          f"cost_source={rec['cost']['cost_source_tally']})")
    print("# NOTE: authoritative spend is /key/info delta, not the line above.")


if __name__ == "__main__":
    main()
