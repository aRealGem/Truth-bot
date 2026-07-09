#!/usr/bin/env python3
"""
Layer B dev-lot — closed-book PCA verdicts over the TRAIN check-worthy subset.

Drives src/truthbot/verdict (adjudicate + normalize) through the pca state machine
on roster.dev (P=mistral, C=dsv4-flash, A=claude-haiku), closed-book (empty evidence
pack ⇒ citations must be [], I4). This is the OPERATIONAL sibling of run_pca_devlot.py
— it exercises the actual Layer B entry point rather than an inline prototype.

IMPORTANT — this is NOT accuracy scoring. claim_set.train.jsonl carries only Layer A
labels (check-worthy/opinion/unimportant); there is no gold verdict to score against.
So this reports the operational shape only: verdict distribution, resolved vs
disagreement vs no_label status split, split/escalation rates, lane tally, model
mismatches, and $/claim. Build a verdict-gold set before spending the read-once
heldout (I6) on a scored pass.

TRAIN only — never reads claim_set.heldout.jsonl (that needs a fresh rc_id under I6).

Env: source the repo .env (LITELLM_PCA_KEY, LITELLM_BASE_URL). No key ⇒ BLOCKED, no
spend. Roster.dev is cheap tiers; pca.yaml's $2.00 ceiling halts a runaway.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))       # sibling helpers
from hydramind import HydraMind
from hydramind.transport import Transport, ProxyCompletion
from hydramind.registry import load_registry
from hydramind.manifest import NullSpendSink
from truthbot.verdict import adjudicator
import proxy_client
import ledger as run_ledger

HERE = Path(__file__).parent
TRAIN = HERE / "claim-set" / "claim_set.train.jsonl"
LEDGER = Path(__file__).resolve().parents[2] / "metrics" / "spend_ledger" / "truthbot.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG)
        return
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]
    claims = [{"sid": r["sid"], "text": r["text"], "context": r.get("context", "")}
              for r in rows if r["label"] == "check-worthy"][:N]

    hm = HydraMind(load_registry(), Transport(
        completion_fn=ProxyCompletion(key_env=proxy_client.resolve_key_env(),
                                      base_url=proxy_client.base_url(),
                                      response_parser=adjudicator.parse_verdict)),
        spend_sink=NullSpendSink(),      # print spend rows; do NOT auto-push to P80
        project=proxy_client.CLIENT)     # truth-bot — the client, not the pca strategy

    verdicts, manifest, notes = adjudicator.adjudicate(hm, claims, roster="dev")

    n = len(verdicts)
    status = Counter(v["status"] for v in verdicts)
    dist = Counter(v["verdict"] for v in verdicts if v["status"] == "resolved")
    # I4: no closed-book verdict may carry a citation (normalize already guards; re-assert)
    leaked = [v["sid"] for v in verdicts if v["citations"]]
    assert not leaked, f"I4 violation — closed-book citations leaked: {leaked}"

    print(f"\n# Layer B dev-lot (roster.dev, closed-book) n={n} check-worthy TRAIN claims")
    print(f"  status          = {dict(status)}")
    print(f"  verdict_dist    = {dict(dist)}  (resolved only)")
    print(f"  split_rate      = {notes['split_rate']:.3f}")
    print(f"  escalation_rate = {notes['escalation_rate']:.3f}  "
          f"(criterion: {notes['split_criterion']})")
    print(f"  disagreement_flagged = {notes['flagged']}")
    print(f"  lane_tally      = {manifest.lane_tally}")
    print(f"  model_mismatches= {manifest.model_mismatches() or '[]'}")
    print(f"  tokens in/out   = {manifest.total_tokens_in}/{manifest.total_tokens_out}")
    print(f"  I4 closed-book  = OK (no citations on any verdict)")
    seats = {}
    for c in manifest.cost_records:
        seats.setdefault(c.role, (c.model, c.returned_model))
    print(f"  seats           = {seats}")
    cost = manifest.total_cost_usd
    print(f"  total spend     = ${cost:.4f}  (${cost / n:.5f}/claim)" if n else "  (no claims)")
    print(f"\n# spend rows (NOT pushed to P80): {json.dumps(manifest.to_spend_records())}")

    out = HERE / "examples"
    out.mkdir(exist_ok=True)
    (out / "manifest.layerb-devlot.json").write_text(manifest.to_json())
    (out / "layerb-devlot-verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print("# manifest → examples/manifest.layerb-devlot.json ; "
          "verdicts → examples/layerb-devlot-verdicts.json")

    rec = run_ledger.append_run(
        LEDGER, manifest, notes=notes,
        config={"key_label": proxy_client.KEY_LABEL, "base_url": proxy_client.base_url()})
    print(f"# ledger ← run {rec['run_id']} (client={rec['client']}, "
          f"${rec['cost']['total_cost_usd']:.4f}, cost_source={rec['cost']['cost_source_tally']})")


if __name__ == "__main__":
    main()
