#!/usr/bin/env python3
"""Score Layer B (closed-book PCA, roster.dev) against the canonical verdict-gold.

Runs the actual Layer B entry point (truthbot.verdict.adjudicate) over exactly the
verdict_gold.train.jsonl sids — pulled by SID from claim_set.train.jsonl (TRAIN only,
never heldout; I6-safe) — then scores predictions vs gold with the closed-book
abstention semantics in scorer/score_verdict.py (decided-accuracy / coverage /
abstain-gap; UNVERIFIABLE and unresolved are abstentions, not misses).

Env: source the repo .env (LITELLM_TRUTHBOT_KEY, LITELLM_BASE_URL). No key ⇒ BLOCKED.
Roster.dev is cheap tiers; pca.yaml's $2.00 ceiling halts a runaway.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scorer"))
from hydramind import HydraMind
from hydramind.transport import Transport, ProxyCompletion
from hydramind.registry import load_registry
from hydramind.manifest import NullSpendSink
from truthbot.verdict import adjudicator
import proxy_client
import score_verdict as sv

TRAIN = HERE / "claim-set" / "claim_set.train.jsonl"
GOLD = HERE / "claim-set" / "verdict_gold.train.jsonl"


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return
    gold = {json.loads(l)["sid"]: json.loads(l)["gold_verdict"]
            for l in GOLD.read_text().splitlines() if l.strip()}
    train = {json.loads(l)["sid"]: json.loads(l)
             for l in TRAIN.read_text().splitlines() if l.strip()}
    missing = [s for s in gold if s not in train]     # heldout/other → must NOT be run
    if missing:
        print(f"WARN {len(missing)} gold sids not in TRAIN (skipped, I6-safe): {missing}")
    sids = [s for s in gold if s in train]
    claims = [{"sid": s, "text": train[s]["text"], "context": train[s].get("context", "")}
              for s in sids]
    print(f"scoring Layer B over {len(claims)} gold claims (roster.dev, closed-book)")

    hm = HydraMind(load_registry(), Transport(
        completion_fn=ProxyCompletion(key_env=proxy_client.resolve_key_env(),
                                      base_url=proxy_client.base_url(),
                                      response_parser=adjudicator.parse_verdict)),
        spend_sink=NullSpendSink(), project=proxy_client.CLIENT)
    verdicts, manifest, notes = adjudicator.adjudicate(hm, claims, roster="dev")
    leaked = [v["sid"] for v in verdicts if v["citations"]]
    assert not leaked, f"I4 violation — closed-book citations leaked: {leaked}"

    preds = {v["sid"]: v for v in verdicts}
    rep = sv.score_verdicts({s: gold[s] for s in sids}, preds)

    (HERE / "examples").mkdir(exist_ok=True)
    (HERE / "examples" / "layerb-vs-gold-verdicts.json").write_text(json.dumps(verdicts, indent=2))

    cost = manifest.total_cost_usd
    print(f"\n# Layer B vs verdict-gold (n={rep['n']})")
    print(f"  decided-accuracy = {rep['decided_accuracy']}  (hit {rep['hit']} / decided {rep['decided']})")
    print(f"  coverage         = {rep['coverage']:.3f}  (committed / n)")
    print(f"  abstain_gap      = {rep['abstain_gap']}  (decidable gold, model abstained → Layer C)")
    print(f"  abstain_ok       = {rep['abstain_ok']}  (gold UNVERIFIABLE, model rightly abstained)")
    print(f"  status split     = {dict(Counter(v['status'] for v in verdicts))}")
    print(f"  gold dist        = {dict(Counter(gold[s] for s in sids))}")
    print("  confusion (gold → pred):")
    for g, row in rep["confusion"].items():
        nz = {k: v for k, v in row.items() if v}
        if nz:
            print(f"    {g:12} {nz}")
    print(f"  spend            = ${cost:.4f} ({cost/len(claims):.5f}/claim)" if claims else "")
    print("# verdicts → examples/layerb-vs-gold-verdicts.json")


if __name__ == "__main__":
    main()
