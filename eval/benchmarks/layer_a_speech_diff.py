#!/usr/bin/env python3
"""Step 1 validation: run ONE real speech through Layer A with confirm_pass=False vs True
and diff the check-worthy queue.

confirm_pass only changes the A1-PASS band (False: PASS -> queue unchecked; True: PASS -> A2,
which can veto). AMBIGUOUS handling is identical either way, so Q_true is a subset of Q_false
and the delta is exactly the A1 lexical false positives A2 caught. A single per-sid memoized
A2 pass (haiku-v2 by default) feeds both run_layer_a calls, so we pay once and stay consistent.

Cross-references checkworthy_gold.jsonl (150-row adjudicated answer key) to quantify the
trade: how much junk did the veto drop, and did it cost any true check-worthy (recall)?

Usage: layer_a_speech_diff.py [speech] [tier]   (default: trump_2026 cheap)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
from hydramind import HydraMind
from hydramind.transport import Transport, ProxyCompletion
from hydramind.registry import load_registry
from hydramind.manifest import NullSpendSink
from truthbot.checkworthy import classifier, pipeline
import proxy_client


def main():
    speech = sys.argv[1] if len(sys.argv) > 1 else "trump_2026"
    tier = sys.argv[2] if len(sys.argv) > 2 else "cheap"   # cheap=haiku-v2 (the chosen A2)
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return

    rows = [json.loads(l) for l in
            (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()]
    sents = [{"sid": r["sid"], "text": r["text"], "context": r.get("context", "")}
             for r in rows if r["speech"] == speech]
    gold = {json.loads(l)["sid"]: json.loads(l)["gold_label"] for l in
            (HERE / "claim-set" / "checkworthy_gold.jsonl").read_text().splitlines() if l.strip()}
    print(f"speech={speech} tier={tier} | sentences={len(sents)} | gold overlap={sum(s['sid'] in gold for s in sents)}")

    hm = HydraMind(load_registry(), Transport(completion_fn=ProxyCompletion(
        key_env=proxy_client.resolve_key_env(), base_url=proxy_client.base_url())),
        spend_sink=NullSpendSink(), project="truth-bot")

    memo: dict[str, dict] = {}
    cost = {"usd": 0.0}

    def classify_fn(items):
        todo = [it for it in items if it["sid"] not in memo]
        if todo:
            out, mani = classifier.classify(hm, todo, tune={"prompt": classifier.A2_SYSTEM}, tier=tier)
            cost["usd"] += mani.total_cost_usd
            for v in out:
                memo[v["sid"]] = v
        return [memo[it["sid"]] for it in items]

    res_off = pipeline.run_layer_a(sents, classify_fn=classify_fn, confirm_pass=False)
    res_on = pipeline.run_layer_a(sents, classify_fn=classify_fn, confirm_pass=True)

    q_off = {r["sid"] for r in res_off.check_worthy_queue}
    q_on = {r["sid"] for r in res_on.check_worthy_queue}
    routes = res_on.a1_routes
    import collections
    print("A1 routes:", dict(collections.Counter(routes.values())))
    print(f"check-worthy queue: confirm_pass=OFF {len(q_off)}  ->  ON {len(q_on)}  "
          f"(vetoed {len(q_off - q_on)}; ON⊆OFF={q_on <= q_off})")

    vetoed = sorted(q_off - q_on)
    by_sid = {s["sid"]: s for s in sents}
    a2 = memo
    print(f"\n## {len(vetoed)} A1-PASS sentences A2 vetoed (lexical false positives caught)")
    gv = collections.Counter()
    recall_loss = []
    for sid in vetoed:
        g = gold.get(sid, "—")
        if g == "check-worthy":
            recall_loss.append(sid)
        gv[g] += 1
        print(f"  [{sid}] A2={a2[sid]['label']:12} gold={g:12} :: {by_sid[sid]['text'][:90]}")
        print(f"        rationale: {a2[sid].get('rationale','')[:110]}")

    print(f"\n## veto vs gold: {dict(gv)}")
    print(f"   correctly vetoed (gold opinion/unimportant): {sum(v for k,v in gv.items() if k in ('opinion','unimportant'))}")
    print(f"   WRONGLY vetoed (gold check-worthy = recall loss): {len(recall_loss)} {recall_loss}")
    print(f"   not in gold (unlabeled, can't score): {gv.get('—',0)}")

    # queue precision vs gold (only sids present in gold)
    def q_prec(q):
        scored = [sid for sid in q if sid in gold]
        tp = sum(gold[sid] == "check-worthy" for sid in scored)
        return tp, len(scored), (tp / len(scored) if scored else 0.0)
    to = q_prec(q_off); tn = q_prec(q_on)
    print(f"\n## check-worthy queue precision vs gold (scored subset)")
    print(f"   OFF: {to[0]}/{to[1]} = {to[2]:.2f}   ON: {tn[0]}/{tn[1]} = {tn[2]:.2f}")
    print(f"\nA2 live spend: ${cost['usd']:.4f} ({len(memo)} sentences classified once)")


if __name__ == "__main__":
    main()
