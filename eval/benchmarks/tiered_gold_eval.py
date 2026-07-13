#!/usr/bin/env python3
"""Score tiered A2 (classify_escalating: haiku, escalate low-confidence -> sonnet) against
the adjudicated checkworthy_gold.jsonl, at a few confidence thresholds. Reports the
check-worthy F1 + anchor scorecard + escalation rate + cost — to see if the tier captures
sonnet's edge on the hard cases while escalating only a small subset."""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
from hydramind import HydraMind
from hydramind.transport import Transport, ProxyCompletion
from hydramind.registry import load_registry
from hydramind.manifest import NullSpendSink
from truthbot.checkworthy import classifier
import proxy_client

ANCHORS = {"trump_2026:0656": "check-worthy", "biden_2022:0025": "check-worthy",
           "biden_2022:0210": "opinion", "trump_2026:0700": "unimportant"}


def score(pred, gold):
    acc = sum(pred[s] == gold[s]["gold_label"] for s in gold) / len(gold)
    tp = sum(pred[s] == "check-worthy" and gold[s]["gold_label"] == "check-worthy" for s in gold)
    fp = sum(pred[s] == "check-worthy" and gold[s]["gold_label"] != "check-worthy" for s in gold)
    fn = sum(pred[s] != "check-worthy" and gold[s]["gold_label"] == "check-worthy" for s in gold)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return acc, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return
    gold = {json.loads(l)["sid"]: json.loads(l) for l in
            (HERE / "claim-set" / "checkworthy_gold.jsonl").read_text().splitlines() if l.strip()}
    rows = {json.loads(l)["sid"]: json.loads(l) for l in
            (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()}
    sents = [{"sid": s, "text": rows[s]["text"], "context": rows[s].get("context", "")} for s in gold]
    hi = {s: g for s, g in gold.items() if not g["needs_review"]}

    hm = HydraMind(load_registry(), Transport(completion_fn=ProxyCompletion(
        key_env=proxy_client.resolve_key_env(), base_url=proxy_client.base_url())),
        spend_sink=NullSpendSink(), project="truth-bot")

    print(f"# tiered A2 (haiku -> escalate low-conf to sonnet) vs gold (n={len(gold)}, hi-conf={len(hi)})")
    print(f"{'thresh':>6} | {'esc%':>5} {'acc':>5} {'cwF1':>5} | hi cwF1 | anchors | cost")
    for th in (0.7, 0.85, 0.95):
        out, info = classifier.classify_escalating(hm, sents, conf_threshold=th)
        pred = {r["sid"]: r["label"] for r in out}
        a, f1 = score(pred, gold)
        _, hf1 = score({s: pred[s] for s in hi}, hi)
        an = sum(pred[sid] == want for sid, want in ANCHORS.items())
        cost = info["manifest_base"].total_cost_usd + (
            info["manifest_escalate"].total_cost_usd if info["manifest_escalate"] else 0.0)
        print(f"{th:>6} | {info['escalate_rate']:>5.0%} {a:.2f} {f1:.2f} | {hf1:.2f}   | "
              f"{an}/4     | ${cost:.4f}")
        # show the anchor labels at this threshold
        print(f"         anchors: { {sid: pred[sid] for sid in ANCHORS} }  "
              f"(escalated: {[r['sid'] for r in out if r['escalated'] and r['sid'] in ANCHORS]})")


if __name__ == "__main__":
    main()
