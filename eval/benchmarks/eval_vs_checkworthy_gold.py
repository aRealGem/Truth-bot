#!/usr/bin/env python3
"""Score Layer A configs against the adjudicated checkworthy_gold.jsonl (the answer key).

Compares v1 prompt (cached: ab_result/sonnet_result, which cover the full corpus) vs v2
prompt (live) on both haiku and sonnet, over the adjudicated gold (now 150 rows). Reports
3-way accuracy + the GATE metric: check-worthy precision/recall/F1 (v1 overshot -> low
recall). Also a high-confidence-only view (drops the needs_review rows) and the 4-anchor
scorecard."""
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
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"acc": acc, "cw_P": p, "cw_R": r, "cw_F1": f1, "n": len(gold)}


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return
    gold = {json.loads(l)["sid"]: json.loads(l) for l in
            (HERE / "claim-set" / "checkworthy_gold.jsonl").read_text().splitlines() if l.strip()}
    sids = list(gold)
    rows = {json.loads(l)["sid"]: json.loads(l) for l in
            (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()}
    sents = [{"sid": s, "text": rows[s]["text"], "context": rows[s].get("context", "")} for s in sids]

    v1h = json.loads((HERE / "ab_result.json").read_text())["new_pred"]
    v1s = json.loads((HERE / "sonnet_result.json").read_text())["sonnet_new"]

    hm = HydraMind(load_registry(), Transport(completion_fn=ProxyCompletion(
        key_env=proxy_client.resolve_key_env(), base_url=proxy_client.base_url())),
        spend_sink=NullSpendSink(), project="truth-bot")
    def run(tier):
        out, mani = classifier.classify(hm, sents, tune={"prompt": classifier.A2_SYSTEM,
                                                         "roles.solo.tier": tier})
        return {v["sid"]: v["label"] for v in out}, mani.total_cost_usd
    v2h, c_h = run("cheap")      # haiku, v2 prompt
    v2s, c_s = run("standard")   # sonnet, v2 prompt

    preds = {"v1 haiku": {s: v1h[s] for s in sids}, "v2 haiku": v2h,
             "v1 sonnet": {s: v1s[s] for s in sids}, "v2 sonnet": v2s}
    hi = {s: g for s, g in gold.items() if not g["needs_review"]}

    print(f"# Layer A vs adjudicated gold (n={len(sids)}; high-conf={len(hi)})")
    print(f"{'config':11} | {'acc':>5} {'cwP':>5} {'cwR':>5} {'cwF1':>5} || high-conf acc/cwF1")
    for name, pred in preds.items():
        a = score(pred, gold); h = score({s: pred[s] for s in hi}, hi)
        print(f"{name:11} | {a['acc']:.2f} {a['cw_P']:.2f} {a['cw_R']:.2f} {a['cw_F1']:.2f} "
              f"|| {h['acc']:.2f} / {h['cw_F1']:.2f}")
    print("\n## anchor scorecard (config -> label on the 4 known cases)")
    for name, pred in preds.items():
        got = {sid: pred[sid] for sid in ANCHORS}
        ok = sum(got[sid] == want for sid, want in ANCHORS.items())
        print(f"  {name:11} {ok}/4  {got}")
    print(f"\n  v2 live spend: haiku ${c_h:.4f} + sonnet ${c_s:.4f}")


if __name__ == "__main__":
    main()
