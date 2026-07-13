#!/usr/bin/env python3
"""Same NEW A2 prompt, Sonnet (tier=standard) vs the cached Haiku-new labels.
Question: is the recall loss a PROMPT problem or a MODEL problem? If Sonnet keeps
the genuine check-worthy claims (NATO, military installation) while still dropping the
true false-positives (Medicare proposal, Jefferson truism), the model tier is the lever."""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
from hydramind import HydraMind
from hydramind.transport import Transport, ProxyCompletion
from hydramind.registry import load_registry
from hydramind.manifest import NullSpendSink
from truthbot.checkworthy import classifier
import proxy_client

# anchors we KNOW the right answer for (NATO is a TRUE row in verdict-gold)
ANCHORS = {
    "biden_2022:0025": "check-worthy",   # NATO founding purpose — substantive historical, keep
    "trump_2026:0656": "check-worthy",   # "military installation, thousands of soldiers…" — keep
    "biden_2022:0210": "not-check-worthy",  # "let Medicare negotiate…" — proposal, drop
    "trump_2026:0700": "not-check-worthy",  # "Jefferson drew his last breath" — truism, drop
}


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return
    rows = [json.loads(l) for l in (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()]
    sents = [{"sid": r["sid"], "text": r["text"], "context": r.get("context", "")} for r in rows]
    txt = {r["sid"]: r["text"] for r in rows}
    ab = json.loads((HERE / "ab_result.json").read_text())
    haiku_new = ab["new_pred"]; haiku_flips = ab["flips_cw_out"]

    hm = HydraMind(load_registry(), Transport(completion_fn=ProxyCompletion(
        key_env=proxy_client.resolve_key_env(), base_url=proxy_client.base_url())),
        spend_sink=NullSpendSink(), project="truth-bot")
    out, manifest = classifier.classify(hm, sents,
        tune={"prompt": classifier.A2_SYSTEM, "roles.solo.tier": "standard"})   # standard -> claude-sonnet
    son = {v["sid"]: v["label"] for v in out}

    def dist(d): return dict(collections.Counter(d.values()))
    agree = sum(1 for s in sents if son[s["sid"]] == haiku_new[s["sid"]])
    print(f"# Sonnet vs Haiku, SAME new prompt — n={len(sents)}")
    print(f"  haiku-new dist : {dist(haiku_new)}")
    print(f"  sonnet-new dist: {dist(son)}")
    print(f"  sonnet vs haiku agreement: {agree}/{len(sents)} ({agree/len(sents):.1%})")

    print("\n## anchor scorecard (known-correct cases)")
    ok = 0
    for sid, want in ANCHORS.items():
        s = son[sid]; got = "check-worthy" if s == "check-worthy" else "not-check-worthy"
        good = got == want; ok += good
        print(f"  {'✅' if good else '❌'} {sid}: sonnet={s:12} (want {want}) | {txt[sid][:60]}")
    print(f"  anchors correct: {ok}/{len(ANCHORS)}")

    print("\n## the 21 sentences Haiku dropped from check-worthy — what does Sonnet say?")
    kept = [sid for sid in haiku_flips if son[sid] == "check-worthy"]
    dropped = [sid for sid in haiku_flips if son[sid] != "check-worthy"]
    print(f"  Sonnet KEEPS as check-worthy (haiku over-dropped): {len(kept)}")
    for sid in kept:
        print(f"    keep {sid}: {txt[sid][:75]}")
    print(f"  Sonnet also drops (both agree non-check-worthy): {len(dropped)}")
    for sid in dropped:
        print(f"    drop {sid}: haiku={haiku_new[sid]:11} sonnet={son[sid]:11} | {txt[sid][:55]}")

    print(f"\n  spend: ${manifest.total_cost_usd:.4f}  model_mismatches: {len(manifest.model_mismatches())}")
    (HERE / "sonnet_result.json").write_text(json.dumps(
        {"sonnet_new": son, "anchors_ok": ok, "sonnet_keeps": kept, "spend_usd": manifest.total_cost_usd}, indent=2))


if __name__ == "__main__":
    main()
