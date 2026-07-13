#!/usr/bin/env python3
"""Ad-hoc A/B: old vs new A2 check-worthiness prompt over the 277-claim set (haiku).
Measures false-positive reduction (check-worthy -> opinion/unimportant) and any
recall movement (opinion/unimportant -> check-worthy), and pins the two known cases.
Requires the truth-bot proxy key. Prints a report + writes ab_result.json."""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from hydramind import HydraMind
from hydramind.transport import Transport, ProxyCompletion
from hydramind.registry import load_registry
from hydramind.manifest import NullSpendSink
from truthbot.checkworthy import classifier
import proxy_client

OLD_PROMPT = (
    'You classify a single sentence from a political transcript for a fact-checking '
    'pipeline. Decide whether it should be verified. Output EXACTLY one label:\n\n'
    '- "check-worthy": a factual, verifiable assertion of public importance (statistic, historical '
    'event, quantitative comparison, causal attribution, or a claim about what someone/some entity '
    'did or said). Also return claim_type in '
    '{statistical, historical, attribution, comparison, other}.\n'
    '- "opinion": opinion, value judgment, rhetoric, aspiration, promise, or a prediction about the '
    'future. claim_type=null.\n'
    '- "unimportant": literally factual but trivial (greeting, ceremony, procedure, personal aside, '
    'truism), not worth a fact-check budget. claim_type=null.\n\n'
    'Judge only the proposition and its speech-act form. Do NOT consider who the speaker is.\n\n'
    'Return JSON only: {"label": "...", "claim_type": "... or null", "confidence": 0.0-1.0, '
    '"rationale": "one clause"}')
NEW_PROMPT = classifier.A2_SYSTEM


def run(hm, sents, prompt):
    out, manifest = classifier.classify(hm, sents, tune={"prompt": prompt})
    return {v["sid"]: v["label"] for v in out}, manifest


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return
    rows = [json.loads(l) for l in (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()]
    sents = [{"sid": r["sid"], "text": r["text"], "context": r.get("context", "")} for r in rows]
    old_label = {r["sid"]: r["label"] for r in rows}   # the frozen (buggy) labels, for reference

    hm = HydraMind(load_registry(), Transport(completion_fn=ProxyCompletion(
        key_env=proxy_client.resolve_key_env(), base_url=proxy_client.base_url())),
        spend_sink=NullSpendSink(), project="truth-bot")

    old_pred, m_old = run(hm, sents, OLD_PROMPT)
    new_pred, m_new = run(hm, sents, NEW_PROMPT)

    def dist(d): return dict(collections.Counter(d.values()))
    flips_cw_out = [s["sid"] for s in sents
                    if old_pred[s["sid"]] == "check-worthy" and new_pred[s["sid"]] != "check-worthy"]
    flips_in_cw = [s["sid"] for s in sents
                   if old_pred[s["sid"]] != "check-worthy" and new_pred[s["sid"]] == "check-worthy"]
    agree = sum(1 for s in sents if old_pred[s["sid"]] == new_pred[s["sid"]])
    txt = {r["sid"]: r["text"] for r in rows}

    print(f"# Layer A A/B (old vs new A2 prompt) — n={len(sents)}  haiku")
    print(f"  old-prompt label dist: {dist(old_pred)}")
    print(f"  new-prompt label dist: {dist(new_pred)}")
    print(f"  old vs new agreement : {agree}/{len(sents)} ({agree/len(sents):.1%})")
    print(f"  check-worthy -> opinion/unimportant (false-positive fixes): {len(flips_cw_out)}")
    for sid in flips_cw_out:
        print(f"    - {sid}: {old_pred[sid]}->{new_pred[sid]}  | {txt[sid][:80]}")
    print(f"  opinion/unimportant -> check-worthy (recall gain OR new false-pos): {len(flips_in_cw)}")
    for sid in flips_in_cw:
        print(f"    + {sid}: {old_pred[sid]}->{new_pred[sid]}  | {txt[sid][:80]}")
    for sid in ("biden_2022:0210", "trump_2026:0700"):
        print(f"  known case {sid}: old={old_pred.get(sid)} new={new_pred.get(sid)}")
    cost = m_old.total_cost_usd + m_new.total_cost_usd
    print(f"  spend: ${cost:.4f}  | model_mismatches old/new: "
          f"{len(m_old.model_mismatches())}/{len(m_new.model_mismatches())}")
    (HERE / "ab_result.json").write_text(json.dumps(
        {"old_pred": old_pred, "new_pred": new_pred, "flips_cw_out": flips_cw_out,
         "flips_in_cw": flips_in_cw, "spend_usd": cost}, indent=2))
    print("  -> ab_result.json")


if __name__ == "__main__":
    main()
