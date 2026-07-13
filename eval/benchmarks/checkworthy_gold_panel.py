#!/usr/bin/env python3
"""Build an independent labeling panel for a stratified check-worthiness sample, as
the basis for an adjudicated Layer-A gold (the answer key we lacked).

Panel = claude-sonnet + mistral (cross-vendor), each labeling the sample with a NEUTRAL,
BALANCED rubric (NOT the overshooting v2 prompt). Direct proxy calls so any model works.
Writes panel_labels.json; a human/agent adjudicator turns it into the gold.

Sample (~55): the 4 known anchors + the 21 haiku boundary flips + clear cases where the
earlier haiku and sonnet runs AGREED (high-confidence check-worthy / opinion / unimportant),
so the set spans the decision boundary and clean cases alike.
"""
from __future__ import annotations
import json, os, re, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); import proxy_client

PANEL = ["claude-sonnet", "mistral"]

NEUTRAL_RUBRIC = """You are an expert fact-checking editor. For a single sentence from a \
political speech, decide whether it is worth sending to a fact-checking pipeline. Output \
EXACTLY one label:

- "check-worthy": a SPECIFIC, verifiable factual assertion whose truth a reasonable person \
could question and that matters to the public — a statistic, a historical or current event, a \
quantitative comparison, a causal claim, or a claim about what a person/entity did, said, or \
funded. A well-known fact STILL counts if it makes a specific, checkable, consequential assertion.
- "opinion": the sentence's MAIN purpose is a value judgment, rhetoric, aspiration, promise, \
prediction, or a policy proposal/recommendation ("we should...", "let's...", "let X do Y"). If \
it embeds a fact but its dominant purpose is to advocate or evaluate, it is opinion.
- "unimportant": factual but not worth checking — a greeting, thanks, ceremony, procedure, \
personal aside, or a trivial/undisputed truism (a famous death date, a sports score) with no \
public stakes.

Judge the DOMINANT purpose and whether the checkable content is specific and consequential. \
Do NOT consider who the speaker is.

Calibration:
- "Core inflation fell to 1.7 percent last quarter." -> check-worthy
- "The military destroyed the enemy's main installation, defended by thousands of troops." -> check-worthy
- "Let Medicare negotiate drug prices, like the VA already does." -> opinion
- "We must protect our democracy." -> opinion
- "Thomas Jefferson drew his last breath." -> unimportant
- "Thank you all for being here tonight." -> unimportant

Return JSON only: {"label":"check-worthy|opinion|unimportant","rationale":"one clause"}"""

_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def label_one(model, key, base, sentence):
    body = json.dumps({"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": NEUTRAL_RUBRIC},
        {"role": "user", "content": sentence}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    m = _OBJ.search(content)
    try:
        lab = json.loads(m.group(0)).get("label", "").strip().lower()
    except Exception:
        lab = ""
    return lab if lab in {"check-worthy", "opinion", "unimportant"} else f"?({content[:30]})"


def build_sample(rows):
    ab = json.loads((HERE / "ab_result.json").read_text())
    son = json.loads((HERE / "sonnet_result.json").read_text())["sonnet_new"]
    hk = ab["new_pred"]
    anchors = ["biden_2022:0025", "trump_2026:0656", "biden_2022:0210", "trump_2026:0700"]
    flips = ab["flips_cw_out"]
    agree = lambda lbl: [s for s in hk if hk[s] == lbl and son.get(s) == lbl]
    clear = agree("check-worthy")[:14] + agree("opinion")[:10] + agree("unimportant")[:10]
    seen, out = set(), []
    for sid in anchors + flips + clear:
        if sid not in seen:
            seen.add(sid); out.append(sid)
    return out


def main():
    if not proxy_client.key_present():
        print(proxy_client.BLOCKED_MSG); return
    key = os.environ[proxy_client.resolve_key_env()]; base = proxy_client.base_url()
    rows = {json.loads(l)["sid"]: json.loads(l) for l in
            (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()}
    sample = build_sample(rows)
    print(f"sample size: {len(sample)}")
    panel = {}
    for i, sid in enumerate(sample, 1):
        text = rows[sid]["text"]
        votes = {m: label_one(m, key, base, text) for m in PANEL}
        panel[sid] = {"text": text, "old_label": rows[sid]["label"], **votes}
        print(f"  [{i:2}/{len(sample)}] {sid}: {votes}")
    (HERE / "panel_labels.json").write_text(json.dumps(panel, indent=2, ensure_ascii=False))
    print("-> panel_labels.json")


if __name__ == "__main__":
    main()
