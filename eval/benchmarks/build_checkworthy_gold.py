#!/usr/bin/env python3
"""Adjudicate the sonnet+mistral panel into checkworthy_gold.jsonl (the Layer-A answer key).

Rule: panel agreement -> that label; a sonnet/mistral split -> the adjudicator (claude) call
recorded in SPLIT_ADJ below, with reasoning. A few unanimous-but-borderline rows are flagged
needs_review. What matters most for the check-worthiness GATE is check-worthy vs not; the
opinion/unimportant split on vague fragments is secondary (both are non-check-worthy).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PANEL = json.loads((HERE / "panel_labels.json").read_text())
ROWS = {json.loads(l)["sid"]: json.loads(l) for l in
        (HERE / "claim-set" / "claim_set.jsonl").read_text().splitlines() if l.strip()}

# adjudicator (claude) calls on the 15 sonnet/mistral splits, with one-line reasoning
SPLIT_ADJ = {
    "biden_2022:0025": ("check-worthy", "substantive historical claim about NATO's purpose; fact-checkable (peace vs containment)"),
    "trump_2026:0056": ("check-worthy", "action/attribution claim ('we ended DEI') — checkable what was done"),
    "trump_2026:0072": ("unimportant",  "sports result, no public stakes (rubric: sports score)"),
    "trump_2026:0096": ("unimportant",  "US hosting World Cup/Olympics is undisputed, celebratory, no stakes"),
    "trump_2026:0336": ("unimportant",  "describes a ceremony he hosted — ceremonial/procedural"),
    "trump_2026:0360": ("opinion",      "partisan blame/rhetoric ('no money because of the Democrats')"),
    "trump_2026:0380": ("opinion",      "explicit opinion ('I think he's a nice guy')"),
    "trump_2026:0428": ("opinion",      "promotional puffery; 'every state' is vague, low fact-check value"),
    "biden_2022:0140": ("check-worthy", "specific historical/legal claim (Buy American Act ~century) — in verdict-gold"),
    "biden_2022:0155": ("check-worthy", "quantitative superlative ('one of the biggest ... in history') — checkable"),
    "biden_2022:0065": ("check-worthy", "asserts what US forces are doing (deploy to defend NATO, not fight in Ukraine)"),
    "biden_2022:0070": ("unimportant",  "vague filler/prediction, no checkable content"),
    "biden_2022:0090": ("unimportant",  "vague filler ('going to take time')"),
    "biden_2022:0035": ("unimportant",  "sentence fragment ('We were ready')"),
    "biden_2022:0060": ("unimportant",  "list fragment ('Economic assistance.')"),
    # round 2 (53 -> 150 expansion): trump splits, mostly the opinion/unimportant boundary
    "trump_2026:0048": ("opinion",      "superlative w/ no measurable referent ('hottest country') — rhetoric"),
    "trump_2026:0076": ("unimportant",  "personal sports anecdote + subjective praise of a goalie, no public stakes"),
    "trump_2026:0084": ("unimportant",  "procedural detail of a personal award anecdote ('I did take a vote')"),
    "trump_2026:0120": ("unimportant",  "live ceremonial medal presentation — ceremonial address, no fact-check stakes"),
    "trump_2026:0152": ("opinion",      "bare value judgment about persons ('great people')"),
    "trump_2026:0164": ("opinion",      "promotional call-to-action (visit a website) — exhortation"),
    "trump_2026:0180": ("opinion",      "vague unquantified magnitude ('a lot of money') — emphatic, not a checkable figure"),
    "trump_2026:0236": ("opinion",      "subjective characterization ('so simple, so big')"),
    "trump_2026:0292": ("opinion",      "emphatic reaction ('a lot of money'), no checkable content of its own"),
    "trump_2026:0312": ("opinion",      "vague comparative ('even worse') with no defined metric — rhetorical amplification"),
    "trump_2026:0408": ("opinion",      "rhetorical counterfactual about the past, not a verifiable assertion"),
    "trump_2026:0484": ("opinion",      "subjective personal reaction ('never seen anything like it'), unfalsifiable"),
    "trump_2026:0536": ("opinion",      "self-quoted prediction ('going to be tough'), personal anecdote"),
    "trump_2026:0544": ("check-worthy", "specific checkable outcome ('found all 28') — quantitative hostage/remains recovery, consequential"),
    "trump_2026:0600": ("unimportant",  "anecdotal specific figure ($1,775) w/ negligible public stakes (setup for a '$1,776' pun)"),
    "trump_2026:0604": ("opinion",      "non-verifiable personal claim about his own conduct — anecdote"),
    "trump_2026:0624": ("opinion",      "deliberately unattributable anecdote ('won't tell you who') of vague praise — self-promotional"),
    "trump_2026:0688": ("unimportant",  "ceremonial honoring of a guest; 'recognition he deserves' is value-laden"),
}
# unanimous but the adjudicator wants a human second look (kept the panel label)
REVIEW_FLAG = {
    "trump_2026:0560": "panel says check-worthy; adjudicator leans opinion (rhetorical metaphor 'kings of the roadside bomb')",
    "trump_2026:0256": "specific $ claim but private personal testimony — hard to verify",
}


def main():
    # a split (sonnet != mistral) with no adjudicator call yet cannot be scored;
    # emit a worklist and refuse to write a partial gold rather than KeyError.
    missing = [sid for sid, v in PANEL.items()
               if v["claude-sonnet"] != v["mistral"] and sid not in SPLIT_ADJ]
    if missing:
        work = {sid: {"text": PANEL[sid]["text"],
                      "claude-sonnet": PANEL[sid]["claude-sonnet"],
                      "mistral": PANEL[sid]["mistral"]} for sid in sorted(missing)}
        p = HERE / "_unadjudicated_splits.json"
        p.write_text(json.dumps(work, indent=2, ensure_ascii=False))
        print(f"BLOCKED: {len(missing)} split(s) need adjudication -> {p.name}")
        print("  add each sid to SPLIT_ADJ with (label, reasoning), then re-run.")
        return
    out = []
    for sid, v in PANEL.items():
        s, m = v["claude-sonnet"], v["mistral"]
        if s == m:                                   # panel agreement
            label, conf, nr, why = s, "high", False, "sonnet+mistral agree"
            if sid in REVIEW_FLAG:
                conf, nr, why = "med", True, REVIEW_FLAG[sid]
        else:                                        # split -> adjudicator
            label, why = SPLIT_ADJ[sid]
            conf, nr = "med", True
        out.append({"sid": sid, "text": ROWS[sid]["text"], "speech": ROWS[sid]["speech"],
                    "gold_label": label, "confidence": conf, "needs_review": nr,
                    "annotators": {"claude-sonnet": s, "mistral": m, "claude-adjudicator": label},
                    "rationale": why})
    out.sort(key=lambda r: r["sid"])
    p = HERE / "claim-set" / "checkworthy_gold.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    import collections
    print(f"wrote {len(out)} rows -> {p.name}")
    print("  gold dist:", dict(collections.Counter(r["gold_label"] for r in out)))
    print("  needs_review:", sum(r["needs_review"] for r in out),
          "| high-conf:", sum(r["confidence"] == "high" for r in out))
    print("  panel agreement:", sum(1 for v in PANEL.values() if v["claude-sonnet"] == v["mistral"]),
          "/", len(PANEL))


if __name__ == "__main__":
    main()
