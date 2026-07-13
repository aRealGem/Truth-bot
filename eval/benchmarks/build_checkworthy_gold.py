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
}
# unanimous but the adjudicator wants a human second look (kept the panel label)
REVIEW_FLAG = {
    "trump_2026:0560": "panel says check-worthy; adjudicator leans opinion (rhetorical metaphor 'kings of the roadside bomb')",
    "trump_2026:0256": "specific $ claim but private personal testimony — hard to verify",
}


def main():
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
