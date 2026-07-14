# Layer A scoring vs the 150-row adjudicated gold

Run: `eval/benchmarks/eval_vs_checkworthy_gold.py` against `checkworthy_gold.jsonl` (150 rows,
115 high-conf / 35 needs_review). Live, 2026-07-13. Spend: haiku $0.17 + sonnet $0.44 = $0.60.

The GATE metric is **check-worthy precision/recall/F1** — how well a config separates the
sentences that must enter the verifier from the opinion/unimportant that must not.

| config    | acc  | cw-P | cw-R | cw-F1 | hi-conf acc / cw-F1 |
|-----------|------|------|------|-------|---------------------|
| v1 haiku  | 0.85 | 0.97 | 0.76 | 0.85  | 0.91 / 0.92 |
| v2 haiku  | 0.84 | 0.84 | 0.89 | 0.86  | 0.90 / 0.90 |
| v1 sonnet | 0.83 | 0.78 | 0.87 | 0.82  | 0.90 / 0.89 |
| v2 sonnet | 0.82 | 0.73 | 0.93 | 0.82  | 0.92 / 0.92 |

Anchor scorecard (4 known cases): v2 sonnet 4/4, v2 haiku 3/4, v1 haiku/sonnet 2/4.

## Findings

1. **The POC "sonnet edge" does not survive at n=150.** On the 53-row POC, sonnet-v2 read
   ~0.90 F1 vs haiku ~0.84. On the 3x-larger gold, v2 **haiku** F1 (0.86 full / 0.90 hi-conf)
   *matches or beats* v2 sonnet (0.82 full / 0.92 hi-conf). Sonnet's only remaining advantages
   are +0.02 hi-conf F1 and the 4th anchor — within noise.

2. **Sonnet-v2 buys recall with precision, which is the wrong trade for the A2 veto.** v2 sonnet
   has the best recall (0.93) but the worst precision (0.73): it over-calls check-worthy. A2's
   job (PR #25, `confirm_pass=True`) is to *reject A1's lexical false positives* — that needs
   precision on the reject side. v2 haiku (P 0.84 / R 0.89) is the balanced choice; v1 haiku is
   even more precise (0.97) but drops 24% of real check-worthy (R 0.76).

3. **Cost:** sonnet is ~2.6x haiku ($0.44 vs $0.17 for 150 sentences) for no F1 gain.

4. **Circularity caveat:** the gold was adjudicated from a sonnet+mistral panel, so if anything
   it is mildly biased *toward* sonnet. Sonnet still does not win — which strengthens the haiku
   conclusion rather than weakening it.

## Recommendation (Layer A A2 model, step 2)

Use **haiku-v2** for the ambiguous+PASS confirmation band. Equal-or-better F1 than sonnet on
the larger gold, better precision (the metric the veto actually needs), at ~40% of the cost.
Reserve sonnet only if a future, larger, independently-labeled gold shows a robust hi-conf edge.
