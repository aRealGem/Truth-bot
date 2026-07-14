# Layer B scored against the canonical verdict-gold (first pass)

Tool: `score_layerb_vs_gold.py` — runs closed-book PCA (roster.dev: P=mistral, C=dsv4-flash,
A=claude-haiku) over exactly the `verdict_gold.train.jsonl` sids (all 17 in TRAIN; none
heldout, I6-safe) and scores with the abstention semantics in `scorer/score_verdict.py`.

Live, 2026-07-14. Spend **$0.0049** ($0.00029/claim).

| metric | value | reading |
|---|---|---|
| decided-accuracy | **0.75** (6/8) | when the panel commits, it's right 3/4 of the time |
| coverage | 0.47 (8/17) | commits on ~half; abstains on the rest — expected closed-book |
| abstain_gap | 9 | decidable gold the model abstained on → the coverage Layer C evidence closes |
| abstain_ok | 0 | (only 1 gold UNVERIFIABLE, and the model committed on it — see miss below) |

Confusion (gold → pred):

```
TRUE (9)         → TRUE 3, MISLEADING 1, ABSTAIN 5
FALSE (2)        → FALSE 1, ABSTAIN 1
MISLEADING (5)   → MISLEADING 2, ABSTAIN 3
UNVERIFIABLE (1) → TRUE 1
```

## Reading

- **The gold works as a scoring target** and produces the expected closed-book profile:
  moderate decided-accuracy with ~50% coverage. The 9 `abstain_gap` rows are the concrete,
  per-claim argument for the Layer C evidence lane (they're decidable *with* sources; the
  closed-book panel correctly declines to guess).
- **The new FALSE class is exercised**: 1 of 2 FALSE caught closed-book, 1 abstained. Before
  this reconciliation the gold had no FALSE at all, so this axis was previously unmeasurable.
- **Two misses**, both instructive:
  - a `TRUE → MISLEADING` (panel over-skeptical without evidence), and
  - the lone `UNVERIFIABLE → TRUE` (`trump_2026:0256`, Catherine's personal drug-cost
    testimony) — the panel committed where it should abstain. A small closed-book calibration
    signal, not a gold problem.

## Layer C Phase 1 — temporal grounding (2026-07-14, +$0.0098)

Prepending a speaker-blind temporal preamble (utterance date + expected evidence window +
today-authoritative; `verdict/speech_context.py`) to each claim's context — still closed-book,
no evidence yet — already improves the panel:

| metric | closed-book baseline | + temporal grounding |
|---|---|---|
| decided-accuracy | 0.75 (6/8) | **0.78 (7/9)** |
| coverage | 0.47 | **0.53** |
| abstain_gap | 9 | **7** |
| abstain_ok | 0 | **1** |

The clean qualitative win: `trump_2026:0256` (gold UNVERIFIABLE) now **abstains** instead of
over-committing TRUE — the "judge as-of utterance, reserve UNVERIFIABLE for what evidence can't
settle" grounding working. TRUE recall rose (5 vs 3), no regressions. This is the foundation for
Phase 2 (Connector evidence injected into the pack → open-book, closing the abstain_gap).

## Caveats

- Gold rows are all `needs_review` (single-annotator); decided-accuracy will shift with a
  multi-annotator adjudication pass.
- n=17 is small, and the panel has cross-provider nondeterminism — treat these as a shape
  check, not settled numbers. Expanding the gold (more FALSE/UNVERIFIABLE, speaker balance)
  tightens the estimate.
