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

## Caveats

- Gold rows are all `needs_review` (single-annotator); decided-accuracy will shift with a
  multi-annotator adjudication pass.
- n=17 is small — treat these as a shape check, not a settled accuracy number. Expanding the
  gold (more FALSE/UNVERIFIABLE, speaker balance) tightens the estimate.
