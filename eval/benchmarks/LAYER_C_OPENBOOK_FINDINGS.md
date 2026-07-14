# Layer C Phase 2 — open-book evidence findings

**Date:** 2026-07-14 · **Scorer:** `score_layerb_vs_gold.py [--open-book]` · **Gold:** 17-row
canonical verdict-gold (TRAIN only, I6-safe) · **Roster:** `dev` (cheap tiers) ·
**Evidence:** Brave + FactCheck connectors, time-scoped per claim to
`expected_claim_window(utterance_date)`.

## Result (vs the closed-book + temporal baseline)

| variant | decided-accuracy | coverage | abstain_gap | spend |
|---|---|---|---|---|
| baseline (closed-book + temporal) | 0.78 | 0.53 | 7 | — |
| **open-book, default contract** | 0.667 (10/15) | **0.882** | **1** | $0.007 |
| open-book + strict-label rubric (A/B, not merged) | 0.727 (8/11) | 0.647 | 5 | $0.023 |

Every claim received evidence (102 items / 17 claims; all 17 had ≥1). The default
open-book pass committed on 15/17 and carried citations on 15.

## What open-book fixed

The existential Layer C goal — coverage collapse / confident staleness — is solved.
Coverage 0.53 → 0.88 and **abstain_gap 7 → 1**: current-events claims the closed-book
panel abstained on (e.g. the Feb-2026 SOTU lines) now resolve against dated,
time-scoped reporting. Net +6 commits, +3 hits.

## What it exposed — severity softening (the Phase 3 lever)

decided-accuracy did **not** hold (0.78 → 0.667). The regression is systematic, not
noise. Confusion (default open-book):

- **FALSE → MISLEADING** (2/2): both FALSE golds downgraded. Partial/technical support
  in the evidence launders a false core assertion into the milder label.
- **MISLEADING → TRUE** (3/5): supporting-ish evidence pulls misleading claims to TRUE.

i.e. given evidence, the panel hedges **down** the severity ladder.

### A/B: does a strict-label rubric fix it?

A principled (non-claim-specific) rubric pinning the TRUE/FALSE/MISLEADING boundaries
recovered accuracy partway (0.667 → 0.727; one FALSE flipped back) **but** cost
coverage (0.882 → 0.647, gap 1 → 5) and introduced 2 disagreement-flags. It slides
along the precision/coverage frontier back toward baseline rather than dominating it.
**Not merged** — verdict-affecting prompt tuning shouldn't be auto-merged, and tuning
to the 17-row TRAIN set risks overfitting. Left for jackie / Phase 3.

## Bottom line

Phase 2 delivered the open-book *mechanism* (time-scoped retrieval → provenanced pack
→ cited, grounded verdicts) and it does exactly what it was meant to: coverage rises,
abstain_gap collapses. The remaining hard problem is **label calibration under
evidence**, which trades against coverage and is not solved by a prompt tweak alone —
that is the Phase 3 objective.
