# Layer C Phase 3 — severity calibration experiment (P67.2)

**Date:** 2026-07-14 · **Question:** gate-2 showed the open-book severity-softening is
**model-level** (proposer + critic unanimously choose the milder label). Can a prompt
move the seats and recover decided-accuracy without collapsing coverage?

**Method:** a `--calib` A/B in `score_layerb_vs_gold.py` swaps the open-book prompt for a
`CALIBRATED_OPEN_BOOK_PROMPTS` **decision procedure** keyed on the claim's *core*
assertion. Live, roster.dev, 21-row TRAIN gold, back-to-back same session.

## Result — the FALSE↔MISLEADING see-saw

| | default | calib-v1 | calib-v2 |
|---|---|---|---|
| decided-accuracy | 0.579 | 0.688 | 0.688 |
| coverage | 0.905 | 0.762 | 0.762 |
| **FALSE → correct** | **0/4** (all→MISLEADING) | **4/4** ✅ | 1/4 |
| **MISLEADING → correct** | 3/6 | 0/6 (→TRUE/FALSE) | **4/6** ✅ |

- **v1** ("a contradicted core is FALSE even with a kernel of truth") — **fixes FALSE
  (0/4 → 4/4)** but collapses MISLEADING to the poles (0/6).
- **v2** ("overstatement of a real fact is MISLEADING, not FALSE") — **recovers
  MISLEADING (0/6 → 4/6)** but re-breaks FALSE (4/4 → 1/4).

## Findings

1. **The softening is prompt-movable.** The prompt decisively controls the
   FALSE↔MISLEADING boundary — v1 took FALSE from 0/4 to 4/4, a categorical effect far
   larger than run noise. Gate-2's "model-level, unanimous" softening *is* reachable by
   the prompt.
2. **But one global instruction can't hold both boundaries.** FALSE and MISLEADING sit on
   opposite sides of the same line; emphasizing one pushes claims across it and breaks the
   other. The cheap roster (haiku / mistral / dsv4-flash) has no stable internal
   FALSE-vs-MISLEADING threshold for a prompt to simply "set."
3. **Aggregate decided-accuracy is not a reliable ranker here.** Two default runs came in
   at 0.667 (gate-2) and 0.579 (this A/B) — a ~0.1 swing at n=21. Only **category-level**
   effects (FALSE 0/4↔4/4, MISLEADING 0/6↔4/6, coverage 0.90↔0.76) are robust.

## Recommendation (not adopted — for jackie / next Phase-3 increment)

Neither calibrated variant is a clean win, so **the default open-book prompt is unchanged**
(the `--calib` harness + `CALIBRATED_OPEN_BOOK_PROMPTS` are committed as opt-in experiment
infrastructure, currently = v2). The see-saw says a single 4-way prompt is the wrong shape.
Next levers, in order of expected payoff:

- **Two-stage decision (structural):** let the panel pick TRUE / {FALSE-or-MISLEADING} /
  UNVERIFIABLE, then run a dedicated **binary discriminator** only on the FALSE-or-MISLEADING
  bucket — "is the core CONTRADICTED (FALSE) or merely OVERSTATED (MISLEADING)?". Isolating
  the hard boundary into its own call should beat trying to hold it inside a 4-way prompt.
- **Stronger tier on the boundary:** route FALSE/MISLEADING adjudication to a frontier
  arbiter; the cheap roster may lack the judgment for it.
- **Bigger gold:** n=21 single-annotator with ~0.1 variance can't rank fine changes. Grow +
  multi-annotate before trusting aggregate accuracy.
- **Reader-facing scale:** the coarse Truthy projection may merge FALSE/MISLEADING anyway —
  worth checking whether this boundary needs to be nailed at all for the published output.
