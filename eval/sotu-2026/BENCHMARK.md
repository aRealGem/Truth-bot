# SOTU 2026 Benchmark — Evaluation Run Summary

Benchmark: 2026 State of the Union address (29 checkable claims).  
Reference: GPT 5.4 Pro extended analysis (`reference.json`).  
Last updated: 2026-04-19.

---

## Run Summary Table

| Run | Dir | Model | Mode | Pop | Gens | Claim Recall | Verdict Agreement | Fitness | Cost | Status |
|-----|-----|-------|------|-----|------|-------------|-------------------|---------|------|--------|
| 1 | `evolution_results/` | claude-sonnet-4-5 / haiku | Evolver (dry-run) | 8 | 5 | N/A (stub) | N/A (stub) | 0.5456 (all identical) | $0.00 | PARKING LOT |
| 2 | `opus-optimized-results/` | claude-opus-4-7 | Standalone fixed-prompt | — | 1 | 100% (29/29) | 80.7% | ~0.492 (simplified) | $2.15 | PARKING LOT |
| 3 | `opus-4-7-results/` | claude-opus-4-7 | Evolver gen-1 seed | 4 | 1 | 100% (29/29) | ~81% | **0.679** (best known) | $0.65 | PARKING LOT |

**GA / prompt evolver approach parked as of 2026-04-19.** Best known result preserved as reference
baseline. New evaluation approach TBD — see `eval/evolver/PARKING_LOT.md`.

---

## Best Known Result (Reference Baseline)

**Run 3 — `opus-4-7-results/`**  
Individual ID: `ecd8b5e_s58c7ac`  
Fitness: **0.679** *(FitnessScorer, 5-dimension formula)*

Preserved as a reference point for comparison against the new approach.

### Scores

| Metric | Score | Weight |
|--------|-------|--------|
| Claim recall | 100% (29/29) | 0.25 |
| Verdict agreement | ~81% | 0.30 |
| Explanation quality | ~67% | 0.20 |
| Source citation | ~33% | 0.15 |
| Parsimony | 0.0 *(calibration bug — now fixed)* | 0.10 |

Mean population fitness: 0.454 (1 of 4 individuals failed with fitness=0.0, likely JSON parse error).

**Note:** Parsimony was always 0.0 due to a calibration bug (target_max=2000 tokens vs. realistic
~15k token runs). Fixed in eval/evolver/fitness.py (new range: 4k-30k). The 0.679 score above
reflects the uncalibrated value; a re-run with the same genome would score higher on parsimony.

### Genome Configuration

**Extraction genome:**

| Gene | Index |
|------|-------|
| persona_idx | 3 |
| methodology_idx | 2 |
| taxonomy_idx | 2 |
| format_idx | 3 |
| filtering_idx | 1 |
| examples_idx | 0 |
| tone_idx | 1 |

**Synthesis genome:**

| Gene | Index |
|------|-------|
| persona_idx | 3 |
| verdict_taxonomy_idx | 2 |
| evidence_weighting_idx | 1 |
| confidence_idx | 1 |
| reasoning_idx | 1 |
| nuance_idx | 1 |
| format_idx | 0 |

**Key finding:** Non-baseline genome (persona_idx=3) beat the baseline across all four individuals
tested, confirming that prompt variation produces meaningful signal even from a single seed generation.

---

## Parking Lot

### Run 1 — evolution_results/ (dry-run, Sonnet/Haiku)

Executed with --dry-run. Returns 3 identical stub claims for every genome, all fitness scores
identical (0.5456), no selection pressure, zero information produced.
See evolution_results/PARKING_LOT.md.

### Run 2 — opus-optimized-results/ (standalone opus_eval.py)

Single-pass fixed prompts with simplified fitness (recall*0.25 + VA*0.30 only, max 0.55). Not
comparable to FitnessScorer results. Useful qualitative baseline only.
See opus-optimized-results/PARKING_LOT.md.

### Run 3 — opus-4-7-results/ (Evolver gen-1 seed) — GA APPROACH PARKED

Parked along with the entire GA/prompt evolver approach. The gen-1 seed population was valid and
the best individual (0.679) is preserved as a reference baseline, but further evolution runs will
not be pursued. New approach TBD — see eval/evolver/PARKING_LOT.md.

---

## Root Cause Analysis — Dry-Run Bug

```
--dry-run flag
    -> CachedRunner.dry_run = True
        -> extract_claims() returns 3 identical STUB claims for every genome
            -> FitnessScorer sees identical inputs for every individual
                -> All fitness scores are identical (0.5456)
                    -> Selection pressure = 0 (random drift only)
                        -> No evolution occurs across all 5 generations
```

Fix applied: Warning block in prompt_evolver.py main(). Preflight check in preflight.py.

---

## Infrastructure Available for New Approach

The following eval infrastructure was built during the GA phase and is ready to reuse:

- eval/evolver/base_eval.py — ModelClient protocol, shared extraction/synthesis prompts,
  BaseEvalRunner with caching and result serialization
- eval/evolver/fitness.py — FitnessScorer (5-dimension, calibrated), verdict taxonomy
  normalization (including compound labels), error direction tracking
- eval/evolver/preflight.py — pre-flight checks for API keys, transcripts, model deprecation,
  budget, writable dirs
- eval/evolver/runner.py — CachedRunner with retry logic (2 attempts on JSON parse error)
- eval/tests/ — 81 tests covering fitness, genome, GA ops, preflight, runner
- eval/opus_eval.py / eval/gpt_eval.py — standalone model eval scripts, ready to run
- eval/march-2025-congress/ — March 2025 Congressional Address test case (10 claims,
  not yet integrated into eval pipeline)
