# SOTU 2026 Benchmark — Evaluation Run Summary

Benchmark: 2026 State of the Union address (29 checkable claims).  
Reference: GPT 5.4 Pro extended analysis (`reference.json`).  
Last updated: 2026-04-27.

---

## Run Summary Table

| Run | Dir | Model | Mode | Pop | Gens | Claim Recall | Verdict Agreement | Fitness | Cost | Status |
|-----|-----|-------|------|-----|------|-------------|-------------------|---------|------|--------|
| 1 | `evolution_results/` | claude-sonnet-4-5 / haiku | Evolver (dry-run) | 8 | 5 | N/A (stub) | N/A (stub) | 0.5456 (all identical) | $0.00 | PARKING LOT |
| 2 | `opus-optimized-results/` | claude-opus-4-7 | Standalone fixed-prompt | — | 1 | 100% (29/29) | 80.7% | ~0.492 (simplified) | $2.15 | PARKING LOT |
| 3 | `opus-4-7-results/` | claude-opus-4-7 | Evolver gen-1 seed | 4 | 1 | 100% (29/29) | ~81% | **0.679** (best known) | $0.65 | PARKING LOT |
| 4 | `metrics/run_summaries/258b5758` | 4-adapter consensus (claude-opus-4-7 + gpt-5.4 + gemini-2.5-pro + grok-4) | Production pipeline (`--mode batch`, `--max-claims 29`) | — | — | 100% (29/29) | 62.8% | **0.5413** | $4.97 | **CURRENT BASELINE 2026-04-27** |
| 5 | (re-score of Run 4) | same | Multi-axis re-score (`score_run.py --all-axes`) | — | — | 100% (29/29) | 62.8% / 60.3% / **69.0%** *(fine / lenient / strict)* | 0.5413 / 0.5341 / **0.5599** | $0.00 | **2026-04-28** — coarse-axis calibration |

**Run 4 is the new current baseline** — first regression scoring against the production
4-adapter consensus pipeline post the 2026-04-26 SOTU fire (Grok `max_tool_calls` cap +
multi-claim `model_reported_sources` backfill). Apples-to-oranges versus runs 2/3 (single-model
standalone) so the −18pp verdict-agreement gap is not a clean regression signal — it conflates
(a) any genuine quality drift since 2026-04-19, (b) consensus-voting label drift (the consensus
algorithm can land on a label different from the strongest individual model's vote), and
(c) the new "Models split" verdict state (2/29 in this run) that did not exist in the
single-model baselines.

**Earlier GA / prompt evolver approach parked as of 2026-04-19.** Best known result (Run 3,
fitness 0.679) preserved as reference baseline. Re-score this run via:

```bash
python eval/sotu-2026/score_run.py \
    --run-id 258b5758-8e25-4bf0-8f34-63778d2f976e \
    --report-id e81546a0-6371-4e96-9e94-3d6213864d5a
```

---

## Current baseline (Run 4) — full scorecard

```
Inputs:
  Extracted claims:       100 (all checkable: 99)
  Published verdicts:     29 (consensus from claims.json)
  Sidecar entries:        39 (OpenAI/Gemini/xAI)
  Sidecar coverage:       13/29 claims have ≥1 sidecar explanation
  Anthropic explanations: excluded (lives in claim HTML — see TODO)
  Token count (sidecar):  262,262

Scores (FitnessScorer, 5-dimension):
  Claim recall:           100.0%   weight 0.25  (29/29 reference claims matched)
  Verdict agreement:       62.8%   weight 0.30
  Explanation quality:     39.5%   weight 0.20  (sidecar-only; Anthropic excluded)
  Source citation:         16.1%   weight 0.15  (sidecar-only; Anthropic excluded)
  Parsimony:                0.0%   weight 0.10  (262,262 tokens, sidecar-only)
  (target_max=30k calibrated for single-model standalone; not meaningful for 4-adapter consensus runs — see TODO)
  ──────────────────────────────────────
  Fitness:                0.5413

Vs baseline (best known: 0.679, claude-opus-4-7 standalone, 2026-04-19):
  Fitness delta:          -0.1377
  Cost:                   $4.97
```

### Known calibration gaps for production-pipeline regression scoring

1. **Anthropic explanations excluded.** They live in the rendered claim HTML pages, not in a
   single canonical JSON. `explanation_quality` and `source_citation_quality` are therefore
   computed from sidecar adapters only (OpenAI/Gemini/xAI = 3 of 4 voters) and undercount
   Anthropic's contribution. Fix-it: persist a per-run consolidated `verdicts.jsonl` covering
   all four adapters, or add an HTML-parse fallback in `score_run.py`.
2. **Parsimony target_max=30k.** Calibrated for single-model standalone runs (~15k tokens
   typical). Production 4-adapter pipeline routinely emits 200k+ in sidecar alone, so
   parsimony will always score 0%. Recalibrating to e.g. 100k–500k would make it a
   meaningful signal again, but changing the calibration mid-flight invalidates baseline
   comparisons. Track separately as a tunable.
3. **Baseline mismatch — partially addressed by Run 5 (2026-04-28).** Runs 2/3 are
   single-model standalone (Anthropic Opus only); Run 4 is 4-adapter consensus. Useful
   trend signal but not a clean A/B. The 2026-04-28 multi-axis re-score (Run 5, below)
   factors out the Mostly-True / Exaggerated / Misleading label-drift axis: under
   `coarse_strict` the Run 4 → Run 3 fitness gap narrows from −0.1377 to −0.1191 (≈14%
   of the gap was lens-dependent). The remaining gap is genuine quality drift between
   the single-model 2026-04-19 baseline and the 4-adapter consensus, not label noise.
   An Anthropic-only single-claim re-score against the same 29-claim transcript would
   still complete the apples-to-apples picture and remains a P1 follow-up, but the
   surprise-magnitude case for it weakened once the lens decomposition landed.

---

## Run 5 — Multi-axis re-score (2026-04-28)

Re-scores Run 4's published verdicts on three comparison axes, made possible by the
2026-04-27 5-bucket coarse-axis projection layer (commit
[`dc64ca0`](https://github.com/aRealGem/Truth-bot/commit/dc64ca0)). Inputs are unchanged
(same `--run-id 258b5758-...` / `--report-id e81546a0-...`); only the verdict-distance
axis varies. No LLM calls, no cost.

```bash
python eval/sotu-2026/score_run.py --all-axes \
    --run-id 258b5758-8e25-4bf0-8f34-63778d2f976e \
    --report-id e81546a0-6371-4e96-9e94-3d6213864d5a
```

### Multi-axis scorecard

| Metric                | fine (6-bucket) | coarse_lenient (5-bucket) | coarse_strict (5-bucket) | weight |
|-----------------------|----------------:|--------------------------:|-------------------------:|-------:|
| Claim recall          | 100.0%          | 100.0%                    | 100.0%                   | 0.25   |
| Verdict agreement     | 62.8%           | 60.3% *(−2.4pp)*          | **69.0%** *(+6.2pp)*     | 0.30   |
| Explanation quality   | 39.5%           | 39.5%                     | 39.5%                    | 0.20   |
| Source citation       | 16.1%           | 16.1%                     | 16.1%                    | 0.15   |
| Parsimony             | 0.0%            | 0.0%                      | 0.0%                     | 0.10   |
| **Fitness**           | 0.5413          | 0.5341                    | **0.5599**               | —      |
| Δ vs 0.679 baseline   | −0.1377         | −0.1449                   | **−0.1191**              | —      |

### Headline finding (counter to the going-in hypothesis)

The plan
([`.cursor/plans/fitnessscorer_coarse-axis_b49ee82d.plan.md`](.cursor/plans/fitnessscorer_coarse-axis_b49ee82d.plan.md))
hypothesised that *Lenient* would lift verdict-agreement because the reference set is
"already coarse-axis-shaped." The empirics flipped the sign: **Strict** lifts (+6.2pp),
**Lenient** drops (−2.4pp). The data explains why.

Reference label distribution (29 claims):

| Reference label | Count | Lenient projection | Strict projection |
|-----------------|------:|--------------------|-------------------|
| FALSE           | 9     | False              | False             |
| MISLEADING      | 7     | Falsey             | Falsey            |
| TRUE            | 4     | True               | True              |
| UNSUPPORTED     | 4     | Unverifiable       | Unverifiable      |
| PARTLY TRUE     | 2     | Truthy             | Truthy            |
| FALSE / MISLEADING | 2  | Falsey             | Falsey            |
| UNVERIFIABLE    | 1     | Unverifiable       | Unverifiable      |

Consensus (4-adapter) label distribution on the same 29 claims:

| Consensus label | Count |
|-----------------|------:|
| False           | 9     |
| **Exaggerated** | **6** |
| True            | 5     |
| Mostly True     | 4     |
| Unverifiable    | 3     |
| Models split    | 2     |
| *(Misleading)*  | 0     |

The crux: the reference set has **9 MISLEADING-family claims and zero EXAGGERATED**,
while consensus has **6 Exaggerated and zero Misleading**. This is the editorial-bar
disagreement. Under each axis:

- **Fine (6-bucket):** consensus `Exaggerated` (pos 2) vs reference `Misleading` (pos 3)
  → distance 1/5 = 0.2, score 0.8. Partial credit for being adjacent on the rhetorical
  scale.
- **Lenient:** consensus `Exaggerated → Truthy` (pos 1) vs reference `Misleading → Falsey`
  (pos 3) → distance 2/4 = 0.5, score 0.5. *Worse* than fine because the projection
  pushes them onto opposite sides of the Truthy/Falsey divide.
- **Strict:** consensus `Exaggerated → Falsey` and reference `Misleading → Falsey`
  collapse into the same bucket → distance 0, score 1.0. *Best* match because Strict's
  editorial bar lines up with the reference's.

### Architectural takeaway: two defaults for two audiences

The published headline pill defaults to **Lenient** (commit
[`dc64ca0`](https://github.com/aRealGem/Truth-bot/commit/dc64ca0)) because that's the
reader-facing surface — Lenient surfaces the agreement *within* the model panel that the
6-bucket axis hides, and respects the consensus's own editorial choice that "Exaggerated"
sits closer to "Mostly True" than to "Misleading."

The regression scorecard, by contrast, is the *reference-facing* surface. Strict is the
better default there because the reference set's editorial bar is closer to Strict's
projection — the reference labels rhetorical claims as `MISLEADING`, where consensus
labels them as `Exaggerated`. Strict-axis scoring puts both on `Falsey` and isolates real
verdict disagreement from label-vocabulary drift.

`score_run.py` keeps `--axis fine` as its CLI default for byte-identical backward
compatibility with the Run 4 numbers, but the *interpretation* default for ongoing
regression tracking should be `--axis coarse_strict` (or `--all-axes` with Strict as the
headline). The fine-axis score remains useful as a fine-grained drift signal between
`Mostly True` ↔ `Exaggerated` ↔ `Misleading`, which Strict deliberately collapses.

### What this rules out — and what it doesn't

- **Rules out:** "the −18pp gap is mostly fine-axis label drift between Mostly True and
  Exaggerated." It isn't. Lenient (which collapses MT + Excg → Truthy) makes the gap
  *worse*, so MT/Excg drift is small.
- **Confirms:** the gap is partially driven by the **Misleading vs Exaggerated**
  consensus-vs-reference editorial-threshold divergence — Strict closes ≈14% of it.
- **Doesn't rule out:** that the remaining ≈86% of the gap (still −0.1191 fitness even
  under Strict) is genuine quality drift between the 2026-04-19 single-model baseline and
  the current 4-adapter consensus pipeline. The Anthropic-only single-claim re-score
  (originally listed under gap #3) is the right next experiment to nail that down.

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
