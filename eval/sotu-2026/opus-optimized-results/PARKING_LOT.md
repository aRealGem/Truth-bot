# PARKING LOT — opus-optimized-results/

**Status:** Baseline reference only. Do not compare fitness scores directly to evolver runs.

## Why

This run was produced by `eval/opus_eval.py` — a standalone single-pass evaluator with
**hand-tuned fixed prompts**, not the genetic evolver.

Two specific problems make this run non-comparable to evolver output:

### 1. Single-pass fixed prompts

There was no evolutionary search. The same prompt was used for every claim. Results reflect
how well a specific hand-crafted prompt performs, not what the best possible prompt achieves.

### 2. Simplified fitness formula

`opus_eval.py` computes an approximate fitness as:

```
fitness = recall × 0.25 + verdict_agreement × 0.30
```

The evolver's `FitnessScorer` uses a richer formula that also weights:
- Explanation quality
- Source citation quality
- Parsimony penalty

The reported fitness of **~0.492** is **not comparable** to the evolver's fitness scale
(where the best Run 3 individual scored **0.679**). The two numbers measure different things.

## What it IS useful for

- Qualitative check: Opus with hand-tuned prompts achieves **100% claim recall (29/29)** and
  **80.7% verdict agreement** — a strong sanity check that the pipeline works.
- Establishing a floor: any evolved individual that scores below this threshold should be
  considered a regression.

## Cost

$2.15 at claude-opus-4-7 pricing (real API calls, full 29-claim synthesis).

## Active reference

Use `opus-4-7-results/` (Run 3) as the active evolver benchmark. See `../BENCHMARK.md`.
