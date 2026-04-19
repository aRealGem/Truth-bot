# eval/evolver — PARKING LOT

**Parked: 2026-04-19**

The genetic algorithm / prompt evolver approach is parked. A new evaluation
approach is in planning — this directory and all its infrastructure remain
intact as a reference and potential parts-bin.

---

## What This Was

A genetic algorithm that evolved extraction and synthesis prompt configurations
(genomes) to maximize a 5-dimension fitness score against a reference benchmark.

**Genome space:** 14 genes (7 extraction + 7 synthesis), each selecting from
2–5 prompt variants. Total search space: ~500k configurations.

**Fitness formula:**
- Claim recall        × 0.25
- Verdict agreement   × 0.30
- Explanation quality × 0.20
- Source citation     × 0.15
- Parsimony           × 0.10

**Best result achieved:** fitness=0.679 (gen-1 seed, 4 individuals,
individual ecd8b5e_s58c7ac). See eval/sotu-2026/BENCHMARK.md.

---

## Why Parked

Decision by operator (Jackie), 2026-04-19. A better approach is forthcoming.

The GA infrastructure itself worked correctly after fixes (dry-run warning,
parsimony calibration, retry logic, preflight checks). The approach was not
abandoned due to technical failure — it was a strategic pivot.

---

## What's Reusable

Everything in this directory is intact and tested:

| File | Reusable for |
|------|--------------|
| base_eval.py | Any model eval — ModelClient protocol, shared prompts, BaseEvalRunner |
| fitness.py | Scoring any extraction+synthesis run against a reference |
| preflight.py | Pre-run sanity checks for any eval script |
| runner.py | CachedRunner with retry — any LLM call loop |
| genome.py | Prompt variant indexing and rendering |
| ga.py | Tournament select, crossover, mutation — reusable for any search |

Tests in eval/tests/ cover all of the above (81 tests, all passing).

---

## Do Not Delete

The opus-4-7 gen-1 seed results in eval/sotu-2026/opus-4-7-results/ are the
current reference baseline (0.679). Preserve them for comparison.
