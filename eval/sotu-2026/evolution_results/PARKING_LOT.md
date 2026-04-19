# PARKING LOT — evolution_results/

**Status:** Not meaningful. Do not use for prompt optimization decisions.

## Why

This run was executed with the `--dry-run` flag:

```
python eval/prompt_evolver.py --generations 5 --population 8 --dry-run
```

In dry-run mode, `CachedRunner.extract_claims()` returns **3 identical stub claims** for every
genome, regardless of the actual prompt content. As a result:

- All 40 individuals (8 pop × 5 gens) received **identical inputs** to the fitness scorer
- All individuals scored exactly **0.5456** — uniform across every generation
- **Selection pressure was zero** — no genome was ever preferred over another
- Evolution ran for 5 generations but **never actually evolved anything**

The files in this directory (`generation_01.json` through `generation_05.json`,
`evolution_log.json`, `best_prompts.json`, etc.) are structurally valid but contain
meaningless data.

## Cost

$0.00 — no API calls were made.

## What to do instead

Use `opus-4-7-results/` as the active reference (Run 3, fitness=0.679, real API calls,
real transcript evaluation). See `../BENCHMARK.md` for the full comparison.
