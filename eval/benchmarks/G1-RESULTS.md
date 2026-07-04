# G1 / equivalence / G5 — first measured run (2026-07-04)

Lane: **L-P** via LiteLLM proxy, virtual key `hydramind-c1` (project=P96.2, max_budget $10).
Model: `claude-haiku` (A2). **Model-fallback guard: PASS** — every call returned the requested
family (no silent reroute; the Flash-registration class of gap is absent for the aliases used).

## G1 — Layer A on heldout (claim_set.heldout, n=94, read once — I6, RC=`rc1-2026-07-04`)

| config | accuracy | macro-F1 | recall_cw | prec_cw |
|---|---|---|---|---|
| A2-only (haiku) | 0.755 | 0.744 | 0.680 | 0.850 |
| **composed A1+A2** | 0.606 | 0.618 | **0.920** | 0.460 |

Provisional bars (record, don't chase): recall_cw ≥ 0.90, macro-F1 ≥ 0.75.
- Composed **meets the recall_cw safety bar (0.920)** — a missed check-worthy claim is a silent
  failure, so recall is the metric that matters most; A1's PASS-override (score ≥ 0.65 ⇒
  check-worthy) is what lifts recall.
- Composed macro-F1 (0.618) is **under bar**: the same PASS-override forces 24 opinions →
  check-worthy (prec_cw 0.46), depressing opinion recall. A2-only has the better F1 (0.744) but
  poor recall_cw (0.680 — haiku under-calls check-worthy).
- Lever for next RC (not chased now): raise A2 tier to `claude-sonnet`, or replace the hard PASS
  override with "A1 drops obvious non-claims, A2 decides the rest." Both are one-line tunes.

Confusion (composed, gold→pred):
```
             check  opini  unimp
check-worthy   23     0      2
opinion        24    19      8
unimportant     3     0     15
```

## Batch≡interactive equivalence (§2 invariant)

10 TRAIN items through **L-P and L-B** (Anthropic Message Batches), identical prompts →
`{"n": 10, "mismatches": [], "equivalent": true}`. Lane is a cost/latency choice only.

## G5 — cost (first datapoint)

`hydramind-c1` spend after the heldout run ≈ **$0.070** → **$0.0746 per 100 sentences** (L-P/haiku).
Batch lane (L-B) bills ~50% of input+output vs interactive.

Run: `PYTHONPATH=.:src python eval/benchmarks/run_g1.py composed --rc <id>` (needs repo `.env`).
Manifest: `examples/manifest.heldout.json`.
