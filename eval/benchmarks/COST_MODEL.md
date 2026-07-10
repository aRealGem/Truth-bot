# Closed-book PCA cost model

Replaces the old "~$120 for 10 speeches" hypothesis (P96.2.1) with a projection
calibrated against the **live-validated** dev-lot. Model + numbers live in
`cost_model.py` (`python eval/benchmarks/cost_model.py`); the dev projection is
pinned to the measured cost by `tests/benchmarks/test_cost_model.py`.

## Formula

Per claim: `proposer + Σ critics + escalation_rate × arbiter`, where each call =
`tokens_in × rate_in + tokens_out × rate_out`. Token profile (per call, closed-book):
proposer 200/90, critic 200/90, arbiter 320/150 — calibrated so the dev roster
reproduces the measured dev-lot cost (~$0.010 / 25 claims, escalation 0.32).

## Projections (list prices 2026-07-09; closed-book Layer B)

| roster | escalation | $/claim | $/100-claim speech | $/10 speeches |
|---|---|---|---|---|
| **DEV (validated)** — mistral / dsv4-flash / claude-haiku | 0.32 | $0.00041 | $0.041 | **$0.41** |
| PROD, opus arbiter (subscription $0) — gpt-5.4-mini / [grok-4.5, dsv4-flash] / opus | 0.32 | $0.00153 | $0.153 | **$1.53** |
| PROD, gpt arbiter — …/ gpt-5.4 | 0.32 | $0.00251 | $0.251 | **$2.51** |
| PROD, gpt arbiter, old gate | 0.80 | $0.00397 | $0.397 | **$3.97** |

**The $120 hypothesis is ~30–80× too high** even for the priciest prod shape.

## What drives cost

- **The every-claim grok critic dominates prod** (~$0.00094/claim at grok-4.5 $2/$6),
  not the DeepInfra seats. This matches the P96.2.1 note ("cost concentrates in Grok").
- **Opus as arbiter is subscription-covered (Lane-Worker) → $0**, so with an Opus
  arbiter the escalation rate doesn't affect dollars (only latency/quota). With a
  *paid* frontier arbiter (gpt-5.4), the `label_mismatch` criterion (escalation 0.32
  vs the old 0.80) cuts the arbiter term ~2.5× — a real saving.
- Dev seats are rounding error: mistral $0.000033/claim, dsv4-flash $0.000034/claim.

## Caveats / confirm at provisioning

- **Closed-book only.** Layer C (evidence-grounded) will raise input tokens a lot —
  widen `TOKENS` before trusting a Layer C number.
- **Prod roster P/A seats are still TBD** in `rosters.yaml`; the PROD rows are
  illustrative shapes, not the committed roster.
- **Rates are list prices** fetched 2026-07-09. Confirm the seats a prod roster
  actually uses — grok and the gpt tier especially (P96.2.1 flagged this).
- **Proxy alias staleness** (see `PROXY_PRICING_AUDIT.md`): the proxy's `grok`,
  `gemini-pro`, `gemini-flash`, `gpt-4o` aliases point at *older* models than the
  rates above assume — reconcile the roster aliases with the intended models.
