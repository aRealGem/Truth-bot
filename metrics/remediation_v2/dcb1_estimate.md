# DC-B1 spend estimate — evidence re-scoring and re-adjudication

**NOTHING HERE HAS BEEN RUN. NO SPEND HAS OCCURRED.**
No `score_evidence()` call was made against a live lane, no key file was sourced,
no model or API call of any kind was issued in producing this document. Both
code deliverables are written, tested offline against stubs, and remain gated:
the re-score script refuses to act without `--go --budget`, and the v2 builder's
scorer is `None` by default with both production flags shipping OFF.

Generated 2026-08-07. Machine-readable sibling: `dcb1_estimate.json`.

## Why this spend is being requested

`build_evidence_pack_v2` wired the R1/R2/R3 retriever shortlists straight into
`consolidate()`, so `verify/relevance.py::score_evidence` — the only writer of
`relevance_score` and `supports_claim` — was never on the v2 path. Measured over
the five rebuilt runs:

| | value |
|---|---|
| items stored | **4,344** |
| items carrying the 0.5 relevance default | **4,344 (100.00%)** |
| items with a null stance | **1,136 (26.2%)**, per speech 20.5–30.2% |

`consolidator._bearing()` requires stance `True`/`False`, so a null item can
never credit `MIN_BEARING_T13 = 2`. Packs holding good evidence therefore
gate-force Unverifiable. The verified example is `trump_2026:0469`, where NPR is
the only crediting item while AP, NBC and two govinfo records sit unscored.

Owner ruling Q-1 approved the permanent fix (**B1b**, wire the scorer into the
v2 builder — done, `scorer=` hook, default off) plus a one-off re-score of the
stored items (**B1a**, this estimate). Both are model spend.

## (a) B1a — re-score the 4,344 stored items

Per item: one cheap Haiku call per **sid** (not per item) through the LiteLLM
proxy, over the pack's items in a single prompt. Stored v2 packs cap at
`PACK_CAP_V2 = 10`, comfortably under the `DEFAULT_SCORE_CAP = 16`, so every sid
is exactly one call and nothing is truncated.

**Lane: `claude-haiku` via the LiteLLM proxy — ON-PROXY.** That matters for
honesty: the funded run's real cost will be **ledger-true**
(`proxy_lane.proxy_key_spend()`), not estimated, and the budget breaker reads
that same ledger. Unlike the phase-3 rebuild, there is no off-proxy R2/R3
component to approximate.

**Method.** `scripts/rescore_stored_packs.py --estimate` builds the *exact*
prompt the funded path would send — the same `relevance.score_payload()`
function `score_evidence` calls — for every sid in every stored artifact, and
measures its character volume. It is a measurement of real payloads, not a
per-claim guess. Tokens are `chars / 4.0`; rates are
`hydramind.models.RATE_TABLE_USD_PER_MTOK["claude-haiku"] = (0.80, 4.00)` USD
per Mtok in/out.

| speech | calls | items | tok_in | tok_out | est USD |
|---|---:|---:|---:|---:|---:|
| gwbush_2006 | 48 | 396 | 26,789 | 4,746 | 0.0404 |
| clinton_1998 | 92 | 792 | 52,335 | 9,476 | 0.0798 |
| obama_2014 | 96 | 799 | 52,134 | 9,572 | 0.0800 |
| biden_2022 | 111 | 885 | 58,600 | 10,622 | 0.0894 |
| trump_2026 | 182 | 1,472 | 98,548 | 17,656 | 0.1495 |
| **TOTAL** | **529** | **4,344** | **288,406** | **52,072** | **0.4391** |

**≈ $0.44 for the whole corpus re-score.**

**Uncertainty.** Two sources, both small and both bounded upward:

1. *Tokenization.* `chars / 4` is an approximation, not a tokenizer run. JSON
   scaffolding and URLs tokenize worse than prose, so the true input count is
   plausibly 10–25% higher. Even at **+50% this line is under $0.70.**
2. *Rate table.* The `claude-haiku` entry is flagged in `hydramind/models.py` as
   a fallback estimate, because Haiku is proxy-priced in practice. The proxy's
   own price is authoritative and the ledger will show it.

The output half is the confident half: the reply is a fixed, tiny JSON shape
(`{"i", "relevance", "supports"}` per item), so volume scales exactly with item
count. Because the lane is on-proxy, this estimate only needs to be good enough
to set a cap — and at this magnitude it plainly is.

## (b) PCA re-adjudication of the gate-flip subset

**This is the honest hard part: the flip set is UNKNOWN until B1a runs.** A claim
only needs re-adjudication if scoring gives it ≥2 bearing Tier-1..3 items where
it previously had fewer. Which claims those are is exactly what B1a discovers.
So this is bounded, not predicted.

### Correction to the brief

The brief called this "the 66 currently gate-forced claims". The artifacts do
not support that number as the candidate pool — **66 is the `newly_gated` count**
(claims the rebuild newly gated relative to the published run), summed from the
`phase3_*_verdict_diff.json` files. The count of claims **currently** gate-forced
in the rebuilt runs is **87**:

| speech | claims | gate-forced now | of which newly gated |
|---|---:|---:|---:|
| gwbush_2006 | 48 | 7 | 5 |
| clinton_1998 | 92 | 14 | 11 |
| obama_2014 | 96 | 12 | 7 |
| biden_2022 | 111 | 16 | 12 |
| trump_2026 | 182 | 38 | 31 |
| **TOTAL** | **529** | **87** | **66** |

The maximum candidate pool is **87** — every currently gate-forced claim could in
principle flip. The 66 is a meaningful subset (the regressions specifically) but
it is not the ceiling, so both are priced below.

### Per-claim cost, verified

From the committed phase-3 run logs, cost / claims per speech:

| speech | claims | run cost | $/claim | note |
|---|---:|---:|---:|---|
| gwbush_2006 | 48 | $3.0823 | **$0.0642** | clean single-session run |
| clinton_1998 | 92 | $6.8453 | **$0.0744** | clean single-session run |
| trump_2026 | 182 | $13.6136 | **$0.0748** | clean single-session run |
| obama_2014 | 96 | $1.8350 | $0.0191 | resumed — **undercounts** |
| biden_2022 | 111 | $4.0042 | $0.0361 | resumed — **undercounts** |

This confirms the brief's **$0.064–0.075/claim** on gpt-5-mini, from the three
runs that completed in one session. The two resumed legs are excluded on
purpose, and the reason is a real accounting gap worth recording:
`phase3_rebuild`'s resume carries `banked_cost` forward from the chunk journal,
but the journal banks only **proxy** spend — the off-proxy R2/R3 estimate from
the interrupted session is lost. Their reported totals are therefore too low and
must not be used to price future work.

### Bounded estimate, at $0.0642–$0.0748/claim

| scenario | claims | low | high |
|---|---:|---:|---:|
| Full pool — every currently gate-forced claim | 87 | $5.59 | $6.51 |
| Newly-gated subset only | 66 | $4.24 | $4.94 |
| Plausible partial — 50% of the pool flips | 44 | $2.83 | $3.29 |
| 50% of the newly-gated subset | 33 | $2.12 | $2.47 |

### Claims to re-adjudicate regardless of flips

Named by Fable for re-adjudication independent of the gate outcome:
`trump_2026:0030`, `trump_2026:0031` (with the exhibit), `trump_2026:0023`,
`trump_2026:0024`, `trump_2026:0343`, plus `clinton_1998:0313` (the CW2-decisive
sid from A3).

**All six are currently DECIDED, not gate-forced** — verified against the
artifacts, in the order listed: FALSE, TRUE, MISLEADING, TRUE, TRUE, TRUE. They
are therefore strictly **additive** to any flip set — no double-counting.

6 claims → **$0.39 – $0.45**.

## (c) Combined total, recommended cap, and scheduling

| line | low | high |
|---|---:|---:|
| B1a re-score, all 4,344 items | $0.44 | $0.44 |
| B1b re-adjudication — 50% partial (44) + 6 named | $3.21 | $3.74 |
| B1b re-adjudication — full pool (87) + 6 named | $5.97 | $6.96 |
| **Combined, midpoint scenario** | **$3.65** | **$4.18** |
| **Combined, worst case** | **$6.41** | **$7.40** |

### Recommended cap: **$8.00**

That covers the worst case (all 87 flip, all 6 extras, high per-claim rate) with
roughly 8% headroom for tokenization drift. It is a cap, not a forecast — the
expected outcome is nearer $4.

### Daily-cap ($20/day) implication

The whole job fits **inside a single day's cap with room to spare** — no
multi-day scheduling is required, which is a change from the phase-3 rebuild
($29.38 total, which did need spreading). Suggested sequencing, still one day:

1. **B1a re-score, ~$0.44.** Cheap enough to run whole. Start with
   `--speech gwbush_2006 --go --budget 0.25` as calibration (48 sids, est.
   $0.04): if ledger-true cost lands far from estimate, stop and re-price before
   the other four.
2. **Join the sidecars, identify the actual flip set.** $0 — pure local analysis
   against `rescored_<speech>.json`. This is the gate that turns the bounded
   estimate above into a real number.
3. **B1b re-adjudication of the flip set + the 6 named claims**, budget set from
   step 2's actual count at $0.075/claim, capped at $7.00.

Steps 1 and 3 are separately budgeted, so step 2's finding can shrink step 3 —
and if the flip set comes back empty, step 3 costs nothing but the 6 named
claims (~$0.45).

## What was built, and what remains gated

| deliverable | state |
|---|---|
| `scripts/rescore_stored_packs.py` | written, 19 offline tests, **not run** (only `--estimate`/plan, both $0) |
| `build_evidence_pack_v2(scorer=...)` | wired, 10 offline tests with a stub scorer, default `None` |
| `pipeline.py --score-evidence` | present, **defaults OFF**, asserted by test |
| `phase3_rebuild.py --score-evidence` | present, **defaults OFF**, asserted by test |

Nothing above spends until someone passes `--go --budget` (B1a) or
`--score-evidence` (B1b), and DC-B1 is signed.
