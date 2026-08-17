# D17-c Stage B decision package

**Status: DRAFT, awaiting the batched owner gate. Nothing here is executed.**

Stage A is closed. This is the package the owner gate decides against: what
Stage A found, what it licenses, what wave 2 would contain, and what each lane
costs. No Stage B work has begun, nothing is published, and the stored packs
are byte-unchanged.

Evidence for every number below: `metrics/remediation_v2/d17c_stage0/`
(`stage_a_census.json`, `stage_a_control.json`, `stage_a_attribution.json`,
`goldens.json`, `b2_settlement.json`). Selector run-sha
`9f1c9a6d975ada4fe4e70170215da9d6796010ac49a1260c1265496c9ce9020c`.

---

## 1. What Stage A established

Treatment (series rows appended to the scoring payload) produced 8 stance
flips across 67 pack items. The control — same 7 claims, same payload path,
same cap, **zero** augmentation — produced **0**. So every flip is
excerpt-attributable and none is rescore noise.

| | treatment | control |
|---|---|---|
| stance flips | 8 | 0 |
| of which on excerpted items | 6 | — |
| ledger spend | $0.053984 | $0.026926 |

Cumulative $0.080910, 54% of the $0.15 ceiling. Realized 1.056× against a
measured-byte projection of $0.0511.

**Attribution scope, as ruled: causal at claim level, descriptive at item
level.** Scoring is whole-pack, so an unexcerpted item still shares a prompt
with its pack's excerpts. Stage B may attribute movement to *a claim's rows*;
it may not attribute movement to *an individual item*.

---

## 2. Flip breakdown, by what each group licenses

### 2a. Pack-reuse — 3 items, ready as-is

Stance resolved from null to definite, reasoning coherent with the rows, no
tension flag. These need no further adjudication to be reused within their
existing pack.

| item | series | stance | the rows say |
|---|---|---|---|
| `biden_2022:0245` E7 | FYFSD | **False** | FY2020 −3,131,917M → FY2021 −2,772,179M: ~$360B improvement, against a claimed "more than one trillion dollars in a single year" |
| `gwbush_2006:0133` E8 | PAYEMS | **True** | 130,255K (Dec 2003) → 134,468K (Dec 2005), +4,213K over 24 months |
| `trump_2026:0221` E9 | CUUR0000SAF112 | **False** | poultry CPI 339.169 (Jan 2025) → 346.613 (Jan 2026), **+2.2%**, against "lower today than when I took office" |

Two of the three **refute** their claim. That is the property that makes the
mechanism worth shipping: the rows decide against the speaker as readily as
for.

### 2b. Reason tension — 2 items, route to panel

Stance flipped to *supports* while the item's own `one_line_why` undercuts it.
Both carry `arithmetic_hinge=True`, so the B2 contract already treats them as
hypotheses rather than proof; `stance_reason_tension=true` names the specific
tension. **These must not be read as settled.**

| item | flipped to | its own stated reasoning |
|---|---|---|
| `biden_2022:0169` E7 | supports | "a gain of 356,000 — **not 369,000**" |
| `trump_2026:0219` E1 | supports | 58.6% "which **rounds to** the claimed 60%" |

Both are the scorer being generous in the speaker's direction. Panel
adjudication, not automatic reuse.

### 2c. `trump_2026:0054` — full escalation

The whole pack escalates together, because this is where the gate moves and
where spillover is visible.

* **Gate:** 1 → 3 bearing Tier-1..3 items, `forced_unverifiable` → **pass**.
  The only claim of the seven whose gate outcome changes.
* **Computed exhibit:** `max(CE16OV @ vintage 2026-02-24) = 164,520 @ 2026-01`,
  which is also the last observation in the window. Verified against
  `goldens.json`, not asserted.
* **E4** (excerpted, CE16OV): null → **True**, coherent with the exhibit.
* **E2** (not excerpted, `spillover_anomaly=true`): its snippet asserts the
  Jan-2026 peak and the rows confirm it, yet the treatment stance came back
  **refuting** — contradicting both. Named, not explained; item-level
  attribution cannot say why.
* **E10** (not excerpted, `spillover_correction=true`): snippet cites Dec-2025
  at 163,992, matching the rows exactly. The **stored** `False` was the
  incoherent stance; moving to `True` reads as a **correction**.

E2 and E10 together are the argument for escalating the pack rather than the
items: the same mechanism produced one incoherent stance and one repair, and
the current design cannot distinguish them at item level.

---

## 3. Wave-2 composition, in dependency order

1. **Stable claim ids (`ids-from-sids`) — FIRST, blocking.** Any re-render
   rotates deep links; doing ids first means one rotation, not two. Nothing
   below may re-render ahead of it.
2. **`series_rows` structured representation.** Stage A appended to `snippet`
   only because the census had to measure against the shipped baseline. The
   production path should carry rows in a dedicated key — cleaner provenance,
   and the scorer stops parsing rows out of prose. **Inherits whole-pack
   semantics** until the isolation ablation is measured, so it must not be
   built on an item-level attribution assumption.
> **PUBLISH NOTE — read before the publish click.** After lane 3, a legacy
> report renders **every source as unverified** until it is re-rendered with a
> classification map. This is deliberate and is lane 3 working as ruled: the
> old behaviour returned "verified" when no classification record existed,
> which is how a URL returning 404 on both FRED and ALFRED wore the
> source-verified badge on the published site, twice. It will look like a
> regression to anyone who does not know it is intentional. It is not.

3. **Badge fail-closed.** `_classify_source_for_render` currently returns
   `"verified"` both when no classification map exists and when a URL is merely
   absent from one — absence of evidence rendering as evidence of
   verification. Invert: no classification record → no `"verified"`; known-dead
   URLs render broken. Rides the stable-ids re-render.
4. **Corrections ledger — OWNER-APPROVED 2026-08-13**, text final at
   `D17-c-corrections-ledger.md`. Three items: the PR #105 "48 are statistical
   series" wording; the `LNS12000000` dead link (both occurrences, with the
   working siblings cited as context only, no correction implied for them); the
   pre-existing dropped-row note. Item 2 says "not retrieved before
   publication" rather than "never retrieved" — the owner chose the weaker,
   equally true form, since the stronger one rests on inference about our own
   pipeline. Browsing-model provenance is deliberately out of the published
   text and stays in the D17 record. **No dependency on stable ids** — this can
   ship ahead of the re-render or alongside it.

---

## 4. Spend estimate per lane

Measured-byte projections use **1.25×** (2.351× retired — it prices estimate
error, not measurement error; 1 of 3 realized factors banked at 1.056×).

| lane | basis | projected | at 1.25× |
|---|---|---|---|
| stable ids | $0, offline — no model call | $0.0000 | $0.0000 |
| `series_rows` rebuild | pack rebuild, not yet measured | **unpriced** | **unpriced** |
| badge fail-closed | $0, render-path change + tests | $0.0000 | $0.0000 |
| corrections ledger | $0, prose | $0.0000 | $0.0000 |
| isolation ablation (D17 cand.) | 67 items scored singly vs packed | ~$0.03 | ~$0.04 |
| Stage A re-run, if goldens change | measured at $0.0539 | $0.0539 | $0.0674 |

**`series_rows` is deliberately unpriced.** It is a pack rebuild rather than a
rescore, so no measured byte count exists for it yet and quoting a number here
would be the same estimate-error mistake that produced 2.351×. It needs a $0
`--estimate` pass before it is costed.

---

## 4a. Proposed S-12 — measure it or say you didn't

**Proposed, not active. Ratified by the owner at publish.**

> Where a fact can be measured, computed or imported, no artifact may assert it
> from a proxy; either the proxy's generational assumption is written down, or
> the proxy is replaced.

Two failures in this wave were the same failure. A guard inferred "no prior
artifact was edited" from *the parent looks unscored* — a proxy whose
generational assumption (every head sits one rebuild above the unscored
artifact) went unwritten until a deeper chain broke it. And a cost estimate
inferred the price of a `series_rows` payload from a constant measured on packs
that had none, running 8.2× over.

**First enforcement point, and the working example to cite:** `costs.py` now
refuses a constant that cannot name the payload schema it was measured under —

```
{constant} declares no payload schema — a constant that cannot name what it
measured is a proxy, not a measurement
```

That is S-12 as a runtime refusal rather than a principle, at the one place it
bites. `scripts/ancestor_locks.py` is the same idea applied to the other
failure: the proxy replaced by the direct check it stood in for.

## 5. What this package does NOT ask for

No Stage B execution, no publish, no render, no verdict rewrite, no gate
application. `obama_2014:0189` remains `window_period_mismatch=true` and
**non-actionable**. Everything stays on `d17c-stage0`.

**Open for the owner gate:** whether `series_rows` is built before or after the
isolation ablation. Building first risks baking in pack-level semantics we have
not measured; measuring first costs ~$0.04 and a round trip.
