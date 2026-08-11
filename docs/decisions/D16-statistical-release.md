# D16(α) — Statistical-agency release in the post-speech band

**Status:** **RATIFIED 2026-08-09** by the owner, the same day as D15. Implemented, tested, and **ENABLED BY DEFAULT**.
**Flag:** `TRUTHBOT_D16_STATISTICAL_RELEASE` — unset means ON. Retained as an OVERRIDE: `=0` reproduces the pre-ratification gate exactly.
**Flip set under the ratified rules:** `scripts/regate_from_rescore.py` → `metrics/remediation_v2/regate_flipset.{json,md}`.
**Deferred alongside it:** `docs/decisions/D17-candidates.md` (FRASER path-level allowance; document-class detection).
**Blast radius measured:** `scripts/measure_d16.py` → `metrics/remediation_v2/d16_blast_radius.json` ($0, no model calls).
**Combined with D15:** `scripts/d15_d16_era_breakdown.py` → `metrics/remediation_v2/d15_d16_era_breakdown.{json,md}` (the M-6 evenhandedness check).
**Code:** `src/truthbot/verdict/statistical_release.py` + `src/truthbot/verify/statistical_agency.py` + `statistical_agency_registry.yaml`, wired in `verdict/consolidator.py`.

---

## 1. The problem: the era rule is silencing the jobs report

Remediation item 1.3 marks every pack item published after the utterance but
inside the fair-game window `post-speech · context-only`: kept, displayed, and
unable to credit `MIN_BEARING_T13`. The target was same-speech fact-checks and
reaction coverage — evidence that could not have existed when the speaker spoke
and that judges the speech with hindsight. That target is correct.

The rule keys on **publication date**, so it also silences a second, entirely
different population: government statistical publications that report
**pre-utterance facts** and merely happen to be printed a few days later. The
January 2006 Employment Situation is published on 3 February 2006 and measures
January 2006. A 31 January 2006 speech about January payrolls cannot be checked
against it — not because the document contains post-utterance world-state, but
because the calendar says so.

Across the five rebuilt runs the post-speech band holds **546 items**. Among
them are BLS payroll and ECI releases, BEA GDP and personal-income releases,
CBO outlooks and monthly budget reviews, and EIA outlooks — all measuring
periods that had already ended when the speaker spoke.

## 2. What was rejected, and why this is different

The blanket form — *any Government-tier post-speech item may credit* — was
**rejected**, because the two motivating examples turned out to be the
principal's own executive documents:

| Claim | Document | Served from |
|---|---|---|
| `gwbush_2006:0217` | ONDCP National Drug Control Strategy (Feb 2006) | `justice.gov`, `files.eric.ed.gov` |
| `clinton_1998:0101` | FY1999 President's Budget | `gpo.gov` |

`principals.principal_relation` keys on **host**, so both read *independent*; a
relation test cannot catch them. Catching them by **document class** is real
work and is deliberately **deferred** (logged D17-candidate).

D16(α) needs no detector, because it **inverts the test**. Instead of asking
what the document is *not*, it asks whether the **publisher's function is
statistical measurement** — and the President's Budget and the ONDCP Strategy
are not statistical-agency records no matter which host serves them. The
exclusion falls out of the allowlist for free.

## 3. The rule — three conditions, all required

### Condition 1 — function

The host resolves through `src/truthbot/verify/statistical_agency_registry.yaml`
(schema `truthbot-statistical-agency-registry v1`), a versioned, **fail-closed**
allowlist in the style of the tier registry. Seeded with the federal
statistical system and the congressional analytical agencies:

| Agency | Hosts |
|---|---|
| BLS, BEA, Census, EIA, USDA-NASS, NCES | `bls.gov`, `bea.gov`, `census.gov`, `eia.gov`, `nass.usda.gov`, `nces.ed.gov` |
| CBO, GAO, CRS | `cbo.gov`, `gao.gov`, `crsreports.congress.gov` |
| FRED / ALFRED | `fred.stlouisfed.org`, `alfred.stlouisfed.org` |
| NCHS / CDC statistical products | `stacks.cdc.gov`, `wonder.cdc.gov`, `data.cdc.gov`, and `cdc.gov` / `archive.cdc.gov` **path-scoped** to `/nchs`, `/mmwr` and the named survey systems |

**Structurally excluded, and this is the point of the design:**

| Excluded | How | Why |
|---|---|---|
| Executive Office of the President units — OMB, ONDCP, CEA | exact dot-label deny + `eop.gov` | they author the President's Budget, the National Drug Control Strategy, the Economic Report of the President |
| anything `*whitehouse*` | host substring deny | every administration's press shop and every NARA mirror of one |
| agency **press offices** | `stat_press_prefixes`, inherited from `tier_registry.yaml` | the newsroom is not the measurement function |
| document **archives** that reprint executive documents alongside statistical ones | explicit deny: `fraser.stlouisfed.org` | it serves the January 2006 Employment Situation (`gwbush_2006:0133`) *and* the OMB budget appendix (`gwbush_2006:0155`) *and* the CEA's Economic Report of the President (`clinton_1998:0167`) — a host that serves both cannot be a function test |
| everything else | the default answer is **no** | it is an allowlist, not a classifier |

Two authoring choices worth stating, because both are load-bearing negatives:

* the NCES entry is scoped to `nces.ed.gov`, **not** `ed.gov` — which is what
  keeps `files.eric.ed.gov` (and with it the ONDCP Strategy) out;
* the CRS entry names `crsreports.congress.gov` only — `congress.gov` as a whole
  also serves the Congressional Record (which D15 classifies as a record of the
  utterance itself), and `everycrsreport.com` is a third-party mirror, not the
  agency.

**Press paths are inherited, never restated.** The tier registry's
`stat_press_prefixes` already encode the hard-won distinction that
`bls.gov/news.release/*` is the jobs report while `bls.gov/newsroom/*` is the
press shop — and that BEA publishes its statistical releases under `/news/`. A
second copy of that list would drift, and the drift would be silent.

### Condition 2 — a parseable data period at or before the utterance

The item must name a data period, in its snippet **or** in its own URL words,
that had **started** on or before the utterance date. Six families, each named
and separately reported:

| Rule | Example | Period start |
|---|---|---|
| `stat-period-month` | "Employment Situation (Jan 2006)" | 1st of that month |
| `stat-period-quarter` | "Q4 2025", "fourth quarter of 1997" | 1st of that quarter |
| `stat-period-fiscal-year` | "FY1998", "fiscal year 2009" | 1 October of the prior year |
| `stat-period-year-data` | "the 2005 MTF survey", "estimates for 2013" | 1 January |
| `stat-period-anchor-year` | "dropped 19% since 2001" | 1 January |
| `stat-period-year-range` | "covering 2003–2006" | 1 January of the first year |

**This REPLACES an earlier heuristic** — *any 4-digit year ≤ the utterance year
anywhere in the snippet* — which **must not survive**: `gwbush_2006:0217`
passed it on the strength of its own **publication year**, "2006". For the same
reason, every calendar-date shape is **masked out before parsing**: the
`[YYYY-MM-DD]` prefix the connectors stamp onto every snippet, bare ISO dates,
`3/4/2022`, "Mar 4, 2022", "4 March 2022". A publication date is not a data
period. The anchor list deliberately omits "in" — "published in 2006" would
rebuild the very heuristic being replaced.

**Fail closed:** no parseable period → no credit. Real statistical releases are
missed by this. `biden_2022:0266`'s CDC Weekly Review names its reference week
as "Feb 23–Mar 1" with no year and is not released; `obama_2014:0080`'s EIA note
says only "trends improving in 2013". That is the intended failure mode — a
false positive lets post-utterance world-state decide a verdict, which is the
harm item 1.3 exists to prevent.

The parser also earns its keep in the negative direction. `biden_2022:0079` is
an **EIA** page — an allowlisted host — published a week after the speech,
announcing the SPR release that *followed* it. It names no data period, so it
stays context-only. Condition 2 is what stops the allowlist from becoming a
blanket pass for its hosts.

### Condition 3 — the S-2 cap, untouched

The item must still be inside the fair-game window
(`utterance < published ≤ utterance + 7`). **S-2 is not modified.** D16 releases
the quota credit of items *already inside* the band; it never moves the band's
edges. An item dated past the cap matches nothing here, and an item dated
*before* the utterance matches nothing either — it already credits.

### Effect, and precedence against D15

A matching item may credit the quota exactly as a pre-utterance item of the same
tier and stance would — on every quota branch, including the D11.2 role-aware
`corroborant` and `primary-record` counts. Nothing else about it changes, except
that its `era_note` becomes `post-speech · statistical release (pre-utterance
data period)`: after the release, "context-only" would be a false statement
about an item the gate actually spent.

**Where D15 and D16 both apply, D15 wins.** A record of the utterance credits
nothing on any branch, and D16 must not hand back what D15 took away. This is
decided on the item at construction, not at the quota, so it is visible in the
journal rather than implied by branch order.

## 4. Measured blast radius ($0)

Same method as D15, deliberately: two stance vintages (`stored` = what the
rebuilt artifacts recorded, `rescored` = with the B1a sidecars overlaid), each
stored pack run through the **real** gate twice, switch off then switch on.

**Corpus totals — 529 claims:**

| | items released | claims touched | bearing | bearing **and** Tier-1..3 | gate outcomes changed |
|---|---:|---:|---:|---:|---:|
| stored stances | 24 | 19 | 16 | 16 | **2** |
| after B1a re-score | 24 | 19 | 18 | 18 | **2** |

**By agency:** BLS 10, CBO 7, BEA 5, EIA 2. Every released item is
GOVERNMENT tier.
**By rule:** `stat-period-month` 19, `stat-period-quarter` 2,
`stat-period-year-data` 2, `stat-period-fiscal-year` 1.

**Per speech (items released / gate Δ):**

| speech | items | claims touched | bearing (stored → rescored) | gate Δ |
|---|---:|---:|---|---:|
| gwbush_2006 | 4 | 3 | 3 → 1 | 0 |
| clinton_1998 | 6 | 5 | 6 → 6 | **2** |
| obama_2014 | 3 | 3 | 3 → 3 | 0 |
| biden_2022 | 7 | 4 | 3 → 5 | 0 |
| trump_2026 | 4 | 4 | 1 → 3 | 0 |

**Direction of every change is one-way: 2 released, 0 newly gated.** That is the
expected signature of a rule that only ever *adds* credit, and it is asserted
rather than assumed — the measurement reports any newly-gated claim as a defect.

**The two claims, both `clinton_1998`, both currently shipping UNVERIFIABLE:**

| sid | claim | released by |
|---|---|---|
| `clinton_1998:0026` | "This year, our deficit is projected to be $10 billion and heading lower." | CBO, January 1998 Economic and Budget Outlook (×2 urls) |
| `clinton_1998:0038` | "Now, if we balance the budget for next year, it is projected that we'll then have a sizable surplus…" | CBO, January 1998 Economic and Budget Outlook |

Both are budget-projection claims answered by the CBO outlook published the day
after the speech, reporting a January baseline. Neither is a verdict the site
currently publishes; releasing them means they become **eligible** for
adjudication, not that they become decided — that costs a panel call.

### The prior upper bound was 13

Under the loose heuristic (any 4-digit year ≤ the utterance year, any
Government-tier host) the estimate was **13** claims. The measured figure under
D16(α) is **2**. The gap is the cost of the two hardenings the reviewer's
objection forced: the function allowlist removes the executive-document hosts,
and the structured period parser removes the items that qualified only on their
own publication year. Fewer claims move, and the ones that do move can be named
and defended one at a time.

## 5. What ratification enabled

Ratified 2026-08-09. The default is now ON, and it:

- restore quota eligibility to 24 statistical-agency items across 19 claims,
  relabelling them in the pack payload so a reader can see why;
- move **2** claims from gated to eligible-for-adjudication (both
  `clinton_1998`, both currently UNVERIFIABLE);
- leave the S-2 fair-game window, the tier ladder and every other gate rule
  exactly as they are.

The suite is green with the flag both off and on, and `consolidate()` accepts
the same switch as an explicit argument, so a measurement never has to set an
environment variable that other consolidations in the process would inherit.

**Confirmed by the ratified re-gate.** Both claims land where this section said
they would: `clinton_1998:0026` and `clinton_1998:0038` are in the released set
under the combined rules, and neither collides with anything D15 takes back
(`metrics/remediation_v2/t1_intersections.md`). Corpus flip set with both rules
active: 23 released / 64 still gated / 65 newly gated / 377 unchanged.

### Open questions for the owner

1. **Ratify the allowlist whole, or host by host?** The measured effect comes
   from four agencies only (BLS, CBO, BEA, EIA). Census, GAO, CRS, FRED/ALFRED,
   NCES, NCHS and USDA-NASS are seeded on function but do no work in this
   corpus — they could be held back without changing a single outcome, at the
   cost of re-opening the question on the next speech.
2. **Is the fail-closed period parser too strict?** It declines
   `biden_2022:0266` (CDC Weekly Review, reference week with no year) and
   `obama_2014:0080` (EIA, "in 2013"). Loosening the anchor list to include
   "in YYYY" would catch both — and would move measurably back toward the bare
   year heuristic this design replaced.
3. **Released ≠ decided.** The 2 claims become eligible; turning that into a
   published verdict is a metered adjudication. Should they be re-run, or left
   released-but-unadjudicated until a broader re-run happens anyway?
4. **D17-candidate stays open.** Document-class detection is what would let a
   *statistical* document served from a non-statistical host (the BLS release
   on FRASER, `gwbush_2006:0133`) be credited. D16(α) declines it on purpose.
