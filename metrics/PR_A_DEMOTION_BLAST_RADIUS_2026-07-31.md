# PR-A (S5 political-communications tier) — retrospective blast-radius measurement

**Date:** 2026-07-31 · **Author:** ccagent (for jackie) · **Cost:** $0 (offline, over stored artifacts)
**Method:** `scripts/measure_tier_demotion_decisiveness.py` re-classifies every evidence URL in
the two published P3-rerun artifacts (`23939712` trump_2026, `7208bbbb` biden_2022) under the
*new* tiering. A URL now resolving to `POLITICAL` (S5) is **demoted** by PR-A. Data:
`metrics/tier_demotion_decisiveness_2026-07-31.json`.

## Headline
PR-A is **not** a narrow "demote the White House press releases" change. Measured against the
live published run:

*Numbers below are AFTER the fixes applied on this branch (senate/uscode/fjc carve-ins +
the "data yes, press no" split — see below).*

| | trump_2026 | biden_2022 | total |
|---|---|---|---|
| evidence items | 1543 | 918 | 2461 |
| **demoted → S5** | 412 | 276 | **688 (28.0%)** |
| demoted **and verdict-decisive** (cited in a decided verdict's rationale) | 190 | 151 | **341** |

- **65%** of decided-with-citations claims (175/269) cite at least one now-demoted source.
- BUT only **5 claims (2%)** are *entirely* sole-sourced on demoted evidence — i.e. would
  collapse to UNVERIFIABLE. The rest are multiply-sourced: they lose a strand of support, not
  the verdict. **Mass verdict-collapse is not the risk; a distribution shift is.**

## What actually gets demoted (the 348 verdict-decisive citations, by class)
| class | count | examples | verdict |
|---|---|---|---|
| federal agency | ~185 | dhs.gov, justice.gov, cbp.gov, treasury, state.gov, energy.gov, dea.gov | mixed |
| **political-comms (correct)** | 85 | whitehouse.gov, *whitehouse.archives.gov, party organs | mixed |
| congressional member/cmte | ~31 | issa.house.gov, waysandmeans.house.gov, budget.house.gov | mixed |
| state/local | 31 | gov.texas.gov, mpdc.dc.gov, ca/ny/oh/mi .gov | mixed |
| military / intl | 15 | uscg.mil, defense.gov, nationalguard.mil | mixed |

Only **~24% (85/348)** are genuinely partisan political communications — the ruling's actual
target. The other ~76% are substantive/operational agency, record, and subnational sources
swept in by the **press-path rule** (`/news/`, `/newsroom`, `/briefing` …) and the
**unmapped-.gov quarantine**.

## Two findings that fall out of this
**1 — Over-demotions fixed on this branch.** The measurement caught `uscode.house.gov` (the
literal US Code — primary law) and `www.fjc.gov` (Federal Judicial Center) quarantined to S5;
both are nonpartisan record/court functions → carved into the registry per the codified
criterion. And per jackie's **"data yes, press no"** ruling (2026-07-31), a new rule promotes
any URL whose path carries a structured-data segment (`stats`, `data`, `timeseries`, …) to S1
even under a press prefix — so `cbp.gov/newsroom/stats/*` (border-encounter *data* on a
`/newsroom` path — the BLS problem one scope level out) now survives, while a genuine press
release/announcement with no data segment (`treasury/news/press-releases/…`, DOJ `/pr/…`) still
demotes. Effect on the register: 708→688 demoted, 348→341 decisive — small, which is itself the
finding: **most of the agency demotions are genuine announcements, not data**, so they are
correctly demoted under the ruling and the composition-shift is largely *intended*.

**2 — The demotion is speaker-asymmetric (this is the one to watch).**
- biden_2022: **151 decisive-demoted, 147 support TRUE verdicts.** His decided TRUEs lean
  heavily on administration/agency sources — including ~40 `*whitehouse.archives.gov` citations
  where the administration confirms its *own* claims (textbook self-sourcing, which PR-A is
  *right* to demote).
- trump_2026: **190 decisive-demoted, mixed** (98 TRUE / 54 FALSE / 38 MISLEADING) — cuts both
  his adverse and favourable verdicts.

So PR-A withdraws evidentiary weight *unevenly* across speakers. That is not a bug — the
underlying evidence really is distributed that way — but it means the aggregate headline shift
is not neutral, and it brushes **I3 (no speaker conditioning)** at the *composition* level. This
is exactly the composition-bias the second-opinion review flagged; the fix is **visibility, not
suppression**: per-run telemetry (packs carrying a quarantined item; decided-vs-UV rate for
claims depending on one) so the skew is on the record, not silent. (Jackie: telemetry =
fast-follow, not a merge gate.)

## Recommendation
PR-A's core is **sound and worth shipping**: it stops political communications (incl. real
self-sourcing) from proving claims true, and it collapses the render/pipeline tier drift
(Finding 2). The over-reach is bounded (2% collapse) and the two clear errors are fixed. Do
**not** auto-regenerate the site off it — route the real verdict-level impact through the gated
15-claim gold before/after run (P129, metered, jackie-approved) before any re-publish.

## Resolved this session (jackie, 2026-07-31)
- **"Data yes, press no"** — implemented as the `data_signal_segments` rule (crux resolved). CBP
  border stats and equivalents survive; DOJ/Treasury/DHS *announcements* still demote.
- Telemetry (composition/asymmetry visibility) = **fast-follow**, not a merge gate.

## Still open for jackie (D7 line-item sign-off — I propose, you rule)
- `ofac.treasury.gov` (sanctions lists/FAQs, non-data paths like `/faqs/added`) → **S1**?
  (primary record; the data-segment rule doesn't catch these paths). Review proposes S1.
- USDA AMS Market News (`mymarketnews.ams.usda.gov`) → add to nonpartisan sources / **S1**?
- `aspe.hhs.gov` → S3 + integrity note (research office, political-appointee-led)?
These are host-level promotions I deliberately did **not** self-apply — they need your D7 sign-off.
