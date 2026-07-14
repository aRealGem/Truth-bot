# Verdict-gold expansion — 2026-07-14 (Phase 3 gate #1)

Expands the canonical verdict-gold **17 → 22 rows** to tighten the Layer B / Layer C
accuracy estimate. All new rows `needs_review: true`, `annotator: claude-expand` —
**verdict calls are for jackie to adjudicate** (political; per the no-auto-merge-verdicts
rule). Every decidable row carries ≥1 authoritative source per the schema.

## Rows added

| sid | verdict | why |
|---|---|---|
| `biden_2022:0030` | TRUE | Putin's invasion premeditated + unprovoked — Nov–Dec 2021 buildup + accurate US pre-invasion intel; widely characterized unprovoked. |
| `trump_2026:0556` | MISLEADING* | "Obliterated Iran's nuclear program" (Op. Midnight Hammer, Jun 2025) — DIA/NBC assessments: one of three sites destroyed, program set back months, not eliminated. *FALSE-leaning — jackie to arbitrate FALSE vs MISLEADING. |
| `trump_2026:0592` | MISLEADING | Ukraine aid "through NATO, they pay us in full" — PURL mechanism is real (allies finance the "vast majority") but "everything/in full" overstates; large prior US direct aid unrepaid. |
| `trump_2026:0600` | UNVERIFIABLE | Out-of-context private payment detail ($1,775 needing approval); no authoritative source. |
| `trump_2026:0100` | UNVERIFIABLE | Individual honoree's WWII combat wound (Luzon) — private biography, no public source. |

New distribution: **TRUE 10 · MISLEADING 7 · FALSE 2 · UNVERIFIABLE 3** (n=22).

## Structural finding — the class gaps are FIXTURE-limited, not annotation-limited

The Phase 3 card (P67.2) named two gaps: **FALSE is Trump-only** and **UNVERIFIABLE was n=1**.
UNVERIFIABLE is now n=3. But the deeper confound persists and **cannot be fixed by
annotation alone**:

```
biden_2022: TRUE 9 · MISLEADING 2                 (no FALSE, no UNVERIFIABLE)
trump_2026: TRUE 1 · MISLEADING 5 · FALSE 2 · UNVERIFIABLE 3
```

FALSE and UNVERIFIABLE are **entirely Trump-side**. I checked every check-worthy
`biden_2022` claim in the frozen TRAIN fixture (14 total): **none is cleanly FALSE.**
The obvious Biden-2022 FALSE — the gun-liability line ("gun manufacturers … the only
industry that can't be sued", which FactCheck.org/PolitiFact rate false) — **is not in
the frozen claim set**, and the schema forbids editing that fixture. `biden_2022:0210`
(let Medicare negotiate…) is normative and already excluded; `0135` (EV chargers/lead
pipes) is labeled `opinion` and never reaches Layer B.

**Why it matters for Phase 3:** with FALSE speaker-locked to Trump, a model that keys on
speaker (or on Trump-era topics) could score well on FALSE for the wrong reason, and the
severity-softening signal (FALSE→MISLEADING) can't be measured on a non-Trump speaker.

**Recommendation (jackie's call):** to get a Biden FALSE we need a **fixture decision**,
not more labeling — either (a) add the gun-liability sentence (and similar) to a future
`biden_2022` fixture rev, or (b) add a second non-Trump speech fixture with fact-checked
false claims. Until then, treat FALSE/UNVERIFIABLE accuracy as Trump-conditional in any
Phase 3 analysis.
