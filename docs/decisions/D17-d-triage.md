# D17-d triage — the decision surface for scope

**Status: input to an owner scope decision. Nothing here is executed, no verdict
moves, $0 spent.**

Evidence: `metrics/remediation_v2/d17d_triage.json` (every claim, its class, and
why), built by `scripts/build_d17d_triage.py` from the five publishing heads.

---

## 1. Why this exists: two things were wearing one label

Owner review of the trump report found the dominant defect is **presentational
and backlog, not adjudicative**.

Across the five speeches there are **132 Unverifiable claims. 128 of them are
gate-withheld** — the evidence gate declining to decide because the retrieved
pack never met the Tier-1..3 bearing quota. That is a statement about *our
retrieval*. Only **4** are the other thing: a claim the gate had no structural
objection to.

Rendered identically, honest abstention read as failed fact-check. "Olympics in
LA: insufficient evidence" looks like a verdict about the Olympics. It is a
verdict about us.

`trump_2026:0054` is the proof this matters. It sat gate-withheld until D17-c
handed the panel the series rows, and it decided **TRUE**. The claim never
changed. What we retrieved did.

**Task 1 fixed the rendering** (`4aacdcc`): gate-withheld now renders as
*"Insufficient qualifying evidence retrieved"* with a claim-card explainer
saying the panel was never asked to rule and that this describes what we
gathered, not whether the claim is true. Keyed on the structured
`provenance_code`, never on prose.

**One limit, recorded not worked around.** `trump_2026:0153` ("I asked Michael
Dell, how do you make all that money?") is a private conversation — permanently
undecidable — and carries *the same* structured gate code. The gate knows the
pack did not qualify; it cannot know that nothing ever could. Both readings are
literally true of it. Separating those is what the `substantive` class below is
for, and it needs this desk pass, not a regex.

---

## 2. Gate-withheld, per speech

| speech | Unverifiable | gate-withheld | other |
|---|---|---|---|
| trump_2026 | 56 | **54** | 2 |
| biden_2022 | 24 | **24** | 0 |
| clinton_1998 | 21 | **21** | 0 |
| obama_2014 | 20 | **19** | 1 |
| gwbush_2006 | 11 | **10** | 1 |
| **total** | **132** | **128** | **4** |

---

## 3. What would actually decide them

| class | n | what it needs |
|---|---|---|
| **web-tier1** | **81** | Tier-1..3 web retrieval. No series will settle it. |
| **substantive** | **35** | Nothing. Permanent honest abstention. |
| **series-core** | **7** | A named statistical series — the D17-c mechanism as built. |
| **compound-split** | **5** | Segmentation first; a checkable core is buried in a compound utterance. |

**The shape of the backlog is not what the trump report suggests.** The largest
class by far is `web-tier1` (63%), and most of it is the human material of a
State of the Union: guests in the gallery, valor citations, named victims. A
Purple Heart is in a record; a 2024 crash was reported; a scholarship is
documentable. These are a *retrieval backlog*, and D17-c's series mechanism does
not touch them.

**`substantive` (27%) is not a backlog at all.** What a doctor said in a room,
what someone thought they were feeling, whether "many, if not most" of an
unmeasured population speak English, what an adversary's *aim* is. No retrieval
reaches these, and the right outcome is that the page says so plainly rather
than implying an unfinished job.

**`series-core` is only 7 claims** — mortgage rates, index records, payrolls, the
federal deficit, the real minimum wage. D17-c built the machinery that decides
exactly these, and there are seven of them left across five speeches.

---

## 4. Cost

| lane | claims | projected |
|---|---|---|
| series-core | 7 | **$0.7623** |
| web-tier1 | 81 | **UNPRICED** |
| compound-split | 5 | **UNPRICED** (and blocked) |
| substantive | 35 | **$0.00** — display work, not retrieval |

**Series lane: $0.7623**, at the measured $0.1089/claim ($0.003124/kchar) from
the d17c-wave2 escape run. Note this **exceeds the $0.50 ceiling wave 2 ran
under** — the series lane alone needs its own authorisation, not headroom.

**web-tier1 is UNPRICED and must stay that way until measured.** No constant
exists for a retrieval-bearing lane on these packs. Borrowing one is exactly
what ran the escalation 8.2× over ($0.3266 against $0.0396), and **S-12 forbids
it**: a constant that cannot name the payload it measured is a proxy, not a
measurement. This lane needs its own $0 estimate pass. It is also the *largest*
lane, so the number that matters most for D17-d scope is the one nobody has.

**compound-split is unpriced and blocked** — segmentation has to land before
retrieval can even be scoped. Cross-refs the logged utterance-segmentation
structural item.

---

## 5. Flagged separately: `trump_2026:0466`

**Not gate-withheld. Verdict TRUE, confidence 0.9. No change proposed.**

> *"And all because she wore the uniform of our nation, she was shot."*

The panel's stated reasoning verified she was **shot while on duty in uniform**.
That is a weakened paraphrase: the claim's core is **causal** — it asserts the
shooter's *motive*. Evidence of the shooting does not reach why the shooter
fired.

This is the same conflation family as ruling (d): a decision procedure that
settles one proposition being credited with settling a neighbouring, harder one.
Logged for owner-visible re-adjudication in D17-d.

---

## 6. What the owner is being asked to scope

1. **Series lane (7 claims, $0.7623)** — smallest, fully understood, needs a
   ceiling above wave 2's.
2. **web-tier1 (81 claims, unpriced)** — the real backlog. Wants a $0 estimate
   pass before any commitment.
3. **substantive (35 claims, $0)** — a display decision, not a retrieval one:
   should these read as permanent abstention rather than sharing a label with an
   unfinished backlog?
4. **compound-split (5)** — deferred behind segmentation.
5. **`0466`** — re-adjudicate or leave.

The classification is desk work on claim text and source knowledge. Where a
class is genuinely arguable the `why` field says so — `trump_2026:0043` (which
investment measure?), `trump_2026:0291` (index series vs survey data),
`obama_2014:0189` (needs the named-anchor work, not just the series). Those are
flagged rather than resolved, because resolving them is a scope decision.
