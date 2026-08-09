# D15 — Utterance-derivative evidence (`utterance-record`)

**Status:** **RATIFIED 2026-08-09** by the owner. Implemented, tested, and **ENABLED BY DEFAULT**.
**Flag:** `TRUTHBOT_D15_UTTERANCE_RECORD` — unset means ON. Retained as an OVERRIDE: `=0` reproduces the pre-ratification gate exactly, which is what makes a regression bisectable without reverting code.
**Flip set under the ratified rules:** `scripts/regate_from_rescore.py` → `metrics/remediation_v2/regate_flipset.{json,md}`.
**Blast radius measured:** `scripts/measure_d15.py` → `metrics/remediation_v2/d15_blast_radius.json` ($0, no model calls).
**Combined with D16(α):** `scripts/d15_d16_era_breakdown.py` → `metrics/remediation_v2/d15_d16_era_breakdown.{json,md}` — the M-6 evenhandedness check. D15 only removes credit and D16(α) only adds it, so neither number means anything on its own; the NET per speech, the decided-rate on both bases, and the era-concentration finding live there.
**Code:** `src/truthbot/verdict/utterance_record.py`, wired in `verdict/consolidator.py` and `verdict/evidential_role.py`.

---

## 1. The problem: a claim witnessing itself

The tier ladder ranks a **document's publisher**. It has no way to ask what the
document *contains*. So a transcript of the speech being fact-checked arrives as
a GOVERNMENT-tier source — the same tier as a Bureau of Labor Statistics series
— and once the stance layer marks it "supports", it credits `MIN_BEARING_T13`
and helps a claim reach a decided verdict.

What that document actually establishes is that *the speaker said the thing*.
It is the assertion under test, reprinted on government letterhead. A claim
cannot be its own witness, and today nothing in the pipeline stops it from
trying.

The worked example is `trump_2026:0469`. Of its ten pack items, two are the
Daily Compilation of Presidential Documents transcript (E7) and the
Congressional Record for 24 February 2026 (E9). Both are GOVERNMENT tier. Both
were stanceless in the original run — and then B1a bought them a stance. After
B1a they are bearing Tier-1 items, and they are two of the credits that let the
claim through. The claim is released on the strength of the President having
said it.

Confirmed classes of instance across the five rebuilt runs:

| Class | Example |
|---|---|
| Daily Compilation of Presidential Documents | `govinfo.gov/.../DCPD-202600136.pdf` |
| Congressional Record of the day | `govinfo.gov/.../CREC-2026-02-24.pdf`, `congress.gov/.../CREC-2006-01-31-...htm` |
| Weekly Compilation of Presidential Documents | `govinfo.gov/.../WCPD-1998-02-02-Pg129-2.pdf` |
| American Presidency Project archive copy | `presidency.ucsb.edu/documents/address-before-joint-session-the-congress-the-state-the-union-21` |
| Same-speech transcripts and recaps | White House / AP / Washington Post / Miller Center full-text transcripts; `trump_2026:0469` E5, an AP recap whose snippet "documents President's wording" |

## 2. The rule

A deterministic evidential role, `utterance-record`, computed by pure code from
three things already on disk: the URL, the snippet, and the **registered speech
date** (`verdict.speech_context.speech_date_for`). No model call, no network,
no new data.

Five independent rules. Each is separately named, separately tested, and
separately reported in the blast-radius census, so a reviewer can ratify or
reject them one at a time rather than as a bloc.

| Rule | Fires when | Anchor |
|---|---|---|
| `dcpd-daily-compilation` | URL carries a `DCPD-YYYYNNNNN` package id of the speech's year, and the item is dated the speech date or the day after | item date, ±1 day |
| `crec-congressional-record` | URL carries a `CREC-YYYY-MM-DD` package id whose **own** date is the speech date | package id, exact |
| `wcpd-weekly-compilation` | URL carries a `WCPD-YYYY-MM-DD` issue dated within the 7 days after the speech (the issue covering that week) | package id, week |
| `presidency-ucsb-address` | An American Presidency Project `/documents/` or `/node/` URL, dated the speech date, whose slug or snippet names the address itself | item date, exact + slug |
| `recap-language` | Recap phrasing ("transcript of", "as delivered", "recap of", "the President's wording") **and** an address token, on an item dated between the speech and the fair-game end | item date, window + two prose cues |

Three deliberate design choices:

1. **Date-anchored, always.** Every rule needs a date it can check against the
   registered speech date. The Congressional Record is published every sitting
   day; only the day of the address is the address. An item with no usable date
   matches nothing.
2. **The package id beats the metadata.** `CREC`/`WCPD` dates are read out of
   the URL, not out of `published_at` — retrievers disagree about the latter
   (the same CREC PDF arrives dated both `1998-01-27` and `1998-01-28`), while
   GPO's package id is the document's own identity.
3. **Prose needs two cues.** `recap-language` is the only rule that reads free
   text, so it demands a recap phrase *and* a token naming the address. This is
   what keeps "as delivered" in an unrelated snippet from firing.

Conservative by construction: a **miss is the intended failure mode**. A false
positive silently destroys real evidence, which is the more expensive error.

### Effect

Quota credit **zero**, on both quota branches — the legacy Tier-1..3 branch, the
D11.2 role-aware branch, and the lenient-era GOVERNMENT branch (which would
otherwise credit a same-day transcript even with a null stance). The item is
still **kept** and still **displayed**, carrying `role: utterance-record` in the
pack payload. Provenance the reader can see; not evidence the gate can spend.
It also joins `verdict_audit.SELF_ROLES`, because a transcript of the speaker
asserting a superlative is the purest case of a verdict resting on the
assertion it was meant to test.

The role rides the axis that already exists (`primary-record` / `corroborant` /
`attribution-only` / `plain-s5`) rather than inventing a parallel mechanism.
One difference, stated explicitly: `utterance-record` is **not** a product of
the D11.2 `f(claim_shape, principal_relation)` table — it is
`f(url, snippet, speech_date)`, so it is assigned by detection and never
returned by `evidential_role()`. Where an item qualifies for both, the D15 label
wins, because it is the stronger statement.

## 3. Measured blast radius ($0)

Two stance vintages, because they disagree and the disagreement is the point.
`stored` is the stances the rebuilt artifacts recorded; `rescored` overlays the
B1a sidecars and is the live state of the corpus. Each stored pack is run
through the **real** gate (`consolidator.consolidate`) twice — switch off, then
switch on — and the answers compared.

**Corpus totals — 529 claims, 4,344 stored items:**

| | items flagged | claims touched | bearing | bearing **and** Tier-1..3 | gate outcomes changed |
|---|---:|---:|---:|---:|---:|
| stored stances | 387 | 241 | 182 | 153 | **48** |
| after B1a re-score | 387 | 241 | 315 | 242 | **50** |

**By rule** (identical in both vintages — detection does not depend on stance):

| rule | items |
|---|---:|
| `crec-congressional-record` | 130 |
| `recap-language` | 104 |
| `dcpd-daily-compilation` | 93 |
| `presidency-ucsb-address` | 39 |
| `wcpd-weekly-compilation` | 21 |

**By tier:** Government 253, Other 57, Political 38, Wire 25, Established 14.

**Per speech:**

| speech | items | bearing (stored → rescored) | T1-3 bearing (stored → rescored) | gate Δ (stored → rescored) |
|---|---:|---|---|---|
| gwbush_2006 | 26 | 12 → 19 | 6 → 9 | 1 → 2 |
| clinton_1998 | 92 | 42 → 82 | 30 → 47 | 10 → 10 |
| obama_2014 | 67 | 32 → 58 | 27 → 47 | 6 → 7 |
| biden_2022 | 45 | 21 → 40 | 17 → 30 | 8 → 8 |
| trump_2026 | 157 | 75 → 116 | 73 → 109 | 23 → 23 |

**Direction of every change is one-way: 50 newly gated, 0 released.** That is
the expected signature of a rule that only ever removes credit, and it is
asserted rather than assumed — a release would mean the exclusion was somehow
*adding* a credit, and the measurement reports any such case as "investigate".

Of the 50 newly-gated claims (rescored vintage), **33 currently ship TRUE**, 16
are already UNVERIFIABLE, and 1 ships FALSE. So ratifying D15 would withhold 33
verdicts the site currently publishes as decided. Withholding costs **$0** — no
panel call is needed to not decide something.

### Two findings that complicate the brief

1. **The `.edu` framing does not hold in this corpus.** All 121 `.edu` items —
   including all 89 `presidency.ucsb.edu` items — are classified tier **Other**,
   not ACADEMIC. Neither Other *nor* ACADEMIC is in `_T13`, so no
   `presidency.ucsb.edu` item can credit the quota today under any stance. The
   American Presidency Project rule is therefore **display hygiene, not gate
   repair**: it labels 39 items honestly and changes no outcome by itself. The
   real quota exposure is the 253 GOVERNMENT-tier items — DCPD, CREC, WCPD.
2. **B1a widened the hole it was dug to fill.** Bearing flagged items go from
   182 to 315 once the B1a scores are overlaid: the re-score is precisely what
   converted stanceless transcripts into quota-crediting evidence. D15 and B1a
   push in opposite directions on the same items, and the gate numbers in the
   B1a flip set should be read with that in mind.

## 4. What ratification enabled

Ratified 2026-08-09. The default is now ON, and it:

- annotate 387 items across 241 claims with `role: utterance-record` in the
  pack payload, visible on the published cards as provenance;
- remove 242 quota credits, moving 50 claims from decided to gated (33 of them
  currently TRUE);
- close the "claim witnesses itself" path permanently, including against a
  future re-score that would otherwise re-open it;
- extend the superlative self-sourcing audit to transcripts.

The suite is green with the flag both off and on, and `consolidate()` accepts
the same switch as an explicit argument, so a measurement never has to set an
environment variable other consolidations in the process would inherit.

**Measured against the ratified re-gate.** With both D15 and D16(α) active over
the B1a+B2 stance vintage the corpus flip set is 23 released / 64 still gated /
65 newly gated / 377 unchanged, from a pre-ratification 33 / 54 / 27 / 415. The
"50 claims moved from decided to gated" above was measured on the B1a-only
vintage with D15 alone; the combined figure supersedes it. See
`metrics/remediation_v2/regate_flipset.json` (`rules` records which
configuration each leg ran) and `metrics/remediation_v2/t1_intersections.md`
for the twelve released claims D15 re-gates for free.

### Open questions for the owner

1. **Ratify the rule set whole, or rule by rule?** `crec` / `dcpd` are
   unambiguous. `wcpd` is the loosest (a weekly issue also carries that week's
   other presidential documents) and `presidency-ucsb-address` currently changes
   no outcome — either could be held back without affecting the other three.
2. **Is 33 withheld TRUEs acceptable?** Each is a claim whose decided verdict
   currently leans on a record of the speech. Withholding is free; the
   alternative is re-retrieval, which is not.
3. **Should a flagged item still be displayed at all,** or displayed with a
   caption? The current answer is "displayed, labelled `utterance-record`".
4. **Retrieval-side follow-up:** the retrievers are still *fetching* these
   documents and spending pack slots on them. D15 stops them counting; it does
   not stop them crowding. That is a separate change.
