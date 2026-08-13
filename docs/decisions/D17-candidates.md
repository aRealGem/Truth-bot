# D17-candidates — three deferrals, logged with their evidence

**Status:** LOGGED ONLY. Nothing here is implemented, proposed for ratification, or scheduled.
**Opened:** 2026-08-09 (T-3, $0 — no model calls, no code changes).
**Why it exists:** D15 and D16(α) were ratified on 2026-08-09. Each was scoped narrowly on purpose, and each left a known, named piece of work on the floor. This file is where those pieces are written down with the evidence already in hand, so a later decision starts from measurement rather than from memory.

---

## How to read this file

These are **candidates**, not proposals. A proposal states a rule, measures its
blast radius, and asks for ratification. None of the three below does any of
that, and none should be implemented off the strength of this document.

What each entry does contain is the thing that is expensive to reconstruct
later: *what was observed, in which artifact, and why the obvious fix was not
taken at the time.* Two of the three were deferred inside a shipped decision
(D16(α) names both in its own text); the third came out of a metered run and
has a measured number attached.

A candidate graduates to a numbered decision only when someone writes the rule
down, measures it against the five rebuilt runs for $0, and puts it in front of
the owner. Until then this file is a memory aid.

---

## D17-a — FRASER path-level allowance

**Deferred from:** D16(α), registry design.
**Code:** `src/truthbot/verify/statistical_agency.py`, `statistical_agency_registry.yaml`.
**Current behaviour:** `fraser.stlouisfed.org` is a **blanket deny** — `(False, "deny:domain:fraser.stlouisfed.org")`.

### What was observed

FRASER (the St. Louis Fed's document archive) is denied at the domain level
because it is genuinely two things at once. In the five rebuilt runs it serves,
from the same host:

| Document | Claim | What it is |
|---|---|---|
| January 2006 Employment Situation | `gwbush_2006:0133` | a real BLS statistical release |
| OMB budget appendix | `gwbush_2006:0155` | an Executive Office of the President document |
| Economic Report of the President (CEA) | `clinton_1998:0167` | an Executive Office of the President document |

D16(α)'s whole test is **publisher function**: does this host exist to measure?
A host that serves both BLS releases and the President's own budget cannot
answer that question at the host level, so the registry fails closed and denies
the domain outright.

### What it costs

Genuine BLS releases archived on FRASER get no credit. That cost is not evenly
distributed — it lands hardest on the **pre-web eras**, where FRASER is
frequently the archive of record and bls.gov simply does not host the document.
A rule that silently penalises 1998 and 2006 relative to 2022 and 2026 is
exactly the era-parity problem the whole remediation exists to fix, so this is
not a cosmetic gap.

### The shape of the fix (not taken)

A **path-level rule** — allow FRASER paths that resolve to BLS/BEA/Census series
and titles, deny the rest — would recover the statistical releases without
re-admitting the budget appendix. That means reading FRASER's path taxonomy,
pinning it in the registry, and testing it against every FRASER URL in the
corpus. Real work; correctly out of scope for a flag-gated era refinement.

### Open questions for whoever picks this up

- Is FRASER's path structure stable enough to pin in a versioned registry, or
  does it need periodic re-verification?
- How many items would this actually recover, per era? (Measurable for $0
  against the five runs — do that first.)
- Does the same argument apply to any other archive host currently denied for
  the same "serves both" reason?

---

## D17-b — Document-class detection

**Deferred from:** D16(α), explicitly, in the module docstring and in `docs/decisions/D16-statistical-release.md` §2.
**Code:** `src/truthbot/verify/principals.py` (`principal_relation`), used from `src/truthbot/pipeline.py`.

### What was observed

`principal_relation` keys on **HOST**. It answers "is this source the speaker's
own organ?" by looking at where the document is served from. That is right for
whitehouse.gov and wrong for anything the executive branch publishes through a
third party:

| Claim | Document | Served from | `principal_relation` reads |
|---|---|---|---|
| `gwbush_2006:0217` | ONDCP National Drug Control Strategy (Feb 2006) | `justice.gov`, `files.eric.ed.gov` | **independent** |
| `clinton_1998:0101` | FY1999 President's Budget | `gpo.gov` | **independent** |

Both are the principal's *own* executive documents. Both read as independent
corroboration. That is a false independence, and it was the reason the blanket
form of D16 — "any Government-tier post-speech item may credit" — had to be
rejected: those two claims were the motivating examples, and the blanket rule
would have let each of them be corroborated by its own author.

### Why the fix was not taken

D16(α) **inverted the test specifically to avoid needing this detector.**
Instead of asking what the document is *not* (not the principal's own), it asks
whether the publisher's *function* is statistical measurement. The President's
Budget and the ONDCP Strategy are not statistical-agency records no matter which
host serves them, so the function test disposes of both without ever
identifying the document class.

That inversion is sound for D16(α)'s narrow purpose and it is **not** a general
fix. Any future rule that needs to know "is this the speaker's own document"
outside the statistical-agency frame hits the same host-keyed blind spot.

### The shape of the fix (not taken)

Author / document-class detection: identify the *authoring body* rather than the
serving host — from the document's own metadata, title conventions, or a pinned
class registry — and let `principal_relation` consult it. Real work, with a real
false-positive risk of its own.

### Open questions for whoever picks this up

- How many pack items across the five runs are the principal's own documents
  served from a third-party host? (Unmeasured. Worth a $0 count before anyone
  designs anything.)
- Does document-class detection belong in `principal_relation`, or beside it as
  a separate axis the evidential-role function consults?

---

## D17-c — Retrieval-contract change: excerpt the series rows

**Deferred from:** B2 (the scoring-prompt fix), 2026-08-08.
**Evidence:** `metrics/remediation_v2/B2_FINDINGS.md`;
`metrics/remediation_v2/b2_subset.json` for the **design** (a truthful pre-run
estimate, 2026-08-08, predating the `haiku-score-2026-08-09` calibration);
`metrics/remediation_v2/d17c_stage0/b2_settlement.json` for the **measurement**
($0.5405 actual, 2.35x the $0.2299 estimate). Cite the one that matches the
question — the subset is not a cost figure.
**Code:** `src/truthbot/verify/relevance.py` (`score_payload`).

### What was observed — with a number

`score_payload` sends the scorer a **source name and 400 characters of
snippet**. It never sends the data table. The pipeline retrieves the URL of a
statistical series and never fetches its contents.

B2 tested the competing diagnosis — that the *prompt* was at fault, that Haiku
was mis-classifying raw series as "context" — by rewriting the prompt to tell
the scorer that a primary series carrying the figure at issue must take a side,
then re-running it over exactly the 227 items that diagnosis predicts it would
fix.

**It moved 17 of 227. 7.5%.** ($0.5404 spent measuring it.)

And the scorer said why, in its own `one_line_why` lines, repeatedly:

> "BLS time-series for employed persons 16+ back to 1948 is the authoritative
> historical record needed to judge whether January 2026 represents an all-time
> high, **but snippet does not state the January 2026 level**."

> "Primary appropriations law for FY1998 provides actual funding level in effect
> near the claim date, **but snippet does not state the 1993 baseline or
> comparison**."

A scorer cannot judge a BLS series it has not been shown. The
mis-classification was a symptom; the missing fetch is the disease. No prompt
closes that gap, and the 7.5% number is what "no prompt closes that gap" looks
like when it is measured instead of argued.

### The shape of the fix (not taken)

**Series-row excerpting**: fetch the primary series, extract the rows relevant
to the claim's period and measure, and put those rows in the snippet before
scoring. This is a change to the *retrieval contract*, not to a prompt and not
to the gate — which is why it was out of scope for B2 and is out of scope here.

It is, on the evidence, the actual fix for statistical claims, and the largest
single lever left on the pipeline's accuracy for that class.

### Open questions for whoever picks this up

- Which series formats are worth supporting first? (BLS and BEA cover most of
  the corpus's statistical claims; a count per source is a $0 exercise.)
- What is the cost delta? Excerpted rows make the score payload larger, and the
  estimator in `src/truthbot/costs.py` should be re-run over a realistic payload
  before anyone commits to a budget.
- Does an excerpt need provenance of its own — the reader should be able to see
  which rows the verdict rested on, not just that a series was consulted.

### Stage 0 findings (2026-08-12, $0 — no model calls)

Scoping only. Stage 0 publishes nothing and re-adjudicates nothing.

**Wave 1 = 84 items across 40 claims** — trump 35, biden 16, obama 15,
clinton 10, gwbush 8. Derived from the shipped heads by
`metrics/remediation_v2/d17c_stage0/wave1.py`, which asserts the split rather
than restating it. That is the 124-item stance-null series population minus the
document publishers (CBO/GAO/NCES/CRS publish tabled *reports*, not series) and
minus NCHS/CDC.

**The D17-c-reachable trump floor is 274/1472 = 18.61%.** Converting every
wave-1 trump item still leaves that residual, against a 15% ceiling — so D17-c
alone does not clear it. The earlier 17.73% figure assumed the document
publishers were in scope; they are not.

**NCHS/CDC is out of wave 1, logged not dropped.** Three stance-null items
corpus-wide fails the ten-item coverage floor. The family is product-scoped:
`wonder.`/`data.cdc.gov` are series-like, `cdc.gov/nchs` and `stacks.cdc.gov`
are documents, and MMWR is a document-with-tables.

**Pilot handler: FRED + ALFRED.** ALFRED is the *vintage axis* of the FRED
handler, not a separate format, so the floor governs the handler — 23 corpus
items between them, which clears it. It covers 9 of the 84 wave-1 items.

**Cost.** An excerpt inflates the scoring prompt only; item count and reply
schema are untouched, so the marginal cost is one term. Against a measured
conservative prompt volume, a whole-pack re-score of the 40 claims with
4,000-character excerpts projects **$0.2992** versus the $0.75 ceiling. The
input side cannot threaten that ceiling: $0.75 would buy ~28,000 characters per
excerpt. The mean/max token delta per item remains a *measurement* pending real
payloads.

**Head lineage (S-9).** The five publishing heads this record is derived from,
so the lineage is greppable:

| speech | run id |
|---|---|
| trump_2026 | `91dd7a34-7a3c-4f40-bcdc-276b2cb15d26` |
| biden_2022 | `ddb05ee3-7d9c-4b2c-beaf-e197b9354379` |
| obama_2014 | `2cbda3e4-c578-442a-aee7-c5c28a388048` |
| clinton_1998 | `49b2e3e8-1667-4460-8989-b265914d4450` |
| gwbush_2006 | `5c923c25-b065-4a9f-80bf-d23db4f9bcd1` |

No heads have been moved to the gitignored `_quarantine/` yet, so there are no
*quarantined*-head run ids to record; these are the live heads. Revisit when
S-9 executes.

**Ledger entry for the next corrections wave (not applied here).** The PR #105
banner's "at most those" is an upper bound and is true, so the published prose
stands. But its "48 are statistical series" is wrong for 12 items —
CBO/GAO/NCES publish tabled reports, not series. Owner-approved as a wording
correction for the next wave, alongside the dropped-row note. **Do not edit the
published sentence outside a corrections wave.**

**CLOSED by ruling: the selection window is now frequency-aware.** The Stage 0
predicate took the 13 most recent observations at or before the utterance date,
which was 13 months on a monthly series and 13 *years* on an annual one —
`FYFSD` selected 2009-09-30 to 2021-09-30. Deterministic either way, but not
the year-over-year window the wording implied. Ruled: trailing K at the series'
native frequency, **K = 25 monthly / 9 quarterly / 13 annual**, frequency taken
as the median spacing of the last four eligible observations and recorded in
the predicate. A flat 13/5/2 was rejected — annual=2 guts the `FYFSD`
cross-administration window, and frequency alone does not fix the `0054`
claim-period mismatch.

Four fixed committed regex rules run against claim text **and context** and
propose an earlier start; the widest proposal wins and every fired rule is
named per item: explicit years → Jan 1 of (min_year − 1); last/past N years →
N+1 years; `record|ever|history|all-time|never` → full eligible history;
`took office|administration|inherited` → trailing 5 years. Excerpts assert at
most 1,500 rows and **halt rather than truncate**. Deepest fixture is
`PAYEMS_current` at 1,051 observations, so the assert does not fire on the
committed set.

The other two questions carried out of Stage 0 are also closed: the
`LNS12000000` dead link is a ledger entry (no substitution), and `units` ships
null with a machine-readable reason at six-of-seven provenance fields.

**Ruled (R1 = (b)): the superlative rule reads the claim's own words.**
`record|ever|history|all-time|never` matches claim TEXT only; the other three
rules stay on text + context. Under text+context it fired on
`biden_2022:0169` — *"369,000 new manufacturing jobs just last year"*, a claim
with no superlative of its own — because a neighbouring sentence carried one,
pulling 997 rows of `MANEMP` for a claim about a single year. A rule keying on
a claim's own assertion should read the claim's own words; the other three
describe a period rather than assert a superlative, so context legitimately
informs them. Only `0169` changed (997 → 25 rows); the other eight goldens are
byte-identical. Matched Fable's pre-registered simulation on every value.

**Wave-2 REQUIRED-recommended: publish badges go fail-closed.** Today
`_classify_source_for_render` returns `"verified"` when no classification map
exists *and* when a URL is simply absent from one — both branches fail OPEN, so
absence of evidence renders as evidence of verification. That is how a URL
returning 404 on both FRED and ALFRED carried the `source-verified` badge on the
published site, twice. Wave 2 should invert it: no classification record → no
`"verified"` badge, and a known-dead URL renders broken. Rides the stable-ids
re-render so deep links rotate once; owner ratification at the Stage B gate.

**Priority bumped: URL-liveness audit and the retrieval contract.** Both were
logged D17 candidates; the Stage A diagnostic raised their priority with
evidence rather than suspicion. `retrieved_at` is an *assembly stamp*, not a
retrieval time — all 20 items across the `0054` and `0055` packs are stamped
inside 309 microseconds, rising ~13µs per item in list order, which is a
serialization loop, not 20 HTTP round-trips. And `metrics/url_cache.jsonl`, the
persistence path named in `url_validation.py`, has never existed on disk or in
git. So nothing in the pipeline ever fetched these URLs, and nothing recorded
that it hadn't. A stance (`supports_claim=True`) was attached to a
browsing-model-authored snippet describing a page that never resolved.

**Production-path candidate: a structured `series_rows` key.** Stage A appends
the excerpt to `snippet` because the census had to measure against the shipped
baseline, and changing the wire shape would have changed both variables at once.
For the production path a dedicated `series_rows` key is cleaner provenance —
the rows stop being prose the scorer has to parse out of a snippet. Logged, not
scheduled.

**Two D17 candidates LOGGED, not implemented (R2).**

1. *Named-era / named-person anchor map.* A deterministic mapping from
   proper-noun temporal anchors ("when Reagan first stood here") to dates. Real
   scope and a new determinism surface; not to be smuggled in under a spend
   ceiling.
2. *`gap_periods` annotation in window provenance.* `CE16OV`, `CPILFESL`,
   `CUUR0000SAF112` and `APU0000708111` each hole at 2025-10 (the shutdown data
   gap); `PAYEMS` is complete. Fable-verified. Deferred so the pre-registered
   run-sha stays binding — annotating provenance would change it.

**Carried limitation: a proper-noun comparison anchor escapes all four rules.**
`obama_2014:0189` compares the minimum wage to *"when Ronald Reagan first stood
here"* (circa 1982). No rule fires, so it takes the default 25-month window
(2011-12-01 to 2013-12-01), which does not reach the period the claim is
actually about. This is the same claim-period mismatch that motivated rejecting
13/5/2, surviving in the ruled ruleset because the anchor is a name rather than
a date. Ruled (R2 = (a) amended): `0189` STAYS in wave 1, the mismatch is
recorded as-is, and no named-anchor rule is implemented. Its census row must
carry `window_period_mismatch=true` and is **NON-ACTIONABLE for any Stage B
consideration**. The flag rides on the census row and deliberately NOT on the
golden payload — adding a field there would change the pre-registered run-sha.

**Stage A is BLOCKED on a payload-shape question, halted before spend.** The
excerpt has no channel to the model. `relevance.score_payload` sends only
`{i, source, snippet}` and truncates `snippet` at `SCORE_SNIPPET_CHARS = 400`;
`Evidence` has no excerpt field and no insertion path exists in `src/`. Routing
the excerpts through `snippet` unchanged would ship 3,200 of 49,655 characters
— **93.6% of every excerpt truncated away** — and would still produce a
complete, plausible-looking flip census measuring 400-character stubs. The
pre-registered $0.0511 projection assumes the full 49,655 characters reach the
model, so the cost model and the committed code disagree. Not resolvable
without a ruling on how excerpts enter the payload; see the report for options.

**Context widens more than claim text alone would.** Running the rules over
`text + context` — as ruled — makes `record|ever|history` fire on
`biden_2022:0169` (*"369,000 new manufacturing jobs just last year"*), whose own
text carries no superlative, pulling the full 997-row `MANEMP` history for a
claim about a single year. Deterministic and within the row cap, but it is
breadth bought from the neighbouring sentences, not from the claim.

**Closeout: the 448/445 census delta.** Reproduced exactly, and it is an
artifact of the hand count rather than a defect in the code — see
`d17c_stage0/delta_closeout.py`. Two mechanisms, together and only together
giving 448 with Census +3, USDA-NASS +1, NCHS −1:

1. *Press-prefix breadth.* The registry's own `press_prefixes` list is five,
   but it is **additive** to `tier_registry.yaml`'s six `stat_press_prefixes`,
   so the shipped `classify_ex` applies nine. The inherited `/newsroom` denies
   three `census.gov` items. This is documented intent, not drift.
2. *Path case-folding.* `statistical_agency._url_path` lowercases. Two items
   turn on it, in opposite directions: `nass.usda.gov/Newsroom/...` is denied
   as press (a hand count matching case-sensitively would admit it), and
   `cdc.gov/MMWR/...` is admitted (a case-sensitive count would deny it).

The shipped behaviour is correct on both counts — a `/Newsroom` press page
*should* be denied and an `/MMWR` document *should* be admitted. Fable's
hypothesis (b), that `quickstats.nass.usda.gov` might not resolve from entry
`nass.usda.gov` by suffix, is **refuted**: it resolves, and the registry
rationale and the code agree.

---

## D17-d — Re-attributing a rationale after a severity flip

**Deferred from:** R-3 (no-blank-rationales), 2026-08-10.
**Code:** `src/truthbot/verdict/discriminator.py` (`apply_discrimination`, `adopt_seat_rationale`).

### What was observed

R-3 fixed the case where a resolver publishes a verdict with **no** rationale.
It did not touch the adjacent case, which is about a rationale that is present
and arguing for a **different label** than the one that shipped.

`apply_discrimination` overrides a resolved FALSE↔MISLEADING label with the
stage-2 discriminator's call. The row keeps the rationale the stage-1 winning
seat wrote — for the label the discriminator just overturned. So a claim can
publish MISLEADING under a sentence explaining why it is FALSE.

This is narrower than it sounds: the discriminator only moves *within* the
adverse pair, so both labels share a factual core and the stored sentence is
rarely nonsense. But it is still a rationale that does not match its verdict.

### Why the fix was not taken

Adopting a different seat's rationale on every severity flip would **change the
published text of already-adjudicated claims** across the corpus, which is a
corrections-ledger event, not a lint fix. R-3 was ruled as a structural repair
for blank rationales; silently rewriting non-blank ones would have exceeded it.

The R-3 machinery is nonetheless already in place: `adopt_seat_rationale` takes
the final label and finds the seat that voted it. Widening the trigger from
"blank" to "blank or label-mismatched" is a one-condition change — the work is
in the ledger and the review, not in the code.

### Open questions for whoever picks this up

- How many rows in the five runs carry a `crm114` override where a *different*
  seat voted the final label with text? ($0 to count — do that first.)
- Is a mismatched-but-adjacent rationale a **correction** (verdict text changed)
  or a **provenance change** (same badge, better attribution)? The ledger's
  vocabulary treats those differently.

---

## D17-e — Computed exhibits as NON-dispositive context on c-eval claims

**Deferred from:** R-1 (the trump_2026:0031 shape correction), 2026-08-10.
**Code:** `src/truthbot/publish/computed_exhibit.py` (`INADMISSIBLE_SHAPES`).
**This is R-2 amendment territory, and explicitly NOT part of this publish.**

### What was observed

R-1 turned on the sharpest case for the current rule and the sharpest case
against it, in the same pair of sentences.

`trump_2026:0031` — *"in the last three months of 2025, it was down to 1.7
percent"* — is now `c-count`, and the exhibit attaches: CPILFESL, ALFRED
vintage 2026-02-24, `(Dec/Sep)^4 - 1 = 1.701%`. The claim states a number, the
exhibit computes that number, and the arithmetic settles it.

Its neighbour `trump_2026:0030` — *"my administration has driven core inflation
down to the lowest level in more than five years"* — is `c-eval` and correctly
so: superlative plus causal attribution. Under R-2 it may carry **no exhibit at
all**, and that is the right call about what can DECIDE it. Arithmetic cannot
establish "lowest in five years" as a characterisation, and it certainly cannot
establish "my administration drove it".

But the same two levels are what the claim is *about*. A reader looking at
0030's FALSE has no way to see the series the verdict rests on, while a reader
looking at 0031's TRUE gets the full derivation — and the two sentences are
about the same statistic.

### The shape of the change (not taken)

Admit an exhibit on a c-eval claim as a **non-dispositive context item**:
rendered, clearly marked as context rather than proof, and structurally barred
from contributing to the verdict or to any gate quota.

That is a genuine amendment to R-2, whose whole content is "never on a C-EVAL
judgment", and it should be argued as one rather than slipped in as a rendering
tweak. The current all-or-nothing rule has the great virtue of being
unambiguous; a "context only" tier is exactly the kind of distinction that
erodes.

### Open questions for whoever picks this up

- Is "non-dispositive" enforceable, or only stated? It needs to be structural —
  a separate field the gate cannot read — or it is a comment.
- How many c-eval claims in the five runs sit adjacent to a c-count claim with
  an exhibit? (The 0030/0031 pattern may be rare enough not to be worth a rule.
  $0 to count.)
- Does a reader distinguish "context" from "proof" on a page that renders both
  in the same visual block? If not, the honest answer may be to keep the ban.

---

## Not in this file

Deliberately excluded, so the absences are legible:

- Anything requiring spend to evaluate. All three entries above can be *sized*
  for $0 against the five rebuilt runs; none has been.
- Any recommendation about sequencing. These are three independent candidates,
  and nothing here claims one should come first.
