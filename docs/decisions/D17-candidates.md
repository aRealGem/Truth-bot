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
**Evidence:** `metrics/remediation_v2/B2_FINDINGS.md`, `metrics/remediation_v2/b2_subset.json`.
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

---

## Not in this file

Deliberately excluded, so the absences are legible:

- Anything requiring spend to evaluate. All three entries above can be *sized*
  for $0 against the five rebuilt runs; none has been.
- Any recommendation about sequencing. These are three independent candidates,
  and nothing here claims one should come first.
