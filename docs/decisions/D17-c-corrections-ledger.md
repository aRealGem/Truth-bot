# Corrections ledger — wave 2, item 4

**Status: owner-approved text, 2026-08-13. Not yet published.**

Approved by the owner in the wording below. Item 2 uses "not retrieved before
publication" rather than "never retrieved" — the weaker and equally true form,
chosen deliberately: the stronger claim rests on inference about our own
pipeline (assembly-stamp spacing plus an absent cache file), and a public
correction should not lean on inference when a directly-supported form says
enough.

The browsing-model provenance is deliberately **out** of the published text. It
is true and it is the root cause, but it is a statement about how the pipeline
works rather than about the correction, and it invites a question the ledger
does not answer. It belongs in the D17 record, where it is already logged.

Publication is not scheduled here. This ships with wave 2 and has no
dependency on the stable-ids work — it can go ahead of it or alongside it,
whichever the owner gate decides.

---

## Published text

> **CORRECTIONS**
>
> 1. In the note published with the source-audit banner, I wrote that 48 of the
>    flagged items are statistical series. That is wrong for 12 of them. The
>    Congressional Budget Office, the Government Accountability Office and the
>    National Center for Education Statistics publish tabled reports, not
>    series. The banner's "at most those" is an upper bound and remains true;
>    what was inaccurate was the description of what the items are.
>
> 2. Two pieces of evidence cited a Federal Reserve Economic Data (FRED)
>    address built from a Bureau of Labor Statistics series number:
>    `fred.stlouisfed.org/series/LNS12000000`. That address does not resolve
>    and never did. It returns "not found" on FRED and on ALFRED, FRED's
>    archive of earlier data vintages.
>
>    It appeared twice: on the claim that more Americans are working today than
>    at any time in the country's history, and on the claim immediately
>    following it, where it was additionally recorded as SUPPORTING that claim.
>    No position should have been recorded for or against a claim on the
>    strength of a source that was not retrieved before publication. Both
>    citations are withdrawn.
>
>    For context, and with no correction implied for either: the same
>    underlying Bureau of Labor Statistics series is cited elsewhere in the same
>    report at its working address, `data.bls.gov/timeseries/LNS12000000`. Those
>    citations stand. The fault was one malformed FRED address, not a bad
>    series.
>
> 3. The previously noted dropped-row correction stands as issued.

---

## What each item rests on

**Item 1** — owner-approved wording correction carried from PR #105. The
12 items are the CBO/GAO/NCES document publishers, identified in the wave-1
scoping work; see `D17-candidates.md`.

**Item 2** — the dead address appears twice in the shipped trump head
`91dd7a34`: `trump_2026:0054` E8 (`supports_claim=None`) and
`trump_2026:0055` E9 (`supports_claim=True`, the stance recorded against an
unresolvable source). "and never did" rests on the ALFRED result specifically:
ALFRED archives earlier data vintages, so a 404 there indicates the address was
never valid rather than having lapsed. The working siblings are
`trump_2026:0054` E2 and `trump_2026:0584` E4, both on `data.bls.gov`; a third
claim, `trump_2026:0584`, restates the employment line using that working
address, which is part of why the fault is the address and not the series.

**Item 3** — pre-existing, unchanged, reissued as-is.
