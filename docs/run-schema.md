# Run schema — verdict vocabulary, fold rules, and the canonical count

*A9 (remediation v2), 2026-08-07. Written so an external reviewer can
reproduce every published count from the artifacts alone, **without reading
`scripts/dc6_package.py`.** If this note and the packager ever disagree, that
is a bug in one of them — say so rather than trusting the code because it is
code.*

Companion note: [`claim-count-reconciliation.md`](claim-count-reconciliation.md)
covers the older 289-vs-277 question (published corpus vs gold eval corpus).
This note covers the five-speech remediation-v2 corpus.

---

## 1. What a run artifact contains

`metrics/pca_runs/<run_id>.json`:

| key | meaning |
|---|---|
| `meta` | speaker, date, `speech_id`, `rebuild_of`, `pipeline_generation` |
| `claims` | one record per check-worthy claim: `{sid, text, context, …}` |
| `rows` | one adjudication row per claim: `{sid, verdict, split, citations, reasoning, evidence_gate, …}` |
| `evidence` | `sid → [pack items]`, **in pack order** — item *n* is `E<n>` |
| `characterization`, `roster`, `composition` | panel + pack composition telemetry |

`claims` and `rows` are joined on `sid`. In a healthy artifact the two sets are
identical; see §5 for the one historical case where they were not.

Citations in `rows[].citations` are E-refs into that claim's `evidence` list by
**position** (`E4` = the 4th item). The published artifact stores no `pack_id`
on the item itself, so position is the addressing.

---

## 2. The verdict vocabulary

A row's outcome is one of six labels. Derivation, in this order (the first rule
that matches wins) — this is `phase3_rebuild.outcome_label`:

1. **`gated-UNVERIFIABLE`** — `evidence_gate` (or legacy `provenance_code`) ==
   `insufficient-qualifying-evidence`. The pack failed the quality gate, so the
   verdict was FORCED, whatever the panel thought. **The panel's opinion is not
   consulted.**
2. **`TRUE` / `MOSTLY TRUE` / `MISLEADING` / `FALSE` / `UNVERIFIABLE`** — the
   panel's own label, taken from `verdict` when it is non-null.
3. **`Models split`** — `verdict` is null and `split` is true: the panel reached
   no plurality.
4. **`No verdict`** — `verdict` is null and `split` is false: no row was
   produced at all.

### Two different Unverifiables

This is the distinction that most often gets flattened, so it is worth being
blunt about it:

* **panel `UNVERIFIABLE`** — the panel looked at a good pack and ruled *this
  claim cannot be checked*. A judgement.
* **`gated-UNVERIFIABLE`** — the pipeline refused to let the panel's verdict
  publish because the pack did not clear the evidence gate. A **withholding**,
  not a judgement.

They **publish the same badge** ("Unverifiable"), which is why a move between
them is invisible on the site and cannot be expressed as a public correction —
`data/corrections.json` accepts only TRUE / FALSE / MISLEADING / UNVERIFIABLE
and rejects an entry whose old and new verdict are equal. Those moves are still
reported, under `non_ledger_changes` in
`metrics/remediation_v2/dc6_corrections_entries.json`.

---

## 3. Folding into the published buckets

Six contract labels fold onto five published display buckets:

| contract label | published bucket |
|---|---|
| `TRUE` | True |
| `MOSTLY TRUE` | Mostly True |
| `MISLEADING` | Misleading |
| `FALSE` | False |
| `UNVERIFIABLE` | Unverifiable |
| `gated-UNVERIFIABLE` | Unverifiable |
| `Models split` | Models split |
| `No verdict` | Models split |

Two folds lose information on purpose, and both are named above: the two
Unverifiables collapse, and `No verdict` renders as a split (from a reader's
point of view "the models did not converge" is the same fact either way).
"Mostly True" is carried through every table even though this corpus contains
none of it — a distribution that silently drops a live label is exactly the
class of bug remediation v2 exists to kill.

---

## 4. What "decided" means

> **decided = every claim NOT in an abstain bucket.**
> **abstain = {Unverifiable, Models split}** (after the §3 fold).

So a decided claim is one carrying a substantive published ruling: True,
Mostly True, Misleading, or False. `decided-rate = decided / total`.

Both Unverifiables count as abstentions. That is deliberate: from the reader's
side, a claim withheld by the gate and a claim the panel could not settle are
the same non-answer.

### The anecdote-adjusted variant (A10)

Some claims are personal anecdotes about private individuals — a guest in the
gallery, a constituent's story. They come back Unverifiable because no
independent public record of a private person exists, which is the *correct*
outcome, not a gate failure. Counting them as abstentions makes a speech look
less decidable in proportion to how many guests it named.

So the decided-rate is reported **both ways**:

* **raw** — every compared claim in the denominator;
* **anecdote-adjusted** — claims whose `layer_a_claim_type` is
  `personal-anecdote` excluded from the denominator entirely.

Both are computed over the same base (the sids the rebuild compared) so they
differ *only* by the exclusion. Claims that carry no `layer_a_claim_type` in the
artifact and do not join to the published `claims.json` by (speaker, normalised
text) stay in the adjusted denominator as non-anecdotes — an assumption, and
reported as one (`join.unresolved`), because guessing "not an anecdote" moves
the number in a known direction.

The two bases can disagree about the headline finding, and when they do the
report says so rather than picking one.

---

## 5. The canonical count — 529

The record disagreed with itself: **529** in the handoff, **530** in commit
`e268dec`'s DC-4' tally, and **183 vs 182** Trump rows. All three were true of
something, which is why the disagreement survived. Measured:

| basis | claims | rows | orphan rows |
|---|---|---|---|
| old (pre-remediation) artifacts | 529 | 530 | `trump_2026:0311` |
| **new (rebuilt) artifacts — canonical** | **529** | **529** | none |
| published `site-pca/data/claims.json` | 530 records | — | 1 renders "(claim text unavailable)" |

### The canonical statement

> **The corpus is 529 claims.** The published 530 is **529 + 1 orphan row**
> (`trump_2026:0311`) — a verdict row with no matching claim record, which the
> site renders as "(claim text unavailable)". The rebuilt artifacts carry no
> orphans: rows without a matching claim record = 0.

### Named exclusions

* `trump_2026:0311` — an orphan row. The pre-remediation Trump artifact carried
  183 rows against 182 claim records; the extra row had a verdict (FALSE) and
  no claim, so the published card had nothing to show. **No reader ever saw a
  fact-check here** — dropping it removes a broken placeholder, not a checked
  claim. It is ledgered as `dropped_rows` in
  `metrics/remediation_v2/dc6_corrections_entries.json`, separately from the
  verdict corrections, because it changes a *count*, not a *verdict*, and has
  no old→new verdict pair to express.

### Why the old denominators differ

The old corpus's decided-rate is over **530** rows and the new over **529**.
The reports do not normalise that away; where a rebuild changed the
denominator, the comparison says so (`denominator_mismatch`,
`raw_matches_section4`) rather than quietly picking one.

---

## 6. Reproducing the counts

Everything below is offline and $0.

```python
import json, glob

RUNS = {"gwbush_2006": "74a89c5f", "clinton_1998": "d0010426",
        "obama_2014": "4de8a551", "biden_2022": "37744fc8",
        "trump_2026": "4ee5a251"}
ABSTAIN = {"UNVERIFIABLE", "gated-UNVERIFIABLE", "Models split", "No verdict"}

def label(row):
    if (row.get("evidence_gate") or row.get("provenance_code") or "") == \
            "insufficient-qualifying-evidence":
        return "gated-UNVERIFIABLE"
    if row.get("verdict") is not None:
        return str(row["verdict"])
    return "Models split" if row.get("split") else "No verdict"

claims = rows = decided = 0
for prefix in RUNS.values():
    art = json.load(open(glob.glob(f"metrics/pca_runs/{prefix}*.json")[0]))
    sids = {c["sid"] for c in art["claims"]}
    assert [r for r in art["rows"] if r["sid"] not in sids] == []   # no orphans
    claims += len(art["claims"])
    rows += len(art["rows"])
    decided += sum(1 for r in art["rows"] if label(r) not in ABSTAIN)

print(claims, rows, decided, round(decided / rows, 4))   # 529 529 420 0.794
```

The invariants above are asserted in `tests/test_canonical_counts.py`; the
whole-package regeneration is `scripts/dc6_package.py`.
