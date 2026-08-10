# T-2 — inventory of every runtime xfail

_2026-08-09 · $0 (test-suite bookkeeping, no model calls)._

Regenerate the raw list with:

```
.venv/bin/python -m pytest -q -rx | grep XFAIL
```

Suite state at the time of writing: **1922 passed, 4 xfailed, 19 deselected.**

## Why this inventory exists

An `xfail` is a promise that something will change. Left unlabelled it decays
into a permanent exemption — the marker says "not yet", nobody records what
"yet" means, and two months later nobody can tell a pending decision from a
quietly abandoned one.

The immediate trigger was worse than decay. The acceptance-gate header still
described `trump_2026:0469` as a **gate defect awaiting the B1a re-score**,
which directly contradicted the owner's ratified reading — that Unverifiable is
the *correct* verdict for that claim. A test header asserting the opposite of a
ratified decision is a live contradiction, not stale prose. It is fixed, and
the rest of the set is inventoried so the same thing cannot happen quietly
somewhere else.

## The four

| # | Test | sid(s) | What it asserts | Tied to |
| --- | --- | --- | --- | --- |
| 1 | `tests/acceptance/test_dc6_acceptance_gate.py::test_beckstrom_0462_split_is_awaiting_a_panel_call` | `trump_2026:0462` | The claim reaches a substantive decided verdict — decided label, not a models-split, not gate-forced | **The adjudication wave.** Flips when a panel breaks the model disagreement. **Not currently in wave scope** — see caveat below |
| 2 | `tests/acceptance/test_dc6_acceptance_gate.py::test_biden_deficit_half_stays_decided` | `biden_2022:0244` | The claim is decided (it shipped decided pre-remediation; the rebuild force-gated it to Unverifiable) | **The adjudication wave.** B1a+B2 already RELEASED it from the quality gate; release only confers eligibility, so the panel call is what flips the marker |
| 3 | `tests/acceptance/test_dc6_acceptance_gate.py::test_murder_rate_pair_is_coherent` | `trump_2026:0023` + `trump_2026:0024` | `verdict_audit.adjacent_coherence_conflicts` reports no conflict between two adjacent claims rating the same 2025 homicide statistic | **The adjudication wave**, which re-adjudicates the pair TOGETHER. Both sids are named extras, so this one **is** in scope |
| 4 | `tests/test_bluesky.py::TestBlueskyPublisher::test_post_report_returns_none_when_unconfigured` | — (not a claim) | `post_report` returns `None` when the publisher is unconfigured; today it raises `NotImplementedError` | **An event, not a decision:** the Bluesky v2 publisher landing in Phase 7 |

All four are `strict=True`. Strict is the point — when the pending thing lands,
the flip surfaces as a loud `XPASS` failure rather than a silent pass nobody
notices.

## What changed in T-2

The Beckstrom pair used to be **one** xfailed case asserting that both `0462`
and `0469` should end up decided, reasoned as "the relevance layer never ran,
so the pack-quality gate force-gated a well-sourced claim — repaired by the B1a
re-score". That reason is now wrong for one of the two sids and incomplete for
the other, so the case is split.

### `trump_2026:0469` — now PASSING

> "Sarah Beckstrom died in order to defend our capital."

The ratified conversion, and the reason it is not a gate failure: the claim has
two parts and they are not equally checkable.

- **Factual core** — she died, shot on National Guard duty near the White House
  — confirmed by NPR (Established), Axios (Other) and a Reuters-origin report
  (Other), plus unscored NBC and AP items. Not thin, and not anybody's press
  shop.
- **Purposive clause** — *in order to defend* — asserts why she was there. The
  only item in the ten-item pack that speaks to purpose at all is
  `mast.house.gov`, a House member's tribute page: **Political-tier**, which
  under the Claim Eval v3 ruling is attribution and never proof.

So Unverifiable is the correct verdict. The test asserts it, and verifies all
three limbs of the rationale against the artifact — verdict, the count of
non-Political bearing supporters, and that the set of purposive supporters is
Political-only — rather than quoting a rationale nobody checks. The rationale
string itself is pinned as `BECKSTROM_0469_RATIONALE` in the module.

### `trump_2026:0462` — stays xfail

> "After a four-month deployment, she voluntarily extended her service, and her
> rank was going to be lifted."

Ten items, six bearing, and it still ships as a models-split with **no verdict
at all**. Nothing deterministic can break a tie between models. It needs a panel
call, and the ratification of its sibling does not touch it.

## Caveat worth surfacing before the wave is scoped

`trump_2026:0462` is **not** in the re-gate's released set and **not** one of
the six named extras. On the wave scope as it currently stands, running the wave
would not touch it, and xfail #1 would still be xfail afterwards.

Two honest options, and this document does not pick one:

1. add `trump_2026:0462` to the wave (one more panel call — it is a split with
   no verdict, which is arguably the most defensible thing a wave can spend on);
2. leave it, and accept that this marker outlives the wave — in which case its
   reason string should eventually name a later event instead.

---

# Update — after the 2026-08-10 wave rulings

_2026-08-10 · $0 for this section (test-suite bookkeeping)._

Suite state: **1986 passed, 2 xfailed, 19 deselected.**

## The two

| # | Test | sid(s) | What it asserts | Tied to |
| --- | --- | --- | --- | --- |
| 1 | `tests/acceptance/test_dc6_acceptance_gate.py::test_no_published_verdict_ships_without_a_rationale` | `biden_2022:0432` | No published verdict, via any resolver path, ships with an empty rationale — publish-blocking by the R-3 ruling | **A panel call that captures seat rationales, or an owner ruling.** There is nothing to adopt: 0432 was tie-routed to MISLEADING in the phase-3 rebuild and again in the wave, and no run in its lineage ever stored a rationale for it (the pre-remediation run has it as a split with no verdict at all). Synthesizing one is precisely what R-3 forbids |
| 2 | `tests/test_bluesky.py::TestBlueskyPublisher::test_post_report_returns_none_when_unconfigured` | — (not a claim) | `post_report` returns `None` when the publisher is unconfigured; today it raises `NotImplementedError` | **An event, not a decision:** the Bluesky v2 publisher landing in Phase 7 |

Both are `strict=True`, for the same reason as before.

## What retired, and how

Three of the four above are gone, and it is worth separating the ways — a
marker that "went away" can mean four different things and only one of them is
the intended lifecycle.

* **#2 `test_biden_deficit_half_stays_decided` — RESOLVED (2026-08-09).** The
  wave's panel decided the claim TRUE, the strict marker announced it as an
  XPASS, and the case became a plain assertion of the outcome. This is the
  lifecycle working exactly as designed.
* **#3 `test_murder_rate_pair_is_coherent` — RENAMED, then REPAIRED.** The wave
  made it pass for the *wrong reason*: the checker went quiet because
  `trump_2026:0023` lost its rationale, not because the claims stopped
  contradicting each other. Renaming it rather than converting it is what kept
  a detection blind spot from being laundered into a green gate — and the
  renamed case is what R-3 was written from. The rationale is now re-emitted
  from stored panel output, the pair ships annotated under the D14 disposition,
  and the case keeps its name because what it proves is unchanged.
* **#1 `test_beckstrom_0462_split_…` — RESOLVED BY POLICY (2026-08-10).** Not by
  a panel call. The wave made the call the marker was waiting for and the seats
  split again, so the escalation question was ruled instead:
  persistent-split-after-2 publishes as Models-Split. The assertion INVERTS —
  the claim is now expected to ship as a split, and the case guards that as the
  stable outcome.

The caveat at the end of the T-2 section was resolved by taking option (1):
`trump_2026:0462` was added to the wave. It did not flip, which is what made
the policy necessary.

## The one that arrived

`biden_2022:0432` was not on anyone's list. It surfaced only because R-3 turned
the no-blank-rationale check into a corpus-wide lint — the same defect as
`trump_2026:0023`, in a second claim, sitting unnoticed through the phase-3
rebuild and the wave. That is the argument for the lint being publish-blocking
rather than advisory: the first instance was found by reading an artifact
closely, and the second was found by a machine in one pass.

---

# Update — after the R-3 escape run (2026-08-10, later)

_2026-08-10 · $0.0602 of a $0.25 cap (two panel calls, no retrieval)._

Suite state: **2000 passed, 1 xfailed, 19 deselected.**

## The one

| # | Test | sid(s) | What it asserts | Tied to |
| --- | --- | --- | --- | --- |
| 1 | `tests/test_bluesky.py::TestBlueskyPublisher::test_post_report_returns_none_when_unconfigured` | — (not a claim) | `post_report` returns `None` when the publisher is unconfigured; today it raises `NotImplementedError` | **An event, not a decision:** the Bluesky v2 publisher landing in Phase 7 |

No claim-level xfail remains. Every marker that was tied to a *decision* has
now been resolved one way or another; the survivor is tied to a feature that
has not been built.

## What retired, and how

* **`test_no_published_verdict_ships_without_a_rationale` — RESOLVED (the
  intended lifecycle).** `biden_2022:0432` got the panel call the marker named,
  the strict marker announced the flip, and the case is now a plain assertion
  that the corpus has no blank-rationale verdict at all. A second case,
  `test_0432_says_why_after_the_escape_run`, pins the specific claim so the
  repair cannot regress silently into the aggregate.

## How the call was made without widening the wave

Both claims sat OUTSIDE the flip set, and `--sids` refuses out-of-set sids by
design. Rather than edit the set — which would have made "the wave" mean two
different things in two reports — `scripts/wave_adjudicate.py` gained an
audited escape: `--extra-sids`, which requires a `--reason`, requires its own
`--tag` (so the run's report, diffs and journals cannot overwrite the wave's),
sources the artifacts from the current publishing head, and records the reason
and the sids in the run report and in the artifact meta. `wave_set()` returns
exactly what it returned before, and a test asserts that.

The same change made `--sids` enforce the contract it always documented: an
out-of-set sid is now REFUSED instead of silently dropped.

## Both claims moved — one of them against expectation

* **`biden_2022:0432` — MISLEADING → Models split.** The old label came from
  the stage-2 discriminator routing a three-way tie, which stored a label and
  no text. The fresh panel split three ways again (True / False /
  Unverifiable), but every seat's rationale is stored this time, so the claim
  publishes as a split that shows all three readings instead of asserting
  Misleading with nothing behind it. A published verdict was withdrawn;
  `r3_corrections_entries.json` records it.

* **`trump_2026:0462` — Models split → UNVERIFIABLE. FLAG FOR THE OWNER.** The
  escalation ruling reasoned that "a third identical panel call would buy the
  same answer at the same price". It did not. The proposer moved from True to
  Unverifiable, leaving a 2-1 plurality with rationale text. The
  persistent-split POLICY is untouched — what stopped holding is its
  precondition on this claim — but the published outcome changed as a result
  of a call made for a different purpose (capturing rationales), and that is an
  owner-revisitable outcome rather than an agent-settled one. The prior
  artifact (`46dfcce8`) is untouched on disk if the owner wants the split back;
  reverting is a re-pin of `RUNS["trump_2026"]` in the acceptance gate.
