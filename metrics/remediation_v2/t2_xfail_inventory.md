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
