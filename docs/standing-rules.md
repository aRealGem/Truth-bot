# Standing rules (S-*) and measurements (M-*)

Definitions for the standing rules the decision docs and kanban notes cite by
number. Until Wave A (2026-08-19) these lived only in the owner/Fable session
and the wiki — `grep S-11` found nothing in-repo, so a reader following a
decision doc had no way to look a rule up. This file is the in-repo home;
decision docs cite the number and link here.

Scope note: these are PROCESS rules about how changes are made and reviewed.
They do not alter pipeline semantics, and where a decision doc records a rule's
*execution* (e.g. D17-candidates' S-9 head-lineage table) that record stays
authoritative for its event.

---

## S-8 — Corrections prose is the owner's voice

Reader-facing corrections-ledger prose is published in the owner's voice.
Agents DRAFT entries (clearly marked, e.g. `DRAFT-FOR-OWNER-REDPEN`, kept out
of the rendered ledger); the owner red-pens and their text replaces the draft
before anything publishes. A correction that ships un-red-penned is a process
violation, not a formatting nit.

## S-9 — Quarantine, not rm; stacked PRs; the owner holds merge

Nothing is deleted on the way out of production: superseded artifacts and
publishing heads move to the gitignored `_quarantine/` (or live on in git
history) so every state has a path back. Work lands as stacked PRs — each
reviewable and revertable on its own — and ONLY the owner merges. A permission
denial on a merge is a hard halt, and the click reverts to the owner.
Execution record: `docs/decisions/D17-candidates.md` ("Head lineage (S-9)")
keeps the live-head run ids greppable for the moment S-9 executes.

## S-11 — One publish wave

The site publishes in WAVES: one reviewed render per wave, no mid-wave
re-renders for isolated fixes. A defect found mid-wave is logged (kanban +
decision doc) and batched to the NEXT wave — e.g. the vp-selfsource-chip
naming/semantics mismatch, logged 2026-08-17 on CW-134 and fixed in Wave A
rather than re-rendering DC-6' for a copy change. Rationale: every published
byte-delta is a reviewable event; drip re-renders make the diff unreviewable.

## M-6 — Evenhandedness, including genre-property disclosure

Any rule or label whose effect CONCENTRATES on one speaker or era is measured
and disclosed rather than silently shipped. The measurement:
`scripts/d15_d16_era_breakdown.py` →
`metrics/remediation_v2/d15_d16_era_breakdown.{json,md}` (see
`docs/decisions/D15-utterance-derivative.md`, `D16-statistical-release.md`) —
net per-speech effect, decided-rate on both bases, era concentration. The
disclosure: where a concentration is a property of a speech's GENRE (rhetoric
dominance, personal-anecdote density), the rendered page says so as a genre
property, never leaving the concentration to read as a finding about the
speaker (the `vp-genre-note` on report pages, Wave A A3).

## M-11 — Rulings bind on pushed artifacts only

Review (Fable) rules only on PUSHED artifacts: report head SHAs with every
result; unpushed work is invisible and does not count as done, verified, or
even claimed. Corollary: "committed locally" is a working state, not a
reviewable one — push before reporting.

Second corollary: **a passing postcondition is not a passing build.** Verifying
the specific thing you set out to change (a tree-equality check, a rendered
diff, a targeted test) says nothing about the state of the suite around it.
Establish that the artifact you produced is green as a whole, not merely
correct in the dimension you were watching.

## M-12 — Publish checklist

The publication surface is `main/site-pca/` served via GitHub Pages (see
STATUS.md). Every publish to main must satisfy, in order:

1. Pre-publish: full suite green on the branch being published.
2. The publish postcondition itself (tree-equality of the rendered site
   against the accepted preview render).
3. **Post-publish: full suite green ON the publish commit.**

Step 3 is not implied by step 1 or step 2. It was added 2026-08-21 after the
D17-c publish (`d63ec5b`) left `main` red: the publish moved the corpus, six
derived constants in the DC-6' acceptance fixture went stale, and the gap was
invisible because the ordered postcondition in step 2 passed. Nothing about
the rendered page was wrong; the build was simply never re-checked as a whole.
