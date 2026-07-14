# Step 1: confirm_pass validated on a real speech (Trump SOTU 2026)

Runner: `layer_a_speech_diff.py trump_2026 cheap` (A2 = haiku-v2, the chosen model). Live,
2026-07-13, A2 spend **$0.16** (142 sentences classified once, memoized across both runs).

Ran the full Trump 2026 speech through `run_layer_a` with `confirm_pass=False` (old: A1-PASS
goes straight to the check-worthy queue) vs `confirm_pass=True` (new default: A1-PASS is
confirmed/vetoed by A2). `confirm_pass` only touches the PASS band, so `Q_true ⊆ Q_false`
and the delta is exactly the A1 lexical false positives A2 caught.

## Headline: the veto nearly doubles queue precision

Check-worthy queue **precision vs the 150-row gold** (scored on the gold-labeled subset):

| | precision | scored |
|---|---|---|
| confirm_pass=OFF | 0.47 | 25/53 |
| confirm_pass=ON  | **0.83** | 20/24 |

A2 vetoed **55** A1-PASS sentences. Of the 29 that are gold-labeled: **24 correctly** (13
opinion + 11 unimportant), 5 were gold check-worthy (recall cost, below). The other **26 are
not in the gold**, but on inspection are overwhelmingly correct vetoes — rhetoric, greetings,
poetry, and procedural filler that A1's lexical prefilter wrongly passed, e.g.:

- `0700` "Thomas Jefferson drew his last breath." → unimportant  (the exact truism the fix targeted)
- `0456` "I'm asking this Congress to pass tough legislation…" → opinion (policy proposal)
- `0468` "Shouldn't have been in our country." → opinion
- `0648` "Nice to have you back, Enrique." → unimportant  (greeting)
- `0716` "From the sun kissed shores of Florida to the endless fields of the Dakotas." → unimportant (poetry)
- `0712` "And when God needs a nation to work his miracles, He knows exactly who to ask." → opinion

So the mechanism does exactly what #25 intended: it stops A1's lexical false positives (the
"should"-proposals and ceremonial truisms) from reaching the expensive PCA panel.

## Recall cost: 5 of 25 queued check-worthy vetoed (0.80) — mostly not A2's fault

| sid | text (truncated) | A2 call | assessment |
|---|---|---|---|
| `0656` | "This was a major military installation protected by thousands of soldiers…" | unimportant ("appears fictional") | **A2 error** — haiku hallucinated a fictional scenario. This is the one anchor sonnet gets right and haiku misses (see gold-150 scoring). |
| `0480` | "Sarah and Andrew, both shot violently in the head, neither was expected to…" | unimportant (personal anecdote) | borderline — specific medical facts, but personal-anecdote framing with no policy stakes. Defensible either way. |
| `0544` | "They found all 28." | unimportant ("too vague/context-dependent") | borderline — a hostage-recovery count in context, vague in isolation. |
| `0184` | "…an unfortunate ruling from the United States Supreme Court, it just came down." | vetoed | leans check-worthy (a real SCOTUS ruling) — A2 slightly over-vetoed. |
| `0424` | "She gets much better bipartisan support than I do." | vetoed | **likely gold error** — subjective comparative, not a checkable fact. A2 is arguably right; flag the gold row for re-review. |

Net: the confirm_pass change trades **+0.36 precision** (0.47→0.83) for **−0.20 recall** on the
queued gold subset — and of that recall loss, only ~1 case (`0656`) is a clear A2 mistake, ~3
are borderline, and 1 (`0424`) is a probable gold labeling error where the veto is correct.

## Decisions & follow-ups

- **Keep `confirm_pass=True` as the default.** Validated: it removes far more junk than it costs
  in real check-worthy, and the junk it removes is exactly the lexical-prefilter overshoot.
- **haiku-v2 stands as the A2 model** (gold-150 scoring). Residual: the `0656`-type
  "narrative-sounds-fictional" miss is haiku's known weakness; if that class recurs at scale,
  it is the argument for sonnet on the PASS/ambiguous band — revisit only if it shows up again.
- **Gold hygiene:** re-review `trump_2026:0424` (subjective → likely opinion, not check-worthy)
  and `trump_2026:0184`. Small, folds into the next gold pass.
