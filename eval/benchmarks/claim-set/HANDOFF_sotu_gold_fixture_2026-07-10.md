# HANDOFF — SOTU gold fixture 2026-07-10

Integration of the architect's manually-verified 15-claim SOTU gold fixture. **Read-only
over the corpus; no merges (Jackie); no kanban writes (cards proposed below only).**

## (a) Files written

| file | role |
|---|---|
| `eval/benchmarks/claim-set/sotu_gold_fixture_2026-07-10.json` | the payload, **verbatim** (15 claims, 7-value interim enum) |
| `eval/benchmarks/claim-set/sotu_gold_fixture_2026-07-10.offsets.json` | span-resolution offsets (sibling), generated |
| `eval/benchmarks/claim-set/resolve_fixture_spans.py` | reproducible span-resolution + attribution-audit tool (read-only over corpus) |

**Contamination guard — VERIFIED SAFE.** Every corpus/claim loader reads a *specific
filename* (`claim_set.train.jsonl`, `claim_set.jsonl`, `verdict_gold.train.jsonl`); no code
globs the `claim-set/` directory into a prompt (checked `src/`, `eval/`, `hydramind/` for
`glob/listdir/iterdir/scandir`). So the distinctly-named `sotu_gold_fixture_*.json` cannot
enter any model context (proposer/critic/arbiter/judge) via existing paths. **If a future
loader globs this directory, that would breach the guard — flag and stop.** Recommend
keeping gold-side fixtures on the `*_fixture_*` / `verdict_gold*` / `_labels_*` naming so a
`claim_set*`-only allowlist keeps them out of prompts.

## (b) Taxonomy mapping — interim → canonical `VerdictLabel` (v0.2.0 six-label)

Canonical (`src/truthbot/models.py::VerdictLabel`): **True · Mostly True · Misleading ·
Exaggerated · False · Unverifiable**. Proposed mapping (for Precious — **not** applied to
the fixture; labels not rewritten):

| interim (architect-interim-1) | → canonical | clean? |
|---|---|---|
| `true` | **True** | ✅ |
| `mostly_true` | **Mostly True** | ✅ |
| `misleading` | **Misleading** | ✅ |
| `false` | **False** | ✅ |
| `true_at_utterance` | **True** | ⚠️ lossy — canonical has no "true-when-said" temporal qualifier; the `reference_period` + `flags` (`revised_upward_post_utterance`, `baseline_selection`) carry that nuance |
| `unverifiable_personal_testimony` | **Unverifiable** | ⚠️ lossy — subtype (personal-testimony) collapses; preserved in `flags` |
| `normative_with_true_premise` | **(no clean target)** | ❌ **does not map** — canonical is factual-only; a "should" with a true premise isn't a truth verdict. Recommend either a new `Normative/Opinion` out-of-scope tag, or scoring only the premise (→ True) and marking the normative wrapper non-scored |

Also note **`Exaggerated`** (canonical) has **no** interim source value — interim folds
exaggeration into `misleading`/`mostly_true`. Precious to decide whether to split it out.

## (c) Span resolution — **15/15 resolved** (see `resolve_fixture_spans.py`)

Anchor match (unordered, case-insensitive) against `_sentences.jsonl` (the in-repo
sentence corpus, both speeches; Trump also has `eval/sotu-2026/transcript.txt` — no
standalone Biden transcript in-repo, so offsets are corpus-sentence-relative by `sid`).

- **14 resolved at exact sentence level in the declared speech** → sids `biden_2022:0025,
  0245, 0305, 0040, 0115, 0125, 0140, 0200, 0210, 0400`; `trump_2026:0132, 0208, 0256`.
- **`trump2026-03` (Jefferson)** resolved at **paragraph level** (`trump_2026:0699`): the
  `1826` anchor is in the adjacent sentence, not the verbatim "drew his last breath" one —
  legitimate per the "sentence/paragraph" spec, flagged as paragraph-scope.
- **`trump2026-05` (Dominican officers)** resolved **cross-speech** at `biden_2022:0325`
  (sentence level) — **confirms the architect's misattribution flag**: it is not in the
  Trump 2026 corpus.

No records failed to resolve.

## (d) Attribution audit — **277/277 match** (read-only)

Every `claim_set.jsonl` record's `speech` field matches its `sid` prefix and its text
matches the `_sentences.jsonl` sentence for that `sid`: `speech_mismatch=0, text_mismatch=0,
missing_sid=0`.

**`trump2026-05` fix (Task 4):** the Dominican-officers claim lives in the 277-set **only**
at `biden_2022:0325`, **correctly attributed to Biden 2022** — there is **no** Trump-2026
copy. So **no repo metadata correction is required**; the misattribution exists solely in
the fixture's own `trump2026-05` cataloguing (which is written verbatim and already carries
`flags:["misattributed_speaker","metadata_correction_required"]`). Recommend the architect
re-key that record to `biden2022-11` / `speech:"biden2022"` in a fixture revision.

## (e) Citation hygiene — proposed primary URLs (`resolve_url:true`)

Resolved from the **named primary** source (no secondary substitution). **Not written into
the fixture** (kept verbatim); proposed here for the architect to fold in. ✅ = confident
canonical; ⚠️ = base resolved, confirm exact issue/slug.

| claim | named source | proposed primary URL |
|---|---|---|
| biden2022-01 | North Atlantic Treaty 1949 | ✅ https://www.nato.int/cps/en/natohq/official_texts_17120.htm |
| biden2022-02 | CBO Budget Outlook Feb 2021 | ✅ https://www.cbo.gov/publication/56991 |
| biden2022-02 | Treasury Final MTS FY2022 | ⚠️ https://fiscal.treasury.gov/reports-statements/mts/ (confirm Sept-2022 final issue) |
| biden2022-03 | CDC COVID Data Tracker | ⚠️ https://covid.cdc.gov/covid-data-tracker/ (needs a Feb-2022 web.archive.org snapshot) |
| biden2022-03 | HHS hospital admissions | ⚠️ https://healthdata.gov/ (community-profile / hospital-admissions series; confirm exact dataset) |
| biden2022-04 | Blinken UNSC 2022-02-17 | ⚠️ https://www.state.gov/ (confirm exact remarks permalink) |
| biden2022-05 | BEA GDP Q4-2021 advance | ✅ https://www.bea.gov/news/2022/gross-domestic-product-fourth-quarter-and-year-2021-advance-estimate |
| biden2022-06 | WEF GCR 2019 | ⚠️ https://www.weforum.org/reports/global-competitiveness-report-2019/ (US profile, infra pillar) |
| biden2022-07 | 41 U.S.C. §8301 | ✅ https://www.law.cornell.edu/uscode/text/41/8301 |
| biden2022-08 | Gotham/Barber/Hill 2018 BMJ GH | ✅ https://gh.bmj.com/content/3/5/e000850 (PubMed 30271626) |
| biden2022-09 | 38 U.S.C. §8126 | ✅ https://www.law.cornell.edu/uscode/text/38/8126 |
| biden2022-09 | SSA §1860D-11(i) | ✅ https://www.ssa.gov/OP_Home/ssact/title18/1860D-11.htm |
| biden2022-10 | ARP ESSER (ED) | ✅ https://oese.ed.gov/offices/american-rescue-plan/american-rescue-plan-elementary-and-secondary-school-emergency-relief/ |
| trump2026-01 | congress.gov H.R.1 119th | ✅ https://www.congress.gov/bill/119th-congress/house-bill/1 |
| trump2026-02 | BLS CPI historical | ⚠️ https://www.bls.gov/cpi/data.htm (confirm exact series id for 9.1% peak / 1920 record) |
| trump2026-03 | Monticello / LoC | ⚠️ https://www.monticello.org/thomas-jefferson/a-day-in-the-life/death/ (confirm slug) |

(The `resolve_url:false` entries already carry URLs — CBS, FactCheck.org, NBC New York — and
are secondary by design for framing/plausibility support; left as-is.)

## Proposed follow-up cards (wiki-first — NOT created)

1. **Fixture-vs-verdict_gold reconciliation.** This fixture (richer `architect-interim-1`
   schema, 15 rows) overlaps my open PR #22 `verdict_gold.train.jsonl` (15 rows, 4-label).
   Decide the canonical Layer-B gold artifact and the taxonomy version (interim-1 →
   VerdictLabel v0.2.0), then converge. Owner: Precious/architect.
2. **Fixture revision r2.** Re-key `trump2026-05` → Biden 2022; fold in the resolved primary
   URLs above; adjudicate the `normative_with_true_premise` mapping + `Exaggerated` split.
3. **Attribution-audit as CI.** Wire `resolve_fixture_spans.py`'s 277-set audit into the
   test suite so any future corpus edit that breaks speaker/sid attribution fails fast.

## PR

Opened against `main`, branch `claude/sotu-gold-fixture` — see the PR this HANDOFF ships in.
No merge (Jackie's call).
