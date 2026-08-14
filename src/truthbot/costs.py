"""Calibrated $0 cost estimation — the SINGLE source every estimator reads.

Every ``--estimate`` surface in the repo used to carry its own copy of the same
three guesses (chars/4, a fixed reply-character shape, a hand-waved multiplier
for free-text fields). Three copies meant three chances to be wrong, and the
same wrongness shipped twice:

  * B1a full re-score  — estimated $0.4391, actual $1.0632 (2.42x)
  * B2 subset re-score — estimated $0.2299, actual $0.5404 (2.35x)

This module holds the constants ONCE, fitted to those two runs' ledger actuals,
so a consumer cannot silently drift from the calibration.


WHAT WENT WRONG (three independent errors, all in the same direction)
---------------------------------------------------------------------
1. The RATE was wrong. ``hydramind.models.RATE_TABLE_USD_PER_MTOK`` priced
   ``claude-haiku`` at $0.80/$4.00 per Mtok, flagged in its own comment as a
   "rough fallback". The LiteLLM proxy that actually billed these runs is
   configured at ``input_cost_per_token: 0.000001`` / ``output_cost_per_token:
   0.000005`` — $1.00/$5.00. A flat 1.25x, on both sides, before any token was
   counted. The table entry has been corrected to match the proxy config.

2. The chars-per-token constant was too generous. 4.0 chars/token is the
   English-prose rule of thumb; this corpus is JSON scaffolding wrapped around
   URLs, source names and clipped snippets, which tokenizes DENSER. The
   ledger-derived value is ~3.13 (see below) — a further 1.28x on the input.

3. The reply was modelled as the JSON we can reconstruct, not the JSON the
   model emits. The old estimator priced ``{"i": N, "relevance": 0.0,
   "supports": null}`` at 46 characters/item ≈ 11.5 tokens. The fitted per-item
   output load is ~32.5 tokens — the model pretty-prints, and whitespace is
   billed. THIS is the dominant term: output is ~66% of B1a's true cost.

Note what error 3 means for the B2 post-mortem. B2's commit message blamed the
free-text ``one_line_why`` for out-running the "3x reply multiplier". The data
says otherwise: the 3x was entirely consumed covering the BASE reply shape that
B1a had too and also under-priced. Measured per item, B1a's stanceless reply is
32.5 output tokens and B2's is ~79 (32.5 base + 146.2 free-text chars ÷ 3.13),
which is 2.4x — LESS than the 3x that was applied. The free-text is real and
large (59% of B2's output tokens) but it was never the multiplier's problem;
the multiplier was paying off a debt the base shape had already run up.


HOW THE CONSTANTS WERE DERIVED (read this before trusting a number)
-------------------------------------------------------------------
ROUTE TAKEN: empirical back-solve from ledger actuals. NOT a tokenizer run.

The venv was inspected first, as the honest route would have been to count real
tokens: no ``tiktoken``, no ``transformers``, no ``tokenizers``, no
``sentencepiece``, and ``anthropic`` 0.97 vendors no local tokenizer (its
``count_tokens`` is a network call, and this work is hard-capped at $0). Adding
one would have meant a network dependency. So the tokens are back-solved from
money that was actually spent, which is the one measurement this repo can make
for free and cannot fool itself about.

The fit has 10 observations — five speeches x two runs, each speech's
``spend_usd`` being an independent ``proxy_key_spend`` delta banked in
``metrics/remediation_v2/rescored[_b2]_<speech>.json``. The model is

    usd = prompt_chars/CHARS_PER_TOKEN * rate_in
        + (items*REPLY_TOKENS_PER_ITEM + freetext_chars/CHARS_PER_TOKEN) * rate_out

with rate_in/rate_out taken as MEASURED (the proxy config), the prompt
characters measured EXACTLY (``relevance.score_payload`` re-run over the
untouched stored artifacts; snippets top out at 207 chars, well under the 400
cap, so nothing is clipped), and the free-text characters measured EXACTLY from
the 1,028 ``one_line_why`` strings B2 stored.

``CHARS_PER_TOKEN`` is solved as a FIXED POINT: the free-text channel is the one
place where the character count is exactly known and the marginal dollar cost is
identified (B1a has no free text, B2 does — the contrast isolates it), so the
fitted tokens-per-free-text-character IS a chars-per-token measurement for this
corpus and model. Iterating until the constant used on the input side equals the
one the output side implies gives 3.1276. It is then applied to the input side
too; prompt JSON tokenizes at least as densely as prose, so if anything this is
the conservative direction.


HONEST UNCERTAINTY — what these numbers are NOT
------------------------------------------------
* This is a FIT TO TWO RUNS ON ONE MODEL (claude-haiku via the local LiteLLM
  proxy), on one corpus (five rebuilt SOTU PCA artifacts), on one prompt family
  (``relevance.score_evidence``). It is not a measured law and must not be
  quoted as one. A different model, a different reply schema, or a corpus with
  different snippet composition invalidates it.
* The input and output shares are only WEAKLY separable from the ledger alone.
  Prompt characters and item count are near-collinear across these ten
  observations; an unconstrained three-parameter regression puts the input
  coefficient at -0.19 +/- 1.21 micro-USD/char, i.e. the ledger cannot see it.
  The split here rests on the fixed-point argument above, not on the regression.
  Total cost is what is pinned down; the in/out attribution is structural.
* Per-RUN residuals are good (see CALIBRATION_RESIDUALS: +1.1% on B1a, -0.3% on
  B2). Per-SPEECH residuals are not: they run to +17% (gwbush_2006 in B1a) and
  -7% (clinton_1998 in B2). gwbush was the first speech in both sequences and
  reads cheap in both, which is what a lagging spend counter looks like — the
  LiteLLM key's spend is written asynchronously, so a delta read right after a
  leg can land short and push the shortfall into the next leg. Treat a
  single-speech estimate as +/-20%, a whole-run estimate as +/-5%.
* Retries are invisible. ``_score_one`` re-sends a pack that comes back
  unchanged; all ten sidecars record zero soft failures, but a first-attempt
  retry that then succeeded would be billed and unlogged, and would sit inside
  these constants as if it were normal volume.

Because the constants are fitted, an estimate built on them is a PLANNING
number, not a promise. The funded paths keep their budget breakers.
"""
from __future__ import annotations

from typing import Optional

# ── provenance ───────────────────────────────────────────────────────────────

#: Bump when the constants below are refitted, so an artifact carrying an
#: estimate can be traced to the calibration that produced it.
CALIBRATION_ID = "haiku-score-2026-08-09"

#: The exact runs the fit was made against. speech-level actuals live in the
#: named sidecars; these totals are their sums.
CALIBRATION_RUNS: tuple[dict, ...] = (
    {"run": "B1a", "date": "2026-08-08", "model": "claude-haiku",
     "sidecars": "metrics/remediation_v2/rescored_<speech>.json",
     "calls": 529, "items": 4344, "prompt_chars": 1153624, "freetext_chars": 0,
     "actual_usd": 1.063245, "old_estimate_usd": 0.4391},
    {"run": "B2", "date": "2026-08-08", "model": "claude-haiku",
     "sidecars": "metrics/remediation_v2/rescored_b2_<speech>.json",
     "calls": 115, "items": 1028, "prompt_chars": 412532, "freetext_chars": 150292,
     "actual_usd": 0.540510, "old_estimate_usd": 0.2299},
)

#: Signed relative error of the calibrated estimator against each run's ledger
#: actual, at the rounded constants published below. Asserted by the tests.
CALIBRATION_RESIDUALS: dict[str, float] = {"B1a": +0.0106, "B2": -0.0029}

#: How far a single estimate may reasonably be off. Run-level is the fitted
#: residual with headroom; speech-level is the observed per-speech spread.
TOLERANCE_RUN = 0.05
TOLERANCE_SPEECH = 0.20


# ── the calibrated constants ─────────────────────────────────────────────────

#: Characters per token for this corpus + model. LEDGER-DERIVED, not a
#: tokenizer run — see the module docstring. The repo's old bare 4.0 is gone:
#: it under-counted tokens by 28%.
CHARS_PER_TOKEN = 3.13

#: Output tokens billed per scored evidence item, EXCLUDING any free-text
#: field. Fitted. Covers the reply's per-item JSON object as the model actually
#: formats it (indentation and newlines included), which is ~2.8x the compact
#: object a reconstruction from the stored scores would suggest.
REPLY_TOKENS_PER_ITEM = 32.5

#: Output tokens per character of free-text reply content. Free text is prose,
#: so this is 1/CHARS_PER_TOKEN by construction — the fixed point the
#: calibration solves for.
REPLY_TOKENS_PER_FREETEXT_CHAR = 1.0 / CHARS_PER_TOKEN

#: Mean characters of ``one_line_why`` per item, over the 1,028 replies B2
#: stored (median 145, p90 186, capped at ONE_LINE_WHY_CHARS=240 twice). Used
#: to PROJECT free-text volume for a run that has not happened yet.
FREETEXT_CHARS_PER_ITEM = 146.2

#: Full-stack PCA cost per claim, USD. LEDGER-DERIVED per-claim actuals from
#: the three non-resumed phase-3 rebuild runs (gwbush $0.0642, clinton $0.0744,
#: trump $0.0748 — metrics/remediation_v2/dcb1_estimate.json). This is NOT a
#: chars/4 estimate and is therefore UNAFFECTED by the recalibration above: it
#: was obtained by dividing money actually spent by claims actually adjudicated.
#: The two resumed legs are excluded because their banked cost drops the
#: off-proxy component. Caveat that survives: the off-proxy R2/R3 share inside
#: these actuals is priced from provider-reported token counts at list rates,
#: so it is token-metered but not ledger-checked.
PER_CLAIM_USD_MEASURED: tuple[float, float] = (0.0642, 0.0748)

#: The planning band a budget cap should use: PER_CLAIM_USD_MEASURED rounded
#: OUTWARD for headroom. Deliberately wider than the measurement.
PER_CLAIM_USD_PLANNING: tuple[float, float] = (0.065, 0.080)

#: Cost per claim to RE-adjudicate on a STORED evidence pack — no new retrieval,
#: just the panel call over evidence already on disk (F8). LEDGER-DERIVED from
#: the three 2026-08 stored-pack reuse runs, measured off the LiteLLM proxy spend
#: ledger (state basis, not a chars/4 estimate):
#:   * R-1 reshape  — 1 claim,  $0.0036 total → $0.0036/claim  (band LOW)
#:   * the wave     — 29 claims, $0.3815 total → $0.0132/claim  (band HIGH)
#:   * R-3 escape   — 2 claims,  $0.0602 total → $0.0301/claim  (OUTLIER)
#: The band is the two clean single-adjudication runs (R-1, wave); the R-3
#: per-claim runs high because trump_2026:0462 took three panel calls to break a
#: persistent split, so it is not a clean single reuse and is excluded from the
#: band while recorded here for honesty. The retrieval-BEARING per-claim constant
#: (PER_CLAIM_USD_MEASURED) is unaffected and stays pinned — pack reuse is the
#: no-retrieval floor beneath it.
PACK_REUSE_USD_MEASURED: tuple[float, float] = (0.0036, 0.0132)

# ── payload schema versioning (D17-c wave 2) ─────────────────────────────────
#
# A per-claim cost constant is only meaningful for the PAYLOAD SHAPE it was
# measured under. PACK_REUSE_USD_MEASURED was measured on packs whose items
# carried {id, source, tier, url, snippet}. D17-c added ``series_rows``, and on
# the escalation run those rows were 91.2% of the panel payload — an 11.3x
# inflation, and 31x on trump_2026:0054 alone (2,986 -> 93,740 characters).
# Priced with the old constant, the run came in 8.2x over: $0.3266 against
# $0.0396. The constant did not drift; it was applied to a payload it never
# measured.
#
# So a constant now NAMES the schema it was measured under, and pricing REFUSES
# on a mismatch rather than warning. A warning would have been ignored at 21:00
# exactly as a warning always is.

#: Payload shape the pack-reuse constants were measured against.
PAYLOAD_SCHEMA_PACK_V2 = "pack-item-payload v2 (no series_rows)"

#: Payload shape once D17-c series excerpts ride on the items.
PAYLOAD_SCHEMA_SERIES_V1 = "pack-item-payload v3 (series_rows)"

#: MEASURED on the d17c-wave2 escape run: $0.3266 of ledger spend over 3 claims
#: and 104,547 payload characters. Ledger-derived, not a chars/4 estimate.
SERIES_PAYLOAD_USD_PER_CLAIM: float = 0.1089
SERIES_PAYLOAD_USD_PER_KCHAR: float = 0.003124

#: Which constant is valid for which payload shape.
_SCHEMA_OF_CONSTANT = {
    "PACK_REUSE_USD_MEASURED": PAYLOAD_SCHEMA_PACK_V2,
    "PER_CLAIM_USD_MEASURED": PAYLOAD_SCHEMA_PACK_V2,
    "SERIES_PAYLOAD_USD_PER_CLAIM": PAYLOAD_SCHEMA_SERIES_V1,
    "SERIES_PAYLOAD_USD_PER_KCHAR": PAYLOAD_SCHEMA_SERIES_V1,
}


class PayloadSchemaMismatch(ValueError):
    """A cost constant was applied to a payload shape it never measured."""


def payload_schema_for(items) -> str:
    """The schema of a pack payload, read from the payload itself."""
    for it in items or []:
        if isinstance(it, dict) and it.get("series_rows"):
            return PAYLOAD_SCHEMA_SERIES_V1
    return PAYLOAD_SCHEMA_PACK_V2


def check_constant_applies(constant_name: str, items) -> None:
    """Refuse — never warn — when a constant does not match the payload.

    This is the guard that would have caught the 8.2x miss before it was spent
    rather than after: PACK_REUSE_USD_MEASURED against a payload carrying
    series_rows is a category error, and the price of missing it is real money.
    """
    want = _SCHEMA_OF_CONSTANT.get(constant_name)
    if want is None:
        raise PayloadSchemaMismatch(
            f"{constant_name} declares no payload schema — a constant that "
            "cannot name what it measured is a proxy, not a measurement")
    got = payload_schema_for(items)
    if want != got:
        raise PayloadSchemaMismatch(
            f"{constant_name} was measured under {want!r} but this payload is "
            f"{got!r}. Re-measure before pricing; do not scale the old number.")


# ── rates ────────────────────────────────────────────────────────────────────

def rates(model: str) -> tuple[float, float]:
    """(USD per Mtok in, USD per Mtok out) for ``model``; (0.0, 0.0) if unknown.

    Delegates to ``hydramind.models.RATE_TABLE_USD_PER_MTOK`` rather than
    keeping a second copy — that table's claude-haiku entry has been corrected
    to the price the LiteLLM proxy is actually configured with, which is what
    billed the calibration runs."""
    from hydramind.models import RATE_TABLE_USD_PER_MTOK

    r = RATE_TABLE_USD_PER_MTOK.get(model)
    return (float(r[0]), float(r[1])) if r else (0.0, 0.0)


# ── the estimator ────────────────────────────────────────────────────────────

def estimate_scoring_cost(*, prompt_chars: int, items: int,
                          freetext_chars: Optional[float] = None,
                          model: str = "claude-haiku") -> dict:
    """Price a relevance-scoring workload from MEASURED prompt volume.

    ``prompt_chars`` is the exact character count of everything sent (system +
    user payload), summed over calls — measure it, never guess it.
    ``items`` is the number of evidence items scored, which is what the reply
    length scales with.
    ``freetext_chars`` is the projected characters of free-text reply content
    (``one_line_why``). None means "the live scorer asks for it" and projects
    ``FREETEXT_CHARS_PER_ITEM * items``; pass 0 for a reply schema that has no
    free-text field (the pre-B2 contract).

    Returns the cost and its parts, so a caller can show its work."""
    if freetext_chars is None:
        freetext_chars = FREETEXT_CHARS_PER_ITEM * items
    r_in, r_out = rates(model)
    tok_in = prompt_chars / CHARS_PER_TOKEN
    tok_out_fixed = REPLY_TOKENS_PER_ITEM * items
    tok_out_free = REPLY_TOKENS_PER_FREETEXT_CHAR * freetext_chars
    tok_out = tok_out_fixed + tok_out_free
    cost = (tok_in * r_in + tok_out * r_out) / 1_000_000.0
    return {
        "model": model,
        "calibration_id": CALIBRATION_ID,
        "prompt_chars": int(prompt_chars),
        "items": int(items),
        "freetext_chars_est": round(freetext_chars),
        "tokens_in_est": round(tok_in),
        "tokens_out_est": round(tok_out),
        "tokens_out_fixed_est": round(tok_out_fixed),
        "tokens_out_freetext_est": round(tok_out_free),
        "rate_in_usd_per_mtok": r_in,
        "rate_out_usd_per_mtok": r_out,
        "cost_usd_est": round(cost, 4),
    }


def uncertainty_note(*, model: str = "claude-haiku") -> str:
    """The caveat an estimate must print. Replaces the old, now-false line
    'token counts are a chars/4 approximation, not a tokenizer run' — which was
    true about the method and silent about the fact that it was 2.4x low."""
    b1a, b2 = CALIBRATION_RUNS
    return (
        f"Uncertainty: token counts are NOT a tokenizer run — no tokenizer is "
        f"installed and this path is $0/offline. {CHARS_PER_TOKEN} chars/token "
        f"and {REPLY_TOKENS_PER_ITEM} output tokens/item are BACK-SOLVED from "
        f"ledger actuals ({CALIBRATION_ID}): the {b1a['run']} run "
        f"(${b1a['actual_usd']:.4f}) and the {b2['run']} run "
        f"(${b2['actual_usd']:.4f}), {b1a['date']}, model {model}, over 10 "
        f"per-speech proxy_key_spend deltas. Back-test residuals "
        f"{CALIBRATION_RESIDUALS['B1a']:+.1%} (B1a) and "
        f"{CALIBRATION_RESIDUALS['B2']:+.1%} (B2). This is a FIT TO TWO RUNS ON "
        f"ONE MODEL, not a measured law: expect +/-{TOLERANCE_RUN:.0%} on a "
        f"whole run and +/-{TOLERANCE_SPEECH:.0%} on a single speech, and "
        f"nothing at all outside this prompt family. Haiku is ON-PROXY, so the "
        f"funded run's real cost stays LEDGER-TRUE (proxy_key_spend) and the "
        f"breaker, not this number, is what stops the spend."
    )
