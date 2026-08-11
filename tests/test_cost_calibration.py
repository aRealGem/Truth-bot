"""The cost calibration (truthbot.costs) — offline, $0, no model anywhere.

Two estimates shipped 2.4x low in a row because three scripts each carried
their own copy of the same wrong guess. What is under test here is therefore
not "does the arithmetic run" but the three things that would let that happen
again:

  * the calibration must REPRODUCE the two ledger actuals it was fitted to;
  * the derivation must stay self-consistent (the free-text channel is what
    back-solves chars-per-token, so the two constants cannot drift apart);
  * a change in truthbot.costs must move EVERY consumer — if an estimator can
    be recalibrated without the others noticing, the third copy is back.

Plus a regression pinning the old bare chars/4 constant, and the rate that was
1.25x low, as gone.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from truthbot import costs

REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    """Load a scripts/ module by path — scripts/ is not a package."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)          # must import clean with no key present
    return mod


def _run(tag: str) -> dict:
    for r in costs.CALIBRATION_RUNS:
        if r["run"] == tag:
            return r
    raise AssertionError(f"no calibration run {tag!r}")


# ── back-test: the calibration reproduces the money that was actually spent ──

@pytest.mark.parametrize("tag", ["B1a", "B2"])
def test_backtest_reproduces_the_ledger_actual(tag):
    """Re-price each run's ACTUAL workload and land on its ACTUAL cost.

    The workload is the one recorded in CALIBRATION_RUNS, not one recomputed
    from today's artifacts, and that distinction matters: B2 rewrote
    _SCORE_SYSTEM (504 -> 1765 chars), so re-measuring B1a's prompt with the
    current code would price a prompt B1a never sent."""
    run = _run(tag)
    est = costs.estimate_scoring_cost(
        prompt_chars=run["prompt_chars"], items=run["items"],
        freetext_chars=run["freetext_chars"], model=run["model"])
    actual = run["actual_usd"]
    residual = (est["cost_usd_est"] - actual) / actual
    assert abs(residual) <= costs.TOLERANCE_RUN, (
        f"{tag}: estimated ${est['cost_usd_est']:.4f} vs ledger ${actual:.4f} "
        f"({residual:+.1%}), outside the stated +/-{costs.TOLERANCE_RUN:.0%}")
    # And the residual the module PUBLISHES must be the one it actually has,
    # so the caveat text cannot go stale while the constants move.
    assert residual == pytest.approx(costs.CALIBRATION_RESIDUALS[tag], abs=0.002)


def test_backtest_beats_the_old_estimator_by_the_missing_factor():
    """The old formula on B1a's workload was 2.42x low. Pin the correction.

    Not a style point: a budget cap set from the old number would have been
    blown by a run that was doing exactly what it said it would do."""
    run = _run("B1a")
    old = (run["prompt_chars"] / 4.0 * 0.80          # chars/4, wrong rate
           + (16 * run["calls"] + 46 * run["items"]) / 4.0 * 4.00) / 1e6
    assert old == pytest.approx(run["old_estimate_usd"], abs=0.001)
    new = costs.estimate_scoring_cost(
        prompt_chars=run["prompt_chars"], items=run["items"],
        freetext_chars=0, model=run["model"])["cost_usd_est"]
    assert new / old > 2.0


# ── the derivation itself ────────────────────────────────────────────────────

def test_chars_per_token_is_back_solved_not_the_old_rule_of_thumb():
    """No tokenizer is installed, so the constant is solved from the ledger.

    Guard both ends: it must not have drifted back to the 4.0 rule of thumb
    that under-counted by 28%, and it must stay inside a physically sane band
    (a value outside it means the fit has been contaminated, not improved)."""
    assert costs.CHARS_PER_TOKEN != 4.0
    assert 2.5 < costs.CHARS_PER_TOKEN < 4.0


def test_the_tokenizer_route_taken_is_the_absent_tokenizer_route():
    """The module claims no tokenizer was available. Verify that claim rather
    than trusting the comment — if one lands in the venv later, this fails and
    the calibration should be redone by counting instead of back-solving."""
    for name in ("tiktoken", "transformers", "tokenizers", "sentencepiece"):
        assert importlib.util.find_spec(name) is None, (
            f"{name} is now installed — truthbot.costs should be re-derived "
            "with a real token count, not a back-solve")


def test_freetext_load_is_the_fixed_point_of_chars_per_token():
    """The free-text channel is the ONE place the char count is exact and the
    marginal dollar is identified, so tokens-per-free-text-char IS the
    chars-per-token measurement. The two must stay reciprocal by construction —
    editing one without the other would silently break the derivation."""
    assert costs.REPLY_TOKENS_PER_FREETEXT_CHAR == pytest.approx(
        1.0 / costs.CHARS_PER_TOKEN)


def test_freetext_is_quantified_separately_from_the_base_reply():
    """The B2 post-mortem blamed one_line_why for out-running a 3x reply
    multiplier. The data says the 3x was already spent on the BASE reply, which
    B1a under-priced too. Both effects are real and must be separable: the base
    load is large enough that it dominated B1a (which had no free text), and
    the free text is the majority of B2's output."""
    b2 = _run("B2")
    est = costs.estimate_scoring_cost(
        prompt_chars=b2["prompt_chars"], items=b2["items"],
        freetext_chars=b2["freetext_chars"], model=b2["model"])
    assert est["tokens_out_freetext_est"] > est["tokens_out_fixed_est"]

    # ...and the free text is worth about 1.4x the base reply per item, NOT the
    # 2x-on-top that a "3x multiplier" assumed. The old multiplier was not too
    # small for the free text; it was covering a different debt.
    per_item_free = b2["freetext_chars"] / b2["items"] / costs.CHARS_PER_TOKEN
    assert 1.0 < per_item_free / costs.REPLY_TOKENS_PER_ITEM < 2.0


def test_an_item_with_no_free_text_is_cheaper_than_one_with():
    run = _run("B2")
    kw = dict(prompt_chars=run["prompt_chars"], items=run["items"],
              model=run["model"])
    bare = costs.estimate_scoring_cost(freetext_chars=0, **kw)
    with_text = costs.estimate_scoring_cost(
        freetext_chars=run["freetext_chars"], **kw)
    assert with_text["cost_usd_est"] > bare["cost_usd_est"]
    assert bare["tokens_out_freetext_est"] == 0


def test_unknown_model_prices_at_zero_rather_than_guessing():
    est = costs.estimate_scoring_cost(prompt_chars=10_000, items=10,
                                      model="no-such-model")
    assert est["cost_usd_est"] == 0.0


# ── rates: one table, corrected ──────────────────────────────────────────────

def test_haiku_rate_matches_the_proxy_that_actually_bills_it():
    """The old (0.80, 4.00) was a self-described "rough fallback" that every
    estimator read as a price. The LiteLLM proxy is configured at
    input_cost_per_token 0.000001 / output_cost_per_token 0.000005."""
    assert costs.rates("claude-haiku") == (1.00, 5.00)


def test_rates_are_not_copied_into_truthbot_costs():
    """costs.rates() must DELEGATE, so there is one table, not two."""
    from hydramind.models import RATE_TABLE_USD_PER_MTOK

    for model, expected in RATE_TABLE_USD_PER_MTOK.items():
        assert costs.rates(model) == (float(expected[0]), float(expected[1]))


# ── the shared-source wiring: one change moves every consumer ────────────────

def test_recalibrating_moves_every_estimator(monkeypatch):
    """Change the calibration in ONE place; every $0 estimator must follow.

    This is the test that would have caught the original bug: b2_primary_series
    imported rescore_stored_packs' private constants, so the two agreed with
    each other and with nothing else. Now they agree because they both read
    truthbot.costs at call time."""
    rs = _load("rescore_stored_packs")
    b2 = _load("b2_primary_series")

    art = {
        "claims": [{"sid": "s:0001", "text": "a claim with some text in it"}],
        "evidence": {"s:0001": [
            {"claim_id": "s:0001", "source_name": "AP",
             "source_url": "https://example.gov/a", "source_tier": "Government",
             "snippet": "a snippet", "supports_claim": None,
             "relevance_score": 0.5}]},
    }
    report = {"per_speech": [{"speech": "gwbush_2006", "sids": ["s:0001"]}]}
    monkeypatch.setattr(b2, "load_artifact", lambda _p: art)
    monkeypatch.setattr(b2, "artifact_path", lambda _s: Path("unused"))

    before = (rs.estimate_speech(art)["cost_usd_est"],
              b2.estimate(report)["cost_usd_est"])
    assert all(c > 0 for c in before)

    monkeypatch.setattr(costs, "REPLY_TOKENS_PER_ITEM",
                        costs.REPLY_TOKENS_PER_ITEM * 10)
    after = (rs.estimate_speech(art)["cost_usd_est"],
             b2.estimate(report)["cost_usd_est"])
    assert all(a > b for a, b in zip(after, before)), (
        f"an estimator ignored the recalibration: {before} -> {after}")


def test_phase3_reads_the_shared_rates_and_per_claim_band():
    p3 = _load("phase3_rebuild")
    assert p3.PER_CLAIM_EST == costs.PER_CLAIM_USD_PLANNING
    assert p3.MODEL_RATES == {m: costs.rates(m)
                              for m in ("gpt-5-mini", "gpt-5.5", "grok-4.3")}
    # The planning band must stay a superset of the measured one: rounding
    # INWARD would quietly shrink a budget cap.
    lo_m, hi_m = costs.PER_CLAIM_USD_MEASURED
    lo_p, hi_p = costs.PER_CLAIM_USD_PLANNING
    assert lo_p >= lo_m and hi_p >= hi_m


def test_per_claim_rate_is_ledger_derived_and_untouched_by_this_recalibration():
    """The Phase-3 per-claim figure is spend/claims from real runs, not a
    chars/4 estimate, so the tokenization fix does not move it. Pinned so a
    future reader does not "helpfully" recalibrate it too."""
    assert costs.PER_CLAIM_USD_MEASURED == (0.0642, 0.0748)


# ── regression: the old constants are gone ───────────────────────────────────

def test_the_old_private_cost_constants_no_longer_exist():
    rs = _load("rescore_stored_packs")
    for dead in ("CHARS_PER_TOKEN", "REPLY_CHARS_PER_ITEM", "REPLY_CHARS_OVERHEAD"):
        assert not hasattr(rs, dead), (
            f"rescore_stored_packs.{dead} is back — cost constants belong in "
            "truthbot.costs, where both misses can be back-tested against")


def test_no_estimator_still_divides_by_a_bare_four():
    """Grep the estimators for the chars/4 idiom. Crude on purpose: the bug was
    a literal, and a literal is what must not come back."""
    for script in ("rescore_stored_packs", "b2_primary_series", "phase3_rebuild"):
        src = (REPO / "scripts" / f"{script}.py").read_text(encoding="utf-8")
        for dead in ("/ 4.0", "/4.0", "CHARS_PER_TOKEN = 4"):
            assert dead not in src, f"{script}.py still has {dead!r}"


def test_calibration_records_its_own_provenance():
    """A future reader must be able to see what this was fitted to without
    digging through git history."""
    assert costs.CALIBRATION_ID
    tags = {r["run"] for r in costs.CALIBRATION_RUNS}
    assert tags == {"B1a", "B2"}
    for r in costs.CALIBRATION_RUNS:
        assert r["date"] and r["model"] and r["sidecars"]
        assert r["actual_usd"] > 0
    note = costs.uncertainty_note()
    assert "not a tokenizer run" in note.lower()
    assert "chars/4" not in note                  # the old, now-false caveat
    assert "two runs on one model" in note.lower()
