"""Every model we can dispatch must have a real price BEFORE it spends.

A missing ``COST_TABLE`` row does not fail — it silently substitutes generic
fallback rates, and in Jul/Aug 2026 that priced gpt-5.5's output at half its
true rate on the highest-volume OpenAI path. This file turns "someone should
re-check the rate table" into something CI enforces.
"""
from __future__ import annotations

import logging

import pytest

from truthbot.metrics.costs import (COST_TABLE, FALLBACK_COST_PER_TOKEN,
                                    estimate_cost, is_priced)
from truthbot.verify import retrievers


def _r2_models() -> set[str]:
    # gpt-5.5 is the built-in default and gpt-5-mini is pinned by the phase3 /
    # rescue / head-retrieve scripts via TRUTHBOT_R2_MODEL.
    return {"gpt-5.5", "gpt-5-mini", *retrievers._R2_FALLBACKS}


@pytest.mark.parametrize("model", sorted(_r2_models()))
def test_every_dispatchable_r2_model_is_priced(model):
    assert is_priced("openai", model), (
        f"R2 can dispatch {model!r} but COST_TABLE has no row for it, so its "
        "spend would be recorded at generic fallback rates")


@pytest.mark.parametrize("model", ["grok-4", "grok-4.3"])
def test_every_dispatchable_r3_model_is_priced(model):
    assert is_priced("xai", model), (
        f"R3 can dispatch {model!r} but COST_TABLE has no row for it")


def test_gpt55_output_rate_is_not_the_fallback():
    """The specific regression: gpt-5.5 output is $30/MTok, fallback says $15."""
    _in, _cached, out = COST_TABLE[("openai", "gpt-5.5")]
    assert out == pytest.approx(30.00 / 1_000_000)
    assert out != FALLBACK_COST_PER_TOKEN[2]


@pytest.mark.parametrize("adapter,model", [
    ("openai", "gpt-5.5"), ("openai", "gpt-5-mini"), ("xai", "grok-4.3"),
])
def test_rate_tables_agree_with_hydramind(adapter, model):
    """The repo prices a model in two tables; they must not drift apart.

    ``truthbot.metrics.costs.COST_TABLE`` prices live telemetry, while
    ``truthbot.costs.rates`` (used by the rebuild/rescue scripts) delegates to
    ``hydramind.models.RATE_TABLE_USD_PER_MTOK``. A model priced differently in
    the two would make a run's own estimate disagree with its own telemetry.
    """
    from hydramind.models import RATE_TABLE_USD_PER_MTOK

    in_rate, _cached, out_rate = COST_TABLE[(adapter, model)]
    hm_in, hm_out = RATE_TABLE_USD_PER_MTOK[model][:2]
    assert in_rate * 1_000_000 == pytest.approx(float(hm_in))
    assert out_rate * 1_000_000 == pytest.approx(float(hm_out))


def test_unpriced_model_logs_an_error_instead_of_pricing_silently(caplog):
    with caplog.at_level(logging.ERROR, logger="truthbot.metrics.costs"):
        cost = estimate_cost("openai", "gpt-does-not-exist", 1_000, 100)
    assert "No cost row" in caplog.text
    # Still returns a number — this runs inside telemetry's finally block, where
    # raising would turn a metering gap into an outage.
    assert cost > 0
