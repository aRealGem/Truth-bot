"""The cost model must reproduce the live-measured dev-lot cost, and prod shapes
must stay far under the old $120/10-speeches hypothesis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "benchmarks"))
import cost_model as cm


def test_dev_projection_reproduces_measured_devlot():
    # live dev-lot: ~$0.010 for 25 claims at escalation 0.32 (label_mismatch)
    per_claim = cm.per_claim(cm.DEV, cm.MEASURED_LABEL_MISMATCH)
    for_25 = per_claim * 25
    assert abs(for_25 - 0.010) < 0.0015, f"{for_25} not within tolerance of measured $0.010"


def test_prod_is_far_under_the_120_hypothesis():
    for roster in (cm.PROD_OPUS_ARBITER, cm.PROD_GPT_ARBITER):
        total = cm.project(roster, cm.MEASURED_LABEL_MISMATCH)["total"]
        assert total < 120, total
        assert total < 10           # in practice single-digit dollars for 10 speeches
    # even at the old higher escalation, still nowhere near $120
    assert cm.project(cm.PROD_GPT_ARBITER, cm.MEASURED_MATERIAL_DISAG)["total"] < 120


def test_grok_dominates_prod_critic_cost():
    # the every-claim grok critic is the prod cost driver, not the dev seats
    assert cm.call_cost("grok-4.5", "critic") > 10 * cm.call_cost("dsv4-flash", "critic")


def test_opus_subscription_zeroes_the_arbiter_term():
    assert cm.call_cost("claude-opus-sub", "arbiter") == 0.0
    # so escalation rate does not change $ when the arbiter is subscription-covered
    lo = cm.project(cm.PROD_OPUS_ARBITER, 0.10)["total"]
    hi = cm.project(cm.PROD_OPUS_ARBITER, 0.90)["total"]
    assert lo == hi
