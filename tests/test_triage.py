"""Triage escalation helpers."""

from __future__ import annotations

import random

from truthbot.models import Confidence, ModelVerdict, VerdictLabel
from truthbot.verify.triage import (
    confidence_numeric,
    should_shadow_sample,
    triage_unanimous_high_conf,
)


def test_confidence_numeric_ordering():
    assert confidence_numeric(Confidence.HIGH) > confidence_numeric(Confidence.MEDIUM)


def test_triage_unanimous_high_conf_requires_two_plus_and_threshold():
    v = ModelVerdict(
        adapter_name="a",
        model_id="m",
        claim_id="c",
        label=VerdictLabel.TRUE,
        confidence=Confidence.HIGH,
        explanation="x",
    )
    assert triage_unanimous_high_conf([v], 0.8) is False
    v2 = ModelVerdict(
        adapter_name="b",
        model_id="m2",
        claim_id="c",
        label=VerdictLabel.TRUE,
        confidence=Confidence.HIGH,
        explanation="y",
    )
    assert triage_unanimous_high_conf([v, v2], 0.8) is True
    w = v.model_copy()
    w.label = VerdictLabel.FALSE
    assert triage_unanimous_high_conf([v, w], 0.8) is False


def test_should_shadow_sample_deterministic():
    rng = random.Random(0)
    assert should_shadow_sample(0.0, rng) is False
    assert should_shadow_sample(1.0, rng) is True
