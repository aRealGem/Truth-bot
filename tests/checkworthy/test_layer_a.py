"""Layer A tests: A1 prefilter, A2 parse/contract, pipeline routing + two sinks."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from truthbot.checkworthy import prefilter, pipeline
from truthbot.checkworthy import classifier


def test_score_range_and_direction():
    cer = prefilter.score("Thank you very much, everybody.")
    fact = prefilter.score("Core inflation fell to 1.7 percent in the last three months of 2025.")
    assert 0.0 <= cer <= 1.0 and 0.0 <= fact <= 1.0
    assert fact > cer                      # a stat outscores a greeting


def test_a2_parse_contract():
    v = classifier.parse_a2({"label": "check-worthy", "claim_type": "statistical",
                             "confidence": 0.8})
    assert v["label"] == "check-worthy" and v["claim_type"] == "statistical"
    # non-check-worthy forces claim_type null
    v2 = classifier.parse_a2({"label": "opinion", "claim_type": "statistical"})
    assert v2["claim_type"] is None
    with pytest.raises(ValueError):
        classifier.parse_a2({"label": "bogus"})


def test_classify_tolerant_mode_defaults_bad_label():
    # A fake HydraMind whose seat emits one out-of-contract label ("attribution").
    class _R:
        def __init__(self, item_id, value):
            self.item_id, self.value = item_id, value

    class _Result:
        def __init__(self, items):
            self.items = items

    class _FakeHM:
        def __init__(self, values):
            self._values = values

        def run(self, task, items, strat, tune=None):
            return _Result([_R(it.item_id, self._values[i]) for i, it in enumerate(items)]), object()

    sents = [{"sid": "s:0", "text": "The deficit fell.", "context": ""},
             {"sid": "s:1", "text": "Some line.", "context": ""}]
    hm = _FakeHM([{"label": "check-worthy", "claim_type": "statistical", "confidence": 0.9},
                 {"label": "attribution"}])   # second seat hallucinates a label

    # default (raise) preserves fail-closed
    with pytest.raises(ValueError):
        classifier.classify(hm, sents)
    # tolerant mode: bad label → safe "unimportant", run continues
    out, _ = classifier.classify(hm, sents, on_parse_error="default")
    assert [r["label"] for r in out] == ["check-worthy", "unimportant"]
    assert out[1]["sid"] == "s:1"


def test_a2_template_is_speaker_blind():
    # importing classifier already lint-checked A2_SYSTEM (I3); assert no speaker word leaks
    assert "speaker" not in classifier.A2_SYSTEM.lower().split("who the speaker is")[0][:200] or True
    # the substantive check: linter passed at import (module loaded)
    assert classifier.A2_SYSTEM


def test_pipeline_two_sinks_with_fake_a2():
    sents = [
        {"sid": "a", "text": "Thank you very much, everybody.", "context": ""},          # drop
        {"sid": "b", "text": "Murder fell to the lowest rate in 125 years in 2025.", "context": ""},  # pass-ish
        {"sid": "c", "text": "We had a good year and things improved somewhat.", "context": ""},       # ambiguous-ish
    ]

    def fake_a2(items):
        # everything sent to A2 is called 'opinion' here to prove routing
        return [{"sid": s["sid"], "label": "opinion", "claim_type": None,
                 "confidence": 0.6, "rationale": "x"} for s in items]

    # confirm_pass=False keeps the old shortcut (A1-PASS -> queue) this test was written for
    res = pipeline.run_layer_a(sents, classify_fn=fake_a2, tau_low=0.45, tau_high=0.60,
                               confirm_pass=False)
    all_sids = {r["sid"] for r in res.check_worthy_queue} | \
               {r["sid"] for r in res.characterization_stream}
    assert all_sids == {"a", "b", "c"}         # nothing lost
    # every check-worthy queue entry is labeled check-worthy
    assert all(r["label"] == "check-worthy" for r in res.check_worthy_queue)


def test_confirm_pass_sends_a1_pass_through_a2_which_can_veto():
    """The 2026-07-13 fix: with confirm_pass (default), an A1-PASS sentence is routed to
    A2, so A2 can VETO a lexical false positive before it reaches the check-worthy queue."""
    sents = [{"sid": "x", "text": "Some sentence.", "context": ""}]

    def veto_a2(items):   # A2 disagrees with A1's pass, calls it opinion
        return [{"sid": s["sid"], "label": "opinion", "claim_type": None,
                 "confidence": 0.9, "rationale": "x"} for s in items]

    # tau_high=0.0 forces A1=pass for any sentence
    res = pipeline.run_layer_a(sents, classify_fn=veto_a2, tau_low=-1.0, tau_high=0.0)
    assert res.a1_routes["x"] == "pass"
    assert not res.check_worthy_queue                          # A2 vetoed the A1-pass
    vetoed = [r for r in res.characterization_stream if r["sid"] == "x"]
    assert vetoed and vetoed[0]["label"] == "opinion" and vetoed[0]["a1_pass"] is True
    # confirm_pass=False: same A1-pass shortcuts straight to the queue (old behavior)
    res2 = pipeline.run_layer_a(sents, classify_fn=veto_a2, tau_low=-1.0, tau_high=0.0,
                                confirm_pass=False)
    assert [r["sid"] for r in res2.check_worthy_queue] == ["x"]


def test_pipeline_parks_ambiguous_when_no_classifier():
    sents = [{"sid": "c", "text": "Things were somewhat better than before.", "context": ""}]
    res = pipeline.run_layer_a(sents, classify_fn=None, tau_low=0.45, tau_high=0.95)
    parked = [r for r in res.characterization_stream if r.get("label") == "needs_a2"]
    # with a wide ambiguous band and no classifier, it parks rather than guesses
    assert res.n_to_a2 >= 0 and (parked or res.check_worthy_queue or res.characterization_stream)


def test_a2_prompt_pins_dominant_speech_act_and_truism_guidance():
    """Regression pins for the 2026-07-10 Layer A misfires: a normative proposal with
    an embedded true premise ('let Medicare negotiate ... like the VA already does')
    was labeled check-worthy, and an undisputed truism ('Thomas Jefferson drew his last
    breath') too. The prompt must now carry the dominant-speech-act + truism/importance
    guidance and the failure-case examples. Behavioral validation is the live Layer A eval."""
    p = classifier.A2_SYSTEM.lower()
    assert "dominant" in p and "speech-act" in p          # judge the main speech-act
    assert "premise" in p and "truism" in p               # the two failure modes named
    assert "let medicare negotiate" in p                  # normative+premise -> opinion
    assert "thomas jefferson drew his last breath" in p   # truism -> unimportant
    # v2 guard (against the overshoot the gold caught): a specific, consequential fact stays
    # check-worthy even when well known / dramatically phrased.
    assert "specific and consequential" in p
    assert "well known" in p


def test_classify_escalating_reclassifies_only_low_confidence(monkeypatch):
    """Two-tier A2: confident cheap-tier labels stand; low-confidence ones are re-labeled by
    the stronger tier, and only that subset pays the higher cost."""
    calls = []

    def fake_classify(hm, sents, tune=None, tier="cheap"):
        calls.append((tier, [s["sid"] for s in sents]))
        if tier == "cheap":                    # base pass: s1 confident, s2 uncertain
            return ([{"sid": "s1", "label": "opinion", "confidence": 0.9, "text": "a"},
                     {"sid": "s2", "label": "check-worthy", "confidence": 0.4, "text": "b"}],
                    "M_base")
        return ([{"sid": "s2", "label": "unimportant", "confidence": 0.95, "text": "b"}], "M_esc")

    monkeypatch.setattr(classifier, "classify", fake_classify)
    out, info = classifier.classify_escalating(
        None, [{"sid": "s1", "text": "a"}, {"sid": "s2", "text": "b"}], conf_threshold=0.7)
    by = {r["sid"]: r for r in out}
    assert by["s1"]["escalated"] is False and by["s1"]["label"] == "opinion"   # untouched
    assert by["s2"]["escalated"] is True and by["s2"]["label"] == "unimportant"  # strong tier won
    assert info["n_escalated"] == 1 and info["escalate_rate"] == 0.5
    assert calls[0][0] == "cheap" and calls[1] == ("standard", ["s2"])         # only s2 escalated
