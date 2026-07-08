"""Task 2 — Layer A → Layer B wiring: classifier.classify must carry the claim
`text` through A2 rows so run_layer_a's check-worthy queue feeds run_layer_b
directly (sid + text preserved). Offline — scripted single-strategy lane."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from hydramind import HydraMind
from hydramind.types import Call, CallResult, Lane
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, call_key
from hydramind.manifest import NullSpendSink

from truthbot.checkworthy import classifier, pipeline as a_pipeline
from truthbot.verdict import pipeline as b_pipeline


def _scripted_checkworthy():
    """Every A2 call comes back 'check-worthy' so items land in the Layer B queue."""
    def fn(call: Call) -> CallResult:
        return CallResult(
            call=call,
            output={"label": "check-worthy", "claim_type": "statistical",
                    "confidence": 0.9, "rationale": "r"},
            lane=Lane.L_P, cost_usd=0.0, cost_source="none",
            tokens_in=10, tokens_out=5, returned_model=call.binding.model)
    return fn


def _hm():
    return HydraMind(load_registry(SPECS_DIR),
                     Transport(completion_fn=_scripted_checkworthy()),
                     spend_sink=NullSpendSink())


def test_classify_carries_text_through_a2():
    sents = [{"sid": "s1", "text": "Core inflation fell to 1.7% in 2025.", "context": "ctx"}]
    rows, _ = classifier.classify(_hm(), sents)
    assert rows[0]["sid"] == "s1"
    assert rows[0]["text"] == "Core inflation fell to 1.7% in 2025."
    assert rows[0]["context"] == "ctx"
    assert rows[0]["label"] == "check-worthy"


def test_layer_a_to_b_pass_preserves_sid_and_text():
    sents = [
        {"sid": "s1", "text": "Core inflation fell to 1.7% in 2025.", "context": ""},
        {"sid": "s2", "text": "Unemployment hit a 50-year low last quarter.", "context": ""},
    ]
    hm = _hm()

    def classify_fn(items):
        rows, _ = classifier.classify(hm, items)
        return rows

    # full_speech routes everything through A2 so the A2 text-carry path is exercised
    a = a_pipeline.run_layer_a(sents, classify_fn=classify_fn, full_speech=True)
    assert {r["sid"] for r in a.check_worthy_queue} == {"s1", "s2"}
    assert all(r.get("text") for r in a.check_worthy_queue)     # text survived A2

    seen = []

    def verdict_fn(claims):
        # the injected Layer B fn sees sid + text directly from the A queue
        for c in claims:
            assert c["sid"] and c["text"]
            seen.append((c["sid"], c["text"]))
        return [{"sid": c["sid"], "status": "resolved", "verdict": "TRUE",
                 "confidence": 0.8, "citations": [], "votes": {}, "split": False,
                 "escalated": False} for c in claims]

    b = b_pipeline.run_layer_b(a.check_worthy_queue, verdict_fn=verdict_fn)
    assert {sid for sid, _ in seen} == {"s1", "s2"}
    assert {t for _, t in seen} == {s["text"] for s in sents}
    assert all(r["status"] == "resolved" for r in b.verdicts)
