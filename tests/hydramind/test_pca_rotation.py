"""Regression: pool-mode arbiter must honor `rotation: round_robin`.

Before the fix, `pca._seat_bindings` resolved the arbiter pool to a single binding
(providers[0]) and the wave-2 loop's `idx % len(arb_bindings)` degenerated to 0, so
every gated item was arbitrated by the same provider (anthropic) — the declared
spread across [anthropic, openai, gemini] never happened, concentrating provider
risk and cost on the priciest, most consequential calls.
"""
from hydramind import HydraMind
from hydramind.types import Call, CallResult, Lane
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, call_key
from hydramind.manifest import NullSpendSink


def _scripted(outputs):
    def fn(call: Call) -> CallResult:
        return CallResult(call=call, output=outputs[call_key(call)], lane=Lane.L_P,
                          cost_usd=0.001, tokens_in=10, tokens_out=5,
                          returned_model=call.binding.model)
    return fn


def test_pool_arbiter_round_robin_spreads_across_providers():
    """3 gated items over the frontier arbiter pool → 3 distinct providers.

    pca.yaml arbiter = {providers: [anthropic, openai, gemini], tier: frontier,
    rotation: round_robin}; frontier tier resolves to claude-opus / gpt-4o /
    gemini-pro. `flow.gate: always` forces every item to the arbiter regardless of
    agreement, so all three seats fire.
    """
    reg = load_registry(SPECS_DIR)
    ids = ["c1", "c2", "c3"]
    outs = {}
    for cid in ids:
        for role in ("proposer", "critic", "arbiter"):
            outs[f"{role}:{cid}"] = {"verdict": "TRUE", "confidence": 0.9, "citations": []}
    hm = HydraMind(reg, Transport(completion_fn=_scripted(outs)), spend_sink=NullSpendSink())
    items = [{"item_id": cid, "payload": {"claim": cid}} for cid in ids]

    _, manifest = hm.run("verdict", items, "pca", tune={"flow.gate": "always"})

    arb_models = [c.model for c in manifest.cost_records if c.role == "arbiter"]
    assert len(arb_models) == 3                       # every item escalated
    # the fix: providers rotate; pre-fix this was ["claude-opus"] * 3
    assert set(arb_models) == {"claude-opus", "gpt-4o", "gemini-pro"}


def test_pool_arbiter_rotation_wraps_when_more_items_than_providers():
    """5 gated items over a 3-provider pool → round-robin wraps (opus appears twice)."""
    reg = load_registry(SPECS_DIR)
    ids = [f"c{i}" for i in range(5)]
    outs = {}
    for cid in ids:
        for role in ("proposer", "critic", "arbiter"):
            outs[f"{role}:{cid}"] = {"verdict": "TRUE", "confidence": 0.9, "citations": []}
    hm = HydraMind(reg, Transport(completion_fn=_scripted(outs)), spend_sink=NullSpendSink())
    items = [{"item_id": cid, "payload": {"claim": cid}} for cid in ids]

    _, manifest = hm.run("verdict", items, "pca", tune={"flow.gate": "always"})

    arb_models = [c.model for c in manifest.cost_records if c.role == "arbiter"]
    assert len(arb_models) == 5
    assert set(arb_models) == {"claude-opus", "gpt-4o", "gemini-pro"}   # all three used
