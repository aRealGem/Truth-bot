#!/usr/bin/env python3
"""
Build 8 — PCA dev-lot on roster.dev (P=mistral, C=dsv4-flash, A=claude-haiku).

Runs n≈20–30 falsifiable TRAIN claims (check-worthy rows; NOT heldout) through the
pca state machine, closed-book (empty evidence pack ⇒ citations must be [], I4).
Reports split rate, escalation rate, $/claim, $/100-sentences-equiv, lane_tally,
model mismatches, and emits P80 spend rows. Also asserts the L-B routing logic
(min_lot=100 won't trigger at this n, but the decision function is verified).

Env: source the repo .env (LITELLM_PCA_KEY, LITELLM_BASE_URL).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hydramind import HydraMind, Wave, Call, Kind, PromptRef, ModelBinding
from hydramind.transport import Transport, ProxyCompletion
from hydramind.anthropic_batch import AnthropicBatchBackend
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.manifest import NullSpendSink

HERE = Path(__file__).parent
TRAIN = HERE / "claim-set" / "claim_set.train.jsonl"
N = 25

_VERDICTS = "TRUE | FALSE | MISLEADING | UNVERIFIABLE"
_CONTRACT = ('Return JSON only: {"verdict": "%s", "confidence": 0.0-1.0, '
             '"citations": [], "reasoning": "one clause"}. Closed-book: no '
             'external evidence is provided, so cite nothing (citations must be []). '
             'If the claim cannot be adjudicated from general knowledge, verdict=UNVERIFIABLE.'
             % _VERDICTS)
PROMPTS = {
    "proposer": "You are the PROPOSER. Assess the factual claim and draft a verdict. " + _CONTRACT,
    "critic":   "You are the CRITIC. Independently and skeptically assess the same claim; "
                "try to find why a naive verdict could be wrong. " + _CONTRACT,
    "arbiter":  "You are the ARBITER. Adjudicate the claim decisively. " + _CONTRACT,
}


def parse_verdict(raw: dict) -> dict:
    v = (raw.get("verdict") or "").strip().upper()
    if v not in {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}:
        v = "UNVERIFIABLE"
    c = raw.get("confidence")
    try:
        c = float(c)
    except (TypeError, ValueError):
        c = None
    return {"verdict": v, "confidence": c, "citations": list(raw.get("citations", []))}


def assert_routing():
    """Verify lane_for_wave: L-P below min_lot, L-B at/above with a batch backend."""
    reg = load_registry(SPECS_DIR)
    spec = reg["pca"]
    tr = Transport(completion_fn=lambda c: None, batch_backend=AnthropicBatchBackend())
    mk = lambda n: Wave(calls=[Call("proposer", f"i{i}", PromptRef.of("p", "{input}"),
                                    ModelBinding("anthropic", "claude-haiku", "cheap"))
                               for i in range(n)], batchable=True, tag="wave1")
    lane_small = tr.lane_for_wave(mk(25), spec).value
    lane_big = tr.lane_for_wave(mk(100), spec).value
    tr_nobatch = Transport(completion_fn=lambda c: None)
    lane_nobackend = tr_nobatch.lane_for_wave(mk(100), spec).value
    assert lane_small == "L-P", lane_small
    assert lane_big == "L-B", lane_big
    assert lane_nobackend == "L-P", lane_nobackend      # no backend ⇒ never L-B
    return {"n25": lane_small, "n100_with_backend": lane_big, "n100_no_backend": lane_nobackend}


def main():
    if not os.environ.get("LITELLM_PCA_KEY"):
        print("BLOCKED: LITELLM_PCA_KEY not in env; source repo .env."); return
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]
    claims = [r for r in rows if r["label"] == "check-worthy"][:N]
    items = [{"item_id": r["sid"],
              "payload": {"claim": r["text"], "context": r.get("context", ""),
                          "evidence_pack_ids": []}} for r in claims]

    routing = assert_routing()
    print(f"# routing assertion PASS: {routing}")

    hm = HydraMind(load_registry(), Transport(
        completion_fn=ProxyCompletion(key_env="LITELLM_PCA_KEY", response_parser=parse_verdict)),
        spend_sink=NullSpendSink())
    result, manifest = hm.run("verdict", items, "pca", roster="dev",
                              tune={"prompts": PROMPTS})

    mism = manifest.model_mismatches()
    n = len(result.items)
    resolved = sum(1 for r in result.items if r.kind.value == "resolved")
    print(f"\n# PCA dev-lot (roster.dev) n={n} claims")
    print(f"  split_rate      = {result.notes['split_rate']:.3f}")
    print(f"  escalation_rate = {result.notes['escalation_rate']:.3f}  "
          f"(criterion: {result.notes['split_criterion']})")
    print(f"  escalation stub = {result.notes['escalation']['trigger']} "
          f"(frontier_threshold={result.notes['escalation']['frontier_confidence_threshold']})")
    print(f"  resolved={resolved}  disagreement_flagged={result.notes['flagged']}")
    print(f"  lane_tally      = {manifest.lane_tally}")
    print(f"  model_mismatches= {mism if mism else '[]'}")
    print(f"  tokens in/out   = {manifest.total_tokens_in}/{manifest.total_tokens_out}")
    # per-seat model summary
    seats = {}
    for c in manifest.cost_records:
        seats.setdefault(c.role, (c.model, c.returned_model))
    print(f"  seats           = {seats}")
    print(f"\n# spend rows (P80): {json.dumps(manifest.to_spend_records())}")
    Path(HERE / "examples" / "manifest.pca-devlot.json").write_text(manifest.to_json())
    print("# manifest → examples/manifest.pca-devlot.json")
    # persist per-item verdicts for provenance
    Path(HERE / "examples" / "pca-devlot-verdicts.json").write_text(json.dumps(
        [{"sid": r.item_id, "kind": r.kind.value, **r.value, "agreement": r.agreement}
         for r in result.items], indent=2))


if __name__ == "__main__":
    main()
