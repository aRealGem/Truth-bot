#!/usr/bin/env python3
"""
Batch≡interactive equivalence check (design §2 invariant, spec §5.4).

Runs the SAME N items through L-P (proxy) and L-B (native batch) with identical
prompts/schemas and diffs the outputs. L-P and L-B differ only in cost/latency,
never behavior — a mismatch must be explained.

The diff logic is pure and unit-tested with fakes; the LIVE run needs both a
proxy virtual key (L-P) and the anthropic SDK + ANTHROPIC_API_KEY (L-B), sourced
from the repo .env (CW-12).
"""
from __future__ import annotations
import json, sys
from pathlib import Path


def diff_outputs(lp: dict, lb: dict, keys=("label", "claim_type", "verdict")) -> list[dict]:
    """Compare two {item_id: output} maps on the given semantic keys.
    Returns a list of mismatch records (empty ⇒ equivalent)."""
    mism = []
    for item_id in sorted(set(lp) | set(lb)):
        a, b = lp.get(item_id, {}), lb.get(item_id, {})
        for k in keys:
            if k in a or k in b:
                if a.get(k) != b.get(k):
                    mism.append({"item_id": item_id, "key": k,
                                 "L-P": a.get(k), "L-B": b.get(k)})
    return mism


def run_live(n=10):  # pragma: no cover - needs live lanes
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hydramind import HydraMind
    from hydramind.transport import Transport, ProxyCompletion
    from hydramind.anthropic_batch import AnthropicBatchBackend
    from truthbot.checkworthy import classifier

    train = [json.loads(l) for l in
             (Path(__file__).parent / "claim-set" / "claim_set.train.jsonl")
             .read_text().splitlines()[:n]]
    sents = [{"sid": r["sid"], "text": r["text"], "context": r.get("context", "")}
             for r in train]

    hm_lp = HydraMind.from_specs_dir(
        transport=Transport(completion_fn=ProxyCompletion(response_parser=classifier.parse_a2)))
    hm_lb = HydraMind.from_specs_dir(
        transport=Transport(
            completion_fn=ProxyCompletion(),
            batch_backend=AnthropicBatchBackend(response_parser=classifier.parse_a2)))

    lp, _ = classifier.classify(hm_lp, sents)
    # force L-B by dropping min_lot below n
    lb, _ = classifier.classify(hm_lb, sents, tune={"batch.min_lot": 1})
    lp_map = {r["sid"]: r for r in lp}
    lb_map = {r["sid"]: r for r in lb}
    mism = diff_outputs(lp_map, lb_map)
    print(json.dumps({"n": n, "mismatches": mism,
                      "equivalent": not mism}, indent=2))


if __name__ == "__main__":
    run_live(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
