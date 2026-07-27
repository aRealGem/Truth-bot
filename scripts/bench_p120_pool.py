#!/usr/bin/env python3
"""P120 A/B bench: Phase R retrieval, SERIAL vs the adaptive POOL, on the gold set.

Runs the SAME gold claims through the exact prod v2 pack builder
(``_build_v2_pack_builder``, grok-fallback) two ways —

  * SERIAL  : ``build_packs_phase(governor=None)`` (pre-P120 behavior)
  * POOLED  : ``build_packs_phase(governor=PoolGovernor(...))`` (P120 L1 trio fan-out
              within a claim + L2 claims-in-flight, sized by Pi pressure)

— and reports wall-clock (total + per claim), the L2 telemetry, and
``decisive_source_recall`` for BOTH sides (reusing scripts/pilot_evidence_v2's
scorer) so we can see the speed win AND confirm quality does not regress.

NOTE: these are two SEPARATE live retrieval passes, so packs are not byte-identical
(web search is non-deterministic run-to-run) — the byte-exact split==inline /
pooled==serial parity is proven offline in the unit tests. Here recall is a
practical no-regression check, and wall-clock is the real measurement.

Retrieval R2/R3 are OFF-proxy, so $ is not on the LiteLLM ledger; the two runs make
the same retriever calls, so cost is ~equal by construction. R1 is subscription.

Usage (repo root; needs repo .env + ~/.env; R2/R3 model overrides for metered runs):
  PYTHONPATH=. .venv/bin/python scripts/bench_p120_pool.py --limit 8 \
      --out metrics/p120_pool_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pilot_evidence_v2 import FIXTURE, SPEECHES, recall_for_pack  # noqa: E402


def _pack_recall(pack, gold_evidence):
    urls = [it.source_url for it in pack.items]
    tiers = [it.tier.value for it in pack.items]
    return recall_for_pack(gold_evidence or [], urls, tiers)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _detail(packs, gold_by_sid):
    """Per-claim {items, recall} — pack size is the diagnostic: if pooled packs are
    the SAME size as serial but recall differs, the delta is source-selection noise;
    if pooled packs are systematically THINNER, that's concurrency contention."""
    out = {}
    for sid, p in packs.items():
        r = _pack_recall(p, gold_by_sid[sid])
        out[sid] = {"items": len(p.items), "recall": r["recall"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="gold claims to bench (0 = all 15)")
    ap.add_argument("--pool-max", type=int, default=3)
    ap.add_argument("--r1-cli-cap", type=int, default=2)
    ap.add_argument("--order", choices=("pooled-first", "serial-first"),
                    default="pooled-first",
                    help="which config runs FIRST — controls the R1-Max-drawdown confound")
    ap.add_argument("--out", default="metrics/p120_pool_bench.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="build claims + assert no contamination; no retrieval spend")
    args = ap.parse_args()

    from truthbot.pipeline import _build_v2_pack_builder
    from truthbot.verdict import retrieval_phase
    from truthbot.verdict.pool_governor import PoolGovernor
    from truthbot.verdict.speech_context import register_speech_date
    from truthbot.verify.retrievers import assert_no_contamination, build_retrieval_prompt
    from truthbot.verdict.speech_context import expected_claim_window as _ecw

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    all_claims = fixture["claims"]
    claims_in = all_claims[:args.limit] if args.limit else all_claims

    # T2.6 contamination guard: nothing from the gold verdicts/rationales/evidence
    # may reach a retriever prompt. Assert BEFORE spending a cent.
    gold_fragments = []
    for c in all_claims:
        gold_fragments.append(str(c.get("verdict_provisional") or ""))
        gold_fragments.append(str(c.get("rationale") or ""))
        for e in c.get("evidence") or []:
            gold_fragments.append(str(e.get("supports") or ""))

    claims, gold_by_sid = [], {}
    for i, c in enumerate(claims_in):
        sid_prefix, utt = SPEECHES[c["speech"]]
        sid = f"{sid_prefix}:9{i:03d}"
        register_speech_date(sid, utt)
        text = c["paraphrase"]
        assert_no_contamination(
            build_retrieval_prompt(text, utterance=utt, window=_ecw(utt)), gold_fragments)
        claims.append({"sid": sid, "text": text, "context": ""})
        gold_by_sid[sid] = c.get("evidence") or []
    print(f"Benched {len(claims)} gold claim(s); contamination guard passed.")
    if args.dry_run:
        print("dry-run: no retrieval performed.")
        return

    def run_serial():
        pb = _build_v2_pack_builder(grok_fallback=True)   # governor=None → pre-P120
        t0 = time.monotonic()
        packs = retrieval_phase.build_packs_phase(claims, pb)
        return packs, time.monotonic() - t0

    def run_pool():
        gov = PoolGovernor(pool_start=1, pool_max=args.pool_max,
                           r1_cli_cap=args.r1_cli_cap, adaptive=True)
        pb = _build_v2_pack_builder(grok_fallback=True, governor=gov)
        t0 = time.monotonic()
        packs = retrieval_phase.build_packs_phase(claims, pb, governor=gov)
        return packs, gov, time.monotonic() - t0

    # Order controls the confound: last run was serial-first (pooled ran second,
    # after R1's Max window was already drawn down). Default here is pooled-first,
    # so if the recall gap flips it points to order/drawdown, not the pool.
    if args.order == "pooled-first":
        print("running POOLED first, then serial...")
        packs_pool, gov, pool_s = run_pool()
        packs_serial, serial_s = run_serial()
    else:
        print("running SERIAL first, then pooled...")
        packs_serial, serial_s = run_serial()
        packs_pool, gov, pool_s = run_pool()

    n = len(claims)
    det_serial = _detail(packs_serial, gold_by_sid)
    det_pool = _detail(packs_pool, gold_by_sid)
    diffs = [
        {"sid": sid, "serial_recall": det_serial[sid]["recall"],
         "pool_recall": det_pool[sid]["recall"],
         "serial_items": det_serial[sid]["items"], "pool_items": det_pool[sid]["items"]}
        for sid in gold_by_sid
        if det_serial[sid]["recall"] != det_pool[sid]["recall"]
    ]
    summary = {
        "claims": n, "order": args.order,
        "serial": {"wall_s": round(serial_s, 1), "per_claim_s": round(serial_s / n, 1),
                   "mean_recall": _mean([d["recall"] for d in det_serial.values()]),
                   "mean_pack_items": _mean([d["items"] for d in det_serial.values()])},
        "pooled": {"wall_s": round(pool_s, 1), "per_claim_s": round(pool_s / n, 1),
                   "mean_recall": _mean([d["recall"] for d in det_pool.values()]),
                   "mean_pack_items": _mean([d["items"] for d in det_pool.values()]),
                   "telemetry": gov.telemetry()},
        "recall_diffs": diffs,
        "speedup_x": round(serial_s / pool_s, 2) if pool_s else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n==== P120 POOL BENCH ====")
    print(json.dumps(summary, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
