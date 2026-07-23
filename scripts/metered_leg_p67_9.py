#!/usr/bin/env python3
"""P67.9 step 2 — the 10-claim METERED LEG for the Phase 3 rerun.

Runs 10 real claims (6 trump_2026 + 4 biden_2022, stride-sampled from the
persisted run artifacts) through the FULL production configuration:

  roster=prod (opus-worker L-W proposer / grok-4.3 critic / gpt-5.5 arbiter),
  evidence shared_pack_v2 (R1/R2/R3 trio -> consolidator + T2.4 gate),
  CRM-114 stage 2, chunked + journaled + proxy-budget-capped.

Purpose: a HARD cost projection for the 289-claim rerun, from ledgers not log
lines. Meters three channels separately:
  * proxy DB delta (`/key/info` before/after each chunk) — grok-4.3 critic +
    gpt-5.5 arbiter + sonnet CRM-114 (the authoritative on-proxy ledger);
  * off-proxy R2 (OpenAI /v1/responses) + R3 (xAI /v1/responses) token usage
    captured from the response envelopes, priced at list;
  * R1 + opus-worker proposer — subscription lanes, $0 marginal.

Chunks journal into metrics/journals/<speech>_p3rerun.jsonl, so the full
rerun (--resume against the same journal) banks this spend instead of
repeating it.

Usage (repo root):
  set -a; . ./.env; . ~/.env; set +a
  TRUTHBOT_R3_MODEL=grok-4.3 PYTHONPATH=. .venv/bin/python \
      scripts/metered_leg_p67_9.py --go [--proxy-cap 3.0]
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

ARTIFACTS = {
    "trump_2026": REPO / "metrics/pca_runs/5c321299-1c27-43b4-8c85-5d7b0b24edda.json",
    "biden_2022": REPO / "metrics/pca_runs/807edf03-484d-42ed-a9cb-b12b5c3d3f52.json",
}
SAMPLE = {"trump_2026": 6, "biden_2022": 4}
JOURNALS = {s: REPO / f"metrics/journals/{s}_p3rerun.jsonl" for s in SAMPLE}
OUT = REPO / "metrics/metered_leg_p67_9.json"

# List prices, USD per Mtok (litellm price map 2026-07-23). Off-proxy lanes are
# estimated from captured usage; the provider LEDGER remains the final word.
R2_RATE = (5.00, 30.00)     # gpt-5.5
R3_RATE = (1.25, 2.50)      # grok-4.3
RERUN_CLAIMS = 289


def _sample(claims: list[dict], n: int) -> list[dict]:
    stride = max(1, len(claims) // n)
    return claims[::stride][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--go", action="store_true", help="actually spend (else list claims and exit)")
    ap.add_argument("--proxy-cap", type=float, default=3.0,
                    help="max on-proxy USD for this leg (preflight per chunk)")
    args = ap.parse_args()

    from truthbot.verdict import adjudicator, proxy_lane, publish_pipeline
    from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
    from truthbot.verify import retrievers as R
    from hydramind.rosters import get_roster

    if not proxy_lane.key_present():
        sys.exit(proxy_lane.BLOCKED_MSG)

    chunks: list[tuple[str, list[dict]]] = []
    for speech, n in SAMPLE.items():
        art = json.loads(ARTIFACTS[speech].read_text(encoding="utf-8"))
        chunk = [{"sid": c["sid"], "text": c["text"], "context": c.get("context", "")}
                 for c in _sample(art["claims"], n)]
        chunks.append((speech, chunk))
        for c in chunk:
            print(f"  [{c['sid']}] {c['text'][:90]}")
    if not args.go:
        print("\n(dry list only — rerun with --go to spend)")
        return

    # ── metered trio ─────────────────────────────────────────────────────────
    usage: dict[str, list] = {"R2": [], "R3": []}

    class MeteredR2(R.OpenAIBrowsingRetriever):
        def _post(self, model, prompt):
            doc = super()._post(model, prompt)
            usage["R2"].append({"model": model, "usage": doc.get("usage") or {}})
            return doc

    class MeteredR3(R.GrokSearchRetriever):
        def _post(self, model, prompt, tool):
            doc = super()._post(model, prompt, tool)
            usage["R3"].append({"model": model, "usage": doc.get("usage") or {}})
            return doc

    trio = (R.ClaudeWorkerRetriever(), MeteredR2(), MeteredR3(model="grok-4.3"))

    def pack_builder(sid: str, text: str, context: str):
        return build_evidence_pack_v2(sid, text, trio, context=context)

    hm = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    roster_note = {"name": "prod", "seats": dict(get_roster("prod").seats)}

    start_spend = proxy_lane.proxy_key_spend()
    print(f"\nproxy key spend at start: ${start_spend:.4f} "
          f"(leg cap ${args.proxy_cap:.2f} on-proxy)")

    result = {"config": {"roster": roster_note, "evidence_mode": "shared_pack_v2",
                         "two_stage": True, "proxy_cap": args.proxy_cap},
              "chunks": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    total_claims = 0
    for idx, (speech, chunk) in enumerate(chunks, 1):
        spent = proxy_lane.proxy_key_spend() - start_spend
        if spent >= args.proxy_cap:
            print(f"BUDGET HALT before chunk {idx}: on-proxy ${spent:.2f} "
                  f">= cap ${args.proxy_cap:.2f}")
            break
        r2_before, r3_before = len(usage["R2"]), len(usage["R3"])
        t0, s0 = time.time(), proxy_lane.proxy_key_spend()
        rows, manifest, notes = adjudicator.adjudicate(
            hm, chunk, roster="prod", pack_builder=pack_builder, two_stage=True)
        s1, t1 = proxy_lane.proxy_key_spend(), time.time()

        proxy_delta = s1 - s0
        self_reported = float(getattr(manifest, "total_cost_usd", 0.0) or 0.0)
        publish_pipeline.append_chunk_journal(
            JOURNALS[speech], 1, rows, notes.get("packs") or {},
            proxy_delta, roster=roster_note)
        verdicts = {r["sid"]: (r.get("verdict") or r.get("status")) for r in rows}
        rec = {"speech": speech, "n_claims": len(chunk),
               "proxy_delta_usd": round(proxy_delta, 4),
               "manifest_self_reported_usd": round(self_reported, 4),
               "wall_s": round(t1 - t0, 1),
               "gate_forced": notes.get("gate_forced_unverifiable", []),
               "verdicts": verdicts,
               "r2_calls": len(usage["R2"]) - r2_before,
               "r3_calls": len(usage["R3"]) - r3_before}
        result["chunks"].append(rec)
        total_claims += len(chunk)
        print(f"chunk {idx} ({speech}, {len(chunk)} claims): proxy ${proxy_delta:.4f} "
              f"(self-reported ${self_reported:.4f}), {t1 - t0:.0f}s\n  {verdicts}")

    # ── cost roll-up + projection ────────────────────────────────────────────
    def _tok_cost(entries, rates):
        tin = sum(int((e["usage"].get("input_tokens")
                       or e["usage"].get("prompt_tokens") or 0)) for e in entries)
        tout = sum(int((e["usage"].get("output_tokens")
                        or e["usage"].get("completion_tokens") or 0)) for e in entries)
        return tin, tout, (tin * rates[0] + tout * rates[1]) / 1e6

    r2_in, r2_out, r2_usd = _tok_cost(usage["R2"], R2_RATE)
    r3_in, r3_out, r3_usd = _tok_cost(usage["R3"], R3_RATE)
    proxy_total = sum(c["proxy_delta_usd"] for c in result["chunks"])
    off_proxy_total = r2_usd + r3_usd
    per_claim = (proxy_total + off_proxy_total) / total_claims if total_claims else 0.0
    projection = per_claim * RERUN_CLAIMS

    result["totals"] = {
        "claims": total_claims,
        "proxy_usd": round(proxy_total, 4),
        "r2_offproxy": {"in": r2_in, "out": r2_out, "usd_est": round(r2_usd, 4)},
        "r3_offproxy": {"in": r3_in, "out": r3_out, "usd_est": round(r3_usd, 4)},
        "subscription_lanes_usd": 0.0,
        "per_claim_usd": round(per_claim, 4),
        "projection_289_claims_usd": round(projection, 2),
        "note": ("off-proxy figures are token-usage estimates at list price; "
                 "reconcile against the OpenAI/xAI ledgers before the go "
                 "decision. Layer A (haiku classify, both speeches) adds "
                 "~$0.7 on prior telemetry and is not part of this leg."),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nTOTALS: proxy ${proxy_total:.4f} + off-proxy est "
          f"${off_proxy_total:.4f} over {total_claims} claims "
          f"=> ${per_claim:.4f}/claim => ${projection:.2f} projected for "
          f"{RERUN_CLAIMS} (panel+retrieval; Layer A extra).")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
