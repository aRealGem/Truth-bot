#!/usr/bin/env python3
"""T2.2 rescue run — re-retrieve the Obama-2014 gate-forced claims under MAX_S5.

The $0 harness (reconsolidate_s5_cap.py) proved the stored packs cannot answer
the retrieval-saturation hypothesis (post-cap data, flips structurally zero).
This is the funded half: for every claim the T2.4 gate forced Unverifiable in
the P131 pilot, rebuild the pack with LIVE retrievers under the S5 saturation
cap (freed slots can now hold quota-crediting T1–3 items), panel-adjudicate
whatever passes the gate, and report per-claim outcomes. Nothing is published;
the output artifact is the report.

Budget discipline (jackie authorized ≤$2.00 total, 2026-08-01): claims run in
chunks; before each chunk the run halts if
  on-proxy delta (/key/info ledger) + off-proxy estimate (R2/R3 usage @ list)
  >= --total-cap (default 1.90, headroom under the $2 grant).
Journals bank every completed chunk (chunk journal + packs journal WITH the
new pre-cap pools), so a halt loses nothing.

Usage (repo root):
  set -a; . ./.env; . ~/.env; set +a
  PYTHONPATH=.:src .venv/bin/python scripts/rescue_gated_s5_p131.py [--go]
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

from datetime import date  # noqa: E402

from truthbot.verdict import speech_context  # noqa: E402

# Remediation v2 (1.3): the first rescue leg never registered the speech
# date, so the era gate silently no-opped and live 2026 BLS pages entered
# 2014 packs. The builders now fail closed without this registration.
speech_context.register_speech_date("obama_2014", date(2014, 1, 28))

ARTIFACT = REPO / "metrics/pca_runs/6fdfde0e-1393-4a7a-9144-abd0fc48b5a1.json"
PACKS_JOURNAL = REPO / "metrics/journals/obama_2014_packs.jsonl"
CHUNK_JOURNAL = REPO / "metrics/journals/obama_2014_s5rescue.jsonl"
RESCUE_PACKS_JOURNAL = REPO / "metrics/journals/obama_2014_s5rescue_packs.jsonl"
OUT = REPO / "metrics/s5_rescue_p131.json"

# List prices, USD per Mtok, PER MODEL (litellm price map 2026-07-23). The
# first leg of this run priced everything at gpt-5.5 rates because the R2
# model was left on its default — the 2026-08-01 ~$3 overrun. Estimation is
# now per recorded model, and the run must be launched with
# TRUTHBOT_R2_MODEL=gpt-5-mini (the pilot economy config); it refuses to
# start otherwise.
MODEL_RATES = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.5": (5.00, 30.00),
    "grok-4.3": (1.25, 2.50),
}
_DEFAULT_RATE = (5.00, 30.00)   # unknown model → price pessimistically
CHUNK_SIZE = 5


class BudgetHalt(RuntimeError):
    """Raised from inside pack_builder when the running estimate crosses the
    cap — the per-CLAIM circuit breaker the first leg lacked (its between-
    chunk check let one 5-claim gpt-5.5 chunk run to ~$5)."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--go", action="store_true",
                    help="actually spend (else list the gated claims and exit)")
    ap.add_argument("--total-cap", type=float, default=1.90,
                    help="halt threshold, on-proxy + off-proxy-est USD")
    args = ap.parse_args()

    from hydramind.rosters import get_roster
    from truthbot.verdict import adjudicator, proxy_lane, publish_pipeline
    from truthbot.verdict.consolidator import GATE_INSUFFICIENT
    from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
    from truthbot.verify import retrievers as R

    if not proxy_lane.key_present():
        sys.exit(proxy_lane.BLOCKED_MSG)
    import os
    if args.go and os.environ.get("TRUTHBOT_R2_MODEL") != "gpt-5-mini":
        sys.exit("REFUSING to spend: TRUTHBOT_R2_MODEL=gpt-5-mini is not set "
                 "(the economy config). The 2026-08-01 leg ran R2 on default "
                 "gpt-5.5 and overspent ~2.5x; this guard makes that "
                 "impossible to repeat by accident.")

    gated_sids = [json.loads(l)["sid"]
                  for l in PACKS_JOURNAL.read_text(encoding="utf-8").splitlines()
                  if l.strip() and json.loads(l).get("gate_code")]
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    by_sid = {c["sid"]: c for c in art["claims"]}
    old_rows = {r["sid"]: r for r in art["rows"]}
    claims = [{"sid": s, "text": by_sid[s]["text"],
               "context": by_sid[s].get("context", "")}
              for s in gated_sids if s in by_sid]
    print(f"{len(claims)} gate-forced claims from the pilot:")
    for c in claims:
        print(f"  [{c['sid']}] {c['text'][:88]}")
    if not args.go:
        print("\n(dry list only — rerun with --go to spend)")
        return

    # Resume: sids already banked in the rescue chunk journal are never re-run.
    done_rows, done_packs, banked_cost, _ = \
        publish_pipeline.load_chunk_journal(CHUNK_JOURNAL)
    done_sids = {r["sid"] for r in done_rows}
    todo = [c for c in claims if c["sid"] not in done_sids]
    if done_sids:
        print(f"resume: {len(done_sids)} sids banked, {len(todo)} to run")

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

    # Pilot economy config: R1+R2 primary, grok joins only the T2.4 rescue
    # round. These claims all gate-failed once, so the retry round (with R3)
    # is expected to fire for most.
    primary = (R.ClaudeWorkerRetriever(), MeteredR2())
    retry = primary + (MeteredR3(model="grok-4.3"),)

    def _offproxy_est() -> float:
        total = 0.0
        for entries in usage.values():
            for e in entries:
                rates = MODEL_RATES.get(str(e.get("model") or ""), _DEFAULT_RATE)
                u = e["usage"]
                tin = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
                tout = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
                total += (tin * rates[0] + tout * rates[1]) / 1e6
        return total

    def pack_builder(sid: str, text: str, context: str):
        # Per-claim circuit breaker: retrieval is where the money goes, so the
        # cap is enforced BEFORE each claim's retrieval, not just per chunk.
        spent = (proxy_lane.proxy_key_spend() - start_spend) + _offproxy_est()
        if spent >= args.total_cap:
            raise BudgetHalt(f"${spent:.2f} >= cap ${args.total_cap:.2f} "
                             f"(before retrieving {sid})")
        pack = build_evidence_pack_v2(sid, text, primary,
                                      retry_retrievers=retry, context=context)
        publish_pipeline.append_packs_journal(RESCUE_PACKS_JOURNAL, sid, pack)
        return pack

    hm = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    roster_note = {"name": "prod", "seats": dict(get_roster("prod").seats)}
    start_spend = proxy_lane.proxy_key_spend()
    print(f"proxy key spend at start: ${start_spend:.4f} "
          f"(total cap ${args.total_cap:.2f} incl. off-proxy est)")

    chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
    all_rows = list(done_rows)
    halted = ""
    for idx, chunk in enumerate(chunks, 1):
        proxy_delta = proxy_lane.proxy_key_spend() - start_spend
        total_so_far = proxy_delta + _offproxy_est() + banked_cost
        if total_so_far >= args.total_cap:
            halted = (f"BUDGET HALT before chunk {idx}: "
                      f"${total_so_far:.2f} >= cap ${args.total_cap:.2f}")
            print(halted)
            break
        t0, s0 = time.time(), proxy_lane.proxy_key_spend()
        try:
            rows, manifest, notes = adjudicator.adjudicate(
                hm, chunk, roster="prod", pack_builder=pack_builder, two_stage=True)
        except BudgetHalt as exc:
            halted = f"BUDGET HALT mid-chunk {idx}: {exc}"
            print(halted)
            break
        s1, t1 = proxy_lane.proxy_key_spend(), time.time()
        publish_pipeline.append_chunk_journal(
            CHUNK_JOURNAL, idx, rows, notes.get("packs") or {}, s1 - s0,
            roster=roster_note if idx == 1 else None)
        all_rows.extend(rows)
        print(f"chunk {idx}/{len(chunks)} ({len(chunk)} claims): "
              f"proxy ${s1 - s0:.4f}, off-proxy est ${_offproxy_est():.4f}, "
              f"{t1 - t0:.0f}s")

    # ── outcome report ───────────────────────────────────────────────────────
    tally = publish_pipeline.verdict_bucket_tally(all_rows)
    per_claim = []
    for row in all_rows:
        sid = row["sid"]
        new_gate = row.get("evidence_gate") or row.get("provenance_code") or ""
        outcome = ("still-gated" if new_gate == GATE_INSUFFICIENT
                   else (row.get("verdict") or
                         ("Models split" if row.get("split") else "No verdict")))
        per_claim.append({
            "sid": sid,
            "old": "gate-forced UNVERIFIABLE",
            "new": outcome,
            "flip_decided": outcome not in
                ("still-gated", "UNVERIFIABLE", "Models split", "No verdict"),
            "text": by_sid.get(sid, {}).get("text", "")[:120],
            "old_reasoning_was_forced": bool(
                old_rows.get(sid, {}).get("provenance_code")),
        })
    proxy_total = proxy_lane.proxy_key_spend() - start_spend
    off_total = _offproxy_est()
    n_flip = sum(1 for p in per_claim if p["flip_decided"])
    n_still = sum(1 for p in per_claim if p["new"] == "still-gated")
    n_panel_uv = sum(1 for p in per_claim if p["new"] == "UNVERIFIABLE")
    result = {
        "run": "s5-rescue (T2.2 funded half)",
        "config": {"roster": roster_note, "evidence": "shared_pack_v2 + MAX_S5",
                   "retry": "grok-fallback", "total_cap": args.total_cap},
        "claims_run": len(all_rows), "halted": halted,
        "outcomes": {"decided": n_flip, "panel_unverifiable": n_panel_uv,
                     "still_gated": n_still,
                     "tally": tally},
        "per_claim": per_claim,
        "spend": {"proxy_usd": round(proxy_total, 4),
                  "offproxy_est_usd": round(off_total, 4),
                  "total_est_usd": round(proxy_total + off_total, 4)},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nOUTCOMES over {len(all_rows)} claims: decided {n_flip} · "
          f"panel-UV {n_panel_uv} · still-gated {n_still}")
    print(f"SPEND: proxy ${proxy_total:.4f} + off-proxy est ${off_total:.4f} "
          f"= ${proxy_total + off_total:.4f} (cap ${args.total_cap:.2f})")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
