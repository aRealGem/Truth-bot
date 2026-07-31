#!/usr/bin/env python3
"""P129 — gold-15 before/after verdict-shift validation for PR-A (S5 tier).

Runs the 15-claim verdict-gold fixture through the FULL production config
(roster=prod, shared_pack_v2, CRM-114 two-stage) and scores the panel verdicts
against the fixture's gold. Run it once on the pre-PR-A checkout (--label before)
and once on main (--label after), SAME DAY, then diff — so the only thing that
changed is the S5 political-communications tiering, not web drift.

This file is intentionally UNTRACKED so it survives `git checkout` between the
two legs. Kept model-agnostic: it just calls adjudicator.adjudicate, so it runs
unchanged on both checkouts.

Env prelude (repo root):
  set -a; . ./.env; . ~/.env; set +a
  TRUTHBOT_R2_MODEL=gpt-5-mini TRUTHBOT_R3_MODEL=grok-4.3 PYTHONPATH=. \
    .venv/bin/python scripts/gold15_p129.py --label after --go [--proxy-cap 3.0]
Without --go it lists the claims + contamination check and exits ($0).
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

FIXTURE = REPO / "eval/benchmarks/claim-set/sotu_gold_fixture_2026-07-10.json"
# fixture speech key -> pca sid speech prefix
SPEECH_SID = {"biden2022": "biden_2022", "trump2026": "trump_2026"}
R2_RATE = (0.25, 2.00)     # gpt-5-mini, USD/Mtok (in, out)
R3_RATE = (1.25, 2.50)     # grok-4.3


def _norm(v: str | None) -> str:
    return (v or "").strip().upper()


def load_gold() -> list[dict]:
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    out = []
    for c in d["claims"]:
        out.append({
            "sid": c["claim_id"],
            "text": c["paraphrase"],
            "context": f"reference period: {c.get('reference_period','')}",
            "gold": _norm(c.get("verdict_provisional")),
            # gold rationale/evidence are the CONTAMINATION set — never fed in.
            "_gold_rationale": str(c.get("rationale") or ""),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, choices=["before", "after"])
    ap.add_argument("--go", action="store_true", help="actually spend (else dry list + exit)")
    ap.add_argument("--proxy-cap", type=float, default=3.0)
    args = ap.parse_args()

    claims = load_gold()
    print(f"P129 gold-15 leg: label={args.label}  n={len(claims)}")
    for c in claims:
        print(f"  [{c['sid']}] gold={c['gold']:<12} {c['text'][:80]}")

    # contamination guard: the claim text we feed must not contain gold rationale.
    from truthbot.verify.retrievers import assert_no_contamination
    frags = [c["_gold_rationale"] for c in claims if c["_gold_rationale"]]
    for c in claims:
        assert_no_contamination(c["text"] + " " + c["context"], frags)
    print("contamination guard: OK (no gold rationale in any claim prompt)")

    out_path = REPO / f"metrics/gold15_p129_{args.label}.json"
    journal = REPO / f"metrics/journals/gold15_p129_{args.label}.jsonl"

    if not args.go:
        print("\n(dry list only — rerun with --go to spend)")
        return

    from truthbot.verdict import adjudicator, proxy_lane, publish_pipeline
    from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
    from truthbot.verify import retrievers as R
    from hydramind.rosters import get_roster

    if not proxy_lane.key_present():
        sys.exit(proxy_lane.BLOCKED_MSG)

    usage = {"R2": [], "R3": []}

    class MeteredR2(R.OpenAIBrowsingRetriever):
        def _post(self, model, prompt):
            doc = super()._post(model, prompt)
            usage["R2"].append(doc.get("usage") or {})
            return doc

    class MeteredR3(R.GrokSearchRetriever):
        def _post(self, model, prompt, tool):
            doc = super()._post(model, prompt, tool)
            usage["R3"].append(doc.get("usage") or {})
            return doc

    trio = (R.ClaudeWorkerRetriever(), MeteredR2(), MeteredR3(model="grok-4.3"))

    def pack_builder(sid, text, context):
        return build_evidence_pack_v2(sid, text, trio, context=context)

    hm = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    roster_note = {"name": "prod", "seats": dict(get_roster("prod").seats)}
    start = proxy_lane.proxy_key_spend()
    print(f"\nproxy spend at start ${start:.4f}  (cap ${args.proxy_cap:.2f})")

    chunk = [{"sid": c["sid"], "text": c["text"], "context": c["context"]} for c in claims]
    rows, manifest, notes = adjudicator.adjudicate(
        hm, chunk, roster="prod", pack_builder=pack_builder, two_stage=True)
    proxy_delta = proxy_lane.proxy_key_spend() - start

    gold = {c["sid"]: c["gold"] for c in claims}
    preds = {r["sid"]: _norm(r.get("verdict") or r.get("status")) for r in rows}

    def _tok(entries, rate):
        ti = sum(int(e.get("input_tokens") or e.get("prompt_tokens") or 0) for e in entries)
        to = sum(int(e.get("output_tokens") or e.get("completion_tokens") or 0) for e in entries)
        return (ti * rate[0] + to * rate[1]) / 1e6
    off_proxy = _tok(usage["R2"], R2_RATE) + _tok(usage["R3"], R3_RATE)

    committed = {"TRUE", "FALSE", "MISLEADING"}
    decided = [(s, preds[s], gold[s]) for s in gold if preds.get(s) in committed]
    hits = sum(1 for _, p, g in decided if p == g)
    result = {
        "label": args.label, "n": len(claims),
        "proxy_usd": round(proxy_delta, 4), "off_proxy_usd_est": round(off_proxy, 4),
        "decided": len(decided), "hits": hits,
        "decided_acc": round(hits / len(decided), 3) if decided else None,
        "coverage": round(len(decided) / len(claims), 3),
        "gold": gold, "preds": preds,
        "gate_forced_uv": notes.get("gate_forced_unverifiable", []),
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    publish_pipeline.append_chunk_journal(journal, 1, rows, notes.get("packs") or {},
                                          proxy_delta, roster=roster_note)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[{args.label}] proxy ${proxy_delta:.4f} + off-proxy est ${off_proxy:.4f} "
          f"= ${proxy_delta + off_proxy:.4f}")
    print(f"decided {len(decided)}/{len(claims)}  acc {result['decided_acc']}  "
          f"coverage {result['coverage']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
