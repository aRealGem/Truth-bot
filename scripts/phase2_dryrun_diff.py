#!/usr/bin/env python
"""Phase 2 remediation dry-run: what would the NEW pipeline rules remove from
the five PUBLISHED evidence packs, and does that loss touch any verdict's
citations? ($0 — no model calls, no retrieval; pipeline modules consumed
read-only.)

Produces the DC-5 decision worksheet Precious/jackie use to approve the
Phase-3 re-adjudication scope:

  * ``metrics/remediation_v2/dc5_worksheet.json``  — full machine-readable
  * ``metrics/remediation_v2/dc5_worksheet.md``    — compact human table

Per published artifact (``metrics/pca_runs/<uuid>.json``), per sid, the stored
pack items (E-number = position + 1 in ``evidence[sid]`` order) are re-judged
under the deterministic rules that landed on ``remediation-v2``:

  removals
    * ``fc-excluded``       — :func:`truthbot.verify.factcheck_exclusion.
      factcheck_exclusion_reason` non-empty (DC-1 fact-check exclusion v2)
    * ``era-violation``     — dated item outside the coded evidence window
      (``expected_claim_window``) OR past the speaker's fair-game window
      (utterance + 7d). BOTH checks are applied, mirroring the new
      ``build_evidence_pack`` filter chain (``_within_window`` +
      ``_within_fair_game``). Dates come from ``era_lint.item_date``
      (published_at, else the ``[YYYY-MM-DD]`` snippet stamp).
    * ``mutable-endpoint``  — :func:`truthbot.verify.mutable_endpoints.
      is_mutable_latest` (live latest-release pointers, era-unsafe)
    * ``s5-capped``         — after re-tiering via ``classify_tier``,
      POLITICAL survivors beyond the first 3 in candidate order

  informational (not removals)
    * ``post-speech``       — utterance < date <= utterance + 7d: the item
      stays but becomes context-only (non-verdict-bearing)
    * ``tier-flip``         — ``classify_tier(url).value`` differs from the
      stored ``source_tier`` (registry drift; affects crediting below)

Candidate set: the stored pack. Where a RICHER pre-cap pool was journaled
(clinton_1998 / gwbush_2006 ``*_packs.jsonl``; obama_2014 rescue leg), the
pool is the candidate set instead — pool items are matched back to pack
E-numbers by URL, and pool-only extras (no E-number) can restore credits a
removal took away. trump/biden are pack-only by design: losses are what this
dry-run measures; gains would need fresh retrieval.

Citation impact per sid (E-refs in ``rows[].citations``; ``\\bE\\d+\\b`` over
``rows[].reasoning``): ``none`` | ``cited-item-lost`` |
``rationale-mentions-lost-item`` | ``context-only-cited``.

Quota per sid — the simple credit rule: credits = surviving items that are
bearing (``supports_claim`` is True/False), not post-speech, and NEW tier in
T1..T3 (Government / Wire / Established; POLITICAL never credits).
``would_gate`` = a currently-decided verdict (TRUE/FALSE/MISLEADING) left with
credits < 2 — the new pipeline would FORCE Unverifiable without fresh
retrieval; the strongest re-adjudication candidates.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))          # hydramind lives at the repo root
sys.path.insert(0, str(REPO / "src"))

from truthbot.models import SourceTier                                # noqa: E402
from truthbot.verdict.era_lint import fair_game_end, item_date        # noqa: E402
from truthbot.verify.context.terms import expected_claim_window       # noqa: E402
from truthbot.verify.factcheck_exclusion import factcheck_exclusion_reason  # noqa: E402
from truthbot.verify.mutable_endpoints import is_mutable_latest       # noqa: E402
from truthbot.verify.source_tiers import classify_tier                # noqa: E402

# ── the five published runs ──────────────────────────────────────────────────

PUBLISHED_RUNS: dict[str, tuple[str, date]] = {
    "trump_2026":   ("23939712-59ea-449d-93f7-a0a0b449efd8", date(2026, 2, 24)),
    "biden_2022":   ("7208bbbb-c802-4155-932f-d0cc66803b24", date(2022, 3, 1)),
    "obama_2014":   ("28965cdf-046e-4c87-a5d1-d21b6529c625", date(2014, 1, 28)),
    "clinton_1998": ("7c59e9e0-0062-487d-84e3-4af15ab94aab", date(1998, 1, 27)),
    "gwbush_2006":  ("92f39851-8870-4609-97f6-458798d5dbb8", date(2006, 1, 31)),
}

#: journals carrying pre-cap "pool" fields keyed by sid (richer candidate set)
POOL_JOURNALS: dict[str, str] = {
    "clinton_1998": "metrics/journals/clinton_1998_packs.jsonl",
    "gwbush_2006":  "metrics/journals/gwbush_2006_packs.jsonl",
    "obama_2014":   "metrics/journals/obama_2014_s5rescue_packs.jsonl",
}

S5_CAP = 3                       # POLITICAL survivors kept, per pack
CREDIT_TIERS = {SourceTier.GOVERNMENT, SourceTier.WIRE, SourceTier.ESTABLISHED}
CREDIT_TIER_NAMES = {t.value for t in CREDIT_TIERS}
DECIDED = {"TRUE", "FALSE", "MISLEADING"}
_EREF_RX = re.compile(r"\bE\d+\b")

REMOVAL_KEYS = ("fc-excluded", "era-violation", "mutable-endpoint", "s5-capped")


# ── per-item evaluation ──────────────────────────────────────────────────────

@dataclass
class ItemEval:
    e: Optional[int]             # E-number in the STORED pack, None = pool-only
    url: str
    stored_tier: str
    new_tier: SourceTier
    from_pool: bool
    bearing: bool                # supports_claim is True/False
    disposition: str = "kept"    # kept | fc-excluded | era-violation | ...
    reason: str = ""
    post_speech: bool = False
    tier_flip: bool = False

    @property
    def removed(self) -> bool:
        return self.disposition != "kept"


def _candidate_items(pack: list[dict], pool: Optional[list[dict]]
                     ) -> list[tuple[Optional[int], dict, bool]]:
    """(e_number, item, from_pool) in candidate order. With a pool, pool order
    rules and each pool item claims the first unclaimed pack position with the
    same URL (item ids are regenerated between legs; URLs are stable). Pack
    items the pool somehow lacks are appended so a loss can never be hidden."""
    if not pool:
        return [(i + 1, it, False) for i, it in enumerate(pack)]
    unclaimed: dict[str, list[int]] = {}
    for i, it in enumerate(pack):
        unclaimed.setdefault(it.get("source_url") or "", []).append(i + 1)
    out: list[tuple[Optional[int], dict, bool]] = []
    for it in pool:
        slots = unclaimed.get(it.get("source_url") or "")
        e = slots.pop(0) if slots else None
        out.append((e, it, e is None))
    for url, slots in unclaimed.items():
        for e in slots:
            out.append((e, pack[e - 1], False))
    return out


def evaluate_claim_items(pack: list[dict], pool: Optional[list[dict]],
                         utterance: date,
                         window: Optional[tuple[date, date]]) -> list[ItemEval]:
    """Apply the new deterministic rules to one claim's candidate set."""
    fg_end = fair_game_end(utterance)
    evals: list[ItemEval] = []
    for e, it, from_pool in _candidate_items(pack, pool):
        url = it.get("source_url") or ""
        stored = it.get("source_tier") or ""
        tier = classify_tier(url)
        ev = ItemEval(
            e=e, url=url, stored_tier=stored, new_tier=tier,
            from_pool=from_pool,
            bearing=it.get("supports_claim") in (True, False),
            tier_flip=bool(stored) and tier.value != stored,
        )
        d = item_date(it.get("published_at"), it.get("snippet") or "")
        if d is not None and utterance < d <= fg_end:
            ev.post_speech = True
        fc = factcheck_exclusion_reason(url)
        if fc:
            ev.disposition, ev.reason = "fc-excluded", fc
        elif d is not None:
            problems = []
            if window is not None and not (window[0] <= d <= window[1]):
                problems.append(
                    f"outside coded window {window[0]}..{window[1]}")
            if d > fg_end:
                problems.append(f"past fair-game end {fg_end}")
            if problems:
                ev.disposition = "era-violation"
                ev.reason = f"dated {d}: " + "; ".join(problems)
        if ev.disposition == "kept" and is_mutable_latest(url):
            ev.disposition, ev.reason = "mutable-endpoint", "live latest-release pointer"
        evals.append(ev)
    # S5 cap: POLITICAL survivors beyond the first 3, in candidate order
    political_kept = 0
    for ev in evals:
        if ev.removed or ev.new_tier is not SourceTier.POLITICAL:
            continue
        political_kept += 1
        if political_kept > S5_CAP:
            ev.disposition = "s5-capped"
            ev.reason = f"POLITICAL survivor #{political_kept} (cap {S5_CAP})"
    return evals


# ── per-claim rollup ─────────────────────────────────────────────────────────

@dataclass
class ClaimResult:
    sid: str
    verdict: Optional[str]
    decided: bool
    pool_used: bool
    impact: str = "none"
    verdict_cited_lost: bool = False
    dispositions: dict[str, int] = field(default_factory=dict)
    lost: list[dict] = field(default_factory=list)
    credits_before_ish: int = 0
    credits_after: int = 0
    would_gate: bool = False

    @property
    def loses_items(self) -> bool:
        """True when the PUBLISHED pack loses an item. ``dispositions`` also
        counts removed pool-only extras (candidate-set picture); ``lost``
        holds only pack items, and pack losses are what the worksheet is
        for — a discarded pool extra was never published."""
        return bool(self.lost)


def analyze_claim(sid: str, pack: list[dict], pool: Optional[list[dict]],
                  row: Optional[dict], utterance: date,
                  window: Optional[tuple[date, date]]) -> ClaimResult:
    verdict = (row or {}).get("verdict")
    res = ClaimResult(sid=sid, verdict=verdict,
                      decided=verdict in DECIDED, pool_used=bool(pool))
    evals = evaluate_claim_items(pack, pool, utterance, window)

    citations = set((row or {}).get("citations") or [])
    rationale_refs = set(_EREF_RX.findall((row or {}).get("reasoning") or ""))

    counts = {k: 0 for k in REMOVAL_KEYS}
    counts.update(kept=0, post_speech=0, tier_flip=0)
    cited_lost = rationale_lost = cited_context_only = False
    for ev in evals:
        counts[ev.disposition] = counts.get(ev.disposition, 0) + 1
        if ev.post_speech:
            counts["post_speech"] += 1
        if ev.tier_flip:
            counts["tier_flip"] += 1
        eref = f"E{ev.e}" if ev.e is not None else None
        if ev.removed:
            was_cited = eref in citations
            in_rat = eref in rationale_refs
            cited_lost |= was_cited
            rationale_lost |= in_rat
            if not ev.from_pool:          # pool-only extras were never in the pack
                res.lost.append({"e": ev.e, "url": ev.url,
                                 "disposition": ev.disposition,
                                 "reason": ev.reason,
                                 "was_cited": was_cited,
                                 "in_rationale": in_rat})
        elif ev.post_speech and eref in citations:
            cited_context_only = True
    res.dispositions = counts
    res.verdict_cited_lost = cited_lost
    res.impact = ("cited-item-lost" if cited_lost
                  else "rationale-mentions-lost-item" if rationale_lost
                  else "context-only-cited" if cited_context_only
                  else "none")

    # quota: before-ish over the stored pack + stored tiers; after over the
    # surviving candidate set with new tiers and post-speech de-crediting
    res.credits_before_ish = sum(
        1 for it in pack
        if it.get("supports_claim") in (True, False)
        and (it.get("source_tier") or "") in CREDIT_TIER_NAMES)
    res.credits_after = sum(
        1 for ev in evals
        if not ev.removed and ev.bearing and not ev.post_speech
        and ev.new_tier in CREDIT_TIERS)
    res.would_gate = res.decided and res.credits_after < 2
    return res


# ── per-artifact / site-wide ─────────────────────────────────────────────────

def load_pools(path: Path) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {}
    if not path.exists():
        return pools
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("pool"):
                pools[rec["sid"]] = rec["pool"]
    return pools


def analyze_artifact(artifact: dict, utterance: date,
                     pools: Optional[dict[str, list[dict]]] = None
                     ) -> list[ClaimResult]:
    window = expected_claim_window(utterance)
    rows = {r.get("sid") or r.get("item_id"): r
            for r in artifact.get("rows") or []}
    evidence = artifact.get("evidence") or {}
    results = []
    for sid in sorted(set(evidence) | set(rows)):
        results.append(analyze_claim(
            sid, evidence.get(sid) or [], (pools or {}).get(sid),
            rows.get(sid), utterance, window))
    return results


def _totals(results: Iterable[ClaimResult]) -> dict:
    rs = list(results)
    return {
        "claims": len(rs),
        "decided": sum(r.decided for r in rs),
        "claims_losing_items": sum(r.loses_items for r in rs),
        "cited_losses": sum(r.impact == "cited-item-lost" for r in rs),
        "rationale_losses": sum(
            r.impact == "rationale-mentions-lost-item" for r in rs),
        "context_only_cited": sum(
            r.impact == "context-only-cited" for r in rs),
        "would_gate": sum(r.would_gate for r in rs),
        # pack items only — removed pool-only extras were never published
        "items_removed": sum(len(r.lost) for r in rs),
        "removed_by_rule": {
            k: sum(1 for r in rs for l in r.lost if l["disposition"] == k)
            for k in REMOVAL_KEYS},
        "post_speech_items": sum(
            r.dispositions.get("post_speech", 0) for r in rs),
        "tier_flips": sum(r.dispositions.get("tier_flip", 0) for r in rs),
    }


def _claim_json(r: ClaimResult) -> dict:
    return {
        "sid": r.sid, "verdict": r.verdict, "impact": r.impact,
        "verdict_cited_lost": r.verdict_cited_lost,
        "pool_used": r.pool_used,
        "dispositions": r.dispositions, "lost": r.lost,
        "quota": {"credits_before_ish": r.credits_before_ish,
                  "credits_after": r.credits_after,
                  "would_gate": r.would_gate},
    }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                  # pragma: no cover
        return "unknown"


def build_worksheet() -> dict:
    per_report = []
    all_results: dict[str, list[ClaimResult]] = {}
    for speech_id, (run_uuid, utterance) in PUBLISHED_RUNS.items():
        artifact = json.loads(
            (REPO / "metrics" / "pca_runs" / f"{run_uuid}.json").read_text())
        pools = (load_pools(REPO / POOL_JOURNALS[speech_id])
                 if speech_id in POOL_JOURNALS else {})
        results = analyze_artifact(artifact, utterance, pools)
        all_results[speech_id] = results
        affected = [r for r in results
                    if r.loses_items or r.would_gate or r.impact != "none"]
        per_report.append({
            "speech_id": speech_id,
            "run_id": run_uuid,
            "utterance": utterance.isoformat(),
            "coded_window": [d.isoformat()
                             for d in expected_claim_window(utterance)],
            "pool_sids": len(pools),
            "totals": _totals(results),
            "claims": [_claim_json(r) for r in affected],
        })

    flat = [r for rs in all_results.values() for r in rs]
    scope_a = sorted(r.sid for r in flat
                     if r.impact == "cited-item-lost" or r.would_gate)
    return {
        "worksheet": "DC-5 Phase-2 regeneration dry-run",
        "generated_from": _git_sha(),
        "rules_note": (
            "removals: fc-excluded (factcheck_exclusion_reason), era-violation "
            "(BOTH coded expected_claim_window AND fair-game utterance+7d), "
            "mutable-endpoint (is_mutable_latest), s5-capped (POLITICAL "
            f"survivors beyond first {S5_CAP} in candidate order). "
            "post-speech (utterance < d <= +7d) items become context-only, "
            "not removed. Credit rule: >=2 bearing, non-post-speech survivors "
            "with NEW tier in {Government, Wire, Established}; POLITICAL "
            "never credits. Pool candidate sets used where journaled "
            "(clinton_1998, gwbush_2006, obama_2014 rescue sids); trump/biden "
            "are pack-only — losses measurable, gains need retrieval."),
        "per_report": per_report,
        "totals": _totals(flat),
        "scope_option_a_minimal": {
            "definition": "every sid with cited-item-lost OR would-gate",
            "count": len(scope_a),
            "sids": scope_a,
        },
    }


# ── markdown rendering ───────────────────────────────────────────────────────

def render_md(ws: dict) -> str:
    L = ["# DC-5 worksheet — Phase-2 $0 regeneration dry-run", "",
         f"Generated from `{ws['generated_from'][:12]}` on branch "
         "remediation-v2. No model calls; published artifacts + journaled "
         "pools re-judged under the new deterministic rules.", "",
         f"Rules: {ws['rules_note']}", "",
         "## Per report", "",
         "| report | claims | decided | losing items | cited losses | "
         "rationale-only losses | context-only cited | would-gate | "
         "items removed | post-speech items | tier flips |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rep in ws["per_report"]:
        t = rep["totals"]
        L.append(
            f"| {rep['speech_id']} | {t['claims']} | {t['decided']} | "
            f"{t['claims_losing_items']} | {t['cited_losses']} | "
            f"{t['rationale_losses']} | {t['context_only_cited']} | "
            f"{t['would_gate']} | {t['items_removed']} | "
            f"{t['post_speech_items']} | {t['tier_flips']} |")
    t = ws["totals"]
    L += ["",
          f"**Site-wide:** {t['claims']} claims ({t['decided']} decided) — "
          f"{t['claims_losing_items']} lose >=1 pack item, "
          f"{t['cited_losses']} lose a CITED item, {t['would_gate']} decided "
          f"claims would now gate to Unverifiable (credits < 2). "
          f"{t['items_removed']} items removed "
          f"({', '.join(f'{k}: {v}' for k, v in t['removed_by_rule'].items())}); "
          f"{t['post_speech_items']} items de-credited as post-speech "
          f"context-only; {t['tier_flips']} tier flips.", ""]
    sa = ws["scope_option_a_minimal"]
    L += ["## Scope option (a) — minimal", "",
          f"{sa['definition']}: **{sa['count']} sids**", ""]
    by_speech: dict[str, list[str]] = {}
    for sid in sa["sids"]:
        by_speech.setdefault(sid.split(":")[0], []).append(sid)
    for speech, sids in by_speech.items():
        L.append(f"- **{speech}** ({len(sids)}): "
                 + ", ".join(s.split(":")[1] for s in sids))
    L.append("")
    return "\n".join(L)


def main() -> None:
    out_dir = REPO / "metrics" / "remediation_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = build_worksheet()
    (out_dir / "dc5_worksheet.json").write_text(
        json.dumps(ws, indent=1) + "\n")
    (out_dir / "dc5_worksheet.md").write_text(render_md(ws))
    print(render_md(ws))
    print(f"wrote {out_dir / 'dc5_worksheet.json'}")
    print(f"wrote {out_dir / 'dc5_worksheet.md'}")


if __name__ == "__main__":
    main()
