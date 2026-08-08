#!/usr/bin/env python3
"""B2 subset — which claims the scoring-prompt fix can actually help. $0.

NO model calls, no keys, no network. Pure arithmetic over the five rebuilt run
artifacts and the B1a re-score sidecars.

WHY
---
Haiku scored raw statistical series as "context" (stance None), so on exactly
the claims where the pack held the BEST possible evidence — the data table —
that evidence credited nothing. trump_2026:0054 is the worked case: the BLS
employed-persons series and FRED LNS12000000 both went True -> None in B1a,
with the snippet "shows the January 2026 peak used to evaluate". clinton_1998:
0101 is the same shape: the DOL budget appendix and two appropriations reports
with explicit figures, all three -> None.

``verify.relevance._SCORE_SYSTEM`` now tells the scorer that a primary series
containing the figure at issue takes a side. This module picks the SUBSET that
fix can move, so the re-score pays only for claims it can help:

    a claim qualifies when at least one of its Tier-1..3 items is CURRENTLY
    stance-None (after the B1a sidecar is overlaid) AND that item looks like a
    primary data / official-record source.

"Looks like" is a curated host list plus a govinfo/congress.gov package-type
check — deterministic, printable, and reviewable BEFORE any money moves. Items
D15 identifies as records of the speech itself are excluded: re-scoring a
transcript buys a stance we have already decided must never credit the quota.

Usage (repo root, always $0):
  PYTHONPATH=.:src .venv/bin/python scripts/b2_primary_series.py
  PYTHONPATH=.:src .venv/bin/python scripts/b2_primary_series.py --write-sids DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regate_from_rescore import load_rescore_sidecar  # noqa: E402
from rescore_stored_packs import (REBUILT_RUNS, artifact_path,  # noqa: E402
                                  load_artifact, sidecar_path)

OUT_DIR = REPO / "metrics" / "remediation_v2"
OUT_STEM = "b2_subset"

#: Tiers that can credit the decided-verdict quota. Restated from the
#: consolidator by IMPORT below — never by copy — so the two cannot drift.

#: Publishers whose pages ARE the measurement: statistical agencies, fiscal
#: scorekeepers, and the official-record hosts. Matched on host suffix, so
#: ``data.bls.gov`` and ``apps.bea.gov`` come along with their parents.
#: Grounded in the actual stance-null Tier-1..3 census, not guessed at.
PRIMARY_HOSTS: frozenset[str] = frozenset({
    # ── statistical agencies ────────────────────────────────────────────────
    "bls.gov", "census.gov", "bea.gov", "eia.gov", "nces.ed.gov",
    "bts.gov", "ers.usda.gov", "nass.usda.gov", "bjs.ojp.gov",
    # ── the Federal Reserve system's series archives ────────────────────────
    "stlouisfed.org", "federalreserve.gov",
    # ── fiscal scorekeepers and auditors ────────────────────────────────────
    "cbo.gov", "gao.gov", "jct.gov", "treasury.gov", "fiscaldata.treasury.gov",
    "usaspending.gov", "omb.gov",
    # ── program administrators publishing their own counts ──────────────────
    "cms.gov", "irs.gov", "ssa.gov", "dol.gov", "doleta.gov", "aspe.hhs.gov",
    "cdc.gov", "nih.gov", "sba.gov", "opm.gov", "energy.gov", "epa.gov",
    "hud.gov", "va.gov", "uscourts.gov", "supremecourt.gov",
    # ── the official record ─────────────────────────────────────────────────
    "federalregister.gov", "congress.gov", "govinfo.gov",
})

#: govinfo/congress.gov carry EVERYTHING, so the host alone is not enough. A
#: package id must name a primary-record collection: the Budget, committee
#: reports and prints, hearings, the Economic Report of the President, the
#: Statutes, the Federal Register, the U.S. Code, the Serial Set.
GOVINFO_PRIMARY_PACKAGES: tuple[str, ...] = (
    "BUDGET-", "ERP-", "CRPT-", "CPRT-", "CHRG-", "CRI-", "SERIALSET-",
    "STATUTE-", "USCODE-", "PLAW-", "BILLS-", "COMPS-", "FR-", "CFR-",
    "GAOREPORTS-", "GPO-CRECB-", "APP-",
)

#: Data-shaped URL cues — a series endpoint, a table, a spreadsheet.
_DATA_CUES: tuple[str, ...] = (
    "/timeseries", "/series/", "/data/", "/tables/", "/table/", ".xlsx",
    ".xls", ".csv", "/graph/", "/pub/", "/appendix",
)


def host_of(url: str) -> str:
    try:
        return (urlsplit(url or "").netloc or "").lower().split(":")[0]
    except ValueError:
        return ""


def host_matches(url: str) -> bool:
    """Is the URL on a primary-record publisher (host or any parent host)?"""
    h = host_of(url)
    if not h:
        return False
    return any(h == p or h.endswith("." + p) for p in PRIMARY_HOSTS)


def package_is_primary(url: str) -> bool:
    """For the catch-all record hosts, does the package id name a primary
    collection? True for anything NOT on those hosts (the host list already
    decided), so this composes as a pure narrowing rule."""
    h = host_of(url)
    if not (h.endswith("govinfo.gov") or h.endswith("congress.gov")):
        return True
    up = (url or "").upper()
    if any(pkg in up for pkg in GOVINFO_PRIMARY_PACKAGES):
        return True
    # congress.gov's own report/bill paths carry no GPO package id.
    low = (url or "").lower()
    return any(seg in low for seg in ("/committee-report/", "/congressional-report/",
                                      "/bill/", "/committee-print/", "/report/"))


def has_data_cue(url: str) -> bool:
    low = (url or "").lower()
    return any(cue in low for cue in _DATA_CUES)


def is_primary_record(url: str, snippet: str = "") -> bool:
    """Does this item look like a primary data series or official record?

    Host-anchored on purpose. A prose-only rule over snippets would sweep in
    every news story that quotes a BLS number, and the whole point is to target
    the sources that CARRY the figure rather than report it."""
    if not host_matches(url):
        return False
    return package_is_primary(url) or has_data_cue(url)


def _item_date(ev_dump: dict) -> Optional[date]:
    from truthbot.verdict import era_lint

    raw = ev_dump.get("published_at")
    dt = None
    if raw:
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            dt = None
    return era_lint.item_date(dt, ev_dump.get("snippet") or "")


def current_stance(sid: str, ev_dump: dict, scored: dict):
    """The stance as it stands TODAY: the B1a sidecar's value when it has one,
    else whatever the artifact recorded."""
    key = (ev_dump.get("source_url") or "").strip().rstrip("/").lower()
    for row in scored.get(sid) or []:
        if (row.get("source_url") or "").strip().rstrip("/").lower() == key:
            return row.get("supports_claim")
    return ev_dump.get("supports_claim")


def derive_subset(speech: str, artifact: dict,
                  sidecar: Optional[dict]) -> dict:
    """The B2 subset for one speech. Pure: no I/O, no mutation, no spend."""
    from truthbot.models import SourceTier
    from truthbot.verdict import speech_context, utterance_record as ur
    from truthbot.verdict.consolidator import _T13

    meta = artifact.get("meta") or {}
    utterance = date.fromisoformat(meta["date"]) if meta.get("date") else None
    if utterance is not None:
        speech_context.register_speech_date(speech, utterance)
    speech_date = speech_context.speech_date_for(f"{speech}:0")
    scored = (sidecar or {}).get("sids") or {}

    sids: list[str] = []
    triggers: list[dict] = []
    by_host: Counter = Counter()
    excluded_d15 = 0
    for sid, evs in (artifact.get("evidence") or {}).items():
        hits = []
        for ev in evs or []:
            url = ev.get("source_url") or ""
            try:
                tier = SourceTier(ev.get("source_tier"))
            except ValueError:
                continue
            if tier not in _T13:
                continue
            if current_stance(sid, ev, scored) is not None:
                continue
            # D15 first: a record of the speech itself can never credit the
            # quota, so buying it a stance buys nothing — and it is an absolute
            # disqualifier, checked BEFORE the primary-source rule so the
            # exclusion is counted honestly rather than hidden behind whichever
            # test happened to run first.
            if ur.utterance_record_rule(url, ev.get("snippet") or "",
                                        speech_date=speech_date,
                                        item_date=_item_date(ev)):
                excluded_d15 += 1
                continue
            if not is_primary_record(url, ev.get("snippet") or ""):
                continue
            hits.append({"sid": sid, "source_url": url, "tier": tier.value,
                         "snippet": (ev.get("snippet") or "")[:140]})
            by_host[host_of(url)] += 1
        if hits:
            sids.append(sid)
            triggers.extend(hits)

    items_in_subset = sum(len(artifact["evidence"].get(s) or []) for s in sids)
    return {"speech": speech, "source_run": artifact.get("run_id"),
            "sids": sorted(sids), "claims": len(sids),
            "trigger_items": len(triggers),
            "items_to_rescore": items_in_subset,
            "excluded_utterance_records": excluded_d15,
            "by_host": dict(by_host.most_common()),
            "triggers": triggers}


def build_report(speeches: list[str]) -> dict:
    rows = []
    for sp in speeches:
        art = load_artifact(artifact_path(sp))
        p = sidecar_path(sp)
        side = (load_rescore_sidecar(p, sp, art.get("run_id", ""))
                if p.exists() else None)
        rows.append(derive_subset(sp, art, side))
    return {"schema": "truthbot-b2-subset v1",
            "generated": datetime.now().astimezone().isoformat(),
            "speeches": speeches, "per_speech": rows,
            "total_claims": sum(r["claims"] for r in rows),
            "total_items_to_rescore": sum(r["items_to_rescore"] for r in rows),
            "total_trigger_items": sum(r["trigger_items"] for r in rows)}


def estimate(report: dict, model: str = "claude-haiku") -> dict:
    """Price the subset from the ACTUAL stored payloads — the same
    ``relevance.score_payload`` bytes the funded run will send."""
    from hydramind.models import RATE_TABLE_USD_PER_MTOK
    from rescore_stored_packs import (CHARS_PER_TOKEN, REPLY_CHARS_OVERHEAD,
                                      REPLY_CHARS_PER_ITEM, claim_texts)
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.relevance import _SCORE_SYSTEM, score_payload

    in_chars = out_chars = calls = items = 0
    for row in report["per_speech"]:
        art = load_artifact(artifact_path(row["speech"]))
        texts = claim_texts(art)
        by_sid = evidence_from_artifact_dict(art.get("evidence") or {})
        for sid in row["sids"]:
            evs = by_sid.get(sid) or []
            if not evs or not texts.get(sid):
                continue
            in_chars += len(_SCORE_SYSTEM) + len(score_payload(texts[sid], evs))
            out_chars += REPLY_CHARS_OVERHEAD + REPLY_CHARS_PER_ITEM * len(evs)
            calls += 1
            items += len(evs)
    tin, tout = in_chars / CHARS_PER_TOKEN, out_chars / CHARS_PER_TOKEN
    r_in, r_out = RATE_TABLE_USD_PER_MTOK.get(model, (0.0, 0.0))
    # The B2 reply carries one_line_why per item, which the B1a reply did not.
    # Assume it roughly triples the per-item reply — deliberately pessimistic,
    # because the number's job is to set a cap, not to be pretty.
    cost = (tin * r_in + tout * 3 * r_out) / 1_000_000.0
    return {"model": model, "calls": calls, "items": items,
            "tokens_in_est": round(tin), "tokens_out_est": round(tout * 3),
            "cost_usd_est": round(cost, 4)}


def render_text(report: dict, est: dict) -> str:
    L: list[str] = []
    A = L.append
    A("B2 subset — claims a primary series could decide but currently cannot ($0)")
    A("  rule: >=1 Tier-1..3 item currently stance-None AND on a primary")
    A("        data/record host (D15 utterance records excluded)")
    A("")
    A(f"  {'speech':<14}{'claims':>8}{'triggers':>10}{'items':>8}{'D15 excl':>10}")
    for r in report["per_speech"]:
        A(f"  {r['speech']:<14}{r['claims']:>8}{r['trigger_items']:>10}"
          f"{r['items_to_rescore']:>8}{r['excluded_utterance_records']:>10}")
    A(f"  {'TOTAL':<14}{report['total_claims']:>8}"
      f"{report['total_trigger_items']:>10}"
      f"{report['total_items_to_rescore']:>8}")
    A("")
    hosts: Counter = Counter()
    for r in report["per_speech"]:
        hosts.update(r["by_host"])
    A(f"  trigger hosts: {dict(hosts.most_common(15))}")
    A("")
    A(f"  ESTIMATE: {est['calls']} calls / {est['items']} items -> "
      f"${est['cost_usd_est']:.4f} ({est['model']}, on-proxy)")
    A("")
    for r in report["per_speech"]:
        if r["sids"]:
            A(f"  {r['speech']}: {' '.join(s.split(':')[1] for s in r['sids'])}")
    return "\n".join(L)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", choices=sorted(REBUILT_RUNS), default=None)
    ap.add_argument("--json", default=None, metavar="PATH")
    ap.add_argument("--write-sids", default=None, metavar="DIR",
                    help="write <speech>.json sid lists for "
                         "rescore_stored_packs.py --only-sids")
    args = ap.parse_args(argv)

    speeches = [args.speech] if args.speech else list(REBUILT_RUNS)
    report = build_report(speeches)
    est = estimate(report)
    report["estimate"] = est
    print(render_text(report, est))

    out = Path(args.json) if args.json else OUT_DIR / f"{OUT_STEM}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    if args.write_sids:
        d = Path(args.write_sids)
        d.mkdir(parents=True, exist_ok=True)
        for r in report["per_speech"]:
            p = d / f"{r['speech']}.json"
            p.write_text(json.dumps(r["sids"], indent=2) + "\n", encoding="utf-8")
            print(f"wrote {p} ({len(r['sids'])} sids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
