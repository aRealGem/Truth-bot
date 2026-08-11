"""Phase-2 dry-run diff (scripts/phase2_dryrun_diff.py): disposition and
citation-impact logic over a synthetic artifact — no metrics/ dependence.

The integration path (the five published artifacts) is exercised separately
and skipped when the untracked run files are absent (e.g. fresh clones / CI).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "phase2_dryrun_diff", REPO / "scripts" / "phase2_dryrun_diff.py")
dryrun = importlib.util.module_from_spec(_SPEC)
sys.modules["phase2_dryrun_diff"] = dryrun   # dataclasses need this at exec
_SPEC.loader.exec_module(dryrun)

UTTERANCE = date(2022, 3, 1)                 # biden_2022-shaped
WINDOW = (date(2020, 1, 1), date(2022, 6, 1))


def _item(url, *, tier="Government", published=None, snippet="",
          supports=True):
    return {"source_url": url, "source_tier": tier,
            "published_at": published, "snippet": snippet,
            "supports_claim": supports}


def test_dispositions_cover_every_new_rule():
    pack = [
        # E1: clean contemporaneous gov item — kept, credits
        _item("https://www.bls.gov/news.release/archives/empsit_02042022.htm",
              published="2022-02-04"),
        # E2: fact-check — excluded regardless of date
        _item("https://www.politifact.com/factchecks/2022/mar/01/x/",
              tier="FactCheck", published="2022-02-20"),
        # E3: dated past fair-game (utterance+7d) — era violation
        _item("https://www.nytimes.com/2022/05/01/x.html",
              tier="Established", published="2022-05-01"),
        # E4: pre-window date carried only in the snippet stamp
        _item("https://example.com/old", tier="Other",
              snippet="[2019-06-01] ancient"),
        # E5: live latest-release pointer — mutable endpoint
        _item("https://www.bls.gov/news.release/empsit.nr0.htm"),
        # E6..E9: POLITICAL by NEW tiering (stored tier says Government for
        # E6 → also a tier-flip); 4th survivor gets s5-capped
        _item("https://www.speaker.gov/newsroom/a", tier="Government"),
        _item("https://www.dnc.org/press/b", tier="Political"),
        _item("https://www.gop.com/press-release/c", tier="Political"),
        _item("https://www.dnc.org/press/d", tier="Political"),
        # E10: post-speech inside fair-game — kept but context-only
        _item("https://www.reuters.com/markets/x", tier="Wire",
              published="2022-03-05", supports=False),
    ]
    evals = dryrun.evaluate_claim_items(pack, None, UTTERANCE, WINDOW)
    by_e = {ev.e: ev for ev in evals}
    assert by_e[1].disposition == "kept" and not by_e[1].post_speech
    assert by_e[2].disposition == "fc-excluded"
    assert by_e[3].disposition == "era-violation"
    assert "fair-game" in by_e[3].reason
    assert by_e[4].disposition == "era-violation"      # snippet-stamped date
    assert "coded window" in by_e[4].reason
    assert by_e[5].disposition == "mutable-endpoint"
    assert [by_e[e].disposition for e in (6, 7, 8)] == ["kept"] * 3
    assert by_e[9].disposition == "s5-capped"
    assert by_e[6].tier_flip                          # Government → Political
    assert by_e[10].disposition == "kept" and by_e[10].post_speech


def test_citation_impact_and_quota():
    pack = [
        _item("https://www.politifact.com/factchecks/2022/x/",  # E1: removed
              tier="FactCheck", published="2022-02-01", supports=False),
        _item("https://www.bls.gov/news.release/archives/empsit_02042022.htm",
              published="2022-02-04", supports=True),            # E2: kept
        _item("https://www.nytimes.com/2022/02/10/x.html",
              tier="Established", published="2022-02-10", supports=True),
    ]
    row = {"sid": "syn:0001", "verdict": "TRUE", "citations": ["E1", "E2"],
           "reasoning": "E1 and E2 agree; E3 adds context."}
    res = dryrun.analyze_claim("syn:0001", pack, None, row, UTTERANCE, WINDOW)
    assert res.impact == "cited-item-lost" and res.verdict_cited_lost
    assert res.lost == [{"e": 1,
                         "url": "https://www.politifact.com/factchecks/2022/x/",
                         "disposition": "fc-excluded",
                         "reason": "domain:politifact.com",
                         "was_cited": True, "in_rationale": True}]
    # E2 + E3 survive, bearing, T1/T3 → 2 credits, no gate
    assert (res.credits_after, res.would_gate) == (2, False)

    # drop E3 → only 1 credit left on a decided verdict → would-gate
    res2 = dryrun.analyze_claim("syn:0002", pack[:2], None,
                                dict(row, citations=["E2"], reasoning=""),
                                UTTERANCE, WINDOW)
    assert res2.impact == "none"
    assert (res2.credits_after, res2.would_gate) == (1, True)

    # same starved pack, UNVERIFIABLE verdict → never would-gate
    res3 = dryrun.analyze_claim("syn:0003", pack[:2], None,
                                dict(row, verdict="UNVERIFIABLE",
                                     citations=[], reasoning=""),
                                UTTERANCE, WINDOW)
    assert not res3.would_gate and not res3.decided


def test_rationale_only_and_context_only_classifications():
    pack = [
        _item("https://www.bls.gov/news.release/archives/a.htm",
              published="2022-02-04"),                            # E1 kept
        _item("https://www.politifact.com/factchecks/2022/y/",    # E2 removed
              tier="FactCheck", published="2022-02-01"),
    ]
    row = {"sid": "syn:0004", "verdict": "TRUE", "citations": ["E1"],
           "reasoning": "E2 also discussed this."}
    res = dryrun.analyze_claim("syn:0004", pack, None, row, UTTERANCE, WINDOW)
    assert res.impact == "rationale-mentions-lost-item"
    assert not res.verdict_cited_lost

    pack2 = [
        _item("https://www.bls.gov/news.release/archives/a.htm",
              published="2022-02-04"),
        _item("https://www.reuters.com/markets/y",                # post-speech
              tier="Wire", published="2022-03-04", supports=False),
    ]
    row2 = {"sid": "syn:0005", "verdict": "TRUE", "citations": ["E1", "E2"],
            "reasoning": ""}
    res2 = dryrun.analyze_claim("syn:0005", pack2, None, row2,
                                UTTERANCE, WINDOW)
    assert res2.impact == "context-only-cited"
    assert res2.dispositions["post_speech"] == 1
    # the post-speech Wire item is de-credited → 1 credit → gates
    assert (res2.credits_after, res2.would_gate) == (1, True)


def test_pool_candidate_set_maps_erefs_and_restores_credits():
    pack = [
        _item("https://www.politifact.com/factchecks/2022/z/",    # E1 removed
              tier="FactCheck", published="2022-02-01", supports=False),
        _item("https://www.bls.gov/news.release/archives/b.htm",
              published="2022-02-04", supports=True),             # E2 kept
    ]
    # pool = pre-cap superset in its own order (ids regenerated → URL match)
    pool = [
        pack[1],
        _item("https://www.apnews.com/article/extra",             # pool-only
              tier="Wire", published="2022-02-15", supports=False),
        pack[0],
    ]
    row = {"sid": "syn:0006", "verdict": "FALSE", "citations": ["E1"],
           "reasoning": ""}
    res = dryrun.analyze_claim("syn:0006", pack, pool, row, UTTERANCE, WINDOW)
    assert res.pool_used
    assert res.impact == "cited-item-lost"            # E1 mapped through pool
    assert [l["e"] for l in res.lost] == [1]          # pool-only extras ≠ lost
    # E2 (gov) + surviving pool-only Wire extra → 2 credits, no gate
    assert (res.credits_after, res.would_gate) == (2, False)


def test_analyze_artifact_totals_shape():
    artifact = {
        "evidence": {"syn:0001": [
            _item("https://www.bls.gov/news.release/archives/a.htm",
                  published="2022-02-04"),
        ]},
        "rows": [{"sid": "syn:0001", "verdict": "TRUE",
                  "citations": ["E1"], "reasoning": ""}],
    }
    results = dryrun.analyze_artifact(artifact, UTTERANCE)
    totals = dryrun._totals(results)
    assert totals["claims"] == 1 and totals["decided"] == 1
    assert totals["claims_losing_items"] == 0
    assert totals["would_gate"] == 1                  # single credit < 2


@pytest.mark.skipif(
    not (REPO / "metrics" / "pca_runs"
         / f"{dryrun.PUBLISHED_RUNS['trump_2026'][0]}.json").exists(),
    reason="published run artifacts not present (untracked metrics/)")
def test_integration_worksheet_builds():
    ws = dryrun.build_worksheet()
    assert {r["speech_id"] for r in ws["per_report"]} == set(
        dryrun.PUBLISHED_RUNS)
    assert ws["totals"]["claims"] == sum(
        r["totals"]["claims"] for r in ws["per_report"])
    assert ws["scope_option_a_minimal"]["count"] == len(
        ws["scope_option_a_minimal"]["sids"])
    json.dumps(ws)                                    # serializable end-to-end
