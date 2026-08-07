"""Deterministic agreed-verdict audit lints (remediation v2, 1.12).

Synthetic fire / does-not-fire pairs per lint, orchestration contract
(``audit_rows`` — decided non-split rows only), the Phase-3 model-pass
selection (pure, seeded), and frozen REAL fixtures from the five published
artifacts (see ``tests/fixtures/verdict_audit/``) guarding against false
positives on verdicts the 2026-07-21 model audit found sound.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from truthbot.verdict import verdict_audit as va

UTT = date(2026, 2, 24)

EV = [{"snippet": "[2026-01-13] BLS: core CPI ran 2.7% annual in Dec 2025.",
       "source_name": "BLS", "source_url": "https://bls.gov/cpi"}]


# ── measure_alignment (queue) ────────────────────────────────────────────────

def test_measure_alignment_fires_when_rationale_ignores_the_measure():
    f = va.lint_measure_alignment(
        "Inflation fell to 1.7 percent this year.",
        "Multiple sources describe broad economic conditions under the "
        "administration and public sentiment about prices.",
        EV, UTT)
    assert f is not None
    assert f.lint == "measure_alignment" and f.action == "queue"


def test_measure_alignment_passes_on_shared_category_or_numeral():
    # shared category (percent, via %)
    assert va.lint_measure_alignment(
        "Inflation fell to 1.7 percent.",
        "BLS data shows core CPI at 2.7%, not the claimed figure.",
        EV, UTT) is None
    # claim numeral echoed
    assert va.lint_measure_alignment(
        "Inflation fell to 1.7 percent.",
        "The 1.7 figure matches no official series.", EV, UTT) is None
    # equivalent: rationale reasons in SOME measure vocabulary
    assert va.lint_measure_alignment(
        "The unemployment rate hit 3.9 percent.",
        "Official data put the annual figure far higher.", EV, UTT) is None


def test_measure_alignment_skips_measureless_claims_and_empty_rationales():
    assert va.lint_measure_alignment(
        "Sarah Beckstrom died defending our capital.",
        "Wire sources confirm the death on patrol.", EV, UTT) is None
    assert va.lint_measure_alignment(
        "Inflation fell to 1.7 percent.", "", EV, UTT) is None


def test_measure_alignment_weak_categories_need_a_numeral():
    # "levels" in a quoted, non-quantified claim must not queue (calibration:
    # the Rehnquist judicial-vacancies quote).
    assert va.lint_measure_alignment(
        'The Chief Justice wrote: "Vacancies cannot remain at such high '
        'levels indefinitely."',
        "Sources confirm he wrote that exact sentence.", EV, UTT) is None
    # but a quantified rate-claim still counts as stating a measure
    f = va.lint_measure_alignment(
        "The unemployment rate hit 3.9 this year.",
        "Sources broadly discuss the labor market and hiring sentiment.",
        EV, UTT)
    assert f is not None and f.action == "queue"


# ── pct_vs_pp ────────────────────────────────────────────────────────────────

def test_pct_vs_pp_fires_on_crossed_units_around_shared_numeral():
    f = va.lint_pct_vs_pp(
        "Unemployment fell 2 percent.",
        "Official series show a drop of 2 percentage points, not more.",
        EV, UTT)
    assert f is not None and f.lint == "pct_vs_pp" and f.action == "flag"
    # and the reverse direction
    f = va.lint_pct_vs_pp(
        "Rates fell by 2 percentage points.",
        "The decline was 2 percent by every measure.", EV, UTT)
    assert f is not None


def test_pct_vs_pp_quiet_without_shared_numeral_or_when_both_present():
    assert va.lint_pct_vs_pp(
        "Unemployment fell 2 percent.",
        "Official series show a drop of 3 percentage points.", EV, UTT) is None
    assert va.lint_pct_vs_pp(
        "Unemployment fell 2 percent.",
        "A 2 percentage-point drop is 2 percent only at a 100% base — the "
        "rationale engages both units.", EV, UTT) is None


# ── quarterly_vs_annual / rate_vs_level / nominal_vs_real ────────────────────

def test_quarterly_vs_annual_fires_only_on_exclusive_mismatch():
    f = va.lint_quarterly_vs_annual(
        "Growth in the last three months hit a record.",
        "Annualized full-year growth was far lower.", EV, UTT)
    assert f is not None and f.action == "flag"
    # rationale that engages BOTH sides is fine
    assert va.lint_quarterly_vs_annual(
        "Growth in the last three months hit a record.",
        "Q4 annualized growth was 2.7%, engaging the quarterly figure.",
        EV, UTT) is None
    # colloquial "three quarters of Americans" is not a calendar quarter
    assert va.lint_quarterly_vs_annual(
        "Three quarters of Americans agree.",
        "Annual polling confirms the share.", EV, UTT) is None


def test_rate_vs_level_fires_only_on_exclusive_mismatch():
    f = va.lint_rate_vs_level(
        "The crime rate fell 30 percent.",
        "FBI data shows the level of offenses rose in absolute terms.",
        EV, UTT)
    assert f is not None and f.action == "flag"
    assert va.lint_rate_vs_level(
        "The crime rate fell 30 percent.",
        "The rate did fall even though the level rose.", EV, UTT) is None
    # institutional "at the federal level" is not a measure
    assert va.lint_rate_vs_level(
        "The crime rate fell 30 percent.",
        "Policy changed at the federal level; the rate data confirms it.",
        EV, UTT) is None


def test_nominal_vs_real_fires_only_on_exclusive_mismatch():
    f = va.lint_nominal_vs_real(
        "Real wages rose under my administration.",
        "Nominal earnings rose 4% over the period.", EV, UTT)
    assert f is not None and f.action == "flag"
    assert va.lint_nominal_vs_real(
        "Real wages rose under my administration.",
        "Nominal earnings rose 4% but inflation-adjusted wages fell.",
        EV, UTT) is None
    # bare "real" ("a real problem") is not the economic modifier
    assert va.lint_nominal_vs_real(
        "We have a real problem at the border.",
        "Nominal spending figures are irrelevant here.", EV, UTT) is None


# ── baseline_selection ───────────────────────────────────────────────────────

def test_baseline_selection_fires_when_rationale_years_miss_the_anchor():
    f = va.lint_baseline_selection(
        "Since I took office, the deficit has been cut in half.",
        "The 2023 deficit was $1.7 trillion against 2026 projections.",
        EV, UTT)  # utterance 2026 → anchors {2018,2019,2020,2021,2024,2025}
    assert f is not None and f.lint == "baseline_selection"
    assert f.action == "flag"


def test_baseline_selection_quiet_on_anchor_year_verbal_engagement_or_no_years():
    # rationale names the term-start-adjacent year
    assert va.lint_baseline_selection(
        "Since I took office, the deficit has been cut in half.",
        "Against the 2024 baseline the deficit halved.", EV, UTT) is None
    # rationale engages the anchor verbally, no year named
    assert va.lint_baseline_selection(
        "By the end of this year, the deficit will be down to less than "
        "half what it was before I took office.",
        "Data tracked below half the pre-Biden baseline when he took office.",
        EV, date(2022, 3, 1)) is None
    # rationale names no years at all → nothing to check
    assert va.lint_baseline_selection(
        "Since I took office, the deficit has been cut in half.",
        "Treasury data confirm a halving of the deficit.", EV, UTT) is None
    # no utterance date → skip (cannot derive term start)
    assert va.lint_baseline_selection(
        "Since I took office, the deficit has been cut in half.",
        "The 2023 deficit was $1.7 trillion.", EV, None) is None


def test_baseline_selection_years_ago_anchor():
    f = va.lint_baseline_selection(
        "Gas cost half as much four years ago.",
        "Prices in 2018 were roughly comparable.", EV, UTT)  # 2026-4=2022±1
    assert f is not None
    assert va.lint_baseline_selection(
        "Gas cost half as much four years ago.",
        "EIA data for 2022 shows prices near half.", EV, UTT) is None


# ── colloquial_recency ───────────────────────────────────────────────────────

def test_colloquial_recency_fires_on_literalist_gap_reading():
    f = va.lint_colloquial_recency(
        "We recently secured the border.",
        "The policy took effect eight months earlier, so it is not recent.",
        EV, UTT)
    assert f is not None and f.action == "flag"


def test_colloquial_recency_quiet_without_literalism():
    assert va.lint_colloquial_recency(
        "We recently secured the border.",
        "DHS data confirms crossings fell after the policy change.",
        EV, UTT) is None
    assert va.lint_colloquial_recency(
        "The deficit doubled last decade.",
        "It happened years earlier — but the claim has no recency word.",
        EV, UTT) is None


# ── invented_referent ────────────────────────────────────────────────────────

def test_invented_referent_fires_on_ungrounded_proper_noun():
    f = va.lint_invented_referent(
        "He rescued fourteen people from the flood.",
        "Reports credit the Colorado Rescue Brigade with the operation.",
        EV, UTT, claim_context="A guest in the gallery was honored.")
    assert f is not None and f.lint == "invented_referent"
    assert "Colorado Rescue Brigade" in f.detail


def test_invented_referent_grounded_by_claim_context_evidence_or_head_token():
    ev = [{"snippet": "Spc. Sarah Beckstrom died of wounds on Guard patrol.",
           "source_name": "DC National Guard"}]
    # phrase in evidence
    assert va.lint_invented_referent(
        "She died defending our capital.",
        "Sources confirm Sarah Beckstrom died on National Guard patrol.",
        ev, UTT) is None
    # head token grounds a name variant ("Remsburg's" ↔ "Cory Remsburg")
    ev2 = [{"snippet": "Details of Remsburg's roadside-bomb injury.",
            "source_name": "CSM"}]
    assert va.lint_invented_referent(
        "For months, he lay in a coma.",
        "Cory Remsburg lay in a coma for three months.", ev2, UTT) is None
    # sentence-initial capitalization is not a proper noun
    assert va.lint_invented_referent(
        "The economy grew.", "Multiple Government sources confirm growth.",
        EV, UTT) is None
    # civic vocabulary / compound modifiers never flag
    assert va.lint_invented_referent(
        "We passed the law.",
        "The White House and the Obama-launched program confirm it, per the "
        "State Dept.", EV, UTT) is None


# ── orchestration ────────────────────────────────────────────────────────────

def _row(sid, verdict="FALSE", reasoning="No measures discussed here at all.",
         **kw):
    return {"sid": sid, "status": "resolved", "verdict": verdict,
            "reasoning": reasoning, **kw}


def test_audit_rows_covers_decided_non_split_rows_only():
    claims = [{"sid": f"s:{i:04d}",
               "text": "Inflation fell to 1.7 percent.",
               "context": ""} for i in range(4)]
    rows = [
        _row("s:0000"),                                   # audited, queues
        _row("s:0001", verdict="UNVERIFIABLE"),           # not decided
        _row("s:0002", split=True),                       # split → skipped
        _row("s:0003", verdict="TRUE",
             reasoning="BLS shows 1.7% exactly."),        # audited, clean
    ]
    out = va.audit_rows(claims, rows, evidence={}, utterance=UTT)
    assert set(out) == {"s:0000", "s:0003"}
    assert out["s:0000"]["audit_queue"] is True
    assert "measure_alignment" in out["s:0000"]["audit_flags"]
    assert out["s:0003"] == {"audit_flags": [], "audit_queue": False}


def test_audit_rows_gates_colloquial_recency_on_adverse_verdicts():
    claims = [{"sid": "s:0000", "text": "We recently secured the border.",
               "context": ""},
              {"sid": "s:0001", "text": "We recently secured the border.",
               "context": ""}]
    reasoning = ("The 1.7% policy took effect months earlier, so it is "
                 "not recent.")
    rows = [_row("s:0000", verdict="FALSE", reasoning=reasoning),
            _row("s:0001", verdict="TRUE", reasoning=reasoning)]
    out = va.audit_rows(claims, rows, evidence={}, utterance=UTT)
    assert "colloquial_recency" in out["s:0000"]["audit_flags"]
    assert "colloquial_recency" not in out["s:0001"]["audit_flags"]


def test_audit_rows_accepts_evidence_packs_and_artifact_dicts():
    from truthbot.models import SourceTier
    from truthbot.verdict.evidence_pack import EvidencePack, PackItem

    claims = [{"sid": "s:0000", "text": "It fell 2 percent.", "context": ""}]
    rows = [_row("s:0000", verdict="TRUE",
                 reasoning="Confirmed by the Fiscal Observatory Council: "
                           "2 percent.")]
    pack = EvidencePack(sid="s:0000", window=None, items=[
        PackItem(pack_id="E1", source_name="Fiscal Observatory Council",
                 source_url="https://x.test", tier=SourceTier.GOVERNMENT,
                 snippet="It fell 2 percent.", retrieved_at="2026-01-01",
                 sha256="x")])
    out = va.audit_rows(claims, rows, {"s:0000": pack}, utterance=UTT)
    assert out["s:0000"]["audit_flags"] == []  # referent grounded via pack
    out2 = va.audit_rows(
        claims, rows,
        {"s:0000": [{"snippet": "It fell 2 percent.",
                     "source_name": "Fiscal Observatory Council"}]},
        utterance=UTT)
    assert out2["s:0000"]["audit_flags"] == []


# ── superlative_self_citation (queue) — A6 / D11 rev-B ───────────────────────

SUPER = "We deported illegal alien criminals at record numbers."


def _ev(*specs):
    """[(tier, snippet[, role]), …] → artifact-shaped pack items."""
    out = []
    for spec in specs:
        tier, snippet = spec[0], spec[1]
        item = {"source_tier": tier, "snippet": snippet,
                "source_name": "src", "source_url": "https://example.gov/x"}
        if len(spec) > 2:
            item["role"] = spec[2]
        out.append(item)
    return out


def test_superlative_fires_when_every_citation_is_self_s5():
    pack = _ev(("Wire", "AP: removals totalled 271,484 in FY2024."),
               ("Political", "DHS touts record 527,000+ removals."),
               ("Political", "DHS: historic removal year."))
    f = va.lint_superlative_self_citation(
        SUPER, "Official DHS releases support record-number removals.",
        pack, UTT, citations=["E2", "E3"])
    assert f is not None
    assert f.lint == "superlative_self_citation" and f.action == va.QUEUE
    # the finding names the effect the D11 rule prescribes
    assert "C-EVAL" in f.detail and "arbiter" in f.detail


def test_superlative_fires_when_every_citation_is_preliminary():
    pack = _ev(("Other", "CCJ PROJECTS a ~21% homicide drop for 2025."),
               ("Wire", "AP: FBI confirmation was not yet available."))
    f = va.lint_superlative_self_citation(
        "Last year the murder rate saw its single largest decline in "
        "recorded history.",
        "Preliminary data showed a large decline.", pack, UTT,
        citations=["E1", "E2"])
    assert f is not None and f.action == va.QUEUE
    assert "preliminary" in f.detail


def test_superlative_cleared_by_one_independent_final_citation():
    pack = _ev(("Political", "DHS touts record removals."),
               ("Government", "DHS OHSS monthly removal tables, FY2025."))
    assert va.lint_superlative_self_citation(
        SUPER, "Reasoning.", pack, UTT, citations=["E1", "E2"]) is None


def test_superlative_skips_non_superlative_claims_and_uncited_rows():
    pack = _ev(("Political", "DHS press release."))
    # no superlative token in the claim
    assert va.lint_superlative_self_citation(
        "We removed 271,484 people in fiscal 2024.", "Reasoning.", pack, UTT,
        citations=["E1"]) is None
    # cited nothing: an abstention / gate-forced row, not this lint's class
    assert va.lint_superlative_self_citation(
        SUPER, "Reasoning.", pack, UTT, citations=[]) is None
    assert va.lint_superlative_self_citation(
        SUPER, "Reasoning.", pack, UTT) is None


def test_superlative_reads_the_evidential_role_as_well_as_the_tier():
    # A GOVERNMENT-tier item can still be the speaker's own record; the D11.2
    # role axis is what says so.
    pack = _ev(("Government", "Agency record of the act.", "primary-record"),
               ("Government", "Agency statement.", "attribution-only"))
    f = va.lint_superlative_self_citation(
        SUPER, "Reasoning.", pack, UTT, citations=["E1", "E2"])
    assert f is not None and "self/S5" in f.detail
    # a CORROBORANT (participant) role is independent enough — no finding
    pack2 = _ev(("Government", "Agency record.", "primary-record"),
                ("Government", "Participant record.", "corroborant"))
    assert va.lint_superlative_self_citation(
        SUPER, "Reasoning.", pack2, UTT, citations=["E1", "E2"]) is None


def test_cited_items_resolves_by_pack_id_then_by_pack_order():
    items = [{"pack_id": "E1", "snippet": "a"}, {"pack_id": "E2", "snippet": "b"}]
    assert [i["snippet"] for i in va.cited_items(items, ["E2"])] == ["b"]
    # artifact shape: no E-ref on the item, so position addresses the pack
    plain = [{"id": "uuid-a", "snippet": "a"}, {"id": "uuid-b", "snippet": "b"}]
    assert [i["snippet"] for i in va.cited_items(plain, ["E2", "E9", "junk"])] == ["b"]


def test_superlative_lint_is_registered_and_reachable_through_run_lints():
    assert "superlative_self_citation" in [n for n, _, _ in va.ALL_LINTS]
    pack = _ev(("Political", "DHS touts record removals."))
    names = [f.lint for f in va.run_lints(
        SUPER, "Official DHS releases support the record framing.",
        pack, UTT, citations=["E1"], adverse=False)]
    assert "superlative_self_citation" in names
    # audit_rows threads the row's citations through for free
    out = va.audit_rows(
        [{"sid": "s:0000", "text": SUPER}],
        [{"sid": "s:0000", "verdict": "TRUE", "citations": ["E1"],
          "reasoning": "Official DHS releases support the record framing."}],
        {"s:0000": pack}, UTT)
    assert out["s:0000"]["audit_queue"] is True
    assert "superlative_self_citation" in out["s:0000"]["audit_flags"]


# ── the shared superlative token list (A6: ONE list, two modules) ────────────

def test_superlative_token_list_is_shared_with_the_shape_lint():
    from truthbot.checkworthy import shape_lint

    assert va.SUPERLATIVE_RX is shape_lint.SUPERLATIVE_RX
    assert va.SUPERLATIVE_TOKENS is shape_lint.SUPERLATIVE_TOKENS


def test_superlative_token_list_carries_the_rev_b_additions():
    from truthbot.checkworthy.shape_lint import SUPERLATIVE_TOKENS, has_superlative

    # the D11 sign-off words survive verbatim …
    for tok in ("largest", "biggest", "smallest", "first", "last", "most",
                "least", "best", "worst", "record", "historic",
                "unprecedented", "strongest", "fastest", "greatest",
                "highest", "lowest"):
        assert tok in SUPERLATIVE_TOKENS
        assert has_superlative(f"It was the {tok} one.")
    # … and the rev-B additions are live
    for tok in ("first-ever", "never before", "in recorded history",
                "in N years"):
        assert tok in SUPERLATIVE_TOKENS
    assert has_superlative("This is the first-ever such program.")
    assert has_superlative("Never before has the deficit fallen this far.")
    assert has_superlative("The single largest decline in recorded history.")
    assert has_superlative("The lowest level in more than five years.")
    assert has_superlative("The lowest number in over 125 years.")


def test_superlative_rev_b_additions_do_not_swallow_forward_looking_plans():
    from truthbot.checkworthy.shape_lint import enforce_shape, has_superlative

    # "in the next N years" is a promise, not a superlative scope.
    assert not has_superlative("We will build it in the next ten years.")
    assert not has_superlative("The program runs within four years.")
    assert enforce_shape("We will hire 5,000 agents in the next four years.",
                         "c-count") == "c-count"


# ── Phase-3 model-pass selection (pure; no model calls) ──────────────────────

def test_select_model_audit_rows_mandatory_plus_seeded_sample():
    rows = (
        [{"sid": "s:0001", "verdict": "MISLEADING",
          "crm114": {"stage1": "FALSE", "final": "MISLEADING"}}]
        + [{"sid": "s:0002", "verdict": "UNVERIFIABLE",
            "provenance_code": "insufficient-qualifying-evidence"}]
        + [{"sid": f"s:{i:04d}", "verdict": "TRUE"} for i in range(10, 30)]
    )
    sel = va.select_model_audit_rows(rows, k=5, seed=42)
    sids = [r["sid"] for r in sel]
    assert sids[:2] == ["s:0001", "s:0002"]      # mandatory coverage first
    assert len(sel) == 7
    # deterministic given (rows, k, seed); different seed → different sample
    assert sids == [r["sid"] for r in va.select_model_audit_rows(rows, 5, 42)]
    assert sids != [r["sid"] for r in va.select_model_audit_rows(rows, 5, 43)]
    # k larger than the pool is clamped, never raises
    assert len(va.select_model_audit_rows(rows, k=999, seed=1)) == 22


def test_agreed_decided_rows_matches_the_one_off_harness_selection():
    rows = [{"sid": "a", "verdict": "TRUE"},
            {"sid": "b", "verdict": "FALSE", "escalated": True},
            {"sid": "c", "verdict": "UNVERIFIABLE"},
            {"sid": "d", "verdict": "MISLEADING"}]
    assert [r["sid"] for r in va.agreed_decided_rows(rows)] == ["a", "d"]


# ── frozen real fixtures (false-positive guard) ──────────────────────────────

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "verdict_audit"

#: name → expected: "sound" fixtures must produce NO queue-action findings;
#: entries with a lint name assert that lint fires (flag action).
SOUND_FIXTURES = [
    "beckstrom_turnaround",   # trump_2026:0461 (model audit: sound)
    "beckstrom_died",         # trump_2026:0469 (model audit: sound)
    "biden_gdp_5_7",          # biden_2022:0115 (model audit: sound)
    "deficit_half",           # biden_2022:0244 (model audit: sound)
    "inflation_1_7",          # trump_2026:0031
    "dei_ended",              # trump_2026:0056
    "college_opportunity",    # obama_2014:0046
    "murder_record_biggest",  # trump_2026:0024 (A6 negative control)
]


def _load_fixture(name):
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text("utf-8"))


def _lint_fixture(fx):
    verdict = str(fx["row"].get("verdict") or "").upper()
    return va.run_lints(
        fx["claim"]["text"], fx["row"].get("reasoning") or "",
        fx["evidence"], date.fromisoformat(fx["utterance"]),
        claim_context=fx["claim"].get("context", "") or "",
        citations=fx["row"].get("citations") or (),
        adverse=verdict in va.ADVERSE)


@pytest.mark.parametrize("name", SOUND_FIXTURES)
def test_sound_fixture_produces_no_queue_findings(name):
    findings = _lint_fixture(_load_fixture(name))
    assert [f for f in findings if f.action == "queue"] == []


def test_sound_true_fixtures_produce_no_findings_at_all():
    # The audit-verified-sound TRUE verdicts must come back completely clean.
    for name in ("beckstrom_died", "biden_gdp_5_7", "deficit_half",
                 "college_opportunity"):
        assert _lint_fixture(_load_fixture(name)) == []


# ── A6 frozen fixtures: the superlative gaming shape, as it really shipped ──

def test_record_removals_fixture_queues_on_self_only_citations():
    """trump_2026:0343 shipped TRUE on a 'record numbers' superlative citing
    E4/E5/E7 — three DHS self-releases — while the AP item (E9) that could
    have borne independently on the count sat UNCITED. Exactly the D11 rev-B
    gaming shape."""
    fx = _load_fixture("record_removals_dhs")
    assert fx["row"]["verdict"] == "TRUE"
    assert fx["row"]["citations"] == ["E4", "E5", "E7"]
    assert all(fx["evidence"][int(c[1:]) - 1]["source_tier"] == "Political"
               for c in fx["row"]["citations"])
    # the AP corroborant is in the pack, just not cited
    assert fx["evidence"][8]["source_tier"] == "Wire"
    findings = {f.lint: f for f in _lint_fixture(fx)}
    assert findings["superlative_self_citation"].action == va.QUEUE


def test_murder_record_pair_fixtures_split_on_the_preliminary_arm():
    """trump_2026:0023 / :0024 rate the SAME statistic. 0023 leans entirely on
    projected/unconfirmed data (CCJ projection + an AP note that FBI
    confirmation was not yet available) and queues; 0024 cites independent,
    unhedged reporting and stays clean. That asymmetry — one queued, one not,
    on one statistic — is why the pair also carries a COHERENCE case in the
    DC-6' acceptance suite."""
    largest = _load_fixture("murder_record_largest")
    biggest = _load_fixture("murder_record_biggest")
    assert largest["claim"]["sid"] == "trump_2026:0023"
    assert biggest["claim"]["sid"] == "trump_2026:0024"
    assert [f.lint for f in _lint_fixture(largest)] == [
        "superlative_self_citation"]
    assert _lint_fixture(biggest) == []


def test_joining_forces_fixture_is_gate_forced_and_skipped_by_audit_rows():
    # obama_2014:0045 shipped UNVERIFIABLE (gate-forced): not a decided row,
    # so the deterministic tier skips it — but the Phase-3 selection contract
    # picks it up through its gate marker.
    fx = _load_fixture("joining_forces")
    assert str(fx["row"]["verdict"]).upper() == "UNVERIFIABLE"
    out = va.audit_rows([fx["claim"]], [fx["row"]], {}, UTT)
    assert fx["row"]["sid"] not in out
    row = dict(fx["row"],
               provenance_code="insufficient-qualifying-evidence")
    assert row in va.select_model_audit_rows([row], k=0, seed=0)


# ── pipeline / publish wiring (remediation v2, 1.12) ─────────────────────────

def test_run_pca_verify_stamps_audit_provenance_and_writes_queue(tmp_path):
    """Offline end-to-end: the audit stage runs post-verdict, pre-bridge —
    rows and bundle provenance carry audit_flags/audit_queue, and queued rows
    land in the re-adjudication JSONL (human inbox; no model call exists
    anywhere on this path)."""
    from truthbot.verdict import publish_pipeline as pp

    sents = [{"sid": "sp:0000",
              "text": "Inflation fell to 1.7 percent in 2025.",
              "context": ""}]

    def classify(rows):
        return [{"sid": s["sid"], "label": "check-worthy", "text": s["text"],
                 "context": s.get("context", "")} for s in rows]

    def adjudicate(chunk):
        rows = [{"sid": c["sid"], "status": "resolved", "verdict": "FALSE",
                 "confidence": 0.9, "citations": [],
                 # rationale ignores the claim's measure → measure_alignment
                 "reasoning": "Sources broadly dispute the rosy picture.",
                 "votes": {"FALSE": 3}} for c in chunk]
        return rows, {"packs": {}, "cost_usd": 0.0}

    qpath = tmp_path / "readjudication_queue.jsonl"
    res = pp.run_pca_verify(sents, layer_a_fn=classify,
                            adjudicate_fn=adjudicate,
                            audit_queue_path=qpath)
    assert res.rows[0]["audit_queue"] is True
    assert "measure_alignment" in res.rows[0]["audit_flags"]
    prov = res.bundles[0].consensus.provenance
    assert prov.audit_queue is True
    assert "measure_alignment" in prov.audit_flags
    recs = [json.loads(l) for l in qpath.read_text().splitlines()]
    assert recs == [{"sid": "sp:0000", "speech_id": "sp",
                     "lints": recs[0]["lints"], "ts": recs[0]["ts"]}]
    assert "measure_alignment" in recs[0]["lints"]


def test_run_pca_verify_clean_rows_write_no_queue_file(tmp_path):
    from truthbot.verdict import publish_pipeline as pp

    sents = [{"sid": "sp:0000",
              "text": "Inflation fell to 1.7 percent in 2025.",
              "context": ""}]

    def classify(rows):
        return [{"sid": s["sid"], "label": "check-worthy", "text": s["text"],
                 "context": s.get("context", "")} for s in rows]

    def adjudicate(chunk):
        rows = [{"sid": c["sid"], "status": "resolved", "verdict": "TRUE",
                 "confidence": 0.9, "citations": [],
                 "reasoning": "BLS confirms 1.7% for 2025.",
                 "votes": {"TRUE": 3}} for c in chunk]
        return rows, {"packs": {}, "cost_usd": 0.0}

    qpath = tmp_path / "readjudication_queue.jsonl"
    res = pp.run_pca_verify(sents, layer_a_fn=classify,
                            adjudicate_fn=adjudicate, audit_queue_path=qpath)
    assert res.rows[0]["audit_flags"] == []
    assert res.rows[0]["audit_queue"] is False
    assert not qpath.exists()          # queue file only appears when needed
    assert res.bundles[0].consensus.provenance.audit_flags == []


def test_provenance_strip_discloses_auto_adjustment_and_audit_marker():
    from truthbot.publish import site
    from truthbot.verdict import bridge

    row = {"sid": "s:0", "status": "resolved", "verdict": "FALSE",
           "confidence": 0.8, "citations": [],
           "reasoning": "Contradicted by BLS.",
           "votes": {"MISLEADING": 2, "FALSE": 1}, "split": False,
           "escalated": False,
           "crm114": {"stage1": "MISLEADING", "final": "FALSE"},
           "audit_flags": ["invented_referent", "measure_alignment"],
           "audit_queue": True}
    claim = {"sid": "s:0", "text": "A claim.",
             "layer_a": {"label": "check-worthy", "source": "A2"}}
    b = bridge.bridge([row], [claim]).bundles[0]
    html = site._pca_provenance_strip(b)
    assert "Auto-adjusted: MISLEADING→FALSE (Severity Classifier)" in html
    assert "audit: invented_referent, measure_alignment · queued for review" in html
    # reader-facing copy never says CRM-114 (established naming policy)
    assert "CRM-114" not in html


def test_claims_json_provenance_carries_audit_fields(tmp_path):
    # the claims.json exporter includes audit_flags/audit_queue so the next
    # regen publishes them (and consistency can later gate on them).
    from types import SimpleNamespace

    from truthbot.publish.site import SitePublisher
    from truthbot.verdict import bridge

    row = {"sid": "s:0", "status": "resolved", "verdict": "TRUE",
           "confidence": 0.9, "citations": [], "reasoning": "ok",
           "votes": {"TRUE": 3}, "split": False, "escalated": False,
           "audit_flags": ["baseline_selection"], "audit_queue": False}
    claim = {"sid": "s:0", "text": "A claim.",
             "layer_a": {"label": "check-worthy", "source": "A2"}}
    b = bridge.bridge([row], [claim]).bundles[0]
    pub = SitePublisher(site_root=tmp_path)
    meta = pub._claim_meta(b, SimpleNamespace(report_id="r1"))
    assert meta["provenance"]["audit_flags"] == ["baseline_selection"]
    assert meta["provenance"]["audit_queue"] is False


def test_olympics_fixture_legitimately_fires_invented_referent():
    # trump_2026:0090 — the single calibration hit over all 530 published
    # rows: the rationale's "Summer Games" appears nowhere in claim, context,
    # or the evidence pack (which says "Summer Olympics"). Flag action only.
    findings = _lint_fixture(_load_fixture("olympics_torch"))
    assert [f.lint for f in findings] == ["invented_referent"]
    assert findings[0].action == "flag"
