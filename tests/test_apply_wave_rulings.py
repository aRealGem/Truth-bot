"""The $0 application of the 2026-08-10 wave rulings.

Pure helpers only — the driver reads five real artifacts and writes new ones,
which is exercised by running it, not by a unit test. What is tested here is
everything that decides WHAT gets written: which row replaces a newly-gated
claim, where an adopted rationale is allowed to come from, and what a coherence
annotation is allowed to say.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def awr():
    spec = importlib.util.spec_from_file_location(
        "apply_wave_rulings", REPO / "scripts" / "apply_wave_rulings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1. applying a newly-gated claim ──────────────────────────────────────────

def test_gated_row_is_the_pipelines_own_gate_row(awr):
    """An applied withholding must be indistinguishable from one the gate
    produced itself — including the gate's rationale sentence, which is what
    keeps this path clean under the no-blank-rationale lint."""
    old = {"sid": "s:1", "status": "resolved", "verdict": "TRUE",
           "reasoning": "the panel's reason", "votes": {"TRUE": 2},
           "by_role": {"proposer": ["TRUE"]}}
    row = awr.gate_row("s:1", old)
    assert row["verdict"] == "UNVERIFIABLE"
    assert row["provenance_code"] == "insufficient-qualifying-evidence"
    assert row["reasoning"].strip()

    from truthbot.verdict import verdict_audit as va
    assert va.blank_rationale_violations([row]) == []


def test_gated_row_keeps_what_it_superseded(awr):
    """The corrections ledger has to say what the claim used to say; an
    artifact that dropped it would make its own entry unverifiable."""
    old = {"sid": "s:1", "status": "resolved", "verdict": "TRUE",
           "reasoning": "the panel's reason", "votes": {"TRUE": 2}}
    row = awr.gate_row("s:1", old)
    assert row["superseded"]["verdict"] == "TRUE"
    assert row["superseded"]["reasoning"] == "the panel's reason"


def test_apply_only_touches_the_deferred_sids_and_keeps_order(awr):
    rows = [{"sid": f"s:{i}", "status": "resolved", "verdict": "TRUE",
             "reasoning": "r"} for i in range(4)]
    out = awr.apply_deferred_gated(rows, {"s:1", "s:3"})
    assert [r["sid"] for r in out] == ["s:0", "s:1", "s:2", "s:3"]
    assert [r["verdict"] for r in out] == [
        "TRUE", "UNVERIFIABLE", "TRUE", "UNVERIFIABLE"]
    assert rows[1]["verdict"] == "TRUE"      # source rows are not mutated


# ── 2. adopting a rationale from the lineage ─────────────────────────────────

def _blank(sid="s:1", verdict="MISLEADING"):
    return {"sid": sid, "status": "resolved", "verdict": verdict,
            "reasoning": "", "crm114": {"stage1": "DISAGREEMENT",
                                        "final": verdict}}


def _ancestor(run_id, sid, verdict, reasoning, by_role=None):
    return {"run_id": run_id,
            "rows": [{"sid": sid, "verdict": verdict, "reasoning": reasoning,
                      "by_role": by_role or {"arbiter": [verdict]}}]}


def test_adoption_copies_the_ancestors_text_verbatim(awr):
    row = _blank()
    prov = awr.adopt_from_lineage(
        row, [_ancestor("run-a", "s:1", "MISLEADING", "the stored reason")])
    assert row["reasoning"] == "the stored reason"
    assert prov["adopted_from"] == "arbiter"
    assert prov["adopted_from_run"] == "run-a"
    assert prov["synthesized"] is False


def test_adoption_refuses_an_ancestor_with_a_different_verdict(awr):
    """A rationale written for TRUE is not a MISLEADING verdict's reason.
    Adopting it would be fabrication with extra steps."""
    row = _blank()
    assert awr.adopt_from_lineage(
        row, [_ancestor("run-a", "s:1", "TRUE", "why it is true")]) is None
    assert row["reasoning"] == ""


def test_adoption_prefers_the_nearest_ancestor(awr):
    row = _blank()
    awr.adopt_from_lineage(row, [
        _ancestor("near", "s:1", "MISLEADING", "the nearer reason"),
        _ancestor("far", "s:1", "MISLEADING", "the older reason")])
    assert row["reasoning"] == "the nearer reason"


def test_adoption_skips_ancestors_that_are_themselves_blank(awr):
    row = _blank()
    awr.adopt_from_lineage(row, [
        _ancestor("near", "s:1", "MISLEADING", ""),
        _ancestor("far", "s:1", "MISLEADING", "the only stored reason")])
    assert row["reasoning"] == "the only stored reason"


def test_adoption_does_nothing_when_no_ancestor_can_source_it(awr):
    """biden_2022:0432's real shape: nothing in the lineage ever wrote a
    rationale. The row stays blank so the publish-blocking lint catches it —
    inventing text is the one thing the R-3 ruling forbids."""
    row = _blank()
    assert awr.adopt_from_lineage(row, []) is None
    assert row["reasoning"] == "" and "rationale_provenance" not in row


def test_reemit_only_targets_rows_the_lint_flags(awr):
    rows = [_blank("s:1"), {"sid": "s:2", "status": "resolved",
                            "verdict": "TRUE", "reasoning": "already said"}]
    ancestors = [{"run_id": "run-a", "rows": [
        {"sid": "s:1", "verdict": "MISLEADING", "reasoning": "adopted",
         "by_role": {"arbiter": ["MISLEADING"]}},
        {"sid": "s:2", "verdict": "TRUE", "reasoning": "DO NOT USE",
         "by_role": {"arbiter": ["TRUE"]}}]}]
    adopted = awr.reemit_blank_rationales(rows, ancestors)
    assert [a["sid"] for a in adopted] == ["s:1"]
    assert rows[1]["reasoning"] == "already said"


# ── 3. the coherence annotation (D14: ANNOTATE, never force agreement) ───────

_PAIR_CLAIMS = [
    {"sid": "sp:1", "text": "Murders fell by the largest amount in recorded history."},
    {"sid": "sp:2", "text": "It was the biggest drop in murders in recorded history."},
]
_PAIR_ROWS = [
    {"sid": "sp:1", "status": "resolved", "verdict": "MISLEADING",
     "reasoning": "the murders decline is real but the recorded-history framing "
                  "rests on a projection"},
    {"sid": "sp:2", "status": "resolved", "verdict": "TRUE",
     "reasoning": "reporting supports the largest single-year murders decline "
                  "in recorded history"},
]


def test_annotation_names_the_other_claim_and_both_verdicts(awr):
    rows = [dict(r) for r in _PAIR_ROWS]
    conflicts = awr.annotate_coherence(_PAIR_CLAIMS, rows)
    assert [c["sids"] for c in conflicts] == [["sp:1", "sp:2"]]
    assert "sp:2" in rows[0]["coherence_note"]
    assert "sp:1" in rows[1]["coherence_note"]
    assert "MISLEADING" in rows[0]["coherence_note"]
    assert "TRUE" in rows[1]["coherence_note"]


def test_annotation_discloses_rather_than_adjudicates(awr):
    """The D14 disposition is ANNOTATE. A note that told the reader which claim
    was right would be resolving the conflict, which is not the ruling."""
    rows = [dict(r) for r in _PAIR_ROWS]
    awr.annotate_coherence(_PAIR_CLAIMS, rows)
    note = rows[0]["coherence_note"].lower()
    assert "disclosed" in note
    for verdict_word in ("correct", "wrong", "should be", "supersed"):
        assert verdict_word not in note


def test_annotating_silences_the_unannotated_conflict_checker(awr):
    """Which is the point: the checker reports UNANNOTATED conflicts, so a
    published-and-annotated pair is correctly no longer a finding."""
    from truthbot.verdict import verdict_audit as va

    rows = [dict(r) for r in _PAIR_ROWS]
    awr.annotate_coherence(_PAIR_CLAIMS, rows)
    assert va.adjacent_coherence_conflicts(_PAIR_CLAIMS, rows) == []


def test_annotation_never_overwrites_an_existing_note(awr):
    rows = [dict(_PAIR_ROWS[0], coherence_note="an owner wrote this"),
            dict(_PAIR_ROWS[1])]
    awr.annotate_coherence(_PAIR_CLAIMS, rows)
    assert rows[0]["coherence_note"] == "an owner wrote this"


# ── the committed measurement ────────────────────────────────────────────────

def test_the_mechanism_artifact_accounts_for_every_gated_claim():
    """The composition the corrections ledger quotes must add up: each shipped
    sid carries exactly one mechanism, and the parts sum to the whole."""
    path = REPO / "metrics" / "remediation_v2" / "deferred_gated_mechanism.json"
    if not path.exists():
        pytest.skip("mechanism artifact not generated in this tree")
    doc = json.loads(path.read_text("utf-8"))
    assert len(doc["mechanism"]) == doc["shipped_total"]
    assert sum(doc["by_mechanism"].values()) == doc["shipped_total"]
    # D16(alpha) is a RELEASE rule; it must never appear as a gating mechanism.
    assert "D16alpha" not in doc["by_mechanism"]
