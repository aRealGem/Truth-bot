"""Corrections-machinery tests (P67.6 / PR-3, remediation T1.5).

Pins: schema validation fails loudly; corrections apply to artifact rows
in-memory with old-verdict guard; the note threads bridge → provenance →
claim-card strip; the Corrections page renders entries and an honest empty
state; every page footer links the page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.publish.corrections import (
    CorrectionsError,
    apply_to_artifact,
    load_corrections,
    note_for,
)
from truthbot.publish.site import _render_corrections

ENTRY = {
    "sid": "biden_2022:0115",
    "speech_id": "biden_2022",
    "old_verdict": "FALSE",
    "new_verdict": "TRUE",
    "reason": "Panel confused quarterly annualized GDP with the 5.7% annual figure.",
    "date": "2026-07-21",
    "source": "agreed-verdict-audit-2026-07-21",
}


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "corrections.json"
    p.write_text(json.dumps(doc))
    return p


def test_load_corrections_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_corrections(tmp_path / "nope.json") == []


def test_load_corrections_validates_schema_and_entries(tmp_path: Path) -> None:
    good = _write(tmp_path, {"schema": "truthbot-corrections v1", "entries": [ENTRY]})
    assert load_corrections(good) == [ENTRY]

    with pytest.raises(CorrectionsError, match="schema"):
        load_corrections(_write(tmp_path, {"schema": "v0", "entries": []}))
    with pytest.raises(CorrectionsError, match="missing"):
        load_corrections(_write(tmp_path, {
            "schema": "truthbot-corrections v1",
            "entries": [{**ENTRY, "reason": ""}]}))
    with pytest.raises(CorrectionsError, match="same verdict"):
        load_corrections(_write(tmp_path, {
            "schema": "truthbot-corrections v1",
            "entries": [{**ENTRY, "new_verdict": "FALSE"}]}))
    with pytest.raises(CorrectionsError, match="duplicate"):
        load_corrections(_write(tmp_path, {
            "schema": "truthbot-corrections v1", "entries": [ENTRY, ENTRY]}))


def _artifact() -> dict:
    return {
        "meta": {"speech_id": "biden_2022", "date": "2022-03-01"},
        "rows": [
            {"sid": "biden_2022:0115", "verdict": "FALSE", "status": "resolved",
             "reasoning": "r", "votes": {"FALSE": 2}, "escalated": False},
            {"sid": "biden_2022:0244", "verdict": "FALSE", "status": "resolved",
             "reasoning": "r", "votes": {"FALSE": 2}, "escalated": False},
        ],
        "claims": [],
    }


def test_apply_to_artifact_rewrites_row_and_stamps_note() -> None:
    artifact = _artifact()
    assert apply_to_artifact(artifact, [ENTRY]) == 1
    row = artifact["rows"][0]
    assert row["verdict"] == "TRUE"
    assert row["corrected"]["old"] == "FALSE"
    assert "Corrected FALSE → TRUE (2026-07-21)" in row["corrected"]["note"]
    # untouched sibling
    assert artifact["rows"][1].get("corrected") is None


def test_apply_guards_old_verdict_mismatch_and_unknown_sid() -> None:
    artifact = _artifact()
    with pytest.raises(CorrectionsError, match="expects old verdict"):
        apply_to_artifact(artifact, [{**ENTRY, "old_verdict": "MISLEADING",
                                      "new_verdict": "TRUE"}])
    with pytest.raises(CorrectionsError, match="absent"):
        apply_to_artifact(_artifact(), [{**ENTRY, "sid": "biden_2022:9999"}])


def test_apply_ignores_other_speeches() -> None:
    artifact = _artifact()
    assert apply_to_artifact(artifact, [{**ENTRY, "speech_id": "trump_2026",
                                         "sid": "trump_2026:0001"}]) == 0


def test_correction_note_threads_into_provenance_and_strip() -> None:
    from truthbot.verdict.bridge import _build_provenance
    from truthbot.models import VerdictProvenance

    row = {"votes": {"FALSE": 2}, "split": False, "escalated": False,
           "corrected": {"note": note_for(ENTRY)}}
    prov = _build_provenance(row, {"layer_a": {"label": "check-worthy"}})
    assert "Corrected FALSE → TRUE" in prov.correction_note

    from tests.test_site_render_aggregates import _make_bundle
    from truthbot.models import VerdictLabel
    from truthbot.publish.site import _pca_provenance_strip

    b = _make_bundle(VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")
    b.consensus.provenance = VerdictProvenance(
        layer_a_label="check-worthy", panel_votes={"True": 2},
        correction_note=note_for(ENTRY))
    html = _pca_provenance_strip(b)
    assert "pca-correction" in html
    assert "Corrected FALSE" in html
    assert 'href="../corrections.html"' in html


def test_corrections_page_renders_entries_and_empty_state() -> None:
    html = _render_corrections([ENTRY])
    assert "biden_2022:0115" in html
    assert "FALSE" in html and "TRUE" in html
    assert "2026-07-21" in html
    assert "never" in html  # "never applied silently" policy sentence

    empty = _render_corrections([])
    assert "No corrections have been issued" in empty


def test_footers_link_corrections_page(tmp_path: Path) -> None:
    from tests.test_site_render_aggregates import _make_bundle, _make_site_report
    from truthbot.models import VerdictLabel
    from truthbot.publish.site import SitePublisher

    pub = SitePublisher(site_root=tmp_path, corrections=[])
    sr = _make_site_report([_make_bundle(
        VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")])
    report_path = pub.publish(sr)
    assert 'href="./corrections.html"' in (tmp_path / "index.html").read_text()
    assert 'href="../corrections.html"' in report_path.read_text()
    assert "No corrections have been issued" in (tmp_path / "corrections.html").read_text()
    assert 'corrections.html" rel="related"' in (tmp_path / "feed.xml").read_text()
