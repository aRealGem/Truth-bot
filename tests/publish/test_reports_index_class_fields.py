"""Step 3f (P-senate): reports.json rows carry the report's occasion class.

Every row gets "class" (report_class of its speech_id) and "class_label"
(report_class_label of that class) so a reader of reports.json alone can
partition or group by occasion without re-deriving from data/report_classes.json.
The fields are additive and ALWAYS present -- a report with no authored class
still gets "unclassified" and its label, never a missing/null field.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from truthbot.publish.site import (SitePublisher, SiteReport, UNCLASSIFIED,
                                   report_class_label)


def _site_report(speech_id: str) -> SiteReport:
    return SiteReport(
        report_id=str(uuid.uuid4()), speaker="Synthetic Speaker",
        role="President", date=datetime(2026, 3, 4), venue="Test Hall",
        transcript_source_url="https://example.org/transcript",
        bundles=[],
        generated_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        speech_id=speech_id)


def _published_row(tmp_path, speech_id: str, sr: SiteReport | None = None) -> dict:
    # Fresh site root per call -- reports_index dedupes on report_id, not
    # speech_id/date, so reusing tmp_path across calls with distinct random
    # report_ids would accumulate rows instead of replacing one.
    root = tmp_path / (speech_id or "no_speech_id") / str(uuid.uuid4())
    sr = sr or _site_report(speech_id)
    SitePublisher(site_root=str(root)).publish(sr)
    rows = json.loads((root / "data" / "reports.json").read_text("utf-8"))
    assert len(rows) == 1
    return rows[0]


def test_presidential_report_gets_its_class_and_label(tmp_path) -> None:
    row = _published_row(tmp_path, "trump_2026")
    assert row["class"] == "presidential_address"
    assert row["class_label"] == "Presidential addresses"


def test_senate_report_gets_its_class_and_label(tmp_path) -> None:
    row = _published_row(tmp_path, "budd_2025-04-02")
    assert row["class"] == "senate_floor"
    assert row["class_label"] == "Senate floor speeches"


def test_unknown_speech_id_is_unclassified_not_omitted(tmp_path) -> None:
    row = _published_row(tmp_path, "stranger_2030")
    assert row["class"] == UNCLASSIFIED
    assert row["class_label"] == report_class_label(UNCLASSIFIED)
    assert row["class_label"]  # non-empty


def test_both_fields_are_present_on_every_row(tmp_path) -> None:
    for speech_id in ("trump_2026", "budd_2025-04-02", "stranger_2030", ""):
        row = _published_row(tmp_path, speech_id)
        assert "class" in row
        assert "class_label" in row
        assert row["class"] is not None
        assert row["class_label"] is not None


def test_existing_fields_are_untouched(tmp_path) -> None:
    """Additive only -- the pre-existing keys are all still present with the
    same values they had before class/class_label were added."""
    sr = _site_report("budd_2025-04-02")
    row = _published_row(tmp_path, "budd_2025-04-02", sr=sr)
    expected = {
        "id":                  sr.report_id,
        "date":                sr.date_str,
        "speaker":             sr.speaker,
        "role":                sr.role,
        "venue":               sr.venue,
        "claim_count":         len(sr.checkable_bundles),
        "panel_roster":        dict(sr.panel_roster or {}),
        "triage_count":        len(sr.characterization or []),
        "verdict_distribution": sr.verdict_distribution,
        "verdict_distribution_lenient": sr.verdict_distribution_lenient,
        "verdict_distribution_strict":  sr.verdict_distribution_strict,
        "model_agreement_rate": round(sr.model_agreement_rate, 3),
        "url":                 sr.report_url,
        "source_of_claims":                          sr.source_of_claims or sr.speaker,
        "source_of_claims_professional_public_title": sr.source_of_claims_professional_public_title or sr.role,
        "event":               sr.event,
        "channel":             sr.channel,
    }
    for key, value in expected.items():
        assert row[key] == value, f"field {key!r} changed: {row[key]!r} != {value!r}"
    # generated_at is present and unchanged in shape (ISO string), just not
    # byte-compared here since the two SiteReport constructions differ in
    # report_id/object identity, not in generated_at itself.
    assert row["generated_at"]
    assert "tier_counts" in row and "tier_fallback" in row
