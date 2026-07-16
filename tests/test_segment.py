"""Tests for the shippable v2 transcript segmenter."""
from __future__ import annotations

from truthbot.ingest import segment


def test_segment_sids_and_context_window():
    text = "The economy grew by 3 percent. Unemployment fell to 4 percent. We must do more."
    recs = segment.segment(text, "trump_2026")
    assert [r["sid"] for r in recs] == [
        "trump_2026:0000", "trump_2026:0001", "trump_2026:0002"]
    assert all(r["speech"] == "trump_2026" for r in recs)
    # context is prev || this || next
    assert recs[1]["context"] == (
        "The economy grew by 3 percent. || Unemployment fell to 4 percent. || "
        "We must do more.")
    # first has empty prev, last has empty next
    assert recs[0]["context"].startswith("||")
    assert recs[2]["context"].endswith("||")


def test_segment_strips_stage_cues_and_short_fragments():
    text = "Thank you. [Applause] The deficit is down. USA"
    recs = segment.segment(text, "s_2026")
    texts = [r["text"] for r in recs]
    assert "The deficit is down." in texts
    # bracketed cue stripped, and the lone "USA" (<8 chars) dropped
    assert not any("Applause" in t for t in texts)
    assert "USA" not in texts


def test_segment_protects_abbreviations():
    text = "The U.S. economy is strong. Dr. Smith agrees with the data."
    recs = segment.segment(text, "s_2026")
    # "U.S." and "Dr." must not trigger a split → exactly 2 sentences
    assert len(recs) == 2
    assert recs[0]["text"] == "The U.S. economy is strong."


def test_segment_strips_hash_header_lines():
    # Miller-Center-format transcripts carry `#` title/source header lines above the
    # body; they must not become sentences.
    text = ("# State of the Union — Biden 2022\n"
            "# Source: https://millercenter.org/x\n\n"
            "The deficit fell by half. Growth was strong.")
    recs = segment.segment(text, "biden_2022")
    texts = [r["text"] for r in recs]
    assert not any(t.startswith("#") for t in texts)
    assert not any("millercenter" in t for t in texts)
    assert texts == ["The deficit fell by half.", "Growth was strong."]


def test_segment_empty_text():
    assert segment.segment("", "s_2026") == []
    assert segment.split_sentences("   ") == []
