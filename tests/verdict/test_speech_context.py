"""Offline tests for Layer B/C temporal grounding (charter: as-of-utterance veracity)."""
from datetime import date

from truthbot.verdict import speech_context as sc


def test_speech_date_resolves_known_speeches():
    assert sc.speech_date_for("biden_2022:0025") == date(2022, 3, 1)
    assert sc.speech_date_for("trump_2026:0020") == date(2026, 2, 24)
    assert sc.speech_date_for("unknown_9999:0001") is None
    assert sc.speech_date_for("") is None


def test_preamble_anchors_on_utterance_date_and_today():
    p = sc.build_temporal_preamble("trump_2026:0020", today=date(2026, 7, 14))
    assert "2026-02-24" in p                     # utterance date
    assert "2026-07-14" in p                     # today authoritative
    assert "AS OF the utterance date" in p
    assert "training cutoff is NOT grounds for UNVERIFIABLE" in p
    assert p.endswith("\n\n")                    # concatenates cleanly before context


def test_preamble_is_speaker_blind_I3():
    p = sc.build_temporal_preamble("biden_2022:0025", today=date(2026, 7, 14))
    low = p.lower()
    assert "biden" not in low and "trump" not in low and "speaker" not in low


def test_reference_period_included_when_given():
    p = sc.build_temporal_preamble("biden_2022:0115", reference_period="calendar year 2021",
                                   today=date(2026, 7, 14))
    assert "calendar year 2021" in p


def test_unknown_speech_yields_empty_preamble():
    assert sc.build_temporal_preamble("mystery:0001", today=date(2026, 7, 14)) == ""
