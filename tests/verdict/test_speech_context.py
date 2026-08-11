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


# ── A4: all five corpus speeches pinned statically ───────────────────────────

def test_all_five_corpus_speeches_are_statically_pinned():
    """Only biden_2022 + trump_2026 used to be in the map; the other three
    resolved solely because a runner called register_speech_date() at run
    time, so any path that skipped the runner ran with NO era gate at all.
    Pinning is what makes the common case safe by default.

    Asserted against the module SOURCE, not the live dict: register_speech_date
    mutates the global (this suite adds several test speeches), so a runtime
    entry would otherwise read as a static pin — which is the exact
    distinction A4 turns on."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(sc.__file__).read_text(encoding="utf-8"))
    literal = next(node.value for node in tree.body
                   if isinstance(node, ast.AnnAssign)
                   and node.target.id == "SPEECH_DATE")
    pinned = {k.value: tuple(a.value for a in v.args)
              for k, v in zip(literal.keys, literal.values)}
    assert pinned == {
        "clinton_1998": (1998, 1, 27),
        "gwbush_2006": (2006, 1, 31),
        "obama_2014": (2014, 1, 28),
        "biden_2022": (2022, 3, 1),
        "trump_2026": (2026, 2, 24),
    }
    for speech, ymd in pinned.items():          # …and the live map agrees
        assert sc.SPEECH_DATE[speech] == date(*ymd)


def test_every_pinned_speech_resolves_through_a_sid():
    for speech in ("clinton_1998", "gwbush_2006", "obama_2014",
                   "biden_2022", "trump_2026"):
        assert sc.speech_date_for(f"{speech}:0313") is not None, speech


def test_pinned_dates_give_the_historical_speeches_a_real_window():
    """window_for() is derived from the utterance date — unpinned, these three
    speeches had no evidence window either, which is the other half of the
    Obama-2014 rescue-leg failure (2026-dated evidence in a 2014 pack)."""
    from truthbot.verdict.evidence_pack import window_for

    for speech, year in (("clinton_1998", 1998), ("gwbush_2006", 2006),
                         ("obama_2014", 2014)):
        win = window_for(f"{speech}:0001")
        assert win is not None and win[0].year <= year <= win[1].year
