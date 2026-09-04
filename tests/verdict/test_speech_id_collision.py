"""The speech_id collision guard (senate-category work).

``_default_speech_id`` is only ``speaker_year``. Two speeches by the same
senator in the same year therefore land on ONE id, and the permissive
registration path would silently file the second speech's claims under the
first speech's utterance date. The real pair that forced this: Elizabeth
Warren's 2025-04-29 floor speech and her 2025-03-05 SASC hearing appearance.
"""
from datetime import date, datetime

import pytest

from truthbot.pipeline import _default_speech_id
from truthbot.verdict import speech_context
from truthbot.verdict.speech_context import (SPEECH_DATE, SpeechIdCollision,
                                             register_speech_date)

WARREN_SPEECH = date(2025, 4, 29)   # floor speech, "first 100 days"
WARREN_HEARING = date(2025, 3, 5)   # SASC hearing appearance


@pytest.fixture(autouse=True)
def _restore_registry():
    """SPEECH_DATE is module-global; keep these tests from leaking into others."""
    before = dict(SPEECH_DATE)
    yield
    SPEECH_DATE.clear()
    SPEECH_DATE.update(before)


def test_default_speech_id_is_only_speaker_and_year():
    """The premise of the guard: both Warren 2025 speeches derive one id."""
    a = _default_speech_id("Elizabeth Warren", datetime(2025, 4, 29))
    b = _default_speech_id("Elizabeth Warren", datetime(2025, 3, 5))
    assert a == b == "elizabeth_warren_2025"


def test_two_warren_2025_speeches_collide_under_the_default_id():
    """THE regression: same derived id, different utterance dates -> refuse."""
    sid = _default_speech_id("Elizabeth Warren", datetime(2025, 4, 29))
    register_speech_date(sid, WARREN_SPEECH, strict=True)
    with pytest.raises(SpeechIdCollision) as exc:
        register_speech_date(sid, WARREN_HEARING, strict=True)
    # The message has to name both dates or it cannot be acted on.
    assert "2025-04-29" in str(exc.value)
    assert "2025-03-05" in str(exc.value)
    # And the first registration must survive the refusal.
    assert SPEECH_DATE[sid] == WARREN_SPEECH


def test_the_same_pair_passes_under_authored_ids():
    """Authored ids carry the date, so the two speeches stay distinct."""
    register_speech_date("warren_2025-04-29", WARREN_SPEECH, strict=True)
    register_speech_date("warren_2025-03-05", WARREN_HEARING, strict=True)
    assert SPEECH_DATE["warren_2025-04-29"] == WARREN_SPEECH
    assert SPEECH_DATE["warren_2025-03-05"] == WARREN_HEARING


def test_strict_rebind_to_the_same_date_is_idempotent():
    register_speech_date("warren_2025-04-29", WARREN_SPEECH, strict=True)
    register_speech_date("warren_2025-04-29", WARREN_SPEECH, strict=True)
    assert SPEECH_DATE["warren_2025-04-29"] == WARREN_SPEECH


def test_default_path_stays_last_write_wins():
    """Non-strict is the documented contract for in-process callers."""
    register_speech_date("scratch_id", WARREN_SPEECH)
    register_speech_date("scratch_id", WARREN_HEARING)
    assert SPEECH_DATE["scratch_id"] == WARREN_HEARING


def test_prepare_speech_threads_strict_to_the_cli_path():
    """The CLI consumes _default_speech_id, so it must register strictly."""
    from truthbot.verdict import publish_pipeline as pp
    seen = {}

    def fake_register(speech_id, utterance, *, strict=False):
        seen["strict"] = strict

    monkey = pytest.MonkeyPatch()
    monkey.setattr(pp, "register_speech_date", fake_register)
    monkey.setattr(pp, "segment", lambda text, sid: [], raising=False)
    try:
        pp.prepare_speech("A sentence.", "warren_2025-04-29", WARREN_SPEECH,
                          strict=True)
    finally:
        monkey.undo()
    assert seen["strict"] is True


def test_the_four_senate_speeches_are_statically_pinned():
    """A re-render never calls the CLI; unpinned ids resolve to None -> no era gate."""
    assert speech_context.SPEECH_DATE["budd_2025-04-02"] == date(2025, 4, 2)
    assert speech_context.SPEECH_DATE["cruz_2026-06-24"] == date(2026, 6, 24)
    assert speech_context.SPEECH_DATE["tillis_2025-01-23"] == date(2025, 1, 23)
    assert speech_context.SPEECH_DATE["warren_2025-04-29"] == date(2025, 4, 29)
