"""Audio-unlock contract pins.

Round 4 fix: the prior ``getCtx() / audioCtx.resume()`` pattern was
fire-and-forget — Safari (and some Chrome variants) left the AudioContext
in ``suspended`` state at oscillator-schedule time, so the report-page
mascot was silent. The Round 4 rewrite returns a Promise from
``unlockAudio()`` and only schedules play after it resolves.

These tests pin the published JS source so a future cleanup can't
silently regress the resume/await pattern. We don't run the JS — these
are string-contract pins (matching the style of the existing
``test_embedded_js_contains_truthy_mute_storage_key`` test).
"""

from __future__ import annotations


def test_embedded_js_uses_promise_returning_unlock_audio() -> None:
    """The JS embedded in published reports must define
    ``unlockAudio()`` returning a Promise that resolves *after*
    ``audioCtx.resume()`` resolves. The play functions consume the
    resolved ctx — they must NOT call a synchronous ``getCtx()``."""
    from truthbot.publish.site import JS

    # The legacy synchronous helper must be gone.
    assert "function getCtx()" not in JS, (
        "Synchronous getCtx() helper resurrected — Safari race window "
        "will reopen. Use unlockAudio() (returns Promise) instead."
    )

    # New helper present.
    assert "function unlockAudio()" in JS

    # ``audioCtx.resume()`` is now consumed via .then() (Promise chain),
    # not fire-and-forget. The exact pattern: ``var p = audioCtx.resume()``
    # followed by ``p.then(``.
    assert "var p = audioCtx.resume();" in JS
    assert "p.then(" in JS

    # speak() awaits unlock before scheduling oscillators (deferred one
    # microtask so AudioContext state has flushed on Safari).
    assert "unlockAudio().then(function(ctx)" in JS
    assert "queueMicrotask(function() { fn(ctx);" in JS
    assert "queueMicrotask(function() { fn(ctx);" in JS


def test_standalone_truthbot_js_has_no_lens_toggle() -> None:
    """Standalone asset must mirror the embedded JS: the editorial-lens
    toggle was removed (remediation v2, 1.8 / DC-4') and must not linger
    in the package-dir mirror either."""
    from pathlib import Path

    js_path = Path(__file__).resolve().parents[1] / "src" / "truthbot" / "publish" / "assets" / "truthbot.js"
    js_src = js_path.read_text(encoding="utf-8")
    assert "DEFAULT_LENS" not in js_src
    assert "editorial-lens" not in js_src
def test_embedded_js_queues_pointerdown_for_first_gesture_autoplay() -> None:
    """``pointerdown`` fires before the subsequent ``click``, which
    matters when the user's first interaction is a navigation link.
    Without it, the AudioContext unlock + oscillator schedule lose
    the race against page navigation."""
    from truthbot.publish.site import JS

    assert "QUEUE_EVENTS" in JS
    assert "'pointerdown'" in JS
    # All four events stay in the queue list — pointerdown joins the
    # original three (click, keydown, touchstart). Pin the four-event
    # contract here so a refactor can't drop any of them.
    for evt in ("'pointerdown'", "'click'", "'keydown'", "'touchstart'"):
        assert evt in JS, f"Queued autoplay missing {evt}"


def test_standalone_truthbot_js_mirrors_unlock_audio_contract() -> None:
    """The standalone asset must match the embedded JS — they ship to
    different surfaces (report pages vs. the mascot fun page) and must
    not drift."""
    from pathlib import Path

    js_path = Path(__file__).resolve().parents[1] / "src" / "truthbot" / "publish" / "assets" / "truthbot.js"
    js_src = js_path.read_text(encoding="utf-8")
    assert "function unlockAudio()" in js_src
    assert "function getCtx()" not in js_src
    assert "unlockAudio().then(function(ctx)" in js_src
    assert "queueMicrotask(function() { fn(ctx);" in js_src
    assert "QUEUE_EVENTS" in js_src
    assert "'pointerdown'" in js_src


def test_play_helpers_accept_pre_unlocked_ctx_argument() -> None:
    """``playHappy/playConfused/playSad`` now take ``ctx`` as a
    parameter (the resolved AudioContext from unlockAudio). The
    earlier signature was zero-arg — we pin the new shape so a refactor
    can't silently revert to the racy fire-and-forget version."""
    from truthbot.publish.site import JS

    assert "function playHappy(ctx)" in JS
    assert "function playConfused(ctx)" in JS
    assert "function playSad(ctx)" in JS
