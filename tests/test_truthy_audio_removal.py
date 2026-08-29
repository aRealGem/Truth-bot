"""Report-page Truthy audio/badge removal contract pins.

History: this file used to pin an audio *unlock* contract (a Promise-returning
``unlockAudio()`` that dodged a Safari autoplay race). That whole subsystem is
gone from report pages.

Why it went: the only visible affordance for the audio was a speaker-icon badge
(``.truthy-tap-hint``) absolutely positioned over the mascot, and that badge
rendered full-size on iOS Safari — covering Truthy instead of sitting in the
corner. The badge *was* the mute control, so removing it without removing the
audio would have left the site able to make noise with no way to silence it.
Both went together, and with nothing left to activate the mascot stopped being
a ``role="button"`` and became a labelled ``role="img"``.

The playground at ``truthy.html`` is untouched — it has its own independent
implementation (``_TRUTHY_FUN_SCRIPT``) and the shared ``JS`` init() no-ops
there, because that page has no ``#truthy-mascot-widget``.

These are string-contract pins (we don't execute the JS) guarding the removal
against a well-meaning future re-add.
"""

from __future__ import annotations

from pathlib import Path

_ASSET = Path(__file__).resolve().parents[1] / "src" / "truthbot" / "publish" / "assets" / "truthbot.js"


def test_report_page_js_has_no_web_audio() -> None:
    """No AudioContext, no oscillators, no play helpers in the shared JS."""
    from truthbot.publish.site import JS

    for token in (
        "AudioContext",
        "webkitAudioContext",
        "createOscillator",
        "function unlockAudio()",
        "function getCtx()",
        "function playHappy",
        "function playConfused",
        "function playSad",
        "function speak()",
        "soundMap",
    ):
        assert token not in JS, (
            f"Report-page audio resurrected via {token!r}. The speaker badge that "
            "controlled it is gone; re-adding sound here would leave no way to mute it."
        )


def test_report_page_js_has_no_mute_state_or_queued_autoplay() -> None:
    """The localStorage mute flag and first-gesture autoplay are gone."""
    from truthbot.publish.site import JS

    for token in (
        "truthy-mute",
        "TRUTHY_MUTE_KEY",
        "DEFAULT_TRUTHY_MUTE",
        "QUEUE_EVENTS",
        "queuedHandler",
        "isTruthyFunPage",
        "data-mute",
    ):
        assert token not in JS, f"Mute/autoplay machinery resurrected via {token!r}"


def test_no_tap_hint_badge_anywhere() -> None:
    """The speaker badge is gone from the markup helper, the CSS and the JS."""
    import truthbot.publish.site as site

    assert not hasattr(site, "_TRUTHY_TAP_HINT"), (
        "_TRUTHY_TAP_HINT is back — this badge rendered oversized on iOS Safari "
        "and covered the mascot."
    )
    assert "truthy-tap-hint" not in site.JS
    assert "tap-hint-label" not in site.JS
    assert "tap-hint-label" not in site.CSS
    # The CSS keeps one prose mention in an explanatory comment; pin that no
    # actual rule targets the class.
    assert ".truthy-tap-hint {" not in site.CSS


def test_report_mascot_is_not_an_interactive_control() -> None:
    """Truthy is presentational on report pages: labelled role="img", not a
    focusable button whose activation does nothing."""
    import truthbot.publish.site as site

    assert "onMascotActivate" not in site.JS
    assert "widget.addEventListener('click'" not in site.JS
    assert "widget.addEventListener('keydown'" not in site.JS


def test_standalone_truthbot_js_is_byte_identical_to_embedded_js() -> None:
    """The shipped asset is written from the ``JS`` constant by the publisher,
    so the committed package-dir copy must match it exactly. Byte equality is a
    stronger no-drift pin than the token-by-token checks this replaced."""
    from truthbot.publish.site import JS

    assert _ASSET.read_text(encoding="utf-8") == JS, (
        "src/truthbot/publish/assets/truthbot.js has drifted from the JS constant. "
        "Regenerate it: Path(asset).write_text(JS, encoding='utf-8')."
    )


def test_standalone_truthbot_js_has_no_lens_toggle() -> None:
    """Pre-existing invariant, kept: the editorial-lens toggle was removed
    (remediation v2, 1.8 / DC-4') and must not linger in the mirror either."""
    js_src = _ASSET.read_text(encoding="utf-8")
    assert "DEFAULT_LENS" not in js_src
    assert "editorial-lens" not in js_src


def test_truthy_fun_page_keeps_its_own_audio() -> None:
    """The easter egg at truthy.html is deliberately NOT part of this removal —
    it is opt-in, users navigate there on purpose, and it is not where the
    oversized-badge bug appeared."""
    from truthbot.publish.site import _TRUTHY_FUN_SCRIPT

    assert "createOscillator" in _TRUTHY_FUN_SCRIPT
    assert "AudioContext" in _TRUTHY_FUN_SCRIPT
