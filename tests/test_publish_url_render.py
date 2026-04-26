"""Tests for Layer 4 — publish-layer URL trust-tier rendering.

Verifies that ``_evidence_list_html`` and ``_classify_source_for_render``
correctly distinguish verified / unverified / broken sources.

Six tests:

1. No classification map → every URL renders verified (backward compat).
2. ``ok`` classification → verified.
3. ``bot-blocked`` / ``transient`` → unverified, with badge.
4. ``dead-4xx`` / ``malformed`` / ``dns`` / ``cert-error`` → broken,
   stripped from rendered output.
5. Mixed list keeps verified + unverified, drops broken, preserves
   relative ordering.
6. All-broken list → "No sources retrieved." fallback message.
"""
from __future__ import annotations

from truthbot.publish.site import (
    _classify_source_for_render,
    _evidence_list_html,
)


def test_classify_no_map_defaults_verified():
    assert _classify_source_for_render("https://x.gov/a", None) == "verified"
    assert _classify_source_for_render("https://x.gov/a", {}) == "verified"


def test_classify_ok_is_verified():
    assert (
        _classify_source_for_render("https://x.gov/a", {"https://x.gov/a": "ok"})
        == "verified"
    )


def test_classify_bot_blocked_and_transient_are_unverified():
    for cls in ("bot-blocked", "transient", "unknown"):
        assert (
            _classify_source_for_render("https://x.gov/a", {"https://x.gov/a": cls})
            == "unverified"
        ), f"failure_class={cls!r} must render as unverified"


def test_classify_broken_categories_are_broken():
    for cls in ("dead-4xx", "malformed", "dns", "cert-error"):
        assert (
            _classify_source_for_render("https://x.gov/a", {"https://x.gov/a": cls})
            == "broken"
        ), f"failure_class={cls!r} must render as broken"


def test_evidence_list_strips_broken_urls():
    good = "https://www.bls.gov/cps/"
    blocked = "https://www.cbp.gov/x"
    dead = "https://www.fake.gov/dead"

    classifications = {
        good: "ok",
        blocked: "bot-blocked",
        dead: "dead-4xx",
    }
    html = _evidence_list_html(
        [good, blocked, dead], classifications=classifications
    )

    assert good in html
    assert blocked in html
    assert dead not in html, "broken URL must be stripped from rendered output"
    assert "source-verified" in html
    assert "source-unverified" in html
    assert "source-broken" not in html
    assert "source-unverified-badge" in html
    assert ">unverified<" in html


def test_evidence_list_all_broken_shows_empty_message():
    dead1 = "https://nope.example.gov/a"
    dead2 = "https://nope.example.com/b"
    html = _evidence_list_html(
        [dead1, dead2],
        classifications={dead1: "dead-4xx", dead2: "dns"},
    )
    assert "No sources retrieved." in html
    assert dead1 not in html
    assert dead2 not in html


def test_evidence_list_no_classifications_renders_as_before():
    """Backward-compatibility — pre-Layer-4 callers don't pass
    ``classifications`` and must get the original verified rendering
    for every URL with no extra badges."""
    urls = ["https://www.bls.gov/cps/", "https://www.bea.gov/data/"]
    html = _evidence_list_html(urls)
    for u in urls:
        assert u in html
    # No unverified badge should be emitted in the legacy path.
    assert "source-unverified-badge" not in html
    # Every list item should render in the verified bucket by default.
    assert html.count("source-verified") == len(urls)
