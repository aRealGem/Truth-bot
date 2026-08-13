"""Tests for Layer 4 — publish-layer URL trust-tier rendering.

Verifies that ``_evidence_list_html`` and ``_classify_source_for_render``
correctly distinguish verified / unverified / broken sources.

Six tests:

1. No classification map → every URL renders UNVERIFIED. This REVERSED in
   wave 2. It previously rendered verified, for backward compatibility with
   pre-Layer-4 reports — and that compatibility is precisely what let a URL
   returning 404 on both FRED and ALFRED wear the source-verified badge on the
   published site, because the run artifact carried no classification map and
   every branch defaulted to verified.
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


def test_classify_no_map_fails_closed_to_unverified():
    """Wave 2: absence of a record is not evidence of verification."""
    assert _classify_source_for_render("https://x.gov/a", None) == "unverified"
    assert _classify_source_for_render("https://x.gov/a", {}) == "unverified"


def test_classify_url_absent_from_an_existing_map_fails_closed():
    """The second fail-open branch: a map exists, this URL is not in it.

    This is the branch that would have caught LNS12000000 had a map been
    present, so it has to fail closed too — otherwise adding classifications
    for *some* URLs silently vouches for the rest."""
    m = {"https://other.gov/x": "ok"}
    assert _classify_source_for_render("https://x.gov/a", m) == "unverified"


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


def test_evidence_list_no_classifications_fails_closed():
    """Wave 2 reversal, stated as the behaviour change it is.

    A caller that passes no ``classifications`` used to get every URL rendered
    verified. It now gets every URL rendered unverified. The URLs are still
    SHOWN — failing closed means declining to vouch for them, not hiding them,
    since a reader can still follow a citation we could not confirm."""
    urls = ["https://www.bls.gov/cps/", "https://www.bea.gov/data/"]
    html = _evidence_list_html(urls)
    for u in urls:
        assert u in html, "failing closed must not drop the citation"
    assert html.count('class="source-unverified"') == len(urls)
    assert 'class="source-verified"' not in html


def test_unverified_tooltip_does_not_vouch_for_an_unchecked_url():
    """Failing closed must not relocate the over-claim into the tooltip.

    The badge copy asserts a URL is "most likely real" — true of one we
    checked and found bot-blocked, and unsupported for one we never checked
    at all."""
    unchecked = _evidence_list_html(["https://www.bls.gov/cps/"])
    assert "not checked at publish time" in unchecked.lower()
    assert "most likely real" not in unchecked

    checked = _evidence_list_html(["https://www.bls.gov/cps/"],
                                  classifications={"https://www.bls.gov/cps/":
                                                   "bot-blocked"})
    assert "most likely real" in checked
