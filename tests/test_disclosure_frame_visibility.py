"""Disclosure-frame visibility pins (Wave B follow-up).

The auxiliary blocks on a report page are default-collapsed `<details>`. That is
a readability decision, and it has a hard boundary: a *disclosure* may be
collapsed only if the disclosing sentence itself stays visible in the
`<summary>`. Anything that must be true whether or not the reader clicks
belongs above the fold of the frame, not inside it.

Three frames are governed here:

* **Genre note** (M-6, `docs/standing-rules.md`) -- the rule requires a
  genre-driven concentration to be "measured and disclosed rather than silently
  shipped". The subtle failure is partial: surfacing the *count* ("17 of the
  corpus's 33 claims fall on this speech") while collapsing the *framing*
  ("that concentration is a property of the speech's rhetorical genre ... not a
  finding about the speaker") would leave the concentration reading as a finding
  about the speaker -- the exact outcome M-6 exists to prevent. So the whole
  genre line is pinned into the summary, not merely its opening sentence.

* **Corrections banner** -- that verdicts were revised, and how many, stays
  visible. Only the mechanics collapse.

* **Evidence coverage** -- the measured stance-null rate against the ceiling
  stays visible, and the one speech published under an owner-ratified exception
  names that exception in its summary too ("shown, not hidden").

These parse the rendered HTML rather than string-matching, because the property
under test is structural: *which side of the `<summary>` boundary* the text
falls on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lxml import html as LH

REPORTS = sorted((Path(__file__).resolve().parents[1] / "site-pca" / "reports").glob("*.html"))


def _docs():
    for p in REPORTS:
        yield p, LH.fromstring(p.read_text(encoding="utf-8"))


def _summary_text(details_el) -> str:
    s = details_el.find("summary")
    return " ".join(s.text_content().split()) if s is not None else ""


def _collapsed_text(details_el) -> str:
    """Everything inside the <details> that is NOT the <summary>."""
    return " ".join(
        " ".join(child.text_content().split())
        for child in details_el
        if child.tag != "summary"
    )


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_every_disclosure_frame_is_default_closed() -> None:
    """The frames are collapsed; none ships with `open`."""
    for p, doc in _docs():
        for el in doc.iter("details"):
            assert el.get("open") is None, f"{p.name}: a <details> ships open"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_genre_note_disclosure_is_fully_inside_the_summary() -> None:
    """M-6: both the concentration count AND the genre-property framing are
    visible without opening the frame."""
    seen = 0
    for p, doc in _docs():
        for el in doc.find_class("vp-genre-details"):
            seen += 1
            summary = _summary_text(el)
            collapsed = _collapsed_text(el)
            assert "beyond the public record fall on this speech" in summary, (
                f"{p.name}: genre concentration is not visible in the summary"
            )
            # When the framing sentence is emitted at all it must ride in the
            # summary next to the count -- never on the collapsed side.
            assert "not a finding about the speaker" not in collapsed, (
                f"{p.name}: the genre-property framing was collapsed while the "
                "concentration count stayed visible. That inverts M-6."
            )
            if "That concentration is a property" in el.text_content():
                assert "not a finding about the speaker" in summary
    assert seen, "no genre disclosure frames found in the rendered reports"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_corrections_headline_is_visible_without_opening() -> None:
    """A reader who never clicks still learns verdicts were revised."""
    seen = 0
    for p, doc in _docs():
        for el in doc.find_class("report-correction-banner"):
            seen += 1
            summary = _summary_text(el)
            assert "Corrections applied" in summary, (
                f"{p.name}: corrections headline is not visible in the summary"
            )
            assert "revised" in summary
    assert seen, "no correction banners found in the rendered reports"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_evidence_coverage_rate_and_exception_are_visible() -> None:
    """The measured rate is in the summary, and the over-ceiling speech names
    its exception there too."""
    seen = 0
    for p, doc in _docs():
        for el in doc.find_class("stance-coverage"):
            seen += 1
            summary = _summary_text(el)
            assert "Evidence coverage" in summary
            assert "carried no stance" in summary, (
                f"{p.name}: stance-null rate is not visible in the summary"
            )
            full = el.text_content()
            if "Published under an exception" in full:
                assert "published under an exception" in summary.lower(), (
                    f"{p.name}: this report is published under a ratified "
                    "exception but the summary does not say so -- that is the "
                    "one notice that must not need a click."
                )
    assert seen, "no evidence-coverage frames found in the rendered reports"
