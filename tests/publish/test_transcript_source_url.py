"""Authored transcript URLs must actually reach the page (FR-0901-02).

The artifacts carry no meta.source_url, so before this wiring the Congressional
Record URLs in data/report_events.json were recorded and never rendered.
"""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
REPORTS = REPO / "site-pca" / "reports"

_spec = importlib.util.spec_from_file_location(
    "rerender_pca_site", REPO / "scripts" / "rerender_pca_site.py")
_rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr)

PUBLISHED_SENATE = {
    "2026-06-24-ted-cruz-425daf.html":
        "CREC-2026-06-24-pt1-PgS3177.htm",
    "2025-04-29-elizabeth-warren-9537ac.html":
        "CREC-2025-04-29-pt1-PgS2644-2.htm",
}
#: The transcript-source row's own marker -- distinct from an evidence citation.
TRANSCRIPT_LABEL = '<span class="lab">Transcript:</span>'

PRESIDENTIAL = ["1998-01-27-bill-clinton-54f0ca.html",
                "2006-01-31-george-w-bush-82f462.html",
                "2014-01-28-barack-obama-bc9c9f.html",
                "2022-03-01-joe-biden-d359c0.html",
                "2026-02-24-donald-trump-583aca.html"]


def test_authored_url_is_returned_for_a_senate_speech():
    url = _rr._transcript_url("warren_2025-04-29")
    assert url.startswith("https://www.govinfo.gov/"), url
    assert "CREC-2025-04-29" in url


def test_a_speech_with_no_authored_url_gets_empty_string():
    """Absent means absent -- the page omits the link rather than guessing."""
    assert _rr._transcript_url("trump_2026") == ""
    assert _rr._transcript_url("") == ""
    assert _rr._transcript_url("nobody_1999") == ""


@pytest.mark.skipif(not REPORTS.exists(), reason="site-pca not rendered")
@pytest.mark.parametrize("page,granule", sorted(PUBLISHED_SENATE.items()))
def test_published_senate_pages_render_the_link(page, granule):
    html = (REPORTS / page).read_text(encoding="utf-8")
    assert TRANSCRIPT_LABEL in html, f"{page}: no Transcript row"
    row = html.split(TRANSCRIPT_LABEL, 1)[1][:400]
    assert granule in row, f"{page}: Transcript row does not link the granule"


@pytest.mark.skipif(not REPORTS.exists(), reason="site-pca not rendered")
@pytest.mark.parametrize("page", PRESIDENTIAL)
def test_presidential_pages_gain_no_transcript_link(page):
    """The wiring must be inert for every speech with no authored URL.

    Asserted on the Transcript ROW, not on the substring "CREC": presidential
    pages legitimately cite the Congressional Record as EVIDENCE, and a bare
    substring scan would confuse a citation with a transcript source.
    """
    assert TRANSCRIPT_LABEL not in (
        REPORTS / page).read_text(encoding="utf-8")
