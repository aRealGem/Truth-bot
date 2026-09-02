"""Disclosure-frame visibility pins (Wave B follow-up).

The auxiliary blocks on a report page are default-collapsed `<details>`. That is
a readability decision, and it has a hard boundary: a *disclosure* may be
collapsed only if the disclosing sentence itself stays visible in the
`<summary>`. Anything that must be true whether or not the reader clicks
belongs above the fold of the frame, not inside it.

Three areas are governed here:

* **Genre note** (M-6, `docs/standing-rules.md`) -- NOT a frame. Owner-ratified
  2026-08-24 (Fable ruling 2026-08-23): the note is rate-based, renders only on
  the speech with the strictly highest beyond-public-record rate, and both of
  its sentences stay fully visible OUTSIDE any `<details>`.

  An earlier revision made the note the `<summary>` of a collapsed frame, on the
  reasoning that a summary is visible without a click. The ruling supersedes
  that: a `<summary>` is still a descendant of `<details>`, and the requirement
  is the note in the open, not merely reachable. The pins below assert the
  stronger property.

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


# The exact ratified note, at the presidential class (5 speeches, trump_2026
# top at 17/182 = 9.3%, median 4.5%). Sentence 1 is owner-ratified as a
# TEMPLATE (FR-0901-02) so one ratified sentence serves every class; the class
# label is a substitution. Sentence 2 is verbatim and must not be edited.
EXPECTED_GENRE_NOTE = (
    "Of this speech's 182 checked claims, 17 (9.3%) were recorded as beyond "
    "the public record — the highest rate of the five presidential addresses "
    "checked (median 4.5%). That concentration is a property of the speech's "
    "rhetorical genre — personal stories, intentions, and unmeasured "
    "superlatives — not a finding about the speaker."
)

TOP_RATE_REPORT = "2026-02-24-donald-trump-583aca.html"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_genre_note_renders_exact_ratified_string_on_top_rate_speech() -> None:
    """(a) The note on trump_2026 is the ratified copy, byte for byte."""
    target = [(p, d) for p, d in _docs() if p.name == TOP_RATE_REPORT]
    assert target, f"{TOP_RATE_REPORT} not rendered"
    p, doc = target[0]
    notes = doc.find_class("vp-genre-note")
    assert len(notes) == 1, f"{p.name}: expected exactly one genre note"
    got = " ".join(notes[0].text_content().split())
    assert got == EXPECTED_GENRE_NOTE, (
        f"\n  expected: {EXPECTED_GENRE_NOTE}\n  got:      {got}"
    )


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_genre_note_absent_on_every_other_report() -> None:
    """(b) Only the highest-rate speech carries the note."""
    carriers = [p.name for p, doc in _docs() if doc.find_class("vp-genre-note")]
    assert carriers == [TOP_RATE_REPORT], (
        f"genre note rendered on unexpected reports: {carriers}"
    )


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_genre_note_is_not_inside_any_details() -> None:
    """(c) M-6 hard constraint: both sentences fully visible, outside <details>.

    A previous revision made this note the <summary> of a collapsed frame. A
    <summary> is a descendant of <details>, so that arrangement fails here --
    which is the point: the ruling requires the note in the open, not merely
    reachable."""
    for p, doc in _docs():
        for el in doc.find_class("vp-genre-note"):
            ancestors = [a.tag for a in el.iterancestors()]
            assert "details" not in ancestors, (
                f"{p.name}: the genre note is inside a <details> "
                f"(ancestors: {ancestors[:6]})"
            )
            assert "summary" not in ancestors, f"{p.name}: note is in a <summary>"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_genre_note_uses_no_inferential_statistics_vocabulary() -> None:
    """Vocabulary guard. With n=5 the maximum possible |z| is
    (n-1)/sqrt(n) = 1.789, so a 2-sigma claim could never fire and a 1-sigma
    claim selects one speech regardless of the data. The note states a RANK,
    which is what this sample size supports."""
    banned = ("standard deviation", "σ", "sigma", "z-score", "outlier")
    for p, doc in _docs():
        for el in doc.find_class("vp-genre-note"):
            text = el.text_content().lower()
            for word in banned:
                assert word not in text, f"{p.name}: note uses {word!r}"


# Byte-level pin of the ratified copy. The literal above is the readable
# assertion; these two are the tamper guard. The string carries three U+2014 em
# dashes, and an em dash is exactly what a latin-1 round-trip, a copy-paste
# through a terminal, or a well-meaning "fix the punctuation" edit silently
# mangles into something that still LOOKS right in a diff.
EXPECTED_GENRE_NOTE_SHA256 = (
    "25d2804037eaf0813158bcfc234e88a15680a17121893fab688af62d0b3119fd")
# 312 -> 326 (FR-0901-02): the class label "presidential addresses" replaced
# the bare "speeches" when the rate statistic became class-partitioned. The
# length moved because the ratified copy gained a substitution, not because
# anyone edited the prose.
EXPECTED_GENRE_NOTE_LEN = 326

#: The ratified pieces pinned at SOURCE, independent of any rendered page.
#: Sentence 2's hash is the one that must never move.
EXPECTED_GENRE_S1_TEMPLATE_SHA256 = (
    "42b388b365b9b48bf71ac1df1e25c81139dd3aa50f8c1814c15c8b01f056fca4")
EXPECTED_GENRE_S2_SHA256 = (
    "5bd503bad02117b631dfe434686b121ccdca4181bd10a34afef007edf621ad66")


def test_ratified_genre_sentences_are_pinned_at_source() -> None:
    """Pins the ratified copy itself, not one rendering of it. Runs even when
    site-pca is absent, so a copy edit cannot slip through on an unrendered
    tree."""
    import hashlib

    from truthbot.publish.site import (_GENRE_NOTE_S1_TEMPLATE,
                                       _GENRE_NOTE_S2)
    assert hashlib.sha256(
        _GENRE_NOTE_S1_TEMPLATE.encode("utf-8")).hexdigest() == (
        EXPECTED_GENRE_S1_TEMPLATE_SHA256), "sentence 1 template was edited"
    assert hashlib.sha256(_GENRE_NOTE_S2.encode("utf-8")).hexdigest() == (
        EXPECTED_GENRE_S2_SHA256), "sentence 2 is VERBATIM and was edited"
    assert _GENRE_NOTE_S2.count("—") == 2
    assert _GENRE_NOTE_S1_TEMPLATE.count("—") == 1


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_genre_note_matches_ratified_hash_and_length() -> None:
    """(b, second form) Pin the rendered note on bytes, not on appearance."""
    import hashlib

    target = [(p, d) for p, d in _docs() if p.name == TOP_RATE_REPORT]
    assert target, f"{TOP_RATE_REPORT} not rendered"
    _p, doc = target[0]
    got = " ".join(doc.find_class("vp-genre-note")[0].text_content().split())
    assert len(got) == EXPECTED_GENRE_NOTE_LEN, (
        f"length {len(got)} != {EXPECTED_GENRE_NOTE_LEN}; "
        f"non-ascii present: {sorted({hex(ord(c)) for c in got if ord(c) > 127})}"
    )
    assert hashlib.sha256(got.encode("utf-8")).hexdigest() == (
        EXPECTED_GENRE_NOTE_SHA256), "rendered note does not match ratified bytes"
    # The three em dashes are load-bearing for the hash; name them so a failure
    # points at the cause instead of at the hash.
    assert got.count("—") == 3, f"expected 3 em dashes, found {got.count(chr(0x2014))}"


#: Inferential-statistics vocabulary the SITE may not use in its own voice.
_BANNED_PROSE = ("SD", "sigma", "standard deviation", "z-score", "outlier",
                 "statistically")


def _own_prose(doc) -> str:
    """Text the site asserts in its OWN voice.

    Two exclusions, both load-bearing, both learned from real hits on the
    published corpus:

    * **URL-bearing material.** The Trump report cites a truncated
      congress.gov path ("...-20260121-SD...") twice. A bare \\bSD\\b over the
      page matches those. A citation is not a copy defect.
    * **Quoted evidence snippets.** The Clinton report quotes an ONDCP source
      reading "household-survey rates statistically unchanged from 1996". That
      is the GOVERNMENT's word inside a quotation. Banning it there would
      force truth-bot to misquote its own evidence, which is a worse failure
      than the one the rule guards against. The rule governs what the site
      concludes, not what its sources said.
    """
    import copy
    import re

    d = copy.deepcopy(doc)
    for el in d.xpath("//a | //script | //style | //code | //pre"
                      " | //*[contains(@class, 'source-snippet')]"
                      " | //*[contains(@class, 'evidence-list')]"
                      " | //blockquote"):
        el.drop_tree()
    text = d.text_content()
    text = re.sub(r"\b(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"\b\S+\.(?:gov|com|org|net|edu)\S*", " ", text)
    return text


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_report_prose_uses_no_inferential_statistics_vocabulary() -> None:
    """(c, second half) The ban covers the page's own prose, not just the note.

    Wider than the note-scoped guard above: the ruling is about what the SITE
    asserts, so a sigma claim smuggled into a neighbouring paragraph breaches
    it just as surely as one inside the note.
    """
    import re

    for p, doc in _docs():
        prose = _own_prose(doc)
        for word in _BANNED_PROSE:
            assert not re.search(rf"\b{re.escape(word)}\b", prose, re.I), (
                f"{p.name}: banned inferential-statistics vocabulary {word!r} "
                f"in the site's own prose")


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_vocabulary_ban_does_not_police_quoted_evidence() -> None:
    """Pins the exclusion above, so nobody 'tightens' the ban onto quotations.

    Without this, a future widening of the scope would start failing on the
    ONDCP snippet and the tempting fix would be to edit the evidence text.
    """
    import re

    hits = [p.name for p, doc in _docs()
            if re.search(r"\bstatistically\b", doc.text_content(), re.I)]
    assert hits, ("expected at least one quoted source to use banned "
                  "vocabulary; if the corpus changed, re-point this pin")


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_frame_titles_are_plain_english_and_parallel() -> None:
    """The two "why" frames are worded as a matched set, in plain English.

    "Why some evidence carried no stance" is deliberately NOT "Where evidence
    coverage falls short": 4 of the 5 published speeches sit UNDER the
    stance-null ceiling, so a shortfall title would assert something the
    measurement contradicts, and would contradict this block's own body copy
    ("a share of null stance is expected and is not a retrieval failure").

    Asserted against RENDERED text, not the module source -- the source also
    carries the comment explaining why the rejected wording was rejected, and a
    source-level scan cannot tell copy from commentary about copy."""
    seen = undecided_frames = stance_frames = 0
    for p, doc in _docs():
        # Statement-triage pages carry no verdict panel and none of these
        # frames -- they are not in scope for this pin.
        if not doc.find_class("verdict-panel"):
            continue
        seen += 1
        text = doc.text_content()
        # PRESENCE is conditional, WORDING is not. A report with no honest
        # abstentions to decompose renders no undecided frame at all -- true of
        # the Senate floor speeches, and not a defect. What this test pins is
        # that wherever the frames DO appear they carry the ratified parallel
        # titles, and that the rejected wording never ships anywhere.
        undecided_frames += "Why some claims are undecided" in text
        stance_frames += "Why some evidence carried no stance" in text
        # Scoped to the site's OWN chrome, not the page text. A speech may say
        # "falls short" -- Warren's does, about an ethics pledge -- and a
        # whole-page scan would police the transcript instead of the UI. Only
        # the frame titles are the site speaking in its own voice.
        chrome = " ".join(
            " ".join(el.text_content().split())
            for el in list(doc.iter("summary"))
            + doc.find_class("stance-coverage-label"))
        assert "falls short" not in chrome, (
            f"{p.name}: rejected wording shipped in a frame title")
        assert "Why undecided" not in chrome.replace(
            "Why some claims are undecided", ""), f"{p.name}: old title survives"
    assert seen, "no full report pages found to check"
    # ...and the pin stays live: both titles must actually be in the corpus,
    # or this test would pass vacuously on a site that rendered neither.
    assert undecided_frames, "no page carries the undecided frame -- pin is dead"
    assert stance_frames, "no page carries the stance frame -- pin is dead"


def test_genre_note_shares_the_frame_rail_without_looking_clickable() -> None:
    """The always-open note sits on the same text rail as the collapsible
    frames, so it does not read as a rendering slip -- but it must NOT carry
    the ▶ marker, which would advertise a control that does not exist."""
    from truthbot.publish.site import CSS

    block = CSS.split(".vp-genre-note {", 1)[1].split("}", 1)[0]
    assert "padding-left" in block, "note no longer aligns to the frames' rail"
    assert "border-left" in block, "note lost its always-open marker rule"
    # No marker pseudo-element anywhere for this class.
    assert ".vp-genre-note::before" not in CSS
    assert ".vp-genre-note::marker" not in CSS
    # And it is never wired into the rotating-marker selector list.
    assert "> .vp-genre-note::before" not in CSS


def test_disclosure_markers_animate_to_their_open_position() -> None:
    """Every collapsible frame's triangle rotates on a transition rather than
    snapping. Two frames already did this; the rest were inconsistent."""
    from truthbot.publish.site import CSS

    markers = (
        ".vp-abstention-summary::before",
        ".vp-anecdote-summary::before",
        ".pca-provenance-summary::before",
        ".report-correction-summary::before",
        ".stance-coverage-summary .stance-coverage-label::before",
    )
    for sel in markers:
        assert sel in CSS, f"{sel} lost its marker rule"
        assert f"details[open] > {sel}" in CSS, (
            f"{sel} does not rotate when its frame is open"
        )
    assert "transition: transform 200ms ease;" in CSS
    # Motion is opt-out for readers who ask for less of it.
    assert "@media (prefers-reduced-motion: reduce)" in CSS


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
            assert "Why some evidence carried no stance" in summary
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
