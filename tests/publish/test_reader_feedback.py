"""Reader-feedback link: fail-closed behaviour, prefill correctness, and the
constraints the published site must not break.

The feature is a plain ``<a>`` to a prefilled form. The tests that matter most
here are the ones asserting what it does NOT do: no page-load traffic, no
script, no form, and nothing at all rendered until the form is configured.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from lxml import html as LH

from truthbot.publish import site
from truthbot.publish.reader_feedback import (CLAIM_TEXT_LIMIT, FIELD_ORDER,
                                              ReaderFeedbackError, is_configured,
                                              is_publishable_id, load_config,
                                              prefill_url, truncate)

REPORTS = sorted(
    (Path(__file__).resolve().parents[2] / "site-pca" / "reports").glob("*.html"))

_FORM = "https://docs.google.com/forms/d/e/TESTFORM/viewform"
_CFG = {"form_url": _FORM,
        "entries": {"claim_url": "111", "claim_id": "222", "claim_text": "333",
                    "verdict": "444", "speaker": "555", "speech_date": "666"}}


@pytest.fixture
def configured(monkeypatch):
    """Turn the feature on for one test, bypassing the lru_cache."""
    monkeypatch.setattr(site, "_feedback_cfg", lambda: _CFG)
    return _CFG


# ── fail closed ──────────────────────────────────────────────────────────────

def test_missing_config_file_is_unconfigured_not_an_error(tmp_path):
    """An installed package with no data/ dir must not crash the renderer."""
    cfg = load_config(tmp_path / "nope.json")
    assert cfg == {"form_url": "", "entries": {}}
    assert not is_configured(cfg)
    assert prefill_url(cfg, claim_url="https://x") == ""


def test_shipped_config_is_wellformed():
    """The shipped config loads, and if it is live it points somewhere sane.

    An earlier version of this test asserted the config was UNCONFIGURED,
    which encoded a temporary state (ships dark, before the form existed) as a
    permanent invariant. It duly failed the moment the form was wired up. The
    durable property is well-formedness, not darkness: fail-closed behaviour is
    covered by the missing-file and empty-config tests above.
    """
    cfg = load_config(site._READER_FEEDBACK_PATH)
    if not is_configured(cfg):
        return                                   # dark is a legitimate state
    url = cfg["form_url"]
    assert url.startswith("https://"), "a feedback form must not be plain http"
    assert "docs.google.com/forms/" in url
    assert url.endswith("/viewform"), (
        "form_url holds everything up to /viewform; the query string is built")
    assert cfg["entries"]["claim_url"].isdigit(), (
        "claim_url must be a numeric Google Forms entry id")


def test_bad_schema_raises(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"schema": "bogus"}), encoding="utf-8")
    with pytest.raises(ReaderFeedbackError):
        load_config(p)


def test_unknown_entry_field_raises(tmp_path):
    """A typo'd field would silently never prefill; fail the build instead."""
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"schema": "truthbot-reader-feedback v1",
                             "entries": {"clam_url": "1"}}), encoding="utf-8")
    with pytest.raises(ReaderFeedbackError):
        load_config(p)


def test_form_url_without_claim_url_entry_is_unconfigured():
    """Without claim_url a response cannot be traced to a claim, which is the
    whole point, so a config that omits it counts as not configured."""
    assert not is_configured({"form_url": _FORM, "entries": {"verdict": "444"}})


def test_link_helper_emits_nothing_when_unconfigured(monkeypatch):
    monkeypatch.setattr(site, "_feedback_cfg", lambda: {"form_url": "", "entries": {}})
    assert site._feedback_link_html(
        cls="c", text="t", aria="a", claim_url="https://x") == ""


# ── prefill correctness ──────────────────────────────────────────────────────

def test_prefill_url_shape_and_values(configured):
    html = site._feedback_link_html(
        cls="claim-feedback-link", text="Something wrong? Welcome feedback!", aria="A",
        claim_url="https://s/claims/trump_2026-0010.html",
        claim_id="trump_2026-0010", claim_text="Unemployment is at a low.",
        verdict="MOSTLY TRUE", speaker="Donald Trump", speech_date="2026-02-24")
    a = LH.fromstring(html)
    assert a.get("target") == "_blank"
    assert a.get("rel") == "noopener"          # house convention, not noreferrer
    assert a.get("class") == "claim-feedback-link"
    href = a.get("href")
    assert href.startswith(f"{_FORM}?usp=pp_url&")
    q = parse_qs(urlsplit(href).query)
    assert q["entry.111"] == ["https://s/claims/trump_2026-0010.html"]
    assert q["entry.222"] == ["trump_2026-0010"]
    assert q["entry.333"] == ["Unemployment is at a low."]
    assert q["entry.444"] == ["MOSTLY TRUE"]
    assert q["entry.555"] == ["Donald Trump"]
    assert q["entry.666"] == ["2026-02-24"]


def test_href_is_percent_encoded_then_html_escaped(configured):
    """Pins the encoding ORDER. Escaping before percent-encoding would turn
    each separator into %26amp%3B and corrupt every field after the first."""
    html = site._feedback_link_html(
        cls="c", text="t", aria="a", claim_url="https://x", claim_id="y")
    assert "&amp;entry." in html
    assert not re.search(r"(?<!amp;)&entry\.", html.replace("&amp;", "AMP")) or True
    assert "&entry." not in html.replace("&amp;entry.", "")
    assert "%26amp%3B" not in html


def test_awkward_characters_round_trip(configured):
    nasty = 'He said "it—rose" & fell #1 50% <b> a/b'
    html = site._feedback_link_html(
        cls="c", text="t", aria="a", claim_url="https://x", claim_text=nasty)
    href = LH.fromstring(html).get("href")
    for token in ("%22", "%E2%80%94", "%26", "%23", "%20", "%3C", "%2F"):
        assert token in href, f"{token} missing from {href}"
    assert parse_qs(urlsplit(href).query)["entry.333"] == [nasty]


def test_claim_text_truncates_on_a_word_boundary(configured):
    long_claim = " ".join(["word"] * 2000)
    html = site._feedback_link_html(
        cls="c", text="t", aria="a", claim_url="https://x", claim_text=long_claim)
    href = LH.fromstring(html).get("href")
    got = parse_qs(urlsplit(href).query)["entry.333"][0]
    assert len(got) <= CLAIM_TEXT_LIMIT + 1      # +1 for the ellipsis
    assert got.endswith("…")
    assert not got[:-1].endswith(" ")            # cut fell on a word boundary
    assert len(href) < 2000, "URL must stay under the practical browser ceiling"


def test_truncate_leaves_short_text_untouched():
    assert truncate("short claim") == "short claim"
    assert truncate("  collapses   whitespace ") == "collapses whitespace"


def test_empty_values_are_omitted_not_sent_blank(configured):
    href = LH.fromstring(site._feedback_link_html(
        cls="c", text="t", aria="a", claim_url="https://x", verdict="")).get("href")
    q = parse_qs(urlsplit(href).query)
    assert "entry.444" not in q


def test_field_order_is_stable():
    """Query-string order must not depend on dict iteration: the rendered site
    is byte-reproducibility checked in CI."""
    a = prefill_url(_CFG, claim_url="u", claim_id="i", verdict="v")
    b = prefill_url(_CFG, verdict="v", claim_id="i", claim_url="u")
    assert a == b
    assert list(FIELD_ORDER)[0] == "claim_url"


# ── what it must NOT do ──────────────────────────────────────────────────────

def test_nothing_phones_home(configured):
    """The site makes no data requests. This feature must not be the exception:
    the control is a link, and a link submits nothing until it is followed."""
    html = site._feedback_link_html(
        cls="c", text="t", aria="a", claim_url="https://x")
    for forbidden in ("<form", "<script", "onclick", "fetch(", "XMLHttpRequest",
                      "sendBeacon", "<input", "<button"):
        assert forbidden not in html
    assert LH.fromstring(html).get("href").startswith("https://")


def test_no_reserved_class_names(configured):
    reserved = {"vp-abstention-chip", "vp-anecdote-note",
                "pca-provenance-summary", "vp-genre-note"}
    html = site._feedback_link_html(
        cls="claim-feedback-link", text="t", aria="a", claim_url="https://x")
    assert not any(r in html for r in reserved)


def test_copy_does_not_invite_a_vote_on_truth(configured):
    """The link asks about OUR check. Wording that reads as 'do you agree with
    this verdict' would turn a fact-check into a poll."""
    html = site._feedback_link_html(
        cls="c", text="Something wrong? Welcome feedback!", aria="a",
        claim_url="https://x")
    text = LH.fromstring(html).text_content().lower()
    for banned in ("do you agree", "vote", "was this fair", "rate this"):
        assert banned not in text


# ── committed tree ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_committed_site_matches_the_configured_state():
    """Written so it needs no edit when the form is switched on: it asserts the
    committed HTML agrees with whatever the committed config says."""
    on = is_configured(load_config(site._READER_FEEDBACK_PATH))
    for p in REPORTS:
        doc = LH.fromstring(p.read_text(encoding="utf-8"))
        links = doc.find_class("claim-feedback-link")
        claims = doc.xpath('//article[contains(@class, "claim")]')
        if on:
            assert len(links) == len(claims), f"{p.name}: one link per claim"
        else:
            assert links == [], f"{p.name}: feature is dark but a link rendered"
            assert "docs.google.com/forms" not in p.read_text(encoding="utf-8")


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_control_is_never_inside_a_details():
    """Complements the M-6 pins: a feedback link buried in a collapsed frame is
    a feedback link nobody uses."""
    for p in REPORTS:
        doc = LH.fromstring(p.read_text(encoding="utf-8"))
        assert doc.xpath(
            '//a[contains(@class, "feedback-link")]/ancestor::details') == []


# ── claim-id guard: no uuid may reach a reader-visible URL ───────────────────

@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_every_published_claim_id_is_speech_derived():
    """Pins the published corpus against the uuid default.

    ``Claim.id`` defaults to a uuid4 (models.py:147). A uuid reaching a
    rendered claim id would put an opaque, unresolvable identifier into a
    public URL. All 529 ids in the corpus are speech-derived; this fails the
    build if that ever stops being true.
    """
    claims_json = Path(__file__).resolve().parents[2] / "site-pca" / "data" / "claims.json"
    rows = json.loads(claims_json.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("claims", rows)
    bad = [r["id"] for r in rows if not is_publishable_id(r["id"])]
    assert bad == [], f"non-speech-derived claim id(s) reached the corpus: {bad[:5]}"

    pages = sorted((claims_json.parents[1] / "claims").glob("*.html"))
    bad_files = [p.stem for p in pages if not is_publishable_id(p.stem)]
    assert bad_files == [], f"non-speech-derived claim page(s): {bad_files[:5]}"


def test_uuid_claim_id_gets_no_feedback_link(configured):
    """A synthetic bundle must not emit a link to an unresolvable claim page."""
    assert not is_publishable_id("3f2b9c1a-4d5e-6f70-8a9b-0c1d2e3f4a5b")
    assert is_publishable_id("trump_2026-0010")
    # And the guard is wired into the renderer, not merely available.
    assert "_publishable_claim_id(claim.id)" in Path(
        site.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", [
    "3f2b9c1a-4d5e-6f70-8a9b-0c1d2e3f4a5b",   # uuid4
    "Trump_2026-0010",                        # uppercase speech id
    "trump_2026-10",                          # too few digits
    "trump_2026",                             # no claim index
    "", None,
])
def test_non_speech_derived_ids_are_rejected(bad):
    assert not is_publishable_id(bad)


# ── the report-level callout ─────────────────────────────────────────────────

def _full_reports():
    """Only the real report pages.

    ``reports/*.html`` also holds triage pages and the redirect stubs left
    behind by re-adjudication. Neither is a report and neither carries a
    callout; a claim article is what distinguishes the real thing.
    """
    for p in REPORTS:
        doc = LH.fromstring(p.read_text(encoding="utf-8"))
        if doc.xpath('//article[contains(@class, "claim")]'):
            yield p, doc

@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_report_carries_exactly_one_feedback_callout():
    """One visible ask per report, never per claim.

    Repeated under all 182 claims an invitation stops reading as openness and
    starts reading as engagement-farming, which is the wrong register for a
    fact-checker. The per-claim links stay quiet and carry the claim-specific
    case; this is the one that asks out loud.
    """
    on = is_configured(load_config(site._READER_FEEDBACK_PATH))
    seen = 0
    for p, doc in _full_reports():
        seen += 1
        callouts = doc.find_class("report-feedback-callout")
        assert len(callouts) == (1 if on else 0), f"{p.name}: {len(callouts)} callouts"
        if on:
            assert len(doc.find_class("report-feedback-link")) == 1
    # 5 presidential + warren_2025-04-29; cruz held (FR-0901-06) and
    # pruned from the site (FR-0901-10).
    assert seen == 6, f"expected the six published reports, walked {seen}"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_callout_sits_high_on_the_page_not_below_the_claims():
    """Placement, not styling, was what made the first version invisible.

    It originally rendered after the claims. On the Trump report that put it at
    99.8% of a 79,000-word document, below all 182 claims: present in the DOM,
    unreachable in practice. Colour cannot fix that, so the position is pinned
    -- it must precede the first claim, and sit in the first quarter of the
    page.
    """
    if not is_configured(load_config(site._READER_FEEDBACK_PATH)):
        pytest.skip("feature dark")
    for p, doc in _full_reports():
        raw = p.read_text(encoding="utf-8")
        at = raw.find("report-feedback-callout")
        first_claim = raw.find('<article class="claim"')
        assert at != -1, f"{p.name}: no callout"
        assert at < first_claim, (
            f"{p.name}: callout renders after the first claim; on a long report "
            f"that is the bottom of the page")
        pct = 100 * at / len(raw)
        assert pct < 25, f"{p.name}: callout at {pct:.1f}% of the document"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_callout_is_visible_not_collapsed():
    """It is the visible ask; behind a <details> it would not be one."""
    for p in REPORTS:
        doc = LH.fromstring(p.read_text(encoding="utf-8"))
        for el in doc.find_class("report-feedback-callout"):
            assert [a.tag for a in el.iterancestors() if a.tag == "details"] == []


def test_callout_copy_invites_correction_not_a_verdict_vote(configured):
    from truthbot.publish.site import _render_report  # noqa: F401  (import pin)

    lead = "Something wrong? Welcome feedback!"
    assert lead in Path(site.__file__).read_text(encoding="utf-8")
    body = Path(site.__file__).read_text(encoding="utf-8")
    for banned in ("do you agree", "vote on", "was this fair", "rate this verdict"):
        assert banned not in body.lower()


def test_claim_link_breaks_out_of_the_metadata_register():
    """.claim-foot is mono/uppercase/letterspaced -- the site's metadata voice.
    Inheriting it is exactly why the link read as a timestamp and went unseen."""
    block = site.CSS.split(".claim-feedback-link {", 1)[1].split("}", 1)[0]
    assert "text-transform: none" in block
    assert "letter-spacing: 0" in block
    assert "var(--sans)" in block
    assert "border:" in block, "the chip border is what makes it read as a control"


def test_callout_uses_the_note_accent_not_a_verdict_colour():
    """The tint must never be a verdict colour.

    An amber panel on a page full of amber "Exaggerated" pills would read as a
    verdict, and a request for feedback is not a finding about anything. The
    resemblance to --v-exaggerated makes this exactly the shortcut a later
    tidy-up would take, so it is pinned.
    """
    block = site.CSS.split(".report-feedback-callout {", 1)[1].split("}", 1)[0]
    assert "var(--note-bg)" in block
    assert "var(--note-border)" in block

    root = site.CSS.split(":root {", 1)[1].split("\n}", 1)[0]
    note = dict(re.findall(r"(--note-[\w-]+):\s*(#[0-9a-fA-F]{3,8})", root))
    assert set(note) == {"--note-bg", "--note-border"}, note
    verdicts = dict(re.findall(r"(--v-[\w-]+):\s*(#[0-9a-fA-F]{3,8})", root))
    assert verdicts, "verdict palette not found; the guard would pass vacuously"
    clash = {k: v for k, v in note.items()
             if v.lower() in {x.lower() for x in verdicts.values()}}
    assert not clash, f"note accent collides with the verdict palette: {clash}"


def test_note_surface_keeps_its_text_above_AA():
    """--ink-dim measures 4.45:1 on the note tint, so it must not appear here."""
    def _lum(h):
        h = h.lstrip("#")
        ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    root = site.CSS.split(":root {", 1)[1].split("\n}", 1)[0]
    tok = dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", root))
    bg = _lum(tok["--note-bg"])
    for name in ("--ink", "--ink-muted"):
        fg = _lum(tok[name])
        hi, lo = max(fg, bg), min(fg, bg)
        assert (hi + 0.05) / (lo + 0.05) >= 4.5, f"{name} below AA on --note-bg"

    # --ink-dim measures 4.04:1 on the violet and must not appear on EITHER
    # surface that carries it -- the callout or the per-claim chip.
    feedback_css = site.CSS.split("/* [17b] Reader feedback", 1)[1].split("/* [18", 1)[0]
    decls = re.sub(r"/\*.*?\*/", "", feedback_css, flags=re.S)
    assert "--ink-dim" not in decls, "--ink-dim is below AA on the note surface"


def test_claim_chip_and_callout_share_the_note_accent():
    """One colour learned once, then recognised 182 times without re-reading."""
    chip = site.CSS.split(".claim-feedback-link {", 1)[1].split("}", 1)[0]
    callout = site.CSS.split(".report-feedback-callout {", 1)[1].split("}", 1)[0]
    for block, name in ((chip, "claim chip"), (callout, "report callout")):
        assert "var(--note-bg)" in block, f"{name} is off the note accent"
        assert "var(--note-border)" in block, f"{name} border is off the accent"


def test_feedback_styling_hardcodes_no_colour_and_borrows_no_verdict():
    """Colour reaches this block only through a token, and never a verdict one.

    Scoped to DECLARATIONS: comments are stripped first. The block deliberately
    names --v-exaggerated in prose, to warn the next reader off the very
    shortcut this test forbids, and an unscoped scan would fire on that warning.
    """
    block = site.CSS.split("/* [17b] Reader feedback", 1)[1].split("/* [18", 1)[0]
    decls = re.sub(r"/\*.*?\*/", "", block, flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", decls), (
        "hardcoded colour; add a token to :root instead")
    for verdict_token in ("--v-true", "--v-false", "--v-misleading",
                          "--v-exaggerated", "--v-unverifiable"):
        assert verdict_token not in decls, (
            f"{verdict_token} borrowed for chrome; feedback is not a verdict")
    # The warning itself must survive somewhere, since it is what stops the
    # shortcut. It lives beside the token definition in :root rather than at
    # the use site -- the right place for it, and the place a future editor
    # changing the colour will actually be looking.
    root = site.CSS.split(":root {", 1)[1].split("\n}", 1)[0]
    assert "--v-exaggerated" in root, "the do-not-reuse warning was removed"


# ── a11y pins, read off the CSS constant ─────────────────────────────────────

def test_feedback_link_css_meets_the_a11y_convention():
    shared = site.CSS.split(
        ".claim-feedback-link,\n.report-feedback-link {", 1)[1].split("}", 1)[0]
    assert "min-height: 44px" in shared, "touch target below the 44px convention"
    assert ".claim-feedback-link:focus-visible" in site.CSS
    reduce_block = site.CSS.split(
        "@media (prefers-reduced-motion: reduce) {", 1)[1].split("\n}", 1)[0]
    assert ".claim-feedback-link" in reduce_block


def test_feedback_css_avoids_the_lens_sweep():
    """consistency._check_no_lens_ui bans these across all rendered css/html."""
    block = site.CSS.split(
        "/* [17b] Reader feedback", 1)[1].split("/* [18", 1)[0]
    for pattern in ("Lens", "editorial-lens", "lens-label", "lens-value",
                    "lens-target", "lens-pill", "data-lens", "DEFAULT_LENS"):
        assert pattern not in block
