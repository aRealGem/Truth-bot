"""Share cards: one per report, one per decided claim.

The card is the only surface on the site that ``consistency.check_site``
cannot read -- it lints quantitative claims in HTML against ``data/*.json``
and a PNG is invisible to it. So a card that computed its own figures would be
free to drift with nothing behind it. The defence is structural rather than a
new lint: the renderer computes nothing, and the tests below pin that it is
handed the same values the HTML renders.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from lxml import html as LH
from PIL import Image

from truthbot.publish import report_cards, site
from truthbot.publish.site import (_CARDED_VERDICTS, _claim_card_path,
                                   _report_card_path, _report_events, _site_url)

REPORTS = sorted(
    (Path(__file__).resolve().parents[2] / "site-pca" / "reports").glob("*.html"))
SITE_PCA = Path(__file__).resolve().parents[2] / "site-pca"

_FIGURES = dict(speaker="Donald Trump", role="President",
                display_date="February 24, 2026", event="2026 State of the Union",
                headline="50% True", ratio_text="61 of 122 decided claims rated True",
                headline_verdict="True",
                distribution={"True": 61, "False": 39, "Misleading": 22,
                              "Unverifiable": 56},
                claim_count=182)


# ── the renderer itself ──────────────────────────────────────────────────────

def test_report_card_is_a_valid_og_sized_png():
    im = Image.open(io.BytesIO(report_cards.render_report_card(**_FIGURES)))
    assert im.format == "PNG"
    assert im.size == (1200, 630), "OpenGraph expects 1200x630"


def test_claim_card_is_a_valid_og_sized_png():
    png = report_cards.render_claim_card(
        claim_text="Unemployment is at a fifty-year low.", verdict="False",
        speaker="Donald Trump", display_date="February 24, 2026")
    im = Image.open(io.BytesIO(png))
    assert im.format == "PNG" and im.size == (1200, 630)


def test_rendering_is_byte_stable():
    """The whole rendered tree is byte-compared across two CI runs, and these
    PNGs are now part of it. Same inputs must give the same bytes."""
    assert (report_cards.render_report_card(**_FIGURES)
            == report_cards.render_report_card(**_FIGURES))


def test_a_long_claim_fits_rather_than_colliding():
    """Auto-fit, not a fixed size.

    The corpus runs from 30 to 500 characters. At a fixed 46px the long claims
    overran the block and struck the verdict pill -- the one collision that
    would make a card misread. The renderer now takes the largest size that
    fits without truncating.
    """
    long_claim = " ".join(["consequential"] * 40)
    png = report_cards.render_claim_card(
        claim_text=long_claim, verdict="Misleading",
        speaker="A Speaker", display_date="January 1, 2026")
    assert Image.open(io.BytesIO(png)).size == (1200, 630)


def test_the_font_is_vendored_not_resolved():
    """Typography must not depend on which machine rendered.

    The determinism test renders twice on ONE machine, so it passes either way
    -- which is exactly why it is the wrong instrument here. The postcondition
    that discriminates is M-12's tree-equality against an accepted render.
    """
    src = Path(report_cards.__file__).read_text(encoding="utf-8")
    assert "/usr/share/fonts" not in src
    assert "load_default" not in src
    assert report_cards._REGULAR.exists() and report_cards._SEMIBOLD.exists()
    assert (report_cards._FONT_DIR / "OFL.txt").exists(), "ship the licence"


def test_a_missing_font_raises_rather_than_degrading(tmp_path, monkeypatch):
    """A silently-skipped card ships a 404 og:image, which nothing reports."""
    monkeypatch.setattr(report_cards, "_SEMIBOLD", tmp_path / "gone.ttf")
    with pytest.raises(report_cards.CardFontError):
        report_cards.render_report_card(**_FIGURES)


def test_cards_borrow_no_colour_outside_the_palette():
    """Every card colour is a token from the site's :root block."""
    root = site.CSS.split(":root {", 1)[1].split("\n}", 1)[0]
    css_hexes = {h.lower() for h in re.findall(r"#([0-9a-fA-F]{6})", root)}
    for name in ("BG", "SURFACE", "INK", "INK_MUTED", "INK_FAINT", "BORDER"):
        rgb = getattr(report_cards, name)
        assert "%02x%02x%02x" % rgb in css_hexes, f"{name} is not a site token"
    for label, rgb in report_cards.VERDICT_RGB.items():
        assert "%02x%02x%02x" % rgb in css_hexes, f"{label} is not a verdict token"


# ── the house card ───────────────────────────────────────────────────────────

def test_house_card_is_a_valid_og_sized_png():
    im = Image.open(io.BytesIO(report_cards.render_house_card()))
    assert im.format == "PNG" and im.size == (1200, 630)


def test_house_card_carries_no_figure_that_could_go_stale():
    """The bug this replaced, stated as a rule.

    The old house card was a hand-made PNG showing the then-latest report --
    "2026-03-04 · 5 claims · Largely False" -- and it kept saying that on every
    page after all three figures were wrong. check_site reads HTML, never
    pixels, so nothing could catch it, and the generator that made it hardcoded
    those figures as literals, so re-running it would have re-applied the
    staleness rather than fixing it.

    A house card with no figures cannot go stale. This asserts the property
    rather than the pixels: no digit appears anywhere in its copy.
    """
    copy = " ".join((report_cards._HOUSE_TAGLINE, report_cards._HOUSE_SUB,
                     *report_cards._HOUSE_STEPS))
    assert not any(ch.isdigit() for ch in copy), (
        f"a figure crept into the house card copy: {copy!r}")


def test_house_card_takes_no_arguments():
    """Constant by construction -- there is no input to pass a figure through."""
    import inspect
    assert not inspect.signature(report_cards.render_house_card).parameters


def test_the_stale_card_generator_is_gone():
    """social-media/gen_assets.py kept the figures as string literals. Removing
    the PNG while leaving the routine that rebuilds it would fix nothing."""
    gen = Path(__file__).resolve().parents[2] / "social-media" / "gen_assets.py"
    if not gen.exists():
        return
    src = gen.read_text(encoding="utf-8")
    assert "def make_social_card" not in src
    for figure in ("Largely False", "LATEST REPORT", "3 of 5 claims"):
        assert figure not in src.split('"""', 2)[-1], f"{figure!r} still live"


def test_house_card_copy_matches_the_page_it_fronts():
    """The card duplicates the index page's tagline rather than importing it,
    because the renderer must not depend on a rendered page. Duplication is
    only safe if something pins the two equal."""
    idx = SITE_PCA / "index.html"
    if not idx.exists():
        pytest.skip("site-pca not rendered")
    tagline = " ".join(
        LH.fromstring(idx.read_text(encoding="utf-8"))
        .xpath('//p[contains(@class,"tagline")]')[0].text_content().split())
    card_copy = f"{report_cards._HOUSE_TAGLINE} {report_cards._HOUSE_SUB}"
    assert card_copy == tagline, (
        f"card says {card_copy!r}, page says {tagline!r}")


def test_the_mascot_is_keyed_not_pasted_flat():
    """The mascot PNG was rendered over --bg but the card panel is --surface
    white, so a flat paste leaves a faintly grey rectangle -- invisible in a
    thumbnail, obvious at full size. Assert the panel stays white right beside
    him rather than trusting the eye."""
    im = Image.open(io.BytesIO(report_cards.render_house_card())).convert("RGB")
    cx, cy, h = report_cards._MASCOT_BOX
    # Probes are taken relative to the PANEL, not by eyeballed offsets from the
    # mascot: the panel is inset 40px, so a naive cx+210 lands at x=1165 and
    # samples the outer --bg margin, which is legitimately not white and made
    # this test fail for a reason that had nothing to do with the mascot.
    panel_r = report_cards.CARD_W - 41
    fig = report_cards._mascot(h)
    probes = ((cx, cy - h // 2 - 15),                  # above the antenna
              (cx, cy + h // 2 + 12),                  # below the base
              (panel_r - 15, cy),                      # right of him, on panel
              (cx - fig.size[0] // 2 - 25, cy - 150))  # left of him
    for probe in probes:
        assert 41 <= probe[0] <= panel_r, f"probe {probe} is off the panel"
        assert im.getpixel(probe) == report_cards.SURFACE, (
            f"{probe} is {im.getpixel(probe)}, not the white panel -- "
            f"the mascot's background is showing through")


def test_the_mascot_clears_the_text_column_and_the_panel():
    """Geometry, asserted rather than eyeballed. The full-body figure is 378px
    wide against a 459px gap, so the margins are real but not generous."""
    cx, cy, h = report_cards._MASCOT_BOX
    fig = report_cards._mascot(h)
    assert fig is not None, "the mascot asset is missing"
    x0, y0 = cx - fig.size[0] // 2, cy - fig.size[1] // 2
    x1, y1 = x0 + fig.size[0], y0 + fig.size[1]
    assert x0 > report_cards._HOUSE_COL_L + report_cards._HOUSE_COL_W, (
        f"the figure starts at x={x0}, inside the text column")
    assert 41 < x0 and x1 < report_cards.CARD_W - 41, "clips the panel sideways"
    assert 41 < y0 and y1 < report_cards.CARD_H - 41, "clips the panel vertically"


def test_the_footer_label_is_not_drawn_over_the_mascot():
    """The footer is drawn AFTER the paste, so an overlap would put faint grey
    text on his cream body rather than behind him -- unreadable, and invisible
    to any test that only checks bounding boxes, since his silhouette is much
    narrower than his box down there. Assert on the alpha channel."""
    from PIL import ImageDraw, Image as _I
    d = ImageDraw.Draw(_I.new("RGB", (10, 10)))
    f = report_cards._font(report_cards._REGULAR, 24)
    w = d.textlength("beta", font=f)
    bx0, by0 = report_cards.CARD_W - 80 - w, report_cards.CARD_H - 100
    cx, cy, h = report_cards._MASCOT_BOX
    fig = report_cards._mascot(h)
    mx0, my0 = cx - fig.size[0] // 2, cy - fig.size[1] // 2
    alpha = fig.getchannel("A")
    for X in range(int(bx0), int(bx0 + w)):
        for Y in range(by0, by0 + 30):
            lx, ly = X - mx0, Y - my0
            if 0 <= lx < fig.size[0] and 0 <= ly < fig.size[1]:
                assert alpha.getpixel((lx, ly)) <= 40, (
                    f"the mascot is opaque at ({X},{Y}), under the footer label")


def test_house_card_text_never_runs_into_the_mascot():
    """The first attempt set a 58px headline flush left; it ran 950px wide and
    drove "fact-checking" through Truthy's face. The column now has a hard
    right edge, and every string is fitted against it."""
    from PIL import ImageDraw, Image as _I
    d = ImageDraw.Draw(_I.new("RGB", (1200, 630)))
    # Where the figure actually sits is asserted by the geometry test above,
    # which measures the rendered silhouette. (An earlier version of this line
    # used the mascot's HEIGHT as its width and computed a left edge 61px off.)
    head_f = report_cards._font(report_cards._SEMIBOLD, 52)
    for line in report_cards._wrap(d, report_cards._HOUSE_TAGLINE, head_f,
                                   report_cards._HOUSE_COL_W, 3):
        assert not line.endswith("…"), "the headline is being truncated"


# ── wiring ───────────────────────────────────────────────────────────────────

def _full_reports():
    """Real report pages only: reports/*.html also holds triage pages and the
    redirect stubs left by re-adjudication, neither of which is a report."""
    for p in REPORTS:
        doc = LH.fromstring(p.read_text(encoding="utf-8"))
        if doc.xpath('//article[contains(@class, "claim")]'):
            yield p, doc


def _og_image(doc):
    v = doc.xpath('//meta[@property="og:image"]/@content')
    return v[0] if v else None


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_each_report_names_its_own_card():
    seen, urls = 0, set()
    for p, doc in _full_reports():
        seen += 1
        slug = p.stem
        img = _og_image(doc)
        assert img == f"{_site_url()}/{_report_card_path(slug)}", p.name
        urls.add(img)
        tw = doc.xpath('//meta[@name="twitter:image"]/@content')
        assert tw and tw[0] == img, f"{p.name}: twitter:image disagrees with og:image"
    # 5 presidential + warren_2025-04-29; cruz held (FR-0901-06) and
    # pruned from the site (FR-0901-10).
    assert seen == 6
    assert len(urls) == seen, "reports are sharing a card"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_no_page_names_a_card_that_does_not_exist():
    """The regression this feature actually shipped.

    _render_claim_page named a claim card unconditionally while generation
    skipped undecided verdicts, so 142 of 529 claim pages pointed at a PNG that
    was never written. Nothing reported it: the page renders fine and only the
    social preview breaks, silently, and only for a reader who shares it.
    """
    missing = []
    for p in list(SITE_PCA.glob("reports/*.html")) + list(SITE_PCA.glob("claims/*.html")):
        img = _og_image(LH.fromstring(p.read_text(encoding="utf-8")))
        if not img:
            continue
        rel = img.split("site-pca/", 1)[-1]
        if not (SITE_PCA / rel).exists():
            missing.append(f"{p.name} -> {rel}")
    assert not missing, f"{len(missing)} page(s) name a missing card: {missing[:3]}"


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_only_decided_claims_get_their_own_card():
    """Undecided claims fall back to the parent report card.

    "We could not check this" is the least shareable thing the site produces
    and would be a quarter of the weight.
    """
    import json
    claims = json.loads((SITE_PCA / "data" / "claims.json").read_text(encoding="utf-8"))
    for c in claims:
        own = SITE_PCA / _claim_card_path(c["id"])
        decided = c["consensus_verdict"] in _CARDED_VERDICTS
        assert own.exists() == decided, (
            f"{c['id']} ({c['consensus_verdict']}): card exists={own.exists()}")


@pytest.mark.skipif(not REPORTS, reason="site-pca not rendered")
def test_chrome_pages_keep_the_house_card():
    for name in ("index.html", "about.html", "corrections.html"):
        p = SITE_PCA / name
        if not p.exists():
            continue
        img = _og_image(LH.fromstring(p.read_text(encoding="utf-8")))
        assert img == f"{_site_url()}/assets/social-card.png", name


# ── the occasion label ───────────────────────────────────────────────────────

def test_event_labels_are_authored_not_derived():
    """"President + Capitol + January" looks like it implies a State of the
    Union, but that is wrong for an inaugural, a farewell, and for a first-year
    address, which is formally an Address to a Joint Session. Mislabelling the
    occasion on our own share card is not a mistake a fact-checker gets to
    make, so every label is a recorded judgement."""
    events = _report_events()
    presidential = {"clinton_1998", "gwbush_2006", "obama_2014",
                    "biden_2022", "trump_2026"}
    assert presidential <= set(events)
    # A presidential address is named by its occasion, which leads with the year.
    for speech_id in presidential:
        label = events[speech_id]
        year = speech_id.rsplit("_", 1)[1]
        assert label.startswith(year), f"{speech_id}: {label!r} does not lead with its year"
    # A Senate floor speech is named by its Congressional Record heading, which
    # is authored from the Record and has no year convention. Pinned verbatim --
    # the point of the test is that these are recorded, not generated.
    assert events["budd_2025-04-02"] == "Fentanyl"
    assert events["cruz_2026-06-24"] == "Dobbs v. Jackson Women's Health Organization"
    assert events["tillis_2025-01-23"] == "Trump Administration"
    assert events["warren_2025-04-29"] == "Trump Administration First 100 Days"


def test_a_speech_with_no_entry_gets_no_label(monkeypatch):
    """Absent means absent, never a guess."""
    monkeypatch.setattr(site, "_report_events", lambda: {})
    assert site._report_events().get("some_2030", "") == ""


# ── icons ────────────────────────────────────────────────────────────────────

_ASSETS = Path(report_cards.__file__).resolve().parent / "assets"


def _pixels(im, opaque_only=False):
    """Pillow 12 deprecates Image.getdata(), and its replacement does not exist
    on the older Pillow CI may resolve -- pyproject pins >=12,<13 but CI never
    reads uv.lock. .load() works on both."""
    px = im.load()
    w, h = im.size
    out = [px[x, y] for y in range(h) for x in range(w)]
    return [p for p in out if p[3] > 128] if opaque_only else out


def test_the_png_favicons_agree_with_favicon_svg():
    """They disagreed for months, invisibly.

    favicon.svg is a green check on a near-white plate; the .ico and the 32px
    PNG were a black Truthy head. Browsers that prefer SVG showed one mark and
    the rest showed another, and nothing failed because no test compared them.
    Asserted on the dominant colours rather than pixels, since the PNGs are
    rasterised and the SVG is not.
    """
    svg = (_ASSETS / "favicon.svg").read_text(encoding="utf-8")
    assert "#65a30d" in svg and "fafaf9" in svg.lower(), (
        "favicon.svg changed; this test encodes the check-on-plate design")
    for name in ("favicon-32.png", "favicon.ico"):
        im = Image.open(_ASSETS / name).convert("RGBA")
        px = _pixels(im, opaque_only=True)
        assert px, f"{name} is fully transparent"
        # the accent must be present, and the plate must dominate
        green = sum(1 for r, g, b, _ in px if g > r + 30 and g > b + 30)
        plate = sum(1 for r, g, b, _ in px if min(r, g, b) > 220)
        assert green, f"{name} has no green check"
        assert plate > green, f"{name} is not a check on a light plate"


def test_the_icons_are_not_the_superseded_flat_mascot():
    """The old head was near-black at every size, which is how it survived --
    a dark blob reads as *something* in a tab. It is gone from the small icons
    because at 16px it read as nothing at all."""
    for name in ("favicon-32.png", "favicon.ico"):
        im = Image.open(_ASSETS / name).convert("RGBA")
        px = _pixels(im, opaque_only=True)
        dark = sum(1 for r, g, b, _ in px if max(r, g, b) < 60)
        assert dark < len(px) * 0.25, f"{name} is still mostly dark"


def test_the_apple_touch_icon_keeps_the_face():
    """180px is the one icon with room for it, and a home-screen icon is where
    the brand should be a face rather than a glyph."""
    im = Image.open(_ASSETS / "apple-touch-icon.png").convert("RGBA")
    assert im.size == (180, 180)
    px = _pixels(im.convert("RGB"))
    mint = sum(1 for r, g, b in px if g > 150 and g > r + 40 and b > r + 20)
    assert mint > 100, "Truthy's lit eyes are missing"
