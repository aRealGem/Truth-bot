"""OpenGraph share cards — one per report, one per claim.

WHY THESE EXIST. A link posted to Bluesky, Signal, Slack or X renders as its
card, and until now every page on the site emitted the SAME static image. So a
report, a claim and the index were visually indistinguishable once shared. On a
platform where the card IS the post's visual, that wastes the only thing a
reader sees before deciding whether to click.

WHAT THIS MODULE MAY NOT DO: compute a figure. Every number on a card arrives as
an argument, and callers must pass the same values the HTML already renders.
``consistency.check_site`` polices quantitative claims in HTML against
``data/*.json``, and it cannot read a PNG -- a card that did its own arithmetic
would be the one surface on the site free to drift, with no lint behind it.
Passing the HTML's own values makes the card linted transitively: drift fails on
the page before the image can lie.

DETERMINISM. The rendered tree is byte-compared across two runs in CI. Nothing
here may vary between renders: no timestamps, no locale-dependent formatting, no
system font lookup. Fonts are vendored beside this module for the same reason --
a card whose typography depends on which machine rendered it would silently
break M-12's tree-equality postcondition.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

# OpenGraph standard. Matches the existing assets/social-card.png.
CARD_W, CARD_H = 1200, 630

#: Claim-card text block: top edge, and the verdict pill's top edge. The gap
#: between them is the budget the claim text must auto-fit into.
TEXT_TOP = 190
PILL_TOP = CARD_H - 190

_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_REGULAR = _FONT_DIR / "Geist-Regular.ttf"
_SEMIBOLD = _FONT_DIR / "Geist-SemiBold.ttf"

# Straight from the CSS :root block. Kept as literals rather than parsed out of
# the stylesheet because a parser would be a second thing to keep correct; the
# test suite pins these against the CSS constant instead.
BG = (250, 250, 249)          # --bg
SURFACE = (255, 255, 255)     # --surface
INK = (12, 10, 9)             # --ink
INK_MUTED = (87, 83, 78)      # --ink-muted
INK_FAINT = (168, 162, 158)   # --ink-faint
BORDER = (231, 229, 228)      # --border

#: Verdict palette, --v-* in the stylesheet. The ONLY chromatic vocabulary.
VERDICT_RGB = {
    "True": (21, 128, 61),
    "Mostly True": (101, 163, 13),
    "Exaggerated": (202, 138, 4),
    "Misleading": (194, 65, 12),
    "False": (153, 27, 27),
    "Unverifiable": (68, 64, 60),
    "Models split": (100, 116, 139),
    "Truthy": (132, 204, 22),
    "Falsey": (234, 88, 12),
}
_DEFAULT_RGB = VERDICT_RGB["Unverifiable"]

#: Bar segment order — worst-to-best reads left-to-right like the report page.
_BAR_ORDER = ("True", "Mostly True", "Truthy", "Exaggerated", "Misleading",
              "Falsey", "False", "Models split", "Unverifiable")


class CardFontError(RuntimeError):
    """A vendored font is missing.

    Deliberately fatal. The renderer is called unconditionally and the page
    names its card's URL either way, so a silently-skipped card ships a 404
    og:image -- worse than a failed publish, because nothing reports it.
    """


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.exists():
        raise CardFontError(
            f"vendored font missing: {path}. Fonts ship with the package so "
            f"card typography cannot depend on the rendering machine; restore "
            f"it rather than falling back to a system face.")
    return ImageFont.truetype(str(path), size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
          max_w: int, max_lines: int) -> list[str]:
    """Greedy word wrap, measured against the real font rather than guessed.

    Overflow is truncated with an ellipsis on the last line: a card that
    silently drops a clause would misquote the speaker, which is the one
    failure a fact-checker cannot ship.
    """
    words = str(text or "").split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and (len(" ".join(lines).split()) < len(words)):
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_w:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines[-1] = last + "…"
    return lines


def _chrome(draw: ImageDraw.ImageDraw) -> None:
    """Panel and wordmark, shared by both card types."""
    draw.rectangle([0, 0, CARD_W, CARD_H], fill=BG)
    draw.rectangle([40, 40, CARD_W - 40, CARD_H - 40], fill=SURFACE, outline=BORDER, width=1)
    wm = _font(_SEMIBOLD, 30)
    draw.text((80, 74), "truth", font=wm, fill=INK)
    x = 80 + draw.textlength("truth", font=wm)
    # The dot carries the site's wordmark accent.
    draw.text((x, 74), "·", font=wm, fill=VERDICT_RGB["False"])
    x += draw.textlength("·", font=wm)
    draw.text((x, 74), "bot", font=wm, fill=INK)


def _footer(draw: ImageDraw.ImageDraw, left: str, right: str) -> None:
    f = _font(_REGULAR, 24)
    draw.text((80, CARD_H - 100), left, font=f, fill=INK_MUTED)
    w = draw.textlength(right, font=f)
    draw.text((CARD_W - 80 - w, CARD_H - 100), right, font=f, fill=INK_FAINT)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    # optimize=True is deterministic in Pillow; no tIME chunk is written.
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_report_card(*, speaker: str, role: str, display_date: str,
                       headline: str, ratio_text: str, headline_verdict: str,
                       distribution: dict, claim_count: int,
                       event: str = "") -> bytes:
    """One card per report. Every figure is supplied, never derived here."""
    img = Image.new("RGB", (CARD_W, CARD_H), BG)
    d = ImageDraw.Draw(img)
    _chrome(d)

    d.text((80, 150), speaker, font=_font(_SEMIBOLD, 64), fill=INK)
    # role, occasion, date -- occasion is authored per speech, never inferred
    meta = " · ".join(x for x in (role, event, display_date) if x)
    d.text((80, 232), meta, font=_font(_REGULAR, 30), fill=INK_MUTED)

    colour = VERDICT_RGB.get(headline_verdict, _DEFAULT_RGB)
    d.text((80, 310), headline, font=_font(_SEMIBOLD, 92), fill=colour)
    d.text((80, 420), ratio_text, font=_font(_REGULAR, 28), fill=INK_MUTED)

    # Stacked distribution bar, same grammar as the report page's verdict bar.
    total = sum(int(v or 0) for v in distribution.values()) or 1
    x, y, bar_w, bar_h = 80, 476, CARD_W - 160, 26
    for label in _BAR_ORDER:
        n = int(distribution.get(label, 0) or 0)
        if n <= 0:
            continue
        seg = max(2, round(bar_w * n / total))
        seg = min(seg, bar_w - (x - 80))
        if seg <= 0:
            break
        d.rectangle([x, y, x + seg, y + bar_h], fill=VERDICT_RGB.get(label, _DEFAULT_RGB))
        x += seg

    _footer(d, f"{claim_count} claims checked", "truth-bot")
    return _png(img)


#: The house card's copy, lifted verbatim from the index page's own tagline and
#: meta description. Duplicated rather than imported because the card must not
#: depend on a rendered page, and pinned equal to the page by the test suite.
_HOUSE_TAGLINE = "Automated political fact-checking"
_HOUSE_SUB = "with multi-model consensus analysis."
_HOUSE_STEPS = (
    "Decompose the speech into checkable claims",
    "Check each against a shared, cited evidence pack",
    "Reconcile a multi-model panel into one verdict",
)

#: House-card geometry. The text column and the mascot must not overlap, and
#: the first attempt did -- a 58px headline set flush left ran 950px wide and
#: drove "fact-checking" straight through Truthy's face. So the column has a
#: hard right edge and every line is wrapped and auto-fitted against it rather
#: than trusted to fit.
_HOUSE_COL_L = 80
_HOUSE_COL_W = 620              # right edge x=700; the mascot starts at 738
#: Centre x, centre y, height. The figure is FULL BODY rather than the head
#: used on the Bluesky avatar, and for a different reason than symmetry: at
#: 1200x630 the head filled 330px of a 548px panel and left the column looking
#: half-empty, and more to the point the head alone omits the clipboard, which
#: is the one part of the mascot that says what the site does. Centre-x splits
#: the gap between the text column's right edge and the panel's.
_MASCOT_BOX = (929, 318, 500)
_MASCOT_ASSET = "truthy-fullbody.png"


def _mascot(height: int) -> Optional[Image.Image]:
    """The site's own Truthy, background keyed out so he sits on the panel.

    The source PNG was rendered over ``--bg`` (250,250,249) and the card panel
    is ``--surface`` white, so pasting it flat would leave a faintly grey
    rectangle -- the kind of edge that is invisible in a thumbnail and obvious
    at full size. An exact-match key is safe here precisely because the source
    is a flat SVG render rather than a photo: the background is one literal
    colour, and Truthy's own cream fill is far enough from it that no part of
    him is keyed away. Anti-aliased rim pixels get a distance-ramped alpha so
    the silhouette stays smooth instead of stair-stepping.

    Returns None if the asset is missing -- unlike a font, the mascot is
    decoration, and a card without him still says everything it needs to.
    """
    src = Path(__file__).resolve().parent / "assets" / "social" / _MASCOT_ASSET
    if not src.exists():
        return None
    im = Image.open(src).convert("RGB")
    px = im.load()
    alpha = Image.new("L", im.size, 255)
    ap = alpha.load()
    br, bg_, bb = BG
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            r, g, b = px[x, y]
            d = abs(r - br) + abs(g - bg_) + abs(b - bb)
            if d <= 2:
                ap[x, y] = 0
            elif d < 24:                      # anti-aliased rim
                ap[x, y] = int(255 * (d - 2) / 22)
    im.putalpha(alpha)
    # Crop to the keyed silhouette first: the source canvas carries a wide
    # transparent margin, so without this `height` would size the padding
    # rather than the figure.
    bbox = im.getchannel("A").getbbox()
    if bbox:
        im = im.crop(bbox)
    w = round(im.size[0] * height / im.size[1])
    return im.resize((w, height), Image.LANCZOS)


def render_house_card() -> bytes:
    """The card for pages that are about the project, not about a report.

    Index, about, corrections, truthy and 404 all share this one.

    IT CARRIES NO FIGURES, AND THAT IS THE POINT. The image this replaced was
    built by hand and embedded the then-latest report -- "2026-03-04 · 5 claims
    · Largely False". By the time it was found it was wrong on every one of
    those figures, because nothing regenerates a hand-made PNG and
    ``consistency.check_site`` cannot read one to complain. A figure-free house
    card cannot go stale, so the failure mode is designed out rather than
    watched for.

    Generated at publish time rather than shipped as a binary for the same
    reason: it uses the vendored font and the shared chrome, so it cannot drift
    from the site it fronts, and there is no blob in the tree that nobody knows
    how to remake.
    """
    img = Image.new("RGB", (CARD_W, CARD_H), BG)
    d = ImageDraw.Draw(img)
    _chrome(d)

    head = _mascot(_MASCOT_BOX[2])
    if head is not None:
        cx, cy, _h = _MASCOT_BOX
        img.paste(head, (cx - head.size[0] // 2, cy - head.size[1] // 2), head)
        d = ImageDraw.Draw(img)

    L, W = _HOUSE_COL_L, _HOUSE_COL_W
    head_f = _font(_SEMIBOLD, 52)
    y = 168
    for line in _wrap(d, _HOUSE_TAGLINE, head_f, W, 3):
        d.text((L, y), line, font=head_f, fill=INK)
        y += 62
    sub_f = _font(_REGULAR, 30)
    y += 4
    for line in _wrap(d, _HOUSE_SUB, sub_f, W, 2):
        d.text((L, y), line, font=sub_f, fill=INK_MUTED)
        y += 40

    # Steps: the longest one sets the size for all three, so they stay a set.
    num_f = _font(_SEMIBOLD, 20)
    indent = 52
    for size in (25, 23, 21, 19):
        step_f = _font(_REGULAR, size)
        if max(d.textlength(s_, font=step_f) for s_ in _HOUSE_STEPS) <= W - indent:
            break
    y = 372
    for i, step in enumerate(_HOUSE_STEPS, 1):
        d.ellipse([L, y, L + 32, y + 32], fill=BG, outline=BORDER, width=1)
        n = str(i)
        d.text((L + 16 - d.textlength(n, font=num_f) / 2, y + 4), n,
               font=num_f, fill=INK_MUTED)
        d.text((L + indent, y + 3), step, font=step_f, fill=INK)
        y += 52

    _footer(d, "arealgem.github.io/Truth-bot", "beta")
    return _png(img)


def render_claim_card(*, claim_text: str, verdict: str, speaker: str,
                      display_date: str) -> bytes:
    """One card per claim. The claim is the hero; the verdict labels it."""
    img = Image.new("RGB", (CARD_W, CARD_H), BG)
    d = ImageDraw.Draw(img)
    _chrome(d)

    meta = " · ".join(x for x in (speaker, display_date) if x)
    d.text((80, 132), meta, font=_font(_REGULAR, 28), fill=INK_MUTED)

    # Auto-fit the claim: take the largest size that fits the block WITHOUT
    # truncating. A fixed size cannot work across a corpus whose claims run
    # from 30 to 500 characters -- at 46px the long ones overran into the
    # verdict pill, which is the one collision that would make a card misread.
    quoted = f"“{claim_text}”"
    top, budget, max_w = TEXT_TOP, PILL_TOP - TEXT_TOP - 30, CARD_W - 160
    body, lines = None, []
    for size in (52, 46, 40, 34, 30, 26):
        f = _font(_SEMIBOLD, size)
        lh = round(size * 1.35)
        cap = max(1, budget // lh)
        cand = _wrap(d, quoted, f, max_w, cap)
        body, lines, line_h = f, cand, lh
        if not cand[-1].endswith("…"):
            break                      # fits whole; stop at the largest size
    y = top
    for line in lines:
        d.text((80, y), line, font=body, fill=INK)
        y += line_h

    colour = VERDICT_RGB.get(verdict, _DEFAULT_RGB)
    pill = _font(_SEMIBOLD, 34)
    label = verdict.upper()
    tw = d.textlength(label, font=pill)
    d.rectangle([80, PILL_TOP, 80 + tw + 56, PILL_TOP + 62], fill=colour)
    d.text((108, PILL_TOP + 14), label, font=pill, fill=(255, 255, 255))

    _footer(d, "checked against the public record", "truth-bot")
    return _png(img)
