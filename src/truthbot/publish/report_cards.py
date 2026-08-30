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
