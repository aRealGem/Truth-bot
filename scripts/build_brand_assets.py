#!/usr/bin/env python3
"""Rebuild every raster derived from the Truthy mascot, from the mascot itself.

WHY THIS EXISTS. These assets were produced interactively three times running,
and each round the constants that mattered -- crop boxes, stamp placement, the
verdict-bar proportions -- lived only in a chat log. The third round was
triggered by a one-line change to the mascot SVG (removing a specular
highlight), which every raster had silently baked in. An asset you cannot
regenerate is an asset that drifts from its source and cannot be corrected
cheaply, so the recipe is now code.

SOURCE OF TRUTH is ``truthbot.publish.site._TRUTHY_SVG`` plus the real site
CSS, rendered by headless Chromium. Nothing here redraws Truthy by hand, so
the brand cannot drift from the mascot the site actually shows.

    python3 scripts/build_brand_assets.py            # rebuild all
    python3 scripts/build_brand_assets.py --check    # report drift, write nothing

NOT run by CI or by publish. These are uploaded by hand to a social profile,
so rebuilding them on every render would be churn no reader ever sees. Run it
when the mascot changes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from truthbot.publish import site                              # noqa: E402
from truthbot.publish.report_cards import (BG, INK, INK_MUTED,  # noqa: E402
                                           VERDICT_RGB, _REGULAR, _SEMIBOLD)

OUT = REPO / "src" / "truthbot" / "publish" / "assets" / "social"
ICONS = REPO / "src" / "truthbot" / "publish" / "assets"

STAMP_RED = VERDICT_RGB["False"]        # the wordmark dot's red; see the README

#: Avatar geometry, in the 1024px frame: Truthy's head, and the circle Bluesky
#: inscribes to crop the square.
HEAD_ELLIPSE = (512, 557, 410, 381)      # cx, cy, rx, ry
CROP_CIRCLE = (512, 512, 512)            # cx, cy, r
#: DERIVED from the SVG, not eyeballed. The avatar viewBox is "25 12 250 250"
#: mapped to 1024px, so scale = 1024/250 about origin (25,12). The bulb is
#: circle cx=152 cy=36 r=5.5 plus its softGlow, and the stalk runs down to the
#: head apex. An earlier hand-guessed box was 10px short on the right and 17px
#: low at the top, which let the stamp graze the bulb while the solver still
#: reported the placement clean.
ANTENNA_KEEPOUT = (485, 63, 555, 188)    # x0, y0, x1, y1

#: Banner BETA placement, measured from the approved asset. No solver: the
#: banner has open space and no circular crop, so nothing constrains it.
BANNER_STAMP = dict(centre=(606, 193), size=64, angle=-9)


# ── rendering the mascot ─────────────────────────────────────────────────────

def _chromium() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome-stable",
                 "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "No Chromium on PATH. The mascot is an SVG in site.py that depends on "
        "the site's own CSS, so it is rasterised by a real browser rather than "
        "re-drawn -- that is what keeps these assets from drifting from him.")


def _page(view_box: str, w: int, h: int, extra_css: str = "") -> str:
    svg = site._TRUTHY_SVG.replace(
        'width="170" height="204" viewBox="0 0 300 360"',
        f'width="{w}" height="{h}" viewBox="{view_box}"')
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{site.CSS}
html,body{{margin:0;padding:0;background:#fafaf9;}}
#wrap{{width:{w}px;height:{h}px;background:#fafaf9;}}
/* The site cycles the LED eyes between moods; a still must pin one frame. */
#mascot .eye-neutral{{opacity:1 !important;}}
#mascot .eye-happy,#mascot .eye-iffy,#mascot .eye-sad{{opacity:0 !important;}}
#mascot *{{animation:none !important;transition:none !important;}}
/* Freezing animations leaves the waving arm untransformed and hanging, so pin
   the wave's own rest frame (the 0%/100% keyframe of index-hero-wave-arm). */
#mascot #armLeftSwing{{transform-box:view-box;transform-origin:88px 253px;
                       transform:rotate(130deg);}}
{extra_css}</style></head><body><div id="wrap">{svg}</div></body></html>"""


def render(view_box: str, w: int, h: int, extra_css: str = "") -> Image.Image:
    with tempfile.TemporaryDirectory() as td:
        html, png = Path(td) / "m.html", Path(td) / "m.png"
        html.write_text(_page(view_box, w, h, extra_css), encoding="utf-8")
        subprocess.run([_chromium(), "--headless", "--disable-gpu",
                        "--hide-scrollbars", f"--screenshot={png}",
                        f"--window-size={w},{h}", str(html)],
                       check=True, capture_output=True)
        if not png.exists():
            raise SystemExit("Chromium produced no screenshot")
        return Image.open(png).convert("RGB").copy()


def keyed(im: Image.Image, crop: bool = True) -> Image.Image:
    """Key the flat --bg background to alpha.

    An exact-match key is safe because the source is a flat SVG render rather
    than a photo: the background is one literal colour and Truthy's cream fill
    is far enough from it that no part of him is removed. Anti-aliased rim
    pixels get a distance-ramped alpha so the silhouette stays smooth.
    """
    im = im.convert("RGB")
    px = im.load()
    alpha = Image.new("L", im.size, 255)
    ap = alpha.load()
    br, bgc, bb = BG
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            r, g, b = px[x, y]
            d = abs(r - br) + abs(g - bgc) + abs(b - bb)
            ap[x, y] = 0 if d <= 2 else (int(255 * (d - 2) / 22) if d < 24 else 255)
    im = im.convert("RGBA")
    im.putalpha(alpha)
    if crop:
        box = im.getchannel("A").getbbox()
        if box:
            im = im.crop(box)
    return im


def stamp_beta(canvas: Image.Image, *, centre, size, angle) -> None:
    """The rotated BETA stamp, in the wordmark's red.

    Red was argued against on the grounds that red means FALSE in the verdict
    palette. That objection is answered by this codebase: the wordmark's own
    separator dot is drawn in the same red. A profile has no verdict pills
    beside it to supply that reading, and red is what makes the mark read as a
    stamp rather than a UI chip.
    """
    font = ImageFont.truetype(str(_SEMIBOLD), size)
    pad = size
    d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    tw = int(d0.textlength("BETA", font=font))
    th = size
    tile = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    box = [pad // 2, pad // 2, tw + pad + pad // 2, th + pad + pad // 2]
    d.rounded_rectangle(box, radius=max(3, size // 8),
                        outline=STAMP_RED + (230,), width=max(3, size // 12))
    d.text((pad + size * 0.12, pad - size * 0.06), "BETA",
           font=font, fill=STAMP_RED + (215,))
    tile = tile.rotate(angle, expand=True, resample=Image.BICUBIC)
    canvas.paste(tile, (centre[0] - tile.size[0] // 2,
                        centre[1] - tile.size[1] // 2), tile)


# ── the assets ───────────────────────────────────────────────────────────────

def build_fullbody() -> Image.Image:
    return render("0 0 300 360", 1200, 1440)


def build_head() -> Image.Image:
    """Head-only, framed by geometry rather than by eye.

    The head is an ellipse at cx=150 cy=148 rx=100 ry=93, so this viewBox
    centres head-and-antenna with enough margin to survive the circle Bluesky
    inscribes in the square. The clipboard is hidden because at avatar scale it
    survives the crop only as an unidentifiable fragment.

    BOTH arms are hidden, not just the right. The left one is raised by the
    wave pose pinned in _page(), which puts its hand inside this crop -- a
    disembodied forearm entering frame at the bottom-left corner.
    """
    return render("25 12 250 250", 1024, 1024,
                  "#armRight,#armLeft,#armLeftSwing,#clipboard"
                  "{display:none !important;}")


def _beta_tile(size: int, angle: float) -> Image.Image:
    font = ImageFont.truetype(str(_SEMIBOLD), size)
    pad = size
    tw = int(ImageDraw.Draw(Image.new("RGB", (10, 10))).textlength("BETA", font=font))
    tile = Image.new("RGBA", (tw + pad * 2, size + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle([pad // 2, pad // 2, tw + pad + pad // 2, size + pad + pad // 2],
                        radius=max(3, size // 8), outline=STAMP_RED + (230,),
                        width=max(3, size // 12))
    d.text((pad + size * 0.12, pad - size * 0.06), "BETA", font=font,
           fill=STAMP_RED + (215,))
    return tile.rotate(angle, expand=True, resample=Image.BICUBIC)


def solve_avatar_stamp(angle: float = -17):
    """Find the largest BETA that fits the white space above-right of the head.

    RE-SOLVED rather than stored, because the constraints pull against each
    other and a hand-nudged constant silently breaks when anything upstream
    moves. Lifting the stamp clear of the skull pushes it toward the circular
    crop, so only a narrow band satisfies both; the band is about 9px wide at
    the crop boundary. A stored constant from an earlier build was re-measured
    at 218x138 against a 176x110 solution and clipped the crop by 6px.

    Constraints: every opaque point outside the head ellipse plus clearance,
    all of the tile inside the crop circle with margin, and clear of the
    antenna.
    """
    hx, hy, hrx, hry = HEAD_ELLIPSE
    ccx, ccy, cr = CROP_CIRCLE
    ax0, ay0, ax1, ay1 = ANTENNA_KEEPOUT
    best = None
    for size in range(56, 27, -2):
        tile = _beta_tile(size, angle)
        alpha = tile.getchannel("A")
        # Every opaque pixel, not a sample. Stride-3 sampling reported a
        # placement clean while 17 of its pixels sat inside the keep-out; a
        # solver whose report cannot be trusted is worse than no solver.
        pts = [(x, y) for x in range(tile.size[0])
               for y in range(tile.size[1]) if alpha.getpixel((x, y)) > 40]
        if not pts:
            continue
        for cy in range(80, 320, 10):
            for cx in range(560, 860, 10):
                ox, oy = cx - tile.size[0] // 2, cy - tile.size[1] // 2
                ok, far = True, 0.0
                for px, py in pts:
                    X, Y = ox + px, oy + py
                    if ((X - hx) / (hrx + 10)) ** 2 + ((Y - hy) / (hry + 10)) ** 2 <= 1:
                        ok = False
                        break
                    if ax0 <= X <= ax1 and ay0 <= Y <= ay1:
                        ok = False
                        break
                    dist = ((X - ccx) ** 2 + (Y - ccy) ** 2) ** 0.5
                    if dist > cr - 8:
                        ok = False
                        break
                    far = max(far, dist)
                if ok:
                    best = dict(centre=(cx, cy), size=size, angle=angle,
                                far=round(far), tile=tile.size)
                    break
            if best:
                break
        if best:
            break
    if not best:
        raise SystemExit("no BETA placement satisfies the avatar constraints")
    return best


def build_avatar(head: Image.Image) -> Image.Image:
    fit = solve_avatar_stamp()
    print(f"  avatar BETA: size {fit['size']} at {fit['centre']}, "
          f"tile {fit['tile']}, farthest corner {fit['far']}/512")
    av = head.copy()
    stamp_beta(av, centre=fit["centre"], size=fit["size"], angle=fit["angle"])
    return av


def build_banner(full: Image.Image) -> Image.Image:
    """1500x500. The lower-left stays empty: Bluesky overlays the avatar there."""
    W, H = 1500, 500
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    wm = ImageFont.truetype(str(_SEMIBOLD), 76)
    x = 120
    d.text((x, 152), "truth", font=wm, fill=INK)
    x += d.textlength("truth", font=wm)
    d.text((x, 152), "·", font=wm, fill=STAMP_RED)
    x += d.textlength("·", font=wm)
    d.text((x, 152), "bot", font=wm, fill=INK)

    tag = ImageFont.truetype(str(_REGULAR), 36)
    d.text((120, 262), "Automated political fact-checking", font=tag, fill=INK_MUTED)
    d.text((120, 308), "with multi-model consensus", font=tag, fill=INK_MUTED)

    fig = keyed(full)
    h = 368
    w = round(fig.size[0] * h / fig.size[1])
    fig = fig.resize((w, h), Image.LANCZOS)
    im.paste(fig, (1240 - w // 2, 56), fig)

    _verdict_bar(im, W, H)
    stamp_beta(im, **BANNER_STAMP)
    return im


def _verdict_bar(im: Image.Image, W: int, H: int) -> None:
    """The site's verdict bar at real corpus proportions.

    A POINT-IN-TIME SNAPSHOT, and the only figure on any brand asset. The
    previous banner encoded 58.9% True against a corpus that was 57.8%, and
    dropped "Models split" entirely -- drift nothing could catch, since these
    files are uploaded by hand and no lint reads a PNG. Recomputed here on
    every build so at least it is true when made; re-run after new reports.
    """
    claims_path = REPO / "site-pca" / "data" / "claims.json"
    if not claims_path.exists():
        return
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    counts = {}
    for c in claims:
        counts[c["consensus_verdict"]] = counts.get(c["consensus_verdict"], 0) + 1
    order = ("True", "Mostly True", "Truthy", "Exaggerated", "Misleading",
             "Falsey", "False", "Models split", "Unverifiable")
    total = sum(counts.values()) or 1
    d = ImageDraw.Draw(im)
    x, bar_h = 0, 10
    for label in order:
        n = counts.get(label, 0)
        if not n:
            continue
        seg = round(W * n / total)
        d.rectangle([x, H - bar_h, min(x + seg, W), H], fill=VERDICT_RGB[label])
        x += seg
    if x < W:                       # rounding crumb
        d.rectangle([x, H - bar_h, W, H], fill=VERDICT_RGB["Unverifiable"])


def build_icons(head: Image.Image) -> Image.Image:
    """apple-touch-icon only. The 16/32px favicons are raster twins of
    favicon.svg (a green check), NOT the face: at 16px Truthy is a smudge, and
    the SVG and the PNG fallbacks used to disagree because nothing compared
    them."""
    S = 180
    c = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    mask = Image.new("L", (S * 4, S * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, S * 4 - 1, S * 4 - 1], radius=int(S * 4 * 0.22), fill=255)
    c.paste(Image.new("RGBA", (S, S), INK + (255,)), (0, 0),
            mask.resize((S, S), Image.LANCZOS))
    fig = keyed(head)
    inner = int(S * 0.84)
    w = round(fig.size[0] * inner / fig.size[1])
    fig = fig.resize((w, inner), Image.LANCZOS)
    c.paste(fig, ((S - w) // 2, int(S * 0.08)), fig)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report which assets would change; write nothing")
    args = ap.parse_args()

    full, head = build_fullbody(), build_head()
    targets = {
        OUT / "truthy-fullbody.png": full,
        OUT / "truthy-avatar.png": build_avatar(head),
        OUT / "truthy-banner.png": build_banner(full),
        ICONS / "apple-touch-icon.png": build_icons(head),
    }
    changed = []
    for path, img in targets.items():
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        new = buf.getvalue()
        if not path.exists() or path.read_bytes() != new:
            changed.append(path.relative_to(REPO))
            if not args.check:
                path.write_bytes(new)
    verb = "would change" if args.check else "rebuilt"
    print(f"{verb}: {len(changed)}")
    for c in changed:
        print(f"  {c}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
