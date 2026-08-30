#!/usr/bin/env python3
"""Generate truth-bot favicons.

The 1200x630 social card USED to be made here and no longer is. It hardcoded
one report's figures as string literals -- "Donald Trump", "2026-03-04",
"5 claims", "Largely False", and a 20/20/60 verdict bar -- so it could only
ever reproduce the same card, which went on being served after all of those
figures were wrong. Re-running it was not a fix; it was how the staleness
would have been re-applied. The house card is now rendered at publish time by
truthbot.publish.report_cards.render_house_card, carries no figures at all,
and uses the vendored font instead of a system one.

The favicon routines below still resolve Liberation fonts by absolute path and
write to a hardcoded OUT directory from another machine, so this script does
not run as-is here. It is kept as the provenance record for the committed
favicon PNGs, not as a working build step.
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUT = "/home/claude/assets-out"
os.makedirs(OUT, exist_ok=True)

# Design tokens (matching site)
BG       = "#fafaf9"
INK      = "#0c0a09"
INK_MUT  = "#57534e"
INK_FAINT= "#a8a29e"
BORDER   = "#e7e5e4"
V_TRUE   = "#15803d"
V_MOST   = "#65a30d"
V_FALSE  = "#991b1b"

# Fonts
SERIF      = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
SANS       = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANS_BOLD  = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
MONO       = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
MONO_BOLD  = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"


def make_truthy_favicon(size, filename):
    """Generate a Truthy head silhouette favicon at the given size."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, int(size * 0.52)
    r = int(size * 0.38)

    # Head circle
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=INK)

    # Visor band (lighter inset)
    v_h = max(int(size * 0.18), 3)
    v_y = cy - v_h // 2 + int(size * 0.02)
    v_margin = max(int(size * 0.08), 1)
    draw.rounded_rectangle([cx-r+v_margin, v_y, cx+r-v_margin, v_y+v_h],
                           radius=max(v_h//2, 1), fill="#3a3a3a")

    # Eyes (two small bright dots)
    eye_r = max(int(size * 0.06), 1)
    eye_y = v_y + v_h // 2
    eye_spread = int(size * 0.14)
    draw.ellipse([cx-eye_spread-eye_r, eye_y-eye_r, cx-eye_spread+eye_r, eye_y+eye_r],
                 fill="#50d8b0")
    draw.ellipse([cx+eye_spread-eye_r, eye_y-eye_r, cx+eye_spread+eye_r, eye_y+eye_r],
                 fill="#50d8b0")

    # Antenna line
    ant_top = cy - r - int(size * 0.12)
    ant_w = max(int(size * 0.04), 1)
    draw.line([(cx, cy-r), (cx, ant_top)], fill=INK, width=ant_w)

    # Antenna dot
    dot_r = max(int(size * 0.06), 1)
    draw.ellipse([cx-dot_r, ant_top-dot_r*2, cx+dot_r, ant_top], fill="#ffd870")

    img.save(os.path.join(OUT, filename), "PNG", optimize=True)
    return img


def make_favicon_ico():
    """Generate multi-resolution .ico file."""
    img16 = make_truthy_favicon(16, "favicon-16.png")
    img32 = make_truthy_favicon(32, "favicon-32.png")

    # Save as ICO with both sizes
    img16_copy = img16.copy()
    img16_copy.save(
        os.path.join(OUT, "favicon.ico"),
        format="ICO",
        sizes=[(16, 16), (32, 32)],
        append_images=[img32]
    )
    print("✓ favicon.ico (16+32)")
    print("✓ favicon-32.png")


def make_apple_touch():
    """180×180 apple touch icon."""
    size = 180
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded background
    draw.rounded_rectangle([0, 0, size-1, size-1], radius=36, fill=BG)
    draw.rounded_rectangle([0, 0, size-1, size-1], radius=36, outline=BORDER, width=2)

    cx, cy = size // 2, int(size * 0.52)
    r = int(size * 0.32)

    # Head
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=INK)

    # Visor
    v_h = int(size * 0.16)
    v_y = cy - v_h // 2 + 3
    v_margin = int(size * 0.06)
    draw.rounded_rectangle([cx-r+v_margin, v_y, cx+r-v_margin, v_y+v_h],
                           radius=v_h//2, fill="#2a2a2a")

    # Eyes
    eye_r = int(size * 0.05)
    eye_y = v_y + v_h // 2
    eye_spread = int(size * 0.12)
    for ex in [cx - eye_spread, cx + eye_spread]:
        draw.ellipse([ex-eye_r, eye_y-eye_r, ex+eye_r, eye_y+eye_r], fill="#50d8b0")
        # Highlight
        hr = max(eye_r // 2, 1)
        draw.ellipse([ex-hr-1, eye_y-hr-1, ex+hr-2, eye_y+hr-2], fill="#a0ffe0")

    # Antenna
    ant_top = cy - r - int(size * 0.1)
    draw.line([(cx, cy-r), (cx, ant_top)], fill=INK, width=3)
    dot_r = int(size * 0.04)
    draw.ellipse([cx-dot_r, ant_top-dot_r*2, cx+dot_r, ant_top], fill="#ffd870")
    # Glow
    gr = dot_r - 1
    draw.ellipse([cx-gr, ant_top-dot_r*2+1, cx+gr-1, ant_top-2], fill="#fff8d0")

    # Ears
    ear_h = int(size * 0.1)
    ear_w = int(size * 0.05)
    ear_y = cy - ear_h // 2
    for ear_x in [cx - r - ear_w + 2, cx + r - 2]:
        draw.ellipse([ear_x, ear_y, ear_x + ear_w, ear_y + ear_h], fill="#c9a158")

    img.save(os.path.join(OUT, "apple-touch-icon.png"), "PNG", optimize=True)
    print("✓ apple-touch-icon.png (180×180)")


if __name__ == "__main__":
    make_favicon_ico()
    make_truthy_favicon(16, "favicon-16.png")
    make_truthy_favicon(32, "favicon-32.png")
    make_apple_touch()
    print(f"\nAll assets in {OUT}/")
