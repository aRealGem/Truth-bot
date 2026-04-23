#!/usr/bin/env python3
"""Generate truth-bot social card (1200×630) and favicons."""

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


def make_social_card():
    """1200×630 OG/Twitter social preview card."""
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # --- Subtle border ---
    draw.rectangle([0, 0, W-1, H-1], outline=BORDER, width=1)

    # --- Top status bar strip ---
    draw.rectangle([0, 0, W, 28], fill=INK)
    f_status = ImageFont.truetype(MONO, 11)
    draw.text((20, 7), "● OPERATIONAL", fill="#4ade80", font=f_status)
    draw.text((170, 7), "PIPELINE V0.2.0", fill="#d6d3d1", font=f_status)
    draw.text((370, 7), "MULTI-MODEL CONSENSUS", fill="#d6d3d1", font=f_status)

    # --- Wordmark ---
    f_word = ImageFont.truetype(SERIF_BOLD, 48)
    draw.text((60, 70), "truth-bot", fill=INK, font=f_word)
    # The signature dot in a warm tone
    dot_x = 60 + draw.textlength("truth-bot", font=f_word)
    draw.text((dot_x, 70), ".", fill="#c9a158", font=f_word)

    # --- Tagline ---
    f_tag = ImageFont.truetype(SANS, 19)
    draw.text((62, 132), "Automated political fact-checking with multi-model consensus analysis.", fill=INK_MUT, font=f_tag)

    # --- Simplified Truthy silhouette (right side) ---
    tx, ty = 920, 120  # center of head
    # Head (large circle)
    r_head = 95
    draw.ellipse([tx-r_head, ty-r_head+15, tx+r_head, ty+r_head+15],
                 fill=None, outline=INK, width=2)
    # Fill head with very light opacity effect
    for i in range(r_head, 0, -1):
        alpha = int(12 + (r_head - i) * 0.15)
        c = f"#{alpha:02x}{alpha:02x}{alpha:02x}"
        # Just use a subtle fill
    draw.ellipse([tx-r_head+2, ty-r_head+17, tx+r_head-2, ty+r_head+13],
                 fill="#f0ebe3", outline=None)
    draw.ellipse([tx-r_head, ty-r_head+15, tx+r_head, ty+r_head+15],
                 fill=None, outline=INK, width=2)

    # Visor band
    v_top = ty - 15
    v_h = 50
    draw.rounded_rectangle([tx-80, v_top, tx+80, v_top+v_h], radius=25,
                           fill="#1a1410", outline=None)
    # Eyes in visor (LED style)
    eye_y = v_top + v_h//2
    for ex in [tx-30, tx+30]:
        draw.rounded_rectangle([ex-10, eye_y-12, ex+10, eye_y+12], radius=6,
                               fill="#50d8b0", outline=None)
        # Highlight
        draw.ellipse([ex-4, eye_y-6, ex+2, eye_y-2], fill="#a0ffe0")

    # Antenna
    draw.line([(tx, ty-r_head+15), (tx+2, ty-r_head-20)], fill=INK, width=3)
    draw.ellipse([tx-4, ty-r_head-28, tx+8, ty-r_head-16], fill="#ffd870")
    # Antenna glow dot
    draw.ellipse([tx-1, ty-r_head-25, tx+5, ty-r_head-19], fill="#fff8d0")

    # Ears
    for ear_x in [tx-r_head-8, tx+r_head-2]:
        draw.ellipse([ear_x, ty-5, ear_x+12, ty+30], fill="#c9a158", outline="#8a7550", width=1)

    # Body (below head)
    body_top = ty + r_head + 10
    draw.rounded_rectangle([tx-65, body_top, tx+65, body_top+55], radius=12,
                           fill="#f0ebe3", outline="#8a7550", width=2)
    # Nameplate
    draw.rounded_rectangle([tx-30, body_top+5, tx+30, body_top+17], radius=2,
                           fill="#c9a158", outline="#8a7550", width=1)
    f_name = ImageFont.truetype(SERIF, 9)
    draw.text((tx-18, body_top+5), "Truthy M.", fill="#3a2e1f", font=f_name)

    # Chest LED
    draw.ellipse([tx+25, body_top+25, tx+41, body_top+41], fill="#5ac075", outline="#2a7840", width=1)
    draw.ellipse([tx+29, body_top+28, tx+35, body_top+34], fill="#b8f5c8")

    # --- "How it works" mini strip ---
    strip_y = 195
    f_how = ImageFont.truetype(SANS, 14)
    f_num = ImageFont.truetype(MONO_BOLD, 12)
    steps = [
        "Decompose speech into claims",
        "Verify each with multiple AI models",
        "Aggregate into consensus verdict",
    ]
    sx = 62
    for i, step in enumerate(steps):
        # Number circle
        draw.ellipse([sx, strip_y, sx+20, strip_y+20], fill=BORDER)
        draw.text((sx+6, strip_y+2), str(i+1), fill=INK_FAINT, font=f_num)
        # Step text
        draw.text((sx+28, strip_y+2), step, fill=INK_MUT, font=f_how)
        text_w = draw.textlength(step, font=f_how)
        # Arrow
        if i < 2:
            arrow_x = sx + 28 + text_w + 15
            draw.text((arrow_x, strip_y+1), "→", fill=INK_FAINT, font=f_how)
            sx = arrow_x + 25
        else:
            sx = sx + 28 + text_w + 15

    # --- Mock report card area ---
    card_y = 270
    draw.rectangle([60, card_y, W-60, card_y+1], fill=BORDER)

    # Report header
    f_section = ImageFont.truetype(MONO, 11)
    draw.text((62, card_y+12), "LATEST REPORT", fill=INK_FAINT, font=f_section)

    f_name_lg = ImageFont.truetype(SANS_BOLD, 26)
    draw.text((62, card_y+35), "Donald Trump", fill=INK, font=f_name_lg)

    f_meta = ImageFont.truetype(SANS, 14)
    draw.text((62, card_y+68), "2026-03-04  ·  Joint Session of Congress  ·  5 claims", fill=INK_MUT, font=f_meta)

    # Verdict pill
    pill_x = 62
    pill_y = card_y + 100
    draw.rounded_rectangle([pill_x, pill_y, pill_x+130, pill_y+28], radius=3,
                           fill=V_FALSE, outline=None)
    f_pill = ImageFont.truetype(SANS_BOLD, 13)
    draw.text((pill_x+10, pill_y+6), "Largely False", fill="#ffffff", font=f_pill)
    f_ratio = ImageFont.truetype(SANS, 13)
    draw.text((pill_x+142, pill_y+7), "3 of 5 claims", fill=INK_MUT, font=f_ratio)

    # Verdict bar (full width)
    bar_y = card_y + 145
    bar_h = 10
    bar_left = 62
    bar_right = W - 62
    bar_total = bar_right - bar_left
    # 20% true, 20% mostly-true, 60% false
    segs = [(0.20, V_TRUE), (0.20, V_MOST), (0.60, V_FALSE)]
    cx = bar_left
    for pct, color in segs:
        seg_w = int(bar_total * pct)
        draw.rectangle([cx, bar_y, cx+seg_w, bar_y+bar_h], fill=color)
        cx += seg_w

    # Legend below bar
    f_legend = ImageFont.truetype(SANS, 12)
    lx = 62
    for label, color, count in [("True", V_TRUE, "1"), ("Mostly True", V_MOST, "1"), ("False", V_FALSE, "3")]:
        draw.rectangle([lx, bar_y+20, lx+10, bar_y+30], fill=color)
        draw.text((lx+14, bar_y+18), f"{label} {count}", fill=INK_MUT, font=f_legend)
        lx += draw.textlength(f"{label} {count}", font=f_legend) + 40

    # --- Bottom verdict color strip (full width) ---
    strip_h = 6
    strip_top = H - strip_h
    total_w = W
    cx = 0
    for pct, color in [(0.25, V_TRUE), (0.25, V_MOST), (0.50, V_FALSE)]:
        seg_w = int(total_w * pct)
        draw.rectangle([cx, strip_top, cx+seg_w, H], fill=color)
        cx += seg_w

    # --- Footer text ---
    f_foot = ImageFont.truetype(MONO, 11)
    draw.text((62, H-28), "PIPELINE V0.2.0  ·  PROMPT 39B42838  ·  GITHUB.COM/AREALGEM/TRUTH-BOT", fill=INK_FAINT, font=f_foot)

    img.save(os.path.join(OUT, "social-card.png"), "PNG", optimize=True)
    print("✓ social-card.png (1200×630)")


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
    make_social_card()
    make_favicon_ico()
    make_truthy_favicon(16, "favicon-16.png")
    make_truthy_favicon(32, "favicon-32.png")
    make_apple_touch()
    print(f"\nAll assets in {OUT}/")
