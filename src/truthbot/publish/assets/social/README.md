# Social brand assets

Committed rather than generated at publish time: these are uploaded by hand to a
social profile, not served by the site, so regenerating them on every render
would be churn for no reader.

| File | Size | Where it goes |
|---|---|---|
| `truthy-avatar.png` | 1024x1024 | Bluesky profile avatar |
| `truthy-banner.png` | 1500x500 | Bluesky profile banner |
| `truthy-fullbody.png` | 1200x1440 | source for the banner; transparent-ish full figure |

## How they were made

All three are rendered from the **live mascot SVG** in the published site, not
redrawn — headless Chromium over a page that inlines `site-pca/truthy.html`'s
`<svg id="mascot">` plus the real `assets/styles.css`. So they cannot drift from
the site's own Truthy in style, only in staleness.

**Avatar** is head-only, framed by geometry rather than by eye: the head is an
ellipse at `cx=150 cy=148 rx=100 ry=93`, so `viewBox="25 12 250 250"` centres
head-and-antenna with enough margin to survive the circle Bluesky inscribes in
the square. `#armRight` and `#clipboard` are hidden — at avatar scale the
clipboard survives the circular crop only as an unidentifiable fragment.

Head-only is deliberate. Bluesky renders avatars at roughly **42px in the feed**,
where a full body becomes an indistinct smudge and the eyes disappear entirely.
The head still reads as a face at that size.

**Banner** uses the full figure in the `hero-wave` state — both arms raised,
clipboard up — with the neutral LED eyes pinned on (the site cycles them in and
out via `true-neutral-cycle`; the still freezes that frame). The bottom rule is
the site's own verdict bar at real corpus proportions. The lower-left is kept
empty on purpose: that is where Bluesky overlays the avatar.

## The BETA stamp

Both carry a rotated BETA stamp in `--v-false` red (#991b1b) at ~215/230 alpha.

Red was argued against first, on the grounds that red means FALSE in the verdict
palette and a red BETA might read as a negative marker. That objection was
wrong, for a reason already sitting in this codebase: the wordmark's `·` is
drawn in that same red, in `report_cards._chrome()` and in the site's
`.wordmark .dot` rule. The brand already borrows verdict-red for chrome. The
context that makes red mean FALSE is a report page with verdict pills beside it;
a profile banner has none. And red is what makes it read as a *stamp* rather
than a UI chip -- the neutral-ink version looked like a badge.

On the avatar the stamp floats in the white space ABOVE-RIGHT of the head,
touching nothing -- clear of the skull, clear of the antenna, inside the crop.

That position was SOLVED, not nudged, and it is worth re-solving rather than
hand-adjusting if anything about the framing changes. The constraints are in
tension: the head is an ellipse at centre (512,557) radii (410,381) in the
1024px frame, and the crop circle is centre (512,512) radius 512. Moving the
stamp up off the skull pushes it toward the circle boundary, so there is only a
narrow band that satisfies both. A grid search over size and position -- every
sampled point outside the head ellipse plus 10px clearance, all four corners
inside the circle with 8px margin, and clear of an antenna keep-out box of
x 480-545 / y 80-190 -- lands on size 48 centred at (635,110), farthest corner
503 against the 512 radius.

It is TIGHT: 9px of margin at the crop boundary. Larger clips; lower touches
the head.

Three earlier placements were wrong and are worth not repeating. Dead-periphery,
in the crescent between head-edge and crop-edge, CLIPS (558 against a 512
radius). The lower jaw fits but reads as something stuck to his face rather than
an annotation on the portrait, and at 96px it blends into the head. Above-right
but bordering the skull overlapped it by ~24px.

It is legible at profile size, a red mark in the corner at 96px, and gone at
42px -- decorative, not load-bearing, which is the right weight for it.

## Regenerating

There is no script yet — these were produced interactively. If they need to
change, the mascot SVG and stylesheet are the source, and the framing constants
above are what matter. Worth a script if it happens a third time.
