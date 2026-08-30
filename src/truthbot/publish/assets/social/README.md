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

## Regenerating

There is no script yet — these were produced interactively. If they need to
change, the mascot SVG and stylesheet are the source, and the framing constants
above are what matter. Worth a script if it happens a third time.
