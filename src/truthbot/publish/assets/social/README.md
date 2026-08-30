# Social brand assets

Committed rather than generated at publish time: these are uploaded by hand to a
social profile, not served by the site, so rebuilding them on every render would
be churn no reader ever sees.

| File | Size | Where it goes |
|---|---|---|
| `truthy-avatar.png` | 1024x1024 | Bluesky profile avatar |
| `truthy-banner.png` | 1500x500 | Bluesky profile banner |
| `truthy-fullbody.png` | 1200x1440 | source for the banner and the house share card |

The sibling `../apple-touch-icon.png` is built by the same script.

## Regenerating

```
python3 scripts/build_brand_assets.py            # rebuild all
python3 scripts/build_brand_assets.py --check    # report drift, write nothing
```

**This README used to say "there is no script yet — these were produced
interactively", and that was the actual problem.** Every constant that mattered
lived in a chat log, so a one-line change to the mascot SVG (removing a specular
highlight from his forehead) meant re-deriving all of them by hand for the third
time. The script is now the source of the recipe; this file only explains the
decisions behind it.

All of it is rendered from the **live mascot SVG** (`site._TRUTHY_SVG`) plus the
real site CSS, via headless Chromium. Nothing redraws Truthy by hand, so the
brand cannot drift from the mascot the site actually shows.

## The decisions worth keeping

**Head-only avatar.** Bluesky renders avatars at roughly **42px in the feed**,
where a full body becomes an indistinct smudge and the eyes vanish. The head
still reads as a face. Both arms are hidden — the left one is raised by the wave
pose, which puts a disembodied forearm in the bottom-left of the crop.

**Full body everywhere else.** On the banner and the 1200x630 house card there is
room, and the head alone omits the clipboard, which is the one part of him that
says what the site does.

**Neutral eyes, pinned.** The site cycles the LED eyes between moods; a still has
to freeze one frame. Freezing animations also leaves the waving arm untransformed
and hanging, so the script pins the wave's own rest keyframe (`rotate(130deg)`).

**The banner's lower-left is deliberately empty** — Bluesky overlays the avatar
there.

**The banner's verdict bar is a point-in-time snapshot** and the only figure on
any brand asset. The previous banner encoded 58.9% True against a corpus that was
57.8%, and dropped `Models split` entirely — drift nothing could catch, since
these files are uploaded by hand and no lint reads a PNG. It is recomputed on
every build, so it is at least true when made. Re-run after new reports land.

## The BETA stamp

Both carry a rotated BETA stamp in `--v-false` red (#991b1b).

Red was argued against first, on the grounds that red means FALSE in the verdict
palette and a red BETA might read as a negative marker. That objection was wrong
for a reason already sitting in this codebase: the wordmark's `·` is drawn in
that same red, in `report_cards._chrome()` and in the site's `.wordmark .dot`
rule. The brand already borrows verdict-red for chrome. The context that makes
red mean FALSE is a report page with verdict pills beside it; a profile has none.
And red is what makes it read as a *stamp* rather than a UI chip — the
neutral-ink version looked like a badge.

**The avatar placement is SOLVED at build time, never stored.** The constraints
pull against each other: the head is an ellipse at centre (512,557) radii
(410,381), the crop circle is centre (512,512) radius 512, and lifting the stamp
clear of the skull pushes it toward the crop boundary. Only a narrow band
satisfies both — about 9px of margin at the boundary. That is far too tight for a
constant that gets hand-nudged; a stored one from an earlier build re-measured at
218x110 against a 176x110 solution and clipped the crop by 6px.

Three earlier placements were wrong and are worth not repeating:

- **Dead-periphery**, in the crescent between head-edge and crop-edge: clips
  (558 against a 512 radius).
- **Lower jaw**: fits, but reads as something stuck to his face rather than an
  annotation on the portrait, and at 96px it blends into the head.
- **Bordering the skull**: overlapped it by ~24px.

It is legible at profile size, a red mark in the corner at 96px, and gone at 42px
— decorative, not load-bearing, which is the right weight for it.

### Verifying it, if you ever need to

Check the stamp against **the tile's own alpha channel**, not against colour in
the composited PNG. Two different colour heuristics were tried here and both were
wrong in ways that looked authoritative: `r-g>40 and r-b>40` matches Truthy's
gold antenna bulb (244,200,106), and `r>110 and g<95 and b<95` matches the dark
shading on his head. Both reported overlaps that did not exist. Measuring the
axis-aligned bounding box of a *rotated* stamp is wrong for the same family of
reasons — its corners are empty space, so it overstates the reach by ~40px.
