# truth-bot UI redesign — implementation brief

## Context

We did a design pass on the truth-bot static site, replacing the current newspaper/editorial CSS with an "accountability dashboard" aesthetic: restrained chrome, verdict colors as the only chroma, Newsreader serif + Geist sans + Geist Mono typography. Truthy McTruthface is integrated into the verdict panel as the aggregate-avatar, with claim-count-aware captions and Web Audio droid sounds (click-to-play, no autoplay, keyboard-accessible).

Two reference HTML files are the design source-of-truth — read them first before touching anything:

- `prototypes/truth-bot-index.html` (reports list)
- `prototypes/truth-bot-report-v2.html` (individual report with Truthy)

Goal of this work: port the prototypes into the Python generator so every page renders this way. **No data-flow changes** — claim extraction, model verdicts, scoring, and mood computation all stay as-is. This is presentation-layer only.

## Files in this handoff package

- `HANDOFF.md` — this brief
- `assets/styles.css` — consolidated stylesheet, ready to drop into the project (replaces existing `assets/styles.css`)
- `assets/truthbot.js` — Truthy state machine + Web Audio sounds (replaces existing `assets/truthbot.js`)
- `prototypes/truth-bot-index.html` — index page reference (self-contained, inline CSS/JS)
- `prototypes/truth-bot-report-v2.html` — report page reference (self-contained, inline CSS/JS)

The CSS in `assets/styles.css` has already been deduplicated and consolidated from both prototypes. The stylesheet is sectioned with comments — read the section index at the top to navigate. The prototypes have inline copies for self-containment; both sources will produce visually identical output.

## Files to modify in the truth-bot repo

- `assets/styles.css` — replace contents with the consolidated stylesheet from this package
- `assets/truthbot.js` — replace contents with the v2 Truthy logic from this package
- Generator templates (whichever Python module emits index / report / about) — update HTML structure to match the prototype markup

Add Google Fonts preconnect + stylesheet link to the `<head>` of every page:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,ital,wght@6..72,0,400;6..72,0,500;6..72,0,600;6..72,0,700;6..72,1,400;6..72,1,500&family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap">
```

## Design tokens

These live in `:root` at the top of `styles.css`. **Never hardcode hex values anywhere else in the codebase.** Change one variable and every bar / pill / swatch / dissent flag / Truthy bubble tint updates.

```css
:root {
  --bg: #fafaf9;
  --surface: #ffffff;
  --surface-warm: #faf8f3;
  --ink: #0c0a09;
  --ink-muted: #57534e;
  --ink-faint: #a8a29e;
  --border: #e7e5e4;
  --border-strong: #d6d3d1;

  /* Verdict palette — ONLY chromatic colors in the design */
  --v-true:         #15803d;
  --v-mostly-true:  #65a30d;
  --v-exaggerated:  #ca8a04;
  --v-misleading:   #c2410c;
  --v-false:        #991b1b;
  --v-unverifiable: #44403c;
}
```

Map the verdict labels in the data model to the corresponding CSS classes. The `.v-{verdict}` classes paint backgrounds (used on bar segments and pills). The `.vt-{verdict}` classes paint text color (used on headline labels and verdict words inside reasoning blocks).

## Truthy generator requirements

The Truthy widget element needs **two attributes** set per page:

- `data-mood` — one of `happy` / `iffy` / `sad` (already computed by the existing aggregate-score logic; keep it)
- `data-claim-count` — integer; the number of claims being aggregated. **Use `1` for single-claim inputs** (a tweet, a single statement) so Truthy says "that" instead of "this". Use the actual count for speeches.

Everything else — the state machine, blink scheduler, eye animations, falling tears, arm/clipboard repositioning, and Web Audio sounds — works off these two attributes. No additional generator-side logic needed.

## Headline verdict logic (per report)

The "Largely False" / "Mostly X" / "Mixed verdict" label above the verdict bar is computed from claim distribution. Implement in Python during page generation:

- max single verdict ≥ 60% → `"Largely {Verdict}"` with `vt-{verdict}` class
- max single verdict ≥ 40% → `"Mostly {Verdict}"` with `vt-{verdict}` class
- otherwise → `"Mixed verdict"` with `.neutral` class (uses ink color, not a verdict color)

## Component checklist

- [ ] Status bar (terminal-style strip, pulsing green live dot)
- [ ] Masthead: full on index (`.wordmark` + tagline + `.top-nav`), compact on report pages (`.wordmark-sm` + `.breadcrumb`)
- [ ] Speech hero: speaker line, headline (`.speech-title` Newsreader 3rem), meta row
- [ ] Verdict panel: 2-column layout (verdict text + Truthy column), collapses to stacked-with-inline-Truthy on mobile
- [ ] Verdict bar with full 6-category legend; zero-count items dim out but stay visible (consistent legend across reports)
- [ ] Source row (transcript, video, models) at bottom of verdict panel
- [ ] TOC with one row per claim: number, verdict pill, truncating text, jump arrow
- [ ] Claim cards: numbered head, serif quote with left rule, caveat callout, model grid, expandable reasoning via `<details>`, evidence list with tier badges, permalink footer
- [ ] Dissent UI: any non-majority model gets `.dissent` class — warm background + "DISSENT" tag in corner via `::after`
- [ ] Methodology callout above footer
- [ ] Footer with pipeline version + GitHub link

## Out of scope — DO NOT implement these now

These are tracked but explicitly deferred to keep MVP scope tight:

1. Iffy-state caption refinement: splitting exaggerated / misleading / mixed into different Truthy voices
2. False-state intensity tiers: "*most* doesn't check out" (60–80%) vs "*almost none* checks out" (>80%)
3. Index page facets: filter aggregate by speaker / org / date range
4. Per-claim share button: copy anchor link, generate social-card screenshot
5. TOC filter for long reports (40+ claims): show only False / Exaggerated etc.
6. Expanding Truthy from 3 moods to 6 (one per verdict)

If you see opportunities to implement any of the above, **don't**. Add a TODO comment instead.

## Acceptance criteria

- Every existing report re-renders via the generator with the new design (no manual editing of output HTML)
- No JavaScript console errors on any page
- Truthy plays the appropriate droid sound on click in Chrome, Safari, and Firefox
- Layout is usable at 360px wide (test with browser dev tools, iPhone SE viewport)
- Verdict colors are the only non-grayscale chroma in the page chrome — the only exception is the red period after "truth-bot" in the wordmark, which is intentional (driven by `--v-false` so it stays in sync)
- All evidence links open in new tabs with `rel="noopener"`
- Keyboard navigation works for the TOC anchors and the Truthy button (Enter/Space triggers speak)

## Suggested execution order

1. Drop in the design tokens and Google Fonts link first — confirm fonts load by inspecting computed style on a heading
2. Update masthead + status bar (cheapest visible win, lets you validate the look before deeper template work)
3. Update report page verdict panel with Truthy integration
4. Update claim card template
5. Update index page (reports list)
6. Sweep for any hardcoded colors that should be CSS variables
7. Run the existing pipeline on existing data; visually diff against prototypes

Read both prototype HTML files end-to-end before starting. The CSS is heavily commented and the structure is what you should mirror exactly.
