The truth-bot site in `site-test/` needs social sharing infrastructure, a favicon, and a few trust improvements. Pre-built image assets and templates are in `social-assets/` — just copy them into place. Nothing here touches the existing stat icons, Truthy mascot, or verdict colors — this is all additive.

## Pre-built assets (in `social-assets/`, ready to copy)
- `social-card.png` — 1200×630 OG/Twitter social preview image
- `favicon.ico` — multi-resolution 16+32px
- `favicon-32.png` — 32×32 PNG favicon
- `apple-touch-icon.png` — 180×180 iOS touch icon
- `feed.xml` — Atom feed template with existing report entry
- `gen_assets.py` — Python script to regenerate images if needed

## Current state (already done, don't redo)
- Stat icons with Approach B (converging bots) are live on index, report, and claim pages
- "How it works" trust strip is live on the index page
- `theme-color` and `color-scheme` meta tags are set

## What's missing and needs implementing

---

### 1. Open Graph + Twitter Card meta tags

Add these to the `<head>` of EVERY page. The values should be page-specific where noted.

**index.html:**
```html
<!-- Open Graph -->
<meta property="og:type" content="website">
<meta property="og:site_name" content="truth-bot">
<meta property="og:title" content="truth-bot — Automated Political Fact-Checking">
<meta property="og:description" content="Multi-model AI consensus analysis of political speeches. Every claim decomposed, verified against primary sources, and scored for accuracy.">
<meta property="og:image" content="./assets/social-card.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="truth-bot: automated political fact-checking with multi-model consensus">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="truth-bot — Automated Political Fact-Checking">
<meta name="twitter:description" content="Multi-model AI consensus analysis of political speeches. Every claim verified against primary sources.">
<meta name="twitter:image" content="./assets/social-card.png">
<meta name="twitter:image:alt" content="truth-bot: automated political fact-checking with multi-model consensus">
```

**Report pages** (e.g. `reports/2026-03-04-donald-trump-165937.html`):
Customize the title and description per report. Use this pattern:
```html
<meta property="og:type" content="article">
<meta property="og:site_name" content="truth-bot">
<meta property="og:title" content="Donald Trump — March 04, 2026 — truth-bot">
<meta property="og:description" content="5 claims checked. Verdict: Largely False (3 of 5 claims). Multi-model AI fact-check with primary source verification.">
<meta property="og:image" content="../assets/social-card.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Donald Trump — March 04, 2026 — truth-bot">
<meta name="twitter:description" content="5 claims checked. Verdict: Largely False (3 of 5 claims). Multi-model AI fact-check.">
<meta name="twitter:image" content="../assets/social-card.png">
```

**Claim pages** (in `claims/`):
```html
<meta property="og:type" content="article">
<meta property="og:site_name" content="truth-bot">
<meta property="og:title" content="Claim: [first 60 chars of claim text] — truth-bot">
<meta property="og:description" content="Verdict: [verdict]. [number] of [total] models agree. Verified against primary government sources.">
<meta property="og:image" content="../assets/social-card.png">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Claim: [first 60 chars] — truth-bot">
<meta name="twitter:description" content="Verdict: [verdict]. Multi-model AI fact-check.">
<meta name="twitter:image" content="../assets/social-card.png">
```

**about.html:**
```html
<meta property="og:type" content="website">
<meta property="og:site_name" content="truth-bot">
<meta property="og:title" content="About — truth-bot">
<meta property="og:description" content="How truth-bot works: atomic claim decomposition, multi-model verification against government primary sources, and transparent consensus scoring.">
<meta property="og:image" content="./assets/social-card.png">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="About — truth-bot">
<meta name="twitter:description" content="How truth-bot works: multi-model AI fact-checking with full methodology transparency.">
<meta name="twitter:image" content="./assets/social-card.png">
```

---

### 2. Social card image (1200×630 PNG)

A pre-built social card is provided at `social-assets/social-card.png` — copy it to `site-test/assets/social-card.png`. No generation needed.

The `social-assets/gen_assets.py` script is also included if you need to regenerate or customize it later. It uses Pillow and the system's Liberation font family.

---

### 3. Favicon + touch icons

Pre-built favicon assets are provided in `social-assets/`:
- `favicon.ico` (16×16 and 32×32 multi-resolution) → copy to `site-test/favicon.ico`
- `favicon-32.png` (32×32) → copy to `site-test/assets/favicon-32.png`
- `apple-touch-icon.png` (180×180) → copy to `site-test/assets/apple-touch-icon.png`

Design: simplified Truthy head silhouette — round head, visor band with green LED eyes, antenna dot. Solid black on transparent background.

Add to `<head>` of ALL pages:
```html
<link rel="icon" href="./favicon.ico" sizes="any">
<link rel="icon" href="./assets/favicon-32.png" type="image/png" sizes="32x32">
<link rel="apple-touch-icon" href="./assets/apple-touch-icon.png">
```
(Use `../` prefix for pages in subdirectories like `reports/` and `claims/`.)

---

### 4. RSS/Atom feed

A pre-built Atom feed template is provided at `social-assets/feed.xml` — copy it to `site-test/feed.xml`.

It contains the existing Trump report entry and a comment marking where the pipeline should append new `<entry>` blocks. Replace `[SITE_URL]` with the production domain when it's set.

Add to `<head>` of `index.html`:
```html
<link rel="alternate" type="application/atom+xml" title="truth-bot feed" href="./feed.xml">
```

---

### 5. Prompt hash in footer (trust signal)

The about page already publishes the full verdict prompt with hash `39b42838`. Surface this in the site footer as a subtle trust signal.

In the `<footer>` of ALL pages, add the prompt hash as a link to the about page methodology section:

Change the existing footer from:
```
Pipeline v0.2.0 · GitHub
```
to:
```
Pipeline v0.2.0 · Prompt 39b42838 · GitHub
```

Where "Prompt 39b42838" links to `about.html#prompt` (add `id="prompt"` to the prompt hash `<h3>` on the about page if not already present).

Style the hash in mono font, same as the pipeline version — `font-family: var(--mono); color: var(--ink-faint)`.

---

### 6. Source tier summary on index page report cards

On the index page, each report card (`.report`) currently shows the verdict bar and claim counts. Add a one-line source tier summary to the `.report-cta` row, showing what kinds of sources backed the analysis.

Example markup to add inside `.report-cta`:
```html
<span class="src-tiers">Sources: 3 gov · 2 wire · 1 news</span>
```

Style it with:
```css
.src-tiers {
  font-family: var(--mono);
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-faint);
}
```

To get the actual tier counts, tally the evidence tier badges from the report's claim pages (`.evidence-tier.tier-gov`, `.tier-wire`, `.tier-news`, `.tier-fc`, etc.). If this is too complex to automate right now, hardcode the counts for the existing Trump report and leave a `<!-- TODO: generate from claims data -->` comment.

---

## Constraints
- Do NOT touch stat icons, Truthy mascot, verdict colors, or animation scripts
- Social card must follow the site's "only chromatic colors are verdicts" design rule
- All new meta tags go in `<head>` AFTER existing meta tags, BEFORE the stylesheet link
- Favicon and social card paths must work for both root pages (./assets/) and subdir pages (../assets/)
- RSS feed uses placeholder `[SITE_URL]` until production domain is configured
