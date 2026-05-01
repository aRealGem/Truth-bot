# Gemini temporal-dismissal — spot check (cached demo + literature)

## External documented failure mode

[`eval/sotu-2026/findings-review.md`](../../eval/sotu-2026/findings-review.md) §C3: Gemini search can run but dismisses real reporting when article dates read as post–training-cutoff, producing **fiction / speculative framing** for genuine 2025–2026 events (example cited: Operation Midnight Hammer narrative).

That pattern is **not** reliably reproducible from the current `site-test/` demo export alone (demo HTML is regenerated from bundled cache focused on verdict presentation).

## Local scan (`site-test/claims/*.html`)

Scripted skim of published Gemini reasoning blocks (`model-name=g`) for dismissal regexes `(fiction|speculative|war game)` over the 2026-04 demo set did **not** surface a clean match; Gemini text in-sample reads like ordinary evidentiary disagreement (e.g. investment totals claim `6970524a-5432-4cbf-b179-da27533b205d.html` — methodological critique, not temporal denial).

## Conclusion for interpretability budgeting

Treat temporal-dismissal as a **risk to watch on fresh live Gemini calls** with post-cutoff current-events claims; use sidecar excerpts + timestamps when validating new adapter releases. Cached static HTML is a poor detector for this failure mode unless the underlying run captured it.
