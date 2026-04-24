"""
One-off rebuild: refresh CSS/feed/static pages and patch existing report/claim
HTML to render the new Beta badge next to every "Pipeline vX.Y.Z" version
string.

Why a patcher: the bundle cache needed to re-render reports/*.html and
claims/*.html from scratch isn't available locally, and re-running the
pipeline would cost real API calls. Since the only markup deltas are surgical
(a `<meta name="generator">` suffix + a `<span class="beta-badge">` injected
after the version string), a regex patcher is sufficient and idempotent.

Idempotent: each patch checks for its marker before rewriting, so the script
is safe to re-run any number of times. When PIPELINE_VERSION crosses 1.0.0,
BETA_BADGE_HTML and BETA_TEXT_SUFFIX both become empty strings, and the
patches become no-ops.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truthbot.publish.site import (  # noqa: E402
    BETA_BADGE_HTML,
    BETA_TEXT_SUFFIX,
    PIPELINE_VERSION,
    SitePublisher,
    _render_404,
    _render_about,
    _render_index,
    _render_truthy,
)

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def refresh_assets_and_static_pages() -> None:
    """Rewrite CSS, truthbot.js, feed.xml, and the four static pages."""
    pub = SitePublisher(site_root=str(SITE))
    pub._ensure_structure()
    pub._copy_assets()  # CSS + JS + icons + social + feed.xml

    reports_index = pub._load_reports_index()
    claims_index = pub._load_claims_index()
    stats = pub._compute_stats(reports_index, claims_index)

    (SITE / "index.html").write_text(
        _render_index(reports_index, stats), encoding="utf-8"
    )
    (SITE / "about.html").write_text(_render_about(), encoding="utf-8")
    (SITE / "truthy.html").write_text(_render_truthy(), encoding="utf-8")
    (SITE / "404.html").write_text(_render_404(), encoding="utf-8")
    print(f"  Refreshed: index.html, about.html, truthy.html, 404.html")


def _patch_version_markup(html: str) -> tuple[str, int]:
    """
    Inject Beta markers for PIPELINE_VERSION in `html`. Returns (new_html, n_patches).

    Three render sites per page:
      1. `<meta name="generator" content="truth-bot X.Y.Z">` (HTML head)
      2. `Pipeline vX.Y.Z` (status bar + index/truthy footer + about prose)
      3. `pipeline vX.Y.Z` (report/claim/about footer — lowercase variant)

    Idempotent: skips any site already carrying `BETA_BADGE_HTML` or the
    `(beta)` suffix.
    """
    if not BETA_BADGE_HTML and not BETA_TEXT_SUFFIX:
        return html, 0

    n = 0
    v = re.escape(PIPELINE_VERSION)

    gen_pattern = re.compile(
        rf'<meta name="generator" content="truth-bot {v}">'
    )
    gen_replacement = (
        f'<meta name="generator" content="truth-bot {PIPELINE_VERSION}{BETA_TEXT_SUFFIX}">'
    )
    html, count = gen_pattern.subn(gen_replacement, html)
    n += count

    pipe_pattern = re.compile(rf'(Pipeline v{v})(?!{re.escape(BETA_BADGE_HTML)})')
    html, count = pipe_pattern.subn(rf'\1{BETA_BADGE_HTML}', html)
    n += count

    lower_pattern = re.compile(rf'(pipeline v{v})(?!{re.escape(BETA_BADGE_HTML)})')
    html, count = lower_pattern.subn(rf'\1{BETA_BADGE_HTML}', html)
    n += count

    return html, n


def patch_reports_and_claims() -> None:
    """Patch every existing report/claim HTML in place."""
    report_dir = SITE / "reports"
    claim_dir = SITE / "claims"

    patched = 0
    touched_files = 0
    for d in (report_dir, claim_dir):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.html")):
            original = p.read_text(encoding="utf-8")
            new_html, n = _patch_version_markup(original)
            if n and new_html != original:
                p.write_text(new_html, encoding="utf-8")
                patched += n
                touched_files += 1

    print(f"  Patched {patched} version markers across {touched_files} HTML files")


def main() -> None:
    print(f"Rebuilding site at {SITE} (PIPELINE_VERSION={PIPELINE_VERSION})")
    print(f"  BETA_BADGE_HTML={BETA_BADGE_HTML!r}")
    print(f"  BETA_TEXT_SUFFIX={BETA_TEXT_SUFFIX!r}")
    refresh_assets_and_static_pages()
    patch_reports_and_claims()
    print("Done.")


if __name__ == "__main__":
    main()
