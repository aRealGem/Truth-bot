"""
One-off: refresh index.html + assets (styles.css, truthbot.js) and post-process
existing report/claim HTML pages to add model adapter names to the
"Model reasoning" <summary> labels. This is needed because the local bundle
cache is empty, so regen_site.py can't rebuild per-claim HTML from scratch.

Safe to re-run: idempotent (won't double-label summaries).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truthbot.publish.site import (  # noqa: E402
    CSS,
    JS,
    SitePublisher,
    _pretty_model_label,
    _render_index,
    _render_about,
    _render_404,
)

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site-test"

def refresh_assets_and_index() -> None:
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    (SITE / "assets" / "styles.css").write_text(CSS, encoding="utf-8")
    (SITE / "assets" / "truthbot.js").write_text(JS, encoding="utf-8")

    pub = SitePublisher(site_root=str(SITE))
    pub._copy_icons()
    reports_index = pub._load_reports_index()
    claims_index = pub._load_claims_index()
    stats = pub._compute_stats(reports_index, claims_index)
    (SITE / "index.html").write_text(_render_index(reports_index, stats), encoding="utf-8")
    (SITE / "about.html").write_text(_render_about(), encoding="utf-8")
    (SITE / "404.html").write_text(_render_404(), encoding="utf-8")
    print(
        "refreshed: index.html, about.html, 404.html, "
        "assets/styles.css, assets/truthbot.js, assets/icons/*.svg"
    )


# Match a single <div class="model"> block through its first Model-reasoning
# <summary>, whether the summary is still plain ("Model reasoning") or already
# carries an older "— adapter" span. Non-greedy so each .model-name pairs with
# its own model's summary.
_MODEL_SUMMARY_RE = re.compile(
    r'(<div class="model-name">)([^<]+)(</div>)'
    r'(?P<middle>.*?)'
    r'<summary>Model reasoning(?P<existing_span><span class="model-reasoning-model">[^<]*</span>)?</summary>',
    re.DOTALL,
)


def _inject_model_label(html: str) -> tuple[str, int]:
    """Label each 'Model reasoning' summary with the pretty provider+model name.

    Rewrites the span every time so we can upgrade older ad-hoc labels
    (e.g. '— anthropic') to the new pretty form ('— Anthropic Claude Opus 4.7').
    """
    changes = 0

    def repl(m: re.Match) -> str:
        nonlocal changes
        open_tag, adapter, close_tag = m.group(1), m.group(2), m.group(3)
        middle = m.group("middle")
        pretty = _pretty_model_label(adapter.strip())
        changes += 1
        return (
            f'{open_tag}{adapter}{close_tag}{middle}'
            f'<summary>Model reasoning'
            f'<span class="model-reasoning-model"> \u2014 {pretty}</span>'
            f'</summary>'
        )

    new_html = _MODEL_SUMMARY_RE.sub(repl, html)
    return new_html, changes


def patch_existing_html() -> None:
    targets = list((SITE / "reports").glob("*.html")) + list((SITE / "claims").glob("*.html"))
    total = 0
    for path in targets:
        before = path.read_text(encoding="utf-8")
        after, n = _inject_model_label(before)
        if n and after != before:
            path.write_text(after, encoding="utf-8")
            total += n
            print(f"patched {path.relative_to(ROOT)} ({n} summaries)")
    print(f"done: {total} summaries labelled across {len(targets)} files")


if __name__ == "__main__":
    refresh_assets_and_index()
    patch_existing_html()
