"""
One-off: refresh index.html + assets (styles.css, truthbot.js) and post-process
existing report/claim HTML pages to add model adapter names to the
"Model reasoning" <summary> labels. This is needed because the local bundle
cache is empty, so regen_site.py can't rebuild per-claim HTML from scratch.

Safe to re-run: idempotent (won't double-label summaries).
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truthbot.publish.site import (  # noqa: E402
    CSS,
    JS,
    SitePublisher,
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
    reports_index = pub._load_reports_index()
    claims_index = pub._load_claims_index()
    stats = pub._compute_stats(reports_index, claims_index)
    (SITE / "index.html").write_text(_render_index(reports_index, stats), encoding="utf-8")
    (SITE / "about.html").write_text(_render_about(), encoding="utf-8")
    (SITE / "404.html").write_text(_render_404(), encoding="utf-8")
    print("refreshed: index.html, about.html, 404.html, assets/styles.css, assets/truthbot.js")


# Match a single <div class="model"> block up through its <summary>Model reasoning</summary>
# and inject the adapter name. Non-greedy so it pairs each name with the nearest
# following summary within the same model block.
_MODEL_SUMMARY_RE = re.compile(
    r'(<div class="model-name">)([^<]+)(</div>)(.*?)<summary>Model reasoning</summary>',
    re.DOTALL,
)


def _inject_model_label(html: str) -> tuple[str, int]:
    """Add ` — {adapter}` to each plain 'Model reasoning' summary. Idempotent."""
    # Load claims.json so we can swap the adapter label for the model_id when present.
    claims_data = []
    claims_json = SITE / "data" / "claims.json"
    if claims_json.exists():
        try:
            claims_data = json.loads(claims_json.read_text(encoding="utf-8"))
        except Exception:
            claims_data = []

    changes = 0

    def repl(m: re.Match) -> str:
        nonlocal changes
        open_tag, adapter, close_tag, middle = m.group(1), m.group(2), m.group(3), m.group(4)
        label = adapter.strip()
        changes += 1
        return (
            f'{open_tag}{adapter}{close_tag}{middle}'
            f'<summary>Model reasoning'
            f'<span class="model-reasoning-model"> \u2014 {label}</span>'
            f'</summary>'
        )

    # Only replace summaries that don't already carry a model-reasoning-model span.
    # We do a two-pass: skip spans already present.
    def guarded_repl(m: re.Match) -> str:
        if 'class="model-reasoning-model"' in m.group(0):
            return m.group(0)
        return repl(m)

    new_html = _MODEL_SUMMARY_RE.sub(guarded_repl, html)
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
