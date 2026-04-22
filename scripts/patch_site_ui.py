"""
One-off: refresh index.html + assets (styles.css, truthbot.js) and post-process
existing report/claim HTML pages to:

  1. Label each "Model reasoning" <summary> with the pretty provider+model name.
  2. Add id="top" to the report hero so per-claim "Top of page" links resolve.
  3. Add id="claim-catalog" + small claim icon to the "Jump to claim"
     section-head (report pages only).
  4. Wrap each .vp-stats child in .vp-stat and prepend a small icon
     (claims / placeholder / consensus).
  5. Prepend a small claim icon inside each .claim-head, grouped in
     .claim-head-lead with the existing .claim-num.
  6. Inject per-claim-card "Back to claim list" + "Top of page" links in
     each .claim-foot (report pages only; standalone claim pages skip this).

Needed because the local bundle cache is empty, so regen_site.py can't
rebuild per-claim HTML from scratch.

Safe to re-run: all patches are idempotent (check for marker classes/ids
before rewriting).
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
    _ICON_BODY_CLAIMS,
    _ICON_BODY_CONSENSUS,
    _icon_svg,
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


# ── Patch 1: model-reasoning label injection (unchanged) ─────────────────────
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


# ── Patch 2: id="top" on the report hero ─────────────────────────────────────
_HERO_RE = re.compile(r'<section class="hero"(?! id=)')


def _inject_hero_id(html: str) -> tuple[str, int]:
    new, n = _HERO_RE.subn('<section class="hero" id="top"', html)
    return new, n


# ── Patch 3: "Jump to claim" section-head ────────────────────────────────────
_TOC_HEAD_RE = re.compile(
    r'<div class="section-head"><span>Jump to claim</span>'
    r'(<span class="sub">[^<]+</span>)'
    r'</div>'
)


def _inject_toc_head(html: str) -> tuple[str, int]:
    """Add id + icon to the 'Jump to claim' section-head."""
    if 'id="claim-catalog"' in html:
        return html, 0
    icon = _icon_svg(_ICON_BODY_CLAIMS, size=18, extra_class="section-head-icon")
    replacement = (
        '<div class="section-head" id="claim-catalog">'
        '<span class="section-head-label">'
        + icon
        + '<span>Jump to claim</span>'
        + '</span>'
        + r'\1'
        + '</div>'
    )
    new, n = _TOC_HEAD_RE.subn(replacement, html)
    return new, n


# ── Patch 4: .vp-stats — wrap children in .vp-stat + add icons ──────────────
# Matches the existing unclassed triple of stats produced before this refactor.
_VP_STATS_RE = re.compile(
    r'<div class="vp-stats">'
    r'<div><div class="vp-stat-num">([^<]+)</div><div class="vp-stat-lbl">Claims checked</div></div>'
    r'<div><div class="vp-stat-num">([^<]+)</div><div class="vp-stat-lbl">Models</div></div>'
    r'<div><div class="vp-stat-num">([^<]+)</div><div class="vp-stat-lbl">Inter-model agreement</div></div>'
    r'</div>'
)


def _inject_vp_stats(html: str) -> tuple[str, int]:
    if 'class="vp-stat-icon"' in html or 'class="vp-stat vp-stat"' in html or '"vp-stat"' in html:
        return html, 0
    icon_claims = _icon_svg(_ICON_BODY_CLAIMS, size=20, extra_class="vp-stat-icon")
    icon_consensus = _icon_svg(_ICON_BODY_CONSENSUS, size=20, extra_class="vp-stat-icon")

    def repl(m: re.Match) -> str:
        claims, models, agree = m.group(1), m.group(2), m.group(3)
        return (
            '<div class="vp-stats">'
            '<div class="vp-stat">'
            + icon_claims
            + f'<div class="vp-stat-num">{claims}</div>'
            '<div class="vp-stat-lbl">Claims checked</div></div>'
            '<div class="vp-stat">'
            '<span class="vp-stat-icon vp-stat-icon-placeholder" aria-hidden="true"></span>'
            + f'<div class="vp-stat-num">{models}</div>'
            '<div class="vp-stat-lbl">Models</div></div>'
            '<div class="vp-stat">'
            + icon_consensus
            + f'<div class="vp-stat-num">{agree}</div>'
            '<div class="vp-stat-lbl">Inter-model agreement</div></div>'
            '</div>'
        )

    new, n = _VP_STATS_RE.subn(repl, html)
    return new, n


# ── Patch 5: .claim-head — prepend icon + wrap in .claim-head-lead ──────────
_CLAIM_HEAD_RE = re.compile(
    r'(<div class="claim-head">)'
    r'(\s*)<span class="claim-num">([^<]+)</span>'
)


def _inject_claim_head_icon(html: str) -> tuple[str, int]:
    if 'class="claim-head-lead"' in html:
        return html, 0
    icon = _icon_svg(_ICON_BODY_CLAIMS, size=18, extra_class="claim-head-icon")

    def repl(m: re.Match) -> str:
        open_div, ws, num_text = m.group(1), m.group(2), m.group(3)
        return (
            open_div
            + '<span class="claim-head-lead">'
            + icon
            + f'<span class="claim-num">{num_text}</span>'
            + '</span>'
        )

    new, n = _CLAIM_HEAD_RE.subn(repl, html)
    return new, n


# ── Patch 6: .claim-foot — inject back-links after permalink ────────────────
_CLAIM_FOOT_RE = re.compile(
    r'(<div class="claim-foot">\s*'
    r'<a href="#claim-(\d+)" class="permalink">claim-\2</a>)'
    r'(\s*<span>Last verified)'
)


def _inject_claim_back_links(html: str) -> tuple[str, int]:
    if 'claim-back-links' in html:
        return html, 0
    back = (
        '<span class="claim-back-links">'
        '<a href="#claim-catalog" class="back-link">&uarr; Back to claim list</a>'
        '<span class="sep">&middot;</span>'
        '<a href="#top" class="back-link">&uarr; Top of page</a>'
        '</span>'
    )

    def repl(m: re.Match) -> str:
        head, _idx, tail = m.group(1), m.group(2), m.group(3)
        return head + back + tail

    new, n = _CLAIM_FOOT_RE.subn(repl, html)
    return new, n


# ── Orchestration ───────────────────────────────────────────────────────────
def _apply_report_patches(html: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    html, counts["labels"] = _inject_model_label(html)
    html, counts["hero_id"] = _inject_hero_id(html)
    html, counts["toc_head"] = _inject_toc_head(html)
    html, counts["vp_stats"] = _inject_vp_stats(html)
    html, counts["claim_head_icon"] = _inject_claim_head_icon(html)
    html, counts["claim_back_links"] = _inject_claim_back_links(html)
    return html, counts


def _apply_claim_page_patches(html: str) -> tuple[str, dict[str, int]]:
    """Standalone claim pages: icon + model labels only; no back-links or TOC."""
    counts: dict[str, int] = {}
    html, counts["labels"] = _inject_model_label(html)
    html, counts["claim_head_icon"] = _inject_claim_head_icon(html)
    return html, counts


def patch_existing_html() -> None:
    report_paths = list((SITE / "reports").glob("*.html"))
    claim_paths = list((SITE / "claims").glob("*.html"))
    total_changes = 0

    for path in report_paths:
        before = path.read_text(encoding="utf-8")
        after, counts = _apply_report_patches(before)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed = sum(counts.values())
            total_changes += changed
            print(f"patched {path.relative_to(ROOT)}: {counts}")

    for path in claim_paths:
        before = path.read_text(encoding="utf-8")
        after, counts = _apply_claim_page_patches(before)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed = sum(counts.values())
            total_changes += changed
            print(f"patched {path.relative_to(ROOT)}: {counts}")

    print(
        f"done: {total_changes} total patches across "
        f"{len(report_paths)} report(s) + {len(claim_paths)} claim page(s)"
    )


if __name__ == "__main__":
    refresh_assets_and_index()
    patch_existing_html()
