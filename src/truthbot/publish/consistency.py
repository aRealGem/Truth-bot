"""Build-time verification that every quantitative figure in site copy is
derived from ``data/*.json`` (remediation T0.8 / card P67.4).

The 2026-07-21 external audit found the landing page asserting "100% Model
Consensus" over reports whose recorded agreement was 47% and 78%, verdict
bars whose segments summed to fewer claims than the report contained, and
header chips computed on a different denominator than the headline two lines
below them. Each of those figures rendered from a *different* source (or a
hand-typed constant). This module re-derives the load-bearing figures from
``data/claims.json`` + ``data/reports.json`` and compares them against what
the HTML actually says; any mismatch is a build failure.

Scope: the checks cover the site's quantitative claim surfaces — index
program stats, per-report verdict bars (both lenses), family rails, header
chips, headline ratios, the anecdote footnote — plus tagline guards for
wording that must stay off the site until later remediation phases restore
it with evidence (T0.5/T0.6). Purely decorative numbers (CSS, dates,
pipeline version strings) are out of scope.

Usage::

    from truthbot.publish.consistency import check_site
    violations = check_site(Path("site-pca"))
    # empty list == consistent site

``scripts/rerender_pca_site.py`` runs this after every render and exits
non-zero on violations; ``tests/test_site_consistency.py`` runs it over the
committed ``site-pca/`` tree so hand-typed numbers cannot merge.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# The two families + abstentions, mirroring site._TRUE_FAMILY /
# site._ADVERSE_FAMILY. Imported lazily in check_site to avoid a cycle.
_ABSTAIN = {"Unverifiable", "Models split"}


def _fmt_pct(numerator: int, denominator: int) -> str:
    """Match site.py's ``format(x, '.0%')`` rendering."""
    return format(numerator / denominator, ".0%") if denominator else "0%"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _claims_for_report(claims: list[dict], report_id: str) -> list[dict]:
    return [c for c in claims if c.get("report_id") == report_id]


def _coarse_dist(report_claims: list[dict], axis: str) -> dict[str, int]:
    """Re-derive one lens's aggregate distribution from claims.json —
    the single bucketing every rendered breakdown must match (T0.2).
    Mirrors SiteReport._coarse_distribution: stored coarse label when
    present, else the fine label projected through the site's maps."""
    from truthbot.publish.site import (COARSE_LENIENT_PROJECTION,
                                       COARSE_STRICT_PROJECTION)
    projection = (COARSE_STRICT_PROJECTION if axis == "strict"
                  else COARSE_LENIENT_PROJECTION)
    dist: dict[str, int] = {}
    for c in report_claims:
        if c.get("consensus_verdict") == "Models split":
            label = "Models split"
        else:
            label = (c.get(f"coarse_{axis}_label")
                     or projection.get(c.get("consensus_verdict", ""), "Unverifiable"))
        dist[label] = dist.get(label, 0) + 1
    return dist


def _families(dist: dict[str, int]) -> tuple[int, int, int]:
    from truthbot.publish.site import _ADVERSE_FAMILY, _TRUE_FAMILY
    t = sum(v for k, v in dist.items() if k in _TRUE_FAMILY)
    f = sum(v for k, v in dist.items() if k in _ADVERSE_FAMILY)
    return t, f, t + f


def _bar_segment_counts(lens_html: str) -> dict[str, int]:
    """Parse ``title="Label: N"`` segment annotations out of one lens block."""
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r'title="([^":]+): (\d+)"', lens_html)}


def _lens_blocks(page: str) -> dict[str, str]:
    """Slice the verdict-panel bar wrap into its strict/lenient lens blocks.

    Nested divs defeat a close-tag regex, so this slices on the block
    *openers*: wrap start → lenient opener bounds the strict block; lenient
    opener → the anecdote note / source row bounds the lenient block. Only
    bar segments carry ``title="Label: N"`` annotations inside the wrap, so
    the slices are safe inputs for _bar_segment_counts."""
    start = page.find('<div class="vp-bar-wrap">')
    if start < 0:
        return {}
    end_markers = [i for i in (page.find('vp-anecdote-note', start),
                               page.find('class="source-row"', start)) if i > 0]
    end = min(end_markers) if end_markers else len(page)
    wrap = page[start:end]
    lenient_at = wrap.find('data-lens-axis="lenient"')
    if lenient_at < 0:
        return {"strict": wrap}
    return {"strict": wrap[:lenient_at], "lenient": wrap[lenient_at:]}


def check_report_page(page: str, report: dict, report_claims: list[dict]) -> list[str]:
    """Verify one report page's figures against claims.json-derived values."""
    violations: list[str] = []
    slug = report.get("url", report.get("id", "?"))
    claim_count = len(report_claims)

    # Report row in reports.json must agree with claims.json (T0.7).
    if report.get("claim_count") != claim_count:
        violations.append(
            f"{slug}: reports.json claim_count={report.get('claim_count')} "
            f"but claims.json has {claim_count} claims for this report")

    blocks = _lens_blocks(page)
    for axis in ("strict", "lenient"):
        dist = _coarse_dist(report_claims, axis)
        # Verdict bar segments (title="Label: N") must reproduce the derived
        # bucketing exactly and sum to claim_count (T0.2 acceptance).
        lens_html = blocks.get(axis)
        if lens_html is None:
            violations.append(f"{slug}: no {axis} verdict bar found")
            continue
        segs = _bar_segment_counts(lens_html)
        if sum(segs.values()) != claim_count:
            violations.append(
                f"{slug} [{axis}]: bar segments sum to {sum(segs.values())}, "
                f"claim_count is {claim_count}")
        derived = {k: v for k, v in dist.items() if v}
        if segs != derived:
            violations.append(
                f"{slug} [{axis}]: bar segments {segs} != derived buckets {derived}")

        # Header chips: family shares over DECIDED claims, same convention as
        # the headline (T0.3).
        t, f, decided = _families(dist)
        # Headline ratio text.
        lean = "true-leaning" if t >= f else "false-leaning"
        fam_count = t if t >= f else f
        expected_ratio = f"{fam_count} of {decided} decided claims {lean}"
        if decided and expected_ratio not in page:
            violations.append(
                f"{slug} [{axis}]: expected headline ratio '{expected_ratio}' not found")

    # Chips render one span per lens inside the two vp-headline-stat frames.
    dist_strict = _coarse_dist(report_claims, "strict")
    dist_lenient = _coarse_dist(report_claims, "lenient")
    for frame_cls, pick in (("vp-stat-truthy", 0), ("vp-stat-false", 1)):
        m = re.search(
            r'class="vp-headline-stat %s.*?data-lens-axis="strict">([\d%%]+)</span>'
            r'.*?data-lens-axis="lenient"[^>]*>([\d%%]+)</span>' % frame_cls,
            page, re.S)
        if not m:
            violations.append(f"{slug}: chip frame {frame_cls} not found")
            continue
        for axis, got in (("strict", m.group(1)), ("lenient", m.group(2))):
            dist = dist_strict if axis == "strict" else dist_lenient
            t, f, decided = _families(dist)
            want = _fmt_pct((t, f)[pick], decided)
            if got != want:
                violations.append(
                    f"{slug} [{axis}]: chip {frame_cls} shows {got}, derived {want}")

    # Anecdote footnote must reconcile with the derived Unverifiable bucket.
    m = re.search(r'vp-anecdote-note[^>]*>(\d+) of the (\d+) Unverifiable', page)
    if m:
        n_anec_uv, uv_shown = int(m.group(1)), int(m.group(2))
        uv_derived = dist_strict.get("Unverifiable", 0)
        anec_uv_derived = sum(
            1 for c in report_claims
            if (c.get("provenance", {}).get("layer_a_claim_type") == "personal-anecdote"
                and (c.get("coarse_strict_label") == "Unverifiable"
                     and c.get("consensus_verdict") != "Models split")))
        if uv_shown != uv_derived:
            violations.append(
                f"{slug}: footnote says {uv_shown} Unverifiable, derived {uv_derived}")
        if n_anec_uv > uv_derived:
            violations.append(
                f"{slug}: footnote counts {n_anec_uv} anecdotes in a "
                f"{uv_derived}-claim Unverifiable bucket")
        if n_anec_uv != anec_uv_derived:
            violations.append(
                f"{slug}: footnote anecdote count {n_anec_uv} != derived "
                f"{anec_uv_derived} from claims.json layer_a_claim_type")
    return violations


def check_site(site_root: Path) -> list[str]:
    """Verify the whole rendered site. Returns a list of violations (empty
    when every checked figure derives cleanly from data/*.json)."""
    site_root = Path(site_root)
    violations: list[str] = []
    reports = _load_json(site_root / "data" / "reports.json")
    claims = _load_json(site_root / "data" / "claims.json")

    # ── Index program stats (T0.1 / T0.7) ────────────────────────────────
    index_html = (site_root / "index.html").read_text(encoding="utf-8")
    reports_claim_sum = sum(r.get("claim_count", 0) for r in reports)
    if reports_claim_sum != len(claims):
        violations.append(
            f"index: reports.json claim_counts sum to {reports_claim_sum}, "
            f"claims.json has {len(claims)} entries")
    m = re.search(r'<div class="num">(\d+)</div><div class="lbl">Claims Checked',
                  index_html)
    if not m:
        violations.append("index: Claims Checked stat not found")
    elif int(m.group(1)) != len(claims):
        violations.append(
            f"index: Claims Checked shows {m.group(1)}, claims.json has {len(claims)}")

    m = re.search(r'<div class="num">(\d+)<span class="unit">%</span></div>'
                  r'<div class="lbl">Model Consensus', index_html)
    if not m:
        violations.append("index: Model Consensus stat not found")
    else:
        want = round(sum(r.get("model_agreement_rate", 0) * r.get("claim_count", 0)
                         for r in reports) / (reports_claim_sum or 1) * 100)
        if int(m.group(1)) != want:
            violations.append(
                f"index: Model Consensus shows {m.group(1)}%, claim-weighted "
                f"mean of reports.json is {want}%")

    # ── Insights page + tagline guards (T0.4 → rebuilt in T4.1) ──────────
    # Valid states: the About redirect stub (no per-seat data) or the v2
    # per-seat page. The v1 pseudo-model page ("Hydramind") must never ship.
    insights = site_root / "model-insights.html"
    if insights.exists():
        text = insights.read_text(encoding="utf-8")
        is_stub = 'http-equiv="refresh"' in text
        is_v2 = "Model panel insights" in text and "panel_by_role" in text
        if "Hydramind" in text or not (is_stub or is_v2):
            violations.append(
                "model-insights.html: expected the v2 per-seat page or the "
                "About redirect stub; the v1 pseudo-model page must not ship")
    for fname, banned in (("index.html", "primary sources"),
                          ("about.html", "comparable accuracy"),
                          ("about.html", "never silently broken")):
        p = site_root / fname
        if p.exists() and banned in p.read_text(encoding="utf-8"):
            violations.append(f"{fname}: banned phrase present: '{banned}'")

    # ── Per-report pages ─────────────────────────────────────────────────
    for report in reports:
        url = report.get("url", "")
        page_path = site_root / url
        if not url or not page_path.exists():
            violations.append(f"report {report.get('id', '?')}: page {url} missing")
            continue
        page = page_path.read_text(encoding="utf-8")
        violations.extend(
            check_report_page(page, report,
                              _claims_for_report(claims, report.get("id"))))
    return violations
