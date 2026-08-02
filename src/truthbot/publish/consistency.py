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
import logging
import re
from datetime import date
from pathlib import Path

from truthbot.publish.aggregation import (COARSE_VERDICT_ORDER,
                                          TIER_LINE_ORDER,
                                          distribution_from_claims,
                                          family_verdict)

logger = logging.getLogger(__name__)

# Abstention buckets (kept for reference by external callers; the family
# math itself delegates to aggregation.family_verdict).
_ABSTAIN = {"Unverifiable", "Models split"}


def _fmt_pct(numerator: int, denominator: int) -> str:
    """Match site.py's ``format(x, '.0%')`` rendering."""
    return format(numerator / denominator, ".0%") if denominator else "0%"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _claims_for_report(claims: list[dict], report_id: str) -> list[dict]:
    return [c for c in claims if c.get("report_id") == report_id]


def _coarse_dist(report_claims: list[dict], axis: str) -> dict[str, int]:
    """Re-derive one lens's aggregate distribution from claims.json — the
    single bucketing every rendered breakdown must match (T0.2). Since
    remediation v2 (1.6) this DELEGATES to the same
    ``aggregation.distribution_from_claims`` the renderer uses; the old
    hand-kept mirror of SiteReport._coarse_distribution is gone."""
    return distribution_from_claims(report_claims, axis)


def _families(dist: dict[str, int]) -> tuple[int, int, int]:
    fam = family_verdict(dist)
    return fam.true_count, fam.adverse_count, fam.decided


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

    # Every stored distribution must sum EXACTLY to the checkable-claim count,
    # and the fine buckets must re-derive from claims.json (PR-A2.0 / T0.1: the
    # Obama-2014 journal tally read 95 of 96 because a split row carries
    # verdict=null — no published aggregate may reproduce that drift class).
    for key in ("verdict_distribution", "verdict_distribution_lenient",
                "verdict_distribution_strict"):
        dist = report.get(key)
        if dist is None:
            continue  # legacy report row predating the coarse exports
        if sum(dist.values()) != claim_count:
            violations.append(
                f"{slug}: {key} sums to {sum(dist.values())}, "
                f"claim_count is {claim_count}")
    fine = report.get("verdict_distribution")
    if fine is not None:
        derived_fine: dict[str, int] = {}
        for c in report_claims:
            label = c.get("consensus_verdict", "")
            derived_fine[label] = derived_fine.get(label, 0) + 1
        stored_fine = {k: v for k, v in fine.items() if v}
        if stored_fine != derived_fine:
            violations.append(
                f"{slug}: verdict_distribution {stored_fine} != "
                f"claims.json-derived {derived_fine}")

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

    # Self-sourced-only abstention chip (PR-A2.1 T1.2): its decomposition must
    # re-derive from claims.json — decided/self-sourced/other(/split) sum to
    # claim_count, with the self-sourced count read off the exported
    # provenance.self_sourced_only flags.
    m = re.search(r'vp-selfsource-chip[^>]*>(\d+) decided · (\d+) unverified — '
                  r'self-sourced only · (\d+) unverifiable — other'
                  r'(?: · (\d+) models split)?', page)
    if m:
        got = [int(g) for g in m.groups() if g is not None]
        dist = _coarse_dist(report_claims, "strict")
        uv = dist.get("Unverifiable", 0)
        split = dist.get("Models split", 0)
        selfsrc = sum(1 for c in report_claims
                      if c.get("provenance", {}).get("self_sourced_only"))
        want = [claim_count - uv - split, selfsrc, uv - selfsrc]
        if split:
            want.append(split)
        if got != want:
            violations.append(
                f"{slug}: self-source chip shows {got}, derived {want}")
        if sum(got) != claim_count:
            violations.append(
                f"{slug}: self-source chip terms sum to {sum(got)}, "
                f"claim_count is {claim_count}")

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


# ── Published run-artifact invariants (remediation v2, 1.4) ──────────────────
#
# metrics/pca_runs/methodology_manifest.json pins every stored artifact to the
# methodology GENERATION it was produced under. Runs labeled with the
# manifest's current_generation must satisfy the current invariants; runs with
# older generations are permanently legacy — reported, never re-assertable —
# which is what blocks re-publishing them as-is.

#: Utterance date per speech, for the era lint over stored artifacts (their
#: meta.date agrees, but the lint must not depend on artifact self-report).
_SPEECH_DATES = {
    "trump_2026": date(2026, 2, 24),
    "biden_2022": date(2022, 3, 1),
    "obama_2014": date(2014, 1, 28),
    "clinton_1998": date(1998, 1, 27),
    "gwbush_2006": date(2006, 1, 31),
}

#: PR-A2.2 / T2.1 saturation cap, mirrored from
#: truthbot.verdict.consolidator.MAX_S5 (imported lazily in the checker to
#: keep this module render-side-import-free).
_MAX_S5_PER_SID = 3


def check_run_artifacts(repo_root) -> list[str]:
    """Assert the current-generation invariants over stored pca_runs artifacts.

    For every run the methodology manifest labels ``current_generation``:
      (i)   per-claim POLITICAL-tier item count <= 3 (the S5 saturation cap),
      (ii)  zero era violations (fair-game window from the speech-date map,
            via :func:`truthbot.verdict.era_lint.lint_pack_items`),
      (iii) zero fact-check URLs in evidence
            (:func:`truthbot.verify.factcheck_exclusion.is_excluded_factchecker`).

    Runs with OLDER generations produce logged report lines, never failures —
    they are legacy by construction and the manifest is what keeps them from
    being re-published as-is. Returns the violation list (empty = pass).

    TODO (D11.2 credit-identity check, gated on generation "v2.4+"): recompute
    the decided-verdict credit set from principal relations and assert no
    decided claim rests solely on the speaker's own record. Not implementable
    over v2.3 artifacts — evidential roles are not stored on artifact evidence
    and the principals recompute belongs to the Phase-3 regeneration.
    """
    from truthbot.verdict.era_lint import lint_pack_items
    from truthbot.verify.factcheck_exclusion import is_excluded_factchecker

    repo_root = Path(repo_root)
    runs_dir = repo_root / "metrics" / "pca_runs"
    manifest = _load_json(runs_dir / "methodology_manifest.json")
    current = manifest["current_generation"]
    violations: list[str] = []

    for run_id, row in manifest["runs"].items():
        path = runs_dir / f"{run_id}.json"
        if not path.exists():
            violations.append(f"{run_id}: manifest row but artifact file missing")
            continue
        if row.get("generation") != current:
            logger.info(
                "pca run %s (%s): legacy generation %r%s — reported, not "
                "re-assertable under %r", run_id[:8], row.get("speech_id"),
                row.get("generation"),
                " [published]" if row.get("published") else "", current)
            continue

        artifact = _load_json(path)
        speech_id = row.get("speech_id", "")
        utterance = _SPEECH_DATES.get(speech_id)
        if utterance is None:
            violations.append(
                f"{run_id}: no utterance date known for speech {speech_id!r} "
                "— the era invariant cannot be asserted (fail closed)")
            continue
        evidence = artifact.get("evidence")
        if evidence is None:
            violations.append(f"{run_id}: current-generation run stores no evidence")
            continue

        for sid, items in evidence.items():
            pol = sum(1 for it in items if it.get("source_tier") == "Political")
            if pol > _MAX_S5_PER_SID:                                    # (i)
                violations.append(
                    f"{run_id} {sid}: {pol} POLITICAL-tier items exceed the "
                    f"<={_MAX_S5_PER_SID} S5 cap")
            era, _, _ = lint_pack_items(sid, items, utterance)           # (ii)
            for v in era:
                violations.append(f"{run_id} {sid}: era violation — {v.message}")
            for it in items:                                             # (iii)
                url = it.get("source_url") or ""
                if url and is_excluded_factchecker(url):
                    violations.append(
                        f"{run_id} {sid}: fact-check URL in evidence: {url}")
    return violations


def _check_index_tier_buckets(index_html: str, reports: list[dict]) -> list[str]:
    """Remediation v2 (1.6): the Sources line on every index card must
    reproduce reports.json tier_counts exactly — every nonzero bucket,
    political included (the old hand-kept order silently dropped it, hiding
    162 sources on the Trump card). Parses the machine-readable
    ``data-tier-counts`` attribute on ``.src-tiers``."""
    violations: list[str] = []
    for r in reports[:20]:  # the index renders the first 20 cards
        url = r.get("url", "")
        slug = url or r.get("id", "?")
        tier_counts = r.get("tier_counts") or {}
        want = {k: v for k, v in tier_counts.items() if v}
        start = index_html.find(f'href="{url}" class="report"')
        if start < 0:
            violations.append(f"index: no report card found for {slug}")
            continue
        card = index_html[start:index_html.find("</a>", start)]
        m = re.search(r'class="src-tiers" data-tier-counts="([^"]*)"', card)
        if not m:
            if want:
                violations.append(
                    f"index card {slug}: no machine-readable Sources chip "
                    f"(data-tier-counts) but tier_counts has {want}")
            continue
        got = {k: int(v) for k, v in
               (pair.split(":") for pair in m.group(1).split() if ":" in pair)}
        if got != want:
            violations.append(
                f"index card {slug}: Sources chip buckets {got} != "
                f"reports.json tier_counts {want}")
        if sum(got.values()) != sum(tier_counts.values()):
            violations.append(
                f"index card {slug}: Sources chip sums to {sum(got.values())}, "
                f"tier_counts sum to {sum(tier_counts.values())}")
    return violations


#: Buckets the aggregate bar can actually render (aggregation.AGGREGATE_BAR_ORDER
#: is the family-grouped union of both axes) — a nonzero count outside this set
#: would silently vanish from every rendered bar.
def _check_bucket_invariants(reports: list[dict], claims: list[dict]) -> list[str]:
    """Remediation v2 (1.6) strict lints (ii)+(iii): per-report bucket sums
    equal claim_count with every nonzero bucket renderable, and the
    site-wide aggregate (sum of per-report distributions) accounts for
    every claim in claims.json exactly once, on every axis."""
    from truthbot.publish.aggregation import AGGREGATE_BAR_ORDER
    renderable = set(AGGREGATE_BAR_ORDER)
    violations: list[str] = []
    totals: dict[str, int] = {"verdict_distribution": 0,
                              "verdict_distribution_lenient": 0,
                              "verdict_distribution_strict": 0}
    for r in reports:
        slug = r.get("url", r.get("id", "?"))
        claim_count = r.get("claim_count", 0)
        for key in totals:
            dist = r.get(key)
            if dist is None:
                continue
            totals[key] += sum(dist.values())
            if sum(dist.values()) != claim_count:   # (ii) — also checked in
                # check_report_page for the fine dist; repeated here so the
                # strict pass reports it even for index-only renders.
                violations.append(
                    f"{slug}: {key} sums to {sum(dist.values())}, "
                    f"claim_count is {claim_count}")
            if key != "verdict_distribution":
                ghost = {k: v for k, v in dist.items()
                         if v and k not in renderable}
                if ghost:
                    violations.append(
                        f"{slug}: {key} buckets {ghost} are outside "
                        "AGGREGATE_BAR_ORDER and would not render")
    for key, total in totals.items():               # (iii)
        if total != len(claims):
            violations.append(
                f"site-wide: {key} buckets sum to {total} across reports.json, "
                f"claims.json has {len(claims)} entries")
    return violations


def check_site(site_root: Path, strict_buckets: bool = True) -> list[str]:
    """Verify the whole rendered site. Returns a list of violations (empty
    when every checked figure derives cleanly from data/*.json).

    ``strict_buckets`` gates the remediation-v2 lints (index Sources-chip
    buckets, per-report/site-wide bucket sums). Default True — every fresh
    render must satisfy them. The COMMITTED site-pca/ tree predates the
    remediation regeneration (its cards were rendered without the political
    bucket), so tests/test_site_consistency.py lints it with
    ``strict_buckets=False`` until the Phase-2 regen flips it to True."""
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

    # ── Remediation-v2 strict bucket lints (1.6) ─────────────────────────
    if strict_buckets:
        violations.extend(_check_index_tier_buckets(index_html, reports))
        violations.extend(_check_bucket_invariants(reports, claims))

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
