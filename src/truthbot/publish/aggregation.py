"""Single source of truth for verdict aggregation (remediation v2, item 1.6).

Every fold from a per-claim verdict to a published bucket — coarse-axis
projection, family (true-leaning vs false-leaning) math, distribution
tallies, and the report-card Sources line — lives here and ONLY here.
``publish/site.py`` (the render layer), ``publish/consistency.py`` (the
build-time checker), and any offline script all import this module, so the
figure a page renders and the figure the checker re-derives can never come
from two drifted implementations.

LEAF MODULE: imports nothing from the publish package (site.py imports *it*,
never the reverse) so consistency.py and scripts can use it without cycles.

The one behavioral fix over the pre-module code (2026-07 audit V6): a claim
whose panel deadlocked ("Models split") or produced no verdict ("No verdict")
NEVER folds into Unverifiable. The old per-call-site fallbacks projected the
fine label (``consensus_label`` is UNVERIFIABLE for those rows) whenever the
stored coarse label was blank, silently laundering a process outcome into an
evidence outcome. :func:`coarse_label` checks the verdict text first, so the
fold is impossible regardless of what the caller stored.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Bucket orders ─────────────────────────────────────────────────────────────

# 5-bucket coarse-axis "Truthy scale" — used by every aggregate display
# (verdict panel, TOC pills, report cards, index totals). Order is most
# positive → most negative so segment-bar renderers can iterate it directly.
# Source rubric: ``eval/sotu-2026/findings-review.md`` Part H.
COARSE_VERDICT_ORDER = ["True", "Truthy", "Unverifiable", "Falsey", "False"]

# Aggregate BAR order — family-grouped union of the coarse and fine axes.
# When the Falsey umbrella left the PCA per-claim pills (2026-07-19), the
# aggregate bars kept iterating COARSE_VERDICT_ORDER, so fine-labeled PCA
# claims (Misleading, Exaggerated, Mostly True) silently vanished from the
# graph while the headline still counted them in its family totals — "95 of
# 132 false-leaning" with only 44 visible on the bar (jackie, 2026-07-20).
# This order includes both axes' labels, contiguous by family (true family →
# abstain → adverse family), so the bar shows every decided claim and the
# family rail's brackets equal the headline's totals by construction.
AGGREGATE_BAR_ORDER = ["True", "Mostly True", "Truthy",
                       "Unverifiable", "Models split",
                       "Exaggerated", "Misleading", "Falsey", "False"]

# ── Projections ───────────────────────────────────────────────────────────────

# Mirror of LENIENT_PROJECTION / STRICT_PROJECTION in
# ``src/truthbot/verify/engine.py``, but keyed on the *fine-axis label string*
# (not the ``VerdictLabel`` enum) so the publish layer can stay string-typed.
# The two must stay in lockstep — see test_consensus_projection.py for the
# canonical mapping invariants.
COARSE_LENIENT_PROJECTION: dict[str, str] = {
    "True":         "True",
    "Mostly True":  "Truthy",
    "Exaggerated":  "Truthy",
    "Misleading":   "Falsey",
    "False":        "False",
    "Unverifiable": "Unverifiable",
}

COARSE_STRICT_PROJECTION: dict[str, str] = {
    "True":         "True",
    "Mostly True":  "Truthy",
    "Exaggerated":  "Falsey",   # diverges from Lenient
    "Misleading":   "Falsey",
    "False":        "False",
    "Unverifiable": "Unverifiable",
}

_PROJECTIONS: dict[str, dict[str, str]] = {
    "lenient": COARSE_LENIENT_PROJECTION,
    "strict":  COARSE_STRICT_PROJECTION,
}

#: Process outcomes that are NEVER folded into an evidence bucket. They pass
#: through every aggregation verbatim (audit V6).
NON_FOLDING_VERDICTS: frozenset[str] = frozenset({"Models split", "No verdict"})


def projection_for(axis: str) -> dict[str, str]:
    """Return the fine→coarse projection map for ``axis`` ('strict'|'lenient')."""
    try:
        return _PROJECTIONS[axis]
    except KeyError:
        raise ValueError(f"unknown projection axis {axis!r} "
                         "(expected 'strict' or 'lenient')") from None


# ── Families ──────────────────────────────────────────────────────────────────

# Family aggregation for headline verdicts (2026-07-19 editorial review):
# a report that is 72% False+Misleading must headline "Largely False", not
# "Mixed verdict" just because no single adverse bucket crossed a threshold.
# True-leaning and false-leaning labels aggregate into two families over
# DECIDED claims; Unverifiable / Models split / No verdict are abstentions
# and stay out of the denominator.
TRUE_FAMILY: frozenset[str] = frozenset({"True", "Mostly True", "Truthy"})
ADVERSE_FAMILY: frozenset[str] = frozenset(
    {"False", "Falsey", "Misleading", "Exaggerated"})


# ── The one folding rule ──────────────────────────────────────────────────────

def fine_label(consensus_verdict: str, consensus_label: str) -> str:
    """Display label on the FINE (6-bucket) axis for one claim.

    PCA split / no-verdict rows carry ``consensus_label=Unverifiable`` (never
    silently dropped) but a distinct verdict text; that text wins so the
    process outcome is what renders and tallies — never "Unverifiable".
    """
    if consensus_verdict in NON_FOLDING_VERDICTS:
        return consensus_verdict
    return consensus_label


def coarse_label(consensus_verdict: str, stored_coarse: str, axis: str) -> str:
    """THE fold from one claim's verdict to its coarse-axis display bucket.

    * "Models split" / "No verdict" return AS-IS — never folded (audit V6:
      the old fine-label fallback projected them to Unverifiable whenever
      the stored coarse field was blank).
    * Otherwise the stored coarse label wins (post-projection bundles).
    * Legacy rows (blank stored field) project the fine label — for resolved
      claims ``consensus_verdict`` IS the fine label string.
    """
    if consensus_verdict in NON_FOLDING_VERDICTS:
        return consensus_verdict
    stored = (stored_coarse or "").strip()
    if stored:
        return stored
    return projection_for(axis).get(consensus_verdict, "Unverifiable")


def project_dist(fine_dist: dict[str, int], axis: str) -> dict[str, int]:
    """Project a 6-bucket fine distribution onto the 5-bucket coarse axis.

    Backfills the coarse fields when a ``reports.json`` entry predates the
    projection layer. Counts mapping to the same coarse bucket are summed
    (e.g. ``Mostly True + Exaggerated → Truthy`` under Lenient). Non-folding
    buckets ("Models split" / "No verdict") pass through verbatim.
    """
    projection = projection_for(axis)
    out: dict[str, int] = {v: 0 for v in COARSE_VERDICT_ORDER}
    out["Models split"] = 0
    for label, cnt in fine_dist.items():
        if label in NON_FOLDING_VERDICTS:
            out[label] = out.get(label, 0) + cnt
        else:
            coarse = projection.get(label, "Unverifiable")
            out[coarse] = out.get(coarse, 0) + cnt
    return out


def distribution_from_claims(rows: list[dict], axis: str) -> dict[str, int]:
    """One axis's aggregate distribution from per-claim rows.

    ``rows`` are claim dicts carrying ``consensus_verdict`` and (optionally)
    ``coarse_{axis}_label`` — the shape of ``data/claims.json`` entries.
    Zero-prefills the five coarse buckets + "Models split" so downstream
    exports keep a stable shape; extra buckets appear only when present.
    """
    dist: dict[str, int] = {v: 0 for v in COARSE_VERDICT_ORDER}
    dist["Models split"] = 0
    for row in rows:
        label = coarse_label(row.get("consensus_verdict", ""),
                             row.get(f"coarse_{axis}_label", ""), axis)
        dist[label] = dist.get(label, 0) + 1
    return dist


# ── Family verdict (the percent-true headline) ───────────────────────────────

@dataclass(frozen=True)
class FamilyVerdict:
    """Family math for one distribution — headline, chips, and rails all
    read these fields so their numerator/denominator can never disagree."""
    label: str        # headline text, e.g. "56% True"
    css: str          # headline CSS class ("vt-true" | "vt-mid" | "vt-false" | "neutral")
    ratio_text: str   # e.g. "5 of 9 decided claims rated True"
    true_count: int
    adverse_count: int
    decided: int      # true_count + adverse_count
    total: int        # every claim, abstentions included


def family_verdict(dist: dict[str, int]) -> FamilyVerdict:
    """Percent-true headline (jackie, 2026-07-25: "just show percent true" —
    supersedes the 2026-07-19 graded bands, whose 'Mostly True' read as an
    endorsement at 55% truthiness).

    The label is the TRUE-family share of DECIDED claims, e.g. '56% True' —
    Unverifiable / Models split / No verdict are abstentions, out of the
    denominator, and disclosed by the ratio text. Color bands (jackie,
    2026-07-25): true-share > 75% green, 50-75% inclusive yellow (vt-mid),
    under 50% red — the words never grade, the number speaks.
    A report with claims but zero decided verdicts headlines 'Unverifiable'.
    """
    total = sum(dist.values())
    t = sum(v for k, v in dist.items() if k in TRUE_FAMILY)
    f = sum(v for k, v in dist.items() if k in ADVERSE_FAMILY)
    decided = t + f
    if total == 0:
        return FamilyVerdict("No claims evaluated", "neutral",
                             "0 claims checked", t, f, decided, total)
    if decided == 0:
        return FamilyVerdict("Unverifiable", "neutral",
                             f"{total} claims checked", t, f, decided, total)
    share = t / decided
    label = f"{round(100 * share)}% True"
    ratio = f"{t} of {decided} decided claims rated True"
    if share > 0.75:
        css = "vt-true"
    elif share >= 0.50:
        css = "vt-mid"
    else:
        css = "vt-false"
    return FamilyVerdict(label, css, ratio, t, f, decided, total)


# ── Report-card Sources line ──────────────────────────────────────────────────

#: (bucket key in tier_counts, reader-facing label) in display order. This is
#: the COMPLETE bucket set of ``site._tier_counts_for_report`` — the old
#: hand-kept order in ``_report_card`` omitted "political" entirely, hiding
#: e.g. 162 press/political sources on the Trump card while the About page
#: promised full sourcing transparency.
TIER_LINE_ORDER: list[tuple[str, str]] = [
    ("gov", "gov"),
    ("wire", "wire"),
    ("news", "news"),
    ("fc", "fc"),
    ("political", "press/political"),
    ("other", "other"),
]


def sources_line(tier_counts: dict[str, int]) -> list[tuple[str, int]]:
    """(label, count) pairs for every NONZERO tier bucket, display order.

    Every bucket that counted a source renders — no bucket is ever silently
    dropped from the Sources line.
    """
    return [(label, tier_counts.get(key, 0))
            for key, label in TIER_LINE_ORDER
            if tier_counts.get(key, 0)]
