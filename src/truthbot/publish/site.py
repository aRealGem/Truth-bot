"""
publish/site.py — Static site publisher for truth-bot.

Generates a complete accountability-dashboard static site from VerdictBundle objects.
All HTML templates are inline Python strings; no external template files required.

Design: accountability dashboard aesthetic — Newsreader serif + Geist sans/mono,
verdict colors as the only chroma, Truthy McTruthface integrated into the verdict panel.

Output structure:
    {SITE_ROOT}/
        index.html
        about.html
        truthy.html
        404.html
        reports/{YYYY-MM-DD}-{speaker-slug}.html
        claims/{claim_id}.html
        assets/styles.css
        assets/truthbot.js
        data/reports.json
        data/claims.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from truthbot.models import VerdictBundle, VerdictLabel
# Bucket orders, projections, family sets, and the one folding rule live in
# ``truthbot.publish.aggregation`` (remediation v2, 1.6) — the single source
# of truth this render layer and the consistency checker both import. The
# public names are re-exported here for backward compat with existing
# importers of ``publish.site``.
from truthbot.publish.aggregation import (
    ADVERSE_FAMILY as _ADVERSE_FAMILY,
    AGGREGATE_BAR_ORDER,
    COARSE_LENIENT_PROJECTION,
    COARSE_STRICT_PROJECTION,
    COARSE_VERDICT_ORDER,
    TIER_LINE_ORDER,  # noqa: F401  (re-export)
    TRUE_FAMILY as _TRUE_FAMILY,
    coarse_label as _agg_coarse_label,
    distribution_from_claims as _agg_distribution_from_claims,
    family_verdict as _agg_family_verdict,
    fine_label as _agg_fine_label,
    project_dist as _agg_project_dist,
    sources_line as _agg_sources_line,
)
from truthbot.verify.principals import PrincipalRelation, principal_relation
from truthbot.verify.source_tiers import TIER_BUCKET, TIER_DISPLAY, classify_tier

logger = logging.getLogger(__name__)

# ── Verdict presentation constants ────────────────────────────────────────────

# CSS class slugs — map label → slug used in .v-{slug} and .vt-{slug}
VERDICT_CSS: dict[str, str] = {
    "True":          "true",
    "Mostly True":   "mostly-true",
    "Misleading":    "misleading",
    "Exaggerated":   "exaggerated",
    "False":         "false",
    "Unverifiable":  "unverifiable",
    # 5-bucket coarse-axis projection (Truthy scale) — used on the headline
    # pill, not the per-model strip. ``Models split`` gets its own slug so
    # aggregate bars can show it as a distinct segment (remediation T0.2:
    # every rendered breakdown must sum to claim_count, split included).
    "Truthy":        "truthy",
    "Falsey":        "falsey",
    "Models split":  "split",
}

# Display order for the verdict bar legend (always show all 6)
VERDICT_ORDER = ["True", "Mostly True", "Exaggerated", "Misleading", "False", "Unverifiable"]

VERDICT_EMOJI: dict[str, str] = {
    "True":          "✅",
    "Mostly True":   "🟢",
    "Misleading":    "⚠️",
    "Exaggerated":   "📊",
    "False":         "❌",
    "Unverifiable":  "❓",
}

STRENGTH_LABEL: dict[str, str] = {
    "strong": "Strong consensus",
    "weak":   "Weak consensus",
    "none":   "Models split",
    "single": "Single model",
}

TIER_TABLE = [
    ("Government",  ".gov, .mil, .int — BLS, BEA, CBO, Census, NATO, etc.",   "Highest"),
    ("Wire",        "AP, Reuters",                                              "High"),
    ("Established", "NYT, WaPo, BBC, NPR, CBS, NBC, ABC",                      "Medium-High"),
    ("Academic",    "Peer-reviewed journals, university presses",               "Medium-High"),
    ("Fact-check",  "PolitiFact, FactCheck.org, Snopes, FullFact",             "Medium"),
    ("Other",       "Blogs, opinion sites, social media, unverified sources",   "Low"),
]

def _url_display_host(url: str) -> str:
    """Return a display-friendly hostname from a URL."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc or url
        return host.removeprefix("www.")
    except Exception:
        return url


GITHUB_URL = "https://github.com/aRealGem/Truth-bot"
PIPELINE_VERSION = "0.2.0"

# Public base URL of the published site. Mirrors ``settings.site_url`` in
# truthbot.config, but read from the environment directly so the render layer
# keeps its zero-config-import convention (same reason SitePublisher reads
# TRUTHBOT_SITE_ROOT itself).
_DEFAULT_SITE_URL = "https://raw.githack.com/aRealGem/Truth-bot/main/site-pca"


def _site_url() -> str:
    """Base URL for absolute self-links (canonical / og:url / feed entries)."""
    return os.environ.get("TRUTHBOT_SITE_URL", _DEFAULT_SITE_URL).rstrip("/")

# Pre-1.0 releases are flagged "Beta" next to the version string everywhere the
# version is rendered. Flips off automatically when PIPELINE_VERSION crosses 1.0.
IS_BETA = PIPELINE_VERSION.split(".", 1)[0] == "0"
BETA_BADGE_HTML = (
    '<span class="beta-badge" aria-label="Beta release">Beta</span>'
    if IS_BETA else ''
)
BETA_TEXT_SUFFIX = ' (beta)' if IS_BETA else ''

# The Atom feed is RENDERED from the reports index at publish time \u2014 see
# ``_render_feed`` (remediation v2, 1.5). The old static FEED_XML_TEMPLATE
# (verbatim [SITE_URL] placeholder, one hand-typed phantom entry, frozen
# <updated> stamp) is gone.

# Google Fonts link tags (exact — do not modify)
_GOOGLE_FONTS = """\
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Newsreader:opsz,ital,wght@6..72,0,400;6..72,0,500;6..72,0,600;6..72,0,700;6..72,1,400;6..72,1,500&family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap\">
"""


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SiteReport:
    """All data needed to render a full report page."""
    report_id: str
    speaker: str        # legacy alias for source_of_claims
    role: str           # legacy alias for source_of_claims_professional_public_title
    date: Optional[datetime]
    venue: str          # physical location (e.g. "U.S. Capitol")
    transcript_source_url: str
    bundles: list[VerdictBundle]
    video_source_url: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Richer speaker/speech identity fields (Change 2)
    source_of_claims: str = ""
    source_of_claims_professional_public_title: str = ""
    event: str = ""    # event name (e.g. "State of the Union Address")
    channel: str = ""  # medium (e.g. "Twitter/X", "Press Release")
    # Non-check-worthy sentence stream from Layer A characterization (Statement
    # Triage). Each record: {sid, speech, idx, text, context, label, source,
    # a1_score}. Default empty → legacy-clean (no triage page rendered).
    characterization: list[dict] = field(default_factory=list)
    # PCA panel composition for this run: {"name": <roster>, "seats": {seat: [alias]}}.
    # A per-RUN fact (one roster judges the whole run), rendered once in the report
    # provenance. Default empty → legacy-clean (no composition block rendered).
    panel_roster: dict = field(default_factory=dict)
    # Stable speech identity (DC-3'): e.g. "obama_2014". When set, the report
    # slug derives its suffix from this id instead of the per-run UUID, so
    # re-rendering the same speech reuses the same URL. Default empty →
    # legacy UUID-suffixed slugs.
    speech_id: str = ""

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y-%m-%d") if self.date else "unknown-date"

    @property
    def display_date(self) -> str:
        return self.date.strftime("%B %d, %Y") if self.date else "Unknown date"

    @property
    def checkable_bundles(self) -> list[VerdictBundle]:
        return [b for b in self.bundles if b.claim.is_checkable]

    @property
    def verdict_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {v: 0 for v in VERDICT_CSS}
        for b in self.checkable_bundles:
            # PCA split / no-verdict claims carry consensus_label=UNVERIFIABLE
            # (never silently dropped) but a distinct verdict text. Count them
            # in their own bucket rather than folding them into Unverifiable —
            # aggregation.fine_label is the one rule for this.
            label = _agg_fine_label(b.consensus.consensus_verdict,
                                    b.consensus.consensus_label.value)
            dist[label] = dist.get(label, 0) + 1
        return dist

    @property
    def verdict_distribution_lenient(self) -> dict[str, int]:
        """5-bucket histogram on the Lenient projection axis.

        Folding is delegated to ``aggregation.coarse_label``: the stored
        ``coarse_lenient_label`` wins when present (post-projection bundles),
        legacy bundles project the fine label on the fly, and split /
        no-verdict rows pass through verbatim (audit V6: never folded to
        Unverifiable).
        """
        return self._coarse_distribution("lenient")

    @property
    def verdict_distribution_strict(self) -> dict[str, int]:
        """5-bucket histogram on the Strict projection axis (Exaggerated → Falsey)."""
        return self._coarse_distribution("strict")

    def _coarse_distribution(self, axis: str) -> dict[str, int]:
        # Delegates to aggregation.distribution_from_claims — the same
        # function consistency.py re-derives every published aggregate with,
        # so the renderer and the checker cannot drift (1.6).
        rows = [
            {
                "consensus_verdict": _agg_fine_label(
                    b.consensus.consensus_verdict,
                    b.consensus.consensus_label.value),
                f"coarse_{axis}_label": getattr(
                    b.consensus, f"coarse_{axis}_label", "") or "",
            }
            for b in self.checkable_bundles
        ]
        return _agg_distribution_from_claims(rows, axis)

    @property
    def model_agreement_rate(self) -> float:
        bundles = self.checkable_bundles
        if not bundles:
            return 0.0
        agreed = sum(1 for b in bundles if b.consensus.agreement)
        return agreed / len(bundles)

    @property
    def report_slug(self) -> str:
        # DC-3' stable slugs: when the report knows its speech identity, the
        # suffix is a deterministic hash of speech_id — every re-render of the
        # same speech lands on the same URL (no more one-orphaned-page-per-
        # publish). Legacy callers without a speech_id keep the per-run UUID
        # prefix, which is unique per publish by construction.
        if self.speech_id:
            short = hashlib.sha1(self.speech_id.encode("utf-8")).hexdigest()[:6]
        else:
            short = self.report_id[:6]  # first 6 chars of UUID — unique per run
        return f"{self.date_str}-{_slug(self.speaker)}-{short}"

    @property
    def report_url(self) -> str:
        return f"reports/{self.report_slug}.html"

    @property
    def triage_slug(self) -> str:
        """Filename stem for this report's Statement Triage page (under reports/)."""
        return f"{self.report_slug}-triage"

    @property
    def truthy_verdict(self):
        """Compute Truthy M. McTruthface aggregate mood from all checkable claims."""
        from truthbot.truthy import evaluate_truthy
        from truthbot.truthy.truthy_score import Rating

        _LABEL_TO_RATING: dict[str, Rating] = {
            "True":          Rating.TRUE,
            "Mostly True":   Rating.MOSTLY_TRUE,
            "Misleading":    Rating.HALF_TRUE,
            "Exaggerated":   Rating.MOSTLY_FALSE,
            "False":         Rating.FALSE,
            "Unverifiable":  Rating.HALF_TRUE,
        }

        ratings = [
            _LABEL_TO_RATING.get(b.consensus.consensus_label.value, Rating.HALF_TRUE)
            for b in self.checkable_bundles
        ]
        return evaluate_truthy(ratings)


# ── Helper functions ──────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _verdict_css(label_str: str) -> str:
    """Map a verdict label string to its CSS slug."""
    return VERDICT_CSS.get(label_str, "unverifiable")


# Family map for per-model dissent flagging on a claim card. Two model
# verdicts in the same family are NOT flagged as disagreeing with the
# bundle consensus, even if their fine-axis labels differ. Families are
# intentionally narrower than the publish-layer LENIENT/STRICT
# projections because the projections are tuned for headline-pill
# bucketing across the full panel, while dissent flagging is tuned for
# "this voter is directionally aligned with the consensus."
#
# Specifically:
#   * ``True`` and ``Mostly True`` are in the same family (truthy). The
#     2026-04 SOTU run had ``[Mostly True, Mostly True, True, True]``
#     → consensus Mostly True, both ``True`` voters flagged as dissent
#     (findings-review C4). With family-aware flagging, the same panel
#     shows zero dissents — directional agreement is honored.
#   * ``Misleading`` and ``False`` are in the same family (falsey).
#   * ``Exaggerated`` lives in its own family rather than collapsing
#     with truthy or falsey because it's editorially the most
#     ambiguous label (Lenient projects → Truthy, Strict projects →
#     Falsey). A ``Mostly True`` consensus with one ``Exaggerated``
#     voter SHOULD show dissent — that's a genuine framing
#     disagreement worth surfacing.
#   * ``Unverifiable`` and any unknown label each get their own
#     family so a defensive vote of "Unverifiable" against a "True"
#     consensus is still flagged.
_VERDICT_FAMILY: dict[str, str] = {
    "True": "truthy",
    "Mostly True": "truthy",
    "Exaggerated": "exaggerated",
    "Misleading": "falsey",
    "False": "falsey",
    "Unverifiable": "unverifiable",
}


def _verdict_family(label_str: str) -> str:
    """Map a fine-axis verdict label to its dissent-flagging family.

    See ``_VERDICT_FAMILY`` for the full mapping + rationale. Unknown
    labels (defensive — should never appear in production) get a unique
    family keyed by the label itself, so every cross-label comparison
    against an unknown stays "dissent."
    """
    return _VERDICT_FAMILY.get(label_str, f"unknown:{label_str}")


# How many normalized leading characters of a caveat form its dedup key.
# Different models often volunteer caveats that share the same opening
# sentence but diverge in phrasing later; collapsing on a normalized prefix
# groups them under a single list item without demanding exact-string
# equality (fix for C9).
_CAVEAT_PREFIX_LEN = 80


def _normalize_caveat_signature(text: str, *, length: int = _CAVEAT_PREFIX_LEN) -> str:
    """
    Return a dedup signature for a caveat.

    Two caveats share a signature when they agree on their first ``length``
    characters of normalized content: lowercased, whitespace collapsed to a
    single space, with leading/trailing punctuation stripped. This is the
    heuristic that lets "Source reliability may vary."  from one model and
    "Source reliability may vary,\\n  as noted above." from another collapse
    into a single caveat-list entry attributed to both.
    """
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    normalized = normalized.strip(".,;:!? \t")
    return normalized[:length]


def _model_attribution(mv: Any) -> str:
    """Display label for a ``ModelVerdict`` in the caveat list.

    Prefers the adapter brand (Anthropic / OpenAI / Google / xAI). Falls
    back to a prettified model_id when the adapter_name is unrecognized so
    legacy rehydrated reports still render a readable label.
    """
    adapter = getattr(mv, "adapter_name", "") or ""
    brand = _ADAPTER_BRAND.get(adapter)
    if brand:
        return brand
    model_id = getattr(mv, "model_id", "") or ""
    return _prettify_model_id(model_id) or (adapter or "Model")


def _render_caveat_block(model_verdicts: list[Any]) -> str:
    """
    Render the per-model caveat callout for a claim card.

    Fixes C8 (no attribution) + C9 (exact-string dedup only). Groups caveats
    by normalized-prefix signature, attributes each surviving caveat to the
    contributing adapter brand(s), and preserves first-seen insertion order
    so the visible transcript matches the underlying model ordering.

    Returns an empty string when no model supplied a non-empty caveat.
    """
    groups: list[dict[str, Any]] = []
    sig_to_idx: dict[str, int] = {}
    for mv in model_verdicts:
        if getattr(mv, "no_response", False):
            continue
        raw = (getattr(mv, "caveats", "") or "").strip()
        if not raw:
            continue
        sig = _normalize_caveat_signature(raw)
        if not sig:
            continue
        label = _model_attribution(mv)
        idx = sig_to_idx.get(sig)
        if idx is None:
            sig_to_idx[sig] = len(groups)
            groups.append({"text": raw, "labels": [label], "seen": {label}})
        else:
            g = groups[idx]
            if label not in g["seen"]:
                g["labels"].append(label)
                g["seen"].add(label)

    if not groups:
        return ""

    items: list[str] = []
    for g in groups:
        attribution = ", ".join(g["labels"])
        items.append(
            '<li class="caveat-item">'
            f'<span class="caveat-attribution">{_esc(attribution)}</span>'
            f'<span class="caveat-text">{_esc(g["text"])}</span>'
            '</li>'
        )

    return (
        '<div class="caveat">'
        '<div class="caveat-label">Model notes</div>'
        f'<ul class="caveat-list">{"".join(items)}</ul>'
        '</div>'
    )


# Provider brand names + the production default model for each adapter.
# Used as fallback labels when a ModelVerdict lacks an explicit model_id
# (e.g. older reports rehydrated after a schema migration).
_ADAPTER_BRAND = {
    "anthropic": "Anthropic",
    "openai":    "OpenAI",
    "gemini":    "Google",
    "grok":      "xAI",
}
_ADAPTER_DEFAULT_MODEL = {
    "anthropic": "claude-opus-4-7",
    "openai":    "gpt-5.4",
    "gemini":    "gemini-2.5-pro",
}
_MODEL_TOKEN_UPPER = {"gpt", "ai"}


def _prettify_model_id(mid: str) -> str:
    """Turn 'claude-opus-4-7' → 'Claude Opus 4.7', 'gpt-5.4' → 'GPT 5.4'."""
    if not mid:
        return ""
    parts = [p for p in mid.split("-") if p]
    out: list[str] = []
    for p in parts:
        if p.isdigit() and out and out[-1][-1:].isdigit():
            out[-1] = f"{out[-1]}.{p}"
        elif p.lower() in _MODEL_TOKEN_UPPER:
            out.append(p.upper())
        else:
            out.append(p[:1].upper() + p[1:])
    return " ".join(out)


def _pretty_model_label(adapter: str, model_id: str = "") -> str:
    """Return a human-friendly provider + model label, e.g. 'Anthropic Claude Opus 4.7'."""
    adapter_key = (adapter or "").strip().lower()
    mid = (model_id or "").strip()
    if not mid:
        mid = _ADAPTER_DEFAULT_MODEL.get(adapter_key, "")
    brand = _ADAPTER_BRAND.get(adapter_key, (adapter_key.capitalize() if adapter_key else ""))
    pretty_mid = _prettify_model_id(mid) if mid else ""
    pieces = [p for p in (brand, pretty_mid) if p]
    return " ".join(pieces) or adapter or "model"


# Family aggregation for headline verdicts: the family sets + the percent-true
# math live in ``truthbot.publish.aggregation`` (single source of truth, 1.6);
# ``_TRUE_FAMILY`` / ``_ADVERSE_FAMILY`` above are re-exported aliases.


def _family_verdict(dist: dict[str, int]) -> tuple[str, str, str]:
    """Percent-true headline — thin wrapper over
    :func:`truthbot.publish.aggregation.family_verdict` (see its docstring
    for the bands + rationale). Returns (label_text, css_class, ratio_text).
    """
    fam = _agg_family_verdict(dist)
    return fam.label, fam.css, fam.ratio_text


def _binary_verdict(dist: dict[str, int]) -> tuple[str, str, str]:
    """Both lenses now show the same percent-true headline (jackie,
    2026-07-25) — the Strict/Lenient distinction lives in the graded vs
    coarse DISTRIBUTIONS below the headline, not in the headline wording.
    One computation, one presentation: delegates to ``_family_verdict``."""
    return _family_verdict(dist)


def _headline_verdict(dist: dict[str, int]) -> tuple[str, str]:
    """Headline verdict label + CSS class for a report (family-aggregated —
    see ``_family_verdict`` for the bands and rationale)."""
    label, css, _ratio = _family_verdict(dist)
    return label, css


# Coarse-axis labels that read naturally on their own and shouldn't get
# the "Mostly"/"Largely" prefix.
_COARSE_ALREADY_QUALIFIED: frozenset[str] = frozenset({"Truthy", "Falsey"})


def _headline_verdict_coarse(dist: dict[str, int]) -> tuple[str, str]:
    """Headline verdict + CSS class for a coarse-axis distribution.

    Family-aggregated like :func:`_headline_verdict` — the coarse buckets
    (True/Truthy vs Falsey/False) fold into the same two families, and
    "Models split" counts as an abstention alongside Unverifiable."""
    label, css, _ratio = _family_verdict(dist)
    return label, css


# Tier rules live in truthbot.verify.source_tiers — ONE implementation shared
# with the pipeline (Claim Eval v3 PR-A). This module used to keep its own
# copies of the domain lists, and they had drifted: the connector counted
# federalreserve.gov and stlouisfed.org as Government while this file did not,
# so a FRASER/FRED source rendered as bottom-tier T6 despite being stored as
# Government in its I5 provenance record.


def _tier_bucket(url: str) -> str:
    """Classify a source URL into one of: gov, wire, news, fc, political, other."""
    return TIER_BUCKET[classify_tier(url)]


def _tier_counts_for_report(site_report) -> dict[str, int]:
    """Tally deduped source URLs per tier bucket across all checkable bundles."""
    seen: set[str] = set()
    counts = {"gov": 0, "wire": 0, "news": 0, "fc": 0, "political": 0, "other": 0}
    for bundle in site_report.checkable_bundles:
        for mv in bundle.model_verdicts:
            for url in mv.web_sources or []:
                if not url or url in seen:
                    continue
                seen.add(url)
                counts[_tier_bucket(url)] += 1
    return counts


def _tier_badge(url: str) -> str:
    """Return an evidence-tier span for a source URL."""
    code, css = TIER_DISPLAY[classify_tier(url)]
    return f'<span class="evidence-tier {css}">{code}</span>'


# Layer 4 — anti-hallucination publish-layer rendering.
#
# A URL's reachability classification (from
# ``truthbot.verify.url_validation.classify_failure``) determines how we
# render it on the static site:
#
#   * ``ok``                       → "verified" (default rendering).
#   * ``bot-blocked`` / ``transient`` / ``unknown`` → "unverified"
#     (muted, small "unverified" badge so readers know we couldn't
#     confirm the URL but the citation likely still exists).
#   * ``dead-4xx`` / ``malformed`` / ``dns`` / ``cert-error`` →
#     "broken" — *skipped from the rendered list entirely*. Even if the
#     cleaned sidecar didn't strip these, the publish layer must as a
#     belt-and-suspenders defense against rendering hallucinated or
#     rotted URLs that destroy reader trust.
_RENDER_AS_BROKEN = frozenset({"dead-4xx", "malformed", "dns", "cert-error"})
_RENDER_AS_UNVERIFIED = frozenset({"bot-blocked", "transient", "unknown"})

# Severity rank for collapsing per-verdict URL classifications into a
# single combined-evidence-list rendering decision. Higher rank wins,
# so if model A says "ok" and model B says "dead-4xx" for the same URL
# the reader sees "broken" (and the URL is stripped) rather than being
# told it's verified. The catch-all default keeps unfamiliar new
# classifications from accidentally being treated as worse than broken.
_CLASSIFICATION_RANK = {
    "ok": 0,
    "transient": 1,
    "unknown": 1,
    "bot-blocked": 1,
    "cert-error": 2,
    "dns": 2,
    "malformed": 2,
    "dead-4xx": 2,
}


def _worse_classification(a: "str | None", b: str) -> str:
    """Return the worse of two classification strings (higher rank)."""
    if a is None:
        return b
    return a if _CLASSIFICATION_RANK.get(a, 0) >= _CLASSIFICATION_RANK.get(b, 0) else b


def _classify_source_for_render(
    url: str, classifications: "dict[str, str] | None"
) -> str:
    """Return one of ``"verified"``, ``"unverified"``, or ``"broken"``.

    Defaults to ``"verified"`` when no classification map is provided —
    preserving pre-Layer-4 rendering behavior on reports generated
    before the URL filter ran.
    """
    if not classifications:
        return "verified"
    cls = classifications.get(url)
    if cls is None:
        return "verified"
    if cls in _RENDER_AS_BROKEN:
        return "broken"
    if cls in _RENDER_AS_UNVERIFIED:
        return "unverified"
    return "verified"


def _evidence_list_html(
    urls: list[str],
    *,
    classifications: "dict[str, str] | None" = None,
) -> str:
    """Render evidence URLs as evidence-list structure.

    When ``classifications`` is provided (mapping URL → failure class
    string from ``classify_failure``), URLs are rendered with one of
    three CSS classes (``source-verified`` / ``source-unverified`` /
    skipped-as-broken) so the static site can visually distinguish them.

    Without ``classifications`` every URL renders as verified (the
    pre-Layer-4 behavior), so older publish runs still look identical.
    """
    if not urls:
        return '<p style="font-size:0.88rem;color:var(--ink-muted)">No sources retrieved.</p>'
    items: list[str] = []
    for url in urls[:10]:
        render_cls = _classify_source_for_render(url, classifications)
        if render_cls == "broken":
            # Defense in depth — never render a known-broken URL even
            # if it slipped through filter-sidecar.
            continue
        badge = _tier_badge(url)
        short = url.replace("https://", "").replace("http://", "")
        if len(short) > 80:
            short = short[:77] + "…"
        unverified_badge = ""
        if render_cls == "unverified":
            unverified_badge = (
                ' <span class="source-unverified-badge" '
                'title="Could not be auto-verified at publish time '
                '(bot-blocked or transient error). The URL is most '
                'likely real but you should confirm before relying on '
                'it.">unverified</span>'
            )
        items.append(
            f'<li class="source-{render_cls}"><span class="ev-mark">→</span>{badge}'
            f'<a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(short)}</a>'
            f'{unverified_badge}</li>'
        )
    if not items:
        return '<p style="font-size:0.88rem;color:var(--ink-muted)">No sources retrieved.</p>'
    return f'<ul class="evidence-list">{"".join(items)}</ul>'


def _sources_consulted_html(sources: list[dict], anchor_base: str = "",
                            self_ids: Optional[set[str]] = None) -> str:
    """Render the FULL retrieved evidence pack (all items, not just cited).

    Independent of what the verdict cited: a claim can have a non-empty pack
    yet zero citations (e.g. Unverifiable), and this list still surfaces every
    source that was consulted. Reuses the ``evidence-list`` / tier styling.

    ``anchor_base`` (per-claim unique) makes each pack item a link target
    (``id="{anchor_base}-E5"``) so E-id mentions in model reasoning can jump
    here (2026-07-19 review follow-up).

    ``self_ids`` (PR-A2.1): pack ids whose source org is the speaker's own
    principal — those items carry a "speaker's own org" badge so the reader
    can see at a glance which records are the claimant's.
    """
    if not sources:
        return ""
    self_ids = self_ids or set()
    items: list[str] = []
    for src in sources:
        url = (src.get("url") or "").strip()
        if not url:
            continue
        name = (src.get("source") or "").strip()
        tier = (src.get("tier") or "").strip()
        snippet = (src.get("snippet") or "").strip()
        badge = _tier_badge(url)
        if str(src.get("id") or "").strip() in self_ids:
            badge += ('<span class="ev-self" title="This source is the speaker&#39;s '
                      'own organization (administration, party, or campaign at the '
                      'time of the speech).">speaker&#39;s own org</span>')
        short = url.replace("https://", "").replace("http://", "")
        if len(short) > 80:
            short = short[:77] + "…"
        name_html = f'<span class="ev-src">{_esc(name)}</span>' if name else ""
        tier_html = f'<span class="ev-src">{_esc(tier)}</span>' if tier else ""
        snippet_html = (
            f'<div class="source-snippet">{_esc(snippet)}</div>' if snippet else ""
        )
        # The pack id (E1, E2, …) is what model reasoning cites — render it so
        # "E5 confirms…" in the write-up is traceable to a concrete source
        # (2026-07-19 review: the ids were captured but never displayed).
        pack_id = (src.get("id") or "").strip()
        id_html = f'<span class="ev-id">[{_esc(pack_id)}]</span>' if pack_id else ""
        li_anchor = f' id="{_esc(anchor_base)}-{_esc(pack_id)}"' if anchor_base and pack_id else ""
        items.append(
            f'<li class="source-verified"{li_anchor}><span class="ev-mark">→</span>{id_html}{badge}'
            f'<a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(short)}</a>'
            f'{name_html}{tier_html}{snippet_html}</li>'
        )
    if not items:
        return ""
    return f'<ul class="evidence-list">{"".join(items)}</ul>'


def _model_cited_unverified_html(urls: list[str]) -> str:
    """Render model-reported URLs that didn't survive the tool-grounding intersection.

    A URL ends up here when the LLM emitted it as a citation but the search
    tool's retrieved-URL set for the same call did not contain it. Could be
    (a) a real URL the harness failed to capture, (b) a plausible URL the
    model pattern-matched on a real domain (tool didn't visit that exact
    path), or (c) outright fabrication. We surface host + path so readers
    can verify each citation themselves, but render it italicized and
    non-clickable to make clear we did NOT vouch for it.
    """
    if not urls:
        return ''
    items: list[str] = []
    for url in urls[:10]:
        # Same short-form transform as the verified tier so reader-side
        # comparison stays apples-to-apples.
        short = url.replace("https://", "").replace("http://", "")
        if len(short) > 80:
            short = short[:77] + "…"
        items.append(
            '<li class="source-model-only">'
            '<span class="ev-mark">!</span>'
            f'<span class="ev-src ev-src-model-only">{_esc(short)}</span>'
            ' <span class="source-unverified-badge" title="'
            'Model emitted this URL as a citation but the search tool did '
            'not return it for this call. Could be a real URL the harness '
            'failed to capture, a plausible URL the model pattern-matched '
            'on a real domain, or fabrication. Verify before relying on it.'
            '">didn’t validate</span></li>'
        )
    return (
        '<p class="evidence-model-only-header">'
        f'Model-cited URLs that didn’t validate ({len(urls)}):</p>'
        f'<ul class="evidence-list evidence-list-model-only">{"".join(items)}</ul>'
    )


def _family_rail_html(dist: dict[str, int], label_order: list[str],
                      rail_class: str = "vp-family-rail") -> str:
    """The Truthy/Falsey family rail above a verdict bar.

    The headline ratio ("95 of 132 decided claims false-leaning") sums the two
    FAMILIES, but the bar segments are per-bucket — so the totals weren't
    visibly derivable from the graph (jackie, 2026-07-20). The rail brackets
    the family groups at the same percentage widths as the segments below
    (families are contiguous in ``COARSE_VERDICT_ORDER``: True/Truthy left,
    abstentions middle, Falsey/False right), labeled with the family totals
    the headline uses. Empty when nothing was decided."""
    total = sum(dist.get(l, 0) for l in label_order) or 1
    t = sum(dist.get(l, 0) for l in label_order if l in _TRUE_FAMILY)
    f = sum(dist.get(l, 0) for l in label_order if l in _ADVERSE_FAMILY)
    abstain = total - t - f
    decided = t + f
    if not decided:
        return ""
    cells: list[str] = []
    if t:
        cells.append(
            f'<div class="fam fam-true" style="width:{t/total*100:.1f}%" '
            f'title="{t} of {decided} decided claims true-leaning">'
            f'Truthy-leaning <span class="n">{t}</span></div>')
    if abstain > 0:
        cells.append(
            f'<div class="fam fam-abstain" style="width:{abstain/total*100:.1f}%" '
            f'title="{abstain} claims not decided (Unverifiable or Models split) — '
            f'excluded from the leaning denominator">{abstain} undecided</div>')
    if f:
        cells.append(
            f'<div class="fam fam-false" style="width:{f/total*100:.1f}%" '
            f'title="{f} of {decided} decided claims false-leaning">'
            f'Falsey-leaning <span class="n">{f}</span></div>')
    return f'<div class="{rail_class}">{"".join(cells)}</div>'


def _verdict_bar_html(
    dist: dict[str, int],
    bar_class: str = "vp-bar",
    order: Optional[list[str]] = None,
    family_rail: bool = False,
) -> str:
    """Render a verdict bar + legend.

    ``order`` controls which labels are iterated and in what order. Defaults
    to the 6-bucket fine axis (``VERDICT_ORDER``) for backward compatibility,
    but every aggregate caller now passes ``COARSE_VERDICT_ORDER`` (plus
    "Models split" implicitly skipped since it carries no semantic position
    on the bar). ``family_rail`` prepends the Truthy/Falsey family rail so the
    headline's leaning totals are traceable to the graph.
    """
    label_order = order if order is not None else VERDICT_ORDER
    total = sum(dist.get(l, 0) for l in label_order) or 1
    segs = []
    for label in label_order:
        count = dist.get(label, 0)
        if count == 0:
            continue
        pct = count / total * 100
        css = _verdict_css(label)
        segs.append(
            f'<div class="seg v-{css}" style="width:{pct:.1f}%" '
            f'title="{_esc(label)}: {count}">{count}</div>'
        )
    parts_aria = [f"{dist.get(l,0)} {l}" for l in label_order if dist.get(l, 0) > 0]
    aria = "Verdict distribution: " + ", ".join(parts_aria)
    bar_html = (
        f'<div class="{bar_class}" role="img" aria-label="{_esc(aria)}">'
        f'{"".join(segs)}</div>'
    )
    legend_items = []
    for label in label_order:
        count = dist.get(label, 0)
        if count == 0 and family_rail:
            # Aggregate mode iterates the union of both axes — hide the
            # unused axis's labels instead of a row of dimmed zeros.
            continue
        css = _verdict_css(label)
        zero_cls = " zero" if count == 0 else ""
        legend_items.append(
            f'<div class="legend-item{zero_cls}">'
            f'<span class="swatch v-{css}"></span>'
            f'{_esc(label)} <span class="ct">{count}</span>'
            '</div>'
        )
    legend_html = f'<div class="vp-legend">{"".join(legend_items)}</div>'
    rail_html = _family_rail_html(dist, label_order) if family_rail else ""
    return rail_html + bar_html + "\n" + legend_html


def _pca_prompt_text() -> str:
    """The ADOPTED verdict prompt set (calibrated open-book PCA seats), rendered
    as one stable text block. This is what the footer hash commits to — it
    changed from the legacy SYNTHESIS_SYSTEM when the PCA engine became the
    production path (About refresh, 2026-07-20)."""
    try:
        from truthbot.verdict.prompts import CALIBRATED_OPEN_BOOK_PROMPTS
        return "\n\n".join(
            f"── {role.upper()} ──\n{CALIBRATED_OPEN_BOOK_PROMPTS[role]}"
            for role in ("proposer", "critic", "arbiter"))
    except Exception:
        return "(prompt unavailable)"


def _prompt_hash() -> str:
    text = _pca_prompt_text()
    if text == "(prompt unavailable)":
        return "unknown"
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# ── Truthy SVG + tap hint ────────────────────────────────────────────────────

_TRUTHY_SVG = (
    '<svg id="mascot" width="170" height="204" viewBox="0 0 300 360" class="state-true">'
    '<defs>'
    '<radialGradient id="headShade" cx="35%" cy="28%" r="75%">'
    '<stop offset="0%" stop-color="#fff8e0"/><stop offset="45%" stop-color="#f2e3c4"/>'
    '<stop offset="85%" stop-color="#c9a876"/><stop offset="100%" stop-color="#a58654"/>'
    '</radialGradient>'
    '<radialGradient id="bodyShade" cx="35%" cy="25%" r="80%">'
    '<stop offset="0%" stop-color="#fff2d5"/><stop offset="50%" stop-color="#ebdfc5"/>'
    '<stop offset="100%" stop-color="#b89968"/>'
    '</radialGradient>'
    '<linearGradient id="rimLight" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#fff8e0" stop-opacity="0"/>'
    '<stop offset="85%" stop-color="#fff8e0" stop-opacity="0"/>'
    '<stop offset="100%" stop-color="#fff4d0" stop-opacity="0.95"/>'
    '</linearGradient>'
    '<radialGradient id="visorShade" cx="50%" cy="35%" r="70%">'
    '<stop offset="0%" stop-color="#3a2e24"/><stop offset="70%" stop-color="#1a1410"/>'
    '<stop offset="100%" stop-color="#0a0604"/>'
    '</radialGradient>'
    '<radialGradient id="brassShade" cx="35%" cy="30%">'
    '<stop offset="0%" stop-color="#f4d98a"/><stop offset="50%" stop-color="#c9a158"/>'
    '<stop offset="100%" stop-color="#7a5e2e"/>'
    '</radialGradient>'
    '<radialGradient id="eyeLedTrue" cx="38%" cy="32%" r="75%">'
    '<stop offset="0%" stop-color="#f0fff8"/><stop offset="40%" stop-color="#a0ffe0"/>'
    '<stop offset="75%" stop-color="#50d8b0"/><stop offset="100%" stop-color="#1e8060"/>'
    '</radialGradient>'
    '<radialGradient id="eyeLedIffy" cx="38%" cy="32%" r="75%">'
    '<stop offset="0%" stop-color="#fffae0"/><stop offset="40%" stop-color="#ffe088"/>'
    '<stop offset="75%" stop-color="#e8a830"/><stop offset="100%" stop-color="#805818"/>'
    '</radialGradient>'
    '<radialGradient id="eyeLedLie" cx="38%" cy="32%" r="80%">'
    '<stop offset="0%" stop-color="#e8f4ff"/><stop offset="40%" stop-color="#9cc0e8"/>'
    '<stop offset="75%" stop-color="#5a82b8"/><stop offset="100%" stop-color="#2a5890"/>'
    '</radialGradient>'
    '<radialGradient id="ledGradTrue" cx="40%" cy="35%">'
    '<stop offset="0%" stop-color="#b8f5c8"/><stop offset="50%" stop-color="#5ac075"/>'
    '<stop offset="100%" stop-color="#2a7840"/>'
    '</radialGradient>'
    '<radialGradient id="ledGradIffy" cx="40%" cy="35%">'
    '<stop offset="0%" stop-color="#fff0a8"/><stop offset="50%" stop-color="#e8b850"/>'
    '<stop offset="100%" stop-color="#8a6520"/>'
    '</radialGradient>'
    '<radialGradient id="ledGradLie" cx="40%" cy="35%">'
    '<stop offset="0%" stop-color="#cce4ff"/><stop offset="50%" stop-color="#5a8ec0"/>'
    '<stop offset="100%" stop-color="#2a4a78"/>'
    '</radialGradient>'
    '<filter id="eyeGlow" x="-50%" y="-50%" width="200%" height="200%">'
    '<feGaussianBlur stdDeviation="2" result="blur"/>'
    '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
    '</filter>'
    '<filter id="softGlow" x="-60%" y="-60%" width="220%" height="220%">'
    '<feGaussianBlur stdDeviation="4"/>'
    '</filter>'
    '<filter id="strongGlow" x="-100%" y="-100%" width="300%" height="300%">'
    '<feGaussianBlur stdDeviation="8" result="blur"/>'
    '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
    '</filter>'
    '<radialGradient id="floorShadowGrad">'
    '<stop offset="0%" stop-color="rgba(0,0,0,0.5)"/>'
    '<stop offset="60%" stop-color="rgba(0,0,0,0.15)"/>'
    '<stop offset="100%" stop-color="rgba(0,0,0,0)"/>'
    '</radialGradient>'
    '<pattern id="scanlines" width="2" height="3" patternUnits="userSpaceOnUse">'
    '<rect width="2" height="3" fill="transparent"/>'
    '<line x1="0" y1="1.5" x2="2" y2="1.5" stroke="#000000" stroke-width="0.4" opacity="0.25"/>'
    '</pattern>'
    '</defs>'
    '<g id="floorShadow">'
    '<ellipse cx="150" cy="352" rx="95" ry="8" fill="url(#floorShadowGrad)"/>'
    '<ellipse cx="150" cy="354" rx="70" ry="5" fill="rgba(0,0,0,0.22)"/>'
    '</g>'
    '<g id="character">'
    '<ellipse cx="115" cy="325" rx="26" ry="10" fill="#4a3a28" opacity="0.25"/>'
    '<rect x="92" y="312" width="48" height="20" rx="9" fill="url(#brassShade)"/>'
    '<rect x="92" y="310" width="48" height="5" rx="2" fill="#f4d98a"/>'
    '<circle cx="100" cy="322" r="1.8" fill="#4a3a28"/>'
    '<circle cx="132" cy="322" r="1.8" fill="#4a3a28"/>'
    '<ellipse cx="185" cy="325" rx="26" ry="10" fill="#4a3a28" opacity="0.25"/>'
    '<rect x="160" y="312" width="48" height="20" rx="9" fill="url(#brassShade)"/>'
    '<rect x="160" y="310" width="48" height="5" rx="2" fill="#f4d98a"/>'
    '<circle cx="168" cy="322" r="1.8" fill="#4a3a28"/>'
    '<circle cx="200" cy="322" r="1.8" fill="#4a3a28"/>'
    '<g id="armLeft"><g id="armLeftSwing" transform="rotate(0 88 253)">'
    '<rect x="81" y="253" width="14" height="48" rx="7" fill="url(#bodyShade)" stroke="#8a7550" stroke-width="1.5"/>'
    '<rect x="81" y="253" width="3" height="48" rx="1.5" fill="#fff2d5" opacity="0.5"/>'
    '<circle cx="88" cy="301" r="8" fill="url(#brassShade)"/>'
    '<circle cx="86" cy="298" r="2.5" fill="#f4d98a" opacity="0.8"/>'
    '</g><circle cx="88" cy="253" r="9" fill="url(#brassShade)"/></g>'
    '<g id="bodyGroup">'
    '<rect x="86" y="244" width="128" height="80" rx="16" fill="#5a4028" opacity="0.3" filter="url(#softGlow)"/>'
    '<rect x="88" y="240" width="124" height="80" rx="14" fill="url(#bodyShade)" stroke="#8a7550" stroke-width="2"/>'
    '<rect x="88" y="240" width="124" height="80" rx="14" fill="url(#rimLight)" opacity="0.8"/>'
    '<rect x="94" y="245" width="112" height="10" rx="5" fill="#fff2d5" opacity="0.4"/>'
    '<line x1="94" y1="262" x2="206" y2="262" stroke="#8a7550" stroke-width="1.5" opacity="0.55"/>'
    '<line x1="94" y1="300" x2="206" y2="300" stroke="#8a7550" stroke-width="1.5" opacity="0.55"/>'
    '<circle cx="98" cy="262" r="2" fill="#8a7550"/>'
    '<circle cx="202" cy="262" r="2" fill="#8a7550"/>'
    '<circle cx="98" cy="300" r="2" fill="#8a7550"/>'
    '<circle cx="202" cy="300" r="2" fill="#8a7550"/>'
    '<g transform="translate(125 252)">'
    '<rect x="-35" y="-7" width="70" height="14" rx="2.5" fill="#8a7550"/>'
    '<rect x="-34" y="-6" width="68" height="12" rx="2" fill="url(#brassShade)" stroke="#5a4028" stroke-width="0.6"/>'
    '<rect x="-33" y="-5.5" width="66" height="2" rx="1" fill="#f4d98a" opacity="0.8"/>'
    '<text x="0" y="3.5" text-anchor="middle" font-family="Georgia,serif"'
    ' font-size="10" font-weight="700" font-style="italic" fill="#3a2e1f" letter-spacing="0.5">Truthy M.</text>'
    '<circle cx="-31" cy="0" r="0.8" fill="#3a2e1f"/>'
    '<circle cx="31" cy="0" r="0.8" fill="#3a2e1f"/>'
    '</g>'
    '<g transform="translate(100, 272)">'
    '<rect x="0" y="0" width="24" height="15" rx="1.5" fill="#f0e8d0" stroke="#8a7550" stroke-width="0.8"/>'
    '<rect x="9" y="1" width="14" height="1.4" fill="#c44545"/>'
    '<rect x="9" y="3.7" width="14" height="1.4" fill="#c44545"/>'
    '<rect x="1" y="6.5" width="22" height="1.4" fill="#c44545"/>'
    '<rect x="1" y="9.3" width="22" height="1.4" fill="#c44545"/>'
    '<rect x="1" y="12.1" width="22" height="1.4" fill="#c44545"/>'
    '<rect x="1" y="1" width="8" height="6" fill="#3a4a78"/>'
    '<circle cx="2.5" cy="2.5" r="0.35" fill="#fff4d0"/>'
    '<circle cx="5" cy="2.5" r="0.35" fill="#fff4d0"/>'
    '<circle cx="7.5" cy="2.5" r="0.35" fill="#fff4d0"/>'
    '<circle cx="3.7" cy="4" r="0.35" fill="#fff4d0"/>'
    '<circle cx="6.2" cy="4" r="0.35" fill="#fff4d0"/>'
    '<circle cx="2.5" cy="5.5" r="0.35" fill="#fff4d0"/>'
    '<circle cx="5" cy="5.5" r="0.35" fill="#fff4d0"/>'
    '<circle cx="7.5" cy="5.5" r="0.35" fill="#fff4d0"/>'
    '<rect x="0" y="0" width="24" height="15" rx="1.5" fill="#5a4028" opacity="0.1"/>'
    '</g>'
    '<circle cx="170" cy="283" r="11" fill="#5a4028"/>'
    '<circle cx="170" cy="283" r="10" fill="url(#brassShade)"/>'
    '<circle id="ledHalo" cx="170" cy="283" r="16" fill="#5ac075" opacity="0.35" filter="url(#strongGlow)"/>'
    '<circle id="led" cx="170" cy="283" r="7" fill="url(#ledGradTrue)"/>'
    '<circle cx="168" cy="281" r="2.5" fill="#ffffff" opacity="0.85"/>'
    '<circle cx="171" cy="285" r="0.8" fill="#ffffff" opacity="0.6"/>'
    '<path d="M 195 252 Q 204 260 198 270" stroke="#5a4028" stroke-width="1.8" fill="none" opacity="0.55"/>'
    '</g>'
    '<g id="headGroup">'
    '<ellipse cx="150" cy="245" rx="90" ry="8" fill="#3a2e1f" opacity="0.4" filter="url(#softGlow)"/>'
    '<ellipse cx="53" cy="148" rx="11" ry="20" fill="url(#brassShade)"/>'
    '<ellipse cx="50" cy="148" rx="6" ry="15" fill="#5a4028"/>'
    '<ellipse cx="48" cy="144" rx="2" ry="6" fill="#2a1e10" opacity="0.8"/>'
    '<ellipse cx="247" cy="148" rx="11" ry="20" fill="url(#brassShade)"/>'
    '<ellipse cx="250" cy="148" rx="6" ry="15" fill="#5a4028"/>'
    '<ellipse cx="252" cy="144" rx="2" ry="6" fill="#2a1e10" opacity="0.8"/>'
    '<ellipse cx="150" cy="148" rx="100" ry="93" fill="url(#headShade)" stroke="#8a7550" stroke-width="2"/>'
    '<ellipse cx="150" cy="148" rx="100" ry="93" fill="url(#rimLight)" opacity="0.7"/>'
    '<ellipse cx="108" cy="95" rx="26" ry="14" fill="#ffffff" opacity="0.55"/>'
    '<ellipse cx="104" cy="92" rx="10" ry="5" fill="#ffffff" opacity="0.9"/>'
    '<ellipse cx="140" cy="80" rx="8" ry="3" fill="#ffffff" opacity="0.4"/>'
    '<path d="M 210 110 L 218 118" stroke="#8a7550" stroke-width="1" opacity="0.5"/>'
    '<circle cx="72" cy="180" r="3" fill="#8a7550"/>'
    '<circle cx="72" cy="180" r="2" fill="url(#brassShade)"/>'
    '<circle cx="228" cy="180" r="3" fill="#8a7550"/>'
    '<circle cx="228" cy="180" r="2" fill="url(#brassShade)"/>'
    '<rect x="60" y="118" width="180" height="72" rx="36" fill="url(#visorShade)"/>'
    '<rect x="62" y="120" width="176" height="3" rx="1.5" fill="#6a5442" opacity="0.7"/>'
    '<rect x="62" y="186" width="176" height="2" rx="1" fill="#000000" opacity="0.5"/>'
    '<ellipse cx="100" cy="135" rx="30" ry="6" fill="#ffffff" opacity="0.08"/>'
    '<g id="eyeLeftGroup" transform="translate(115 154) rotate(0)"><g class="eye-shape">'
    '<rect class="eye-neutral eye-led" x="-14" y="-16" width="28" height="32" rx="8" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
    '<path class="eye-happy eye-led" d="M -16 4 L -16 -1 Q 0 -16 16 -1 L 16 4 Q 0 -4 -16 4 Z" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
    '<rect class="eye-iffy eye-led" x="-16" y="-6" width="32" height="12" rx="5" fill="url(#eyeLedIffy)" filter="url(#eyeGlow)"/>'
    '<rect class="eye-sad eye-led" x="-17" y="-17" width="34" height="34" rx="8" fill="url(#eyeLedLie)" filter="url(#eyeGlow)"/>'
    '<rect x="-19" y="-20" width="38" height="40" fill="url(#scanlines)" pointer-events="none" opacity="0.7"/>'
    '</g></g>'
    '<g id="eyeRightGroup" transform="translate(185 154) rotate(0)"><g class="eye-shape">'
    '<rect class="eye-neutral eye-led" x="-14" y="-16" width="28" height="32" rx="8" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
    '<path class="eye-happy eye-led" d="M -16 4 L -16 -1 Q 0 -16 16 -1 L 16 4 Q 0 -4 -16 4 Z" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
    '<rect class="eye-iffy eye-led" x="-16" y="-6" width="32" height="12" rx="5" fill="url(#eyeLedIffy)" filter="url(#eyeGlow)"/>'
    '<rect class="eye-sad eye-led" x="-17" y="-17" width="34" height="34" rx="8" fill="url(#eyeLedLie)" filter="url(#eyeGlow)"/>'
    '<rect x="-19" y="-20" width="38" height="40" fill="url(#scanlines)" pointer-events="none" opacity="0.7"/>'
    '</g></g>'
    # Tears: lower-inner sad-eye (lie pose: eye groups at y=170).
    '<g transform="translate(119 172)"><g id="tearLeft">'
    '<rect x="-3" y="0" width="6" height="6" rx="0.8" fill="#9cc8e8"/>'
    '<rect x="-6" y="6" width="6" height="6" rx="0.8" fill="#b8dcf0"/>'
    '<rect x="0"  y="6" width="6" height="6" rx="0.8" fill="#b8dcf0"/>'
    '<rect x="-6" y="12" width="6" height="6" rx="0.8" fill="#7eb4d8"/>'
    '<rect x="0"  y="12" width="6" height="6" rx="0.8" fill="#7eb4d8"/>'
    '<rect x="-3" y="18" width="6" height="6" rx="0.8" fill="#4a86b8"/>'
    '</g></g>'
    '<g transform="translate(181 172)"><g id="tearRight">'
    '<rect x="-3" y="0" width="6" height="6" rx="0.8" fill="#9cc8e8"/>'
    '<rect x="-6" y="6" width="6" height="6" rx="0.8" fill="#b8dcf0"/>'
    '<rect x="0"  y="6" width="6" height="6" rx="0.8" fill="#b8dcf0"/>'
    '<rect x="-6" y="12" width="6" height="6" rx="0.8" fill="#7eb4d8"/>'
    '<rect x="0"  y="12" width="6" height="6" rx="0.8" fill="#7eb4d8"/>'
    '<rect x="-3" y="18" width="6" height="6" rx="0.8" fill="#4a86b8"/>'
    '</g></g>'
    '<g id="antenna">'
    '<path d="M 150 62 Q 148 50 152 38" stroke="url(#brassShade)" stroke-width="3" fill="none" stroke-linecap="round"/>'
    '<circle cx="152" cy="36" r="5.5" fill="#f4c86a" filter="url(#softGlow)" opacity="0.6"/>'
    '<circle cx="152" cy="36" r="4" fill="#ffd870"/>'
    '<circle cx="151" cy="35" r="1.5" fill="#fff8d0"/>'
    '</g>'
    '</g>'
    '<g id="armRight">'
    '<g id="armRightSwing" transform="rotate(-110 212 253)">'
    '<rect x="205" y="253" width="14" height="46" rx="7" fill="url(#bodyShade)" stroke="#8a7550" stroke-width="1.5"/>'
    '<rect x="205" y="253" width="3" height="46" rx="1.5" fill="#fff2d5" opacity="0.5"/>'
    '<circle cx="212" cy="299" r="8" fill="url(#brassShade)"/>'
    '<circle cx="210" cy="297" r="2.5" fill="#f4d98a" opacity="0.8"/>'
    '</g>'
    '<circle cx="212" cy="253" r="9" fill="url(#brassShade)"/>'
    '</g>'
    '<g id="clipboard" transform="translate(228 218) rotate(-8)">'
    '<rect x="1" y="2" width="38" height="50" rx="3" fill="#3a2e1f" opacity="0.35"/>'
    '<rect x="0" y="0" width="38" height="50" rx="3" fill="#d4b585" stroke="#5a4028" stroke-width="1.5"/>'
    '<rect x="0" y="0" width="38" height="4" rx="1.5" fill="#e8c89a"/>'
    '<rect x="12" y="-3" width="14" height="7" rx="1.5" fill="#5a4028"/>'
    '<rect x="15" y="-5" width="8" height="5" rx="1" fill="#8a7550"/>'
    '<rect x="3" y="6" width="32" height="42" rx="1" fill="#fff8e8"/>'
    '<line x1="6" y1="14" x2="26" y2="14" stroke="#8a7550" stroke-width="0.7" opacity="0.55"/>'
    '<path d="M 29 12 L 31 15 L 34 10" stroke="#3a8a50" stroke-width="1.2" fill="none" stroke-linecap="round"/>'
    '<line x1="6" y1="22" x2="26" y2="22" stroke="#8a7550" stroke-width="0.7" opacity="0.55"/>'
    '<path d="M 29 19 L 34 24 M 34 19 L 29 24" stroke="#c44545" stroke-width="1.2" stroke-linecap="round"/>'
    '<line x1="6" y1="30" x2="26" y2="30" stroke="#8a7550" stroke-width="0.7" opacity="0.55"/>'
    '<path d="M 29 28 L 31 31 L 34 26" stroke="#3a8a50" stroke-width="1.2" fill="none" stroke-linecap="round"/>'
    '<line x1="6" y1="38" x2="22" y2="38" stroke="#8a7550" stroke-width="0.7" opacity="0.55"/>'
    '</g>'
    '</g>'
    '</svg>'
)

_TRUTHY_TAP_HINT = (
    '<div class="truthy-tap-hint">'
    '<svg class="icon" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3a4.5 4.5 0 00-2.5-4.03v8.05a4.5 4.5 0 002.5-4.02zM14 3.23v2.06a7 7 0 010 13.42v2.06a9 9 0 000-17.54z"/>'
    '</svg>'
    '<span class="tap-hint-label">Tap</span>'
    '</div>'
)


def _initial_bubble(mood: str, claim_count: int) -> tuple[str, str]:
    """Return (initial_text, css_class) prior to JS activation."""
    state_map = {"happy": "true", "iffy": "iffy", "sad": "lie"}
    state = state_map.get(mood, "iffy")
    bubble_class_map = {"true": "is-true", "iffy": "is-iffy", "lie": "is-lie"}
    captions_single = {
        "true": "That checks out. Sources match!",
        "iffy": "Hmm… let me double-check my sources.",
        "lie":  "Oh no… that isn't true.",
    }
    captions_multi = {
        "true": "All sources check out. Looking good!",
        "iffy": "Mixed signals — some hold up, some don't.",
        "lie":  "Oh no… most of this doesn't check out.",
    }
    caps = captions_single if claim_count == 1 else captions_multi
    return caps.get(state, ""), bubble_class_map.get(state, "is-iffy")
def _verdict_panel(site_report) -> str:
    """Build the full .verdict-panel section for a report page."""
    claim_count = len(site_report.checkable_bundles)
    model_count, model_hint = _models_engaged(site_report)
    agree_rate  = site_report.model_agreement_rate
    # 5-bucket Truthy-scale aggregates rendered side-by-side; the Lens
    # chip swaps them in lockstep with the per-claim headline pills.
    # Strict is the published default (matches the per-claim pill + lens chip).
    dist_lenient = site_report.verdict_distribution_lenient
    dist_strict  = site_report.verdict_distribution_strict
    # Lens semantics (2026-07-19): Lenient = the simple Truthy/Falsey lean,
    # Strict = the graded family bands — two presentations of one computation.
    headline_lenient, hcls_lenient, ratio_text_lenient = _binary_verdict(dist_lenient)
    headline_strict,  hcls_strict,  ratio_text_strict  = _family_verdict(dist_strict)

    # Mascot mood derives from the published headline (remediation T0.3), not
    # the independent truthy-score rollup. Since 2026-07-25 the headline TEXT
    # is a percentage ("56% True" — every label ends with "True"), so the
    # mood keys off the headline's color class, which carries the lean:
    # >75% true-share greens (happy), <50% reds (sad), the 50-75% yellow
    # band (vt-mid) is iffy — as are Unverifiable / no-claims ("neutral").
    if hcls_strict == "vt-true":
        mood = "happy"
    elif hcls_strict == "vt-false":
        mood = "sad"
    else:  # neutral: coin-flip percentage / Unverifiable / no claims
        mood = "iffy"
    state_map = {"happy": "true", "iffy": "iffy", "sad": "lie"}
    svg_state = "state-" + state_map.get(mood, "iffy")

    bubble_text, bubble_cls = _initial_bubble(mood, claim_count)

    svg_html = _TRUTHY_SVG.replace('class="state-true"', 'class="' + svg_state + '"')
    aria_mood = {"happy": "happy", "iffy": "uncertain", "sad": "sad"}.get(mood, "uncertain")

    widget = (
        '<div class="vp-truthy-col">'
        + '<div class="truthy-frame" id="truthy-mascot-widget"'
        + ' data-mood="' + mood + '" data-claim-count="' + str(claim_count) + '"'
        + ' role="button" tabindex="0"'
        + ' aria-label="Truthy McTruthface, the truth-bot mascot. Currently ' + aria_mood + '. Click to hear.">'
        + svg_html
        + _TRUTHY_TAP_HINT
        + '</div>'
        + '<div class="truthy-bubble ' + bubble_cls + '" id="truthy-bubble">' + _esc(bubble_text) + '</div>'
        + '</div>'
    )

    # Two paired headline+ratio blocks, one per lens. Strict is the
    # published default (2026-04-30 editorial flip from Lenient) so it
    # ships first and visible; Lenient ships ``hidden`` and the lens
    # chip flips them. Non-JS clients therefore see Strict.
    text_col = (
        '<div class="vp-text-col">'
        + '<div class="vp-headline-lens" data-lens-axis="strict">'
        + '<div class="vp-verdict ' + hcls_strict + '">' + _esc(headline_strict) + '</div>'
        + '<div class="vp-ratio">' + _esc(ratio_text_strict) + '</div>'
        + '</div>'
        + '<div class="vp-headline-lens" data-lens-axis="lenient" hidden>'
        + '<div class="vp-verdict ' + hcls_lenient + '">' + _esc(headline_lenient) + '</div>'
        + '<div class="vp-ratio">' + _esc(ratio_text_lenient) + '</div>'
        + '</div>'
        + '</div>'
    )

    # Headline-stats frames: "Truthy or better" + "False or worse",
    # promoted out of the stats grid into two prominent block frames
    # above the aggregate stats. Family logic and denominator are
    # IDENTICAL to the headline (remediation T0.3): the two families
    # over decided claims, abstentions (Unverifiable / Models split)
    # excluded — the chips and the "N of M decided" ratio can never
    # disagree. Both frames are lens-aware via the paired
    # data-lens-axis pattern; Strict is the published default.
    def _pct(numerator: int, total: int) -> str:
        return format(numerator / total, '.0%') if total else "0%"

    # Family math comes from the same FamilyVerdict the headline used (1.6) —
    # chips and headline literally share one computation.
    _fam_strict = _agg_family_verdict(dist_strict)
    _fam_lenient = _agg_family_verdict(dist_lenient)
    t_strict,  f_strict,  decided_strict  = (
        _fam_strict.true_count, _fam_strict.adverse_count, _fam_strict.decided)
    t_lenient, f_lenient, decided_lenient = (
        _fam_lenient.true_count, _fam_lenient.adverse_count, _fam_lenient.decided)
    truthy_pct_strict  = _pct(t_strict,  decided_strict)
    truthy_pct_lenient = _pct(t_lenient, decided_lenient)
    false_pct_strict   = _pct(f_strict,  decided_strict)
    false_pct_lenient  = _pct(f_lenient, decided_lenient)

    truthy_frame_title = (
        "True-leaning family (True + Mostly True + Truthy) over decided "
        "claims — same families and denominator as the headline; "
        "Unverifiable and Models split are excluded."
    )
    false_frame_title = (
        "False-leaning family (False + Falsey + Misleading + Exaggerated) "
        "over decided claims — same families and denominator as the "
        "headline; Unverifiable and Models split are excluded."
    )

    headline_stats_html = (
        '  <div class="vp-headline-stats">\n'
        + '    <div class="vp-headline-stat vp-stat-truthy" title="' + _esc(truthy_frame_title) + '">\n'
        + '      <div class="vp-stat-icon">' + _icon_svg(_ICON_BODY_TRUTHY_RATE, size=42) + '</div>\n'
        + '      <div class="vp-stat-body">\n'
        + '        <div class="vp-stat-num">'
        + '<span class="lens-target" data-lens-axis="strict">' + truthy_pct_strict + '</span>'
        + '<span class="lens-target" data-lens-axis="lenient" hidden>' + truthy_pct_lenient + '</span>'
        + '</div>\n'
        + '        <div class="vp-stat-lbl">Truthy or better</div>\n'
        + '        <div class="vp-stat-hint">true-leaning / decided claims</div>\n'
        + '      </div>\n'
        + '    </div>\n'
        + '    <div class="vp-headline-stat vp-stat-false" title="' + _esc(false_frame_title) + '">\n'
        + '      <div class="vp-stat-icon">' + _icon_svg(_ICON_BODY_FALSE_RATE, size=42) + '</div>\n'
        + '      <div class="vp-stat-body">\n'
        + '        <div class="vp-stat-num">'
        + '<span class="lens-target" data-lens-axis="strict">' + false_pct_strict + '</span>'
        + '<span class="lens-target" data-lens-axis="lenient" hidden>' + false_pct_lenient + '</span>'
        + '</div>\n'
        + '        <div class="vp-stat-lbl">False or worse</div>\n'
        + '        <div class="vp-stat-hint">false-leaning / decided claims</div>\n'
        + '      </div>\n'
        + '    </div>\n'
        + '  </div>\n'
    )

    panel_stats_html = (
        '  <div class="stats stats-4">\n'
        '    <div class="stat">'
        + _icon_svg(_ICON_BODY_CLAIMS, size=32)
        + '<div class="num">' + str(claim_count) + '</div>'
        + '<div class="lbl">Claims Checked</div></div>\n'
        '    <div class="stat" title="' + _esc(model_hint) + '">'
        + _icon_svg(_ICON_BODY_MODELS_ENGAGED, size=32)
        + '<div class="num">' + str(model_count) + '</div>'
        + '<div class="lbl">Models Engaged</div></div>\n'
        '    <div class="stat">'
        + _icon_svg(_ICON_BODY_MODEL_CONSENSUS, size=32)
        + '<div class="num">' + format(agree_rate, '.0%') + '</div>'
        + '<div class="lbl">Model Consensus</div></div>\n'
        '    <div class="stat">'
        + _icon_svg(_ICON_BODY_LEADERS, size=32)
        + '<div class="num">1</div>'
        + '<div class="lbl">Leaders Reviewed</div></div>\n'
        '  </div>\n'
    )

    # Lens-aware verdict bar + legend. Same paired-element pattern as the
    # headline above. Both axes are 5-bucket so the segment colors match
    # the per-claim pill palette (Truthy / Falsey gradient stops).
    #
    # Each block now carries a ``vp-lens-caption`` so the reader knows
    # which lens they're seeing (the legend below the bar lists buckets,
    # not the active rubric). Strict block is rendered first and visible
    # by default — Lenient ships ``hidden`` and the lens chip flips it.
    bar_html_lenient = _verdict_bar_html(dist_lenient, order=AGGREGATE_BAR_ORDER,
                                         family_rail=True)
    bar_html_strict  = _verdict_bar_html(dist_strict,  order=AGGREGATE_BAR_ORDER,
                                         family_rail=True)
    bar_html = (
        '<div class="vp-bar-lens" data-lens-axis="strict">'
        + '<div class="vp-lens-caption">Strict lens</div>'
        + bar_html_strict
        + '</div>'
        + '<div class="vp-bar-lens" data-lens-axis="lenient" hidden>'
        + '<div class="vp-lens-caption">Lenient lens</div>'
        + bar_html_lenient
        + '</div>'
    )

    model_names = sorted({mv.adapter_name for b in site_report.checkable_bundles for mv in b.model_verdicts})
    model_str = ' · '.join(model_names) if model_names else 'Multi-model'
    src_row_parts: list[str] = []
    if site_report.transcript_source_url:
        t_host = _url_display_host(site_report.transcript_source_url)
        src_row_parts.append(
            '<span><span class="lab">Transcript:</span>'
            + '<a href="' + _esc(site_report.transcript_source_url)
            + '" target="_blank" rel="noopener">' + _esc(t_host) + ' &#x2197;</a></span>'
        )
    if getattr(site_report, 'video_source_url', ''):
        v_host = _url_display_host(site_report.video_source_url)
        src_row_parts.append(
            '<span><span class="lab">Video:</span>'
            + '<a href="' + _esc(site_report.video_source_url)
            + '" target="_blank" rel="noopener">' + _esc(v_host) + ' &#x2197;</a></span>'
        )
    src_row_parts.append('<span><span class="lab">Models:</span>' + _esc(model_str) + '</span>')
    source_row_html = '<div class="source-row">' + ''.join(src_row_parts) + '</div>\n'

    # Guest-anecdote footnote: break out how much of the Unverifiable bucket is
    # the anecdote genre (no public record to check) vs data claims the
    # evidence failed to settle. Anecdote-pilled claims whose panel deadlocked
    # sit in the Models-split bar bucket, not Unverifiable, so they get their
    # own clause — the footnote's arithmetic must reconcile with the bar it
    # sits under (remediation T0.2; the old count lumped both together and
    # could exceed the Unverifiable segment).
    anec_bundles = [b for b in site_report.checkable_bundles if _is_anecdote_unverifiable(b)]
    n_anec_split = sum(1 for b in anec_bundles
                       if b.consensus.consensus_verdict == "Models split")
    n_anec_uv = len(anec_bundles) - n_anec_split
    uv_bucket = dist_strict.get("Unverifiable", 0)
    anecdote_note_html = ""
    if anec_bundles:
        unit = "claim" if uv_bucket == 1 else "claims"
        verb = "is a guest anecdote" if n_anec_uv == 1 else "are guest anecdotes"
        split_clause = ""
        if n_anec_split:
            split_clause = f", plus {n_anec_split} more among the Models-split claims"
        anecdote_note_html = (
            '<p class="vp-anecdote-note" style="font-size:0.85rem;color:var(--ink-muted)">'
            f'{n_anec_uv} of the {uv_bucket} Unverifiable {unit} {verb}{split_clause} — '
            'private individuals\' stories with no independent public record to check.</p>\n'
        )

    # Honest-abstention chip (PR-A2.1 / T1.2): decompose the abstentions so an
    # evidence-AVAILABILITY abstention ("only witness is the claimant") is never
    # read as an integrity signal. Terms sum to claim_count; every number is
    # re-derivable from claims.json (consistency.py checks it). Rendered only
    # when the sub-state exists so legacy reports are byte-identical.
    n_split = dist_strict.get("Models split", 0)
    n_selfsrc = sum(1 for b in site_report.checkable_bundles
                    if _is_self_sourced_unverified(b))
    selfsource_chip_html = ""
    if n_selfsrc:
        n_decided = claim_count - uv_bucket - n_split
        parts = [f"{n_decided} decided",
                 f"{n_selfsrc} unverified — self-sourced only",
                 f"{uv_bucket - n_selfsrc} unverifiable — other"]
        if n_split:
            parts.append(f"{n_split} models split")
        selfsource_chip_html = (
            '<p class="vp-selfsource-chip" '
            f'title="{_esc(SELF_SOURCED_TITLE)}">' + _esc(" · ".join(parts))
            + '</p>\n'
        )

    return (
        '<section class="verdict-panel">\n'
        + '  <div class="vp-headline">' + text_col + widget + '</div>\n'
        + headline_stats_html
        + panel_stats_html
        + '  <div class="vp-bar-wrap">' + bar_html + '</div>\n'
        + selfsource_chip_html
        + anecdote_note_html
        + source_row_html
        + '</section>\n'
    )


# ── Run manifest panel (roadmap [4]) ──────────────────────────────────────────
#
# Per-run provenance + degraded-consensus disclosure on every report.
# Surfaces (a) actual model-id used per adapter (catches gpt-5.4 vs
# gpt-4.1 fallback / opus-vs-fallback-to-sonnet etc.), (b) per-adapter
# coverage (% of claims that produced a non-failed verdict), (c) the
# tool-URL grounding rate (model_reported_sources → web_sources
# survival rate), and (d) the consensus-strength distribution across
# claims.
#
# Editorial framing follows the 2026-05-01 strip-audit findings: lead
# the manifest with COVERAGE PARITY as the credibility signal — a
# clean structural number that tells the reader whether the panel was
# whole at run time. Tool-URL grounding rate goes in a footer caveat,
# de-emphasized, because the metric mixes harness-capture artefacts
# with citation-discipline differences and isn't reliably interpreted
# as "fabrication rate" (see metrics/adapter_interpretability/strip_audit_2026-05.md).
#
# A degraded-consensus banner surfaces above the manifest body when
# any adapter failed to produce a verdict on at least one claim;
# threshold for banner is 1+ no_response on any adapter regardless of
# total. Coverage column highlights the same row(s) for visual
# consistency.

_DEGRADED_COVERAGE_THRESHOLD = 1.0  # any miss flips the banner


def _adapter_run_stats(site_report) -> "list[dict[str, Any]]":
    """Aggregate per-adapter stats across all checkable bundles in a report.

    Returns one dict per unique adapter, sorted by adapter_name, with
    coverage / model-id / mode / tool-URL grounding / no_response
    counters. Coverage denominator is ``len(checkable_bundles)`` —
    every bundle is expected to carry one verdict per registered
    adapter; genuinely missing or ``no_response`` rows count against
    coverage, while PCA split claims (panel voted, no consensus — the
    bridge emits zero ModelVerdicts for them) count as covered and are
    tallied separately in ``split_contributed`` (1.7).
    """
    from collections import defaultdict
    bundles = list(site_report.checkable_bundles)
    total_claims = len(bundles) or 1

    per_adapter: "dict[str, dict[str, Any]]" = defaultdict(
        lambda: {
            "name": "",
            "model_ids": defaultdict(int),
            "modes": defaultdict(int),
            "tiers": defaultdict(int),
            "verdicts_total": 0,
            "no_response": 0,
            "split_contributed": 0,
            "mrs_total": 0,
            "web_total": 0,
        }
    )

    # Each bundle ought to contribute one verdict per adapter; we track
    # per-bundle adapter coverage so a missing adapter row (no MV at
    # all on that bundle) counts as a no_response too.
    seen_per_bundle: "dict[str, set[str]]" = defaultdict(set)

    for bundle in bundles:
        for mv in bundle.model_verdicts:
            a = mv.adapter_name
            slot = per_adapter[a]
            slot["name"] = a
            slot["verdicts_total"] += 1
            seen_per_bundle[bundle.claim.id].add(a)
            if getattr(mv, "no_response", False):
                slot["no_response"] += 1
                continue
            slot["model_ids"][mv.model_id or "?"] += 1
            mode = getattr(mv, "synthesis_mode", "") or "unknown"
            slot["modes"][mode] += 1
            tier = getattr(mv, "tier", "") or "unknown"
            slot["tiers"][tier] += 1
            slot["mrs_total"] += len(getattr(mv, "model_reported_sources", None) or [])
            slot["web_total"] += len(mv.web_sources or [])

    # Backfill when an adapter produced ZERO verdicts on a bundle. Two very
    # different causes used to look identical here (remediation v2, 1.7):
    # a PCA split claim bridges with model_verdicts=[] because the panel
    # VOTED but did not converge — that is a disclosed process outcome, not
    # a coverage hole — while a genuine engine miss really is one. Classify
    # per bundle via the consensus provenance: panel_votes non-empty means
    # the panel ran, so the claim counts as covered ("split_contributed");
    # only true absence counts as no_response and can degrade the run.
    all_adapters = set(per_adapter.keys())
    for bundle in bundles:
        present = seen_per_bundle.get(bundle.claim.id, set())
        prov = getattr(bundle.consensus, "provenance", None)
        panel_ran = bool(prov and prov.panel_votes)
        for missing in all_adapters - present:
            if panel_ran:
                per_adapter[missing]["split_contributed"] += 1
            else:
                per_adapter[missing]["no_response"] += 1

    rows: "list[dict[str, Any]]" = []
    for name in sorted(per_adapter):
        slot = per_adapter[name]
        present = total_claims - slot["no_response"]
        coverage_pct = present / total_claims if total_claims else 1.0
        # Most-common model_id (ties broken by alphabetical order for
        # determinism in tests).
        model_ids = slot["model_ids"]
        if model_ids:
            top_count = max(model_ids.values())
            model_id_top = sorted(
                [m for m, c in model_ids.items() if c == top_count]
            )[0]
            extra_models = len(model_ids) - 1
        else:
            model_id_top = ""
            extra_models = 0
        modes = slot["modes"]
        modes_str = " + ".join(sorted(modes.keys())) if modes else ""
        mrs_total = slot["mrs_total"]
        web_total = slot["web_total"]
        if mrs_total > 0:
            grounding_pct = web_total / mrs_total
        else:
            grounding_pct = None  # nothing to ground — render as "—"
        rows.append(
            {
                "name": name,
                "coverage_present": present,
                "coverage_total": total_claims,
                "coverage_pct": coverage_pct,
                "model_id": model_id_top,
                "extra_models": extra_models,
                "modes_str": modes_str,
                "mrs_total": mrs_total,
                "web_total": web_total,
                "grounding_pct": grounding_pct,
                "split_contributed": slot["split_contributed"],
                "degraded": coverage_pct < _DEGRADED_COVERAGE_THRESHOLD,
            }
        )
    return rows


def _consensus_strength_distribution(site_report) -> "dict[str, int]":
    """Tally consensus_strength values across all checkable bundles."""
    from collections import defaultdict
    counts: "dict[str, int]" = defaultdict(int)
    for bundle in site_report.checkable_bundles:
        s = (getattr(bundle.consensus, "consensus_strength", "") or "none").lower()
        counts[s] += 1
    return dict(counts)


def _run_manifest_html(site_report) -> str:
    """Render the per-run provenance + degraded-consensus aside.

    Empty when there are no checkable bundles (degenerate report).
    Otherwise always renders — the panel doubles as an audit-trail
    even when no adapter degraded.
    """
    rows = _adapter_run_stats(site_report)
    if not rows:
        return ""

    total_claims = rows[0]["coverage_total"]
    degraded_rows = [r for r in rows if r["degraded"]]

    banner_html = ""
    if degraded_rows:
        bits = []
        for r in degraded_rows:
            missing = r["coverage_total"] - r["coverage_present"]
            bits.append(
                f"{_esc(r['name'])} contributed "
                f"{r['coverage_present']} of {r['coverage_total']} claims "
                f"({missing} unavailable)"
            )
        banner_html = (
            '<div class="run-manifest-banner" role="status" '
            'aria-live="polite">'
            '<span class="run-manifest-banner-icon" aria-hidden="true">!</span>'
            '<span class="run-manifest-banner-text"><strong>Degraded consensus.</strong> '
            + ' · '.join(bits) + '.</span>'
            '</div>'
        )

    body_rows = []
    for r in rows:
        model_label = _pretty_model_label(r["name"], r["model_id"]) if r["model_id"] else "—"
        if r["extra_models"] > 0:
            model_label += f" <span class=\"run-manifest-extra\">+{r['extra_models']} more</span>"
        cov_pct_int = int(round(r["coverage_pct"] * 100))
        cov_text = (
            f'{r["coverage_present"]}/{r["coverage_total"]} '
            f'<span class="run-manifest-pct">({cov_pct_int}%)</span>'
        )
        if r["degraded"]:
            cov_text = f'<strong>{cov_text}</strong>'
        if r.get("split_contributed"):
            _n_split = r["split_contributed"]
            cov_text += (
                f' <span class="run-manifest-pct">· {_n_split} split '
                f'(panel voted, no consensus)</span>'
            )
        if r["grounding_pct"] is None:
            grounding_text = '<span class="run-manifest-pct">—</span>'
        else:
            g_int = int(round(r["grounding_pct"] * 100))
            grounding_text = (
                f'{r["web_total"]}/{r["mrs_total"]} '
                f'<span class="run-manifest-pct">({g_int}%)</span>'
            )
        row_class = ' class="degraded"' if r["degraded"] else ''
        body_rows.append(
            f'<tr{row_class}>'
            f'<td>{_esc(_adapter_pretty(r["name"]))}</td>'
            f'<td>{model_label}</td>'
            f'<td>{_esc(r["modes_str"])}</td>'
            f'<td>{cov_text}</td>'
            f'<td>{grounding_text}</td>'
            '</tr>'
        )

    strength_dist = _consensus_strength_distribution(site_report)
    strength_pretty = {
        "strong": "strong",
        "weak": "weak",
        "single": "single-vote",
        "none": "no-consensus",
    }
    strength_parts = [
        f'{count} {strength_pretty.get(k, k)}'
        for k, count in sorted(strength_dist.items(), key=lambda kv: -kv[1])
        if count > 0
    ]
    strength_html = ""
    if strength_parts:
        strength_html = (
            '<p class="run-manifest-meta">'
            '<span class="run-manifest-meta-label">Consensus strength:</span> '
            + ' · '.join(strength_parts)
            + '</p>'
        )

    # PCA runs (panel_roster present) headline the DISTINCT seat models —
    # counting the bridge's single reconciled adapter row as "1 model"
    # under-reported a 3-model panel (1.7; same fix class as
    # ``_models_engaged``). Legacy multi-adapter runs keep the row count.
    seats = (getattr(site_report, "panel_roster", None) or {}).get("seats") or {}
    seat_models = {m for ms in seats.values() for m in (ms or []) if m}
    if seat_models:
        n_seat = len(seat_models)
        summary_text = (
            f'Run manifest · {n_seat} seat model{"s" if n_seat != 1 else ""} '
            f'· {total_claims} claim{"s" if total_claims != 1 else ""}'
        )
        adapter_th = "Panel"
    else:
        summary_text = (
            f'Run manifest · {len(rows)} model{"s" if len(rows) != 1 else ""} '
            f'· {total_claims} claim{"s" if total_claims != 1 else ""}'
        )
        adapter_th = "Adapter"
    if degraded_rows:
        summary_text += f' · {len(degraded_rows)} degraded'

    return (
        '<aside class="run-manifest">'
        + banner_html
        + '<details class="run-manifest-details"'
        + (' open' if degraded_rows else '')
        + '>'
        + f'<summary class="run-manifest-summary">{_esc(summary_text)}</summary>'
        + '<div class="run-manifest-body">'
        + '<table class="run-manifest-table">'
        + '<thead><tr>'
        + f'<th>{adapter_th}</th>'
        + '<th>Model</th>'
        + '<th>Mode</th>'
        + '<th>Coverage</th>'
        + '<th>Tool-URL grounding</th>'
        + '</tr></thead>'
        + '<tbody>' + ''.join(body_rows) + '</tbody>'
        + '</table>'
        + strength_html
        + '<p class="run-manifest-caveat">'
        + '<strong>Tool-URL grounding</strong> is the share of model-emitted '
        + 'citation URLs that intersected the search tool’s retrieved-URL '
        + 'set for the same call. Lower numbers can reflect harness-capture '
        + 'asymmetry in the multi-claim batch path as much as model-citation '
        + 'discipline; URLs that didn’t intersect appear per-claim with '
        + 'a “didn’t validate” caveat.'
        + '</p>'
        + '</div>'
        + '</details>'
        + '</aside>'
    )


# Seat display order + reader-facing labels for the PCA panel composition.
# The panel is proposer → critic → arbiter; unknown seats fall through in
# roster order after these three.
_PCA_SEAT_ORDER = ("proposer", "critic", "arbiter")
_PCA_SEAT_LABELS = {
    "proposer": "Proposer",
    "critic": "Critic",
    "arbiter": "Arbiter",
}


def _panel_composition_html(site_report) -> str:
    """Render the "PCA panel composition" block once per report.

    A per-RUN provenance fact: which model fills each PCA seat
    (proposer/critic/arbiter). Reads ``site_report.panel_roster`` —
    ``{"name": <roster>, "seats": {seat: [alias, …]}}``. Renders nothing when
    the roster is absent or carries no seats (legacy-clean).

    Note: this surfaces roster COMPOSITION only. Per-seat vote attribution
    ("model X voted False") is discarded at panel collapse and is not shown here.
    """
    roster = getattr(site_report, "panel_roster", None) or {}
    seats = roster.get("seats") or {}
    # Only seats with at least one concrete alias are worth rendering.
    filled = {s: [a for a in (aliases or []) if a] for s, aliases in seats.items()}
    filled = {s: a for s, a in filled.items() if a}
    if not filled:
        return ""

    # Ordered seats: proposer/critic/arbiter first, then any extras in roster order.
    ordered = [s for s in _PCA_SEAT_ORDER if s in filled]
    ordered += [s for s in filled if s not in _PCA_SEAT_ORDER]

    seat_rows = []
    for seat in ordered:
        label = _PCA_SEAT_LABELS.get(seat, seat.replace("_", " ").title())
        models_str = ", ".join(_esc(a) for a in filled[seat])
        seat_rows.append(
            '<li class="panel-composition-seat">'
            f'<span class="panel-composition-role">{_esc(label)}</span>'
            f'<span class="panel-composition-model">{models_str}</span>'
            '</li>'
        )

    name = roster.get("name") or ""
    name_html = (
        f'<span class="panel-composition-roster">roster: {_esc(name)}</span>'
        if name else ""
    )

    return (
        '<aside class="panel-composition">'
        '<div class="panel-composition-head">'
        '<span class="panel-composition-title">PCA panel composition</span>'
        + name_html
        + '</div>'
        + '<ul class="panel-composition-list">'
        + ''.join(seat_rows)
        + '</ul>'
        '</aside>'
    )


def _models_engaged(site_report) -> tuple[int, str]:
    """Distinct models that actually touched verdicts for this report.

    PCA reports (panel_roster present): the distinct models across the
    proposer/critic/arbiter seats, plus the Severity Classifier when any claim
    carries a stage-2 override. Counting adapter names under-reports a 3-model
    panel as 1 — the bridge emits ONE reconciled ModelVerdict per claim
    (2026-07-19 review find). Display-only: consensus always comes from the
    panel votes, never from this counter. Legacy reports keep the
    adapter-name count. Returns (count, composition hint)."""
    roster = getattr(site_report, "panel_roster", None) or {}
    seats = roster.get("seats") or {}
    if seats:
        models = {m for ms in seats.values() for m in (ms or [])}
        crm_engaged = any(
            getattr(getattr(b.consensus, "provenance", None), "crm114_final", "")
            for b in site_report.checkable_bundles
        )
        if crm_engaged:
            return len(models) + 1, (
                f"{len(models)} panel seat models "
                f"({', '.join(sorted(models))}) + the Severity Classifier")
        return len(models), f"{len(models)} panel seat models ({', '.join(sorted(models))})"
    n = len({mv.adapter_name for b in site_report.checkable_bundles for mv in b.model_verdicts})
    return n, f"{n} model adapter{'s' if n != 1 else ''}"


def _status_bar(model_count: int = 0, stamp: Optional[str] = None) -> str:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model_str = f"{model_count} Model{'s' if model_count != 1 else ''}" if model_count else "Multi-model"
    # Editorial-lens chip toggles the headline-pill projection between
    # Strict (default since 2026-04-30) and Lenient. The chip is hidden
    # by default and the toggle JS reveals it on pages that have any
    # claim pills to flip.
    lens_chip = (
        '    <button type="button" class="editorial-lens" data-lens="strict" hidden '
        'title="Toggle the report verdict between the Strict lens (graded: Largely/Mostly '
        'True/False over decided claims, Mixed for coin-flips) and the Lenient lens '
        '(simple overall lean: Truthy or Falsey). Same claims, same counts — two '
        'presentations.">\n'
        '      <span class="lens-label">Lens:</span>\n'
        '      <span class="lens-value">Strict</span>\n'
        '    </button>\n'
    )
    return (
        '<div class="status-bar">\n'
        '  <div class="row">\n'
        '    <span class="live">Operational</span>\n'
        f'    <span>Pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}</span>\n'
        f'    <span>{model_str}</span>\n'
        + lens_chip +
        f'    <span class="stamp">{_esc(stamp)}</span>\n'
        '  </div>\n'
        '</div>\n'
    )


def _masthead_full(rel: str = "./") -> str:
    return (
        '<header class="masthead">\n'
        '  <div class="wrap masthead-row">\n'
        '    <div>\n'
        f'      <div class="wordmark"><a href="{rel}index.html" style="color:inherit;text-decoration:none">'
        'truth-bot<span class="dot">.</span></a></div>\n'
        '      <p class="tagline">Automated political fact-checking with multi-model consensus analysis.</p>\n'
        '    </div>\n'
        '    <nav class="top-nav">\n'
        f'      <a href="{rel}index.html">Reports</a>\n'
        f'      <a href="{rel}about.html">About</a>\n'
        f'      <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub ↗</a>\n'
        '    </nav>\n'
        '  </div>\n'
        '</header>\n'
    )


def _masthead_compact(rel: str = "../") -> str:
    return (
        '<header class="masthead">\n'
        '  <div class="wrap mast-row">\n'
        f'    <a href="{rel}index.html" class="wordmark-sm">truth-bot<span class="dot">.</span></a>\n'
        '    <nav class="breadcrumb">\n'
        f'      <a href="{rel}index.html"><span class="arrow">←</span>All reports</a>\n'
        '    </nav>\n'
        '  </div>\n'
        '</header>\n'
    )


# Default OG/Twitter description used when a page doesn't provide one.
# "primary sources" RESTORED 2026-07-25 (remediation T3.3): the Phase 3
# artifacts verified with zero fact-check items in any pack (T2.1 exclusion
# holds end-to-end) and a green strict era lint — the claim is now true of
# what actually ships. It was removed 2026-07-21 (T0.5, D4) while packs
# still mixed in fact-check domains.
_DEFAULT_OG_DESCRIPTION = (
    "Automated fact-checking of political speeches. Every claim is checked "
    "by a multi-model AI panel against a shared pack of cited primary "
    "sources — sources linked inline, disagreements disclosed."
)


def _social_head(
    rel: str,
    og_title: str,
    og_description: str,
    og_type: str = "website",
    og_image_alt: str = "truth-bot: automated political fact-checking with multi-model consensus",
    include_feed_link: bool = False,
) -> str:
    """Emit favicon links, Open Graph meta, Twitter Card meta, and optional feed link.

    `rel` is the path prefix to reach the site root from the page
    (`./` for root pages, `../` for pages in `reports/` or `claims/`).
    """
    twitter_desc = og_description
    parts = [
        f'  <link rel="icon" href="{rel}favicon.ico" sizes="any">\n',
        f'  <link rel="icon" href="{rel}assets/favicon-32.png" type="image/png" sizes="32x32">\n',
        f'  <link rel="apple-touch-icon" href="{rel}assets/apple-touch-icon.png">\n',
        f'  <meta property="og:type" content="{_esc(og_type)}">\n',
        '  <meta property="og:site_name" content="truth-bot">\n',
        f'  <meta property="og:title" content="{_esc(og_title)}">\n',
        f'  <meta property="og:description" content="{_esc(og_description)}">\n',
        f'  <meta property="og:image" content="{rel}assets/social-card.png">\n',
        '  <meta property="og:image:width" content="1200">\n',
        '  <meta property="og:image:height" content="630">\n',
        f'  <meta property="og:image:alt" content="{_esc(og_image_alt)}">\n',
        '  <meta name="twitter:card" content="summary_large_image">\n',
        f'  <meta name="twitter:title" content="{_esc(og_title)}">\n',
        f'  <meta name="twitter:description" content="{_esc(twitter_desc)}">\n',
        f'  <meta name="twitter:image" content="{rel}assets/social-card.png">\n',
        f'  <meta name="twitter:image:alt" content="{_esc(og_image_alt)}">\n',
    ]
    if include_feed_link:
        parts.append(
            f'  <link rel="alternate" type="application/atom+xml" '
            f'title="truth-bot feed" href="{rel}feed.xml">\n'
        )
    return ''.join(parts)


def _page_index(
    title: str,
    body: str,
    footer: str = "",
    model_count: int = 0,
    og_title: str = "truth-bot — Automated Political Fact-Checking",
    og_description: str = _DEFAULT_OG_DESCRIPTION,
    og_type: str = "website",
) -> str:
    foot_html = (
        '<footer class="foot wrap">\n' + footer + '\n</footer>\n'
        if footer else ''
    )
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}{BETA_TEXT_SUFFIX}">\n'
        # Tint mobile browser chrome to match the page background.
        # Keep in sync with --bg in CSS.
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        + _social_head("./", og_title, og_description, og_type=og_type, include_feed_link=True)
        + f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
        + _GOOGLE_FONTS + '\n'
        '  <link rel="stylesheet" href="./assets/styles.css">\n'
        '</head>\n'
        '<body>\n'
        + _status_bar(model_count)
        + _masthead_full(rel="./")
        + '<main class="wrap">\n'
        + body
        + '\n</main>\n'
        + foot_html
        + '<script src="./assets/truthbot.js"></script>\n'
        '</body>\n'
        '</html>'
    )


def _page_report(
    title: str,
    body: str,
    footer: str = "",
    model_count: int = 0,
    analyzed_at: Optional[str] = None,
    og_title: Optional[str] = None,
    og_description: str = _DEFAULT_OG_DESCRIPTION,
    og_type: str = "article",
) -> str:
    stamp = f"Analyzed {analyzed_at}" if analyzed_at else "Analyzed " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    foot_html = (
        '<footer class="foot wrap">\n' + footer + '\n</footer>\n'
        if footer else ''
    )
    _og_title = og_title or f"{title.removesuffix(' — truth-bot')} — truth-bot"
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}{BETA_TEXT_SUFFIX}">\n'
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        + _social_head("../", _og_title, og_description, og_type=og_type)
        + f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
        + _GOOGLE_FONTS + '\n'
        '  <link rel="stylesheet" href="../assets/styles.css">\n'
        '</head>\n'
        '<body>\n'
        + _status_bar(model_count, stamp)
        + _masthead_compact(rel="../")
        + '<main class="wrap">\n'
        + body
        + '\n</main>\n'
        + foot_html
        + '<script src="../assets/truthbot.js"></script>\n'
        '</body>\n'
        '</html>'
    )


def _page_about(
    title: str,
    body: str,
    footer: str = "",
    og_title: Optional[str] = None,
    og_description: str = _DEFAULT_OG_DESCRIPTION,
    og_type: str = "website",
) -> str:
    foot_html = (
        '<footer class="foot wrap">\n' + footer + '\n</footer>\n'
        if footer else ''
    )
    _og_title = og_title or f"{title.removesuffix(' — truth-bot')} — truth-bot"
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}{BETA_TEXT_SUFFIX}">\n'
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        + _social_head("./", _og_title, og_description, og_type=og_type)
        + f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
        + _GOOGLE_FONTS + '\n'
        '  <link rel="stylesheet" href="./assets/styles.css">\n'
        '</head>\n'
        '<body>\n'
        + _status_bar()
        + _masthead_full(rel="./")
        + '<main class="wrap">\n'
        + body
        + '\n</main>\n'
        + foot_html
        + '<script src="./assets/truthbot.js"></script>\n'
        '</body>\n'
        '</html>'
    )


def _page_truthy(
    title: str,
    body: str,
    footer: str = "",
    og_title: Optional[str] = None,
    og_description: str = _DEFAULT_OG_DESCRIPTION,
) -> str:
    """Fun / mascot page shell — same chrome as about, no truthbot.js (inline _TRUTHY_FUN_SCRIPT in body)."""
    foot_html = (
        '<footer class="foot wrap">\n' + footer + '\n</footer>\n'
        if footer else ''
    )
    _og_title = og_title or f"{title.removesuffix(' — truth-bot')} — truth-bot"
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}{BETA_TEXT_SUFFIX}">\n'
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        + _social_head("./", _og_title, og_description, og_type="website")
        + f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
        + _GOOGLE_FONTS + '\n'
        '  <link rel="stylesheet" href="./assets/styles.css">\n'
        '</head>\n'
        '<body>\n'
        + _status_bar()
        + _masthead_full(rel="./")
        + '<main class="wrap">\n'
        + body
        + '\n</main>\n'
        + foot_html
        + '</body>\n'
        '</html>'
    )


# ── Claim + report building blocks ───────────────────────────────────────────


def _is_pca_bundle(bundle: VerdictBundle) -> bool:
    """A PCA (single reconciled-judge) bundle: at most one card AND a recorded
    panel vote tally. Legacy multi-adapter bundles have >1 card and empty
    provenance, so they take the classic per-model-agreement path unchanged."""
    prov = getattr(bundle.consensus, "provenance", None)
    return bool(prov and prov.panel_votes) and len(bundle.model_verdicts) <= 1


def _pca_vote_tally(votes: dict) -> str:
    """``{'True': 2, 'Misleading': 1}`` → ``'True ×2, Misleading ×1'`` (count desc)."""
    items = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{lbl} ×{n}" for lbl, n in items)


def _pca_agreement_summary(bundle: VerdictBundle) -> str:
    """Reconciled-judge replacement for the "N of M agree" tally.

    Speaks panel-vote vocabulary honestly instead of the vacuous "1 of 1 agree":
    a resolved claim reports how many seats backed the winning label; a split
    claim (which renders zero cards) reports the tie instead of a blank strip."""
    prov = bundle.consensus.provenance
    votes = prov.panel_votes
    total = sum(votes.values())
    if prov.panel_split or bundle.consensus.consensus_verdict == "Models split":
        return f'Panel split &mdash; {_esc(_pca_vote_tally(votes))}'
    top = max(votes.values()) if votes else 0
    seats = "seat" if total == 1 else "seats"
    return f'<span class="num">{top} of {total}</span> {seats} agree'


_SEAT_ORDER = ("proposer", "critic", "arbiter")


def _pca_seat_line(prov, roster: Optional[dict] = None) -> str:
    """Per-seat predictions: 'proposer (mistral): Misleading · critic (dsv4-flash):
    False · …'. Uses provenance.panel_by_role (captured since 2026-07-19); seat →
    model names come from the report-level roster when available. Empty string for
    older bundles with no by_role — the tally line still renders."""
    by_role = getattr(prov, "panel_by_role", None) or {}
    if not by_role:
        return ""
    seats = dict((roster or {}).get("seats") or {})
    ordered = [r for r in _SEAT_ORDER if r in by_role] + sorted(
        r for r in by_role if r not in _SEAT_ORDER)
    bits = []
    for role in ordered:
        labels = "/".join(by_role[role])
        models = ", ".join(seats.get(role) or [])
        who = f"{role} ({models})" if models else role
        bits.append(f"{who}: {labels}")
    return " · ".join(bits)


#: Pill text + reader-facing copy for the guest-anecdote treatment. A private
#: person's story that comes back Unverifiable is a GENRE limit (no independent
#: public record exists to check), not a failed verification — so it gets its
#: own pill instead of the same Unverifiable used for data claims the evidence
#: couldn't settle (2026-07-20, jackie review).
ANECDOTE_PILL = "Anecdote"
ANECDOTE_TITLE = (
    "Guest anecdote — a private individual's personal story. No independent "
    "public record exists to check it against, so truth-bot does not rate it. "
    "This is a limit of the genre, not a failed verification."
)


def _is_anecdote_unverifiable(bundle: VerdictBundle) -> bool:
    """True when this claim is a personal-anecdote that came back Unverifiable —
    the case that renders with the Anecdote pill. An anecdote the evidence CAN
    settle (e.g. press independently investigated it) keeps its real verdict."""
    prov = bundle.consensus.provenance
    return (prov.layer_a_claim_type == "personal-anecdote"
            and bundle.consensus.consensus_label == VerdictLabel.UNVERIFIABLE)


#: Honest-abstention sub-state (PR-A2.1 / Evidential Role Axis Phase 1).
#: "The only witness is the claimant" is a FINDING, not an evasion: the quality
#: gate failed, the pack's only bearing items are the speaker's own
#: organization, and no independent S1–S3 item bears on the core assertion —
#: so the reader is told exactly that instead of a bare "Unverifiable".
#: Display only; verdicts, gates and weights are untouched in this phase.
SELF_SOURCED_PILL = "Unverified — self-sourced only"
SELF_SOURCED_TITLE = (
    "The evidence gate failed and every source bearing on this claim is the "
    "speaker's own organization (administration, party, or campaign at the "
    "time of the speech). Self-records can confirm a claim was made — never, "
    "on their own, that it is true. No independent top-tier source was found."
)
#: Mirrors ``verdict.consolidator.GATE_INSUFFICIENT`` — site.py is string-typed
#: by design (like the coarse projections); the pin lives in the render tests.
GATE_INSUFFICIENT = "insufficient-qualifying-evidence"
_INDEPENDENT_TIERS = ("Government", "Wire", "Established")  # S1–S3 tier values


def _self_source_ids(bundle: VerdictBundle) -> set[str]:
    """Pack ids of consulted sources whose org is the speaker's own principal
    (era-scoped SELF relation), for the per-item badge on the source strip."""
    return {
        str(s.get("id") or "").strip()
        for s in (getattr(bundle, "sources_consulted", None) or [])
        if principal_relation(s.get("url", ""), bundle.speaker, bundle.date_str)
        is PrincipalRelation.SELF
    } - {""}


def _is_self_sourced_unverified(bundle: VerdictBundle) -> bool:
    """True when the claim renders "Unverified — self-sourced only" (T1.1):
    gate-forced Unverifiable AND ≥1 bearing (on-core) SELF item AND zero
    independent bearing S1–S3 items. Anecdotes keep their genre pill — the
    genre limit is the more specific finding."""
    if _is_anecdote_unverifiable(bundle):
        return False
    consensus = bundle.consensus
    if (consensus.consensus_label != VerdictLabel.UNVERIFIABLE
            or consensus.consensus_verdict == "Models split"
            or getattr(consensus.provenance, "evidence_gate", "") != GATE_INSUFFICIENT):
        return False
    has_bearing_self = False
    for src in getattr(bundle, "sources_consulted", None) or []:
        if src.get("supports_claim") is None:  # not bearing on the core assertion
            continue
        rel = principal_relation(src.get("url", ""), bundle.speaker, bundle.date_str)
        if rel is PrincipalRelation.SELF:
            has_bearing_self = True
        elif (src.get("tier") or "") in _INDEPENDENT_TIERS:
            return False  # an independent S1–S3 item bears — not self-sourced-only
    return has_bearing_self


def _pca_provenance_strip(bundle: VerdictBundle, roster: Optional[dict] = None,
                          rel: str = "../") -> str:
    """The Layer A → PCA panel → CRM-114 chain, rendered as a compact strip.

    Surfaces provenance that used to live only as buried reasoning text (CRM-114)
    or nowhere at all (Layer A label, per-seat tally). When per-seat labels were
    captured (panel_by_role), a second line names what each seat predicted."""
    prov = bundle.consensus.provenance
    parts: list[str] = []
    if prov.layer_a_label:
        qualifiers = ", ".join(
            q for q in (prov.layer_a_source, prov.layer_a_claim_type,
                        getattr(prov, "layer_a_claim_shape", "")) if q)
        src = f" ({qualifiers})" if qualifiers else ""
        parts.append(f"Layer A: {prov.layer_a_label}{src}")
    if prov.panel_votes:
        parts.append(f"PCA panel: {_pca_vote_tally(prov.panel_votes)}")
    if prov.crm114_final:
        stage1 = prov.crm114_stage1 or "?"
        parts.append(f"Severity Classifier: {stage1}→{prov.crm114_final}")
    if not parts:
        return ""
    chain = _esc(" → ".join(parts))
    seat_line = _pca_seat_line(prov, roster)
    seat_html = (
        f'<div class="pca-seats">{_esc(seat_line)}</div>' if seat_line else ""
    )
    # Post-publication correction (T1.5): shown wherever the verdict is,
    # linked to the public changelog — a correction is never silent.
    corr_html = ""
    if getattr(prov, "correction_note", ""):
        corr_html = (
            f'<div class="pca-correction">⚠ {_esc(prov.correction_note)} '
            f'· <a href="{rel}corrections.html">Corrections</a></div>'
        )
    return (
        '<div class="pca-provenance" '
        'title="Pipeline provenance: check-worthiness routing, the PCA panel seat '
        'tally, each seat&#39;s own prediction, and any Severity Classifier '
        'stage-2 override.">'
        f'{chain}{seat_html}{corr_html}</div>'
    )


def _claim_card(bundle: VerdictBundle, idx: int, total: int, rel: str = "../",
                standalone: bool = False, panel_roster: Optional[dict] = None) -> str:
    claim = bundle.claim
    consensus = bundle.consensus
    # Folding through aggregation (1.6): a split / no-verdict claim reads its
    # verdict text on EVERY axis — including data-fine-label, which used to
    # echo "Unverifiable" for split rows (audit V6).
    fine_label = _agg_fine_label(consensus.consensus_verdict,
                                 consensus.consensus_label.value)
    # Headline defaults to the Strict 5-bucket projection (2026-04-30
    # editorial flip from Lenient). Older cached bundles (pre-projection-
    # layer) carry blank coarse fields; on this surface they ECHO the fine
    # label (passed as the stored value below) so toggling is a visual no-op
    # rather than a broken render — while split rows still pass through
    # coarse_label's non-folding rule on every axis.
    _stored_lenient = (consensus.coarse_lenient_label or "").strip()
    _stored_strict = (consensus.coarse_strict_label or "").strip()
    lenient_attr = _agg_coarse_label(
        fine_label, _stored_lenient or fine_label, "lenient")
    strict_attr = _agg_coarse_label(
        fine_label, _stored_strict or fine_label, "strict")
    label = strict_attr
    css = _verdict_css(label)
    fine_css = _verdict_css(fine_label)
    lenient_css = _verdict_css(lenient_attr)
    strict_css = _verdict_css(strict_attr)
    # Guest-anecdote treatment: swap the pill TEXT on both lens axes (so the
    # Lenient/Strict toggle can't restore "Unverifiable") but keep the
    # Unverifiable color family; the dashed pill border marks the genre.
    anecdote = _is_anecdote_unverifiable(bundle)
    pill_title = ("Headline shows the 5-bucket coarse projection. Per-model strip below uses "
                  "the 6-bucket fine scale. Use the Editorial lens chip to toggle Lenient/Strict.")
    anecdote_cls = ""
    if anecdote:
        label = lenient_attr = strict_attr = ANECDOTE_PILL
        pill_title = ANECDOTE_TITLE
        anecdote_cls = " pill-anecdote"
    # Self-sourced-only treatment (PR-A2.1): same both-axes text swap as the
    # anecdote pill so the lens toggle can't restore a bare "Unverifiable".
    elif _is_self_sourced_unverified(bundle):
        label = lenient_attr = strict_attr = SELF_SOURCED_PILL
        pill_title = SELF_SOURCED_TITLE
        anecdote_cls = " pill-self-sourced"
    n = str(idx).zfill(2)

    context_html = ''
    if claim.category:
        context_html = f'<div class="claim-context"><span>{_esc(claim.category)}</span></div>'

    caveat_html = _render_caveat_block(bundle.model_verdicts)

    # Dissent is computed on the *family* axis (2026-05-01 follow-up to
    # findings-review C4): {True, Mostly True} share a family, as do
    # {Misleading, False}; Exaggerated and Unverifiable each stand
    # alone. A True voter against a Mostly True consensus is no longer
    # flagged — directional agreement is honored. The per-model strip
    # still RENDERS the fine-axis label; only the dissent CSS class
    # (and the "N of M agree" tally) reads on the family axis. See
    # ``_verdict_family`` for the full mapping + rationale.
    majority_label = fine_label
    majority_family = _verdict_family(majority_label)

    triage_badge = ""
    if getattr(bundle, "triage_skipped_frontier", False):
        triage_badge = (
            '<span class="claim-pill triage-only" '
            'title="Unanimous high-confidence triage; frontier models were skipped">Triage</span>'
        )

    # E-id anchor plumbing (2026-07-19 review follow-up): pack ids mentioned in
    # reasoning ("E5 confirms…") become links to the matching "Sources consulted"
    # item. Only ids actually present in the retrieved pack are linkified.
    _anchor_base = "ev-" + re.sub(r"[^A-Za-z0-9_-]", "-", str(bundle.claim.id))
    _pack_ids = {str(s.get("id") or "").strip()
                 for s in (getattr(bundle, "sources_consulted", None) or [])}
    _pack_ids.discard("")

    def _link_pack_ids(escaped: str) -> str:
        if not _pack_ids:
            return escaped
        return re.sub(
            r"\bE\d{1,3}\b",
            lambda m: (f'<a class="ev-ref" href="#{_anchor_base}-{m.group(0)}">{m.group(0)}</a>'
                       if m.group(0) in _pack_ids else m.group(0)),
            escaped)

    def _reasoning_paragraphs(text: str) -> str:
        if not text:
            return ""
        # Display-time only: the bridge annotates overrides in the explanation as
        # "CRM-114: ..." for internal audit; readers see "Severity Classifier".
        # (Stored explanation and internal identifiers are left untouched.)
        text = text.replace("CRM-114", "Severity Classifier")
        parts = [seg.strip() for seg in re.split(r"\n\s*\n", text.strip()) if seg.strip()]
        if not parts:
            return ""
        return "".join(f'<p>{_link_pack_ids(_esc(seg))}</p>' for seg in parts)

    model_cards = []
    agreeing = 0
    all_urls: list[str] = []
    seen_urls: set[str] = set()
    combined_classifications: dict[str, str] = {}
    for mv in bundle.model_verdicts:
        mv_label = mv.label.value
        mv_css = _verdict_css(mv_label)
        if getattr(mv, 'no_response', False):
            # Model failed to respond
            model_cards.append(
                '<div class="model no-response">'
                f'  <div class="model-name">{_esc(mv.adapter_name)}</div>'
                '  <div class="model-verdict" style="color:var(--ink-faint)">Requested / Failed</div>'
                '</div>'
            )
        else:
            dissent = (
                " dissent"
                if _verdict_family(mv_label) != majority_family
                else ""
            )
            if not dissent:
                agreeing += 1
            reasoning_html = ""
            reasoning_text = _reasoning_paragraphs(getattr(mv, 'explanation', '').strip())
            if reasoning_text:
                mid_label = _pretty_model_label(mv.adapter_name, getattr(mv, "model_id", ""))
                reasoning_html = (
                    '<details class="model-reasoning">'
                    f'  <summary>Model reasoning<span class="model-reasoning-model"> — {_esc(mid_label)}</span></summary>'
                    f'  <div class="model-reasoning-body">{reasoning_text}</div>'
                    '</details>'
                )
            # Per-model tier/mode chip removed (2026-04-29). Editorial intent
            # has always been frontier for all final outcomes; the only legit
            # non-frontier exception is the bundle-level "Triage" pill above
            # (rendered when ``triage_skipped_frontier=True``). Surfacing
            # ``mv.tier`` / ``mv.synthesis_mode`` per-model just made it look
            # like Anthropic/OpenAI batch verdicts were "less than frontier"
            # next to live Grok/Gemini verdicts that didn't render a chip,
            # which is the opposite of the truth. Engine still records
            # tier/mode on each ModelVerdict for telemetry; consumers that
            # care can read claims.json or the bundle cache directly.
            model_cards.append(
                f'<div class="model{dissent}">'
                f'  <div class="model-name">{_esc(mv.adapter_name)}</div>'
                f'  <div class="model-verdict vt-{mv_css}">{VERDICT_EMOJI.get(mv_label, "")} {_esc(mv_label)}</div>'
                f'  {reasoning_html}'
                '</div>'
            )
        for url in mv.web_sources:
            if url not in seen_urls:
                seen_urls.add(url)
                all_urls.append(url)
        # Layer 4 — merge per-verdict URL classifications so the
        # combined evidence list can render the three trust tiers. If
        # multiple verdicts disagree on a URL's classification (rare),
        # the *worse* verdict wins so the reader is never told a URL is
        # verified when one model's run found it broken.
        for url, cls in (mv.url_classifications or {}).items():
            existing = combined_classifications.get(url)
            combined_classifications[url] = _worse_classification(
                existing, cls
            )

    # Second pass — collect model-reported URLs that did NOT survive the
    # tool-grounding intersection (apply_url_grounding) for ANY model. A
    # URL validated by even one model is treated as validated and stays
    # out of this list. Surfaced separately under the evidence block
    # with a "didn't validate" caveat so readers see the audit trail
    # without us implying we vouched for them.
    unverified_urls: list[str] = []
    seen_unverified: set[str] = set()
    for mv in bundle.model_verdicts:
        for url in (getattr(mv, 'model_reported_sources', None) or []):
            if not url or url in seen_urls or url in seen_unverified:
                continue
            seen_unverified.add(url)
            unverified_urls.append(url)

    total_models = len(bundle.model_verdicts)
    dissenting = total_models - agreeing
    dissent_note = f" · {dissenting} dissent{'s' if dissenting > 1 else ''}" if dissenting else ""

    # PCA (reconciled-judge) bundles speak panel-vote vocabulary + a provenance
    # strip instead of the legacy "N of M agree" per-adapter tally; a split claim
    # (zero cards) shows its tie rather than a blank strip. Legacy multi-adapter
    # bundles are untouched.
    if _is_pca_bundle(bundle):
        grid_html = (
            f'<div class="model-grid">{"".join(model_cards)}</div>'
            if model_cards
            else '<div class="model-grid model-grid-empty">'
                 '<div class="model no-response">'
                 '<div class="model-verdict" style="color:var(--ink-faint)">'
                 'No single verdict — panel did not converge</div></div></div>'
        )
        models_block = (
            '<div class="models">'
            '  <div class="models-head">'
            '    <span class="models-label">Reconciled judgment</span>'
            f'    <span class="models-agreement">{_pca_agreement_summary(bundle)}</span>'
            '  </div>'
            f'  {_pca_provenance_strip(bundle, panel_roster)}'
            f'  {grid_html}'
            '</div>'
        )
    else:
        models_block = (
            '<div class="models">'
            '  <div class="models-head">'
            '    <span class="models-label">Model consensus</span>'
            f'    <span class="models-agreement"><span class="num">{agreeing} of {total_models}</span> agree{_esc(dissent_note)}</span>'
            '  </div>'
            f'  <div class="model-grid">{"".join(model_cards)}</div>'
            '</div>'
        )

    # Full retrieved pack (ALL items, not just cited) — rides on the bundle so
    # a claim with a non-empty pack but zero citations still shows its real
    # sources rather than a bare "No sources retrieved."
    consulted = list(getattr(bundle, "sources_consulted", None) or [])
    consulted_html = ""
    if consulted:
        consulted_inner = _sources_consulted_html(
            consulted, anchor_base=_anchor_base,
            self_ids=_self_source_ids(bundle))
        if consulted_inner:
            # Collapsed by default (2026-07-19 review): the snippet verbiage is
            # audit detail, not first-read content — one click away, not in the way.
            consulted_html = (
                '<details class="evidence-details">'
                f'  <summary class="evidence-summary">Sources consulted ({len(consulted)})</summary>'
                f'  <div class="evidence">{consulted_inner}</div>'
                '</details>'
            )

    unverified_block = _model_cited_unverified_html(unverified_urls)
    if all_urls:
        evidence_inner = (
            f'{_evidence_list_html(all_urls[:10], classifications=combined_classifications or None)}'
            f'{unverified_block}'
        )
    elif unverified_urls:
        # Suppress the "No sources retrieved." note when the model DID
        # cite URLs but none survived intersection — the unverified
        # block is the audit trail in that case.
        evidence_inner = unverified_block
    elif consulted_html:
        # Pack was retrieved but nothing was cited (e.g. Unverifiable). Point
        # the reader at the "Sources consulted" section instead of asserting
        # "No sources retrieved." as if the search came back empty.
        evidence_inner = (
            '<p style="font-size:0.88rem;color:var(--ink-muted)">'
            'No sources were cited for this verdict — see '
            '<em>Sources consulted</em> below for the full retrieved pack.</p>'
        )
    else:
        evidence_inner = _evidence_list_html([], classifications=None)
    evidence_html = (
        '<details class="evidence-details">'
        '  <summary class="evidence-summary">Combined evidence / sources list</summary>'
        f'  <div class="evidence">{evidence_inner}</div>'
        '</details>'
    )

    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    permalink = f"#{'claim-' + str(idx)}" if standalone else f"{rel}claims/{claim.id}.html"

    # Back links only appear on the in-report claim cards (standalone=False),
    # since standalone claim pages are their own scroll scope and "#claim-catalog"
    # / "#top" anchors don't exist there.
    back_links_html = ''
    if not standalone:
        back_links_html = (
            '    <span class="claim-back-links">'
            '      <a href="#claim-catalog" class="back-link">&uarr; Back to claim list</a>'
            '      <span class="sep">&middot;</span>'
            '      <a href="#top" class="back-link">&uarr; Top of page</a>'
            '    </span>'
        )

    return (
        f'<article class="claim" id="claim-{idx}">'
        '<div class="claim-head">'
        '  <span class="claim-head-lead">'
        + _icon_svg(_ICON_BODY_CLAIMS, size=18, extra_class="claim-head-icon")
        + f'    <span class="claim-num">Claim {n} / {str(total).zfill(2)}</span>'
        '  </span>'
        f'  <span class="claim-pill claim-pill-headline lens-pill v-{css}{anecdote_cls}"'
        f' data-fine-label="{_esc(fine_label)}" data-fine-css="{_esc(fine_css)}"'
        f' data-coarse-lenient="{_esc(lenient_attr)}" data-coarse-lenient-css="{_esc(lenient_css)}"'
        f' data-coarse-strict="{_esc(strict_attr)}" data-coarse-strict-css="{_esc(strict_css)}"'
        f' title="{_esc(pill_title)}">'
        f'{_esc(label)}</span>'
        f'  {triage_badge}'
        '</div>'
        '<div class="claim-body">'
        f'  {_claim_quote_html(claim)}'
        f'  {context_html}'
        f'  {caveat_html}'
        f'  {models_block}'
        f'  {evidence_html}'
        f'  {consulted_html}'
        '  <div class="claim-foot">'
        f'    <a href="#claim-{idx}" class="permalink">claim-{idx}</a>'
        + back_links_html
        + f'    <span>Last verified {gen_ts}</span>'
        '  </div>'
        '</div>'
        '</article>'
    )

def _claim_quote_html(claim) -> str:
    """The claim quote, rendered INSIDE its surrounding transcript sentences.

    Half the Obama-2014 claims (49/96) open with deictic words — "Tonight, I'm
    announcing we'll launch six more this year." — that are unreadable as a
    bare quote (jackie, 2026-08-01: unacceptable). The PCA panel always judged
    with the surrounding sentences (``adjudicator`` feeds ``claim.context``);
    the reader now sees the same thing: neighbors greyed, the checked claim
    emphasized. Bundles without context (legacy) render the bare quote
    exactly as before.

    ``claim.context`` format is the segmenter's ``prev || claim || next``;
    when the claim text isn't a clean element of it, the whole context renders
    below the quote rather than being dropped.
    """
    text = (claim.text or "").strip()
    ctx = (getattr(claim, "context", "") or "").strip()
    bare = f'<blockquote class="claim-quote">"{_esc(text)}"</blockquote>'
    if not ctx or ctx == text:
        return bare
    parts = [p.strip() for p in ctx.split("||") if p.strip()]
    if text not in parts or len(parts) < 2:
        return (bare
                + '<div class="claim-context-fallback">'
                  '<span class="ccq-label">In context</span> '
                + _esc(" … ".join(p for p in parts if p != text)) + '</div>')
    i = parts.index(text)
    before = " ".join(f'<span class="ccq-side">{_esc(p)}</span>'
                      for p in parts[:i])
    after = " ".join(f'<span class="ccq-side">{_esc(p)}</span>'
                     for p in parts[i + 1:])
    mid = f'<span class="ccq-claim">"{_esc(text)}"</span>'
    return (
        '<blockquote class="claim-quote claim-quote-ctx" '
        'title="The checked claim, emphasized, inside the transcript sentences '
        'around it — the same context the verdict panel judged with.">'
        + " ".join(s for s in (before, mid, after) if s)
        + '</blockquote>'
    )


def _toc(bundles: list[VerdictBundle]) -> str:
    items = []
    for i, b in enumerate(bundles, 1):
        consensus = b.consensus
        # Folding through aggregation (1.6): split / no-verdict rows keep
        # their verdict text on every axis (never "Unverifiable"); legacy
        # bundles with blank coarse fields project the fine label on the fly.
        fine_label = _agg_fine_label(consensus.consensus_verdict,
                                     consensus.consensus_label.value)
        coarse_lenient = _agg_coarse_label(
            fine_label, consensus.coarse_lenient_label, "lenient")
        coarse_strict = _agg_coarse_label(
            fine_label, consensus.coarse_strict_label, "strict")
        # Default text is Strict (matches the published default lens
        # since the 2026-04-30 flip).
        default_label = coarse_strict
        default_css   = _verdict_css(default_label)
        fine_css      = _verdict_css(fine_label)
        lenient_css   = _verdict_css(coarse_lenient)
        strict_css    = _verdict_css(coarse_strict)
        toc_anecdote_cls = ""
        if _is_anecdote_unverifiable(b):
            # Mirror the claim card's guest-anecdote pill on both lens axes.
            default_label = coarse_lenient = coarse_strict = ANECDOTE_PILL
            toc_anecdote_cls = " pill-anecdote"
        items.append(
            f'<a class="toc-item" href="#claim-{i}">'
            f'  <span class="toc-num">{str(i).zfill(2)}</span>'
            f'  <span class="toc-pill lens-pill v-{default_css}{toc_anecdote_cls}"'
            f' data-fine-label="{_esc(fine_label)}" data-fine-css="{_esc(fine_css)}"'
            f' data-coarse-lenient="{_esc(coarse_lenient)}" data-coarse-lenient-css="{_esc(lenient_css)}"'
            f' data-coarse-strict="{_esc(coarse_strict)}" data-coarse-strict-css="{_esc(strict_css)}">'
            f'{_esc(default_label)}</span>'
            f'  <span class="toc-text">"{_esc(b.claim.text)}"</span>'
            '  <span class="toc-jump">↓</span>'
            '</a>'
        )
    return '<nav class="toc">' + "".join(items) + '</nav>'


def _report_card(r: dict) -> str:
    claim_count = r.get("claim_count", 0)

    # 5-bucket coarse-axis aggregates for both lenses. Falls back to
    # projecting the legacy 6-bucket distribution if a report predates
    # the projection layer (older reports.json entries).
    fine_dist = r.get("verdict_distribution", {}) or {}
    dist_lenient = (r.get("verdict_distribution_lenient")
                    or _agg_project_dist(fine_dist, "lenient"))
    dist_strict = (r.get("verdict_distribution_strict")
                   or _agg_project_dist(fine_dist, "strict"))

    def _card_axis_html(d: dict[str, int], axis: str = "strict") -> tuple[str, str, str, str, str]:
        """Return (headline_html, ratio_text, segs_html, counts_html, rail_html)
        for one axis. Strict lens = graded family bands; Lenient lens = simple
        Truthy/Falsey. rail_html is the family rail tying the headline's
        leaning totals to the bar."""
        if axis == "lenient":
            headline, cls, ratio_text = _binary_verdict(d)
        else:
            headline, cls, ratio_text = _family_verdict(d)
        # Every bucket renders, Models split included — the card bar must sum
        # to claim_count just like the report-page bar (remediation T0.2).
        total_named = sum(d.values()) or 1
        segs_inner: list[str] = []
        counts_inner: list[str] = []
        for label in AGGREGATE_BAR_ORDER:
            count = d.get(label, 0)
            if not count:
                continue
            segs_inner.append(
                f'<div class="seg v-{_verdict_css(label)}" '
                f'style="width:{count/total_named*100:.1f}%"></div>'
            )
            counts_inner.append(
                f'<div class="ct"><span class="swatch v-{_verdict_css(label)}"></span>'
                f'{_esc(label)} <span class="n">{count}</span></div>'
            )
        head_html = (
            f'<span class="label {cls}">{_esc(headline)}</span>'
            f'<span class="ratio">{_esc(ratio_text)}</span>'
        )
        rail = _family_rail_html(d, AGGREGATE_BAR_ORDER, rail_class="report-family-rail")
        return head_html, ratio_text, "".join(segs_inner), "".join(counts_inner), rail

    head_lenient, _ratio_lenient, segs_lenient, counts_lenient, rail_lenient = _card_axis_html(dist_lenient, axis="lenient")
    head_strict,  _ratio_strict,  segs_strict,  counts_strict,  rail_strict  = _card_axis_html(dist_strict, axis="strict")

    meta_bits = []
    if r.get("date"):
        meta_bits.append(_esc(r["date"]))
    if r.get("venue"):
        meta_bits.append(_esc(r["venue"]))
    meta = '<span class="sep">·</span>'.join(meta_bits)

    tier_counts = r.get("tier_counts") or {}
    # Every nonzero bucket ships, via aggregation.sources_line (1.6):
    # "other" since remediation F6, and "press/political" since remediation
    # v2 — the old hand-kept order omitted the political bucket entirely,
    # hiding 162 sources on the Trump card. The data-tier-counts attribute
    # is the machine-readable mirror consistency.check_site lints against
    # reports.json tier_counts.
    _tier_pairs = _agg_sources_line(tier_counts)
    _tier_parts = [f'{count} {label}' for label, count in _tier_pairs]
    _tier_attr = " ".join(
        f"{key}:{tier_counts.get(key, 0)}"
        for key, _label in TIER_LINE_ORDER if tier_counts.get(key, 0)
    )
    src_tiers_html = (
        f'    <span class="src-tiers" data-tier-counts="{_esc(_tier_attr)}">'
        f'Sources: {" · ".join(_tier_parts)}</span>'
        if _tier_parts else ''
    )

    return (
        f'<a href="{_esc(r.get("url", "#"))}" class="report">'
        '  <div class="report-top">'
        '    <div>'
        f'      <div class="report-headline">{_esc(r.get("speaker", ""))}</div>'
        f'      <div class="report-meta">{meta}</div>'
        '    </div>'
        '    <div class="verdict-pill">'
        f'      <span class="lens-target" data-lens-axis="strict">{head_strict}</span>'
        f'      <span class="lens-target" data-lens-axis="lenient" hidden>{head_lenient}</span>'
        '    </div>'
        '  </div>'
        '  <div class="report-bar-row">'
        f'    <div class="report-bar-caption lens-target" data-lens-axis="strict">Strict lens</div>'
        f'    <div class="report-bar-caption lens-target" data-lens-axis="lenient" hidden>Lenient lens</div>'
        '  </div>'
        f'  <div class="lens-target" data-lens-axis="strict">{rail_strict}</div>'
        f'  <div class="lens-target" data-lens-axis="lenient" hidden>{rail_lenient}</div>'
        f'  <div class="report-bar lens-target" data-lens-axis="strict">{segs_strict}</div>'
        f'  <div class="report-bar lens-target" data-lens-axis="lenient" hidden>{segs_lenient}</div>'
        f'  <div class="report-counts lens-target" data-lens-axis="strict">{counts_strict}</div>'
        f'  <div class="report-counts lens-target" data-lens-axis="lenient" hidden>{counts_lenient}</div>'
        '  <div class="report-cta">'
        f'    <span class="src">{claim_count} claim{"s" if claim_count != 1 else ""}</span>'
        + src_tiers_html +
        '    <span class="read">Read full report →</span>'
        '  </div>'
        '</a>'
    )


# ``_project_dist`` moved to ``aggregation.project_dist`` (1.6) — with the
# non-folding fix: a "Models split" fine bucket now passes through instead of
# being projected to Unverifiable.


def _agg_bar(
    verdict_totals: dict[str, int],
    order: Optional[list[str]] = None,
) -> str:
    """Site-wide aggregate verdict bar + legend.

    ``order`` defaults to the 6-bucket ``VERDICT_ORDER`` for backward
    compat, but the index renderer now passes ``COARSE_VERDICT_ORDER`` for
    both Lenient and Strict aggregate views (rendered side-by-side and
    swapped by the lens toggle).
    """
    label_order = order if order is not None else VERDICT_ORDER
    total = sum(verdict_totals.get(l, 0) for l in label_order) or 1
    segs = []
    legend = []
    for label in label_order:
        count = verdict_totals.get(label, 0)
        if count:
            segs.append(
                f'<div class="seg v-{_verdict_css(label)}" style="width:{count/total*100:.1f}%"'
                f' title="{_esc(label)}: {count}">{count}</div>'
            )
        zero = " zero" if not count else ""
        legend.append(
            f'<div class="legend-item{zero}"><span class="swatch v-{_verdict_css(label)}"></span>'
            f'{_esc(label)} <span class="ct">{count}</span></div>'
        )
    aria = ", ".join(
        f"{verdict_totals.get(l,0)} {l}" for l in label_order if verdict_totals.get(l, 0)
    )
    return (
        '<div class="agg">'
        '  <div class="agg-label">Verdict distribution</div>'
        f'  <div class="agg-bar" role="img" aria-label="Verdict distribution: {aria}">'
        f'    {"".join(segs)}'
        '  </div>'
        f'  <div class="agg-legend">{"".join(legend)}</div>'
        '</div>'
    )

# ── CSS / JS assets ────────────────────────────────────────────────────────

# Toggle: include .how-strip in the page-load rise animation cascade (80ms delay).
# Flip to False if the staggered reveal ever feels too busy; this appends an
# override at the bottom of CSS that neutralizes the animation.
HOW_STRIP_RISE = True

CSS = """\
/* ─────────────────────────────────────────────────────────────────────
   truth-bot — consolidated stylesheet
   Accountability dashboard aesthetic. Verdict colors are sacred —
   they are the ONLY chromatic colors in the design.

   Sections:
     [00] Design tokens
     [01] Reset & base
     [02] Status bar
     [03] Layout primitives
     [04] Masthead (full + compact variants)
     [05] Section heads
     [06] Index page — aggregate stats & verdict bar
     [07] Index page — report cards
     [08] Report page — speech hero
     [09] Report page — verdict panel
     [10] Report page — Truthy column & speech bubble
     [11] Report page — TOC
     [12] Report page — claim cards
     [13] Report page — caveat callout
     [14] Report page — model verdict matrix
     [15] Report page — expandable reasoning
     [16] Report page — evidence list
     [17] Report page — claim footer & methodology
     [18] Footer
     [19] Verdict color utilities
     [20] Truthy SVG internal animations
     [21] Page-load choreography
     [22] Responsive
   ───────────────────────────────────────────────────────────────────── */


/* [00] Design tokens ─────────────────────────────────────────────────── */
:root {
  --bg: #fafaf9;
  --surface: #ffffff;
  --surface-hover: #fdfcfa;
  --surface-warm: #faf8f3;
  --ink: #0c0a09;
  --ink-muted: #57534e;
  --ink-faint: #a8a29e;
  --border: #e7e5e4;
  --border-strong: #d6d3d1;

  /* Verdict palette — the ONLY chromatic colors in the design.
     Change one variable, every bar / pill / swatch / dissent flag /
     Truthy bubble tint updates. Never hardcode these hex values
     anywhere else in the codebase. */
  --v-true:         #15803d;
  --v-mostly-true:  #65a30d;
  --v-exaggerated:  #ca8a04;
  --v-misleading:   #c2410c;
  --v-false:        #991b1b;
  --v-unverifiable: #44403c;
  /* 5-bucket coarse-axis projection (Truthy scale). Used on the headline
     pill only — the per-model strip stays on the 6-bucket palette above.
     truthy sits between true (green) and exaggerated (amber); falsey
     sits between misleading (orange) and false (red). */
  --v-truthy:       #84cc16;
  --v-falsey:       #ea580c;
  /* percent-true headline mid band (50-75% inclusive, jackie 2026-07-25) */
  --v-mid:          #ca8a04;
  /* Models split — panel deadlock. Its own cool slate, distinct from the
     warm-gray Unverifiable, so the aggregate bars can show the split bucket
     as a real segment (T0.2) rather than silently dropping it. */
  --v-split:        #64748b;

  --serif: 'Newsreader', Georgia, 'Times New Roman', serif;
  --sans:  'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --mono:  'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace;

  --max-w: 960px;
}


/* [01] Reset & base ──────────────────────────────────────────────────── */
* { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; scroll-behavior: smooth; scroll-padding-top: 1.5rem; }
body {
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
a { color: inherit; text-decoration: none; }


/* [02] Status bar ────────────────────────────────────────────────────── */
/* Thin terminal-style strip at the top. Signals "monitoring tool". */
.status-bar {
  background: var(--ink);
  color: #d6d3d1;
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 0.55rem 1.25rem;
}
.status-bar .row {
  max-width: var(--max-w);
  margin: 0 auto;
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  align-items: center;
}
.status-bar .live::before {
  content: "●";
  color: #4ade80;
  margin-right: 0.45rem;
  animation: pulse 2.4s ease-in-out infinite;
}
.status-bar .stamp { margin-left: auto; color: #a8a29e; }
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}


/* [03] Layout primitives ─────────────────────────────────────────────── */
.wrap {
  max-width: var(--max-w);
  margin: 0 auto;
  padding: 0 1.25rem;
}


/* [04] Masthead ──────────────────────────────────────────────────────── */
/* Two variants: .masthead (full, used on index) and the compact form
   inside report pages with .wordmark-sm + .breadcrumb. */
header.masthead {
  padding: 3.25rem 0 2.25rem;
  border-bottom: 1px solid var(--border);
}

/* Full masthead (index page) */
.masthead-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 2rem;
  flex-wrap: wrap;
}
.wordmark {
  font-family: var(--serif);
  font-size: 2.6rem;
  font-weight: 500;
  letter-spacing: -0.025em;
  line-height: 1;
  color: var(--ink);
}
/* The lone chrome accent: a red period after "truth-bot". */
.wordmark .dot { color: var(--v-false); }
.tagline {
  margin-top: 0.65rem;
  color: var(--ink-muted);
  font-size: 0.95rem;
  max-width: 42ch;
}

/* Top nav (index page) */
nav.top-nav {
  display: flex;
  gap: 1.4rem;
  align-items: center;
  font-family: var(--mono);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding-top: 0.85rem;
}
nav.top-nav a {
  color: var(--ink-muted);
  transition: color 120ms ease;
  padding: 0.25rem 0;
  border-bottom: 1px solid transparent;
}
nav.top-nav a:hover {
  color: var(--ink);
  border-bottom-color: var(--ink);
}

/* Compact masthead override (report pages add .compact OR set padding inline).
   When a report page uses the smaller wordmark + breadcrumb pattern,
   reduce the masthead padding via the .mast-row container. */
.mast-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}
header.masthead:has(.mast-row) {
  padding: 1.5rem 0 1.25rem;
}
.wordmark-sm {
  font-family: var(--serif);
  font-size: 1.4rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.wordmark-sm .dot { color: var(--v-false); }
.breadcrumb {
  font-family: var(--mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
}
.breadcrumb a { color: var(--ink-muted); transition: color 120ms ease; }
.breadcrumb a:hover { color: var(--ink); }
.breadcrumb .arrow { margin-right: 0.4rem; }


/* [05] Section heads ─────────────────────────────────────────────────── */
.section-head {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-muted);
  margin: 2.75rem 0 1rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}
.section-head .sub { color: var(--ink-faint); }


/* [06] Index page — aggregate stats & verdict bar ────────────────────── */
.stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border: 1px solid var(--border);
  background: var(--surface);
  margin-bottom: 1rem;
}
.stat {
  padding: 1.4rem 1.6rem;
  border-right: 1px solid var(--border);
}
.stat:last-child { border-right: none; }
.stat .num {
  font-family: var(--serif);
  font-size: 2.9rem;
  font-weight: 400;
  line-height: 1;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.025em;
}
.stat .num .unit {
  font-size: 1.5rem;
  color: var(--ink-muted);
  margin-left: 0.05rem;
}
.stat .lbl {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ink-muted);
  margin-top: 0.6rem;
}

/* Aggregate verdict bar (index page hero) */
.agg {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 1.4rem 1.6rem;
}
.agg-label {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ink-muted);
  margin-bottom: 0.85rem;
}
.agg-bar {
  display: flex;
  height: 34px;
  overflow: hidden;
}
.agg-bar .seg {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-family: var(--mono);
  font-size: 0.78rem;
  font-weight: 500;
  transition: filter 200ms ease;
}
.agg-bar .seg:hover { filter: brightness(1.12); }

.agg-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 1.4rem;
  margin-top: 1rem;
  font-family: var(--mono);
  font-size: 0.74rem;
  color: var(--ink-muted);
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}
.legend-item.zero { color: var(--ink-faint); }
.swatch {
  width: 8px;
  height: 8px;
  border-radius: 1px;
  flex-shrink: 0;
}
.legend-item .ct {
  color: var(--ink);
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}
.legend-item.zero .ct { color: var(--ink-faint); }


/* Index page — hero Truthy layout */
.index-hero {
  display: flex;
  align-items: center;
  gap: 2rem;
  padding: 1.5rem 0 1rem;
  flex-wrap: nowrap;
}
a.hero-truthy-link {
  color: inherit;
  text-decoration: none;
  flex-shrink: 0;
  border-radius: 0.5rem;
  outline-offset: 3px;
}
a.hero-truthy-link:focus-visible {
  outline: 2px solid var(--ink-muted);
}
a.hero-truthy-link:hover .hero-truthy-wrap {
  filter: brightness(1.04);
}
.truthy-fun-h1 {
  font-family: var(--serif);
  font-size: 2.1rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  margin: 0 0 0.35rem;
  color: var(--ink);
}
.truthy-sound-row {
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
  flex-wrap: wrap;
  margin-bottom: 1.25rem;
}
.truthy-sound-label {
  font-family: var(--mono);
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  color: var(--ink-muted);
  user-select: none;
}
.truthy-sound-toggle {
  flex-shrink: 0;
  width: 2.5rem;
  height: 2.5rem;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  background: var(--surface);
  color: var(--ink-muted);
  cursor: pointer;
  transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
}
.truthy-sound-toggle:hover {
  background: var(--surface-hover);
  color: var(--ink);
  border-color: var(--ink-muted);
}
.truthy-sound-toggle:focus-visible {
  outline: 2px solid var(--ink-muted);
  outline-offset: 2px;
}
.truthy-sound-toggle-icons { position: relative; width: 22px; height: 22px; }
.truthy-sound-toggle-icons svg {
  position: absolute;
  left: 0;
  top: 0;
  display: block;
}
.truthy-sound-toggle .icon-on { opacity: 1; }
.truthy-sound-toggle .icon-off { opacity: 0; pointer-events: none; }
.truthy-sound-toggle.is-muted .icon-on { opacity: 0; }
.truthy-sound-toggle.is-muted .icon-off { opacity: 1; }
.truthy-fun-notes {
  margin-top: 2rem;
  padding: 1.25rem 1.5rem;
  border: 1px solid var(--border);
  background: var(--surface-warm);
  font-family: var(--sans);
  font-size: 0.9rem;
  line-height: 1.55;
  color: var(--ink-muted);
  max-width: 40rem;
}
.truthy-fun-notes-lead {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--ink);
  line-height: 1.4;
}
.truthy-fun-notes-mascot {
  margin: 0 0 1rem;
  font-size: 0.82rem;
  font-style: italic;
  color: var(--ink-muted);
  line-height: 1.45;
}
.truthy-fun-notes-outro {
  margin: 0;
  font-size: 0.9rem;
  line-height: 1.55;
  color: var(--ink-muted);
}
/* Hero bubble: tail points LEFT toward Truthy (overrides default upward tail).
   width: fit-content + column align-items:flex-start so the bubble hugs caption
   text instead of stretching to the full flex column width. */
.index-hero .truthy-bubble {
  width: fit-content;
  max-width: min(92vw, 260px);
  box-sizing: border-box;
  text-align: left;
}
.index-hero .truthy-bubble::before,
.index-hero .truthy-bubble::after {
  left: -8px;
  top: 1rem;
  transform: none;
  border-left: none;
  border-top: 8px solid transparent;
  border-bottom: 8px solid transparent;
}
.index-hero .truthy-bubble::before { border-right: 8px solid var(--border); border-bottom-color: transparent; }
.index-hero .truthy-bubble::after  { border-right: 8px solid var(--surface-warm); left: -7px; border-bottom-color: transparent; }
.index-hero .truthy-bubble.is-true::before { border-right-color: rgba(21, 128, 61, 0.3); border-bottom-color: transparent; }
.index-hero .truthy-bubble.is-iffy::before { border-right-color: rgba(202, 138, 4, 0.4); border-bottom-color: transparent; }
.index-hero .truthy-bubble.is-lie::before  { border-right-color: rgba(153, 27, 27, 0.3); border-bottom-color: transparent; }
.hero-truthy-wrap {
  flex-shrink: 0;
  animation: hero-truthy-float 3.2s ease-in-out infinite;
}
.hero-truthy-wrap svg {
  width: 280px;
  height: 336px;
}
@keyframes hero-truthy-float {
  0%, 100% { transform: translateY(0); }
  50%      { transform: translateY(-6px); }
}
/* Counter-animate the floor shadow so it stays put in the viewport
   while Truthy bobs, and subtly breathes (a touch smaller when he's up). */
.index-hero #floorShadow,
.truthy-frame #floorShadow {
  transform-box: fill-box;
  transform-origin: 150px 353px;
  animation: hero-shadow-breathe 3.2s ease-in-out infinite;
}
@keyframes hero-shadow-breathe {
  0%, 100% { transform: translateY(0)   scale(1);    opacity: 1; }
  50%      { transform: translateY(6px) scale(0.95); opacity: 0.9; }
}
/* Index hero: gentle wave on left arm while both arms stay raised (happy state) */
@keyframes index-hero-wave-arm {
  0%, 100% { transform: rotate(130deg); }
  50%      { transform: rotate(150deg); }
}
.index-hero #mascot.state-true.hero-wave #armLeftSwing {
  transform-box: view-box;
  transform-origin: 88px 253px;
  animation: index-hero-wave-arm 0.9s ease-in-out infinite;
}
.hero-truthy-col {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}
.stat-wide {
  border-right: none;
}
.stat-breakdown {
  font-size: 0.82rem;
  color: var(--ink-muted);
  margin-top: 0.25rem;
  line-height: 1.45;
}
.stat-breakdown strong { color: var(--ink); font-weight: 600; }


/* [07] Index page — report cards ─────────────────────────────────────── */
.reports { display: flex; flex-direction: column; }
.report {
  background: var(--surface);
  border: 1px solid var(--border);
  border-bottom: none;
  padding: 1.6rem;
  transition: background 150ms ease;
  display: block;
  color: inherit;
}
.report:last-child { border-bottom: 1px solid var(--border); }
.report:hover { background: var(--surface-hover); }

.report-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1.5rem;
  margin-bottom: 1.1rem;
  flex-wrap: wrap;
}
.report-headline {
  font-family: var(--serif);
  font-size: 1.7rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  line-height: 1.15;
  color: var(--ink);
}
.report-meta {
  font-family: var(--mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--ink-muted);
  margin-top: 0.5rem;
}
.report-meta .sep { margin: 0 0.5rem; color: var(--ink-faint); }

.verdict-pill {
  text-align: right;
  flex-shrink: 0;
}
.verdict-pill .label {
  font-family: var(--serif);
  font-size: 1.35rem;
  font-weight: 500;
  letter-spacing: -0.015em;
  line-height: 1.1;
  display: block;
}
.verdict-pill .label.neutral { color: var(--ink); }
.verdict-pill .ratio {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
  margin-top: 0.35rem;
}

/* Paired editorial-lens blocks: ``hidden`` must win over display:flex on
   `.report-bar` / `.report-counts` / etc. Otherwise index report cards
   stack *both* Strict and Lenient bars at once (user only wants one bar). */
[data-lens-axis][hidden] {
  display: none !important;
}

/* Slim verdict bar inside a report card (vs. the chunky one in the verdict panel) */
.report-bar {
  display: flex;
  height: 6px;
  overflow: hidden;
  margin: 0.25rem 0 1rem;
}
.report-bar .seg { transition: filter 200ms ease; }
.report-bar-row {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.6rem;
}
.report-bar-caption {
  font-family: var(--mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
}

.report-counts {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem 1.4rem;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--ink-muted);
}
.report-counts .ct {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.report-counts .ct .n {
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

.report-cta {
  margin-top: 1.1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
.report-cta .src { color: var(--ink-faint); }
.report-cta .src-tiers {
  font-family: var(--mono);
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-faint);
}
.report-cta .read {
  color: var(--ink);
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  transition: gap 200ms ease;
}
.report:hover .report-cta .read { gap: 0.7rem; }


/* [08] Report page — speech hero ─────────────────────────────────────── */
.hero { padding: 2.5rem 0 1rem; }
.hero-overline {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-muted);
  margin-bottom: 0.75rem;
}
.speaker-name {
  font-family: var(--serif);
  font-size: 3.2rem;
  font-weight: 500;
  line-height: 1.02;
  letter-spacing: -0.03em;
  color: var(--ink);
}
.speech-title {
  font-family: var(--serif);
  font-style: italic;
  font-size: 1.7rem;
  font-weight: 400;
  line-height: 1.2;
  letter-spacing: -0.015em;
  color: var(--ink-muted);
  margin-top: 0.4rem;
}
.speech-meta {
  margin-top: 1rem;
  font-family: var(--mono);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--ink-muted);
}
.speech-meta .sep { margin: 0 0.5rem; color: var(--ink-faint); }


/* [09] Report page — verdict panel ───────────────────────────────────── */
.verdict-panel {
  margin-top: 2rem;
  background: var(--surface);
  border: 1px solid var(--border);
}
.vp-headline {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 2rem;
  padding: 1.75rem 1.75rem 1.5rem;
  border-bottom: 1px solid var(--border);
  align-items: stretch;
}
.vp-text-col {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 1.25rem;
  min-width: 0;
}
.vp-verdict {
  font-family: var(--serif);
  font-size: 2.4rem;
  font-weight: 500;
  letter-spacing: -0.025em;
  line-height: 1.05;
}
.vp-verdict.neutral { color: var(--ink); }
.vp-ratio {
  font-family: var(--mono);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
  margin-top: 0.5rem;
}

/* Big verdict bar inside the panel */
.vp-bar-wrap { padding: 1.5rem 1.75rem; }
.vp-lens-caption {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-muted);
  margin-bottom: 0.5rem;
}
.vp-bar {
  display: flex;
  height: 38px;
  overflow: hidden;
}
.vp-bar .seg {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-family: var(--mono);
  font-size: 0.82rem;
  font-weight: 500;
  transition: filter 200ms ease;
}
.vp-bar .seg:hover { filter: brightness(1.12); }
.vp-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.4rem;
  margin-top: 1.1rem;
  font-family: var(--mono);
  font-size: 0.74rem;
  color: var(--ink-muted);
}

/* Source row at the bottom of the verdict panel */
.source-row {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  padding: 1rem 1.75rem;
  border-top: 1px solid var(--border);
  background: var(--surface-warm);
  font-family: var(--mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
.source-row a {
  color: var(--ink);
  border-bottom: 1px solid var(--border-strong);
  padding-bottom: 1px;
  transition: border-color 150ms ease;
}
.source-row a:hover { border-bottom-color: var(--ink); }
.source-row .lab { color: var(--ink-faint); margin-right: 0.5rem; }


/* [10] Report page — Truthy column & speech bubble ───────────────────── */
.vp-truthy-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  min-width: 200px;
}
.truthy-frame {
  position: relative;
  cursor: pointer;
  transition: filter 200ms ease;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  /* Match index-hero bob; shadow counter-animates inside the SVG (#floorShadow). */
  animation: hero-truthy-float 3.2s ease-in-out infinite;
}
.truthy-frame:hover { filter: brightness(1.04); }
.truthy-frame:active { filter: brightness(0.98); }
.truthy-tap-hint {
  position: absolute;
  bottom: -4px;
  right: -4px;
  background: var(--ink);
  color: var(--bg);
  font-family: var(--mono);
  font-size: 0.55rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.25rem 0.45rem;
  border-radius: 2px;
  display: flex;
  align-items: center;
  gap: 0.3rem;
  pointer-events: none;
  opacity: 0;
  transition: opacity 200ms ease;
}
.truthy-frame:hover .truthy-tap-hint { opacity: 1; }
.truthy-tap-hint .icon { width: 8px; height: 8px; }

/* Editorial speech bubble — Truthy's voice in Newsreader italic.
   Tail points up at Truthy. Border tints to match mood. */
.truthy-bubble {
  position: relative;
  background: var(--surface-warm);
  border: 1px solid var(--border);
  padding: 0.7rem 1rem;
  max-width: 240px;
  font-family: var(--serif);
  font-style: italic;
  font-size: 0.95rem;
  line-height: 1.4;
  color: var(--ink);
  text-align: center;
}
.truthy-bubble::before,
.truthy-bubble::after {
  content: "";
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  width: 0;
  height: 0;
  border-left: 8px solid transparent;
  border-right: 8px solid transparent;
}
.truthy-bubble::before {
  top: -9px;
  border-bottom: 8px solid var(--border);
}
.truthy-bubble::after {
  top: -7px;
  border-bottom: 8px solid var(--surface-warm);
}
.truthy-bubble.is-true { border-color: rgba(21, 128, 61, 0.3); }
.truthy-bubble.is-true::before { border-bottom-color: rgba(21, 128, 61, 0.3); }
.truthy-bubble.is-iffy { border-color: rgba(202, 138, 4, 0.4); }
.truthy-bubble.is-iffy::before { border-bottom-color: rgba(202, 138, 4, 0.4); }
.truthy-bubble.is-lie { border-color: rgba(153, 27, 27, 0.3); }
.truthy-bubble.is-lie::before { border-bottom-color: rgba(153, 27, 27, 0.3); }


/* [11] Report page — TOC ─────────────────────────────────────────────── */
.toc {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 0.5rem 0;
}
.toc-item {
  display: grid;
  grid-template-columns: 2.5rem auto 1fr auto;
  gap: 1rem;
  align-items: center;
  padding: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--border);
  transition: background 120ms ease;
  color: inherit;
}
.toc-item:last-child { border-bottom: none; }
.toc-item:hover { background: var(--surface-warm); }
.toc-num {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--ink-faint);
  letter-spacing: 0.06em;
  font-variant-numeric: tabular-nums;
}
.toc-pill {
  font-family: var(--mono);
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #fff;
  padding: 0.2rem 0.55rem;
  border-radius: 2px;
  font-weight: 500;
  white-space: nowrap;
}
.toc-text {
  font-size: 0.92rem;
  color: var(--ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.toc-jump {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--ink-faint);
}


/* [12] Report page — claim cards ─────────────────────────────────────── */
.claim {
  background: var(--surface);
  border: 1px solid var(--border);
  margin-bottom: 1.25rem;
  scroll-margin-top: 1rem;
}
.claim-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  padding: 0.85rem 1.5rem;
  border-bottom: 1px solid var(--border);
  background: var(--surface-warm);
}
.claim-num {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-muted);
  font-variant-numeric: tabular-nums;
}
/* Family rail — brackets the Truthy/Falsey family groups above a verdict bar
   at the same widths as the segments, so the headline's "N of M decided
   claims X-leaning" totals are visibly derivable from the graph. */
.vp-family-rail, .report-family-rail {
  display: flex;
  gap: 2px;
  margin-bottom: 0.35rem;
  font-family: var(--mono);
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.vp-family-rail .fam, .report-family-rail .fam {
  border-top: 2px solid;
  padding-top: 0.2rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}
.vp-family-rail .fam .n, .report-family-rail .fam .n { font-weight: 700; }
.fam.fam-true    { border-color: var(--v-truthy); color: var(--v-true); }
.fam.fam-false   { border-color: var(--v-falsey); color: var(--v-false); text-align: right; }
.fam.fam-abstain { border-color: var(--border-strong, #d6d3d1); color: var(--ink-faint); text-align: center; border-top-style: dashed; }

.claim-pill {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #fff;
  padding: 0.25rem 0.7rem;
  border-radius: 2px;
  font-weight: 500;
}
/* Guest-anecdote genre marker: keeps the Unverifiable color family but a
   dashed outline signals "no public record exists", not "we failed". */
.claim-pill.pill-anecdote, .toc-pill.pill-anecdote {
  outline: 1.5px dashed rgba(255,255,255,0.65);
  outline-offset: -3.5px;
}
/* Self-sourced-only abstention (PR-A2.1): Unverifiable color family, solid
   double-rule outline — "the only witness is the claimant", not "we failed". */
.claim-pill.pill-self-sourced {
  outline: 3px double rgba(255,255,255,0.65);
  outline-offset: -4.5px;
}
/* Per-item marker on the Sources-consulted strip: this record is the
   speaker's own organization (era-scoped principal match). */
.ev-self {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-muted);
  border: 1px solid var(--border-strong);
  border-radius: 2px;
  padding: 0.05rem 0.3rem;
  margin-right: 0.4rem;
  white-space: nowrap;
}
/* Abstention-decomposition chip under the verdict bars (PR-A2.1 T1.2). */
.vp-selfsource-chip {
  font-family: var(--mono);
  font-size: 0.8rem;
  color: var(--ink-muted);
  margin: 0.35rem 0 0;
}
.claim-body { padding: 1.75rem 1.75rem 1.5rem; }
.claim-quote {
  font-family: var(--serif);
  font-size: 1.4rem;
  font-weight: 400;
  line-height: 1.4;
  letter-spacing: -0.012em;
  color: var(--ink);
  padding-left: 1.25rem;
  border-left: 3px solid var(--border-strong);
}
/* Claim-in-context (2026-08-01): the checked sentence emphasized inside its
   greyed transcript neighbors — deictic quotes ("we'll launch six more") are
   unreadable bare, and the panel always judged with this context. */
.claim-quote-ctx { font-size: 1.15rem; }
.claim-quote-ctx .ccq-side {
  color: var(--ink-muted);
  font-size: 0.92rem;
  font-weight: 400;
}
.claim-quote-ctx .ccq-claim {
  color: var(--ink);
  font-size: 1.32rem;
  font-weight: 700;
}
.claim-context-fallback {
  margin-top: 0.6rem;
  padding-left: 1.25rem;
  font-size: 0.92rem;
  color: var(--ink-muted);
}
.claim-context-fallback .ccq-label {
  font-family: var(--mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-right: 0.5rem;
}
.claim-context {
  margin-top: 1rem;
  font-family: var(--mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--ink-muted);
}
.claim-context .sep { margin: 0 0.5rem; color: var(--ink-faint); }


/* [13] Report page — caveat callout ──────────────────────────────────── */
/* Always visible, never collapsible. Editorial annotation style. */
.caveat {
  margin: 1.5rem 0 0;
  background: #fefbf3;
  border-left: 3px solid var(--v-exaggerated);
  padding: 0.9rem 1.1rem;
}
.caveat-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--v-exaggerated);
  font-weight: 600;
  margin-bottom: 0.35rem;
}
.caveat-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.caveat-item {
  display: block;
  font-size: 0.92rem;
  line-height: 1.55;
  color: var(--ink);
}
.caveat-item + .caveat-item {
  margin-top: 0.45rem;
  padding-top: 0.45rem;
  border-top: 1px dashed color-mix(in srgb, var(--v-exaggerated) 35%, transparent);
}
.caveat-attribution {
  display: inline-block;
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--v-exaggerated);
  font-weight: 600;
  margin-right: 0.45rem;
}
.caveat-attribution::after {
  content: "\2022";
  margin-left: 0.45rem;
  color: var(--ink-faint);
  font-weight: 400;
}
.caveat-text {
  font-size: 0.92rem;
  color: var(--ink);
  line-height: 1.55;
}


/* [14] Report page — model verdict matrix ────────────────────────────── */
.models { margin-top: 1.75rem; }
.models-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.65rem;
}
.models-label,
.models-agreement {
  font-family: var(--mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ink-muted);
}
.models-agreement .num {
  color: var(--ink);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.pca-provenance {
  font-family: var(--mono);
  font-size: 0.66rem;
  color: var(--ink-muted);
  margin-bottom: 0.6rem;
  line-height: 1.5;
  word-break: break-word;
}
.pca-seats {
  margin-top: 0.15rem;
  color: var(--ink-faint);
}
/* Post-publication correction note (T1.5) — amber, can't be missed. */
.pca-correction {
  margin-top: 0.25rem;
  color: var(--v-exaggerated);
}
.pca-correction a { color: var(--v-exaggerated); text-decoration: underline; }
.corrections-table td.mono { font-family: var(--mono); font-size: 0.8rem; }
.corrections-note { color: var(--ink-muted); }
/* Pipeline diagram (About, T4.2) — structural only, no figures. */
.pipeline-diagram {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem;
  margin: 1rem 0; font-family: var(--sans); font-size: 0.82rem;
}
.pipeline-diagram .pd-node {
  border: 1px solid var(--rule, #ccc); border-radius: 6px;
  padding: 0.35rem 0.6rem; text-align: center; line-height: 1.25;
}
.pipeline-diagram .pd-node small { color: var(--ink-muted); font-size: 0.72rem; }
.pipeline-diagram .pd-arrow { color: var(--ink-faint); }
.seat-insights td .ct { font-weight: 600; }
.report-correction-banner {
  margin: 0.75rem 0 1.25rem;
  padding: 0.6rem 0.9rem;
  border-left: 3px solid var(--v-exaggerated);
  background: color-mix(in srgb, var(--v-exaggerated) 8%, transparent);
  font-size: 0.9rem;
}
.report-correction-banner a { text-decoration: underline; }
/* Statement Triage — set-aside (non-check-worthy) sentence stream */
.triage-group { margin: 0 0 1.5rem; }
.triage-list {
  list-style: none;
  margin: 0.5rem 0 0;
  padding: 0;
  border-top: 1px solid var(--border);
}
.triage-item {
  padding: 0.6rem 0;
  border-bottom: 1px solid var(--border);
}
.triage-text { line-height: 1.5; }
.triage-meta {
  font-family: var(--mono);
  font-size: 0.66rem;
  color: var(--ink-muted);
  margin-top: 0.25rem;
}
.triage-tag {
  display: inline-block;
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 600;
  padding: 0.05rem 0.35rem;
  border: 1px solid var(--border);
  border-radius: 3px;
  margin-right: 0.4rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.model-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border: 1px solid var(--border);
}
.model {
  padding: 0.75rem 0.85rem;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  background: var(--surface);
  position: relative;
}
.model:last-child { border-right: none; }
/* Dissenting model: warm bg + small DISSENT tag in the corner */
.model.dissent { background: #fefbf3; }
.model.dissent::after {
  content: "DISSENT";
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  font-family: var(--mono);
  font-size: 0.55rem;
  letter-spacing: 0.08em;
  color: var(--v-exaggerated);
  font-weight: 600;
}
/* No-response model: muted bg + FAILED tag in the corner */
.model.no-response { background: #f9f8f7; }
.model.no-response::after {
  content: "FAILED";
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  font-family: var(--mono);
  font-size: 0.55rem;
  letter-spacing: 0.08em;
  color: var(--ink-faint);
  font-weight: 600;
}
.model-name {
  font-family: var(--mono);
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
  font-weight: 500;
}
.model-verdict {
  font-size: 0.85rem;
  font-weight: 500;
}



/* [15] Report page — per-model reasoning (native <details>) ─────────── */
details.model-reasoning {
  margin-top: 0.65rem;
  border: 1px solid var(--border);
}
details.model-reasoning > summary {
  list-style: none;
  cursor: pointer;
  padding: 0.6rem 0.85rem;
  font-family: var(--mono);
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
  display: flex;
  align-items: center;
  gap: 0.4rem;
  transition: background 120ms ease;
}
details.model-reasoning > summary .model-reasoning-model {
  font-family: var(--mono);
  font-weight: 400;
  font-size: 0.82em;
  color: var(--ink-muted);
  letter-spacing: 0.02em;
}
details.model-reasoning > summary::-webkit-details-marker { display: none; }
details.model-reasoning > summary::before {
  content: "▸";
  transition: transform 200ms ease;
  color: var(--ink-faint);
  display: inline-block;
}
details.model-reasoning[open] > summary::before { transform: rotate(90deg); }
details.model-reasoning > summary:hover { background: var(--surface-warm); }
.model-reasoning-body {
  padding: 0 0.85rem 0.85rem;
  font-size: 0.88rem;
  line-height: 1.6;
  color: var(--ink);
}
.model-reasoning-body p { margin: 0.4rem 0 0; }
.model-reasoning-body p:first-child { margin-top: 0.2rem; }

details.model-tier-wrap {
  margin-top: 0.45rem;
  border: 1px dashed var(--border);
}
details.model-tier-wrap > summary.model-tier-sum {
  list-style: none;
  cursor: pointer;
  padding: 0.35rem 0.6rem;
  font-family: var(--mono);
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-muted);
}
details.model-tier-wrap > summary::-webkit-details-marker { display: none; }
.model-tier-body {
  padding: 0 0.6rem 0.5rem;
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--ink);
}

/* [16] Report page — evidence list ───────────────────────────────────── */
.evidence { margin-top: 1.5rem; }
.evidence-label {
  font-family: var(--mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-muted);
  margin-bottom: 0.55rem;
}
.evidence-list { list-style: none; }
.evidence-list li {
  padding: 0.55rem 0;
  border-top: 1px solid var(--border);
  display: flex;
  gap: 0.65rem;
  align-items: baseline;
  font-size: 0.88rem;
  line-height: 1.4;
}
.evidence-list li:first-child { border-top: none; }
.evidence-list .ev-mark {
  font-family: var(--mono);
  color: var(--ink-faint);
  font-size: 0.75rem;
}
.evidence-list .ev-id {
  font-family: var(--mono);
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--ink-muted);
  margin: 0 0.35rem 0 0.15rem;
}
/* E-id references inside model reasoning → jump links to the pack item. */
.model-reasoning-body a.ev-ref {
  font-family: var(--mono);
  font-size: 0.8em;
  font-weight: 600;
  color: var(--ink);
  border-bottom: 1px dashed var(--border-strong);
}
.model-reasoning-body a.ev-ref:hover { border-bottom-style: solid; }
/* Briefly spotlight the pack item a clicked E-id lands on. */
.evidence-list li:target { background: var(--surface-raised, rgba(0,0,0,0.05)); }
.evidence-list a {
  color: var(--ink);
  border-bottom: 1px solid var(--border-strong);
  padding-bottom: 1px;
}
.evidence-list a:hover { border-bottom-color: var(--ink); }
.evidence-list .ev-src {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--ink-faint);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-left: 0.5rem;
}
/* Source-tier badges (preserved from existing schema) */
.evidence-tier {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.05em;
  padding: 0.1rem 0.35rem;
  color: #fff;
  border-radius: 1px;
  font-weight: 600;
  margin-right: 0.4rem;
}
.tier-gov   { background: #1565c0; }
.tier-news  { background: #4a148c; }
.tier-fc    { background: #e65100; }
.tier-other { background: #546e7a; }
/* T7 political communications — warm brown, deliberately NOT the verdict red,
   which means a FALSE ruling rather than an untrusted source. */
.tier-political { background: #6d4c41; }
.evidence-list .source-snippet {
  display: block;
  width: 100%;
  margin-top: 0.35rem;
  font-size: 0.82rem;
  line-height: 1.4;
  color: var(--ink-muted);
}


/* Collapsible evidence/sources (native <details>) */
details.evidence-details {
  margin-top: 1.5rem;
  border: 1px solid var(--border);
}
details.evidence-details > summary.evidence-summary {
  list-style: none;
  cursor: pointer;
  padding: 0.6rem 1rem;
  font-family: var(--mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ink-muted);
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: background 120ms ease;
  user-select: none;
}
details.evidence-details > summary.evidence-summary::-webkit-details-marker { display: none; }
details.evidence-details > summary.evidence-summary::before {
  content: "▶";
  font-size: 0.6rem;
  color: var(--ink-faint);
  transition: transform 200ms ease;
  display: inline-block;
}
details.evidence-details[open] > summary.evidence-summary::before { transform: rotate(90deg); }
details.evidence-details > summary.evidence-summary:hover { background: var(--surface-warm); }
details.evidence-details .evidence { padding: 0.5rem 1rem 1rem; }


/* [17] Report page — claim footer & methodology ──────────────────────── */
.claim-foot {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
  flex-wrap: wrap;
  gap: 0.5rem;
}
.permalink {
  color: var(--ink-faint);
  transition: color 120ms ease;
}
.permalink:hover { color: var(--ink); }
.permalink::before {
  content: "#";
  margin-right: 0.15rem;
  color: var(--ink-faint);
}

.methodology {
  margin-top: 3rem;
  background: var(--surface-warm);
  border: 1px solid var(--border);
  padding: 1.25rem 1.5rem;
  font-size: 0.88rem;
  color: var(--ink-muted);
  line-height: 1.6;
}
.methodology strong { color: var(--ink); font-weight: 600; }
.methodology a {
  color: var(--ink);
  border-bottom: 1px solid var(--border-strong);
}

/* PCA panel composition — which model fills each seat, once per run. */
.panel-composition {
  margin-top: 1rem;
  padding: 0.75rem 1.25rem;
  background: var(--surface-warm);
  border: 1px solid var(--border);
  font-size: 0.85rem;
  color: var(--ink-muted);
}
.panel-composition-head {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-bottom: 0.5rem;
}
.panel-composition-title {
  font-family: var(--mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-muted);
}
.panel-composition-roster {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--ink-faint);
}
.panel-composition-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.25rem;
}
.panel-composition-seat {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
}
.panel-composition-role {
  font-family: var(--mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-faint);
}
.panel-composition-model { color: var(--ink); }


/* [18] Footer ────────────────────────────────────────────────────────── */
footer.foot {
  margin-top: 4rem;
  padding: 1.5rem 0 2.5rem;
  border-top: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--ink-muted);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}
footer.foot a {
  color: var(--ink);
  border-bottom: 1px solid var(--border-strong);
  padding-bottom: 1px;
}
footer.foot .footer-hash {
  font-family: var(--mono);
  color: var(--ink-faint);
  border-bottom: none;
  text-decoration: none;
}
footer.foot .footer-hash:hover {
  text-decoration: underline;
  color: var(--ink-muted);
}


/* [18a] Beta release badge ──────────────────────────────────────────── */
/* Rendered inline next to every "Pipeline vX.Y.Z" version string while the
   project is pre-1.0 (see IS_BETA in site.py). Auto-hidden once version ≥ 1.0
   because the render sites concat an empty string instead of the span. */
.beta-badge {
  display: inline-block;
  margin-left: 0.4em;
  padding: 0.05em 0.45em;
  font-family: var(--mono);
  font-size: 0.82em;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink);
  background: var(--surface-warm, #f5f4ef);
  border: 1px solid var(--border-strong, #d6d3d1);
  border-radius: 3px;
  line-height: 1.3;
  vertical-align: baseline;
}
.status-bar .beta-badge {
  color: #d6d3d1;
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.25);
}
footer.foot .beta-badge {
  text-transform: uppercase;
  letter-spacing: 0.08em;
}


/* [18b] HR dividers & dim helper ────────────────────────────────────── */
hr.rule {
  border: none;
  border-top: 1px solid var(--border-strong);
  margin: 2rem 0;
}
hr.rule-light {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0;
}
/* Muted body copy — used for footnotes, about-page prose, empty states */
.dim {
  color: var(--ink-muted);
  font-size: 0.88rem;
  line-height: 1.6;
}
.dim a {
  color: var(--ink);
  border-bottom: 1px solid var(--border-strong);
  padding-bottom: 1px;
}
.dim a:hover { border-bottom-color: var(--ink); }


/* [19] Verdict color utilities ───────────────────────────────────────── */
/* Background paint */
.v-true         { background: var(--v-true); }
.v-mostly-true  { background: var(--v-mostly-true); }
.v-exaggerated  { background: var(--v-exaggerated); }
.v-misleading   { background: var(--v-misleading); }
.v-false        { background: var(--v-false); }
.v-unverifiable { background: var(--v-unverifiable); }
/* 5-bucket coarse-axis paint (headline pill only). */
.v-truthy       { background: var(--v-truthy); }
.v-falsey       { background: var(--v-falsey); }
.v-split        { background: var(--v-split); }
/* Text paint */
.vt-true         { color: var(--v-true); }
.vt-mostly-true  { color: var(--v-mostly-true); }
.vt-exaggerated  { color: var(--v-exaggerated); }
.vt-misleading   { color: var(--v-misleading); }
.vt-false        { color: var(--v-false); }
.vt-mid          { color: var(--v-mid); }
.vt-unverifiable { color: var(--v-unverifiable); }
.vt-truthy       { color: var(--v-truthy); }
.vt-falsey       { color: var(--v-falsey); }
.vt-split        { color: var(--v-split); }

/* ── Editorial-lens chip (status bar) ──────────────────────────────────────
   Toggles the headline pill between the Lenient and Strict 5-bucket
   projections. Hidden by default; the toggle JS reveals it once it has
   wired up at least one ``.claim-pill-headline`` element on the page. */
.editorial-lens {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.1rem 0.55rem;
  border: 1px solid var(--rule);
  border-radius: 999px;
  background: transparent;
  color: inherit;
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  cursor: pointer;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}
.editorial-lens:hover,
.editorial-lens:focus-visible {
  background: rgba(0, 0, 0, 0.04);
  border-color: var(--ink-faint);
  outline: none;
}
.editorial-lens .lens-label {
  color: var(--ink-faint);
  text-transform: uppercase;
}
.editorial-lens .lens-value {
  font-weight: 600;
}
.editorial-lens[data-lens="strict"] .lens-value {
  color: var(--v-falsey);
}
.editorial-lens[data-lens="lenient"] .lens-value {
  color: var(--v-truthy);
}


/* [20] Truthy SVG internal animations ────────────────────────────────── */
/* These drive Truthy's idle behavior, eye states, and pose changes.
   The state classes (.state-true, .state-iffy, .state-lie) are toggled
   on the #mascot SVG by truthbot.js based on the data-mood attribute. */

@keyframes idle {
  0%, 100% { transform: translateY(0); }
  50%      { transform: translateY(-2.5px); }
}
#character {
  animation: idle 4s ease-in-out infinite;
  transform-origin: center bottom;
}

@keyframes antenna-sway {
  0%, 100% { transform: rotate(-2deg); }
  50%      { transform: rotate(2deg); }
}
#antenna {
  animation: antenna-sway 3s ease-in-out infinite;
  transform-origin: 150px 62px;
  transform-box: fill-box;
}

.eye-led { opacity: 0; transition: opacity 0.35s ease; }

@keyframes true-happy-cycle {
  0%, 70% { opacity: 1; }
  78%, 88% { opacity: 0; }
  96%, 100% { opacity: 1; }
}
@keyframes true-neutral-cycle {
  0%, 70% { opacity: 0; }
  78%, 88% { opacity: 1; }
  96%, 100% { opacity: 0; }
}
@keyframes happy-pulse {
  0%, 100% { transform: scale(1); }
  50%      { transform: scale(1.08); }
}
.state-true .eye-happy {
  animation:
    true-happy-cycle 4s ease-in-out infinite,
    happy-pulse 2.2s ease-in-out infinite;
  transform-origin: center;
  transform-box: fill-box;
}
.state-true .eye-neutral { animation: true-neutral-cycle 4s ease-in-out infinite; }
.state-iffy .eye-iffy { opacity: 1; }

@keyframes sad-wander {
  0%   { transform: translate(-4px, 0.5px); }
  25%  { transform: translate(-3px, 2px); }
  50%  { transform: translate( 4px, 2.5px); }
  75%  { transform: translate( 3px, 1.2px); }
  100% { transform: translate(-4px, 0.5px); }
}
.state-lie .eye-sad {
  opacity: 1;
  animation: sad-wander 4.2s ease-in-out infinite;
  transform-origin: center;
  transform-box: fill-box;
}
.state-lie #eyeRightGroup .eye-sad { animation-delay: -1.3s; }

.eye-shape {
  transform-origin: center;
  transform-box: fill-box;
  transition: transform 0.09s ease-out;
}
#mascot.blinking .eye-shape { transform: scaleY(0.06); }

@keyframes tear-fall {
  0%   { transform: translateY(-4px); opacity: 0; }
  18%  { opacity: 1; }
  100% { transform: translateY(72px); opacity: 0; }
}
.state-lie #tearLeft,
.state-lie #tearRight {
  animation: tear-fall 2.2s ease-in infinite;
  transform-origin: center;
  transform-box: fill-box;
}
.state-lie #tearRight { animation-delay: 0.7s; }
#tearLeft, #tearRight { opacity: 0; }

/* Report Truthy is rendered smaller than index-hero (170px vs 280px wide SVG).
   Scale tears in .truthy-frame so on-screen tear mass tracks the hero (~280/170
   ≈ 1.65); extra lift on <=740px when the SVG is only 110px wide. */
.truthy-frame #mascot.state-lie #tearLeft,
.truthy-frame #mascot.state-lie #tearRight {
  transform-box: fill-box;
  transform-origin: center top;
  animation-name: tear-fall-report;
}
.truthy-frame #mascot.state-lie #tearRight { animation-delay: 0.7s; }

@keyframes tear-fall-report {
  0%   { transform: translateY(-4px) scale(1.82); opacity: 0; }
  18%  { opacity: 1; }
  100% { transform: translateY(76px) scale(1.82); opacity: 0; }
}

@keyframes tear-fall-report-sm {
  0%   { transform: translateY(-4px) scale(2.62); opacity: 0; }
  18%  { opacity: 1; }
  100% { transform: translateY(82px) scale(2.62); opacity: 0; }
}

#armLeftSwing,
#armRightSwing,
#eyeLeftGroup,
#eyeRightGroup,
#headGroup,
#bodyGroup,
#clipboard {
  transition: transform 0.55s cubic-bezier(0.34, 1.56, 0.64, 1);
}
#led, #ledHalo { transition: fill 0.3s; }

/* Triggered by JS when user clicks Truthy: brief LED flash */
@keyframes ledFlash {
  0%   { filter: brightness(1); }
  30%  { filter: brightness(1.6) saturate(1.3); }
  100% { filter: brightness(1); }
}
#mascot.speaking #led,
#mascot.speaking #ledHalo {
  animation: ledFlash 0.7s ease-out;
}


/* [21] Page-load choreography ────────────────────────────────────────── */
@keyframes rise {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
/* Staggered reveal of major content blocks */
.stats, .how-strip, .agg, .hero, .verdict-panel, .toc, .reports .report, .claim {
  animation: rise 480ms cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
}
/* Index page stagger */
.stats     { animation-delay: 50ms; }
.how-strip { animation-delay: 80ms; }
.agg       { animation-delay: 130ms; }
.reports .report:nth-of-type(1) { animation-delay: 240ms; }
.reports .report:nth-of-type(2) { animation-delay: 320ms; }
.reports .report:nth-of-type(3) { animation-delay: 400ms; }
/* Report page stagger */
.hero          { animation-delay: 60ms; }
.verdict-panel { animation-delay: 140ms; }
.toc           { animation-delay: 220ms; }


/* [22] Responsive ────────────────────────────────────────────────────── */
@media (max-width: 740px) {
  /* Masthead variants */
  .wordmark { font-size: 2rem; }
  header.masthead { padding: 2.25rem 0 1.5rem; }
  .masthead-row { flex-direction: column; gap: 0.5rem; }
  nav.top-nav { padding-top: 0.5rem; }

  /* Status bar */
  .status-bar .stamp { margin-left: 0; }

  /* Index aggregate stats — single column. Report .stats.stats-4 and
     .stats.stats-5 own their own breakpoints (further down in this
     stylesheet) so they're excluded here. */
  .stats:not(.stats-4):not(.stats-5) { grid-template-columns: 1fr; }
  .stat {
    border-right: none;
    border-bottom: 1px solid var(--border);
    padding: 1.1rem 1.3rem;
  }
  .stat:last-child { border-bottom: none; }
  .stat .num { font-size: 2.4rem; }
  /* Index hero stacks on mobile */
  .index-hero { flex-direction: column; gap: 1rem; padding: 1rem 0 0.5rem; flex-wrap: wrap; }
  .hero-truthy-wrap svg { width: 180px; height: 216px; }
  /* Mobile hero stacks vertically; revert the bubble tail to point UP at Truthy. */
  .index-hero .truthy-bubble::before,
  .index-hero .truthy-bubble::after {
    left: 50%;
    top: -9px;
    transform: translateX(-50%);
    border-left: 8px solid transparent;
    border-right: 8px solid transparent !important;
    border-top: none;
    border-bottom: 8px solid var(--border);
  }
  .index-hero .truthy-bubble::after { top: -7px; border-bottom-color: var(--surface-warm); }
  .index-hero .truthy-bubble.is-true::before { border-right-color: transparent !important; border-bottom-color: rgba(21, 128, 61, 0.3); }
  .index-hero .truthy-bubble.is-iffy::before { border-right-color: transparent !important; border-bottom-color: rgba(202, 138, 4, 0.4); }
  .index-hero .truthy-bubble.is-lie::before  { border-right-color: transparent !important; border-bottom-color: rgba(153, 27, 27, 0.3); }
  .index-hero .truthy-bubble { max-width: min(92vw, 240px); }

  /* Report card layout collapses verdict pill below headline */
  .report-top { flex-direction: column; gap: 0.85rem; }
  .verdict-pill { text-align: left; }

  /* Speech hero */
  .speaker-name { font-size: 2.2rem; }
  .speech-title { font-size: 1.3rem; }

  /* Verdict panel: Truthy goes inline next to bubble (bubble tail re-points) */
  .vp-headline {
    grid-template-columns: 1fr;
    gap: 1.5rem;
    padding: 1.5rem 1.25rem 1.25rem;
  }
  .vp-truthy-col {
    flex-direction: row;
    align-items: flex-start;
    gap: 1rem;
    min-width: 0;
  }
  .vp-truthy-col .truthy-frame svg { width: 110px; height: 132px; }
  .truthy-frame #mascot.state-lie #tearLeft,
  .truthy-frame #mascot.state-lie #tearRight {
    animation-name: tear-fall-report-sm;
  }
  .truthy-bubble {
    max-width: none;
    flex: 1;
    text-align: left;
  }
  /* Re-point the bubble tail leftward toward Truthy */
  .truthy-bubble::before,
  .truthy-bubble::after {
    left: -8px;
    top: 1rem;
    transform: none;
    border-left: none;
    border-top: 8px solid transparent;
    border-bottom: 8px solid transparent;
  }
  .truthy-bubble::before { border-right: 8px solid var(--border); }
  .truthy-bubble::after  { border-right: 8px solid var(--surface-warm); left: -7px; }
  .truthy-bubble.is-true::before { border-right-color: rgba(21, 128, 61, 0.3); border-bottom-color: transparent; }
  .truthy-bubble.is-iffy::before { border-right-color: rgba(202, 138, 4, 0.4); border-bottom-color: transparent; }
  .truthy-bubble.is-lie::before  { border-right-color: rgba(153, 27, 27, 0.3); border-bottom-color: transparent; }

  .vp-bar-wrap, .source-row { padding-left: 1.25rem; padding-right: 1.25rem; }

  /* Model grid stacks */
  .model-grid { grid-template-columns: 1fr; }
  .model {
    border-right: none;
    border-bottom: 1px solid var(--border);
  }
  .model:last-child { border-bottom: none; }

  /* TOC compresses */
  .toc-item { grid-template-columns: 2rem auto 1fr; }
  .toc-jump { display: none; }
  .toc-text { white-space: normal; }

  /* Claim cards */
  .claim-body { padding: 1.25rem; }
  .claim-quote { font-size: 1.15rem; padding-left: 1rem; }
}


/* [23] Stat icons (landing page) ─────────────────────────────────────── */
.stat {
  display: flex;
  flex-direction: column;
}
.stat-icon {
  color: var(--ink-muted);
  margin-bottom: 0.5rem;
  opacity: 0.7;
}

/* [24] How-it-works trust strip (landing page) ───────────────────────── */
.how-strip {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem 1.25rem;
  margin-bottom: 1rem;
  border: 1px solid var(--border);
  background: var(--surface-warm);
  font-family: var(--sans);
  font-size: 0.78rem;
  color: var(--ink-muted);
  line-height: 1.45;
}
.how-step {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  flex: 1;
}
.how-num {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--ink-faint);
  background: var(--border);
  width: 1.3rem;
  height: 1.3rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  flex-shrink: 0;
}
.how-text {
  flex: 1;
}
.how-sep {
  color: var(--ink-faint);
  font-family: var(--mono);
  font-size: 0.8rem;
  padding-top: 0.15rem;
  flex-shrink: 0;
}

/* [25] Larger landing-page stat icons (hero scale) ───────────────────── */
.stat-icon-lg {
  margin-bottom: 0.75rem;
  opacity: 0.8;
}

/* [27] Inline icons in section heads + claim cards ───────────────────── */
.section-head-label {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}
.section-head-icon {
  color: var(--ink-muted);
  opacity: 0.75;
  margin-bottom: 0; /* override default .stat-icon margin */
  flex-shrink: 0;
}
.claim-head-lead {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  min-width: 0;
}
.claim-head-icon {
  color: var(--ink-muted);
  opacity: 0.7;
  margin-bottom: 0;
  flex-shrink: 0;
}

/* [28] Per-claim back-links in .claim-foot ───────────────────────────── */
.claim-back-links {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  font-size: 0.68rem;
  color: var(--ink-faint);
}
.claim-back-links .back-link {
  color: var(--ink-muted);
  text-decoration: none;
  border-bottom: 1px dotted var(--border);
  transition: color 120ms ease, border-color 120ms ease;
}
.claim-back-links .back-link:hover {
  color: var(--ink);
  border-bottom-color: var(--ink-muted);
}
.claim-back-links .sep {
  color: var(--ink-faint);
  user-select: none;
}

/* [29] Small-screen overrides for stat icons + how-strip ─────────────── */
@media (max-width: 600px) {
  .stat { flex-direction: row; align-items: center; gap: 1rem; }
  .stat-icon { flex-shrink: 0; margin-bottom: 0; }
  /* Shrink the hero-scale icons so text still wraps nicely on phones */
  .stat-icon-lg { width: 32px; height: 32px; }
  .how-strip { flex-direction: column; gap: 0.5rem; }
  .how-sep { display: none; }
  /* Allow the claim-foot back-links to wrap below on narrow screens */
  .claim-foot {
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: flex-start;
  }
  .claim-back-links {
    order: 3;
    width: 100%;
    justify-content: flex-start;
  }
}

/* [30] Four-column report stats variant */
.stats.stats-4 {
  grid-template-columns: repeat(4, 1fr);
}
.verdict-panel > .stats.stats-4 {
  border-top: 1px solid var(--border);
  margin: 0 1.25rem 1rem;
}
@media (max-width: 700px) {
  .stats.stats-4 { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
  .stats.stats-4 { grid-template-columns: 1fr; }
}

/* [30b] Headline-stats frames — promoted % truthy / % false above the
   aggregate stats grid. Two prominent block frames so the verdict
   percentages aren't competing visually with claim/model/leader counts. */
.vp-headline-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  padding: 1rem 1.25rem 0.5rem;
}
.vp-headline-stat {
  position: relative;
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.1rem 1rem 1.4rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}
.vp-headline-stat::before {
  /* Tinted left accent strip — Truthy frame uses the truthy color,
     False frame uses the falsey color. Matches the per-claim pill
     palette so the two frames are visually distinct. */
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 5px;
}
.vp-headline-stat.vp-stat-truthy::before { background: var(--v-truthy); }
.vp-headline-stat.vp-stat-false::before  { background: var(--v-falsey); }
.vp-headline-stat .vp-stat-icon { color: var(--ink-muted); flex: 0 0 auto; }
.vp-headline-stat.vp-stat-truthy .vp-stat-icon { color: var(--v-truthy); }
.vp-headline-stat.vp-stat-false  .vp-stat-icon { color: var(--v-falsey); }
.vp-headline-stat .vp-stat-body { display: flex; flex-direction: column; gap: 0.1rem; min-width: 0; }
.vp-headline-stat .vp-stat-num {
  font-family: var(--serif);
  font-size: 2.2rem;
  font-weight: 500;
  line-height: 1.05;
  letter-spacing: -0.02em;
}
.vp-headline-stat .vp-stat-lbl {
  font-family: var(--mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink);
  margin-top: 0.15rem;
}
.vp-headline-stat .vp-stat-hint {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--ink-faint);
  margin-top: 0.1rem;
}
@media (max-width: 600px) {
  .vp-headline-stats { grid-template-columns: 1fr; }
}

/* [16] Model Panel Insights ─────────────────────────────────────────── */

.insights-strip {
  margin: 0 0 1.4rem;
}
.insight-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border: 1px solid var(--border);
  background: var(--surface);
}
.insight-card {
  padding: 1.1rem 1.25rem;
  border-right: 1px solid var(--border);
}
.insight-card:last-child { border-right: none; }
.insight-card-eyebrow {
  font-family: var(--mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
  margin-bottom: 0.4rem;
}
.insight-card-headline {
  font-family: var(--serif);
  font-size: 1.05rem;
  font-weight: 600;
  line-height: 1.3;
  margin-bottom: 0.3rem;
  color: var(--ink);
}
.insight-card-figure {
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--ink-muted);
}
@media (max-width: 700px) {
  .insight-cards { grid-template-columns: 1fr; }
  .insight-card { border-right: none; border-bottom: 1px solid var(--border); }
  .insight-card:last-child { border-bottom: none; }
}

/* Per-model summary table on the dedicated insights page */
.insights-summary {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
  margin: 0.75rem 0 1.5rem;
}
.insights-summary th, .insights-summary td {
  padding: 0.5rem 0.7rem;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
.insights-summary thead th {
  font-family: var(--mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
}
.insights-summary .num-cell { text-align: right; font-variant-numeric: tabular-nums; }
.insights-meta {
  font-size: 0.85rem;
  color: var(--ink-muted);
}

/* Bias chart — paired horizontal bars centered on a midpoint */
.bias-chart {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin: 0.5rem 0 1.5rem;
}
.bias-row {
  display: grid;
  grid-template-columns: 13rem 1fr 4rem;
  align-items: center;
  gap: 0.75rem;
}
.bias-row-label { font-weight: 600; }
.bias-track {
  position: relative;
  height: 14px;
  background: var(--surface-warm);
  border: 1px solid var(--border);
  border-radius: 2px;
}
.bias-mid {
  position: absolute;
  left: 50%;
  top: -2px;
  bottom: -2px;
  width: 1px;
  background: var(--ink-faint);
}
.bias-fill {
  position: absolute;
  top: 0;
  bottom: 0;
}
.bias-fill-lenient { left: 50%; background: var(--v-truthy); }
.bias-fill-strict  { right: 50%; background: var(--v-falsey); }
.bias-row-figure {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
  text-align: right;
}
@media (max-width: 600px) {
  .bias-row { grid-template-columns: 1fr; }
  .bias-row-figure { text-align: left; }
}

/* Pairwise agreement matrix */
.agreement-matrix {
  border-collapse: collapse;
  font-size: 0.92rem;
  margin: 0.5rem 0 1.5rem;
}
.agreement-matrix th, .agreement-matrix td {
  padding: 0.45rem 0.7rem;
  border: 1px solid var(--border);
  text-align: center;
}
.agreement-matrix thead th { background: var(--surface); }
.agreement-matrix .agg-self {
  background: var(--surface);
  color: var(--ink-faint);
}
.agreement-matrix .agg-cell {
  font-variant-numeric: tabular-nums;
}
.agreement-matrix .agg-n {
  display: block;
  font-size: 0.7rem;
  color: var(--ink-muted);
  font-family: var(--mono);
}

/* Extreme split cards */
.insights-extremes { margin-top: 1.25rem; }
.extreme-card {
  border: 1px solid var(--border);
  padding: 0.85rem 1rem;
  margin-bottom: 0.75rem;
  background: var(--surface);
}
.extreme-head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.4rem;
  font-size: 0.78rem;
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-muted);
}
.extreme-diff {
  background: var(--ink);
  color: var(--bg);
  padding: 0.1rem 0.4rem;
  border-radius: 2px;
}
.extreme-odd-label {
  margin-left: auto;
  color: var(--ink);
  text-transform: none;
  letter-spacing: 0;
  font-family: var(--serif);
  font-size: 0.92rem;
}
.extreme-text {
  margin: 0.25rem 0;
  font-size: 0.96rem;
  line-height: 1.4;
}
.extreme-meta {
  margin: 0;
  font-size: 0.8rem;
  color: var(--ink-muted);
}
.extreme-speaker { color: var(--ink-faint); }
.insights-method { margin-top: 1.5rem; font-size: 0.92rem; }

"""

if not HOW_STRIP_RISE:
    # Neutralize the rise animation so the how-strip appears immediately.
    CSS += (
        "\n/* how-strip rise override (HOW_STRIP_RISE=False) */\n"
        ".how-strip { animation: none; }\n"
    )

JS = """\
/* ─────────────────────────────────────────────────────────────────────
   truthbot.js — Truthy McTruthface state machine + Web Audio droid sounds

   Reads two attributes from #truthy-mascot-widget:
     data-mood        : 'happy' | 'iffy' | 'sad'   (computed by the pipeline)
     data-claim-count : integer; if 1, uses singular "that" wording,
                        otherwise uses aggregate "this" wording

   Updates #truthy-bubble text to match mood + count.
   Click (or Enter/Space when focused) plays the appropriate droid sound.

   No dependencies. Safe to load at the bottom of <body> or in <head>
   (DOMContentLoaded wrapper handles either case).
   ───────────────────────────────────────────────────────────────────── */

(function() {
  'use strict';

  function init() {
    var mascot       = document.getElementById('mascot');
    var widget       = document.getElementById('truthy-mascot-widget');
    if (!mascot || !widget) return;  // graceful no-op if Truthy isn't on this page

    var led          = document.getElementById('led');
    var ledHalo      = document.getElementById('ledHalo');
    var eyeLeftGroup = document.getElementById('eyeLeftGroup');
    var eyeRightGroup= document.getElementById('eyeRightGroup');
    var headGroup    = document.getElementById('headGroup');
    var bodyGroup    = document.getElementById('bodyGroup');
    var armLeftSwing = document.getElementById('armLeftSwing');
    var armRightSwing= document.getElementById('armRightSwing');
    var clipboard    = document.getElementById('clipboard');
    var bubble       = document.getElementById('truthy-bubble');

    /* ─── Captions: claim-count-aware ─── */
    var captionsSingle = {
      true: "That checks out. Sources match!",
      iffy: "Hmm… let me double-check my sources.",
      lie:  "Oh no… that isn't true."
    };
    var captionsMulti = {
      true: "All sources check out. Looking good!",
      iffy: "Mixed signals — some hold up, some don't.",
      lie:  "Oh no… most of this doesn't check out."
    };
    function getCaption(state, count) {
      return (count === 1 ? captionsSingle : captionsMulti)[state] || "";
    }
    var bubbleClassMap = { true: 'is-true', iffy: 'is-iffy', lie: 'is-lie' };

    var claimCount = parseInt(widget.getAttribute('data-claim-count'), 10);
    if (isNaN(claimCount)) claimCount = 0;  // 0 → uses multi-claim phrasing

    /* ─── State setter ─── */
    function setState(state) {
      mascot.classList.remove('state-true', 'state-iffy', 'state-lie');
      mascot.classList.add('state-' + state);

      if (bubble) {
        bubble.textContent = getCaption(state, claimCount);
        bubble.classList.remove('is-true', 'is-iffy', 'is-lie');
        bubble.classList.add(bubbleClassMap[state]);
      }

      if (state === 'true') {
        led.setAttribute('fill', 'url(#ledGradTrue)');
        ledHalo.setAttribute('fill', '#5ac075');
        eyeLeftGroup.setAttribute('transform', 'translate(115 154) rotate(0)');
        eyeRightGroup.setAttribute('transform', 'translate(185 154) rotate(0)');
        headGroup.setAttribute('transform', 'translate(0,0)');
        bodyGroup.setAttribute('transform', 'translate(0,0)');
        armLeftSwing.setAttribute('transform', 'rotate(135 88 253)');
        armRightSwing.setAttribute('transform', 'rotate(-135 212 253)');
        if (clipboard) clipboard.setAttribute('transform', 'translate(228 218) rotate(-8)');
      } else if (state === 'iffy') {
        led.setAttribute('fill', 'url(#ledGradIffy)');
        ledHalo.setAttribute('fill', '#e8b850');
        eyeLeftGroup.setAttribute('transform', 'translate(115 156) rotate(-10)');
        eyeRightGroup.setAttribute('transform', 'translate(185 156) rotate(10)');
        headGroup.setAttribute('transform', 'rotate(-7 150 170)');
        bodyGroup.setAttribute('transform', 'translate(0,0)');
        armLeftSwing.setAttribute('transform', 'rotate(0 88 253)');
        armRightSwing.setAttribute('transform', 'rotate(-110 212 253)');
        if (clipboard) clipboard.setAttribute('transform', 'translate(238 224) rotate(-3)');
      } else if (state === 'lie') {
        led.setAttribute('fill', 'url(#ledGradLie)');
        ledHalo.setAttribute('fill', '#5a8ec0');
        eyeLeftGroup.setAttribute('transform', 'translate(115 170) rotate(0)');
        eyeRightGroup.setAttribute('transform', 'translate(185 170) rotate(0)');
        headGroup.setAttribute('transform', 'translate(0,7)');
        bodyGroup.setAttribute('transform', 'translate(0,3)');
        armLeftSwing.setAttribute('transform', 'rotate(8 88 253)');
        armRightSwing.setAttribute('transform', 'rotate(35 212 253)');
        if (clipboard) clipboard.setAttribute('transform', 'translate(174 298) rotate(40)');
      }
    }

    /* ─── Idle blink scheduler ─── */
    function doBlink() {
      mascot.classList.add('blinking');
      setTimeout(function() { mascot.classList.remove('blinking'); }, 110);
    }
    function scheduleBlink() {
      var d = 2500 + Math.random() * 4500;
      setTimeout(function() {
        doBlink();
        if (Math.random() < 0.2) setTimeout(doBlink, 280);  // 20% chance of double-blink
        scheduleBlink();
      }, d);
    }
    scheduleBlink();

    /* ─── Web Audio droid sounds ─────────────────────────────────────
       Synthesized via Web Audio API. No audio files needed,
       no licensing, no network round-trips. All sounds resolve in
       <500ms.

       Autoplay-policy contract: browsers (especially Safari) leave a
       freshly-created AudioContext in ``suspended`` until a user
       gesture explicitly resumes it. ``audioCtx.resume()`` returns
       a Promise. The earlier implementation called resume() and
       *immediately* scheduled oscillators against ``ctx.currentTime``
       — on Safari and some Chrome variants the context was still
       suspended at schedule time, so the oscillator silently
       no-op'd. The fix: ``unlockAudio()`` returns a Promise, and the
       play functions are only invoked after that Promise resolves.
       ──────────────────────────────────────────────────────────── */
    var audioCtx = null;
    function unlockAudio() {
      if (!audioCtx) {
        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) { return Promise.resolve(null); }
      }
      if (audioCtx.state === 'suspended') {
        var p = audioCtx.resume();
        // Some old Safari versions return undefined from resume().
        if (p && typeof p.then === 'function') {
          return p.then(function() { return audioCtx; },
                        function() { return audioCtx; });
        }
      }
      return Promise.resolve(audioCtx);
    }

    // Happy: bright rising arpeggio (C5 → E5 → G5 → C6) with square wave
    function playHappy(ctx) {
      var notes = [523.25, 659.25, 783.99, 1046.50];
      notes.forEach(function(freq, i) {
        var t0 = ctx.currentTime + i * 0.07;
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'square';
        osc.frequency.setValueAtTime(freq, t0);
        gain.gain.setValueAtTime(0, t0);
        gain.gain.linearRampToValueAtTime(0.12, t0 + 0.01);
        gain.gain.linearRampToValueAtTime(0, t0 + 0.10);
        osc.connect(gain).connect(ctx.destination);
        osc.start(t0);
        osc.stop(t0 + 0.12);
      });
    }

    // Confused: triangle wave bending up to ~620Hz then dropping to ~330Hz
    function playConfused(ctx) {
      var t0 = ctx.currentTime;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(440, t0);
      osc.frequency.exponentialRampToValueAtTime(620, t0 + 0.18);
      osc.frequency.exponentialRampToValueAtTime(330, t0 + 0.42);
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(0.14, t0 + 0.02);
      gain.gain.linearRampToValueAtTime(0, t0 + 0.45);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t0);
      osc.stop(t0 + 0.5);
    }

    // Sad: descending minor third (G4 → Eb4) with downward pitch bend on each note
    function playSad(ctx) {
      var notes = [392.00, 311.13];
      notes.forEach(function(freq, i) {
        var t0 = ctx.currentTime + i * 0.20;
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, t0);
        osc.frequency.linearRampToValueAtTime(freq * 0.93, t0 + 0.25);
        gain.gain.setValueAtTime(0, t0);
        gain.gain.linearRampToValueAtTime(0.15, t0 + 0.03);
        gain.gain.linearRampToValueAtTime(0, t0 + 0.28);
        osc.connect(gain).connect(ctx.destination);
        osc.start(t0);
        osc.stop(t0 + 0.32);
      });
    }

    var soundMap = { true: playHappy, iffy: playConfused, lie: playSad };

    /* ─── Speak handler ──────────────────────────────────────────────
       Awaits the AudioContext unlock Promise before scheduling
       oscillators. Browsers that silently dropped the prior
       fire-and-forget pattern now actually emit sound.
       ──────────────────────────────────────────────────────────── */
    function speak() {
      var match = mascot.className.match(/state-(true|iffy|lie)/);
      if (!match) return;
      var state = match[1];
      var fn = soundMap[state];
      if (!fn) return;
      mascot.classList.add('speaking');
      setTimeout(function() { mascot.classList.remove('speaking'); }, 700);
      unlockAudio().then(function(ctx) {
        if (!ctx) return;
        /* Defer oscillator scheduling one microtask so ``resume()``'s
           state transition has flushed on Safari / some Chrome builds. */
        queueMicrotask(function() { fn(ctx); });
      });
    }

    /* ─── Initialize ─── */
    var mood = widget.getAttribute('data-mood') || 'iffy';
    var stateMap = { happy: 'true', iffy: 'iffy', sad: 'lie' };
    setState(stateMap[mood] || 'iffy');

    /* ─── Site-wide mute state + queued first-gesture autoplay ─────
       Default: ``mute === 'off'`` (sound enabled). On report and
       index pages we attempt a one-shot mood sound on the user's
       first interaction with the page (browser autoplay policies
       block AudioContext.start() until a gesture). On the dedicated
       Truthy fun page we keep the legacy "tap = always plays"
       behavior so the page stays a playground.

       Persistence: localStorage["truthy-mute"] in {"on", "off"}.
       ─────────────────────────────────────────────────────────── */
    var TRUTHY_MUTE_KEY = 'truthy-mute';
    var DEFAULT_TRUTHY_MUTE = 'off';
    var path = (window.location && window.location.pathname) || '';
    /* The dedicated Truthy fun page keeps the legacy "tap always plays"
       behavior; everywhere else uses the mute toggle. Detection is by
       URL path substring so query strings / hashes don't trip it up. */
    var isTruthyFunPage = path.indexOf('truthy.html') !== -1;

    function readMute() {
      try {
        var v = localStorage.getItem(TRUTHY_MUTE_KEY);
        return (v === 'on' || v === 'off') ? v : DEFAULT_TRUTHY_MUTE;
      } catch (e) { return DEFAULT_TRUTHY_MUTE; }
    }
    function writeMute(v) {
      try { localStorage.setItem(TRUTHY_MUTE_KEY, v); } catch (e) { /* ignore */ }
    }

    var tapHintLabel = widget.querySelector('.tap-hint-label');
    function updateTapHintLabel(mute) {
      if (!tapHintLabel) return;
      if (isTruthyFunPage) {
        tapHintLabel.textContent = 'Tap';
      } else if (mute === 'on') {
        tapHintLabel.textContent = 'Muted';
      } else {
        tapHintLabel.textContent = 'Tap to mute';
      }
    }
    if (tapHintLabel) widget.setAttribute('data-mute', isTruthyFunPage ? 'na' : readMute());
    updateTapHintLabel(readMute());

    /* Queued first-gesture autoplay. Suppressed on the fun page
       (legacy behavior). Removed if the user explicitly taps the
       mascot before any other gesture (taking explicit control of
       the mute toggle should not also fire the queued play).

       ``pointerdown`` fires *before* the subsequent ``click``, which
       matters when the user's first gesture is on a navigation link:
       click navigates the page away, while pointerdown gives the
       AudioContext unlock + oscillator schedule a head start. */
    var queuedHandler = null;
    var QUEUE_EVENTS = ['pointerdown', 'click', 'keydown', 'touchstart'];
    function removeQueued() {
      if (!queuedHandler) return;
      QUEUE_EVENTS.forEach(function(evt) {
        document.removeEventListener(evt, queuedHandler, true);
      });
      queuedHandler = null;
    }
    function setupQueuedAutoplay() {
      if (isTruthyFunPage) return;
      if (readMute() === 'on') return;
      queuedHandler = function() { removeQueued(); speak(); };
      QUEUE_EVENTS.forEach(function(evt) {
        document.addEventListener(evt, queuedHandler, true);
      });
    }
    setupQueuedAutoplay();

    function onMascotActivate(e) {
      if (isTruthyFunPage) {
        speak();
        return;
      }
      /* User explicitly took control before any queued autoplay
         could fire — cancel it so the click only does the mute
         toggle, not also a play. */
      removeQueued();
      if (e && e.stopPropagation) e.stopPropagation();
      var current = readMute();
      var next = (current === 'on') ? 'off' : 'on';
      writeMute(next);
      widget.setAttribute('data-mute', next);
      updateTapHintLabel(next);
      if (next === 'off') speak();  // unmuting always plays once
    }

    widget.addEventListener('click', onMascotActivate);
    widget.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onMascotActivate(e);
      }
    });
  }

  // Run init immediately if DOM is already parsed; otherwise wait
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ─────────────────────────────────────────────────────────────────────
   Editorial-lens toggle — flips every Truthy-scale display between the
   Strict (default since 2026-04-30) and Lenient 5-bucket coarse-axis
   projections.

   Two render patterns are toggled together so the page never goes
   internally inconsistent (e.g. headline says "Mostly Truthy" while the
   verdict bar still shows the Strict aggregate):

   1) PER-PILL SWAP — in-place text+class rewrite on individual pills.
      Used by the per-claim headline pill (claim card) and the
      per-claim TOC mini-pill on report pages. Both wear ``.lens-pill``
      and carry the data-coarse-{lenient,strict} attribute pair.

   2) PAIRED-AXIS SWAP — show/hide complementary blocks pre-rendered
      server-side. Used by aggregate views: the verdict-panel headline
      + ratio + bar, the per-report cards on the index, and any future
      lens-aware aggregate. Each block wears ``[data-lens-axis="X"]``
      and the toggle simply flips the ``hidden`` attribute.

   The per-model strip pills (Anthropic / OpenAI / Gemini / xAI) are
   NEVER touched — they keep the 6-bucket fine labels for audit.

   Body data attribute ``document.body.dataset.lens`` is also set so
   any lens-aware CSS rule can react.

   Persistence: ``localStorage.editorial-lens`` ∈ {"lenient","strict"}.
   Default: strict (2026-04-30 editorial flip from Lenient — Strict
   tracks more closely with the reference set per FitnessScorer Run 5
   and stays the conservative default for non-JS clients). Stored
   user preference still wins on revisit.
   No-op if the page has nothing toggleable (e.g. about, 404).
   ───────────────────────────────────────────────────────────────────── */
(function() {
  'use strict';

  var STORAGE_KEY = 'editorial-lens';
  var DEFAULT_LENS = 'strict';
  var ALL_PILL_CSS_CLASSES = [
    'v-true', 'v-mostly-true', 'v-exaggerated', 'v-misleading',
    'v-false', 'v-unverifiable', 'v-truthy', 'v-falsey', 'v-split'
  ];

  function readLens() {
    try {
      var v = localStorage.getItem(STORAGE_KEY);
      return (v === 'strict' || v === 'lenient') ? v : DEFAULT_LENS;
    } catch (e) {
      return DEFAULT_LENS;
    }
  }

  function writeLens(lens) {
    try { localStorage.setItem(STORAGE_KEY, lens); } catch (e) { /* ignore */ }
  }

  function applyLensToPill(pill, lens) {
    var label, cssSlug;
    if (lens === 'strict') {
      label = pill.getAttribute('data-coarse-strict') || pill.getAttribute('data-fine-label') || '';
      cssSlug = pill.getAttribute('data-coarse-strict-css') || pill.getAttribute('data-fine-css') || 'unverifiable';
    } else {
      label = pill.getAttribute('data-coarse-lenient') || pill.getAttribute('data-fine-label') || '';
      cssSlug = pill.getAttribute('data-coarse-lenient-css') || pill.getAttribute('data-fine-css') || 'unverifiable';
    }
    if (!label) return;
    pill.textContent = label;
    for (var i = 0; i < ALL_PILL_CSS_CLASSES.length; i++) {
      pill.classList.remove(ALL_PILL_CSS_CLASSES[i]);
    }
    pill.classList.add('v-' + cssSlug);
  }

  function applyLensToAxisPairs(lens) {
    /* Show the block tagged with the active lens, hide the other.
       Idempotent — safe to call repeatedly. */
    var blocks = document.querySelectorAll('[data-lens-axis]');
    for (var i = 0; i < blocks.length; i++) {
      var axis = blocks[i].getAttribute('data-lens-axis');
      if (axis === lens) {
        blocks[i].hidden = false;
      } else {
        blocks[i].hidden = true;
      }
    }
  }

  function applyLens(lens) {
    /* 1) per-pill text+class swap (headline pill + TOC pill) */
    var pills = document.querySelectorAll('.lens-pill');
    for (var i = 0; i < pills.length; i++) {
      applyLensToPill(pills[i], lens);
    }
    /* 2) paired-axis show/hide for aggregate displays */
    applyLensToAxisPairs(lens);
    /* 3) body data-attr so any lens-aware CSS rule can react */
    if (document.body) document.body.setAttribute('data-lens', lens);
    /* 4) chip state */
    var chip = document.querySelector('.editorial-lens');
    if (chip) {
      chip.setAttribute('data-lens', lens);
      var valEl = chip.querySelector('.lens-value');
      if (valEl) valEl.textContent = (lens === 'strict') ? 'Strict' : 'Lenient';
      chip.setAttribute('aria-pressed', lens === 'strict' ? 'true' : 'false');
    }
  }

  function init() {
    var pills = document.querySelectorAll('.lens-pill');
    var axisBlocks = document.querySelectorAll('[data-lens-axis]');
    var chip = document.querySelector('.editorial-lens');
    var hasToggleableContent = pills.length > 0 || axisBlocks.length > 0;
    if (!hasToggleableContent) {
      if (chip) chip.hidden = true;
      return;
    }
    var lens = readLens();
    applyLens(lens);
    if (chip) {
      chip.hidden = false;
      chip.addEventListener('click', function() {
        var current = chip.getAttribute('data-lens') || DEFAULT_LENS;
        var next = (current === 'lenient') ? 'strict' : 'lenient';
        writeLens(next);
        applyLens(next);
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ── E-id jump links: reveal targets hidden inside collapsed <details> ──
   Reasoning cites pack ids as anchors into "Sources consulted", which is
   collapsed by default; plain fragment navigation won't open a closed
   <details>, so open every ancestor before the browser scrolls. */
(function() {
  'use strict';
  function revealHashTarget() {
    var id = location.hash && location.hash.slice(1);
    var el = id && document.getElementById(id);
    if (!el || !el.closest) return;
    var d = el.closest('details');
    while (d) {
      d.open = true;
      d = d.parentElement ? d.parentElement.closest('details') : null;
    }
    el.scrollIntoView({block: 'center'});
  }
  window.addEventListener('hashchange', revealHashTarget);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', revealHashTarget);
  } else {
    revealHashTarget();
  }
})();
"""

# ── Page renderers ──────────────────────────────────────────────────────────
from collections import defaultdict as _defaultdict


def _disambiguate_report_urls(reports: list[dict]) -> list[dict]:
    """
    Ensure every report card in the index links to a unique URL.

    Old entries in reports.json may have been stored without the 6-char
    run-id suffix (before the slug was updated).  Group by base slug
    (stripping any trailing -[0-9a-f]{6}); if a group has >1 member,
    EVERY member gets its url rewritten to include the first 6 chars of
    its stored report id.  Single-member groups are left untouched so
    existing links keep working.

    The hash is derived from the report's id (stable across re-runs).
    """
    def _base(r: dict) -> str:
        name = r.get("url", "").split("/")[-1].removesuffix(".html")
        return re.sub(r"-[0-9a-f]{6}$", "", name)

    groups: dict[str, list] = _defaultdict(list)
    for r in reports:
        groups[_base(r)].append(r)

    result: list[dict] = []
    for r in reports:
        base = _base(r)
        if len(groups[base]) > 1:
            short = r.get("id", "")[:6]
            r = {**r, "url": f"reports/{base}-{short}.html"}
        result.append(r)
    return result



# ── Index page hero animation script ────────────────────────────────────────


_HERO_SCRIPT = r"""<script>
(function(){
  var mascot = document.getElementById('mascot');
  var bubble = document.getElementById('hero-bubble');
  if (!mascot || !bubble) return;
  var led = document.getElementById('led');
  var ledHalo = document.getElementById('ledHalo');
  var eyeL = document.getElementById('eyeLeftGroup');
  var eyeR = document.getElementById('eyeRightGroup');
  var head = document.getElementById('headGroup');
  var body = document.getElementById('bodyGroup');
  var armL = document.getElementById('armLeftSwing');
  var armR = document.getElementById('armRightSwing');
  var clipboard = document.getElementById('clipboard');
  var STEP_MS = 3000;
  var steps = [
    { state: 'true', cls: 'is-true', text: "I'm Truthy and honesty makes me happy!", dur: STEP_MS },
    { state: 'iffy', cls: 'is-iffy', text: "I'll evaluate all claims thoroughly.", dur: STEP_MS },
    { state: 'lie',  cls: 'is-lie',  text: "Lies make me very sad.", dur: STEP_MS }
  ];
  function pose(state) {
    mascot.classList.remove('state-true', 'state-iffy', 'state-lie');
    mascot.classList.add('state-' + state);
    mascot.classList.remove('hero-wave');
    if (state === 'true') {
      if (led) led.setAttribute('fill', 'url(#ledGradTrue)');
      if (ledHalo) ledHalo.setAttribute('fill', '#5ac075');
      if (eyeL) eyeL.setAttribute('transform', 'translate(115 154) rotate(0)');
      if (eyeR) eyeR.setAttribute('transform', 'translate(185 154) rotate(0)');
      if (head) head.setAttribute('transform', 'translate(0,0)');
      if (body) body.setAttribute('transform', 'translate(0,0)');
      mascot.classList.add('hero-wave');
      if (armL) armL.setAttribute('transform', 'rotate(135 88 253)');
      if (armR) armR.setAttribute('transform', 'rotate(-135 212 253)');
      if (clipboard) clipboard.setAttribute('transform', 'translate(228 218) rotate(-8)');
    } else if (state === 'iffy') {
      if (led) led.setAttribute('fill', 'url(#ledGradIffy)');
      if (ledHalo) ledHalo.setAttribute('fill', '#e8b850');
      if (eyeL) eyeL.setAttribute('transform', 'translate(115 156) rotate(-10)');
      if (eyeR) eyeR.setAttribute('transform', 'translate(185 156) rotate(10)');
      if (head) head.setAttribute('transform', 'rotate(-7 150 170)');
      if (body) body.setAttribute('transform', 'translate(0,0)');
      if (armL) armL.setAttribute('transform', 'rotate(0 88 253)');
      if (armR) armR.setAttribute('transform', 'rotate(-110 212 253)');
      if (clipboard) clipboard.setAttribute('transform', 'translate(238 224) rotate(-3)');
    } else {
      if (led) led.setAttribute('fill', 'url(#ledGradLie)');
      if (ledHalo) ledHalo.setAttribute('fill', '#5a8ec0');
      if (eyeL) eyeL.setAttribute('transform', 'translate(115 170) rotate(0)');
      if (eyeR) eyeR.setAttribute('transform', 'translate(185 170) rotate(0)');
      if (head) head.setAttribute('transform', 'translate(0,7)');
      if (body) body.setAttribute('transform', 'translate(0,3)');
      if (armL) armL.setAttribute('transform', 'rotate(8 88 253)');
      if (armR) armR.setAttribute('transform', 'rotate(35 212 253)');
      if (clipboard) clipboard.setAttribute('transform', 'translate(174 298) rotate(40)');
    }
  }
  function showStep(step) {
    pose(step.state);
    bubble.classList.remove('is-true', 'is-iffy', 'is-lie');
    bubble.classList.add(step.cls);
    bubble.textContent = step.text;
    bubble.style.opacity = '1';
    bubble.style.transform = 'translateY(0)';
    if (step.state === 'iffy') {
      setTimeout(blinkOnce, 600);
    }
  }
  function cycle(idx) {
    showStep(steps[idx]);
    var dur = steps[idx].dur || STEP_MS;
    var next = (idx + 1) % steps.length;
    setTimeout(function(){ cycle(next); }, dur);
  }
  function blinkOnce() {
    mascot.classList.add('blinking');
    setTimeout(function() { mascot.classList.remove('blinking'); }, 110);
  }
  (function scheduleBlink() {
    setTimeout(function() {
      blinkOnce();
      if (Math.random() < 0.2) setTimeout(blinkOnce, 280);
      scheduleBlink();
    }, 2500 + Math.random() * 4500);
  })();
  setTimeout(function() { cycle(0); }, 400);
})();
</script>
"""

# Truthy fun page: same hero cycle as index + Web Audio (first click on toggle unlocks; mirrors truthbot.js).
_TRUTHY_FUN_SCRIPT = r"""<script>
(function(){
  var mascot = document.getElementById('mascot');
  var bubble = document.getElementById('hero-bubble');
  if (!mascot || !bubble) return;
  var led = document.getElementById('led');
  var ledHalo = document.getElementById('ledHalo');
  var eyeL = document.getElementById('eyeLeftGroup');
  var eyeR = document.getElementById('eyeRightGroup');
  var head = document.getElementById('headGroup');
  var body = document.getElementById('bodyGroup');
  var armL = document.getElementById('armLeftSwing');
  var armR = document.getElementById('armRightSwing');
  var clipboard = document.getElementById('clipboard');

  var audioCtx = null;
  function getCtx() {
    if (!audioCtx) {
      try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      } catch (e) { return null; }
    }
    return audioCtx;
  }
  function playHappy() {
    var ctx = getCtx(); if (!ctx) return;
    var notes = [523.25, 659.25, 783.99, 1046.50];
    notes.forEach(function(freq, i) {
      var t0 = ctx.currentTime + i * 0.07;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(freq, t0);
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(0.12, t0 + 0.01);
      gain.gain.linearRampToValueAtTime(0, t0 + 0.10);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t0);
      osc.stop(t0 + 0.12);
    });
  }
  function playConfused() {
    var ctx = getCtx(); if (!ctx) return;
    var t0 = ctx.currentTime;
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(440, t0);
    osc.frequency.exponentialRampToValueAtTime(620, t0 + 0.18);
    osc.frequency.exponentialRampToValueAtTime(330, t0 + 0.42);
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(0.14, t0 + 0.02);
    gain.gain.linearRampToValueAtTime(0, t0 + 0.45);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t0);
    osc.stop(t0 + 0.5);
  }
  function playSad() {
    var ctx = getCtx(); if (!ctx) return;
    var notes = [392.00, 311.13];
    notes.forEach(function(freq, i) {
      var t0 = ctx.currentTime + i * 0.20;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t0);
      osc.frequency.linearRampToValueAtTime(freq * 0.93, t0 + 0.25);
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(0.15, t0 + 0.03);
      gain.gain.linearRampToValueAtTime(0, t0 + 0.28);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t0);
      osc.stop(t0 + 0.32);
    });
  }
  var soundMap = { true: playHappy, iffy: playConfused, lie: playSad };

  var btn = document.getElementById('truthy-sound-toggle');
  var audioUnlocked = false;
  var soundsOn = false;

  function refreshToggleUi() {
    if (!btn) return;
    var muted = !audioUnlocked || !soundsOn;
    btn.classList.toggle('is-muted', muted);
    btn.setAttribute('aria-pressed', (!muted).toString());
    if (!audioUnlocked) {
      btn.setAttribute('aria-label', 'Turn on droid sounds');
    } else if (soundsOn) {
      btn.setAttribute('aria-label', 'Mute');
    } else {
      btn.setAttribute('aria-label', 'Unmute');
    }
  }
  function unlockAudio() {
    if (audioUnlocked) return;
    var c = getCtx();
    if (!c) return;
    if (c.state === 'suspended') c.resume();
    audioUnlocked = true;
    soundsOn = true;
    refreshToggleUi();
  }
  if (btn) {
    btn.addEventListener('click', function() {
      if (!audioUnlocked) {
        unlockAudio();
        return;
      }
      soundsOn = !soundsOn;
      refreshToggleUi();
    });
  }
  refreshToggleUi();

  var STEP_MS = 3000;
  var steps = [
    { state: 'true', cls: 'is-true', text: "I'm Truthy and honesty makes me happy!", dur: STEP_MS },
    { state: 'iffy', cls: 'is-iffy', text: "I'll evaluate all claims thoroughly.", dur: STEP_MS },
    { state: 'lie',  cls: 'is-lie',  text: "Lies make me very sad.", dur: STEP_MS }
  ];
  function pose(state) {
    mascot.classList.remove('state-true', 'state-iffy', 'state-lie');
    mascot.classList.add('state-' + state);
    mascot.classList.remove('hero-wave');
    if (state === 'true') {
      if (led) led.setAttribute('fill', 'url(#ledGradTrue)');
      if (ledHalo) ledHalo.setAttribute('fill', '#5ac075');
      if (eyeL) eyeL.setAttribute('transform', 'translate(115 154) rotate(0)');
      if (eyeR) eyeR.setAttribute('transform', 'translate(185 154) rotate(0)');
      if (head) head.setAttribute('transform', 'translate(0,0)');
      if (body) body.setAttribute('transform', 'translate(0,0)');
      mascot.classList.add('hero-wave');
      if (armL) armL.setAttribute('transform', 'rotate(135 88 253)');
      if (armR) armR.setAttribute('transform', 'rotate(-135 212 253)');
      if (clipboard) clipboard.setAttribute('transform', 'translate(228 218) rotate(-8)');
    } else if (state === 'iffy') {
      if (led) led.setAttribute('fill', 'url(#ledGradIffy)');
      if (ledHalo) ledHalo.setAttribute('fill', '#e8b850');
      if (eyeL) eyeL.setAttribute('transform', 'translate(115 156) rotate(-10)');
      if (eyeR) eyeR.setAttribute('transform', 'translate(185 156) rotate(10)');
      if (head) head.setAttribute('transform', 'rotate(-7 150 170)');
      if (body) body.setAttribute('transform', 'translate(0,0)');
      if (armL) armL.setAttribute('transform', 'rotate(0 88 253)');
      if (armR) armR.setAttribute('transform', 'rotate(-110 212 253)');
      if (clipboard) clipboard.setAttribute('transform', 'translate(238 224) rotate(-3)');
    } else {
      if (led) led.setAttribute('fill', 'url(#ledGradLie)');
      if (ledHalo) ledHalo.setAttribute('fill', '#5a8ec0');
      if (eyeL) eyeL.setAttribute('transform', 'translate(115 170) rotate(0)');
      if (eyeR) eyeR.setAttribute('transform', 'translate(185 170) rotate(0)');
      if (head) head.setAttribute('transform', 'translate(0,7)');
      if (body) body.setAttribute('transform', 'translate(0,3)');
      if (armL) armL.setAttribute('transform', 'rotate(8 88 253)');
      if (armR) armR.setAttribute('transform', 'rotate(35 212 253)');
      if (clipboard) clipboard.setAttribute('transform', 'translate(174 298) rotate(40)');
    }
  }
  function showStep(step) {
    pose(step.state);
    bubble.classList.remove('is-true', 'is-iffy', 'is-lie');
    bubble.classList.add(step.cls);
    bubble.textContent = step.text;
    bubble.style.opacity = '1';
    bubble.style.transform = 'translateY(0)';
    if (audioUnlocked && soundsOn) {
      var fn = soundMap[step.state];
      if (fn) fn();
    }
    if (step.state === 'iffy') {
      setTimeout(blinkOnce, 600);
    }
  }
  function cycle(idx) {
    showStep(steps[idx]);
    var dur = steps[idx].dur || STEP_MS;
    var next = (idx + 1) % steps.length;
    setTimeout(function(){ cycle(next); }, dur);
  }
  function blinkOnce() {
    mascot.classList.add('blinking');
    setTimeout(function() { mascot.classList.remove('blinking'); }, 110);
  }
  (function scheduleBlink() {
    setTimeout(function() {
      blinkOnce();
      if (Math.random() < 0.2) setTimeout(blinkOnce, 280);
      scheduleBlink();
    }, 2500 + Math.random() * 4500);
  })();
  setTimeout(function() { cycle(0); }, 400);
})();
</script>
"""

# ── Shared icon bodies (monochrome, currentColor only) ────────────────────
# Kept in sync with src/truthbot/publish/assets/icons/*.svg. Bodies only — no
# outer <svg> wrapper — so they can be sized/classed per context via _icon_svg.
_ICON_BODY_LEADERS = (
    '<circle cx="12" cy="5.5" r="3" fill="currentColor"/>'
    '<path d="M8 10.5c0-1 1.2-2 4-2s4 1 4 2v1.5H8z" fill="currentColor"/>'
    '<rect x="4" y="14" width="16" height="2" rx="0.5" fill="currentColor" opacity="0.7"/>'
    '<rect x="6" y="17" width="12" height="5" rx="0.8" fill="currentColor" opacity="0.4"/>'
    '<rect x="10" y="16" width="4" height="6.5" rx="0.5" fill="currentColor" opacity="0.55"/>'
)
_ICON_BODY_CLAIMS = (
    '<path d="M4 4h12a2 2 0 012 2v7a2 2 0 01-2 2h-3.5l-3 3v-3H4a2 2 0 01-2-2V6a2 2 0 012-2z" '
    'fill="currentColor" opacity="0.35"/>'
    '<line x1="5" y1="8" x2="13" y2="8" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
    '<line x1="5" y1="11" x2="10" y2="11" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
    '<circle cx="17" cy="15" r="4" stroke="currentColor" stroke-width="1.8" fill="none"/>'
    '<line x1="20" y1="18" x2="22.5" y2="20.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
)
# Three bot heads in a row (see assets/icons/icon-models-engaged.svg).
_ICON_BODY_MODELS_ENGAGED = (
    '<line x1="4.5" y1="5.5" x2="4.5" y2="7" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>'
    '<circle cx="4.5" cy="5" r="0.8" fill="currentColor"/>'
    '<rect x="1.5" y="7.5" width="6" height="5" rx="1.8" fill="currentColor" opacity="0.25"/>'
    '<rect x="2" y="9.5" width="5" height="2" rx="1" fill="currentColor" opacity="0.65"/>'
    '<circle cx="3.5" cy="10.5" r="0.8" fill="currentColor"/>'
    '<circle cx="5.5" cy="10.5" r="0.8" fill="currentColor"/>'
    '<line x1="12" y1="5.5" x2="12" y2="7" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>'
    '<circle cx="12" cy="5" r="0.8" fill="currentColor"/>'
    '<rect x="9" y="7.5" width="6" height="5" rx="1.8" fill="currentColor" opacity="0.25"/>'
    '<rect x="9.5" y="9.5" width="5" height="2" rx="1" fill="currentColor" opacity="0.65"/>'
    '<circle cx="11" cy="10.5" r="0.8" fill="currentColor"/>'
    '<circle cx="13" cy="10.5" r="0.8" fill="currentColor"/>'
    '<line x1="19.5" y1="5.5" x2="19.5" y2="7" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>'
    '<circle cx="19.5" cy="5" r="0.8" fill="currentColor"/>'
    '<rect x="16.5" y="7.5" width="6" height="5" rx="1.8" fill="currentColor" opacity="0.25"/>'
    '<rect x="17" y="9.5" width="5" height="2" rx="1" fill="currentColor" opacity="0.65"/>'
    '<circle cx="18.5" cy="10.5" r="0.8" fill="currentColor"/>'
    '<circle cx="20.5" cy="10.5" r="0.8" fill="currentColor"/>'
    '<line x1="2" y1="14" x2="22" y2="14" stroke="currentColor" stroke-width="0.5" opacity="0.2"/>'
)
# Bots converging to checkmark (see assets/icons/icon-model-consensus.svg).
# Check-mark inside a circle. Used by the "Truthy or better" headline
# frame above the verdict panel's aggregate stats grid. Matches the
# existing 24x24 grid the other ``_ICON_BODY_*`` constants use.
_ICON_BODY_TRUTHY_RATE = (
    '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5" '
    'fill="currentColor" fill-opacity="0.12"/>'
    '<path d="M 7.5 12.5 L 11 16 L 17 8.5" stroke="currentColor" stroke-width="2" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
)


# X-mark inside a circle — the negative counterpart to TRUTHY_RATE,
# used by the "False or worse" headline frame.
_ICON_BODY_FALSE_RATE = (
    '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5" '
    'fill="currentColor" fill-opacity="0.12"/>'
    '<path d="M 8 8 L 16 16 M 16 8 L 8 16" stroke="currentColor" stroke-width="2" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
)


_ICON_BODY_MODEL_CONSENSUS = (
    '<line x1="4.5" y1="4.5" x2="4.5" y2="6" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>'
    '<circle cx="4.5" cy="4" r="0.8" fill="currentColor"/>'
    '<rect x="1.5" y="6.5" width="6" height="5" rx="1.8" fill="currentColor" opacity="0.25"/>'
    '<rect x="2" y="8.5" width="5" height="2" rx="1" fill="currentColor" opacity="0.65"/>'
    '<circle cx="3.5" cy="9.5" r="0.8" fill="currentColor"/>'
    '<circle cx="5.5" cy="9.5" r="0.8" fill="currentColor"/>'
    '<line x1="12" y1="4.5" x2="12" y2="6" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>'
    '<circle cx="12" cy="4" r="0.8" fill="currentColor"/>'
    '<rect x="9" y="6.5" width="6" height="5" rx="1.8" fill="currentColor" opacity="0.25"/>'
    '<rect x="9.5" y="8.5" width="5" height="2" rx="1" fill="currentColor" opacity="0.65"/>'
    '<circle cx="11" cy="9.5" r="0.8" fill="currentColor"/>'
    '<circle cx="13" cy="9.5" r="0.8" fill="currentColor"/>'
    '<line x1="19.5" y1="4.5" x2="19.5" y2="6" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>'
    '<circle cx="19.5" cy="4" r="0.8" fill="currentColor"/>'
    '<rect x="16.5" y="6.5" width="6" height="5" rx="1.8" fill="currentColor" opacity="0.25"/>'
    '<rect x="17" y="8.5" width="5" height="2" rx="1" fill="currentColor" opacity="0.65"/>'
    '<circle cx="18.5" cy="9.5" r="0.8" fill="currentColor"/>'
    '<circle cx="20.5" cy="9.5" r="0.8" fill="currentColor"/>'
    '<line x1="4.5" y1="12.5" x2="10" y2="17" stroke="currentColor" stroke-width="1" opacity="0.35" '
    'stroke-linecap="round"/>'
    '<line x1="12" y1="12.5" x2="12" y2="17" stroke="currentColor" stroke-width="1" opacity="0.35" '
    'stroke-linecap="round"/>'
    '<line x1="19.5" y1="12.5" x2="14" y2="17" stroke="currentColor" stroke-width="1" opacity="0.35" '
    'stroke-linecap="round"/>'
    '<circle cx="12" cy="18.5" r="3" fill="currentColor" opacity="0.12"/>'
    '<circle cx="12" cy="18.5" r="3" stroke="currentColor" stroke-width="1" fill="none" opacity="0.5"/>'
    '<path d="M10.3 18.5l1.1 1.3 2.2-2.6" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" '
    'stroke-linejoin="round" fill="none"/>'
)


def _icon_svg(body: str, size: int = 28, extra_class: str = "") -> str:
    """Render a monochrome stat-style icon at the requested pixel size.

    The icon always uses the 24x24 viewBox and currentColor — size/class is
    context-dependent (landing hero vs report stats strip vs inline badge).
    """
    cls = "stat-icon" + ((" " + extra_class.strip()) if extra_class else "")
    return (
        '<svg class="' + cls + '" width="' + str(size) + '" height="' + str(size) + '" '
        'viewBox="0 0 24 24" fill="none" aria-hidden="true">' + body + '</svg>'
    )

def _render_index(reports: list[dict], stats: dict) -> str:
    """Render the landing page from the reports index."""
    reports = _disambiguate_report_urls(reports)
    total_claims    = stats.get("total_claims", 0)
    total_leaders   = stats.get("total_leaders", len({r.get("source_of_claims") or r.get("speaker", "") for r in reports}))
    avg_consensus   = stats.get("avg_consensus", stats.get("model_agreement_rate", 0))

    avg_pct = str(round(avg_consensus * 100))

    # Truthy hero — full SVG, state-true by default; inline script drives animation
    hero_html = (
        '<div class="index-hero">'
        '<a class="hero-truthy-link" href="./truthy.html" '
        'aria-label="Meet Truthy McTruthface — fun page">'
        '<div class="hero-truthy-wrap">'
        + _TRUTHY_SVG
        + '</div></a>'
        '<div class="hero-truthy-col">'
        '<div class="truthy-bubble is-true" id="hero-bubble" '
        'style="opacity:1;transition:opacity 100ms ease,transform 100ms ease">'
        "I&rsquo;m Truthy and honesty makes me happy!"
        '</div>'
        '</div>'
        '</div>'
    )

    stats_html = (
        '<div class="section-head"><span>Program stats</span><span class="sub">All time</span></div>'
        '<div class="stats">'
        + '<div class="stat">'
        + _icon_svg(_ICON_BODY_LEADERS, size=48, extra_class="stat-icon-lg")
        + '<div class="num">' + str(total_leaders) + '</div>'
        + '<div class="lbl">Leaders Reviewed</div></div>'
        + '<div class="stat">'
        + _icon_svg(_ICON_BODY_CLAIMS, size=48, extra_class="stat-icon-lg")
        + '<div class="num">' + str(total_claims) + '</div>'
        + '<div class="lbl">Claims Checked</div></div>'
        + '<div class="stat">'
        + _icon_svg(_ICON_BODY_MODEL_CONSENSUS, size=48, extra_class="stat-icon-lg")
        + '<div class="num">' + avg_pct + '<span class="unit">%</span></div>'
        + '<div class="lbl">Model Consensus</div></div>'
        + '</div>'
    )

    # How-it-works trust strip: reassures readers about the pipeline in one glance.
    how_strip_html = (
        '<div class="how-strip">'
        '<div class="how-step">'
        '<span class="how-num">1</span>'
        '<span class="how-text">Speech is decomposed into atomic, verifiable claims</span>'
        '</div>'
        '<div class="how-sep" aria-hidden="true">&rarr;</div>'
        '<div class="how-step">'
        '<span class="how-num">2</span>'
        '<span class="how-text">Each claim is checked by a multi-model panel against a shared, cited evidence pack</span>'
        '</div>'
        '<div class="how-sep" aria-hidden="true">&rarr;</div>'
        '<div class="how-step">'
        '<span class="how-num">3</span>'
        '<span class="how-text">Verdicts are aggregated into a transparent consensus score</span>'
        '</div>'
        '</div>'
    )

    cards_html = '<div class="reports">'
    if reports:
        for r in reports[:20]:
            cards_html += _report_card(r)
    else:
        cards_html += '<p class="dim">No reports yet.</p>'
    cards_html += '</div>'

    # Model-insights strip retired with the vestigial insights page
    # (remediation T0.4) — it summarized a single pseudo-model with 0%
    # dissent by construction. Returns with the Phase 4 per-seat rebuild.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        hero_html
        + stats_html
        + how_strip_html
        + '<hr class="rule">'
        + '<div class="section-head"><span>Latest truthiness reviews</span>'
        + '<span class="sub">Feed</span></div>'
        + cards_html
        + _HERO_SCRIPT
    )
    _phash = _prompt_hash()
    footer = (
        '<span>Last updated: ' + now + '</span>'
        + '<span>Pipeline v' + PIPELINE_VERSION + BETA_BADGE_HTML
        + f' · Prompt <a class="footer-hash" href="./about.html#prompt">{_phash}</a>'
        + ' · <a href="./corrections.html">Corrections</a>'
        + ' · <a href="./model-insights.html">Panel insights</a>'
        + ' · <a href="' + GITHUB_URL + '" target="_blank" rel="noopener">GitHub</a></span>'
    )
    return _page_index(
        "Latest Reports",
        body,
        footer,
        og_title="truth-bot — Automated Political Fact-Checking",
        og_description=(
            "Automated fact-checking of political speeches. Every claim is checked "
            "by a multi-model AI panel against a shared, cited evidence pack — "
            "sources linked inline, disagreements disclosed."
        ),
        og_type="website",
    )


def _render_report(site_report: SiteReport) -> str:
    """Render a full per-speech report page."""
    src_link = ""
    if site_report.transcript_source_url:
        src_link = (
            ' · <a href="' + _esc(site_report.transcript_source_url) +
            '" target="_blank" rel="noopener">Transcript source</a>'
        )

    toc_html = _toc(site_report.checkable_bundles) if len(site_report.checkable_bundles) > 2 else ""

    claim_blocks = "\n".join(
        _claim_card(b, i, len(site_report.checkable_bundles), rel="../",
                    panel_roster=getattr(site_report, "panel_roster", None))
        for i, b in enumerate(site_report.checkable_bundles, 1)
    )

    phash = _prompt_hash()
    gen_ts = site_report.generated_at.strftime("%Y-%m-%d %H:%M UTC")

    claim_count = len(site_report.checkable_bundles)
    toc_section_head = ''
    if toc_html:
        # id="claim-catalog" is the anchor target for per-claim "Back to claim list" links.
        toc_section_head = (
            '<div class="section-head" id="claim-catalog">'
            + '<span class="section-head-label">'
            + _icon_svg(_ICON_BODY_CLAIMS, size=18, extra_class="section-head-icon")
            + '<span>Jump to claim</span>'
            + '</span>'
            + '<span class="sub">' + str(claim_count) + ' claim' + ('s' if claim_count != 1 else '') + ' evaluated</span>'
            + '</div>'
        )

    _model_count, _model_hint = _models_engaged(site_report)
    _roster_seats = (getattr(site_report, "panel_roster", None) or {}).get("seats") or {}
    if _roster_seats:
        _comp = " / ".join(
            f"{role}={', '.join(ms or [])}" for role, ms in _roster_seats.items())
        methodology_html = (
            '<aside class="methodology">'
            '<strong>How this report was generated.</strong> truth-bot extracts factual '
            'claims from the source transcript, screens them for check-worthiness, and '
            'routes each checkable claim to a proposer \u2192 critic \u2192 arbiter panel '
            'of language models (' + _esc(_comp) + '), grounded in a retrieved evidence '
            'pack cited per claim. Label disagreements escalate to the arbiter; a '
            'dedicated Severity Classifier model re-examines the False-vs-Misleading '
            'boundary. Verdicts, seat votes, and sources are shown on every claim page. '
            'Truthy McTruthface\u2019s mood reflects the aggregate score across all claims. '
            '<a href="../about.html">Read the full methodology \u2192</a>'
            '</aside>'
        )
    else:
        _model_word = str(_model_count) + ' frontier language model' + ('s' if _model_count != 1 else '')
        methodology_html = (
            '<aside class="methodology">'
            '<strong>How this report was generated.</strong> truth-bot extracts factual claims '
            'from the source transcript, submits each independently to ' + _model_word + ' '
            'with the instruction to verify against publicly cited sources, and aggregates '
            'verdicts using a simple majority rule. Caveats are surfaced when models flag '
            'ambiguity or framing concerns. Truthy McTruthface\u2019s mood reflects the aggregate '
            'score across all claims. '
            '<a href="../about.html">Read the full methodology \u2192</a>'
            '</aside>'
        )

    # Build hero elements conditionally (omit empty fields per spec)
    # Prefer new decomposed fields; fall back to legacy fields for backward compat.
    _overline = site_report.source_of_claims_professional_public_title or site_report.role
    _name = site_report.source_of_claims or site_report.speaker
    _event = site_report.event  # event name -> speech-title
    _venue = site_report.venue
    _channel = site_report.channel

    _hero_parts = ['<section class="hero" id="top">']
    if _overline:
        _hero_parts.append('<div class="hero-overline">' + _esc(_overline) + '</div>')
    _hero_parts.append('<h1 class="speaker-name">' + _esc(_name) + '</h1>')
    if _event:
        _hero_parts.append('<div class="speech-title">' + _esc(_event) + '</div>')
    _meta_spans: list[str] = []
    if site_report.date:
        _meta_spans.append('<span>' + _esc(site_report.display_date) + '</span>')
    if _venue:
        _meta_spans.append('<span>' + _esc(_venue) + '</span>')
    if _channel:
        _meta_spans.append('<span>' + _esc(_channel) + '</span>')
    if _meta_spans:
        _hero_parts.append('<div class="speech-meta">' + '<span class="sep">&middot;</span>'.join(_meta_spans) + '</div>')
    _hero_parts.append('</section>')
    hero_html = '\n'.join(_hero_parts)

    # Statement Triage cross-link — only when the non-check-worthy stream exists.
    triage_link_html = ""
    if site_report.characterization:
        _n_triage = len(site_report.characterization)
        triage_link_html = (
            '<aside class="methodology">'
            '<strong>Statement Triage.</strong> Of the sentences in this speech, '
            + str(_n_triage) + ' were set aside as non-check-worthy '
            '(pleasantries, opinion, or otherwise unimportant) before fact-checking. '
            '<a href="' + _esc(site_report.triage_slug) + '.html">'
            'See what we set aside and why →</a>'
            '</aside>'
        )

    # Report-level correction banner (D2): derived entirely from the bundles'
    # correction notes — count and latest date come from data, never typed.
    _corrected = [b for b in site_report.checkable_bundles
                  if getattr(b.consensus.provenance, "correction_note", "")]
    correction_banner = ""
    if _corrected:
        _dates = sorted(
            m.group(1) for b in _corrected
            for m in [re.search(r"\((\d{4}-\d{2}-\d{2})\)",
                                b.consensus.provenance.correction_note)] if m)
        _latest = _dates[-1] if _dates else ""
        _n = len(_corrected)
        correction_banner = (
            '<aside class="report-correction-banner">'
            f'<strong>Corrections applied:</strong> {_n} verdict'
            f'{"s" if _n != 1 else ""} on this report '
            f'{"were" if _n != 1 else "was"} revised'
            + (f' on {_esc(_latest)}' if _latest else '')
            + ' following a reasoning audit. Aggregates and the headline reflect '
              'the corrected verdicts; each change is logged on the '
              '<a href="../corrections.html">Corrections page</a> and marked on '
              'the claim\'s provenance strip.</aside>'
        )

    body = (
        hero_html
        + _verdict_panel(site_report)
        + correction_banner
        + toc_section_head
        + toc_html
        + '<div class="section-head">'
        + '<span>Claims, in order spoken</span>'
        + '<span class="sub">Anchor links shareable</span>'
        + '</div>'
        + claim_blocks
        + methodology_html
        + triage_link_html
        + _run_manifest_html(site_report)
        + _panel_composition_html(site_report)
    )
    footer = (
        '<span>truth-bot · pipeline v' + PIPELINE_VERSION + BETA_BADGE_HTML + '</span>'
        + f'<span>Prompt <a class="footer-hash" href="../about.html#prompt">{phash}</a>'
        + ' · <a href="../corrections.html">Corrections</a></span>'
        + '<span>Source: <a href="' + GITHUB_URL + '" target="_blank" rel="noopener">'
        + 'github.com/aRealGem/Truth-bot</a></span>'
    )
    _headline, _ = _headline_verdict(site_report.verdict_distribution)
    _n_claims = len(site_report.checkable_bundles)
    _report_og_desc = (
        f"{_n_claims} claim{'s' if _n_claims != 1 else ''} checked. "
        f"Verdict: {_headline}. "
        "Multi-model AI fact-check with primary source verification."
    )
    _report_og_title = (
        f"{site_report.speaker} — {site_report.display_date} — truth-bot"
    )
    return _page_report(
        _esc(site_report.speaker) + " — " + _esc(site_report.display_date),
        body,
        footer=footer,
        og_title=_report_og_title,
        og_description=_report_og_desc,
        og_type="article",
    )


# Human-readable descriptions for the stage that set a sentence aside.
_TRIAGE_SOURCE_LABELS = {
    "A1": "Lexical prefilter (Stage A1)",
    "A2": "Check-worthiness classifier (Stage A2)",
}


def _falsifiability_note_html(n_claims: int, n_set_aside: int) -> str:
    """The falsifiability-ratio genre statistic (P67.10 / T4.2): what share of
    a speech is checkable at all. Derived at render time (T0.8)."""
    total = n_claims + n_set_aside
    if not total or not n_claims:
        return ""
    ratio = n_claims / total
    return (
        '<aside class="methodology">'
        f'<strong>Falsifiability ratio.</strong> {n_claims} of the {total} '
        f'sentences in this speech ({format(ratio, ".1%")}) made a checkable '
        'factual claim. This is a statistic about the <em>genre</em> — '
        'political speech is mostly narrative, applause lines, and aspiration '
        '— not about the speaker\'s accuracy, which the report itself '
        'measures.</aside>'
    )


def _render_statement_triage(site_report: SiteReport) -> str:
    """Render the Statement Triage page — the non-check-worthy sentences that the
    pipeline recorded but never published, grouped by the stage that set them aside.

    Returns "" when ``site_report.characterization`` is empty (legacy-clean: no
    page should be written in that case)."""
    records = site_report.characterization
    if not records:
        return ""

    total = len(records)
    # Group by source stage (A1 lexical prefilter vs A2 classifier). Unknown
    # sources fall into their own bucket rather than being dropped.
    groups: dict[str, list[dict]] = {}
    for rec in records:
        src = str(rec.get("source") or "other")
        groups.setdefault(src, []).append(rec)

    report_url = f"../reports/{site_report.report_slug}.html"
    breadcrumb = (
        '<div class="breadcrumb">'
        '<a href="../index.html">Reports</a> › '
        f'<a href="{report_url}">{_esc(site_report.speaker)} — '
        f'{_esc(site_report.display_date)}</a> › Statement Triage</div>'
    )

    intro = (
        '<section class="hero" id="top">'
        '<h1 class="speaker-name">Statement Triage</h1>'
        f'<div class="speech-title">{_esc(site_report.speaker)} — '
        f'{_esc(site_report.display_date)}</div>'
        '</section>'
        '<aside class="methodology">'
        '<strong>What we set aside.</strong> The pipeline reads every sentence of '
        'the speech, but only fact-checks the ones that assert a verifiable claim. '
        f'The {total} sentence' + ('s' if total != 1 else '') + ' below were '
        'recorded as <em>non-check-worthy</em> — pleasantries, opinion, or '
        'otherwise unimportant — and set aside before verification. We surface '
        'them here so it is clear what was excluded and which stage excluded it.'
        '</aside>'
        + _falsifiability_note_html(len(site_report.checkable_bundles), total)
    )

    # Deterministic stage order: A1 first, then A2, then any others alphabetically.
    def _stage_key(s: str) -> tuple[int, str]:
        order = {"A1": 0, "A2": 1}
        return (order.get(s, 2), s)

    sections: list[str] = []
    for src in sorted(groups, key=_stage_key):
        recs = groups[src]
        stage_label = _TRIAGE_SOURCE_LABELS.get(src, f"Stage {src}")
        rows: list[str] = []
        for rec in recs:
            text = _esc(str(rec.get("text", "")))
            label = _esc(str(rec.get("label", "")))
            a1 = rec.get("a1_score")
            try:
                a1_str = f"{float(a1):.2f}"
            except (TypeError, ValueError):
                a1_str = "—"
            meta = (
                f'<span class="triage-tag">{_esc(src)}</span>'
                f'label: {label} · a1_score: {a1_str}'
            )
            rows.append(
                '<li class="triage-item">'
                f'<div class="triage-text">{text}</div>'
                f'<div class="triage-meta">{meta}</div>'
                '</li>'
            )
        sections.append(
            '<div class="triage-group">'
            '<div class="section-head">'
            f'<span>{_esc(stage_label)}</span>'
            f'<span class="sub">{len(recs)} set aside</span>'
            '</div>'
            '<ul class="triage-list">' + "".join(rows) + '</ul>'
            '</div>'
        )

    body = breadcrumb + intro + "".join(sections)

    phash = _prompt_hash()
    footer = (
        f'<span>truth-bot · pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}</span>'
        f'<span>Prompt <a class="footer-hash" href="../about.html#prompt">{phash}</a></span>'
        f'<span>Source: <a href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'github.com/aRealGem/Truth-bot</a></span>'
    )
    return _page_report(
        f"Statement Triage — {_esc(site_report.speaker)} — {_esc(site_report.display_date)}",
        body,
        footer=footer,
        og_title=f"Statement Triage — {site_report.speaker} — {site_report.display_date} — truth-bot",
        og_description=(
            f"{total} non-check-worthy sentences set aside from {site_report.speaker}'s "
            f"{site_report.display_date} remarks, and the pipeline stage that excluded each."
        ),
        og_type="article",
    )


def _render_claim_page(bundle: VerdictBundle, site_report: SiteReport) -> str:
    """Render a standalone per-claim permalink page."""
    report_url = f"../reports/{site_report.report_slug}.html"
    body = (
        f'<div class="breadcrumb">'
        f'<a href="../index.html">Reports</a> › '
        f'<a href="{report_url}">{_esc(site_report.speaker)} — '
        f'{_esc(site_report.display_date)}</a> › Claim</div>'
        f'{_claim_card(bundle, 1, 1, rel="../", standalone=True, panel_roster=getattr(site_report, "panel_roster", None))}'
    )
    phash = _prompt_hash()
    gen_ts = site_report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    footer = (
        f'<span>truth-bot · pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}</span>'
        f'<span>Prompt <a class="footer-hash" href="../about.html#prompt">{phash}</a></span>'
        f'<span>Source: <a href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'github.com/aRealGem/Truth-bot</a></span>'
    )
    _claim_text_trunc = bundle.claim.text[:60]
    _claim_og_title = f"Claim: {_claim_text_trunc} — truth-bot"
    _verdict_label = bundle.consensus.consensus_verdict
    # Agreement meta speaks panel-vote vocabulary on PCA bundles (1.7): the
    # bridge collapses the panel to ONE reconciled ModelVerdict (zero on a
    # split), so the old adapter tally read "1 of 1 models agree" — or
    # "0 of 0" on splits. panel_votes preserves the real seat tally. Legacy
    # multi-adapter bundles (no votes) keep the adapter-count wording.
    _votes = dict(getattr(bundle.consensus.provenance, "panel_votes", {}) or {})
    if _votes and _verdict_label == "Models split":
        _agree_text = "Panel split — no consensus."
    elif _votes:
        _n_seats = sum(_votes.values())
        _agree_text = (
            f"{max(_votes.values())} of {_n_seats} "
            f"seat{'s' if _n_seats != 1 else ''} agree."
        )
    else:
        _total_models = len(bundle.model_verdicts)
        _agree_models = sum(
            1 for mv in bundle.model_verdicts
            if mv.label.value == _verdict_label
        )
        _agree_text = (
            f"{_agree_models} of {_total_models} "
            f"model{'s' if _total_models != 1 else ''} agree."
        )
    _claim_og_desc = (
        f"Verdict: {_verdict_label}. "
        f"{_agree_text} "
        "Checked against a shared, cited evidence pack."
    )
    return _page_report(
        f"Claim: {_claim_text_trunc}",
        body,
        footer=footer,
        og_title=_claim_og_title,
        og_description=_claim_og_desc,
        og_type="article",
    )


def _render_truthy() -> str:
    """Fun Truthy page: shared SVG + index-style hero cycle, toggle-only droid sounds, masthead chrome."""
    # Standard volume / volume-mute (stroke icons, currentColor).
    _sound_icons = (
        '<span class="truthy-sound-toggle-icons" aria-hidden="true">'
        '<svg class="icon-on" width="22" height="22" viewBox="0 0 24 24" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>'
        '<path d="M15.54 8.46a5 5 0 010 7.07"/>'
        '<path d="M19.07 4.93a10 10 0 010 14.14"/>'
        '</svg>'
        '<svg class="icon-off" width="22" height="22" viewBox="0 0 24 24" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>'
        '<line x1="22" y1="9" x2="16" y2="15"/>'
        '<line x1="16" y1="9" x2="22" y2="15"/>'
        '</svg>'
        '</span>'
    )
    hero_block = (
        '<h1 class="truthy-fun-h1">Truthy</h1>'
        '<div class="truthy-sound-row" role="group" aria-labelledby="truthy-sound-label">'
        '<span class="truthy-sound-label" id="truthy-sound-label">Toggle droid sounds: </span>'
        '<button type="button" class="truthy-sound-toggle is-muted" id="truthy-sound-toggle" '
        'aria-pressed="false" aria-label="Turn on droid sounds">'
        + _sound_icons
        + '</button>'
        '</div>'
        '<div class="index-hero">'
        '<div class="hero-truthy-wrap">'
        + _TRUTHY_SVG
        + '</div>'
        '<div class="hero-truthy-col">'
        '<div class="truthy-bubble is-true" id="hero-bubble" '
        'style="opacity:1;transition:opacity 100ms ease,transform 100ms ease">'
        "I&rsquo;m Truthy and honesty makes me happy!"
        '</div>'
        '</div>'
        '</div>'
    )
    notes = (
        '<div class="truthy-fun-notes">'
        '<p class="truthy-fun-notes-lead">Truthy M. -- The M. stands for McTruthface!</p>'
        '<p class="truthy-fun-notes-mascot">Our citizen-funded fact-checking mascot.</p>'
        '<p class="truthy-fun-notes-outro">A fact-check for every citizen. Funded by We The People, powered by AI, '
        'made in the USA. Because liberty and justice for all starts with the truth. &#x1F1FA;&#x1F1F8;</p>'
        '</div>'
    )
    body = hero_block + notes + _TRUTHY_FUN_SCRIPT
    _phash = _prompt_hash()
    footer = (
        '<span><a href="./index.html">Back to reports</a></span>'
        f'<span>Pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}'
        f' &middot; Prompt <a class="footer-hash" href="./about.html#prompt">{_phash}</a>'
        f' &middot; <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a></span>'
    )
    return _page_truthy(
        "Meet Truthy",
        body,
        footer,
        og_title="Meet Truthy McTruthface — truth-bot",
        og_description=(
            "Truthy McTruthface is truth-bot's citizen-funded fact-checking mascot. "
            "A fact-check for every citizen."
        ),
    )


def _insights_strip_html(insights: "ModelPanelInsights | None") -> str:
    """Compact 3-card highlight strip for the landing page.

    Renders nothing when ``insights`` is None or has no models — the
    strip is purely additive context, so a fresh / empty corpus
    shouldn't break the index. The CTA always points to
    ``./model-insights.html`` regardless.
    """
    if insights is None or not insights.per_model:
        return ""
    cards: list[str] = []
    top_pair = insights.top_pair
    if top_pair is not None:
        cards.append(
            '<div class="insight-card">'
            '<div class="insight-card-eyebrow">Strongest pairwise agreement</div>'
            '<div class="insight-card-headline">'
            + _esc(_adapter_pretty(top_pair.a))
            + ' &harr; '
            + _esc(_adapter_pretty(top_pair.b))
            + '</div>'
            '<div class="insight-card-figure">'
            + format(top_pair.agreement_rate, '.0%')
            + ' identical fine-label calls</div>'
            '</div>'
        )
    most_div = insights.most_divergent
    if most_div is not None and most_div.dissent_rate > 0:
        cards.append(
            '<div class="insight-card">'
            '<div class="insight-card-eyebrow">Most divergent on the panel</div>'
            '<div class="insight-card-headline">'
            + _esc(most_div.pretty_name)
            + '</div>'
            '<div class="insight-card-figure">'
            + format(most_div.dissent_rate, '.0%')
            + ' of claims diverge from consensus</div>'
            '</div>'
        )
    most_lenient = insights.most_lenient
    most_strict  = insights.most_strict
    if (most_lenient is not None and most_strict is not None
            and most_lenient.adapter != most_strict.adapter):
        cards.append(
            '<div class="insight-card">'
            '<div class="insight-card-eyebrow">Truthy-axis bias spread</div>'
            '<div class="insight-card-headline">'
            + _esc(most_lenient.pretty_name)
            + ' &uarr; &nbsp; vs &nbsp; '
            + _esc(most_strict.pretty_name)
            + ' &darr;'
            + '</div>'
            '<div class="insight-card-figure">'
            + format(most_lenient.truthy_bias, '+.2f')
            + ' &nbsp;&middot;&nbsp; '
            + format(most_strict.truthy_bias, '+.2f')
            + '</div>'
            '</div>'
        )
    if not cards:
        return ""
    return (
        '<section class="insights-strip" aria-labelledby="insights-strip-head">\n'
        '  <div class="section-head">'
        '<span id="insights-strip-head">Model panel insights</span>'
        '<span class="sub"><a href="./model-insights.html">'
        'Full breakdown &rarr;</a></span></div>\n'
        '  <div class="insight-cards">' + ''.join(cards) + '</div>\n'
        '</section>\n'
    )


def _adapter_pretty(adapter: str) -> str:
    """Insights-side adapter pretty name (delegates to ``insights`` map)."""
    from truthbot.publish.insights import _adapter_brand
    return _adapter_brand(adapter)


def _bias_bar_html(stat: "ModelStat") -> str:
    """Two-sided horizontal bar for the truthy bias number.

    Bias range is theoretically -2..+2 but in practice clusters near
    -0.5..+0.5. We clamp to -1..+1 for display so the chart axis is
    stable across vintages.
    """
    bias = max(-1.0, min(1.0, stat.truthy_bias))
    half_pct = abs(bias) * 50.0
    side = 'lenient' if bias >= 0 else 'strict'
    label_color = (
        'var(--v-truthy)' if bias > 0
        else ('var(--v-falsey)' if bias < 0 else 'var(--ink)')
    )
    return (
        '<div class="bias-row">'
        '  <div class="bias-row-label">' + _esc(stat.pretty_name) + '</div>\n'
        '  <div class="bias-track" aria-hidden="true">\n'
        '    <div class="bias-mid"></div>\n'
        '    <div class="bias-fill bias-fill-' + side + '" '
        'style="width:' + format(half_pct, '.2f') + '%"></div>\n'
        '  </div>\n'
        '  <div class="bias-row-figure" style="color:' + label_color + '">'
        + format(stat.truthy_bias, '+.2f')
        + '</div>\n'
        '</div>'
    )


def _agreement_matrix_html(insights: "ModelPanelInsights") -> str:
    """4x4 agreement table. Diagonal is greyed out."""
    adapters = sorted({m.adapter for m in insights.per_model})
    if not adapters:
        return ""
    pretty = {a: _adapter_pretty(a) for a in adapters}
    by_pair = {
        (p.a, p.b): p for p in insights.pairwise
    }
    head_cells = "".join(
        f'<th scope="col">{_esc(pretty[a])}</th>' for a in adapters
    )
    rows: list[str] = []
    for a in adapters:
        cells: list[str] = [f'<th scope="row">{_esc(pretty[a])}</th>']
        for b in adapters:
            if a == b:
                cells.append('<td class="agg-self">&mdash;</td>')
                continue
            key = (min(a, b), max(a, b))
            pair = by_pair.get(key)
            if pair is None:
                cells.append('<td class="agg-empty">&mdash;</td>')
            else:
                cells.append(
                    '<td class="agg-cell">'
                    + format(pair.agreement_rate, '.0%')
                    + '<span class="agg-n">n=' + str(pair.claims_both_present) + '</span>'
                    + '</td>'
                )
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (
        '<table class="agreement-matrix">\n'
        '  <thead><tr><th></th>' + head_cells + '</tr></thead>\n'
        '  <tbody>' + ''.join(rows) + '</tbody>\n'
        '</table>\n'
    )


def _extreme_split_card(e: "ExtremeSplit") -> str:
    others = ", ".join(
        _esc(_adapter_pretty(a)) + ': <strong>' + _esc(lbl) + '</strong>'
        for a, lbl in e.other_labels.items()
    )
    direction_word = "lone optimist" if e.direction == "optimist" else "lone pessimist"
    href = '../' + e.claim_url if e.claim_url else ''
    title_html = _esc(e.claim_text)
    speaker_meta = ''
    if e.speaker:
        speaker_meta = (
            ' &middot; <span class="extreme-speaker">'
            + _esc(e.speaker)
            + (' &middot; ' + _esc(e.date) if e.date else '')
            + '</span>'
        )
    return (
        '<article class="extreme-card">\n'
        '  <header class="extreme-head">\n'
        '    <span class="extreme-diff">&Delta;' + str(e.diff) + '</span>\n'
        '    <span class="extreme-odd">'
        + _esc(_adapter_pretty(e.odd_one_out))
        + ' as ' + direction_word + '</span>\n'
        '    <span class="extreme-odd-label">'
        + _esc(e.odd_label)
        + '</span>\n'
        '  </header>\n'
        '  <p class="extreme-text">' + (
            f'<a href="{href}">{title_html}</a>' if href else title_html
        ) + '</p>\n'
        '  <p class="extreme-meta">vs ' + others + speaker_meta + '</p>\n'
        '</article>'
    )


def _render_model_insights(insights: "ModelPanelInsights | None") -> str:
    """Dedicated model-insights deep-dive page."""
    from truthbot.publish.insights import EXTREME_DIFF_THRESHOLD
    if insights is None or not insights.per_model:
        body = (
            '<section class="prose">\n'
            '  <h1>Model panel insights</h1>\n'
            '  <p>Not enough claims yet to compute panel insights. Check back '
            'after the next report run.</p>\n'
            '</section>\n'
        )
    else:
        bias_rows = "\n".join(_bias_bar_html(m) for m in insights.per_model)
        per_model_table_rows = "\n".join(
            '<tr>'
            f'<td><strong>{_esc(m.pretty_name)}</strong></td>'
            f'<td class="num-cell">{m.claims_seen}</td>'
            f'<td class="num-cell">{m.dissent_count}</td>'
            f'<td class="num-cell">{m.dissent_rate:.0%}</td>'
            f'<td class="num-cell">{m.truthy_bias:+.2f}</td>'
            f'<td class="num-cell">{m.extreme_lone_optimist}</td>'
            f'<td class="num-cell">{m.extreme_lone_pessimist}</td>'
            '</tr>'
            for m in insights.per_model
        )
        extreme_html = ""
        if insights.top_extreme_splits:
            extreme_html = (
                '<section class="prose insights-extremes">\n'
                '  <h2>Top extreme splits</h2>\n'
                '  <p class="insights-meta">'
                'Claims where exactly one model was the lone outlier '
                f'(&Delta; ≥ {EXTREME_DIFF_THRESHOLD} points on the truthy axis). '
                'Sorted by magnitude.</p>\n'
                + ''.join(_extreme_split_card(e) for e in insights.top_extreme_splits)
                + '</section>\n'
            )
        body = (
            '<section class="prose">\n'
            '  <h1>Model panel insights</h1>\n'
            f'  <p class="insights-meta">{insights.total_claims} distinct claims '
            f'across {len(insights.per_model)} frontier models. '
            'All numbers update on every report publish.</p>\n'
            '  <h2>Per-model summary</h2>\n'
            '  <table class="insights-summary">\n'
            '    <thead><tr>'
            '<th>Model</th>'
            '<th class="num-cell">Claims</th>'
            '<th class="num-cell">Dissents</th>'
            '<th class="num-cell">Dissent %</th>'
            '<th class="num-cell">Truthy bias</th>'
            '<th class="num-cell">Lone &uarr;</th>'
            '<th class="num-cell">Lone &darr;</th>'
            '</tr></thead>\n'
            '    <tbody>'
            + per_model_table_rows +
            '</tbody>\n'
            '  </table>\n'
            '  <h2>Truthy bias</h2>\n'
            '  <p class="insights-meta">'
            'Average signed gap between this model&rsquo;s truthy-axis '
            'score and the panel mean, per claim. Positive = leaner '
            'toward Truthy; negative = stricter.</p>\n'
            '  <div class="bias-chart">' + bias_rows + '</div>\n'
            '  <h2>Pairwise agreement</h2>\n'
            '  <p class="insights-meta">'
            'Share of co-checked claims where the two models cast '
            'identical fine-label verdicts.</p>\n'
            + _agreement_matrix_html(insights) +
            '</section>\n'
            + extreme_html +
            '<section class="prose insights-method">\n'
            '  <h2>Method</h2>\n'
            '  <p>Truthy-axis scores: True (+2), Mostly True (+1), '
            'Unverifiable (0), Exaggerated/Misleading (-1), False (-2). '
            'Dissents are counted against the published consensus '
            'verdict for each claim. Pairwise agreement uses the full '
            '6-bucket fine label, not the projected 5-bucket Truthy '
            'scale, so it&rsquo;s a strict measurement of label '
            'identity.</p>\n'
            '  <p>The <a href="https://github.com/truth-bot/truth-bot/blob/main/eval/opus_vs_rest_scan.py">'
            'Opus-vs-rest scan</a> is the standalone variant that '
            'inspired this page; both share their constants via '
            '<code>truthbot.publish.insights.LABEL_SCORE</code>.</p>\n'
            '  <p><a href="./about.html">&larr; About this site</a></p>\n'
            '</section>\n'
        )
    _phash = _prompt_hash()
    footer = (
        '<span><a href="./index.html">Back to reports</a></span>'
        f'<span>Pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}'
        f' &middot; Prompt <a class="footer-hash" href="./about.html#prompt">{_phash}</a>'
        f' &middot; <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a></span>'
    )
    return _page_about(
        "Model panel insights",
        body,
        footer,
        og_title="Model panel insights — truth-bot",
        og_description=(
            "How frontier models agree, dissent, and skew on truth-bot's "
            "fact-check panel — pairwise agreement, truthy bias, lone-outlier "
            "splits."
        ),
    )


def _render_model_insights_v2(reports: list[dict], claims: list[dict]) -> str:
    """Per-seat panel insights rebuilt from published provenance (P67.10 /
    T4.1) — replaces the retired v1 page, which summarized one reconciled
    pseudo-model at 0% dissent by construction. Every figure derives from
    claims.json at build time (T0.8)."""
    from truthbot.publish.seat_insights import compute_seat_insights

    insights = compute_seat_insights(claims)
    by_id = {r.get("id"): r for r in reports}
    label_order = ["True", "Misleading", "False", "Unverifiable"]

    sections: list[str] = []
    for rid, ins in insights.items():
        r = by_id.get(rid) or {}
        roster = (r.get("panel_roster") or {}).get("seats") or {}
        title = f'{_esc(r.get("speaker", "Unknown"))} — {_esc(r.get("date", ""))}'
        url = r.get("url", "#")
        seat_rows = []
        for role in ("proposer", "critic", "arbiter"):
            seat = ins.seats.get(role)
            if seat is None:
                continue
            model = ", ".join(roster.get(role, [])) or "—"
            counts = " · ".join(
                f'{lbl} <span class="ct">{seat.label_counts.get(lbl, 0)}</span>'
                for lbl in label_order if seat.label_counts.get(lbl, 0))
            seat_rows.append(
                f'<tr><td><strong>{_esc(role.capitalize())}</strong></td>'
                f'<td class="mono">{_esc(model)}</td>'
                f'<td>{counts}</td>'
                f'<td>{format(seat.rate("False"), ".1%")}</td>'
                f'<td>{seat.total}</td></tr>')
        arb = ins.arbiter_sided
        arb_line = ""
        if ins.escalated:
            arb_line = (
                f'<p>Arbiter side-taking on the {ins.escalated} escalated claims: '
                f'sided with the proposer {arb.get("proposer", 0)}, with the critic '
                f'{arb.get("critic", 0)}, took a third position {arb.get("neither", 0)}.</p>')
        ov_line = ""
        if ins.overrides:
            parts = [f'{_esc(k)} <span class="ct">{v}</span>'
                     for k, v in sorted(ins.overrides.items())]
            ov_line = ('<p>Severity-Classifier stage-2 overrides: '
                       + " · ".join(parts) + '.</p>')
        sections.append(
            f'<h3 style="margin-top:1.5rem"><a href="./{_esc(url)}">{title}</a></h3>'
            f'<p class="dim">{ins.n_claims} claims · {ins.escalated} escalated to '
            f'the arbiter ({format(ins.escalation_rate, ".1%")})</p>'
            f'<table class="tier-table seat-insights">'
            f'<tr><th>Seat</th><th>Model</th><th>Seat predictions</th>'
            f'<th>False-rate</th><th>Claims voted</th></tr>'
            + "".join(seat_rows) + '</table>'
            + arb_line + ov_line)

    body = (
        '<h2>Model panel insights</h2><hr class="rule">'
        '<p>What each seat of the verdict panel actually predicted, per report '
        '— computed from the published per-claim provenance '
        '(<span class="mono">panel_by_role</span>), not from the reconciled '
        'verdicts. Disagreement between seats is the pipeline\'s error-'
        'catching mechanism: an escalation means the proposer and critic '
        'differed and the arbiter decided; a Severity-Classifier override '
        'means a second-stage model re-graded a boundary call. Both are '
        'disclosed per claim on its provenance strip.</p>'
        + "".join(sections)
        + '<p class="dim" style="margin-top:1.5rem">Method notes: seat '
        'predictions are each seat\'s own verdict before reconciliation; the '
        'False-rate is the share of that seat\'s votes reading False. See '
        '<a href="./about.html">About</a> for the full pipeline.</p>'
    )
    footer = (
        f'<span>truth-bot · pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}</span>'
        f'<span><a href="./corrections.html">Corrections</a></span>'
        f'<span>Source: <a href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'github.com/aRealGem/Truth-bot</a></span>'
    )
    return _page_about(
        "Model Panel Insights",
        body,
        footer=footer,
        og_title="Model panel insights — truth-bot",
        og_description="Per-seat verdict distributions, escalation rates, "
                       "arbiter side-taking, and severity overrides for every "
                       "published report.",
        og_type="website",
    )


def _render_corrections(entries: list[dict], notes: Optional[list[dict]] = None) -> str:
    """The public Corrections page (P67.6 / T1.5) — a fact-checking-norm
    changelog: claim id, old → new verdict, reason, date. Rendered on every
    publish (empty state included) so the page exists before its first entry
    and readers can always find the correction policy."""
    if entries:
        rows = "".join(
            f'<tr><td class="mono">{_esc(e["sid"])}</td>'
            f'<td>{_esc(e["speech_id"])}</td>'
            f'<td><span class="vt-{_verdict_css(e["old_verdict"].capitalize())}">'
            f'{_esc(e["old_verdict"].upper())}</span> → '
            f'<span class="vt-{_verdict_css(e["new_verdict"].capitalize())}">'
            f'{_esc(e["new_verdict"].upper())}</span></td>'
            f'<td>{_esc(e["reason"])}</td>'
            f'<td>{_esc(e["date"])}</td></tr>'
            for e in entries
        )
        table = (
            '<table class="tier-table corrections-table">'
            '<tr><th>Claim</th><th>Report</th><th>Verdict</th>'
            '<th>Reason</th><th>Date</th></tr>'
            f'{rows}</table>'
        )
    else:
        table = ('<p class="dim">No corrections have been issued for the '
                 'currently published reports.</p>')
    notes_html = "".join(
        f'<p class="corrections-note"><strong>{_esc(n["date"])}</strong> — '
        f'{_esc(n["text"])}</p>'
        for n in (notes or [])
    )
    body = (
        '<h2>Corrections</h2><hr class="rule">'
        '<p>When a published verdict is found to be wrong — a reasoning error, '
        'evidence outside the claim\'s era, a misread referent — it is corrected '
        'publicly, per fact-checking norms: the claim keeps a visible correction '
        'note on its provenance strip, and every change is logged here with the '
        'old and new verdict, the reason, and the date. Corrections are never '
        'applied silently.</p>'
        + notes_html
        + table
    )
    footer = (
        f'<span>truth-bot · pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}</span>'
        f'<span>Source: <a href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'github.com/aRealGem/Truth-bot</a></span>'
    )
    return _page_about(
        "Corrections",
        body,
        footer=footer,
        og_title="Corrections — truth-bot",
        og_description="Public changelog of corrected verdicts: claim, old and "
                       "new verdict, reason, and date.",
        og_type="website",
    )


def _render_model_insights_redirect() -> str:
    """Redirect stub shipped at model-insights.html while the page is retired
    (remediation T0.4). The v1 insights page predated the PCA pipeline: it
    showed a single reconciled pseudo-model with 0% dissent by construction
    and a claim total that disagreed with the rest of the site. The Phase 4
    rebuild will regenerate it from per-seat provenance (panel_by_role)."""
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '  <meta charset="utf-8">\n'
        '  <meta http-equiv="refresh" content="0; url=./about.html">\n'
        '  <link rel="canonical" href="./about.html">\n'
        '  <title>Model insights — moved — truth-bot</title>\n'
        '</head>\n<body>\n'
        '  <p>The model-insights page is being rebuilt from per-seat panel '
        'provenance. <a href="./about.html">Continue to About</a>.</p>\n'
        '</body>\n</html>\n'
    )


def _render_about() -> str:
    """Render the about/method page (PCA-era architecture; refreshed 2026-07-20)."""
    prompt_text = _pca_prompt_text()
    phash = _prompt_hash()

    tier_rows = "".join(
        f'<tr><td><strong>{_esc(t)}</strong></td><td>{_esc(d)}</td><td>{_esc(q)}</td></tr>'
        for t, d, q in TIER_TABLE
    )
    tier_table = (
        f'<table class="tier-table">'
        f'<tr><th>Tier</th><th>Sources</th><th>Trust weight</th></tr>'
        f'{tier_rows}</table>'
    )

    models_list = (
        "<ul>"
        "<li><strong>Proposer</strong> — Claude Opus 4.8: drafts the initial verdict</li>"
        "<li><strong>Critic</strong> — Grok 4.3: independently re-judges the same "
        "evidence, hunting for why a naive verdict could be wrong</li>"
        "<li><strong>Arbiter</strong> — GPT-5.5: adjudicates only when proposer and "
        "critic disagree</li>"
        "<li><strong>Severity Classifier</strong> — Claude Sonnet 4.6: a second-stage check on "
        "False-vs-Misleading boundary calls and panel ties</li>"
        "<li><strong>Evidence researchers</strong> — three independent web-search lanes "
        "(Claude Opus native search, GPT browsing, Grok search) whose shortlists are merged "
        "by a deterministic consolidator: URL dedup, era gates, fact-check-site exclusion, "
        "source-tier quotas. No model ranks another model's findings</li>"
        "<li><strong>Triage</strong> — Claude Haiku 4.5: check-worthiness classification "
        "of every sentence before any claim is judged</li>"
        "</ul>"
        "<p class=\"dim\" style=\"margin-top:0.5rem\">Three different model families from three "
        "different vendors sit in the verdict seats, so a single vendor's blind spot can't "
        "silently decide a claim. The exact roster used for a report is recorded in its "
        "\"Panel composition\" section. Evidence is gathered fresh per claim, time-scoped "
        "to what was knowable when the words were spoken, and fact-checking organizations "
        "are excluded from evidence packs — the panel reaches its own verdicts from "
        "primary sources rather than inheriting another checker's ruling.</p>"
    )

    limitations = (
        "<ul>"
        "<li><strong>One panel, one pass:</strong> The proposer→critic→arbiter structure "
        "and the second-stage Severity Classifier are the accuracy mechanism, and boundary "
        "calls (False vs Misleading) remain the hardest cases.</li>"
        "<li><strong>Retrieval-bounded:</strong> Verdicts are grounded in an evidence pack "
        "(up to ten items) assembled at run time. If retrieval misses the decisive source "
        "and the pack fails the quality bar, the claim is forced to Unverifiable — the "
        "panel is instructed not to fill gaps from memory.</li>"
        "<li><strong>No cross-claim context:</strong> Each claim is judged independently. "
        "Recurring rhetoric may be rated inconsistently across speeches.</li>"
        "<li><strong>Training-data bias:</strong> Model judgments may reflect the slant of "
        "their training data. Cross-vendor seats partially mitigate this; a bias shared by "
        "all three vendors would not be caught.</li>"
        "<li><strong>As-of-utterance judging:</strong> Claims are judged against evidence "
        "from their own era — a claim true when spoken is not False because reality moved "
        "later. The \"Last verified\" stamp on each claim shows when the check ran.</li>"
        "</ul>"
    )

    body = (
        f'<h2>About truth-bot</h2><hr class="rule">'
        f'<h3>What this is</h3>'
        f'<p>truth-bot is an automated political fact-checker. It segments a speech into '
        f'sentences, filters them to specific, verifiable, consequential claims, retrieves '
        f'era-appropriate evidence for each claim from the open web, and adjudicates every '
        f'claim through a structured panel of language models that must ground its verdict '
        f'in the retrieved evidence and cite it.</p>'
        f'<h3 style="margin-top:1.5rem">The pipeline</h3>'
        f'<p><strong>1 · Check-worthiness triage.</strong> Every sentence is classified as '
        f'check-worthy, opinion, or unimportant. Opinions, aspirations, pleasantries, and '
        f'rhetoric are set aside — visibly, not silently: each report links a '
        f'<em>Statement Triage</em> page listing everything excluded and why. Check-worthy '
        f'claims also get a type (statistical, historical, attribution, comparison, '
        f'personal-anecdote, other). The classifier is not told who the speaker is, '
        f'though a sentence may name its own speaker.</p>'
        f'<p style="margin-top:0.75rem"><strong>2 · Evidence retrieval.</strong> For each '
        f'claim, a small model writes targeted search queries (the era\'s fiscal year, the '
        f'specific statistic or program named), and the pipeline fetches candidates via web '
        f'search plus fact-check databases — time-scoped to the claim\'s era so a 2026 '
        f'article cannot decide a 2022 claim. Candidates are scored for relevance to the '
        f'claim, deduplicated, stripped of non-evidence (homepages, listing pages), and '
        f'capped at six items ranked by relevance, then source trust. Every pack item '
        f'carries a URL, retrieval timestamp, and content hash.</p>'
        f'<p style="margin-top:0.75rem"><strong>3 · The verdict panel (PCA).</strong> A '
        f'<em>proposer</em> drafts a verdict from the evidence; a <em>critic</em> '
        f'independently re-judges the same evidence; when they disagree, an <em>arbiter</em> '
        f'decides. Verdicts use a four-label contract — True, False, Misleading, '
        f'Unverifiable — and must cite pack items by id (the E1, E2… ids you see in '
        f'reasoning and source lists; citations outside the pack are rejected). A '
        f'genuine tie is either resolved by the Severity Classifier — recorded on the '
        f'claim\'s provenance strip — or published as "Panel split"; a tie is never '
        f'dropped without a visible trace. The panel is speaker-blind in its inputs: '
        f'the speaker\'s name is withheld as metadata, though the claim text itself '
        f'may still identify the speaker.</p>'
        f'<p style="margin-top:0.75rem"><strong>4 · Severity check.</strong> Because small '
        f'models tend to soften a contradicted claim to "Misleading," False-vs-Misleading '
        f'boundary calls and tie-routed rows pass through a second-stage Severity '
        f'Classifier on a stronger model. Its overrides are shown on the claim card\'s '
        f'provenance strip.</p>'
        f'<p style="margin-top:0.75rem"><strong>Guest anecdotes.</strong> A private '
        f'individual\'s personal story told from the stage usually has no public record to '
        f'check against. Those claims still run the full panel, but when they come back '
        f'unverifiable they are labeled <em>Anecdote</em> — a limit of the genre, not a '
        f'failed verification. An anecdote the press independently investigated gets a '
        f'real verdict.</p>'
        f'<div class="pipeline-diagram" aria-label="Pipeline diagram">'
        f'<span class="pd-node">Transcript</span><span class="pd-arrow">→</span>'
        f'<span class="pd-node">Check-worthiness triage<br><small>A1 + A2, speaker withheld</small></span>'
        f'<span class="pd-arrow">→</span>'
        f'<span class="pd-node">Era-scoped retrieval<br><small>queries + connectors</small></span>'
        f'<span class="pd-arrow">→</span>'
        f'<span class="pd-node">Evidence pack<br><small>deduped · ranked · hashed</small></span>'
        f'<span class="pd-arrow">→</span>'
        f'<span class="pd-node">Panel<br><small>proposer → critic → arbiter</small></span>'
        f'<span class="pd-arrow">→</span>'
        f'<span class="pd-node">Severity check</span><span class="pd-arrow">→</span>'
        f'<span class="pd-node">Report</span>'
        f'</div>'
        f'<h3 style="margin-top:1.5rem">How to read a report</h3>'
        f'<p><strong>Per-claim pill.</strong> Each claim headlines the panel\'s own verdict '
        f'(True, False, Misleading, Unverifiable — or Anecdote / Panel split). The '
        f'provenance strip beneath the verdict shows the full chain: how the claim was '
        f'routed, what each seat predicted, the vote tally, and any Severity Classifier '
        f'override. E-ids in the reasoning link to the exact evidence item cited.</p>'
        f'<p style="margin-top:0.75rem"><strong>Report headline &amp; leaning totals.</strong> '
        f'Claims aggregate into two families — true-leaning (True) and false-leaning (False '
        f'+ Misleading) — over <em>decided</em> claims only; Unverifiable, Anecdote, and '
        f'Panel split are abstentions and stay out of the denominator. The headline band is '
        f'the dominant family\'s share of decided claims: ≥70% "Largely," ≥55% "Mostly," '
        f'under 55% "Mixed verdict." The family rail above each verdict bar brackets the '
        f'same totals on the graph itself, so the headline\'s "N of M decided claims '
        f'X-leaning" is always visibly derivable.</p>'
        f'<p style="margin-top:0.75rem"><strong>Display conventions.</strong> Aggregate '
        f'bars show every claim, including a distinct <em>Models split</em> segment for '
        f'panel deadlocks — segments always sum to the report\'s claim count. Guest '
        f'anecdotes keep their Unverifiable (or Models split) bucket on the bar; the '
        f'Anecdote pill and the footnote beneath the bar break out how many of those '
        f'abstentions are anecdotes.</p>'
        f'<p style="margin-top:0.75rem"><strong>Going deeper.</strong> The '
        f'<a href="./model-insights.html">Model panel insights</a> page shows what each '
        f'seat predicted per report — dissent, escalations, arbiter side-taking, and '
        f'severity overrides. Each report\'s <em>Statement Triage</em> page carries its '
        f'falsifiability ratio: the share of the speech that made a checkable claim at '
        f'all, a statistic about the genre rather than the speaker.</p>'
        f'<p style="margin-top:0.75rem"><strong>The Lens chip</strong> flips between two '
        f'presentations of the same computation: <em>Lenient</em> shows the simple '
        f'Truthy/Falsey lean; <em>Strict</em> (the default) shows the graded bands. The two '
        f'lenses share the Mixed band and the decided-claims denominator — they can never '
        f'disagree about whether a report is a toss-up.</p>'
        f'<h3 style="margin-top:1.5rem">Who\'s on the panel</h3>'
        f'{models_list}'
        f'<h3 style="margin-top:1.5rem">Source tier hierarchy</h3>'
        f'<p>Evidence items carry a trust tier assigned from the source\'s registered '
        f'domain. Relevance to the claim ranks first; tier breaks ties and is what the '
        f'panel is told to weigh on conflicting evidence.</p>'
        f'{tier_table}'
        f'<h3 style="margin-top:1.5rem">Known limitations</h3>'
        f'{limitations}'
        f'<h3 id="prompt" style="margin-top:1.5rem">Full verdict prompts (hash: {phash})</h3>'
        f'<p class="dim">Verbatim system prompts for the three panel seats — the calibrated '
        f'open-book set, including the decision procedure and the absolute-claim rule. The '
        f'hash in every report footer commits to exactly this text.</p>'
        f'<pre>{_esc(prompt_text)}</pre>'
        f'<hr class="rule-light">'
        f'<p class="dim"><a href="{GITHUB_URL}" target="_blank">GitHub</a> · '
        f'Pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}</p>'
    )
    footer = (
        f'<span>truth-bot · pipeline v{PIPELINE_VERSION}{BETA_BADGE_HTML}</span>'
        f'<span>Prompt <a class="footer-hash" href="./about.html#prompt">{phash}</a></span>'
        f'<span>Source: <a href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'github.com/aRealGem/Truth-bot</a></span>'
    )
    return _page_about(
        "About",
        body,
        footer=footer,
        og_title="About — truth-bot",
        og_description=(
            "How truth-bot works: check-worthiness triage, era-scoped evidence retrieval, "
            "a speaker-blind proposer-critic-arbiter model panel that must cite its "
            "evidence, and a second-stage severity check."
        ),
        og_type="website",
    )


def _render_404() -> str:
    body = (
        '<h2>404 — Page not found</h2>'
        '<p class="dim">The page you requested does not exist.</p>'
        '<p><a href="index.html">Return to reports</a></p>'
    )
    return _page_about(
        "404 Not Found",
        body,
        og_title="404 Not Found — truth-bot",
        og_description="The page you requested does not exist on truth-bot.",
        og_type="website",
    )


# ── Atom feed (remediation v2, 1.5) ──────────────────────────────────────────


def _iso_utc(dt: datetime) -> str:
    """ISO-8601 UTC with Z suffix; naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _feed_display_date(date_str: str) -> str:
    """'2026-03-04' → 'March 04, 2026'; anything unparseable passes through."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
    except (TypeError, ValueError):
        return date_str or "Unknown date"


def _render_feed(reports: list[dict], site_url: str) -> str:
    """Render the Atom feed — one <entry> per reports.json row (the caller
    passes the freshly sorted index, newest speech first).

    Every value derives from the index row: link/id from the report url,
    <published> from the speech date, <updated> from the row's per-publish
    ``generated_at`` stamp (speech-date fallback for legacy rows), the
    summary from the strict-lens family verdict via ``aggregation``. The
    feed-level <updated> is the max entry <updated>. All text is
    XML-escaped. Replaces the static template whose phantom entry and
    [SITE_URL] placeholder shipped verbatim (1.5).
    """
    from xml.sax.saxutils import escape, quoteattr

    entry_blocks: list[str] = []
    updated_stamps: list[str] = []
    for r in reports:
        url = str(r.get("url", ""))
        stem = Path(url).stem
        title = f"{r.get('speaker', '')} — {_feed_display_date(r.get('date', ''))}"
        published = (f"{r['date']}T00:00:00Z" if r.get("date")
                     else str(r.get("generated_at", "")))
        updated = str(r.get("generated_at", "") or published)
        updated_stamps.append(updated)
        claim_count = r.get("claim_count", 0)
        dist = (r.get("verdict_distribution_strict")
                or _agg_project_dist(r.get("verdict_distribution") or {}, "strict"))
        fam = _agg_family_verdict(dist)
        summary = (
            f"{claim_count} claim{'s' if claim_count != 1 else ''} checked. "
            f"Verdict: {fam.label} — {fam.ratio_text}. "
            "Multi-model AI fact-check with cited sources."
        )
        entry_blocks.append(
            "  <entry>\n"
            f"    <title>{escape(title)}</title>\n"
            f"    <link href={quoteattr(f'{site_url}/{url}')} "
            'rel="alternate" type="text/html"/>\n'
            f"    <id>urn:truth-bot:report:{escape(stem)}</id>\n"
            f"    <published>{escape(published)}</published>\n"
            f"    <updated>{escape(updated)}</updated>\n"
            f'    <summary type="text">{escape(summary)}</summary>\n'
            '    <category term="fact-check"/>\n'
            '    <category term="speech"/>\n'
            "  </entry>\n"
        )

    feed_updated = (max(updated_stamps) if updated_stamps
                    else _iso_utc(datetime.now(timezone.utc)))
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        f"  <title>truth-bot{BETA_TEXT_SUFFIX}</title>\n"
        "  <subtitle>Automated political fact-checking with multi-model "
        "consensus</subtitle>\n"
        f"  <link href={quoteattr(site_url + '/feed.xml')} rel=\"self\" "
        'type="application/atom+xml"/>\n'
        f"  <link href={quoteattr(site_url + '/')} rel=\"alternate\" "
        'type="text/html"/>\n'
        f"  <link href={quoteattr(site_url + '/corrections.html')} "
        'rel="related" type="text/html" title="Corrections"/>\n'
        f"  <updated>{escape(feed_updated)}</updated>\n"
        "  <id>urn:truth-bot:feed</id>\n"
        "  <author>\n    <name>truth-bot pipeline</name>\n  </author>\n"
        f'  <generator version="{PIPELINE_VERSION}">'
        f"truth-bot{BETA_TEXT_SUFFIX}</generator>\n"
        "  <rights>Data sourced from public speeches and cited web "
        "evidence.</rights>\n\n"
        + "".join(entry_blocks)
        + "</feed>\n"
    )


# ── SitePublisher ─────────────────────────────────────────────────────────────

class SitePublisher:
    """
    Generates and maintains a static fact-check website.

    Parameters
    ----------
    site_root:
        Root directory for the generated site. Reads TRUTHBOT_SITE_ROOT
        env var if not provided; falls back to ./site/.
    """

    def __init__(self, site_root: Optional[str | Path] = None,
                 corrections: Optional[list[dict]] = None,
                 correction_notes: Optional[list[dict]] = None) -> None:
        import os
        if site_root:
            self._root = Path(site_root)
        else:
            self._root = Path(os.environ.get("TRUTHBOT_SITE_ROOT", "./site"))
        # Public corrections entries + editorial notes (P67.6 / T1.5) —
        # rendered on corrections.html each publish. Empty → empty-state page.
        self._corrections: list[dict] = list(corrections or [])
        self._correction_notes: list[dict] = list(correction_notes or [])

    # ── Public API ────────────────────────────────────────────────────────────

    def publish(self, site_report: SiteReport) -> Path:
        """
        Generate/update all site files for a new or updated report.

        Returns the absolute path to the report HTML page.
        """
        self._ensure_structure()
        self._copy_assets()

        # Backfill speaker/date onto bundles that bridged without them
        # (PR-A2.1). The bridge only threads speaker/date_str when the claim
        # dicts carry them; the offline artifact path ({"sid","text","context",
        # "layer_a"}) does not, leaving bundle.speaker="Unknown". The report
        # knows both, and the principal-relation display (self-sourced-only
        # pill/badges/chip) needs them per bundle.
        for bundle in site_report.bundles:
            if not bundle.speaker or bundle.speaker == "Unknown":
                bundle.speaker = site_report.speaker
            if not bundle.date_str:
                bundle.date_str = site_report.date_str

        # Write report page
        report_html = _render_report(site_report)
        report_path = self._root / "reports" / f"{site_report.report_slug}.html"
        self._write(report_path, report_html)

        # Write Statement Triage page (non-check-worthy sentence stream). Only
        # emitted when the report carries a characterization list — legacy
        # reports (empty list) render no triage page and no cross-link.
        triage_html = _render_statement_triage(site_report)
        if triage_html:
            triage_path = self._root / "reports" / f"{site_report.triage_slug}.html"
            self._write(triage_path, triage_html)

        # Write per-claim pages
        for bundle in site_report.checkable_bundles:
            claim_html = _render_claim_page(bundle, site_report)
            claim_path = self._root / "claims" / f"{bundle.claim.id}.html"
            self._write(claim_path, claim_html)

        # Update data files
        reports_index = self._load_reports_index()
        claims_index = self._load_claims_index()

        # Remove stale entry for this report (re-add updated)
        reports_index = [r for r in reports_index if r.get("id") != site_report.report_id]
        claims_index  = [c for c in claims_index  if c.get("report_id") != site_report.report_id]

        reports_index.insert(0, self._report_meta(site_report))
        for bundle in site_report.checkable_bundles:
            claims_index.append(self._claim_meta(bundle, site_report))

        # Reverse-chronological by SPEECH date (jackie, 2026-08-01): with a
        # multi-president corpus, publish order is meaningless to a reader —
        # the index and reports.json list newest speech first. Undated
        # reports sink to the end, stable within a date.
        reports_index.sort(key=lambda r: r.get("date") or "", reverse=True)

        self._write_reports_index(reports_index)
        self._write_claims_index(claims_index)

        # Atom feed renders from the sorted index (1.5) — one entry per
        # report, absolute links via TRUTHBOT_SITE_URL.
        self._write_feed(reports_index)

        # Regenerate index
        stats = self._compute_stats(reports_index, claims_index)
        index_html = _render_index(reports_index, stats)
        self._write(self._root / "index.html", index_html)

        # About + 404 (regenerate on each publish for prompt-hash freshness)
        self._write(self._root / "about.html", _render_about())
        self._write(self._root / "truthy.html", _render_truthy())
        self._write(self._root / "404.html",   _render_404())
        # model-insights v2 (P67.10 / T4.1): rebuilt from per-seat provenance.
        # Falls back to the About redirect stub when the claims index carries
        # no panel_by_role data (fresh site, legacy corpus).
        has_seats = any((c.get("provenance") or {}).get("panel_by_role")
                        for c in claims_index)
        self._write(
            self._root / "model-insights.html",
            _render_model_insights_v2(reports_index, claims_index)
            if has_seats else _render_model_insights_redirect(),
        )
        self._write(self._root / "corrections.html",
                    _render_corrections(self._corrections,
                                        self._correction_notes))

        return report_path.resolve()

    def site_url(self, site_report: SiteReport, base_url: str = "http://expressionpi.home.arpa/truthbot") -> str:
        return f"{base_url.rstrip('/')}/reports/{site_report.report_slug}.html"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_structure(self) -> None:
        for sub in ("reports", "claims", "assets", "data", "assets/icons"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)

    def _copy_assets(self) -> None:
        self._write(self._root / "assets" / "styles.css", CSS)
        self._write(self._root / "assets" / "truthbot.js", JS)
        self._copy_icons()
        self._copy_social_assets()

    def _copy_icons(self) -> None:
        """Copy package-shipped icon SVGs to the site's assets/icons/ folder."""
        src_dir = Path(__file__).resolve().parent / "assets" / "icons"
        dst_dir = self._root / "assets" / "icons"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            return
        for svg in src_dir.glob("*.svg"):
            dst = dst_dir / svg.name
            dst.write_bytes(svg.read_bytes())
            logger.debug("Copied icon %s -> %s", svg.name, dst)

    def _copy_social_assets(self) -> None:
        """Copy social card + favicon PNGs to assets/, and favicon.ico to site root."""
        src_dir = Path(__file__).resolve().parent / "assets"
        assets_dir = self._root / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for name in ("social-card.png", "favicon-32.png", "apple-touch-icon.png"):
            src = src_dir / name
            if src.exists():
                (assets_dir / name).write_bytes(src.read_bytes())
                logger.debug("Copied social asset %s", name)
        ico_src = src_dir / "favicon.ico"
        if ico_src.exists():
            (self._root / "favicon.ico").write_bytes(ico_src.read_bytes())
            logger.debug("Copied favicon.ico to site root")

    def _write_feed(self, reports: list[dict]) -> None:
        """Render + write the Atom feed from the freshly sorted reports index
        — one entry per published report (1.5). Called from ``publish`` after
        the index is written, never from the asset copier: the feed is data,
        not a static asset."""
        (self._root / "feed.xml").write_text(
            _render_feed(reports, _site_url()), encoding="utf-8")
        logger.debug("Wrote feed.xml (%d entries)", len(reports))

    def _write(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote %s (%d B)", path, len(content))

    def _load_reports_index(self) -> list[dict]:
        p = self._root / "data" / "reports.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _write_reports_index(self, reports: list[dict]) -> None:
        p = self._root / "data" / "reports.json"
        p.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_claims_index(self) -> list[dict]:
        p = self._root / "data" / "claims.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _write_claims_index(self, claims: list[dict]) -> None:
        p = self._root / "data" / "claims.json"
        p.write_text(json.dumps(claims, indent=2, ensure_ascii=False), encoding="utf-8")

    def _report_meta(self, sr: SiteReport) -> dict:
        return {
            "id":                  sr.report_id,
            "date":                sr.date_str,
            # Publish stamp for the feed's per-entry <updated> (1.5).
            # Deterministic when the caller sets SiteReport.generated_at
            # (e.g. from artifact data); otherwise it is the dataclass
            # default — wall clock at SiteReport construction.
            "generated_at":        _iso_utc(sr.generated_at),
            "speaker":             sr.speaker,
            "role":                sr.role,
            "venue":               sr.venue,
            "claim_count":         len(sr.checkable_bundles),
            # P67.10: seat naming for the per-seat insights page, and the
            # set-aside count so the falsifiability ratio is derivable from
            # reports.json alone.
            "panel_roster":        dict(getattr(sr, "panel_roster", None) or {}),
            "triage_count":        len(getattr(sr, "characterization", None) or []),
            "verdict_distribution": sr.verdict_distribution,
            # 5-bucket coarse-axis distributions for lens-aware aggregate
            # rendering on the index page (and external consumers of
            # reports.json that want the Truthy-scale histogram). The
            # 6-bucket ``verdict_distribution`` above is kept for backward
            # compat with anyone already reading reports.json.
            "verdict_distribution_lenient": sr.verdict_distribution_lenient,
            "verdict_distribution_strict":  sr.verdict_distribution_strict,
            "model_agreement_rate": round(sr.model_agreement_rate, 3),
            "url":                 sr.report_url,
            "tier_counts":         _tier_counts_for_report(sr),
            # New decomposed speaker/speech fields
            "source_of_claims":                          sr.source_of_claims or sr.speaker,
            "source_of_claims_professional_public_title": sr.source_of_claims_professional_public_title or sr.role,
            "event":               sr.event,
            "channel":             sr.channel,
        }

    def _claim_meta(self, bundle: VerdictBundle, sr: SiteReport) -> dict:
        return {
            "id":                    bundle.claim.id,
            "report_id":             sr.report_id,
            "claim_text":            bundle.claim.text,
            # Surrounding transcript sentences ("prev || claim || next") — the
            # context the panel judged with and the pages now display
            # (2026-08-01). Empty on legacy bundles.
            "claim_context":         getattr(bundle.claim, "context", "") or "",
            "consensus_verdict":     bundle.consensus.consensus_verdict,
            "consensus_strength":    bundle.consensus.consensus_strength,
            # 5-bucket coarse-axis projections (Truthy scale). Default empty
            # strings on legacy bundles deserialize cleanly; downstream
            # consumers can detect "post-projection" data by checking for
            # non-empty ``coarse_lenient_label``.
            "coarse_lenient_label":   bundle.consensus.coarse_lenient_label,
            "coarse_lenient_strength": bundle.consensus.coarse_lenient_strength,
            "coarse_strict_label":     bundle.consensus.coarse_strict_label,
            "coarse_strict_strength":  bundle.consensus.coarse_strict_strength,
            "model_verdicts_summary": [
                {"adapter": mv.adapter_name, "label": mv.label.value,
                 "confidence": mv.confidence.value}
                for mv in bundle.model_verdicts
            ],
            # PCA pipeline provenance (empty on legacy multi-adapter bundles). Lets
            # downstream consumers reconstruct per-claim panel agreement — the tally
            # the reconciled-judge card collapses away. See VerdictProvenance.
            "provenance": {
                "layer_a_label":   bundle.consensus.provenance.layer_a_label,
                "layer_a_source":  bundle.consensus.provenance.layer_a_source,
                "layer_a_claim_type": bundle.consensus.provenance.layer_a_claim_type,
                "layer_a_claim_shape": getattr(bundle.consensus.provenance,
                                               "layer_a_claim_shape", ""),
                "panel_votes":     dict(bundle.consensus.provenance.panel_votes),
                "panel_split":     bundle.consensus.provenance.panel_split,
                "panel_escalated": bundle.consensus.provenance.panel_escalated,
                "crm114_stage1":   bundle.consensus.provenance.crm114_stage1,
                "crm114_final":    bundle.consensus.provenance.crm114_final,
                "panel_by_role":   {k: list(v) for k, v in
                                    getattr(bundle.consensus.provenance,
                                            "panel_by_role", {}).items()},
                # PR-A2.1: the T2.4 gate code and the rendered honest-abstention
                # sub-state, exported so the report chip's decomposition is
                # re-derivable from claims.json alone (consistency.py checks it).
                "evidence_gate":   getattr(bundle.consensus.provenance,
                                           "evidence_gate", ""),
                "self_sourced_only": _is_self_sourced_unverified(bundle),
            },
            "url": f"claims/{bundle.claim.id}.html",
        }

    def _compute_stats(self, reports: list[dict], claims: list[dict] | None = None) -> dict:
        # Canonical claim count is the claims index itself (remediation T0.7)
        # — every surfaced figure derives from the same source. The per-report
        # claim_count sum must reconcile; drift is surfaced loudly (the
        # consistency checker fails the build on it) rather than papered over.
        reports_claim_sum = sum(r.get("claim_count", 0) for r in reports)
        total_claims = len(claims) if claims else reports_claim_sum
        if claims and reports_claim_sum != len(claims):
            logger.warning(
                "claim-count drift: claims.json has %d entries but reports.json "
                "claim_counts sum to %d", len(claims), reports_claim_sum)
        # Site-wide consensus = CLAIM-WEIGHTED mean of per-report panel
        # agreement (remediation T0.1). The old path averaged per-claim
        # agreement from model_verdicts_summary, which under the PCA
        # pipeline holds a single reconciled pseudo-model that matches
        # consensus by construction — rendering "100% Model Consensus"
        # over reports whose real agreement was 47% and 78%.
        if reports_claim_sum:
            agree_rate = sum(
                r.get("model_agreement_rate", 0) * r.get("claim_count", 0)
                for r in reports
            ) / reports_claim_sum
        elif reports:
            agree_rate = sum(r.get("model_agreement_rate", 0) for r in reports) / len(reports)
        else:
            agree_rate = 0.0
        verdict_totals: dict[str, int] = {v: 0 for v in VERDICT_CSS}
        for r in reports:
            for label, cnt in r.get("verdict_distribution", {}).items():
                verdict_totals[label] = verdict_totals.get(label, 0) + cnt

        # 5-bucket coarse-axis aggregates for the lens-aware index. Both
        # axes are summed across whichever per-report fields exist; legacy
        # reports.json entries that predate the projection layer simply
        # contribute 0s. The renderer falls back to projecting from the
        # 6-bucket verdict_totals at render time when a particular report
        # is missing both coarse fields, keeping mixed-vintage indexes
        # internally consistent.
        verdict_totals_lenient: dict[str, int] = {v: 0 for v in COARSE_VERDICT_ORDER}
        verdict_totals_lenient["Models split"] = 0
        verdict_totals_strict: dict[str, int] = {v: 0 for v in COARSE_VERDICT_ORDER}
        verdict_totals_strict["Models split"] = 0
        for r in reports:
            for label, cnt in (r.get("verdict_distribution_lenient") or {}).items():
                verdict_totals_lenient[label] = verdict_totals_lenient.get(label, 0) + cnt
            for label, cnt in (r.get("verdict_distribution_strict") or {}).items():
                verdict_totals_strict[label] = verdict_totals_strict.get(label, 0) + cnt
        # Fallback projection for legacy reports.json entries that have only
        # the 6-bucket verdict_distribution: aggregation.project_dist per
        # axis (1.6) — its non-folding rule keeps a legacy "Models split"
        # fine bucket out of Unverifiable, unlike the old inline fold.
        for r in reports:
            if r.get("verdict_distribution_lenient") and r.get("verdict_distribution_strict"):
                continue
            fine = r.get("verdict_distribution") or {}
            if not r.get("verdict_distribution_lenient"):
                for label, cnt in _agg_project_dist(fine, "lenient").items():
                    verdict_totals_lenient[label] = (
                        verdict_totals_lenient.get(label, 0) + cnt
                    )
            if not r.get("verdict_distribution_strict"):
                for label, cnt in _agg_project_dist(fine, "strict").items():
                    verdict_totals_strict[label] = (
                        verdict_totals_strict.get(label, 0) + cnt
                    )

        # Distinct leaders reviewed
        distinct_leaders = len({
            r.get("source_of_claims") or r.get("speaker", "")
            for r in reports
            if r.get("source_of_claims") or r.get("speaker")
        })

        # Per-model agreement stats computed from claims index
        model_agree: dict[str, list[bool]] = {}
        per_claim_agree: list[float] = []
        if claims:
            for c in claims:
                consensus = c.get("consensus_verdict", "")
                mvs = c.get("model_verdicts_summary", [])
                if mvs:
                    n_agree = sum(1 for mv in mvs if mv.get("label") == consensus)
                    per_claim_agree.append(n_agree / len(mvs))
                    for mv in mvs:
                        adapter = mv.get("adapter", "")
                        model_agree.setdefault(adapter, []).append(mv.get("label") == consensus)

        # per_claim_agree is kept for the per-model divergence stats below,
        # but no longer feeds the site-wide consensus figure (see above).
        avg_consensus = agree_rate

        model_rates = {a: sum(v) / len(v) for a, v in model_agree.items() if v}
        mean_rate = sum(model_rates.values()) / len(model_rates) if model_rates else 0.0
        models_above = sorted(a for a, r in model_rates.items() if r > mean_rate)
        model_lowest = min(model_rates, key=lambda a: model_rates[a]) if model_rates else None
        # If all rates are equal, "most often diverging" is not meaningful
        if model_rates and len(set(round(v, 4) for v in model_rates.values())) == 1:
            model_lowest = None

        # Per-model panel insights — computed best-effort from the
        # claims index. None when claims are unavailable; renderers
        # treat None as "skip".
        insights = None
        if claims:
            try:
                from truthbot.publish.insights import compute_model_panel_insights
                reports_by_id = {r.get("id"): r for r in reports if r.get("id")}
                insights = compute_model_panel_insights(
                    claims, reports_by_id=reports_by_id
                )
            except Exception:  # pragma: no cover — defensive
                logger.debug("compute_model_panel_insights failed", exc_info=True)
                insights = None

        return {
            "total_speeches": len(reports),
            "total_leaders": distinct_leaders,
            "total_claims": total_claims,
            "model_agreement_rate": agree_rate,
            "avg_consensus": avg_consensus,
            "verdict_totals": verdict_totals,
            "verdict_totals_lenient": verdict_totals_lenient,
            "verdict_totals_strict":  verdict_totals_strict,
            "models_above_mean": models_above,
            "model_lowest": model_lowest,
            "insights": insights,
        }

    def summary(self) -> dict:
        """Return site stats dict for CLI printing."""
        reports = self._load_reports_index()
        claims  = self._load_claims_index()
        # Measure total site weight
        total_bytes = sum(
            f.stat().st_size
            for f in self._root.rglob("*.html")
        )
        total_bytes += sum(
            f.stat().st_size
            for f in self._root.rglob("*.json")
        )
        total_bytes += sum(
            f.stat().st_size
            for f in (self._root / "assets").rglob("*") if f.is_file()
        )
        return {
            "reports": len(reports),
            "claims":  len(claims),
            "total_kb": round(total_bytes / 1024, 1),
            "root":    str(self._root.resolve()),
        }


# ── Offline Statement-Triage backfill ─────────────────────────────────────────
# The immediate $0 win for the CURRENT live site: read a persisted PCA replay
# artifact (metrics/pca_runs/<run_id>.json), build a SiteReport from its meta +
# characterization stream, and render ONLY the Statement Triage page into an
# existing site root — no bundles, no LLM spend, no full re-publish.


def _site_report_from_artifact(artifact: dict) -> SiteReport:
    """Build a minimal SiteReport (meta + characterization only) from a persisted
    ``metrics/pca_runs/<run_id>.json`` artifact dict. Bundles are empty — this
    report is only ever used to render the Statement Triage page."""
    from datetime import datetime

    meta = artifact.get("meta", {}) or {}
    date_val = None
    if meta.get("date"):
        try:
            date_val = datetime.strptime(str(meta["date"]), "%Y-%m-%d")
        except Exception:
            date_val = None
    return SiteReport(
        report_id=str(artifact.get("run_id", "") or ""),
        speaker=meta.get("speaker", "") or "",
        role=meta.get("role", "") or "",
        date=date_val,
        venue=meta.get("venue", "") or "",
        transcript_source_url=meta.get("source_url", "") or "",
        bundles=[],
        characterization=list(artifact.get("characterization", []) or []),
        speech_id=str(meta.get("speech_id", "") or ""),
    )


def _match_existing_report_slug(site_root: Path, speaker: str, date_str: str) -> Optional[str]:
    """Find the published report slug for a speaker+date in an existing site root,
    by consulting data/reports.json. Returns None if no match / no index."""
    idx_path = site_root / "data" / "reports.json"
    if not idx_path.exists():
        return None
    try:
        reports = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for r in reports:
        r_speaker = r.get("source_of_claims") or r.get("speaker") or ""
        r_date = r.get("date_str") or r.get("date") or ""
        if r_speaker == speaker and str(r_date) == str(date_str):
            # slug is the report url stem: reports/<slug>.html
            url = r.get("url") or r.get("report_url") or ""
            stem = Path(str(url)).name
            if stem.endswith(".html"):
                return stem[: -len(".html")]
            if r.get("slug"):
                return str(r["slug"])
    return None


def backfill_statement_triage(
    artifact_path: str | Path,
    site_root: str | Path,
    *,
    report_slug: Optional[str] = None,
) -> Optional[Path]:
    """Render the Statement Triage page for a persisted PCA run into an existing site.

    Parameters
    ----------
    artifact_path:
        Path to a ``metrics/pca_runs/<run_id>.json`` replay artifact.
    site_root:
        Existing published site root (e.g. ``site-pca/``). The triage page is
        written under ``<site_root>/reports/``.
    report_slug:
        Filename stem of the already-published report page (without ``.html``),
        so the triage page is named ``<report_slug>-triage.html`` and its
        breadcrumb links back correctly. When omitted, it is looked up from the
        site's ``data/reports.json`` by matching speaker + date; if that fails,
        it falls back to the slug derived from the artifact's own run_id.

    Returns the path to the written triage page, or ``None`` when the artifact
    carries no characterization stream (nothing to render)."""
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    site_root = Path(site_root)
    sr = _site_report_from_artifact(artifact)
    if not sr.characterization:
        return None

    # Resolve the slug the existing report page uses, so links line up.
    slug = report_slug or _match_existing_report_slug(site_root, sr.speaker, sr.date_str)
    if slug:
        # Point the SiteReport's derived slugs at the existing report by faking
        # report_id so report_slug == slug (report_slug = "<date>-<speaker>-<id[:6]>").
        # Simplest robust path: render with an explicit slug override.
        html = _render_statement_triage_with_slug(sr, slug)
        triage_stem = f"{slug}-triage"
    else:
        html = _render_statement_triage(sr)
        triage_stem = sr.triage_slug

    out_dir = site_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{triage_stem}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path.resolve()


def _render_statement_triage_with_slug(site_report: SiteReport, report_slug: str) -> str:
    """Same as :func:`_render_statement_triage`, but the breadcrumb back-link
    targets an explicit existing ``report_slug`` rather than the one derived from
    the SiteReport's own (possibly mismatched) report_id."""
    class _Proxy:
        # Lightweight shim: forwards everything to site_report, overrides report_slug.
        def __init__(self, base, slug):
            object.__setattr__(self, "_base", base)
            object.__setattr__(self, "_slug", slug)

        def __getattr__(self, name):
            if name == "report_slug":
                return object.__getattribute__(self, "_slug")
            return getattr(object.__getattribute__(self, "_base"), name)

    return _render_statement_triage(_Proxy(site_report, report_slug))


def _cli_backfill_statement_triage(argv: Optional[list[str]] = None) -> int:
    """CLI: render a Statement Triage page from a persisted artifact into a site.

    Usage:
        python -m truthbot.publish.site \\
            --artifact metrics/pca_runs/<run_id>.json \\
            --site-root site-pca/ \\
            [--report-slug 2026-02-24-donald-trump-0c33d1]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="truthbot.publish.site",
        description="Offline Statement-Triage backfill (renders one page, no re-run).",
    )
    parser.add_argument("--artifact", required=True,
                        help="Path to metrics/pca_runs/<run_id>.json")
    parser.add_argument("--site-root", required=True,
                        help="Existing site root (e.g. site-pca/)")
    parser.add_argument("--report-slug", default=None,
                        help="Slug of the published report page (without .html). "
                             "Auto-detected from data/reports.json if omitted.")
    args = parser.parse_args(argv)

    out = backfill_statement_triage(
        args.artifact, args.site_root, report_slug=args.report_slug
    )
    if out is None:
        print("No characterization stream in artifact; nothing to render.")
        return 1
    print(f"Wrote Statement Triage page: {out}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_cli_backfill_statement_triage())
