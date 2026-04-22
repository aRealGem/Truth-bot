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
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from truthbot.models import VerdictBundle, VerdictLabel

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
    ("Government",  ".gov, .mil — BLS, BEA, CBO, Census, CDC, etc.",          "Highest"),
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
            label = b.consensus.consensus_label.value
            dist[label] = dist.get(label, 0) + 1
        return dist

    @property
    def model_agreement_rate(self) -> float:
        bundles = self.checkable_bundles
        if not bundles:
            return 0.0
        agreed = sum(1 for b in bundles if b.consensus.agreement)
        return agreed / len(bundles)

    @property
    def report_slug(self) -> str:
        short = self.report_id[:6]  # first 6 chars of UUID — unique per run
        return f"{self.date_str}-{_slug(self.speaker)}-{short}"

    @property
    def report_url(self) -> str:
        return f"reports/{self.report_slug}.html"

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


# Provider brand names + the production default model for each adapter.
# Used as fallback labels when a ModelVerdict lacks an explicit model_id
# (e.g. older reports rehydrated after a schema migration).
_ADAPTER_BRAND = {
    "anthropic": "Anthropic",
    "openai":    "OpenAI",
    "gemini":    "Google",
}
_ADAPTER_DEFAULT_MODEL = {
    "anthropic": "claude-opus-4-7",
    "openai":    "gpt-5.4-pro",
    "gemini":    "gemini-2.5-pro",
}
_MODEL_TOKEN_UPPER = {"gpt", "ai"}


def _prettify_model_id(mid: str) -> str:
    """Turn 'claude-opus-4-7' → 'Claude Opus 4.7', 'gpt-5.4-pro' → 'GPT 5.4 Pro'."""
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


# Categories whose display labels already contain a qualifier word.
# Never prepend "Mostly" or "Largely" to these — it reads as double-qualified.
_ALREADY_QUALIFIED: frozenset[str] = frozenset({"Mostly True"})


def _headline_verdict(dist: dict[str, int]) -> tuple[str, str]:
    """
    Compute the headline verdict label and CSS class for a report.
    Returns (label_text, css_class).

    Rules (applied in order):
      - 0 claims                    → 'No claims evaluated' / neutral
      - 2+ categories tie for max   → 'Mixed verdict' / neutral
      - dominant ≥ 60%              → 'Largely {label}' / vt-{slug}
      - dominant ≥ 40%              → label (if already qualified) or 'Mostly {label}'
      - otherwise                   → 'Mixed verdict' / neutral

    The tie check prevents max() from silently picking a winner when the data
    is genuinely split (e.g. 2 True + 2 False).  The ALREADY_QUALIFIED guard
    prevents double-prefixing labels like "Mostly True" into "Mostly Mostly True".
    """
    total = sum(dist.values())
    if total == 0:
        return "No claims evaluated", "neutral"
    max_count = max(dist.values())
    # Tie: two or more categories share the top count → always Mixed
    if sum(1 for v in dist.values() if v == max_count) > 1:
        return "Mixed verdict", "neutral"
    max_label = max(dist, key=lambda k: dist[k])
    max_pct = max_count / total
    css = _verdict_css(max_label)
    if max_label in _ALREADY_QUALIFIED:
        # Label is self-descriptive; only apply it if it genuinely dominates
        if max_pct >= 0.40:
            return max_label, f"vt-{css}"
        return "Mixed verdict", "neutral"
    if max_pct >= 0.60:
        return f"Largely {max_label}", f"vt-{css}"
    elif max_pct >= 0.40:
        return f"Mostly {max_label}", f"vt-{css}"
    else:
        return "Mixed verdict", "neutral"


def _tier_badge(url: str) -> str:
    """Return an evidence-tier span for a source URL."""
    lower = url.lower()
    if any(d in lower for d in (".gov", ".mil")):
        return '<span class="evidence-tier tier-gov">T1·Gov</span>'
    if any(d in lower for d in ("apnews.com", "reuters.com")):
        return '<span class="evidence-tier tier-news">T2·Wire</span>'
    if any(d in lower for d in ("nytimes.com", "washingtonpost.com", "bbc.", "npr.org",
                                  "nbcnews.com", "cbsnews.com", "abcnews.go.com")):
        return '<span class="evidence-tier tier-news">T3·News</span>'
    if any(d in lower for d in ("politifact.com", "factcheck.org", "snopes.com", "fullfact.org")):
        return '<span class="evidence-tier tier-fc">T5·FC</span>'
    return '<span class="evidence-tier tier-other">T6</span>'


def _evidence_list_html(urls: list[str]) -> str:
    """Render evidence URLs as evidence-list structure."""
    if not urls:
        return '<p style="font-size:0.88rem;color:var(--ink-muted)">No sources retrieved.</p>'
    items = []
    for url in urls[:10]:
        badge = _tier_badge(url)
        short = url.replace("https://", "").replace("http://", "")
        if len(short) > 80:
            short = short[:77] + "…"
        items.append(
            f'<li><span class="ev-mark">→</span>{badge}'
            f'<a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(short)}</a></li>'
        )
    return f'<ul class="evidence-list">{"".join(items)}</ul>'


def _verdict_bar_html(dist: dict[str, int], bar_class: str = "vp-bar") -> str:
    """Render a full 6-category verdict bar + legend."""
    total = sum(dist.values()) or 1
    segs = []
    for label in VERDICT_ORDER:
        count = dist.get(label, 0)
        if count == 0:
            continue
        pct = count / total * 100
        css = _verdict_css(label)
        segs.append(
            f'<div class="seg v-{css}" style="width:{pct:.1f}%" '
            f'title="{_esc(label)}: {count}">{count}</div>'
        )
    parts_aria = [f"{dist.get(l,0)} {l}" for l in VERDICT_ORDER if dist.get(l, 0) > 0]
    aria = "Verdict distribution: " + ", ".join(parts_aria)
    bar_html = (
        f'<div class="{bar_class}" role="img" aria-label="{_esc(aria)}">'
        f'{"".join(segs)}</div>'
    )
    legend_items = []
    for label in VERDICT_ORDER:
        count = dist.get(label, 0)
        css = _verdict_css(label)
        zero_cls = " zero" if count == 0 else ""
        legend_items.append(
            f'<div class="legend-item{zero_cls}">'
            f'<span class="swatch v-{css}"></span>'
            f'{_esc(label)} <span class="ct">{count}</span>'
            '</div>'
        )
    legend_html = f'<div class="vp-legend">{"".join(legend_items)}</div>'
    return bar_html + "\n" + legend_html


def _prompt_hash() -> str:
    try:
        from truthbot.verify.adapters.base import SYNTHESIS_SYSTEM
        return hashlib.sha256(SYNTHESIS_SYSTEM.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


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
    'Tap'
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
    tv = site_report.truthy_verdict
    mood = tv.mood
    state_map = {"happy": "true", "iffy": "iffy", "sad": "lie"}
    svg_state = "state-" + state_map.get(mood, "iffy")

    claim_count = len(site_report.checkable_bundles)
    model_count = len({mv.adapter_name for b in site_report.checkable_bundles for mv in b.model_verdicts})
    agree_rate  = site_report.model_agreement_rate
    dist        = site_report.verdict_distribution
    headline, h_cls = _headline_verdict(dist)

    total   = sum(dist.values()) or 1
    max_lbl = max(dist, key=lambda k: dist[k]) if dist else ""
    ratio_text = str(dist.get(max_lbl, 0)) + " of " + str(total) + " claims rated " + max_lbl.lower() if max_lbl else str(total) + " claims checked"

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

    text_col = (
        '<div class="vp-text-col">'
        + '<div>'
        + '<div class="vp-verdict ' + h_cls + '">' + _esc(headline) + '</div>'
        + '<div class="vp-ratio">' + _esc(ratio_text) + '</div>'
        + '</div>'
        + '</div>'
    )

    panel_stats_html = (
        '  <div class="stats stats-4">\n'
        '    <div class="stat">'
        + _icon_svg(_ICON_BODY_CLAIMS, size=32)
        + '<div class="num">' + str(claim_count) + '</div>'
        + '<div class="lbl">Claims Checked</div></div>\n'
        '    <div class="stat">'
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

    bar_html = _verdict_bar_html(dist)

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

    return (
        '<section class="verdict-panel">\n'
        + '  <div class="vp-headline">' + text_col + widget + '</div>\n'
        + panel_stats_html
        + '  <div class="vp-bar-wrap">' + bar_html + '</div>\n'
        + source_row_html
        + '</section>\n'
    )


def _status_bar(model_count: int = 0, stamp: Optional[str] = None) -> str:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model_str = f"{model_count} Model{'s' if model_count != 1 else ''}" if model_count else "Multi-model"
    return (
        '<div class="status-bar">\n'
        '  <div class="row">\n'
        '    <span class="live">Operational</span>\n'
        f'    <span>Pipeline v{PIPELINE_VERSION}</span>\n'
        f'    <span>{model_str}</span>\n'
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


def _page_index(title: str, body: str, footer: str = "", model_count: int = 0) -> str:
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
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}">\n'
        # Tint mobile browser chrome to match the page background.
        # Keep in sync with --bg in CSS.
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
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


def _page_report(title: str, body: str, footer: str = "", model_count: int = 0, analyzed_at: Optional[str] = None) -> str:
    stamp = f"Analyzed {analyzed_at}" if analyzed_at else "Analyzed " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
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
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}">\n'
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
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


def _page_about(title: str, body: str, footer: str = "") -> str:
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
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}">\n'
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
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


def _page_truthy(title: str, body: str, footer: str = "") -> str:
    """Fun / mascot page shell — same chrome as about, no truthbot.js (inline _TRUTHY_FUN_SCRIPT in body)."""
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
        f'  <meta name="generator" content="truth-bot {PIPELINE_VERSION}">\n'
        '  <meta name="theme-color" content="#fafaf9">\n'
        '  <meta name="color-scheme" content="light">\n'
        f'  <title>{_esc(title.removesuffix(" — truth-bot"))} — truth-bot</title>\n'
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


def _claim_card(bundle: VerdictBundle, idx: int, total: int, rel: str = "../", standalone: bool = False) -> str:
    claim = bundle.claim
    consensus = bundle.consensus
    label = consensus.consensus_label.value
    css = _verdict_css(label)
    n = str(idx).zfill(2)

    context_html = ''
    if claim.category:
        context_html = f'<div class="claim-context"><span>{_esc(claim.category)}</span></div>'

    caveat_html = ''
    caveats = []
    seen = set()
    for mv in bundle.model_verdicts:
        cav = mv.caveats.strip()
        if cav and cav not in seen:
            seen.add(cav)
            caveats.append(cav)
    if caveats:
        caveat_html = (
            '<div class="caveat">'
            '  <div class="caveat-label">Caveat</div>'
            f'  <div class="caveat-text">{_esc(" ".join(caveats))}</div>'
            '</div>'
        )

    majority_label = label

    triage_badge = ""
    if getattr(bundle, "triage_decision", False):
        triage_badge = (
            '<span class="claim-pill triage-only" '
            'title="Unanimous high-confidence triage; frontier models were skipped">Triage</span>'
        )

    def _reasoning_paragraphs(text: str) -> str:
        if not text:
            return ""
        parts = [seg.strip() for seg in re.split(r"\n\s*\n", text.strip()) if seg.strip()]
        if not parts:
            return ""
        return "".join(f'<p>{_esc(seg)}</p>' for seg in parts)

    model_cards = []
    agreeing = 0
    all_urls: list[str] = []
    seen_urls: set[str] = set()
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
            dissent = " dissent" if mv_label != majority_label else ""
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

    total_models = len(bundle.model_verdicts)
    dissenting = total_models - agreeing
    dissent_note = f" · {dissenting} dissent{'s' if dissenting > 1 else ''}" if dissenting else ""

    evidence_html = (
        '<details class="evidence-details">'
        '  <summary class="evidence-summary">Combined evidence / sources list</summary>'
        '  <div class="evidence">'
        f'{_evidence_list_html(all_urls[:10])}'
        '  </div>'
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
        f'  <span class="claim-pill v-{css}">{_esc(label)}</span>'
        f'  {triage_badge}'
        '</div>'
        '<div class="claim-body">'
        f'  <blockquote class="claim-quote">"{_esc(claim.text)}"</blockquote>'
        f'  {context_html}'
        f'  {caveat_html}'
        '  <div class="models">'
        '    <div class="models-head">'
        '      <span class="models-label">Model consensus</span>'
        f'      <span class="models-agreement"><span class="num">{agreeing} of {total_models}</span> agree{_esc(dissent_note)}</span>'
        '    </div>'
        f'    <div class="model-grid">{"".join(model_cards)}</div>'
        '  </div>'
        f'  {evidence_html}'
        '  <div class="claim-foot">'
        f'    <a href="#claim-{idx}" class="permalink">claim-{idx}</a>'
        + back_links_html
        + f'    <span>Last verified {gen_ts}</span>'
        '  </div>'
        '</div>'
        '</article>'
    )

def _toc(bundles: list[VerdictBundle]) -> str:
    items = []
    for i, b in enumerate(bundles, 1):
        label = b.consensus.consensus_label.value
        css = _verdict_css(label)
        items.append(
            f'<a class="toc-item" href="#claim-{i}">'
            f'  <span class="toc-num">{str(i).zfill(2)}</span>'
            f'  <span class="toc-pill v-{css}">{_esc(label)}</span>'
            f'  <span class="toc-text">"{_esc(b.claim.text)}"</span>'
            '  <span class="toc-jump">↓</span>'
            '</a>'
        )
    return '<nav class="toc">' + "".join(items) + '</nav>'


def _report_card(r: dict) -> str:
    dist = r.get("verdict_distribution", {})
    claim_count = r.get("claim_count", 0)
    headline, cls = _headline_verdict(dist)
    total = sum(dist.values()) or 1
    max_label = max(dist, key=lambda k: dist[k]) if dist else ""
    ratio = f"{dist.get(max_label, 0)} of {total} claims" if max_label else f"{claim_count} claims"

    segs = []
    for label in VERDICT_ORDER:
        count = dist.get(label, 0)
        if not count:
            continue
        segs.append(f'<div class="seg v-{_verdict_css(label)}" style="width:{count/total*100:.1f}%"></div>')

    counts = []
    for label in VERDICT_ORDER:
        count = dist.get(label, 0)
        if count:
            counts.append(
                f'<div class="ct"><span class="swatch v-{_verdict_css(label)}"></span>'
                f'{_esc(label)} <span class="n">{count}</span></div>'
            )

    meta_bits = []
    if r.get("date"):
        meta_bits.append(_esc(r["date"]))
    if r.get("venue"):
        meta_bits.append(_esc(r["venue"]))
    meta = '<span class="sep">·</span>'.join(meta_bits)

    return (
        f'<a href="{_esc(r.get("url", "#"))}" class="report">'
        '  <div class="report-top">'
        '    <div>'
        f'      <div class="report-headline">{_esc(r.get("speaker", ""))}</div>'
        f'      <div class="report-meta">{meta}</div>'
        '    </div>'
        '    <div class="verdict-pill">'
        f'      <span class="label {cls}">{_esc(headline)}</span>'
        f'      <span class="ratio">{_esc(ratio)}</span>'
        '    </div>'
        '  </div>'
        f'  <div class="report-bar">{"".join(segs)}</div>'
        f'  <div class="report-counts">{"".join(counts)}</div>'
        '  <div class="report-cta">'
        f'    <span class="src">{claim_count} claim{"s" if claim_count != 1 else ""}</span>'
        '    <span class="read">Read full report →</span>'
        '  </div>'
        '</a>'
    )


def _agg_bar(verdict_totals: dict[str, int]) -> str:
    total = sum(verdict_totals.values()) or 1
    segs = []
    legend = []
    for label in VERDICT_ORDER:
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
    aria = ", ".join(f"{verdict_totals.get(l,0)} {l}" for l in VERDICT_ORDER if verdict_totals.get(l,0))
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

/* Slim verdict bar inside a report card (vs. the chunky one in the verdict panel) */
.report-bar {
  display: flex;
  height: 6px;
  overflow: hidden;
  margin: 0.5rem 0 1rem;
}
.report-bar .seg { transition: filter 200ms ease; }

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
/* Text paint */
.vt-true         { color: var(--v-true); }
.vt-mostly-true  { color: var(--v-mostly-true); }
.vt-exaggerated  { color: var(--v-exaggerated); }
.vt-misleading   { color: var(--v-misleading); }
.vt-false        { color: var(--v-false); }
.vt-unverifiable { color: var(--v-unverifiable); }


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

  /* Index aggregate stats — single column (report .stats.stats-4 uses 700/480 breakpoints) */
  .stats:not(.stats-4) { grid-template-columns: 1fr; }
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
       no licensing, no network round-trips.
       AudioContext is lazily created on first user gesture (browsers
       block autoplay otherwise). All sounds resolve in <500ms.
       ──────────────────────────────────────────────────────────── */
    var audioCtx = null;
    function getCtx() {
      if (!audioCtx) {
        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) { return null; }
      }
      if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
      return audioCtx;
    }

    // Happy: bright rising arpeggio (C5 → E5 → G5 → C6) with square wave
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

    // Confused: triangle wave bending up to ~620Hz then dropping to ~330Hz
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

    // Sad: descending minor third (G4 → Eb4) with downward pitch bend on each note
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

    /* ─── Speak handler ─── */
    function speak() {
      var match = mascot.className.match(/state-(true|iffy|lie)/);
      if (!match) return;
      var state = match[1];
      var fn = soundMap[state];
      if (fn) fn();
      mascot.classList.add('speaking');
      setTimeout(function() { mascot.classList.remove('speaking'); }, 700);
    }

    /* ─── Initialize ─── */
    var mood = widget.getAttribute('data-mood') || 'iffy';
    var stateMap = { happy: 'true', iffy: 'iffy', sad: 'lie' };
    setState(stateMap[mood] || 'iffy');

    widget.addEventListener('click', speak);
    widget.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        speak();
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
        '<span class="how-text">Each claim is checked by multiple AI models using primary sources</span>'
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
    footer = (
        '<span>Last updated: ' + now + '</span>'
        + '<span>Pipeline v' + PIPELINE_VERSION
        + ' · <a href="' + GITHUB_URL + '" target="_blank" rel="noopener">GitHub</a></span>'
    )
    return _page_index("Latest Reports", body, footer)


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
        _claim_card(b, i, len(site_report.checkable_bundles), rel="../")
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

    _model_count = len({mv.adapter_name for b in site_report.checkable_bundles for mv in b.model_verdicts})
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

    body = (
        hero_html
        + _verdict_panel(site_report)
        + toc_section_head
        + toc_html
        + '<div class="section-head">'
        + '<span>Claims, in order spoken</span>'
        + '<span class="sub">Anchor links shareable</span>'
        + '</div>'
        + claim_blocks
        + methodology_html
    )
    footer = (
        '<span>truth-bot · pipeline v' + PIPELINE_VERSION + '</span>'
        + '<span>Source: <a href="' + GITHUB_URL + '" target="_blank" rel="noopener">'
        + 'github.com/aRealGem/Truth-bot</a></span>'
    )
    return _page_report(
        _esc(site_report.speaker) + " — " + _esc(site_report.display_date),
        body,
        footer=footer,
    )


def _render_claim_page(bundle: VerdictBundle, site_report: SiteReport) -> str:
    """Render a standalone per-claim permalink page."""
    report_url = f"../reports/{site_report.report_slug}.html"
    body = (
        f'<div class="breadcrumb">'
        f'<a href="../index.html">Reports</a> › '
        f'<a href="{report_url}">{_esc(site_report.speaker)} — '
        f'{_esc(site_report.display_date)}</a> › Claim</div>'
        f'{_claim_card(bundle, 1, 1, rel="../", standalone=True)}'
    )
    phash = _prompt_hash()
    gen_ts = site_report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    footer = (
        f'<span>truth-bot · pipeline v{PIPELINE_VERSION}</span>'
        f'<span>Source: <a href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'github.com/aRealGem/Truth-bot</a></span>'
    )
    return _page_report(f"Claim: {bundle.claim.text[:60]}", body, footer=footer)


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
    footer = (
        '<span><a href="./index.html">Back to reports</a></span>'
        f'<span>Pipeline v{PIPELINE_VERSION} &middot; '
        f'<a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a></span>'
    )
    return _page_truthy("Meet Truthy", body, footer)


def _render_about() -> str:
    """Render the about/method page."""
    try:
        from truthbot.verify.adapters.base import SYNTHESIS_SYSTEM
        prompt_text = SYNTHESIS_SYSTEM
    except Exception:
        prompt_text = "(prompt unavailable)"

    phash = hashlib.sha256(prompt_text.encode()).hexdigest()[:8]

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
        "<li><strong>Anthropic</strong> claude-opus-4-7 — primary verifier</li>"
        "<li><strong>OpenAI</strong> gpt-5.4-pro — pending API availability</li>"
        "<li><strong>Google</strong> gemini-2.5-pro — pending API key</li>"
        "<li><strong>xAI</strong> grok-4 — pending API key</li>"
        "</ul>"
    )

    limitations = (
        "<ul>"
        "<li><strong>Small corpus:</strong> Each claim is verified independently with no "
        "cross-claim context. Recurring rhetoric may be rated inconsistently across speeches.</li>"
        "<li><strong>Hallucinated citations:</strong> Language models can fabricate plausible-"
        "looking URLs. All cited sources should be independently verified before drawing "
        "conclusions.</li>"
        "<li><strong>Training-data bias:</strong> Model verdicts may reflect the political "
        "slant of their training data. The multi-model consensus is designed to partially "
        "mitigate this, but systematic bias in all four providers would not be caught.</li>"
        "<li><strong>Temporal grounding:</strong> Web search retrieves recent results but "
        "may miss updated statistics or corrections published after the pipeline run. "
        "The generation timestamp on each report indicates when verification occurred.</li>"
        "</ul>"
    )

    body = (
        f'<h2>About truth-bot</h2><hr class="rule">'
        f'<h3>What this is</h3>'
        f'<p>truth-bot is an automated political fact-checker that decomposes speeches and '
        f'public statements into atomic, individually verifiable claims, then runs each claim '
        f'through multiple large language models simultaneously. Each model performs live web '
        f'searches and returns a structured verdict. The results are aggregated into a consensus '
        f'verdict with an explicit strength score.</p>'
        f'<h3 style="margin-top:1.5rem">How verdicts are produced</h3>'
        f'<p>Each claim is sent to all configured LLM providers in parallel. Every provider '
        f'is instructed to search Tier 1 government sources first (BLS, BEA, CBP, etc.) before '
        f'citing secondary sources. The providers return a structured JSON verdict including a '
        f'label, confidence level, explanation, source URLs, and a self-reported caveats field '
        f'flagging source-quality gaps.</p>'
        f'<p style="margin-top:0.75rem">Consensus is computed by majority vote. '
        f'Three or more models returning the same label = "Strong consensus." '
        f'Two models agreeing = "Weak consensus." No majority = "Models split."</p>'
        f'<h3 style="margin-top:1.5rem">Models</h3>'
        f'{models_list}'
        f'<h3 style="margin-top:1.5rem">Source tier hierarchy</h3>'
        f'{tier_table}'
        f'<h3 style="margin-top:1.5rem">Known limitations</h3>'
        f'{limitations}'
        f'<h3 style="margin-top:1.5rem">Full verdict prompt (hash: {phash})</h3>'
        f'<p class="dim">Verbatim prompt sent to each model for verdict synthesis.</p>'
        f'<pre>{_esc(prompt_text)}</pre>'
        f'<hr class="rule-light">'
        f'<p class="dim"><a href="{GITHUB_URL}" target="_blank">GitHub</a> · '
        f'Pipeline v{PIPELINE_VERSION}</p>'
    )
    footer = (
        f'<span>truth-bot · pipeline v{PIPELINE_VERSION}</span>'
        f'<span>Source: <a href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'github.com/aRealGem/Truth-bot</a></span>'
    )
    return _page_about("About", body, footer=footer)


def _render_404() -> str:
    body = (
        '<h2>404 — Page not found</h2>'
        '<p class="dim">The page you requested does not exist.</p>'
        '<p><a href="index.html">Return to reports</a></p>'
    )
    return _page_about("404 Not Found", body)


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

    def __init__(self, site_root: Optional[str | Path] = None) -> None:
        import os
        if site_root:
            self._root = Path(site_root)
        else:
            self._root = Path(os.environ.get("TRUTHBOT_SITE_ROOT", "./site"))

    # ── Public API ────────────────────────────────────────────────────────────

    def publish(self, site_report: SiteReport) -> Path:
        """
        Generate/update all site files for a new or updated report.

        Returns the absolute path to the report HTML page.
        """
        self._ensure_structure()
        self._copy_assets()

        # Write report page
        report_html = _render_report(site_report)
        report_path = self._root / "reports" / f"{site_report.report_slug}.html"
        self._write(report_path, report_html)

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

        self._write_reports_index(reports_index)
        self._write_claims_index(claims_index)

        # Regenerate index
        stats = self._compute_stats(reports_index, claims_index)
        index_html = _render_index(reports_index, stats)
        self._write(self._root / "index.html", index_html)

        # About + 404 (regenerate on each publish for prompt-hash freshness)
        self._write(self._root / "about.html", _render_about())
        self._write(self._root / "truthy.html", _render_truthy())
        self._write(self._root / "404.html",   _render_404())

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
            "speaker":             sr.speaker,
            "role":                sr.role,
            "venue":               sr.venue,
            "claim_count":         len(sr.checkable_bundles),
            "verdict_distribution": sr.verdict_distribution,
            "model_agreement_rate": round(sr.model_agreement_rate, 3),
            "url":                 sr.report_url,
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
            "consensus_verdict":     bundle.consensus.consensus_verdict,
            "consensus_strength":    bundle.consensus.consensus_strength,
            "model_verdicts_summary": [
                {"adapter": mv.adapter_name, "label": mv.label.value,
                 "confidence": mv.confidence.value}
                for mv in bundle.model_verdicts
            ],
            "url": f"claims/{bundle.claim.id}.html",
        }

    def _compute_stats(self, reports: list[dict], claims: list[dict] | None = None) -> dict:
        total_claims = sum(r.get("claim_count", 0) for r in reports)
        if reports:
            agree_rate = sum(r.get("model_agreement_rate", 0) for r in reports) / len(reports)
        else:
            agree_rate = 0.0
        verdict_totals: dict[str, int] = {v: 0 for v in VERDICT_CSS}
        for r in reports:
            for label, cnt in r.get("verdict_distribution", {}).items():
                verdict_totals[label] = verdict_totals.get(label, 0) + cnt

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

        avg_consensus = (sum(per_claim_agree) / len(per_claim_agree)
                         if per_claim_agree else agree_rate)

        model_rates = {a: sum(v) / len(v) for a, v in model_agree.items() if v}
        mean_rate = sum(model_rates.values()) / len(model_rates) if model_rates else 0.0
        models_above = sorted(a for a, r in model_rates.items() if r > mean_rate)
        model_lowest = min(model_rates, key=lambda a: model_rates[a]) if model_rates else None
        # If all rates are equal, "most often diverging" is not meaningful
        if model_rates and len(set(round(v, 4) for v in model_rates.values())) == 1:
            model_lowest = None

        return {
            "total_speeches": len(reports),
            "total_leaders": distinct_leaders,
            "total_claims": total_claims,
            "model_agreement_rate": agree_rate,
            "avg_consensus": avg_consensus,
            "verdict_totals": verdict_totals,
            "models_above_mean": models_above,
            "model_lowest": model_lowest,
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
