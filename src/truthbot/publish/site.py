"""
publish/site.py — Static site publisher for truth-bot.

Generates a complete newspaper-aesthetic static site from VerdictBundle objects.
All HTML templates are inline Python strings; no external template files required.

Output structure:
    {SITE_ROOT}/
        index.html
        about.html
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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from truthbot.models import VerdictBundle, VerdictLabel

logger = logging.getLogger(__name__)

# ── Verdict presentation constants ────────────────────────────────────────────

VERDICT_COLOR: dict[str, str] = {
    "True":          "#2e7d32",
    "Mostly True":   "#558b2f",
    "Misleading":    "#e65100",
    "Exaggerated":   "#bf360c",
    "False":         "#b71c1c",
    "Unverifiable":  "#546e7a",
}

VERDICT_BG: dict[str, str] = {
    "True":          "#e8f5e9",
    "Mostly True":   "#f1f8e9",
    "Misleading":    "#fff3e0",
    "Exaggerated":   "#fbe9e7",
    "False":         "#ffebee",
    "Unverifiable":  "#eceff1",
}

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

STRENGTH_COLOR: dict[str, str] = {
    "strong": "#2e7d32",
    "weak":   "#f57c00",
    "none":   "#b71c1c",
    "single": "#546e7a",
}

TIER_TABLE = [
    ("Government",  ".gov, .mil — BLS, BEA, CBO, Census, CDC, etc.",          "Highest"),
    ("Wire",        "AP, Reuters",                                              "High"),
    ("Established", "NYT, WaPo, BBC, NPR, CBS, NBC, ABC",                      "Medium-High"),
    ("Academic",    "Peer-reviewed journals, university presses",               "Medium-High"),
    ("Fact-check",  "PolitiFact, FactCheck.org, Snopes, FullFact",             "Medium"),
    ("Other",       "Blogs, opinion sites, social media, unverified sources",   "Low"),
]

GITHUB_URL = "https://github.com/aRealGem/Truth-bot"
PIPELINE_VERSION = "0.2.0"


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SiteReport:
    """All data needed to render a full report page."""
    report_id: str
    speaker: str
    role: str
    date: Optional[datetime]
    venue: str
    transcript_source_url: str
    bundles: list[VerdictBundle]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
        dist: dict[str, int] = {v: 0 for v in VERDICT_COLOR}
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
        return f"{self.date_str}-{_slug(self.speaker)}"

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
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _chip(text: str, bg: str, color: str = "#fff", small: bool = False) -> str:
    sz = "0.72rem" if small else "0.82rem"
    return (f'<span class="chip" style="background:{bg};color:{color};'
            f'font-size:{sz}">{_esc(text)}</span>')


def _verdict_chip(label_str: str, large: bool = False) -> str:
    bg = VERDICT_COLOR.get(label_str, "#546e7a")
    emoji = VERDICT_EMOJI.get(label_str, "")
    sz = "1rem" if large else "0.82rem"
    pad = "0.35rem 0.9rem" if large else "0.2rem 0.6rem"
    return (f'<span class="chip verdict-chip" style="background:{bg};color:#fff;'
            f'font-size:{sz};padding:{pad}">{emoji} {_esc(label_str)}</span>')


def _strength_chip(strength: str, agreeing: int, total: int) -> str:
    label = STRENGTH_LABEL.get(strength, strength)
    color = STRENGTH_COLOR.get(strength, "#546e7a")
    count = f" {agreeing}/{total}" if total > 1 else ""
    return _chip(f"{label}{count}", color)


def _stacked_bar(dist: dict[str, int]) -> str:
    total = sum(dist.values()) or 1
    segs = []
    for label, count in dist.items():
        if count == 0:
            continue
        pct = count / total * 100
        bg = VERDICT_COLOR.get(label, "#546e7a")
        segs.append(
            f'<div class="bar-seg" style="width:{pct:.1f}%;background:{bg}" '
            f'title="{_esc(label)}: {count}"><span>{count}</span></div>'
        )
    return f'<div class="stacked-bar">{"".join(segs)}</div>'


def _tier_badge(url: str) -> str:
    lower = url.lower()
    if any(d in lower for d in (".gov", ".mil")):
        return _chip("T1·Gov", "#1565c0", small=True)
    if any(d in lower for d in ("apnews.com", "reuters.com")):
        return _chip("T2·Wire", "#1b5e20", small=True)
    if any(d in lower for d in ("nytimes.com", "washingtonpost.com", "bbc.", "npr.org",
                                  "nbcnews.com", "cbsnews.com", "abcnews.go.com")):
        return _chip("T3·News", "#4a148c", small=True)
    if any(d in lower for d in ("politifact.com", "factcheck.org", "snopes.com", "fullfact.org")):
        return _chip("T5·FC", "#e65100", small=True)
    return _chip("T6", "#546e7a", small=True)


def _evidence_links(urls: list[str]) -> str:
    if not urls:
        return "<p class='dim'>No sources retrieved.</p>"
    items = []
    for url in urls[:10]:
        badge = _tier_badge(url)
        short = url.replace("https://", "").replace("http://", "")[:70]
        items.append(
            f'<li>{badge} <a href="{_esc(url)}" target="_blank" rel="noopener">'
            f'{_esc(short)}</a></li>'
        )
    return f'<ul class="evidence-list">{"".join(items)}</ul>'


def _prompt_hash() -> str:
    try:
        from truthbot.verify.adapters.base import SYNTHESIS_SYSTEM
        return hashlib.sha256(SYNTHESIS_SYSTEM.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """\
/* truth-bot static site — newspaper/editorial aesthetic */
:root {
  --paper: #f8f6f0;
  --ink: #1a1a1a;
  --ink-light: #555;
  --rule: #c8c0b0;
  --accent: #8b0000;
  --link: #1a4a8a;
  --chip-r: 3px;
  --max-w: 900px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; }
body { background: var(--paper); color: var(--ink);
       font-family: Georgia, "Times New Roman", serif;
       line-height: 1.65; padding: 0 1rem; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
h1,h2,h3,h4 { font-family: Georgia, serif; font-weight: 700; line-height: 1.25; }
code, pre { font-family: "Courier New", monospace; font-size: 0.85rem; }
pre { background: #f0ede6; border: 1px solid var(--rule); padding: 1rem;
      overflow-x: auto; white-space: pre-wrap; border-radius: 4px; }

/* Layout */
.wrap { max-width: var(--max-w); margin: 0 auto; padding: 1.5rem 0; }
.rule { border: none; border-top: 2px solid var(--ink); margin: 1.5rem 0; }
.rule-light { border: none; border-top: 1px solid var(--rule); margin: 1rem 0; }

/* Masthead */
header.masthead { text-align: center; padding: 2rem 0 1.5rem;
                   border-bottom: 3px double var(--ink); margin-bottom: 2rem; }
header.masthead h1 { font-size: 2.8rem; letter-spacing: -0.02em; color: var(--accent); }
header.masthead .tagline { color: var(--ink-light); font-style: italic; margin-top: 0.3rem; }
nav.top-nav { margin-top: 0.75rem; font-family: system-ui, sans-serif; font-size: 0.85rem; }
nav.top-nav a { margin: 0 0.6rem; color: var(--ink-light); }

/* Verdict chips */
.chip { display: inline-block; padding: 0.2rem 0.55rem; border-radius: var(--chip-r);
         font-family: system-ui, sans-serif; font-weight: 600; white-space: nowrap; }
.verdict-chip { font-size: 0.85rem; }

/* Stacked bar */
.stacked-bar { display: flex; height: 24px; border-radius: 3px; overflow: hidden;
                border: 1px solid var(--rule); margin: 0.5rem 0; }
.bar-seg { display: flex; align-items: center; justify-content: center;
            color: #fff; font-size: 0.7rem; font-weight: 700;
            font-family: system-ui, sans-serif; overflow: hidden; min-width: 0; }
.bar-seg span { padding: 0 2px; }

/* Stats grid */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
               gap: 1rem; margin: 1rem 0; }
.stat-box { background: #fff; border: 1px solid var(--rule); border-radius: 4px;
             padding: 0.75rem 1rem; text-align: center; }
.stat-box .num { font-size: 2rem; font-weight: 700; color: var(--accent);
                  font-family: system-ui, sans-serif; line-height: 1; }
.stat-box .lbl { font-size: 0.75rem; color: var(--ink-light);
                  font-family: system-ui, sans-serif; margin-top: 0.2rem; }

/* Report card (index) */
.report-card { background: #fff; border: 1px solid var(--rule); border-radius: 4px;
                padding: 1rem 1.25rem; margin-bottom: 1rem; }
.report-card h3 { font-size: 1.05rem; margin-bottom: 0.2rem; }
.report-card .meta { font-size: 0.8rem; color: var(--ink-light);
                      font-family: system-ui, sans-serif; margin-bottom: 0.5rem; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.4rem; }

/* Claim block */
.claim-block { background: #fff; border: 1px solid var(--rule); border-radius: 4px;
                padding: 1.25rem; margin-bottom: 1.5rem; }
.claim-text { font-size: 1.08rem; font-weight: 600; margin-bottom: 0.75rem;
               line-height: 1.5; }
.claim-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
               margin-bottom: 0.75rem; }
.claim-context { font-size: 0.85rem; color: var(--ink-light); font-style: italic;
                  border-left: 3px solid var(--rule); padding-left: 0.75rem;
                  margin: 0.5rem 0 0.75rem; }

/* Caveats — PROMINENT, never hidden */
.caveats { background: #fffde7; border: 1px solid #f9a825; border-radius: 4px;
             padding: 0.6rem 0.9rem; margin: 0.6rem 0; font-size: 0.88rem; }
.caveats .cav-label { font-weight: 700; font-family: system-ui, sans-serif;
                        color: #e65100; margin-bottom: 0.25rem; font-size: 0.78rem;
                        text-transform: uppercase; letter-spacing: 0.04em; }

/* Model verdict row */
.model-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 0.5rem; margin: 0.75rem 0; }
.model-card { border: 1px solid var(--rule); border-radius: 4px; padding: 0.6rem 0.75rem;
               font-family: system-ui, sans-serif; font-size: 0.82rem; }
.model-card .model-name { font-weight: 700; color: var(--ink-light); font-size: 0.75rem;
                            text-transform: uppercase; letter-spacing: 0.03em;
                            margin-bottom: 0.3rem; }
.model-card .model-verdict { margin-bottom: 0.25rem; }

/* Expandable sections */
.expandable { margin-top: 0.5rem; }
.expand-btn { background: none; border: 1px solid var(--rule); border-radius: 3px;
               cursor: pointer; font-family: system-ui, sans-serif; font-size: 0.78rem;
               color: var(--link); padding: 0.2rem 0.6rem; }
.expand-btn:hover { background: #f0ede6; }
.expand-content { display: none; margin-top: 0.5rem; }
.expand-content.open { display: block; }

/* Model reasoning expandable */
.model-reasoning { margin-top: 0.75rem; }
.reasoning-block { border: 1px solid var(--rule); border-radius: 4px;
                    padding: 0.75rem; margin-bottom: 0.5rem;
                    font-size: 0.87rem; background: #faf9f6; }
.reasoning-block .r-model { font-weight: 700; font-family: system-ui, sans-serif;
                              font-size: 0.75rem; text-transform: uppercase;
                              color: var(--ink-light); margin-bottom: 0.4rem; }
.reasoning-block .r-text { margin-bottom: 0.5rem; }

/* Evidence list */
.evidence-list { list-style: none; margin: 0.3rem 0; padding: 0; }
.evidence-list li { margin: 0.2rem 0; font-size: 0.82rem;
                     font-family: system-ui, sans-serif; display: flex;
                     align-items: center; gap: 0.4rem; flex-wrap: wrap; }

/* About page */
.tier-table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.88rem; }
.tier-table th { background: var(--ink); color: #fff; padding: 0.4rem 0.6rem;
                  text-align: left; font-family: system-ui, sans-serif; }
.tier-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--rule); }
.tier-table tr:nth-child(even) td { background: #f0ede6; }

/* Footer */
footer { border-top: 2px solid var(--ink); margin-top: 3rem; padding: 1rem 0;
          font-size: 0.78rem; color: var(--ink-light); font-family: system-ui, sans-serif;
          display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between; }

/* Breadcrumb */
.breadcrumb { font-family: system-ui, sans-serif; font-size: 0.82rem;
               color: var(--ink-light); margin-bottom: 1rem; }
.breadcrumb a { color: var(--link); }

/* Permalink */
.permalink { font-family: system-ui, sans-serif; font-size: 0.75rem;
              color: var(--ink-light); margin-top: 0.5rem; }

/* dim text */
.dim { color: var(--ink-light); font-size: 0.85rem; font-family: system-ui, sans-serif; }

/* Responsive */
@media (max-width: 600px) {
  header.masthead h1 { font-size: 1.9rem; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .model-row { grid-template-columns: 1fr 1fr; }
}
"""

# ── JavaScript ────────────────────────────────────────────────────────────────

JS = """\
/* truth-bot minimal JS — expand/collapse only */
(function() {
  document.querySelectorAll('.expand-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var targetId = btn.getAttribute('data-target');
      var el = document.getElementById(targetId);
      if (!el) return;
      var open = el.classList.toggle('open');
      btn.textContent = open ? (btn.getAttribute('data-close') || 'Hide') : btn.getAttribute('data-open');
    });
  });
})();
"""


# ── HTML shell ────────────────────────────────────────────────────────────────

def _page(title: str, body: str, rel: str = "./", extra_head: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="generator" content="truth-bot {PIPELINE_VERSION}">
  <title>{_esc(title)} — truth-bot</title>
  <link rel="stylesheet" href="{rel}assets/styles.css">
  {extra_head}
</head>
<body>
<header class="masthead">
  <div class="wrap">
    <h1><a href="{rel}index.html" style="color:inherit;text-decoration:none">truth-bot</a></h1>
    <p class="tagline">Automated political fact-checking · Multi-model consensus</p>
    <nav class="top-nav">
      <a href="{rel}index.html">Reports</a>
      <a href="{rel}about.html">About</a>
      <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
    </nav>
  </div>
</header>
<div class="wrap">
{body}
</div>
<script src="{rel}assets/truthbot.js"></script>
</body>
</html>"""


# ── Per-claim HTML block (shared by report + claim pages) ─────────────────────

def _claim_block(bundle: VerdictBundle, rel: str = "../", standalone: bool = False) -> str:
    claim = bundle.claim
    consensus = bundle.consensus
    label_str = consensus.consensus_label.value
    total_models = len(bundle.model_verdicts)
    agreeing = len(bundle.agreeing_models)

    # Consensus strength line
    strength_chip = _strength_chip(consensus.consensus_strength, agreeing, total_models)
    verdict_chip = _verdict_chip(label_str, large=True)

    # Context window
    context_html = ""
    if claim.context:
        context_html = (f'<div class="claim-context">'
                        f'…{_esc(claim.context[:300])}…</div>')

    # Caveats — PROMINENT, never hidden
    caveats_html = ""
    all_caveats = [mv.caveats for mv in bundle.model_verdicts if mv.caveats.strip()]
    if all_caveats:
        cav_items = "".join(
            f'<div class="cav-label">{_esc(mv.adapter_name.upper())} · CAVEAT</div>'
            f'<p>{_esc(mv.caveats)}</p>'
            for mv in bundle.model_verdicts if mv.caveats.strip()
        )
        caveats_html = f'<div class="caveats">{cav_items}</div>'

    # Per-model verdict cards
    model_cards = []
    for mv in bundle.model_verdicts:
        vc = _verdict_chip(mv.label.value)
        conf_chip = _chip(mv.confidence.value, "#e0e0e0", "#333", small=True)
        model_cards.append(
            f'<div class="model-card">'
            f'<div class="model-name">{_esc(mv.adapter_name)} · {_esc(mv.model_id)}</div>'
            f'<div class="model-verdict">{vc} {conf_chip}</div>'
            f'</div>'
        )
    model_row_html = f'<div class="model-row">{"".join(model_cards)}</div>' if model_cards else ""

    # Expandable reasoning section
    expand_id = f"reasoning-{claim.id[:8]}"
    reasoning_blocks = []
    for mv in bundle.model_verdicts:
        links = _evidence_links(mv.web_sources)
        reasoning_blocks.append(
            f'<div class="reasoning-block">'
            f'<div class="r-model">{_esc(mv.adapter_name)} / {_esc(mv.model_id)} — '
            f'{_verdict_chip(mv.label.value)} {_chip(mv.confidence.value, "#e0e0e0", "#333", True)}'
            f'</div>'
            f'<p class="r-text">{_esc(mv.explanation)}</p>'
            f'<strong style="font-family:system-ui;font-size:0.78rem">Sources</strong>'
            f'{links}'
            f'</div>'
        )

    expandable_html = ""
    if reasoning_blocks:
        expandable_html = (
            f'<div class="expandable">'
            f'<button class="expand-btn" data-target="{expand_id}" '
            f'data-open="See each model\'s reasoning" data-close="Hide reasoning">'
            f'See each model\'s reasoning</button>'
            f'<div class="expand-content model-reasoning" id="{expand_id}">'
            f'{"".join(reasoning_blocks)}'
            f'</div></div>'
        )

    # Permalink
    permalink = (f'<div class="permalink"><a href="{rel}claims/{claim.id}.html">'
                 f'Permalink: /claims/{claim.id[:16]}…</a></div>')

    return (
        f'<div class="claim-block" id="claim-{claim.id[:8]}">'
        f'<p class="claim-text">{_esc(claim.text)}</p>'
        f'{context_html}'
        f'<div class="claim-meta">{verdict_chip} {strength_chip} '
        f'<span class="dim">{_esc(claim.category or "")}</span></div>'
        f'{caveats_html}'
        f'{model_row_html}'
        f'{expandable_html}'
        f'{permalink}'
        f'</div>'
    )


# ── Truthy mascot widget ─────────────────────────────────────────────────────

_TRUTHY_WIDGET_CSS = """
.truthy-widget{max-width:220px;margin:2rem auto 1rem;text-align:center}
.truthy-score-line{font-family:system-ui,sans-serif;font-size:0.85rem;color:#555;margin-top:0.5rem;font-style:italic}
.truthy-mood-label{font-weight:700;font-family:system-ui,sans-serif;font-size:1rem;margin-top:0.25rem}
.truthy-mood-happy{color:#2e7d32}.truthy-mood-iffy{color:#f57c00}.truthy-mood-sad{color:#b71c1c}
#truthy-caption{font-family:system-ui,sans-serif;font-size:0.88rem;color:#666;text-align:center;margin-top:0.4rem;font-style:italic;min-height:1.2em}
@keyframes idle{0%,100%{transform:translateY(0)}50%{transform:translateY(-2.5px)}}
#character{animation:idle 4s ease-in-out infinite;transform-origin:center bottom}
@keyframes antenna-sway{0%,100%{transform:rotate(-2deg)}50%{transform:rotate(2deg)}}
#antenna{animation:antenna-sway 3s ease-in-out infinite;transform-origin:150px 62px;transform-box:fill-box}
.eye-led{opacity:0;transition:opacity 0.35s ease}
@keyframes true-happy-cycle{0%,70%{opacity:1}78%,88%{opacity:0}96%,100%{opacity:1}}
@keyframes true-neutral-cycle{0%,70%{opacity:0}78%,88%{opacity:1}96%,100%{opacity:0}}
@keyframes happy-pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.08)}}
.state-true .eye-happy{animation:true-happy-cycle 4s ease-in-out infinite,happy-pulse 2.2s ease-in-out infinite;transform-origin:center;transform-box:fill-box}
.state-true .eye-neutral{animation:true-neutral-cycle 4s ease-in-out infinite}
.state-iffy .eye-iffy{opacity:1}
@keyframes sad-wander{0%{transform:translate(-4px,.5px)}25%{transform:translate(-3px,2px)}50%{transform:translate(4px,2.5px)}75%{transform:translate(3px,1.2px)}100%{transform:translate(-4px,.5px)}}
.state-lie .eye-sad{opacity:1;animation:sad-wander 4.2s ease-in-out infinite;transform-origin:center;transform-box:fill-box}
.state-lie #eyeRightGroup .eye-sad{animation-delay:-1.3s}
.eye-shape{transform-origin:center;transform-box:fill-box;transition:transform 0.09s ease-out}
#mascot.blinking .eye-shape{transform:scaleY(0.06)}
@keyframes tear-fall{0%{transform:translateY(-4px);opacity:0}18%{opacity:1}100%{transform:translateY(38px);opacity:0}}
.state-lie #tearLeft,.state-lie #tearRight{animation:tear-fall 2.2s ease-in infinite;transform-origin:center;transform-box:fill-box}
.state-lie #tearRight{animation-delay:0.7s}
#tearLeft,#tearRight{opacity:0}
#armLeftSwing,#armRightSwing,#eyeLeftGroup,#eyeRightGroup,#headGroup,#bodyGroup,#clipboard{transition:transform 0.55s cubic-bezier(.34,1.56,.64,1)}
#led,#ledHalo{transition:fill 0.3s}
"""

_TRUTHY_WIDGET_JS = """
(function(){
  var mascot=document.getElementById('mascot');
  var led=document.getElementById('led');
  var ledHalo=document.getElementById('ledHalo');
  var eyeLeftGroup=document.getElementById('eyeLeftGroup');
  var eyeRightGroup=document.getElementById('eyeRightGroup');
  var headGroup=document.getElementById('headGroup');
  var bodyGroup=document.getElementById('bodyGroup');
  var armLeftSwing=document.getElementById('armLeftSwing');
  var armRightSwing=document.getElementById('armRightSwing');
  var clipboard=document.getElementById('clipboard');
  var caption=document.getElementById('truthy-caption');
  var captions={true:'"That checks out. Sources match! 🎉"',iffy:'"Hmm… let me double-check my sources."',lie:'"Oh no… that isn’t true."'};
  function setState(state){
    mascot.classList.remove('state-true','state-iffy','state-lie');
    mascot.classList.add('state-'+state);
    if(caption)caption.textContent=captions[state];
    if(state==='true'){
      led.setAttribute('fill','url(#ledGradTrue)');
      ledHalo.setAttribute('fill','#5ac075');
      eyeLeftGroup.setAttribute('transform','translate(115 154) rotate(0)');
      eyeRightGroup.setAttribute('transform','translate(185 154) rotate(0)');
      headGroup.setAttribute('transform','translate(0,0)');
      bodyGroup.setAttribute('transform','translate(0,0)');
      armLeftSwing.setAttribute('transform','rotate(135 88 253)');
      armRightSwing.setAttribute('transform','rotate(-135 212 253)');
      clipboard.setAttribute('transform','translate(228 218) rotate(-8)');
    }else if(state==='iffy'){
      led.setAttribute('fill','url(#ledGradIffy)');
      ledHalo.setAttribute('fill','#e8b850');
      eyeLeftGroup.setAttribute('transform','translate(115 156) rotate(-10)');
      eyeRightGroup.setAttribute('transform','translate(185 156) rotate(10)');
      headGroup.setAttribute('transform','rotate(-7 150 170)');
      bodyGroup.setAttribute('transform','translate(0,0)');
      armLeftSwing.setAttribute('transform','rotate(0 88 253)');
      armRightSwing.setAttribute('transform','rotate(-110 212 253)');
      clipboard.setAttribute('transform','translate(238 224) rotate(-3)');
    }else if(state==='lie'){
      led.setAttribute('fill','url(#ledGradLie)');
      ledHalo.setAttribute('fill','#5a8ec0');
      eyeLeftGroup.setAttribute('transform','translate(115 170) rotate(0)');
      eyeRightGroup.setAttribute('transform','translate(185 170) rotate(0)');
      headGroup.setAttribute('transform','translate(0,7)');
      bodyGroup.setAttribute('transform','translate(0,3)');
      armLeftSwing.setAttribute('transform','rotate(8 88 253)');
      armRightSwing.setAttribute('transform','rotate(35 212 253)');
      clipboard.setAttribute('transform','translate(174 298) rotate(40)');
    }
  }
  function doBlink(){mascot.classList.add('blinking');setTimeout(function(){mascot.classList.remove('blinking')},110)}
  function scheduleBlink(){var d=2500+Math.random()*4500;setTimeout(function(){doBlink();if(Math.random()<0.2)setTimeout(doBlink,280);scheduleBlink()},d)}
  scheduleBlink();
  var widget=document.getElementById('truthy-mascot-widget');
  if(widget){
    var mood=widget.getAttribute('data-mood')||'iffy';
    var reasoning=widget.getAttribute('data-reasoning')||'';
    var stateMap={happy:'true',iffy:'iffy',sad:'lie'};
    setState(stateMap[mood]||'iffy');
    var scoreEl=document.getElementById('truthy-score-line');
    var moodEl=document.getElementById('truthy-mood-label');
    if(scoreEl)scoreEl.textContent=reasoning;
    if(moodEl){
      moodEl.textContent=mood==='happy'?'😊 Mostly True':mood==='sad'?'😢 Mostly False':'🤔 Mixed';
      moodEl.className='truthy-mood-label truthy-mood-'+mood;
    }
  }
})();
"""


def _truthy_widget_html(mood: str, reasoning: str) -> str:
    """Return inline style + widget div + init script for Truthy mascot."""
    esc_mood = _esc(mood)
    esc_reasoning = _esc(reasoning)
    svg = (
        '<svg id="mascot" width="200" height="240" viewBox="0 0 300 360" class="state-true">'
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
        '<filter id="softGlow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="4"/></filter>'
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
        '<ellipse cx="150" cy="340" rx="95" ry="10" fill="url(#floorShadowGrad)"/>'
        '<ellipse cx="150" cy="342" rx="70" ry="6" fill="rgba(0,0,0,0.25)"/>'
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
        '<g id="eyeLeftGroup" transform="translate(115 154) rotate(0)">'
        '<g class="eye-shape">'
        '<rect class="eye-neutral eye-led" x="-14" y="-16" width="28" height="32" rx="8" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
        '<path class="eye-happy eye-led" d="M -16 4 L -16 -1 Q 0 -16 16 -1 L 16 4 Q 0 -4 -16 4 Z" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
        '<rect class="eye-iffy eye-led" x="-16" y="-6" width="32" height="12" rx="5" fill="url(#eyeLedIffy)" filter="url(#eyeGlow)"/>'
        '<rect class="eye-sad eye-led" x="-17" y="-17" width="34" height="34" rx="8" fill="url(#eyeLedLie)" filter="url(#eyeGlow)"/>'
        '<rect x="-19" y="-20" width="38" height="40" fill="url(#scanlines)" pointer-events="none" opacity="0.7"/>'
        '</g></g>'
        '<g id="eyeRightGroup" transform="translate(185 154) rotate(0)">'
        '<g class="eye-shape">'
        '<rect class="eye-neutral eye-led" x="-14" y="-16" width="28" height="32" rx="8" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
        '<path class="eye-happy eye-led" d="M -16 4 L -16 -1 Q 0 -16 16 -1 L 16 4 Q 0 -4 -16 4 Z" fill="url(#eyeLedTrue)" filter="url(#eyeGlow)"/>'
        '<rect class="eye-iffy eye-led" x="-16" y="-6" width="32" height="12" rx="5" fill="url(#eyeLedIffy)" filter="url(#eyeGlow)"/>'
        '<rect class="eye-sad eye-led" x="-17" y="-17" width="34" height="34" rx="8" fill="url(#eyeLedLie)" filter="url(#eyeGlow)"/>'
        '<rect x="-19" y="-20" width="38" height="40" fill="url(#scanlines)" pointer-events="none" opacity="0.7"/>'
        '</g></g>'
        '<g transform="translate(102 193)"><g id="tearLeft">'
        '<rect x="-1.5" y="0" width="3" height="3" rx="0.4" fill="#9cc8e8"/>'
        '<rect x="-3" y="3" width="3" height="3" rx="0.4" fill="#b8dcf0"/>'
        '<rect x="0" y="3" width="3" height="3" rx="0.4" fill="#b8dcf0"/>'
        '<rect x="-3" y="6" width="3" height="3" rx="0.4" fill="#7eb4d8"/>'
        '<rect x="0" y="6" width="3" height="3" rx="0.4" fill="#7eb4d8"/>'
        '<rect x="-1.5" y="9" width="3" height="3" rx="0.4" fill="#4a86b8"/>'
        '</g></g>'
        '<g transform="translate(198 193)"><g id="tearRight">'
        '<rect x="-1.5" y="0" width="3" height="3" rx="0.4" fill="#9cc8e8"/>'
        '<rect x="-3" y="3" width="3" height="3" rx="0.4" fill="#b8dcf0"/>'
        '<rect x="0" y="3" width="3" height="3" rx="0.4" fill="#b8dcf0"/>'
        '<rect x="-3" y="6" width="3" height="3" rx="0.4" fill="#7eb4d8"/>'
        '<rect x="0" y="6" width="3" height="3" rx="0.4" fill="#7eb4d8"/>'
        '<rect x="-1.5" y="9" width="3" height="3" rx="0.4" fill="#4a86b8"/>'
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
        '<circle cx="210" cy="296" r="2.5" fill="#f4d98a" opacity="0.8"/>'
        '</g><circle cx="212" cy="253" r="9" fill="url(#brassShade)"/></g>'
        '<g id="clipboard" transform="translate(238 224) rotate(0)">'
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
    return (
        f'<style>{_TRUTHY_WIDGET_CSS}</style>'
        f'<div style="text-align:center;margin:1.5rem 0">'
        f'<div class="truthy-widget" id="truthy-mascot-widget"'
        f' data-mood="{esc_mood}" data-reasoning="{esc_reasoning}">'
        f'{svg}'
        f'<div class="truthy-score-line" id="truthy-score-line"></div>'
        f'<div class="truthy-mood-label" id="truthy-mood-label"></div>'
        f'<div id="truthy-caption"></div>'
        f'</div>'
        f'</div>'
        f'<script>{_TRUTHY_WIDGET_JS}</script>'
    )


# ── Page renderers ────────────────────────────────────────────────────────────

def _render_index(reports: list[dict], stats: dict) -> str:
    """Render the landing page from the reports index."""
    cards = []
    for r in reports[:10]:
        dist = r.get("verdict_distribution", {})
        bar = _stacked_bar(dist)
        chips = " ".join(_verdict_chip(k) + f"<small> {v}</small>"
                         for k, v in dist.items() if v > 0)
        cards.append(
            f'<div class="report-card">'
            f'<h3><a href="{_esc(r["url"])}">{_esc(r["speaker"])}</a></h3>'
            f'<p class="meta">{_esc(r["date"])} · {_esc(r.get("venue",""))} · '
            f'{r.get("claim_count",0)} claims</p>'
            f'{bar}'
            f'<div class="chip-row">{chips}</div>'
            f'</div>'
        )

    total_claims = stats.get("total_claims", 0)
    total_speeches = stats.get("total_speeches", 0)
    agree_rate = stats.get("model_agreement_rate", 0)

    stat_boxes = (
        f'<div class="stat-box"><div class="num">{total_speeches}</div>'
        f'<div class="lbl">Speeches</div></div>'
        f'<div class="stat-box"><div class="num">{total_claims}</div>'
        f'<div class="lbl">Claims checked</div></div>'
        f'<div class="stat-box"><div class="num">{agree_rate:.0%}</div>'
        f'<div class="lbl">Model agreement rate</div></div>'
    )
    for label, count in stats.get("verdict_totals", {}).items():
        if count:
            emoji = VERDICT_EMOJI.get(label, "")
            stat_boxes += (
                f'<div class="stat-box"><div class="num">{count}</div>'
                f'<div class="lbl">{emoji} {_esc(label)}</div></div>'
            )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f'<h2 style="margin-bottom:1rem">Latest Reports</h2>'
        f'<div class="stats-grid">{stat_boxes}</div>'
        f'<hr class="rule">'
        f'{"".join(cards) if cards else "<p class=dim>No reports yet.</p>"}'
        f'<hr class="rule-light">'
        f'<p class="dim"><a href="about.html">About this project</a> · '
        f'<a href="{GITHUB_URL}" target="_blank">GitHub</a></p>'
    )
    footer = (
        f'<span>Last updated: {now}</span>'
        f'<span>Pipeline v{PIPELINE_VERSION} · '
        f'<a href="{GITHUB_URL}" target="_blank">GitHub</a></span>'
    )
    return _page("Latest Reports",
                 body + f'</div><footer class="wrap">{footer}',
                 rel="./")


def _render_report(site_report: SiteReport) -> str:
    """Render a full per-speech report page."""
    dist = site_report.verdict_distribution
    bar = _stacked_bar(dist)
    chips = " ".join(_verdict_chip(k) + f"<small> {v}</small>"
                     for k, v in dist.items() if v > 0)
    agree_rate = site_report.model_agreement_rate
    checkable = len(site_report.checkable_bundles)
    total = len(site_report.bundles)

    summary_card = (
        f'<div class="stats-grid">'
        f'<div class="stat-box"><div class="num">{total}</div>'
        f'<div class="lbl">Claims</div></div>'
        f'<div class="stat-box"><div class="num">{checkable}</div>'
        f'<div class="lbl">Checkable</div></div>'
        f'<div class="stat-box"><div class="num">{agree_rate:.0%}</div>'
        f'<div class="lbl">Model agreement</div></div>'
        f'</div>'
        f'<div class="chip-row">{chips}</div>'
        f'{bar}'
    )

    src_link = ""
    if site_report.transcript_source_url:
        src_link = (f' · <a href="{_esc(site_report.transcript_source_url)}" '
                    f'target="_blank" rel="noopener">Transcript source</a>')

    claim_blocks = "\n".join(
        _claim_block(b, rel="../") for b in site_report.checkable_bundles
    )

    phash = _prompt_hash()
    gen_ts = site_report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    verdict = site_report.truthy_verdict
    widget_html = _truthy_widget_html(verdict.mood, verdict.reasoning)

    body = (
        f'<div class="breadcrumb"><a href="../index.html">Reports</a> › '
        f'{_esc(site_report.speaker)}</div>'
        f'<h2>{_esc(site_report.speaker)}</h2>'
        f'<p class="dim">{_esc(site_report.display_date)}'
        f'{" · " + _esc(site_report.venue) if site_report.venue else ""}'
        f'{" · " + _esc(site_report.role) if site_report.role else ""}'
        f'{src_link}</p>'
        f'<hr class="rule">'
        f'{summary_card}'
        f'<hr class="rule">'
        f'{widget_html}'
        f'<h3 style="margin-bottom:1rem">Claims</h3>'
        f'{claim_blocks}'
    )
    footer = (
        f'<span>Generated: {gen_ts} · Pipeline v{PIPELINE_VERSION} · '
        f'Prompt hash: {phash}</span>'
        f'<span><a href="../index.html">All reports</a> · '
        f'<a href="{GITHUB_URL}" target="_blank">GitHub</a></span>'
    )
    return _page(
        f"{site_report.speaker} — {site_report.display_date}",
        body + f'</div><footer class="wrap">{footer}',
        rel="../",
    )


def _render_claim_page(bundle: VerdictBundle, site_report: SiteReport) -> str:
    """Render a standalone per-claim permalink page."""
    report_url = f"../reports/{site_report.report_slug}.html"
    body = (
        f'<div class="breadcrumb">'
        f'<a href="../index.html">Reports</a> › '
        f'<a href="{report_url}">{_esc(site_report.speaker)} — '
        f'{_esc(site_report.display_date)}</a> › Claim</div>'
        f'{_claim_block(bundle, rel="../", standalone=True)}'
    )
    phash = _prompt_hash()
    gen_ts = site_report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    footer = (
        f'<span>Generated: {gen_ts} · Pipeline v{PIPELINE_VERSION} · '
        f'Prompt hash: {phash}</span>'
        f'<span><a href="{report_url}">Back to report</a> · '
        f'<a href="{GITHUB_URL}" target="_blank">GitHub</a></span>'
    )
    return _page(
        f"Claim: {bundle.claim.text[:60]}…",
        body + f'</div><footer class="wrap">{footer}',
        rel="../",
    )


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
        f'<span>Pipeline v{PIPELINE_VERSION} · Prompt hash: {phash}</span>'
        f'<span><a href="{GITHUB_URL}" target="_blank">GitHub</a></span>'
    )
    return _page("About", body + f'</div><footer class="wrap">{footer}', rel="./")


def _render_404() -> str:
    body = (
        '<h2>404 — Page not found</h2>'
        '<p class="dim">The page you requested does not exist.</p>'
        '<p><a href="index.html">Return to reports</a></p>'
    )
    return _page("404 Not Found", body, rel="./")


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
        stats = self._compute_stats(reports_index)
        index_html = _render_index(reports_index, stats)
        self._write(self._root / "index.html", index_html)

        # About + 404 (regenerate on each publish for prompt-hash freshness)
        self._write(self._root / "about.html", _render_about())
        self._write(self._root / "404.html",   _render_404())

        return report_path.resolve()

    def site_url(self, site_report: SiteReport, base_url: str = "http://expressionpi.home.arpa/truthbot") -> str:
        return f"{base_url.rstrip('/')}/reports/{site_report.report_slug}.html"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_structure(self) -> None:
        for sub in ("reports", "claims", "assets", "data"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)

    def _copy_assets(self) -> None:
        self._write(self._root / "assets" / "styles.css", CSS)
        self._write(self._root / "assets" / "truthbot.js", JS)

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

    def _compute_stats(self, reports: list[dict]) -> dict:
        total_claims = sum(r.get("claim_count", 0) for r in reports)
        if reports:
            agree_rate = sum(r.get("model_agreement_rate", 0) for r in reports) / len(reports)
        else:
            agree_rate = 0.0
        verdict_totals: dict[str, int] = {v: 0 for v in VERDICT_COLOR}
        for r in reports:
            for label, cnt in r.get("verdict_distribution", {}).items():
                verdict_totals[label] = verdict_totals.get(label, 0) + cnt
        return {
            "total_speeches": len(reports),
            "total_claims": total_claims,
            "model_agreement_rate": agree_rate,
            "verdict_totals": verdict_totals,
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
