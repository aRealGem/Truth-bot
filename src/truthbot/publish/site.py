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
