"""Canonical source-tier classification (Claim Eval v3 PR-A / D7).

**This is the single implementation.** It used to be two: ``classify_tier`` in
the Brave connector decided the tier the pipeline stored, and ``_tier_bucket``
in :mod:`truthbot.publish.site` re-derived a tier from the URL at render time to
draw its badges. They shared a *matcher* (:mod:`truthbot.domains`) but kept
separate *domain lists*, and had already drifted: the connector counted
``federalreserve.gov`` and ``stlouisfed.org`` as Government, the renderer did
not, so a FRASER/FRED source was Government in the evidence pack and badged
bottom-tier on the published site. Both now call in here.

That drift matters more than a wrong badge. ``tier`` is one of exactly four
fields invariant **I5** requires on every evidence item
(``hydramind.invariants._REQUIRED_PROVENANCE``), so it is part of the integrity
record — the site must not contradict it. See ``docs/integrity-invariants.md``.

Why the path rules exist (jackie's ruling, 2026-07-29): partisan government
press releases are admissible only to confirm a claim was **made**, never to
prove it **true**. Tiering was domain-only, so every ``.gov`` host classified
Government — top tier, ``TIER_WEIGHTS`` 1.0, and eligible to trigger the
rubric's automatic-FALSE override on a contradiction. The new bottom tier
``SourceTier.POLITICAL`` fixes that. It is the tier jackie's design note calls
**S5**; the site badges it **T7** because that is its actual rank (last). S5 and
T7 are the same tier under two numbering schemes.

Rules are config, not code (``source_tiers.json``), mirroring
:mod:`truthbot.verify.factcheck_exclusion`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from truthbot.domains import host_matches, url_host, url_matches_any
from truthbot.models import SourceTier

_TIERS_PATH = Path(__file__).resolve().parent / "source_tiers.json"

# ``.int`` = treaty-established intergovernmental orgs (nato.int, un.int) —
# primary-source class; was silently OTHER until the 2026-07-21 pilot.
# ``stlouisfed.org`` = Federal Reserve Bank of St. Louis (FRASER + FRED): a Fed
# property on .org and the canonical host of archival government statistics —
# ranked OTHER until the Nixon probe (2026-07-24) showed its BLS-release
# archives couldn't credit the T2.4 quota.
GOV_DOMAINS = (".gov", ".mil", ".int", "federalreserve.gov", "stlouisfed.org")
WIRE_DOMAINS = ("apnews.com", "reuters.com")
ESTABLISHED_DOMAINS = (
    "nytimes.com", "washingtonpost.com", "bbc.com", "bbc.co.uk",
    "nbcnews.com", "cbsnews.com", "abcnews.go.com", "npr.org",
)
FACTCHECK_DOMAINS = ("politifact.com", "factcheck.org", "snopes.com", "fullfact.org")


class _Config:
    """Parsed ``source_tiers.json``. Tuples so the cached instance is immutable."""

    __slots__ = ("political_domains", "political_paths", "gov_press_paths",
                 "stat_domains", "stat_data_paths", "stat_press_paths",
                 "gov_substantive_paths", "established_gov_domains",
                 "data_signal_segments", "data_hub_host_labels",
                 "quarantine_unmapped_gov")

    def __init__(self, doc: dict) -> None:
        pol = doc.get("political") or {}
        self.political_domains = tuple(d.lower() for d in pol.get("domains") or ())
        self.political_paths = tuple(
            (e["domain"].lower(), e["prefix"].lower()) for e in pol.get("path_prefixes") or ()
        )
        self.gov_press_paths = tuple(
            p.lower() for p in (doc.get("gov_press_paths") or {}).get("prefixes") or ()
        )
        stat = doc.get("nonpartisan_sources") or {}
        self.stat_domains = tuple(d.lower() for d in stat.get("domains") or ())
        self.stat_data_paths = tuple(
            p.lower() for p in (stat.get("data_path_prefixes") or {}).get("prefixes") or ()
        )
        self.stat_press_paths = tuple(
            p.lower() for p in (stat.get("press_path_prefixes") or {}).get("prefixes") or ()
        )
        self.gov_substantive_paths = tuple(
            p.lower() for p in (doc.get("gov_substantive_paths") or {}).get("prefixes") or ()
        )
        self.established_gov_domains = tuple(
            d.lower() for d in (doc.get("established_gov_domains") or {}).get("domains") or ()
        )
        self.data_signal_segments = frozenset(
            s.lower() for s in (doc.get("data_signal_segments") or {}).get("segments") or ()
        )
        self.data_hub_host_labels = frozenset(
            s.lower() for s in (doc.get("data_hub_host_labels") or {}).get("labels") or ()
        )
        self.quarantine_unmapped_gov = bool(doc.get("quarantine_unmapped_gov", True))


@lru_cache(maxsize=1)
def _config() -> _Config:
    return _Config(json.loads(_TIERS_PATH.read_text(encoding="utf-8")))


def _url_path(url: str) -> str:
    """Lowercased path of ``url`` (``'/'``-normalised), or ``''`` if unparseable."""
    try:
        path = urlsplit(url if "://" in url else f"//{url}").path or "/"
    except ValueError:
        return ""
    return path.lower()


def _starts_with_any(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(p) for p in prefixes)


def _host_leading_label(host: str) -> str:
    """First dot-delimited label of ``host`` (``datahub.hhs.gov`` → ``datahub``)."""
    return host.split(".", 1)[0]


def _path_segments(path: str) -> tuple[str, ...]:
    """Non-empty ``'/'``-delimited segments of ``path`` (``/newsroom/stats/x`` →
    ``('newsroom', 'stats', 'x')``). Exact-segment matching, not substring, so
    ``/news/data-shows-x`` does not count ``data-shows-x`` as a data segment."""
    return tuple(s for s in path.split("/") if s)


def _gov_tier(url: str, host: str, path: str, cfg: _Config) -> SourceTier:
    """Tier a government-class host, applying D7's path rules.

    Order matters. The nonpartisan-source carve-out is checked first, and its
    data paths first within that, because ``bls.gov/news.release/*`` is real
    data on a press-looking path — the exact case a blanket "``/news`` means
    demote" rule would destroy. The same reasoning extends the carve-out past
    statistical agencies to courts, legislative records and science agencies:
    measured against stored run artifacts, the quarantine alone was demoting
    ``supremecourt.gov`` opinions and PubMed Central papers to S5.
    """
    # D7 disposition: a government host explicitly capped at S3, overriding even
    # its data/substantive paths — e.g. aspe.hhs.gov, an appointee-led research
    # office that is credible-secondary, not primary nonpartisan record.
    if any(host_matches(host, d) for d in cfg.established_gov_domains):
        return SourceTier.ESTABLISHED

    # Open-data hubs: the data signal is in the HOST, not the path. P129 caught
    # datahub.hhs.gov/Hospital/COVID-19-Reported-Patient-Impact quarantined to S5
    # because no path segment said "data" — but the whole host IS a data hub.
    # A leading label like ``data``/``datahub`` marks the host as structured data.
    if _host_leading_label(host) in cfg.data_hub_host_labels:
        return SourceTier.GOVERNMENT

    if any(host_matches(host, d) for d in cfg.stat_domains):
        if _starts_with_any(path, cfg.stat_data_paths):
            return SourceTier.GOVERNMENT          # S1 — data, whatever it looks like
        if _starts_with_any(path, cfg.stat_press_paths):
            return SourceTier.ESTABLISHED         # S3 — the agency's own press shop
        return SourceTier.GOVERNMENT              # nonpartisan by default, not quarantined

    # "Data yes, press no" (D7, jackie 2026-07-31). A structured-data or
    # statistical-record path survives even when it sits UNDER a press prefix:
    # ``cbp.gov/newsroom/stats/nationwide-encounters`` is border-encounter data
    # on a newsroom path — the BLS case one scope level out, for an enforcement
    # agency that is not on the nonpartisan-source list. Checked BEFORE the
    # press-path demotion so a data segment wins; a genuine press release or
    # announcement (no data segment) still falls through to S5.
    if cfg.data_signal_segments.intersection(_path_segments(path)):
        return SourceTier.GOVERNMENT

    if _starts_with_any(path, cfg.gov_press_paths):
        return SourceTier.POLITICAL               # S5 — an agency press release

    if _starts_with_any(path, cfg.gov_substantive_paths):
        return SourceTier.GOVERNMENT

    # D7 quarantine: an unmapped path fails CLOSED, so a newly-invented press
    # path cannot leak into the top tier just by not being listed yet.
    #
    # Scoped to .gov ONLY, which is what D7 actually says. An earlier revision
    # applied it to the whole government class and demoted
    # nato.int/cps/en/natohq/* to S5 — .int is treaty-established
    # intergovernmental orgs (NATO, UN), a primary-source class with no US
    # partisan press shop to guard against. Same for .mil document paths. The
    # quarantine exists to contain *political messaging*, and that risk lives
    # on .gov.
    if cfg.quarantine_unmapped_gov and host_matches(host, ".gov"):
        return SourceTier.POLITICAL
    return SourceTier.GOVERNMENT


def classify_tier(url: str) -> SourceTier:
    """Assign a trust tier to ``url``.

    Host-suffix matching (:mod:`truthbot.domains`), never substring — a
    substring rule once made ``www.govtech.com`` rank Government because it
    contains ``.gov``, letting a trade magazine win pack slots. Paths are
    consulted only for government-class hosts and the explicit political
    path rules.
    """
    host = url_host(url)
    if not host:
        return SourceTier.OTHER
    cfg = _config()
    path = _url_path(url)

    # Political communications outrank every other rule: whitehouse.gov is S5
    # on all paths, as are party and campaign organs.
    if any(host_matches(host, d) for d in cfg.political_domains):
        return SourceTier.POLITICAL
    if any(host_matches(host, d) and path.startswith(p) for d, p in cfg.political_paths):
        return SourceTier.POLITICAL

    if url_matches_any(url, GOV_DOMAINS):
        return _gov_tier(url, host, path, cfg)
    if url_matches_any(url, WIRE_DOMAINS):
        return SourceTier.WIRE
    if url_matches_any(url, ESTABLISHED_DOMAINS):
        return SourceTier.ESTABLISHED
    if url_matches_any(url, FACTCHECK_DOMAINS):
        return SourceTier.FACTCHECK
    return SourceTier.OTHER


#: Display metadata per tier, consumed by the site renderer. ``code`` is the
#: badge's rank label — POLITICAL is T7 because it ranks last; the design note
#: calls the same tier S5.
TIER_DISPLAY: dict[SourceTier, tuple[str, str]] = {
    SourceTier.GOVERNMENT: ("T1·Gov", "tier-gov"),
    SourceTier.WIRE: ("T2·Wire", "tier-news"),
    SourceTier.ESTABLISHED: ("T3·News", "tier-news"),
    SourceTier.ACADEMIC: ("T4·Acad", "tier-news"),
    SourceTier.FACTCHECK: ("T5·FC", "tier-fc"),
    SourceTier.OTHER: ("T6", "tier-other"),
    SourceTier.POLITICAL: ("T7·Pol", "tier-political"),
}

#: Coarse bucket key per tier, for the site's per-report source tallies.
TIER_BUCKET: dict[SourceTier, str] = {
    SourceTier.GOVERNMENT: "gov",
    SourceTier.WIRE: "wire",
    SourceTier.ESTABLISHED: "news",
    SourceTier.ACADEMIC: "news",
    SourceTier.FACTCHECK: "fc",
    SourceTier.OTHER: "other",
    SourceTier.POLITICAL: "political",
}
