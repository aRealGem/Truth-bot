"""Fact-checker exclusion for model-facing evidence (P67.7 / remediation T2.1).

Locked project invariant: fact-checker VERDICTS are gold-side only — they
never enter model context. Under evidence-mode ``shared_pack_v1`` the pack
deliberately carried fact-check rulings (129/178 Trump claims had one; the
audit's F5 circularity finding); ``shared_pack_v2`` excludes them at BOTH
retrieval (the PR-5 retrievers call :func:`is_excluded_factchecker` on every
candidate) and consolidation (:mod:`truthbot.verdict.consolidator` filters
again, so a retriever that forgets is caught).

The list is MAINTAINED IN CONFIG, not code: ``factcheck_blocklist.json``
next to this module (domains match by registered-domain suffix via
``truthbot.domains.host_matches``; ``path_prefixes`` entries block only that
path on an otherwise-allowed domain — reuters.com stays a Tier-2 wire
source, reuters.com/fact-check/* does not).

Gold-side evaluation code may keep using these sources freely.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from truthbot.domains import host_matches, url_host

_BLOCKLIST_PATH = Path(__file__).resolve().parent / "factcheck_blocklist.json"


@dataclass(frozen=True)
class _Rules:
    domains: tuple[str, ...]
    prefixes: tuple[tuple[str, str], ...]
    path_patterns: tuple[re.Pattern, ...]
    verticals: tuple[tuple[str, re.Pattern], ...]
    allowlist: tuple[tuple[str, str], ...]


@lru_cache(maxsize=1)
def _blocklist() -> _Rules:
    doc = json.loads(_BLOCKLIST_PATH.read_text(encoding="utf-8"))
    return _Rules(
        domains=tuple(d.lower() for d in doc.get("domains") or ()),
        prefixes=tuple((e["domain"].lower(), e["prefix"].lower())
                       for e in doc.get("path_prefixes") or ()),
        path_patterns=tuple(re.compile(p, re.IGNORECASE)
                            for p in doc.get("path_patterns") or ()),
        verticals=tuple((e["domain"].lower(),
                         re.compile(e["pattern"], re.IGNORECASE))
                        for e in doc.get("verticals") or ()),
        allowlist=tuple((e["domain"].lower(), e["prefix"].lower())
                        for e in doc.get("allowlist") or ()),
    )


def blocked_domains() -> tuple[str, ...]:
    """The configured full-domain blocklist (for docs/tests/telemetry)."""
    return _blocklist().domains


def _url_path(url: str) -> str:
    try:
        return (urlsplit(url if "://" in url else f"//{url}").path or "").lower()
    except ValueError:
        return ""


def factcheck_exclusion_reason(url: str) -> str:
    """The rule that excludes ``url`` from model-facing packs, or '' when
    allowed. Evaluation order (remediation v2, 1.1 / DC-1): allowlist wins →
    blocked domains → blocked path prefixes → per-domain vertical regexes →
    the generic fact-check path regex (URL path only, never the host — a
    'factcheck.house.gov'-style HOST is a domain-rule question)."""
    host = url_host(url)
    if not host:
        return ""
    rules = _blocklist()
    path = _url_path(url)
    if any(host_matches(host, d) and path.startswith(p)
           for d, p in rules.allowlist):
        return ""
    for d in rules.domains:
        if host_matches(host, d):
            return f"domain:{d}"
    for d, p in rules.prefixes:
        if host_matches(host, d) and path.startswith(p):
            return f"path-prefix:{d}{p}"
    for d, rx in rules.verticals:
        if host_matches(host, d) and rx.search(path):
            return f"vertical:{d}"
    if any(rx.search(path) for rx in rules.path_patterns):
        return "path-regex"
    return ""


def is_excluded_factchecker(url: str) -> bool:
    """True when ``url`` is excluded from model-facing evidence — see
    :func:`factcheck_exclusion_reason` for the rule that fired."""
    return bool(factcheck_exclusion_reason(url))


# Query-constraint tokens (T2.2): generated queries must never steer retrieval
# toward fact-checker coverage. Checked case-insensitively on whole queries.
FACTCHECK_QUERY_TOKENS = ("fact check", "fact-check", "factcheck")


def query_violates_constraints(query: str, forbidden_terms: tuple[str, ...] = ()) -> str:
    """Return a reason string when ``query`` violates T2.2, else ''.

    ``forbidden_terms`` carries speaker-name tokens supplied by the CALLER
    (code-side validation — the query-generation model itself remains
    speaker-blind and is never shown these terms)."""
    q = (query or "").lower()
    for tok in FACTCHECK_QUERY_TOKENS:
        if tok in q:
            return f"contains fact-check token {tok!r}"
    for term in forbidden_terms:
        t = (term or "").strip().lower()
        if len(t) >= 3 and t in q:
            return f"contains speaker term {term!r}"
    return ""
