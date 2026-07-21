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
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from truthbot.domains import host_matches, url_host

_BLOCKLIST_PATH = Path(__file__).resolve().parent / "factcheck_blocklist.json"


@lru_cache(maxsize=1)
def _blocklist() -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    doc = json.loads(_BLOCKLIST_PATH.read_text(encoding="utf-8"))
    domains = tuple(d.lower() for d in doc.get("domains") or ())
    prefixes = tuple((e["domain"].lower(), e["prefix"].lower())
                     for e in doc.get("path_prefixes") or ())
    return domains, prefixes


def blocked_domains() -> tuple[str, ...]:
    """The configured full-domain blocklist (for docs/tests/telemetry)."""
    return _blocklist()[0]


def is_excluded_factchecker(url: str) -> bool:
    """True when ``url`` belongs to a blocked fact-checker domain, or sits
    under a blocked fact-check path of an otherwise-allowed domain."""
    host = url_host(url)
    if not host:
        return False
    domains, prefixes = _blocklist()
    if any(host_matches(host, d) for d in domains):
        return True
    try:
        path = (urlsplit(url if "://" in url else f"//{url}").path or "").lower()
    except ValueError:
        return False
    return any(host_matches(host, d) and path.startswith(p) for d, p in prefixes)


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
