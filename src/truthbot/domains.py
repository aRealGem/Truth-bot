"""
Registered-domain matching for source-tier classification.

The tier rules used to be plain substring tests against the whole URL, which
misfires on lookalike hosts: ``".gov" in "https://www.govtech.com/..."`` is
True, so a trade magazine ranked as Government tier and won evidence-pack
slots over on-topic sources. Matching here is against the URL's *hostname*
only, and a pattern matches when the host equals it or is a subdomain of it
(label-boundary suffix), never mid-string.

Shared by the Brave connector's ``_classify_tier`` and the site renderer's
``_tier_bucket`` / ``_tier_badge`` so the two stay in sync by construction.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

_HOST_RX = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")


def url_host(url: str) -> str:
    """Lowercased hostname of ``url``, or ``''`` when unparseable.

    Accepts scheme-less inputs (``"bls.gov/data"``) so callers can pass bare
    domains as well as full URLs."""
    if not url:
        return ""
    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"//{candidate}"
    try:
        host = urlsplit(candidate).hostname
    except ValueError:
        return ""
    host = (host or "").lower().rstrip(".")
    return host if _HOST_RX.match(host) else ""


def host_matches(host: str, pattern: str) -> bool:
    """True when ``host`` is ``pattern`` or a subdomain of it.

    ``pattern`` may be a registered domain (``"apnews.com"``), a bare TLD
    rule written with a leading dot (``".gov"``), or a host. Matching is on
    dot-label boundaries: ``"www.bls.gov"`` matches ``".gov"``;
    ``"www.govtech.com"`` does not."""
    if not host or not pattern:
        return False
    p = pattern.lower().lstrip(".")
    return host == p or host.endswith("." + p)


def url_matches_any(url: str, patterns: tuple[str, ...] | list[str]) -> bool:
    """True when the URL's hostname matches any pattern (see ``host_matches``)."""
    host = url_host(url)
    return any(host_matches(host, p) for p in patterns)


#: Query params that mark a listing/pagination/search page rather than an article.
_LISTING_QUERY_KEYS = frozenset({"pagenum", "page", "paged", "s", "q", "query", "search"})


def is_substantive_url(url: str) -> bool:
    """False for homepages, section indexes, and pagination/search listings —
    URLs that can never BE evidence, only point at a site.

    Empirical driver (jackie, 2026-07-20, trump_2026:0107): the FactCheck
    connector's ``site:``-scoped queries return ``snopes.com/`` (the homepage)
    and ``snopes.com/fact-check/?pagenum=3`` (a listing index) when Brave has
    no article-level match, and the domain filter happily passed both into a
    cap-6 evidence pack. Rules: a URL with an empty path is a homepage; a
    shallow path (single segment) carrying a listing/pagination/search query
    param is an index page. Real articles (``/fact-check/<slug>``, deep dated
    paths, even single-segment slug articles without listing params) pass."""
    from urllib.parse import parse_qs

    if not url_host(url):
        return False
    try:
        parts = urlsplit(url if "://" in url else f"//{url}")
    except ValueError:
        return False
    path = parts.path.strip("/")
    if not path:
        return False                                   # bare homepage
    if len(path.split("/")) <= 1:
        q = parse_qs(parts.query)
        if any(k in q for k in _LISTING_QUERY_KEYS):
            return False                               # section listing / search page
    return True
