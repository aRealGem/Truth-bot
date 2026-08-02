"""Mutable "latest" endpoint blocklist (remediation v2, item 1.3).

Some agency URLs are LIVE pointers whose content silently tracks the newest
release — e.g. ``bls.gov/news.release/empsit.htm`` always shows the CURRENT
Employment Situation, whatever its retrieval date said. Admitting one into an
era-scoped pack plants evidence whose content will drift out of the claim's
era (the Obama-2014 packs carried live BLS pages showing 2026 data).

Policy: such URLs are DROPPED at consolidation with telemetry
(``dropped["mutable-latest-endpoint"]``). Archive-dated variants (the same
release under an ``/archives/`` path or with an embedded release date) are
immutable and pass. Deterministic rewrite-to-archive is not attempted: the
archive URL embeds the release date, which is not derivable offline
(release-calendar table = explicit future decision, DC'd 2026-08-02).

Config: ``mutable_endpoints.json`` next to this module —
``[{"domain", "prefix", "immutable_markers": [..]}]``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

_CONFIG = Path(__file__).with_name("mutable_endpoints.json")


@lru_cache(maxsize=1)
def _rules() -> list[dict]:
    return json.loads(_CONFIG.read_text())["endpoints"]


def is_mutable_latest(url: str) -> bool:
    """True when ``url`` is a live latest-release pointer (era-unsafe)."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path or "/"
    for rule in _rules():
        if host != rule["domain"] and not host.endswith("." + rule["domain"]):
            continue
        if not path.startswith(rule["prefix"]):
            continue
        if any(m in path for m in rule.get("immutable_markers", [])):
            continue
        return True
    return False
