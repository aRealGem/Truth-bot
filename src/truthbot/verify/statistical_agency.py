"""Function-based statistical-agency allowlist — D16 condition 1. FAIL-CLOSED.

Loads ``statistical_agency_registry.yaml`` (schema
``truthbot-statistical-agency-registry v1``) and exposes
:func:`classify_ex`, which returns ``(allowed, reason)`` — the reason string
names the rule that decided, so the D16 measurement can distinguish "denied
because it is a press page" from "denied because nobody listed this host".

Deliberately modelled on :mod:`truthbot.verify.tier_registry` (same YAML
header style, same longest-suffix host matching, same load-time guards), with
ONE structural difference that is the whole point:

  * the tier registry is a CLASSIFIER — every host gets some tier, and unknown
    government hosts fall through to a quarantine;
  * this registry is an ALLOWLIST — the answer for an unlisted host is NO, and
    the deny rules run BEFORE the allowlist so a structural exclusion can never
    lose to a later-added entry.

The press-path list is INHERITED from the tier registry's
``stat_press_prefixes`` rather than restated (see the YAML header): that list
already encodes the distinction between ``bls.gov/news.release/*`` — the jobs
report — and ``bls.gov/newsroom/*`` — the press shop. A second copy would
drift, and the drift would be silent.

This module answers ONE question: is this URL a record published by a body
whose FUNCTION is statistical measurement? It says nothing about the document's
content, its date, or whether it may credit anything. The date test is
:mod:`truthbot.verdict.statistical_release`; the quota effect is the
consolidator's.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from truthbot.domains import url_host

_REGISTRY_PATH = Path(__file__).resolve().parent / "statistical_agency_registry.yaml"

SCHEMA = "truthbot-statistical-agency-registry v1"


@dataclass(frozen=True)
class Entry:
    domain: str
    agency: str
    rationale: str
    date: str
    #: When non-empty the path MUST start with one of these — the fail-closed
    #: form, used for hosts (cdc.gov) that are only PARTLY a statistical
    #: function.
    allow_paths: tuple[str, ...] = ()
    #: Denied path prefixes, checked before ``allow_paths``.
    deny_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DenyDomain:
    domain: str
    rationale: str
    date: str


@dataclass(frozen=True)
class Registry:
    schema: str
    version: str
    entries_by_domain: dict[str, Entry]
    deny_by_domain: dict[str, DenyDomain]
    deny_host_substrings: tuple[str, ...]
    deny_host_labels: frozenset[str]
    #: Extra press prefixes from THIS file; the tier registry's
    #: ``stat_press_prefixes`` are added to them at load time.
    press_prefixes: tuple[str, ...]

    def entries(self) -> list[Entry]:
        return list(self.entries_by_domain.values())

    def agencies(self) -> set[str]:
        return {e.agency for e in self.entries_by_domain.values()}


def _tuple_paths(raw, field: str, domain: str) -> tuple[str, ...]:
    out = []
    for p in raw.get(field) or ():
        p = str(p).lower().strip()
        if not p.startswith("/"):
            raise ValueError(
                f"statistical_agency_registry: {domain!r} {field} entry {p!r} "
                "must be a path prefix starting with '/'")
        out.append(p)
    return tuple(out)


def _parse_entry(raw: dict) -> Entry:
    domain = str(raw["domain"]).lower().strip()
    if not domain or "*" in domain or "/" in domain:
        raise ValueError(
            f"statistical_agency_registry: entry domain {domain!r} must be a "
            "bare host suffix — this registry has no wildcards, on purpose: "
            "an allowlist that can match a class is an allowlist nobody reads")
    agency = str(raw.get("agency", "")).strip()
    rationale = str(raw.get("rationale", "")).strip()
    if not agency or not rationale:
        raise ValueError(
            f"statistical_agency_registry: {domain!r} needs both an agency and "
            "a rationale — every host here widens the only door through which "
            "post-speech evidence can reach a verdict")
    return Entry(domain=domain, agency=agency, rationale=rationale,
                 date=str(raw.get("date", "")),
                 allow_paths=_tuple_paths(raw, "allow_paths", domain),
                 deny_paths=_tuple_paths(raw, "deny_paths", domain))


@lru_cache(maxsize=1)
def load_registry() -> Registry:
    from truthbot.verify.tier_registry import load_registry as load_tiers

    doc = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(
            f"statistical_agency_registry: unexpected schema {doc.get('schema')!r}")
    deny = doc.get("deny") or {}

    entries: dict[str, Entry] = {}
    for raw in doc.get("entries") or ():
        e = _parse_entry(raw)
        if e.domain in entries:
            raise ValueError(
                f"statistical_agency_registry: duplicate entry {e.domain!r}")
        entries[e.domain] = e

    denies: dict[str, DenyDomain] = {}
    for raw in deny.get("domains") or ():
        d = DenyDomain(domain=str(raw["domain"]).lower().strip(),
                       rationale=str(raw.get("rationale", "")).strip(),
                       date=str(raw.get("date", "")))
        if not d.rationale:
            raise ValueError(f"statistical_agency_registry: deny domain "
                             f"{d.domain!r} needs a rationale")
        denies[d.domain] = d

    # A host may not be both allowed and denied — the deny would win silently
    # and the entry would be a lie on disk.
    overlap = sorted(set(entries) & set(denies))
    if overlap:
        raise ValueError(f"statistical_agency_registry: {overlap} appear in "
                         "BOTH entries and deny.domains")

    press = tuple(str(p).lower() for p in deny.get("press_prefixes") or ())
    # Inherited, never copied — see the module docstring.
    press = tuple(dict.fromkeys(press + load_tiers().stat_press_prefixes))

    return Registry(
        schema=str(doc["schema"]),
        version=str(doc.get("version", "")),
        entries_by_domain=entries,
        deny_by_domain=denies,
        deny_host_substrings=tuple(
            str(s).lower() for s in deny.get("host_substrings") or ()),
        deny_host_labels=frozenset(
            str(s).lower() for s in deny.get("host_labels") or ()),
        press_prefixes=press,
    )


# ── URL helpers (same shapes as tier_registry, deliberately) ────────────────

def _url_path(url: str) -> str:
    try:
        path = urlsplit(url if "://" in url else f"//{url}").path or "/"
    except ValueError:
        return ""
    return path.lower()


def _match_suffix(host: str, table: dict) -> tuple[str, object] | None:
    """Longest dot-label suffix present in ``table`` (so ``www.govtech.com``
    can never match a ``.gov`` rule)."""
    labels = host.split(".")
    for i in range(len(labels) - 1):
        candidate = ".".join(labels[i:])
        hit = table.get(candidate)
        if hit is not None:
            return candidate, hit
    return None


def _first_prefix(path: str, prefixes: tuple[str, ...]) -> str | None:
    for p in prefixes:
        if path.startswith(p):
            return p
    return None


# ── the question ────────────────────────────────────────────────────────────

def classify_ex(url: str) -> tuple[bool, str]:
    """Is ``url`` a statistical-agency record, and WHY — ``(allowed, reason)``.

    Reason strings name the deciding rule::

        (True,  "entry:bls.gov")
        (True,  "entry:cdc.gov/path:/nchs")
        (False, "deny:host-substring:whitehouse")
        (False, "deny:host-label:omb")
        (False, "deny:domain:fraser.stlouisfed.org")
        (False, "deny:press-prefix:/newsroom")
        (False, "deny:path-not-allowed:cdc.gov")
        (False, "not-listed")

    Evaluation order is the YAML header's: structural denies, then press
    paths, then the allowlist, then the fail-closed default. An unparseable
    URL is denied (``"no-host"``), like everything else nobody vouched for.
    """
    host = url_host(url)
    if not host:
        return False, "no-host"
    reg = load_registry()

    # 1-2. Structural denies. First, so nothing added later can outrank them.
    for token in reg.deny_host_substrings:
        if token in host:
            return False, f"deny:host-substring:{token}"
    hit = reg.deny_host_labels.intersection(host.split("."))
    if hit:
        return False, f"deny:host-label:{sorted(hit)[0]}"

    # 3. Explicit, documented suffix denials.
    denied = _match_suffix(host, reg.deny_by_domain)
    if denied is not None:
        return False, f"deny:domain:{denied[0]}"

    path = _url_path(url)

    # 4. The agency's own press shop is not its measurement function.
    p = _first_prefix(path, reg.press_prefixes)
    if p is not None:
        return False, f"deny:press-prefix:{p}"

    # 5. The allowlist itself.
    matched = _match_suffix(host, reg.entries_by_domain)
    if matched is None:
        return False, "not-listed"          # 6. fail closed
    domain, entry = matched
    assert isinstance(entry, Entry)
    p = _first_prefix(path, entry.deny_paths)
    if p is not None:
        return False, f"deny:entry-path:{domain}{p}"
    if entry.allow_paths:
        p = _first_prefix(path, entry.allow_paths)
        if p is None:
            return False, f"deny:path-not-allowed:{domain}"
        return True, f"entry:{domain}/path:{p}"
    return True, f"entry:{domain}"


def is_statistical_agency(url: str) -> bool:
    """Boolean form of :func:`classify_ex`."""
    return classify_ex(url)[0]


def agency_for(url: str) -> str:
    """The allowlisted agency short name for ``url``, or ``""`` if denied."""
    allowed, _ = classify_ex(url)
    if not allowed:
        return ""
    matched = _match_suffix(url_host(url), load_registry().entries_by_domain)
    return matched[1].agency if matched else ""      # type: ignore[union-attr]
