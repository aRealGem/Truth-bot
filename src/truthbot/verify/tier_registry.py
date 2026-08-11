"""Versioned tier registry — THE source of tier truth (remediation v2, 1.2).

Loads ``tier_registry.yaml`` (schema ``truthbot-tier-registry v1``) and exposes
:func:`classify_tier_ex`, which returns ``(SourceTier, reason)`` — the reason
string names the matched rule so telemetry can distinguish, e.g., a real
POLITICAL classification from the fail-closed quarantine of an unmapped
government host (``"quarantine-unmapped-gov"``).

The public pipeline/renderer API is unchanged: callers keep importing
``classify_tier`` from :mod:`truthbot.verify.source_tiers`, which is now a
facade over this module. See the YAML header for the full evaluation order.

DC-2a (jackie, 2026-08-02) behavior deltas carried by the registry data:
  1. all nine executive-mirror hosts classify POLITICAL,
  2. protected statistical/record functions classify GOVERNMENT regardless of
     press framing (``press: none`` entries / explicit path_classes),
  3. unmapped ``.mil``/``.int`` fail CLOSED to quarantine (was fail-open
     GOVERNMENT); census-enumerated hosts got explicit government entries,
  4. justice.gov is government-base with press paths (incl. ``/opa/pr`` and
     the ``pr`` path segment) demoting via the generic press classes.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from truthbot.domains import url_host
from truthbot.models import SourceTier

_REGISTRY_PATH = Path(__file__).resolve().parent / "tier_registry.yaml"

#: Quarantine maps to POLITICAL for pack semantics (matches prior behavior);
#: the reason string is what distinguishes it for telemetry (S-6).
QUARANTINE_REASON = "quarantine-unmapped-gov"

_TIER_BY_NAME = {
    "government": SourceTier.GOVERNMENT,
    "wire": SourceTier.WIRE,
    "established": SourceTier.ESTABLISHED,
    "academic": SourceTier.ACADEMIC,
    "factcheck": SourceTier.FACTCHECK,
    "other": SourceTier.OTHER,
    "political": SourceTier.POLITICAL,
}

#: TLD classes treated as government-class hosts (dot-label suffix match).
_GOV_TLDS = ("gov", "mil", "int")


@dataclass(frozen=True)
class PathClass:
    prefix: str
    tier: SourceTier
    note: str = ""


@dataclass(frozen=True)
class Entry:
    domain: str
    tier_name: str                    # includes "quarantine"
    rationale: str
    date: str
    press: str = "demote"             # demote | established | none
    path_classes: tuple[PathClass, ...] = ()


@dataclass(frozen=True)
class PatternRule:
    """A host GLOB that classifies a whole class of hosts (Phase A, A3).

    ``entry`` carries identical semantics to a domain :class:`Entry` — the
    same tier/press/path_class machinery runs on a pattern hit — so a pattern
    rule can express anything an entry can, for every host in the class,
    without enumerating them.

    Specificity is ``labels`` (dot-labels in the pattern), which is what ranks
    a pattern against a suffix entry. NOTE fnmatch semantics: ``*`` matches
    dots too, so ``clintonwhitehouse*.archives.gov`` also matches
    ``clintonwhitehousefoo.bar.archives.gov``. Write patterns to the full host
    shape.
    """
    pattern: str
    entry: Entry
    labels: int

    def matches(self, host: str) -> bool:
        return fnmatch.fnmatchcase(host, self.pattern)


@dataclass(frozen=True)
class Registry:
    schema: str
    version: str
    entries_by_domain: dict[str, Entry]
    data_segments: frozenset[str]
    press_prefixes: tuple[str, ...]
    press_segments: frozenset[str]
    substantive_prefixes: tuple[str, ...]
    data_hub_host_labels: frozenset[str]
    stat_data_prefixes: tuple[str, ...]
    stat_press_prefixes: tuple[str, ...]
    #: Host-glob rules, in YAML order (first match wins among patterns).
    patterns: tuple[PatternRule, ...] = ()
    unknown_non_gov: str = "other"
    unknown_gov: str = "quarantine"

    def entries(self) -> list[Entry]:
        return list(self.entries_by_domain.values())


def _parse_entry(raw: dict) -> Entry:
    tier_name = str(raw["tier"]).lower()
    if tier_name != "quarantine" and tier_name not in _TIER_BY_NAME:
        raise ValueError(f"tier_registry: unknown tier {tier_name!r} "
                         f"for {raw.get('domain')!r}")
    pcs = []
    for pc in raw.get("path_classes") or ():
        pcs.append(PathClass(prefix=str(pc["prefix"]).lower(),
                             tier=_TIER_BY_NAME[str(pc["tier"]).lower()],
                             note=str(pc.get("note", ""))))
    return Entry(
        domain=str(raw["domain"]).lower(),
        tier_name=tier_name,
        rationale=str(raw.get("rationale", "")),
        date=str(raw.get("date", "")),
        press=str(raw.get("press", "demote")).lower(),
        path_classes=tuple(pcs),
    )


def _parse_pattern(raw: dict) -> PatternRule:
    """Build a :class:`PatternRule`, refusing patterns that are too broad.

    FAIL-CLOSED GUARDS — a pattern rule must never silently widen the
    registry's reach:
      * it must actually contain a wildcard (``*`` / ``?`` / ``[``); a literal
        host belongs in ``entries`` where the duplicate check applies,
      * its final TWO dot-labels must be wildcard-free, so no rule can claim a
        whole TLD or registry (``*.gov``, ``*``) by accident. The class a
        pattern names has to be a class, not the internet.
    """
    pattern = str(raw["pattern"]).lower().strip()
    if not any(c in pattern for c in "*?["):
        raise ValueError(
            f"tier_registry: pattern {pattern!r} has no wildcard — a literal "
            "host belongs in entries:, not patterns:")
    labels = pattern.split(".")
    if len(labels) < 3 or any(c in lbl for lbl in labels[-2:] for c in "*?["):
        raise ValueError(
            f"tier_registry: pattern {pattern!r} is too broad — the last two "
            "dot-labels must be literal, so a pattern can never claim a whole "
            "TLD or registry")
    entry = _parse_entry({**raw, "domain": pattern})
    return PatternRule(pattern=pattern, entry=entry, labels=len(labels))


@lru_cache(maxsize=1)
def load_registry() -> Registry:
    doc = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    if doc.get("schema") != "truthbot-tier-registry v1":
        raise ValueError(f"tier_registry: unexpected schema {doc.get('schema')!r}")
    defaults = doc.get("defaults") or {}
    unknown = doc.get("unknown") or {}
    entries: dict[str, Entry] = {}
    for raw in doc.get("entries") or ():
        e = _parse_entry(raw)
        if e.domain in entries:
            raise ValueError(f"tier_registry: duplicate entry {e.domain!r}")
        entries[e.domain] = e
    patterns = tuple(_parse_pattern(raw) for raw in doc.get("patterns") or ())
    seen_patterns: set[str] = set()
    for p in patterns:
        if p.pattern in seen_patterns:
            raise ValueError(f"tier_registry: duplicate pattern {p.pattern!r}")
        seen_patterns.add(p.pattern)
    return Registry(
        schema=str(doc["schema"]),
        version=str(doc.get("version", "")),
        entries_by_domain=entries,
        patterns=patterns,
        data_segments=frozenset(s.lower() for s in defaults.get("data_segments") or ()),
        press_prefixes=tuple(p.lower() for p in defaults.get("press_prefixes") or ()),
        press_segments=frozenset(s.lower() for s in defaults.get("press_segments") or ()),
        substantive_prefixes=tuple(
            p.lower() for p in defaults.get("substantive_prefixes") or ()),
        data_hub_host_labels=frozenset(
            s.lower() for s in defaults.get("data_hub_host_labels") or ()),
        stat_data_prefixes=tuple(
            p.lower() for p in defaults.get("stat_data_prefixes") or ()),
        stat_press_prefixes=tuple(
            p.lower() for p in defaults.get("stat_press_prefixes") or ()),
        unknown_non_gov=str(unknown.get("non_gov", "other")),
        unknown_gov=str(unknown.get("gov_mil_int", "quarantine")),
    )


# ── URL helpers ──────────────────────────────────────────────────────────────

def _url_path(url: str) -> str:
    """Lowercased path of ``url`` (``'/'``-normalised), or ``''`` if unparseable."""
    try:
        path = urlsplit(url if "://" in url else f"//{url}").path or "/"
    except ValueError:
        return ""
    return path.lower()


def _path_segments(path: str) -> tuple[str, ...]:
    """Exact-segment matching, never substring: ``/news/data-shows-x`` does not
    count ``data-shows-x`` as a data segment."""
    return tuple(s for s in path.split("/") if s)


def _first_prefix(path: str, prefixes: tuple[str, ...]) -> str | None:
    for p in prefixes:
        if path.startswith(p):
            return p
    return None


def _is_gov_class_host(host: str) -> bool:
    return host.rsplit(".", 1)[-1] in _GOV_TLDS


def _match_entry(host: str, reg: Registry) -> Entry | None:
    """Longest-suffix entry for ``host`` (dot-label boundaries, so
    ``www.govtech.com`` never matches a ``.gov`` rule)."""
    return (_match_entry_ex(host, reg) or (None, 0))[0]


def _match_entry_ex(host: str, reg: Registry) -> tuple[Entry, int] | None:
    """Longest-suffix entry plus HOW MANY dot-labels it matched.

    The label count is the entry's specificity, which is what a pattern rule
    is ranked against (see :func:`classify_tier_ex`)."""
    labels = host.split(".")
    # candidate suffixes from most-specific to least: a.b.c.d → a.b.c.d,
    # b.c.d, c.d, d — first hit is the longest match.
    for i in range(len(labels) - 1):
        candidate = ".".join(labels[i:])
        e = reg.entries_by_domain.get(candidate)
        if e is not None:
            return e, len(labels) - i
    return None


def _match_pattern(host: str, reg: Registry) -> PatternRule | None:
    """First pattern rule (YAML order) whose glob matches the WHOLE host."""
    for rule in reg.patterns:
        if rule.matches(host):
            return rule
    return None


# ── the generic government-class pipeline ("data yes, press no") ────────────

def _generic_gov(host: str, path: str, reg: Registry,
                 *, substantive: bool, fallback: tuple[SourceTier, str],
                 ) -> tuple[SourceTier, str]:
    """Shared path logic for demote-class entries, quarantine entries and
    unmapped government hosts. Order: data-hub host label, data segment,
    press prefix, press segment, (optionally) substantive prefix, fallback."""
    label = host.split(".", 1)[0]
    if label in reg.data_hub_host_labels:
        return SourceTier.GOVERNMENT, f"data-hub-host:{label}"
    segs = _path_segments(path)
    hit = reg.data_segments.intersection(segs)
    if hit:
        return SourceTier.GOVERNMENT, f"data-segment:{sorted(hit)[0]}"
    p = _first_prefix(path, reg.press_prefixes)
    if p is not None:
        return SourceTier.POLITICAL, f"press-prefix:{p}"
    hit = reg.press_segments.intersection(segs)
    if hit:
        return SourceTier.POLITICAL, f"press-segment:{sorted(hit)[0]}"
    if substantive:
        p = _first_prefix(path, reg.substantive_prefixes)
        if p is not None:
            return SourceTier.GOVERNMENT, f"substantive-prefix:{p}"
    return fallback


def _apply_entry(entry: Entry, host: str, path: str, reg: Registry,
                 base_reason: str) -> tuple[SourceTier, str]:
    """Entry semantics — shared by domain entries and pattern rules, so a
    pattern classifies EXACTLY as the equivalent entry would (Phase A, A3)."""
    for pc in entry.path_classes:
        if path.startswith(pc.prefix):
            return pc.tier, f"{base_reason}/path:{pc.prefix}"
    if entry.tier_name == "quarantine":
        # partisan-by-construction host class: generic rules apply
        # (data yes, press no, record paths survive), fallback POLITICAL.
        return _generic_gov(
            host, path, reg, substantive=True,
            fallback=(SourceTier.POLITICAL, f"{base_reason}/quarantine"))
    tier = _TIER_BY_NAME[entry.tier_name]
    if tier is not SourceTier.GOVERNMENT:
        return tier, base_reason
    if entry.press == "none":
        # DC-2a delta 2: protected statistical/record function — press
        # framing never demotes.
        return SourceTier.GOVERNMENT, base_reason
    if entry.press == "established":
        # the nonpartisan carve-out: real data on press-looking paths
        # stays GOVERNMENT (bls.gov/news.release/* is the jobs report);
        # a genuine press-shop path is ESTABLISHED — demoted, not
        # condemned; default GOVERNMENT, never quarantined.
        p = _first_prefix(path, reg.stat_data_prefixes)
        if p is not None:
            return SourceTier.GOVERNMENT, f"stat-data-prefix:{p}"
        p = _first_prefix(path, reg.stat_press_prefixes)
        if p is not None:
            return SourceTier.ESTABLISHED, f"stat-press-prefix:{p}"
        return SourceTier.GOVERNMENT, base_reason
    # press == "demote": generic data/press classes still apply.
    return _generic_gov(host, path, reg, substantive=False,
                        fallback=(SourceTier.GOVERNMENT, base_reason))


def classify_tier_ex(url: str) -> tuple[SourceTier, str]:
    """Classify ``url`` and say WHY — ``(tier, reason)``.

    Reason strings name the matched rule: ``"entry:bls.gov"``,
    ``"entry:senate.gov/path:/newsroom"``,
    ``"pattern:clintonwhitehouse*.archives.gov"``, ``"press-prefix:/news/"``,
    ``"data-segment:stats"``, ``"quarantine-unmapped-gov"``,
    ``"unmapped-non-gov"``.

    PRECEDENCE (Phase A, A3 — entries and patterns rank TOGETHER by
    specificity, measured in dot-labels matched):

      1. the most specific host rule wins. An exact entry for the full host
         (3 labels for ``clintonwhitehouse3.archives.gov``) beats a pattern
         matching the same host, so every explicit entry keeps working
         untouched; a pattern (3 labels) beats a SHORTER suffix entry
         (``archives.gov``, 2 labels), which is the whole point — the mirror
         class must not inherit the National Archives' Government tier;
      2. ties go to the exact entry — a pattern never overrides an explicit
         decision about a specific host;
      3. below both, unchanged: unmapped ``.gov``/``.mil``/``.int`` run the
         generic pipeline and fall back to POLITICAL/``quarantine-unmapped-
         gov``, unmapped non-gov is OTHER.

    Patterns are therefore ADDITIVE ONLY. A host matching no entry and no
    pattern still follows the existing unknown policy, so nothing is widened
    silently; and ``_parse_pattern`` refuses a pattern whose last two labels
    are not literal, so no rule can claim a whole TLD.
    """
    host = url_host(url)
    if not host:
        return SourceTier.OTHER, "no-host"
    reg = load_registry()
    path = _url_path(url)

    matched = _match_entry_ex(host, reg)
    rule = _match_pattern(host, reg)
    if rule is not None and (matched is None or rule.labels > matched[1]):
        return _apply_entry(rule.entry, host, path, reg,
                            f"pattern:{rule.pattern}")
    if matched is not None:
        entry = matched[0]
        return _apply_entry(entry, host, path, reg, f"entry:{entry.domain}")

    if _is_gov_class_host(host):
        # Unmapped government-class host: the generic path rules still apply
        # (identical to today's .gov handling), but the FALLBACK fails closed
        # for .mil/.int too now (DC-2a delta 3; they used to fail open).
        return _generic_gov(host, path, reg, substantive=True,
                            fallback=(SourceTier.POLITICAL, QUARANTINE_REASON))
    return SourceTier.OTHER, "unmapped-non-gov"
