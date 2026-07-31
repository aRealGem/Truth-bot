"""Composition-bias telemetry for the S5 quarantine (Claim Eval v3, fast-follow).

**Why this exists.** PR-A demotes political communications to S5, and the T2.4
evidence-quality gate then abstains when a claim's only real support was demoted.
That is the *intended* behaviour — but abstention is not free. If quarantine hits
correlate with claim topic (sanctions claims lean on Treasury pages, immigration
claims on DHS pages), then failing closed silently changes **which** claims get
decided, and it can do so unevenly across speakers. Measured on the published
run, the demotion really was speaker-asymmetric: biden_2022's decisive-demoted
citations were ~97% TRUE-supporting, trump_2026's were mixed.

That skew is not a bug to suppress — the underlying evidence genuinely is
distributed that way, and I3 forbids conditioning on *who* is speaking. The
remedy is **visibility**: every run records how often packs carry a quarantined
item and how the decided-vs-Unverifiable rate differs between claims that depend
on one and claims that don't. A shift you can see in the run record is a finding;
the same shift unrecorded is an accusation waiting to happen.

Pure functions over already-built rows + evidence — no I/O, no model calls, so
this can run inside the publish pipeline and offline over stored artifacts alike.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from truthbot.models import SourceTier

#: Verdicts that count as the panel having committed. Anything else
#: (UNVERIFIABLE, disagreement, unresolved) is an abstention.
COMMITTED = frozenset({"TRUE", "FALSE", "MISLEADING"})

_QUARANTINED = SourceTier.POLITICAL.value


def _tier_of(item: Any) -> str:
    """Tier value of an evidence item, tolerating Evidence objects and dicts."""
    if isinstance(item, Mapping):
        raw = item.get("source_tier") or item.get("tier")
    else:
        raw = getattr(item, "source_tier", None) or getattr(item, "tier", None)
    return getattr(raw, "value", raw) or ""


def _supports(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("supports_claim")
    return getattr(item, "supports_claim", None)


def _verdict_of(row: Mapping) -> str:
    return str(row.get("verdict") or row.get("status") or "").strip().upper()


def _speaker_of(sid: str) -> str:
    """Speech id from a sid like ``trump_2026:0010`` (``''`` if unprefixed)."""
    return sid.split(":", 1)[0] if ":" in sid else ""


def _rate(numer: int, denom: int) -> float | None:
    return round(numer / denom, 3) if denom else None


def _url_of(item: Any) -> str:
    if isinstance(item, Mapping):
        return item.get("source_url") or item.get("url") or ""
    return getattr(item, "source_url", "") or getattr(item, "url", "") or ""


def composition_report(
    rows: Iterable[Mapping],
    evidence_by_sid: Mapping[str, list] | None = None,
    *,
    tier_fn=None,
) -> dict:
    """Quarantine-exposure and decided-rate telemetry for one run.

    Returns a dict with an ``overall`` block, a ``by_speaker`` block, and a
    ``decided_rate_gap`` — the headline number: decided-rate among claims whose
    pack carries NO quarantined item, minus the rate among claims that depend on
    one. A large positive gap means the quarantine is materially shifting which
    claims get decided, and the per-speaker table shows whether it lands evenly.

    By default tiers are read from each item's **stored** provenance (the I5
    ``tier`` field), so a run's telemetry describes what that run actually did.
    Pass ``tier_fn=classify_tier`` to re-derive tiers from the URL under *current*
    rules instead — the retrospective "what would this run look like today" view,
    which is the only way to read pre-PR-A artifacts (their stored tiers predate
    the S5 tier, so they would otherwise report zero exposure).
    """
    evidence_by_sid = evidence_by_sid or {}
    buckets: dict[str, dict[str, int]] = {}

    def _tier(item: Any) -> str:
        if tier_fn is None:
            return _tier_of(item)
        derived = tier_fn(_url_of(item))
        return getattr(derived, "value", derived) or ""

    def _bucket(key: str) -> dict[str, int]:
        return buckets.setdefault(key, {
            "claims": 0, "decided": 0,
            "evidence_items": 0, "quarantined_items": 0,
            "exposed": 0, "exposed_decided": 0,          # pack has >=1 S5 item
            "unexposed": 0, "unexposed_decided": 0,
            "sole_quarantined": 0, "sole_quarantined_decided": 0,
        })

    for row in rows:
        sid = str(row.get("sid") or "")
        items = evidence_by_sid.get(sid) or []
        n_quar = sum(1 for i in items if _tier(i) == _QUARANTINED)
        decided = _verdict_of(row) in COMMITTED

        # "Sole-quarantined": every item that actually bears on the claim
        # (supports or contradicts) is quarantined — the collapse-risk shape.
        bearing = [i for i in items if _supports(i) is not None]
        sole = bool(bearing) and all(_tier(i) == _QUARANTINED for i in bearing)

        for key in ("__all__", _speaker_of(sid)):
            if not key:
                continue
            b = _bucket(key)
            b["claims"] += 1
            b["evidence_items"] += len(items)
            b["quarantined_items"] += n_quar
            if decided:
                b["decided"] += 1
            if n_quar:
                b["exposed"] += 1
                b["exposed_decided"] += int(decided)
            else:
                b["unexposed"] += 1
                b["unexposed_decided"] += int(decided)
            if sole:
                b["sole_quarantined"] += 1
                b["sole_quarantined_decided"] += int(decided)

    def _summarize(b: dict[str, int]) -> dict:
        exp_rate = _rate(b["exposed_decided"], b["exposed"])
        unexp_rate = _rate(b["unexposed_decided"], b["unexposed"])
        return {
            "claims": b["claims"],
            "decided": b["decided"],
            "decided_rate": _rate(b["decided"], b["claims"]),
            "evidence_items": b["evidence_items"],
            "quarantined_items": b["quarantined_items"],
            "quarantined_item_share": _rate(b["quarantined_items"], b["evidence_items"]),
            "packs_exposed": b["exposed"],
            "pack_exposure_rate": _rate(b["exposed"], b["claims"]),
            "decided_rate_exposed": exp_rate,
            "decided_rate_unexposed": unexp_rate,
            # positive => claims depending on quarantined evidence are decided LESS often
            "decided_rate_gap": (round(unexp_rate - exp_rate, 3)
                                 if exp_rate is not None and unexp_rate is not None else None),
            "sole_quarantined": b["sole_quarantined"],
            "sole_quarantined_decided": b["sole_quarantined_decided"],
        }

    overall = _summarize(_bucket("__all__"))   # created empty if there were no rows
    by_speaker = {k: _summarize(v) for k, v in sorted(buckets.items()) if k != "__all__"}
    return {
        "schema": "truthbot-composition-telemetry v1",
        "quarantined_tier": _QUARANTINED,
        "overall": overall,
        "by_speaker": by_speaker,
    }


def format_report(report: Mapping) -> str:
    """Human-readable one-block summary, for run logs and review packages."""
    o = report.get("overall") or {}
    lines = [
        "composition telemetry (S5 quarantine exposure)",
        f"  claims {o.get('claims')}  decided {o.get('decided')} "
        f"(rate {o.get('decided_rate')})",
        f"  quarantined evidence {o.get('quarantined_items')}/{o.get('evidence_items')} "
        f"(share {o.get('quarantined_item_share')})",
        f"  packs exposed {o.get('packs_exposed')} (rate {o.get('pack_exposure_rate')})",
        f"  decided-rate exposed {o.get('decided_rate_exposed')} vs "
        f"unexposed {o.get('decided_rate_unexposed')}  GAP {o.get('decided_rate_gap')}",
        f"  sole-quarantined claims {o.get('sole_quarantined')} "
        f"(decided {o.get('sole_quarantined_decided')})",
    ]
    for sp, s in (report.get("by_speaker") or {}).items():
        lines.append(
            f"  [{sp}] exposure {s.get('pack_exposure_rate')} "
            f"decided-rate {s.get('decided_rate')} gap {s.get('decided_rate_gap')}"
        )
    return "\n".join(lines)
