"""The decidability axis — is this claim checkable at all? (D17-d, Q2)

``data/decidability.json`` at the repo root is the system of record:

    {"schema": "truthbot-decidability v1",
     "entries": [
        {"sid": "trump_2026:0153",
         "speech_id": "trump_2026",
         "decidability": "undecidable-from-public-record",
         "provenance": "owner-ratified",
         "date": "2026-08-14",
         "why": "A private conversation with Michael Dell. Nothing public could settle what was said.",
         "review_trigger": "A published account by either party, or a contemporaneous record of the exchange."}]}

WHY THIS EXISTS. The evidence gate records that a pack did not qualify. It
cannot record whether anything ever could. So a documented valor citation and a
private hospital-room conversation left the gate identical, and the page told
readers both were the same kind of unknowable.

WHY IT IS RECORDED, NOT DERIVED. The D17-d probes tried to derive it from the
pipeline's own structured fields and failed, measurably:

* ``claim_type``/``claim_shape`` are utterance-form taxonomies. Where they
  committed at all they ran ANTI-correlated with decidability — precision 0.235
  and 0.105 against a 0.633 majority-class prior — and every committed error ran
  one way, predicting "undecidable" for a documentable claim.
* Pack anatomy (R7) separates nothing: web-tier1 and substantive differ by 0.07
  items and 0.09 quota credits, and the tier signal is faintly inverted. The
  corpus is conditioned on gate rejection, so the discriminating anatomy is
  already flattened.
* The desk's own classes were not recoverable: compound-split 0 of 5,
  series-core 1 of 7.

Deriving "cannot be verified" from those fields would be admissibility keyed on
the wrong axis (ruling (d)) relocated into a new mechanism. So decidability is
**recorded with provenance**, never inferred from shape.

FAIL CLOSED. ``publishable_entries`` returns ONLY ``owner-ratified`` rows. A
``desk``, ``rule`` or ``model`` assignment is a working note: it is stored, it is
auditable, and it may not reach a page. This mirrors the wave-2 badge rule — no
classification record, no badge — and it is what keeps ccagent's judgement from
being published as the system's.

NEVER SAYS NEVER. ``undecidable-from-public-record`` REQUIRES a
``review_trigger`` naming what would reopen it. A fact-checker does not get to
call a question permanently closed without saying what would change its mind;
the validator enforces that rather than trusting the copy.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "truthbot-decidability v1"

_REQUIRED = ("sid", "speech_id", "decidability", "provenance", "date", "why")

#: The axis. Four values, each naming what the claim's relation to the public
#: record actually is — not what shape the utterance took.
VALUES = {
    # Retrieval would settle it; the lane simply has not been run.
    "retrievable-pending-lane",
    # Retrieval ran and what came back did not qualify. A statement about the
    # pack, which is what the evidence gate has always meant.
    "retrieved-insufficient",
    # No public record reaches it: a private exchange, an unmeasured
    # population, an attribution of interior state. Requires review_trigger.
    "undecidable-from-public-record",
    # A checkable core is buried in a compound utterance; segmentation first.
    "needs-decomposition",
}

#: Who assigned it. Only ``owner-ratified`` may be published.
PROVENANCE = {"desk", "owner-ratified", "rule", "model"}

#: The one publishable provenance. Everything else is a working note.
PUBLISHABLE_PROVENANCE = "owner-ratified"

#: The value whose copy would otherwise assert permanence.
_NEEDS_REVIEW_TRIGGER = "undecidable-from-public-record"


class DecidabilityError(ValueError):
    """decidability.json is malformed — fail the build, don't guess."""


def load_decidability(path: Path) -> list[dict]:
    """Load + validate the registry. Missing file → no entries (empty)."""
    path = Path(path)
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise DecidabilityError(f"{path}: unknown schema {doc.get('schema')!r}")
    entries = doc.get("entries") or []
    seen: set[str] = set()
    for e in entries:
        missing = [k for k in _REQUIRED if not e.get(k)]
        if missing:
            raise DecidabilityError(
                f"{path}: entry {e.get('sid', '?')} missing {missing}")
        if e["decidability"] not in VALUES:
            raise DecidabilityError(
                f"{path}: {e['sid']} bad decidability={e['decidability']!r} "
                f"(expected one of {sorted(VALUES)})")
        if e["provenance"] not in PROVENANCE:
            raise DecidabilityError(
                f"{path}: {e['sid']} bad provenance={e['provenance']!r} "
                f"(expected one of {sorted(PROVENANCE)})")
        if e["decidability"] == _NEEDS_REVIEW_TRIGGER and not e.get("review_trigger"):
            raise DecidabilityError(
                f"{path}: {e['sid']} is {_NEEDS_REVIEW_TRIGGER} with no "
                "review_trigger. A claim may not be recorded as beyond the "
                "public record without naming what would reopen it.")
        if e["sid"] in seen:
            raise DecidabilityError(f"{path}: duplicate entry for {e['sid']}")
        seen.add(e["sid"])
    return entries


def publishable_entries(entries: list[dict]) -> list[dict]:
    """Fail closed: only owner-ratified assignments may reach a page."""
    return [e for e in entries
            if e.get("provenance") == PUBLISHABLE_PROVENANCE]


def by_sid(entries: list[dict], *, publishable_only: bool = True) -> dict[str, dict]:
    """sid → entry lookup.

    Keyed by sid ON PURPOSE. Claim-level metadata carried ON an object gets
    dropped by reconstruction paths — the offline artifact path rebuilds claims
    as ``{sid, text, context, layer_a}`` and bundles come back with
    ``speaker='Unknown'``. A registry keyed by sid cannot be dropped that way,
    because nothing has to carry it.
    """
    rows = publishable_entries(entries) if publishable_only else entries
    return {e["sid"]: e for e in rows}


def decidability_for(entries: list[dict], sid: str,
                     *, publishable_only: bool = True) -> str | None:
    """The recorded decidability for ``sid``, or None if nothing is recorded.

    None means "we have not ratified an answer" — render must treat it as
    absence, never as a default class.
    """
    entry = by_sid(entries, publishable_only=publishable_only).get(sid)
    return entry["decidability"] if entry else None


def summary(entries: list[dict]) -> dict:
    """Counts by value and provenance — for the build log and the audit."""
    out: dict = {"total": len(entries),
                 "publishable": len(publishable_entries(entries)),
                 "by_value": {}, "by_provenance": {}}
    for e in entries:
        out["by_value"][e["decidability"]] = \
            out["by_value"].get(e["decidability"], 0) + 1
        out["by_provenance"][e["provenance"]] = \
            out["by_provenance"].get(e["provenance"], 0) + 1
    return out
