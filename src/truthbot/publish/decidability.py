"""The decidability axis — is this claim checkable at all? (D17-d, Q2)

``data/decidability.json`` at the repo root is the system of record:

    {"schema": "truthbot-decidability v2",
     "entries": [
        {"sid": "trump_2026:0153",
         "speech_id": "trump_2026",
         "decidability": "undecidable-from-public-record",
         "provenance": "owner-ratified",
         "date": "2026-08-14",
         "why": "A private conversation with Michael Dell. Nothing public could settle what was said.",
         "review_trigger": "A published account by either party, or a contemporaneous record of the exchange.",
         "reason_code": "PRIVATE-EVENT"}]}

v2 adds three OPTIONAL fields and changes nothing else, so v1 still loads:
``reason_code`` (the reader-facing explanation, from
``truthbot.publish.reason_codes``), ``reason_code_2`` (AUDIT-ONLY, for a
genuinely dual row — renders never read it), and ``review_after`` (defers
re-review). All three are legal ONLY on ``undecidable-from-public-record``
rows; on any other value they would assert that the public record cannot reach
a claim we merely under-retrieved, which is the exact conflation this axis
exists to prevent.

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

SCHEMA = "truthbot-decidability v2"

#: v1 is still accepted on read. The v2 bump ADDS optional fields
#: (``reason_code``, ``reason_code_2``, ``review_after``) and changes nothing
#: about how a v1 row is interpreted, so refusing v1 would break older registries
#: to no purpose. Anything written from here on carries :data:`SCHEMA`.
_ACCEPTED_SCHEMAS = {"truthbot-decidability v1", SCHEMA}

_REQUIRED = ("sid", "speech_id", "decidability", "provenance", "date", "why")

#: Reason codes are the SUBSTANTIVE species' explanation of why no source could
#: settle a claim, so they are meaningless — and misleading — on any other value.
#: A gate-withheld row carrying one would be asserting the public record cannot
#: reach a claim we simply under-retrieved.
_REASON_CODE_ONLY_ON = "undecidable-from-public-record"

#: ``reason_code_2`` exists for genuinely dual rows (trump_2026:0514 is both a
#: mass attribution and a counterfactual). It is AUDIT-ONLY: renders read
#: ``reason_code`` and never the second, so a dual row still shows one
#: explanation rather than two competing ones.
_OPTIONAL_CODE_FIELDS = ("reason_code", "reason_code_2")

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


def load_decidability(path: Path, *, reason_codes=None) -> list[dict]:
    """Load + validate the registry. Missing file → no entries (empty).

    ``reason_codes`` is an optional registry from
    ``truthbot.publish.reason_codes.load_reason_codes``. Pass it to validate
    that every ``reason_code`` on a row is actually defined; omit it to check
    the v2 fields' shape only."""
    path = Path(path)
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") not in _ACCEPTED_SCHEMAS:
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
        _check_reason_codes(path, e, reason_codes)
        if e["sid"] in seen:
            raise DecidabilityError(f"{path}: duplicate entry for {e['sid']}")
        seen.add(e["sid"])
    return entries


def _check_reason_codes(path, e: dict, registry) -> None:
    """Validate the v2 optional fields. Fail closed, never coerce.

    ``registry`` is an optional reason-code registry (see
    ``truthbot.publish.reason_codes.load_reason_codes``). When supplied, codes
    are checked for MEMBERSHIP too — an undefined code is a build failure, not a
    blank label."""
    primary = e.get("reason_code")
    secondary = e.get("reason_code_2")

    for field in _OPTIONAL_CODE_FIELDS:
        val = e.get(field)
        if val is None:
            continue
        if not isinstance(val, str) or not val.strip():
            raise DecidabilityError(
                f"{path}: {e['sid']} has a non-string/empty {field}")
        if e["decidability"] != _REASON_CODE_ONLY_ON:
            raise DecidabilityError(
                f"{path}: {e['sid']} is {e['decidability']!r} but carries "
                f"{field}={val!r}. Reason codes explain why the PUBLIC RECORD "
                f"cannot reach a claim; on any other value they would assert "
                "that about a claim we merely under-retrieved.")
        if registry is not None:
            from truthbot.publish.reason_codes import known
            if val not in known(registry):
                raise DecidabilityError(
                    f"{path}: {e['sid']} {field}={val!r} is not a defined "
                    "reason code. A code nobody defined cannot be explained to "
                    "a reader.")

    if secondary is not None and primary is None:
        raise DecidabilityError(
            f"{path}: {e['sid']} has reason_code_2 with no reason_code — a "
            "secondary code without a primary has nothing to be secondary to.")
    if secondary is not None and secondary == primary:
        raise DecidabilityError(
            f"{path}: {e['sid']} repeats {primary!r} as both codes")

    after = e.get("review_after")
    if after is not None and (not isinstance(after, str) or not after.strip()):
        raise DecidabilityError(
            f"{path}: {e['sid']} has a non-string/empty review_after")


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
