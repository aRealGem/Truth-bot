"""Era-scoped principal relations: is a source the speaker's own organization?
(PR-A2.1, the ``principal_relation`` input to the Evidential Role Axis.)

The tier ladder stays global and speaker-neutral — ``whitehouse.gov`` is S5
for everyone. What the ladder cannot express is the *claim-relative* fact that
for one particular speaker the same domain is the speaker's own press shop:
the only witness being the claimant is a finding worth displaying (Phase 1)
and, post-D11, an input to evidential-role quota logic (Phase 3).

Invariant I3 discipline — RELATIONAL, never CONDITIONAL: the speaker enters
:func:`principal_relation` as an *argument* to a total function computed
identically for every speaker; every fact that names a person lives in the
versioned data table (``principals.json``, the same precedent as
``source_tiers.json`` naming ``obamawhitehouse.archives.gov``). No code path
branches on who is being analyzed. See ``docs/integrity-invariants.md``.

Era scoping: a domain is SELF only while the utterance date falls inside one
of the speaker's eras, so ``whitehouse.gov`` is SELF for Obama on 2014-01-28
and INDEPENDENT for every other speaker on that same date. Unknown speakers
and undated claims fail OPEN to INDEPENDENT — this module must never manufacture
a self-sourcing finding it cannot ground.

``PARTICIPANT`` (a named participant in the claimed event publishing on its
own domain) is part of the enum contract now but is not computed until the
Phase 3 role axis lands (D11): it needs per-claim participant entities, which
no deterministic source provides yet.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Union

from truthbot.domains import host_matches, url_host

_PRINCIPALS_PATH = Path(__file__).resolve().parent / "principals.json"


class PrincipalRelation(str, Enum):
    SELF = "self"                # source org = speaker's administration/party/campaign
    PARTICIPANT = "participant"  # named event participant on its own domain (Phase 3)
    INDEPENDENT = "independent"  # everything else — normal tier handling


class _Era:
    __slots__ = ("start", "end", "domains")

    def __init__(self, doc: dict) -> None:
        self.start = date.fromisoformat(doc["start"])
        end = doc.get("end")
        self.end = date.fromisoformat(end) if end else None
        self.domains = tuple(
            d.lower()
            for key in ("administration_domains", "party_domains", "campaign_domains")
            for d in doc.get(key) or ())

    def covers(self, when: date) -> bool:
        return self.start <= when and (self.end is None or when < self.end)


@lru_cache(maxsize=1)
def _eras_by_speaker() -> dict[str, tuple[_Era, ...]]:
    doc = json.loads(_PRINCIPALS_PATH.read_text(encoding="utf-8"))
    table: dict[str, tuple[_Era, ...]] = {}
    for entry in doc.get("principals") or ():
        eras = tuple(_Era(e) for e in entry.get("eras") or ())
        for name in (entry.get("speaker", ""), *(entry.get("aliases") or ())):
            key = _norm(name)
            if key:
                table[key] = eras
    return table


def _norm(speaker: str) -> str:
    return " ".join((speaker or "").casefold().split())


def _coerce_date(when: Union[date, datetime, str, None]) -> Optional[date]:
    if isinstance(when, datetime):
        return when.date()
    if isinstance(when, date):
        return when
    if isinstance(when, str) and when.strip():
        try:
            return date.fromisoformat(when.strip()[:10])
        except ValueError:
            return None
    return None


def principal_relation(url: str, speaker: str,
                       utterance_date: Union[date, datetime, str, None],
                       participants: Sequence[str] = (),
                       ) -> PrincipalRelation:
    """Relation of the source behind ``url`` to ``speaker`` at utterance time.

    Identically computed for every speaker (I3): looks the speaker up in the
    data table, finds the era covering ``utterance_date``, and suffix-matches
    the URL host against that era's principal domains. Any gap — unknown
    speaker, unparseable/absent date, no covering era, no host — returns
    INDEPENDENT (fail-open: display code may under-flag, never over-flag).

    ``participants`` (PR-A2.3, D11.5.1): explicit per-claim participant
    domains — orgs NAMED in the claimed event, publishing on their own domain.
    Deterministic and caller-supplied (never inferred here, never by an LLM);
    SELF wins over PARTICIPANT when both match. Empty (the default, and the
    only value any live caller passes until the participant-entity lane
    exists) preserves the exact A2.1 behavior.
    """
    host = url_host(url)
    when = _coerce_date(utterance_date)
    if not host or when is None:
        return PrincipalRelation.INDEPENDENT
    for era in _eras_by_speaker().get(_norm(speaker), ()):
        if era.covers(when) and any(host_matches(host, d) for d in era.domains):
            return PrincipalRelation.SELF
    if any(host_matches(host, d.lower()) for d in participants or ()):
        return PrincipalRelation.PARTICIPANT
    return PrincipalRelation.INDEPENDENT
