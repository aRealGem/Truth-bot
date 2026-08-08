"""Deterministic evidence consolidator — shared_pack_v2 (P67.7 / T2.3-T2.4).

Design note: wiki ``projects:truthbot:evidence-v2-design`` (published before
this code, per the wiki-first rule). The consolidator takes the retriever
shortlists (PR-5: R1 Opus/Lane-Worker native search, R2 GPT-5.5 browsing,
R3 pending D1) and assembles the pack with NO small-model re-scoring — every
step is pure code, so the same shortlists always produce the same pack:

  1. round-robin merge of the shortlists (each retriever's own order is
     preserved; no retriever can dominate the head of the pack),
  2. URL dedup (first drawn wins),
  3. era filter — the originally-coded window AND the fair-game window
     (utterance + 7 days), via :mod:`truthbot.verdict.era_lint`,
  4. fact-checker exclusion (T2.1) — enforced here even though the
     retrievers filter too (a retriever that forgets is caught),
  5. junk-URL filter (homepages / listing pages),
  6. tier quotas: at least ``MIN_BEARING_T13`` Tier-1..3 items bearing on
     the core assertion (stance supports/refutes), at most ``MAX_T6`` OTHER
     items; excess OTHER items are dropped lowest-priority-first,
  7. cap at ``PACK_CAP_V2`` (10) items, ordered by (draw round, tier rank).

Quality gate (T2.4): quota unmet → the caller performs ONE targeted
re-retrieval and consolidates again; still unmet → the claim's verdict is
FORCED Unverifiable with provenance code ``insufficient-qualifying-evidence``
(``GATE_INSUFFICIENT``). No silent thin-pack verdicts.

evidence_pack v2.0 payload per item: {url, date, tier, stance, one_line_why}.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional, Sequence

from truthbot.domains import is_substantive_url
from truthbot.models import Evidence, SourceTier
from truthbot.verify.factcheck_exclusion import factcheck_exclusion_reason
from truthbot.verify.mutable_endpoints import is_mutable_latest
from truthbot.verify.tier_registry import QUARANTINE_REASON, classify_tier_ex

from . import era_lint

PACK_CAP_V2 = 10
MIN_BEARING_T13 = 2          # ≥2 Tier-1..3 items bearing on the core assertion
MAX_T6 = 2                   # ≤2 OTHER-tier items
MAX_S5 = 3                   # ≤3 POLITICAL items (PR-A2.2 / T2.1, symmetric with MAX_T6)
GATE_INSUFFICIENT = "insufficient-qualifying-evidence"

SCHEMA_VERSION = "evidence_pack v2.0"

_TIER_RANK = {
    SourceTier.GOVERNMENT: 0,
    SourceTier.WIRE: 1,
    SourceTier.ESTABLISHED: 2,
    SourceTier.ACADEMIC: 3,
    SourceTier.FACTCHECK: 4,   # unreachable in v2 (excluded), kept for totality
    SourceTier.OTHER: 5,
    SourceTier.POLITICAL: 6,   # S5 — ranks below OTHER (Claim Eval v3 PR-A / D7)
}

# Tiers that can credit the decided-verdict quota. POLITICAL is deliberately
# ABSENT: a partisan press release may show a claim was made, never that it is
# true, so it must not be one of the MIN_BEARING_T13 items that let a claim
# reach a decided verdict.
_T13 = {SourceTier.GOVERNMENT, SourceTier.WIRE, SourceTier.ESTABLISHED}


POST_SPEECH_NOTE = "post-speech · context-only"

# ── Scoring-coverage telemetry (remediation v2 Phase A, A1) ──────────────────
#
# The v2 pack path NEVER scores relevance or stance. ``verify.relevance.
# score_evidence`` — the only writer of ``relevance_score`` / ``supports_claim``
# — is reachable from the legacy v1 provider (pipeline._build_open_book_provider)
# and the R4 archive retriever ONLY; ``build_evidence_pack_v2`` wires the
# R1/R2/R3 shortlists straight into :func:`consolidate`. So every v2 item keeps
# the pydantic default relevance (models.py Evidence.relevance_score = 0.5) and
# whatever stance the retriever's own JSON claimed — with retrievers.py mapping
# stance "context" → None. Since ``_bearing`` requires True/False, those nulls
# can never credit MIN_BEARING_T13 and the pack gate-forces Unverifiable.
#
# None of that was VISIBLE on disk: nothing recorded how much of a pack was
# unscored. These helpers make the coverage a first-class, journaled number so
# the condition is measurable (and lintable) instead of inferred.

#: ``models.Evidence.relevance_score``'s pydantic default. An item still
#: carrying it was never seen by the relevance layer. ``None`` counts as
#: unscored too (``PackItem.relevance_score`` defaults to None on packs built
#: with no relevance layer at all).
DEFAULT_RELEVANCE_SCORE = 0.5

#: Keys of the per-pack scoring telemetry dict, in report order.
SCORING_KEYS = ("items", "relevance_scored", "relevance_default",
                "stance_supports", "stance_refutes", "stance_null")


def _field(obj, name):
    """Read ``name`` off an attribute-style object OR a mapping.

    Evidence, PackItem, ConsolidatedItem-wrapped Evidence and the plain dicts
    stored in a run artifact's ``evidence`` map all answer the same two
    questions; one accessor lets live packs and stored artifacts share ONE
    telemetry implementation (so the lint over old artifacts and the field
    written by new runs can never drift apart)."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    ev = getattr(obj, "evidence", None)
    if ev is not None and not hasattr(obj, name):
        obj = ev
    return getattr(obj, name, None)


def scoring_telemetry(items: Iterable) -> dict:
    """Scoring coverage for one pack's items.

    ``{items, relevance_scored, relevance_default, stance_supports,
    stance_refutes, stance_null}`` — counts only, so the journal line stays
    small. ``relevance_default`` counts items whose relevance is the untouched
    0.5 default (or None, i.e. no relevance layer at all); everything else is
    ``relevance_scored``. Stance counts partition the pack by
    ``supports_claim`` True / False / None."""
    tel = dict.fromkeys(SCORING_KEYS, 0)
    for it in items:
        tel["items"] += 1
        rel = _field(it, "relevance_score")
        if rel is None or rel == DEFAULT_RELEVANCE_SCORE:
            tel["relevance_default"] += 1
        else:
            tel["relevance_scored"] += 1
        stance = _field(it, "supports_claim")
        if stance is True:
            tel["stance_supports"] += 1
        elif stance is False:
            tel["stance_refutes"] += 1
        else:
            tel["stance_null"] += 1
    return tel


def scoring_telemetry_from_artifact(evidence: Mapping) -> dict:
    """The SAME telemetry, summed over a STORED run artifact's ``evidence``
    dict (``{sid: [evidence dict, …]}``).

    Artifacts predating the ``EvidencePack.scoring`` field carry no telemetry
    of their own, so the lint recomputes it from what is actually on disk.
    Adds ``packs`` plus the two derived rates the fitness lint thresholds on
    (0.0 on an empty artifact — no division by zero, no crash)."""
    tel = dict.fromkeys(SCORING_KEYS, 0)
    packs = 0
    for items in (evidence or {}).values():
        packs += 1
        one = scoring_telemetry(items or [])
        for k in SCORING_KEYS:
            tel[k] += one[k]
    n = tel["items"]
    tel["packs"] = packs
    tel["scored_rate"] = (tel["relevance_scored"] / n) if n else 0.0
    tel["stance_null_rate"] = (tel["stance_null"] / n) if n else 0.0
    return tel


@dataclass(frozen=True)
class ConsolidatedItem:
    """One v2 pack item plus its merge provenance."""
    evidence: Evidence
    draw_round: int          # which round-robin round drew it
    retriever: str           # shortlist label it came from
    # Evidential role (PR-A2.3): non-empty only on role-aware consolidations
    # (claim_shape + relation_of supplied). Rides into the panel payload so an
    # attribution-only self record is visibly non-probative to the seats.
    role: str = ""
    # Post-speech band (remediation v2, 1.3): dated after the utterance but
    # within the fair-game window. Admitted for context, NEVER verdict-bearing
    # (cannot credit the quota) — same-speech fact-checks and reaction
    # coverage live in exactly this band.
    post_speech: bool = False
    # D15 (flag-gated, default OFF): which ``verdict.utterance_record`` rule
    # found this item to be a record of the SPEECH ITSELF ('' = no rule, or the
    # flag is off). When set, ``role`` is forced to ``utterance-record`` and the
    # item credits nothing. Kept ALONGSIDE ``role`` so the journal records WHICH
    # rule fired, not merely that one did.
    utterance_rule: str = ""

    def to_payload_v2(self) -> dict:
        ev = self.evidence
        stance = {True: "supports", False: "refutes"}.get(ev.supports_claim, "context")
        payload = {
            "url": ev.source_url,
            "date": ev.published_at.date().isoformat() if ev.published_at else None,
            "tier": ev.source_tier.value,
            "stance": stance,
            # B2: when the scorer stated the COMPARISON it made, that is a far
            # better line than the snippet — it says why this item bears on the
            # claim rather than merely what the page says. Falls back to the
            # snippet, which is what every item scored before B2 carries.
            "one_line_why": ((ev.one_line_why or "").strip()
                             or (ev.snippet or "").strip())[:200],
        }
        if getattr(ev, "arithmetic_hinge", False):
            # The stance came from arithmetic the SCORER performed over the
            # series, so it is a hypothesis for the seats to check, not a
            # settled reading. Saying so in the payload is the point: the panel
            # must not treat it as proof (R-2 computed-exhibit routing).
            payload["arithmetic_hinge"] = True
        if self.role and self.role != "normal":
            payload["role"] = self.role
        if self.post_speech:
            payload["era_note"] = POST_SPEECH_NOTE
        return payload


@dataclass
class ConsolidationResult:
    sid: str
    items: list[ConsolidatedItem] = field(default_factory=list)
    quota_met: bool = False
    gate_code: str = ""              # GATE_INSUFFICIENT when forced-UV applies
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count
    # Per-item fact-check exclusion log (remediation v2, 1.1): every FC drop
    # is recorded — {"url", "reason", "retriever"} — and journaled with the
    # pack. Exclusions are never silent.
    excluded_fc: list[dict] = field(default_factory=list)
    # Quarantine telemetry (remediation v2, 1.2 / S-6): kept items whose tier
    # came from the fail-closed quarantine of an unmapped government-class
    # host (tier_registry reason "quarantine-unmapped-gov"). They are KEPT —
    # classified POLITICAL, so they can never credit the quota — but journaled
    # so an unmapped-host burst is visible instead of silently bottom-tiered.
    quarantined: list[str] = field(default_factory=list)
    retrospective: int = 0           # lenient mode: admitted post-era items
    # The full post-filter/post-quota candidate list BEFORE the pack cap, in
    # final order (PR-A2.2). Persisting this is what makes "would claim X have
    # decided under a different cap/quota?" answerable offline later — the
    # Obama-2014 measurement could NOT be re-run locally because only capped
    # packs were stored.
    pre_cap_items: list[ConsolidatedItem] = field(default_factory=list)
    # Evidential-role tally over the kept pool (PR-A2.3), e.g.
    # {"normal": 6, "primary-record": 2, "attribution-only": 1}. Empty on
    # non-role-aware consolidations.
    role_tally: dict[str, int] = field(default_factory=dict)
    # D15 utterance-record telemetry (flag-gated, default OFF): one
    # {"url", "rule"} per kept item classified as a record of the speech
    # itself. ALWAYS empty with the flag off — which is how a reader can tell
    # "D15 found nothing here" from "D15 was not running". Unlike role_tally
    # this is populated on role-aware AND legacy consolidations, because the
    # exclusion does not depend on the claim shape.
    utterance_records: list[dict] = field(default_factory=list)

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def to_payload(self) -> list[dict]:
        return [it.to_payload_v2() for it in self.items]


def _bearing(ev: Evidence) -> bool:
    """An item 'bears on the core assertion' when the stance layer classified
    it as supports or refutes (not ambiguous context)."""
    return ev.supports_claim is True or ev.supports_claim is False


def _d15_on(explicit: Optional[bool]) -> bool:
    """Is the D15 utterance-record exclusion active for this consolidation?

    ``None`` (the default) defers to the ``TRUTHBOT_D15_UTTERANCE_RECORD``
    environment flag, which is OFF unless set — so production behaviour is
    unchanged until the switch is thrown. An explicit True/False is the same
    switch as an argument, for tests and the $0 blast-radius measurement."""
    from truthbot.verdict import utterance_record

    return utterance_record.flag_enabled() if explicit is None else bool(explicit)


def _round_robin(shortlists: Sequence[tuple[str, Sequence[Evidence]]]):
    """Yield (draw_round, retriever_label, evidence) interleaving the lists."""
    idx = 0
    while True:
        emitted = False
        for label, items in shortlists:
            if idx < len(items):
                emitted = True
                yield idx, label, items[idx]
        if not emitted:
            return
        idx += 1


def consolidate(
    sid: str,
    shortlists: Sequence[tuple[str, Sequence[Evidence]]],
    *,
    utterance: Optional[date],
    window: Optional[tuple[date, date]] = None,
    max_items: int = PACK_CAP_V2,
    era_mode: str = "strict",
    claim_shape: str = "",
    relation_of=None,
    utterance_record: Optional[bool] = None,
) -> ConsolidationResult:
    """Assemble a shared_pack_v2 pack from retriever shortlists.

    ``shortlists`` is an ordered sequence of ``(retriever_label, items)``;
    each retriever's list is in ITS preference order. Deterministic: no
    randomness, no model calls, stable ordering throughout.

    ``era_mode`` (historical-era policy, wiki projects:truthbot:
    historical-era-design): "strict" (default) DROPS items dated outside the
    coded window / fair-game end. "lenient" (pre-web speeches) ADMITS them —
    ranked behind contemporaneous sources — and lets a GOVERNMENT-tier item
    dated within the era count toward the quota even when its stance is
    neutral (a 1973 BLS table IS bearing evidence). Fact-checker exclusion,
    dedup, and provenance rules are identical in both modes.

    Evidential-role axis (PR-A2.3, D11-approved): ``claim_shape`` is the
    Layer A shape (c-exist/c-count/c-eval/c-third; '' = legacy) and
    ``relation_of`` an ``Evidence -> PrincipalRelation`` callable the caller
    closes over speaker + utterance (the consolidator itself stays
    speaker-ignorant; the callable is identical machinery for every speaker —
    I3-relational). When BOTH are supplied the quota is role-aware per the
    D11.2 table: a PRIMARY-RECORD self item may fill at most one slot, a
    PARTICIPANT corroborant fills the independent slot, an ATTRIBUTION-ONLY
    item satisfies nothing, and a decided verdict always needs at least one
    non-self credit. With either absent, quota behavior is bit-for-bit
    today's.

    D15 utterance-record exclusion (PROPOSED, FLAG-GATED, DEFAULT OFF — see
    ``docs/decisions/D15-utterance-derivative.md``): ``utterance_record=None``
    (the default) defers to ``TRUTHBOT_D15_UTTERANCE_RECORD``, which is unset
    in production, so NOTHING about the gate changes until the owner ratifies.
    Switched on, items ``verdict.utterance_record`` identifies as records of
    THIS speech — the DCPD transcript, the day's Congressional Record, the
    Weekly Compilation issue, the archive copy of the address, same-speech
    recap coverage — take role ``utterance-record`` and credit the quota ZERO
    on BOTH quota branches. They stay in the pack and stay displayed:
    provenance, never proof. A claim may not witness itself."""
    result = ConsolidationResult(sid=sid)
    seen: set[str] = set()
    kept: list[ConsolidatedItem] = []
    era_class: dict[int, int] = {}   # id(item) -> 0 contemporaneous / 1 undated / 2 retro
    d15_on = _d15_on(utterance_record)
    # The REGISTERED speech date is the anchor every D15 rule checks against
    # (a Congressional Record from another day is a different day's business).
    # Resolved once per pack, and only when the flag is on, so the flag-off
    # path does not even take the import.
    d15_speech_date = None
    if d15_on:
        from truthbot.verdict import speech_context, utterance_record as _ur

        d15_speech_date = speech_context.speech_date_for(sid)

    def _drop(reason: str) -> None:
        result.dropped[reason] = result.dropped.get(reason, 0) + 1

    def _contemporaneous(d: Optional[date]) -> Optional[bool]:
        """True/False for dated items, None for undated."""
        if d is None:
            return None
        if window is not None and not (window[0] <= d <= window[1]):
            return False
        if utterance is not None and d > era_lint.fair_game_end(utterance):
            return False
        return True

    for draw_round, label, ev in _round_robin(shortlists):
        url = (ev.source_url or "").strip()
        if not url:
            _drop("empty-url")
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            _drop("duplicate-url")
            continue
        seen.add(key)
        fc_reason = factcheck_exclusion_reason(url)
        if not fc_reason and ev.source_tier == SourceTier.FACTCHECK:
            fc_reason = "tier:factcheck"
        if fc_reason:
            _drop("factcheck-excluded")
            result.excluded_fc.append(
                {"url": url, "reason": fc_reason, "retriever": label})
            continue
        if not is_substantive_url(url):
            _drop("non-substantive-url")
            continue
        if is_mutable_latest(url):
            # Live latest-release pointers drift out of the claim's era no
            # matter what date retrieval saw (remediation v2, 1.3).
            _drop("mutable-latest-endpoint")
            continue
        d = era_lint.item_date(ev.published_at, ev.snippet or "")
        contemp = _contemporaneous(d)
        if contemp is False and era_mode != "lenient":
            if window is not None and d is not None and not (window[0] <= d <= window[1]):
                _drop("outside-coded-window")
            else:
                _drop("after-fair-game-window")
            continue
        if contemp is False:
            result.retrospective += 1
        post = (utterance is not None and d is not None
                and utterance < d <= era_lint.fair_game_end(utterance))
        # D15: decided HERE, from the URL + snippet + registered speech date,
        # so the role is fixed before any ordering or capping and cannot
        # depend on where the item happened to land in the pack.
        u_rule = ""
        if d15_on:
            u_rule = _ur.utterance_record_rule(
                url, ev.snippet or "", speech_date=d15_speech_date, item_date=d)
        item = ConsolidatedItem(evidence=ev, draw_round=draw_round,
                                retriever=label, post_speech=post,
                                utterance_rule=u_rule,
                                role=_ur.ROLE if u_rule else "")
        era_class[id(item)] = 0 if contemp else (1 if contemp is None else 2)
        # Quarantine telemetry (1.2): computed once per kept item; cheap and
        # additive — the classification itself is unchanged.
        if classify_tier_ex(url)[1] == QUARANTINE_REASON:
            result.quarantined.append(url)
        kept.append(item)

    # T6 quota: keep at most MAX_T6 OTHER items, dropping the lowest-priority
    # (latest-drawn, then worst-tier — here all OTHER, so latest-drawn) first.
    others = [it for it in kept if it.evidence.source_tier == SourceTier.OTHER]
    if len(others) > MAX_T6:
        to_drop = {id(it) for it in others[MAX_T6:]}
        kept = [it for it in kept if id(it) not in to_drop]
        result.dropped["t6-quota"] = len(others) - MAX_T6

    # S5 saturation quota (PR-A2.2 / T2.1): at most MAX_S5 POLITICAL items,
    # first-drawn kept, symmetric with the T6 rule. S5 items can never credit
    # the decided-verdict quota, so saturation buys nothing epistemically —
    # but on official-act claims the press-shop coverage is so dense it can
    # crowd bearing T1–3 items past the pack cap (the T2.3 retrieval-
    # saturation hypothesis). Runs BEFORE the cap so freed slots backfill.
    pols = [it for it in kept if it.evidence.source_tier == SourceTier.POLITICAL]
    if len(pols) > MAX_S5:
        to_drop = {id(it) for it in pols[MAX_S5:]}
        kept = [it for it in kept if id(it) not in to_drop]
        result.dropped["s5-quota"] = len(pols) - MAX_S5

    # Final order: draw round, then tier rank — the round-robin merge is the
    # primary ranking (T2.3), tier breaks ties within a round. Lenient mode
    # prepends the era class so contemporaneous sources ALWAYS outrank
    # undated, which outrank retrospective (strict ordering is unchanged —
    # strict packs never contain retrospective items).
    if era_mode == "lenient":
        kept.sort(key=lambda it: (era_class.get(id(it), 1), it.draw_round,
                                  _TIER_RANK[it.evidence.source_tier]))
    else:
        kept.sort(key=lambda it: (it.draw_round, _TIER_RANK[it.evidence.source_tier]))
    # Evidential-role annotation (PR-A2.3): computed once per item, attached
    # to both the capped pack and the pre-cap pool so payloads and journals
    # agree. Empty when the caller didn't opt into the role axis.
    role_aware = bool(claim_shape) and relation_of is not None
    if role_aware:
        from dataclasses import replace

        from truthbot.verdict.evidential_role import evidential_role
        # A D15 utterance record keeps the role it was constructed with: it is
        # the stronger statement (credit 0 on every branch), and it is decided
        # by the document's identity rather than by the claim's shape, so the
        # D11.2 table must not overwrite it.
        roles = {id(it): (it.role if it.utterance_rule
                          else evidential_role(claim_shape,
                                               relation_of(it.evidence)).value)
                 for it in kept}
        kept = [replace(it, role=roles[id(it)]) for it in kept]
        # id()s changed with replace(); carry era classes across.
        era_class = {id(it): era_class.get(old_id, 1)
                     for it, old_id in zip(kept, list(roles))}
        for it in kept:
            result.role_tally[it.role] = result.role_tally.get(it.role, 0) + 1

    result.utterance_records = [{"url": it.evidence.source_url,
                                 "rule": it.utterance_rule}
                                for it in kept if it.utterance_rule]

    result.pre_cap_items = list(kept)
    result.items = kept[:max_items]
    if len(kept) > max_items:
        result.dropped["pack-cap"] = len(kept) - max_items

    def _quota_credit(it: ConsolidatedItem) -> bool:
        if it.utterance_rule:
            # D15: a record of the speech itself is provenance, not proof. It
            # cannot credit the quota on ANY tier, in EITHER era mode — a
            # transcript is a GOVERNMENT document and would otherwise sail
            # through both the T1-3 branch and the lenient-GOVERNMENT branch.
            return False
        if it.post_speech:
            # Post-speech band is context-only — it can inform, never decide
            # (remediation v2, 1.3).
            return False
        if it.evidence.source_tier in _T13 and _bearing(it.evidence):
            return True
        # Lenient: an era-contemporaneous GOVERNMENT document counts even
        # when the stance layer called it neutral "context" — archival
        # statistical PDFs rarely take an explicit side, but a 1973 BLS
        # table IS bearing evidence (Nixon probe, 2026-07-24).
        return (era_mode == "lenient"
                and it.evidence.source_tier == SourceTier.GOVERNMENT
                and era_class.get(id(it)) == 0)

    if not role_aware:
        credits = sum(1 for it in result.items if _quota_credit(it))
        result.quota_met = credits >= MIN_BEARING_T13
    else:
        # D11.2 quota: independent credits are today's rule but restricted to
        # non-self, non-participant sources; a CORROBORANT fills the
        # independent slot regardless of base tier; a PRIMARY-RECORD self item
        # contributes at most ONE credit; ATTRIBUTION-ONLY and PLAIN-S5
        # satisfy nothing. A decided verdict always needs ≥1 credit that is
        # not the speaker's own record.
        independent = sum(1 for it in result.items
                          if it.role == "normal" and _quota_credit(it))
        corroborants = sum(1 for it in result.items
                           if it.role == "corroborant" and _bearing(it.evidence)
                           and not it.post_speech)
        primary = sum(1 for it in result.items
                      if it.role == "primary-record" and _bearing(it.evidence)
                      and not it.post_speech)
        credits = independent + corroborants + min(1, primary)
        result.quota_met = (credits >= MIN_BEARING_T13
                            and (independent >= 1 or corroborants >= 1))
    if not result.quota_met:
        result.gate_code = GATE_INSUFFICIENT
    return result
