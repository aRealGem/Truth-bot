"""v2 (HydraMind PCA) publish path — Layer A → adjudicate → bridge → VerdictBundles.

This is the orchestration that lets the new PCA verdict stack (Layer A check-worthy
filter + open-book PCA panel + CRM-114 stage-2) drive the EXISTING publisher, in
place of the legacy ``VerificationEngine``. It is deliberately a pure router with
INJECTED lane functions (``layer_a_fn``, ``adjudicate_fn``) — the same discipline as
``checkworthy.pipeline.run_layer_a`` / ``verdict.pipeline.run_layer_b`` — so the
whole flow is unit-testable offline with fakes. The live HydraMind + evidence
provider are constructed by the caller (the ``publish`` CLI) and closed over by the
two functions; nothing here talks to a network.

Flow (per ``run_pca_verify``):
  1. Layer A over the segmented sentences → a check-worthy queue (everything else
     goes to the characterization stream, which the publisher can surface later).
  2. Chunk the check-worthy claims into rate-limit-aware batches and adjudicate each
     chunk through the PCA panel (open-book + CRM-114 when the caller wired a
     provider). Chunking is the concession to the proxy's 429 ceiling — one small
     batch per ``adjudicate_fn`` call, sequential, leaning on the bounded backoff
     already in ``ProxyCompletion``.
  3. Bridge the accumulated rows + evidence packs → ``VerdictBundle`` list via
     ``verdict.bridge`` (offline), preserving check-worthy order.

Speaker-blind (I3) all the way through Layer A/B; the speaker is attached only at
bridge/publish time (cosmetic, post-verdict).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from truthbot.checkworthy.pipeline import LayerAResult, run_layer_a
from truthbot.models import Evidence, VerdictBundle
from truthbot.verdict import bridge as bridge_mod
from truthbot.verdict.evidence_pack import EvidencePack
from truthbot.verdict.speech_context import register_speech_date

# layer_a_fn: (sentences[{"sid","text","context"}]) -> A2 rows (classifier.classify's
#             first return value). None → no A2 lane (A1 lexical routing only).
LayerAFn = Callable[[list[dict]], list[dict]]

# adjudicate_fn: (claims chunk [{"sid","text","context"}]) -> (rows, notes).
#                Wraps adjudicator.adjudicate (dropping the manifest, or folding its
#                cost into notes["cost_usd"] — the caller decides).
AdjudicateFn = Callable[[list[dict]], tuple[list[dict], dict]]


@dataclass
class PcaVerifyResult:
    """Everything the publisher (and telemetry) needs from a v2 verify pass."""

    bundles: list[VerdictBundle] = field(default_factory=list)
    evidence: dict[str, list[Evidence]] = field(default_factory=dict)
    characterization: list[dict] = field(default_factory=list)  # non-check-worthy stream
    # Raw adjudication rows + the claim dicts fed to the bridge. Retained so the
    # orchestrator can persist a replay artifact — re-bridging {rows, claims} in
    # offline reproduces the bundles with no LLM spend (see _run_publish_pca).
    rows: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    n_sentences: int = 0
    n_check_worthy: int = 0
    n_chunks: int = 0
    cost_usd: float = 0.0
    # PCA panel composition for THIS run: {"name": <roster>, "seats": {seat: [alias]}}.
    # Per-RUN fact (the whole run uses one roster), captured once from the first
    # non-empty adjudicate notes. Empty → legacy-clean (no composition rendered).
    roster: Optional[dict] = None


def _chunk(items: list, size: int) -> list[list]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


class BudgetHalt(RuntimeError):
    """Raised BEFORE a chunk when the preflight probe says headroom is below
    the projected chunk cost (P67.3 option 3). Completed rows ride on
    ``partial_result`` like any other mid-run failure — but nothing was lost:
    the halt fires before spend, not after a 429."""


# ── P67.3 chunk journal (option 1) ───────────────────────────────────────────
#
# One JSONL line per completed chunk: {"chunk", "rows", "evidence", "cost_usd",
# "roster"?}. Evidence is serialized like the run artifact (Evidence dicts) so
# a resumed run rebuilds identical packs and the offline re-bridge path works
# on a journal alone.

def append_chunk_journal(path, chunk_idx: int, rows: list[dict],
                         packs: dict, cost_usd: float,
                         roster: Optional[dict] = None) -> None:
    import json
    from pathlib import Path

    rec = {
        "chunk": chunk_idx,
        "rows": rows,
        "evidence": {sid: [ev.model_dump(mode="json") for ev in
                           bridge_mod._pack_to_evidence(sid, pack)]
                     for sid, pack in (packs or {}).items()},
        "cost_usd": cost_usd,
    }
    if roster:
        rec["roster"] = roster
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def packs_from_evidence_dict(evidence_by_sid: dict,
                             gate_codes: Optional[dict] = None) -> dict:
    """sid → serialized-Evidence list → sid → EvidencePack. Shared decode for the
    chunk journal and the P120 Phase R packs journal. ``gate_codes`` (optional)
    restores each pack's T2.4 gate flag so a RESUMED gate-failed pack still forces
    Unverifiable instead of silently reopening as a thin pack."""
    from truthbot.models import Evidence
    from truthbot.verdict.evidence_pack import EvidencePack, PackItem, _sha256

    gate_codes = gate_codes or {}
    packs: dict[str, EvidencePack] = {}
    for sid, evs in (evidence_by_sid or {}).items():
        items = []
        for i, d in enumerate(evs, start=1):
            ev = Evidence.model_validate(d)
            items.append(PackItem(
                pack_id=f"E{i}", source_name=ev.source_name,
                source_url=ev.source_url, tier=ev.source_tier,
                snippet=ev.snippet,
                retrieved_at=ev.retrieved_at.isoformat(),
                sha256=_sha256(ev.source_url, ev.snippet),
                supports_claim=ev.supports_claim,
                relevance_score=ev.relevance_score,
                published_at=(ev.published_at.date().isoformat()
                              if ev.published_at else None)))
        packs[sid] = EvidencePack(sid=sid, window=None, items=items,
                                  gate_code=gate_codes.get(sid, ""))
    return packs


def load_chunk_journal(path) -> tuple[list[dict], dict, float, Optional[dict]]:
    """(rows, packs, cost_usd, roster) accumulated from a prior run's journal.
    Missing file → empty (fresh run)."""
    import json
    from pathlib import Path

    p = Path(path)
    rows: list[dict] = []
    packs: dict = {}
    cost = 0.0
    roster: Optional[dict] = None
    if not p.exists():
        return rows, packs, cost, roster
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.extend(rec.get("rows") or [])
        cost += float(rec.get("cost_usd") or 0.0)
        roster = roster or rec.get("roster")
        packs.update(packs_from_evidence_dict(rec.get("evidence") or {}))
    return rows, packs, cost, roster


# ── P120 B1 phase-split: Phase R packs journal ───────────────────────────────
#
# One JSONL line per built pack: {"sid", "gate_code", "evidence": [Evidence…]}.
# Evidence is serialized exactly like the chunk journal so ``packs_from_evidence_dict``
# reloads it. gate_code is persisted (unlike the chunk journal) because a Phase R
# pack that gate-failed must still force Unverifiable when a resumed run reloads it
# BEFORE the panel has run.

def append_packs_journal(path, sid: str, pack) -> None:
    import json
    from pathlib import Path

    rec = {"sid": sid,
           "gate_code": getattr(pack, "gate_code", "") or "",
           "evidence": [ev.model_dump(mode="json")
                        for ev in bridge_mod._pack_to_evidence(sid, pack)]}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_packs_journal(path) -> dict:
    """sid → EvidencePack accumulated from a Phase R packs journal (resume).
    Missing file → empty (fresh phase)."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}
    evidence_by_sid: dict = {}
    gate_codes: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sid = rec.get("sid")
        if not sid:
            continue
        evidence_by_sid[sid] = rec.get("evidence") or []
        gate_codes[sid] = rec.get("gate_code") or ""
    return packs_from_evidence_dict(evidence_by_sid, gate_codes)


def verdict_bucket_tally(rows: list[dict]) -> dict[str, int]:
    """Canonical verdict-bucket tally for journal / run-artifact rows (PR-A2.0).

    A PCA disagreement row carries ``verdict=None`` with ``split=True`` — it is
    kept, not dropped (see ``verdict.bridge``), so any tally keyed naively on
    ``row["verdict"]`` under-counts by every split claim. That is exactly how
    the Obama-2014 measurement read 95 named buckets against 96 check-worthy
    claims (T0.1). Every journal consumer must tally through this helper so
    ``sum(tally.values()) == n_check_worthy`` holds by construction:

      verdict present            → that verdict's bucket
      verdict None + split       → "Models split"
      verdict None, not split    → "No verdict" (malformed row — visible, not lost)
    """
    tally: dict[str, int] = {}
    for row in rows:
        verdict = row.get("verdict")
        if verdict is None:
            label = "Models split" if row.get("split") else "No verdict"
        else:
            label = str(verdict)
        tally[label] = tally.get(label, 0) + 1
    return tally


def claims_from_queue(queue: list[dict]) -> list[dict]:
    """Check-worthy queue rows → adjudicate/bridge claim dicts. Carries sid/text/
    context plus each claim's Layer A routing provenance (label + which stage passed
    it) so the bridge can record it — the queue row is the only place that survives.
    Shared by ``run_pca_verify`` and the P120 split Phase R driver so both build
    identical claim identity from the same queue."""
    return [{"sid": r["sid"], "text": r.get("text", ""),
             "context": r.get("context", ""),
             "layer_a": {"label": r.get("label", ""), "source": r.get("source", ""),
                         "claim_type": r.get("claim_type") or ""}}
            for r in queue]


def run_pca_verify(
    sentences: list[dict],
    *,
    layer_a_fn: Optional[LayerAFn] = None,
    adjudicate_fn: AdjudicateFn,
    chunk_size: int = 6,
    confirm_pass: bool = True,
    on_progress: Optional[Callable[[int, int, list[dict]], None]] = None,
    resume_rows: Optional[list[dict]] = None,
    resume_packs: Optional[dict] = None,
    journal_path=None,
    budget_check: Optional[Callable[[], float]] = None,
    budget_safety: float = 1.5,
    prebuilt_layer_a: Optional[LayerAResult] = None,
) -> PcaVerifyResult:
    """Segmented sentences → published-ready ``VerdictBundle``s via the PCA stack.

    Args:
      sentences:    Layer A input, ``[{"sid","text","context"}]`` (see ``ingest.segment``).
      layer_a_fn:   A2 classify lane (live). None runs A1-only routing (A1-PASS →
                    queue, ambiguous parked) — useful for offline tests.
      adjudicate_fn: per-chunk PCA lane (live), returns ``(rows, notes)`` where
                    ``notes["packs"]`` (open-book) maps sid → EvidencePack and an
                    optional ``notes["cost_usd"]`` is summed into the result.
      chunk_size:   check-worthy claims per adjudicate call (rate-limit control).
      confirm_pass: pass through to Layer A (A2 confirms the A1-PASS band).
      on_progress:  optional callback(chunk_idx, n_chunks, chunk_rows) for CLI logging.
      resume_rows:  P67.3 resume — rows from a prior run's journal; their sids
                    are NEVER re-adjudicated (their spend is already banked).
      resume_packs: sid → EvidencePack matching ``resume_rows`` (journal-loaded).
      journal_path: P67.3 option 1 — when set, every completed chunk's rows +
                    packs + cost are appended to this JSONL immediately, so a
                    mid-run failure loses at most the in-flight chunk.
      budget_check: P67.3 option 3 — callable returning remaining headroom in
                    USD. Probed before every chunk; when headroom < projected
                    chunk cost × ``budget_safety`` (rolling mean of completed
                    chunks), the run halts EARLY with ``BudgetHalt`` — before
                    spend, with everything journaled.

    Returns a ``PcaVerifyResult``. Pure/offline given offline ``*_fn``s.
    On ANY mid-run exception the completed rows/claims ride on the exception
    as ``exc.partial_result`` (a PcaVerifyResult without bundles), so callers
    and tooling can always recover banked spend.
    """
    layer_a = (prebuilt_layer_a if prebuilt_layer_a is not None
               else run_layer_a(sentences, classify_fn=layer_a_fn,
                                confirm_pass=confirm_pass))
    queue = layer_a.check_worthy_queue

    result = PcaVerifyResult(
        characterization=layer_a.characterization_stream,
        n_sentences=len(sentences),
        n_check_worthy=len(queue),
    )
    if not queue:
        return result

    # Claims for adjudicate + the bridge's Claim reconstruction (sid/text/context
    # + Layer A routing provenance) — see claims_from_queue.
    claims = claims_from_queue(queue)
    # P67.3 resume: sids with journaled rows never hit the lane again.
    all_rows: list[dict] = list(resume_rows or [])
    packs: dict[str, EvidencePack] = dict(resume_packs or {})
    done_sids = {r.get("sid") for r in all_rows}
    todo = [c for c in claims if c["sid"] not in done_sids]

    chunks = _chunk(todo, chunk_size)
    result.n_chunks = len(chunks)

    chunk_costs: list[float] = []
    try:
        for idx, chunk in enumerate(chunks, 1):
            if budget_check is not None:
                projected = (sum(chunk_costs) / len(chunk_costs)
                             if chunk_costs else 0.0) * budget_safety
                headroom = budget_check()
                if chunk_costs and headroom < projected:
                    raise BudgetHalt(
                        f"budget preflight: headroom ${headroom:.2f} < "
                        f"projected chunk cost ${projected:.2f} "
                        f"(halting before chunk {idx}/{len(chunks)}; "
                        f"completed work journaled)")
            rows, notes = adjudicate_fn(chunk)
            all_rows.extend(rows)
            chunk_packs = dict(notes.get("packs") or {})
            packs.update(chunk_packs)
            chunk_cost = float(notes.get("cost_usd", 0.0) or 0.0)
            chunk_costs.append(chunk_cost)
            result.cost_usd += chunk_cost
            # Capture the PCA roster composition once — it's identical across
            # chunks, so take the first non-empty one and never overwrite it.
            if result.roster is None:
                roster_note = notes.get("roster")
                if roster_note:
                    result.roster = roster_note
            if journal_path is not None:
                append_chunk_journal(journal_path, idx, rows, chunk_packs,
                                     chunk_cost,
                                     roster=result.roster if idx == 1 else None)
            if on_progress is not None:
                on_progress(idx, len(chunks), rows)
    except Exception as exc:
        # The partial-result channel (P67.3): whatever completed — including
        # resumed rows — survives on the exception. With a journal_path it is
        # also already on disk.
        partial = PcaVerifyResult(
            characterization=result.characterization,
            rows=all_rows, claims=claims,
            n_sentences=result.n_sentences,
            n_check_worthy=result.n_check_worthy,
            n_chunks=result.n_chunks, cost_usd=result.cost_usd,
            roster=result.roster)
        exc.partial_result = partial
        raise

    out = bridge_mod.bridge(all_rows, claims, packs)
    result.bundles = out.bundles
    result.evidence = out.evidence
    result.rows = all_rows
    result.claims = claims
    return result


def build_pca_lane_fns(
    hm_classify,
    hm_verdict,
    provider,
    *,
    pack_builder=None,
    crm114: bool = True,
    roster: str = "dev",
    a2_tier: str = "cheap",
    disc_tier: str = "standard",
    layer_a_batch: int = 25,
    layer_a_pause_s: float = 1.0,
    sleep_fn=None,
) -> tuple[LayerAFn, AdjudicateFn]:
    """Bind the live HydraMind lanes into the ``(layer_a_fn, adjudicate_fn)`` pair
    ``run_pca_verify`` expects.

    Takes TWO engines because the lanes parse responses differently (see
    ``proxy_lane.build_hydramind``): ``hm_classify`` (identity parser) drives Layer A's
    ``parse_a2``; ``hm_verdict`` (``parse_verdict`` parser) drives the PCA panel + CRM-114.

    ``layer_a_fn`` runs the A2 classifier in paced batches; ``adjudicate_fn`` runs the
    PCA panel (open-book + CRM-114 stage-2 when ``provider`` is set — the discriminator
    is evidence-only, so it's forced off closed-book) and folds the run manifest's
    cost into ``notes["cost_usd"]`` so the orchestrator can total spend. Imports
    are local so offline importers of this module don't pull the classifier.

    Layer A over a full speech is hundreds of A2 calls; ``classifier.classify`` dispatches
    a batch as one unpaced burst. We split it into ``layer_a_batch``-sized calls with a
    ``layer_a_pause_s`` gap between them to bound the burst on a shared proxy."""
    import time

    from truthbot.checkworthy import classifier
    from truthbot.verdict import adjudicator

    # shared_pack_v2 (P67.9): a pack_builder supersedes the v1 provider; the
    # CRM-114 stage judges on evidence either way, so it stays on for both.
    two_stage = bool(crm114) and (provider is not None or pack_builder is not None)
    _sleep = sleep_fn or time.sleep

    def layer_a_fn(sentences: list[dict]) -> list[dict]:
        batch = max(1, int(layer_a_batch))
        rows: list[dict] = []
        n = len(sentences)
        for i in range(0, n, batch):
            batch_rows, _manifest = classifier.classify(
                hm_classify, sentences[i:i + batch], tier=a2_tier, on_parse_error="default")
            rows.extend(batch_rows)
            if layer_a_pause_s and i + batch < n:
                _sleep(layer_a_pause_s)
        return rows

    def adjudicate_fn(chunk: list[dict]) -> tuple[list[dict], dict]:
        rows, manifest, notes = adjudicator.adjudicate(
            hm_verdict, chunk, roster=roster, evidence_provider=provider,
            pack_builder=pack_builder,
            two_stage=two_stage, disc_tier=disc_tier)
        notes = dict(notes or {})
        try:
            notes["cost_usd"] = float(getattr(manifest, "total_cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            notes["cost_usd"] = 0.0
        # Record the PCA panel composition (roster name + seat→alias map) so the
        # publisher can surface WHICH models judged the run. Guarded: a bad roster
        # name must never break a live run (fall back to name-only, empty seats).
        try:
            from hydramind.rosters import get_roster
            notes["roster"] = {"name": roster, "seats": dict(get_roster(roster).seats)}
        except Exception:
            notes["roster"] = {"name": roster, "seats": {}}
        return rows, notes

    return layer_a_fn, adjudicate_fn


def prepare_speech(text: str, speech_id: str, utterance: date) -> list[dict]:
    """Segment a transcript and register its utterance date for temporal grounding.

    Convenience for the CLI: registers ``speech_id → utterance`` (so the temporal
    preamble + evidence window resolve even for a non-fixture speech) and returns
    the Layer A sentence inventory. Import kept local so ``ingest`` isn't pulled in
    when only ``run_pca_verify`` is used (offline tests)."""
    from truthbot.ingest.segment import segment

    register_speech_date(speech_id, utterance)
    return segment(text, speech_id)
