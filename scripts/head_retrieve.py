"""D17-d: retrieve evidence for NAMED claims off the PUBLISHING HEAD — journal only.

    scripts/head_retrieve.py --speech SPEECH --sids SID [SID ...] --tag TAG
        [--estimate] [--retrieve-only]
        [--go --budget USD --reason TEXT] [--chunk-size N]

Default (no --go/--estimate): print the plan ($0) and exit.

WHY THIS SCRIPT EXISTS
----------------------
The two halves of "retrieve for named sids off the head" lived in different
scripts and neither could do it alone:

  * ``wave_adjudicate.py`` sources the publishing head and carries the audited
    escape, but re-gates STORED packs — it never retrieves;
  * ``phase3_rebuild.py`` has the real R1/R2/R3 retrieval and the per-claim
    budget breaker, but sources the PRE-WAVE phase-3 artifact, whose verdicts
    were superseded by the wave and the rulings pass.

Both halves are IMPORTED here rather than reimplemented. What is new is the
wiring and an honest cost split.

WHAT THIS SCRIPT CANNOT DO, BY CONSTRUCTION
-------------------------------------------
It cannot write a ``pca_runs`` artifact. Not "does not by default" — the
artifact writers are not imported, so no code path reaches them. That absence
is a safety property, because writing one would silently MOVE THE PUBLISHING
HEAD: ``reshape_rerun_0031.shipping_artifact`` selects the unique leaf whose
lineage passes through the rulings pass, so an artifact with
``rebuild_of=<current head>`` makes today's head a child and becomes the head
itself. ``update_manifest``'s ``published=false`` does not protect —
``shipping_artifact`` never reads the manifest. A three-claim pricing probe
must not be able to promote itself into the thing the site renders.

So the deliverable is journals plus a report: the old->new verdict diff is
COMPUTED and PRINTED, never applied. Promoting it is a separate, deliberate,
owner-gated decision.

THE COST STORY HAS THREE LANES, NOT TWO
---------------------------------------
  * on-proxy, MEASURED: the panel, plus the Haiku relevance/stance scorer.
    ``proxy_key_spend()`` is the LiteLLM ledger and is the truth for this lane.
  * off-proxy, ESTIMATED: R2 (OpenAI browsing) and R3 (grok-4.3), priced from
    provider-reported token counts at list rates. Reconciled against nothing.
  * R1 (``ClaudeWorkerRetriever``): shells out to the ``claude`` CLI with
    ``ANTHROPIC_API_KEY`` popped, so it runs on the Max subscription. It costs
    no dollars and has no meter; ``--budget`` does not bound it, rate limits do.

These are reported SEPARATELY and never fused into one headline number. The
budget breaker enforces the pessimistic sum (proxy + off-proxy estimate +
banked), so an estimate can only ever make the cap fire EARLIER.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import phase3_rebuild as p3            # noqa: E402  (needs the path inserts)
import wave_adjudicate as wa           # noqa: E402
from regate_from_rescore import claim_shape_map  # noqa: E402

# NOTE: ``reshape_rerun_0031`` is deliberately NOT imported at module level. It
# imports wave_adjudicate at import time, and wave_adjudicate breaks that cycle
# with a lazy import; a third module-level edge would reintroduce it. The head
# is reached ONLY through wa.source_artifact.

#: Hardcoded filename infix, NOT part of --tag. No --tag value can therefore
#: produce a wave or phase3 journal name, so this runner's journals cannot
#: collide with (or be resumed from) a banked journal of either.
RUNNER_TAG_INFIX = "headret"

#: Tags that name another lane's artifacts. Refused outright.
RESERVED_TAGS = frozenset({"wave", "p3rebuild", "p3rerun", "packs", "r3",
                           "d17c-wave2", "pilot", "s5rescue"})

#: Pinned in every journal's header record so a resume can prove it is
#: continuing the same run against the same head.
JOURNAL_SCHEMA = "truthbot-head-retrieve v1"

#: The D17-d web-tier1 backlog this probe exists to price.
WEB_TIER1_LANE_N = 81


# ── $0: refusals, paths, sourcing ────────────────────────────────────────────

def tag_refusal(tag: str) -> Optional[str]:
    """--tag names this run's journals and report. None = clear to run."""
    if not tag:
        return ("REFUSING: --tag is REQUIRED and has no default. It names the "
                "journals and the report; a default would let two unrelated "
                "runs append to one journal. No spend attempted.")
    if tag in RESERVED_TAGS:
        return (f"REFUSING --tag {tag!r}: that tag names another lane's "
                "artifacts. No spend attempted.")
    if not all(ch.isalnum() or ch in "-_" for ch in tag):
        return (f"REFUSING --tag {tag!r}: use only letters, digits, dash and "
                "underscore — it becomes a filename. No spend attempted.")
    return None


def reason_refusal(reason: str, go: bool) -> Optional[str]:
    """A funded run buys retrieval for claims nobody costed, so it is audited
    like wave_adjudicate's escape: say why, in the artifacts, before spending."""
    if go and not (reason or "").strip():
        return ("REFUSING to spend: --reason is REQUIRED with --go. This run "
                "buys retrieval for hand-named claims; without a recorded "
                "reason the spend is unreconstructible afterwards. No spend "
                "attempted.")
    return None


#: Why scoring is ON by default (and why --allow-unscored is loud): the most
#: expensive silent failure available here. ``build_evidence_pack_v2`` wires the
#: R1/R2/R3 shortlists straight into ``consolidate`` and NEVER scores relevance
#: or stance — see the "Scoring-coverage telemetry" comment in
#: ``verdict/consolidator.py``. Every unscored item keeps the pydantic default
#: relevance and a null stance; ``_bearing`` requires True/False, so nulls can
#: never credit ``MIN_BEARING_T13`` and the pack GATE-FORCES Unverifiable. An
#: unscored run pays the full retrieval bill and buys back the exact verdicts
#: the claims already carry. The wiring lives in run_head_retrieve.


def journal_paths(speech: str, tag: str) -> tuple[Path, Path]:
    base = wa.JOURNAL_DIR
    stem = f"{speech}_{RUNNER_TAG_INFIX}_{tag}"
    return base / f"{stem}.jsonl", base / f"{stem}_packs.jsonl"


def report_path(tag: str) -> Path:
    return wa.OUT_DIR / f"{RUNNER_TAG_INFIX}_{tag}_report.json"


def head_source(speech: str) -> tuple[Path, dict]:
    """The publishing head, and nothing else.

    ``head=True`` is a LITERAL, and there is no --source/--artifact/--phase3
    flag anywhere in this script's CLI. The pre-wave phase-3 artifact is not
    discouraged here; it is unreachable."""
    return wa.source_artifact(speech, head=True)


def head_refusal(speech: str, art: dict) -> Optional[str]:
    """Belt to ``shipping_artifact``'s braces: assert the negative.

    ``shipping_artifact`` already refuses ambiguous or non-rulings lineage.
    This survives someone swapping the accessor out."""
    run_id = str(art.get("run_id") or "")
    pinned = (p3.SPEECHES.get(speech) or {}).get("run_id", "")
    if pinned and run_id == pinned:
        return (f"REFUSING: resolved artifact {run_id[:8]} IS the pinned "
                f"pre-wave phase-3 run for {speech}. Retrieving against it "
                "would discard every ruling that landed since. No spend "
                "attempted.")
    if not (art.get("meta") or {}).get("rulings"):
        # The head itself carries the rulings block; a descendant inherits the
        # lineage but need not restate it, so this is a warn-shaped refusal
        # only when the block is absent AND no rebuild_of chain exists.
        if not (art.get("meta") or {}).get("rebuild_of"):
            return (f"REFUSING: {run_id[:8]} carries no rulings block and no "
                    "lineage. It is not the publishing head. No spend "
                    "attempted.")
    return None


def shapes_for(art: dict, speech: str) -> tuple[dict, int]:
    """sid -> claim_shape, resolved the way the rebuild's registry saw it.

    NOT from ``art["claims"][*].layer_a.claim_shape``: the head carries None
    there for every claim (verified on trump_2026 head 799e71b6). Registering
    the head's claims directly would register ZERO shapes, and with
    --legacy-quota-ok would silently run the LEGACY evidential-role quota —
    a wrong answer that looks like a right one."""
    return claim_shape_map(art, speech)


def journal_header(art: dict, speech: str, tag: str, sids: list[str]) -> dict:
    """First line of a fresh chunk journal — pins what this run is continuing.

    Safe alongside the shared loader: ``load_chunk_journal`` reads
    ``rec.get("rows") or []`` and ``float(rec.get("cost_usd") or 0.0)``, so a
    header record contributes no rows and no cost."""
    return {"schema": JOURNAL_SCHEMA, "tag": tag, "speech": speech,
            "source_run_id": str(art.get("run_id") or ""),
            "sids": sorted(sids),
            "generated": datetime.now(timezone.utc).isoformat()}


def read_journal_header(path: Path) -> Optional[dict]:
    if not Path(path).exists():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
            except ValueError:
                return None
            return rec if rec.get("schema") == JOURNAL_SCHEMA else None
    return None


def header_refusal(path: Path, art: dict, speech: str) -> Optional[str]:
    """Refuse to resume a journal that belongs to a different head or script.

    Rows retrieved against two different heads must never merge into one run:
    the head moves when rulings land, and a verdict decided against the old one
    is a superseded verdict wearing a current run's tag."""
    p = Path(path)
    if not p.exists() or not p.read_text(encoding="utf-8").strip():
        return None
    header = read_journal_header(p)
    if header is None:
        return (f"REFUSING to resume {p.name}: it has no "
                f"{JOURNAL_SCHEMA!r} header, so it was written by another "
                "script or by hand. No spend attempted.")
    now = str(art.get("run_id") or "")
    was = str(header.get("source_run_id") or "")
    if was != now:
        return (f"REFUSING to resume {p.name}: it was written against head "
                f"{was[:8]}, but the publishing head is now {now[:8]}. The "
                "head moved; those rows are against a superseded artifact. "
                "Start a new --tag. No spend attempted.")
    if str(header.get("speech") or "") != speech:
        return (f"REFUSING to resume {p.name}: it is a "
                f"{header.get('speech')!r} journal, not {speech!r}. No spend "
                "attempted.")
    return None


def requested_complete(requested: list[str],
                       rows: list[dict]) -> tuple[bool, list[str]]:
    """Completeness over the REQUESTED sids — not over the whole speech.

    ``phase3_rebuild``'s guard is ``full_sids <= have_sids`` across every claim
    in the artifact. For a named subset that can never be satisfied, so copying
    it here would mean the run is permanently 'incomplete'. The correct scope
    is the claim set the operator actually asked and paid for (the same shape
    wave_adjudicate uses for its wave set)."""
    have = {r.get("sid") for r in rows}
    missing = [s for s in requested if s not in have]
    return (not missing), missing


# ── $0: the cost model ───────────────────────────────────────────────────────

def spend_split(proxy_usd: float, offproxy_usd: float, usage: dict,
                r1_calls: int, banked_usd: float) -> dict:
    """Three lanes, labelled, never fused into a single headline.

    A subscription call reported as ``$0.00`` is a claim the lane is free, so
    R1 is ``None`` (UNPRICED) rather than zero."""
    billed = proxy_usd + offproxy_usd + banked_usd
    return {
        "proxy": {
            "usd": round(proxy_usd, 6),
            "basis": "litellm proxy ledger (proxy_key_spend, double-read, "
                     "rounded up)",
            "confidence": "ledger-true"},
        "offproxy": {
            "usd": round(offproxy_usd, 6),
            "basis": "provider-reported token counts at MODEL_RATES list "
                     "rates (R2 gpt-5-mini, R3 grok-4.3)",
            "confidence": "ESTIMATE — token-metered, reconciled against "
                          "nothing",
            "calls": {"R2": len(usage.get("R2") or []),
                      "R3": len(usage.get("R3") or [])}},
        "r1_worker": {
            "usd": None,
            "basis": "Claude Max subscription (claude CLI, ANTHROPIC_API_KEY "
                     "popped)",
            "confidence": "UNPRICED — not billed, not estimable from here",
            "calls": r1_calls},
        "banked": {"usd": round(banked_usd, 6),
                   "basis": "prior sessions of this tag's chunk journal"},
        "billed_total_usd": round(billed, 6),
        "estimated_share": round(offproxy_usd / billed, 4) if billed else 0.0,
        "ceiling_basis": "billed_total_usd is what the breaker enforces "
                         "against --budget",
    }


def print_spend_split(split: dict, cap: Optional[float]) -> None:
    off, r1 = split["offproxy"], split["r1_worker"]
    print("\nSPEND (three lanes, reported separately):")
    print(f"  on-proxy  ${split['proxy']['usd']:.4f}  ledger-true "
          "(panel + relevance/stance scorer)")
    print(f"  off-proxy ${off['usd']:.4f}  ESTIMATE at list price "
          f"(R2 x{off['calls']['R2']}, R3 x{off['calls']['R3']})")
    print(f"  R1 worker  UNPRICED   x{r1['calls']} calls on the Max "
          "subscription — no meter, not bounded by --budget")
    if split["banked"]["usd"]:
        print(f"  banked    ${split['banked']['usd']:.4f}  prior sessions")
    cap_s = f" of cap ${cap:.2f}" if cap else ""
    # The share is computed over THIS session's lanes only. Banked spend from a
    # prior session is not decomposable from the chunk journal (it records one
    # cost figure), so claiming a share over a mostly-banked total would assert
    # "0% estimated" about a bill that was in fact 87% estimate.
    live = split["proxy"]["usd"] + split["offproxy"]["usd"]
    share = (f"{split['estimated_share']:.0%} of it is NOT ledger-checked"
             if live > 0 else
             "share unknown — this total is prior-session banked spend, whose "
             "lane split the journal does not preserve")
    print(f"  billed total ${split['billed_total_usd']:.4f}{cap_s}  ({share})")


def lane_projection(split: dict, n_probe: int, n_lane: int) -> dict:
    """Project the measured probe onto the backlog lane. A POINT projection.

    n=3 gives no variance, so this deliberately does not emit a band: a band
    implies a spread that was never measured."""
    if n_probe <= 0:
        return {}
    scale = n_lane / n_probe
    proxy = split["proxy"]["usd"] * scale
    off = split["offproxy"]["usd"] * scale
    # Project the CUMULATIVE billed total, not just this session's lanes: on a
    # resumed run the retrieval spend sits in ``banked`` and projecting only
    # the live lanes would price the lane at a fraction of its real cost (the
    # phase-P resume of the 2026-08-14 probe projected $0.00 that way).
    banked = split["banked"]["usd"]
    total = (split["proxy"]["usd"] + split["offproxy"]["usd"] + banked) * scale
    return {"n_probe": n_probe, "n_lane": n_lane,
            "ledger_true_usd": round(proxy, 4),
            "estimated_usd": round(off, 4),
            "banked_usd": round(banked * scale, 4),
            "total_usd": round(total, 4),
            "estimated_share": split["estimated_share"],
            "r1_calls_projected": split["r1_worker"]["calls"] * n_lane // max(n_probe, 1),
            "kind": "point projection from n=%d — NOT a band; n=%d gives no "
                    "variance" % (n_probe, n_probe)}


def print_lane_projection(proj: dict) -> None:
    if not proj:
        return
    print(f"\n{proj['n_lane']}-claim web-tier1 lane, projected from "
          f"n={proj['n_probe']}:")
    print(f"  ledger-true component  ${proj['ledger_true_usd']:.2f}")
    print(f"  ESTIMATED component    ${proj['estimated_usd']:.2f}  "
          f"({proj['estimated_share']:.0%} of the total)")
    print(f"  UNPRICED component     ~{proj['r1_calls_projected']} R1 worker "
          "calls (subscription capacity, no dollar figure)")
    if proj.get("banked_usd"):
        print(f"  banked (prior session) ${proj['banked_usd']:.2f}  "
              "not decomposable into lanes from here")
    print(f"  -> ${proj['total_usd']:.2f} total. {proj['kind']}.")


def estimate_report(speech: str, sids: list[str]) -> str:
    """$0 projection. Says what is measured, what is borrowed, and what is not
    known — and refuses to invent the number this probe exists to measure."""
    lo, hi = p3.PER_CLAIM_EST
    n = len(sids)
    return "\n".join([
        "Head-retrieval cost projection ($0 — constants only, no calls):",
        "",
        "  BORROWED CONSTANT (measured on the wrong payload):",
        f"    ${lo:.3f}-{hi:.3f}/claim — truthbot.costs.PER_CLAIM_USD_PLANNING,",
        "    ledger-derived from the 2026-08-01 FULL-SPEECH phase-3 rebuilds",
        "    (gwbush/clinton at the gpt-5-mini economy config). It was NOT",
        "    measured on the D17-d web-tier1 backlog, so it may not be quoted",
        "    as this lane's rate.",
        "",
        "  THIS LANE'S RATE: UNMEASURED.",
        "    That is what this probe exists to measure. Emitting a band here",
        "    would recreate the guess the probe is meant to replace.",
        "",
        "  CAP-SIZING ONLY (borrowed constant x claims):",
        f"    {speech}: {n} claim(s) -> ${n * lo:.3f} - ${n * hi:.3f}",
        "    Size --budget above this, with headroom for one claim's full",
        "    retry stack (R1 + R2 + an R3 rescue round + panel).",
        "",
        "  NOT INCLUDED: R1 worker calls run on the Max subscription. They",
        "    cost no dollars, carry no meter, and --budget does not bound",
        "    them — rate limits do.",
    ])


# ── $0: plan ─────────────────────────────────────────────────────────────────

def print_plan(speech: str, art: dict, src_path: Path, sids: list[str],
               shapes: dict, n_sidecar: int, chunk_j: Path, packs_j: Path,
               scoring: bool) -> None:
    print(f"\nPLAN ($0) — head retrieval, journal only")
    print(f"  speech       {speech}")
    print(f"  head         {str(art.get('run_id'))[:8]} ({src_path.name})")
    print(f"  claims       {len(sids)} named sid(s)")
    for s in sids:
        print(f"                 {s}  shape={shapes.get(s) or '(none)'}")
    print(f"  shapes       {n_sidecar} filled from the backfill sidecar "
          "(the head carries none)")
    print(f"  scoring      {'ON — relevance/stance lane (+1 Haiku call/claim, on-proxy)' if scoring else 'OFF — packs will gate-force Unverifiable'}")
    print(f"  journals     {chunk_j.name}")
    print(f"               {packs_j.name}")
    print("  artifact     NONE — this script has no artifact writer. The "
          "verdict diff is")
    print("               computed and printed; promoting it is a separate "
          "owner decision.")


# ── the funded path ──────────────────────────────────────────────────────────

def _count_r1(primary) -> dict:
    """Count R1 shortlist calls without touching retrievers.py.

    ``ClaudeWorkerRetriever`` has no ``_post`` seam to subclass (it shells out
    to the CLI), so the instance's bound method is wrapped. ``retry`` reuses the
    SAME instance, so one patch covers both rounds."""
    counter = {"n": 0}
    r1 = primary[0]
    original = r1.shortlist

    def counted(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    r1.shortlist = counted
    return counter


def run_head_retrieve(args) -> int:
    import os

    from hydramind.rosters import get_roster
    from truthbot.verdict import (adjudicator, proxy_lane, publish_pipeline,
                                  shape_registry, speech_context)
    from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
    from truthbot.verdict.retrieval_phase import (build_packs_phase,
                                                  packs_only_builder)
    from truthbot.verify.principals import principal_relation

    speech, tag = args.speech, args.tag

    # ── refusals, cheapest first, all before any key or network touch ──
    for refusal in (tag_refusal(tag),
                    reason_refusal(args.reason, args.go)):
        if refusal:
            print(refusal)
            return 2

    src_path, art = head_source(speech)
    refusal = head_refusal(speech, art)
    if refusal:
        print(refusal)
        return 2

    # Speech date from the HEAD's meta, not from the SPEECHES table: the era
    # gate fails closed and window_for() returns None without a registered
    # date, and the retrieval path must not be the one place that trusts a
    # static table over the artifact in hand.
    meta_date = (art.get("meta") or {}).get("date") or ""
    if meta_date:
        speech_context.register_speech_date(speech,
                                            date.fromisoformat(str(meta_date)[:10]))

    shapes, n_sidecar = shapes_for(art, speech)

    claims_by_sid = {c.get("sid"): c for c in (art.get("claims") or [])}
    all_claims = [{"sid": c["sid"], "text": c.get("text", ""),
                   "context": c.get("context", "") or ""}
                  for c in (art.get("claims") or [])]
    try:
        selected = p3.select_claims(all_claims, args.sids, 0)
    except p3.UnknownSid as exc:
        print(f"--sids: {exc}")
        return 2
    sids = [c["sid"] for c in selected]

    # Register ONLY the selected claims' shapes, then guard on that scope.
    shape_registry.register_claim_shapes(
        [{"sid": s, "layer_a": {"claim_shape": shapes.get(s) or ""}}
         for s in sids])
    n_shaped = sum(1 for s in sids if shapes.get(s))
    refusal = p3.shape_refusal(n_shaped, len(sids), args.legacy_quota_ok)
    if refusal:
        print(refusal)
        return 2

    chunk_journal, packs_journal = journal_paths(speech, tag)
    refusal = header_refusal(chunk_journal, art, speech)
    if refusal:
        print(refusal)
        return 2

    scoring = not args.allow_unscored
    print_plan(speech, art, src_path, sids, shapes, n_sidecar,
               chunk_journal, packs_journal, scoring)

    if args.estimate:
        print()
        print(estimate_report(speech, sids))
        return 0
    if not args.go:
        print("\n($0 plan only — add --estimate for the cost projection, or "
              "--go --budget USD to spend)")
        return 0

    # ── funded from here ──
    if not proxy_lane.key_present():
        print(proxy_lane.BLOCKED_MSG)
        return 2
    refusal = p3.go_refusal(os.environ, args.budget)
    if refusal:
        print(refusal)
        return 2

    scorer = None
    if scoring:
        from truthbot.verify.relevance import build_scorer
        scorer = build_scorer()
        if scorer is None:
            print("REFUSING: no LiteLLM proxy key for the relevance/stance "
                  "lane. Without it every retrieved item keeps a null stance, "
                  "the pack gate-forces Unverifiable, and the run buys back "
                  "the verdicts it started with. No spend attempted.")
            return 2
    else:
        print("! --allow-unscored: packs will carry null stance and "
              "GATE-FORCE Unverifiable. This measures retrieval cost only.")

    done_rows, _done_packs, banked_cost, _ = \
        publish_pipeline.load_chunk_journal(chunk_journal)
    if read_journal_header(chunk_journal) is None:
        chunk_journal.parent.mkdir(parents=True, exist_ok=True)
        with chunk_journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(journal_header(art, speech, tag, sids),
                                ensure_ascii=False) + "\n")
    todo = p3.pending_claims(selected, done_rows)
    if done_rows:
        print(f"\nresume: {len(done_rows)} sid(s) banked "
              f"(${banked_cost:.4f} prior), {len(todo)} to run")

    primary, retry, offproxy_est, usage = p3.metered_offproxy_retrievers()
    r1_counter = _count_r1(primary)

    speaker = (art.get("meta") or {}).get("speaker", "")
    utterance = date.fromisoformat(str(meta_date)[:10]) if meta_date else None

    def relation_of(ev):
        return principal_relation(ev.source_url, speaker, utterance)

    def build_pack(sid: str, text: str, context: str):
        return build_evidence_pack_v2(
            sid, text, primary, retry_retrievers=retry, context=context,
            claim_shape=shape_registry.shape_for(sid), relation_of=relation_of,
            scorer=scorer)

    start_spend = proxy_lane.proxy_key_spend()
    print(f"proxy key spend at start: ${start_spend:.4f} "
          f"(HARD cap ${args.budget:.2f}, incl. off-proxy estimate + banked)")

    pack_builder = p3.make_pack_builder(
        build_pack=build_pack, cap=args.budget, start_spend=start_spend,
        offproxy_est=offproxy_est, banked_cost=banked_cost,
        # NOT packs_journal=: build_packs_phase already journals every pack it
        # builds, and it also owns the resume path. Passing it here too wrote
        # each pack TWICE (harmless to a resume — load_packs_journal builds a
        # dict — but it overstates the retrieval work to anyone reading the
        # journal, which is the artifact a lane gets priced from).
        packs_journal=None)

    halted = ""

    # ── Phase R: retrieve. Every pack is journaled the instant it completes,
    # so a halt here loses at most the in-flight claim. ──
    def _progress(done: int, n: int, sid: str) -> None:
        print(f"  phase R {done}/{n}: {sid} "
              f"(off-proxy est ${offproxy_est():.4f})", flush=True)

    print(f"\nPHASE R — retrieval for {len(todo)} claim(s)")
    try:
        packs = build_packs_phase(
            todo, pack_builder, journal_path=packs_journal,
            resume_packs=publish_pipeline.load_packs_journal(packs_journal),
            on_progress=_progress)
    except p3.BudgetHalt as exc:
        halted = f"BUDGET HALT in phase R: {exc}"
        print(halted)
        packs = publish_pipeline.load_packs_journal(packs_journal)

    proxy_now = wa.settled_delta(proxy_lane, start_spend)
    split = spend_split(proxy_now, offproxy_est(), usage,
                        r1_counter["n"], banked_cost)
    print_spend_split(split, args.budget)

    # Bank phase R's spend into the CHUNK journal so a resume inherits it.
    # Without this the cap does not survive a resume: phase R writes only to
    # the packs journal, so load_chunk_journal returns banked_cost=0 and the
    # next run starts its ceiling from scratch — $0.45 of retrieval followed by
    # a resume would authorise another full --budget. Recorded as a rows-less
    # record so the shared loader sums the cost and ignores everything else.
    phase_r_spend = proxy_now + offproxy_est()
    if phase_r_spend > 0:
        publish_pipeline.append_chunk_journal(
            chunk_journal, 0, [], {}, phase_r_spend)
        banked_cost += phase_r_spend

    if args.retrieve_only:
        print("\n--retrieve-only: stopping after phase R. No panel call, no "
              "verdicts. Packs are journaled and a resume will not re-buy "
              "them.")
        _write_report(args, speech, tag, art, sids, split, [], halted,
                      chunk_journal, packs_journal, shapes, phase="R")
        return 1 if halted else 0

    # ── Phase P: adjudicate from the prebuilt packs. No retrieval. ──
    lookup = packs_only_builder(packs)

    def guarded(sid: str, text: str, context: str):
        spent = ((proxy_lane.proxy_key_spend() - start_spend)
                 + offproxy_est() + banked_cost)
        if spent >= args.budget:
            raise p3.BudgetHalt(f"${spent:.2f} >= cap ${args.budget:.2f} "
                                f"(before the panel call for {sid})")
        return lookup(sid, text, context)

    hm = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    roster_note = {"name": "prod", "seats": dict(get_roster("prod").seats)}
    chunk_size = max(1, min(int(args.chunk_size or 1), p3.CHUNK_SIZE))
    runnable = [c for c in todo if c["sid"] in packs]
    chunks = [runnable[i:i + chunk_size]
              for i in range(0, len(runnable), chunk_size)]
    all_rows = list(done_rows)

    print(f"\nPHASE P — panel over {len(runnable)} prebuilt pack(s), "
          f"chunk size {chunk_size}")
    for idx, chunk in enumerate(chunks, 1):
        if halted:
            break
        running = ((proxy_lane.proxy_key_spend() - start_spend)
                   + offproxy_est() + banked_cost)
        if running >= args.budget:
            halted = (f"BUDGET HALT before chunk {idx}: ${running:.2f} >= cap "
                      f"${args.budget:.2f}")
            print(halted)
            break
        t0, s0 = time.time(), proxy_lane.proxy_key_spend()
        try:
            rows, _manifest, notes = p3._adjudicate_chunk(
                adjudicator, hm, chunk, guarded, idx)
        except p3.BudgetHalt as exc:
            halted = f"BUDGET HALT mid-chunk {idx}: {exc}"
            print(halted)
            break
        except p3.ChunkFailed as exc:
            halted = f"TRANSIENT HALT at chunk {idx}: {exc}"
            print(halted)
            break
        s1, t1 = proxy_lane.proxy_key_spend(), time.time()
        publish_pipeline.append_chunk_journal(
            chunk_journal, idx, rows, notes.get("packs") or {}, s1 - s0,
            roster=roster_note if not done_rows and idx == 1 else None)
        all_rows.extend(rows)
        print(f"  chunk {idx}/{len(chunks)} ({len(chunk)} claim(s)): "
              f"proxy ${s1 - s0:.4f}, {t1 - t0:.0f}s", flush=True)

    proxy_total = wa.settled_delta(proxy_lane, start_spend)
    split = spend_split(proxy_total, offproxy_est(), usage,
                        r1_counter["n"], banked_cost)
    print_spend_split(split, args.budget)

    complete, missing = requested_complete(sids, all_rows)
    banked_rows = [r for r in all_rows if r.get("sid") in set(sids)]
    old_rows = {r.get("sid"): r for r in (art.get("rows") or [])}
    diff = p3.build_verdict_diff(
        [old_rows[s] for s in sids if s in old_rows], banked_rows,
        art.get("claims") or [])
    p3.print_diff(diff, partial=not complete)
    if not complete:
        print(f"\nINCOMPLETE — {len(missing)} of {len(sids)} requested sid(s) "
              f"not banked: {', '.join(missing)}")
        print(f"  chunk journal: {chunk_journal}")
        print(f"  packs journal: {packs_journal}")
        print("  Resume with the same --tag (re-spends only on unbanked sids).")

    proj = lane_projection(split, len(banked_rows), WEB_TIER1_LANE_N)
    print_lane_projection(proj)

    print("\nNO ARTIFACT WRITTEN — by construction, not by default. The diff "
          "above is\ncomputed, not applied; promoting it is a separate owner "
          "decision.")
    _write_report(args, speech, tag, art, sids, split, banked_rows, halted,
                  chunk_journal, packs_journal, shapes, phase="RP",
                  diff=diff, projection=proj, complete=complete,
                  missing=missing)
    return 1 if halted else 0


def _write_report(args, speech, tag, art, sids, split, rows, halted,
                  chunk_journal, packs_journal, shapes, *, phase: str,
                  diff: Optional[dict] = None,
                  projection: Optional[dict] = None,
                  complete: bool = False,
                  missing: Optional[list] = None) -> Path:
    """The deliverable. Records what was measured AND what it does not cover."""
    decided = [r.get("sid") for r in rows
               if str(r.get("verdict") or "").upper() not in ("", "UNVERIFIABLE")]
    report = {
        "schema": "truthbot-head-retrieve-report v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "tag": tag, "speech": speech, "phase": phase,
        "source_run_id": str(art.get("run_id") or ""),
        "source_note": "current publishing head, selected by lineage",
        "reason": (args.reason or "").strip(),
        "sids": list(sids),
        "claim_shapes": {s: shapes.get(s) or "" for s in sids},
        "scored": not args.allow_unscored,
        "cap_usd": args.budget,
        "spend": split,
        "projection": projection or {},
        "complete": complete,
        "missing_sids": list(missing or []),
        "halted": halted,
        "journals": {"chunk": str(chunk_journal), "packs": str(packs_journal)},
        "verdict_diff": diff or {},
        "artifact_written": False,
        "artifact_note": ("This runner has no artifact writer. Writing one "
                          "would make this run the publishing head "
                          "(shipping_artifact selects the unique rulings-"
                          "descended leaf), so promotion is a separate, "
                          "deliberate, owner-gated step."),
        # Recorded so the follow-up is documented rather than discovered later.
        "decidability_now_stale": decided,
        "decidability_note": ("These sids were retrievable-pending-lane on the "
                              "decidability axis and now have a decided "
                              "verdict. This runner does NOT edit "
                              "data/decidability.json — that axis is "
                              "owner-ratified-gated by design."),
        "evidence_note": ("build_evidence_pack_v2 REPLACES the head's stored "
                          "evidence for a retrieved sid; it does not union "
                          "with it. The d17d_webtier1_estimate assumption "
                          "'a retrieval pass adds N new items' does not "
                          "describe this run."),
    }
    out = report_path(tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"run report -> {out}")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", required=True, choices=sorted(p3.SPEECHES),
                    help="which speech's publishing head to source")
    ap.add_argument("--sids", nargs="+", required=True,
                    help="the claims to retrieve for — REQUIRED; there is no "
                         "whole-speech mode")
    ap.add_argument("--tag", default="",
                    help="REQUIRED, no default: names this run's journals and "
                         "report")
    ap.add_argument("--reason", default="",
                    help="why these claims are being bought — REQUIRED with "
                         "--go, recorded in the report")
    ap.add_argument("--go", action="store_true",
                    help="actually spend (else print the plan, $0)")
    ap.add_argument("--budget", type=float, default=None,
                    help="HARD halt cap in USD — REQUIRED with --go")
    ap.add_argument("--estimate", action="store_true",
                    help="$0 cost projection and exit")
    ap.add_argument("--retrieve-only", action="store_true",
                    help="stop after phase R — price retrieval alone, no panel")
    ap.add_argument("--chunk-size", type=int, default=1,
                    help="panel chunk size (default 1: per-claim banking IS "
                         "the measurement for a pricing probe)")
    ap.add_argument("--allow-unscored", action="store_true",
                    help="run WITHOUT the relevance/stance lane. Packs then "
                         "carry null stance and gate-force Unverifiable — "
                         "deliberate only")
    ap.add_argument("--legacy-quota-ok", action="store_true",
                    help="run the legacy evidential-role quota DELIBERATELY")
    ap.add_argument("--write-artifact", action="store_true",
                    help="REFUSES — see the message; artifact promotion is a "
                         "separate owner-gated step")
    args = ap.parse_args(argv)

    if args.write_artifact:
        print(
            "REFUSING --write-artifact: this runner has no artifact writer, "
            "deliberately.\n"
            "Writing a pca_runs artifact with rebuild_of=<current head> makes "
            "today's head\na child, so the new file becomes the sole leaf "
            "descending from the rulings\npass — i.e. THE PUBLISHING HEAD that "
            "the site renders. update_manifest's\npublished=false does not "
            "protect: shipping_artifact never reads the manifest.\n"
            "Promoting retrieved verdicts is an owner decision, not a side "
            "effect of\npricing a lane. The report carries the computed diff; "
            "act on that.")
        return 2

    return run_head_retrieve(args)


if __name__ == "__main__":
    raise SystemExit(main())
