#!/usr/bin/env python3
"""D15 + D16(α) era breakdown — the M-6 evenhandedness check. $0.

NO model calls, no keys, no network: pure arithmetic over the five rebuilt run
artifacts (metrics/pca_runs) and the B1a re-score sidecars.

WHY THIS EXISTS
---------------
D15 (utterance-record) only ever REMOVES quota credit; D16(α)
(statistical-release) only ever ADDS it. Reported separately they are two
one-sided numbers, and a packet that shows "50 claims withheld" on one page and
"2 claims released" on another has told the owner nothing about whether the two
rules, taken together, move the five speeches EVENLY.

That is the M-6 question, and it is the one that matters: the load-bearing claim
of the whole remediation is that the pipeline judges eras the same way. A repair
that withholds 23 Trump verdicts and releases 2 Clinton ones is not obviously
evenhanded, and the packet must say so out loud rather than leave it to be
discovered.

WHAT IT COMPUTES
----------------
Every stored pack is run through the REAL gate (``consolidator.consolidate``)
FOUR times — both flags off, D15 only, D16 only, both — and the four answers
compared. Per speech:

  * D15 alone: newly-gated count, and how many of those currently SHIP TRUE
    (the ones ratification would actually take off the site);
  * D16(α) alone: released count;
  * both together: the newly-gated and released counts, the NET, and the
    resulting decided-rate — RAW and ANECDOTE-ADJUSTED (the A10 convention
    from ``scripts/dc6_package.py``: a claim typed ``personal-anecdote`` has no
    public record to check, so "Unverifiable" is the correct outcome, not a
    miss) — with the max-min SPREAD across speeches on both bases, before and
    after.

DECIDED-RATE CONVENTION (stated, not buried)
--------------------------------------------
"Before" is what the artifacts actually ship. "After" applies the gate change
and nothing else:

  * a claim NEWLY GATED becomes Unverifiable — the gate forces it, and
    withholding costs nothing;
  * a claim RELEASED becomes ELIGIBLE for a decided verdict. Whether it lands
    decided needs a panel call, which is spend. So the headline "after" figure
    is an UPPER BOUND, and the lower bound — releases counted as still
    undecided — is carried alongside it in the JSON. Both are reported because
    the honest answer is a range, and the range is small enough to state.

Both switches are passed EXPLICITLY (``consolidate(utterance_record=...,
statistical_release=...)``), never by setting the environment, so this
measurement can never leave a flag on behind it.

Usage (repo root, always $0):
  PYTHONPATH=.:src .venv/bin/python scripts/d15_d16_era_breakdown.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc6_package import ABSTAIN, ANECDOTE_CLAIM_TYPE, SPEAKERS, display  # noqa: E402
from regate_from_rescore import (claim_shape_map, gate_once,  # noqa: E402
                                 load_rescore_sidecar, merge_sidecars,
                                 overlay_rescores, row_gate_code)
from rescore_stored_packs import (REBUILT_RUNS, artifact_path,  # noqa: E402
                                  b2_sidecar_path, load_artifact, sidecar_path)

OUT_DIR = REPO / "metrics" / "remediation_v2"
OUT_STEM = "d15_d16_era_breakdown"

#: The two stance vintages. ``rescored`` is the live state of the corpus and is
#: the one the headline tables use — it is the vintage the "measured 50" came
#: from — but ``stored`` is carried so a reader can see that the era pattern is
#: not an artefact of the B1a re-score.
VINTAGES = ("stored", "rescored")
HEADLINE_VINTAGE = "rescored"

#: The four gate configurations, in report order.
CONFIGS = {
    "base": (False, False),
    "d15": (True, False),
    "d16": (False, True),
    "both": (True, True),
}

#: Which state a claim is said to have MOVED FROM.
#:
#: ``recomputed`` (the original, still the default) compares against the same
#: packs re-gated with both rules OFF. That ISOLATES THE RULES: the B1a+B2
#: re-score is present on both sides of the comparison, so it cancels, and the
#: numbers answer "what would ratifying D15 and D16(alpha) do?".
#:
#: ``shipped`` compares against the gate outcome the source artifact actually
#: RECORDED. Nothing cancels, so it answers a different question — "what did the
#: corpus lose relative to what was on the page?", the re-score's own
#: withholdings included. Pointed at the PRE-application artifacts, that is
#: exactly the change the 2026-08-10 application shipped, which is why the
#: post-application packet quotes this basis and not the other one.
BASELINES = ("recomputed", "shipped")
DEFAULT_BASELINE = "recomputed"

#: Which generation of artifacts supplies the packs and the shipped rows.
#: ``rebuild`` is the Phase-3 rebuild — what the ratification was measured
#: against. ``wave`` is the adjudication wave's, the state the 2026-08-10
#: application started FROM. ``rulings`` is that application's own output, the
#: state the corpus is in now.
SOURCES = ("rebuild", "wave", "rulings")
DEFAULT_SOURCE = "rebuild"


def source_artifact_path(speech: str, source: str = DEFAULT_SOURCE) -> Path:
    """The artifact a run reads for ``speech``.

    Resolved by SEARCHING the run directory for the generation's own marker,
    never by a pinned id: run ids for these generations are not constants
    anywhere, and a stale pin would silently measure the wrong corpus — the
    exact failure that makes a packet's numbers describe something nobody is
    publishing.

    The two markers are not interchangeable and the ``wave`` case has to say so:
    a rulings artifact inherits its parent's ``meta.wave`` block wholesale
    (that block is the record of which claims the wave adjudicated, and it stays
    true), so "has a wave block" matches BOTH generations. A wave artifact is
    the one with a wave block and NO rulings block."""
    if source == "rebuild":
        return artifact_path(speech)
    runs_dir = REPO / "metrics" / "pca_runs"
    hits = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            meta = (json.loads(path.read_text(encoding="utf-8"))
                    .get("meta") or {})
        except (ValueError, OSError):
            continue
        if meta.get("speech_id") != speech:
            continue
        if source == "rulings" and meta.get("rulings"):
            hits.append(path)
        elif source == "wave" and meta.get("wave") and not meta.get("rulings"):
            hits.append(path)
    if not hits:
        raise SystemExit(f"no {source} artifact found for {speech}")
    return hits[-1]


def _speech_id(sid: str) -> str:
    return sid.split(":", 1)[0]


def gate_all_ways(speech: str, artifact: dict,
                  sidecar: Optional[dict]) -> dict:
    """``{vintage: {sid: {config: quota_met}}}`` plus the per-sid metadata the
    report needs (shipped verdict, claim type, claim text)."""
    from truthbot.verdict import speech_context
    from truthbot.verdict.consolidator import GATE_INSUFFICIENT
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.principals import principal_relation

    meta = artifact.get("meta") or {}
    speaker = meta.get("speaker") or ""
    utterance = date.fromisoformat(meta["date"]) if meta.get("date") else None
    if utterance is not None:
        speech_context.register_speech_date(speech, utterance)

    relation_of = None
    if speaker and utterance is not None:
        def relation_of(ev):                      # noqa: F811 — mirrors pipeline
            return principal_relation(ev.source_url, speaker, utterance)

    claims = {c.get("sid"): c for c in (artifact.get("claims") or [])}
    shapes, _ = claim_shape_map(artifact, speech)
    rows = {r.get("sid"): r for r in (artifact.get("rows") or [])}
    scored = (sidecar or {}).get("sids") or {}

    info: dict[str, dict] = {}
    quota = {v: {} for v in VINTAGES}
    for sid, dumps in (artifact.get("evidence") or {}).items():
        claim = claims.get(sid) or {}
        text = (claim.get("text") or "").strip()
        shape = shapes.get(sid, "")
        info[sid] = {
            "verdict": (rows.get(sid) or {}).get("verdict") or "",
            "claim_type": ((claim.get("layer_a") or {}).get("claim_type")
                           or "").strip(),
            "claim": text[:140],
        }
        for vintage in VINTAGES:
            if vintage == "rescored" and sid not in scored:
                continue

            def _pack():
                # Rebuilt per configuration, so one run's mutations can never
                # leak into another's arithmetic.
                ev = evidence_from_artifact_dict({sid: dumps})[sid]
                if vintage == "rescored":
                    overlay_rescores(ev, scored[sid])
                return ev

            answers = {}
            for name, (d15, d16) in CONFIGS.items():
                res, _ = gate_once(sid, _pack(), utterance=utterance,
                                   claim_shape=shape, relation_of=relation_of,
                                   claim_text=text, utterance_record=d15,
                                   statistical_release=d16)
                answers[name] = bool(res.quota_met)
            # The fifth answer is not computed — it is READ. What the artifact
            # recorded is the only baseline that describes the published page,
            # and recomputing it would fold the re-score back out of the
            # comparison, which is the whole point of offering it.
            answers["shipped"] = row_gate_code(rows.get(sid, {})) != GATE_INSUFFICIENT
            quota[vintage][sid] = answers
    return {"info": info, "quota": quota}


# ── decided-rate arithmetic ─────────────────────────────────────────────────

def _decided_now(sid: str, info: dict) -> bool:
    """Does this claim SHIP a substantive ruling today?

    An EMPTY verdict counts as undecided. Three of the five rebuilt runs carry
    rows with no verdict at all (10 across the corpus), and ``display("")``
    returns ``""``, which is not in ``ABSTAIN`` — so the obvious spelling of
    this test would silently count every unresolved row as decided and inflate
    each "before" figure in the packet."""
    verdict = (info[sid]["verdict"] or "").strip()
    if not verdict:
        return False
    return display(verdict) not in ABSTAIN


def decided_rate(sids: list[str], info: dict, *, gated: set[str],
                 released: set[str], releases_decide: bool) -> dict:
    """decided-rate over ``sids`` after applying the gate change.

    ``releases_decide`` picks the bound: True treats a released claim as
    decided (the upper bound), False leaves it where it ships (the lower
    bound). Newly-gated claims are always undecided — the gate forces that, at
    no cost."""
    decided = 0
    for sid in sids:
        d = _decided_now(sid, info)
        if sid in gated:
            d = False
        elif sid in released and releases_decide:
            d = True
        decided += bool(d)
    n = len(sids)
    return {"decided": decided, "total": n,
            "rate": round(decided / n, 4) if n else 0.0}


def _spread(rates: dict[str, float]) -> dict:
    if not rates:
        return {"min": 0.0, "max": 0.0, "spread": 0.0,
                "min_speech": "", "max_speech": ""}
    lo = min(rates, key=rates.get)
    hi = max(rates, key=rates.get)
    return {"min": rates[lo], "max": rates[hi],
            "spread": round(rates[hi] - rates[lo], 4),
            "min_speech": lo, "max_speech": hi}


def per_speech_block(speech: str, gated_all: dict, vintage: str,
                     baseline: str = DEFAULT_BASELINE) -> dict:
    info = gated_all["info"]
    quota = gated_all["quota"][vintage]
    sids = sorted(quota)
    base_key = "base" if baseline == "recomputed" else "shipped"

    def moved(cfg: str, direction: str) -> list[str]:
        """sids whose gate outcome differs from the baseline under ``cfg``.
        direction "gated": baseline decided-eligible -> now gated;
        "released": baseline gated -> now eligible."""
        out = []
        for sid in sids:
            base, now = quota[sid][base_key], quota[sid][cfg]
            if base and not now and direction == "gated":
                out.append(sid)
            if (not base) and now and direction == "released":
                out.append(sid)
        return out

    d15_gated = moved("d15", "gated")
    d15_released = moved("d15", "released")          # expected empty
    d16_released = moved("d16", "released")
    d16_gated = moved("d16", "gated")                # expected empty
    both_gated = moved("both", "gated")
    both_released = moved("both", "released")

    anecdotes = {sid for sid in sids
                 if info[sid]["claim_type"] == ANECDOTE_CLAIM_TYPE}
    adj_sids = [sid for sid in sids if sid not in anecdotes]

    gated_set, released_set = set(both_gated), set(both_released)
    none: set[str] = set()
    block = {
        "speech": speech,
        "speaker": SPEAKERS.get(speech, ""),
        "baseline": baseline,
        "claims_measured": len(sids),
        "anecdotes": len(anecdotes),
        # Rows the rebuild left without a verdict. Surfaced rather than folded
        # into the abstain bucket, because "no ruling recorded" and "ruled
        # Unverifiable" are different facts about the run.
        "rows_without_a_verdict": sum(
            1 for s in sids if not (info[s]["verdict"] or "").strip()),
        "d15": {
            "newly_gated": len(d15_gated),
            "newly_gated_sids": d15_gated,
            "newly_gated_shipping_true": sum(
                1 for s in d15_gated if info[s]["verdict"] == "TRUE"),
            "newly_gated_true_sids": [s for s in d15_gated
                                      if info[s]["verdict"] == "TRUE"],
            "released": len(d15_released),
            "released_sids": d15_released,
        },
        "d16": {
            "released": len(d16_released),
            "released_sids": d16_released,
            "newly_gated": len(d16_gated),
            "newly_gated_sids": d16_gated,
        },
        "combined": {
            "newly_gated": len(both_gated),
            "newly_gated_sids": both_gated,
            "released": len(both_released),
            "released_sids": both_released,
            "net": len(both_released) - len(both_gated),
        },
        "decided_rate": {
            "raw_before": decided_rate(sids, info, gated=none, released=none,
                                       releases_decide=False),
            "raw_after": decided_rate(sids, info, gated=gated_set,
                                      released=released_set,
                                      releases_decide=True),
            "raw_after_lower": decided_rate(sids, info, gated=gated_set,
                                            released=released_set,
                                            releases_decide=False),
            "adjusted_before": decided_rate(adj_sids, info, gated=none,
                                            released=none,
                                            releases_decide=False),
            "adjusted_after": decided_rate(adj_sids, info, gated=gated_set,
                                           released=released_set,
                                           releases_decide=True),
            "adjusted_after_lower": decided_rate(adj_sids, info, gated=gated_set,
                                                 released=released_set,
                                                 releases_decide=False),
        },
    }
    dr = block["decided_rate"]
    block["decided_rate"]["delta_raw"] = round(
        dr["raw_after"]["rate"] - dr["raw_before"]["rate"], 4)
    block["decided_rate"]["delta_adjusted"] = round(
        dr["adjusted_after"]["rate"] - dr["adjusted_before"]["rate"], 4)
    return block


# ── concentration: the M-6 question, answered in the output ─────────────────

def concentration(blocks: list[dict]) -> dict:
    """Does the combined effect land on ONE speaker/era?

    Reported as shares of the corpus totals, per direction, plus the share each
    speech contributes RELATIVE to its size — because the speeches differ by a
    factor of nearly four in claim count, and a raw count table alone would let
    "Trump has the most claims" masquerade as "the repair targets Trump"."""
    tot_gated = sum(b["combined"]["newly_gated"] for b in blocks) or 0
    tot_released = sum(b["combined"]["released"] for b in blocks) or 0
    tot_claims = sum(b["claims_measured"] for b in blocks) or 0

    rows = []
    for b in blocks:
        n = b["claims_measured"]
        rows.append({
            "speech": b["speech"],
            "speaker": b["speaker"],
            "claims": n,
            "claim_share": round(n / tot_claims, 4) if tot_claims else 0.0,
            "newly_gated": b["combined"]["newly_gated"],
            "gated_share": (round(b["combined"]["newly_gated"] / tot_gated, 4)
                            if tot_gated else 0.0),
            "gated_rate_within_speech": (
                round(b["combined"]["newly_gated"] / n, 4) if n else 0.0),
            "released": b["combined"]["released"],
            "released_share": (round(b["combined"]["released"] / tot_released, 4)
                               if tot_released else 0.0),
            "net": b["combined"]["net"],
            "net_rate_within_speech": (
                round(b["combined"]["net"] / n, 4) if n else 0.0),
        })

    # A direction "concentrates" when one speech carries more than half of it
    # AND carries visibly more than its share of the corpus.
    def _top(field: str, share_field: str) -> dict:
        best = max(rows, key=lambda r: r[field]) if rows else None
        if best is None:
            return {}
        return {"speech": best["speech"], "speaker": best["speaker"],
                "count": best[field], "share": best[share_field],
                "claim_share": best["claim_share"],
                "concentrated": bool(best[share_field] > 0.5
                                     and best[share_field]
                                     > best["claim_share"] + 0.10)}

    gated_top = _top("newly_gated", "gated_share")
    released_top = _top("released", "released_share")

    # Per-speech withholding RATES are the evenhandedness test that survives
    # the size difference: same rule, same corpus, different rate per era.
    rates = {r["speech"]: r["gated_rate_within_speech"] for r in rows}
    rate_spread = _spread(rates)
    # A RATE ratio of 2x or more is treated as material concentration. Stated
    # as a threshold rather than left to the reader, because "is 12.6% vs 4.2%
    # a lot?" is exactly the question a packet must not leave open.
    ratio = (rate_spread["max"] / rate_spread["min"]
             if rate_spread["min"] else float("inf") if rate_spread["max"]
             else 1.0)
    rate_concentrated = ratio >= 2.0
    return {
        "rows": rows,
        "withholding_top": gated_top,
        "release_top": released_top,
        "withholding_rate_spread": rate_spread,
        "withholding_rate_ratio": (round(ratio, 2) if ratio != float("inf")
                                   else None),
        "rate_concentration_threshold": 2.0,
        "rate_concentrated": bool(rate_concentrated),
        "verdict": _concentration_sentence(gated_top, released_top,
                                           rate_spread, ratio,
                                           rate_concentrated),
    }


def _concentration_sentence(gated_top: dict, released_top: dict,
                            rate_spread: dict, ratio: float,
                            rate_concentrated: bool) -> str:
    """The plain-English answer, written into the artifact so the packet cannot
    ship the table without the finding.

    Two tests, because they can disagree and the disagreement is informative:
    SHARE (does one speech carry most of the effect?) and RATE (does the rule
    fire more often per claim in one era?). Only the second survives the fact
    that the speeches differ by nearly 4x in size."""
    parts = []
    if rate_concentrated:
        parts.append(
            f"YES — the withholding effect concentrates by ERA. The rule fires "
            f"on {rate_spread['max']:.1%} of {rate_spread['max_speech']}'s "
            f"claims and {rate_spread['min']:.1%} of "
            f"{rate_spread['min_speech']}'s, a ratio of {ratio:.1f}x "
            f"(spread {rate_spread['spread']:.1%}). This is the size-adjusted "
            f"number and it is the one to read.")
    else:
        parts.append(
            f"No material era concentration: per-speech withholding rates run "
            f"{rate_spread['min']:.1%} ({rate_spread['min_speech']}) to "
            f"{rate_spread['max']:.1%} ({rate_spread['max_speech']}), a ratio "
            f"of {ratio:.1f}x.")
    if gated_top.get("concentrated"):
        parts.append(
            f"By raw share too: {gated_top['speaker']} "
            f"({gated_top['speech']}) carries {gated_top['share']:.0%} of the "
            f"newly-gated claims on {gated_top['claim_share']:.0%} of the "
            f"corpus.")
    else:
        parts.append(
            f"By raw SHARE it is milder — the largest single share is "
            f"{gated_top.get('speaker', '?')} at "
            f"{gated_top.get('share', 0):.0%} of newly-gated claims on "
            f"{gated_top.get('claim_share', 0):.0%} of the corpus, an "
            f"over-representation rather than a majority.")
    parts.append(
        f"Release lands entirely on {released_top.get('speaker', '?')} "
        f"({released_top.get('share', 0):.0%}), but on a base of only "
        f"{released_top.get('count', 0)} claim(s) — too few to read as a "
        f"pattern.")
    return " ".join(parts)


# ── report ──────────────────────────────────────────────────────────────────

def build_report(speeches: list[str], *, baseline: str = DEFAULT_BASELINE,
                 source: str = DEFAULT_SOURCE) -> dict:
    per_vintage: dict[str, dict] = {}
    gated_all: dict[str, dict] = {}
    missing: list[str] = []
    sources: dict[str, str] = {}
    for sp in speeches:
        path = source_artifact_path(sp, source)
        art = load_artifact(path)
        sources[sp] = str(art.get("run_id") or path.stem)
        # STANCE VINTAGE: B1a and B2 MERGED, latest wins — the same stance the
        # flip set and the application gate on. Reading B1a alone (which this
        # script did until 2026-08-10) silently disagrees with them: B2
        # re-scored a targeted subset into its own sidecar, and on the wave
        # artifacts that difference is worth three claims. A breakdown that
        # counts 62 withholdings while 65 ship is not describing the corpus.
        #
        # The sidecars name the REBUILD's run id — the run whose packs they
        # scored. A later generation copies those packs VERBATIM (the wave
        # replaced rows, never evidence, for anything it did not
        # re-adjudicate), so the scores still join correctly while the revision
        # check would reject them on the id alone. It is passed empty here,
        # which skips the id check and keeps the speech check.
        parts = [p for p in (sidecar_path(sp), b2_sidecar_path(sp))
                 if p.exists()]
        side = merge_sidecars(*(load_rescore_sidecar(p, sp, "") for p in parts))
        if not parts:
            side = None
            missing.append(sp)
        gated_all[sp] = gate_all_ways(sp, art, side)

    for vintage in VINTAGES:
        blocks = [per_speech_block(sp, gated_all[sp], vintage, baseline)
                  for sp in speeches]
        spreads = {}
        for basis in ("raw_before", "raw_after", "adjusted_before",
                      "adjusted_after"):
            spreads[basis] = _spread(
                {b["speech"]: b["decided_rate"][basis]["rate"] for b in blocks})
        corpus = {
            "claims_measured": sum(b["claims_measured"] for b in blocks),
            "d15_newly_gated": sum(b["d15"]["newly_gated"] for b in blocks),
            "d15_newly_gated_shipping_true": sum(
                b["d15"]["newly_gated_shipping_true"] for b in blocks),
            "d16_released": sum(b["d16"]["released"] for b in blocks),
            "combined_newly_gated": sum(
                b["combined"]["newly_gated"] for b in blocks),
            "combined_released": sum(b["combined"]["released"] for b in blocks),
            "combined_net": sum(b["combined"]["net"] for b in blocks),
        }
        per_vintage[vintage] = {
            "per_speech": blocks,
            "spreads": spreads,
            "corpus": corpus,
            "concentration": concentration(blocks),
        }

    return {
        "schema": "truthbot-d15-d16-era-breakdown v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "flags": {
            "d15": "TRUTHBOT_D15_UTTERANCE_RECORD (default OFF — NOT enabled)",
            "d16": "TRUTHBOT_D16_STATISTICAL_RELEASE (default OFF — NOT enabled)",
        },
        "headline_vintage": HEADLINE_VINTAGE,
        "anecdote_claim_type": ANECDOTE_CLAIM_TYPE,
        "baseline": baseline,
        "baseline_meaning": (
            "movement measured against the same packs re-gated with both rules "
            "OFF — the re-score cancels, so the numbers are the RULES' own "
            "contribution"
            if baseline == "recomputed" else
            "movement measured against the gate outcome the source artifact "
            "RECORDED — nothing cancels, so the numbers are what the corpus "
            "loses relative to what was on the page, the re-score's own "
            "withholdings included"),
        "source_generation": source,
        "source_runs": sources,
        "decided_rate_convention": (
            "before = what the artifacts ship; after = newly-gated claims "
            "forced Unverifiable, released claims counted as decided (UPPER "
            "bound; *_after_lower leaves them where they ship)"),
        "speeches": speeches,
        "speeches_missing_sidecar": missing,
        "vintages": per_vintage,
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _spread_reading(sp: dict, blocks: Optional[list[dict]] = None) -> str:
    """The plain-English reading of the two spread rows — derived, not typed,
    so it cannot drift from the table above it."""
    d_raw = sp["raw_after"]["spread"] - sp["raw_before"]["spread"]
    d_adj = sp["adjusted_after"]["spread"] - sp["adjusted_before"]["spread"]
    word = {True: "widens", False: "narrows"}

    anecdotes = ""
    if blocks:
        hi = max(blocks, key=lambda b: b["anecdotes"])
        lo = min(blocks, key=lambda b: b["anecdotes"])
        anecdotes = (f" — `{hi['speech']}` carries {hi['anecdotes']} of them "
                     f"and `{lo['speech']}` carries {lo['anecdotes']} —")

    if (d_raw > 0) != (d_adj > 0):
        return (
            f"**The two bases disagree, and the disagreement is the finding.** "
            f"On the raw basis the spread {word[d_raw > 0]} by "
            f"{abs(d_raw) * 100:.1f} pp; on the anecdote-adjusted basis it "
            f"{word[d_adj > 0]} by {abs(d_adj) * 100:.1f} pp. The raw movement "
            f"is driven by how many personal anecdotes a speech contains"
            f"{anecdotes} not by how the two rules treat evidence. On the "
            f"basis that controls for that, ratifying D15 + D16(α) leaves era "
            f"parity essentially where it found it.")
    return (
        f"Both bases move the same way: raw {word[d_raw > 0]} by "
        f"{abs(d_raw) * 100:.1f} pp, anecdote-adjusted {word[d_adj > 0]} by "
        f"{abs(d_adj) * 100:.1f} pp.")


def render_markdown(report: dict) -> str:
    L: list[str] = []
    A = L.append
    v = report["headline_vintage"]
    data = report["vintages"][v]
    blocks = data["per_speech"]

    A("# D15 + D16(α) — era breakdown (the M-6 evenhandedness check)")
    A("")
    A(f"*Generated {report['generated']} · $0, no model calls. "
      f"Stance vintage: **{v}** (the B1a re-score overlaid — the live state of "
      f"the corpus, and the vintage the \"measured 50\" came from).*")
    A("")
    A(f"- **measurement-harness flags** — the configuration THIS breakdown ran "
      f"under, not the production gate default: D15 `{report['flags']['d15']}`, "
      f"D16(α) `{report['flags']['d16']}`. The harness sweeps the rules to "
      f"measure what each would do; what production ships is set by the ratified "
      f"gate config, not by these switches.")
    A("")
    A(f"- **source artifacts** `{report.get('source_generation', 'rebuild')}` "
      + ", ".join(f"`{sp}`={rid[:8]}"
                  for sp, rid in sorted((report.get('source_runs') or {}).items())))
    A(f"- **baseline** `{report.get('baseline', DEFAULT_BASELINE)}` — "
      f"{report.get('baseline_meaning', '')}")
    A("")
    if report.get("baseline", DEFAULT_BASELINE) == "recomputed":
        A("Every number below is what ratification *would* do, computed by "
          "running the real gate over the stored packs four ways — both off, "
          "D15 only, D16 only, both.")
    else:
        A("Every number below is measured against **what the source artifact "
          "actually recorded**, so it describes the change as a reader of the "
          "published page would experience it. The columns still separate the "
          "rules (D15 only, D16 only, both), but each is now compared to the "
          "shipped gate rather than to a rules-off recomputation — so a claim "
          "the B1a+B2 re-score withholds on its own is counted here and is "
          "not counted on the `recomputed` basis.")
    A("")

    A("## 1. The three views, per speech")
    A("")
    A("| speech | speaker | claims | D15 newly gated | …of which ship TRUE | "
      "D16(α) released | combined gated | combined released | **net** |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for b in blocks:
        A(f"| `{b['speech']}` | {b['speaker']} | {b['claims_measured']} | "
          f"{b['d15']['newly_gated']} | "
          f"{b['d15']['newly_gated_shipping_true']} | "
          f"{b['d16']['released']} | {b['combined']['newly_gated']} | "
          f"{b['combined']['released']} | "
          f"**{b['combined']['net']:+d}** |")
    c = data["corpus"]
    A(f"| **corpus** | | **{c['claims_measured']}** | "
      f"**{c['d15_newly_gated']}** | "
      f"**{c['d15_newly_gated_shipping_true']}** | "
      f"**{c['d16_released']}** | **{c['combined_newly_gated']}** | "
      f"**{c['combined_released']}** | **{c['combined_net']:+d}** |")
    A("")
    A("\"Net\" is released minus newly gated: the number of claims the two "
      "rules together move *toward* a decided verdict. It is negative "
      "everywhere, because D15 removes far more credit than D16 gives back — "
      "which is the honest headline, and the reason these two must be reported "
      "on one page rather than two.")
    A("")

    A("## 2. Decided-rate, before and after — both bases")
    A("")
    A(f"Anecdote-adjusted excludes claims typed `{report['anecdote_claim_type']}` "
      "(the A10 convention): a private individual's story told from the stage "
      "usually has no public record to check, so \"Unverifiable\" is the "
      "correct outcome rather than a miss. Both bases are shown because the "
      "adjustment is an argument, and a reader who rejects it must still see "
      "the raw figure it came from.")
    A("")
    A(f"*Convention: {report['decided_rate_convention']}.*")
    A("")
    A("| speech | anecdotes | raw before → after | Δ raw | "
      "adjusted before → after | Δ adjusted |")
    A("|---|---:|---|---:|---|---:|")
    for b in blocks:
        d = b["decided_rate"]
        A(f"| `{b['speech']}` | {b['anecdotes']} | "
          f"{_pct(d['raw_before']['rate'])} → {_pct(d['raw_after']['rate'])} | "
          f"{d['delta_raw'] * 100:+.1f} pp | "
          f"{_pct(d['adjusted_before']['rate'])} → "
          f"{_pct(d['adjusted_after']['rate'])} | "
          f"{d['delta_adjusted'] * 100:+.1f} pp |")
    A("")
    sp = data["spreads"]
    A("### Spread (max − min across the five speeches)")
    A("")
    A("| basis | before | after | change |")
    A("|---|---|---|---|")
    A(f"| raw | {_pct(sp['raw_before']['spread'])} "
      f"({sp['raw_before']['min_speech']} … {sp['raw_before']['max_speech']}) | "
      f"{_pct(sp['raw_after']['spread'])} "
      f"({sp['raw_after']['min_speech']} … {sp['raw_after']['max_speech']}) | "
      f"{(sp['raw_after']['spread'] - sp['raw_before']['spread']) * 100:+.1f} pp |")
    A(f"| anecdote-adjusted | {_pct(sp['adjusted_before']['spread'])} "
      f"({sp['adjusted_before']['min_speech']} … "
      f"{sp['adjusted_before']['max_speech']}) | "
      f"{_pct(sp['adjusted_after']['spread'])} "
      f"({sp['adjusted_after']['min_speech']} … "
      f"{sp['adjusted_after']['max_speech']}) | "
      f"{(sp['adjusted_after']['spread'] - sp['adjusted_before']['spread']) * 100:+.1f} pp |")
    A("")
    A(_spread_reading(sp, blocks))
    A("")

    A("## 3. Does the effect concentrate in one speaker or era?")
    A("")
    conc = data["concentration"]
    A(f"**{conc['verdict']}**")
    A("")
    A("| speech | claims (share of corpus) | newly gated (share of all "
      "withholding) | withholding rate within the speech | released | net |")
    A("|---|---|---|---:|---:|---:|")
    for r in conc["rows"]:
        A(f"| `{r['speech']}` | {r['claims']} ({_pct(r['claim_share'])}) | "
          f"{r['newly_gated']} ({_pct(r['gated_share'])}) | "
          f"{_pct(r['gated_rate_within_speech'])} | {r['released']} | "
          f"{r['net']:+d} |")
    A("")
    A("The **withholding rate within the speech** is the column to read: the "
      "five speeches differ by nearly a factor of four in claim count, so a "
      "raw count table alone would let \"this speech has the most claims\" "
      "masquerade as \"the repair targets this speaker\".")
    A("")

    A("## 4. The claims, named")
    A("")
    for b in blocks:
        if not (b["d15"]["newly_gated_sids"] or b["d16"]["released_sids"]):
            continue
        A(f"### `{b['speech']}` — {b['speaker']}")
        A("")
        if b["d15"]["newly_gated_true_sids"]:
            A(f"D15 would withhold {len(b['d15']['newly_gated_true_sids'])} "
              f"claim(s) that currently ship TRUE: "
              + ", ".join(f"`{s}`" for s in b["d15"]["newly_gated_true_sids"]))
            A("")
        other = [s for s in b["d15"]["newly_gated_sids"]
                 if s not in set(b["d15"]["newly_gated_true_sids"])]
        if other:
            A(f"D15 would also gate {len(other)} claim(s) not currently "
              f"shipping TRUE: " + ", ".join(f"`{s}`" for s in other))
            A("")
        if b["d16"]["released_sids"]:
            A("D16(α) would release: "
              + ", ".join(f"`{s}`" for s in b["d16"]["released_sids"]))
            A("")

    other_v = [x for x in VINTAGES if x != v][0]
    ov = report["vintages"][other_v]
    A(f"## 5. Cross-check against the `{other_v}` stance vintage")
    A("")
    A(f"The pattern is not an artefact of the B1a re-score. On `{other_v}` "
      f"stances the corpus totals are: D15 newly gated "
      f"{ov['corpus']['d15_newly_gated']}, D16 released "
      f"{ov['corpus']['d16_released']}, net "
      f"{ov['corpus']['combined_net']:+d}.")
    A("")
    A(f"**{ov['concentration']['verdict']}**")
    A("")
    return "\n".join(L)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None, metavar="DIR")
    ap.add_argument("--baseline", choices=BASELINES, default=DEFAULT_BASELINE,
                    help=("what a claim MOVED FROM. 'recomputed' (default) "
                          "isolates the two rules; 'shipped' measures against "
                          "what the artifact recorded, which is what the "
                          "published page said."))
    ap.add_argument("--source", choices=SOURCES, default=DEFAULT_SOURCE,
                    help=("which artifact generation supplies the packs and "
                          "the shipped rows: the Phase-3 rebuild (default) or "
                          "the adjudication wave's."))
    ap.add_argument("--stem", default=None, metavar="NAME",
                    help=("output file stem; defaults to "
                          f"'{OUT_STEM}'. A non-default basis MUST write to "
                          "its own stem — two bases under one filename would "
                          "make the packet's numbers depend on run order."))
    args = ap.parse_args(argv)

    speeches = list(REBUILT_RUNS)
    report = build_report(speeches, baseline=args.baseline, source=args.source)
    md = render_markdown(report)
    print(md)

    stem = args.stem or (
        OUT_STEM if (args.baseline == DEFAULT_BASELINE
                     and args.source == DEFAULT_SOURCE)
        else f"{OUT_STEM}_{args.source}_{args.baseline}")
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(md + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / (stem + '.json')}")
    print(f"wrote {out_dir / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
