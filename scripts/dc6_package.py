#!/usr/bin/env python3
"""DC-6 review package — the human-judgement surface for the Phase-3 rebuild ($0).

Phase 3 re-adjudicated all five SOTU speeches on generation
``v2.3-role-axis-s5cap``. The new artifacts are committed but UNPUBLISHED
(``methodology_manifest.json`` marks them ``published: false``); the live
``site-pca/`` tree still renders the old runs. Before anything is published a
human has to be able to see, in one place:

* how many verdicts moved, in which direction, and on which claims;
* whether the era-parity claim survives — i.e. whether the modern speeches got
  gated harder than the historical ones (old vs new decided-rate, side by side);
* what the rebuild cost, split into ledger-true proxy spend and off-proxy
  ESTIMATES at list rates;
* what the public corrections ledger would say if this shipped.

Everything here is derived deterministically from artifacts already on disk —
the five ``metrics/remediation_v2/phase3_<speech>_verdict_diff.json`` files, the
``metrics/pca_runs/`` artifacts they name, the Phase-2 dry-run worksheet, and
the run logs. NO model or API calls. Nothing under ``site-pca/`` is touched.

Usage (repo root)::

    PYTHONPATH=. .venv/bin/python scripts/dc6_package.py \
        --new-site /tmp/dc6-site [--old-site site-pca] [--write-archive]

Outputs (all under ``metrics/remediation_v2/``)::

    dc6_review.json / dc6_review.md         corpus + per-speech + every changed claim
    dc6_corrections_entries.json            one entry per changed verdict
    dc6_corrections_ledger_proposed.json    the PROPOSED post-reset data/corrections.json

**Publish gate (A7).** This report is the human-judgement surface; the machine
gate beside it is the named acceptance suite
``tests/acceptance/test_dc6_acceptance_gate.py``, run as::

    .venv/bin/python -m pytest -m acceptance -q

It asserts the hand-adjudicated cases against the STAGED artifacts and must
pass before anything here is published. Cases still blocked on the B1a
re-score are marked ``xfail(strict=True)`` and will announce themselves as
XPASS the moment that repair lands.

``--write-archive`` additionally writes ``data/corrections-archive-<date>.json``
(a copy of the current ledger). It is purely additive; ``data/corrections.json``
is never modified by this script — the reset is a reviewable proposal, applied
at publish time under jackie's gate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

DIFF_DIR = REPO / "metrics" / "remediation_v2"
RUNS_DIR = REPO / "metrics" / "pca_runs"

#: Speech order used in every report section — chronological by utterance.
SPEECH_ORDER = ["clinton_1998", "gwbush_2006", "obama_2014", "biden_2022",
                "trump_2026"]

SPEAKERS = {"clinton_1998": "Bill Clinton", "gwbush_2006": "George W. Bush",
            "obama_2014": "Barack Obama", "biden_2022": "Joe Biden",
            "trump_2026": "Donald Trump"}

#: Display vocabulary, in the site's family order (true → abstain → adverse).
#: "Mostly True" is carried even though this corpus has none, because the fine
#: axis can emit it and a distribution table that silently drops a live label
#: is exactly the class of bug remediation v2 exists to kill.
DISPLAY_ORDER = ["True", "Mostly True", "Misleading", "False", "Unverifiable",
                 "Models split"]

#: Buckets that are NOT a substantive ruling. decided-rate = 1 - share of these.
ABSTAIN = {"Unverifiable", "Models split"}

#: Change classes, most consequential first — a claim that moved between two
#: substantive verdicts is the one a reader must look at.
CLASS_ORDER = ["decided_to_decided_changed", "newly_gated", "newly_decided",
               "split_changes", "other"]

CLASS_TITLE = {
    "decided_to_decided_changed": "Decided → decided (verdict flipped between substantive rulings)",
    "newly_gated": "Newly gated (was decided, now withheld as Unverifiable)",
    "newly_decided": "Newly decided (was withheld/split, now a substantive ruling)",
    "split_changes": "Split changes (model-split status changed)",
    "other": "Other / unclassified",
}

#: Verdict-contract label → published display label.
LABEL_DISPLAY = {
    "TRUE": "True",
    "MOSTLY TRUE": "Mostly True",
    "MISLEADING": "Misleading",
    "FALSE": "False",
    "UNVERIFIABLE": "Unverifiable",
    "GATED-UNVERIFIABLE": "Unverifiable",
    "MODELS SPLIT": "Models split",
    "NO VERDICT": "Models split",
}

#: The four verdicts data/corrections.json's loader accepts (mirrors
#: truthbot.publish.corrections._VALID_VERDICTS — imported at runtime for the
#: real check; duplicated here only so the module reads standalone).
LEDGER_VERDICTS = {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}

REBUILD_DATE = "2026-08-06"
GENERATION = "v2.3-role-axis-s5cap"
CORRECTION_SOURCE = f"phase3-rebuild-{REBUILD_DATE} ({GENERATION})"

#: Which generation of per-speech verdict diffs a run reads. The Phase-3
#: rebuild's and the adjudication wave's live side by side under the same
#: directory: the wave re-adjudicated 29 of the rebuild's claims, so its diffs
#: describe the same speeches and must not overwrite the rebuild's record.
PHASE3_DIFF_GLOB = "phase3_*_verdict_diff.json"
WAVE_DIFF_GLOB = "wave_*_verdict_diff.json"
#: The 2026-08-10 rulings pass — the wave's DEFERRED newly-gated set applied,
#: plus the R-3 rationale re-emit and the D14 coherence annotation. A third
#: generation of diffs beside the other two, for the same reason: it describes
#: the same speeches and must not overwrite either earlier record.
RULINGS_DIFF_GLOB = "rulings_*_verdict_diff.json"

WAVE_DATE = "2026-08-09"
WAVE_SOURCE = f"adjudication-wave-{WAVE_DATE} ({GENERATION})"
WAVE_FLIPSET = DIFF_DIR / "regate_flipset.json"

#: Mechanical release clauses for the wave, by why the claim was in it. These
#: are FACTS about the mechanism (S-8): which rule released the claim, and that
#: it was re-adjudicated on the unified pipeline. They deliberately contain no
#: lineage narrative and no editorial characterisation of any verdict — the
#: owner writes that paragraph, and a generator that drafted it would be
#: putting words in the publication's mouth.
WAVE_RELEASE_CLAUSE = (
    "released from the evidence gate by the D16(alpha) statistical-release "
    "and D15 utterance-record rules ratified 2026-08-09, applied to the "
    "B1a+B2 stance re-score of the stored evidence pack; re-adjudicated on "
    "the unified pipeline with no new retrieval")
WAVE_EXTRA_CLAUSE = (
    "re-adjudicated on the unified pipeline at the owner's designation, on "
    "the B1a+B2 stance re-score of the stored evidence pack, with no new "
    "retrieval")
WAVE_SPLIT_CLAUSE = (
    "shipped as a models-split with no verdict, which no deterministic "
    "re-gate can settle; re-adjudicated on the unified pipeline with no new "
    "retrieval")
WAVE_CLAUSE_BY_REASON = {
    "released": WAVE_RELEASE_CLAUSE,
    "named-extra": WAVE_EXTRA_CLAUSE,
    "models-split extra": WAVE_SPLIT_CLAUSE,
}

RULINGS_DATE = "2026-08-10"
RULINGS_SOURCE = f"wave-rulings-{RULINGS_DATE} ({GENERATION})"
RULINGS_MECHANISM_PATH = DIFF_DIR / "deferred_gated_mechanism.json"

#: Per-claim MECHANISM clauses for the applied withholdings. The wave RELEASED
#: claims and had one clause for all of them; this pass WITHHOLDS them, and the
#: reason differs per claim — so the clause is chosen from the measured
#: attribution rather than shared. Facts only (S-8): which mechanism withdrew
#: the qualifying evidence, and that no panel call was made or needed.
RULINGS_MECHANISM_CLAUSE = {
    "re-score": (
        "withheld by the evidence gate after the B1a+B2 stance re-score of the "
        "stored evidence pack: with real relevance and stance scores the pack "
        "no longer meets the Tier-1..3 bearing quota. Neither ratified rule is "
        "required to reach this outcome"),
    "D15": (
        "withheld by the evidence gate under the D15 utterance-record rule "
        "(ratified 2026-08-09), applied to the B1a+B2 stance re-score of the "
        "stored evidence pack: items that reproduce the utterance itself — the "
        "transcript, the Congressional Record of the day, the official "
        "compilation — carry no quota credit, and without them the pack no "
        "longer meets the bearing quota"),
    "D16alpha": (
        "withheld by the evidence gate under the D16(alpha) statistical-release "
        "rule (ratified 2026-08-09), applied to the B1a+B2 stance re-score of "
        "the stored evidence pack"),
    "D15+D16alpha (interaction)": (
        "withheld by the evidence gate under the D15 utterance-record and "
        "D16(alpha) statistical-release rules acting together (both ratified "
        "2026-08-09), applied to the B1a+B2 stance re-score of the stored "
        "evidence pack: neither rule alone withdraws enough qualifying "
        "evidence to fail the bearing quota"),
}

#: The two NON-gating changes this pass makes, both provenance-level.
RULINGS_RATIONALE_CLAUSE = (
    "the published rationale was re-emitted from stored panel output: the "
    "stage-2 discriminator resolved this claim out of a three-way tie and "
    "recorded no rationale text, so the rationale of the seat that reached the "
    "same verdict in the prior run was adopted verbatim and attributed. The "
    "verdict is unchanged")
RULINGS_COHERENCE_CLAUSE = (
    "an adjacent-claim coherence annotation was added: this claim and its "
    "neighbour rate the same statistic and carry different published verdicts, "
    "which is now disclosed on both. Neither verdict was changed")

#: The ratified rationale for trump_2026:0469 (2026-08-09). The claim is NOT in
#: the wave — it stays Unverifiable by ratification, not by defect — but if it
#: ever reaches the ledger for any reason, this is the reason it carries, and
#: it must not be re-derived by whatever produced the entry.
BECKSTROM_SID = "trump_2026:0469"
BECKSTROM_RATIONALE = (
    "purposive clause uncheckable; factual core confirmed; the sole purposive "
    "support is Political-tier, which under Claim Eval v3 is attribution, "
    "never proof")

# ── Spend ledger ──────────────────────────────────────────────────────────
# proxy_usd  = LiteLLM proxy key spend — LEDGER-TRUE (the proxy billed it).
# offproxy_usd = models called OUTSIDE the proxy, costed at published list
#                rates from token counts — an ESTIMATE, not a receipt.
# stated_usd = the per-speech figure quoted in the DC-6 brief; carried so the
#              report can flag where the brief and the run logs disagree
#              instead of quietly preferring one.
SPEND: dict[str, dict[str, Any]] = {
    "gwbush_2006": {"legs": 1, "proxy_usd": 0.2479, "offproxy_usd": 2.8344,
                    "stated_usd": 3.08,
                    "note": "single leg, 10 chunks"},
    "clinton_1998": {"legs": 1, "proxy_usd": 0.8791, "offproxy_usd": 5.9663,
                     "stated_usd": 6.85,
                     "note": "single leg, 19 chunks"},
    "obama_2014": {"legs": 2, "proxy_usd": 0.7577, "offproxy_usd": 6.4780,
                   "stated_usd": 7.24,
                   "note": ("leg 1 banked 80/96 rows (proxy $0.6586, off-proxy "
                            "est $5.4007) before an L-W worker failure; leg 2 "
                            "ran the remaining 16 (proxy $0.0991, off-proxy est "
                            "$1.0773)")},
    "biden_2022": {"legs": 2, "proxy_usd": 0.7361, "offproxy_usd": 7.6371,
                   "stated_usd": 8.00,
                   "note": ("leg 1 banked 60/111 rows (proxy $0.3480, off-proxy "
                            "est $4.3690) before a browsing-model timeout; leg 2 "
                            "ran the remaining 51 (proxy $0.3881, off-proxy est "
                            "$3.2681)")},
    "trump_2026": {"legs": 1, "proxy_usd": 1.7272, "offproxy_usd": 11.8864,
                   "stated_usd": 13.61,
                   "note": "single leg, 37 chunks"},
}

#: Claim-shape backfill sidecars (haiku on the L-P proxy) — ledger-true.
SHAPE_BACKFILL_USD = 0.63

#: Corpus total quoted in the DC-6 brief, for the same flag-don't-smooth check.
STATED_TOTAL_USD = 38.8

#: The two speeches whose runs were interrupted and RESUMED. Named, not
#: inferred from ``legs``, because the disclosure below is about these specific
#: runs' accounting and must not silently start or stop applying if the leg
#: bookkeeping changes.
RESUMED_SPEECHES = ("obama_2014", "biden_2022")

#: DC-B1 carry-forward obligation → the DC-6' final ledger.
#:
#: Two different honesty problems live in one total, and both have to be said
#: out loud every time the total is quoted:
#:
#: 1. MIXED BASIS. ``proxy`` is a receipt; ``off-proxy`` is an estimate. Adding
#:    them is the only way to get a corpus number, but the sum is not a
#:    ledger-true figure and must never be presented as one.
#: 2. RESUMED-LEG UNDERCOUNT. ``phase3_rebuild`` banks only PROXY spend in the
#:    chunk journal (``append_chunk_journal`` is handed the ``proxy_key_spend``
#:    delta), so a resumed session carries the prior leg's proxy cost forward
#:    and drops that leg's off-proxy estimate entirely. The per-speech figures
#:    in this table are reconstructed by hand from BOTH legs' logs, which
#:    recovers what the runner's own SPEND line lost — but only down to the
#:    last chunk that got banked. Both interrupted legs died *inside* the next
#:    chunk, after its retrieval had already run, and that retrieval was never
#:    printed or journalled. It is unrecoverable.
#:
#: Hence: the off-proxy component, and therefore the corpus total, is a LOWER
#: BOUND. This is exactly why the DC-B1 estimate prices re-adjudication off the
#: three single-session runs only (gwbush/clinton/trump, $0.0642–$0.0748 per
#: claim) and excludes the two resumed ones.
SPEND_BASIS_DISCLOSURE = (
    "MIXED COST BASIS, AND A KNOWN UNDERCOUNT — this total is a lower bound, "
    "not a receipt. `proxy` is LEDGER-TRUE: the LiteLLM proxy key was billed "
    "and the figure is read back from that ledger. `off-proxy` is an ESTIMATE: "
    "models called outside the proxy, costed from token counts at published "
    "list rates. The two are summed here because there is no better corpus "
    "number, but they are not the same kind of evidence. Separately, "
    f"{' and '.join(RESUMED_SPEECHES)} were interrupted and RESUMED, and "
    "phase3_rebuild banks only PROXY spend in the chunk journal "
    "(append_chunk_journal is handed the proxy_key_spend delta), so a resumed "
    "session carries the prior leg's proxy cost forward and DROPS that leg's "
    "off-proxy estimate — which is why those two runs' self-reported totals "
    "($1.8350 obama_2014, $4.0042 biden_2022) are far below the per-leg "
    "figures reconstructed in this table. That reconstruction recovers only "
    "what was banked: both legs died inside the following chunk, after its "
    "retrieval had already run and before anything was journalled, and that "
    "spend is unrecoverable. The off-proxy component — and therefore the "
    "corpus total — is a LOWER BOUND, and the two resumed runs' per-claim "
    "rates must not be used to price future work (DC-B1 prices off the three "
    "single-session runs only)."
)


# ── loading ───────────────────────────────────────────────────────────────
def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_diffs(diff_dir: Path = DIFF_DIR,
               pattern: str = PHASE3_DIFF_GLOB) -> list[dict]:
    """The per-speech verdict diffs, in SPEECH_ORDER.

    ``pattern`` picks WHICH generation of diffs. It defaults to the Phase-3
    rebuild's; :data:`WAVE_DIFF_GLOB` picks the adjudication wave's. Two globs
    in one directory rather than two directories, because the wave's diffs
    describe the same speeches and must sit BESIDE the rebuild's record
    instead of replacing it."""
    found = {}
    for path in sorted(Path(diff_dir).glob(pattern)):
        doc = _read_json(path)
        found[doc["speech_id"]] = doc
    order = [s for s in SPEECH_ORDER if s in found]
    order += [s for s in sorted(found) if s not in SPEECH_ORDER]
    return [found[s] for s in order]


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> Optional[dict]:
    """Artifact for a run id (accepts the 8-char short form)."""
    for path in Path(runs_dir).glob("*.json"):
        if path.stem == run_id or path.stem.startswith(run_id):
            return _read_json(path)
    return None


def display(label: str) -> str:
    """Verdict-contract label → published display label."""
    return LABEL_DISPLAY.get(str(label).strip().upper(), str(label))


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ── aggregation ───────────────────────────────────────────────────────────
def aggregate(diffs: Iterable[dict]) -> dict:
    """Corpus + per-speech totals over the five verdict diffs.

    ``claims`` is the diff's ``n_compared`` — the number of sids the rebuild
    actually re-adjudicated. It is NOT necessarily the old run's row count:
    see ``coverage`` below, which surfaces any sid the old run carried and the
    new one does not (that gap is a publish-blocking fact, not a rounding
    error, so it is reported rather than absorbed)."""
    per_speech: dict[str, dict] = {}
    totals = Counter()
    for d in diffs:
        counts = dict(d["counts"])
        changed = sum(v for k, v in counts.items() if k != "unchanged")
        row = {
            "speech_id": d["speech_id"],
            "speaker": SPEAKERS.get(d["speech_id"], ""),
            "old_run_id": d.get("rebuild_of", ""),
            "new_run_id": d.get("new_run_id", ""),
            "claims": d["n_compared"],
            "unchanged": counts.get("unchanged", 0),
            "decided_to_decided_changed": counts.get("decided_to_decided_changed", 0),
            "newly_gated": counts.get("newly_gated", 0),
            "newly_decided": counts.get("newly_decided", 0),
            "split_changes": counts.get("split_changes", 0),
            "other": counts.get("other", 0),
            "changed_total": changed,
            "gate_forced_new": d.get("gate_forced_new", 0),
        }
        per_speech[d["speech_id"]] = row
        for key in ("claims", "unchanged", "decided_to_decided_changed",
                    "newly_gated", "newly_decided", "split_changes", "other",
                    "changed_total", "gate_forced_new"):
            totals[key] += row[key]
    return {"corpus": dict(totals), "per_speech": per_speech}


def _dist_from_tally(tally: dict) -> dict[str, int]:
    """Verdict-contract tally → display distribution over DISPLAY_ORDER."""
    out = {label: 0 for label in DISPLAY_ORDER}
    for label, count in (tally or {}).items():
        out[display(label)] = out.get(display(label), 0) + count
    return out


def decided_rate(dist: dict[str, int]) -> dict:
    """decided = every claim NOT in an abstain bucket (Unverifiable / split)."""
    total = sum(dist.values())
    decided = sum(c for label, c in dist.items() if label not in ABSTAIN)
    return {"decided": decided, "total": total,
            "rate": round(decided / total, 4) if total else 0.0}


def distributions(diffs: Iterable[dict]) -> dict:
    """Old vs new verdict distribution + decided-rate, per speech and corpus.

    NOTE the denominators: ``old_tally`` counts the OLD artifact's rows and
    ``new_tally`` the NEW artifact's rows. Where a rebuild dropped a sid the
    two differ, and the decided-rate comparison is over different denominators
    — flagged per speech via ``denominator_mismatch`` rather than normalised
    away."""
    per_speech: dict[str, dict] = {}
    old_corpus, new_corpus = Counter(), Counter()
    for d in diffs:
        old = _dist_from_tally(d["old_tally"])
        new = _dist_from_tally(d["new_tally"])
        old_corpus.update(old)
        new_corpus.update(new)
        old_rate, new_rate = decided_rate(old), decided_rate(new)
        per_speech[d["speech_id"]] = {
            "speech_id": d["speech_id"],
            "speaker": SPEAKERS.get(d["speech_id"], ""),
            "old": old, "new": new,
            "old_decided": old_rate, "new_decided": new_rate,
            "decided_rate_delta": round(new_rate["rate"] - old_rate["rate"], 4),
            "denominator_mismatch": old_rate["total"] != new_rate["total"],
        }
    old_c = {label: old_corpus.get(label, 0) for label in DISPLAY_ORDER}
    new_c = {label: new_corpus.get(label, 0) for label in DISPLAY_ORDER}
    old_rate, new_rate = decided_rate(old_c), decided_rate(new_c)
    # Era parity is the load-bearing claim of the whole remediation: a
    # methodology that gates the modern speech harder than the historical ones
    # produces a politically-shaped result. Measure the SPREAD of decided-rates
    # across speeches, old vs new — narrowing = more parity, widening = less.
    old_rates = {s: v["old_decided"]["rate"] for s, v in per_speech.items()}
    new_rates = {s: v["new_decided"]["rate"] for s, v in per_speech.items()}

    def _spread(rates: dict[str, float]) -> dict:
        if not rates:
            return {"min": 0.0, "max": 0.0, "spread": 0.0, "min_speech": "",
                    "max_speech": ""}
        lo = min(rates, key=rates.get)
        hi = max(rates, key=rates.get)
        return {"min": rates[lo], "max": rates[hi],
                "spread": round(rates[hi] - rates[lo], 4),
                "min_speech": lo, "max_speech": hi}

    old_spread, new_spread = _spread(old_rates), _spread(new_rates)
    return {
        "per_speech": per_speech,
        "corpus": {"old": old_c, "new": new_c,
                   "old_decided": old_rate, "new_decided": new_rate,
                   "decided_rate_delta": round(new_rate["rate"] - old_rate["rate"], 4)},
        "era_parity": {
            "old_spread": old_spread,
            "new_spread": new_spread,
            "spread_delta": round(new_spread["spread"] - old_spread["spread"], 4),
            "narrowed": new_spread["spread"] < old_spread["spread"],
        },
    }


# ── Anecdote-adjusted parity (remediation v2 Phase A, A10) ───────────────────
#
# The raw decided-rate treats every Unverifiable as a gate failure. For one
# genre it is not: a private individual's story told from the stage usually has
# no public record to check against, so "Unverifiable" is the CORRECT outcome,
# not a miss. The site already says this (the anecdote footnote under every
# verdict bar); the review generator did not, which meant the era-parity number
# — the load-bearing claim of the whole remediation — was being read off a
# denominator that penalises a speech for how many guests its author thanked.
#
# Trump's rebuild carries 52 anecdotes out of 182 claims and Bush's carries 1
# out of 48, so this is not a rounding correction: it is the difference between
# comparing methodologies and comparing speechwriting styles. Both bases are
# reported side by side — the adjustment is an argument, and a reader who
# rejects it must still be able to see the raw figure it was derived from.

#: Layer-A claim type marking the anecdote genre. Same string the renderer
#: keys on (``site._is_anecdote_unverifiable``) and the consistency checker
#: re-derives the anecdote footnote from (``consistency.py``, the
#: ``layer_a_claim_type`` read) — one convention, three consumers.
ANECDOTE_CLAIM_TYPE = "personal-anecdote"


def _site_claim_type_index(site_root: Path) -> dict[tuple[str, str], str]:
    """``(speaker, normalised claim text) -> layer_a_claim_type`` from a
    published site tree.

    The fallback join for artifacts that do not carry the provenance inline.
    Keyed on text rather than claim id because ids are minted per render
    (``uuid4``) — the same reason ``badge_diff`` keys on text.
    """
    site_root = Path(site_root)
    claims_path = site_root / "data" / "claims.json"
    reports_path = site_root / "data" / "reports.json"
    if not claims_path.exists() or not reports_path.exists():
        return {}
    reports = _read_json(reports_path)
    speaker_by_report = {r.get("id") or r.get("report_id"): r.get("speaker", "")
                         for r in reports}
    index: dict[tuple[str, str], str] = {}
    for c in _read_json(claims_path):
        ctype = (c.get("provenance") or {}).get("layer_a_claim_type") or ""
        if not ctype:
            continue
        key = (speaker_by_report.get(c.get("report_id"), ""),
               _norm_text(c.get("claim_text", "")))
        index[key] = ctype
    return index


def claim_types_for_speech(run: dict, speech_id: str,
                           site_index: dict[tuple[str, str], str]
                           ) -> tuple[dict[str, str], dict]:
    """``(sid -> claim_type, join report)`` for one run artifact.

    Prefers the artifact's own ``claims[].layer_a.claim_type``; falls back to
    the published claims.json by (speaker, normalised claim text). A sid that
    resolves through NEITHER is reported by sid, never silently counted as
    "not an anecdote" — that would quietly inflate the adjusted denominator
    with exactly the claims we failed to classify.
    """
    speaker = SPEAKERS.get(speech_id, "")
    types: dict[str, str] = {}
    from_artifact = from_site = 0
    unresolved: list[str] = []
    for claim in run.get("claims") or []:
        sid = claim.get("sid", "")
        if not sid:
            continue
        ctype = ((claim.get("layer_a") or {}).get("claim_type") or "").strip()
        if ctype:
            types[sid] = ctype
            from_artifact += 1
            continue
        ctype = site_index.get((speaker, _norm_text(claim.get("text", ""))), "")
        if ctype:
            types[sid] = ctype
            from_site += 1
        else:
            unresolved.append(sid)
    return types, {"from_artifact": from_artifact, "from_claims_json": from_site,
                   "unresolved": len(unresolved),
                   "unresolved_sids": sorted(unresolved)}


def _rate(labels: Iterable[str]) -> dict:
    """decided-rate over a bare sequence of verdict-contract labels."""
    labels = [display(x) for x in labels]
    total = len(labels)
    decided = sum(1 for x in labels if x not in ABSTAIN)
    return {"decided": decided, "total": total,
            "rate": round(decided / total, 4) if total else 0.0}


def anecdote_parity(diffs: Iterable[dict], runs_dir: Path = RUNS_DIR,
                    site_root: Optional[Path] = None) -> dict:
    """Decided-rate per speech and corpus-wide, RAW and anecdote-adjusted.

    Both bases are computed over the diff's ``per_sid`` list — the sids the
    rebuild actually compared — so the raw and adjusted figures share a
    denominator and differ ONLY by the anecdote exclusion. Where that set is
    smaller than section 4's tally (a sid the rebuild dropped),
    ``raw_matches_section4`` says so instead of the two quietly disagreeing.
    """
    site_root = Path(site_root) if site_root is not None else REPO / "site-pca"
    site_index = _site_claim_type_index(site_root)

    per_speech: dict[str, dict] = {}
    join_totals = {"from_artifact": 0, "from_claims_json": 0, "unresolved": 0}
    unresolved_sids: list[str] = []
    corpus: dict[str, list[str]] = {"old_raw": [], "new_raw": [],
                                    "old_adj": [], "new_adj": []}
    for d in diffs:
        speech_id = d["speech_id"]
        run = load_run(d.get("new_run_id", ""), runs_dir) or {}
        types, join = claim_types_for_speech(run, speech_id, site_index)
        for key in join_totals:
            join_totals[key] += join[key]
        unresolved_sids += join["unresolved_sids"]

        rows = d.get("per_sid") or []
        anecdote_sids = {r["sid"] for r in rows
                         if types.get(r["sid"]) == ANECDOTE_CLAIM_TYPE}
        kept = [r for r in rows if r["sid"] not in anecdote_sids]

        old_raw, new_raw = _rate([r["old"] for r in rows]), _rate([r["new"] for r in rows])
        old_adj, new_adj = _rate([r["old"] for r in kept]), _rate([r["new"] for r in kept])
        corpus["old_raw"] += [r["old"] for r in rows]
        corpus["new_raw"] += [r["new"] for r in rows]
        corpus["old_adj"] += [r["old"] for r in kept]
        corpus["new_adj"] += [r["new"] for r in kept]

        # How many of the excluded anecdotes were in fact abstentions — the
        # size of the effect the adjustment is arguing about.
        anecdote_rows = [r for r in rows if r["sid"] in anecdote_sids]
        per_speech[speech_id] = {
            "speech_id": speech_id,
            "speaker": SPEAKERS.get(speech_id, ""),
            "claims_compared": len(rows),
            "anecdotes": len(anecdote_sids),
            "anecdote_share": (round(len(anecdote_sids) / len(rows), 4)
                               if rows else 0.0),
            "anecdotes_abstained_new": sum(
                1 for r in anecdote_rows if display(r["new"]) in ABSTAIN),
            "old_raw": old_raw, "new_raw": new_raw,
            "old_adjusted": old_adj, "new_adjusted": new_adj,
            "delta_raw": round(new_raw["rate"] - old_raw["rate"], 4),
            "delta_adjusted": round(new_adj["rate"] - old_adj["rate"], 4),
            "raw_matches_section4":
                sum(d["old_tally"].values()) == len(rows)
                and sum(d["new_tally"].values()) == len(rows),
            "join": join,
        }

    def _spread(rates: dict[str, float]) -> dict:
        if not rates:
            return {"min": 0.0, "max": 0.0, "spread": 0.0, "min_speech": "",
                    "max_speech": ""}
        lo, hi = min(rates, key=rates.get), max(rates, key=rates.get)
        return {"min": rates[lo], "max": rates[hi],
                "spread": round(rates[hi] - rates[lo], 4),
                "min_speech": lo, "max_speech": hi}

    spreads = {}
    for basis, field in (("old_raw", "old_raw"), ("new_raw", "new_raw"),
                         ("old_adjusted", "old_adjusted"),
                         ("new_adjusted", "new_adjusted")):
        spreads[basis] = _spread({s: v[field]["rate"]
                                  for s, v in per_speech.items()})

    return {
        "anecdote_claim_type": ANECDOTE_CLAIM_TYPE,
        "per_speech": per_speech,
        "corpus": {
            "old_raw": _rate(corpus["old_raw"]),
            "new_raw": _rate(corpus["new_raw"]),
            "old_adjusted": _rate(corpus["old_adj"]),
            "new_adjusted": _rate(corpus["new_adj"]),
        },
        "spread": {
            **spreads,
            "raw_spread_delta": round(spreads["new_raw"]["spread"]
                                      - spreads["old_raw"]["spread"], 4),
            "adjusted_spread_delta": round(spreads["new_adjusted"]["spread"]
                                           - spreads["old_adjusted"]["spread"], 4),
            "raw_narrowed": (spreads["new_raw"]["spread"]
                             < spreads["old_raw"]["spread"]),
            "adjusted_narrowed": (spreads["new_adjusted"]["spread"]
                                  < spreads["old_adjusted"]["spread"]),
        },
        "join": {**join_totals, "unresolved_sids": sorted(unresolved_sids),
                 "site_root": str(site_root),
                 "site_index_size": len(site_index)},
    }


def coverage(diffs: Iterable[dict], runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Per speech: old row count vs new row count vs sids compared.

    The verdict diff iterates the NEW rows, so a sid the old run adjudicated
    and the new one never saw is invisible in it. That is precisely the kind
    of silent drop this package must not smooth over."""
    out = []
    for d in diffs:
        old = load_run(d.get("rebuild_of", ""), runs_dir) or {}
        new = load_run(d.get("new_run_id", ""), runs_dir) or {}
        old_sids = [r.get("sid") for r in (old.get("rows") or [])]
        new_sids = {r.get("sid") for r in (new.get("rows") or [])}
        dropped = sorted(s for s in old_sids if s not in new_sids)
        old_claim_sids = {c.get("sid") for c in (old.get("claims") or [])}
        old_verdicts = {r.get("sid"): r.get("verdict")
                        for r in (old.get("rows") or [])}
        out.append({
            "speech_id": d["speech_id"],
            "old_rows": len(old_sids),
            "old_claims": len(old_claim_sids),
            "new_rows": len(new_sids),
            "compared": d["n_compared"],
            "dropped_sids": dropped,
            "dropped_detail": [
                {"sid": s, "old_verdict": old_verdicts.get(s),
                 # An old row with no matching claim record is an ORPHAN: the
                 # publisher had no text for it and rendered "(claim text
                 # unavailable)". Dropping it removes a broken card, not data.
                 "had_claim_record": s in old_claim_sids}
                for s in dropped],
            "added_sids": sorted(new_sids - set(old_sids)),
        })
    return out


# ── changed claims ────────────────────────────────────────────────────────
def changed_claims(diffs: Iterable[dict], runs_dir: Path = RUNS_DIR,
                   text_limit: int = 140, rationale_limit: int = 200) -> list[dict]:
    """Every claim whose outcome moved, with the NEW panel rationale."""
    rows: list[dict] = []
    for d in diffs:
        new_run = load_run(d.get("new_run_id", ""), runs_dir) or {}
        reasoning = {r.get("sid"): (r.get("reasoning") or "")
                     for r in (new_run.get("rows") or [])}
        full_text = {c.get("sid"): (c.get("text") or "")
                     for c in (new_run.get("claims") or [])}
        for entry in d["per_sid"]:
            if entry["category"] == "unchanged":
                continue
            sid = entry["sid"]
            text = full_text.get(sid) or entry.get("text") or ""
            rows.append({
                "sid": sid,
                "speech_id": d["speech_id"],
                "speaker": SPEAKERS.get(d["speech_id"], ""),
                "claim_text": _truncate(text, text_limit),
                "claim_text_full": " ".join(str(text).split()),
                "old_label": entry["old"],
                "new_label": entry["new"],
                "old_verdict": display(entry["old"]),
                "new_verdict": display(entry["new"]),
                "change_class": entry["category"],
                "rationale": _truncate(reasoning.get(sid, ""), rationale_limit),
            })
    rows.sort(key=lambda r: (CLASS_ORDER.index(r["change_class"])
                             if r["change_class"] in CLASS_ORDER else 99,
                             SPEECH_ORDER.index(r["speech_id"])
                             if r["speech_id"] in SPEECH_ORDER else 99,
                             r["sid"]))
    return rows


# ── corrections entries ───────────────────────────────────────────────────
_DISPOSITION_LABEL = {
    "fc-excluded": "fact-check-excluded",
    "era-violation": "era-invalid (published after the speech)",
    "s5-capped": "over-cap political",
    "mutable-endpoint": "mutable live-endpoint",
}

_CLASS_CAUSE = {
    "decided_to_decided_changed":
        "the panel reached a different substantive verdict on the rebuilt evidence pack",
    "newly_gated":
        "the rebuilt pack no longer clears the evidence gate "
        "(insufficient-qualifying-evidence), so the claim is reported "
        "Unverifiable instead of decided",
    "newly_decided":
        "the rebuilt pack cleared the evidence gate, so the panel returned a "
        "substantive verdict where the prior run withheld one",
    "split_changes":
        "the panel's model votes changed split status on the rebuilt pack",
    "other": "",
}

GENERIC_CAUSE = ("re-adjudicated under the unified v2.3 pipeline (evidence pack "
                 "rebuilt under the corrected tier registry, era gate, and "
                 "political-source cap)")


def load_dc5_dispositions(path: Path = DIFF_DIR / "dc5_worksheet.json") -> dict[str, dict]:
    """sid → the Phase-2 dry-run's projected evidence losses for that claim.

    This is the dry run's PROJECTION over the old pack, not a measurement of
    the rebuilt pack. Reasons that quote it say so."""
    path = Path(path)
    if not path.exists():
        return {}
    doc = _read_json(path)
    out: dict[str, dict] = {}
    for report in doc.get("per_report") or []:
        for claim in report.get("claims") or []:
            disp = claim.get("dispositions") or {}
            removed = {k: v for k, v in disp.items()
                       if k in _DISPOSITION_LABEL and v}
            if removed:
                out[claim["sid"]] = removed
    return out


def _dc5_clause(removed: dict[str, int]) -> str:
    parts = [f"{n} {_DISPOSITION_LABEL[k]}"
             for k, n in sorted(removed.items(), key=lambda kv: -kv[1])]
    total = sum(removed.values())
    return (f"the Phase-2 dry run projected the prior pack losing {total} "
            f"evidence item(s) under the corrected rules ("
            + ", ".join(parts) + ")")


def build_reason(change: dict, dispositions: dict[str, dict],
                 extra_clauses: Optional[dict[str, str]] = None,
                 head_date: str = REBUILD_DATE,
                 head_generation: str = GENERATION) -> str:
    """Factual, specific where derivable; the approved generic line otherwise.

    ``extra_clauses`` (sid → clause) is how the adjudication wave states the
    MECHANISM that put a claim in front of the panel — which ratified rule
    released it, or that the owner named it. It is appended as another factual
    clause, never as narrative: the S-8 constraint on this generator is that it
    assembles facts and leaves the lineage paragraph to the owner."""
    clauses = []
    cause = _CLASS_CAUSE.get(change["change_class"], "")
    # "decided" in the diff's vocabulary includes panel UNVERIFIABLE (the panel
    # ruled that the claim cannot be checked) — distinct from a gate-forced
    # withholding. Saying "a different substantive verdict" for a move into or
    # out of panel-UNVERIFIABLE would misdescribe it, so name it precisely.
    if change["change_class"] == "decided_to_decided_changed":
        old_u = str(change["old_label"]).strip().upper() == "UNVERIFIABLE"
        new_u = str(change["new_label"]).strip().upper() == "UNVERIFIABLE"
        if new_u:
            cause = ("the panel itself returned Unverifiable on the rebuilt "
                     "evidence pack (a panel ruling, not an evidence-gate "
                     "withholding)")
        elif old_u:
            cause = ("the panel reached a substantive verdict on the rebuilt "
                     "evidence pack where it previously ruled the claim "
                     "unverifiable")
    extra = (extra_clauses or {}).get(change["sid"])
    if extra:
        clauses.append(extra)
    if cause:
        clauses.append(cause)
    removed = dispositions.get(change["sid"])
    if removed:
        clauses.append(_dc5_clause(removed))
    # The one claim whose reason is RATIFIED rather than derived. It is not in
    # the wave, but if it ever reaches an entry the ratified rationale travels
    # with it instead of being re-invented from the change class.
    if change["sid"] == BECKSTROM_SID:
        clauses.append(BECKSTROM_RATIONALE)
    if not clauses:
        return GENERIC_CAUSE[0].upper() + GENERIC_CAUSE[1:] + "."
    head = (f"Re-adjudicated on the unified {head_generation} pipeline "
            f"({head_date}): ")
    return head + "; ".join(clauses) + "."


def ledger_verdicts(change: dict) -> tuple[Optional[str], Optional[str], str]:
    """(old, new, reason-if-unrepresentable) in the ledger's vocabulary.

    ``data/corrections.json``'s loader accepts only TRUE/FALSE/MISLEADING/
    UNVERIFIABLE and rejects an entry whose old and new verdict are equal. Two
    real transitions in this corpus fall outside that vocabulary:

    * anything touching "Models split" — the ledger has no split label;
    * panel UNVERIFIABLE → gate-forced UNVERIFIABLE — a real provenance change
      (the claim is now withheld by the evidence gate rather than by the
      panel) that collapses to the same published badge.

    Those changes are still reported in the DC-6 review; they just cannot be
    expressed as a public correction, which is a statement about the ledger's
    vocabulary, not a reason to drop them."""
    old = str(change["old_label"]).strip().upper()
    new = str(change["new_label"]).strip().upper()
    if old == "GATED-UNVERIFIABLE":
        old = "UNVERIFIABLE"
    if new == "GATED-UNVERIFIABLE":
        new = "UNVERIFIABLE"
    if old not in LEDGER_VERDICTS or new not in LEDGER_VERDICTS:
        return None, None, ("verdict outside the corrections vocabulary "
                            f"({change['old_verdict']} → {change['new_verdict']})")
    if old == new:
        return None, None, ("published badge unchanged — provenance-only move "
                            f"({change['old_label']} → {change['new_label']})")
    return old, new, ""


def dropped_rows(diffs: Iterable[dict], runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Rows the rebuild DROPPED: present in the old artifact, gone from the new.

    The verdict diff is built over the sids in the NEW artifact, so a dropped
    row is invisible to every count derived from it — which is exactly how the
    530-vs-529 discrepancy survived as long as it did. This walks the old
    artifact instead.

    A dropped row is an ORPHAN when the old artifact had no claim record for
    it. trump_2026:0311 is the corpus's one orphan: a row with no claim, which
    the published claims.json still carries and the site still renders as
    "(claim text unavailable)". Dropping it removes a placeholder, not a
    fact-check — but it MOVES A PUBLISHED COUNT, so it is ledgered."""
    out: list[dict] = []
    for d in diffs:
        speech_id = d["speech_id"]
        old = load_run(d.get("rebuild_of", ""), runs_dir) or {}
        new = load_run(d.get("new_run_id", ""), runs_dir) or {}
        if not old or not new:
            continue
        new_sids = {r.get("sid") for r in new.get("rows") or []}
        old_claim_sids = {c.get("sid") for c in old.get("claims") or []}
        for r in old.get("rows") or []:
            sid = r.get("sid")
            if sid in new_sids:
                continue
            orphan = sid not in old_claim_sids
            out.append({
                "sid": sid,
                "speech_id": speech_id,
                "old_label": outcome_label_local(r),
                "kind": "orphan_row" if orphan else "dropped_claim",
                "reason": (
                    "Orphan row: the prior artifact carried a verdict row with "
                    "NO matching claim record, so the published claims.json "
                    "shows it as \"(claim text unavailable)\". The rebuild "
                    "emits rows only for real claims, so it is gone. This is a "
                    "count correction, not a verdict correction — no reader "
                    "ever saw a fact-check here."
                    if orphan else
                    "The rebuild produced no row for this sid; it is absent "
                    "from the republished corpus."),
                "published_effect": (
                    "published row count 530 → 529; claim count unchanged at 529"
                    if orphan else "published row count reduced by one"),
            })
    out.sort(key=lambda e: e["sid"])
    return out


def outcome_label_local(row: dict) -> str:
    """``phase3_rebuild.outcome_label`` without importing the runner (which
    carries spend guards). Same three rules: gate-forced UV first, then the
    verdict, then split/no-verdict."""
    if (row.get("evidence_gate") or row.get("provenance_code") or "") \
            == "insufficient-qualifying-evidence":
        return "gated-UNVERIFIABLE"
    if row.get("verdict") is not None:
        return str(row["verdict"])
    return "Models split" if row.get("split") else "No verdict"


def canonical_counts(diffs: Iterable[dict], runs_dir: Path = RUNS_DIR,
                     site_root: Optional[Path] = None) -> dict:
    """THE canonical corpus count, with every exclusion named (A9).

    The record disagreed with itself: 529 in the handoff, 530 in commit
    e268dec's DC-4' tally, 183 vs 182 Trump rows. All three were true of
    something, which is why nobody could close it. Measured here, once:

    * old artifacts: 529 claims, 530 rows — the extra row is trump_2026:0311,
      an orphan with no claim record;
    * new artifacts: 529 claims, 529 rows — no orphans;
    * published claims.json: 530 records, of which exactly one renders
      "(claim text unavailable)".

    So: **529 claims is canonical.** 530 was always 529 + 1 orphan row."""
    site_root = Path(site_root) if site_root is not None else REPO / "site-pca"
    old_claims = old_rows = new_claims = new_rows = 0
    old_orphans: list[str] = []
    new_orphans: list[str] = []
    for d in diffs:
        old = load_run(d.get("rebuild_of", ""), runs_dir) or {}
        new = load_run(d.get("new_run_id", ""), runs_dir) or {}
        for run, orphans in ((old, old_orphans), (new, new_orphans)):
            claim_sids = {c.get("sid") for c in run.get("claims") or []}
            orphans += [r.get("sid") for r in run.get("rows") or []
                        if r.get("sid") not in claim_sids]
        old_claims += len(old.get("claims") or [])
        old_rows += len(old.get("rows") or [])
        new_claims += len(new.get("claims") or [])
        new_rows += len(new.get("rows") or [])

    published = _site_claims(site_root)
    placeholders = [c.get("id") for c in published
                    if "(claim text unavailable)" in (c.get("claim_text") or "")]
    return {
        "canonical_claims": new_claims,
        "statement": (
            f"The corpus is {new_claims} claims. The published "
            f"{len(published)} is {new_claims} + {len(placeholders)} orphan "
            f"row ({', '.join(sorted(old_orphans)) or 'none'}) — a verdict row "
            "with no claim record, rendered as \"(claim text unavailable)\". "
            "The rebuilt artifacts carry no orphans."),
        "old": {"claims": old_claims, "rows": old_rows,
                "orphan_rows": sorted(old_orphans)},
        "new": {"claims": new_claims, "rows": new_rows,
                "orphan_rows": sorted(new_orphans)},
        "published": {"records": len(published),
                      "placeholder_records": len(placeholders),
                      "placeholder_ids": placeholders},
        "named_exclusions": [
            {"sid": sid,
             "why": "row with no claim record (orphan); never a fact-check"}
            for sid in sorted(old_orphans)],
    }


def correction_entries(changes: Iterable[dict],
                       dispositions: Optional[dict[str, dict]] = None,
                       date: str = REBUILD_DATE,
                       dropped: Optional[list[dict]] = None,
                       extra_clauses: Optional[dict[str, str]] = None,
                       source: str = CORRECTION_SOURCE,
                       generation: str = GENERATION) -> dict:
    """One record per changed verdict, split into ledger-eligible entries and
    changes the ledger schema cannot express — plus ``dropped_rows``, the
    rows that left the corpus entirely (A9). A drop has no old→new verdict
    pair, so it can never be a ledger entry; it is recorded here because it
    moves a PUBLISHED COUNT and the count is what the record disagreed about."""
    dispositions = dispositions or {}
    entries: list[dict] = []
    non_ledger: list[dict] = []
    for change in changes:
        old, new, blocked = ledger_verdicts(change)
        reason = build_reason(change, dispositions, extra_clauses,
                              head_date=date, head_generation=generation)
        if blocked:
            non_ledger.append({
                "sid": change["sid"],
                "speech_id": change["speech_id"],
                "claim_text": change.get("claim_text_full")
                or change.get("claim_text", ""),
                "old_label": change["old_label"],
                "new_label": change["new_label"],
                "old_verdict": change["old_verdict"],
                "new_verdict": change["new_verdict"],
                "change_class": change["change_class"],
                "reason": reason,
                "excluded_because": blocked,
            })
            continue
        entries.append({
            "sid": change["sid"],
            "speech_id": change["speech_id"],
            # The claim's own words, carried on the entry (S-8): an entry that
            # states only "TRUE → MISLEADING" makes the reader look the claim
            # up somewhere else to know what moved. It is a FACT off the
            # artifact, not a characterisation.
            "claim_text": change.get("claim_text_full")
            or change.get("claim_text", ""),
            "old_verdict": old,
            "new_verdict": new,
            "reason": reason,
            "date": date,
            "source": source,
        })
    entries.sort(key=lambda e: e["sid"])
    non_ledger.sort(key=lambda e: e["sid"])
    return {
        "schema": "truthbot-dc6-corrections-entries v1",
        "generated": date,
        "generation": generation,
        "usage": ("PUBLICATION RECORD, not an input to apply_to_artifact. The "
                  "rebuilt artifacts already carry the new verdicts, so "
                  "applying these entries would fail closed on the old_verdict "
                  "check. Render the rebuilt corpus with --corrections skip."),
        "changed_total": len(entries) + len(non_ledger),
        "ledger_eligible": len(entries),
        "not_ledger_representable": len(non_ledger),
        "dropped_total": len(dropped or []),
        "entries": entries,
        "non_ledger_changes": non_ledger,
        "dropped_rows": list(dropped or []),
    }


# ── the adjudication wave's corrections ───────────────────────────────────
def wave_extra_clauses(diffs: Iterable[dict]) -> dict[str, str]:
    """sid → the mechanical clause naming why the wave adjudicated that claim.

    Read off the wave diffs' own ``reasons`` map rather than recomputed, so the
    clause an entry carries and the reason the runner recorded when it spent
    the money are the same statement."""
    out: dict[str, str] = {}
    for d in diffs:
        for sid, reason in (d.get("reasons") or {}).items():
            clause = WAVE_CLAUSE_BY_REASON.get(reason)
            if clause:
                out[sid] = clause
    return out


def build_wave_corrections(diff_dir: Path = DIFF_DIR,
                           runs_dir: Path = RUNS_DIR) -> tuple[dict, list[dict]]:
    """Corrections entries for the adjudication wave — (entries_doc, changes).

    Deliberately NOT the full DC-6 review: era parity, anecdote parity and the
    badge diff are corpus-wide measurements, and running them over a 29-claim
    diff would produce numbers that look corpus-wide and are not.

    The DC-5 dispositions are also deliberately absent. They project what the
    Phase-2 dry run expected the OLD packs to lose under the corrected rules —
    a statement about retrieval, and this wave did no retrieval. Quoting them
    here would attach a projection to a change it did not cause."""
    diffs = load_diffs(diff_dir, WAVE_DIFF_GLOB)
    changes = changed_claims(diffs, runs_dir)
    entries_doc = correction_entries(
        changes, dispositions={}, date=WAVE_DATE,
        extra_clauses=wave_extra_clauses(diffs),
        source=WAVE_SOURCE, generation=GENERATION)
    entries_doc["usage"] = (
        "PUBLICATION RECORD for the adjudication wave, not an input to "
        "apply_to_artifact — the wave artifacts already carry these verdicts. "
        "FACTS ONLY (S-8): sid, claim text, old verdict, new verdict and the "
        "mechanical reason. The lineage paragraph is the owner's to write and "
        "is deliberately absent.")
    entries_doc["wave"] = {
        "date": WAVE_DATE,
        "speeches": [d["speech_id"] for d in diffs],
        "claims_adjudicated": sum(len(d.get("wave_sids") or []) for d in diffs),
        "claims_changed": len(changes),
    }
    return entries_doc, changes


# ── the 2026-08-10 rulings pass ───────────────────────────────────────────
def load_mechanism(path: Path = RULINGS_MECHANISM_PATH) -> dict[str, str]:
    """sid → the MEASURED mechanism that withholds it.

    Read from the artifact ``apply_wave_rulings.py`` writes, never recomputed
    here: the attribution the ledger states and the attribution the applier
    acted on have to be the same measurement, or an entry can describe a cause
    that did not produce the change it is reporting."""
    path = Path(path)
    if not path.exists():
        return {}
    return dict((_read_json(path).get("mechanism") or {}))


def rulings_extra_clauses(diffs: Iterable[dict],
                          mechanism: dict[str, str]) -> dict[str, str]:
    """sid → the mechanical clause for what this pass did to that claim.

    Three kinds of change, and they are told apart by what the diff shows
    rather than by a list: a claim whose mechanism is known was WITHHELD; a
    claim whose outcome did not move was touched at the provenance level only
    (rationale re-emit and/or coherence annotation), which the ledger must
    still record because the published page changes."""
    out: dict[str, str] = {}
    for d in diffs:
        moved = {e["sid"] for e in d.get("per_sid") or []
                 if e.get("category") != "unchanged"}
        for sid in d.get("ruling_sids") or []:
            mech = mechanism.get(sid) or (d.get("mechanism") or {}).get(sid) or ""
            if sid in moved and mech in RULINGS_MECHANISM_CLAUSE:
                out[sid] = RULINGS_MECHANISM_CLAUSE[mech]
            elif sid not in moved:
                out[sid] = RULINGS_RATIONALE_CLAUSE
    return out


def rulings_provenance_changes(diffs: Iterable[dict],
                               runs_dir: Path = RUNS_DIR) -> list[dict]:
    """The rows this pass changed WITHOUT changing the published verdict.

    ``changed_claims`` walks the verdict diff, so a claim whose badge did not
    move is invisible to it — and both R-3 changes (an adopted rationale, a
    coherence annotation) are exactly that shape. They are still publication
    changes: the page a reader sees gains a reason, or a disclosure. The
    ledger's schema cannot express them as corrections (no old→new verdict
    pair), so they are reported here, in the same place, rather than dropped."""
    out: list[dict] = []
    for d in diffs:
        new_run = load_run(d.get("new_run_id", ""), runs_dir) or {}
        rows = {r.get("sid"): r for r in (new_run.get("rows") or [])}
        texts = {c.get("sid"): (c.get("text") or "")
                 for c in (new_run.get("claims") or [])}
        moved = {e["sid"] for e in d.get("per_sid") or []
                 if e.get("category") != "unchanged"}
        for sid in d.get("ruling_sids") or []:
            if sid in moved:
                continue
            row = rows.get(sid) or {}
            kinds = []
            if row.get("rationale_provenance"):
                kinds.append("rationale re-emitted from stored panel output")
            if str(row.get("coherence_note") or "").strip():
                kinds.append("adjacent-claim coherence annotation added")
            if not kinds:
                continue
            out.append({
                "sid": sid,
                "speech_id": d["speech_id"],
                "claim": texts.get(sid, "")[:140],
                "verdict": row.get("verdict"),
                "kinds": kinds,
                "rationale_provenance": row.get("rationale_provenance") or {},
                "excluded_because": (
                    "published verdict unchanged — a provenance change, which "
                    "data/corrections.json's schema (old verdict → new verdict) "
                    "cannot express"),
            })
    return out


def build_rulings_corrections(diff_dir: Path = DIFF_DIR,
                              runs_dir: Path = RUNS_DIR) -> tuple[dict, list[dict]]:
    """Corrections entries for the 2026-08-10 rulings pass — (doc, changes).

    Same shape and same S-8 constraint as the wave's: facts per claim — sid,
    claim text, old verdict, new verdict, and the mechanical reason — with the
    lineage paragraph deliberately absent for the owner to write.

    What is new is that the mechanical reason is PER CLAIM. The wave released
    claims for one reason and could share a clause; this pass withholds them
    for three different ones (the re-score alone, D15, or the two ratified
    rules composing), and a ledger that said "the ratified rules" for all 65
    would be attributing 26 of them to rules that had nothing to do with it."""
    diffs = load_diffs(diff_dir, RULINGS_DIFF_GLOB)
    mechanism = load_mechanism(diff_dir / RULINGS_MECHANISM_PATH.name)
    changes = changed_claims(diffs, runs_dir)
    entries_doc = correction_entries(
        changes, dispositions={}, date=RULINGS_DATE,
        extra_clauses=rulings_extra_clauses(diffs, mechanism),
        source=RULINGS_SOURCE, generation=GENERATION)
    entries_doc["usage"] = (
        "PUBLICATION RECORD for the 2026-08-10 wave rulings, not an input to "
        "apply_to_artifact — the rulings artifacts already carry these "
        "outcomes. FACTS ONLY (S-8): sid, claim text, old verdict, new verdict "
        "and the mechanical reason, with the MECHANISM attributed per claim "
        "from the measured attribution in deferred_gated_mechanism.json. The "
        "lineage paragraph is the owner's to write and is deliberately absent.")
    by_mech: dict[str, int] = {}
    for change in changes:
        m = mechanism.get(change["sid"], "")
        if m:
            by_mech[m] = by_mech.get(m, 0) + 1
    provenance = rulings_provenance_changes(diffs, runs_dir)
    entries_doc["provenance_changes"] = provenance
    entries_doc["rulings"] = {
        "date": RULINGS_DATE,
        "speeches": [d["speech_id"] for d in diffs],
        "claims_touched": sum(len(d.get("ruling_sids") or []) for d in diffs),
        "claims_changed": len(changes),
        "by_mechanism": by_mech,
        "provenance_only_changes": len(provenance),
    }
    return entries_doc, changes


# ── ledger clean-slate reset ──────────────────────────────────────────────
def proposed_ledger(current: dict, entries: list[dict],
                    archive_name: str, date: str = REBUILD_DATE,
                    n_non_ledger: int = 0) -> dict:
    """The PROPOSED post-reset data/corrections.json.

    Clean slate: every pre-existing entry and note is archived to
    ``archive_name`` and the live ledger carries the rebuild entries plus one
    factual note. The note states only what is verifiable from this package —
    it makes no claim about which way any individual verdict moved before
    2026-08-06."""
    n_old = len(current.get("entries") or [])
    n_notes = len(current.get("notes") or [])
    note = (
        f"On {date} the entire five-speech corpus was re-adjudicated from "
        f"scratch on the unified {GENERATION} pipeline (corrected source-tier "
        "registry, fail-closed era gate, and per-claim cap on political "
        f"sources). {len(entries)} claims are published with a verdict that "
        "differs from the previously published run and are listed below. The "
        f"{n_old} correction entr{'y' if n_old == 1 else 'ies'} and "
        f"{n_notes} note{'' if n_notes == 1 else 's'} issued before that date "
        "described the superseded run; they are archived verbatim in "
        f"{archive_name} and no longer describe what this site publishes."
        + (f" A further {n_non_ledger} claims changed in a way this ledger's "
           "vocabulary cannot express (model-split transitions, and claims "
           "withheld by the evidence gate rather than by the panel); they are "
           "itemised in the DC-6 review package."
           if n_non_ledger else "")
    )
    return {
        "schema": current.get("schema", "truthbot-corrections v1"),
        "notes": [{"date": date, "text": note}],
        "entries": list(entries),
    }


# ── badge diff ────────────────────────────────────────────────────────────
def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _site_claims(site_root: Path) -> list[dict]:
    """(speaker, claim_text, verdict) triples out of a rendered site tree."""
    site_root = Path(site_root)
    claims = _read_json(site_root / "data" / "claims.json")
    reports = _read_json(site_root / "data" / "reports.json")
    speaker_by_report = {r.get("id") or r.get("report_id"): r.get("speaker", "")
                         for r in reports}
    return [{
        "id": c.get("id", ""),
        "speaker": speaker_by_report.get(c.get("report_id"), ""),
        "claim_text": c.get("claim_text", ""),
        "verdict": c.get("consensus_verdict", ""),
    } for c in claims]


def badge_diff(old_claims: list[dict], new_claims: list[dict]) -> dict:
    """Old vs new published badges, keyed on (speaker, normalised claim text).

    Claim ids are minted per render (``uuid4`` in rerender_pca_site.py), so an
    id-keyed diff matches nothing and reports "everything changed" — vacuous.
    ``id_overlap`` proves that in-band: it is the count of ids common to both
    sites, and it should be 0 for two independent renders."""
    def key(c):
        return (c["speaker"], _norm_text(c["claim_text"]))

    old_by, new_by = {}, {}
    dup_old, dup_new = 0, 0
    for c in old_claims:
        if key(c) in old_by:
            dup_old += 1
        old_by[key(c)] = c
    for c in new_claims:
        if key(c) in new_by:
            dup_new += 1
        new_by[key(c)] = c

    matched = sorted(set(old_by) & set(new_by))
    changes = []
    for k in matched:
        if old_by[k]["verdict"] != new_by[k]["verdict"]:
            changes.append({
                "speaker": k[0],
                "claim_text": _truncate(new_by[k]["claim_text"], 140),
                "old_verdict": old_by[k]["verdict"],
                "new_verdict": new_by[k]["verdict"],
            })
    only_old = [{"speaker": k[0], "claim_text": _truncate(old_by[k]["claim_text"], 140),
                 "verdict": old_by[k]["verdict"]}
                for k in sorted(set(old_by) - set(new_by))]
    only_new = [{"speaker": k[0], "claim_text": _truncate(new_by[k]["claim_text"], 140),
                 "verdict": new_by[k]["verdict"]}
                for k in sorted(set(new_by) - set(old_by))]
    id_overlap = len({c["id"] for c in old_claims} & {c["id"] for c in new_claims})
    return {
        "keyed_on": "(speaker, normalised claim_text)",
        "old_claims": len(old_claims),
        "new_claims": len(new_claims),
        "matched": len(matched),
        "only_old": len(only_old),
        "only_new": len(only_new),
        "verdict_changes": len(changes),
        "duplicate_keys_old": dup_old,
        "duplicate_keys_new": dup_new,
        "id_overlap": id_overlap,
        "id_keying_would_be_vacuous": id_overlap == 0,
        "changes": changes,
        "only_old_claims": only_old,
        "only_new_claims": only_new,
    }


def renderer_selection(runs_dir: Path = RUNS_DIR) -> dict[str, str]:
    """speech_id → run id the renderer WILL choose, by its own rule.

    Mirrors ``scripts/rerender_pca_site.py``: newest-mtime evidence-bearing
    artifact per speech_id. Recomputing it here is how we assert the staged
    render actually consumed the five rebuilds rather than trusting the log."""
    latest: dict[str, str] = {}
    for path in sorted(Path(runs_dir).glob("*.json"),
                       key=lambda p: p.stat().st_mtime):
        try:
            doc = _read_json(path)
        except (ValueError, OSError):
            continue
        if "evidence" not in doc:
            continue
        speech = (doc.get("meta") or {}).get("speech_id") or path.stem
        latest[speech] = path.stem
    return latest


def reconcile(badge: dict, agg: dict, entries_doc: dict) -> dict:
    """Does the rendered badge diff agree with the per-speech verdict diffs?

    They count different things where the published badge collapses two
    contract labels onto one word (panel UNVERIFIABLE and gate-forced
    UNVERIFIABLE both render "Unverifiable"), so the expected identity is::

        badge verdict_changes == changed_total - badge-invisible changes

    where the badge-invisible set is exactly the provenance-only moves the
    corrections builder already had to set aside."""
    changed_total = agg["corpus"]["changed_total"]
    invisible = [c for c in entries_doc["non_ledger_changes"]
                 if c["old_verdict"] == c["new_verdict"]]
    expected = changed_total - len(invisible)
    return {
        "per_speech_changed_total": changed_total,
        "badge_invisible_changes": len(invisible),
        "badge_invisible_sids": [c["sid"] for c in invisible],
        "expected_badge_changes": expected,
        "actual_badge_changes": badge["verdict_changes"],
        "agree": expected == badge["verdict_changes"],
        "delta": badge["verdict_changes"] - expected,
    }


# ── spend ─────────────────────────────────────────────────────────────────
def spend_table(diffs: Iterable[dict]) -> dict:
    per_speech = []
    proxy = offproxy = stated = 0.0
    for d in diffs:
        sid = d["speech_id"]
        s = SPEND.get(sid, {})
        p, o = float(s.get("proxy_usd", 0.0)), float(s.get("offproxy_usd", 0.0))
        st = float(s.get("stated_usd", 0.0))
        proxy += p
        offproxy += o
        stated += st
        per_speech.append({
            "speech_id": sid,
            "speaker": SPEAKERS.get(sid, ""),
            "old_run_id": d.get("rebuild_of", ""),
            "new_run_id": d.get("new_run_id", ""),
            "claims": d["n_compared"],
            "legs": s.get("legs", 1),
            "proxy_usd_ledger_true": round(p, 4),
            "offproxy_usd_ESTIMATE": round(o, 4),
            "log_total_usd": round(p + o, 4),
            "brief_stated_usd": st,
            "delta_vs_brief": round((p + o) - st, 4),
            "note": s.get("note", ""),
        })
    log_total = proxy + offproxy
    discrepancies = [r for r in per_speech if abs(r["delta_vs_brief"]) >= 0.01]
    return {
        "per_speech": per_speech,
        "shape_backfill_usd_ledger_true": SHAPE_BACKFILL_USD,
        "proxy_usd_ledger_true": round(proxy + SHAPE_BACKFILL_USD, 4),
        "offproxy_usd_ESTIMATE": round(offproxy, 4),
        "log_derived_total_usd": round(log_total + SHAPE_BACKFILL_USD, 4),
        "brief_stated_per_speech_sum_usd": round(stated, 2),
        "brief_stated_total_usd": STATED_TOTAL_USD,
        "cost_basis": {
            "proxy": "ledger-true — billed by the LiteLLM proxy key",
            "off_proxy": ("ESTIMATE — models called outside the proxy, costed "
                          "from token counts at published list rates"),
            "mixed": True,
            "offproxy_is_lower_bound": True,
            "resumed_speeches": list(RESUMED_SPEECHES),
            "disclosure": SPEND_BASIS_DISCLOSURE,
        },
        #: Top-level too: the DC-6' final ledger reads this key directly, and a
        #: disclosure nested three levels down is a disclosure waiting to be
        #: dropped by whatever assembles the next document.
        "basis_disclosure": SPEND_BASIS_DISCLOSURE,
        "discrepancies": [
            f"{r['speech_id']}: run logs total ${r['log_total_usd']:.4f} vs "
            f"${r['brief_stated_usd']:.2f} stated in the DC-6 brief "
            f"({r['delta_vs_brief']:+.4f})" for r in discrepancies
        ] + ([
            f"corpus: run logs total ${log_total + SHAPE_BACKFILL_USD:.4f} "
            f"(incl. ${SHAPE_BACKFILL_USD:.2f} shape backfill) vs "
            f"~${STATED_TOTAL_USD:.2f} stated in the brief; the brief's "
            f"per-speech figures themselves sum to ${stated:.2f} BEFORE the "
            "backfill, so the stated total appears to double-count the "
            "backfill as already included"
        ] if abs((log_total + SHAPE_BACKFILL_USD) - STATED_TOTAL_USD) >= 0.01 else []),
    }


# ── markdown ──────────────────────────────────────────────────────────────
def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def render_markdown(review: dict) -> str:
    agg, dist = review["aggregate"], review["distributions"]
    L: list[str] = []
    A = L.append
    A("# DC-6 review package — Phase-3 rebuild vs published corpus")
    A("")
    A(f"Generated `{review['generated']}` · generation `{review['generation']}` · "
      "$0 (derived from artifacts on disk; no model calls).")
    A("")
    A("**This is a review surface, not a publish.** The rebuilt artifacts are "
      "committed with `published: false`; `site-pca/` still serves the old runs. "
      "Read the changed-claim tables below and decide.")
    A("")

    # Headline
    c = agg["corpus"]
    A("## 1. Headline")
    A("")
    A(f"- Claims re-adjudicated: **{c['claims']}**")
    A(f"- Unchanged: **{c['unchanged']}** · Changed: **{c['changed_total']}** "
      f"({c['changed_total'] / c['claims'] * 100:.1f}%)")
    A("- Decided → decided (flipped between substantive verdicts): "
      f"**{c['decided_to_decided_changed']}**")
    A(f"- Newly gated (decided → withheld): **{c['newly_gated']}**")
    A(f"- Newly decided (withheld → decided): **{c['newly_decided']}**")
    A(f"- Split changes: **{c['split_changes']}** · Other: **{c['other']}**")
    A(f"- Gate-forced Unverifiable in the new runs: **{c['gate_forced_new']}**")
    A("")
    corpus = dist["corpus"]
    A(f"- Corpus decided-rate: **{_pct(corpus['old_decided']['rate'])}** "
      f"({corpus['old_decided']['decided']}/{corpus['old_decided']['total']}) → "
      f"**{_pct(corpus['new_decided']['rate'])}** "
      f"({corpus['new_decided']['decided']}/{corpus['new_decided']['total']}), "
      f"{corpus['decided_rate_delta'] * 100:+.1f} pts")
    A("")

    # Flags
    if review.get("flags"):
        A("## 2. Flags — read these before anything else")
        A("")
        for flag in review["flags"]:
            A(f"- **{flag}**")
        A("")
    else:
        A("## 2. Flags")
        A("")
        A("- none")
        A("")

    # Per-speech
    A("## 3. Per-speech change counts")
    A("")
    rows = [[r["speech_id"], r["speaker"], r["claims"], r["unchanged"],
             r["decided_to_decided_changed"], r["newly_gated"],
             r["newly_decided"], r["split_changes"], r["changed_total"]]
            for r in (agg["per_speech"][s] for s in SPEECH_ORDER
                      if s in agg["per_speech"])]
    rows.append(["**corpus**", "", c["claims"], c["unchanged"],
                 c["decided_to_decided_changed"], c["newly_gated"],
                 c["newly_decided"], c["split_changes"], c["changed_total"]])
    L += _table(["speech", "speaker", "claims", "unchanged", "dec→dec",
                 "newly gated", "newly decided", "splits", "changed"], rows)
    A("")

    # Distribution
    A("## 4. Verdict distribution — old vs new")
    A("")
    A("Old = the published run's rows; new = the rebuilt run's rows. "
      "`decided` = every claim that is not Unverifiable and not a model split.")
    A("")
    for sid in SPEECH_ORDER:
        if sid not in dist["per_speech"]:
            continue
        d = dist["per_speech"][sid]
        A(f"### {d['speaker']} — `{sid}`")
        A("")
        rows = [["old"] + [d["old"].get(v, 0) for v in DISPLAY_ORDER]
                + [f"{d['old_decided']['decided']}/{d['old_decided']['total']}",
                   _pct(d["old_decided"]["rate"])],
                ["new"] + [d["new"].get(v, 0) for v in DISPLAY_ORDER]
                + [f"{d['new_decided']['decided']}/{d['new_decided']['total']}",
                   _pct(d["new_decided"]["rate"])]]
        L += _table(["run"] + DISPLAY_ORDER + ["decided", "decided-rate"], rows)
        A("")
        A(f"decided-rate change: **{d['decided_rate_delta'] * 100:+.1f} pts**"
          + ("  ⚠ old and new denominators differ — see flags"
             if d["denominator_mismatch"] else ""))
        A("")
    A("### Corpus")
    A("")
    rows = [["old"] + [corpus["old"].get(v, 0) for v in DISPLAY_ORDER]
            + [f"{corpus['old_decided']['decided']}/{corpus['old_decided']['total']}",
               _pct(corpus["old_decided"]["rate"])],
            ["new"] + [corpus["new"].get(v, 0) for v in DISPLAY_ORDER]
            + [f"{corpus['new_decided']['decided']}/{corpus['new_decided']['total']}",
               _pct(corpus["new_decided"]["rate"])]]
    L += _table(["run"] + DISPLAY_ORDER + ["decided", "decided-rate"], rows)
    A("")
    A("**Era parity** — decided-rate, oldest speech to newest, old → new:")
    A("")
    parity = [f"`{sid}` {_pct(dist['per_speech'][sid]['old_decided']['rate'])} → "
              f"{_pct(dist['per_speech'][sid]['new_decided']['rate'])}"
              for sid in SPEECH_ORDER if sid in dist["per_speech"]]
    A("- " + "  ·  ".join(parity))
    A("")
    ep = dist["era_parity"]
    A(f"Spread across speeches (max − min decided-rate): "
      f"**{_pct(ep['old_spread']['spread'])}** old "
      f"(`{ep['old_spread']['max_speech']}` {_pct(ep['old_spread']['max'])} vs "
      f"`{ep['old_spread']['min_speech']}` {_pct(ep['old_spread']['min'])}) → "
      f"**{_pct(ep['new_spread']['spread'])}** new "
      f"(`{ep['new_spread']['max_speech']}` {_pct(ep['new_spread']['max'])} vs "
      f"`{ep['new_spread']['min_speech']}` {_pct(ep['new_spread']['min'])}), "
      f"{ep['spread_delta'] * 100:+.1f} pts — "
      f"**{'NARROWED' if ep['narrowed'] else 'WIDENED'}**.")
    A("")
    if not ep["narrowed"]:
        A("> The unified pipeline did NOT equalise decided-rates across eras; "
          "it spread them further apart. A reader should not read this rebuild "
          "as having produced era parity in outcome — it produced one "
          "methodology, applied uniformly, whose per-speech decided-rates "
          "differ more than before. Judge the publish on that basis.")
        A("")

    # ── A10: anecdote-adjusted parity ───────────────────────────────────────
    ap = review.get("anecdote_parity")
    if ap:
        A("### Anecdote-adjusted parity")
        A("")
        A("The raw decided-rate above counts every Unverifiable as a claim the "
          "pipeline failed to settle. For one genre that is the wrong reading: "
          f"a claim typed `{ap['anecdote_claim_type']}` is a private "
          "individual's story told from the stage, and it usually has no public "
          "record to check — Unverifiable is the EXPECTED outcome, not a gate "
          "failure. The site already says so in the footnote under every "
          "verdict bar; this section applies the same logic to the parity "
          "number, because anecdote counts differ enormously between these "
          "speeches and an unadjusted spread partly measures speechwriting "
          "style rather than methodology.")
        A("")
        A("Both bases are shown. The adjustment is an ARGUMENT — a reader who "
          "rejects it can read the raw column and ignore the rest.")
        A("")
        rows = []
        for sid in SPEECH_ORDER:
            r = ap["per_speech"].get(sid)
            if not r:
                continue
            rows.append([
                f"`{sid}`", str(r["claims_compared"]),
                f"{r['anecdotes']} ({_pct(r['anecdote_share'])})",
                f"{_pct(r['old_raw']['rate'])} → {_pct(r['new_raw']['rate'])}",
                f"{_pct(r['old_adjusted']['rate'])} → "
                f"{_pct(r['new_adjusted']['rate'])}",
                f"{(r['new_adjusted']['rate'] - r['new_raw']['rate']) * 100:+.1f} pts",
            ])
        c = ap["corpus"]
        rows.append([
            "**corpus**", str(c["new_raw"]["total"]),
            str(c["new_raw"]["total"] - c["new_adjusted"]["total"]),
            f"{_pct(c['old_raw']['rate'])} → {_pct(c['new_raw']['rate'])}",
            f"{_pct(c['old_adjusted']['rate'])} → {_pct(c['new_adjusted']['rate'])}",
            f"{(c['new_adjusted']['rate'] - c['new_raw']['rate']) * 100:+.1f} pts",
        ])
        L += _table(["speech", "compared", "anecdotes", "decided-rate raw "
                     "(old → new)", "anecdote-adjusted (old → new)",
                     "new: adj − raw"], rows)
        A("")
        for basis, label in (("raw", "Raw"), ("adjusted", "Anecdote-adjusted")):
            old_s = ap["spread"][f"old_{basis}"]
            new_s = ap["spread"][f"new_{basis}"]
            A(f"- **{label} spread** (max − min decided-rate across the five "
              f"speeches): **{_pct(old_s['spread'])}** old "
              f"(`{old_s['max_speech']}` {_pct(old_s['max'])} vs "
              f"`{old_s['min_speech']}` {_pct(old_s['min'])}) → "
              f"**{_pct(new_s['spread'])}** new "
              f"(`{new_s['max_speech']}` {_pct(new_s['max'])} vs "
              f"`{new_s['min_speech']}` {_pct(new_s['min'])}), "
              f"{ap['spread'][f'{basis}_spread_delta'] * 100:+.1f} pts — "
              f"**{'NARROWED' if ap['spread'][f'{basis}_narrowed'] else 'WIDENED'}**.")
        A("")
        if ap["spread"]["raw_narrowed"] == ap["spread"]["adjusted_narrowed"]:
            A("> Both bases agree on the direction, so the era-parity finding "
              "above does not depend on how anecdotes are counted.")
        else:
            A("> The two bases DISAGREE on direction. The era-parity finding "
              "above is therefore conditional on treating anecdote "
              "Unverifiables as gate failures — read both rows before drawing "
              "a conclusion.")
        A("")
        j = ap["join"]
        A(f"Provenance of the anecdote flag: **{j['from_artifact']}** claim(s) "
          f"carried `layer_a.claim_type` in the rebuilt artifact, "
          f"**{j['from_claims_json']}** were joined from the published "
          f"`claims.json` by (speaker, normalised claim text), and "
          f"**{j['unresolved']}** could not be resolved by either route"
          + (f" (`{'`, `'.join(j['unresolved_sids'][:10])}`)"
             if j["unresolved_sids"] else "")
          + ". Unresolved claims stay in the adjusted denominator as "
            "non-anecdotes — reported here rather than assumed away, because "
            "counting an unclassified claim as 'not an anecdote' is a guess "
            "that moves the number in a known direction.")
        A("")

    # Canonical count reconciliation (A9)
    cc = review.get("canonical_counts")
    if cc:
        A("## 4b. Canonical claim count")
        A("")
        A(cc["statement"])
        A("")
        L += _table(["basis", "claims", "rows", "orphan rows"], [
            ["old artifacts", str(cc["old"]["claims"]), str(cc["old"]["rows"]),
             ", ".join(cc["old"]["orphan_rows"]) or "—"],
            ["new artifacts (canonical)", str(cc["new"]["claims"]),
             str(cc["new"]["rows"]),
             ", ".join(cc["new"]["orphan_rows"]) or "—"],
            ["published claims.json", str(cc["published"]["records"]), "—",
             f"{cc['published']['placeholder_records']} placeholder record(s)"],
        ])
        A("")
        A("Named exclusions:")
        for ex in cc["named_exclusions"]:
            A(f"- `{ex['sid']}` — {ex['why']}")
        if not cc["named_exclusions"]:
            A("- none")
        A("")
        A("\"Decided\" in the parity metric means a substantive published "
          "ruling: True / Mostly True / Misleading / False. Unverifiable "
          "(panel OR gate-forced) and Models split are abstentions. The fold "
          "rules and the anecdote-adjusted variant are documented in "
          "`docs/run-schema.md`, which is written so an external reviewer can "
          "reproduce these counts without reading this packager.")
        A("")

    # Changed claims
    A("## 5. Every changed claim")
    A("")
    A("Ordered by consequence: verdicts that flipped between two substantive "
      "rulings first, then claims newly withheld, then claims newly decided, "
      "then split changes. Rationale is the NEW panel's reasoning.")
    A("")
    by_class: dict[str, list[dict]] = {}
    for change in review["changed_claims"]:
        by_class.setdefault(change["change_class"], []).append(change)
    for cls in CLASS_ORDER:
        items = by_class.get(cls) or []
        if not items:
            continue
        A(f"### {CLASS_TITLE[cls]} — {len(items)} claim(s)")
        A("")
        for sid in SPEECH_ORDER:
            speech_items = [i for i in items if i["speech_id"] == sid]
            if not speech_items:
                continue
            A(f"#### {SPEAKERS.get(sid, sid)} — `{sid}` ({len(speech_items)})")
            A("")
            for item in speech_items:
                A(f"- **`{item['sid']}`** · {item['old_verdict']} → "
                  f"**{item['new_verdict']}**")
                A(f"  - claim: {item['claim_text']}")
                A(f"  - new rationale: {item['rationale'] or '(none recorded)'}")
            A("")

    # Spend
    spend = review["spend"]
    A("## 6. Spend + provenance")
    A("")
    A("**Cost basis — read this before quoting any number below.** "
      + spend["basis_disclosure"])
    A("")
    rows = [[r["speech_id"], f"`{r['old_run_id'][:8]}` → `{r['new_run_id'][:8]}`",
             r["claims"], r["legs"], f"${r['proxy_usd_ledger_true']:.4f}",
             f"${r['offproxy_usd_ESTIMATE']:.4f}", f"${r['log_total_usd']:.4f}",
             f"${r['brief_stated_usd']:.2f}", f"{r['delta_vs_brief']:+.4f}"]
            for r in spend["per_speech"]]
    rows.append(["shape backfill", "(haiku sidecars)", "—", "—",
                 f"${spend['shape_backfill_usd_ledger_true']:.2f}", "$0.0000",
                 f"${spend['shape_backfill_usd_ledger_true']:.2f}", "—", "—"])
    rows.append(["**total**", "", "", "",
                 f"${spend['proxy_usd_ledger_true']:.4f}",
                 f"${spend['offproxy_usd_ESTIMATE']:.4f}",
                 f"${spend['log_derived_total_usd']:.4f}",
                 f"${spend['brief_stated_total_usd']:.2f}", ""])
    L += _table(["speech", "old run → new run", "claims", "legs",
                 "proxy (ledger-true)", "off-proxy (ESTIMATE)", "log total",
                 "brief stated", "Δ"], rows)
    A("")
    for note in (r for r in spend["per_speech"] if r["legs"] > 1):
        A(f"- `{note['speech_id']}`: {note['note']}")
    A("")
    if spend["discrepancies"]:
        A("**Spend discrepancies (not smoothed):**")
        A("")
        for d in spend["discrepancies"]:
            A(f"- {d}")
        A("")

    # Render + badge diff
    if review.get("render"):
        r = review["render"]
        A("## 7. Staged render + badge diff")
        A("")
        A(f"- site root: `{r['site_root']}` (staged; `site-pca/` untouched)")
        A("- artifacts picked by the renderer: "
          + ", ".join(f"`{a['speech_id']}`→`{a['run_id'][:8]}`"
                      for a in r["artifacts"]))
        A(f"- all five NEW artifacts selected: **{r['picked_all_new']}**")
        verdict = ("PASS — 0 violations" if not r["violations"]
                   else f"{len(r['violations'])} VIOLATION(S)")
        A(f"- `check_site(strict_buckets=True)`: **{verdict}**")
        for v in r["violations"]:
            A(f"  - {v}")
        A("")
        b = review["badge_diff"]
        A(f"- badge diff keyed on {b['keyed_on']}: matched **{b['matched']}**, "
          f"only-old **{b['only_old']}**, only-new **{b['only_new']}**, "
          f"verdict changes **{b['verdict_changes']}**")
        A(f"- id overlap between the two renders: **{b['id_overlap']}** — "
          f"id-keying would be vacuous: **{b['id_keying_would_be_vacuous']}**")
        for c_ in b["only_old_claims"]:
            A(f"  - only-old: {c_['speaker']} — {c_['claim_text']} [{c_['verdict']}]")
        for c_ in b["only_new_claims"]:
            A(f"  - only-new: {c_['speaker']} — {c_['claim_text']} [{c_['verdict']}]")
        A("")
        rec = review["reconciliation"]
        A(f"- reconciliation: per-speech diffs report {rec['per_speech_changed_total']} "
          f"changes; {rec['badge_invisible_changes']} of them are invisible on the "
          f"published badge (panel-Unverifiable → gate-forced-Unverifiable), so the "
          f"badge diff should show {rec['expected_badge_changes']} and shows "
          f"{rec['actual_badge_changes']} — **{'AGREE' if rec['agree'] else 'DISAGREE'}**")
        A("")

    # Corrections
    corr = review["corrections"]
    A("## 8. Proposed corrections ledger")
    A("")
    A(f"- changed verdicts: **{corr['changed_total']}**")
    A(f"- expressible as public corrections: **{corr['ledger_eligible']}**")
    A(f"- not expressible in the ledger vocabulary: **{corr['not_ledger_representable']}** "
      "(model-split transitions; and claims that moved from panel-Unverifiable "
      "to gate-forced-Unverifiable, which publish the same badge)")
    A(f"- archive target: `{corr['archive_path']}` "
      f"({corr['archived_entries']} entries + {corr['archived_notes']} note(s))")
    A(f"- proposed live ledger: {corr['proposed_entries']} entries + 1 note")
    A("")
    A("`data/corrections.json` is NOT modified by this script. The reset is "
      "applied at publish time, under the gate.")
    A("")
    A("**Publish mechanics:** these entries are a PUBLICATION RECORD of what "
      "the rebuild changed — they must not be re-applied to the artifacts. "
      "`apply_to_artifact` fails closed when an entry's `old_verdict` does not "
      "match the row, and the rebuilt rows already carry the NEW verdicts. "
      "Render with `--corrections skip`; the corrections page still publishes "
      "the full ledger and its note.")
    A("")
    return "\n".join(L) + "\n"


# ── driver ────────────────────────────────────────────────────────────────
def build_review(new_site: Optional[Path] = None,
                 old_site: Optional[Path] = None,
                 diff_dir: Path = DIFF_DIR,
                 runs_dir: Path = RUNS_DIR) -> tuple[dict, dict, dict]:
    diffs = load_diffs(diff_dir)
    agg = aggregate(diffs)
    dist = distributions(diffs)
    anecdote = anecdote_parity(diffs, runs_dir)
    cov = coverage(diffs, runs_dir)
    changes = changed_claims(diffs, runs_dir)
    dispositions = load_dc5_dispositions(diff_dir / "dc5_worksheet.json")
    dropped = dropped_rows(diffs, runs_dir)
    entries_doc = correction_entries(changes, dispositions, dropped=dropped)
    counts = canonical_counts(diffs, runs_dir, old_site)

    flags: list[str] = []
    ep = dist["era_parity"]
    if not ep["narrowed"]:
        flags.append(
            "era parity: the spread of decided-rates across the five speeches "
            f"WIDENED from {ep['old_spread']['spread'] * 100:.1f} pts to "
            f"{ep['new_spread']['spread'] * 100:.1f} pts "
            f"({ep['new_spread']['min_speech']} is now the least-decided at "
            f"{ep['new_spread']['min'] * 100:.1f}% vs "
            f"{ep['new_spread']['max_speech']} at "
            f"{ep['new_spread']['max'] * 100:.1f}%). The rebuild unified the "
            "METHOD; it did not equalise the OUTCOME")
    # A10: the era-parity verdict can differ between the two bases. When it
    # does, say so — a spread that only widens because one speech had more
    # guests is not the same finding as one that widens on the merits.
    asp = anecdote["spread"]
    if asp["raw_narrowed"] != asp["adjusted_narrowed"]:
        flags.append(
            "era parity flips between bases: the decided-rate spread "
            f"{'NARROWED' if asp['raw_narrowed'] else 'WIDENED'} raw "
            f"({asp['raw_spread_delta'] * 100:+.1f} pts) but "
            f"{'NARROWED' if asp['adjusted_narrowed'] else 'WIDENED'} with "
            f"personal anecdotes excluded "
            f"({asp['adjusted_spread_delta'] * 100:+.1f} pts) — the parity "
            "reading depends on whether you count anecdotes as gate failures")
    if anecdote["join"]["unresolved"]:
        flags.append(
            f"anecdote parity: {anecdote['join']['unresolved']} claim(s) "
            "carry no layer_a_claim_type in the artifact AND did not join to "
            "the published claims.json by (speaker, normalised text) — they "
            "are counted in the adjusted denominator as non-anecdotes, which "
            "is an assumption, not a measurement: "
            + ", ".join(anecdote["join"]["unresolved_sids"][:10]))
    for sid, row in anecdote["per_speech"].items():
        if not row["raw_matches_section4"]:
            flags.append(
                f"{sid}: the anecdote-parity denominator "
                f"({row['claims_compared']} compared sids) differs from "
                "section 4's tally — section 4 counts every row in each "
                "artifact, this section counts the sids the rebuild compared")

    for c in cov:
        if c["dropped_sids"]:
            orphans = [x["sid"] for x in c["dropped_detail"]
                       if not x["had_claim_record"]]
            detail = "; ".join(
                f"{x['sid']} (old verdict {x['old_verdict']}, "
                + ("ORPHAN — the published run had no claim record for it and "
                   "rendered '(claim text unavailable)'"
                   if not x["had_claim_record"] else
                   "had a claim record — real content loss")
                + ")" for x in c["dropped_detail"])
            flags.append(
                f"{c['speech_id']}: {len(c['dropped_sids'])} sid(s) present in "
                f"the published run ({c['old_rows']} rows) are ABSENT from the "
                f"rebuild ({c['new_rows']} rows) and were never re-adjudicated "
                f"— {detail}. Publishing changes that report's claim count from "
                f"{c['old_rows']} to {c['new_rows']}"
                + (f"; {len(orphans)} of the dropped sid(s) were orphan rows, so "
                   "the drop removes a broken card rather than losing a checked "
                   "claim" if orphans else ""))
        if c["added_sids"]:
            flags.append(f"{c['speech_id']}: {len(c['added_sids'])} sid(s) are "
                         f"new in the rebuild — {', '.join(c['added_sids'])}")

    review: dict[str, Any] = {
        "schema": "truthbot-dc6-review v1",
        "generated": REBUILD_DATE,
        "generation": GENERATION,
        "aggregate": agg,
        "distributions": dist,
        "anecdote_parity": anecdote,
        "coverage": cov,
        "canonical_counts": counts,
        "dropped_rows": dropped,
        "changed_claims": changes,
        "spend": spend_table(diffs),
    }

    if new_site is not None:
        from truthbot.publish.consistency import check_site
        new_claims = _site_claims(new_site)
        reports = _read_json(Path(new_site) / "data" / "reports.json")
        selected = renderer_selection(runs_dir)
        picked, wrong = [], []
        for d in diffs:
            expected = d.get("new_run_id", "")
            actual = selected.get(d["speech_id"], "")
            picked.append({"speech_id": d["speech_id"], "run_id": actual,
                           "expected_run_id": expected, "ok": actual == expected})
            if actual != expected:
                wrong.append(f"{d['speech_id']}: renderer selects "
                             f"{actual[:8] or '(none)'}, rebuild is {expected[:8]}")
        violations = [str(v) for v in check_site(Path(new_site), strict_buckets=True)]
        review["render"] = {
            "site_root": str(new_site),
            "reports": len(reports),
            "claims": len(new_claims),
            "artifacts": picked,
            "picked_all_new": not wrong,
            "selection_errors": wrong,
            "violations": violations,
        }
        flags += wrong
        if old_site is not None:
            badge = badge_diff(_site_claims(old_site), new_claims)
            review["badge_diff"] = badge
            rec = reconcile(badge, agg, entries_doc)
            review["reconciliation"] = rec
            if not rec["agree"]:
                flags.append(
                    f"badge diff ({rec['actual_badge_changes']}) and per-speech "
                    f"verdict diffs ({rec['expected_badge_changes']} expected "
                    f"after {rec['badge_invisible_changes']} badge-invisible "
                    f"move(s)) DISAGREE by {rec['delta']:+d} — investigate "
                    "before publishing")
            if badge["only_old"] or badge["only_new"]:
                flags.append(
                    f"badge diff: {badge['only_old']} claim(s) present only in "
                    f"the published site and {badge['only_new']} only in the "
                    "staged render")
            if violations:
                flags.append(f"check_site(strict_buckets=True): "
                             f"{len(violations)} violation(s)")

    flags += review["spend"]["discrepancies"]
    review["flags"] = flags
    return review, entries_doc, {"diffs": diffs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new-site", default=None,
                    help="staged render root (e.g. /tmp/dc6-site); enables the "
                         "consistency gate + badge diff")
    ap.add_argument("--old-site", default=str(REPO / "site-pca"),
                    help="published site root for the badge diff (read-only)")
    ap.add_argument("--out-dir", default=str(DIFF_DIR))
    ap.add_argument("--write-archive", action="store_true",
                    help=f"also write data/corrections-archive-{REBUILD_DATE}.json")
    ap.add_argument("--wave", action="store_true",
                    help=("read the adjudication wave's verdict diffs "
                          f"({WAVE_DIFF_GLOB}) and write ONLY "
                          "wave_corrections_entries.json — facts per changed "
                          "verdict. The corpus-wide review sections are "
                          "skipped: they would report a 29-claim slice as if "
                          "it were the corpus."))
    ap.add_argument("--rulings", action="store_true",
                    help=("read the 2026-08-10 rulings diffs "
                          f"({RULINGS_DIFF_GLOB}) and write ONLY "
                          "rulings_corrections_entries.json — facts per "
                          "changed verdict, with the withholding MECHANISM "
                          "attributed per claim."))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rulings:
        entries_doc, _changes = build_rulings_corrections()
        (out_dir / "rulings_corrections_entries.json").write_text(
            json.dumps(entries_doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"rulings corrections → {out_dir}/rulings_corrections_entries.json")
        r = entries_doc["rulings"]
        print(f"  {r['claims_touched']} claims touched across "
              f"{len(r['speeches'])} speech(es); {r['claims_changed']} changed")
        print(f"  mechanism: {json.dumps(r['by_mechanism'])}")
        print(f"  {entries_doc['ledger_eligible']} ledger-eligible, "
              f"{entries_doc['not_ledger_representable']} not representable, "
              f"{r['provenance_only_changes']} provenance-only")
        for e in entries_doc["entries"][:5]:
            print(f"    {e['sid']:<22} {e['old_verdict']} → {e['new_verdict']}")
        if len(entries_doc["entries"]) > 5:
            print(f"    … {len(entries_doc['entries']) - 5} more")
        for e in entries_doc["provenance_changes"]:
            print(f"    {e['sid']:<22} verdict unchanged — {', '.join(e['kinds'])}")
        return
    if args.wave:
        entries_doc, changes = build_wave_corrections()
        (out_dir / "wave_corrections_entries.json").write_text(
            json.dumps(entries_doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"wave corrections → {out_dir}/wave_corrections_entries.json")
        w = entries_doc["wave"]
        print(f"  {w['claims_adjudicated']} claims adjudicated across "
              f"{len(w['speeches'])} speech(es); {w['claims_changed']} changed")
        print(f"  {entries_doc['ledger_eligible']} ledger-eligible, "
              f"{entries_doc['not_ledger_representable']} not representable")
        for e in entries_doc["entries"]:
            print(f"    {e['sid']:<22} {e['old_verdict']} → {e['new_verdict']}")
        for e in entries_doc["non_ledger_changes"]:
            print(f"    {e['sid']:<22} {e['old_label']} → {e['new_label']}  "
                  f"({e['excluded_because']})")
        return
    new_site = Path(args.new_site) if args.new_site else None
    old_site = Path(args.old_site) if (args.old_site and new_site) else None

    review, entries_doc, _ = build_review(new_site, old_site)

    ledger_path = REPO / "data" / "corrections.json"
    current = _read_json(ledger_path)
    archive_name = f"data/corrections-archive-{REBUILD_DATE}.json"
    proposed = proposed_ledger(current, entries_doc["entries"], archive_name,
                               n_non_ledger=entries_doc["not_ledger_representable"])

    review["corrections"] = {
        "changed_total": entries_doc["changed_total"],
        "ledger_eligible": entries_doc["ledger_eligible"],
        "not_ledger_representable": entries_doc["not_ledger_representable"],
        "archive_path": archive_name,
        "archived_entries": len(current.get("entries") or []),
        "archived_notes": len(current.get("notes") or []),
        "proposed_entries": len(proposed["entries"]),
    }

    (out_dir / "dc6_corrections_entries.json").write_text(
        json.dumps(entries_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "dc6_corrections_ledger_proposed.json").write_text(
        json.dumps(proposed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "dc6_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "dc6_review.md").write_text(render_markdown(review), encoding="utf-8")

    if args.write_archive:
        (REPO / archive_name).write_text(
            json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"archived current ledger → {archive_name}")

    print(f"DC-6 package → {out_dir}/dc6_review.{{json,md}}, "
          f"dc6_corrections_entries.json, dc6_corrections_ledger_proposed.json")
    c = review["aggregate"]["corpus"]
    print(f"  {c['claims']} claims · {c['changed_total']} changed "
          f"({c['decided_to_decided_changed']} dec→dec, {c['newly_gated']} gated, "
          f"{c['newly_decided']} decided, {c['split_changes']} split)")
    print(f"  corrections: {entries_doc['ledger_eligible']} ledger-eligible, "
          f"{entries_doc['not_ledger_representable']} not representable")
    ap = review["anecdote_parity"]
    print(f"  decided-rate (new): raw {ap['corpus']['new_raw']['rate']:.1%} · "
          f"anecdote-adjusted {ap['corpus']['new_adjusted']['rate']:.1%} · "
          f"spread raw {ap['spread']['new_raw']['spread']:.1%} vs adjusted "
          f"{ap['spread']['new_adjusted']['spread']:.1%} · "
          f"{ap['join']['unresolved']} claim(s) unjoined")
    for flag in review["flags"]:
        print(f"  FLAG: {flag}")


if __name__ == "__main__":
    main()
