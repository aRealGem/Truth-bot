"""Build-time verification that every quantitative figure in site copy is
derived from ``data/*.json`` (remediation T0.8 / card P67.4).

The 2026-07-21 external audit found the landing page asserting "100% Model
Consensus" over reports whose recorded agreement was 47% and 78%, verdict
bars whose segments summed to fewer claims than the report contained, and
header chips computed on a different denominator than the headline two lines
below them. Each of those figures rendered from a *different* source (or a
hand-typed constant). This module re-derives the load-bearing figures from
``data/claims.json`` + ``data/reports.json`` and compares them against what
the HTML actually says; any mismatch is a build failure.

Scope: the checks cover the site's quantitative claim surfaces — index
program stats, per-report verdict bars (strict axis — the single published
presentation since remediation v2, 1.8), family rails, header
chips, headline ratios, the anecdote footnote — plus tagline guards for
wording that must stay off the site until later remediation phases restore
it with evidence (T0.5/T0.6), and a sweep for retired UI that must never
render again (R-1: the editorial-lens chip). Purely decorative numbers
(CSS, dates, pipeline version strings) are out of scope.

Usage::

    from truthbot.publish.consistency import check_site
    violations = check_site(Path("site-pca"))
    # empty list == consistent site

``scripts/rerender_pca_site.py`` runs this after every render and exits
non-zero on violations; ``tests/test_site_consistency.py`` runs it over the
committed ``site-pca/`` tree so hand-typed numbers cannot merge.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from truthbot.publish.aggregation import (COARSE_VERDICT_ORDER,
                                          TIER_LINE_ORDER,
                                          distribution_from_claims,
                                          family_verdict)

logger = logging.getLogger(__name__)

# Abstention buckets (kept for reference by external callers; the family
# math itself delegates to aggregation.family_verdict).
_ABSTAIN = {"Unverifiable", "Models split"}


def _fmt_pct(numerator: int, denominator: int) -> str:
    """Match site.py's ``format(x, '.0%')`` rendering."""
    return format(numerator / denominator, ".0%") if denominator else "0%"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _claims_for_report(claims: list[dict], report_id: str) -> list[dict]:
    return [c for c in claims if c.get("report_id") == report_id]


def _coarse_dist(report_claims: list[dict], axis: str) -> dict[str, int]:
    """Re-derive one axis's aggregate distribution from claims.json — the
    single bucketing every rendered breakdown must match (T0.2). Since
    remediation v2 (1.6) this DELEGATES to the same
    ``aggregation.distribution_from_claims`` the renderer uses; the old
    hand-kept mirror of SiteReport._coarse_distribution is gone."""
    return distribution_from_claims(report_claims, axis)


def _families(dist: dict[str, int]) -> tuple[int, int, int]:
    fam = family_verdict(dist)
    return fam.true_count, fam.adverse_count, fam.decided


def _bar_segment_counts(wrap_html: str) -> dict[str, int]:
    """Parse ``title="Label: N"`` segment annotations out of the bar wrap."""
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r'title="([^":]+): (\d+)"', wrap_html)}


def _bar_wrap_block(page: str) -> str | None:
    """Slice out the verdict-panel bar wrap (single-block parsing).

    Nested divs defeat a close-tag regex, so this slices on markers: wrap
    start → the anecdote note / source row. Only bar segments carry
    ``title="Label: N"`` annotations inside the wrap, so the slice is a safe
    input for _bar_segment_counts. ONE block, ONE axis (remediation v2, 1.8 /
    DC-4' / R-1): a fresh render emits a single strict bar and there is no
    per-axis branch left to parse. The committed pre-remediation pages still
    carry a hidden second bar whose annotations are byte-identical to strict's
    (the PCA verdict contract projects the same on both axes), so the whole-wrap
    slice stays exact on them too — duplicate ``Label: N`` pairs collapse in
    the dict."""
    start = page.find('<div class="vp-bar-wrap">')
    if start < 0:
        return None
    end_markers = [i for i in (page.find('vp-anecdote-note', start),
                               page.find('class="source-row"', start)) if i > 0]
    end = min(end_markers) if end_markers else len(page)
    return page[start:end]


def check_report_page(page: str, report: dict, report_claims: list[dict]) -> list[str]:
    """Verify one report page's figures against claims.json-derived values."""
    violations: list[str] = []
    slug = report.get("url", report.get("id", "?"))
    claim_count = len(report_claims)

    # Report row in reports.json must agree with claims.json (T0.7).
    if report.get("claim_count") != claim_count:
        violations.append(
            f"{slug}: reports.json claim_count={report.get('claim_count')} "
            f"but claims.json has {claim_count} claims for this report")

    # Every stored distribution must sum EXACTLY to the checkable-claim count,
    # and the fine buckets must re-derive from claims.json (PR-A2.0 / T0.1: the
    # Obama-2014 journal tally read 95 of 96 because a split row carries
    # verdict=null — no published aggregate may reproduce that drift class).
    for key in ("verdict_distribution", "verdict_distribution_lenient",
                "verdict_distribution_strict"):
        dist = report.get(key)
        if dist is None:
            continue  # legacy report row predating the coarse exports
        if sum(dist.values()) != claim_count:
            violations.append(
                f"{slug}: {key} sums to {sum(dist.values())}, "
                f"claim_count is {claim_count}")
    fine = report.get("verdict_distribution")
    if fine is not None:
        derived_fine: dict[str, int] = {}
        for c in report_claims:
            label = c.get("consensus_verdict", "")
            derived_fine[label] = derived_fine.get(label, 0) + 1
        stored_fine = {k: v for k, v in fine.items() if v}
        if stored_fine != derived_fine:
            violations.append(
                f"{slug}: verdict_distribution {stored_fine} != "
                f"claims.json-derived {derived_fine}")

    # Single-axis pass (remediation v2, 1.8 / DC-4'): the published surface
    # renders the STRICT distribution only, so the checker re-derives and
    # asserts that one axis. (Committed pre-remediation pages still carry a
    # hidden lenient twin whose figures are identical under the PCA verdict
    # contract — see _bar_wrap_block.)
    dist_strict = _coarse_dist(report_claims, "strict")
    # Verdict bar segments (title="Label: N") must reproduce the derived
    # bucketing exactly and sum to claim_count (T0.2 acceptance).
    wrap_html = _bar_wrap_block(page)
    if wrap_html is None:
        violations.append(f"{slug}: no verdict bar found")
    else:
        segs = _bar_segment_counts(wrap_html)
        if sum(segs.values()) != claim_count:
            violations.append(
                f"{slug}: bar segments sum to {sum(segs.values())}, "
                f"claim_count is {claim_count}")
        derived = {k: v for k, v in dist_strict.items() if v}
        if segs != derived:
            violations.append(
                f"{slug}: bar segments {segs} != derived buckets {derived}")

    # Headline ratio text: family shares over DECIDED claims, same
    # convention as the headline (T0.3).
    t, f, decided = _families(dist_strict)
    lean = "true-leaning" if t >= f else "false-leaning"
    fam_count = t if t >= f else f
    expected_ratio = f"{fam_count} of {decided} decided claims {lean}"
    if decided and expected_ratio not in page:
        violations.append(
            f"{slug}: expected headline ratio '{expected_ratio}' not found")

    # Header chips inside the two vp-headline-stat frames. Fresh renders
    # carry one plain value; committed pre-remediation pages wrap it in a
    # lens-target span (the leading optional-span matches the strict value,
    # which renders first there).
    for frame_cls, pick in (("vp-stat-truthy", 0), ("vp-stat-false", 1)):
        # Slice the frame FIRST and search inside it. The old pattern ran
        # `.*?` across the whole page under re.S, so a frame carrying no
        # vp-stat-num silently matched the NEXT frame's number and reported
        # the other chip's percentage as this one's.
        start = re.search(r'class="vp-headline-stat %s\b' % frame_cls, page)
        if not start:
            violations.append(f"{slug}: chip frame {frame_cls} not found")
            continue
        rest = page[start.end():]
        nxt = re.search(r'class="vp-headline-stat ', rest)
        frame = rest[:nxt.start()] if nxt else rest
        m = re.search(r'vp-stat-num">(?:<span[^>]*>)?([\d%%]+)', frame)
        if not m:
            # The small-sample guard replaces the percentage with a caveat:
            # under 10 decided claims there is deliberately no number to
            # check. That is the ONLY licensed reason for a missing chip.
            if "Small sample" in frame:
                continue
            violations.append(f"{slug}: chip frame {frame_cls} not found")
            continue
        got = m.group(1)
        want = _fmt_pct((t, f)[pick], decided)
        if got != want:
            violations.append(
                f"{slug}: chip {frame_cls} shows {got}, derived {want}")

    # Honest-abstention chip (PR-A2.1 T1.2, decomposed further by D17-d;
    # class renamed vp-selfsource-chip -> vp-abstention-chip in Wave A A3 in
    # LOCKSTEP with this parser). Its decomposition must re-derive from
    # claims.json. NOTE on the defect this replaced (Fable-ratified 2026-08-20):
    # the old regex required the terms in one fixed consecutive order --
    # decided, self-sourced-only, unverifiable-other -- so it failed on every
    # published page: two reports never emitted a self-sourced term at all, and
    # on the other three the gate term sat between the self-sourced and other
    # terms. The check was inert on all five live reports. The parser now
    # speaks the chip's actual grammar: "N decided" plus any of the abstention
    # terms plus an optional split term, all summing to claim_count.
    m = re.search(r'vp-abstention-chip[^>]*>([^<]+)</p>', page)
    if m:
        _CHIP_TERMS = {
            "decided": "decided",
            "unverified — self-sourced only": "selfsrc",
            "insufficient qualifying evidence retrieved": "gate",
            # F3: the reason-coded species, re-derived from the reason_code
            # exported onto each claim's provenance.
            "beyond the public record": "coded",
            "unverifiable — other": "other",
            "models split": "split",
        }
        got: dict[str, int] = {}
        chip_ok = True
        for part in m.group(1).split(" · "):
            pm = re.fullmatch(r"(\d+) (.+)", part.strip())
            key = _CHIP_TERMS.get(pm.group(2)) if pm else None
            if key is None or key in got:
                violations.append(
                    f"{slug}: abstention chip carries unparseable term {part!r}")
                chip_ok = False
                break
            got[key] = int(pm.group(1))
        if chip_ok:
            dist = _coarse_dist(report_claims, "strict")
            uv = dist.get("Unverifiable", 0)
            split = dist.get("Models split", 0)
            # F3: the reason-coded species takes precedence over every other
            # abstention sub-state, exactly as it does in site.py.
            def _coded(c):
                return bool(c.get("provenance", {}).get("reason_code")) and \
                    c.get("coarse_strict_label") == "Unverifiable" and \
                    c.get("consensus_verdict") != "Models split"
            coded = sum(1 for c in report_claims if _coded(c))
            selfsrc = sum(1 for c in report_claims
                          if c.get("provenance", {}).get("self_sourced_only")
                          and not _coded(c))
            # Mirrors site.py's chip term exactly: gate-withheld (UNVERIFIABLE,
            # not split, gate code insufficient) MINUS the narrower sub-states,
            # each excluded inside the same predicate chain.
            gate = sum(
                1 for c in report_claims
                if (c.get("provenance", {}).get("evidence_gate")
                    == "insufficient-qualifying-evidence"
                    and c.get("coarse_strict_label") == "Unverifiable"
                    and c.get("consensus_verdict") != "Models split"
                    and not c.get("provenance", {}).get("self_sourced_only")
                    and c.get("provenance", {}).get("layer_a_claim_type")
                    != "personal-anecdote"
                    and not _coded(c)))
            want = {"decided": claim_count - uv - split}
            if selfsrc:
                want["selfsrc"] = selfsrc
            if gate:
                want["gate"] = gate
            if coded:
                want["coded"] = coded
            other = uv - selfsrc - gate - coded
            if other:
                want["other"] = other
            if split:
                want["split"] = split
            if got != want:
                violations.append(
                    f"{slug}: abstention chip shows {got}, derived {want}")
            if sum(got.values()) != claim_count:
                violations.append(
                    f"{slug}: abstention chip terms sum to {sum(got.values())}, "
                    f"claim_count is {claim_count}")

    # Anecdote footnote must reconcile with the derived Unverifiable bucket.
    m = re.search(r'vp-anecdote-note[^>]*>(\d+) of the (\d+) Unverifiable', page)
    if m:
        n_anec_uv, uv_shown = int(m.group(1)), int(m.group(2))
        uv_derived = dist_strict.get("Unverifiable", 0)
        anec_uv_derived = sum(
            1 for c in report_claims
            if (c.get("provenance", {}).get("layer_a_claim_type") == "personal-anecdote"
                and (c.get("coarse_strict_label") == "Unverifiable"
                     and c.get("consensus_verdict") != "Models split")))
        if uv_shown != uv_derived:
            violations.append(
                f"{slug}: footnote says {uv_shown} Unverifiable, derived {uv_derived}")
        if n_anec_uv > uv_derived:
            violations.append(
                f"{slug}: footnote counts {n_anec_uv} anecdotes in a "
                f"{uv_derived}-claim Unverifiable bucket")
        if n_anec_uv != anec_uv_derived:
            violations.append(
                f"{slug}: footnote anecdote count {n_anec_uv} != derived "
                f"{anec_uv_derived} from claims.json layer_a_claim_type")
    return violations


# ── Published run-artifact invariants (remediation v2, 1.4) ──────────────────
#
# metrics/pca_runs/methodology_manifest.json pins every stored artifact to the
# methodology GENERATION it was produced under. Runs labeled with the
# manifest's current_generation must satisfy the current invariants; runs with
# older generations are permanently legacy — reported, never re-assertable —
# which is what blocks re-publishing them as-is.

def _utterance_date(speech_id: str):
    """Utterance date for the era lint over stored artifacts, or None.

    Remediation v2 Phase A (A4): resolves through
    ``verdict.speech_context.speech_date_for`` — the ONE map — instead of the
    private mirror this module used to keep. Two maps meant a speech could be
    pinned for the lint and unpinned for the pipeline (or the reverse) with
    nothing to notice; the mirror was in fact more complete than the pipeline's
    own map for three of the five speeches. It covers static pins AND runner
    registrations (``register_speech_date``), which is precisely the "statically
    pinned or runner-registered" test the publish gate applies below. Never
    reads the artifact's self-reported meta.date: a run must not certify its
    own era.
    """
    from truthbot.verdict.speech_context import speech_date_for

    return speech_date_for(f"{speech_id}:0") if speech_id else None

#: PR-A2.2 / T2.1 saturation cap, mirrored from
#: truthbot.verdict.consolidator.MAX_S5 (imported lazily in the checker to
#: keep this module render-side-import-free).
_MAX_S5_PER_SID = 3


# ── Fitness to gate (remediation v2 Phase A, A1) ─────────────────────────────
#
# The consolidator's quota (MIN_BEARING_T13) only credits items whose stance is
# supports/refutes, and the T2.4 gate FORCES Unverifiable when the quota is
# unmet. That machinery is only meaningful if the stance and relevance layers
# actually RAN. On the v2 path they do not: build_evidence_pack_v2 wires the
# R1/R2/R3 retrievers straight into consolidate(), and score_evidence — the
# only writer of relevance_score/supports_claim — sits on the legacy v1
# provider and the R4 archive retriever. A run in that state cannot be trusted
# to distinguish "no qualifying evidence exists" from "nobody looked".
#
# Such a run is UNFIT TO GATE: its gate-forced Unverifiables are artifacts of
# missing scoring, not findings. This is a REPORTED condition over stored
# artifacts (check_run_fitness) and a HARD gate at publish time
# (check_publish_gate) — deliberately not an artifact-lint violation, so
# check_run_artifacts keeps meaning "the current-generation invariants hold".

#: Stance-null ceiling. Above this share of items carrying supports_claim=None,
#: too much of the pack is invisible to _bearing() for the quota — and hence
#: for the forced-Unverifiable gate — to mean anything.
#:
#: WHY 0.15: the ten stored runs (5 published + 5 rebuilt) sit at 20.5%–34.2%
#: stance-null, so any ceiling below 20.5% flags all ten. 0.15 is set one
#: meaningful step BELOW the best observed run rather than at it: a ceiling
#: tuned to the corpus (e.g. 0.35) would certify today's behavior as the
#: standard, which is exactly the failure this lint exists to catch. 0.15
#: still tolerates a genuinely ambiguous minority — a real stance layer that
#: honestly returns "context" on ~1 item in 7 stays fit — while refusing a
#: pack where a quarter of the evidence never got a stance at all. Every
#: existing run failing this is the CORRECT outcome: they are all unfit.
UNFIT_STANCE_NULL_RATE = 0.15

#: F13 (D-B): owner-ratified, registry-keyed exceptions to the stance-null
#: publish gate. A speech listed here PUBLISHES despite being unfit, with its
#: exception version and expiry disclosed on the report and printed by the gate.
#: This is DATA, not a CLI bypass: the four unlisted speeches keep hard
#: enforcement, a second unfit speech that is not listed still refuses, and
#: --allow-unfit-gate waives nothing on a real publish. The threshold
#: (:data:`UNFIT_STANCE_NULL_RATE`) is untouched — an exception discloses that a
#: speech is over the line, it does not move the line. Expiry is a condition, not
#: a date: when it is met the entry must be removed and the speech must clear the
#: ceiling or stop publishing.
#: Expiry is a REVIEW condition, not a single fix. It said "when the D17
#: retrieval-contract fix lands" until 2026-08-11, when measuring the null
#: population showed that fix reaches only the statistical-series items (48 of
#: 309) and leaves the rate at ~17.7% — still over the ceiling. An expiry
#: condition that cannot be met is worse than none, so the exception is now
#: reviewed at each publish against the measured rate.
STANCE_NULL_GATE_EXCEPTIONS: dict[str, dict] = {
    "trump_2026": {
        "version": "dc6'-2026-08-11",
        "expiry": "reviewed at each publish against the measured stance-null "
                  "rate; no single retrieval fix clears it (the series-excerpt "
                  "fix reaches 48 of 309 items, leaving ~17.7%)",
    },
}

#: Cohort → the one-phrase gloss that says what that cohort IS. The fitness
#: finding is stated over 17 stored run artifacts, and 17 is a number nobody
#: can interpret on its own: read against the five published reports it looks
#: like triple-counting, and read against "the corpus" it looks like a bigger
#: failure than it is. So the denominator travels with the number everywhere
#: it is stated — in the report artifact, in the lint's own output, and in the
#: DC-B1 packet. The glosses are here, once, so the three cannot drift.
RUN_COHORT_GLOSS = {
    "published": "live on the site",
    "rebuilt": "staged, unpublished",
    "superseded": "retained per archive-never-delete",
}

#: Cohort order for the composition line: newest/most consequential first.
RUN_COHORT_ORDER = ("published", "rebuilt", "superseded")

#: The A1 finding, in ONE place. ``{composition}`` is filled by
#: :func:`fitness_composition` from the manifest, so the artifact, the lint and
#: any future re-statement all quote the same sentence with the same
#: denominator rather than a hand-copied one.
A1_FINDING = (
    "The v2 evidence path never scores relevance or stance: "
    "verify/relevance.py::score_evidence is reachable only from the legacy v1 "
    "provider (pipeline._build_open_book_provider) and the R4 archive "
    "retriever, while build_evidence_pack_v2 wires R1/R2/R3 straight into "
    "consolidate(). Result: relevance_score == 0.5 on 100% of items in ALL "
    "stored runs, and supports_claim null on 20.5-34.2% (retrievers.py maps "
    "stance 'context' -> None). consolidator._bearing() requires True/False, "
    "so null items cannot credit MIN_BEARING_T13=2 and the T2.4 gate forces "
    "Unverifiable. Every stored run is therefore unfit to gate — and \"every\" "
    "means {composition}, not five: the finding covers the whole stored "
    "record, of which the published site is one cohort."
)


def run_cohort(manifest_row: dict, current_generation: str) -> str:
    """Which cohort a manifest row belongs to: published / rebuilt / superseded.

    ``published`` is whatever the live site renders (any generation — the
    published corpus is deliberately NOT all one vintage). Everything else
    splits on generation: an unpublished run on the *current* generation is a
    Phase-3 ``rebuilt`` artifact awaiting the DC-6 publish decision; an
    unpublished run on an older generation is ``superseded`` and retained only
    because the archive is never deleted from.
    """
    if manifest_row.get("published"):
        return "published"
    if (manifest_row.get("generation") or "") == current_generation:
        return "rebuilt"
    return "superseded"


def fitness_composition(rows: list[dict]) -> str:
    """The denominator, spelled out: "17 stored run artifacts = 5 published …".

    Computed from the rows themselves so it can never disagree with the report
    it annotates.
    """
    counts = Counter(r.get("cohort", "") for r in rows)
    parts = [f"{counts[c]} {c} ({RUN_COHORT_GLOSS[c]})"
             for c in RUN_COHORT_ORDER if counts.get(c)]
    return (f"{len(rows)} stored run artifact{'' if len(rows) == 1 else 's'} = "
            + " + ".join(parts))


def fitness_finding(rows: list[dict]) -> str:
    """:data:`A1_FINDING` with this checkout's composition substituted in."""
    return A1_FINDING.format(composition=fitness_composition(rows))


def is_fit_to_gate(artifact) -> tuple[bool, str]:
    """Is this run artifact's evidence scored well enough to GATE on?

    Returns ``(fit, reason)``; ``reason`` is a one-line human explanation in
    both directions. Unfit when (a) not a single item carries a real relevance
    score — the whole run kept the 0.5 pydantic default, i.e. the relevance
    layer never ran — or (b) the stance-null rate exceeds
    :data:`UNFIT_STANCE_NULL_RATE`. Accepts the parsed artifact dict; works on
    artifacts that predate ``EvidencePack.scoring`` because the telemetry is
    recomputed from the stored evidence with the very same function the
    pipeline writes.
    """
    from truthbot.verdict.consolidator import scoring_telemetry_from_artifact

    evidence = artifact.get("evidence")
    if evidence is None:
        return False, "no evidence stored — nothing to gate on"
    tel = scoring_telemetry_from_artifact(evidence)
    n = tel["items"]
    if not n:
        return False, "evidence map is empty — nothing to gate on"
    if not tel["relevance_scored"]:
        return False, (
            f"relevance is entirely default: 0 of {n} items scored "
            f"(all carry the {0.5} pydantic default) — the relevance layer "
            "never ran on this run")
    if tel["stance_null_rate"] > UNFIT_STANCE_NULL_RATE:
        return False, (
            f"stance-null rate {tel['stance_null_rate']:.1%} exceeds the "
            f"{UNFIT_STANCE_NULL_RATE:.0%} ceiling "
            f"({tel['stance_null']} of {n} items carry no stance) — those "
            "items cannot credit the quota, so the forced-Unverifiable gate "
            "is measuring retrieval silence, not evidence")
    return True, (
        f"{tel['relevance_scored']} of {n} items relevance-scored, "
        f"stance-null {tel['stance_null_rate']:.1%} "
        f"(<= {UNFIT_STANCE_NULL_RATE:.0%})")


def run_fitness_report(repo_root) -> list[dict]:
    """Machine-readable fitness row per stored run artifact.

    One dict per manifest run whose artifact is present in this checkout:
    ``{run_id, speech_id, generation, published, cohort, items, packs, scored,
    scored_rate, stance_null, stance_null_rate, fit_to_gate, reason}``.
    Absent artifacts are skipped for the same reason check_run_artifacts skips
    them (CI clones legitimately carry manifest rows without the large files).

    ``cohort`` (published / rebuilt / superseded, see :func:`run_cohort`) is
    carried on every row so the headline count always has a denominator
    attached — see :func:`fitness_composition`.
    """
    from truthbot.verdict.consolidator import scoring_telemetry_from_artifact

    repo_root = Path(repo_root)
    runs_dir = repo_root / "metrics" / "pca_runs"
    manifest = _load_json(runs_dir / "methodology_manifest.json")
    current_generation = manifest.get("current_generation", "")
    rows: list[dict] = []
    for run_id, row in manifest["runs"].items():
        path = runs_dir / f"{run_id}.json"
        if not path.exists():
            continue
        artifact = _load_json(path)
        tel = scoring_telemetry_from_artifact(artifact.get("evidence") or {})
        fit, reason = is_fit_to_gate(artifact)
        rows.append({
            "run_id": run_id,
            "speech_id": row.get("speech_id", ""),
            "generation": row.get("generation", ""),
            "published": bool(row.get("published")),
            "cohort": run_cohort(row, current_generation),
            "packs": tel["packs"],
            "items": tel["items"],
            "relevance_scored": tel["relevance_scored"],
            "relevance_default": tel["relevance_default"],
            "scored_rate": round(tel["scored_rate"], 6),
            "stance_supports": tel["stance_supports"],
            "stance_refutes": tel["stance_refutes"],
            "stance_null": tel["stance_null"],
            "stance_null_rate": round(tel["stance_null_rate"], 6),
            "fit_to_gate": fit,
            "reason": reason,
        })
    return rows


def check_run_fitness(repo_root) -> list[str]:
    """REPORTED (not violating) unfit-to-gate conditions over stored runs.

    Returned as its OWN list so ``check_run_artifacts() == []`` keeps meaning
    "the current-generation invariants hold" — a fitness problem is a
    different class of defect (the evidence was never scored) from an
    invariant breach (the pack broke a rule), and conflating them would make
    the existing suite's ``violations == []`` assertion vacuous. The teeth are
    at publish time: :func:`check_publish_gate`.

    The FIRST line is the tally with its denominator spelled out
    ("N of M stored run artifacts unfit to gate — M = 5 published … "), and
    every run line names its cohort. Without that, a reader meeting "17" beside
    a five-report site has no way to tell whether the lint is counting reports,
    speeches or artifacts.
    """
    rows = run_fitness_report(repo_root)
    unfit = [r for r in rows if not r["fit_to_gate"]]
    if not unfit:
        return []
    lines = [f"{len(unfit)} of {len(rows)} stored run artifacts unfit to gate "
             f"— {fitness_composition(rows)}"]
    for row in unfit:
        lines.append(
            f"{row['run_id'][:8]} ({row['speech_id']}, {row['cohort']}"
            f"{'' if row['published'] else ', unpublished'}): unfit-to-gate — "
            f"{row['reason']}")
    return lines


def check_publish_gate(artifact, label: str = "") -> list[str]:
    """HARD publish-time gate: an unfit-to-gate run must not be published.

    Returns violations (empty = publishable). The publish path
    (``scripts/rerender_pca_site.py``) refuses to render an artifact this
    rejects. Kept out of :func:`check_run_artifacts` on purpose — storing an
    unfit run is fine and necessary (it is the evidence for the finding);
    PUBLISHING its gate-forced verdicts as fact-check results is not.
    """
    fit, reason = is_fit_to_gate(artifact)
    if fit:
        return []
    # F13: a speech with a ratified, registry-keyed exception publishes despite
    # being unfit — the notice is surfaced via :func:`publish_gate_notice`, not as
    # a violation. Every other unfit speech still refuses.
    if _gate_exception(artifact):
        return []
    name = label or (artifact.get("meta") or {}).get("speech_id") or "artifact"
    return [f"{name}: unfit-to-gate, refusing to publish — {reason}"]


def _gate_exception(artifact) -> dict | None:
    sid = (artifact.get("meta") or {}).get("speech_id") or ""
    return STANCE_NULL_GATE_EXCEPTIONS.get(sid)


def publish_gate_notice(artifact, label: str = "") -> str:
    """F13: the disclosure the gate PRINTS instead of refusing, when a speech is
    unfit but publishes under a ratified exception. Empty for a fit speech or an
    unfit speech with no exception (which :func:`check_publish_gate` refuses)."""
    fit, reason = is_fit_to_gate(artifact)
    if fit:
        return ""
    exc = _gate_exception(artifact)
    if not exc:
        return ""
    name = label or (artifact.get("meta") or {}).get("speech_id") or "artifact"
    return (f"{name}: PUBLISHED UNDER RATIFIED EXCEPTION {exc['version']} "
            f"(expires when {exc['expiry']}) — {reason}")


def check_ledger_completeness(net_ledger: dict,
                             published_entries: list[dict],
                             published_resolution: list[dict] | None = None
                             ) -> list[str]:
    """HARD publish-time gate (F6, extended F9): the corrections ledger accounts
    for every changed verdict, and the published changelog is exactly the
    net-VISIBLE set — the ledger-eligible entries UNION the non-ledger moves whose
    verdict actually crossed a models-split boundary (old != new). The
    net-UNCHANGED non-ledger churn stays prose-only and is deliberately NOT on the
    page.

    Returns violations (empty = publishable). ``net_ledger`` is the DC-6' net
    record; ``published_entries`` and ``published_resolution`` are
    ``data/corrections.json``'s ``entries`` and ``resolution_state_changes``. The
    failure this blocks is a silent drop — a claim whose verdict moved but which
    the public record fails to show — and its inverse, a phantom correction.
    """
    v: list[str] = []
    missing = net_ledger.get("completeness_missing") or []
    phantom = net_ledger.get("completeness_phantom") or []
    if not net_ledger.get("completeness_ok", False) or missing or phantom:
        v.append(
            "corrections ledger is incomplete: set(changed) != set(ledgered)"
            + (f" — {len(missing)} changed sid(s) not ledgered: {missing}" if missing else "")
            + (f" — {len(phantom)} ledgered sid(s) never changed: {phantom}" if phantom else ""))
    mism = net_ledger.get("head_mismatches") or []
    if mism:
        v.append(f"corrections ledger disagrees with the publishing heads: {mism}")
    # The published set must be EXACTLY entries UNION net-visible non-ledger — no
    # more (a phantom), no fewer (an omitted change), and the net-unchanged churn
    # must NOT leak onto the page.
    expected = set(net_ledger.get("published_expected")
                   or [e.get("sid") for e in (net_ledger.get("entries") or [])])
    published_sids = ({e.get("sid") for e in (published_entries or [])}
                      | {e.get("sid") for e in (published_resolution or [])})
    if expected != published_sids:
        only_net = sorted(expected - published_sids)
        only_pub = sorted(published_sids - expected)
        v.append(
            "data/corrections.json does not match the net ledger's published set "
            "(entries UNION net-visible non-ledger)"
            + (f" — missing from changelog: {only_net}" if only_net else "")
            + (f" — extra in changelog: {only_pub}" if only_pub else ""))
    return v


def check_run_artifacts(repo_root) -> list[str]:
    """Assert the current-generation invariants over stored pca_runs artifacts.

    Over EVERY manifest row, any generation, artifact present or not (Phase A,
    A4):
      (0)   the row's ``speech_id`` resolves to an utterance date — statically
            pinned in ``verdict.speech_context.SPEECH_DATE`` or runner-
            registered. An unresolvable speech disables era gating entirely,
            so it fails closed here rather than publishing ungated.

    For every run the methodology manifest labels ``current_generation``:
      (i)   per-claim POLITICAL-tier item count <= 3 (the S5 saturation cap),
      (ii)  zero era violations (fair-game window from the speech-date map,
            via :func:`truthbot.verdict.era_lint.lint_pack_items`),
      (iii) zero fact-check URLs in evidence
            (:func:`truthbot.verify.factcheck_exclusion.is_excluded_factchecker`).

    Runs with OLDER generations produce logged report lines, never failures —
    they are legacy by construction and the manifest is what keeps them from
    being re-published as-is. Returns the violation list (empty = pass).

    TODO (D11.2 credit-identity check, gated on generation "v2.4+"): recompute
    the decided-verdict credit set from principal relations and assert no
    decided claim rests solely on the speaker's own record. Not implementable
    over v2.3 artifacts — evidential roles are not stored on artifact evidence
    and the principals recompute belongs to the Phase-3 regeneration.
    """
    from truthbot.verdict.era_lint import lint_pack_items
    from truthbot.verify.factcheck_exclusion import is_excluded_factchecker

    repo_root = Path(repo_root)
    runs_dir = repo_root / "metrics" / "pca_runs"
    manifest = _load_json(runs_dir / "methodology_manifest.json")
    current = manifest["current_generation"]
    violations: list[str] = []

    # Phase A (A1): scoring-coverage fitness is REPORTED here, never a
    # violation — see check_run_fitness. The hard gate is check_publish_gate.
    for line in check_run_fitness(repo_root):
        logger.info("pca run %s", line)

    # Phase A (A4) — speech-date fail-closed, asserted over EVERY manifest row
    # regardless of generation or whether the artifact file is in the checkout.
    # A speech_id that resolves to no utterance date (neither statically pinned
    # in speech_context.SPEECH_DATE nor runner-registered) disables the era gate
    # wholesale: era_lint has nothing to compare against, so post-speech
    # evidence rides in unchallenged. This is the check that would have caught
    # obama_2014, gwbush_2006 and clinton_1998 running unpinned, era-gated only
    # by whichever runner happened to call register_speech_date() first.
    for run_id, row in manifest["runs"].items():
        speech_id = row.get("speech_id", "")
        if _utterance_date(speech_id) is None:
            violations.append(
                f"{run_id}: speech_id {speech_id!r} resolves to no utterance "
                "date — neither statically pinned in "
                "verdict.speech_context.SPEECH_DATE nor runner-registered. The "
                "era gate cannot run for this speech; pin it before publishing "
                "(fail closed)")

    for run_id, row in manifest["runs"].items():
        path = runs_dir / f"{run_id}.json"
        if not path.exists():
            # Absent data is NOT an integrity violation: run artifacts are
            # large and some are untracked, so a fresh clone (CI) legitimately
            # carries manifest rows without their files. The write-then-record
            # ordering is guaranteed at the source instead — the Phase-3
            # runner writes the artifact before it adds the manifest row.
            logger.info("pca run %s (%s): artifact not in this checkout — "
                        "skipped, nothing to assert", run_id[:8],
                        row.get("speech_id"))
            continue
        if row.get("generation") != current:
            logger.info(
                "pca run %s (%s): legacy generation %r%s — reported, not "
                "re-assertable under %r", run_id[:8], row.get("speech_id"),
                row.get("generation"),
                " [published]" if row.get("published") else "", current)
            continue

        artifact = _load_json(path)
        speech_id = row.get("speech_id", "")
        utterance = _utterance_date(speech_id)
        if utterance is None:
            # Already reported by the A4 pass above; skip the era invariant
            # rather than assert it against nothing.
            continue
        evidence = artifact.get("evidence")
        if evidence is None:
            violations.append(f"{run_id}: current-generation run stores no evidence")
            continue

        for sid, items in evidence.items():
            pol = sum(1 for it in items if it.get("source_tier") == "Political")
            if pol > _MAX_S5_PER_SID:                                    # (i)
                violations.append(
                    f"{run_id} {sid}: {pol} POLITICAL-tier items exceed the "
                    f"<={_MAX_S5_PER_SID} S5 cap")
            era, _, _ = lint_pack_items(sid, items, utterance)           # (ii)
            for v in era:
                violations.append(f"{run_id} {sid}: era violation — {v.message}")
            for it in items:                                             # (iii)
                url = it.get("source_url") or ""
                if url and is_excluded_factchecker(url):
                    violations.append(
                        f"{run_id} {sid}: fact-check URL in evidence: {url}")

        # Standing agreed-verdict audit coverage (remediation v2, 1.12) —
        # REPORT-ONLY: rows the Severity Classifier auto-adjusted
        # (crm114.final set) that carry no audit stamp (the pre-audit
        # artifacts all do, until the Phase-3 regeneration re-runs them
        # through publish_pipeline.apply_verdict_audit).
        # TODO(Phase 3 publish gate): promote this report line to a hard
        # violation — a published crm114-overridden row without audit
        # coverage must fail the build once the regen lands.
        uncovered = sum(
            1 for r in artifact.get("rows") or []
            if ((r.get("crm114") or {}).get("final")) and "audit_flags" not in r)
        if uncovered:
            logger.info(
                "pca run %s (%s): %d crm114-overridden row(s) lack "
                "agreed-verdict audit coverage (report-only until the "
                "Phase-3 gate)", run_id[:8], speech_id, uncovered)
    return violations


# _check_index_tier_buckets (remediation v2, 1.6) was removed in the site
# readability pass: it validated the homepage card's ``.src-tiers`` chip
# against ``reports.json`` tier_counts, and that chip was removed from
# ``_report_card`` entirely (source-tier detail now lives only on the
# report page) — there is no longer a homepage-rendered surface for this
# check to validate.


#: Buckets the aggregate bar can actually render (aggregation.AGGREGATE_BAR_ORDER
#: is the family-grouped union of both axes) — a nonzero count outside this set
#: would silently vanish from every rendered bar.
def _check_bucket_invariants(reports: list[dict], claims: list[dict]) -> list[str]:
    """Remediation v2 (1.6) strict lints (ii)+(iii): per-report bucket sums
    equal claim_count with every nonzero bucket renderable, and the
    site-wide aggregate (sum of per-report distributions) accounts for
    every claim in claims.json exactly once, on every axis."""
    from truthbot.publish.aggregation import AGGREGATE_BAR_ORDER
    renderable = set(AGGREGATE_BAR_ORDER)
    violations: list[str] = []
    totals: dict[str, int] = {"verdict_distribution": 0,
                              "verdict_distribution_lenient": 0,
                              "verdict_distribution_strict": 0}
    for r in reports:
        slug = r.get("url", r.get("id", "?"))
        claim_count = r.get("claim_count", 0)
        for key in totals:
            dist = r.get(key)
            if dist is None:
                continue
            totals[key] += sum(dist.values())
            if sum(dist.values()) != claim_count:   # (ii) — also checked in
                # check_report_page for the fine dist; repeated here so the
                # strict pass reports it even for index-only renders.
                violations.append(
                    f"{slug}: {key} sums to {sum(dist.values())}, "
                    f"claim_count is {claim_count}")
            if key != "verdict_distribution":
                ghost = {k: v for k, v in dist.items()
                         if v and k not in renderable}
                if ghost:
                    violations.append(
                        f"{slug}: {key} buckets {ghost} are outside "
                        "AGGREGATE_BAR_ORDER and would not render")
    for key, total in totals.items():               # (iii)
        if total != len(claims):
            violations.append(
                f"site-wide: {key} buckets sum to {total} across reports.json, "
                f"claims.json has {len(claims)} entries")
    return violations


#: Lens-UI fingerprints. The first entry is the user-visible WORD — matched on
#: word boundaries so "Zelenskyy"/"Lenskyy" in a source URL or headline cannot
#: masquerade as the retired chip. The rest are the markup hooks the toggle
#: used to hang off; they are not reader-visible on their own, but any of them
#: surviving means a lens code path is still rendering.
_LENS_UI_PATTERNS: list[tuple[str, str]] = [
    (r"\bLens\b", 'the user-visible word "Lens"'),
    (r"editorial-lens", "the editorial-lens chip class"),
    (r"lens-label", "the lens-label span"),
    (r"lens-value", "the lens-value span"),
    (r"lens-target", "a lens-target span"),
    (r"lens-pill", "a lens-pill class"),
    (r"data-lens", "a data-lens* attribute"),
    (r"DEFAULT_LENS", "the DEFAULT_LENS JS constant"),
]


def _check_no_lens_ui(site_root: Path) -> list[str]:
    """R-1: NO lens UI anywhere on the rendered site (owner ruling).

    The Strict/Lenient toggle is gone and there is exactly one grading
    posture, so no rendered page — HTML, CSS, or JS — may show the word
    "Lens" or carry the markup the toggle hung off. Walks every asset the
    publisher writes rather than a hand-listed set of pages, because the last
    two removal passes each left a remnant on a surface nobody thought to
    check (the status-bar chip, then the paired-axis CSS).

    STRICT-GATED on purpose: the COMMITTED ``site-pca/`` tree predates the
    removal and still renders the chip on every page, which is precisely why
    this lint is only asserted against fresh renders.
    """
    violations: list[str] = []
    for path in sorted(site_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".html", ".css", ".js"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(site_root).as_posix()
        for pattern, what in _LENS_UI_PATTERNS:
            if re.search(pattern, text):
                violations.append(f"{rel}: lens UI remnant — {what} is present")
    return violations


# ── Rendered tier == stored tier (remediation v2 Phase A, A5 / C-3(a)) ───────
#
# The renderer used to classify every source URL at RENDER time, so a published
# report could show a different evidence hierarchy than the artifact its
# verdicts were written from (414/1543 and 272/918 items on the two published
# modern reports). Now every pack item renders its stored tier and advertises
# it as ``data-stored-tier``; this lint proves the badge next to it agrees.
#
# Two figures come out of the sweep:
#   * VIOLATIONS — a badge that contradicts the stored tier sitting on the same
#     element. There is no benign version of this; it means the join silently
#     stopped working.
#   * the FALLBACK RATE — the share of rendered tier badges that had no stored
#     tier to use (``data-tier-src="classified"``). Reported, never asserted: a
#     model can legitimately cite a URL that never entered the pack. It is the
#     magnitude that matters, which is why it is surfaced rather than hidden.

#: One rendered pack item: its stored tier plus the badge/source attributes.
_LI_STORED_TIER_RE = re.compile(
    r'<li\b[^>]*\bdata-stored-tier="([^"]*)"[^>]*>(.*?)</li>', re.S)
_BADGE_RE = re.compile(
    r'<span class="evidence-tier [^"]*" data-tier-src="(stored|classified)">'
    r'([^<]*)</span>')


def tier_render_telemetry(site_root: Path, reports: list[dict]) -> dict:
    """Rendered-tier join telemetry over the report pages, per report + total.

    ``{"per_report": {slug: {...}}, "total": {...}}`` where each row is
    ``{joined, fallback, total, fallback_rate}`` counted off the rendered
    ``data-tier-src`` attributes. Independent of reports.json — it measures the
    HTML that actually shipped, so a renderer that reports one number and
    prints another cannot hide.
    """
    site_root = Path(site_root)
    per_report: dict[str, dict] = {}
    tot_joined = tot_fallback = 0
    for report in reports:
        url = report.get("url", "")
        page_path = site_root / url
        if not url or not page_path.exists():
            continue
        page = page_path.read_text(encoding="utf-8")
        srcs = [m.group(1) for m in _BADGE_RE.finditer(page)]
        joined = sum(1 for s in srcs if s == "stored")
        fallback = len(srcs) - joined
        tot_joined += joined
        tot_fallback += fallback
        per_report[url] = {
            "joined": joined, "fallback": fallback, "total": len(srcs),
            "fallback_rate": round(fallback / len(srcs), 6) if srcs else 0.0}
    total = tot_joined + tot_fallback
    return {"per_report": per_report,
            "total": {"joined": tot_joined, "fallback": tot_fallback,
                      "total": total,
                      "fallback_rate": (round(tot_fallback / total, 6)
                                        if total else 0.0)}}


def _check_rendered_tiers(site_root: Path, reports: list[dict]) -> list[str]:
    """A5 lint: every JOINED item renders the tier the artifact stored.

    Also logs the per-report and site-wide fallback rate — visible in the
    render log next to the publisher's own printed figure.
    """
    from truthbot.models import SourceTier
    from truthbot.verify.source_tiers import TIER_DISPLAY

    site_root = Path(site_root)
    violations: list[str] = []
    for report in reports:
        url = report.get("url", "")
        page_path = site_root / url
        if not url or not page_path.exists():
            continue
        page = page_path.read_text(encoding="utf-8")
        for m in _LI_STORED_TIER_RE.finditer(page):
            stored_raw, body = m.group(1), m.group(2)
            badge = _BADGE_RE.search(body)
            if badge is None:
                violations.append(
                    f"{url}: pack item stores tier {stored_raw!r} but renders "
                    "no evidence-tier badge")
                continue
            src_attr, code = badge.group(1), badge.group(2)
            try:
                want = TIER_DISPLAY[SourceTier(stored_raw)][0]
            except (ValueError, KeyError):
                violations.append(
                    f"{url}: pack item carries unknown stored tier "
                    f"{stored_raw!r}")
                continue
            if code != want:
                violations.append(
                    f"{url}: pack item stores tier {stored_raw!r} (badge "
                    f"{want}) but renders {code} — the render is not showing "
                    "the tier the panel adjudicated on")
            if src_attr != "stored":
                violations.append(
                    f"{url}: pack item has a stored tier {stored_raw!r} but "
                    f"its badge is marked {src_attr!r} — the join was skipped")

    tel = tier_render_telemetry(site_root, reports)
    for slug, row in tel["per_report"].items():
        logger.info("tier join %s: %d/%d stored, fallback rate %.1f%%",
                    slug, row["joined"], row["total"],
                    100 * row["fallback_rate"])
    logger.info("tier join site-wide: %d/%d stored, fallback rate %.1f%%",
                tel["total"]["joined"], tel["total"]["total"],
                100 * tel["total"]["fallback_rate"])
    return violations


def check_feed(site_root: Path, reports: list[dict]) -> list[str]:
    """Validate feed.xml against the reports index (remediation v2, 1.5).

    Violations: the legacy [SITE_URL] placeholder anywhere, unparseable XML,
    entry count != len(reports), an entry link whose page is missing under
    ``site_root``, duplicate entry ids, or a feed-level <updated> that is
    not the max entry <updated>."""
    import xml.etree.ElementTree as ET
    from urllib.parse import urlparse

    site_root = Path(site_root)
    feed_path = site_root / "feed.xml"
    if not feed_path.exists():
        return ["feed.xml: file missing"]
    text = feed_path.read_text(encoding="utf-8")
    violations: list[str] = []
    if "[SITE_URL]" in text:
        violations.append("feed.xml: legacy [SITE_URL] placeholder present")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        violations.append(f"feed.xml: XML parse error: {exc}")
        return violations
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = root.findall("a:entry", ns)
    if len(entries) != len(reports):
        violations.append(
            f"feed.xml: {len(entries)} entries, reports.json has "
            f"{len(reports)} reports")
    ids = [e.findtext("a:id", default="", namespaces=ns) for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        violations.append(f"feed.xml: duplicate entry ids: {sorted(dupes)}")
    updated = [e.findtext("a:updated", default="", namespaces=ns)
               for e in entries]
    feed_updated = root.findtext("a:updated", default="", namespaces=ns)
    if entries and feed_updated != max(updated):
        violations.append(
            f"feed.xml: feed <updated> {feed_updated!r} != max entry "
            f"<updated> {max(updated)!r}")
    for e, eid in zip(entries, ids):
        link = e.find("a:link", ns)
        href = link.get("href", "") if link is not None else ""
        # Entry links are {site_url}/reports/<slug>.html — the trailing two
        # path segments locate the page under the site root.
        rel = "/".join(urlparse(href).path.split("/")[-2:])
        if not rel or not (site_root / rel).exists():
            violations.append(
                f"feed.xml: entry {eid or href}: linked page {rel!r} "
                "missing under site root")
    return violations


def check_site(site_root: Path, strict_buckets: bool = True) -> list[str]:
    """Verify the whole rendered site. Returns a list of violations (empty
    when every checked figure derives cleanly from data/*.json).

    ``strict_buckets`` gates the remediation-v2 lints (index Sources-chip
    buckets, per-report/site-wide bucket sums, feed validity, the R-1
    no-lens-UI sweep, and the A5 rendered-tier == stored-tier join). Default
    True — every fresh render must satisfy them.
    The COMMITTED site-pca/ tree predates the remediation regeneration (its
    cards were rendered without the political bucket, and every page still
    carries the retired lens chip), so tests/test_site_consistency.py lints it
    with ``strict_buckets=False`` until the Phase-2 regen flips it to True."""
    site_root = Path(site_root)
    violations: list[str] = []
    reports = _load_json(site_root / "data" / "reports.json")
    claims = _load_json(site_root / "data" / "claims.json")

    # ── Index program stats (T0.1 / T0.7) ────────────────────────────────
    index_html = (site_root / "index.html").read_text(encoding="utf-8")
    reports_claim_sum = sum(r.get("claim_count", 0) for r in reports)
    if reports_claim_sum != len(claims):
        violations.append(
            f"index: reports.json claim_counts sum to {reports_claim_sum}, "
            f"claims.json has {len(claims)} entries")
    m = re.search(r'<div class="num">(\d+)</div><div class="lbl">Claims Checked',
                  index_html)
    if not m:
        violations.append("index: Claims Checked stat not found")
    elif int(m.group(1)) != len(claims):
        violations.append(
            f"index: Claims Checked shows {m.group(1)}, claims.json has {len(claims)}")

    m = re.search(r'<div class="num">(\d+)<span class="unit">%</span></div>'
                  r'<div class="lbl">Model Consensus', index_html)
    if not m:
        violations.append("index: Model Consensus stat not found")
    else:
        want = round(sum(r.get("model_agreement_rate", 0) * r.get("claim_count", 0)
                         for r in reports) / (reports_claim_sum or 1) * 100)
        if int(m.group(1)) != want:
            violations.append(
                f"index: Model Consensus shows {m.group(1)}%, claim-weighted "
                f"mean of reports.json is {want}%")

    # ── Insights page + tagline guards (T0.4 → rebuilt in T4.1) ──────────
    # Valid states: the About redirect stub (no per-seat data) or the v2
    # per-seat page. The v1 pseudo-model page ("Hydramind") must never ship.
    insights = site_root / "model-insights.html"
    if insights.exists():
        text = insights.read_text(encoding="utf-8")
        is_stub = 'http-equiv="refresh"' in text
        is_v2 = "Model panel insights" in text and "panel_by_role" in text
        if "Hydramind" in text or not (is_stub or is_v2):
            violations.append(
                "model-insights.html: expected the v2 per-seat page or the "
                "About redirect stub; the v1 pseudo-model page must not ship")
    banned_pairs = [("index.html", "primary sources"),
                    ("about.html", "comparable accuracy"),
                    ("about.html", "never silently broken")]
    if strict_buckets:
        # Remediation v2 (1.9) About reconciliation: copy that contradicts
        # shipped behavior must not re-enter. Gated with the other v2 lints
        # because the COMMITTED pre-regen about.html still carries all three
        # (the Phase-2 regen rewrites it and flips the flag to True).
        banned_pairs += [("about.html", "capped at six"),
                         ("about.html", "fact-check databases"),
                         ("about.html", "Panel split")]
    for fname, banned in banned_pairs:
        p = site_root / fname
        if p.exists() and banned in p.read_text(encoding="utf-8"):
            violations.append(f"{fname}: banned phrase present: '{banned}'")

    # ── Corrections page state (remediation v2, 1.11) ────────────────────
    # The entries table and the empty-state sentence are mutually exclusive
    # by construction in _render_corrections; both at once means the caller
    # rendered notes/entries inconsistently (the --corrections skip bug:
    # audit note + "No corrections have been issued" on one page).
    corrections_page = site_root / "corrections.html"
    if corrections_page.exists():
        text = corrections_page.read_text(encoding="utf-8")
        if ("corrections-table" in text
                and "No corrections have been issued" in text):
            violations.append(
                "corrections.html: renders BOTH the corrections entries "
                "table and the empty-state sentence")

    # ── Remediation-v2 strict lints (1.5 + 1.6) ──────────────────────────
    # Gated because the COMMITTED site-pca/ tree predates the regeneration
    # (old static feed.xml, cards without the political bucket); every
    # fresh render runs them (default True).
    if strict_buckets:
        violations.extend(_check_bucket_invariants(reports, claims))
        violations.extend(check_feed(site_root, reports))
        violations.extend(_check_no_lens_ui(site_root))
        violations.extend(_check_rendered_tiers(site_root, reports))

    # ── Per-report pages ─────────────────────────────────────────────────
    for report in reports:
        url = report.get("url", "")
        page_path = site_root / url
        if not url or not page_path.exists():
            violations.append(f"report {report.get('id', '?')}: page {url} missing")
            continue
        page = page_path.read_text(encoding="utf-8")
        violations.extend(
            check_report_page(page, report,
                              _claims_for_report(claims, report.get("id"))))
    return violations
