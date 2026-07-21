"""
Layer B / PCA verdict prompts — the frozen proposer→critic→arbiter system prompts
and the closed-book JSON output contract.

Closed-book: no external evidence is provided, so the panel adjudicates from general
knowledge and MUST cite nothing (citations == []; enforced by I4 in pca.reduce). If a
claim cannot be adjudicated from general knowledge the verdict is UNVERIFIABLE.

Speaker-blind (I3): none of these prompts reference the speaker/source — linted at
import, so a speaker conditional fails module load, not a run (mirrors A2_SYSTEM in
checkworthy/classifier.py).

These are lifted verbatim from eval/benchmarks/run_pca_devlot.py, which proved the
closed-book P→C→A shape on roster.dev; this module is now their single source of truth.
"""
from __future__ import annotations

from hydramind.invariants import lint_template_for_speaker_conditionals

# Closed-book 4-label set. Deliberately coarser than the full 6-bucket
# models.py::VerdictLabel — without evidence, TRUE/MOSTLY_TRUE and
# MISLEADING/EXAGGERATED gradations can't be grounded, so we avoid false precision.
# The map up to VerdictLabel is deferred to Layer C (evidence-grounded).
VERDICTS = "TRUE | FALSE | MISLEADING | UNVERIFIABLE"

_CONTRACT = (
    'Return JSON only: {"verdict": "%s", "confidence": 0.0-1.0, '
    '"citations": [], "reasoning": "one clause"}. Closed-book: no external '
    'evidence is provided, so cite nothing (citations must be []). '
    'If the claim cannot be adjudicated from general knowledge, verdict=UNVERIFIABLE.'
    % VERDICTS
)

PROMPTS = {
    "proposer": "You are the PROPOSER. Assess the factual claim and draft a verdict. " + _CONTRACT,
    "critic":   "You are the CRITIC. Independently and skeptically assess the same claim; "
                "try to find why a naive verdict could be wrong. " + _CONTRACT,
    "arbiter":  "You are the ARBITER. Adjudicate the claim decisively. " + _CONTRACT,
}

# ── Open-book (Layer C) contract ──────────────────────────────────────────────
# Evidence is supplied in the payload under "evidence" as a list of items, each
# with an "id" (E1, E2, ...). The panel grounds the verdict in that evidence and
# cites the ids it relied on. I4 (pca.reduce) enforces citations ⊆ provided ids, so
# a cite must reference a supplied item — never a fabricated URL. If the evidence is
# absent or insufficient to settle the claim, the verdict is UNVERIFIABLE with
# citations []. Still speaker-blind (linted below).
_OPEN_CONTRACT = (
    'Evidence items are provided in the input under "evidence"; each has an "id" '
    '(E1, E2, ...), a source, a trust tier, and a dated snippet. Ground your verdict '
    'in that evidence and judge the claim as of its utterance date. '
    'Return JSON only: {"verdict": "%s", "confidence": 0.0-1.0, '
    '"citations": ["E1", ...], "reasoning": "one clause"}. Set "citations" to the '
    'ids of the evidence items you relied on — cite ONLY provided ids, never a bare '
    'URL or an id not in the evidence list. Weigh higher-trust tiers (Government, '
    'Wire) above lower ones on conflict. An item may carry a "stance" ("supports" or '
    '"refutes") — how that source bears on the claim; treat a direct refutation or '
    'confirmation from a trustworthy source as strong evidence, but do not let stance '
    'override the trust-tier ordering above when sources conflict. If the provided '
    'evidence is absent or insufficient to settle the claim, verdict=UNVERIFIABLE '
    'with citations [].'
    % VERDICTS
)

OPEN_BOOK_PROMPTS = {
    "proposer": "You are the PROPOSER. Assess the factual claim and draft a verdict. " + _OPEN_CONTRACT,
    "critic":   "You are the CRITIC. Independently and skeptically assess the same claim "
                "against the SAME evidence; try to find why a naive verdict could be wrong. "
                + _OPEN_CONTRACT,
    "arbiter":  "You are the ARBITER. Adjudicate the claim decisively on the evidence. "
                + _OPEN_CONTRACT,
}

# ── Calibrated open-book prompts (ADOPTED open-book default — P67 Track B) ─────
# Adopted 2026-07-19 after the Track B matched-prompt eval: dev+calib decided-acc
# 0.5625 vs plain 0.50 and vs the frontier panel 0.474-0.50; also the only config
# that correctly abstains on the unverifiable row. adjudicator.adjudicate uses this
# set whenever an evidence provider is present; OPEN_BOOK_PROMPTS is retained as the
# plain A/B baseline (score_layerb_vs_gold.py --plain).
# Gate-2 diagnosis: given open-book evidence the seats UNANIMOUSLY soften severity
# — FALSE→MISLEADING (a false core claim with a kernel of truth) and MISLEADING→TRUE
# (a distorted-but-supported claim). The softening is model-level, so the fix has to
# move the seats' own judgment. This is a DECISION PROCEDURE keyed on the claim's CORE
# assertion (not a definition list — that A/B abstained more without fixing the bug).
# It targets BOTH softening directions and is careful NOT to induce abstention (which
# would inflate decided-accuracy via a denominator effect). Speaker-blind (linted).
_CALIB_PROCEDURE = (
    'Classify by the claim\'s CORE assertion, judged against the evidence: '
    '(1) State the single central factual assertion, ignoring rhetoric. '
    '(2) FALSE if the evidence CONTRADICTS the core assertion — the stated fact did not '
    'happen, or the reverse is true — even when a peripheral detail is accurate. Do not '
    'soften a contradicted core to MISLEADING because it contains a kernel of truth. '
    '(2b) ABSOLUTE-CLAIM RULE: when the core assertion is an absolute or universal — '
    'zero, none, only, all, every, ended, eliminated, completely stopped or destroyed, '
    'biggest/lowest in history — evidence of material counterexamples CONTRADICTS that '
    'core, and the verdict is FALSE. The underlying trend or event being real does NOT '
    'soften an absolute to MISLEADING: "we ended X" is FALSE when X demonstrably '
    'continues, even if X was substantially reduced. '
    '(3) MISLEADING if the core is REAL but the evidence shows it is exaggerated, '
    'cherry-picked, stripped of context, or spun to create a false impression. '
    'Overstating or distorting a true underlying fact is MISLEADING — NOT FALSE (unless '
    'the claim states an absolute; see 2b); reserve FALSE for a core the evidence '
    'actually contradicts. Also do NOT call such a claim TRUE. '
    '(4) TRUE only when the evidence supports the core assertion without material '
    'exaggeration or distortion. '
    '(5) UNVERIFIABLE only when the provided evidence cannot settle the core assertion — '
    'not as a hedge when a label is uncomfortable. '
    'Distinguish contradiction (FALSE) from overstatement of a real fact (MISLEADING); '
    'pick the label the evidence warrants, and do not default toward the middle. '
)

CALIBRATED_OPEN_BOOK_PROMPTS = {
    "proposer": "You are the PROPOSER. Assess the factual claim and draft a verdict. "
                + _CALIB_PROCEDURE + _OPEN_CONTRACT,
    "critic":   "You are the CRITIC. Independently and skeptically assess the same claim "
                "against the SAME evidence; test whether the core assertion is actually "
                "FALSE rather than merely misleading. " + _CALIB_PROCEDURE + _OPEN_CONTRACT,
    "arbiter":  "You are the ARBITER. Adjudicate the claim decisively on the evidence. "
                + _CALIB_PROCEDURE + _OPEN_CONTRACT,
}

# I3 guard at load — a speaker/source conditional in any seat prompt fails the import.
for _role, _tmpl in {**PROMPTS, **OPEN_BOOK_PROMPTS, **CALIBRATED_OPEN_BOOK_PROMPTS}.items():
    lint_template_for_speaker_conditionals(f"PCA_{_role.upper()}", _tmpl)
