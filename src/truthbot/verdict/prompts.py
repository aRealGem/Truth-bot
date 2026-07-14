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
    'Wire) above lower ones on conflict. If the provided evidence is absent or '
    'insufficient to settle the claim, verdict=UNVERIFIABLE with citations [].'
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

# I3 guard at load — a speaker/source conditional in any seat prompt fails the import.
for _role, _tmpl in {**PROMPTS, **OPEN_BOOK_PROMPTS}.items():
    lint_template_for_speaker_conditionals(f"PCA_{_role.upper()}", _tmpl)
