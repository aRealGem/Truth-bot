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

# I3 guard at load — a speaker/source conditional in any seat prompt fails the import.
for _role, _tmpl in PROMPTS.items():
    lint_template_for_speaker_conditionals(f"PCA_{_role.upper()}", _tmpl)
