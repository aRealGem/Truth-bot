"""
Fitness function for evolved prompt genomes.

Scores each individual against eval/sotu-2026/reference.json on:
  1. Claim recall      (0.25): fraction of 29 reference claims identified
  2. Verdict agreement (0.30): label accuracy on matched claims
  3. Explanation quality (0.20): keyword/entity overlap with reference explanations
  4. Source citation quality (0.15): reference to authoritative source types
  5. Parsimony         (0.10): token efficiency (fewer tokens = higher score)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Reference data ─────────────────────────────────────────────────────────────

_REFERENCE_PATH = Path(__file__).parent.parent / "sotu-2026" / "reference.json"


def load_reference() -> list[dict]:
    """Load the 29 reference claims from reference.json."""
    with open(_REFERENCE_PATH) as f:
        return json.load(f)


# ── Verdict distance matrix ────────────────────────────────────────────────────
# Maps (reference_label, predicted_label) → penalty [0.0, 1.0]
# 0.0 = perfect match, 1.0 = maximum disagreement

# Normalized labels: map reference.json verdicts to VerdictLabel-like strings
_LABEL_NORMALIZE: dict[str, str] = {
    "TRUE": "true",
    "FALSE": "false",
    "PARTLY TRUE": "mostly_true",
    "UNSUPPORTED": "unverifiable",
    "MISLEADING": "misleading",
    "EXAGGERATED": "exaggerated",
    "FALSE / MISLEADING": "misleading",
    "UNVERIFIABLE": "unverifiable",
}

_TRUTHBOT_LABEL_NORMALIZE: dict[str, str] = {
    "True": "true",
    "Mostly True": "mostly_true",
    "Misleading": "misleading",
    "Exaggerated": "exaggerated",
    "False": "false",
    "Unverifiable": "unverifiable",
}

# Ordered list (most positive → most negative)
_LABEL_ORDER = ["true", "mostly_true", "exaggerated", "misleading", "unverifiable", "false"]
_LABEL_POS = {label: i for i, label in enumerate(_LABEL_ORDER)}


def verdict_distance(ref_verdict: str, pred_verdict: str) -> float:
    """
    Return a penalty in [0, 1] for the distance between two verdict labels.
    0 = perfect match, 1 = opposite ends (true ↔ false).
    """
    ref_norm = _LABEL_NORMALIZE.get(ref_verdict.upper().strip(), "unverifiable")
    pred_norm = _TRUTHBOT_LABEL_NORMALIZE.get(pred_verdict.strip(), None)
    if pred_norm is None:
        # Try normalizing via the reference map too
        pred_norm = _LABEL_NORMALIZE.get(pred_verdict.upper().strip(), "unverifiable")

    if ref_norm == pred_norm:
        return 0.0

    ref_pos = _LABEL_POS.get(ref_norm, 3)
    pred_pos = _LABEL_POS.get(pred_norm, 3)
    max_dist = len(_LABEL_ORDER) - 1
    return abs(ref_pos - pred_pos) / max_dist


def verdict_agreement_score(ref_verdict: str, pred_verdict: str) -> float:
    """1.0 = exact match, approaching 0 as verdicts diverge."""
    return 1.0 - verdict_distance(ref_verdict, pred_verdict)


# ── Fuzzy claim matching ───────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_claim_similarity(claim_a: str, claim_b: str) -> float:
    """
    Compute token Jaccard similarity between two claim strings.
    Returns 0.0–1.0.
    Falls back to thefuzz if available for a combined score.
    """
    a_tokens = set(_normalize_text(claim_a).split())
    b_tokens = set(_normalize_text(claim_b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    try:
        from thefuzz import fuzz  # type: ignore
        ratio = fuzz.token_set_ratio(claim_a, claim_b) / 100.0
        return (jaccard + ratio) / 2
    except ImportError:
        return jaccard


def match_claims_to_reference(
    extracted_claims: list[dict],
    reference: list[dict],
    threshold: float = 0.20,
) -> list[dict]:
    """
    For each reference claim, find the best-matching extracted claim.

    Returns a list of match dicts:
      {
        "ref_id": int,
        "ref_claim": str,
        "ref_verdict": str,
        "matched_claim": str | None,
        "similarity": float,
        "matched": bool,
      }
    """
    results = []
    for ref_item in reference:
        ref_text = ref_item["claim"]
        best_sim = 0.0
        best_match = None
        for ext in extracted_claims:
            ext_text = ext.get("text", "") if isinstance(ext, dict) else str(ext)
            sim = fuzzy_claim_similarity(ref_text, ext_text)
            if sim > best_sim:
                best_sim = sim
                best_match = ext_text
        results.append({
            "ref_id": ref_item["id"],
            "ref_claim": ref_text,
            "ref_verdict": ref_item["verdict"],
            "ref_explanation": ref_item.get("explanation", ""),
            "matched_claim": best_match,
            "similarity": best_sim,
            "matched": best_sim >= threshold,
        })
    return results


# ── Explanation quality scoring ────────────────────────────────────────────────

# Keywords indicating data-backed, source-cited explanations
_DATA_KEYWORDS = [
    r"\b\d+(\.\d+)?%",           # percentages
    r"\$\d+",                     # dollar amounts
    r"\b\d{4}\b",                 # years
    r"\bBLS\b", r"\bBEA\b", r"\bEIA\b", r"\bCBP\b", r"\bFBI\b",
    r"\bFactCheck\b", r"\bPolitiFact\b", r"\bAP\b", r"\bReuters\b",
    r"\bFreddie Mac\b", r"\bAAA\b", r"\bGasBuddy\b",
    r"\bcredible\b", r"\bprimary\b", r"\bofficial\b", r"\bdata\b",
    r"\bpercent\b", r"\bmillion\b", r"\bbillion\b", r"\btrillion\b",
]

_DATA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _DATA_KEYWORDS]


def explanation_quality_score(explanation: str) -> float:
    """
    Score explanation quality by counting data signals.
    Returns 0.0–1.0, saturating at 5+ signals.
    """
    if not explanation:
        return 0.0
    hits = sum(1 for pat in _DATA_PATTERNS if pat.search(explanation))
    # Also reward length (more detail = better)
    length_score = min(len(explanation) / 500.0, 1.0)
    signal_score = min(hits / 5.0, 1.0)
    return 0.6 * signal_score + 0.4 * length_score


# ── Source citation quality ────────────────────────────────────────────────────

_AUTHORITATIVE_SOURCES = [
    r"\bBLS\b", r"\bBEA\b", r"\bEIA\b", r"\bCBP\b", r"\bFBI\b", r"\bCBO\b",
    r"\bCensus\b", r"\bFederal Reserve\b", r"\bFred\b", r"\bFreddie Mac\b",
    r"\bAP\b", r"\bReuters\b", r"\bNYT\b", r"\bWashington Post\b",
    r"\bNew York Times\b", r"\bCBS\b", r"\bBBC\b",
    r"\bPolitiFact\b", r"\bFactCheck\.org\b", r"\bSnopes\b",
    r"\bNY Fed\b", r"\bFederal Reserve\b",
    r"\bgov\b", r"\bgovernment\b",
]

_SOURCE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _AUTHORITATIVE_SOURCES]


def source_citation_score(explanation: str) -> float:
    """
    Score whether the explanation references authoritative sources.
    Returns 0.0–1.0.
    """
    if not explanation:
        return 0.0
    hits = sum(1 for pat in _SOURCE_PATTERNS if pat.search(explanation))
    return min(hits / 3.0, 1.0)


# ── Parsimony scoring ──────────────────────────────────────────────────────────

def parsimony_score(token_count: int, target_min: int = 500, target_max: int = 2000) -> float:
    """
    Reward lower token counts. Returns 1.0 at/below target_min, 0.0 at target_max.
    """
    if token_count <= target_min:
        return 1.0
    if token_count >= target_max:
        return 0.0
    return 1.0 - (token_count - target_min) / (target_max - target_min)


# ── Main fitness scorer ────────────────────────────────────────────────────────

class FitnessScorer:
    """
    Compute fitness scores for an Individual given its pipeline outputs.

    Usage:
        scorer = FitnessScorer()
        scores = scorer.score(
            extracted_claims=[{"text": "...", "is_checkable": True, ...}, ...],
            verdicts=[{"claim_text": "...", "label": "False", "explanation": "..."}, ...],
            token_count=1200,
        )
    """

    def __init__(self, reference: list[dict] | None = None):
        self._reference = reference or load_reference()

    def score(
        self,
        extracted_claims: list[dict],
        verdicts: list[dict],
        token_count: int = 0,
    ) -> dict[str, float]:
        """
        Score a complete pipeline run against the reference.

        Parameters
        ----------
        extracted_claims:
            Raw claim dicts from extraction (each with at least "text").
        verdicts:
            Verdict dicts (each with "claim_text", "label", "explanation").
        token_count:
            Total tokens used (extraction + synthesis) for parsimony scoring.

        Returns
        -------
        dict with keys: claim_recall, verdict_agreement, explanation_quality,
        source_citation_quality, parsimony, fitness
        """
        checkable = [c for c in extracted_claims if c.get("is_checkable", True)]

        # 1. Claim recall
        matches = match_claims_to_reference(checkable, self._reference)
        recall = sum(1 for m in matches if m["matched"]) / len(self._reference)

        # 2. Verdict agreement -- only on matched claims that also have a verdict
        verdict_scores = []
        for match in matches:
            if not match["matched"]:
                continue
            # Find corresponding verdict
            verdict = self._find_verdict(match["matched_claim"], verdicts)
            if verdict is None:
                continue
            score = verdict_agreement_score(match["ref_verdict"], verdict["label"])
            verdict_scores.append(score)
        verdict_agreement = (
            sum(verdict_scores) / len(verdict_scores) if verdict_scores else 0.0
        )

        # 3. Explanation quality -- average over all verdicts
        expl_scores = [
            explanation_quality_score(v.get("explanation", ""))
            for v in verdicts
        ]
        expl_quality = sum(expl_scores) / len(expl_scores) if expl_scores else 0.0

        # 4. Source citation quality
        src_scores = [
            source_citation_score(v.get("explanation", ""))
            for v in verdicts
        ]
        src_quality = sum(src_scores) / len(src_scores) if src_scores else 0.0

        # 5. Parsimony
        pars = parsimony_score(token_count)

        # Weighted composite
        w = {
            "claim_recall": 0.25,
            "verdict_agreement": 0.30,
            "explanation_quality": 0.20,
            "source_citation_quality": 0.15,
            "parsimony": 0.10,
        }
        fitness = (
            w["claim_recall"] * recall
            + w["verdict_agreement"] * verdict_agreement
            + w["explanation_quality"] * expl_quality
            + w["source_citation_quality"] * src_quality
            + w["parsimony"] * pars
        )

        return {
            "claim_recall": round(recall, 4),
            "verdict_agreement": round(verdict_agreement, 4),
            "explanation_quality": round(expl_quality, 4),
            "source_citation_quality": round(src_quality, 4),
            "parsimony": round(pars, 4),
            "fitness": round(fitness, 4),
            "matched_count": sum(1 for m in matches if m["matched"]),
            "total_extracted": len(checkable),
        }

    def score_extraction_only(self, extracted_claims: list[dict], token_count: int = 0) -> dict:
        """
        Score only extraction quality when synthesis hasn't been run.
        Verdict agreement is 0, synthesis scores are 0.
        """
        checkable = [c for c in extracted_claims if c.get("is_checkable", True)]
        matches = match_claims_to_reference(checkable, self._reference)
        recall = sum(1 for m in matches if m["matched"]) / len(self._reference)
        pars = parsimony_score(token_count)

        fitness = 0.25 * recall + 0.10 * pars
        return {
            "claim_recall": round(recall, 4),
            "verdict_agreement": 0.0,
            "explanation_quality": 0.0,
            "source_citation_quality": 0.0,
            "parsimony": round(pars, 4),
            "fitness": round(fitness, 4),
            "matched_count": sum(1 for m in matches if m["matched"]),
            "total_extracted": len(checkable),
        }

    def _find_verdict(self, claim_text: str | None, verdicts: list[dict]) -> dict | None:
        """Find the best-matching verdict for a claim text."""
        if not claim_text or not verdicts:
            return None
        best_sim = 0.0
        best = None
        for v in verdicts:
            vtext = v.get("claim_text", "")
            sim = fuzzy_claim_similarity(claim_text, vtext)
            if sim > best_sim:
                best_sim = sim
                best = v
        return best if best_sim > 0.1 else None
