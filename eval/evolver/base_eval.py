"""
Shared base class for opus_eval.py and gpt_eval.py.

Defines:
  - EXTRACTION_SYSTEM, EXTRACTION_USER: shared prompt constants
  - SYNTHESIS_SYSTEM, SYNTHESIS_USER: shared prompt constants
  - ModelClient protocol
  - BaseEvalRunner: full extraction + synthesis + scoring pipeline
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── Shared prompt constants ────────────────────────────────────────────────────
# Single source of truth. Both opus_eval.py and gpt_eval.py import from here.

EXTRACTION_SYSTEM = (
    "You are a senior researcher at a major fact-checking organization (PolitiFact / FactCheck.org). "
    "You have 20 years of experience decomposing political speeches into discrete, independently verifiable factual claims.\n\n"
    "A claim is checkable if it: (1) references a specific measurable outcome (number, percentage, dollar amount, rate, "
    "count, or named comparison like 'lowest in X years'), (2) could be confirmed or refuted by a public data source "
    "(BLS, BEA, EIA, CBP, Census, Freddie Mac, court records), (3) is NOT purely a statement of intent, opinion, or "
    "prediction without empirical basis, (4) is NOT a rhetorical flourish or vague superlative.\n\n"
    "EXTRACTION RULES:\n"
    "1. Extract ATOMIC claims - split compound sentences with multiple assertions\n"
    "2. Restate each claim as a clear, self-contained declarative sentence\n"
    "3. Preserve numbers, percentages, time ranges, and named entities exactly\n"
    "4. Include comparative claims ('lowest since X', 'up Y% since Z') and measurable causal claims\n"
    "5. Do NOT merge multiple assertions into one claim\n"
    "6. Aim for completeness - a missed checkable claim is a failure\n\n"
    "TRICKY CASES:\n"
    "- 'Illegal immigration is at record lows' -> CHECKABLE (measurable benchmark)\n"
    "- 'Fentanyl seizures are down 56%' -> CHECKABLE (specific, measurable)\n"
    "- 'We ended the fentanyl crisis' -> NOT checkable (unmeasurable causal)\n"
    "- 'We have the strongest economy ever' -> NOT checkable (vague superlative)\n\n"
    "OUTPUT FORMAT: Return ONLY a valid JSON array. Each element:\n"
    '{"text":"<claim as standalone sentence>","category":"<inflation|jobs_employment|energy_prices|immigration_border|crime_statistics|mortgage_housing|investment_trade|drug_interdiction|foreign_policy|elections_voting|federal_budget|healthcare_drugs|food_prices|other>","is_checkable":true,"check_confidence":<0.0-1.0>}\n'
    "No preamble. No markdown. Pure JSON array only."
)

EXTRACTION_USER = (
    "Extract all checkable factual claims from this speech transcript.\n\n"
    "TRANSCRIPT:\n{transcript}\n\n"
    "Return ONLY the JSON array."
)

SYNTHESIS_SYSTEM = (
    "You are the chief fact-checking editor at a major nonpartisan news organization. "
    "You evaluate political claims against primary government data, wire services, and established research "
    "with intellectual honesty.\n\n"
    "VERDICT TAXONOMY:\n"
    "  True         - Confirmed accurate by authoritative primary sources\n"
    "  Mostly True  - Broadly correct but with notable caveats or missing context\n"
    "  Misleading   - Technically accurate but framed to convey a false impression; selective facts; cherry-picked timeframe\n"
    "  Exaggerated  - Directionally correct but scale/magnitude significantly overstated\n"
    "  False        - Directly refuted by primary data or multiple credible sources\n"
    "  Unverifiable - Cannot be confirmed or denied with available public evidence\n\n"
    "SOURCE TRUST HIERARCHY:\n"
    "  1. Government primary data (BLS, BEA, EIA, CBP, Census, Freddie Mac) - highest\n"
    "  2. Wire services (AP, Reuters) - high\n"
    "  3. Established outlets (NYT, WaPo, NPR, CBS, BBC) - moderate\n"
    "  4. Fact-checking orgs (PolitiFact, FactCheck.org) - moderate\n\n"
    "REASONING STEPS - work through each before issuing a verdict:\n"
    "  1. CLAIM ANALYSIS: What exactly does this claim assert? What specific measurable assertion is made?\n"
    "  2. EVIDENCE REVIEW: What does the best available evidence say? Note source tier and conflicts.\n"
    "  3. DISCREPANCY CHECK: Gap between what was claimed and what evidence shows? Quantify if possible.\n"
    "  4. FRAMING CHECK: Even if technically accurate, does the claim create a false impression through "
    "selective framing or cherry-picked timeframes?\n"
    "  5. VERDICT SELECTION: Apply taxonomy. When in doubt between adjacent labels, choose the one that "
    "better serves the reader's accurate understanding.\n\n"
    "SPECIAL RULES:\n"
    "- Inherently unmeasurable -> Unverifiable\n"
    "- Technically true but misleading -> Misleading (NOT True)\n"
    "- Comparison claims -> verify BOTH current value AND historical benchmark\n\n"
    "OUTPUT FORMAT: Write your reasoning first (free text), then end with a JSON object as the final element:\n"
    '{"label":"<verdict>","confidence":"High|Medium|Low","explanation":"<one paragraph for general audience>","support_count":<int>,"contradict_count":<int>}\n'
    "The JSON must be the last thing in your response. No text after the closing brace."
)

SYNTHESIS_USER = (
    "Claim: {claim_text}\n\n"
    "Evidence:\n{evidence_block}\n\n"
    "[Reference verdict from independent fact-checkers: {reference_verdict}]\n\n"
    "Work through the reasoning steps, then output your JSON verdict."
)


# ── Model client protocol ──────────────────────────────────────────────────────

@runtime_checkable
class ModelClient(Protocol):
    """Protocol for model-agnostic API client."""

    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        """
        Call the model and return (response_text, input_tokens, output_tokens).
        Raises on API error. Returns empty string on parse/content issues.
        """
        ...


# ── Shared helper functions ────────────────────────────────────────────────────

def _ck(prefix: str, content: str) -> str:
    return f"{prefix}_{hashlib.sha256(content.encode()).hexdigest()[:16]}"


def _load_cache(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def build_evidence(ref_claim: dict) -> str:
    """Build evidence block from a reference claim dict."""
    parts = []
    explanation = ref_claim.get("explanation", "")
    if explanation:
        parts.append(f"[Fact-checker note] {explanation}")
    sources = ref_claim.get("sources", [])
    if sources:
        parts.append(f"[Referenced sources: {', '.join(str(s) for s in sources)}]")
    return "\n\n".join(parts) if parts else "[No external evidence -- rely on training knowledge]"


def extract_json_obj(text: str) -> dict | None:
    last = text.rfind("}")
    if last == -1:
        return None
    first = text.rfind("{", 0, last + 1)
    if first == -1:
        return None
    try:
        return json.loads(text[first:last + 1])
    except Exception:
        return None


def extract_json_arr(text: str) -> list | None:
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            pass
    f, l = text.find("["), text.rfind("]")
    if f == -1 or l == -1:
        return None
    try:
        return json.loads(text[f:l + 1])
    except Exception:
        return None


# ── Base eval runner ───────────────────────────────────────────────────────────

class BaseEvalRunner:
    """
    Full extraction + synthesis + scoring pipeline.

    Used by both opus_eval.py (AnthropicClient) and gpt_eval.py (OpenAIClient).
    """

    MAX_TRANSCRIPT_CHARS = 12_000
    SYNTHESIS_MAX_TOKENS = 1500
    EXTRACTION_MAX_TOKENS = 4096
    INTER_CALL_DELAY = 0.35

    def __init__(
        self,
        client: ModelClient,
        model: str,
        results_dir: Path,
        cache_prefix: str,
    ) -> None:
        self._client = client
        self._model = model
        self._results_dir = results_dir
        self._cache_prefix = cache_prefix
        self._cache_dir = results_dir / "runner_cache"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, transcript: str, reference: list[dict]) -> dict:
        """Full extraction + synthesis + scoring. Returns a results dict."""
        from evolver.fitness import (
            FitnessScorer,
            match_claims_to_reference,
            verdict_agreement_score,
        )

        # 1. Extraction
        extracted, total_it, total_ot = self._run_extraction(transcript)

        # 2. Match to reference
        ref_by_id = {r["id"]: r for r in reference}
        match_results = match_claims_to_reference(extracted, reference, threshold=0.15)
        matched = [m for m in match_results if m["matched"]]
        recall = len(matched) / len(reference) if reference else 0.0
        logger.info("Recall: %d/%d (%.1f%%)", len(matched), len(reference), recall * 100)

        # 3. Synthesis
        verdicts, syn_it, syn_ot = self._run_synthesis(matched, ref_by_id)
        total_it += syn_it
        total_ot += syn_ot

        # 4. Score
        results = self._score_results(
            extracted, matched, verdicts,
            total_it, total_ot, recall, reference
        )

        # 5. Save
        out_path = self._save_results(results)
        self._print_summary(results)
        logger.info("Saved -> %s", out_path)
        return results

    def _run_extraction(self, transcript: str) -> tuple[list[dict], int, int]:
        """Run claim extraction. Returns (claims, input_tokens, output_tokens)."""
        key = _ck(f"{self._cache_prefix}_ext", transcript[:self.MAX_TRANSCRIPT_CHARS] + self._model)
        cached = _load_cache(self._cache_dir / f"{key}.json")
        if cached:
            logger.info("Extraction: cache hit (%d claims)", len(cached.get("claims", [])))
            return cached["claims"], cached.get("it", 0), cached.get("ot", 0)

        logger.info("Extraction: calling %s ...", self._model)
        raw, it, ot = self._client.complete(
            system=EXTRACTION_SYSTEM,
            user=EXTRACTION_USER.format(transcript=transcript[:self.MAX_TRANSCRIPT_CHARS]),
            max_tokens=self.EXTRACTION_MAX_TOKENS,
        )
        logger.info("Extraction: %d in / %d out tokens", it, ot)
        claims = extract_json_arr(raw) or []
        claims = [c for c in claims if isinstance(c, dict) and c.get("is_checkable", True)]
        logger.info("Extraction: %d checkable claims", len(claims))
        _save_cache(self._cache_dir / f"{key}.json", {
            "claims": claims, "raw": raw, "it": it, "ot": ot, "model": self._model
        })
        return claims, it, ot

    def _run_synthesis(
        self,
        matched: list[dict],
        ref_by_id: dict,
    ) -> tuple[list[dict], int, int]:
        """
        Run verdict synthesis for all matched claims.
        Returns (verdicts, total_input_tokens, total_output_tokens).
        """
        verdicts = []
        total_it = total_ot = 0

        for i, m in enumerate(matched):
            claim_text = m["matched_claim"] or m["ref_claim"]
            ref_verdict = m["ref_verdict"]
            ref_full = ref_by_id.get(m["ref_id"], {})
            evidence = build_evidence(ref_full)

            logger.info("Synthesis [%d/%d]: %s ...", i + 1, len(matched), claim_text[:55])

            key = _ck(f"{self._cache_prefix}_syn", self._model + "|" + claim_text + "|" + evidence)
            cached = _load_cache(self._cache_dir / f"{key}.json")
            if cached:
                verdicts.append(self._make_verdict_record(
                    claim_text, ref_verdict, cached
                ))
                total_it += cached.get("it", 0)
                total_ot += cached.get("ot", 0)
                continue

            try:
                raw, it, ot = self._client.complete(
                    system=SYNTHESIS_SYSTEM,
                    user=SYNTHESIS_USER.format(
                        claim_text=claim_text,
                        evidence_block=evidence,
                        reference_verdict=ref_verdict,
                    ),
                    max_tokens=self.SYNTHESIS_MAX_TOKENS,
                )
                v = extract_json_obj(raw)
                if not v:
                    logger.warning("JSON parse failed for: %s", claim_text[:60])
                    v = {
                        "label": "Unverifiable",
                        "confidence": "Low",
                        "explanation": "Parse failure.",
                        "support_count": 0,
                        "contradict_count": 0,
                    }
                result = {**v, "raw": raw[:500], "it": it, "ot": ot}
            except Exception as e:
                logger.error("Synthesis error: %s", e)
                result = {
                    "label": "Unverifiable", "confidence": "Low",
                    "explanation": str(e), "support_count": 0,
                    "contradict_count": 0, "it": 0, "ot": 0,
                }

            _save_cache(self._cache_dir / f"{key}.json", result)
            verdicts.append(self._make_verdict_record(claim_text, ref_verdict, result))
            total_it += result.get("it", 0)
            total_ot += result.get("ot", 0)
            time.sleep(self.INTER_CALL_DELAY)

        return verdicts, total_it, total_ot

    def _make_verdict_record(self, claim_text: str, ref_verdict: str, v: dict) -> dict:
        from evolver.fitness import verdict_agreement_score
        agr = verdict_agreement_score(ref_verdict, v.get("label", "Unverifiable"))
        return {
            "claim_text": claim_text,
            "ref_verdict": ref_verdict,
            "model_label": v.get("label", "?"),
            "confidence": v.get("confidence", "?"),
            "agreement_score": agr,
            "explanation": v.get("explanation", ""),
        }

    def _score_results(
        self,
        extracted: list[dict],
        matched: list[dict],
        verdicts: list[dict],
        total_it: int,
        total_ot: int,
        recall: float,
        reference: list[dict],
    ) -> dict:
        """Compute unified fitness scores using FitnessScorer."""
        from evolver.fitness import FitnessScorer

        scorer = FitnessScorer(reference)
        total_tokens = total_it + total_ot

        # Build verdict dicts in FitnessScorer format
        fitness_verdicts = [
            {
                "claim_text": v["claim_text"],
                "label": v["model_label"],
                "explanation": v["explanation"],
                "support_count": 0,
                "contradict_count": 0,
            }
            for v in verdicts
        ]
        scores = scorer.score(extracted, fitness_verdicts, token_count=total_tokens)

        # Also compute the simplified formula for backwards compatibility / reporting
        va = (
            sum(v["agreement_score"] for v in verdicts) / len(verdicts)
            if verdicts else 0.0
        )
        fitness_approx = recall * 0.25 + va * 0.30

        return {
            "model": self._model,
            "recall": recall,
            "matched": len(matched),
            "total_ref": len(reference),
            "verdict_agreement": va,
            "fitness_approx": fitness_approx,   # simplified formula (max 0.55)
            "fitness_full": scores["fitness"],  # FitnessScorer 5-dim (max 1.0)
            "scores": scores,
            "total_it": total_it,
            "total_ot": total_ot,
            "verdicts": verdicts,
        }

    def _save_results(self, results: dict) -> Path:
        out = self._results_dir / "results.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        return out

    def _print_summary(self, results: dict) -> None:
        verdicts = results.get("verdicts", [])
        total_ref = results.get("total_ref", 1)
        matched = results.get("matched", 0)
        recall = results.get("recall", 0.0)
        va = results.get("verdict_agreement", 0.0)
        total_it = results.get("total_it", 0)
        total_ot = results.get("total_ot", 0)

        print()
        print("=" * 72)
        print(f"  EVAL -- {self._model}")
        print("=" * 72)
        print(f"  Claim recall:       {matched}/{total_ref}  ({recall*100:.1f}%)")
        print(f"  Verdict agreement:  {va:.4f}  ({va*100:.1f}%)")
        print(f"  Fitness (approx):   {results.get('fitness_approx', 0.0):.4f}")
        print(f"  Fitness (full):     {results.get('fitness_full', 0.0):.4f}")
        print(f"  Tokens:             {total_it:,} in / {total_ot:,} out")
        print()
        header_label = "MODEL VERDICT"
        print(f"  {'OK':3}  {'REF VERDICT':<22}  {header_label:<16}  CLAIM (first 55 chars)")
        print(f"  {'-'*3}  {'-'*22}  {'-'*16}  {'-'*55}")
        for v in sorted(verdicts, key=lambda x: x.get("agreement_score", 0)):
            icon = "YES" if v.get("agreement_score", 0) >= 0.8 else (
                "~  " if v.get("agreement_score", 0) >= 0.4 else "NO "
            )
            print(
                f"  {icon:3}  {v.get('ref_verdict', '?'):<22}  "
                f"{v.get('model_label', '?'):<16}  "
                f"{v.get('claim_text', '')[:55]}"
            )
        print()
        print(f"  Results -> {self._results_dir / 'results.json'}")
        print("=" * 72)
