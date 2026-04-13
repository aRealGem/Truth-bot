"""
Verification engine — orchestrates evidence gathering and verdict synthesis.

For each claim:
  1. Query all configured source connectors in parallel (thread pool)
  2. Collect and deduplicate evidence
  3. Send evidence + claim to LLM for verdict synthesis
  4. Return a Verdict with label, confidence, and explanation

This module is a stub — the orchestration logic is defined, but the LLM
synthesis call returns a placeholder verdict until an API key is set.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from truthbot.models import Claim, Confidence, Evidence, SourceTier, Verdict, VerdictLabel
from truthbot.verify.sources.base import SourceConnector

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM = """You are an expert fact-checker. Given a claim and a set of evidence snippets,
determine the verdict according to this taxonomy:

  - True: Accurate and supported by primary sources
  - Mostly True: Accurate but missing nuance
  - Misleading: Technically accurate framing that implies something false
  - Exaggerated: Directionally correct but overstated
  - False: Contradicted by credible evidence
  - Unverifiable: Insufficient evidence

Respond with a JSON object:
{
  "label": "<verdict>",
  "confidence": "High|Medium|Low",
  "explanation": "<one paragraph explanation>",
  "support_count": <int>,
  "contradict_count": <int>
}"""


class VerificationEngine:
    """
    Orchestrates multi-source evidence gathering and LLM verdict synthesis.

    Parameters
    ----------
    connectors:
        List of SourceConnector instances to query. If empty, uses defaults.
    api_key:
        Anthropic API key for verdict synthesis.
    model:
        LLM model identifier.
    max_workers:
        Thread pool size for parallel source queries.
    """

    def __init__(
        self,
        connectors: Optional[list[SourceConnector]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_workers: int = 4,
    ) -> None:
        import os
        from truthbot.config import settings

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model or settings.llm_model
        self._max_workers = max_workers

        if connectors is not None:
            self._connectors = connectors
        else:
            self._connectors = self._default_connectors()

    def verify(self, claim: Claim) -> tuple[list[Evidence], Verdict]:
        """
        Gather evidence and synthesize a verdict for a single claim.

        Parameters
        ----------
        claim:
            The claim to verify.

        Returns
        -------
        tuple[list[Evidence], Verdict]
            All evidence gathered and the synthesized verdict.
        """
        evidence = self._gather_evidence(claim)
        verdict = self._synthesize_verdict(claim, evidence)
        return evidence, verdict

    def verify_many(
        self, claims: list[Claim]
    ) -> list[tuple[Claim, list[Evidence], Verdict]]:
        """
        Verify a list of claims, returning results in input order.

        Parameters
        ----------
        claims:
            Claims to verify.

        Returns
        -------
        list[tuple[Claim, list[Evidence], Verdict]]
            Results for each claim.
        """
        results = []
        for claim in claims:
            if not claim.is_checkable:
                verdict = self._unverifiable_verdict(claim, "Opinion or value judgment — not checkable.")
                results.append((claim, [], verdict))
                continue
            evidence, verdict = self.verify(claim)
            results.append((claim, evidence, verdict))
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gather_evidence(self, claim: Claim) -> list[Evidence]:
        """Query all connectors in parallel and collect results."""
        all_evidence: list[Evidence] = []
        available = [c for c in self._connectors if c.is_available()]

        if not available:
            logger.warning("No evidence sources available for claim %s", claim.id)
            return []

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(available))) as pool:
            futures = {pool.submit(c.search, claim): c for c in available}
            for future in as_completed(futures):
                connector = futures[future]
                try:
                    results = future.result()
                    all_evidence.extend(results)
                    logger.debug(
                        "%s returned %d evidence items for claim %s",
                        connector.source_name,
                        len(results),
                        claim.id,
                    )
                except Exception as exc:
                    logger.error(
                        "Connector %s raised an exception: %s",
                        connector.source_name,
                        exc,
                    )

        return all_evidence

    def _synthesize_verdict(self, claim: Claim, evidence: list[Evidence]) -> Verdict:
        """
        Use LLM to synthesize a verdict from collected evidence.

        Falls back to a stub verdict if no API key is set.
        """
        if not self._api_key or not evidence:
            return self._stub_verdict(claim, evidence)

        try:
            return self._call_llm_for_verdict(claim, evidence)
        except Exception as exc:
            logger.error("Verdict synthesis failed for claim %s: %s", claim.id, exc)
            return self._stub_verdict(claim, evidence)

    def _call_llm_for_verdict(self, claim: Claim, evidence: list[Evidence]) -> Verdict:
        """Make the Anthropic API call to synthesize a verdict."""
        import json
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        evidence_text = "\n\n".join(
            f"[{i+1}] {e.source_name} ({e.source_tier.value})\n{e.snippet}"
            for i, e in enumerate(evidence[:10])
        )
        user_msg = (
            f"Claim: {claim.text}\n\n"
            f"Evidence:\n{evidence_text}"
        )

        message = client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = json.loads(message.content[0].text)
        label = VerdictLabel(raw["label"])
        confidence = Confidence(raw["confidence"])

        # Determine best source tier
        best_tier = max(
            (e.source_tier for e in evidence),
            key=lambda t: list(SourceTier).index(t),
            default=None,
        )

        return Verdict(
            claim_id=claim.id,
            label=label,
            confidence=confidence,
            explanation=raw["explanation"],
            evidence_ids=[e.id for e in evidence],
            support_count=raw.get("support_count", 0),
            contradict_count=raw.get("contradict_count", 0),
            primary_source_tier=best_tier,
        )

    def _stub_verdict(self, claim: Claim, evidence: list[Evidence]) -> Verdict:
        """Return a placeholder verdict (no API key or no evidence)."""
        return Verdict(
            claim_id=claim.id,
            label=VerdictLabel.UNVERIFIABLE,
            confidence=Confidence.LOW,
            explanation=(
                "Verdict synthesis is not available (stub mode). "
                "Configure ANTHROPIC_API_KEY for live fact-checking."
            ),
            evidence_ids=[e.id for e in evidence],
        )

    def _unverifiable_verdict(self, claim: Claim, reason: str) -> Verdict:
        """Mark a claim as unverifiable with a given reason."""
        return Verdict(
            claim_id=claim.id,
            label=VerdictLabel.UNVERIFIABLE,
            confidence=Confidence.HIGH,
            explanation=reason,
        )

    def _default_connectors(self) -> list[SourceConnector]:
        """Build the default connector stack from configured env vars."""
        from truthbot.verify.sources.brave import BraveSearchConnector
        from truthbot.verify.sources.factcheck import FactCheckConnector
        from truthbot.verify.sources.government import GovernmentDataConnector

        return [
            GovernmentDataConnector(),
            FactCheckConnector(),
            BraveSearchConnector(),
        ]
