"""
Verification engine — orchestrates evidence gathering and multi-model verdict synthesis.

For each claim:
  1. Query all configured source connectors in parallel (thread pool)
  2. Collect and deduplicate evidence
  3. Fan out claim + evidence to all active LLM adapters in parallel
  4. Build a consensus verdict from the individual model verdicts
  5. Return ConsensusVerdict
"""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    Evidence,
    ModelVerdict,
    SourceTier,
    Verdict,
    VerdictLabel,
)
from truthbot.verify.sources.base import SourceConnector

logger = logging.getLogger(__name__)

# Tie-breaking order: most conservative wins (lowest index = most conservative)
_TIE_BREAK_ORDER = [
    VerdictLabel.FALSE,
    VerdictLabel.MISLEADING,
    VerdictLabel.EXAGGERATED,
    VerdictLabel.MOSTLY_TRUE,
    VerdictLabel.UNVERIFIABLE,
    VerdictLabel.TRUE,
]


class VerificationEngine:
    """
    Orchestrates multi-source evidence gathering and multi-model verdict synthesis.

    Parameters
    ----------
    connectors:
        List of SourceConnector instances to query. If empty, uses defaults.
    max_workers:
        Thread pool size for parallel source queries and adapter calls.
    """

    def __init__(
        self,
        connectors: Optional[list[SourceConnector]] = None,
        max_workers: int = 4,
        # Legacy params accepted but ignored
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._max_workers = max_workers

        if connectors is not None:
            self._connectors = connectors
        else:
            self._connectors = self._default_connectors()

        self._adapters = self._build_adapters()

    def _build_adapters(self):
        """Instantiate all available adapters, skipping those without API keys."""
        from truthbot.verify.adapters.anthropic import AnthropicAdapter
        from truthbot.verify.adapters.gemini import GeminiAdapter
        from truthbot.verify.adapters.grok import GrokAdapter
        from truthbot.verify.adapters.openai import OpenAIAdapter
        from truthbot.verify.adapters.base import AdapterUnavailable

        adapter_classes = [AnthropicAdapter, OpenAIAdapter, GeminiAdapter, GrokAdapter]
        active = []
        skipped = []

        for cls in adapter_classes:
            try:
                adapter = cls()
                active.append(adapter)
            except AdapterUnavailable as exc:
                skipped.append(f"{cls.adapter_name} ({exc})")
            except Exception as exc:
                skipped.append(f"{cls.adapter_name} (init error: {exc})")

        active_names = [a.adapter_name for a in active]
        logger.info(
            "Active adapters: %s. Skipped: %s",
            active_names if active_names else "none",
            skipped if skipped else "none",
        )

        return active

    def verify(self, claim: Claim) -> tuple[list[Evidence], ConsensusVerdict]:
        """
        Gather evidence and synthesize a consensus verdict for a single claim.

        Parameters
        ----------
        claim:
            The claim to verify.

        Returns
        -------
        tuple[list[Evidence], ConsensusVerdict]
            All evidence gathered and the consensus verdict.
        """
        evidence = self._gather_evidence(claim)

        if not self._adapters:
            logger.warning(
                "No LLM adapters active — returning stub verdict for claim %s", claim.id
            )
            stub = self._stub_verdict(claim, evidence)
            # Wrap stub in ConsensusVerdict
            return evidence, ConsensusVerdict(
                claim_id=claim.id,
                model_verdicts=[],
                consensus_label=stub.label,
                confidence=stub.confidence,
                agreement=True,
                explanation=stub.explanation,
            )

        # Fan out to all adapters in parallel
        model_verdicts: list[ModelVerdict] = []
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(self._adapters))) as pool:
            futures = {
                pool.submit(adapter.call, claim, evidence): adapter
                for adapter in self._adapters
            }
            for future in as_completed(futures):
                adapter = futures[future]
                try:
                    mv = future.result()
                    model_verdicts.append(mv)
                except Exception as exc:
                    logger.error(
                        "Adapter %s raised an exception for claim %s: %s",
                        adapter.adapter_name,
                        claim.id,
                        exc,
                    )

        if not model_verdicts:
            stub = self._stub_verdict(claim, evidence)
            return evidence, ConsensusVerdict(
                claim_id=claim.id,
                model_verdicts=[],
                consensus_label=stub.label,
                confidence=stub.confidence,
                agreement=True,
                explanation=stub.explanation,
            )

        consensus = _build_consensus(claim.id, model_verdicts)
        return evidence, consensus

    def verify_many(
        self, claims: list[Claim]
    ) -> list[tuple[Claim, list[Evidence], ConsensusVerdict]]:
        """
        Verify a list of claims, returning results in input order.

        Parameters
        ----------
        claims:
            Claims to verify.

        Returns
        -------
        list[tuple[Claim, list[Evidence], ConsensusVerdict]]
            Results for each claim.
        """
        results = []
        for claim in claims:
            if not claim.is_checkable:
                verdict = self._unverifiable_verdict(
                    claim, "Opinion or value judgment — not checkable."
                )
                # Wrap in ConsensusVerdict
                consensus = ConsensusVerdict(
                    claim_id=claim.id,
                    model_verdicts=[],
                    consensus_label=verdict.label,
                    confidence=verdict.confidence,
                    agreement=True,
                    explanation=verdict.explanation,
                )
                results.append((claim, [], consensus))
                continue
            evidence, consensus = self.verify(claim)
            results.append((claim, evidence, consensus))
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

    def _stub_verdict(self, claim: Claim, evidence: list[Evidence]) -> Verdict:
        """Return a placeholder verdict (no adapters active or no evidence)."""
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


def _build_consensus(claim_id: str, model_verdicts: list[ModelVerdict]) -> ConsensusVerdict:
    """
    Build a ConsensusVerdict from a list of ModelVerdicts.

    Uses majority vote with tie-breaking (most conservative wins).
    """
    label_counts = Counter(mv.label for mv in model_verdicts)
    max_count = max(label_counts.values())
    tied = [label for label, count in label_counts.items() if count == max_count]

    if len(tied) == 1:
        consensus_label = tied[0]
    else:
        # Most conservative wins
        consensus_label = min(tied, key=lambda l: _TIE_BREAK_ORDER.index(l))

    total = len(model_verdicts)
    all_agree = len(label_counts) == 1
    majority_agree = max_count > total / 2

    if total == 1:
        confidence = model_verdicts[0].confidence
    elif all_agree:
        confidence = Confidence.HIGH
    elif majority_agree:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    agreement = all_agree

    parts = ", ".join(f"{mv.adapter_name}({mv.label.value})" for mv in model_verdicts)
    if all_agree:
        suffix = "unanimous"
    elif majority_agree:
        suffix = "majority"
    else:
        suffix = "split"

    explanation = (
        f"Model verdicts: {parts}. Consensus: {consensus_label.value} [{suffix}]."
    )

    return ConsensusVerdict(
        claim_id=claim_id,
        model_verdicts=model_verdicts,
        consensus_label=consensus_label,
        confidence=confidence,
        agreement=agreement,
        explanation=explanation,
    )
