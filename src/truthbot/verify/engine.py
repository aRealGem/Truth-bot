"""
Verification engine — orchestrates evidence gathering and multi-model verdict synthesis.

For each claim:
  1. Check VerdictBundle cache (key = hash(claim.text + speaker + date))
  2. Evidence via ``EvidenceProvider`` (``TRUTHBOT_EVIDENCE_SOURCE`` / DataHoover stub / connectors).
  3. Optional triage tier, then frontier fan-out to all active LLM adapters via asyncio.gather,
     with per-adapter 120s timeout and full error isolation
  4. Build a VerdictBundle with per-model verdicts and consensus output
  5. Write bundle to cache and return
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    Evidence,
    ModelVerdict,
    SourceTier,
    Verdict,
    VerdictBundle,
    VerdictLabel,
)
from truthbot.verify.evidence_provider import EvidenceProvider, build_evidence_provider
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

_ADAPTER_TIMEOUT_SECONDS = 120.0


def _cache_key(claim_text: str, speaker: str, date_str: str) -> str:
    """Deterministic cache key for a (claim, speaker, date) triple."""
    raw = f"{claim_text.strip().lower()}|{speaker.strip().lower()}|{date_str.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _build_consensus(claim_id: str, model_verdicts: list[ModelVerdict]) -> ConsensusVerdict:
    """
    Build a ConsensusVerdict from a list of ModelVerdicts.

    Strength rules:
      single  — exactly 1 model active
      strong  — ≥3 models return the same label
      weak    — exactly 2 models agree, others split
      none    — no majority (all different or symmetric split)

    Tie-breaking: most conservative label wins.
    """
    n = len(model_verdicts)

    if n == 0:
        return ConsensusVerdict(
            claim_id=claim_id,
            model_verdicts=[],
            consensus_label=VerdictLabel.UNVERIFIABLE,
            consensus_verdict="Models split",
            confidence=Confidence.LOW,
            agreement=False,
            consensus_strength="none",
            explanation="No model verdicts returned.",
        )

    label_counts = Counter(mv.label for mv in model_verdicts)
    max_count = max(label_counts.values())
    tied_labels = [lbl for lbl, cnt in label_counts.items() if cnt == max_count]

    # Pick consensus label (ties broken conservatively)
    if len(tied_labels) == 1:
        consensus_label = tied_labels[0]
    else:
        consensus_label = min(tied_labels, key=lambda l: _TIE_BREAK_ORDER.index(l))

    all_agree = len(label_counts) == 1

    # Consensus strength
    if n == 1:
        strength = "single"
    elif max_count >= 3:
        strength = "strong"
    elif max_count == 2:
        # Only "weak" if the 2 agreeing models are a strict plurality, others split
        strength = "weak"
    else:
        strength = "none"

    # Confidence
    if n == 1:
        confidence = model_verdicts[0].confidence
    elif all_agree:
        confidence = Confidence.HIGH
    elif max_count > n / 2:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    # Explanation
    parts = ", ".join(f"{mv.adapter_name}({mv.label.value})" for mv in model_verdicts)
    if all_agree:
        suffix = "unanimous"
    elif strength == "strong":
        suffix = "strong majority"
    elif strength == "weak":
        suffix = "weak majority"
    else:
        suffix = "split — no consensus"

    explanation = (
        f"Model verdicts: {parts}. "
        f"Consensus: {consensus_label.value} [{suffix}]."
    )

    return ConsensusVerdict(
        claim_id=claim_id,
        model_verdicts=model_verdicts,
        consensus_label=consensus_label,
        consensus_verdict=(consensus_label.value if strength != "none" else "Models split"),
        confidence=confidence,
        agreement=all_agree,
        consensus_strength=strength,
        explanation=explanation,
    )


class VerificationEngine:
    """
    Orchestrates evidence gathering and multi-model verdict synthesis.

    Parameters
    ----------
    connectors:
        Connectors used when ``evidence_source`` is ``connectors`` (see ``Settings``).
    evidence_provider:
        Optional override; otherwise chosen from ``TRUTHBOT_EVIDENCE_SOURCE``.
    max_workers:
        Thread pool size for evidence gathering.
    cache_dir:
        Directory for the VerdictBundle disk cache. Defaults to settings.cache_dir/bundles.
    """

    def __init__(
        self,
        connectors: Optional[list[SourceConnector]] = None,
        max_workers: int = 4,
        cache_dir: Optional[Path] = None,
        # Legacy params accepted but ignored
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        evidence_provider: Optional[EvidenceProvider] = None,
        inject_evidence: bool = True,
        triage_enabled: bool = False,
        triage_threshold: float = 0.8,
        triage_shadow_rate: float = 0.0,
        triage_seed: Optional[int] = None,
        run_id: Optional[str] = None,
        verify_mode: str = "live",
    ) -> None:
        self._max_workers = max_workers
        self._connectors = connectors if connectors is not None else self._default_connectors()
        from truthbot.config import settings

        src = settings.evidence_source
        self._evidence_provider = evidence_provider or build_evidence_provider(
            source=src,
            connectors=self._connectors,
            max_workers=max_workers,
        )
        self._inject_evidence = inject_evidence
        self._triage_enabled = triage_enabled
        self._triage_threshold = triage_threshold
        self._triage_shadow_rate = triage_shadow_rate
        self._triage_rng = random.Random(triage_seed if triage_seed is not None else 0)
        self._triage_adapters: list = []
        if triage_enabled:
            from truthbot.verify.triage import build_triage_adapters

            self._triage_adapters = build_triage_adapters()
        self._run_id = run_id
        self._verify_mode = verify_mode
        self._adapters = self._build_adapters()
        self._bundle_cache = self._init_cache(cache_dir)

    # ── Public interface ──────────────────────────────────────────────────────

    def verify_bundle(
        self,
        claim: Claim,
        speaker: str = "",
        date_str: str = "",
    ) -> VerdictBundle:
        """
        Full pipeline for one claim: cache check → evidence → adapter fan-out → bundle.

        Parameters
        ----------
        claim:
            The claim to verify.
        speaker:
            Speaker name (used in cache key).
        date_str:
            Speech date as YYYY-MM-DD string (used in cache key).

        Returns
        -------
        VerdictBundle
            Full per-model verdicts, consensus, and cache metadata.
        """
        key = _cache_key(claim.text, speaker, date_str)

        # Cache hit
        if self._bundle_cache is not None:
            cached = self._bundle_cache.get(key)
            if cached:
                try:
                    bundle = VerdictBundle.model_validate_json(cached)
                    bundle.cache_hit = True
                    logger.info("Cache HIT for claim %s (key %s)", claim.id, key[:8])
                    return bundle
                except Exception as exc:
                    logger.warning("Cache entry corrupt, re-verifying: %s", exc)

        from truthbot.verify.triage import (
            run_triage_fan_out,
            should_shadow_sample,
            triage_unanimous_high_conf,
        )

        evidence = self._evidence_provider.get_evidence(claim)

        shadow = (
            self._triage_enabled
            and should_shadow_sample(self._triage_shadow_rate, self._triage_rng)
        )

        if self._triage_enabled and self._triage_adapters and not shadow:
            triage_verdicts = run_triage_fan_out(
                self._triage_adapters,
                claim,
                evidence,
                inject_evidence=self._inject_evidence,
                run_id=self._run_id,
            )
            if triage_unanimous_high_conf(triage_verdicts, self._triage_threshold):
                for v in triage_verdicts:
                    v.tier = "triage"
                    v.synthesis_mode = self._verify_mode
                consensus = _build_consensus(claim.id, triage_verdicts)
                bundle = VerdictBundle(
                    claim=claim,
                    speaker=speaker,
                    date_str=date_str,
                    model_verdicts=triage_verdicts,
                    consensus=consensus,
                    evidence_count=len(evidence),
                    cache_hit=False,
                    triage_skipped_frontier=True,
                )
                if self._bundle_cache is not None:
                    try:
                        self._bundle_cache.set(key, bundle.model_dump_json())
                    except Exception as exc:
                        logger.warning("Failed to write bundle to cache: %s", exc)
                return bundle

        model_verdicts = self._run_fan_out(claim, evidence)
        if shadow and self._triage_enabled:
            for v in model_verdicts:
                v.tier = "frontier_shadow"

        consensus = _build_consensus(claim.id, model_verdicts)

        bundle = VerdictBundle(
            claim=claim,
            speaker=speaker,
            date_str=date_str,
            model_verdicts=model_verdicts,
            consensus=consensus,
            evidence_count=len(evidence),
            cache_hit=False,
            triage_skipped_frontier=False,
        )

        if self._bundle_cache is not None:
            try:
                self._bundle_cache.set(key, bundle.model_dump_json())
            except Exception as exc:
                logger.warning("Failed to write bundle to cache: %s", exc)

        return bundle

    # ── Split-path helpers for batch-mode submit/reconcile ────────────────────

    @property
    def adapters(self) -> list:
        """Read-only view of the active frontier adapters."""
        return list(self._adapters)

    @property
    def evidence_provider(self) -> EvidenceProvider:
        """Expose the evidence provider so batch mode can prefetch before submit."""
        return self._evidence_provider

    def maybe_resolve_early(
        self,
        claim: Claim,
        speaker: str = "",
        date_str: str = "",
    ) -> tuple[Optional[VerdictBundle], list[Evidence]]:
        """
        Try to resolve a claim without frontier fan-out (cache hit or triage short-circuit).

        Returns a tuple ``(bundle, evidence)`` — if ``bundle`` is non-None the
        claim is done (already cached or resolved unanimously at triage); the
        caller should NOT dispatch a frontier call. Otherwise ``bundle is None``
        and the caller should submit the claim to the batch/live frontier with
        the returned evidence list.

        Side effect: successful triage bundles are written to the on-disk
        bundle cache just like the normal ``verify_bundle`` flow.
        """
        key = _cache_key(claim.text, speaker, date_str)

        if self._bundle_cache is not None:
            cached = self._bundle_cache.get(key)
            if cached:
                try:
                    bundle = VerdictBundle.model_validate_json(cached)
                    bundle.cache_hit = True
                    logger.info("maybe_resolve_early: cache HIT for %s", claim.id)
                    return bundle, []
                except Exception as exc:
                    logger.warning("maybe_resolve_early: cache entry corrupt: %s", exc)

        from truthbot.verify.triage import (
            run_triage_fan_out,
            should_shadow_sample,
            triage_unanimous_high_conf,
        )

        evidence = self._evidence_provider.get_evidence(claim)
        shadow = (
            self._triage_enabled
            and should_shadow_sample(self._triage_shadow_rate, self._triage_rng)
        )
        if self._triage_enabled and self._triage_adapters and not shadow:
            triage_verdicts = run_triage_fan_out(
                self._triage_adapters,
                claim,
                evidence,
                inject_evidence=self._inject_evidence,
                run_id=self._run_id,
            )
            if triage_unanimous_high_conf(triage_verdicts, self._triage_threshold):
                for v in triage_verdicts:
                    v.tier = "triage"
                    v.synthesis_mode = self._verify_mode
                consensus = _build_consensus(claim.id, triage_verdicts)
                bundle = VerdictBundle(
                    claim=claim,
                    speaker=speaker,
                    date_str=date_str,
                    model_verdicts=triage_verdicts,
                    consensus=consensus,
                    evidence_count=len(evidence),
                    cache_hit=False,
                    triage_skipped_frontier=True,
                )
                if self._bundle_cache is not None:
                    try:
                        self._bundle_cache.set(key, bundle.model_dump_json())
                    except Exception as exc:
                        logger.warning("Failed to write bundle to cache: %s", exc)
                return bundle, evidence

        return None, evidence

    def finalize_bundle(
        self,
        claim: Claim,
        speaker: str,
        date_str: str,
        model_verdicts: list[ModelVerdict],
        evidence_count: int = 0,
    ) -> VerdictBundle:
        """
        Build a ``VerdictBundle`` from externally-collected model verdicts and cache it.

        Used by the batch reconcile path: after parsing provider batch results
        and merging any live-sidecar verdicts, call this to produce a
        consensus + cached bundle that the SitePublisher can consume.
        """
        consensus = _build_consensus(claim.id, model_verdicts)
        bundle = VerdictBundle(
            claim=claim,
            speaker=speaker,
            date_str=date_str,
            model_verdicts=model_verdicts,
            consensus=consensus,
            evidence_count=evidence_count,
            cache_hit=False,
            triage_skipped_frontier=False,
        )
        if self._bundle_cache is not None:
            key = _cache_key(claim.text, speaker, date_str)
            try:
                self._bundle_cache.set(key, bundle.model_dump_json())
            except Exception as exc:
                logger.warning("finalize_bundle: cache write failed: %s", exc)
        return bundle

    def verify(self, claim: Claim) -> tuple[list[Evidence], ConsensusVerdict]:
        """
        Legacy interface: returns (evidence, ConsensusVerdict).
        Prefer verify_bundle() for new code.
        """
        evidence = self._evidence_provider.get_evidence(claim)
        if not self._adapters:
            logger.warning("No LLM adapters active — returning stub verdict for claim %s", claim.id)
            stub = self._stub_verdict(claim, evidence)
            return evidence, ConsensusVerdict(
                claim_id=claim.id,
                model_verdicts=[],
                consensus_label=stub.label,
                consensus_verdict=stub.label.value,
                confidence=stub.confidence,
                agreement=True,
                consensus_strength="none",
                explanation=stub.explanation,
            )
        model_verdicts = self._run_fan_out(claim, evidence)
        if not model_verdicts:
            stub = self._stub_verdict(claim, evidence)
            return evidence, ConsensusVerdict(
                claim_id=claim.id,
                model_verdicts=[],
                consensus_label=stub.label,
                consensus_verdict=stub.label.value,
                confidence=stub.confidence,
                agreement=True,
                consensus_strength="none",
                explanation=stub.explanation,
            )
        return evidence, _build_consensus(claim.id, model_verdicts)

    def verify_many(
        self, claims: list[Claim]
    ) -> list[tuple[Claim, list[Evidence], ConsensusVerdict]]:
        """Verify a list of claims sequentially, skipping non-checkable ones."""
        results = []
        for claim in claims:
            if not claim.is_checkable:
                consensus = ConsensusVerdict(
                    claim_id=claim.id,
                    model_verdicts=[],
                    consensus_label=VerdictLabel.UNVERIFIABLE,
                    confidence=Confidence.HIGH,
                    agreement=True,
                    consensus_strength="none",
                    explanation="Opinion or value judgment — not checkable.",
                )
                results.append((claim, [], consensus))
                continue
            evidence, consensus = self.verify(claim)
            results.append((claim, evidence, consensus))
        return results

    # ── Async fan-out ─────────────────────────────────────────────────────────

    async def _async_fan_out(
        self, claim: Claim, evidence: list[Evidence]
    ) -> list[ModelVerdict]:
        """Dispatch all active adapters concurrently with per-adapter timeout."""

        async def run_one(adapter) -> ModelVerdict:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    lambda a=adapter: a.call(
                        claim,
                        evidence,
                        inject_evidence=self._inject_evidence,
                        run_id=self._run_id,
                    )
                ),
                timeout=_ADAPTER_TIMEOUT_SECONDS,
            )

        tasks = [run_one(adapter) for adapter in self._adapters]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        model_verdicts: list[ModelVerdict] = []
        for adapter, result in zip(self._adapters, results):
            if isinstance(result, Exception):
                logger.error(
                    "Adapter %s failed for claim %s: %s",
                    adapter.adapter_name,
                    claim.id,
                    result,
                )
            else:
                model_verdicts.append(result)

        return model_verdicts

    def _run_fan_out(self, claim: Claim, evidence: list[Evidence]) -> list[ModelVerdict]:
        """Sync wrapper around the async fan-out. Handles nested event loop edge cases."""
        if not self._adapters:
            return []
        try:
            return asyncio.run(self._async_fan_out(claim, evidence))
        except RuntimeError:
            # Nested event loop (e.g. pytest-asyncio, Jupyter) — fall back to thread pool
            logger.debug("Falling back to ThreadPoolExecutor (nested event loop detected)")
            model_verdicts: list[ModelVerdict] = []
            with ThreadPoolExecutor(max_workers=len(self._adapters)) as pool:
                futures = {
                    pool.submit(
                        a.call,
                        claim,
                        evidence,
                        inject_evidence=self._inject_evidence,
                        run_id=self._run_id,
                    ): a
                    for a in self._adapters
                }
                for future in as_completed(futures, timeout=_ADAPTER_TIMEOUT_SECONDS + 5):
                    adapter = futures[future]
                    try:
                        model_verdicts.append(future.result(timeout=_ADAPTER_TIMEOUT_SECONDS))
                    except Exception as exc:
                        logger.error(
                            "Adapter %s failed for claim %s: %s",
                            adapter.adapter_name,
                            claim.id,
                            exc,
                        )
            return model_verdicts

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_adapters(self) -> list:
        """Instantiate all available adapters, skipping those without API keys."""
        from truthbot.verify.adapters.anthropic import AnthropicAdapter
        from truthbot.verify.adapters.base import AdapterUnavailable
        from truthbot.verify.adapters.gemini import GeminiAdapter
        from truthbot.verify.adapters.grok import GrokAdapter
        from truthbot.verify.adapters.openai import OpenAIAdapter

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

        logger.info(
            "Active adapters: %s. Skipped: %s",
            [a.adapter_name for a in active] if active else "none",
            skipped if skipped else "none",
        )
        return active

    def _init_cache(self, cache_dir: Optional[Path]):
        """Initialise the diskcache for VerdictBundle objects."""
        try:
            import diskcache
            from truthbot.config import settings
            path = cache_dir or (settings.cache_dir / "bundles")
            path.mkdir(parents=True, exist_ok=True)
            return diskcache.Cache(str(path))
        except Exception as exc:
            logger.warning("VerdictBundle cache unavailable: %s", exc)
            return None

    def save_bundle_to_cache(self, bundle: VerdictBundle) -> None:
        """Write a completed bundle to the on-disk VerdictBundle cache."""
        if self._bundle_cache is None:
            return
        key = _cache_key(bundle.claim.text, bundle.speaker, bundle.date_str)
        try:
            self._bundle_cache.set(key, bundle.model_dump_json())
        except Exception as exc:
            logger.warning("save_bundle_to_cache failed: %s", exc)

    def _gather_evidence(self, claim: Claim) -> list[Evidence]:
        """Legacy hook — delegates to ``EvidenceProvider``."""
        return self._evidence_provider.get_evidence(claim)

    def _stub_verdict(self, claim: Claim, evidence: list[Evidence]) -> Verdict:
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
        return Verdict(
            claim_id=claim.id,
            label=VerdictLabel.UNVERIFIABLE,
            confidence=Confidence.HIGH,
            explanation=reason,
        )

    def _default_connectors(self) -> list[SourceConnector]:
        from truthbot.verify.sources.brave import BraveSearchConnector
        from truthbot.verify.sources.factcheck import FactCheckConnector
        from truthbot.verify.sources.government import GovernmentDataConnector

        return [
            GovernmentDataConnector(),
            FactCheckConnector(),
            BraveSearchConnector(),
        ]
