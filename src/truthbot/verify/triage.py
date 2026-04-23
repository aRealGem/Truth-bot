"""
Cheap-tier triage models — optional pre-pass before frontier fan-out.

Escalation: all triage models must agree on ``VerdictLabel`` and each mapped
confidence must be >= ``threshold`` (default 0.8: High=1.0, Medium=0.7, Low=0.4).
"""

from __future__ import annotations

import logging
import os
import random
from typing import TYPE_CHECKING

from truthbot.models import Claim, Confidence, Evidence, ModelVerdict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def confidence_numeric(conf: Confidence) -> float:
    return {
        Confidence.HIGH: 1.0,
        Confidence.MEDIUM: 0.7,
        Confidence.LOW: 0.4,
    }[conf]


def triage_unanimous_high_conf(verdicts: list[ModelVerdict], threshold: float) -> bool:
    if len(verdicts) < 2:
        return False
    labels = {v.label for v in verdicts}
    if len(labels) != 1:
        return False
    return min(confidence_numeric(v.confidence) for v in verdicts) >= threshold - 1e-9


def build_triage_adapters() -> list:
    """Instantiate cheap-tier adapters (skip any that cannot init)."""
    from truthbot.verify.adapters.anthropic import AnthropicAdapter
    from truthbot.verify.adapters.base import AdapterUnavailable
    from truthbot.verify.adapters.gemini import GeminiAdapter
    from truthbot.verify.adapters.grok import GrokAdapter
    from truthbot.verify.adapters.openai import OpenAIAdapter

    class TriageAnthropic(AnthropicAdapter):
        model_id = os.environ.get("TRUTHBOT_TRIAGE_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    class TriageOpenAI(OpenAIAdapter):
        model_id = os.environ.get("TRUTHBOT_TRIAGE_OPENAI_MODEL", "gpt-4o-mini")

    class TriageGemini(GeminiAdapter):
        model_id = os.environ.get("TRUTHBOT_TRIAGE_GEMINI_MODEL", "gemini-2.5-flash")

    class TriageGrok(GrokAdapter):
        model_id = os.environ.get("TRUTHBOT_TRIAGE_GROK_MODEL", "grok-3-mini")

    out: list = []
    for cls in (TriageAnthropic, TriageOpenAI, TriageGemini, TriageGrok):
        try:
            out.append(cls())
        except AdapterUnavailable:
            logger.debug("Triage: skip %s (no key)", cls.__name__)
        except Exception as exc:
            logger.warning("Triage: skip %s (%s)", cls.__name__, exc)
    return out


def run_triage_fan_out(
    adapters: list,
    claim: Claim,
    evidence: list[Evidence],
    *,
    inject_evidence: bool,
    run_id: str | None = None,
) -> list[ModelVerdict]:
    """Synchronous parallel triage calls (thread pool)."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not adapters:
        return []

    async def _async_triage() -> list[ModelVerdict]:
        async def one(ad) -> ModelVerdict:
            return await asyncio.to_thread(
                lambda a=ad: a.call(
                    claim,
                    evidence,
                    inject_evidence=inject_evidence,
                    telemetry_tier="triage",
                    run_id=run_id,
                )
            )

        return list(await asyncio.gather(*[one(a) for a in adapters]))

    try:
        return asyncio.run(_async_triage())
    except RuntimeError:
        verdicts: list[ModelVerdict] = []
        with ThreadPoolExecutor(max_workers=len(adapters)) as pool:
            futs = {
                pool.submit(
                    a.call,
                    claim,
                    evidence,
                    inject_evidence=inject_evidence,
                    telemetry_tier="triage",
                    run_id=run_id,
                ): a
                for a in adapters
            }
            for fut in as_completed(futs):
                try:
                    verdicts.append(fut.result())
                except Exception as exc:
                    logger.error("Triage adapter failed: %s", exc)
        return verdicts


def should_shadow_sample(shadow_rate: float, rng: random.Random) -> bool:
    return shadow_rate > 0 and rng.random() < shadow_rate
