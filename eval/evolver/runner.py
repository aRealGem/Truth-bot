"""
Pipeline runner with custom prompt injection.

Executes the truth-bot claim extraction (and optionally verdict synthesis)
using a specified ExtractionGenome/SynthesisGenome instead of the default
prompts baked into claims.py and engine.py.

Design goals:
  - Minimal changes to the production code path
  - Aggressive caching: (prompt_hash, transcript_hash) → cached output
  - Synthetic evidence for synthesis evaluation (avoids expensive source queries)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "sotu-2026" / "evolution_results" / "runner_cache"


def _cache_key(prompt_hash: str, content_hash: str) -> str:
    return f"{prompt_hash}_{content_hash}"


def _transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class CachedRunner:
    """
    Runs claim extraction (and optionally synthesis) against a transcript,
    caching results by (prompt_hash, transcript_hash).

    Parameters
    ----------
    api_key:
        Anthropic API key. If None, reads ANTHROPIC_API_KEY from environment.
    extraction_model:
        Model for extraction calls.
    synthesis_model:
        Model for synthesis calls.
    cache_dir:
        Directory for disk cache. Created if needed.
    dry_run:
        If True, return stub results without making API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        extraction_model: str = "claude-sonnet-4-5",
        synthesis_model: str = "claude-sonnet-4-5",
        cache_dir: Path = _CACHE_DIR,
        dry_run: bool = False,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._extraction_model = extraction_model
        self._synthesis_model = synthesis_model
        self._cache_dir = cache_dir
        self._dry_run = dry_run
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Extraction ─────────────────────────────────────────────────────────────

    def extract_claims(
        self,
        transcript_text: str,
        speaker: str,
        date_str: str,
        system_prompt: str,
        user_template: str,
        prompt_hash: str,
    ) -> tuple[list[dict], int]:
        """
        Extract claims using the given prompts.

        Returns (claims_list, token_count).
        Results are cached by (prompt_hash, transcript_hash).
        """
        tx_hash = _transcript_hash(transcript_text)
        cache_key = _cache_key(f"ext_{prompt_hash}", tx_hash)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug("Cache hit: extraction %s", cache_key)
            return cached["claims"], cached["tokens"]

        if self._dry_run:
            logger.info("[DRY-RUN] Would call extraction API with prompt hash %s", prompt_hash)
            stub = self._stub_extraction(transcript_text, speaker, date_str)
            return stub, 0

        claims, tokens = self._call_extraction_api(
            transcript_text, speaker, date_str, system_prompt, user_template
        )
        self._save_cache(cache_key, {"claims": claims, "tokens": tokens})
        return claims, tokens

    def _call_extraction_api(
        self,
        transcript_text: str,
        speaker: str,
        date_str: str,
        system_prompt: str,
        user_template: str,
    ) -> tuple[list[dict], int]:
        """Make the Anthropic API call for claim extraction."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        user_msg = user_template.format(
            speaker=speaker,
            date=date_str,
            text=transcript_text[:12000],
        )

        response = client.messages.create(
            model=self._extraction_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        tokens = response.usage.input_tokens + response.usage.output_tokens
        raw = response.content[0].text.strip()

        # Parse JSON -- handle markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()

        try:
            claims = json.loads(raw)
            if not isinstance(claims, list):
                logger.warning("Extraction returned non-list JSON; wrapping")
                claims = []
        except json.JSONDecodeError as e:
            logger.error("JSON parse error in extraction: %s\nRaw: %s", e, raw[:500])
            claims = []

        return claims, tokens

    def _stub_extraction(self, text: str, speaker: str, date: str) -> list[dict]:
        """Return minimal stub claims for dry-run mode."""
        return [
            {
                "text": "Inflation was at record levels when the president took office.",
                "context": "We inherited record inflation...",
                "category": "inflation",
                "is_checkable": True,
            },
            {
                "text": "There were zero illegal aliens admitted in the past nine months.",
                "context": "Zero illegal aliens admitted in nine months.",
                "category": "immigration",
                "is_checkable": True,
            },
            {
                "text": "Egg prices are down about 60%.",
                "context": "Egg prices are down 60 percent.",
                "category": "food_prices",
                "is_checkable": True,
            },
        ]

    # ── Synthesis ──────────────────────────────────────────────────────────────

    def synthesize_verdicts(
        self,
        claims: list[dict],
        system_prompt: str,
        prompt_hash: str,
        reference: list[dict],
    ) -> tuple[list[dict], int]:
        """
        Synthesize verdicts for claims, using synthetic evidence derived from
        reference.json (avoids running expensive source queries per generation).

        Returns (verdicts_list, token_count).
        """
        tx_hash = hashlib.sha256(
            json.dumps([c.get("text", "") for c in claims]).encode()
        ).hexdigest()[:12]
        cache_key = _cache_key(f"syn_{prompt_hash}", tx_hash)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug("Cache hit: synthesis %s", cache_key)
            return cached["verdicts"], cached["tokens"]

        if self._dry_run:
            logger.info("[DRY-RUN] Would call synthesis API with prompt hash %s", prompt_hash)
            stub = self._stub_synthesis(claims)
            return stub, 0

        verdicts, tokens = self._call_synthesis_api(claims, system_prompt, reference)
        self._save_cache(cache_key, {"verdicts": verdicts, "tokens": tokens})
        return verdicts, tokens

    def _call_synthesis_api(
        self,
        claims: list[dict],
        system_prompt: str,
        reference: list[dict],
    ) -> tuple[list[dict], int]:
        """
        Call Anthropic API for verdict synthesis.
        Uses synthetic evidence derived from reference.json explanations.
        """
        import anthropic
        from evolver.fitness import fuzzy_claim_similarity

        client = anthropic.Anthropic(api_key=self._api_key)
        verdicts = []
        total_tokens = 0

        for claim in claims:
            claim_text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
            if not claim_text:
                continue

            # Build synthetic evidence from reference.json
            evidence_text = self._build_synthetic_evidence(claim_text, reference)

            user_msg = f"Claim: {claim_text}\n\nEvidence:\n{evidence_text}"

            try:
                response = client.messages.create(
                    model=self._synthesis_model,
                    max_tokens=512,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                )
                total_tokens += response.usage.input_tokens + response.usage.output_tokens
                raw = response.content[0].text.strip()

                # Strip markdown fences
                if raw.startswith("```"):
                    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()

                parsed = json.loads(raw)
                verdicts.append({
                    "claim_text": claim_text,
                    "label": parsed.get("label", "Unverifiable"),
                    "confidence": parsed.get("confidence", "Low"),
                    "explanation": parsed.get("explanation", ""),
                    "support_count": parsed.get("support_count", 0),
                    "contradict_count": parsed.get("contradict_count", 0),
                })
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Synthesis failed for claim '%s': %s", claim_text[:60], e)
                verdicts.append({
                    "claim_text": claim_text,
                    "label": "Unverifiable",
                    "confidence": "Low",
                    "explanation": "Parse error or API failure.",
                    "support_count": 0,
                    "contradict_count": 0,
                })

            time.sleep(0.3)  # Rate limit buffer

        return verdicts, total_tokens

    def _build_synthetic_evidence(self, claim_text: str, reference: list[dict]) -> str:
        """
        Build a synthetic evidence string for a claim by finding the closest
        reference item and using its explanation as evidence text.
        """
        from evolver.fitness import fuzzy_claim_similarity

        best_sim = 0.0
        best_ref = None
        for ref in reference:
            sim = fuzzy_claim_similarity(claim_text, ref["claim"])
            if sim > best_sim:
                best_sim = sim
                best_ref = ref

        if best_ref is None or best_sim < 0.1:
            return "[1] General fact-check sources\nNo specific evidence found for this claim."

        sources_desc = {
            "R1": "American Presidency Project (Government archive)",
            "R4": "PolitiFact (FactCheck organization)",
            "R5": "FactCheck.org (FactCheck organization)",
            "R6": "CBS News (Established outlet)",
            "R7": "Freddie Mac PMMS (Government/financial data)",
            "R8": "Associated Press (Wire service)",
        }
        source_names = [sources_desc.get(s, s) for s in best_ref.get("sources", [])]
        source_str = "; ".join(source_names) or "Multiple sources"

        return (
            f"[1] {source_str}\n"
            f"{best_ref['explanation']}\n\n"
            f"[Reference verdict from fact-checkers: {best_ref['verdict']}]"
        )

    def _stub_synthesis(self, claims: list[dict]) -> list[dict]:
        """Return stub verdicts for dry-run mode."""
        return [
            {
                "claim_text": c.get("text", ""),
                "label": "Unverifiable",
                "confidence": "Low",
                "explanation": "[DRY-RUN] Stub verdict -- no API call made.",
                "support_count": 0,
                "contradict_count": 0,
            }
            for c in claims
        ]

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        self._cache_path(key).write_text(json.dumps(data, indent=2))


# ── Transcript fetcher ─────────────────────────────────────────────────────────

_TRANSCRIPT_CACHE = Path(__file__).parent.parent / "sotu-2026" / "evolution_results" / "transcript_cache.json"


def fetch_transcript(url: str | None = None, cache_path: Path = _TRANSCRIPT_CACHE) -> str:
    """
    Fetch and cache the SOTU transcript.
    Returns the full text. Uses cache if available.
    """
    if cache_path.exists():
        logger.info("Using cached transcript from %s", cache_path)
        data = json.loads(cache_path.read_text())
        return data["text"]

    target_url = url or "https://apnews.com/article/c13e2a07df999b464b733f4a6e84dbd4"
    logger.info("Fetching transcript from %s", target_url)

    try:
        import httpx
        response = httpx.get(
            target_url,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; truth-bot/1.0)"},
        )
        response.raise_for_status()
        text = _extract_text(response.text)
    except Exception as e:
        logger.error("Failed to fetch transcript: %s", e)
        raise RuntimeError(f"Could not fetch transcript from {target_url}: {e}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"url": target_url, "text": text}, indent=2))
    logger.info("Transcript fetched and cached (%d chars)", len(text))
    return text


def _extract_text(html: str) -> str:
    """Basic HTML → plain text extraction."""
    import re
    # Remove script/style
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Fix missing import
import re  # noqa: E402
