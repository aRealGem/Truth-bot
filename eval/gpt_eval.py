#!/usr/bin/env python3
"""GPT-based standalone evaluation for Truth Bot.
Apples-to-apples comparison vs opus_eval.py -- identical prompts, OpenAI API.
Results saved to: eval/sotu-2026/gpt-5.4-results/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))

# Import shared prompt constants (single source of truth)
from evolver.base_eval import (
    EXTRACTION_SYSTEM,  # noqa: F401  (re-exported for callers that import from here)
    EXTRACTION_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
    BaseEvalRunner,
    ModelClient,
)
from evolver.fitness import load_reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpt_eval")

DEFAULT_MODEL = "gpt-5.4"
TRANSCRIPT_PATH = EVAL_DIR / "sotu-2026" / "transcript.txt"
RESULTS_DIR = EVAL_DIR / "sotu-2026" / "gpt-5.4-results"


def load_env(root: Path) -> None:
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


class OpenAIClient:
    """ModelClient backed by the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""
        it = response.usage.prompt_tokens
        ot = response.usage.completion_tokens
        return raw, it, ot


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GPT-based standalone evaluator -- apples-to-apples vs opus_eval.py"
    )
    ap.add_argument("--transcript", default=str(TRANSCRIPT_PATH))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    load_env(EVAL_DIR.parent)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight sanity checks
    from evolver.preflight import PreflightChecker
    _checker = PreflightChecker()
    _preflight = _checker.run_all(
        provider="openai",
        transcript_path=args.transcript,
        model=args.model,
        results_dir=results_dir,
    )
    for w in _preflight.warnings:
        logger.warning("PREFLIGHT: %s", w)
    for e in _preflight.errors:
        logger.error("PREFLIGHT: %s", e)
    if not _preflight.passed:
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENAI_API_KEY not set. "
            "Set it in your environment or in a .env file at the project root."
        )

    client = OpenAIClient(api_key=api_key, model=args.model)
    runner = BaseEvalRunner(
        client=client,
        model=args.model,
        results_dir=results_dir,
        cache_prefix="gpt",
    )

    transcript = Path(args.transcript).read_text()
    reference = load_reference()
    logger.info("Model: %s | Ref claims: %d", args.model, len(reference))

    runner.run(transcript, reference)


if __name__ == "__main__":
    main()
