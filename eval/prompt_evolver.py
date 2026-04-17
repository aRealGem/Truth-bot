#!/usr/bin/env python3
"""
Genetic algorithm-based prompt optimizer for truth-bot.

Tunes claim extraction and verdict synthesis prompts against the GPT 5.4 Pro
Extended reference analysis of the 2026 SOTU (29 claims in
eval/sotu-2026/reference.json).

Usage examples
--------------
# Dry run -- show what would happen without making API calls:
    python eval/prompt_evolver.py --generations 5 --population 8 --dry-run

# Real run (5 generations, population 8, budget $5):
    python eval/prompt_evolver.py --generations 5 --population 8 --budget 5.0

# Extraction-only mode (no synthesis evaluation):
    python eval/prompt_evolver.py --generations 3 --population 6 --no-synthesis

# Use a local transcript file instead of fetching:
    python eval/prompt_evolver.py --transcript-file /path/to/sotu.txt

# Skip transcript fetch (use previously cached):
    python eval/prompt_evolver.py --use-cached-transcript

Environment variables
---------------------
ANTHROPIC_API_KEY   -- required for live runs
BRAVE_API_KEY       -- used by truth-bot source connectors (optional for evolver)

Output
------
eval/sotu-2026/evolution_results/
  generation_NN.json    -- per-generation population snapshots
  evolution_log.json    -- full history
  best_prompts.json     -- top-performing prompts with component breakdown
  fitness_report.md     -- human-readable convergence report
  comparison_table.md   -- evolved output vs. reference (29 claims)
  runner_cache/         -- cached API responses (keyed by prompt+content hash)
  transcript_cache.json -- cached SOTU transcript

Integration
-----------
Set these env vars to use the winning prompts in production:
  TRUTHBOT_EXTRACTION_PROMPT_FILE=/path/to/extraction_prompt.txt
  TRUTHBOT_SYNTHESIS_PROMPT_FILE=/path/to/synthesis_prompt.txt

Or copy the prompt strings from best_prompts.json directly into:
  src/truthbot/extract/claims.py  (_SYSTEM_PROMPT)
  src/truthbot/verify/engine.py   (_SYNTHESIS_SYSTEM)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the eval/ directory and its subdirs are on the path
_EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(_EVAL_DIR))

# Load .env from project root
_PROJECT_ROOT = _EVAL_DIR.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE)
    except ImportError:
        # Manual parse if python-dotenv not installed
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genetic algorithm prompt optimizer for truth-bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--generations", "-g", type=int, default=5,
        help="Number of generations to run (default: 5)",
    )
    parser.add_argument(
        "--population", "-p", type=int, default=8,
        help="Population size per generation (default: 8, min 4, max 12)",
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=0.20,
        help="Per-gene mutation probability (default: 0.20)",
    )
    parser.add_argument(
        "--tournament-k", type=int, default=3,
        help="Tournament selection size (default: 3)",
    )
    parser.add_argument(
        "--elitism", type=int, default=2,
        help="Number of elite individuals to carry forward unchanged (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be evaluated without making API calls",
    )
    parser.add_argument(
        "--budget", type=float, default=10.0,
        help="Maximum estimated API cost in USD before stopping (default: $10.00)",
    )
    parser.add_argument(
        "--no-synthesis", action="store_true",
        help="Skip synthesis prompt evaluation (extraction only, ~5x cheaper)",
    )
    parser.add_argument(
        "--transcript-url", type=str, default=None,
        help="URL to fetch SOTU transcript from (default: AP transcript)",
    )
    parser.add_argument(
        "--transcript-file", type=str, default=None,
        help="Path to a local transcript text file (skips fetch)",
    )
    parser.add_argument(
        "--use-cached-transcript", action="store_true",
        help="Use previously cached transcript (error if not available)",
    )
    parser.add_argument(
        "--extraction-model", type=str, default="claude-sonnet-4-5",
        help="Model for extraction runs (default: claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--synthesis-model", type=str, default="claude-sonnet-4-5",
        help="Model for synthesis runs (default: claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Output directory (default: eval/sotu-2026/evolution_results/)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--show-prompts", action="store_true",
        help="Print the full prompts for the baseline and exit (no evolution)",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Score the baseline prompts against reference.json and exit",
    )
    return parser.parse_args()


def show_prompts() -> None:
    """Print baseline prompts for inspection."""
    from evolver.genome import ExtractionGenome, SynthesisGenome
    ext = ExtractionGenome.baseline()
    syn = SynthesisGenome.baseline()
    print("=" * 70)
    print("EXTRACTION SYSTEM PROMPT (baseline)")
    print("=" * 70)
    print(ext.render_system_prompt())
    print()
    print("=" * 70)
    print("SYNTHESIS SYSTEM PROMPT (baseline)")
    print("=" * 70)
    print(syn.render_system_prompt())


def score_baseline(results_dir: Path) -> None:
    """Score the baseline prompts in dry-run mode."""
    from evolver.fitness import FitnessScorer, load_reference
    from evolver.genome import ExtractionGenome, SynthesisGenome
    from evolver.runner import CachedRunner

    reference = load_reference()
    scorer = FitnessScorer(reference)
    runner = CachedRunner(dry_run=True, cache_dir=results_dir / "runner_cache")

    ext = ExtractionGenome.baseline()
    syn = SynthesisGenome.baseline()

    claims, _ = runner.extract_claims(
        transcript_text="[baseline scoring stub]",
        speaker="President Donald Trump",
        date_str="2026-02-24",
        system_prompt=ext.render_system_prompt(),
        user_template=ext.render_user_template(),
        prompt_hash=ext.hash(),
    )
    verdicts, _ = runner.synthesize_verdicts(
        claims=claims,
        system_prompt=syn.render_system_prompt(),
        prompt_hash=syn.hash(),
        reference=reference,
    )
    scores = scorer.score(claims, verdicts, token_count=0)
    print("\nBaseline scores (dry-run, stub data):")
    for k, v in scores.items():
        print(f"  {k}: {v}")


def print_dry_run_plan(args: argparse.Namespace) -> None:
    """Print what would be evaluated without making any API calls."""
    from evolver.ga import build_seed_population
    from evolver.genome import Individual

    pop = build_seed_population(min(max(args.population, 4), 12))
    n_gens = args.generations
    n_pop = len(pop)
    n_claims = 29

    print(f"\n{'='*60}")
    print("DRY-RUN PLAN")
    print(f"{'='*60}")
    print(f"Generations:       {n_gens}")
    print(f"Population size:   {n_pop}")
    print(f"Mutation rate:     {args.mutation_rate:.0%}")
    print(f"Tournament k:      {args.tournament_k}")
    print(f"Elitism:           {args.elitism}")
    print(f"Synthesis eval:    {'yes' if not args.no_synthesis else 'no'}")
    print(f"Budget cap:        ${args.budget:.2f}")
    print(f"Extraction model:  {args.extraction_model}")
    print(f"Synthesis model:   {args.synthesis_model}")
    print()

    # API call estimates
    unique_extraction_prompts = n_gens * n_pop  # worst case (no cache hits)
    ext_calls = unique_extraction_prompts
    syn_calls = n_gens * n_pop * n_claims if not args.no_synthesis else 0
    print(f"API calls estimated (worst case, no cache):")
    print(f"  Extraction:   {ext_calls} calls × ~4000 tokens each")
    print(f"  Synthesis:    {syn_calls} calls × ~500 tokens each" if not args.no_synthesis else "  Synthesis:    0 (disabled)")
    ext_cost = ext_calls * 4000 * 0.000003
    syn_cost = syn_calls * 500 * 0.000003
    total = ext_cost + syn_cost
    print(f"  Estimated cost (sonnet, rough): ${total:.2f}")
    print()
    print("Generation 0 seed population:")
    for i, ind in enumerate(pop):
        print(f"  [{i}] {ind.id()} -- ext genes: {ind.extraction.to_dict()} | syn genes: {ind.synthesis.to_dict()}")
    print()
    print("Gene pool sizes:")
    from evolver.ga import _GENE_POOL_SIZES
    for gene, size in _GENE_POOL_SIZES.items():
        print(f"  {gene}: {size} variants")
    print()
    print("Would output to:", args.results_dir or "eval/sotu-2026/evolution_results/")
    print(f"{'='*60}\n")


def get_transcript(args: argparse.Namespace) -> str:
    """Load or fetch the SOTU transcript."""
    from evolver.runner import fetch_transcript, _TRANSCRIPT_CACHE

    # Local file override
    if args.transcript_file:
        path = Path(args.transcript_file)
        if not path.exists():
            print(f"ERROR: Transcript file not found: {args.transcript_file}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text()
        print(f"Loaded transcript from file: {path} ({len(text)} chars)")
        return text

    # Cache-only mode
    if args.use_cached_transcript:
        if not _TRANSCRIPT_CACHE.exists():
            print(
                "ERROR: --use-cached-transcript specified but no cache found.\n"
                f"Cache expected at: {_TRANSCRIPT_CACHE}\n"
                "Run without this flag first to fetch and cache the transcript.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Normal fetch (uses cache if available)
    return fetch_transcript(url=args.transcript_url)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("prompt_evolver")

    import random
    if args.seed is not None:
        random.seed(args.seed)
        logger.info("Random seed: %d", args.seed)

    # --show-prompts: print baseline prompts and exit
    if args.show_prompts:
        show_prompts()
        return

    # Results directory
    results_dir = Path(args.results_dir) if args.results_dir else (
        Path(__file__).parent / "sotu-2026" / "evolution_results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # --score-only: score baseline and exit
    if args.score_only:
        score_baseline(results_dir)
        return

    # --dry-run: print plan and exit
    if args.dry_run:
        # Set results_dir on args for the plan printer
        args.results_dir = str(results_dir)
        print_dry_run_plan(args)
        # Also do a quick dry-run evaluation so caching behaviour is exercised
        logger.info("Running dry-run evaluation (stub data only)...")
        _run_dry_evaluation(args, results_dir)
        return

    # Check API key for live runs
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not set. Either set it in .env or environment.\n"
            "Use --dry-run to test without API calls.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch/load transcript
    logger.info("Loading SOTU transcript...")
    transcript_text = get_transcript(args)
    logger.info("Transcript loaded (%d chars)", len(transcript_text))

    # Build config
    from evolver.ga import EvolutionConfig, GeneticEvolver

    config = EvolutionConfig(
        generations=args.generations,
        population_size=min(max(args.population, 4), 12),
        mutation_rate=args.mutation_rate,
        tournament_k=args.tournament_k,
        elitism_count=args.elitism,
        dry_run=False,
        budget_usd=args.budget,
        extraction_model=args.extraction_model,
        synthesis_model=args.synthesis_model,
        eval_synthesis=not args.no_synthesis,
        results_dir=results_dir,
    )

    logger.info(
        "Starting evolution: %d generations × %d individuals, budget $%.2f",
        config.generations,
        config.population_size,
        config.budget_usd,
    )

    evolver = GeneticEvolver(config=config, api_key=api_key)
    best = evolver.run(transcript_text)

    # Generate reports
    from evolver.report import generate_fitness_report, generate_comparison_table
    from evolver.fitness import load_reference

    log_path = results_dir / "evolution_log.json"
    evolution_log = json.loads(log_path.read_text()) if log_path.exists() else []
    reference = load_reference()

    report = generate_fitness_report(
        evolution_log=evolution_log,
        best_individual=best.to_dict(),
        results_dir=results_dir,
    )
    generate_comparison_table(
        best_individual=best.to_dict(),
        reference=reference,
        runner_cache_dir=results_dir / "runner_cache",
        results_dir=results_dir,
    )

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best individual: {best.id()}")
    print(f"  Fitness:             {best.fitness:.4f}")
    print(f"  Claim recall:        {best.claim_recall:.4f}")
    print(f"  Verdict agreement:   {best.verdict_agreement:.4f}")
    print(f"  Explanation quality: {best.explanation_quality:.4f}")
    print(f"  Source citation:     {best.source_citation_quality:.4f}")
    print(f"  Parsimony:           {best.parsimony:.4f}")
    print()
    print(f"Outputs written to: {results_dir}")
    print()
    print("Next step: review best_prompts.json and integrate winning prompts")
    print("  into src/truthbot/extract/claims.py and src/truthbot/verify/engine.py")


def _run_dry_evaluation(args: argparse.Namespace, results_dir: Path) -> None:
    """Run a dry evaluation to exercise the full code path with stub data."""
    from evolver.ga import build_seed_population, EvolutionConfig, GeneticEvolver

    config = EvolutionConfig(
        generations=args.generations,
        population_size=min(max(args.population, 4), 12),
        mutation_rate=args.mutation_rate,
        tournament_k=args.tournament_k,
        elitism_count=args.elitism,
        dry_run=True,
        budget_usd=args.budget,
        extraction_model=args.extraction_model,
        synthesis_model=args.synthesis_model,
        eval_synthesis=not args.no_synthesis,
        results_dir=results_dir,
    )

    evolver = GeneticEvolver(config=config)
    # Use a short stub transcript
    stub_transcript = (
        "Fellow Americans, inflation was at record levels when I took office. "
        "We now have zero illegal aliens admitted in the past nine months. "
        "Fentanyl flow is down 56%. Murder rate is the lowest in 125 years. "
        "Core inflation is the lowest in more than five years. "
        "Gas prices are below $2.30 in most states. "
        "Egg prices are down about 60 percent. "
        "Oil production is up by more than 600,000 barrels per day. "
        "We secured more than $18 trillion in investment commitments. "
        "Mortgage rates are the lowest in four years. "
        "Tariffs are paid by foreign countries, not our people."
    )
    best = evolver.run(stub_transcript)

    from evolver.report import generate_fitness_report, generate_comparison_table
    from evolver.fitness import load_reference

    log_path = results_dir / "evolution_log.json"
    evolution_log = json.loads(log_path.read_text()) if log_path.exists() else []
    reference = load_reference()

    generate_fitness_report(evolution_log, best.to_dict(), results_dir)
    generate_comparison_table(best.to_dict(), reference, results_dir / "runner_cache", results_dir)

    print(f"\n[DRY-RUN] Completed. Results written to: {results_dir}")
    print(f"[DRY-RUN] Best (stub) individual: {best.id()} -- fitness: {best.fitness:.4f}")
    print("\nFiles generated:")
    for f in sorted(list(results_dir.glob("*.json")) + list(results_dir.glob("*.md"))):
        print(f"  {f}")


if __name__ == "__main__":
    main()
