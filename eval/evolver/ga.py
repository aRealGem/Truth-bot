"""
Genetic algorithm for prompt optimization.

Population -> Selection (tournament) -> Crossover -> Mutation -> Elitism -> repeat.

Mutation uses claude-3-5-haiku to rewrite individual gene values,
keeping the spirit of the gene but varying phrasing, emphasis, and detail.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evolver.genome import (
    ExtractionGenome,
    Individual,
    SynthesisGenome,
    EXTRACTION_PERSONA_VARIANTS,
    EXTRACTION_METHODOLOGY_VARIANTS,
    EXTRACTION_TAXONOMY_VARIANTS,
    EXTRACTION_FORMAT_VARIANTS,
    EXTRACTION_FILTERING_VARIANTS,
    EXTRACTION_EXAMPLES_VARIANTS,
    EXTRACTION_TONE_VARIANTS,
    SYNTHESIS_PERSONA_VARIANTS,
    SYNTHESIS_VERDICT_TAXONOMY_VARIANTS,
    SYNTHESIS_EVIDENCE_WEIGHTING_VARIANTS,
    SYNTHESIS_CONFIDENCE_VARIANTS,
    SYNTHESIS_REASONING_VARIANTS,
    SYNTHESIS_NUANCE_VARIANTS,
    SYNTHESIS_FORMAT_VARIANTS,
)

logger = logging.getLogger(__name__)


# ── Mutation helpers ───────────────────────────────────────────────────────────

_MUTATION_MODEL = "claude-3-5-haiku-20241022"

# Fallback: cycle to next variant index instead of calling LLM
_GENE_POOL_SIZES: dict[str, int] = {
    "extraction.persona_idx": len(EXTRACTION_PERSONA_VARIANTS),
    "extraction.methodology_idx": len(EXTRACTION_METHODOLOGY_VARIANTS),
    "extraction.taxonomy_idx": len(EXTRACTION_TAXONOMY_VARIANTS),
    "extraction.format_idx": len(EXTRACTION_FORMAT_VARIANTS),
    "extraction.filtering_idx": len(EXTRACTION_FILTERING_VARIANTS),
    "extraction.examples_idx": len(EXTRACTION_EXAMPLES_VARIANTS),
    "extraction.tone_idx": len(EXTRACTION_TONE_VARIANTS),
    "synthesis.persona_idx": len(SYNTHESIS_PERSONA_VARIANTS),
    "synthesis.verdict_taxonomy_idx": len(SYNTHESIS_VERDICT_TAXONOMY_VARIANTS),
    "synthesis.evidence_weighting_idx": len(SYNTHESIS_EVIDENCE_WEIGHTING_VARIANTS),
    "synthesis.confidence_idx": len(SYNTHESIS_CONFIDENCE_VARIANTS),
    "synthesis.reasoning_idx": len(SYNTHESIS_REASONING_VARIANTS),
    "synthesis.nuance_idx": len(SYNTHESIS_NUANCE_VARIANTS),
    "synthesis.format_idx": len(SYNTHESIS_FORMAT_VARIANTS),
}


def _rotate_gene(current_idx: int, pool_size: int) -> int:
    """Pick a random different index from the pool."""
    if pool_size <= 1:
        return current_idx
    candidates = [i for i in range(pool_size) if i != current_idx]
    return random.choice(candidates)


# ── Genetic operators ──────────────────────────────────────────────────────────

def tournament_select(population: list[Individual], k: int = 3) -> Individual:
    """
    Tournament selection: pick k random individuals, return the fittest.
    """
    competitors = random.sample(population, min(k, len(population)))
    return max(competitors, key=lambda ind: ind.fitness)


def crossover(parent_a: Individual, parent_b: Individual, generation: int) -> tuple[Individual, Individual]:
    """
    Single-point crossover on gene indices.
    Each gene is independently inherited from either parent (uniform crossover).
    """
    child_a = Individual(generation=generation)
    child_b = Individual(generation=generation)

    child_a.parent_hashes = [parent_a.id(), parent_b.id()]
    child_b.parent_hashes = [parent_a.id(), parent_b.id()]

    # Extraction genes
    for gene in ExtractionGenome.GENE_NAMES:
        if random.random() < 0.5:
            setattr(child_a.extraction, gene, getattr(parent_a.extraction, gene))
            setattr(child_b.extraction, gene, getattr(parent_b.extraction, gene))
        else:
            setattr(child_a.extraction, gene, getattr(parent_b.extraction, gene))
            setattr(child_b.extraction, gene, getattr(parent_a.extraction, gene))

    # Synthesis genes
    for gene in SynthesisGenome.GENE_NAMES:
        if random.random() < 0.5:
            setattr(child_a.synthesis, gene, getattr(parent_a.synthesis, gene))
            setattr(child_b.synthesis, gene, getattr(parent_b.synthesis, gene))
        else:
            setattr(child_a.synthesis, gene, getattr(parent_b.synthesis, gene))
            setattr(child_b.synthesis, gene, getattr(parent_a.synthesis, gene))

    return child_a, child_b


def mutate(
    individual: Individual,
    mutation_rate: float = 0.2,
    api_key: Optional[str] = None,
    use_llm_mutation: bool = False,
) -> Individual:
    """
    Apply mutation to an individual.

    For each gene, with probability mutation_rate:
      - Rotate to a different variant index (fast, no API cost)
      - OR use LLM to generate a novel variant (if use_llm_mutation=True and api_key set)

    Returns a new Individual (original is not modified).
    """
    mutant = copy.deepcopy(individual)
    mutant.mutation_log = []

    # Extraction mutations
    for gene in ExtractionGenome.GENE_NAMES:
        if random.random() < mutation_rate:
            pool_size = _GENE_POOL_SIZES[f"extraction.{gene}"]
            old_idx = getattr(mutant.extraction, gene)
            new_idx = _rotate_gene(old_idx, pool_size)
            setattr(mutant.extraction, gene, new_idx)
            mutant.mutation_log.append(f"extraction.{gene}: {old_idx} -> {new_idx}")

    # Synthesis mutations
    for gene in SynthesisGenome.GENE_NAMES:
        if random.random() < mutation_rate:
            pool_size = _GENE_POOL_SIZES[f"synthesis.{gene}"]
            old_idx = getattr(mutant.synthesis, gene)
            new_idx = _rotate_gene(old_idx, pool_size)
            setattr(mutant.synthesis, gene, new_idx)
            mutant.mutation_log.append(f"synthesis.{gene}: {old_idx} -> {new_idx}")

    mutant.evaluated = False
    mutant.fitness = 0.0
    return mutant


# ── Seed population ────────────────────────────────────────────────────────────

def build_seed_population(size: int) -> list[Individual]:
    """
    Build the generation-0 population.
    - Individual 0: all-baseline (current production prompts)
    - Individuals 1–3: hand-crafted alternatives
    - Remaining: random
    """
    population: list[Individual] = []

    # Baseline
    baseline = Individual(
        extraction=ExtractionGenome.baseline(),
        synthesis=SynthesisGenome.baseline(),
        generation=0,
    )
    population.append(baseline)

    # Hand-crafted variant 1: investigative + aggressive extraction + step-by-step synthesis
    v1 = Individual(
        extraction=ExtractionGenome(
            persona_idx=1,          # investigative journalist
            methodology_idx=1,      # aggressive extraction
            taxonomy_idx=1,         # expanded taxonomy
            format_idx=1,           # richer fields
            filtering_idx=1,        # explicit checkability
            examples_idx=1,         # one-shot example
            tone_idx=2,             # detailed/verbose
        ),
        synthesis=SynthesisGenome(
            persona_idx=1,              # senior research analyst
            verdict_taxonomy_idx=1,     # richer taxonomy
            evidence_weighting_idx=1,   # explicit tier weighting
            confidence_idx=1,           # calibrated confidence
            reasoning_idx=1,            # step-by-step
            nuance_idx=1,               # nuance handling
            format_idx=1,               # with source emphasis
        ),
        generation=0,
    )
    population.append(v1)

    # Hand-crafted variant 2: data journalist + conservative extraction + priority synthesis
    v2 = Individual(
        extraction=ExtractionGenome(
            persona_idx=4,          # data journalist
            methodology_idx=3,      # source-matchable
            taxonomy_idx=3,         # fine-grained topics
            format_idx=0,           # baseline format
            filtering_idx=2,        # strict checkability
            examples_idx=2,         # two examples
            tone_idx=3,             # skeptical
        ),
        synthesis=SynthesisGenome(
            persona_idx=2,              # policy researcher
            verdict_taxonomy_idx=3,     # priority-ordered taxonomy
            evidence_weighting_idx=2,   # source skepticism
            confidence_idx=2,           # conservative confidence
            reasoning_idx=2,            # evidence inventory
            nuance_idx=2,               # misleading detection
            format_idx=2,               # minimal output
        ),
        generation=0,
    )
    population.append(v2)

    # Hand-crafted variant 3: fact-checker + conservative + calibrated
    v3 = Individual(
        extraction=ExtractionGenome(
            persona_idx=3,          # fact-check org researcher
            methodology_idx=2,      # conservative/high-precision
            taxonomy_idx=2,         # grouped categories
            format_idx=3,           # with confidence/notes
            filtering_idx=1,        # explicit checkability
            examples_idx=0,         # zero-shot
            tone_idx=1,             # terse
        ),
        synthesis=SynthesisGenome(
            persona_idx=3,              # investigative journalist
            verdict_taxonomy_idx=2,     # reference-aligned labels
            evidence_weighting_idx=1,   # tier weighting
            confidence_idx=1,           # calibrated
            reasoning_idx=1,            # step-by-step
            nuance_idx=1,               # nuance handling
            format_idx=0,               # baseline format
        ),
        generation=0,
    )
    population.append(v3)

    # Fill remaining with random individuals
    while len(population) < size:
        population.append(Individual(
            extraction=ExtractionGenome.random(),
            synthesis=SynthesisGenome.random(),
            generation=0,
        ))

    return population[:size]


# ── Evolution loop ─────────────────────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    """Configuration for the genetic algorithm run."""
    generations: int = 5
    population_size: int = 8
    mutation_rate: float = 0.2
    tournament_k: int = 3
    elitism_count: int = 2
    use_llm_mutation: bool = False  # LLM mutation is costly; off by default
    dry_run: bool = False
    budget_usd: float = 10.0
    extraction_model: str = "claude-sonnet-4-5"
    synthesis_model: str = "claude-sonnet-4-5"
    eval_synthesis: bool = True  # Whether to run synthesis evaluation
    results_dir: Path = field(default_factory=lambda: Path(
        __file__).parent.parent / "sotu-2026" / "evolution_results"
    )


class GeneticEvolver:
    """
    Runs the full genetic algorithm evolution loop.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        api_key: Optional[str] = None,
    ):
        self.config = config
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._results_dir = config.results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._evolution_log: list[dict] = []
        self._estimated_cost_usd = 0.0

    def run(self, transcript_text: str) -> Individual:
        """
        Run the evolution loop.
        Returns the best individual found.
        """
        from evolver.runner import CachedRunner
        from evolver.fitness import load_reference
        from evolver.fitness import FitnessScorer

        reference = load_reference()
        runner = CachedRunner(
            api_key=self._api_key,
            extraction_model=self.config.extraction_model,
            synthesis_model=self.config.synthesis_model,
            dry_run=self.config.dry_run,
        )
        scorer = FitnessScorer(reference=reference)

        logger.info("Initializing seed population (size=%d)", self.config.population_size)
        population = build_seed_population(self.config.population_size)

        best_ever: Optional[Individual] = None

        for gen in range(self.config.generations):
            logger.info("=== Generation %d/%d ===", gen + 1, self.config.generations)

            # Budget check
            if self._estimated_cost_usd >= self.config.budget_usd:
                logger.warning(
                    "Budget cap reached ($%.2f). Stopping early.",
                    self.config.budget_usd,
                )
                break

            # Evaluate unevaluated individuals
            population = self._evaluate_population(
                population, transcript_text, runner, scorer, reference
            )

            # Sort by fitness (descending)
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Log generation results
            gen_log = self._log_generation(gen, population)
            self._evolution_log.append(gen_log)
            self._save_generation_file(gen, population)

            best_gen = population[0]
            logger.info(
                "Gen %d best: fitness=%.4f recall=%.3f verdict=%.3f expl=%.3f src=%.3f pars=%.3f [id=%s]",
                gen + 1,
                best_gen.fitness,
                best_gen.claim_recall,
                best_gen.verdict_agreement,
                best_gen.explanation_quality,
                best_gen.source_citation_quality,
                best_gen.parsimony,
                best_gen.id(),
            )

            if best_ever is None or best_gen.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(best_gen)

            # Final generation: don't breed a new population
            if gen == self.config.generations - 1:
                break

            # Breed next generation
            population = self._breed_next_generation(population, gen + 1)

        # Save final outputs
        self._save_evolution_log()
        if best_ever:
            self._save_best_prompts(best_ever, population)

        return best_ever or population[0]

    def _evaluate_population(
        self,
        population: list[Individual],
        transcript_text: str,
        runner: "CachedRunner",
        scorer: "FitnessScorer",
        reference: list[dict],
    ) -> list[Individual]:
        """Evaluate all unevaluated individuals in the population."""
        for ind in population:
            if ind.evaluated:
                continue

            logger.info("Evaluating individual %s ...", ind.id())

            # Extraction
            system_prompt = ind.extraction.render_system_prompt()
            user_template = ind.extraction.render_user_template()
            claims, ext_tokens = runner.extract_claims(
                transcript_text=transcript_text,
                speaker="President Donald Trump",
                date_str="2026-02-24",
                system_prompt=system_prompt,
                user_template=user_template,
                prompt_hash=ind.extraction.hash(),
            )
            ind.extraction_token_count = ext_tokens

            # Estimate cost: ~$3/MTok input, ~$15/MTok output for sonnet
            # Rough: assume 70/30 split
            ext_cost = ext_tokens * 0.000006
            self._estimated_cost_usd += ext_cost

            # Synthesis (optional)
            verdicts = []
            syn_tokens = 0
            if self.config.eval_synthesis and claims:
                syn_prompt = ind.synthesis.render_system_prompt()
                verdicts, syn_tokens = runner.synthesize_verdicts(
                    claims=claims,
                    system_prompt=syn_prompt,
                    prompt_hash=ind.synthesis.hash(),
                    reference=reference,
                )
                ind.synthesis_token_count = syn_tokens
                syn_cost = syn_tokens * 0.000006
                self._estimated_cost_usd += syn_cost

            # Score
            scores = scorer.score(
                extracted_claims=claims,
                verdicts=verdicts,
                token_count=ext_tokens + syn_tokens,
            )

            ind.claim_recall = scores["claim_recall"]
            ind.verdict_agreement = scores["verdict_agreement"]
            ind.explanation_quality = scores["explanation_quality"]
            ind.source_citation_quality = scores["source_citation_quality"]
            ind.parsimony = scores["parsimony"]
            ind.compute_fitness()
            ind.evaluated = True

            logger.info(
                "  -> fitness=%.4f recall=%.3f verdict=%.3f expl=%.3f src=%.3f matched=%d/%d cost_total=$%.3f",
                ind.fitness,
                ind.claim_recall,
                ind.verdict_agreement,
                ind.explanation_quality,
                ind.source_citation_quality,
                scores["matched_count"],
                29,
                self._estimated_cost_usd,
            )

        return population

    def _breed_next_generation(
        self, population: list[Individual], generation: int
    ) -> list[Individual]:
        """
        Breed the next generation via elitism + tournament selection + crossover + mutation.
        """
        next_gen: list[Individual] = []

        # Elitism: carry forward top performers unchanged
        elites = population[: self.config.elitism_count]
        for elite in elites:
            kept = copy.deepcopy(elite)
            kept.generation = generation
            kept.parent_hashes = [elite.id()]
            kept.mutation_log = ["[elite carry-forward]"]
            next_gen.append(kept)
            logger.debug("Elite carried: %s (fitness=%.4f)", elite.id(), elite.fitness)

        # Fill rest via selection + crossover + mutation
        while len(next_gen) < self.config.population_size:
            parent_a = tournament_select(population, self.config.tournament_k)
            parent_b = tournament_select(population, self.config.tournament_k)
            child_a, child_b = crossover(parent_a, parent_b, generation)
            child_a = mutate(child_a, self.config.mutation_rate, self._api_key)
            child_b = mutate(child_b, self.config.mutation_rate, self._api_key)
            next_gen.append(child_a)
            if len(next_gen) < self.config.population_size:
                next_gen.append(child_b)

        return next_gen[: self.config.population_size]

    def _log_generation(self, gen: int, population: list[Individual]) -> dict:
        return {
            "generation": gen + 1,
            "population_size": len(population),
            "best_fitness": population[0].fitness if population else 0.0,
            "mean_fitness": sum(i.fitness for i in population) / len(population) if population else 0.0,
            "estimated_cost_usd": round(self._estimated_cost_usd, 4),
            "individuals": [ind.to_dict() for ind in population],
        }

    def _save_generation_file(self, gen: int, population: list[Individual]) -> None:
        path = self._results_dir / f"generation_{gen + 1:02d}.json"
        data = {
            "generation": gen + 1,
            "individuals": [ind.to_dict() for ind in population],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.debug("Saved generation file: %s", path)

    def _save_evolution_log(self) -> None:
        path = self._results_dir / "evolution_log.json"
        path.write_text(json.dumps(self._evolution_log, indent=2))
        logger.info("Evolution log saved: %s", path)

    def _save_best_prompts(
        self, best: Individual, final_population: list[Individual]
    ) -> None:
        path = self._results_dir / "best_prompts.json"
        data = {
            "best_individual": best.to_dict(),
            "extraction_system_prompt": best.extraction.render_system_prompt(),
            "extraction_user_template": best.extraction.render_user_template(),
            "synthesis_system_prompt": best.synthesis.render_system_prompt(),
            "top_5": [ind.to_dict() for ind in sorted(
                final_population, key=lambda i: i.fitness, reverse=True
            )[:5]],
            "integration_instructions": (
                "To use these prompts in production:\n"
                "1. Set TRUTHBOT_EXTRACTION_PROMPT_FILE=path/to/extraction_prompt.txt\n"
                "2. Set TRUTHBOT_SYNTHESIS_PROMPT_FILE=path/to/synthesis_prompt.txt\n"
                "OR update _SYSTEM_PROMPT in src/truthbot/extract/claims.py\n"
                "and _SYNTHESIS_SYSTEM in src/truthbot/verify/engine.py\n"
                "with the 'extraction_system_prompt' and 'synthesis_system_prompt' values above."
            ),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Best prompts saved: %s", path)


# Re-export load_reference for convenience
def load_reference() -> list[dict]:
    from evolver.fitness import load_reference as _lr
    return _lr()
