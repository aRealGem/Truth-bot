"""Tests for eval/evolver/ga.py"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evolver.genome import ExtractionGenome, SynthesisGenome, Individual
from evolver.ga import (
    mutate,
    crossover,
    tournament_select,
    build_seed_population,
    _GENE_POOL_SIZES,
)


def _make_individual(fitness: float = 0.5) -> Individual:
    ind = Individual(
        extraction=ExtractionGenome.baseline(),
        synthesis=SynthesisGenome.baseline(),
    )
    ind.fitness = fitness
    ind.evaluated = True
    return ind


# ── Mutation ───────────────────────────────────────────────────────────────────

def test_mutation_changes_genes_at_rate_1():
    """mutation_rate=1.0 should change all genes (when pool size > 1)."""
    random.seed(42)
    ind = _make_individual()
    mutant = mutate(ind, mutation_rate=1.0)
    # At least some genes should have changed
    changed = 0
    for gene in ExtractionGenome.GENE_NAMES:
        pool_size = _GENE_POOL_SIZES[f"extraction.{gene}"]
        if pool_size > 1:
            # _rotate_gene always picks a different index
            if getattr(mutant.extraction, gene) != getattr(ind.extraction, gene):
                changed += 1
    assert changed > 0


def test_mutation_preserves_genes_at_rate_0():
    """mutation_rate=0.0 should not change any genes."""
    random.seed(0)
    ind = _make_individual()
    # Set some non-zero values
    ind.extraction.persona_idx = 2
    ind.synthesis.persona_idx = 1
    mutant = mutate(ind, mutation_rate=0.0)
    for gene in ExtractionGenome.GENE_NAMES:
        assert getattr(mutant.extraction, gene) == getattr(ind.extraction, gene)
    for gene in SynthesisGenome.GENE_NAMES:
        assert getattr(mutant.synthesis, gene) == getattr(ind.synthesis, gene)


def test_mutation_returns_new_object():
    ind = _make_individual()
    mutant = mutate(ind, mutation_rate=0.0)
    assert mutant is not ind
    assert mutant.extraction is not ind.extraction


def test_mutation_sets_evaluated_false():
    ind = _make_individual()
    assert ind.evaluated is True
    mutant = mutate(ind, mutation_rate=0.0)
    assert mutant.evaluated is False


# ── Crossover ─────────────────────────────────────────────────────────────────

def test_crossover_genes_from_parents():
    random.seed(7)
    parent_a = Individual(
        extraction=ExtractionGenome(persona_idx=0, methodology_idx=0),
        synthesis=SynthesisGenome(persona_idx=0),
    )
    parent_b = Individual(
        extraction=ExtractionGenome(persona_idx=3, methodology_idx=2),
        synthesis=SynthesisGenome(persona_idx=2),
    )
    child_a, child_b = crossover(parent_a, parent_b, generation=1)
    # Each gene in each child must come from one of the two parents
    for gene in ExtractionGenome.GENE_NAMES:
        val_a = getattr(parent_a.extraction, gene)
        val_b = getattr(parent_b.extraction, gene)
        child_a_val = getattr(child_a.extraction, gene)
        child_b_val = getattr(child_b.extraction, gene)
        assert child_a_val in (val_a, val_b), f"child_a.{gene} = {child_a_val} not from parents"
        assert child_b_val in (val_a, val_b), f"child_b.{gene} = {child_b_val} not from parents"


def test_crossover_returns_two_children():
    parent_a = _make_individual()
    parent_b = _make_individual()
    result = crossover(parent_a, parent_b, generation=1)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_crossover_children_not_evaluated():
    parent_a = _make_individual()
    parent_b = _make_individual()
    child_a, child_b = crossover(parent_a, parent_b, generation=1)
    assert child_a.evaluated is False
    assert child_b.evaluated is False


# ── Tournament selection ───────────────────────────────────────────────────────

def test_tournament_select_returns_fittest_when_k_equals_population():
    pop = [_make_individual(fitness=float(i) / 10) for i in range(5)]
    # With k = len(pop), we always see the whole population → fittest wins
    random.seed(0)
    winner = tournament_select(pop, k=len(pop))
    assert winner.fitness == max(ind.fitness for ind in pop)


def test_tournament_select_works_with_k1():
    pop = [_make_individual(fitness=float(i) / 10) for i in range(5)]
    random.seed(0)
    winner = tournament_select(pop, k=1)
    assert winner in pop


# ── Seed population ───────────────────────────────────────────────────────────

def test_build_seed_population_baseline_is_first():
    pop = build_seed_population(4)
    assert len(pop) >= 1
    first = pop[0]
    # Baseline: all indices 0
    for gene in ExtractionGenome.GENE_NAMES:
        assert getattr(first.extraction, gene) == 0, f"extraction.{gene} should be 0"
    for gene in SynthesisGenome.GENE_NAMES:
        assert getattr(first.synthesis, gene) == 0, f"synthesis.{gene} should be 0"


def test_build_seed_population_has_correct_size():
    pop = build_seed_population(6)
    assert len(pop) == 6


def test_dry_run_produces_identical_results_for_any_genome(tmp_path, sample_transcript):
    """Documents the known dry-run limitation: genomes are indistinguishable."""
    from evolver.runner import CachedRunner

    runner = CachedRunner(dry_run=True, cache_dir=tmp_path)
    g_baseline = ExtractionGenome.baseline()
    g_random = ExtractionGenome.random()

    claims_a, _ = runner.extract_claims(
        sample_transcript, "Speaker", "2026-01-01",
        g_baseline.render_system_prompt(),
        g_baseline.render_user_template(),
        g_baseline.hash(),
    )
    claims_b, _ = runner.extract_claims(
        sample_transcript, "Speaker", "2026-01-01",
        g_random.render_system_prompt(),
        g_random.render_user_template(),
        g_random.hash(),
    )
    # dry-run: both return the same fixed stub claims
    assert claims_a == claims_b, "DRY-RUN: genomes produce identical stub claims -- known limitation"
