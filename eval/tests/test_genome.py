"""Tests for eval/evolver/genome.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evolver.genome import (
    ExtractionGenome,
    SynthesisGenome,
    Individual,
    EXTRACTION_PERSONA_VARIANTS,
    EXTRACTION_METHODOLOGY_VARIANTS,
    EXTRACTION_TAXONOMY_VARIANTS,
    EXTRACTION_FORMAT_VARIANTS,
    EXTRACTION_FILTERING_VARIANTS,
    EXTRACTION_EXAMPLES_VARIANTS,
    EXTRACTION_TONE_VARIANTS,
    SYNTHESIS_PERSONA_VARIANTS,
)


def test_baseline_is_all_zeros():
    g = ExtractionGenome.baseline()
    for name in ExtractionGenome.GENE_NAMES:
        assert getattr(g, name) == 0, f"{name} should be 0"


def test_hash_is_stable():
    g = ExtractionGenome.baseline()
    h1 = g.hash()
    h2 = g.hash()
    assert h1 == h2


def test_different_genomes_have_different_hashes():
    g0 = ExtractionGenome.baseline()
    # v1 has persona_idx=1
    g1 = ExtractionGenome(persona_idx=1)
    assert g0.hash() != g1.hash()


def test_random_genome_is_valid():
    g = ExtractionGenome.random()
    pools = ExtractionGenome.GENE_POOLS
    for name in ExtractionGenome.GENE_NAMES:
        idx = getattr(g, name)
        pool_size = len(pools[name])
        assert 0 <= idx < pool_size, f"{name} idx {idx} out of bounds for pool size {pool_size}"


def test_to_dict_from_dict_roundtrip():
    g = ExtractionGenome(persona_idx=2, methodology_idx=1, taxonomy_idx=3)
    d = g.to_dict()
    g2 = ExtractionGenome.from_dict(d)
    for name in ExtractionGenome.GENE_NAMES:
        assert getattr(g, name) == getattr(g2, name), f"Mismatch on {name}"


def test_render_system_prompt_includes_persona():
    g = ExtractionGenome(persona_idx=1)  # investigative journalist
    prompt = g.render_system_prompt()
    assert EXTRACTION_PERSONA_VARIANTS[1] in prompt


def test_render_system_prompt_baseline():
    g = ExtractionGenome.baseline()
    prompt = g.render_system_prompt()
    assert EXTRACTION_PERSONA_VARIANTS[0] in prompt
    assert len(prompt) > 100


def test_individual_id_is_stable():
    ind = Individual(
        extraction=ExtractionGenome.baseline(),
        synthesis=SynthesisGenome.baseline(),
    )
    id1 = ind.id()
    id2 = ind.id()
    assert id1 == id2


def test_individual_compute_fitness_weighted_sum():
    ind = Individual()
    ind.claim_recall = 0.8
    ind.verdict_agreement = 0.6
    ind.explanation_quality = 0.5
    ind.source_citation_quality = 0.4
    ind.parsimony = 1.0

    fitness = ind.compute_fitness()

    w = Individual.WEIGHTS
    expected = (
        w["claim_recall"] * 0.8
        + w["verdict_agreement"] * 0.6
        + w["explanation_quality"] * 0.5
        + w["source_citation_quality"] * 0.4
        + w["parsimony"] * 1.0
    )
    assert abs(fitness - expected) < 1e-9
    assert abs(ind.fitness - expected) < 1e-9


def test_individual_to_dict_has_id():
    ind = Individual()
    d = ind.to_dict()
    assert "id" in d
    assert "fitness" in d
    assert "extraction_genome" in d
    assert "synthesis_genome" in d


def test_individual_from_dict_roundtrip():
    ind = Individual(
        extraction=ExtractionGenome(persona_idx=2),
        synthesis=SynthesisGenome(persona_idx=1),
    )
    ind.fitness = 0.55
    ind.claim_recall = 0.7
    d = ind.to_dict()
    ind2 = Individual.from_dict(d)
    assert ind2.extraction.persona_idx == 2
    assert ind2.synthesis.persona_idx == 1
    assert abs(ind2.fitness - 0.55) < 1e-9


def test_synthesis_genome_baseline_all_zeros():
    g = SynthesisGenome.baseline()
    for name in SynthesisGenome.GENE_NAMES:
        assert getattr(g, name) == 0
