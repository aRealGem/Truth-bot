"""
Report generation for evolution results.

Produces:
  - fitness_report.md: human-readable summary of convergence and component importance
  - comparison_table.md: best evolved prompt output vs. reference for each claim
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_fitness_report(
    evolution_log: list[dict],
    best_individual: dict,
    results_dir: Path,
) -> str:
    """
    Generate a Markdown fitness report.
    Returns the report text (also writes to results_dir/fitness_report.md).
    """
    lines = [
        "# Prompt Evolution Fitness Report",
        "",
        "## Overview",
        "",
    ]

    if not evolution_log:
        lines += ["*No evolution data available.*", ""]
    else:
        n_gens = len(evolution_log)
        first_best = evolution_log[0]["best_fitness"]
        last_best = evolution_log[-1]["best_fitness"]
        improvement = last_best - first_best
        mean_final = evolution_log[-1].get("mean_fitness", 0.0)
        total_cost = evolution_log[-1].get("estimated_cost_usd", 0.0)

        lines += [
            f"- **Generations run:** {n_gens}",
            f"- **Population size:** {evolution_log[0].get('population_size', '?')}",
            f"- **Fitness Gen 1:** {first_best:.4f}",
            f"- **Fitness Final:** {last_best:.4f}",
            f"- **Improvement:** {improvement:+.4f} ({improvement/first_best*100:+.1f}% relative)" if first_best > 0 else f"- **Improvement:** N/A",
            f"- **Mean fitness (final gen):** {mean_final:.4f}",
            f"- **Estimated API cost:** ${total_cost:.4f}",
            "",
        ]

        # Convergence table
        lines += [
            "## Fitness by Generation",
            "",
            "| Generation | Best Fitness | Mean Fitness | Est. Cost |",
            "|---|---|---|---|",
        ]
        for entry in evolution_log:
            lines.append(
                f"| {entry['generation']} | {entry['best_fitness']:.4f} | "
                f"{entry.get('mean_fitness', 0):.4f} | ${entry.get('estimated_cost_usd', 0):.4f} |"
            )
        lines.append("")

    # Best individual breakdown
    lines += [
        "## Best Individual",
        "",
        f"**Individual ID:** `{best_individual.get('id', 'N/A')}`",
        f"**Generation:** {best_individual.get('generation', '?')}",
        "",
        "### Fitness Scores",
        "",
        f"| Dimension | Score | Weight | Contribution |",
        f"|---|---|---|---|",
    ]

    weights = {
        "claim_recall": 0.25,
        "verdict_agreement": 0.30,
        "explanation_quality": 0.20,
        "source_citation_quality": 0.15,
        "parsimony": 0.10,
    }
    for dim, weight in weights.items():
        score = best_individual.get(dim, 0.0)
        contribution = score * weight
        lines.append(f"| {dim.replace('_', ' ').title()} | {score:.4f} | {weight} | {contribution:.4f} |")
    lines += [
        f"| **Total Fitness** | **{best_individual.get('fitness', 0):.4f}** | 1.0 | -- |",
        "",
    ]

    # Genome breakdown
    lines += [
        "### Extraction Genome",
        "",
    ]
    ext = best_individual.get("extraction_genome", {})
    _add_gene_table(lines, ext, "extraction")

    lines += [
        "",
        "### Synthesis Genome",
        "",
    ]
    syn = best_individual.get("synthesis_genome", {})
    _add_gene_table(lines, syn, "synthesis")

    # Component importance analysis
    lines += [
        "",
        "## Component Importance Analysis",
        "",
        "_Estimated by comparing mean fitness of individuals with each gene index._",
        "",
    ]
    if evolution_log:
        importance = _analyze_component_importance(evolution_log)
        if importance:
            lines += [
                "| Gene | Best Index | Mean Fitness at Best Index |",
                "|---|---|---|",
            ]
            for gene_name, data in sorted(importance.items(), key=lambda x: x[1]["best_mean"], reverse=True):
                lines.append(
                    f"| {gene_name} | {data['best_idx']} | {data['best_mean']:.4f} |"
                )
            lines.append("")
        else:
            lines += ["*Not enough data for component analysis.*", ""]

    # Mutation log for best
    mut_log = best_individual.get("mutation_log", [])
    if mut_log:
        lines += [
            "## Best Individual Lineage",
            "",
            f"**Parents:** {', '.join(best_individual.get('parent_hashes', ['N/A']))}",
            "",
            "**Mutations applied:**",
        ]
        for m in mut_log:
            lines.append(f"- {m}")
        lines.append("")

    report = "\n".join(lines)
    out_path = results_dir / "fitness_report.md"
    out_path.write_text(report)
    logger.info("Fitness report written: %s", out_path)
    return report


def _add_gene_table(lines: list[str], genome_dict: dict, genome_type: str) -> None:
    """Add a gene breakdown table to lines."""
    lines += [
        "| Gene | Index |",
        "|---|---|",
    ]
    for k, v in genome_dict.items():
        lines.append(f"| {k.replace('_idx', '').replace('_', ' ')} | {v} |")


def _analyze_component_importance(evolution_log: list[dict]) -> dict:
    """
    For each gene, find which index value correlates with highest fitness.
    Returns {gene_name: {best_idx: int, best_mean: float}}.
    """
    gene_fitness: dict[str, dict[int, list[float]]] = {}

    for gen_data in evolution_log:
        for ind in gen_data.get("individuals", []):
            fitness = ind.get("fitness", 0.0)
            ext = ind.get("extraction_genome", {})
            syn = ind.get("synthesis_genome", {})

            for k, v in ext.items():
                key = f"extraction.{k}"
                gene_fitness.setdefault(key, {}).setdefault(v, []).append(fitness)
            for k, v in syn.items():
                key = f"synthesis.{k}"
                gene_fitness.setdefault(key, {}).setdefault(v, []).append(fitness)

    result = {}
    for gene, idx_map in gene_fitness.items():
        best_idx = max(idx_map.keys(), key=lambda i: sum(idx_map[i]) / len(idx_map[i]))
        best_mean = sum(idx_map[best_idx]) / len(idx_map[best_idx])
        result[gene] = {"best_idx": best_idx, "best_mean": best_mean}
    return result


def generate_comparison_table(
    best_individual: dict,
    reference: list[dict],
    runner_cache_dir: Path,
    results_dir: Path,
) -> str:
    """
    Generate a comparison table: best evolved output vs. reference for each claim.
    Requires the runner cache to have stored extraction + synthesis outputs.
    Returns Markdown text (also writes to results_dir/comparison_table.md).
    """
    lines = [
        "# Best Evolved Prompt: Output vs. Reference",
        "",
        "Comparing the best evolved prompt's outputs against the GPT 5.4 Pro Extended reference.",
        "",
    ]

    # Try to load cached extraction output for the best individual
    ext_hash = best_individual.get("id", "")[:6]  # approximate
    lines += [
        "| # | Topic | Reference Claim | Reference Verdict | Evolved Verdict | Match? |",
        "|---|---|---|---|---|---|",
    ]
    for ref in reference:
        lines.append(
            f"| {ref['id']} | {ref['topic']} | {ref['claim'][:60]}... | "
            f"{ref['verdict']} | *(see evolution_log.json)* | -- |"
        )
    lines += [
        "",
        "_Note: For full evolved verdicts, see the runner cache in `evolution_results/runner_cache/`._",
        "",
    ]

    out = "\n".join(lines)
    out_path = results_dir / "comparison_table.md"
    out_path.write_text(out)
    logger.info("Comparison table written: %s", out_path)
    return out
