# Prompt Evolution Fitness Report

## Overview

- **Generations run:** 1
- **Population size:** 4
- **Fitness Gen 1:** 0.6786
- **Fitness Final:** 0.6786
- **Improvement:** +0.0000 (+0.0% relative)
- **Mean fitness (final gen):** 0.4534
- **Estimated API cost:** $0.6494

## Fitness by Generation

| Generation | Best Fitness | Mean Fitness | Est. Cost |
|---|---|---|---|
| 1 | 0.6786 | 0.4534 | $0.6494 |

## Best Individual

**Individual ID:** `ecd8b5e_s58c7ac`
**Generation:** 0

### Fitness Scores

| Dimension | Score | Weight | Contribution |
|---|---|---|---|
| Claim Recall | 1.0000 | 0.25 | 0.2500 |
| Verdict Agreement | 0.8138 | 0.3 | 0.2441 |
| Explanation Quality | 0.6721 | 0.2 | 0.1344 |
| Source Citation Quality | 0.3333 | 0.15 | 0.0500 |
| Parsimony | 0.0000 | 0.1 | 0.0000 |
| **Total Fitness** | **0.6786** | 1.0 | -- |

### Extraction Genome

| Gene | Index |
|---|---|
| persona | 3 |
| methodology | 2 |
| taxonomy | 2 |
| format | 3 |
| filtering | 1 |
| examples | 0 |
| tone | 1 |

### Synthesis Genome

| Gene | Index |
|---|---|
| persona | 3 |
| verdict taxonomy | 2 |
| evidence weighting | 1 |
| confidence | 1 |
| reasoning | 1 |
| nuance | 1 |
| format | 0 |

## Component Importance Analysis

_Estimated by comparing mean fitness of individuals with each gene index._

| Gene | Best Index | Mean Fitness at Best Index |
|---|---|---|
| extraction.persona_idx | 3 | 0.6786 |
| extraction.methodology_idx | 2 | 0.6786 |
| extraction.taxonomy_idx | 2 | 0.6786 |
| extraction.format_idx | 3 | 0.6786 |
| extraction.tone_idx | 1 | 0.6786 |
| synthesis.persona_idx | 3 | 0.6786 |
| synthesis.verdict_taxonomy_idx | 2 | 0.6786 |
| extraction.examples_idx | 0 | 0.6624 |
| synthesis.format_idx | 0 | 0.6624 |
| extraction.filtering_idx | 0 | 0.6462 |
| synthesis.evidence_weighting_idx | 0 | 0.6462 |
| synthesis.confidence_idx | 0 | 0.6462 |
| synthesis.reasoning_idx | 0 | 0.6462 |
| synthesis.nuance_idx | 0 | 0.6462 |
