"""
Prompt genome definitions.

Each prompt is modelled as a set of independent "genes" -- modular string
components that can be swapped, crossed over, or mutated independently.
A full prompt is rendered by concatenating the active gene values.

Two genome types:
    ExtractionGenome  -- controls the claim-extraction system+user prompts
    SynthesisGenome   -- controls the verdict-synthesis system prompt
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import ClassVar


# ── Extraction genes ───────────────────────────────────────────────────────────

EXTRACTION_PERSONA_VARIANTS: list[str] = [
    # v0 -- baseline (current prompt's persona)
    "You are a professional fact-checker.",
    # v1
    "You are an investigative journalist specializing in political accountability.",
    # v2
    "You are a nonpartisan policy analyst at a nonpartisan research institute.",
    # v3
    "You are a senior researcher at a major fact-checking organization such as PolitiFact or FactCheck.org.",
    # v4
    "You are a data journalist who specializes in statistical claims and empirical policy assertions.",
]

EXTRACTION_METHODOLOGY_VARIANTS: list[str] = [
    # v0 -- baseline
    """Your job is to extract verifiable factual claims from political speech transcripts.

Rules:
1. Extract only atomic claims -- one specific assertion per item
2. Exclude opinions, predictions, and value judgments
3. Normalize claims to third-person ("The speaker claimed X" → just "X")
4. Include statistical claims, policy claims, historical claims, comparative claims
5. Exclude rhetorical questions and vague platitudes""",
    # v1 -- granular / aggressive extraction
    """Your job is to decompose every factual assertion in a political speech into discrete, checkable atomic claims.

Rules:
1. Be exhaustive -- if a sentence contains multiple numbers or assertions, split them into separate claims
2. Include statistical claims, comparative claims ("lowest in X years"), causal claims ("because of policy X"), and historical claims
3. Exclude pure opinions, predictions without a stated basis, and vague platitudes
4. Preserve specificity: exact numbers, percentages, time ranges, and named entities are vital
5. Do NOT merge multiple assertions into one claim""",
    # v2 -- conservative / high-precision
    """Your job is to extract only the most clearly verifiable factual claims from political speech.

Rules:
1. Prioritize precision over recall -- only extract claims where there is a clear right/wrong answer
2. Focus on numeric claims (statistics, dollar amounts, percentages), named comparisons ("lowest since X"), and direct cause-effect assertions backed by named policies
3. Skip anything subjective, speculative, or where "true/false" depends on framing
4. Each claim must be independently verifiable without additional context
5. Prefer quality over quantity -- 20 strong claims beat 40 borderline ones""",
    # v3 -- emphasis on source-matchability
    """Your job is to extract factual claims that can be checked against government data, news archives, or research.

Rules:
1. Extract claims that reference specific measurable outcomes (rates, counts, percentages, dollar figures)
2. Note the data source category each claim would require (e.g., BLS for employment, EIA for energy, CBP for border)
3. Capture claims involving time comparisons ("up X% since...", "lowest in Y years", "record high/low")
4. Skip vague boasts with no measurable anchor
5. Include attributional claims ("Biden did X", "Democrats caused Y") only when they reference measurable outcomes""",
]

EXTRACTION_TAXONOMY_VARIANTS: list[str] = [
    # v0 -- baseline categories
    'Category options: economy, immigration, healthcare, crime, foreign_policy, environment, education, other',
    # v1 -- expanded
    'Category options: economy, inflation, jobs, energy, immigration, crime, healthcare, foreign_policy, elections, social_policy, environment, education, budget_taxes, other',
    # v2 -- grouped
    'Category options: economic_performance (GDP, jobs, wages, inflation), immigration_border, crime_safety, energy_environment, foreign_policy_defense, domestic_policy, elections_democracy, other',
    # v3 -- fine-grained per-topic
    'Category must be one of: inflation, jobs_employment, energy_prices, immigration_border, crime_statistics, mortgage_housing, investment_trade, drug_interdiction, foreign_policy, elections_voting, federal_budget, healthcare_drugs, food_prices, other',
]

EXTRACTION_FORMAT_VARIANTS: list[str] = [
    # v0 -- baseline
    """For each claim, output a JSON object with these fields:
  - "text": The claim restated as a clear, standalone declarative sentence
  - "context": The surrounding quote from the original transcript (max 200 chars)
  - "category": Subject category
  - "is_checkable": true if this is a factual assertion, false if it's an opinion or value judgment

Return a JSON array of claim objects. Nothing else.""",
    # v1 -- richer fields
    """For each claim, output a JSON object with exactly these fields:
  - "text": The claim as a standalone declarative sentence, fully self-contained
  - "context": Verbatim surrounding quote from transcript (max 200 chars)
  - "category": Subject category from the allowed list
  - "is_checkable": boolean -- true only for empirically testable assertions
  - "specificity": "high" | "medium" | "low" -- how precisely the claim is stated
  - "data_source_hint": the type of primary source needed to verify (e.g., "BLS", "EIA", "CBP", "Freddie Mac", "general")

Return ONLY a valid JSON array. No preamble, no markdown fences.""",
    # v2 -- minimal fields, lean output
    """Return a JSON array. Each element has exactly:
  - "text": claim as a standalone sentence
  - "category": one of the category options
  - "is_checkable": boolean

No other fields. No markdown. Pure JSON array.""",
    # v3 -- with confidence and reasoning
    """For each claim output a JSON object with:
  - "text": claim as a self-contained declarative sentence
  - "context": surrounding transcript quote (≤200 chars)
  - "category": subject category
  - "is_checkable": boolean
  - "check_confidence": 0.0–1.0 float indicating how confidently this can be verified
  - "verification_note": brief note on what would be needed to verify this claim

Return ONLY a valid JSON array.""",
]

EXTRACTION_FILTERING_VARIANTS: list[str] = [
    # v0 -- baseline (implicit in rules)
    "",
    # v1 -- explicit checkability criteria
    """
A claim is checkable (is_checkable: true) if it meets ALL of:
  - References a specific, measurable outcome (number, percentage, rate, comparison)
  - Could in principle be confirmed or refuted by a public data source
  - Is not purely a statement of intent, opinion, or prediction
  - Is not a rhetorical flourish or metaphor

Mark as NOT checkable (is_checkable: false):
  - Pure value judgments ("we are the greatest nation")
  - Future predictions without stated data basis
  - Rhetorical questions
  - Vague superlatives without quantification""",
    # v2 -- stricter
    """
Checkability threshold: STRICT. Only mark is_checkable: true if:
  - There is a named statistic (%, dollar amount, count, rate) that can be looked up
  - OR there is a named comparison to a historical benchmark ("highest since 1985", "lowest in five years")
  - OR there is a named causal claim with a measurable outcome ("Policy X caused Y% change")

When in doubt, mark is_checkable: false rather than true.""",
]

EXTRACTION_EXAMPLES_VARIANTS: list[str] = [
    # v0 -- zero-shot (no examples)
    "",
    # v1 -- one example
    """
Example input: "Thanks to our energy policies, gasoline is now below $2 in most states."
Example output:
[
  {
    "text": "Gasoline prices are below $2 per gallon in most U.S. states.",
    "context": "Thanks to our energy policies, gasoline is now below $2 in most states.",
    "category": "energy_prices",
    "is_checkable": true
  }
]""",
    # v2 -- two examples including a non-checkable one
    """
Example 1:
Input: "We built the greatest economy in the history of our nation."
Output: [] (no checkable claims -- this is a vague superlative opinion)

Example 2:
Input: "Fentanyl seizures at the border are down 56 percent in the past year."
Output:
[
  {
    "text": "Fentanyl seizures at the U.S. border decreased by 56% in the past year.",
    "context": "Fentanyl seizures at the border are down 56 percent in the past year.",
    "category": "immigration_border",
    "is_checkable": true
  }
]""",
]

EXTRACTION_TONE_VARIANTS: list[str] = [
    # v0 -- neutral
    "",
    # v1 -- terse / high density
    "Be terse. Maximize claim density. Short text fields are preferred over verbose ones.",
    # v2 -- detailed / verbose
    "Be thorough. Prefer detailed claim text that preserves numerical specificity and full context over brevity.",
    # v3 -- skeptical framing
    "Approach the transcript with appropriate skepticism: flag claims that contain inherently unmeasurable assertions (e.g., 'total illegal flow') even if they appear specific.",
]


# ── Synthesis genes ────────────────────────────────────────────────────────────

SYNTHESIS_PERSONA_VARIANTS: list[str] = [
    # v0 -- baseline
    "You are an expert fact-checker.",
    # v1
    "You are a senior research analyst at a nonpartisan fact-checking organization.",
    # v2
    "You are an evidence-based policy researcher who specializes in adjudicating political claims.",
    # v3
    "You are a seasoned investigative journalist with deep expertise in data-driven fact verification.",
]

SYNTHESIS_VERDICT_TAXONOMY_VARIANTS: list[str] = [
    # v0 -- baseline
    """Given a claim and a set of evidence snippets, determine the verdict according to this taxonomy:
  - True: Accurate and supported by primary sources
  - Mostly True: Accurate but missing nuance
  - Misleading: Technically accurate framing that implies something false
  - Exaggerated: Directionally correct but overstated
  - False: Contradicted by credible evidence
  - Unverifiable: Insufficient evidence""",
    # v1 -- richer definitions
    """Assign one verdict label from this taxonomy:
  - True: The claim is accurate as stated; primary/authoritative sources confirm it
  - Mostly True: The core assertion is correct but lacks important nuance, caveats, or context
  - Misleading: Technically accurate framing designed to imply a false conclusion
  - Exaggerated: The direction is correct but the magnitude is significantly overstated
  - False: The claim is directly contradicted by credible primary or secondary sources
  - Unverifiable: The claim is inherently unmeasurable or no reliable evidence exists

Distinguish carefully between False (clear contradiction) and Misleading (technically true but deceptive framing).""",
    # v2 -- aligned to reference schema (TRUE/FALSE/PARTLY TRUE/UNSUPPORTED/MISLEADING)
    """Assess the claim using these verdict labels (pick the single best fit):
  - True: Confirmed accurate by authoritative sources
  - Mostly True: Broadly correct but with notable caveats or partial inaccuracies
  - Misleading: Technically accurate but framed to convey a false impression
  - Exaggerated: Directionally correct but the scale or degree is overstated
  - False: Directly refuted by primary data or credible independent sources
  - Unverifiable: Cannot be confirmed or denied with available public evidence

When a claim mixes accurate and inaccurate elements, choose the label that best captures the dominant character.""",
    # v3 -- with explicit priority ordering
    """Apply this verdict taxonomy with the following priority order:
  1. False -- if primary data directly contradicts the specific number or fact
  2. Misleading -- if technically accurate but contextually deceptive
  3. Exaggerated -- if directionally right but the magnitude is wrong
  4. Mostly True -- if correct with meaningful caveats
  5. True -- if confirmed by multiple independent authoritative sources
  6. Unverifiable -- only if no evidence can confirm or deny the specific claim

Prefer False over Misleading when there is a clear factual error; prefer Misleading over Exaggerated when framing is the primary issue.""",
]

SYNTHESIS_EVIDENCE_WEIGHTING_VARIANTS: list[str] = [
    # v0 -- baseline (implicit)
    "",
    # v1 -- explicit tier weighting
    """When weighing evidence, apply this trust hierarchy:
  1. Government primary data (BLS, BEA, EIA, CBP, Census, Freddie Mac) -- highest weight
  2. Wire services (AP, Reuters) -- high weight
  3. Established outlets (NYT, WaPo, CBS, BBC) -- moderate weight
  4. Fact-checking organizations (PolitiFact, FactCheck.org, Snopes) -- moderate weight
  5. Other sources -- low weight

If high-tier sources conflict with low-tier sources, defer to high-tier.""",
    # v2 -- source skepticism
    """Evidence quality hierarchy:
  - Primary data (government statistics, official reports): trust fully
  - Mainstream fact-checkers aggregating primary data: trust their synthesis
  - Individual news outlets: trust for reporting facts, be cautious on interpretation
  - Press releases and advocacy material: note but do not weight heavily

When only low-quality evidence is available, default to Unverifiable rather than guessing.""",
]

SYNTHESIS_CONFIDENCE_VARIANTS: list[str] = [
    # v0 -- baseline (implicit)
    "",
    # v1 -- calibrated
    """Calibrate confidence as follows:
  - High: Multiple independent authoritative sources agree; the verdict is clear-cut
  - Medium: Evidence points one direction but with some ambiguity or missing data
  - Low: Limited evidence, conflicting sources, or inherent measurement difficulties""",
    # v2 -- conservative
    """Default to Medium confidence unless:
  - High: At least two independent primary/government sources confirm the verdict unambiguously
  - Low: The claim is inherently unmeasurable, or sources conflict significantly""",
]

SYNTHESIS_REASONING_VARIANTS: list[str] = [
    # v0 -- holistic
    "",
    # v1 -- step-by-step
    """Before assigning a verdict, reason step-by-step:
  1. What exactly does the claim assert? (identify the specific measurable assertion)
  2. What does the best available evidence say about that assertion?
  3. Is there a discrepancy between what was claimed and what the evidence shows?
  4. Does context or framing affect the truth value?
  5. Then assign the verdict label.""",
    # v2 -- evidence inventory
    """Structure your reasoning as:
  CLAIM ANALYSIS: What specific, measurable assertion is being made?
  EVIDENCE SUMMARY: What do the sources say?
  DISCREPANCY CHECK: Does the evidence match, contradict, or nuance the claim?
  VERDICT RATIONALE: Why this label over alternatives?
  Then output the JSON verdict.""",
]

SYNTHESIS_NUANCE_VARIANTS: list[str] = [
    # v0 -- baseline (implicit)
    "",
    # v1 -- explicit nuance handling
    """Special handling for tricky claim types:
  - Inherently unmeasurable claims (e.g., "total illegal drug flow"): default to Unverifiable
  - Technically-true-but-misleading claims: use Misleading, not True
  - Claims mixing a true element with a false element: assess the overall impression conveyed
  - Comparison claims ("lowest in X years"): check both the current value AND the historical comparison
  - Causal claims ("because of policy X, Y happened"): verify Y independently of the causal attribution""",
    # v2 -- misleading detection emphasis
    """Pay special attention to misleading framing:
  - A claim is Misleading if it is literally accurate but omits critical context that would change the listener's conclusion
  - A claim is Exaggerated if the number/magnitude is wrong but the direction is right
  - A claim is False if the stated number or fact is directly incorrect per primary sources
  - Do not assign True to a claim where important context is omitted -- use Mostly True or Misleading""",
]

SYNTHESIS_FORMAT_VARIANTS: list[str] = [
    # v0 -- baseline
    """Respond with a JSON object:
{
  "label": "<verdict>",
  "confidence": "High|Medium|Low",
  "explanation": "<one paragraph explanation>",
  "support_count": <int>,
  "contradict_count": <int>
}""",
    # v1 -- with source citation emphasis
    """Respond with this JSON object exactly:
{
  "label": "<verdict label>",
  "confidence": "High|Medium|Low",
  "explanation": "<paragraph explaining verdict, citing specific data points, numbers, and named sources>",
  "support_count": <number of evidence items supporting the claim>,
  "contradict_count": <number of evidence items contradicting the claim>,
  "key_source": "<name of the single most authoritative source cited>"
}""",
    # v2 -- minimal
    """Respond with JSON only:
{
  "label": "<verdict>",
  "confidence": "High|Medium|Low",
  "explanation": "<concise explanation citing at least one specific data point>",
  "support_count": <int>,
  "contradict_count": <int>
}""",
]


# ── Genome dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ExtractionGenome:
    """
    A full set of genes for the claim extraction prompt.
    Each gene is an index into its corresponding variants list.
    """

    persona_idx: int = 0
    methodology_idx: int = 0
    taxonomy_idx: int = 0
    format_idx: int = 0
    filtering_idx: int = 0
    examples_idx: int = 0
    tone_idx: int = 0

    # Gene names in order -- used for crossover / mutation
    GENE_NAMES: ClassVar[list[str]] = [
        "persona_idx",
        "methodology_idx",
        "taxonomy_idx",
        "format_idx",
        "filtering_idx",
        "examples_idx",
        "tone_idx",
    ]
    GENE_POOLS: ClassVar[dict[str, list]] = {
        "persona_idx": EXTRACTION_PERSONA_VARIANTS,
        "methodology_idx": EXTRACTION_METHODOLOGY_VARIANTS,
        "taxonomy_idx": EXTRACTION_TAXONOMY_VARIANTS,
        "format_idx": EXTRACTION_FORMAT_VARIANTS,
        "filtering_idx": EXTRACTION_FILTERING_VARIANTS,
        "examples_idx": EXTRACTION_EXAMPLES_VARIANTS,
        "tone_idx": EXTRACTION_TONE_VARIANTS,
    }

    def render_system_prompt(self) -> str:
        """Assemble the full system prompt from active gene values."""
        parts = [
            EXTRACTION_PERSONA_VARIANTS[self.persona_idx],
            "\n\n",
            EXTRACTION_METHODOLOGY_VARIANTS[self.methodology_idx],
            "\n\n",
            EXTRACTION_TAXONOMY_VARIANTS[self.taxonomy_idx],
        ]
        if self.filtering_idx > 0:
            parts += ["\n\n", EXTRACTION_FILTERING_VARIANTS[self.filtering_idx]]
        if self.examples_idx > 0:
            parts += ["\n\n", EXTRACTION_EXAMPLES_VARIANTS[self.examples_idx]]
        if self.tone_idx > 0:
            parts += ["\n\n", EXTRACTION_TONE_VARIANTS[self.tone_idx]]
        parts += ["\n\n", EXTRACTION_FORMAT_VARIANTS[self.format_idx]]
        return "".join(parts)

    def render_user_template(self) -> str:
        """Return the user message template (speaker/date/text slots)."""
        return (
            "Extract all verifiable factual claims from the following transcript.\n"
            "Speaker: {speaker}\nDate: {date}\n\nTranscript:\n{text}\n\n"
            "Return a JSON array of claim objects."
        )

    def to_dict(self) -> dict:
        return {
            "persona_idx": self.persona_idx,
            "methodology_idx": self.methodology_idx,
            "taxonomy_idx": self.taxonomy_idx,
            "format_idx": self.format_idx,
            "filtering_idx": self.filtering_idx,
            "examples_idx": self.examples_idx,
            "tone_idx": self.tone_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionGenome":
        return cls(**{k: v for k, v in d.items() if k in cls.GENE_NAMES})

    def hash(self) -> str:
        """Stable hash of the rendered system prompt for cache keying."""
        content = self.render_system_prompt()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @classmethod
    def random(cls) -> "ExtractionGenome":
        """Create a random genome (uniform over variant indices)."""
        return cls(
            persona_idx=random.randrange(len(EXTRACTION_PERSONA_VARIANTS)),
            methodology_idx=random.randrange(len(EXTRACTION_METHODOLOGY_VARIANTS)),
            taxonomy_idx=random.randrange(len(EXTRACTION_TAXONOMY_VARIANTS)),
            format_idx=random.randrange(len(EXTRACTION_FORMAT_VARIANTS)),
            filtering_idx=random.randrange(len(EXTRACTION_FILTERING_VARIANTS)),
            examples_idx=random.randrange(len(EXTRACTION_EXAMPLES_VARIANTS)),
            tone_idx=random.randrange(len(EXTRACTION_TONE_VARIANTS)),
        )

    @classmethod
    def baseline(cls) -> "ExtractionGenome":
        """Return the baseline genome (all index 0 = current production prompts)."""
        return cls()


@dataclass
class SynthesisGenome:
    """
    A full set of genes for the verdict synthesis prompt.
    """

    persona_idx: int = 0
    verdict_taxonomy_idx: int = 0
    evidence_weighting_idx: int = 0
    confidence_idx: int = 0
    reasoning_idx: int = 0
    nuance_idx: int = 0
    format_idx: int = 0

    GENE_NAMES: ClassVar[list[str]] = [
        "persona_idx",
        "verdict_taxonomy_idx",
        "evidence_weighting_idx",
        "confidence_idx",
        "reasoning_idx",
        "nuance_idx",
        "format_idx",
    ]
    GENE_POOLS: ClassVar[dict[str, list]] = {
        "persona_idx": SYNTHESIS_PERSONA_VARIANTS,
        "verdict_taxonomy_idx": SYNTHESIS_VERDICT_TAXONOMY_VARIANTS,
        "evidence_weighting_idx": SYNTHESIS_EVIDENCE_WEIGHTING_VARIANTS,
        "confidence_idx": SYNTHESIS_CONFIDENCE_VARIANTS,
        "reasoning_idx": SYNTHESIS_REASONING_VARIANTS,
        "nuance_idx": SYNTHESIS_NUANCE_VARIANTS,
        "format_idx": SYNTHESIS_FORMAT_VARIANTS,
    }

    def render_system_prompt(self) -> str:
        parts = [
            SYNTHESIS_PERSONA_VARIANTS[self.persona_idx],
            " ",
            SYNTHESIS_VERDICT_TAXONOMY_VARIANTS[self.verdict_taxonomy_idx],
        ]
        if self.evidence_weighting_idx > 0:
            parts += ["\n\n", SYNTHESIS_EVIDENCE_WEIGHTING_VARIANTS[self.evidence_weighting_idx]]
        if self.confidence_idx > 0:
            parts += ["\n\n", SYNTHESIS_CONFIDENCE_VARIANTS[self.confidence_idx]]
        if self.reasoning_idx > 0:
            parts += ["\n\n", SYNTHESIS_REASONING_VARIANTS[self.reasoning_idx]]
        if self.nuance_idx > 0:
            parts += ["\n\n", SYNTHESIS_NUANCE_VARIANTS[self.nuance_idx]]
        parts += ["\n\n", SYNTHESIS_FORMAT_VARIANTS[self.format_idx]]
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "persona_idx": self.persona_idx,
            "verdict_taxonomy_idx": self.verdict_taxonomy_idx,
            "evidence_weighting_idx": self.evidence_weighting_idx,
            "confidence_idx": self.confidence_idx,
            "reasoning_idx": self.reasoning_idx,
            "nuance_idx": self.nuance_idx,
            "format_idx": self.format_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SynthesisGenome":
        return cls(**{k: v for k, v in d.items() if k in cls.GENE_NAMES})

    def hash(self) -> str:
        content = self.render_system_prompt()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @classmethod
    def random(cls) -> "SynthesisGenome":
        return cls(
            persona_idx=random.randrange(len(SYNTHESIS_PERSONA_VARIANTS)),
            verdict_taxonomy_idx=random.randrange(len(SYNTHESIS_VERDICT_TAXONOMY_VARIANTS)),
            evidence_weighting_idx=random.randrange(len(SYNTHESIS_EVIDENCE_WEIGHTING_VARIANTS)),
            confidence_idx=random.randrange(len(SYNTHESIS_CONFIDENCE_VARIANTS)),
            reasoning_idx=random.randrange(len(SYNTHESIS_REASONING_VARIANTS)),
            nuance_idx=random.randrange(len(SYNTHESIS_NUANCE_VARIANTS)),
            format_idx=random.randrange(len(SYNTHESIS_FORMAT_VARIANTS)),
        )

    @classmethod
    def baseline(cls) -> "SynthesisGenome":
        return cls()


@dataclass
class Individual:
    """One member of the GA population: a pair of genomes + fitness scores."""

    extraction: ExtractionGenome = field(default_factory=ExtractionGenome.baseline)
    synthesis: SynthesisGenome = field(default_factory=SynthesisGenome.baseline)

    # Fitness dimensions (populated after evaluation)
    claim_recall: float = 0.0
    verdict_agreement: float = 0.0
    explanation_quality: float = 0.0
    source_citation_quality: float = 0.0
    parsimony: float = 0.0
    fitness: float = 0.0

    # Bookkeeping
    evaluated: bool = False
    extraction_token_count: int = 0
    synthesis_token_count: int = 0
    generation: int = 0
    parent_hashes: list[str] = field(default_factory=list)
    mutation_log: list[str] = field(default_factory=list)

    # Weights for composite fitness
    WEIGHTS: ClassVar[dict[str, float]] = {
        "claim_recall": 0.25,
        "verdict_agreement": 0.30,
        "explanation_quality": 0.20,
        "source_citation_quality": 0.15,
        "parsimony": 0.10,
    }

    def compute_fitness(self) -> float:
        """Weighted sum of all fitness dimensions."""
        self.fitness = (
            self.WEIGHTS["claim_recall"] * self.claim_recall
            + self.WEIGHTS["verdict_agreement"] * self.verdict_agreement
            + self.WEIGHTS["explanation_quality"] * self.explanation_quality
            + self.WEIGHTS["source_citation_quality"] * self.source_citation_quality
            + self.WEIGHTS["parsimony"] * self.parsimony
        )
        return self.fitness

    def id(self) -> str:
        return f"e{self.extraction.hash()[:6]}_s{self.synthesis.hash()[:6]}"

    def to_dict(self) -> dict:
        return {
            "id": self.id(),
            "generation": self.generation,
            "extraction_genome": self.extraction.to_dict(),
            "synthesis_genome": self.synthesis.to_dict(),
            "fitness": self.fitness,
            "claim_recall": self.claim_recall,
            "verdict_agreement": self.verdict_agreement,
            "explanation_quality": self.explanation_quality,
            "source_citation_quality": self.source_citation_quality,
            "parsimony": self.parsimony,
            "evaluated": self.evaluated,
            "extraction_token_count": self.extraction_token_count,
            "synthesis_token_count": self.synthesis_token_count,
            "parent_hashes": self.parent_hashes,
            "mutation_log": self.mutation_log,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Individual":
        ind = cls(
            extraction=ExtractionGenome.from_dict(d["extraction_genome"]),
            synthesis=SynthesisGenome.from_dict(d["synthesis_genome"]),
        )
        ind.fitness = d.get("fitness", 0.0)
        ind.claim_recall = d.get("claim_recall", 0.0)
        ind.verdict_agreement = d.get("verdict_agreement", 0.0)
        ind.explanation_quality = d.get("explanation_quality", 0.0)
        ind.source_citation_quality = d.get("source_citation_quality", 0.0)
        ind.parsimony = d.get("parsimony", 0.0)
        ind.evaluated = d.get("evaluated", False)
        ind.generation = d.get("generation", 0)
        ind.parent_hashes = d.get("parent_hashes", [])
        ind.mutation_log = d.get("mutation_log", [])
        return ind
