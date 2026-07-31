"""
Core Pydantic data models for truth-bot.

All data that flows through the pipeline lives in one of these types.
Keeping models centralized prevents circular imports and makes serialization
(JSON, disk cache) straightforward.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class VerdictLabel(str, Enum):
    """The six possible verdict labels in the truth-bot rubric."""

    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    MISLEADING = "Misleading"
    EXAGGERATED = "Exaggerated"
    FALSE = "False"
    UNVERIFIABLE = "Unverifiable"


class Confidence(str, Enum):
    """Confidence in the verdict: how much evidence supports the rating."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SourceTier(str, Enum):
    """
    Trust hierarchy for evidence sources (descending trust).

    Used by the scoring rubric to weight conflicting evidence.
    """

    GOVERNMENT = "Government"       # BLS, FRED, CBO, Census, etc.
    WIRE = "Wire"                   # AP, Reuters
    ESTABLISHED = "Established"     # NYT, WaPo, BBC, etc.
    ACADEMIC = "Academic"           # Peer-reviewed, major NGOs
    FACTCHECK = "FactCheck"         # PolitiFact, FactCheck.org, Snopes
    OTHER = "Other"
    # Declaration order IS the trust ranking — rubric._score_evidence tracks
    # best_tier via ``list(SourceTier).index``. POLITICAL must stay LAST.
    POLITICAL = "Political"         # White House / agency press shops, party + campaign organs


# ── Core models ───────────────────────────────────────────────────────────────


class Transcript(BaseModel):
    """A raw speech or statement ingested into the pipeline."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="Full normalized transcript text")
    speaker: str = Field(default="Unknown", description="Name or title of the speaker")
    date: Optional[datetime] = Field(None, description="Date of the speech/statement")
    venue: Optional[str] = Field(None, description="Location or context (e.g. 'State of the Union')")
    source_url: Optional[str] = Field(None, description="URL where the transcript was obtained")
    word_count: int = Field(default=0)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Transcript text cannot be empty")
        return v.strip()

    def model_post_init(self, __context: Any) -> None:
        if self.word_count == 0:
            self.word_count = len(self.text.split())


class Claim(BaseModel):
    """
    An atomic, verifiable factual claim extracted from a transcript.

    A claim should be a single, specific assertion that can in principle
    be checked against evidence — not an opinion, not a value judgment.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transcript_id: str = Field(..., description="ID of the source transcript")
    text: str = Field(..., description="The claim as a self-contained sentence")
    speaker: str = Field(default="Unknown")
    context: Optional[str] = Field(None, description="Surrounding text for context")
    category: Optional[str] = Field(
        None,
        description="Subject category e.g. 'economy', 'immigration', 'healthcare'",
    )
    is_checkable: bool = Field(
        True, description="False if LLM judged it an opinion or value statement"
    )
    speech_date: Optional[datetime] = Field(
        None,
        description=(
            "Date the claim was spoken, copied from the source Transcript at "
            "extract time. Read by the temporal-preamble helper to anchor "
            "model reasoning in the correct era (fixes C10 wrong-term errors)."
        ),
    )
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Claim text cannot be empty")
        return v.strip()


class Evidence(BaseModel):
    """A single piece of evidence retrieved to evaluate a claim."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(..., description="The claim this evidence relates to")
    source_name: str = Field(..., description="Publication or dataset name")
    source_url: str = Field(..., description="Direct URL to the evidence")
    source_tier: SourceTier = Field(SourceTier.OTHER)
    snippet: str = Field(..., description="Relevant excerpt or summary")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = Field(
        None,
        description=(
            "Publication date of the source, when the connector could determine "
            "one (e.g. Brave's page_age). None = undated. Used at pack build to "
            "drop dated items outside the claim's era window (Layer C) — how a "
            "2026 PolitiFact piece stops landing in a 2022 evidence pack."
        ),
    )
    supports_claim: Optional[bool] = Field(
        None,
        description="True=supports, False=contradicts, None=ambiguous",
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0–1 relevance to the claim",
    )


class Verdict(BaseModel):
    """The final verdict on a single claim."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(...)
    label: VerdictLabel = Field(...)
    confidence: Confidence = Field(...)
    explanation: str = Field(..., description="Human-readable explanation of the verdict")
    evidence_ids: list[str] = Field(default_factory=list)
    model_id: Optional[str] = None
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    # Raw scores used internally by the rubric
    support_count: int = Field(default=0, description="Number of supporting evidence items")
    contradict_count: int = Field(default=0, description="Number of contradicting evidence items")
    primary_source_tier: Optional[SourceTier] = Field(
        None,
        description="Highest-trust tier among the evidence",
    )


class Report(BaseModel):
    """
    A complete fact-check report for one transcript.

    Contains the original transcript, all extracted claims, the evidence
    gathered, and the final verdicts — ready to publish.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transcript: Transcript
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    report_url: Optional[str] = None
    bluesky_thread_url: Optional[str] = None
    rss_feed_url: Optional[str] = None

    # Convenience counts
    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def checkable_claims(self) -> int:
        return sum(1 for c in self.claims if c.is_checkable)

    @property
    def verdict_summary(self) -> dict[str, int]:
        """Count of each verdict label across all verdicts."""
        counts: dict[str, int] = {label.value: 0 for label in VerdictLabel}
        for v in self.verdicts:
            counts[v.label.value] += 1
        return counts

    def verdict_for(self, claim_id: str) -> Optional[Verdict]:
        """Look up the verdict for a specific claim ID."""
        for v in self.verdicts:
            if v.claim_id == claim_id:
                return v
        return None


class ModelVerdict(BaseModel):
    """Verdict produced by a single LLM adapter."""
    adapter_name: str
    model_id: str
    claim_id: str
    label: VerdictLabel
    confidence: Confidence
    explanation: str
    caveats: str = Field(default="", description="Source-quality notes from the adapter")
    web_sources: list[str] = Field(default_factory=list)

    @field_validator("web_sources", mode="before")
    @classmethod
    def _sanitize_web_sources(cls, value: Any) -> list[str]:
        """Sanitize model-emitted web_sources URLs.

        Empirical fix (Phase 3b follow-up): live Gemini output
        occasionally contains doubled-scheme URLs like
        ``httpshttps://www.ebc.com/...``. These are clearly concatenation
        artifacts; we collapse them to the last scheme present. URLs that
        still lack a recognizable scheme after normalization are dropped
        rather than rendered as trusted sources.
        """
        import re

        _double_scheme_rx = re.compile(r"^https?(?=https?://)", re.IGNORECASE)

        if value is None or not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            url = item.strip()
            if not url:
                continue
            # Collapse ``httpshttps://``/``httphttps://``/``httpshttp://``
            # /``httphttp://`` prefixes down to the inner scheme. Anchored
            # at start; runs at most twice to cover ``httpshttpshttps://``
            # edge cases without unbounded looping.
            for _ in range(3):
                new = _double_scheme_rx.sub("", url, count=1)
                if new == url:
                    break
                url = new
            if url.lower().startswith(("http://", "https://")):
                out.append(url)
        return out
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    no_response: bool = Field(default=False, description="True when the adapter failed/timed out and returned no verdict")
    tier: str = Field(
        default="frontier",
        description="frontier | triage | frontier_shadow",
    )
    synthesis_mode: str = Field(
        default="live",
        description="live | batch (billing / latency mode for this call)",
    )
    cached_input_tokens: int = Field(default=0, description="Input tokens billed at cache rate, if reported")
    input_tokens: int = Field(
        default=0,
        description=(
            "Total input tokens reported by the provider for the API call that "
            "produced this verdict. For multi-claim batch calls, the full call "
            "usage is attributed to the index-0 verdict; siblings carry 0 so "
            "costs.estimate_cost does not N-count a single API call."
        ),
    )
    output_tokens: int = Field(
        default=0,
        description="Total output tokens reported by the provider (index-0 only for batch).",
    )
    batch_call_index: int = Field(
        default=0,
        description=(
            "Position of this verdict within a multi-claim API call (0 = first). "
            "Telemetry attributes the call's full usage to the index-0 verdict; "
            "siblings carry zero so cost is not N-counted."
        ),
    )
    batch_call_id: str = Field(
        default="",
        description="Opaque identifier for the multi-claim API call that produced this verdict (custom_id).",
    )
    tool_call_count: int = Field(
        default=0,
        description=(
            "Number of provider-side tool/search invocations (e.g. OpenAI "
            "``web_search_call`` output items, Anthropic ``server_tool_use`` "
            "content blocks) that fired during the API call that produced this "
            "verdict. For multi-claim batch calls the full call's count is "
            "attributed to the index-0 verdict; siblings carry 0 so per-run "
            "totals don't N-count a single API call. Fix for C6 (batch tool-"
            "call undercount): previously hardcoded to 0 for every batch row."
        ),
    )
    temporal_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Post-hoc temporal-alignment flags attached by "
            "``verify.context.validator.apply_temporal_flags``. A non-empty "
            "list means the model's reasoning referenced dates outside the "
            "expected claim window (e.g. cited Trump-I 2017 data for a 2026 "
            "claim — fix for C10 wrong-term errors). Consumed by the "
            "adjudication layer (Phase 3e) and the family-aware consensus "
            "weighting (Phase 3c)."
        ),
    )
    model_reported_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Raw URL list emitted by the model in its JSON output, before "
            "the anti-hallucination ground-truth intersection ran (Layer 1 "
            "of the anti-hallucination defense-in-depth). ``web_sources`` "
            "is the post-intersection subset; this field preserves the "
            "model's full claim for audit, fabrication-rate metrics, and "
            "the future cross-model consensus rescue path (Phase 3c — if "
            "≥2 model families independently emit the same non-tool-"
            "retrieved URL it likely isn't a hallucination)."
        ),
    )
    stripped_source_count: int = Field(
        default=0,
        description=(
            "Number of *distinct* model-reported URLs that were stripped "
            "by the ground-truth intersection because they did not appear "
            "in the search tool's retrieved-URL set. Combined with "
            "``len(model_reported_sources)`` this yields the per-call "
            "fabrication rate. Always ``<= len(model_reported_sources)`` "
            "after dedup; a value > 0 with ``tool_call_count == 0`` means "
            "the model fabricated citations from training data without "
            "running search."
        ),
    )
    url_classifications: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-URL reachability classification (Layer 4 of the anti-"
            "hallucination defense-in-depth). Keys are URLs from "
            "``web_sources``; values are the strings emitted by "
            "``truthbot.verify.url_validation.classify_failure`` — one of "
            "``ok`` / ``bot-blocked`` / ``transient`` / ``dead-4xx`` / "
            "``malformed`` / ``dns`` / ``cert-error`` / ``unknown``. "
            "Populated by ``classify_verdicts_in_place`` (or loaded from "
            "the cleaned-sidecar ``url_filter_classification`` audit "
            "field). The publish layer reads this to render the three "
            "trust tiers (verified / unverified / broken). An empty dict "
            "means the URLs were never classified, so the renderer falls "
            "back to the legacy verified-by-default rendering."
        ),
    )


class VerdictProvenance(BaseModel):
    """Pipeline provenance for a PCA (single reconciled-judge) verdict.

    The PCA lane collapses a rich adjudication row to one reconciled ``ModelVerdict``
    for display; this record preserves the structured evidence that collapse would
    otherwise discard, so per-claim agreement and the Layer A→panel→CRM-114 chain are
    reconstructable from the published bundle (see docs/pca-provenance-single-judge.md).

    All fields default empty so bundles that predate this layer — and legacy
    multi-adapter bundles that never had a PCA row — deserialize cleanly; the renderer
    treats an empty ``panel_votes`` as "not PCA mode" and falls back to the classic
    per-model strip.
    """
    layer_a_label: str = Field(default="", description="Check-worthy routing label, e.g. 'check-worthy'")
    layer_a_source: str = Field(default="", description="Which Layer A stage routed the claim: 'A1' | 'A2'")
    layer_a_claim_type: str = Field(
        default="",
        description=(
            "A2 claim_type for check-worthy rows (statistical | historical | "
            "attribution | comparison | personal-anecdote | other). Empty on "
            "A1-routed rows and pre-capture bundles. 'personal-anecdote' drives "
            "the distinct render treatment for private-person guest stories "
            "that come back Unverifiable (no independent public record)."
        ),
    )
    panel_votes: dict[str, int] = Field(
        default_factory=dict,
        description="Per-label seat tally from the PCA panel, e.g. {'True': 2, 'Misleading': 1}",
    )
    panel_split: bool = Field(default=False, description="Panel reached no plurality (genuine split)")
    panel_escalated: bool = Field(default=False, description="Escalated to the arbiter seat")
    crm114_stage1: str = Field(default="", description="CRM-114 stage-1 label (pre-override)")
    crm114_final: str = Field(default="", description="CRM-114 final label (post stage-2 override)")
    panel_by_role: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-seat labels by role, e.g. {'proposer': ['Misleading'], "
                    "'critic': ['False']} (a critic may be a panel, hence lists). "
                    "Default empty — bundles from runs predating by_role capture "
                    "deserialize cleanly and the renderer falls back to the tally.",
    )
    correction_note: str = Field(
        default="",
        description=(
            "Non-empty when this verdict was corrected post-publication via the "
            "public Corrections process (P67.6 / remediation T1.5): "
            "'Corrected OLD → NEW (YYYY-MM-DD): reason'. Rendered on the "
            "provenance strip and indexed on corrections.html."
        ),
    )
    evidence_gate: str = Field(
        default="",
        description=(
            "Pack-quality gate code (P67.7 / T2.4). "
            "'insufficient-qualifying-evidence' = the shared_pack_v2 "
            "consolidator could not meet tier quotas even after one targeted "
            "re-retrieval, so the verdict was FORCED Unverifiable. Empty on "
            "quota-met packs and all pre-v2 runs."
        ),
    )


class ConsensusVerdict(BaseModel):
    """Aggregated verdict across all active adapters."""
    claim_id: str
    model_verdicts: list[ModelVerdict]
    consensus_label: VerdictLabel
    consensus_verdict: str = Field(
        default="",
        description="Consensus verdict text; for split cases may be 'Models split'",
    )
    confidence: Confidence
    agreement: bool
    consensus_strength: str = Field(
        default="none",
        description="strong (≥3 agree) | weak (exactly 2 agree) | none (split) | single (1 model)",
    )
    explanation: str
    scored_at: datetime = Field(default_factory=datetime.utcnow)

    # ── 5-bucket coarse-axis projection (Truthy scale) ────────────────────────
    # Two parallel lenses are computed at consensus time from the existing
    # 6-bucket model labels (no model-side change). Lenient is the published
    # default; Strict is published alongside for the client-side toggle.
    # Older bundles that predate this layer deserialize cleanly with empty
    # defaults; the renderer falls back to the fine-axis pill when these are
    # blank.
    coarse_lenient_label: str = Field(
        default="",
        description=(
            "Lenient 5-bucket projection of consensus: "
            "'True' | 'Truthy' | 'Unverifiable' | 'Falsey' | 'False' | 'Models split' | ''"
        ),
    )
    coarse_lenient_strength: str = Field(
        default="none",
        description="strong | weak | none | single — strength on the Lenient projected axis",
    )
    coarse_strict_label: str = Field(
        default="",
        description=(
            "Strict 5-bucket projection of consensus (Exaggerated→Falsey): "
            "'True' | 'Truthy' | 'Unverifiable' | 'Falsey' | 'False' | 'Models split' | ''"
        ),
    )
    coarse_strict_strength: str = Field(
        default="none",
        description="strong | weak | none | single — strength on the Strict projected axis",
    )

    # ── PCA pipeline provenance ───────────────────────────────────────────────
    # Populated by the bridge for PCA (reconciled-judge) bundles; empty on legacy
    # multi-adapter bundles. See VerdictProvenance.
    provenance: VerdictProvenance = Field(default_factory=VerdictProvenance)

    @property
    def dissenting_models(self) -> list[str]:
        return [mv.adapter_name for mv in self.model_verdicts if mv.label != self.consensus_label]

class VerdictBundle(BaseModel):
    """
    Complete fact-check bundle for a single claim: all per-model verdicts,
    consensus output, and cache metadata.

    This is the primary output unit of the VerificationEngine.
    """
    bundle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim: Claim
    speaker: str = Field(default="", description="Speaker name, used in cache key")
    date_str: str = Field(default="", description="Speech date YYYY-MM-DD, used in cache key")
    model_verdicts: list[ModelVerdict]
    consensus: ConsensusVerdict
    evidence_count: int = Field(default=0)
    sources_consulted: list[dict] = Field(
        default_factory=list,
        description="Full retrieved evidence pack (ALL items, not just cited): each {id, source, url, tier, snippet}",
    )
    cache_hit: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    triage_skipped_frontier: bool = Field(
        default=False,
        description="True when unanimous high-confidence triage skipped frontier models",
    )

    @property
    def agreeing_models(self) -> list[str]:
        return [mv.adapter_name for mv in self.model_verdicts
                if mv.label == self.consensus.consensus_label]

    @property
    def dissenting_models(self) -> list[str]:
        return self.consensus.dissenting_models
