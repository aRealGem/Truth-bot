"""
Base class and shared utilities for LLM adapters.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from truthbot.models import Claim, Confidence, Evidence, ModelVerdict, VerdictLabel
from truthbot.verify.context import apply_temporal_flags, build_temporal_preamble

logger = logging.getLogger(__name__)


# Verdict-label aliases we've seen models emit in the wild that don't match
# the canonical ``VerdictLabel`` enum verbatim. Keys are lowercased + stripped
# of dashes/underscores. Values are the canonical enum member we map to.
#
# Mapping rationale:
#   - ``mostly false`` → ``MISLEADING``  (partially-true rubric slot; the 6-label
#                                         rubric has ``MOSTLY_TRUE`` but no
#                                         ``MOSTLY_FALSE``, and ``MISLEADING``
#                                         is the closest partial-false bucket)
#   - ``half true`` / ``half false``  → ``MISLEADING``
#   - ``pants on fire``               → ``FALSE`` (PolitiFact term)
#   - ``no evidence`` / ``n/a``       → ``UNVERIFIABLE``
#
# Always logs a warning when a normalization fires so we can track how often
# models drift off-schema.
_VERDICT_LABEL_ALIASES: dict[str, VerdictLabel] = {
    "mostlyfalse":    VerdictLabel.MISLEADING,
    "halftrue":       VerdictLabel.MISLEADING,
    "halffalse":      VerdictLabel.MISLEADING,
    "pantsonfire":    VerdictLabel.FALSE,
    "noevidence":     VerdictLabel.UNVERIFIABLE,
    "na":             VerdictLabel.UNVERIFIABLE,
    "notapplicable":  VerdictLabel.UNVERIFIABLE,
    "cantverify":     VerdictLabel.UNVERIFIABLE,
    "cannotverify":   VerdictLabel.UNVERIFIABLE,
    "needscontext":   VerdictLabel.MISLEADING,
    "partlytrue":     VerdictLabel.MISLEADING,
    "partiallytrue":  VerdictLabel.MISLEADING,
    "partlyfalse":    VerdictLabel.MISLEADING,
    "partiallyfalse": VerdictLabel.MISLEADING,
    "overstated":     VerdictLabel.EXAGGERATED,
    "exaggeration":   VerdictLabel.EXAGGERATED,
}


def _canonicalize(raw: str) -> str:
    return "".join(c for c in raw.lower() if c.isalnum())


def normalize_verdict_label(raw: str) -> VerdictLabel:
    """Map a raw model-emitted label string to a canonical ``VerdictLabel``.

    Tries, in order:
      1. Exact ``VerdictLabel(raw)`` match (canonical capitalization).
      2. Case-insensitive match against enum values
         (``"true"`` → ``VerdictLabel.TRUE`` etc.).
      3. Alias map (``"mostly false"``, ``"pants on fire"``, …).

    Raises ``ValueError`` for unknown labels so callers keep their existing
    ``except Exception`` → ``UNVERIFIABLE`` path for truly off-schema output.
    """
    if not isinstance(raw, str):
        raise ValueError(f"label must be a string, got {type(raw).__name__}")

    try:
        return VerdictLabel(raw)
    except ValueError:
        pass

    key = _canonicalize(raw)
    for member in VerdictLabel:
        if _canonicalize(member.value) == key:
            logger.warning(
                "normalize_verdict_label: accepted non-canonical case/punct '%s' → %s",
                raw, member.name,
            )
            return member

    if key in _VERDICT_LABEL_ALIASES:
        mapped = _VERDICT_LABEL_ALIASES[key]
        logger.warning(
            "normalize_verdict_label: mapped alias '%s' → %s (no exact enum match)",
            raw, mapped.name,
        )
        return mapped

    raise ValueError(f"'{raw}' is not a recognized verdict label")

# OpenAI automatic prompt caching requires a stable system prefix ≥ ~1024 tokens.
# Keep this suffix byte-identical across every claim for the same model family.
_OPENAI_OPERATIONAL_SUFFIX = (
    "\n\nOperational constraints (OpenAI): Use at most 3 web searches. "
    "Keep reasoning brief. Return ONLY the JSON object."
)

SYNTHESIS_SYSTEM = """You are an expert fact-checker. Given a claim and a set of evidence snippets, \
use your web search tool to research the claim and return a verdict.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — MANDATORY PRIMARY-SOURCE SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before citing any aggregator, news outlet, or commentary site, you MUST attempt to retrieve \
relevant Tier 1 government primary sources if the claim touches any of the following domains:

  • Jobs, unemployment, wages, inflation, labor force  → search site:bls.gov
  • GDP, trade, personal income                        → search site:bea.gov
  • Federal budget, deficit, national debt             → search site:cbo.gov OR site:treasury.gov
  • Social Security, Medicare                          → search site:ssa.gov OR site:cms.gov
  • Health data, disease statistics                    → search site:cdc.gov OR site:hhs.gov
  • Census data, demographics, population              → search site:census.gov
  • Energy, oil, gas, electricity production           → search site:eia.gov
  • Education data, graduation rates, test scores      → search site:nces.ed.gov
  • Crime statistics                                   → search site:bjs.ojp.gov OR site:fbi.gov
  • Immigration, border data                           → search site:dhs.gov OR site:cbp.gov
  • Weather, climate, ocean, satellites                → search site:noaa.gov OR site:climate.gov
  • Geological hazards, maps, minerals                 → search site:usgs.gov
  • Consumer protection, antitrust, fraud              → search site:ftc.gov
  • Communications, spectrum, broadband                → search site:fcc.gov
  • Transportation safety, aviation, highways         → search site:ntsb.gov OR site:fhwa.dot.gov
  • Public transit statistics                          → search site:transit.dot.gov OR site:bts.gov
  • Agriculture, crops, food safety                  → search site:usda.gov OR site:fda.gov
  • Environmental regulations, emissions               → search site:epa.gov
  • Federal workforce, pay scales                      → search site:opm.gov
  • Workplace safety                                   → search site:osha.gov
  • Housing programs, fair housing                     → search site:hud.gov
  • Securities markets, corporate filings              → search site:sec.gov
  • Veterans benefits, health, education               → search site:va.gov
  • International affairs, treaties, aid               → search site:state.gov OR site:usaid.gov
  • Defense budgets, military personnel (public)     → search site:defense.gov OR site:dod.mil
  • Small business statistics                          → search site:sba.gov
  • Patents, trademarks, IP                            → search site:uspto.gov
  • Federal courts, opinions (public records)        → search site:courtlistener.com OR site:uscourts.gov
  • IRS tax statistics, forms guidance                 → search site:irs.gov
  • Federal procurement, contracts                     → search site:usaspending.gov OR site:sam.gov
  • NASA missions, Earth science                       → search site:nasa.gov
  • National security public datasets                  → search site:odni.gov OR site:dhs.gov
  • Nuclear regulation                                 → search site:nrc.gov
  • Mine safety                                        → search site:msha.gov
  • Railroad safety                                    → search site:dot.gov OR site:frs.dot.gov
  • Pipeline safety                                    → search site:phmsa.dot.gov
  • Maritime administration                            → search site:maritime.dot.gov
  • Federal Reserve economic data                      → search site:federalreserve.gov OR site:stlouisfed.org
  • FDIC bank data                                     → search site:fdic.gov
  • NCUA credit union data                             → search site:ncua.gov
  • Treasury interest rates, debt                      → search site:fiscaldata.treasury.gov
  • USAspending / federal awards                       → search site:usaspending.gov
  • Grants.gov program data                            → search site:grants.gov
  • Regulations.gov rulemaking                         → search site:regulations.gov
  • Federal Register                                   → search site:federalregister.gov
  • Congress bills, votes (public)                     → search site:congress.gov
  • GAO reports                                        → search site:gao.gov
  • OMB budget guidance                                → search site:whitehouse.gov/omb OR site:omb.gov
  • CFTC derivatives markets                             → search site:cftc.gov
  • FinCEN financial crimes (public)                   → search site:fincen.gov
  • BIS export controls (public)                       → search site:bis.doc.gov
  • ITA trade enforcement                              → search site:trade.gov
  • NOAA fisheries                                     → search site:fisheries.noaa.gov
  • USFS forestry                                      → search site:fs.usda.gov
  • NPS visitation, parks                              → search site:nps.gov
  • BLM land use                                       → search site:blm.gov
  • FEMA disaster programs                             → search site:fema.gov
  • ATF firearms statistics (public)                   → search site:atf.gov
  • DEA drug scheduling (public)                      → search site:dea.gov
  • SAMHSA behavioral health                           → search site:samhsa.gov
  • NIH research grants directory                      → search site:reporter.nih.gov
  • NSF science statistics                             → search site:nsf.gov
  • DOE energy programs                                → search site:energy.gov
  • NIST standards                                     → search site:nist.gov
  • USITC trade investigations                         → search site:usitc.gov
  • Ex-Im Bank financing (public)                      → search site:exim.gov
  • OPIC / DFC development finance (public)          → search site:dfc.gov
  • Peace Corps programs                               → search site:peacecorps.gov
  • Smithsonian collections (public)                   → search site:si.edu
  • Library of Congress collections                    → search site:loc.gov
  • National Archives records                          → search site:archives.gov
  • Census economic indicators                         → search site:census.gov/economic-indicators
  • BEA regional accounts                              → search site:apps.bea.gov
  • BLS CPI/PPI methodology                          → search site:bls.gov/cpi
  • CMS open data                                      → search site:data.cms.gov
  • HRSA health workforce                              → search site:hrsa.gov
  • AHRQ quality measures                              → search site:ahrq.gov
  • FDA drug approvals database                        → search site:accessdata.fda.gov
  • USDA nutrition database                            → search site:fdc.nal.usda.gov
  • EPA enforcement/compliance                         → search site:echo.epa.gov
  • DOT crash statistics (NHTSA)                       → search site:nhtsa.gov OR site:crashstats.nhtsa.dot.gov
  • FAA operations data                                → search site:faa.gov/data
  • TSA throughput (public)                            → search site:tsa.gov
  • CBP travel statistics                              → search site:cbp.gov/newsroom/stats
  • ICE detention statistics (public)                  → search site:ice.gov
  • USCIS immigration statistics                       → search site:uscis.gov/tools/reports-and-studies
  • State visa bulletin                                → search site:travel.state.gov
  • USTR trade agreements                              → search site:ustr.gov
  • U.S. Mint coin production                          → search site:usmint.gov
  • Bureau of Engraving currency                       → search site:bep.gov
  • USPS service performance                         → search site:usps.com/household/service-performance
  • FCC broadband maps                                 → search site:broadbandmap.fcc.gov
  • NTIA broadband programs                            → search site:ntia.doc.gov
  • NSF NCSES science indicators                       → search site:ncses.nsf.gov
  • ED data express                                    → search site:ed.gov/data
  • College Scorecard                                  → search site:collegescorecard.ed.gov
  • OSHA enforcement data                              → search site:enforcedata.dol.gov
  • MSHA mine enforcement                              → search site:arlweb.msha.gov
  • BLS JOLTS, CES, CPS program pages                  → search site:bls.gov/jlt
  • Treasury OFAC sanctions lists                    → search site:ofac.treasury.gov
  • FinCEN advisories (public)                         → search site:fincen.gov/resources
  • SEC EDGAR company search                           → search site:sec.gov/edgar
  • CPSC recalls                                       → search site:cpsc.gov/Recalls
  • NHTSA recalls                                      → search site:nhtsa.gov/recalls
  • USDA FSIS recalls                                  → search site:fsis.usda.gov/recalls
  • FDA recalls                                        → search site:fda.gov/safety/recalls-market-withdrawals-safety-alerts
  • EPA recalls / advisories                           → search site:epa.gov/enforcement
  • NOAA hurricane advisories                          → search site:nhc.noaa.gov
  • USGS earthquake feeds                              → search site:earthquake.usgs.gov
  • US drought monitor                                 → search site:droughtmonitor.unl.edu
  • USDA crop progress                                 → search site:nass.usda.gov
  • EIA weekly petroleum                               → search site:eia.gov/petroleum
  • EIA electricity                                    → search site:eia.gov/electricity
  • FRED macro series                                  → search site:fred.stlouisfed.org
  • Census building permits                            → search site:census.gov/construction
  • HUD point-in-time homelessness                     → search site:huduser.gov
  • VA open data                                       → search site:data.va.gov
  • SSA actuarial publications                         → search site:ssa.gov/oact
  • CMS rate filings                                   → search site:cms.gov/marketplace
  • IRS SOI tax stats                                  → search site:irs.gov/statistics/soi-tax-stats
  • Treasury fiscal service                            → search site:fiscal.treasury.gov
  • BLS QCEW employer data                           → search site:bls.gov/cew
  • OSHA injury illness rates                          → search site:bls.gov/iif
  • BTS airline on-time                                → search site:transtats.bts.gov
  • FAA ASIAS (public summaries)                       → search site:faa.gov/data
  • TSA checkpoint travel numbers                      → search site:tsa.gov/travel/passenger-volumes
  • CBP enforcement statistics                         → search site:cbp.gov/newsroom/stats
  • USCIS processing times                             → search site:egov.uscis.gov/processing-times
  • State human rights reports (public)                → search site:state.gov/reports
  • USAID data                                         → search site:data.usaid.gov
  • World Bank open data (cross-check U.S. claims)     → search site:data.worldbank.org
  • IMF data (cross-check macro claims)                → search site:imf.org/en/data
  • OECD data (cross-check policy comparisons)         → search site:oecd.org
  • UN population data                                 → search site:population.un.org
  • WHO health statistics                              → search site:who.int/data
  • IEA energy statistics                              → search site:iea.org/data-statistics

Rules:
  - If a Tier 1 search returns relevant results, those MUST appear as primary evidence.
  - Aggregators, news outlets, and commentary (Tier 2–5) may be cited in addition, never instead.
  - If a Tier 1 search returns no relevant results, note it explicitly in the caveats field \
and then proceed with lower-tier sources.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — VERDICT SYNTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After gathering evidence, determine the verdict using this taxonomy:

  - True:          Accurate and supported by primary sources
  - Mostly True:   Accurate but missing important nuance or context
  - Misleading:    Technically accurate framing that implies something false
  - Exaggerated:   Directionally correct but substantially overstated
  - False:         Contradicted by credible evidence
  - Unverifiable:  Insufficient evidence to confirm or deny

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — VERDICT VALIDATION (before returning output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before returning your verdict, review your evidence list:
  - If the claim falls into any domain listed in Step 1 AND your evidence contains zero Tier 1 \
government sources, you must either:
      (a) perform an additional targeted Tier 1 search using the site: operators above, or
      (b) add a caveat: "No Tier 1 primary source retrieved despite relevant domain; \
verdict rests on secondary analysis."
  - A verdict backed only by aggregators or commentary on a quantitative government data claim \
is not acceptable without this caveat.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION DISCIPLINE — anti-hallucination requirement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The "web_sources" array MUST contain ONLY URLs that the web_search tool \
returned during this exact call. Do NOT fabricate URLs from training data, \
do NOT guess URL patterns even for sites you know well (bls.gov, cbp.gov, \
whitehouse.gov, etc.), and do NOT reconstruct URLs from a snippet \
description. A URL you cite must be one your tool retrieved verbatim.

If you want to reference a source you remember but did not retrieve, \
describe the source in `caveats` (e.g. "Per BLS CPI release Feb 2026, \
unverified at retrieval time") rather than emitting a URL for it.

If the web_search tool returned zero relevant URLs for this claim, return \
"web_sources": [] and either:
  - set "confidence": "Low" with a verdict label other than Unverifiable if \
the claim is otherwise supported by retrieved evidence text, OR
  - set "label": "Unverifiable" if no retrieved evidence supports a verdict.

Fabricated URLs are stripped automatically by a downstream ground-truth \
intersection. URLs you fabricate will NOT appear in the published report; \
they will appear in fabrication-rate telemetry against your model. \
Citation discipline is enforced.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY raw JSON (no markdown fences, no preamble):
{
  "label": "<verdict>",
  "confidence": "High|Medium|Low",
  "explanation": "<one paragraph citing specific sources and data points>",
  "caveats": "<source-quality notes, or empty string if none>",
  "web_sources": ["<url1>", "<url2>", ...]
}"""

# Stable OpenAI system prefix (shared rubric + provider suffix) for prompt-cache hits.
OPENAI_SYNTHESIS_SYSTEM = SYNTHESIS_SYSTEM + _OPENAI_OPERATIONAL_SUFFIX

EVIDENCE_CAVEAT = (
    "Pre-gathered evidence (use at your discretion — consider source credibility; "
    "you may rely on your own web search instead):\n\n"
)


def build_user_message(
    claim: Claim,
    evidence: list[Evidence],
    *,
    inject_evidence: bool = True,
) -> str:
    """
    Build the per-claim user message. When inject_evidence is True and snippets exist,
    include them with a caveat that models may ignore them and use web search instead.

    Every message is prefixed with a ``build_temporal_preamble`` block that
    anchors the model's reasoning to the speech era (fixes C10 wrong-term
    errors) and declares post-training-cutoff search results as primary
    evidence (fixes C3 "this must be fiction"). The preamble lives in the
    user message rather than the system prompt so that Anthropic/OpenAI
    prompt caching on the stable SYNTHESIS_SYSTEM prefix continues to hit.
    """
    preamble = build_temporal_preamble(claim)
    if not inject_evidence:
        return (
            f"{preamble}"
            f"Claim: {claim.text}\n\n"
            f"Speaker: {claim.speaker}\n\n"
            "Use your web search tool to research this claim. "
            "No pre-gathered evidence was supplied."
        )

    evidence_text = "\n\n".join(
        f"[{i+1}] {e.source_name} ({e.source_tier.value})\n{e.snippet}"
        for i, e in enumerate(evidence[:5])
    )
    if evidence_text:
        return (
            f"{preamble}"
            f"Claim: {claim.text}\n\n"
            f"Speaker: {claim.speaker}\n\n"
            f"{EVIDENCE_CAVEAT}"
            f"Evidence:\n{evidence_text}"
        )
    return (
        f"{preamble}"
        f"Claim: {claim.text}\n\n"
        f"Speaker: {claim.speaker}\n\n"
        "No pre-gathered evidence available. Please use web search to research this claim."
    )


# ── Multi-claim batching helpers ──────────────────────────────────────────────

_MULTI_CLAIM_PREAMBLE = (
    "You will verify {n} claims in a single request. Return ONLY a JSON array "
    "with one object per claim, keyed by \"claim_id\" matching the IDs below. "
    "Do NOT merge, reorder, or omit claims. Each object must include the same "
    "fields as a single-claim verdict (label, confidence, explanation, caveats, "
    "web_sources).\n\n"
    "CRITICAL — per-claim web_sources attribution:\n"
    "Each claim's `web_sources` array MUST list the URLs the web_search tool "
    "retrieved while researching THAT specific claim, exactly as the "
    "single-claim CITATION DISCIPLINE rule above requires. Do NOT collapse "
    "URLs into a single shared list across claims, do NOT leave `web_sources` "
    "empty just because you researched multiple claims in one request, and do "
    "NOT omit the field. If you genuinely retrieved no URL relevant to a "
    "particular claim, set its `web_sources` to [] and lower that claim's "
    "confidence accordingly. The CITATION DISCIPLINE rule applies per-claim "
    "in this multi-claim format, not request-wide.\n\n"
)

_MULTI_CLAIM_OUTPUT_SCHEMA = (
    "\n\nRespond with ONLY a raw JSON array (no markdown fences, no preamble):\n"
    "[\n"
    "  {\n"
    "    \"claim_id\": \"<id>\",\n"
    "    \"label\": \"True|Mostly True|Misleading|Exaggerated|False|Unverifiable\",\n"
    "    \"confidence\": \"High|Medium|Low\",\n"
    "    \"explanation\": \"<one paragraph>\",\n"
    "    \"caveats\": \"<source-quality notes, or empty string>\",\n"
    "    \"web_sources\": [\"<url1>\", \"<url2>\", ...]\n"
    "  },\n"
    "  ...one object per claim, in the order listed above...\n"
    "]\n\n"
    "Field rules: every object MUST include the `web_sources` field "
    "(do not omit). It is a JSON array of URL strings retrieved by the "
    "web_search tool while researching THAT claim, or `[]` when no URL was "
    "retrieved for that claim. JSON does not allow comments, so do NOT "
    "include any `//` inline annotations in your output."
)


def _short_claim_tag(claim_id: str) -> str:
    """A stable, short label for a claim used inside multi-claim prompts."""
    return claim_id[:12]


def build_multi_user_message(
    claims: list[Claim],
    evidence_by_claim: dict[str, list[Evidence]],
    *,
    inject_evidence: bool = True,
    max_evidence_per_claim: int = 5,
) -> str:
    """
    Build a single user message enumerating multiple claims for one LLM call.

    The message preserves per-claim structure (speaker, pre-gathered evidence)
    and asks the model to return a JSON array keyed by ``claim_id``. Callers
    should not invoke this with an empty list.
    """
    if not claims:
        raise ValueError("build_multi_user_message requires at least one claim")

    # Temporal preamble: one block at the top of the multi-claim message.
    # Uses claims[0] as the representative claim — all claims in a single
    # multi-claim request share a transcript (and therefore a speech_date +
    # speaker) via the BatchDispatcher's chunking contract.
    blocks: list[str] = [
        build_temporal_preamble(claims[0]),
        _MULTI_CLAIM_PREAMBLE.format(n=len(claims)),
    ]
    for idx, claim in enumerate(claims, start=1):
        tag = _short_claim_tag(claim.id)
        evidence = evidence_by_claim.get(claim.id, []) if inject_evidence else []
        evidence = evidence[:max_evidence_per_claim]

        block = [
            f"Claim #{idx}  (claim_id={claim.id}, tag={tag})",
            f"  Speaker: {claim.speaker}",
            f"  Text: {claim.text}",
        ]
        if evidence:
            ev_lines = "\n".join(
                f"    [{i+1}] {e.source_name} ({e.source_tier.value}) — {e.snippet}"
                for i, e in enumerate(evidence)
            )
            block.append(f"  {EVIDENCE_CAVEAT.strip()}")
            block.append(ev_lines)
        else:
            block.append(
                "  No pre-gathered evidence; use your web search tool to research this claim."
            )
        blocks.append("\n".join(block))

    blocks.append(_MULTI_CLAIM_OUTPUT_SCHEMA)
    return "\n\n".join(blocks)


def _extract_json_array(text: str) -> list:
    """Parse a JSON array out of raw model text, tolerating markdown fences."""
    if not text:
        raise json.JSONDecodeError("empty response", text or "", 0)
    cleaned = text.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("verdicts", "results", "claims", "items"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            return [parsed]
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("verdicts", "results", "claims", "items"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]
                return [parsed]
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No JSON array found in response", cleaned, 0)


def _normalize_url_for_compare(url: str) -> str:
    """Normalize ``url`` into a comparison-only key.

    Used by :func:`ground_truth_web_sources` to decide whether a model-reported
    URL matches one of the URLs the search tool actually returned. Comparison
    is liberal — most cosmetic differences (case, fragment, default port,
    leading ``www.``, trailing ``/``) are collapsed so that a model that lightly
    rewrites a tool URL still matches.

    The original URL strings are *never* mutated in caller-visible output;
    this is purely a comparison key.

    Returns ``""`` for inputs that aren't a recognizable HTTP/HTTPS URL — the
    caller treats those as non-matching (i.e. stripped).
    """
    if not isinstance(url, str):
        return ""
    s = url.strip()
    if not s:
        return ""
    if not s.lower().startswith(("http://", "https://")):
        return ""
    from urllib.parse import urlparse, urlunparse

    try:
        p = urlparse(s)
    except Exception:
        return ""
    scheme = (p.scheme or "").lower()
    netloc = (p.netloc or "").lower()
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    elif scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path or ""
    if path == "/":
        path = ""
    elif len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunparse((scheme, netloc, path, p.params or "", p.query or "", ""))


def ground_truth_web_sources(
    model_reported: Optional[list[str]],
    tool_retrieved: Optional[list[str]],
) -> tuple[list[str], int]:
    """Filter ``model_reported`` to URLs that intersect ``tool_retrieved``.

    This is the **anti-hallucination intersection** invoked by every adapter
    after the model emits its JSON verdict. Models occasionally fabricate
    URLs that *look* like real citations (correct domain pattern, plausible
    slug) but were never returned by the web_search tool. We strip those.

    Comparison is performed via :func:`_normalize_url_for_compare`; matching
    URLs are returned in their **original model-reported form** (so the
    publish layer still shows what the model emitted, not a normalized
    variant). Order from ``model_reported`` is preserved; duplicates are
    collapsed (first occurrence wins).

    Returns
    -------
    (kept, stripped_count)
        ``kept`` — the subset of ``model_reported`` URLs that match at least
        one URL in ``tool_retrieved``, deduplicated.
        ``stripped_count`` — number of distinct model-reported URLs that did
        NOT match, including malformed entries. Useful for fabrication-rate
        telemetry.

    Edge cases
    ----------
    * ``model_reported`` is None / empty → ``([], 0)``.
    * ``tool_retrieved`` is None / empty → strict mode: every model-reported
      URL is stripped. (Caller is expected to combine this with
      ``tool_call_count`` to decide whether to mark the verdict
      ``Unverifiable`` per the CITATION DISCIPLINE prompt block.)
    """
    reported = list(model_reported or [])
    if not reported:
        return [], 0

    truth_keys: set[str] = set()
    for u in tool_retrieved or []:
        k = _normalize_url_for_compare(u)
        if k:
            truth_keys.add(k)

    kept: list[str] = []
    seen_kept: set[str] = set()
    stripped_keys: set[str] = set()

    for raw in reported:
        key = _normalize_url_for_compare(raw)
        if not key:
            stripped_keys.add(repr(raw))
            continue
        if key in truth_keys:
            if key not in seen_kept:
                kept.append(raw)
                seen_kept.add(key)
        else:
            stripped_keys.add(key)

    return kept, len(stripped_keys)


def apply_url_grounding(
    raw: dict,
    tool_retrieved: Optional[list[str]],
    *,
    fallback_limit: int = 10,
    tool_call_count: int = 0,
) -> tuple[list[str], list[str], int]:
    """Compute the three URL fields for a single ``ModelVerdict`` (Layer 1d).

    Returns ``(web_sources, model_reported_sources, stripped_source_count)``:
        * ``web_sources``           — what the publish layer renders.
        * ``model_reported_sources``— the model's raw self-reported URLs
          before intersection (audit trail / Phase 3c consensus).
        * ``stripped_source_count`` — distinct model-reported URLs that
          failed the ground-truth intersection (fabrication-rate metric).

    Behavior:
      * Model omitted ``web_sources`` entirely → fall back to up to
        ``fallback_limit`` tool-retrieved URLs in ``web_sources``,
        empty ``model_reported_sources``, zero stripped count. These
        URLs are tool-grounded by definition so this is not a
        fabrication.
      * Model emitted ``web_sources`` AND
        ``tool_call_count > 0`` AND ``tool_retrieved`` is empty
        (trust-when-fired) → bypass intersection and trust the
        model. Search/grounding tool fired but the harness extractor
        returned no URLs (e.g., OpenAI Responses API JSON-output mode
        produces no inline ``url_citation`` annotations; Gemini's
        ``vertexaisearch`` redirect resolver requires a session
        cookie our harness can't supply, so
        :func:`resolve_gemini_redirect` drops every URL it sees).
        The model's emitted URLs are what the search tool returned
        in-context; the strip would be a harness-attribution false
        positive. xAI / Anthropic prove the relaxed semantics are
        safe: their adapters capture URLs cleanly and run with 0%
        strip under the same anti-fabrication intent. The fallback
        only fires when the tool actually invoked, so it does NOT
        weaken anti-fabrication for runs where the model declined to
        search. Diagnosis +
        evidence: ``metrics/adapter_interpretability/strip_audit_2026-05.md``.
      * Model emitted ``web_sources`` and the harness captured at
        least one tool URL → run :func:`ground_truth_web_sources`
        intersection (strict mode); the survivor set becomes
        ``web_sources``, the raw list stays on
        ``model_reported_sources``, and the strip count is the
        fabrication-rate metric (subject to harness-completeness
        caveats — see docs).
      * ``tool_call_count`` defaults to 0 so legacy callers see no
        behavior change.
    """
    raw_ws = raw.get("web_sources", None)
    tool_retrieved = list(tool_retrieved or [])
    if raw_ws is None:
        return list(tool_retrieved[:fallback_limit]), [], 0
    model_reported = [u for u in (raw_ws or []) if isinstance(u, str)]
    if tool_call_count > 0 and not tool_retrieved and model_reported:
        # Trust-when-fired: harness saw nothing, tool actually ran,
        # model has URLs to attest. Bypass intersection.
        return list(model_reported), list(model_reported), 0
    kept, stripped = ground_truth_web_sources(model_reported, tool_retrieved)
    # Diagnostic for the "strip everything" case (kept=[] AND model_reported
    # non-empty AND stripped > 0). Disambiguates the two ways this can
    # happen post-fallback:
    #   (A) tool_call_count == 0 — model declined to invoke search;
    #       strict strip is correctly anti-fabrication. Fix path is
    #       prompt-side ("tool_choice=required" or stricter copy).
    #   (B) tool_call_count > 0 AND tool_retrieved non-empty but no
    #       overlap with model_reported — URL near-miss (e.g.,
    #       .htm vs .pdf same-release). Fix path is fuzzy intersection.
    # The trust-when-fired branch above already handles the third case
    # (tool fired + retrieved empty), so it doesn't appear here. Sample
    # URLs are logged so operators can eyeball whether the model is
    # citing real-but-non-matching URLs (case B) or pure assertion
    # (case A). Diagnostic only — no behavior change.
    if stripped > 0 and not kept and model_reported:
        logger.warning(
            "apply_url_grounding strip-no-keep: tool_count=%d "
            "retrieved=%d reported=%d stripped=%d "
            "sample_reported=%s sample_retrieved=%s",
            tool_call_count,
            len(tool_retrieved),
            len(model_reported),
            stripped,
            model_reported[:2],
            tool_retrieved[:2],
        )
    return kept, model_reported, stripped


def parse_multi_claim_json(text: str, claims: list[Claim]) -> dict[str, dict]:
    """
    Parse a multi-claim model response into a ``{claim_id: raw_verdict_dict}`` map.

    Keys by ``claim_id`` when the model returns it; otherwise falls back to
    positional matching against ``claims``. Missing entries are simply absent
    from the returned map — callers are responsible for filling them as
    ``UNVERIFIABLE no_response=True``.
    """
    rows = _extract_json_array(text)

    by_id: dict[str, dict] = {}
    leftover: list[dict] = []
    claim_ids = {c.id for c in claims}

    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = row.get("claim_id") or row.get("id")
        if isinstance(cid, str) and cid in claim_ids:
            by_id.setdefault(cid, row)
        else:
            leftover.append(row)

    if leftover:
        missing = [c for c in claims if c.id not in by_id]
        for claim, row in zip(missing, leftover):
            by_id.setdefault(claim.id, row)

    return by_id


def build_multi_verdicts(
    claims: list[Claim],
    raw_by_claim: dict[str, dict],
    *,
    adapter_name: str,
    model_id: str,
    synthesis_mode: str = "batch",
    tier: str = "frontier",
    call_usage: Optional[dict[str, Any]] = None,
    batch_call_id: str = "",
    tool_retrieved_urls: Optional[list[str]] = None,
) -> list[ModelVerdict]:
    """
    Convert a parsed multi-claim response into one ``ModelVerdict`` per claim.

    Usage (input/output/cached tokens) is attributed to the first verdict in
    the returned list via ``batch_call_index=0``; siblings get index > 0 and
    zero usage so ``costs.estimate_cost`` does not N-count a single API call.

    ``call_usage`` keys honored: ``input_tokens``, ``output_tokens``,
    ``cached_input_tokens``, ``tool_call_count``. Missing keys default to 0.
    ``tool_call_count`` is attributed to the index-0 verdict (like tokens) so
    per-run aggregation does not N-count a single API call's tool usage.

    Anti-hallucination Layer 1d: ``tool_retrieved_urls`` is the set of URLs
    the search tool actually returned during this batch call (a single URL
    set covers all claims since they share one API call). Each verdict's
    model-reported ``web_sources`` is intersected against that set via
    :func:`apply_url_grounding`; matching URLs become ``web_sources``, the
    raw list is preserved on ``model_reported_sources``, and
    ``stripped_source_count`` records how many distinct URLs were rejected
    (fabrication signal). When ``tool_retrieved_urls`` is ``None`` (legacy
    callers that haven't been wired yet), grounding is skipped and the
    pre-Layer-1d behavior is preserved.

    Defensive ``model_reported_sources`` backfill (2026-04-26): when the
    model emits empty / missing ``web_sources`` but the search tool DID
    retrieve URLs during the chunk, every claim in the chunk gets
    ``model_reported_sources`` populated with up to 10 of those tool URLs.
    This preserves the audit trail / cross-claim consensus signal for
    OpenAI / Gemini / xAI multi-claim — they routinely drop per-claim
    attribution despite the search tool firing 6-27 times per chunk. We
    deliberately do NOT fan tool URLs out to siblings' ``web_sources``
    (visible publish-layer field) because that would falsely suggest each
    claim was independently grounded; the index-0 verdict still gets the
    legacy visible-grounding fallback so the report shows at least one
    cited source per chunk. Trade-off: attribution fidelity for siblings,
    visible grounding for index-0.
    """
    usage = call_usage or {}

    # Batch-level trust-when-fired observability (2026-05-01). The actual
    # fallback now lives inside :func:`apply_url_grounding` so it applies
    # universally to every adapter that calls it (Anthropic / Gemini /
    # Grok / OpenAI single batch + live, AND build_multi_verdicts here).
    # We keep ONE WARNING per batch — not N per claim — so operators can
    # spot the harness-gap signature in stderr without N-flooding the
    # log. See metrics/adapter_interpretability/strip_audit_2026-05.md.
    batch_tool_count = int(usage.get("tool_call_count", 0) or 0)
    if (
        isinstance(tool_retrieved_urls, list)
        and len(tool_retrieved_urls) == 0
        and batch_tool_count > 0
    ):
        logger.warning(
            "%s build_multi_verdicts trust-when-fired: search tool fired "
            "(count=%d) but harness extracted no URLs; trusting "
            "model-emitted web_sources for this batch (claims=%d, "
            "batch_call_id=%s).",
            adapter_name,
            batch_tool_count,
            len(claims),
            batch_call_id or "",
        )

    out: list[ModelVerdict] = []
    for idx, claim in enumerate(claims):
        raw = raw_by_claim.get(claim.id)
        if raw is None:
            out.append(
                ModelVerdict(
                    adapter_name=adapter_name,
                    model_id=model_id,
                    claim_id=claim.id,
                    label=VerdictLabel.UNVERIFIABLE,
                    confidence=Confidence.LOW,
                    explanation="batch partial response: no verdict for this claim",
                    tier=tier,
                    synthesis_mode=synthesis_mode,
                    no_response=True,
                    batch_call_index=idx,
                    batch_call_id=batch_call_id,
                )
            )
            continue
        try:
            label = normalize_verdict_label(raw["label"])
            confidence = Confidence(raw["confidence"])
        except Exception as exc:
            logger.warning(
                "%s multi-claim parse: bad label/confidence for %s: %s",
                adapter_name,
                claim.id,
                exc,
            )
            out.append(
                ModelVerdict(
                    adapter_name=adapter_name,
                    model_id=model_id,
                    claim_id=claim.id,
                    label=VerdictLabel.UNVERIFIABLE,
                    confidence=Confidence.LOW,
                    explanation=f"Failed to parse verdict fields: {exc}",
                    tier=tier,
                    synthesis_mode=synthesis_mode,
                    no_response=True,
                    batch_call_index=idx,
                    batch_call_id=batch_call_id,
                )
            )
            continue

        if idx == 0:
            cached_t = int(usage.get("cached_input_tokens", 0) or 0)
            input_t = int(usage.get("input_tokens", 0) or 0)
            output_t = int(usage.get("output_tokens", 0) or 0)
            tool_count = int(usage.get("tool_call_count", 0) or 0)
        else:
            cached_t = 0
            input_t = 0
            output_t = 0
            tool_count = 0
        if tool_retrieved_urls is None:
            ws = list(raw.get("web_sources", []) or [])
            mrs: list[str] = []
            stripped = 0
        elif raw.get("web_sources") is None:
            # Model omitted ``web_sources`` for this claim entirely. We do
            # NOT fan tool URLs out to siblings' ``web_sources`` (publish-
            # layer field) because that would falsely suggest each claim
            # was independently grounded. Two backfills happen instead:
            #   1. ``model_reported_sources`` gets the chunk's tool URLs
            #      for every claim, so audit trails / cross-claim
            #      consensus see grounding signal even when multi-claim
            #      providers drop per-claim attribution.
            #   2. Index-0 (the "call owner") also gets ``web_sources``
            #      populated, preserving the legacy visible-grounding
            #      behavior so the report shows at least one cited source
            #      per chunk.
            tool_urls = list((tool_retrieved_urls or [])[:10])
            ws = list(tool_urls) if (idx == 0 and tool_urls) else []
            mrs = list(tool_urls)
            stripped = 0
        else:
            # apply_url_grounding handles trust-when-fired internally
            # when ``tool_call_count > 0`` and ``tool_retrieved_urls``
            # is empty. Pass the BATCH-level tool count (not the
            # idx==0-only attributed count) so siblings benefit too.
            ws, mrs, stripped = apply_url_grounding(
                raw, tool_retrieved_urls, tool_call_count=batch_tool_count
            )
            # Same defensive backfill when the model emitted an explicit
            # empty ``web_sources: []`` array and the tool DID retrieve
            # URLs. Distinguishes "model said nothing was relevant" from
            # "model researched but didn't bother attributing" — both look
            # identical post-grounding (ws=[], mrs=[]), so we only backfill
            # when the tool actually fired (tool_retrieved_urls non-empty).
            if not ws and not mrs and tool_retrieved_urls:
                tool_urls = list(tool_retrieved_urls[:10])
                mrs = list(tool_urls)
                if idx == 0:
                    ws = list(tool_urls)
        verdict = ModelVerdict(
            adapter_name=adapter_name,
            model_id=model_id,
            claim_id=claim.id,
            label=label,
            confidence=confidence,
            explanation=raw.get("explanation", ""),
            caveats=raw.get("caveats", ""),
            web_sources=ws,
            model_reported_sources=mrs,
            stripped_source_count=stripped,
            tier=tier,
            synthesis_mode=synthesis_mode,
            cached_input_tokens=cached_t,
            input_tokens=input_t,
            output_tokens=output_t,
            batch_call_index=idx,
            batch_call_id=batch_call_id,
            tool_call_count=tool_count,
        )
        apply_temporal_flags(verdict, claim)
        out.append(verdict)
    return out


class AdapterUnavailable(Exception):
    """Raised when a required API key is missing."""
    pass


class LLMAdapter(ABC):
    """Abstract base class for LLM fact-checking adapters."""

    adapter_name: str
    model_id: str
    required_env_key: str

    def __init__(self) -> None:
        if not os.environ.get(self.required_env_key):
            raise AdapterUnavailable(f"{self.required_env_key} not set")

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the required API key is present in the environment."""
        return bool(os.environ.get(cls.required_env_key))

    @abstractmethod
    def call(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
        telemetry_tier: str = "frontier",
        run_id: Optional[str] = None,
    ) -> ModelVerdict:
        """Call the LLM with claim + evidence and return a ModelVerdict."""
        ...

    def _build_user_message(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
    ) -> str:
        """Build the user message string for the LLM prompt."""
        return build_user_message(claim, evidence, inject_evidence=inject_evidence)

    # ── Batch API support (optional, provider-specific) ───────────────────────

    #: True when the adapter ships a production ``build_batch_payload``/``parse_batch_response``
    #: pair and can be routed through the BatchDispatcher.
    supports_batch: bool = False

    #: Maximum number of atomic claims the adapter can safely fold into a single
    #: request. Default ``1`` means the adapter only handles single-claim mode;
    #: concrete adapters that implement ``build_multi_batch_payload`` should
    #: raise this to the empirically-safe ceiling (context window / tool-call
    #: budget / cost). ``BatchDispatcher`` clamps the user-requested chunk size
    #: at submit time.
    max_claims_per_request: int = 1

    def build_batch_payload(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
    ) -> dict:
        """
        Return the provider-specific request body for a batch submission.

        The returned dict should be ready to hand to the provider's batch SDK
        (e.g. as ``params`` in an Anthropic Message Batches request, or as
        ``body`` in an OpenAI batch JSONL line). Providers that do not
        implement batch mode should leave the default ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{self.adapter_name} adapter does not implement build_batch_payload"
        )

    def parse_batch_response(
        self,
        raw_response: Any,
        claim: Claim,
    ) -> ModelVerdict:
        """
        Parse a provider-specific batch response envelope into a ``ModelVerdict``.

        ``raw_response`` is the result payload as returned by the provider batch
        API for a single request (shape varies per provider).
        """
        raise NotImplementedError(
            f"{self.adapter_name} adapter does not implement parse_batch_response"
        )

    # ── Multi-claim batching (optional, gated on max_claims_per_request > 1) ──

    def build_multi_batch_payload(
        self,
        claims: list[Claim],
        evidence_by_claim: dict[str, list[Evidence]],
        *,
        inject_evidence: bool = True,
        max_evidence_per_claim: int = 5,
    ) -> dict:
        """
        Return the provider-specific request body for a multi-claim submission.

        Implementations should keep the system prompt byte-identical to the
        single-claim version so that provider-side prompt caches continue to
        hit, and scale ``max_tokens`` / tool-call budgets proportionally to
        ``len(claims)``.
        """
        raise NotImplementedError(
            f"{self.adapter_name} adapter does not implement build_multi_batch_payload"
        )

    def parse_multi_batch_response(
        self,
        raw_response: Any,
        claims: list[Claim],
        *,
        batch_call_id: str = "",
    ) -> list[ModelVerdict]:
        """
        Parse a multi-claim response envelope into one ``ModelVerdict`` per claim.

        Missing claims (partial responses) should be filled with
        ``UNVERIFIABLE no_response=True``; usage should be attributed to the
        first verdict only (``batch_call_index=0``).
        """
        raise NotImplementedError(
            f"{self.adapter_name} adapter does not implement parse_multi_batch_response"
        )

    # ── Live multi-claim call (Phase E — non-batch-API multi-claim) ──────────

    def call_multi(
        self,
        claims: list[Claim],
        evidence_by_claim: dict[str, list[Evidence]],
        *,
        inject_evidence: bool = True,
        max_evidence_per_claim: int = 5,
        telemetry_tier: str = "frontier",
        run_id: Optional[str] = None,
    ) -> list[ModelVerdict]:
        """
        Call the provider with N claims in one request and return N verdicts.

        Default implementation loops ``self.call`` per claim — preserves
        byte-identical behavior for adapters that don't override this method
        (Anthropic, OpenAI today). Concrete overrides on Grok/Gemini use
        ``build_multi_user_message`` + ``parse_multi_claim_json`` +
        ``build_multi_verdicts`` to fold the SYNTHESIS_SYSTEM rubric across N
        claims in one API call, attributing the full call usage to the
        index-0 verdict so ``costs.estimate_cost`` bills once per call.

        ``max_evidence_per_claim`` is a hint to overriding implementations;
        the default ignores it (single-claim ``build_user_message`` caps
        evidence to 5 internally).
        """
        return [
            self.call(
                claim,
                evidence_by_claim.get(claim.id, []),
                inject_evidence=inject_evidence,
                telemetry_tier=telemetry_tier,
                run_id=run_id,
            )
            for claim in claims
        ]
