"""
Base class and shared utilities for LLM adapters.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from truthbot.models import Claim, Evidence, ModelVerdict

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
    """
    if not inject_evidence:
        return (
            f"Claim: {claim.text}\n\n"
            f"Speaker: {claim.speaker}\n\n"
            "Use your web search tool to research this claim. "
            "No pre-gathered evidence was supplied."
        )

    evidence_text = "\n\n".join(
        f"[{i+1}] {e.source_name} ({e.source_tier.value})\n{e.snippet}"
        for i, e in enumerate(evidence[:10])
    )
    if evidence_text:
        return (
            f"Claim: {claim.text}\n\n"
            f"Speaker: {claim.speaker}\n\n"
            f"{EVIDENCE_CAVEAT}"
            f"Evidence:\n{evidence_text}"
        )
    return (
        f"Claim: {claim.text}\n\n"
        f"Speaker: {claim.speaker}\n\n"
        "No pre-gathered evidence available. Please use web search to research this claim."
    )


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
