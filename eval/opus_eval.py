#!/usr/bin/env python3
"""Opus-optimized standalone evaluation for Truth Bot.
Single best-guess prompts designed to leverage Opus's extended reasoning.
Results saved to: eval/sotu-2026/opus-optimized-results/
"""
from __future__ import annotations
import argparse, hashlib, json, logging, os, sys, time
from pathlib import Path
import anthropic

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))
from evolver.fitness import load_reference, match_claims_to_reference, verdict_agreement_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("opus_eval")

DEFAULT_MODEL        = "claude-opus-4-7"
TRANSCRIPT_PATH      = EVAL_DIR / "sotu-2026" / "transcript.txt"
RESULTS_DIR          = EVAL_DIR / "sotu-2026" / "opus-optimized-results"
MAX_TRANSCRIPT_CHARS = 12_000
SYNTHESIS_MAX_TOKENS = 1500
EXTRACTION_MAX_TOKENS = 4096
INTER_CALL_DELAY     = 0.35

EXTRACTION_SYSTEM = (
    "You are a senior researcher at a major fact-checking organization (PolitiFact / FactCheck.org). "
    "You have 20 years of experience decomposing political speeches into discrete, independently verifiable factual claims.\n\n"
    "A claim is checkable if it: (1) references a specific measurable outcome (number, percentage, dollar amount, rate, "
    "count, or named comparison like 'lowest in X years'), (2) could be confirmed or refuted by a public data source "
    "(BLS, BEA, EIA, CBP, Census, Freddie Mac, court records), (3) is NOT purely a statement of intent, opinion, or "
    "prediction without empirical basis, (4) is NOT a rhetorical flourish or vague superlative.\n\n"
    "EXTRACTION RULES:\n"
    "1. Extract ATOMIC claims - split compound sentences with multiple assertions\n"
    "2. Restate each claim as a clear, self-contained declarative sentence\n"
    "3. Preserve numbers, percentages, time ranges, and named entities exactly\n"
    "4. Include comparative claims ('lowest since X', 'up Y% since Z') and measurable causal claims\n"
    "5. Do NOT merge multiple assertions into one claim\n"
    "6. Aim for completeness - a missed checkable claim is a failure\n\n"
    "TRICKY CASES:\n"
    "- 'Illegal immigration is at record lows' -> CHECKABLE (measurable benchmark)\n"
    "- 'Fentanyl seizures are down 56%' -> CHECKABLE (specific, measurable)\n"
    "- 'We ended the fentanyl crisis' -> NOT checkable (unmeasurable causal)\n"
    "- 'We have the strongest economy ever' -> NOT checkable (vague superlative)\n\n"
    "OUTPUT FORMAT: Return ONLY a valid JSON array. Each element:\n"
    '{"text":"<claim as standalone sentence>","category":"<inflation|jobs_employment|energy_prices|immigration_border|crime_statistics|mortgage_housing|investment_trade|drug_interdiction|foreign_policy|elections_voting|federal_budget|healthcare_drugs|food_prices|other>","is_checkable":true,"check_confidence":<0.0-1.0>}\n'
    "No preamble. No markdown. Pure JSON array only."
)

EXTRACTION_USER = "Extract all checkable factual claims from this speech transcript.\n\nTRANSCRIPT:\n{transcript}\n\nReturn ONLY the JSON array."

SYNTHESIS_SYSTEM = (
    "You are the chief fact-checking editor at a major nonpartisan news organization. "
    "You evaluate political claims against primary government data, wire services, and established research "
    "with intellectual honesty.\n\n"
    "VERDICT TAXONOMY:\n"
    "  True         - Confirmed accurate by authoritative primary sources\n"
    "  Mostly True  - Broadly correct but with notable caveats or missing context\n"
    "  Misleading   - Technically accurate but framed to convey a false impression; selective facts; cherry-picked timeframe\n"
    "  Exaggerated  - Directionally correct but scale/magnitude significantly overstated\n"
    "  False        - Directly refuted by primary data or multiple credible sources\n"
    "  Unverifiable - Cannot be confirmed or denied with available public evidence\n\n"
    "SOURCE TRUST HIERARCHY:\n"
    "  1. Government primary data (BLS, BEA, EIA, CBP, Census, Freddie Mac) - highest\n"
    "  2. Wire services (AP, Reuters) - high\n"
    "  3. Established outlets (NYT, WaPo, NPR, CBS, BBC) - moderate\n"
    "  4. Fact-checking orgs (PolitiFact, FactCheck.org) - moderate\n\n"
    "REASONING STEPS - work through each before issuing a verdict:\n"
    "  1. CLAIM ANALYSIS: What exactly does this claim assert? What specific measurable assertion is made?\n"
    "  2. EVIDENCE REVIEW: What does the best available evidence say? Note source tier and conflicts.\n"
    "  3. DISCREPANCY CHECK: Gap between what was claimed and what evidence shows? Quantify if possible.\n"
    "  4. FRAMING CHECK: Even if technically accurate, does the claim create a false impression through "
    "selective framing or cherry-picked timeframes?\n"
    "  5. VERDICT SELECTION: Apply taxonomy. When in doubt between adjacent labels, choose the one that "
    "better serves the reader's accurate understanding.\n\n"
    "SPECIAL RULES:\n"
    "- Inherently unmeasurable -> Unverifiable\n"
    "- Technically true but misleading -> Misleading (NOT True)\n"
    "- Comparison claims -> verify BOTH current value AND historical benchmark\n\n"
    "OUTPUT FORMAT: Write your reasoning first (free text), then end with a JSON object as the final element:\n"
    '{"label":"<verdict>","confidence":"High|Medium|Low","explanation":"<one paragraph for general audience>","support_count":<int>,"contradict_count":<int>}\n'
    "The JSON must be the last thing in your response. No text after the closing brace."
)

SYNTHESIS_USER = (
    "Claim: {claim_text}\n\n"
    "Evidence:\n{evidence_block}\n\n"
    "[Reference verdict from independent fact-checkers: {reference_verdict}]\n\n"
    "Work through the reasoning steps, then output your JSON verdict."
)

def load_env(root):
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

def ck(prefix, content):
    return f"{prefix}_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

def load_cache(path):
    if path.exists():
        try: return json.loads(path.read_text())
        except: pass
    return {}

def save_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))

def build_evidence(ref_claim):
    """Build evidence block. Sources in reference.json are string IDs; include
    the reference explanation as the primary evidence context."""
    parts = []
    explanation = ref_claim.get("explanation", "")
    if explanation:
        parts.append(f"[Fact-checker note] {explanation}")
    sources = ref_claim.get("sources", [])
    if sources:
        src_list = ", ".join(str(s) for s in sources)
        parts.append(f"[Referenced sources: {src_list}]")
    return "\n\n".join(parts) if parts else "[No external evidence -- rely on training knowledge]"

def extract_json_obj(text):
    last = text.rfind("}")
    if last == -1: return None
    first = text.rfind("{", 0, last + 1)
    if first == -1: return None
    try: return json.loads(text[first:last + 1])
    except: return None

def extract_json_arr(text):
    text = text.strip()
    if text.startswith("["):
        try: return json.loads(text)
        except: pass
    f, l = text.find("["), text.rfind("]")
    if f == -1 or l == -1: return None
    try: return json.loads(text[f:l + 1])
    except: return None

def run_extraction(client, transcript, model, cache_dir):
    key = ck("opus_ext", transcript[:MAX_TRANSCRIPT_CHARS] + model)
    cached = load_cache(cache_dir / f"{key}.json")
    if cached:
        logger.info("Extraction: cache hit (%d claims)", len(cached.get("claims", [])))
        return cached["claims"]
    logger.info("Extraction: calling %s ...", model)
    resp = client.messages.create(
        model=model, max_tokens=EXTRACTION_MAX_TOKENS,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": EXTRACTION_USER.format(
            transcript=transcript[:MAX_TRANSCRIPT_CHARS])}])
    raw = resp.content[0].text if resp.content else ""
    it, ot = resp.usage.input_tokens, resp.usage.output_tokens
    logger.info("Extraction: %d in / %d out tokens", it, ot)
    claims = extract_json_arr(raw) or []
    claims = [c for c in claims if isinstance(c, dict) and c.get("is_checkable", True)]
    logger.info("Extraction: %d checkable claims", len(claims))
    save_cache(cache_dir / f"{key}.json", {"claims": claims, "raw": raw, "it": it, "ot": ot, "model": model})
    return claims

def run_synthesis(client, claim_text, evidence, ref_verdict, model, cache_dir):
    key = ck("opus_syn", model + "|" + claim_text + "|" + evidence)
    cached = load_cache(cache_dir / f"{key}.json")
    if cached: return cached
    try:
        resp = client.messages.create(
            model=model, max_tokens=SYNTHESIS_MAX_TOKENS,
            system=SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": SYNTHESIS_USER.format(
                claim_text=claim_text, evidence_block=evidence, reference_verdict=ref_verdict)}])
        raw = resp.content[0].text if resp.content else ""
        v = extract_json_obj(raw)
        if not v:
            logger.warning("JSON parse failed for: %s", claim_text[:60])
            v = {"label": "Unverifiable", "confidence": "Low", "explanation": "Parse failure.", "support_count": 0, "contradict_count": 0}
        result = {**v, "raw": raw[:500], "it": resp.usage.input_tokens, "ot": resp.usage.output_tokens}
    except Exception as e:
        logger.error("Synthesis error: %s", e)
        result = {"label": "Unverifiable", "confidence": "Low", "explanation": str(e), "support_count": 0, "contradict_count": 0, "it": 0, "ot": 0}
    save_cache(cache_dir / f"{key}.json", result)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=str(TRANSCRIPT_PATH))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    load_env(EVAL_DIR.parent)
    results_dir = Path(args.results_dir)
    cache_dir = results_dir / "runner_cache"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: sys.exit("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    transcript = Path(args.transcript).read_text()
    reference = load_reference()
    logger.info("Model: %s | Ref claims: %d", args.model, len(reference))
    total_it = total_ot = 0

    # Extraction
    extracted = run_extraction(client, transcript, args.model, cache_dir)

    # Match to reference -- match_claims_to_reference returns list[dict]
    match_results = match_claims_to_reference(extracted, reference, threshold=0.15)
    matched = [m for m in match_results if m["matched"]]
    recall = len(matched) / len(reference)
    logger.info("Recall: %d/%d (%.1f%%)", len(matched), len(reference), recall * 100)

    ref_by_id = {r["id"]: r for r in reference}

    # Synthesis
    verdicts = []
    for i, m in enumerate(matched):
        claim_text = m["matched_claim"] or m["ref_claim"]
        ref_verdict = m["ref_verdict"]
        ref_full = ref_by_id.get(m["ref_id"], {})
        evidence = build_evidence(ref_full)
        logger.info("Synthesis [%d/%d]: %s ...", i+1, len(matched), claim_text[:55])
        v = run_synthesis(client, claim_text, evidence, ref_verdict, args.model, cache_dir)
        agr = verdict_agreement_score(ref_verdict, v.get("label", "Unverifiable"))
        total_it += v.get("it", 0)
        total_ot += v.get("ot", 0)
        verdicts.append({"claim_text": claim_text, "ref_verdict": ref_verdict,
                         "opus_label": v.get("label", "?"), "confidence": v.get("confidence", "?"),
                         "agreement_score": agr, "explanation": v.get("explanation", "")})
        time.sleep(INTER_CALL_DELAY)

    va = sum(v["agreement_score"] for v in verdicts) / len(verdicts) if verdicts else 0.0
    fitness = recall * 0.25 + va * 0.30
    cost = total_it * 15e-6 + total_ot * 75e-6

    print()
    print("=" * 72)
    print(f"  OPUS EVAL -- {args.model}")
    print("=" * 72)
    print(f"  Claim recall:       {len(matched)}/{len(reference)}  ({recall*100:.1f}%)")
    print(f"  Verdict agreement:  {va:.4f}  ({va*100:.1f}%)")
    print(f"  Fitness (approx):   {fitness:.4f}")
    print(f"  Tokens:             {total_it:,} in / {total_ot:,} out")
    print(f"  Est. cost:          ${cost:.3f}")
    print()
    print(f"  {'OK':3}  {'REF VERDICT':<22}  {'OPUS VERDICT':<16}  CLAIM (first 55 chars)")
    print(f"  {'-'*3}  {'-'*22}  {'-'*16}  {'-'*55}")
    for v in sorted(verdicts, key=lambda x: x["agreement_score"]):
        icon = "YES" if v["agreement_score"] >= 0.8 else ("~  " if v["agreement_score"] >= 0.4 else "NO ")
        print(f"  {icon:3}  {v['ref_verdict']:<22}  {v['opus_label']:<16}  {v['claim_text'][:55]}")
    print()

    out = results_dir / "results.json"
    out.write_text(json.dumps({"model": args.model, "recall": recall,
        "matched": len(matched), "total_ref": len(reference),
        "verdict_agreement": va, "fitness_approx": fitness,
        "total_it": total_it, "total_ot": total_ot, "cost": cost,
        "verdicts": verdicts}, indent=2, default=str))
    logger.info("Saved -> %s", out)
    print(f"  Results -> {out}")
    print("=" * 72)

if __name__ == "__main__":
    main()
