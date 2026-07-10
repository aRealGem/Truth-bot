#!/usr/bin/env python3
"""
Closed-book PCA cost model (P96.2.1 follow-up).

Turns the live-validated dev-lot economics into a parameterized projection for the
production run, replacing the old "~$120 / 10 speeches" hypothesis with a real
number. Calibrated so the dev roster reproduces the measured dev-lot cost
(~$0.010 / 25 claims, escalation 0.32) to within a few percent — see
tests/benchmarks/test_cost_model.py.

Model (per claim):  proposer + Σ critics + escalation_rate x arbiter
Each call's cost = tokens_in x rate_in + tokens_out x rate_out.

SCOPE: closed-book Layer B (short, ~0-cache prompts). Layer C (evidence-grounded)
will push input tokens up substantially — widen TOKENS before trusting a Layer C
projection. All non-dev rates are list prices (fetched 2026-07-09); confirm at
provisioning, especially the seats a prod roster actually uses.
"""
from __future__ import annotations

# ── rates: USD per 1M tokens, (input, output) ─────────────────────────────────
# DeepInfra list prices verified 2026-07-09 (deepinfra.com). Others fetched same
# day from the vendor pricing pages; treat as list, confirm at provisioning.
RATES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # DeepInfra
    "dsv4-flash":       (0.09, 0.18),
    "deepseek-v4-pro":  (1.30, 2.60),
    "deepseek-v3":      (0.32, 0.89),
    "deepseek-r1":      (0.50, 2.15),
    "mistral":          (0.075, 0.20),
    # Anthropic — claude-haiku (1.00/5.00) reproduces the measured proxy arbiter
    # cost. Opus via the Max/Lane-Worker subscription is effectively $0 for the
    # arbiter seat (its API list price is not what prod pays).
    "claude-haiku":     (1.00, 5.00),
    "claude-opus-sub":  (0.0, 0.0),     # subscription-covered (Lane-Worker)
    "claude-opus-api":  (15.0, 75.0),   # API list, for comparison only
    # xAI — latest grok-4.5. NB proxy alias `grok` currently -> grok-2-latest (stale).
    "grok-4.5":         (2.00, 6.00),
    # Google — gemini 3.1 pro / 3.5 flash. NB proxy `gemini-pro`->1.5-pro (stale).
    "gemini-pro":       (2.00, 12.00),
    "gemini-flash":     (1.50, 9.00),
    # OpenAI — GPT-5.x. NB proxy has gpt-4o (superseded).
    "gpt-5.4":          (2.50, 15.00),
    "gpt-5.4-mini":     (0.75, 4.50),
    "gpt-5.6-luna":     (1.00, 6.00),
}

# ── per-call token profile, (in, out), calibrated from the closed-book dev-lot ──
TOKENS: dict[str, tuple[int, int]] = {
    "proposer": (200, 90),
    "critic":   (200, 90),
    "arbiter":  (320, 150),   # arbiter also reads the proposer + critic verdicts
}

# ── rosters: seat -> alias(es) ────────────────────────────────────────────────
DEV = {"proposer": "mistral", "critics": ["dsv4-flash"], "arbiter": "claude-haiku"}
# Illustrative prod shapes (prod roster P/A seats are still TBD in rosters.yaml).
# P96 note: cost concentrates in grok (every-claim critic) + gpt; Opus is subsidised.
PROD_OPUS_ARBITER = {"proposer": "gpt-5.4-mini", "critics": ["grok-4.5", "dsv4-flash"],
                     "arbiter": "claude-opus-sub"}
PROD_GPT_ARBITER = {"proposer": "gpt-5.4-mini", "critics": ["grok-4.5", "dsv4-flash"],
                    "arbiter": "gpt-5.4"}

MEASURED_LABEL_MISMATCH = 0.32   # dev-lot escalation under the label_mismatch criterion
MEASURED_MATERIAL_DISAG = 0.80   # prior gate (label OR |Δconf|) — for sensitivity


def call_cost(model: str, role: str) -> float:
    ti, to = TOKENS[role]
    ri, ro = RATES_USD_PER_MTOK[model]
    return (ti * ri + to * ro) / 1_000_000.0


def per_claim(roster: dict, escalation: float) -> float:
    c = call_cost(roster["proposer"], "proposer")
    c += sum(call_cost(m, "critic") for m in roster["critics"])
    c += escalation * call_cost(roster["arbiter"], "arbiter")
    return c


def project(roster: dict, escalation: float,
            claims_per_speech: int = 100, speeches: int = 10) -> dict:
    pc = per_claim(roster, escalation)
    return {"per_claim": pc, "per_speech": pc * claims_per_speech,
            "total": pc * claims_per_speech * speeches}


def _fmt(name: str, roster: dict, esc: float) -> str:
    p = project(roster, esc)
    return (f"  {name:22} esc={esc:.2f}  ${p['per_claim']:.5f}/claim  "
            f"${p['per_speech']:.3f}/100-claim-speech  ${p['total']:.2f}/10-speeches")


if __name__ == "__main__":
    print("# Closed-book PCA cost projections (list prices 2026-07-09; confirm at provisioning)")
    print(_fmt("DEV (validated)", DEV, MEASURED_LABEL_MISMATCH))
    print(_fmt("PROD opus-arbiter", PROD_OPUS_ARBITER, MEASURED_LABEL_MISMATCH))
    print(_fmt("PROD gpt-arbiter", PROD_GPT_ARBITER, MEASURED_LABEL_MISMATCH))
    print("  -- escalation sensitivity (old 0.80 gate) --")
    print(_fmt("PROD gpt-arbiter", PROD_GPT_ARBITER, MEASURED_MATERIAL_DISAG))
    print("  vs the old $120/10-speeches hypothesis — even the priciest shape is far under it.")
