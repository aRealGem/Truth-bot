#!/usr/bin/env python3
"""Single-claim live OpenAI probe (interpretability; optional API cost).

Requires OPENAI_API_KEY. When TRUTHBOT_OPENAI_RESPONSES_PROBE is truthy, emits a
structured WARNING from OpenAIAdapter (see
metrics/adapter_interpretability/openai_responses_probe.md).

Usage:
  TRUTHBOT_OPENAI_RESPONSES_PROBE=1 uv run python scripts/openai_responses_probe.py "claim text"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from truthbot.models import Claim, Evidence
from truthbot.verify.adapters.base import AdapterUnavailable
from truthbot.verify.adapters.openai import OpenAIAdapter


def main() -> int:
    p = argparse.ArgumentParser(description="One OpenAI live Responses + web_search probe.")
    p.add_argument("claim_text", help="Atomic claim sentence to verify")
    p.add_argument("--speaker", default="Probe")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    try:
        adapter = OpenAIAdapter()
    except AdapterUnavailable as exc:
        print(f"OpenAI adapter unavailable: {exc}", file=sys.stderr)
        return 2

    claim = Claim(
        transcript_id="probe",
        text=args.claim_text.strip(),
        speaker=args.speaker,
    )
    evidence: list[Evidence] = []

    verdict = adapter.call(claim, evidence, inject_evidence=True, telemetry_tier="frontier")
    payload = verdict.model_dump(mode="json")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
