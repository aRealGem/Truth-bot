"""One-off: telemetry-capture reconcile (no publish) for v-p1-p2 OpenAI batch.

Mirrors `_run_publish_batch_reconcile` in pipeline.py but stops before the
SitePublisher.publish step. Confirms the OpenAI batch verdicts merge into
the sidecar and run_summary, gives us baseline calibration data before we
start mutating adapter code in the anti-hallucination defense-in-depth
phase.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from truthbot.config import settings
from truthbot.metrics.telemetry import finalize_run
from truthbot.verify.adapters.base import AdapterUnavailable
from truthbot.verify.batch import reconcile_run
from truthbot.verify.engine import VerificationEngine

RUN_ID = "ed7be4ad-3f2e-4010-a674-be2f8a17589e"

adapters_by_name = {}
for mod_name, cls_name in (
    ("truthbot.verify.adapters.anthropic", "AnthropicAdapter"),
    ("truthbot.verify.adapters.openai", "OpenAIAdapter"),
    ("truthbot.verify.adapters.gemini", "GeminiAdapter"),
    ("truthbot.verify.adapters.grok", "GrokAdapter"),
):
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        adapters_by_name[cls.adapter_name] = cls()
    except AdapterUnavailable as exc:
        print(f"skipping {cls_name}: {exc}")
    except Exception as exc:
        print(f"failed to build {cls_name}: {exc}")

print(f"adapters loaded: {sorted(adapters_by_name.keys())}")
engine = VerificationEngine(run_id=RUN_ID, verify_mode="batch")
result = reconcile_run(
    settings.metrics_dir,
    RUN_ID,
    adapters_by_name=adapters_by_name,
    engine=engine,
)
status = result["status"]
print(f"status: {status}")

if status == "pending":
    print("PENDING (unexpected — we expected the OpenAI batch to be done):")
    for provider, st in result.get("pending_providers", []):
        print(f"  {provider}: {st}")
    sys.exit(2)

bundles = result.get("bundles", [])
triaged = result.get("triaged_bundles", [])
print(f"bundles (reconciled): {len(bundles)}")
print(f"triaged (cached/live): {len(triaged)}")

try:
    fin = finalize_run(RUN_ID)
    print(f"total_cost_usd: {fin['total_cost_usd']:.6f}")
    print(f"total_input_tokens: {fin.get('total_input_tokens', '?')}")
    print(f"total_output_tokens: {fin.get('total_output_tokens', '?')}")
    print(f"total_calls: {fin.get('total_calls', '?')}")
except Exception as exc:
    print(f"finalize_run failed: {exc}")

# Inspect the reconciled OpenAI verdicts to validate Phase 2b telemetry
# and look for fabrication signals + temporal flags.
print()
print("=" * 60)
print("Reconciled bundle inspection (post-OpenAI batch)")
print("=" * 60)
all_b = list(triaged) + list(bundles)
print(f"total bundles: {len(all_b)}")
for b in all_b:
    print()
    print(f"claim {b.claim.id[:8]}: {b.claim.text[:90]}")
    for mv in b.model_verdicts:
        flags = mv.temporal_flags or []
        print(
            f"  {mv.adapter_name:10s} {mv.model_id:25s} "
            f"label={mv.label:14s} conf={mv.confidence:6s} "
            f"tools={mv.tool_call_count} "
            f"src={len(mv.web_sources)} "
            f"flags={flags}"
        )
        for u in mv.web_sources[:3]:
            print(f"      - {u[:110]}")
