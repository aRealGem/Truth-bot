"""
Verification-time context helpers shared by all adapters.

Everything in this package produces strings that get injected into user
messages (NOT the system prompt) so provider-side prompt caches on the
SYNTHESIS_SYSTEM prefix continue to hit, plus post-hoc validators that
decorate model verdicts with quality flags without rewriting them.
"""

from truthbot.verify.context.temporal import build_temporal_preamble
from truthbot.verify.context.validator import (
    TemporalFinding,
    apply_temporal_flags,
    scan_text,
)

__all__ = [
    "TemporalFinding",
    "apply_temporal_flags",
    "build_temporal_preamble",
    "scan_text",
]
