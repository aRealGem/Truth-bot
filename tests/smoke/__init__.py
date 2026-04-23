"""Live 2-claim smoke tests against real provider APIs.

**These tests spend real money** and are skipped by default. Run with:

    pytest tests/smoke/ -m live -v

They validate that each provider's full submit → poll → reconcile (batch)
or live call (sidecar) path returns sensible verdicts on two hardcoded
claims with opposite truth values:

  - True  : "The United States landed astronauts on the Moon in 1969."
  - False : "The Eiffel Tower is located in Berlin, Germany."

Every test captures actual token usage and wall-clock time; the aggregate
numbers land in STATUS.md via ``p2-status-note``.
"""
