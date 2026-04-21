# TODO

## Migrate test_verify.py / test_pipeline.py / bluesky stub from legacy Verdict schema to ConsensusVerdict

**Affected files:**
- `tests/test_verify.py`
- `tests/test_pipeline.py`
- `src/truthbot/publish/bluesky.py` (stub references legacy `Verdict` / `.label` / `.consensus_label`)

**Schema drift:** Tests were written against the old `Verdict` model with `.label` and `.consensus_label` string attributes; the current schema uses `ConsensusVerdict` with a `.consensus_label` `VerdictLabel` enum and `ModelVerdict` objects, causing 9 test failures.

**Do not fix until:** schema is stable and adapters are fully integrated.
