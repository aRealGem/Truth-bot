# Truth-bot Status — 2026-04-21 15:56 EDT

## Current State
- **Pipeline:** Phases 1–5 are complete. Multi-adapter verification (Anthropic + OpenAI fallback to gpt-4.1), telemetry logging (`metrics/adapter_calls.jsonl`), and the static site publisher are all working end-to-end via `truthbot publish`.
- **Latest publish:** `site-test/` contains the 2026-03-04 Trump SOTU run (5 claims, 3 False / 2 True-ish verdicts). Material is ready to rsync to ExpressionPi once a live document root is configured.
- **Telemetry:** Anthropic calls average ~19s / $0.65 each; OpenAI adapter currently lands on gpt-4.1 fallback because gpt-5.4 isn’t GA yet. `truthbot metrics summary` reports cleanly from the JSONL log.
- **Working tree:** `src/truthbot/publish/site.py` + the regenerated `site-test` assets contain a WIP UI refresh (provider/model labels inside claim cards). The content renders fine locally, but the file picked up mojibake (“Ã¢ÂÂ” etc.) and needs to be re-encoded before shipping.

## Next Steps
1. **Phase 6 — ExpressionPi deploy:** set `TRUTHBOT_SITE_ROOT` to the nginx docroot on ExpressionPi, rerun `truthbot publish`, and wire nginx to serve `/truthbot` with daily rsync/cron refresh.
2. **Adapter coverage:** add valid OPENAI/GEMINI/XAI keys (or better fallbacks) so that consensus verdicts once again include more than Anthropic+OpenAI fallback, and update telemetry cost tables as models change.
3. **Automated runs:** define a nightly job (cron or systemd timer) that ingests the latest transcript, runs `truthbot publish`, and copies reports to ExpressionPi + Bluesky once Phase 7 lands.
4. **Site refresh cleanup:** fix the encoding on `publish/site.py` (re-save as UTF-8) and commit the provider/model labeling improvements alongside regenerated assets.

## Blockers / Risks
- **Provider keys:** GEMINI_API_KEY and XAI_API_KEY are still unset. OPENAI_API_KEY works but flagship `gpt-5.4` is unavailable, so we’re stuck on gpt-4.1 fallback until GA.
- **ExpressionPi availability:** we need a confirmed docroot + nginx stanza before pushing the static site live.
- **Bluesky publisher:** intentionally stubbed (`NotImplementedError`) until the site deployment is stable; Phase 7 is still queued.

## Session note — 2026-04-21 evening
- Fresh clone of `aRealGem/Truth-bot` landed on `main @ b1c3c7b`; existing `.env` preserved.
- `.venv` created with Python 3.13.12; `pip install -e ".[dev]"` succeeded.
- (Superseded by 2026-04-22 morning) Earlier run was **175 passed, 9 failed** due to test/schema drift; see morning note for resolution.

## Session note — 2026-04-22 morning
- **Tests:** Schema drift resolved in `tests/test_verify.py` and `tests/test_pipeline.py` (`ConsensusVerdict` / `.consensus_label` vs legacy `Verdict` on engine returns; `DummyEngine` returns `ConsensusVerdict` to match `Pipeline.run()`).
- **Bluesky:** `test_post_report_returns_none_when_unconfigured` is **xfail (strict)** until Phase 7 implements `post_report` (see `TODO.md`).
- **Baseline:** `pytest` → **183 passed, 1 xfailed, 0 failed**. No `--deselect` workaround needed.

### Recommendation (today)
1. **Next Steps #1** — Phase 6 ExpressionPi deploy (docroot + nginx + `TRUTHBOT_SITE_ROOT`).
2. **Next Steps #4** — UTF-8 cleanup in `src/truthbot/publish/site.py` (mojibake) before shipping UI refresh.
3. **Phase 7** — implement Bluesky `post_report`, then remove the strict xfail on the unconfigured test.
