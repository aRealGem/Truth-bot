# Truth-bot Status — 2026-04-22 evening

## Session note — 2026-04-22 evening (batch-API push)

### Shipped tonight
- **Real batch API implementation.** ``truthbot publish --mode batch`` now submits per-claim requests to each provider's native batch API instead of falling through to live calls:
  - Anthropic Message Batches (``client.messages.batches.create/retrieve/results``) with ``web_search_20250305`` tool + ephemeral prompt caching.
  - OpenAI Batch API against ``/v1/responses`` with ``web_search_preview`` (writes JSONL, uploads via ``/files``, creates batch, downloads output file at reconcile time).
  - Gemini has ``build_batch_payload``/``parse_batch_response`` wired but ``supports_batch=False`` pending a pinned google-genai with verified ``batches.create`` + GoogleSearch support; it falls out of batch-mode consensus until then.
  - xAI (Grok): no public batch API — runs **live during submit** as a sidecar, verdicts spooled to ``metrics/batch_sidecar/<run_id>.jsonl`` and merged at reconcile.
- **Two-phase CLI.** ``truthbot batch poll <run_id>`` returns a short status string; the new ``truthbot batch reconcile <run_id>`` polls → fetches results → parses → merges sidecar → builds consensus → caches ``VerdictBundle``s → publishes the site. Pipeline ``_run_publish`` split cleanly so ``--mode live`` is byte-for-byte unchanged.
- **Cost-reporting honesty.** ``costs.estimate_cost`` now gates the 50% batch discount on a real ``batch_job_id`` (not just the ``mode`` string). ``telemetry.measure`` threads the provider batch ID through. Scaffolding/misconfigured calls are billed full price. Existing `test_estimate_cost_batch_multiplier` updated to cover both gates.
- **Total-claim preservation.** ``--max-claims 0`` (the new default) means "verify every checkable claim". Extractor safety cap raised to 500; transcript budget raised to 200 K chars (covered a full SOTU instead of truncating to ~25% of one). Pipeline always prints ``N claims extracted total, M checkable, K selected``.
- **Tests.** 263 passed / 1 xfailed (up from 257 / 1). New ``tests/test_batch_mode.py`` adds mocked Anthropic submit→poll→reconcile round-trip, pending-path early-return, real-``batch_job_id`` telemetry assertion, sidecar JSONL round-trip, and error-row handling.
- **Backlog updated.** Added entries for "Multi-claim batching", "Gemini batch transport", and "xAI Grok batch" in ``PROJECT_BOARD.md``.

### Pre-submission blocker — **API keys in `.env` are truncated**
Three keys got eaten by a terminal line-wrap on paste (all end at column 80 with a stray ``>`` redirection char). Until these are re-pasted, **no live or batch API call can succeed** — any SOTU run right now would burn only the preamble request and 401 before anything useful happens.

Offending keys (check lengths, not values):
- ``ANTHROPIC_API_KEY`` — currently ~55 char body, real keys are ~108.
- ``OPENAI_API_KEY`` — currently ~55 char body, real ``sk-proj-*`` keys are ~156.
- ``XAI_API_KEY`` — currently ~59 char body, real ``xai-*`` keys are ~84.
- ``GEMINI_API_KEY`` looks intact (50 chars, no trailing ``>``).

### Morning runbook — SOTU 2026-02-24 overnight batch

Before running anything:

1. **Re-paste** the three keys above into ``.env`` (one line each, no line breaks, no trailing ``>``). Terminal paste works best if you widen the window first or use an editor.
2. **Verify** with a cheap ping per provider (these each cost well under a cent):
   ```bash
   set -a && . ./.env && set +a
   .venv/bin/python -c "import anthropic; anthropic.Anthropic().messages.create(model='claude-haiku-4-5', max_tokens=8, messages=[{'role':'user','content':'ping'}]); print('anthropic ok')"
   .venv/bin/python -c "import openai; openai.OpenAI().responses.create(model='gpt-5.4-nano', input='ping', max_output_tokens=16); print('openai ok')"
   .venv/bin/python -c "import openai; openai.OpenAI(api_key=__import__('os').environ['XAI_API_KEY'], base_url='https://api.x.ai/v1').chat.completions.create(model='grok-4', messages=[{'role':'user','content':'ping'}], max_tokens=8); print('xai ok')"
   ```
3. **Submit** the SOTU run (all 29 claims, no cap, triage on):
   ```bash
   set -a && . ./.env && set +a
   .venv/bin/truthbot publish \
     --transcript eval/sotu-2026/transcript.txt \
     --speaker "Donald Trump" --role "President" \
     --date 2026-02-24 --venue "State of the Union" \
     --site-root site-test --mode batch --triage --max-claims 0
   ```
   The command prints the ``run_id`` + both poll commands and exits immediately (no waiting).
4. **Reconcile & publish** after batches complete (Anthropic batches usually finish in 5–60 min, OpenAI up to 24 h):
   ```bash
   .venv/bin/truthbot batch poll <run_id>         # quick status check
   .venv/bin/truthbot batch reconcile <run_id>    # merge + site regen
   .venv/bin/truthbot metrics summary --run-id <run_id>
   ```
   Append totals (claims extracted, cost, duration) to this file.

## Current State

## Current State (pre-batch push, retained for history)
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
