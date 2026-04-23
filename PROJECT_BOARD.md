# Project board

Lightweight kanban for scoped work. **Manual:** when a plan starts, move the row **Backlog → WIP**. When its PR merges, move **WIP → Done** and paste the merged PR link.

Plan files usually live under `.cursor/plans/*.plan.md` (Cursor); links below use that path.

## Backlog

| Item | Plan / pointer | Notes |
|------|------------------|-------|
| **truthbot costs pull** reconciler | — | Pull vendor-side usage/cost via **admin APIs** (not browser scraping): **Anthropic** Admin Usage & Cost API; **OpenAI** `/v1/organization/usage` and `/v1/organization/costs` (org admin key); **Google AI Studio** has no billing API equivalent — **Vertex** spend is reconciled via **Cloud Billing export → BigQuery**. **xAI:** no public billing API as of 2026-04-22; skip for v1 or treat console as manual (no Computer Use / credentialed dashboard automation). Write rollups under `metrics/vendor_costs/<provider>-<yyyy-mm>.json`. Join to `metrics/adapter_calls.jsonl` by `run_id`, `adapter_name`, day; emit `metrics/reports/reconcile-<yyyy-mm>.md` (estimated vs billed per adapter). |
| **Bluesky v2 publisher** | [TODO.md](TODO.md) Phase 7 | `post_report` + remove strict xfail on `tests/test_bluesky.py` unconfigured case. |
| **ExpressionPi deploy** | [STATUS.md](STATUS.md) Next Steps #1 | `TRUTHBOT_SITE_ROOT`, nginx `/truthbot`, rsync/cron. |

## WIP

| Item | Plan | PR |
|------|------|-----|
| *(none)* | | |

## Done

| Item | Plan | PR |
|------|------|-----|
| DataHoover evidence provider + cost optimizations + 2026-04-22 cost-table refresh | [.cursor/plans/datahoover-hook-plus-costs_02ead614.plan.md](.cursor/plans/datahoover-hook-plus-costs_02ead614.plan.md) · [.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md](.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md) | [#3](https://github.com/aRealGem/Truth-bot/pull/3) |
| Project board (kanban) | [.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md](.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md) | [#2](https://github.com/aRealGem/Truth-bot/pull/2) |
| Historical SOTU transcripts corpus (Nixon–Trump) | — | [#1](https://github.com/aRealGem/Truth-bot/pull/1) |
