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
| Cost-table refresh + project board | [.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md](.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md) | *(add PR URL when opened)* |

## Done

| Item | Plan | PR |
|------|------|-----|
| DataHoover hook + cost optimizations | [.cursor/plans/datahoover-hook-plus-costs_02ead614.plan.md](.cursor/plans/datahoover-hook-plus-costs_02ead614.plan.md) | *(add merged PR URL when merged)* |
