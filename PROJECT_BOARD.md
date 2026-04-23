# Project board

Lightweight kanban for scoped work. **Manual:** when a plan starts, move the row **Backlog → WIP**. When its PR merges, move **WIP → Done** and paste the merged PR link.

Plan files usually live under `.cursor/plans/*.plan.md` (Cursor); links below use that path.

## Backlog

| Item | Plan / pointer | Notes |
|------|------------------|-------|
| **truthbot costs pull** reconciler | — | Pull vendor-side usage/cost via **admin APIs** (not browser scraping): **Anthropic** Admin Usage & Cost API; **OpenAI** `/v1/organization/usage` and `/v1/organization/costs` (org admin key); **Google AI Studio** has no billing API equivalent — **Vertex** spend is reconciled via **Cloud Billing export → BigQuery**. **xAI:** no public billing API as of 2026-04-22; skip for v1 or treat console as manual (no Computer Use / credentialed dashboard automation). Write rollups under `metrics/vendor_costs/<provider>-<yyyy-mm>.json`. Join to `metrics/adapter_calls.jsonl` by `run_id`, `adapter_name`, day; emit `metrics/reports/reconcile-<yyyy-mm>.md` (estimated vs billed per adapter). |
| **Bluesky v2 publisher** | [TODO.md](TODO.md) Phase 7 | `post_report` + remove strict xfail on `tests/test_bluesky.py` unconfigured case. |
| **ExpressionPi deploy** | [STATUS.md](STATUS.md) Next Steps #1 | `TRUTHBOT_SITE_ROOT`, nginx `/truthbot`, rsync/cron. |
| **Absolute URLs for social + feed (`[SITE_URL]` substitution)** | — | Follow-up to [#4](https://github.com/aRealGem/Truth-bot/pull/4). At publish time (probably wired into ExpressionPi deploy) rewrite the relative `og:image` / `twitter:image` URLs in every page and resolve the `[SITE_URL]` placeholder in `site-test/feed.xml` to the real production origin. Without this, Twitter/Facebook/etc. scrapers resolve the image path against whatever origin they hit (raw.githack, Pages) instead of the canonical site, and the Atom `<link href>`s stay broken. Options: (a) post-publish rewrite pass in `SitePublisher` gated on a `TRUTHBOT_BASE_URL` env var, (b) emit absolute URLs directly from the generator when `TRUTHBOT_BASE_URL` is set, leaving relative as the local/dev fallback. Also consider adding `<link rel="canonical">` while we're in there. |

## WIP

| Item | Plan | PR |
|------|------|-----|
| *(none)* | | |

## Done

| Item | Plan | PR |
|------|------|-----|
| Social sharing infra, favicon, Atom feed, prompt-hash footer, src-tiers chip | — | [#4](https://github.com/aRealGem/Truth-bot/pull/4) |
| DataHoover evidence provider + cost optimizations + 2026-04-22 cost-table refresh | [.cursor/plans/datahoover-hook-plus-costs_02ead614.plan.md](.cursor/plans/datahoover-hook-plus-costs_02ead614.plan.md) · [.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md](.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md) | [#3](https://github.com/aRealGem/Truth-bot/pull/3) |
| Project board (kanban) | [.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md](.cursor/plans/cost-table-refresh-and-board_be12e335.plan.md) | [#2](https://github.com/aRealGem/Truth-bot/pull/2) |
| Historical SOTU transcripts corpus (Nixon–Trump) | — | [#1](https://github.com/aRealGem/Truth-bot/pull/1) |
