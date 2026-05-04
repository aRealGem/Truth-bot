# Claude orientation — Truth-bot

This file is read first by every Claude Code session. Keep it short and evergreen — point at canonical docs rather than duplicating them.

## What this project is

Truth-bot is a multi-model fact-checking pipeline. It ingests a transcript of claims, dispatches verification across multiple LLM providers (Anthropic, OpenAI, Google, xAI) at two tiers (cheap **triage** + expensive **frontier**), reconciles per-claim consensus across the model panel, and publishes a static site (`site-test/` for local demos; production target lives wherever `TRUTHBOT_SITE_ROOT` points).

Code lives under `src/truthbot/`:
- Pipeline entry: [`src/truthbot/pipeline.py`](src/truthbot/pipeline.py)
- Adapters (per-provider): [`src/truthbot/verify/adapters/`](src/truthbot/verify/adapters/)
- Batch + dispatch: [`src/truthbot/verify/batch.py`](src/truthbot/verify/batch.py), [`engine.py`](src/truthbot/verify/engine.py)
- Publishing: [`src/truthbot/publish/site.py`](src/truthbot/publish/site.py)

## Where to look first

In rough order of how often you'll need them:

1. [**`PROJECT_BOARD.md`**](PROJECT_BOARD.md) — kanban (Backlog P0→P4, WIP, Done). Tells you what's prioritized, what's in flight, and what's already shipped (with PR links).
2. [**`STATUS.md`**](STATUS.md) — dated session log. Tells you what changed most recently and why.
3. [**`TODO.md`**](TODO.md) — phase-numbered long-running threads (publishers, infra).
4. **Open PRs** — `gh pr list --state open` for in-flight work; `gh pr view <n>` for context.
5. [**`eval/sotu-2026/`**](eval/sotu-2026/) — current canonical eval set + runbooks. The temporal-regressions runbook ([`temporal-regressions-runbook.md`](eval/sotu-2026/temporal-regressions-runbook.md)) is the live-validation playbook for accuracy regressions; **it costs real API budget to run.**
6. [**`metrics/adapter_interpretability/`**](metrics/adapter_interpretability/) — ad-hoc audit notes (e.g., [`strip_audit_2026-05.md`](metrics/adapter_interpretability/strip_audit_2026-05.md)) explaining what the harness's metrics actually mean. Read these before drawing conclusions from `metrics/run_summaries/*`.

## External tracking — P67

Truth-bot is tracked as **card `P67`** in the user's external cass-wip kanban (CSV-backed, accessed via the `dokuwiki` MCP server tools: `mcp__dokuwiki__csv_get`, `csv_list`, `csv_append_note`). P67 is the high-level, user-facing status surface — independent of this repo's `PROJECT_BOARD.md`, which remains the source of truth for in-repo work tracking.

When something notable happens (a PR merges, regression-set scoring shifts, a blocker is hit or cleared, a sub-project spins up), append a dated entry to P67's Notes via `csv_append_note(card_id="P67", text="…")`. The text gets auto-prefixed with the UTC date.

If a sub-project warrants its own card (a substantial workstream that should appear at the same kanban level as P67), discover the convention with `csv_list(area="Project")` and create the card via the appropriate `csv_*` tool — but default to extending P67 rather than fragmenting unless the work is genuinely independent.

## Conventions

- **Tests**: `uv run pytest -q`. Live tests are gated behind a `live` pytest marker and excluded by default — never opt in (`-m live`) without explicit user confirmation; they spend real API money.
- **Live pipeline runs**: same rule. The runbook in `eval/sotu-2026/temporal-regressions-runbook.md` describes the canonical live-validation procedure but it isn't free. Estimate cost; ask first.
- **Site artifacts under `site-test/`**: changes must be **additive**. If a regen would drop existing entries from `site-test/data/claims.json`, surface the conflict — don't silently commit a non-additive regen. The site is regeneratable from cache via [`scripts/republish_site_test_from_cache.py`](scripts/republish_site_test_from_cache.py).
- **Commits**: short imperative subject (`feat(verify): …`, `fix(pipeline): …`, `chore(metrics): …`); one logical change per commit; never amend, always make a new commit; include a `Co-Authored-By: Claude <noreply@anthropic.com>` trailer.
- **`.gitignore`** covers runtime telemetry/cache: `metrics/ab_probe_*/`, `metrics/temporal_regressions_*/`, `truthbot_cache_*/`, `.claude/`. Don't fight these — they're meant to stay untracked.
- **Branches**: long-running work on `claude/<topic>-<slug>` branches → open a draft PR early so it's visible → mark ready when validation passes → squash-merge → delete branch.

## Common tasks

- **Test suite**: `uv run pytest -q` (~1.5s; 786+ tests).
- **Re-publish demo from cache (no LLM spend)**: `uv run python scripts/republish_site_test_from_cache.py --skip-rebuild`.
- **Run the temporal-regressions live validation (COSTS MONEY)**: see [`eval/sotu-2026/temporal-regressions-runbook.md`](eval/sotu-2026/temporal-regressions-runbook.md). Confirm with user before running.
- **Fresh dependency install**: `uv sync --extra dev`.

## Background you'll want once

- **Two verification tiers**: `triage` (cheap, fast — Grok-4-fast / Haiku / Flash / mini-tier OpenAI) decides which claims short-circuit and which escape to `frontier` (Opus / Sonnet / GPT-5 / Pro). Adapter `_call_with_fallback` paths handle this.
- **Two consensus axes**: 6-bucket fine (model-facing rubric) and 5-bucket coarse "Truthy" scale (reader-facing), with **Lenient** (default) and **Strict** (regression-facing) projections.
- **URL grounding** is set-intersection between model-reported URLs and `tool_retrieved_urls` captured during the same API call — *not* HTTP HEAD validation. "Stripped" means the model claimed a citation the search tool never returned for this call. See [`strip_audit_2026-05.md`](metrics/adapter_interpretability/strip_audit_2026-05.md) before reading too much into "fabrication rate" numbers.
- **Verdict taxonomy** + source trust hierarchy: see [`README.md`](README.md).
