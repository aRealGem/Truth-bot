# Temporal-regressions live-run runbook (operator-only)

## What this is

Four claims from the 2026-04 ~117-claim SOTU run that the pipeline got
materially wrong (published "False" when ground truth lives in the
[Truthy, True] band). They are pinned in
[`temporal-regressions.json`](./temporal-regressions.json) as a
regression set targeting the 2026-04→05 fix wave:

- Phase 1 temporal grounding ([`23bb092`](https://github.com/aRealGem/Truth-bot/commit/23bb092))
- Grok triage cap ([`a319480`](https://github.com/aRealGem/Truth-bot/commit/a319480))
- "Model-cited (unverified)" tier ([`f447fa7`](https://github.com/aRealGem/Truth-bot/commit/f447fa7))
- Run-manifest panel ([`fab3a25`](https://github.com/aRealGem/Truth-bot/commit/fab3a25))
- Trust-when-fired harness fallback ([`ea10e34`](https://github.com/aRealGem/Truth-bot/commit/ea10e34))

The regression set itself is data + a schema validator + unit tests.
Actually exercising the pipeline against it requires a **live** run
(no truthbot_cache reuse) because cached HTML cannot reproduce the
OpenAI / Gemini temporal-dismissal failure mode that caused the
original wrong labels — see
[`metrics/adapter_interpretability/gemini_temporal_spot_sample.md`](../../metrics/adapter_interpretability/gemini_temporal_spot_sample.md).

## Why "live only"

Cached bundles encode the model's outputs from the original run. To
test the *new* pipeline's behavior on the same claims, the model has
to run again with the new prompts, the new tool-grounding policy, and
the new harness extraction path. Re-publishing from cache would just
replay the broken 2026-04 verdicts.

## Procedure

1. **Sync to head and confirm clean state.**

   ```bash
   git fetch && git checkout claude/repo-status-report-xli20
   git pull
   uv sync --extra dev
   uv run python -m pytest eval/tests/test_temporal_regressions.py -q
   ```

   Expect 13 passed.

2. **Build the 4-claim transcript.** A focused mini-transcript that
   contains just the four sentences:

   ```text
   Marco Rubio received 100% of Senate confirmation votes.
   The White House announced TrumpRx.gov in February 2026, bringing
   Most-Favored-Nation drug pricing to American patients.
   The Venezuela operation involved Russian and Chinese military
   technology supplied to the Maduro government.
   The Helicoide prison was closed and hundreds of political prisoners
   were released.
   ```

   Save as `data/temporal-regressions-mini-transcript.txt` (or wherever
   your local convention puts ad-hoc transcripts).

3. **Run a live publish with explicit cache bypass.**

   ```bash
   TRUTHBOT_FORCE_LIVE=1 \
     uv run truthbot publish \
       --transcript data/temporal-regressions-mini-transcript.txt \
       --speaker "Donald Trump" \
       --date 2026-02-24 \
       --venue "U.S. Capitol" \
       --event "2026 SOTU Address" \
       --output-dir metrics/temporal_regressions_$(date +%Y%m%d_%H%M)/
   ```

   (If `TRUTHBOT_FORCE_LIVE` isn't a real flag in your harness, the
   equivalent is whatever the local convention is to skip
   `truthbot_cache/`. If unsure, simply move
   `truthbot_cache/bundles/cache.db` aside before the run.)

4. **Read the run_summary and per-claim verdicts.**

   ```bash
   cat metrics/temporal_regressions_*/run_summaries/*.json | jq .fabrication
   cat metrics/temporal_regressions_*/claims.jsonl | \
     jq -c '{id: .claim_id, text: .claim.text[:80], strict: .consensus.coarse_strict_label, fine: .consensus.consensus_label.value}'
   ```

5. **Score against the pin.**

   For each of the four cases (matched on claim text), check the
   verdict against `test_acceptance` in `temporal-regressions.json`:

   - `consensus.consensus_label.value` ∈ `fine_label_in`?
   - `consensus.coarse_strict_label` ∈ `strict_label_in`?
   - `consensus.confidence` ≥ `min_confidence`?
   - For TrumpRx / Venezuela / Helicoide:
     `count(model_verdicts where tool_call_count > 0) ≥ min_adapters_with_tool_calls`?

   A matching helper script lives at the bottom of this doc.

## Acceptance criteria (per case)

| ID | Min strict label | Min fine label | Tools required |
|---|---|---|---|
| `rubio-100-percent-2026` | `Truthy` | `Mostly True` | n/a (pre-cutoff) |
| `trumprx-mfn-2026-02` | `Truthy` | `Mostly True` | ≥ 3 of 4 adapters fire web_search |
| `venezuela-russian-chinese-tech-2026` | `Falsey` | `Exaggerated` | ≥ 3 of 4 adapters fire web_search |
| `helicoide-prisoner-release-2026` | `Truthy` | `Mostly True` | ≥ 3 of 4 adapters fire web_search |

**Pass:** all 4 within Strict ±1 of ground truth.
**Partial:** 2-3 within ±1; one or two still flat-False indicates the
fix didn't fully land for those cases.
**Fail:** all 4 still flat-False — re-open the temporal-grounding
investigation; the trust-when-fired fallback alone wasn't enough and
you need to revisit the Phase 1 system-prompt copy or Gemini-specific
temporal-dismissal handling.

## Cost & time budget

- 4 claims × 4 adapters live ≈ $0.40-$0.60 per run (most of it Grok
  + Anthropic; OpenAI/Gemini are cheap on this volume)
- ~2-4 minutes wall-clock with normal API latency
- Run twice for variance (LLM outputs are non-deterministic even at
  temperature 0); consensus should be stable across runs

## Scoring helper (snippet)

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("eval/sotu-2026")))
from temporal_regressions import load_temporal_regressions  # noqa: E402

CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}


def score(claims_jsonl: Path) -> int:
    _, cases = load_temporal_regressions()
    rows = [json.loads(l) for l in claims_jsonl.read_text().splitlines() if l.strip()]
    pass_count = 0
    for case in cases:
        # Match on substring of claim text — claims are short enough that
        # a substring-match against the first 30 chars is reliable.
        anchor = case.claim[:30].lower()
        match = next(
            (r for r in rows if anchor in r["claim"]["text"].lower()),
            None,
        )
        if not match:
            print(f"FAIL  {case.id}: no matching claim in run output")
            continue
        consensus = match["consensus"]
        fine = consensus["consensus_label"]["value"]
        strict = consensus.get("coarse_strict_label") or fine
        confidence = consensus.get("confidence", "Low")
        ok_fine = fine in case.test_acceptance["fine_label_in"]
        ok_strict = strict in case.test_acceptance["strict_label_in"]
        ok_conf = (
            CONFIDENCE_RANK.get(confidence, 0)
            >= CONFIDENCE_RANK[case.test_acceptance["min_confidence"]]
        )
        # Tool-call requirement
        ok_tools = True
        min_tools = case.test_acceptance.get("min_adapters_with_tool_calls")
        if min_tools is not None:
            with_tools = sum(
                1 for mv in match["model_verdicts"]
                if (mv.get("tool_call_count") or 0) > 0
            )
            ok_tools = with_tools >= min_tools
        passed = ok_fine and ok_strict and ok_conf and ok_tools
        pass_count += int(passed)
        marker = "PASS " if passed else "FAIL "
        print(f"{marker}{case.id}  fine={fine}  strict={strict}  "
              f"conf={confidence}  ok={passed}")
    print(f"\n{pass_count} / {len(cases)} cases pass")
    return 0 if pass_count == len(cases) else 1


if __name__ == "__main__":
    sys.exit(score(Path(sys.argv[1])))
```

Save as `eval/sotu-2026/score_temporal_regressions.py` if you decide
to keep it; this doc inlines it for now.

## Reporting back

Once you've run the live probe, post:

1. The run output directory path
2. Per-case PASS/FAIL line from the scoring helper
3. The `run_summary.fabrication` block (so we can confirm the
   trust-when-fired fallback fired and produced 0% strip on
   OpenAI/Gemini for these claims too)
4. Total cost (from the run_summary)

If 4/4 pass, we ship the next 29-claim refresh with confidence. If
any fail, post the failure mode and we'll pick which knob to turn —
prompt copy in `verify/context/temporal.py`, Gemini-specific
temporal-dismissal preamble, or revisit the trust-when-fired
threshold.
