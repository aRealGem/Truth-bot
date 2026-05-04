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

Uses [`find_matching_bundle`](./temporal_regressions.py) (schema v2,
2026-05-01) to AND-match the case's `match_keywords` against the
bundle's claim text. Robust to extractor splits and "%" → "percent"
normalization that broke the runbook's earlier first-30-char anchor.

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("eval/sotu-2026")))
from temporal_regressions import (  # noqa: E402
    find_matching_bundle, load_temporal_regressions,
)

CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}


def score(bundles: "list[dict]") -> int:
    _, cases = load_temporal_regressions()
    pass_count = 0
    for case in cases:
        match = find_matching_bundle(case, bundles)
        if not match:
            print(f"FAIL  {case.id}: no matching claim in run output")
            continue
        consensus = match.get("consensus", {})
        fine_label = consensus.get("consensus_label")
        if isinstance(fine_label, dict):
            fine = fine_label.get("value", "")
        else:
            fine = fine_label or ""
        strict = consensus.get("coarse_strict_label") or fine
        confidence = consensus.get("confidence", "Low")
        if isinstance(confidence, dict):
            confidence = confidence.get("value", "Low")

        ok_fine = fine in case.test_acceptance["fine_label_in"]
        ok_strict = strict in case.test_acceptance["strict_label_in"]
        ok_conf = (
            CONFIDENCE_RANK.get(confidence, 0)
            >= CONFIDENCE_RANK[case.test_acceptance["min_confidence"]]
        )
        ok_tools = True
        min_tools = case.test_acceptance.get("min_adapters_with_tool_calls")
        if min_tools is not None:
            with_tools = sum(
                1 for mv in match.get("model_verdicts", [])
                if (mv.get("tool_call_count") or 0) > 0
            )
            ok_tools = with_tools >= min_tools

        passed = ok_fine and ok_strict and ok_conf and ok_tools
        pass_count += int(passed)
        marker = "PASS " if passed else "FAIL "
        detail = f"fine={fine}  strict={strict}  conf={confidence}"
        if min_tools is not None:
            detail += f"  tools_OK={ok_tools}"
        print(f"{marker}{case.id:40s}  {detail}")
    print(f"\n{pass_count} / {len(cases)} cases pass")
    return 0 if pass_count == len(cases) else 1


if __name__ == "__main__":
    # bundles can come from claims.jsonl OR the diskcache bundle store.
    src = Path(sys.argv[1])
    if src.is_file():
        bundles = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    else:
        import diskcache
        cache = diskcache.Cache(str(src / "bundles"))
        bundles = []
        for k in cache.iterkeys():
            v = cache.get(k)
            if v:
                try:
                    bundles.append(json.loads(v))
                except Exception:
                    pass
        cache.close()
    sys.exit(score(bundles))
```

## What the 2026-05-01 live run taught us

Run `cbc335a1-…` (4-claim live, ~$0.40, 0% strip rate) scored **0/4**
on the regression set. That's mixed news worth understanding:

- **Harness work succeeded.** Strip rate 0.0%; P1 + P2 + universal
  trust-when-fired are doing exactly what they were built to do.
- **Substance work hasn't started.** The 0/4 came from:
  - **Tool-firing on post-cutoff cases:** OpenAI / Gemini / xAI in
    live mode emitted **0 model_reported URLs** between them on this
    run. The models declined to invoke search for events they thought
    they knew from training data. That's the C3 finding (temporal
    dismissal), and it's a **prompt-side** fix, not a harness fix.
  - **Adjudication on Rubio:** "Models split" + Low confidence. This
    is the C4 finding — fine-axis tie-break can't capture directional
    agreement when the model panel literally objects to "100% but
    really 99-0". The family-aware dissent fix in `a006dd9` cleans up
    the *display* of that disagreement, not the consensus *verdict*.

So the regression set's purpose has shifted. It was originally
designed to validate the fix-wave; it now serves as the **acceptance
test for a separate substance-track work**:

1. Stronger temporal-grounding prompt that forces tool invocation
   for post-cutoff dates (or `tool_choice="required"` per-adapter).
2. Adjudication discipline — likely a consensus rule that promotes
   "Models split" → coarse-axis projection when the split is
   intra-truthy / intra-falsey (the literal C4 fix the family-aware
   dissent helper already builds the vocabulary for).
3. Possibly per-adapter prompt tuning on Gemini's "speculative
   fiction" framing for post-cutoff content.

Those three items form the substance track. Until they ship, the
regression set will read as 0/4 even on a clean harness. That's not
failure of the fix-wave — it's the fix-wave hitting its design ceiling.

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
