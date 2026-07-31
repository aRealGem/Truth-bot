# P129 — gold-15 before/after verdict-shift validation for PR-A

**Date:** 2026-07-31 · **Cost:** $2.34 metered (before $1.00 + after $1.34, proxy+off-proxy).
**Method:** the 15-claim verdict-gold fixture run through the full prod config (opus/grok/gpt +
CRM-114, shared_pack_v2), **same day**, on two checkouts — BEFORE = pre-PR-A `e3c73ec` (no S5
tier), AFTER = `main` (PR-A merged). Runner: `scripts/gold15_p129.py` (untracked). Artifacts:
`metrics/gold15_p129_{before,after}.json`, journals `metrics/journals/gold15_p129_*.jsonl`.

## Headline numbers
| | decided | decided-acc | coverage |
|---|---|---|---|
| BEFORE (pre-PR-A) | 14/15 | 0.714 | 0.93 |
| AFTER (PR-A) | 12/15 | 0.667 | 0.80 |

4 of 15 claims changed; **3 went decided → UNVERIFIABLE**, all via the T2.4 evidence-quality
gate. On its face that's an accuracy + coverage dip — but **the dip does not decompose to a
clean PR-A regression.** n=15 (each claim ≈ 7 pts), and the three movers are three different
stories:

## The three decided → Unverifiable claims, explained from the packs
1. **`trump2026-05`** (officers' biographies) — **PR-A working as intended.** Its decisive
   "support" was a NYC **mayor's-office press transcript** (`nyc.gov/mayors-office/news/…`), now
   S5. Demoting self-serving political comms → gate abstains. And BEFORE had it *wrong*
   (MISLEADING vs gold TRUE), so decided-wrong → abstain is neutral-to-good.
2. **`biden2022-06`** (infrastructure ranked 13th) — **NOT a PR-A effect.** Its AFTER pack was
   2 items, both `weforum.org` (OTHER tier), **zero government, zero political**. It gate-forced
   for pure retrieval thinness; BEFORE happened to retrieve enough to decide. This is same-day
   retrieval nondeterminism/web-drift, not tiering.
3. **`biden2022-03`** (vaccination/hospitalization stats) — **a fixable over-demotion bug.** Its
   pack was 9 Government + 1 Political, but the one clearly-supporting data item was
   `datahub.hhs.gov/Hospital/COVID-19-Reported-Patient-Impact/…` — an HHS **open-data hub** — now
   quarantined to S5 because the data signal is in its **hostname, not its path** (the
   `data_signal_segments` rule only inspects path segments). (Contributing: 8 CDC gov items came
   back `supports=None` — a relevance/bearing issue independent of tiering.)

## Read
- **No claim was wrongly flipped by PR-A's core ruling.** The single verdict genuinely changed
  by the S5 demotion (`trump2026-05`) was defensible and corrected a wrong call.
- The apparent 0.714→0.667 dip is **1 retrieval-noise abstention + 1 fixable-over-demotion
  abstention**. Neither indicts the ruling.
- This 15-claim slice is **too small/noisy to claim an accuracy delta** in either direction. The
  definitive signal remains the full 293-claim gold from the P67.9 rerun (**0.643 → 0.700**).

## Recommendations
1. **Small follow-up fix (correctness):** treat open-data-hub hosts as data — e.g. a hostname
   signal for a leading `data`/`datahub`/`datasets` label, or add `datahub.hhs.gov` (and peers
   like `healthdata.gov`, already OK) to `nonpartisan_sources`. This is the one real bug P129
   surfaced. ~$0, a quick follow-up PR.
2. **PR-A stays merged** — validation shows the core ruling behaving as designed, no misfire.
3. The T2.4 gate forcing UV when a claim's only real support is demoted/thin is the
   "correct-but-must-be-seen" outcome P129 exists to catch. Accept it; the fast-follow
   composition telemetry keeps it visible per run.
4. **No re-publish** without a further explicit go-ahead. The live site is unchanged.
