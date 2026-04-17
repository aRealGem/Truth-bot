# eval/ — Truth-Bot Evaluation Framework

This directory contains evaluation test cases for benchmarking truth-bot's performance against reference fact-checks.

## Structure

Each subdirectory is a named test case:

```
eval/
├── README.md              # This file
└── sotu-2026/             # 2026 State of the Union address
    ├── README.md          # Test case overview and scorecard
    ├── reference.md       # Full reference fact-check (clean markdown)
    ├── reference.json     # Structured claim data for automated comparison
    ├── transcript_urls.md # Source transcript URLs
    └── sources.md         # All reference sources with descriptions
```

## How Evaluation Works

Each test case provides:

1. **Reference fact-check** (`reference.md`, `reference.json`) — the "gold standard" to compare against. This is a human- or AI-produced fact-check of the same input that truth-bot will process.
2. **Source transcript URLs** (`transcript_urls.md`) — the input truth-bot should process.
3. **Structured claim data** (`reference.json`) — machine-readable claims with verdicts, for automated comparison against truth-bot's output.

To evaluate truth-bot on a test case:
1. Feed it the transcript URL(s) from `transcript_urls.md`
2. Compare its extracted claims and verdicts against `reference.json`
3. Score by: claim coverage, verdict agreement, explanation quality

## Reference Quality

The SOTU 2026 reference was produced by **GPT 5.4 Pro with extended thinking** (Feb 26, 2026), cross-checked against:
- PolitiFact
- FactCheck.org
- CBS News
- Associated Press
- Primary government data sources (BLS, BEA, CBP, EIA, Freddie Mac)

Overall confidence rating: **0.78** (per the model's self-assessment; higher for multi-source agreement, lower for inherently unmeasurable claims).

## Adding New Test Cases

1. Create a new subdirectory: `eval/<slug>/`
2. Add the five standard files (see structure above)
3. Add an entry to this README if needed
4. Ensure `reference.json` follows the schema defined in the SOTU 2026 example
