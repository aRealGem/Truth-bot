# truth-bot 🔍

Automated political rhetoric fact-checker. Feed it a transcript; get back a structured, scored, shareable fact-check report.

## What It Does

1. **Ingest** — Accepts speech transcripts as text, files, or URLs. Normalizes format and extracts metadata (speaker, date, venue).
2. **Extract** — Uses an LLM (Anthropic Claude) to decompose rhetoric into atomic, verifiable claims.
3. **Verify** — Checks each claim against credible sources: government data APIs (BLS, FRED, Census, CBO), Brave Search, and existing fact-check databases (PolitiFact, FactCheck.org).
4. **Score** — Assigns each claim a verdict from the taxonomy below, with a confidence level and supporting evidence.
5. **Publish** — Generates an HTML report, Bluesky thread, RSS feed entry, and JSON API response.

## Verdict Taxonomy

| Verdict | Meaning |
|---|---|
| `True` | Accurate and supported by primary sources |
| `Mostly True` | Accurate but missing nuance or context |
| `Misleading` | Technically accurate framing that implies something false |
| `Exaggerated` | Directionally correct but overstated |
| `False` | Contradicted by credible evidence |
| `Unverifiable` | Insufficient evidence to confirm or deny |

Confidence levels: **High** / **Medium** / **Low**

## Architecture

```
Transcript → [Ingest] → [Extract Claims] → [Verify per Claim]
                                               ↓
                                    [Score + Rubric] → [Cache]
                                               ↓
                              [Publish: Web / Bluesky / RSS / API]
```

Source trust hierarchy (descending):
1. Government primary data (BLS, FRED, CBO, Census)
2. Wire services (AP, Reuters)
3. Established outlets (NYT, WaPo, BBC)
4. Academic / NGO
5. Other

## Setup

### Requirements

- Python 3.11+
- API keys (see below)

### Install

```bash
git clone git@github.com:jackiemclean/truth-bot.git
cd truth-bot
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude API key for claim extraction + verdict synthesis |
| `BRAVE_API_KEY` | ✅ | Brave Search API for web evidence gathering |
| `BLUESKY_HANDLE` | optional | Your Bluesky handle (e.g. `yourname.bsky.social`) |
| `BLUESKY_APP_PASSWORD` | optional | Bluesky app password (not your main password) |
| `FRED_API_KEY` | optional | FRED (Federal Reserve) economic data API |

### Run

```bash
# Check a transcript file
truthbot --transcript speech.txt --speaker "Speaker Name" --date 2025-01-20

# Or pipe text
echo "Unemployment is at a 50-year low." | truthbot --speaker "Politician"

# Output formats
truthbot --transcript speech.txt --output html --output-dir ./reports/
truthbot --transcript speech.txt --post-bluesky
```

## Development

```bash
pytest -v                    # Run all tests
ruff check src/ tests/       # Lint
black src/ tests/            # Format
```

## Project Structure

```
src/truthbot/
├── config.py          — Settings from environment variables
├── models.py          — Pydantic data models (Transcript, Claim, Evidence, Verdict, Report)
├── pipeline.py        — End-to-end orchestrator
├── ingest/            — Transcript ingestion and normalization
├── extract/           — LLM-powered claim extraction
├── verify/            — Evidence gathering and verdict synthesis
│   └── sources/       — Pluggable source connectors
├── scoring/           — Verdict rubric and confidence scoring
├── cache/             — Claim deduplication and caching
└── publish/           — Output: HTML, Bluesky, RSS, JSON API
```

## Status

🚧 **Alpha** — Core architecture is in place; LLM integration and source connectors are stubbed and ready for implementation.

## License

MIT
