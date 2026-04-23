# Historical SOTU Transcripts

Small corpus of ten U.S. presidential **State of the Union** addresses, one per president from Nixon through Trump (2026), packaged in [`sotus_2nd_year_last_term.zip`](sotus_2nd_year_last_term.zip). Intended as evaluation / fact-check benchmark material for truth-bot.

## Contents

| File | President | Date | Words |
|------|-----------|------|------:|
| `nixon_1974_sotu.txt`   | Richard Nixon       | 1974-01-30 | ~5,156 |
| `ford_1975_sotu.txt`    | Gerald Ford         | 1975-01-15 | ~4,107 |
| `carter_1978_sotu.txt`  | Jimmy Carter        | 1978-01-19 | ~4,559 |
| `reagan_1986_sotu.txt`  | Ronald Reagan       | 1986-02-04 | ~3,472 |
| `ghwbush_1990_sotu.txt` | George H.W. Bush    | 1990-01-31 | ~3,727 |
| `clinton_1998_sotu.txt` | Bill Clinton        | 1998-01-27 | ~7,290 |
| `gwbush_2006_sotu.txt`  | George W. Bush      | 2006-01-31 | ~5,278 |
| `obama_2014_sotu.txt`   | Barack Obama        | 2014-01-28 | ~6,838 |
| `biden_2022_sotu.txt`   | Joe Biden           | 2022-03-01 | ~6,486 |
| `trump_2026_sotu.txt`   | Donald Trump        | 2026-02-24 | ~10,539 |

The zip also contains [`manifest.json`](#manifest), which lists each file with its president, date, year, word count, source URL, source name, and retrieval date.

## Selection rule

One SOTU per president, chosen as the **second full calendar year of the president's last (or only) term** — i.e. the speech delivered roughly 13 months into a term that the president ultimately did not extend. Rationale: this minimizes "first-SOTU victory-lap" boilerplate and captures each administration once it has substantive legislative and policy record to report on.

Applied to the modern post-Nixon era:

- Two-term presidents (Reagan, Clinton, GW Bush, Obama): year 2 of **second** term (1986, 1998, 2006, 2014).
- Single-term presidents (Carter, GHW Bush, Biden): year 2 of their **only** term (1978, 1990, 2022).
- Non-consecutive two-term (Trump): year 2 of **current** (second, non-consecutive) term (2026).
- **Nixon 1974**: year 2 of his second term (his last SOTU in office).

### Exception: Ford 1975

Gerald Ford **was never elected** to the presidency. He completed the remainder of Nixon's second term after Nixon's resignation in August 1974, and **lost** the 1976 general election before a "year 2 of his last term" could occur. The selection rule therefore has no canonical match for Ford.

We include **Ford's 1975 SOTU** — his first and (by the timing of his loss) only full SOTU as president — as the closest analog. It breaks the "year 2 / last term" pattern but preserves one-speech-per-president coverage. Consumers that need strict rule conformance should exclude `ford_1975_sotu.txt`.

## Source

All transcripts pulled from the **Miller Center (University of Virginia)** Presidential Speeches archive:
<https://millercenter.org/the-presidency/presidential-speeches>

Each `.txt` embeds its specific source URL in the second line of its header. The same URL is mirrored in `manifest.json` under `source_url`. Retrieved on **2026-04-22**.

## File format

Every `.txt` is UTF-8 and starts with two `#`-prefixed header lines:

```
# State of the Union — <President> <Year>
# Source: https://millercenter.org/...
```

…followed by a blank line and the full speech body. Downstream tokenizers should **skip `#`-prefixed lines** at the top of each file when counting words or feeding the text to ingestion.

## Manifest

`manifest.json` is a JSON array with one object per transcript:

```json
{
  "filename": "trump_2026_sotu.txt",
  "president": "Donald Trump",
  "date": "2026-02-24",
  "year": 2026,
  "word_count": 10539,
  "source_url": "https://millercenter.org/the-presidency/presidential-speeches/february-24-2026-state-union-address",
  "source_name": "Miller Center (University of Virginia)",
  "retrieved_on": "2026-04-22"
}
```

`word_count` excludes the two `#` header lines.
