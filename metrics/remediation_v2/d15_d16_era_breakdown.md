# D15 + D16(α) — era breakdown (the M-6 evenhandedness check)

*Generated 2026-08-09T15:16:31.984117+00:00 · $0, no model calls. Stance vintage: **rescored** (the B1a re-score overlaid — the live state of the corpus, and the vintage the "measured 50" came from).*

- **D15** `TRUTHBOT_D15_UTTERANCE_RECORD (default OFF — NOT enabled)`
- **D16(α)** `TRUTHBOT_D16_STATISTICAL_RELEASE (default OFF — NOT enabled)`

Both flags are OFF in the committed tree. Every number below is what ratification *would* do, computed by running the real gate over the stored packs four ways — both off, D15 only, D16 only, both.

## 1. The three views, per speech

| speech | speaker | claims | D15 newly gated | …of which ship TRUE | D16(α) released | combined gated | combined released | **net** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gwbush_2006` | George W. Bush | 48 | 2 | 2 | 0 | 2 | 0 | **-2** |
| `clinton_1998` | Bill Clinton | 92 | 10 | 8 | 2 | 10 | 2 | **-8** |
| `obama_2014` | Barack Obama | 96 | 7 | 3 | 0 | 7 | 0 | **-7** |
| `biden_2022` | Joe Biden | 111 | 8 | 6 | 0 | 8 | 0 | **-8** |
| `trump_2026` | Donald Trump | 182 | 23 | 14 | 0 | 23 | 0 | **-23** |
| **corpus** | | **529** | **50** | **33** | **2** | **50** | **2** | **-48** |

"Net" is released minus newly gated: the number of claims the two rules together move *toward* a decided verdict. It is negative everywhere, because D15 removes far more credit than D16 gives back — which is the honest headline, and the reason these two must be reported on one page rather than two.

## 2. Decided-rate, before and after — both bases

Anecdote-adjusted excludes claims typed `personal-anecdote` (the A10 convention): a private individual's story told from the stage usually has no public record to check, so "Unverifiable" is the correct outcome rather than a miss. Both bases are shown because the adjustment is an argument, and a reader who rejects it must still see the raw figure it came from.

*Convention: before = what the artifacts ship; after = newly-gated claims forced Unverifiable, released claims counted as decided (UPPER bound; *_after_lower leaves them where they ship).*

| speech | anecdotes | raw before → after | Δ raw | adjusted before → after | Δ adjusted |
|---|---:|---|---:|---|---:|
| `gwbush_2006` | 1 | 83.3% → 79.2% | -4.2 pp | 83.0% → 78.7% | -4.3 pp |
| `clinton_1998` | 7 | 83.7% → 77.2% | -6.5 pp | 83.5% → 81.2% | -2.4 pp |
| `obama_2014` | 23 | 81.2% → 77.1% | -4.2 pp | 78.1% → 76.7% | -1.4 pp |
| `biden_2022` | 9 | 82.9% → 77.5% | -5.4 pp | 81.4% → 77.5% | -3.9 pp |
| `trump_2026` | 52 | 73.1% → 65.4% | -7.7 pp | 79.2% → 78.5% | -0.8 pp |

### Spread (max − min across the five speeches)

| basis | before | after | change |
|---|---|---|---|
| raw | 10.6% (trump_2026 … clinton_1998) | 13.8% (trump_2026 … gwbush_2006) | +3.2 pp |
| anecdote-adjusted | 5.5% (obama_2014 … clinton_1998) | 4.5% (obama_2014 … clinton_1998) | -1.0 pp |

**The two bases disagree, and the disagreement is the finding.** On the raw basis the spread widens by 3.2 pp; on the anecdote-adjusted basis it narrows by 1.0 pp. The raw movement is driven by how many personal anecdotes a speech contains — `trump_2026` carries 52 of them and `gwbush_2006` carries 1 — not by how the two rules treat evidence. On the basis that controls for that, ratifying D15 + D16(α) leaves era parity essentially where it found it.

## 3. Does the effect concentrate in one speaker or era?

**YES — the withholding effect concentrates by ERA. The rule fires on 12.6% of trump_2026's claims and 4.2% of gwbush_2006's, a ratio of 3.0x (spread 8.5%). This is the size-adjusted number and it is the one to read. By raw SHARE it is milder — the largest single share is Donald Trump at 46% of newly-gated claims on 34% of the corpus, an over-representation rather than a majority. Release lands entirely on Bill Clinton (100%), but on a base of only 2 claim(s) — too few to read as a pattern.**

| speech | claims (share of corpus) | newly gated (share of all withholding) | withholding rate within the speech | released | net |
|---|---|---|---:|---:|---:|
| `gwbush_2006` | 48 (9.1%) | 2 (4.0%) | 4.2% | 0 | -2 |
| `clinton_1998` | 92 (17.4%) | 10 (20.0%) | 10.9% | 2 | -8 |
| `obama_2014` | 96 (18.1%) | 7 (14.0%) | 7.3% | 0 | -7 |
| `biden_2022` | 111 (21.0%) | 8 (16.0%) | 7.2% | 0 | -8 |
| `trump_2026` | 182 (34.4%) | 23 (46.0%) | 12.6% | 0 | -23 |

The **withholding rate within the speech** is the column to read: the five speeches differ by nearly a factor of four in claim count, so a raw count table alone would let "this speech has the most claims" masquerade as "the repair targets this speaker".

## 4. The claims, named

### `gwbush_2006` — George W. Bush

D15 would withhold 2 claim(s) that currently ship TRUE: `gwbush_2006:0033`, `gwbush_2006:0189`

### `clinton_1998` — Bill Clinton

D15 would withhold 8 claim(s) that currently ship TRUE: `clinton_1998:0027`, `clinton_1998:0134`, `clinton_1998:0135`, `clinton_1998:0195`, `clinton_1998:0225`, `clinton_1998:0227`, `clinton_1998:0236`, `clinton_1998:0243`

D15 would also gate 2 claim(s) not currently shipping TRUE: `clinton_1998:0090`, `clinton_1998:0350`

D16(α) would release: `clinton_1998:0026`, `clinton_1998:0038`

### `obama_2014` — Barack Obama

D15 would withhold 3 claim(s) that currently ship TRUE: `obama_2014:0045`, `obama_2014:0123`, `obama_2014:0125`

D15 would also gate 4 claim(s) not currently shipping TRUE: `obama_2014:0114`, `obama_2014:0126`, `obama_2014:0153`, `obama_2014:0255`

### `biden_2022` — Joe Biden

D15 would withhold 6 claim(s) that currently ship TRUE: `biden_2022:0019`, `biden_2022:0137`, `biden_2022:0171`, `biden_2022:0284`, `biden_2022:0420`, `biden_2022:0431`

D15 would also gate 2 claim(s) not currently shipping TRUE: `biden_2022:0146`, `biden_2022:0154`

### `trump_2026` — Donald Trump

D15 would withhold 14 claim(s) that currently ship TRUE: `trump_2026:0098`, `trump_2026:0099`, `trump_2026:0102`, `trump_2026:0106`, `trump_2026:0111`, `trump_2026:0255`, `trump_2026:0340`, `trump_2026:0343`, `trump_2026:0482`, `trump_2026:0638`, `trump_2026:0643`, `trump_2026:0659`, `trump_2026:0660`, `trump_2026:0664`

D15 would also gate 9 claim(s) not currently shipping TRUE: `trump_2026:0153`, `trump_2026:0256`, `trump_2026:0279`, `trump_2026:0325`, `trump_2026:0329`, `trump_2026:0379`, `trump_2026:0402`, `trump_2026:0405`, `trump_2026:0514`

## 5. Cross-check against the `stored` stance vintage

The pattern is not an artefact of the B1a re-score. On `stored` stances the corpus totals are: D15 newly gated 48, D16 released 2, net -46.

**YES — the withholding effect concentrates by ERA. The rule fires on 12.6% of trump_2026's claims and 2.1% of gwbush_2006's, a ratio of 6.1x (spread 10.6%). This is the size-adjusted number and it is the one to read. By raw SHARE it is milder — the largest single share is Donald Trump at 48% of newly-gated claims on 34% of the corpus, an over-representation rather than a majority. Release lands entirely on Bill Clinton (100%), but on a base of only 2 claim(s) — too few to read as a pattern.**

