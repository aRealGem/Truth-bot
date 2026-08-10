# D15 + D16(α) — era breakdown (the M-6 evenhandedness check)

*Generated 2026-08-10T18:24:41.102558+00:00 · $0, no model calls. Stance vintage: **rescored** (the B1a re-score overlaid — the live state of the corpus, and the vintage the "measured 50" came from).*

- **D15** `TRUTHBOT_D15_UTTERANCE_RECORD (default OFF — NOT enabled)`
- **D16(α)** `TRUTHBOT_D16_STATISTICAL_RELEASE (default OFF — NOT enabled)`

- **source artifacts** `wave` `biden_2022`=8577979b, `clinton_1998`=fcbc8db2, `gwbush_2006`=0ae0f3b8, `obama_2014`=91d400ba, `trump_2026`=9c4262a7
- **baseline** `shipped` — movement measured against the gate outcome the source artifact RECORDED — nothing cancels, so the numbers are what the corpus loses relative to what was on the page, the re-score's own withholdings included

Every number below is measured against **what the source artifact actually recorded**, so it describes the change as a reader of the published page would experience it. The columns still separate the rules (D15 only, D16 only, both), but each is now compared to the shipped gate rather than to a rules-off recomputation — so a claim the B1a+B2 re-score withholds on its own is counted here and is not counted on the `recomputed` basis.

## 1. The three views, per speech

| speech | speaker | claims | D15 newly gated | …of which ship TRUE | D16(α) released | combined gated | combined released | **net** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gwbush_2006` | George W. Bush | 48 | 4 | 4 | 0 | 4 | 0 | **-4** |
| `clinton_1998` | Bill Clinton | 92 | 16 | 15 | 2 | 13 | 0 | **-13** |
| `obama_2014` | Barack Obama | 96 | 10 | 5 | 2 | 10 | 0 | **-10** |
| `biden_2022` | Joe Biden | 111 | 11 | 9 | 2 | 11 | 0 | **-11** |
| `trump_2026` | Donald Trump | 182 | 27 | 17 | 6 | 27 | 0 | **-27** |
| **corpus** | | **529** | **68** | **50** | **12** | **65** | **0** | **-65** |

"Net" is released minus newly gated: the number of claims the two rules together move *toward* a decided verdict. It is negative everywhere, because D15 removes far more credit than D16 gives back — which is the honest headline, and the reason these two must be reported on one page rather than two.

## 2. Decided-rate, before and after — both bases

Anecdote-adjusted excludes claims typed `personal-anecdote` (the A10 convention): a private individual's story told from the stage usually has no public record to check, so "Unverifiable" is the correct outcome rather than a miss. Both bases are shown because the adjustment is an argument, and a reader who rejects it must still see the raw figure it came from.

*Convention: before = what the artifacts ship; after = newly-gated claims forced Unverifiable, released claims counted as decided (UPPER bound; *_after_lower leaves them where they ship).*

| speech | anecdotes | raw before → after | Δ raw | adjusted before → after | Δ adjusted |
|---|---:|---|---:|---|---:|
| `gwbush_2006` | 1 | 85.4% → 77.1% | -8.3 pp | 85.1% → 76.6% | -8.5 pp |
| `clinton_1998` | 7 | 90.2% → 76.1% | -14.1 pp | 90.6% → 80.0% | -10.6 pp |
| `obama_2014` | 23 | 84.4% → 77.1% | -7.3 pp | 82.2% → 76.7% | -5.5 pp |
| `biden_2022` | 9 | 85.6% → 76.6% | -9.0 pp | 84.3% → 78.4% | -5.9 pp |
| `trump_2026` | 52 | 77.5% → 66.5% | -11.0 pp | 84.6% → 82.3% | -2.3 pp |

### Spread (max − min across the five speeches)

| basis | before | after | change |
|---|---|---|---|
| raw | 12.8% (trump_2026 … clinton_1998) | 10.6% (trump_2026 … gwbush_2006) | -2.2 pp |
| anecdote-adjusted | 8.4% (obama_2014 … clinton_1998) | 5.7% (gwbush_2006 … trump_2026) | -2.7 pp |

Both bases move the same way: raw narrows by 2.2 pp, anecdote-adjusted narrows by 2.7 pp.

## 3. Does the effect concentrate in one speaker or era?

**No material era concentration: per-speech withholding rates run 8.3% (gwbush_2006) to 14.8% (trump_2026), a ratio of 1.8x. By raw SHARE it is milder — the largest single share is Donald Trump at 42% of newly-gated claims on 34% of the corpus, an over-representation rather than a majority. Release lands entirely on George W. Bush (0%), but on a base of only 0 claim(s) — too few to read as a pattern.**

| speech | claims (share of corpus) | newly gated (share of all withholding) | withholding rate within the speech | released | net |
|---|---|---|---:|---:|---:|
| `gwbush_2006` | 48 (9.1%) | 4 (6.2%) | 8.3% | 0 | -4 |
| `clinton_1998` | 92 (17.4%) | 13 (20.0%) | 14.1% | 0 | -13 |
| `obama_2014` | 96 (18.1%) | 10 (15.4%) | 10.4% | 0 | -10 |
| `biden_2022` | 111 (21.0%) | 11 (16.9%) | 9.9% | 0 | -11 |
| `trump_2026` | 182 (34.4%) | 27 (41.5%) | 14.8% | 0 | -27 |

The **withholding rate within the speech** is the column to read: the five speeches differ by nearly a factor of four in claim count, so a raw count table alone would let "this speech has the most claims" masquerade as "the repair targets this speaker".

## 4. The claims, named

### `gwbush_2006` — George W. Bush

D15 would withhold 4 claim(s) that currently ship TRUE: `gwbush_2006:0033`, `gwbush_2006:0134`, `gwbush_2006:0189`, `gwbush_2006:0217`

### `clinton_1998` — Bill Clinton

D15 would withhold 15 claim(s) that currently ship TRUE: `clinton_1998:0006`, `clinton_1998:0026`, `clinton_1998:0027`, `clinton_1998:0038`, `clinton_1998:0101`, `clinton_1998:0107`, `clinton_1998:0134`, `clinton_1998:0135`, `clinton_1998:0195`, `clinton_1998:0210`, `clinton_1998:0211`, `clinton_1998:0225`, `clinton_1998:0227`, `clinton_1998:0236`, `clinton_1998:0243`

D15 would also gate 1 claim(s) not currently shipping TRUE: `clinton_1998:0358`

D16(α) would release: `clinton_1998:0090`, `clinton_1998:0350`

### `obama_2014` — Barack Obama

D15 would withhold 5 claim(s) that currently ship TRUE: `obama_2014:0001`, `obama_2014:0045`, `obama_2014:0123`, `obama_2014:0125`, `obama_2014:0198`

D15 would also gate 5 claim(s) not currently shipping TRUE: `obama_2014:0004`, `obama_2014:0070`, `obama_2014:0114`, `obama_2014:0126`, `obama_2014:0189`

D16(α) would release: `obama_2014:0153`, `obama_2014:0255`

### `biden_2022` — Joe Biden

D15 would withhold 9 claim(s) that currently ship TRUE: `biden_2022:0019`, `biden_2022:0100`, `biden_2022:0137`, `biden_2022:0171`, `biden_2022:0284`, `biden_2022:0376`, `biden_2022:0420`, `biden_2022:0427`, `biden_2022:0431`

D15 would also gate 2 claim(s) not currently shipping TRUE: `biden_2022:0124`, `biden_2022:0211`

D16(α) would release: `biden_2022:0146`, `biden_2022:0154`

### `trump_2026` — Donald Trump

D15 would withhold 17 claim(s) that currently ship TRUE: `trump_2026:0098`, `trump_2026:0099`, `trump_2026:0102`, `trump_2026:0106`, `trump_2026:0111`, `trump_2026:0255`, `trump_2026:0340`, `trump_2026:0341`, `trump_2026:0343`, `trump_2026:0482`, `trump_2026:0638`, `trump_2026:0643`, `trump_2026:0659`, `trump_2026:0660`, `trump_2026:0664`, `trump_2026:0665`, `trump_2026:0667`

D15 would also gate 10 claim(s) not currently shipping TRUE: `trump_2026:0043`, `trump_2026:0054`, `trump_2026:0057`, `trump_2026:0130`, `trump_2026:0137`, `trump_2026:0153`, `trump_2026:0256`, `trump_2026:0450`, `trump_2026:0487`, `trump_2026:0514`

D16(α) would release: `trump_2026:0279`, `trump_2026:0325`, `trump_2026:0329`, `trump_2026:0379`, `trump_2026:0402`, `trump_2026:0405`

## 5. Cross-check against the `stored` stance vintage

The pattern is not an artefact of the B1a re-score. On `stored` stances the corpus totals are: D15 newly gated 50, D16 released 0, net -48.

**YES — the withholding effect concentrates by ERA. The rule fires on 12.6% of trump_2026's claims and 2.1% of gwbush_2006's, a ratio of 6.1x (spread 10.6%). This is the size-adjusted number and it is the one to read. By raw SHARE it is milder — the largest single share is Donald Trump at 48% of newly-gated claims on 34% of the corpus, an over-representation rather than a majority. Release lands entirely on George W. Bush (0%), but on a base of only 0 claim(s) — too few to read as a pattern.**

