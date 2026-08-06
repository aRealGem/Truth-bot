# DC-6 review package — Phase-3 rebuild vs published corpus

Generated `2026-08-06` · generation `v2.3-role-axis-s5cap` · $0 (derived from artifacts on disk; no model calls).

**This is a review surface, not a publish.** The rebuilt artifacts are committed with `published: false`; `site-pca/` still serves the old runs. Read the changed-claim tables below and decide.

## 1. Headline

- Claims re-adjudicated: **529**
- Unchanged: **388** · Changed: **141** (26.7%)
- Decided → decided (flipped between substantive verdicts): **49**
- Newly gated (decided → withheld): **66**
- Newly decided (withheld → decided): **17**
- Split changes: **9** · Other: **0**
- Gate-forced Unverifiable in the new runs: **87**

- Corpus decided-rate: **90.4%** (479/530) → **79.4%** (420/529), -11.0 pts

## 2. Flags — read these before anything else

- **era parity: the spread of decided-rates across the five speeches WIDENED from 4.3 pts to 10.6 pts (trump_2026 is now the least-decided at 73.1% vs clinton_1998 at 83.7%). The rebuild unified the METHOD; it did not equalise the OUTCOME**
- **trump_2026: 1 sid(s) present in the published run (183 rows) are ABSENT from the rebuild (182 rows) and were never re-adjudicated — trump_2026:0311 (old verdict FALSE, ORPHAN — the published run had no claim record for it and rendered '(claim text unavailable)'). Publishing changes that report's claim count from 183 to 182; 1 of the dropped sid(s) were orphan rows, so the drop removes a broken card rather than losing a checked claim**
- **badge diff: 1 claim(s) present only in the published site and 0 only in the staged render**
- **biden_2022: run logs total $8.3732 vs $8.00 stated in the DC-6 brief (+0.3732)**
- **corpus: run logs total $39.7802 (incl. $0.63 shape backfill) vs ~$38.80 stated in the brief; the brief's per-speech figures themselves sum to $38.78 BEFORE the backfill, so the stated total appears to double-count the backfill as already included**

## 3. Per-speech change counts

| speech | speaker | claims | unchanged | dec→dec | newly gated | newly decided | splits | changed |
|---|---|---|---|---|---|---|---|---|
| clinton_1998 | Bill Clinton | 92 | 69 | 8 | 11 | 4 | 0 | 23 |
| gwbush_2006 | George W. Bush | 48 | 37 | 3 | 5 | 3 | 0 | 11 |
| obama_2014 | Barack Obama | 96 | 75 | 7 | 7 | 4 | 3 | 21 |
| biden_2022 | Joe Biden | 111 | 91 | 3 | 12 | 3 | 2 | 20 |
| trump_2026 | Donald Trump | 182 | 116 | 28 | 31 | 3 | 4 | 66 |
| **corpus** |  | 529 | 388 | 49 | 66 | 17 | 9 | 141 |

## 4. Verdict distribution — old vs new

Old = the published run's rows; new = the rebuilt run's rows. `decided` = every claim that is not Unverifiable and not a model split.

### Bill Clinton — `clinton_1998`

| run | True | Mostly True | Misleading | False | Unverifiable | Models split | decided | decided-rate |
|---|---|---|---|---|---|---|---|---|
| old | 72 | 0 | 9 | 2 | 8 | 1 | 83/92 | 90.2% |
| new | 71 | 0 | 5 | 1 | 14 | 1 | 77/92 | 83.7% |

decided-rate change: **-6.5 pts**

### George W. Bush — `gwbush_2006`

| run | True | Mostly True | Misleading | False | Unverifiable | Models split | decided | decided-rate |
|---|---|---|---|---|---|---|---|---|
| old | 37 | 0 | 5 | 0 | 6 | 0 | 42/48 | 87.5% |
| new | 35 | 0 | 4 | 1 | 8 | 0 | 40/48 | 83.3% |

decided-rate change: **-4.2 pts**

### Barack Obama — `obama_2014`

| run | True | Mostly True | Misleading | False | Unverifiable | Models split | decided | decided-rate |
|---|---|---|---|---|---|---|---|---|
| old | 78 | 0 | 3 | 4 | 10 | 1 | 85/96 | 88.5% |
| new | 71 | 0 | 5 | 2 | 15 | 3 | 78/96 | 81.2% |

decided-rate change: **-7.3 pts**

### Joe Biden — `biden_2022`

| run | True | Mostly True | Misleading | False | Unverifiable | Models split | decided | decided-rate |
|---|---|---|---|---|---|---|---|---|
| old | 96 | 0 | 4 | 1 | 7 | 3 | 101/111 | 91.0% |
| new | 86 | 0 | 5 | 1 | 17 | 2 | 92/111 | 82.9% |

decided-rate change: **-8.1 pts**

### Donald Trump — `trump_2026`

| run | True | Mostly True | Misleading | False | Unverifiable | Models split | decided | decided-rate |
|---|---|---|---|---|---|---|---|---|
| old | 94 | 0 | 31 | 43 | 14 | 1 | 168/183 | 91.8% |
| new | 74 | 0 | 21 | 38 | 45 | 4 | 133/182 | 73.1% |

decided-rate change: **-18.7 pts**  ⚠ old and new denominators differ — see flags

### Corpus

| run | True | Mostly True | Misleading | False | Unverifiable | Models split | decided | decided-rate |
|---|---|---|---|---|---|---|---|---|
| old | 377 | 0 | 52 | 50 | 45 | 6 | 479/530 | 90.4% |
| new | 337 | 0 | 40 | 43 | 99 | 10 | 420/529 | 79.4% |

**Era parity** — decided-rate, oldest speech to newest, old → new:

- `clinton_1998` 90.2% → 83.7%  ·  `gwbush_2006` 87.5% → 83.3%  ·  `obama_2014` 88.5% → 81.2%  ·  `biden_2022` 91.0% → 82.9%  ·  `trump_2026` 91.8% → 73.1%

Spread across speeches (max − min decided-rate): **4.3%** old (`trump_2026` 91.8% vs `gwbush_2006` 87.5%) → **10.6%** new (`clinton_1998` 83.7% vs `trump_2026` 73.1%), +6.3 pts — **WIDENED**.

> The unified pipeline did NOT equalise decided-rates across eras; it spread them further apart. A reader should not read this rebuild as having produced era parity in outcome — it produced one methodology, applied uniformly, whose per-speech decided-rates differ more than before. Judge the publish on that basis.

## 5. Every changed claim

Ordered by consequence: verdicts that flipped between two substantive rulings first, then claims newly withheld, then claims newly decided, then split changes. Rationale is the NEW panel's reasoning.

### Decided → decided (verdict flipped between substantive rulings) — 49 claim(s)

#### Bill Clinton — `clinton_1998` (8)

- **`clinton_1998:0007`** · True → **Misleading**
  - claim: Crime has dropped for a record five years in a row, and the welfare rolls are at their lowest levels in 27 years.
  - new rationale: FBI data support a fifth consecutive crime decline, but government welfare data show 1997 rolls/recipiency were not the lowest in 27 years, contradicting the absolute welfare claim
- **`clinton_1998:0057`** · Misleading → **True**
  - claim: Last year I proposed and you passed 220,000 new Pell grant scholarships for deserving students.
  - new rationale: Contemporaneous White House signing statement and Clinton's own signing remarks (Nov 1997) both attest the enacted Pell increase made an additional 220,000 students eligible, matching the claim's cor…
- **`clinton_1998:0071`** · Misleading → **True**
  - claim: Thanks to the actions of this Congress last year, we will soon have, for the very first time, a voluntary national test based on national s…
  - new rationale: Government evidence confirms P.L. 105-78 (enacted Nov 1997) funded development of the first-ever voluntary national tests in 4th-grade reading and 8th-grade math on NAEP standards, matching the claim…
- **`clinton_1998:0156`** · Misleading → **True**
  - claim: Millions of Americans between the ages of 55 and 65 have lost their health insurance.
  - new rationale: Census and EBRI show ~3 million near-elderly (55–64) uninsured in 1996, so 'millions' lacking coverage—with documented erosion of retiree/employer coverage driving loss—is supported without material…
- **`clinton_1998:0170`** · Misleading → **False**
  - claim: Since then, about 15 million people have taken advantage of it, and I've met a lot of them all across this country.
  - new rationale: official FMLA survey/Commission evidence put actual FMLA leave use far below the claimed roughly 15 million by the utterance date
- **`clinton_1998:0192`** · False → **Misleading**
  - claim: I think every American should know that most juvenile crime is committed between the hours of 3 in the afternoon and 8 at night.
  - new rationale: the federal data showed an after-school peak but only about 20% of juvenile violent crimes in the relevant afternoon period, not a majority between 3 p.m. and 8 p.m.
- **`clinton_1998:0270`** · True → **Misleading**
  - claim: Under the leadership of Vice President Gore, we've reduced the federal payroll by 300,000 workers, cut 16,000 pages of regulation, eliminat…
  - new rationale: real NPR-era workforce, regulatory, and program cuts were documented, but government data and GAO/Congress reviews show the 300,000 figure and sweeping agency-improvement attribution were overstated…
- **`clinton_1998:0337`** · Unverifiable → **True**
  - claim: Now, think about this: The entire store of human knowledge now doubles every five years.
  - new rationale: contemporaneous sources stated that broadly available information and medical knowledge doubled about every five years, supporting the claim as made

#### George W. Bush — `gwbush_2006` (3)

- **`gwbush_2006:0154`** · True → **Misleading**
  - claim: Every year of my Presidency, we've reduced the growth of nonsecurity discretionary spending, and last year you passed bills that cut this s…
  - new rationale: the absolute claim that growth was reduced every year and that the prior year’s bills cut the category is contradicted by budget data showing nonsecurity/nondefense discretionary authority or funding…
- **`gwbush_2006:0218`** · Misleading → **False**
  - claim: There are fewer abortions in America than at any point in the last three decades, and the number of children born to teenage mothers has be…
  - new rationale: The teen-birth half is true (12th straight annual decline, E4/E6), but the claim's leading absolute — abortions lower than at any point in three decades — is contradicted, since government and Guttma…
- **`gwbush_2006:0249`** · Misleading → **True**
  - claim: We're removing debris and repairing highways and rebuilding stronger levees.
  - new rationale: Contemporary government/wire evidence confirms all three ongoing efforts — debris removal (E4, E5), highway repair (E1, E3), and levee rebuilding to stronger design (E6, E9, E10) — with only E7 tempe…

#### Barack Obama — `obama_2014` (7)

- **`obama_2014:0114`** · True → **Unverifiable**
  - claim: And today, Detroit Manufacturing Systems has more than 700 employees.
  - new rationale: All supporting items merely restate the speech's own figure; no independent employment record confirms >700, though E5's >600-in-first-year growth is loosely consistent.
- **`obama_2014:0125`** · Unverifiable → **True**
  - claim: She put herself through college.
  - new rationale: Official White House guest bio and independent reporting confirm DeMars put herself through college.
- **`obama_2014:0157`** · Misleading → **True**
  - claim: Last year, I also pledged to connect 99 percent of our students to high-speed broadband over the next four years.
  - new rationale: Obama made the 2013 ConnectED pledge to connect 99% of students to high-speed broadband within five years, which reasonably corresponded to roughly four years remaining at the January 2014 utterance
- **`obama_2014:0189`** · True → **Misleading**
  - claim: Today, the federal minimum wage is worth about twenty percent less than it was when Ronald Reagan first stood here.
  - new rationale: The real minimum wage did decline since Reagan's era, but the best inflation-adjusted estimates put the drop at ~12-16% ($3.35/1981 ≈ $8.24-$8.61 in 2013-14 dollars vs $7.25), so 'about twenty percen…
- **`obama_2014:0218`** · True → **Unverifiable**
  - claim: Just one week earlier, Amanda said, that surgery would’ve meant bankruptcy.
  - new rationale: provided evidence only documents that the SOTU contained the assertion, not independent evidence that Amanda actually said the surgery would have meant bankruptcy
- **`obama_2014:0221`** · True → **Misleading**
  - claim: More than nine million Americans have signed up for private health insurance or Medicaid coverage.
  - new rationale: official contemporaneous figures showed about 2.15 million marketplace plan selections and roughly 6.3–6.6 million Medicaid/CHIP applications or determinations, which did not establish more than nine…
- **`obama_2014:0223`** · False → **Misleading**
  - claim: Because of this law, no American can ever again be dropped or denied coverage for a preexisting condition like asthma, back pain, or cancer.
  - new rationale: the ACA broadly barred preexisting-condition denials starting in 2014, but the absolute claim is contradicted by documented exceptions such as grandfathered individual plans and transitional non-enfo…

#### Joe Biden — `biden_2022` (3)

- **`biden_2022:0167`** · Misleading → **True**
  - claim: Ford is investing $11 billion to build electric vehicles, creating 11,000 jobs across the country.
  - new rationale: Ford and SK announced about $11.4 billion in EV/battery investment and nearly/about 11,000 new jobs before the utterance date
- **`biden_2022:0211`** · True → **Misleading**
  - claim: Look, the American Rescue Plan is helping millions of families on Affordable Care Act plans save $2,400 a year on their health care premium…
  - new rationale: ARP did lower ACA marketplace premiums for millions, and $2,400/year matches an illustrative family-of-four calculation, but evidence shows average savings were lower for typical individuals/househol…
- **`biden_2022:0431`** · Unverifiable → **True**
  - claim: He loved building Legos with their daughter.
  - new rationale: Gold Star tribute and official SOTU record corroborate that Heath loved building Legos with his daughter.

#### Donald Trump — `trump_2026` (28)

- **`trump_2026:0017`** · Misleading → **False**
  - claim: Today our border is secure, our spirit is restored, inflation is plummeting, incomes are rising fast, the roaring economy is roaring like n…
  - new rationale: the absolute 'roaring like never before' is contradicted by BEA data showing GDP decelerating (1.4% Q4, 2.2% year vs 2.8%), while 'inflation plummeting' and 'incomes rising fast' overstate merely eas…
- **`trump_2026:0019`** · True → **Misleading**
  - claim: After four years in which millions and millions of illegal aliens poured across our borders totally unvetted and unchecked, we now have the…
  - new rationale: encounters/releases were reported at historic lows and official releases echoed the framing, but the evidence supports at most a recent historic low/zero releases, not the sweeping claim of the stron…
- **`trump_2026:0023`** · True → **Misleading**
  - claim: And last year the murder rate saw its single largest decline in recorded history.
  - new rationale: available 2025 data showed a large preliminary decline, but the record-setting national claim was only a projection and AP reported FBI confirmation was not yet available as of the utterance
- **`trump_2026:0024`** · Misleading → **True**
  - claim: This is the biggest decline, think of it, in recorded history—the lowest number in over 125 years.
  - new rationale: contemporary CCJ-based reporting supported that 2025 had about a 21% homicide decline, the largest one-year drop on record, and the lowest homicide rate since records back to 1900
- **`trump_2026:0031`** · False → **True**
  - claim: And in the last three months of 2025, it was down to 1.7 percent.
  - new rationale: Primary BLS data and outlet analyses confirm the Q4-2025 three-month annualized core CPI rate was ~1.7%, its lowest since early 2021, matching the claim's specific 'last three months' framing (distin…
- **`trump_2026:0046`** · Misleading → **False**
  - claim: Think of it, much less than $1 trillion for four years versus much more than $18 trillion for one year.
  - new rationale: The core $18T-in-one-year figure is contradicted by BEA FDI data (~$151–232B/yr) and audits showing even the White House's own tracker (~$9.6T) is largely unenforceable pledges, not secured investmen…
- **`trump_2026:0053`** · True → **Misleading**
  - claim: American natural gas production is at an all-time high because I kept my promise to drill baby drill.
  - new rationale: U.S. natural gas production was at record levels, but the evidence attributes the growth to continuing shale-region trends and prior record-setting years rather than uniquely to the speaker’s new “dr…
- **`trump_2026:0054`** · True → **False**
  - claim: More Americans are working today than at any time in the history of our country.
  - new rationale: Government data shows the all-time employment peak was Dec 2025 (163,992K) with Jan/Feb 2026 declines, so 'today' is below the record, contradicting this absolute claim.
- **`trump_2026:0057`** · Misleading → **Unverifiable**
  - claim: We cut a record number of job killing regulations and, in one year, we have lifted 2.4 million Americans, a record, off of food stamps.
  - new rationale: the cited SNAP tables and deregulation guidance are pointers whose snippets contain no actual figures to confirm the 2.4-million or either 'record' claim
- **`trump_2026:0092`** · False → **Misleading**
  - claim: And Los Angeles is going to be safe just like Washington D.C. is now one of the safest cities in the country.
  - new rationale: DC's dramatic, real crime decline (30-year low, homicides down ~32%) is spun into the exaggerated impression that DC is now 'one of the safest cities in the country,' which the evidence of substantia…
- **`trump_2026:0106`** · False → **True**
  - claim: I was there.
  - new rationale: contemporary reports document the speaker visiting and surveying the Kerr County/Camp Mystic flood site in person after the July 4, 2025 disaster
- **`trump_2026:0109`** · True → **Unverifiable**
  - claim: As the waters threatened to sweep her away, 11-year-old Millie Kate McClelland closed her eyes and prayed to God.
  - new rationale: the provided evidence shows the line was said or repeated in transcripts and coverage, but does not independently verify that the child actually closed her eyes and prayed during the flood
- **`trump_2026:0137`** · Misleading → **Unverifiable**
  - claim: Megan is here this evening and she's happy to tell you that she is so, so much richer because, with no tax on tips, no tax on overtime and…
  - new rationale: the tip/overtime/CTC provisions are real and $5,000 is plausible per IRS caps, but no evidence discloses Megan's actual household tax figures to confirm the >$5,000 / halved-bill magnitude
- **`trump_2026:0153`** · True → **Unverifiable**
  - claim: You know, I asked Michael Dell, how do you make all that money?
  - new rationale: provided evidence corroborates Dell’s public dorm-room computer origin story and the speech transcript, but not the private assertion that Trump personally asked Dell this question
- **`trump_2026:0186`** · False → **True**
  - claim: But the good news is that almost all countries and corporations want to keep the deal that they already made.
  - new rationale: Wire and government sources confirm partners (EU explicitly, others per USTR) want to keep the existing deals with no partner signalling withdrawal, supporting the core assertion despite one lower-ti…
- **`trump_2026:0215`** · Misleading → **False**
  - claim: Their policies created the high prices.
  - new rationale: Independent economic analyses attribute the price surge mainly to supply-chain shocks, not the prior administration's policies, contradicting the sole-causation core (only partisan GOP research suppo…
- **`trump_2026:0219`** · Misleading → **True**
  - claim: The price of eggs is down 60 percent.
  - new rationale: Government BLS/USDA data show a ~59–60% peak-to-current retail decline (Mar 2025 $6.23 → Jan 2026 $2.58), matching the claim, though YoY was only ~34%.
- **`trump_2026:0231`** · True → **Misleading**
  - claim: With our government giving them hundreds and hundreds of billions of dollars a year as their stock prices soared 1,000, 1,200, 1,400 and ev…
  - new rationale: Real underlying facts (hundreds of billions in federal payments to private insurers, large multi-year insurer stock gains) but exaggerated and cherry-picked — ACA subsidies themselves run ~$125B/yr,…
- **`trump_2026:0241`** · False → **True**
  - claim: Other presidents tried to do it, but they never could.
  - new rationale: Government sources confirm Medicare gained negotiation authority 'for the first time' under the IRA, so prior administrations' drug-pricing efforts never achieved it.
- **`trump_2026:0319`** · Misleading → **True**
  - claim: The Somali pirates who ransacked Minnesota, remind us that there are large parts of the world where bribery, corruption, and lawlessness ar…
  - new rationale: Core assertions — massive Somali-linked fraud in Minnesota ('ransacked') and Somalia among the most corrupt nations where corruption is the norm — are both directly supported; 'pirates' is rhetorical…
- **`trump_2026:0343`** · False → **True**
  - claim: Her heartbroken mother is in the gallery to remind everyone in this chamber exactly why we are deporting illegal alien criminals from our c…
  - new rationale: Official DHS releases contemporaneous with the utterance support record-number removals of illegal-alien criminals, backing the core assertion.
- **`trump_2026:0450`** · Unverifiable → **Misleading**
  - claim: No one will ever forget—there were people on that train, no one will ever forget the expression of terror on Iryna's face as she looked up…
  - new rationale: the core assertion that Iryna looked up at her attacker in terror in her final seconds is contradicted by affidavit-based reporting that surveillance showed the attacker struck her from behind with n…
- **`trump_2026:0487`** · True → **Unverifiable**
  - claim: This was a conversation that I had with her that night with her son laying hopelessly in bed, blood all over.
  - new rationale: provided evidence confirms the remark was made and the shooting occurred but does not independently verify the private conversation or bedside details
- **`trump_2026:0509`** · Misleading → **False**
  - claim: In my first 10 months I ended eight wars, including Cambodia.
  - new rationale: the absolute claim of ending eight wars is contradicted — independent review finds several conflicts unresolved, exaggerated, or predating his term, and even the real Cambodia-Thailand case is a frag…
- **`trump_2026:0566`** · True → **Misleading**
  - claim: We stopped them from hanging a lot of them with the threat of serious violence.
  - new rationale: higher-trust reporting contradicted the core assertion that Trump’s threats had halted planned hangings, noting Iran was still signaling fast trials/executions and calling the claimed halt false
- **`trump_2026:0579`** · True → **Misleading**
  - claim: Also, we just approved $1 trillion budget.
  - new rationale: the recent enacted appropriations measure was about $1.2 trillion and overall federal/discretionary budget figures were not a newly approved $1 trillion budget
- **`trump_2026:0635`** · True → **Misleading**
  - claim: We're working closely with the new president of Venezuela, Delcy Rodriguez, to unleash extraordinary economic gains for both of our countri…
  - new rationale: Real economic coordination (OFAC license, oil deals) exists, but framing Rodriguez as elected 'new president' and the relationship as close partnership overstates an unelected interim leader installe…
- **`trump_2026:0642`** · Misleading → **True**
  - claim: But since the raid, we have worked with the new leadership, and they have ordered the closure of that vile prison and released hundreds of…
  - new rationale: Evidence confirms ordered closure of El Helicoide and hundreds of political prisoners released since the raid, with more pending under the amnesty law.

### Newly gated (was decided, now withheld as Unverifiable) — 66 claim(s)

#### Bill Clinton — `clinton_1998` (11)

- **`clinton_1998:0021`** · True → **Unverifiable**
  - claim: We have the smallest government in 35 years, but a more progressive one.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0026`** · True → **Unverifiable**
  - claim: This year, our deficit is projected to be $10 billion and heading lower.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0028`** · Misleading → **Unverifiable**
  - claim: Tonight I come before you to announce that the federal deficit, once so incomprehensibly large that it had 11 zeros, will be, simply, zero.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0029`** · True → **Unverifiable**
  - claim: I will submit to Congress for 1999 the first balanced budget in 30 years.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0038`** · True → **Unverifiable**
  - claim: Now, if we balance the budget for next year, it is projected that we'll then have a sizable surplus in the years that immediately follow.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0077`** · Misleading → **Unverifiable**
  - claim: Now, with these teachers—listen—with these teachers, we will actually be able to reduce class size in the first, second, and third grades t…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0090`** · True → **Unverifiable**
  - claim: In the last five years, we have led the way in opening new markets, with 240 trade agreements that remove foreign barriers to products bear…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0132`** · True → **Unverifiable**
  - claim: I'm pleased to report we have also met that goal, two full years ahead of schedule.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0194`** · Misleading → **Unverifiable**
  - claim: Drug use is on the decline.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0226`** · True → **Unverifiable**
  - claim: His father was a decorated Vietnam vet.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`clinton_1998:0364`** · False → **Unverifiable**
  - claim: Nearly 200 years ago, a tattered flag, its broad stripes and bright stars still gleaming through the smoke of a fierce battle, moved Franci…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.

#### George W. Bush — `gwbush_2006` (5)

- **`gwbush_2006:0024`** · True → **Unverifiable**
  - claim: In 1945, there were about two dozen lonely democracies in the world.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`gwbush_2006:0133`** · True → **Unverifiable**
  - claim: Our economy is healthy and vigorous and growing faster than other major industrialized nations.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`gwbush_2006:0147`** · True → **Unverifiable**
  - claim: In the last five years, the tax relief you passed has left $880 billion in the hands of American workers, investors, small businesses, and…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`gwbush_2006:0155`** · True → **Unverifiable**
  - claim: This year my budget will cut it again, and reduce or eliminate more than 140 programs that are performing poorly or not fulfilling essentia…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`gwbush_2006:0156`** · True → **Unverifiable**
  - claim: By passing these reforms, we will save the American taxpayer another $14 billion next year and stay on track to cut the deficit in half by…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.

#### Barack Obama — `obama_2014` (7)

- **`obama_2014:0055`** · True → **Unverifiable**
  - claim: With the economy picking up speed, companies say they intend to hire more people this year.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`obama_2014:0121`** · True → **Unverifiable**
  - claim: But first, this Congress needs to restore the unemployment insurance you just let expire for 1.6 million people.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`obama_2014:0158`** · True → **Unverifiable**
  - claim: Tonight, I can announce that with the support of the FCC and companies like Apple, Microsoft, Sprint, and Verizon, we’ve got a down payment…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`obama_2014:0177`** · True → **Unverifiable**
  - claim: In the year since I asked this Congress to raise the minimum wage, five states have passed laws to raise theirs.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`obama_2014:0255`** · True → **Unverifiable**
  - claim: When I took office, nearly 180,000 Americans were serving in Iraq and Afghanistan.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`obama_2014:0280`** · True → **Unverifiable**
  - claim: American diplomacy, backed by the threat of force, is why Syria’s chemical weapons are being eliminated, and we will continue to work with…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`obama_2014:0283`** · True → **Unverifiable**
  - claim: As we gather here tonight, Iran has begun to eliminate its stockpile of higher levels of enriched uranium.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.

#### Joe Biden — `biden_2022` (12)

- **`biden_2022:0045`** · True → **Unverifiable**
  - claim: Putin is now isolated from the world more than ever.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0051`** · True → **Unverifiable**
  - claim: The US Department of Justice is assembling a dedicated task force to go after the crimes of Russian oligarchs.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0154`** · True → **Unverifiable**
  - claim: Intel’s CEO, Pat Gelsinger, who is here tonight, told me they are ready to increase their investment from $20 billion to $100 billion.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0168`** · True → **Unverifiable**
  - claim: GM is making the largest investment in its history—$7 billion to build electric vehicles, creating 4,000 jobs in Michigan.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0194`** · True → **Unverifiable**
  - claim: Top business leaders and most Americans support my plan.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0200`** · Misleading → **Unverifiable**
  - claim: Insulin costs about $10 a vial to make.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0244`** · True → **Unverifiable**
  - claim: By the end of this year, the deficit will be down to less than half what it was before I took office.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0285`** · True → **Unverifiable**
  - claim: And we’re launching the “Test to Treat” initiative so people can get tested at a pharmacy, and if they’re positive, receive antiviral pills…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0311`** · True → **Unverifiable**
  - claim: We’ve sent 475 million vaccine doses to 112 countries, more than any other nation.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0362`** · True → **Unverifiable**
  - claim: Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democr…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0366`** · Models split → **Unverifiable**
  - claim: We’ve set up joint patrols with Mexico and Guatemala to catch more human traffickers.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`biden_2022:0437`** · True → **Unverifiable**
  - claim: The VA is pioneering new ways of linking toxic exposures to diseases, already helping more veterans get benefits.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.

#### Donald Trump — `trump_2026` (31)

- **`trump_2026:0035`** · Misleading → **Unverifiable**
  - claim: Mortgage rates are the lowest in four years and falling fast, and the annual cost of a typical new mortgage is down almost $5,000 just sinc…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0040`** · Misleading → **Unverifiable**
  - claim: The stock market has set 53 all-time record highs since the election.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0090`** · False → **Unverifiable**
  - claim: I'm also pleased to say that the next time the Olympic torch is lit, it will be here in America for the 2028 Olympics and it's the summer v…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0100`** · True → **Unverifiable**
  - claim: He was badly wounded and almost killed by enemy machine guns in Luzon.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0110`** · Unverifiable → **Unverifiable**
  - claim: She thought she was going to die.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0161`** · Misleading → **Unverifiable**
  - claim: So, with modest additional contributions, these young people's accounts could grow to over $100,000 or more by the time they turn 18.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0191`** · Misleading → **Unverifiable**
  - claim: Many of the wars I settled was because of the threat of tariffs.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0279`** · Unverifiable → **Unverifiable**
  - claim: She placed bids on 20 homes and lost all of those bids to gigantic investment firms that bypassed inspection, paid all cash and turned thos…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0294`** · True → **Unverifiable**
  - claim: Your 401(k)s are way up.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0310`** · False → **Unverifiable**
  - claim: There's been no more stunning example than Minnesota, where members of the Somali community have pillaged an estimated $19 billion from the…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0325`** · True → **Unverifiable**
  - claim: Delilah Coleman was only five years old in June 2024 when an 18-wheel tractor-trailer plowed into her stopped car traveling at 60 miles an…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0326`** · True → **Unverifiable**
  - claim: The driver was an illegal alien let in by Joe Biden and given a commercial driver's license by open borders politicians in California.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0327`** · True → **Unverifiable**
  - claim: Doctors said Delilah would never be able to walk or talk, have a good life.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0328`** · True → **Unverifiable**
  - claim: She wouldn't even be able to eat again.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0329`** · True → **Unverifiable**
  - claim: But against all odds, she is now in the first grade, learning to walk.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0334`** · Misleading → **Unverifiable**
  - claim: Many, if not most, illegal aliens do not speak English and cannot read even the most basic road signs as to direction, speed, danger or loc…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0400`** · True → **Unverifiable**
  - claim: In 2021, Sage was 14 when school officials in Virginia sought to socially transition her to a new gender, treating her as a boy and hiding…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0402`** · True → **Unverifiable**
  - claim: Before long, a confused Sage ran away from home.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0428`** · True → **Unverifiable**
  - claim: And students and educators in every state have joined the First Lady's efforts in the presidential AI challenge, keeping America's next gen…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0469`** · True → **Unverifiable**
  - claim: Sarah Beckstrom died in order to defend our capital.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0553`** · Misleading → **Unverifiable**
  - claim: And we're working very hard to end the ninth war, the killing and slaughter between Russia and Ukraine, where 25,000 soldiers are dying eac…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0559`** · Misleading → **Unverifiable**
  - claim: They've killed and maimed thousands of American service members and hundreds of thousands and even millions of people with what's called ro…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0564`** · Misleading → **Unverifiable**
  - claim: And just over the last couple of months with the protests, they've killed at least, it looks like 32,000 protests, 32,000 protesters in the…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0569`** · True → **Unverifiable**
  - claim: After Midnight Hammer, they were warned to make no future attempts to rebuild their weapons program in a particular, nuclear weapons, yet t…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0616`** · False → **Unverifiable**
  - claim: And with our new military campaign, we have stopped record amounts of drugs coming into our country and virtually stopped it completely com…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0639`** · True → **Unverifiable**
  - claim: But after Enrique ran for office and opposed Maduro, he was kidnaped by Maduro's security forces and thrown into the regime's really infamo…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0652`** · True → **Unverifiable**
  - claim: But the deeds of one warrior that night will live forever in the eternal chronicles of military valor, Chief Warrant Officer Five, Eric Slo…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0666`** · True → **Unverifiable**
  - claim: The success of the entire mission and the lives of his fellow warriors hinge on Eric's ability to take the searing pain, it was unbelievabl…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0683`** · True → **Unverifiable**
  - claim: In the skies over Korea in 1952, Royce was in the dogfight of a lifetime, legendary dogfight.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0685`** · True → **Unverifiable**
  - claim: It was his first aerial combat of the war and despite being massively outnumbered and outgunned, Royce led the takedown of four enemy jets…
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.
- **`trump_2026:0686`** · True → **Unverifiable**
  - claim: His story was secret for over 50 years.
  - new rationale: Forced UNVERIFIABLE: the evidence pack failed the quality gate (too few qualifying Tier-1..3 sources bearing on the core assertion) after one targeted re-retrieval.

### Newly decided (was withheld/split, now a substantive ruling) — 17 claim(s)

#### Bill Clinton — `clinton_1998` (4)

- **`clinton_1998:0056`** · Unverifiable → **True**
  - claim: Since then, this Congress—across party lines—and the American people have responded, in the most important year for education in a generati…
  - new rationale: Each cited initiative — 3,000 charters, America Reads with college tutors, Head Start toward a million, classroom connectivity — is corroborated by government records, and the claim frames them as co…
- **`clinton_1998:0113`** · Unverifiable → **True**
  - claim: Recent months have brought serious financial problems to Thailand, Indonesia, South Korea, and beyond.
  - new rationale: IMF bailouts for South Korea and Indonesia plus contemporaneous reporting confirm serious 1997–98 financial crises across Thailand, Indonesia, and South Korea.
- **`clinton_1998:0173`** · Unverifiable → **True**
  - claim: Last year I cohosted the very first White House Conference on Child Care with one of our foremost experts, America's First Lady.
  - new rationale: Multiple sources confirm Clinton co-hosted the first-ever White House Conference on Child Care with the First Lady on Oct 23, 1997 — the year before the Jan 1998 utterance.
- **`clinton_1998:0332`** · Unverifiable → **True**
  - claim: This year Hillary and I launched the White House Millennium Program to promote America's creativity and innovation, and to preserve our her…
  - new rationale: contemporary official and news records show Clinton and Hillary Clinton launched the White House Millennium Program in 1997 to celebrate American creativity and preserve heritage and culture into the…

#### George W. Bush — `gwbush_2006` (3)

- **`gwbush_2006:0095`** · Unverifiable → **True**
  - claim: The same is true of Iran, a nation now held hostage by a small clerical elite that is isolating and repressing its people.
  - new rationale: Contemporaneous HRW, UN, and State Department reporting confirm Iran was governed by a clerical elite (Guardian Council vetting) that repressed and isolated its people.
- **`gwbush_2006:0206`** · Unverifiable → **True**
  - claim: We've made a good start in the early grades with the No Child Left Behind Act, which is raising standards and lifting test scores across ou…
  - new rationale: Government NAEP data show real early-grade (4th) math (+3) and reading (+1) gains through 2005, supporting the hedged 'good start in the early grades' claim, though NCLB causation is contested.
- **`gwbush_2006:0245`** · Unverifiable → **True**
  - claim: And this good work is being led by our First Lady, Laura Bush.
  - new rationale: Multiple contemporaneous government and official sources confirm Laura Bush led the Helping America's Youth initiative as of the utterance date.

#### Barack Obama — `obama_2014` (4)

- **`obama_2014:0045`** · Unverifiable → **True**
  - claim: The Joining Forces alliance that Michelle and Jill Biden launched has already encouraged employers to hire or train nearly 400,000 veterans…
  - new rationale: 290,000 hired/trained by April 2013 (E4) trajectory supports ~400,000 by the January 2014 utterance, and the figure appears in official SOTU records.
- **`obama_2014:0086`** · Models split → **True**
  - claim: It’s not just oil and natural gas production that’s booming; we’re becoming a global leader in solar, too.
  - new rationale: Contemporary data show record US solar installs (surpassing Germany, joining top-tier installers) and rapid growth, supporting the hedged 'becoming a global leader' claim.
- **`obama_2014:0087`** · Unverifiable → **True**
  - claim: Every four minutes, another American home or business goes solar; every panel pounded into place by a worker whose job can’t be outsourced.
  - new rationale: GTM Research documents one U.S. solar system installed every four minutes in 2013, and BLS/DOE evidence confirms installation is domestic, non-outsourceable labor.
- **`obama_2014:0198`** · Unverifiable → **True**
  - claim: Right now, it helps about half of all parents at some point.
  - new rationale: CBPP and CEA reports confirm about half of taxpayers with children claim the EITC at some point over a multi-decade period, matching the claim.

#### Joe Biden — `biden_2022` (3)

- **`biden_2022:0127`** · Models split → **Misleading**
  - claim: That’s why it was so important to pass the Bipartisan Infrastructure Law—the most sweeping investment to rebuild America in history.
  - new rationale: The core—a massive historic infrastructure investment—is real, but the absolute superlative 'most sweeping in history' overstates it; the one independent source (E10) calibrates to 'largest in decade…
- **`biden_2022:0216`** · Unverifiable → **True**
  - claim: Many families pay up to $14,000 a year for child care per child.
  - new rationale: High-cost-area child care demonstrably reaches ~$14k+ per child, supporting the 'up to $14,000' framing.
- **`biden_2022:0432`** · Models split → **Misleading**
  - claim: But cancer from prolonged exposure to burn pits ravaged Heath’s lungs and body.
  - new rationale: (none recorded)

#### Donald Trump — `trump_2026` (3)

- **`trump_2026:0030`** · Models split → **False**
  - claim: But in 12 months, my administration has driven core inflation down to the lowest level in more than five years.
  - new rationale: the latest supported core CPI figure was a recent low but only framed as lowest since March 2021, not more than five years, while core PCE was not at a five-year low
- **`trump_2026:0099`** · Unverifiable → **True**
  - claim: He fought bravely in the famous Battle of Manila, worked so hard.
  - new rationale: Independent biographical reporting confirms Taggart served under MacArthur and fought in the real Battle of Manila/Luzon, matching the core assertion without material exaggeration.
- **`trump_2026:0341`** · Unverifiable → **True**
  - claim: Her mother Jacqueline went home to look for her and she found her lying dead in a bathtub bleeding profusely after being stabbed 25 times.
  - new rationale: Contemporary sentencing coverage confirms mother found 16-yr-old cheerleader Lizbeth Medina stabbed to death in a bathtub before the Christmas parade, matching the claim.

### Split changes (model-split status changed) — 9 claim(s)

#### Barack Obama — `obama_2014` (3)

- **`obama_2014:0034`** · False → **Models split**
  - claim: Upward mobility has stalled.
  - new rationale: (none recorded)
- **`obama_2014:0070`** · True → **Models split**
  - claim: Over the past five years, my administration has made more loans to small business owners than any other.
  - new rationale: (none recorded)
- **`obama_2014:0285`** · True → **Models split**
  - claim: Unprecedented inspections help the world verify, every day, that Iran is not building a bomb.
  - new rationale: (none recorded)

#### Joe Biden — `biden_2022` (2)

- **`biden_2022:0385`** · True → **Models split**
  - claim: I signed 80 bipartisan bills into law last year.
  - new rationale: (none recorded)
- **`biden_2022:0397`** · True → **Models split**
  - claim: I believe in recovery, and I celebrate the 23 million Americans in recovery.
  - new rationale: (none recorded)

#### Donald Trump — `trump_2026` (4)

- **`trump_2026:0015`** · Misleading → **Models split**
  - claim: Today, our border is secure.
  - new rationale: (none recorded)
- **`trump_2026:0043`** · False → **Models split**
  - claim: In four long years, the last administration got less than $1 trillion in new investment in the United States.
  - new rationale: (none recorded)
- **`trump_2026:0257`** · True → **Models split**
  - claim: But a few weeks ago, she logged on to the website and got that same drug that cost costs $4,000, got it for under $500, a reduction of much…
  - new rationale: (none recorded)
- **`trump_2026:0462`** · True → **Models split**
  - claim: After a four-month deployment, she voluntarily extended her service, and her rank was going to be lifted.
  - new rationale: (none recorded)

## 6. Spend + provenance

`proxy` is **ledger-true** (billed by the LiteLLM proxy key). `off-proxy` is an **ESTIMATE** — models called outside the proxy, costed from token counts at published list rates.

| speech | old run → new run | claims | legs | proxy (ledger-true) | off-proxy (ESTIMATE) | log total | brief stated | Δ |
|---|---|---|---|---|---|---|---|---|
| clinton_1998 | `7c59e9e0` → `d0010426` | 92 | 1 | $0.8791 | $5.9663 | $6.8454 | $6.85 | -0.0046 |
| gwbush_2006 | `92f39851` → `74a89c5f` | 48 | 1 | $0.2479 | $2.8344 | $3.0823 | $3.08 | +0.0023 |
| obama_2014 | `28965cdf` → `4de8a551` | 96 | 2 | $0.7577 | $6.4780 | $7.2357 | $7.24 | -0.0043 |
| biden_2022 | `7208bbbb` → `37744fc8` | 111 | 2 | $0.7361 | $7.6371 | $8.3732 | $8.00 | +0.3732 |
| trump_2026 | `23939712` → `4ee5a251` | 182 | 1 | $1.7272 | $11.8864 | $13.6136 | $13.61 | +0.0036 |
| shape backfill | (haiku sidecars) | — | — | $0.63 | $0.0000 | $0.63 | — | — |
| **total** |  |  |  | $4.9780 | $34.8022 | $39.7802 | $38.80 |  |

- `obama_2014`: leg 1 banked 80/96 rows (proxy $0.6586, off-proxy est $5.4007) before an L-W worker failure; leg 2 ran the remaining 16 (proxy $0.0991, off-proxy est $1.0773)
- `biden_2022`: leg 1 banked 60/111 rows (proxy $0.3480, off-proxy est $4.3690) before a browsing-model timeout; leg 2 ran the remaining 51 (proxy $0.3881, off-proxy est $3.2681)

**Spend discrepancies (not smoothed):**

- biden_2022: run logs total $8.3732 vs $8.00 stated in the DC-6 brief (+0.3732)
- corpus: run logs total $39.7802 (incl. $0.63 shape backfill) vs ~$38.80 stated in the brief; the brief's per-speech figures themselves sum to $38.78 BEFORE the backfill, so the stated total appears to double-count the backfill as already included

## 7. Staged render + badge diff

- site root: `/tmp/dc6-site` (staged; `site-pca/` untouched)
- artifacts picked by the renderer: `clinton_1998`→`d0010426`, `gwbush_2006`→`74a89c5f`, `obama_2014`→`4de8a551`, `biden_2022`→`37744fc8`, `trump_2026`→`4ee5a251`
- all five NEW artifacts selected: **True**
- `check_site(strict_buckets=True)`: **PASS — 0 violations**

- badge diff keyed on (speaker, normalised claim_text): matched **529**, only-old **1**, only-new **0**, verdict changes **139**
- id overlap between the two renders: **0** — id-keying would be vacuous: **True**
  - only-old: Donald Trump — (claim text unavailable) [False]

- reconciliation: per-speech diffs report 141 changes; 2 of them are invisible on the published badge (panel-Unverifiable → gate-forced-Unverifiable), so the badge diff should show 139 and shows 139 — **AGREE**

## 8. Proposed corrections ledger

- changed verdicts: **141**
- expressible as public corrections: **125**
- not expressible in the ledger vocabulary: **16** (model-split transitions; and claims that moved from panel-Unverifiable to gate-forced-Unverifiable, which publish the same badge)
- archive target: `data/corrections-archive-2026-08-06.json` (17 entries + 1 note(s))
- proposed live ledger: 125 entries + 1 note

`data/corrections.json` is NOT modified by this script. The reset is applied at publish time, under the gate.

**Publish mechanics:** these entries are a PUBLICATION RECORD of what the rebuild changed — they must not be re-applied to the artifacts. `apply_to_artifact` fails closed when an entry's `old_verdict` does not match the row, and the rebuilt rows already carry the NEW verdicts. Render with `--corrections skip`; the corrections page still publishes the full ledger and its note.

