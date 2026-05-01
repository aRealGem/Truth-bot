# OpenAI / Gemini "stripped sources" — strip-mechanism audit (2026-05-01)

## tl;dr

The 38.9% aggregate "fabrication rate" reported in
`metrics/ab_probe_20260430/arm_b/run_summaries/bd9e2552-…json` is
produced by **set-intersection in [`apply_url_grounding`](../../src/truthbot/verify/adapters/base.py)**
between model-reported URLs and the **tool-retrieved** URL set captured
during the same API call — *not* by HTTP HEAD validation. The
"stripped" signal therefore means "model claimed a citation the search
tool never returned for this call," which conflates **(a)** real URLs
the harness failed to capture, **(b)** real URLs from the same domain
the tool didn't fetch (model pattern-matched a plausible path), and
**(c)** wholly fabricated paths. Operator-side curl probing is required
to disentangle (a/b/c); from this sandbox both `*.gov` and
`web.archive.org` return 403 to all probes, so external classification
is not possible here.

## Mechanism (corrected)

[`ground_truth_web_sources`](../../src/truthbot/verify/adapters/base.py)
normalizes every URL via `_normalize_url_for_compare` (lower-case
host, drop `www.`, drop `:443`/`:80`, drop trailing `/`, **preserve
path/query**), then keeps only model-reported URLs whose normalized
form appears verbatim in the tool-retrieved set. Misses become
`stripped_source_count`. There is no network call, no HEAD/GET, no
domain whitelist.

Concretely, exact-path matching means a model citation of
`/news.release/archives/cpi_12182025.htm` is stripped even if the tool
retrieved `/news.release/archives/cpi_12182025.pdf` for the same
release. Examples below confirm this.

## Sample (arm-B run `bd9e2552`, OpenAI batch sidecar)

8 verdicts, 26 model-reported URLs, **25 stripped (96%)**. Every
stripped URL is on a real US-government domain
(`bls.gov` / `eia.gov` / `congress.gov` / `cbp.gov` / `fbi.gov`).

| claim | label | kept | stripped | smoking-gun pattern |
|---|---|---:|---:|---|
| `c3727673…` | False | 0 | 3 | three plausible BLS CPI release URLs (2026 paths) |
| `0231c9de…` | False | 1 (`cpi_12182025.pdf`) | 3 (`cpi_12182025.htm`, `cpi_01132026.htm`, `cpi_02132026.htm`) | model cited htm + future months for the same BLS-release pattern; tool only retrieved the December PDF |
| `45f8c4b7…` | Mostly True | 0 | 2 | two `eia.gov/todayinenergy/detail.php?id=…` paths |
| `a271b130…` | False | 0 | 3 | congress.gov congressional-record + CRS paths |
| `5fbfe50f…` | False | 0 | 4 | cbp.gov ports-of-entry + I-92 + monthly-update paths |
| `85d4bce2…` | Exaggerated | 0 | 5 | cbp.gov enforcement-stats + multiple monthly-update paths |
| `17a10bf1…` | Mostly True | 0 | 3 | fbi.gov UCR + crime-statistics paths |
| `ce1a5f0f…` | Unverifiable | 0 | 2 | fbi.gov UCR (duplicates of 17a10bf1) |

For **xAI batch** in the same run — `55 model-reported / 49 web_sources / 0 stripped` — the strip rate is 0%, almost certainly because xAI's adapter routes the same model that produced the verdict through the same search context, so the "tool-retrieved" set already contains every URL the model cites. This is a **harness symmetry artefact**, not a credibility difference between the models.

## Probe constraint (sandbox)

| target | egress observation |
|---|---|
| `*.gov` (all 13 sampled URLs) | uniform `403` on HEAD, `403` on ranged GET |
| `web.archive.org` CDX API | initial round returned non-empty bodies that parsed to `[]` (no captures) for known-good URLs (`bls.gov/cpi.htm`, `bls.gov/`); subsequent round returned `403 Forbidden` directly |

Curl reachability from this sandbox is therefore **not informative**.
Operator runs needed (see "Operator follow-up" below).

## Recommendation

Two interventions, in order:

### 1. Surface `model_reported_sources` as a separate "Model-cited (unverified)" tier in published reports — *correct regardless of audit outcome*

The intersection itself is doing the right thing; the credibility issue
is the **binary hide-or-show** behavior in
[`publish/site.py`](../../src/truthbot/publish/site.py). Today, when
`web_sources=[]` and `model_reported_sources=[…]`, readers see no
citation at all even though the model produced one. That's worse than
showing the citation with a "didn't validate" caveat.

Proposed render: under the existing combined evidence/sources block,
add a second sub-list "Model-cited URLs that didn't validate (n)" with
domain-only, non-clickable, italic styling. Distinguishes the tier
visually; preserves audit trail; honest about validator state.

This applies whether the URLs turn out to be (a) real-but-uncaptured,
(b) real-pattern-near-matches, or (c) fabricated — readers can click
through (or not) and judge for themselves. Closes Cursor's brief
item 2.

### 2. Operator follow-up — complete the (a/b/c) classification from MBP

Run from a non-blocked egress (`/Users/<you>/.../Truth-bot`):

```bash
cat > /tmp/strip_urls_2026-05.txt <<'EOF'
https://www.bls.gov/opub/ted/2026/consumer-prices-up-2-4-percent-over-the-year-ended-january-2026.htm
https://www.bls.gov/news.release/archives/cpi_01132026.htm
https://www.bls.gov/news.release/archives/cpi_02132026.htm
https://www.bls.gov/news.release/archives/cpi_12182025.pdf
https://www.eia.gov/todayinenergy/detail.php?id=55099
https://www.eia.gov/todayinenergy/detail.php?id=65184
https://www.congress.gov/congressional-record/volume-171/issue-41/senate-section/article/S1466-4
https://www.cbp.gov/border-security/ports-entry/overview?language=es
https://www.cbp.gov/newsroom/national-media-release/cbp-releases-february-2025-monthly-update
https://www.cbp.gov/newsroom/national-media-release/cbp-releases-march-2024-monthly-update
https://www.fbi.gov/news/press-releases/fbi-releases-2024-reported-crimes-in-the-nation-statistics
https://www.fbi.gov/services/cjis/ucr/
https://www.cbp.gov/newsroom/stats/cbp-enforcement-statistics
EOF

while IFS= read -r u; do
  code=$(curl -sI -L --max-time 10 -A "Mozilla/5.0" -o /dev/null -w "%{http_code}" "$u")
  printf '%-3s  %s\n' "$code" "$u"
done < /tmp/strip_urls_2026-05.txt
```

Classify each URL:
- **(a)** `200/204/206` on HEAD or ranged-GET → real-and-reachable, harness should have captured it
- **(b)** `403/405/429` on HEAD with `200` on ranged-GET → real, HEAD-blocked; harness captured under same constraint
- **(c)** `404` on both → fabricated path, model invented a plausible URL

If majority **(a)** + **(b)**: the OpenAI tool-URL-capture path in
[`verify/adapters/openai.py`](../../src/truthbot/verify/adapters/openai.py)
is missing URLs the tool actually visited (probable harness bug). The
[`_walk_output_for_urls`](../../src/truthbot/verify/adapters/openai.py)
helper added in `a319480` may help here — extending it to walk
`tool_call`-typed output items in addition to whatever it currently
walks.

If majority **(c)**: the strip is doing real anti-fabrication work,
and intervention 1 is the only correct response. Tighten the OpenAI
prompt's CITATION DISCIPLINE block to forbid path-pattern guessing
("if the search tool did not return a specific URL for this fact,
emit no `web_sources` for that fact").

## Related code

- [`src/truthbot/verify/adapters/base.py:564-624`](../../src/truthbot/verify/adapters/base.py)
  — `ground_truth_web_sources`
- [`src/truthbot/verify/adapters/base.py:627-659`](../../src/truthbot/verify/adapters/base.py)
  — `apply_url_grounding`
- [`src/truthbot/verify/adapters/base.py:694-859`](../../src/truthbot/verify/adapters/base.py)
  — `build_multi_verdicts` (where the defensive MRS backfill happens)
- [`src/truthbot/verify/adapters/openai.py`](../../src/truthbot/verify/adapters/openai.py)
  — `_walk_output_for_urls` (potential harness fix locus)
- [`src/truthbot/publish/site.py`](../../src/truthbot/publish/site.py)
  — render locus for intervention 1
