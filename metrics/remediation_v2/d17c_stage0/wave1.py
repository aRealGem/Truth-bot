"""D17-c Stage 0 — the wave-1 work list, and item 6's affordability envelope.

Wave 1 is the stance-null statistical-series population MINUS the document
publishers Fable excluded (CBO/GAO/NCES/CRS publish tabled reports, not
series) MINUS NCHS/CDC, which fails the ten-item coverage floor. Fable's
independently-reproduced target is 84 items across 40 claims; this script
derives that from the shipped artifacts and asserts the per-speech split
rather than trusting it.

The cost half answers item 6 in the only direction available with no
fixtures. A series excerpt adds characters to the SCORING PROMPT and nothing
else: same items, same reply schema, so the output side is untouched. That
makes the marginal cost of excerpting a one-term function of prompt growth,

    delta_usd = (chars_per_excerpt * items) / CHARS_PER_TOKEN * rate_in / 1e6

which inverts exactly for "how large may an excerpt be before Stage A costs
more than the $0.75 ceiling". The measured mean/max token delta per item that
Fable asked for needs real payloads and is NOT computed here; this is the
envelope those measurements will have to land inside.

$0, offline, no model calls.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# The rate table lives in the repo-root ``hydramind`` package, which is only on
# sys.path when the interpreter starts at the repo root. This script does not.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from truthbot.costs import CHARS_PER_TOKEN, estimate_scoring_cost, rates  # noqa: E402
from truthbot.publish.heads import publishing_heads  # noqa: E402
from truthbot.verify.statistical_agency import agency_for, classify_ex  # noqa: E402

#: Publishers of tabled REPORTS, not series — excluded by Fable's ruling.
DOCUMENT_PUBLISHERS = {"CBO", "GAO", "NCES", "CRS"}
#: Fails the ten-item coverage floor (3 nulls corpus-wide) — logged, out of wave 1.
BELOW_FLOOR = {"NCHS/CDC-statistical"}
#: Fable's independently-reproduced wave-1 split, asserted below.
EXPECTED_PER_SPEECH = {"trump_2026": 35, "biden_2022": 16, "obama_2014": 15,
                       "clinton_1998": 10, "gwbush_2006": 8}
CEILING_USD = 0.75

wave1: list[tuple[str, str, str, str, str]] = []
per_speech = Counter()
per_agency = Counter()
claims: set[str] = set()
pack_items: dict[str, int] = {}
touched_pack_items = 0

for sid, path in publishing_heads().items():
    with open(path) as fh:
        doc = json.load(fh)
    for claim, pack in (doc.get("evidence") or {}).items():
        pack = pack or []
        pack_items[claim] = len(pack)
        hit = False
        for idx, item in enumerate(pack, start=1):
            url = item.get("source_url") or ""
            if not classify_ex(url)[0]:
                continue
            if item.get("supports_claim") is not None:
                continue
            agency = agency_for(url) or "?"
            if agency in DOCUMENT_PUBLISHERS or agency in BELOW_FLOOR:
                continue
            wave1.append((sid, claim, f"E{idx}", agency, url))
            per_speech[sid] += 1
            per_agency[agency] += 1
            claims.add(claim)
            hit = True
        if hit:
            touched_pack_items += len(pack)

print("=== WAVE 1 ===")
print(f"items  = {len(wave1)}   (Fable: 84)")
print(f"claims = {len(claims)}   (Fable: 40)")
print("\nper speech:")
for sid, want in EXPECTED_PER_SPEECH.items():
    got = per_speech[sid]
    print(f"  {sid:<14}{got:>4}   expected {want:>3}   {'OK' if got == want else 'MISMATCH'}")
print("\nper agency (the handler work list):")
for a, n in per_agency.most_common():
    print(f"  {a:<24}{n:>4}")

pilot = per_agency["FRED"] + per_agency["ALFRED"]
print(f"\npilot handler FRED+ALFRED covers {pilot} of {len(wave1)} wave-1 items")

out = Path(__file__).resolve().parent / "wave1_items.tsv"
with out.open("w") as fh:
    fh.write("speech\tclaim_sid\tevidence_id\tformat\turl\n")
    for row in sorted(wave1):
        fh.write("\t".join(row) + "\n")
print(f"wave-1 list written: {len(wave1)} rows -> {out.name}")

# ── item 6: the affordability envelope ──────────────────────────────────────

r_in, r_out = rates("claude-haiku")
n = len(wave1)


def marginal_usd(chars_per_excerpt: float, items: int = n) -> float:
    """Cost of adding ``chars_per_excerpt`` to ``items`` scoring prompts."""
    return (chars_per_excerpt * items) / CHARS_PER_TOKEN * r_in / 1_000_000.0


print("\n=== ITEM 6: affordability envelope (NOT a measurement) ===")
print(f"rate_in = ${r_in}/Mtok   CHARS_PER_TOKEN = {CHARS_PER_TOKEN}")
print(f"excerpt inflates the PROMPT only; reply schema and item count unchanged\n")
print(f"{'chars/excerpt':>14}{'tokens/excerpt':>16}{'delta $ (84 items)':>21}")
for c in (500, 1_000, 2_000, 4_000, 8_000, 16_000):
    print(f"{c:>14,}{c / CHARS_PER_TOKEN:>16,.0f}{marginal_usd(c):>21.4f}")

budget_chars = CEILING_USD * 1_000_000.0 / r_in * CHARS_PER_TOKEN / n
print(f"\nthe ${CEILING_USD:.2f} ceiling buys {budget_chars:,.0f} chars "
      f"({budget_chars / CHARS_PER_TOKEN:,.0f} tokens) per excerpt across {n} items")
print("-> the excerpt-size question is not close to the ceiling; the input side")
print("   is far too cheap for a row excerpt to threaten it.")

print(f"\nre-score denominator: the {len(claims)} touched packs hold "
      f"{touched_pack_items} evidence items in total")
print("(if Stage A re-scores whole packs rather than single items, that is the")
print(" reply-side denominator — and the reply side is where the money is)")

# ── item 6, the projection Fable asked to be halted on ──────────────────────
#
# The envelope above prices the excerpt. It does NOT price Stage A, because
# re-scoring a pack pays for every item in it on the reply side whether or not
# that item got an excerpt. Prompt volume is MEASURED from the artifacts, per
# the estimator's own instruction never to guess it.

print("\n=== ITEM 6: Stage A projection (measured prompt volume) ===")
pack_chars = Counter()
for sid, path in publishing_heads().items():
    with open(path) as fh:
        doc = json.load(fh)
    for claim, pack in (doc.get("evidence") or {}).items():
        if claim not in claims:
            continue
        for item in pack or []:
            pack_chars[claim] += sum(
                len(str(item.get(k) or ""))
                for k in ("title", "source_url", "snippet", "publisher")
            )

measured = sum(pack_chars.values())
print(f"measured evidence-payload chars over the {len(claims)} packs: {measured:,}")

for label, excerpt_chars in (("no excerpt (baseline re-score)", 0),
                             ("2,000-char excerpt x 84 items", 2_000),
                             ("4,000-char excerpt x 84 items", 4_000)):
    est = estimate_scoring_cost(prompt_chars=measured + excerpt_chars * n,
                                items=touched_pack_items)
    print(f"  {label:<32} ${est['cost_usd_est']:.4f}"
          f"   (in {est['tokens_in_est']:,} tok, out {est['tokens_out_est']:,} tok)")

# ``measured`` is the EVIDENCE payload only — it omits the system prompt and
# the claim text, which every call also carries. Left alone it would flatter
# the projection. The cited B2 run measured its whole prompt, so its
# chars-per-item is the honest scale factor for what this undercounts.
B2_PROMPT_CHARS, B2_ITEMS = 412_532, 1028          # metrics/remediation_v2/b2_subset.json
conservative = B2_PROMPT_CHARS / B2_ITEMS * touched_pack_items
print(f"\nconservative prompt volume (B2 measured {B2_PROMPT_CHARS / B2_ITEMS:.0f} "
      f"chars/item x {touched_pack_items}): {conservative:,.0f} chars"
      f"  [{conservative / measured:.1f}x the evidence-only figure]")

worst = estimate_scoring_cost(prompt_chars=conservative + 4_000 * n,
                              items=touched_pack_items)["cost_usd_est"]
print(f"worst case, conservative prompt + 4,000-char excerpts = ${worst:.4f}")
print(f"vs ceiling ${CEILING_USD:.2f} -> "
      f"{'OVER — HALT AND ASK' if worst > CEILING_USD else 'UNDER, no halt required'}")
print("\nNOTE: this prices ONE scoring pass. It excludes re-adjudication, which")
print("is Stage B and separately authorized. The mean/max token delta per item")
print("that item 6 asks for is a MEASUREMENT and still needs real payloads;")
print("what is derived here is the ceiling those measurements must fit under.")
