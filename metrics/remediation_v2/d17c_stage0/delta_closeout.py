"""D17-c Stage 0 — closing the 448-vs-445 statistical-agency census delta.

Two counts of the same corpus disagreed by three items: Fable's 448 against
445 from the shipped ``classify_ex``. A three-item gap in the population that
defines the whole work list is worth closing exactly rather than splitting.

It closes exactly, and the shipped code is right. The gap is an artifact of how
a hand count matches, and it decomposes into two independent mechanisms that
reproduce Fable's per-agency deltas (Census +3, USDA-NASS +1, NCHS -1) only
when applied together:

1. **Press-prefix breadth.** The statistical-agency registry declares five
   ``press_prefixes``, but the file says they are ADDITIVE to the six
   ``stat_press_prefixes`` it inherits from ``tier_registry.yaml`` — "one list,
   one meaning; a second copy would drift". The shipped union is nine. The
   inherited ``/newsroom`` is what denies three census.gov items.

2. **Path case-folding.** ``statistical_agency._url_path`` lowercases before
   matching. Exactly two corpus items turn on that, and they move in OPPOSITE
   directions, which is why neither mechanism alone reproduces the delta:
   ``nass.usda.gov/Newsroom/...`` is correctly denied as a press page, and
   ``cdc.gov/MMWR/...`` is correctly admitted as a document.

Fable's second hypothesis — that ``quickstats.nass.usda.gov`` might fail to
resolve from entry ``nass.usda.gov`` by suffix, putting the registry rationale
and the code in disagreement — is REFUTED below: it resolves.

$0, offline, no model calls. Run from anywhere.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from truthbot.publish.heads import publishing_heads          # noqa: E402
from truthbot.verify import statistical_agency as sa         # noqa: E402

#: The registry's OWN additive list — what a hand count would naturally apply.
FIVE = ("/media/releases", "/media/press", "/news/press", "/opa/pr", "/about/new")

_orig_path, _orig_reg = sa._url_path, sa.load_registry
_REGISTRY = _orig_reg()


def census(*, five_only: bool, case_sensitive: bool) -> Counter:
    """Per-agency census under a stated matching convention."""
    reg = dataclasses.replace(_REGISTRY, press_prefixes=FIVE) if five_only else _REGISTRY
    sa.load_registry = lambda: reg

    def patched(url: str) -> str:
        try:
            path = urlsplit(url if "://" in url else f"//{url}").path or "/"
        except ValueError:
            return ""
        return path if case_sensitive else path.lower()

    sa._url_path = patched
    try:
        counts: Counter = Counter()
        for _sid, path in publishing_heads().items():
            with open(path) as fh:
                doc = json.load(fh)
            for _claim, pack in (doc.get("evidence") or {}).items():
                for item in pack or []:
                    url = item.get("source_url") or ""
                    if sa.classify_ex(url)[0]:
                        counts[sa.agency_for(url) or "?"] += 1
        return counts
    finally:
        sa._url_path, sa.load_registry = _orig_path, _orig_reg


shipped = census(five_only=False, case_sensitive=False)

print("=== the delta, mechanism by mechanism ===")
for label, five, case in (
    ("shipped        (9 prefixes, case-insensitive)", False, False),
    ("prefixes only  (5 prefixes, case-insensitive)", True, False),
    ("case only      (9 prefixes, case-SENSITIVE)  ", False, True),
    ("BOTH           (5 prefixes, case-SENSITIVE)  ", True, True),
):
    counts = census(five_only=five, case_sensitive=case)
    delta = {k: counts[k] - shipped[k]
             for k in set(shipped) | set(counts) if counts[k] != shipped[k]}
    print(f"  {label}  total={sum(counts.values()):>4}  delta={delta or '{}'}")

both = census(five_only=True, case_sensitive=True)
expected = {"Census": 3, "USDA-NASS": 1, "NCHS/CDC-statistical": -1}
actual = {k: both[k] - shipped[k]
          for k in set(shipped) | set(both) if both[k] != shipped[k]}
print(f"\nreproduces Fable's 448?  {sum(both.values()) == 448}")
print(f"reproduces Fable's per-agency delta?  {actual == expected}")
assert sum(both.values()) == 448 and actual == expected, actual

# ── hypothesis (b): does quickstats resolve by suffix? ──────────────────────

print("\n=== hypothesis (b): quickstats.nass.usda.gov suffix resolution ===")
for url in ("https://quickstats.nass.usda.gov/results/ABC123",
            "https://quickstats.nass.usda.gov/",
            "https://www.nass.usda.gov/Publications/x.pdf"):
    allowed, reason = sa.classify_ex(url)
    print(f"  {str(allowed):<6} {reason:<28} agency={sa.agency_for(url)!r}")
print("-> RESOLVES by suffix. The registry rationale and the code AGREE;")
print("   hypothesis (b) is refuted, and the USDA-NASS +1 is case-folding.")

# ── the two items the case rule turns on ────────────────────────────────────

print("\n=== the two case-sensitive items ===")
for url in ("https://www.nass.usda.gov/Newsroom/2026/01-30-2026.php",
            "https://www.cdc.gov/MMWR/preview/mmwrhtml/mm5524a2.htm"):
    ci = sa.classify_ex(url)
    sa._url_path = lambda u: urlsplit(u).path or "/"
    cs = sa.classify_ex(url)
    sa._url_path = _orig_path
    print(f"  {url}")
    print(f"    shipped (lowercased): {ci}")
    print(f"    case-sensitive      : {cs}")
