import json
from collections import Counter, defaultdict

from truthbot.publish.heads import publishing_heads
from truthbot.verify.statistical_agency import agency_for, classify_ex

SP = ["gwbush_2006", "clinton_1998", "obama_2014", "biden_2022", "trump_2026"]
grid = defaultdict(Counter)
nullgrid = defaultdict(Counter)
tot_items = tot_null = tot_series = tot_series_null = 0
claims_touched = set()
per_speech_items = Counter()
per_speech_null = Counter()
enum = []

for sid, p in publishing_heads().items():
    d = json.load(open(p))
    for cl, pack in (d.get("evidence") or {}).items():
        for idx, e in enumerate(pack or [], start=1):
            tot_items += 1
            per_speech_items[sid] += 1
            isnull = e.get("supports_claim") is None
            if isnull:
                tot_null += 1
                per_speech_null[sid] += 1
            u = e.get("source_url") or ""
            if classify_ex(u)[0]:
                a = agency_for(u) or "?"
                grid[a][sid] += 1
                tot_series += 1
                if isnull:
                    nullgrid[a][sid] += 1
                    tot_series_null += 1
                    claims_touched.add(cl)
                    enum.append((sid, cl, f"E{idx}", a, u))

hdr = f"{'format':<20}" + "".join(f"{s.split('_')[0][:8]:>9}" for s in SP) + f"{'TOT':>7}"
print("=== ITEM 1: CENSUS per-format x per-speech (SHIPPED classify_ex) ===")
print(hdr)
for a in sorted(grid, key=lambda x: -sum(grid[x].values())):
    print(f"{a:<20}" + "".join(f"{grid[a][s]:>9}" for s in SP)
          + f"{sum(grid[a].values()):>7}")
print(f"{'ALL':<20}" + "".join(f"{sum(grid[a][s] for a in grid):>9}" for s in SP)
      + f"{tot_series:>7}")

print()
print("=== stance-NULL subset (the D17-c work list) ===")
print(hdr)
for a in sorted(nullgrid, key=lambda x: -sum(nullgrid[x].values())):
    print(f"{a:<20}" + "".join(f"{nullgrid[a][s]:>9}" for s in SP)
          + f"{sum(nullgrid[a].values()):>7}")
print(f"{'ALL':<20}" + "".join(f"{sum(nullgrid[a][s] for a in nullgrid):>9}" for s in SP)
      + f"{tot_series_null:>7}")
print(f"claims touched by the null-series set: {len(claims_touched)}")

print()
print("=== ITEM 3: recompute the trump floor from artifacts ===")
t_items = per_speech_items["trump_2026"]
t_null = per_speech_null["trump_2026"]
t_series_null = sum(nullgrid[a]["trump_2026"] for a in nullgrid)
resid = t_null - t_series_null
print(f"trump items={t_items} null={t_null} rate={t_null / t_items * 100:.2f}%")
print(f"trump stance-null on statistical series (D17-c target) = {t_series_null}")
print(f"residual after converting ALL of them = {resid}/{t_items} = "
      f"{resid / t_items * 100:.2f}%  (ceiling 15%)")
print(f"clears 15%? {resid / t_items * 100 <= 15.0}")

with open("/tmp/d17c_enum.tsv", "w") as fh:
    fh.write("speech\tclaim_sid\tevidence_id\tformat\turl\n")
    for r in sorted(enum):
        fh.write("\t".join(r) + "\n")
print(f"\nITEM 2 enumeration written: {len(enum)} rows -> /tmp/d17c_enum.tsv")
