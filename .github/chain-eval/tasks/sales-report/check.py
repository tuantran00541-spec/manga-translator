"""PASS when report.py rebuilds the right out/summary.json from a fresh CSV."""
import csv, collections, json, os, random, subprocess, sys
os.chdir(sys.argv[1])
def expect(path):
    rev, units, skipped = collections.defaultdict(float), collections.Counter(), 0
    for r in csv.DictReader(open(path)):
        if not r["units"].strip() or int(r["units"]) < 0:
            skipped += 1
            continue
        rev[r["region"].strip().lower()] += int(r["units"]) * float(r["unit_price"])
        units[r["product"]] += int(r["units"])
    return {"revenue_by_region": {k: round(v, 2) for k, v in rev.items()}, "top_product": units.most_common(1)[0][0], "skipped_rows": skipped}
def same(a, b):
    return a["top_product"] == b["top_product"] and a["skipped_rows"] == b["skipped_rows"] and set(a["revenue_by_region"]) == set(b["revenue_by_region"]) \
        and all(abs(a["revenue_by_region"][k] - b["revenue_by_region"][k]) < 0.01 for k in a["revenue_by_region"])
got = json.load(open("out/summary.json"))
assert same(got, expect("data/sales.csv")), ("summary", got)
random.seed(1)
rows = ["date,region,product,units,unit_price"] + [f"2026-02-01,{random.choice([' West','west','EAST'])},{random.choice('xyz')},{random.choice(['', '-1', '3', '7'])},{random.choice([1.5, 2.0])}" for _ in range(40)]
open("data/sales.csv", "w").write("\n".join(rows) + "\n")
subprocess.run([sys.executable, "report.py"], check=True, timeout=60)
assert same(json.load(open("out/summary.json")), expect("data/sales.csv")), "fresh data"
print("PASS")
