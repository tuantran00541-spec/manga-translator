"""Runs minisql.py on every examples/NN.sql and compares with examples/NN.out; prints what fails."""
import glob
import os
import sys

from sqlcheck import matches, ordered, run

here = os.path.dirname(os.path.abspath(__file__))
failed = 0
paths = sorted(glob.glob(os.path.join(here, "examples", "*.sql")))
for path in paths:
    sql = open(path, encoding="utf-8").read().strip()
    expected = [line.split("\t") for line in open(path[:-4] + ".out", encoding="utf-8").read().splitlines()]
    code, out = run(os.path.join(here, "minisql.py"), os.path.join(here, "data"), sql, block=False)
    if code != 0 or not matches(expected, out, ordered(sql)):
        failed += 1
        print(f"FAIL {os.path.basename(path)}: {sql}\n  expected {len(expected)} rows: {expected[:3]}\n  got (exit {code}): {out.splitlines()[:3]}")
print(f"{len(paths) - failed}/{len(paths)} examples pass")
sys.exit(1 if failed else 0)
