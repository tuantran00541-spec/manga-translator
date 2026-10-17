"""PASS when out/stuck.txt lists exactly the stuck requests."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
want = open(os.path.join(here, "expected.txt")).read().split()
got = open(os.path.join(sys.argv[1], "out/stuck.txt")).read().split()
assert got == want, (len(got), len(want), sorted(set(got) ^ set(want))[:5])
print("PASS")
