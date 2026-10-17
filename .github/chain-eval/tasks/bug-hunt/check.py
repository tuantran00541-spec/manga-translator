"""PASS when the given tests and hidden edge cases pass."""
import copy, os, subprocess, sys
root = sys.argv[1]
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=root, capture_output=True, text=True, timeout=120)
assert r.returncode == 0, r.stdout[-800:]
sys.path.insert(0, root)
from ranges import merge_ranges, total_length
assert merge_ranges([]) == [] and total_length([]) == 0
assert merge_ranges([[1, 2], [2, 3]]) == [[1, 3]], "touching"
assert merge_ranges([[1, 10], [2, 3], [4, 5]]) == [[1, 10]], "contained"
data = [[5, 6], [1, 3], [2, 4]]
before = copy.deepcopy(data)
assert merge_ranges(data) == [[1, 4], [5, 6]] and data == before, "input kept"
assert total_length([[5, 6], [1, 3], [2, 4]]) == 4
print("PASS")
