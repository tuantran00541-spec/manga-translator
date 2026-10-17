"""PASS when wc.py keeps its old output and the new options behave."""
import json, os, subprocess, sys
root = sys.argv[1]
def run(*args):
    return subprocess.run([sys.executable, "wc.py", *args], cwd=root, capture_output=True, text=True, timeout=60)
assert run("sample.txt").stdout == "lines: 4\nwords: 12\nunique: 8\nthe 3\ncat 2\nsat 2\nran 1\na 1\n", run("sample.txt").stdout
assert run("sample.txt", "--top", "2").stdout.splitlines()[3:] == ["the 3", "cat 2"]
data = json.loads(run("--json", "--top", "3", "sample.txt").stdout)
assert data == {"lines": 4, "words": 12, "unique": 8, "top": [["the", 3], ["cat", 2], ["sat", 2]]}, data
assert json.loads(run("sample.txt", "--json").stdout)["top"][4] == ["a", 1]
bad = run("nope.txt")
assert bad.returncode == 2 and bad.stderr.strip() == "error: no such file: nope.txt", (bad.returncode, bad.stderr)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=root, capture_output=True, text=True, timeout=120)
assert r.returncode == 0 and len([f for f in os.listdir(os.path.join(root, "tests")) if f.startswith("test_")]) >= 2, r.stdout[-500:]
print("PASS")
