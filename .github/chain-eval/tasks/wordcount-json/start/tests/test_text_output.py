import subprocess, sys


def test_text_output_is_unchanged():
    out = subprocess.run([sys.executable, "wc.py", "sample.txt"], capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[:4] == ["lines: 4", "words: 12", "unique: 8", "the 3"]
