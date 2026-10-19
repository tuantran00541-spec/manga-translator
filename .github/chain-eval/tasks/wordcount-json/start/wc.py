import sys
from collections import Counter

def main():
    path = sys.argv[1]
    text = open(path, encoding="utf-8").read()
    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    words = [w for w in words if w]
    counts = Counter(words)
    print(f"lines: {len(text.splitlines())}")
    print(f"words: {len(words)}")
    print(f"unique: {len(counts)}")
    for word, n in counts.most_common(5):
        print(f"{word} {n}")

main()
