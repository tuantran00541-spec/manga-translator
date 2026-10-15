"""Fetch the hard-case training set (Total-Text hard shards + webtoon + AnimeText
hard) and append it to an existing kiuyha-data directory as extra training data.

The zip holds images/ and labels/ in YOLO format (class 0 = text), the same
label format build.py writes. Filenames carry distinct prefixes (tt-hard-,
wt-, at_), so nothing collides with the base set.

Usage:
    python .github/kiuyha/fetch_hard.py kiuyha-data

Needs nothing beyond the stdlib (urllib + zipfile).
"""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://github.com/tuantran00541-spec/manga-translator"
       "/releases/download/kiuyha-hard-v1/kiuyha-hard-v1.zip")


def main(out):
    root = Path(out)
    zip_path = root / "kiuyha-hard-v1.zip"
    print(f"downloading {URL}", flush=True)
    urllib.request.urlretrieve(URL, zip_path)
    print(f"extracting {zip_path.stat().st_size / 1e6:.0f} MB", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        added = sum(1 for n in z.namelist() if n.startswith("images/"))
        z.extractall(root)
    zip_path.unlink()
    # The zip holds flat images/ + labels/; kiuyha-data wants images/train/.
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    for p in (root / "images").glob("*.jpg"):
        p.rename(root / "images" / "train" / p.name)
    for p in (root / "labels").glob("*.txt"):
        p.rename(root / "labels" / "train" / p.name)
    n_img = len(list((root / "images" / "train").glob("*.jpg")))
    n_lbl = len(list((root / "labels" / "train").glob("*.txt")))
    print(f"added {added} hard images; "
          f"kiuyha-data now holds {n_img} train images / {n_lbl} train labels",
          flush=True)
    stats_path = root / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats["train"] = n_img
        stats["train_hard_added"] = added
        stats_path.write_text(json.dumps(stats, indent=1))
    print(json.dumps({"hard_train_total": n_img}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
