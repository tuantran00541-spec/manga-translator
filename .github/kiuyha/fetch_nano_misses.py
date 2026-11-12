import json
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://github.com/tuantran00541-spec/manga-translator"
       "/releases/download/kiuyha-nano-miss-v1/kiuyha-nano-miss-v1.zip")


def main(out):
    root = Path(out)
    zip_path = root / "kiuyha-nano-miss-v1.zip"
    print(f"downloading {URL}", flush=True)
    urllib.request.urlretrieve(URL, zip_path)
    print(f"extracting {zip_path.stat().st_size / 1e6:.0f} MB", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    zip_path.unlink()
    manifest = json.loads(
        (Path(__file__).parent / "nano_misses.json").read_text())
    expect = len({(r["chapter"], r["source_page"], r["slice_index"])
                  for r in manifest["kept"]})
    n_img = len(list((root / "images" / "train").glob("nano-miss-*.jpg")))
    n_lbl = len(list((root / "labels" / "train").glob("nano-miss-*.txt")))
    assert n_img == expect == n_lbl, (
        f"nano-miss slices: images {n_img}, labels {n_lbl}, "
        f"expected {expect}")
    n_train = len(list((root / "images" / "train").glob("*.jpg")))
    stats_path = root / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats["train"] = n_train
        stats["train_nano_miss_slices"] = n_img
        stats_path.write_text(json.dumps(stats, indent=1))
    print(f"added {n_img} nano-miss slices "
          f"({len(manifest['kept'])} miss boxes); "
          f"kiuyha-data now holds {n_train} train images", flush=True)
    print(json.dumps({"nano_miss_slices": n_img,
                      "train_total": n_train}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
