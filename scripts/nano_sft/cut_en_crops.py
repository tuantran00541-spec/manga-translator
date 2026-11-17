#!/usr/bin/env python3
"""Cut EN crops for pilot DP pairs on the runner.

The build workflow only cuts EN crops for its own (greedy) pairs; the
DP re-pair adds pairs whose EN boxes were never cut. This cuts them
from the re-downloaded EN chapter zip.
Usage: cut_en_crops.py <pairs_jsonl> <en_zip> <out_dir> <chapter>

NOTE (2026-11-17, TMPT ch11-20): MangaOwl image URLs change extension
between fetches (build zip had 001.webp, re-download zip did not), so
the zip entry is resolved by STEM (001), not by the exact en_img name
from the build run. Crop filenames use the stem either way.
"""
import json
import os
import sys
import zipfile

from PIL import Image


def main():
    pairs_path, en_zip, out_dir, ch = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    crops_dir = os.path.join(out_dir, "en-crops")
    os.makedirs(crops_dir, exist_ok=True)
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8") if l.strip()]
    n_cut = 0
    with zipfile.ZipFile(en_zip) as zf:
        # Resolve by stem: re-downloaded zips may carry a different
        # extension for the same image index (see module docstring).
        by_stem = {os.path.splitext(n)[0]: n for n in zf.namelist()}
        cache = {}
        for p in pairs:
            en_img = p["en_img"]
            x1, y1, x2, y2 = [int(v) for v in p["en_box"]]
            stem = os.path.splitext(en_img)[0]
            name = f"{ch}_{stem}_{x1}_{y1}.png"
            path = os.path.join(crops_dir, name)
            if os.path.exists(path):
                continue
            entry = by_stem.get(stem)
            if entry is None:
                raise KeyError(f"no zip entry with stem {stem!r} "
                               f"(pairs say {en_img!r}; zip has "
                               f"{sorted(by_stem)[:5]}...)")
            if entry not in cache:
                with zf.open(entry) as f:
                    cache[entry] = Image.open(f).convert("RGB").copy()
            cache[entry].crop((x1, y1, x2, y2)).save(path)
            n_cut += 1
    print(f"cut {n_cut} EN crops for {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
