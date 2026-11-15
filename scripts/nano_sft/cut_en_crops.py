#!/usr/bin/env python3
"""Cut EN crops for pilot DP pairs on the runner.

The build workflow only cuts EN crops for its own (greedy) pairs; the
DP re-pair adds pairs whose EN boxes were never cut. This cuts them
from the re-downloaded EN chapter zip.
Usage: cut_en_crops.py <pairs_jsonl> <en_zip> <out_dir> <chapter>
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
        cache = {}
        for p in pairs:
            en_img = p["en_img"]
            x1, y1, x2, y2 = [int(v) for v in p["en_box"]]
            stem = os.path.splitext(en_img)[0]
            name = f"{ch}_{stem}_{x1}_{y1}.png"
            path = os.path.join(crops_dir, name)
            if os.path.exists(path):
                continue
            if en_img not in cache:
                with zf.open(en_img) as f:
                    cache[en_img] = Image.open(f).convert("RGB").copy()
            cache[en_img].crop((x1, y1, x2, y2)).save(path)
            n_cut += 1
    print(f"cut {n_cut} EN crops for {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
