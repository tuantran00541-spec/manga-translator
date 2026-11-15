#!/usr/bin/env python3
"""Cut VI crops for pilot pairs on the runner (VM must not bulk-download VI).

Inputs: pairs jsonl from the en-vi-pilot-build run (one chapter) + the
chapter's VI images (downloaded on the runner by download_vi_series.py).
Outputs: VI crops named <ch>_<stem>_<x1>_<y1>.png (same convention as
cut_fresh_crops.py) + vi_sizes.json {vi_img: [w, h]} for the re-pair step.
Usage: cut_vi_crops.py <pairs_jsonl> <vi_chapter_dir> <out_dir> <chapter>
Chapter is a CLI arg (builder pairs jsonl has no chapter field).
"""
import json
import os
import sys

from PIL import Image


def main():
    pairs_path, vi_dir, out_dir, ch = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    crops_dir = os.path.join(out_dir, "vi-crops")
    os.makedirs(crops_dir, exist_ok=True)
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8") if l.strip()]
    sizes = {}
    n_cut = 0
    for p in pairs:
        vi_img = p["vi_img"]
        x1, y1, x2, y2 = [int(v) for v in p["vi_box"]]
        stem = os.path.splitext(vi_img)[0]
        name = f"{ch}_{stem}_{x1}_{y1}.png"
        path = os.path.join(crops_dir, name)
        if not os.path.exists(path):
            im = Image.open(os.path.join(vi_dir, vi_img)).convert("RGB")
            if vi_img not in sizes:
                sizes[vi_img] = list(im.size)
            im.crop((x1, y1, x2, y2)).save(path)
            n_cut += 1
        elif vi_img not in sizes:
            sizes[vi_img] = list(Image.open(os.path.join(vi_dir, vi_img)).size)
    with open(os.path.join(out_dir, "vi_sizes.json"), "w", encoding="utf-8") as f:
        json.dump(sizes, f, ensure_ascii=False)
    print(f"cut {n_cut} VI crops for {len(pairs)} pairs; sizes for {len(sizes)} images", flush=True)


if __name__ == "__main__":
    main()
