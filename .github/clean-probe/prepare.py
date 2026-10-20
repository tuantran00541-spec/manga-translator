"""Download the probe's chapters with the current app, keep the dev slices and pick the hidden ones, cleaned by today's pipeline too."""
import json
import shutil
import sys
from pathlib import Path

import cv2

from app.config import KIUYHA_TEXT_MODEL
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.image_io import read_image
from app.manifest_utils import load_manifest_raw
from app.processing_pipeline_factory import build_processing_pipeline

DEV = ("c1a90001", "https://asurascans.com/comics/shadow-slave-05c7df14/chapter/1")
HIDDEN = {"solo": ("c1a90002", "https://asurascans.com/comics/solo-swordmaster-3ec3b16f/chapter/1"),
          "academy": ("c1a90003", "https://asurascans.com/comics/the-academy-s-weapon-replicator-05c7df14/chapter/1"),
          "hero": ("c1a90004", "https://asurascans.com/comics/the-hero-cannot-rest-3ec3b16f/chapter/1")}
WITH_TEXT, WITHOUT_TEXT = 24, 3  # hidden slices per chapter


def spread(items: list, n: int) -> list:
    return items if len(items) <= n else [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def main(out: Path) -> int:
    pipeline = build_processing_pipeline()
    dev = out / "work" / "dev"
    dev.mkdir(parents=True)
    for i, page in enumerate(pipeline.download_chapter(DEV[1], DEV[0], workers=2)["pages"]):
        cv2.imwrite(str(dev / f"{i:03d}.png"), read_image(Path(page["original"])))
    kiuyha = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    raw, main_clean = out / "hidden" / "raw", out / "hidden" / "main"
    raw.mkdir(parents=True), main_clean.mkdir(parents=True)
    picked = {}
    for tag, (cid, url) in HIDDEN.items():
        pages = pipeline.download_chapter(url, cid, workers=2)["pages"]
        pipeline.process_pages(cid, list(range(len(pages))))
        pages = load_manifest_raw(cid)["pages"]
        texts, empty = [], []
        for i, page in enumerate(pages):
            (texts if kiuyha.text_boxes(read_image(Path(page["original"]))) else empty).append(i)
        chosen = spread(texts, WITH_TEXT) + spread(empty, WITHOUT_TEXT)
        picked[tag] = {"slices": len(pages), "with_text": len(texts), "chosen": chosen}
        for i in chosen:
            name = f"{tag}-{i:03d}.png"
            cv2.imwrite(str(raw / name), read_image(Path(pages[i]["original"])))
            clean = pages[i].get("clean")
            if clean:
                cv2.imwrite(str(main_clean / name), read_image(Path(clean)))
    (out / "hidden" / "picked.json").write_text(json.dumps(picked, indent=1))
    print(json.dumps(picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
