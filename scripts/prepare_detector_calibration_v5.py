from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from app.downloader.registry import download_chapter
from app.downloader.slicer import slice_image
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_LETTERBOX_VALUE


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read calibration image: {path}")
    return image


def _letterbox_rgb(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Empty calibration image")
    scale = DETECTOR_INPUT_SIZE / max(h, w)
    nh = max(1, int(h * scale))
    nw = max(1, int(w * scale))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full(
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3),
        DETECTOR_LETTERBOX_VALUE,
        dtype=np.uint8,
    )
    pad_x = (DETECTOR_INPUT_SIZE - nw) // 2
    pad_y = (DETECTOR_INPUT_SIZE - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas


def _variants(image: np.ndarray, window_height: int) -> list[tuple[str, np.ndarray]]:
    h = image.shape[0]
    if h <= 1:
        return [("full", image)]
    target = max(64, min(int(window_height), h))
    if h <= target:
        crop_h = max(64, int(round(h * 0.75)))
        crop_h = min(crop_h, h)
        return [
            ("full", image),
            ("top", image[:crop_h]),
            ("bottom", image[h - crop_h :]),
        ]
    return [
        ("full", image),
        ("top", image[:target]),
        ("bottom", image[h - target :]),
    ]


def _even_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    count = max(1, min(int(count), total))
    if count == 1:
        return [total // 2]
    return sorted({int(round(v)) for v in np.linspace(0, total - 1, count)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-slices", type=int, default=16)
    parser.add_argument("--window-height", type=int, default=1344)
    args = parser.parse_args()

    root = Path(args.output_dir)
    if root.exists():
        shutil.rmtree(root)
    raw_dir = root / "raw"
    sliced_dir = root / "sliced"
    tensor_dir = root / "tensors"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sliced_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = download_chapter(args.chapter_url, raw_dir)
    slice_paths: list[Path] = []
    for source_index, raw_path in enumerate(raw_paths):
        items = slice_image(
            raw_path,
            sliced_dir,
            f"{source_index:03d}",
            return_metadata=True,
        )
        for item in items:
            path = Path(item["path"]) if isinstance(item, dict) else Path(item)
            slice_paths.append(path)

    chosen = _even_indices(len(slice_paths), args.base_slices)
    records = []
    tensor_index = 0
    for slice_index in chosen:
        image = _read_image(slice_paths[slice_index])
        for variant_name, variant in _variants(image, args.window_height):
            canvas = _letterbox_rgb(variant)
            out_path = tensor_dir / f"{tensor_index:03d}.npy"
            np.save(out_path, canvas, allow_pickle=False)
            records.append(
                {
                    "tensor": out_path.name,
                    "slice_index": slice_index,
                    "variant": variant_name,
                    "source_shape": list(image.shape[:2]),
                    "variant_shape": list(variant.shape[:2]),
                }
            )
            tensor_index += 1

    metadata = {
        "chapter_url": args.chapter_url,
        "source_pages": len(raw_paths),
        "slices": len(slice_paths),
        "base_slices": len(chosen),
        "calibration_tensors": len(records),
        "input_size": DETECTOR_INPUT_SIZE,
        "window_height": args.window_height,
        "records": records,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "records"}, indent=2))
    if len(records) < 12:
        raise RuntimeError(f"Too few calibration tensors: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
