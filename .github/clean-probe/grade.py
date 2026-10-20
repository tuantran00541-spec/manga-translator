"""Score cleaned hidden slices: text and ghosts left by the CTD letter model, and art changed away from any text."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector import ctd_mask
from app.detector.kiuyha_detector import KiuyhaTextDetector
from scripts.verify import GHOST, GHOST_MIN_PX, GHOST_SHARE, LEFT_MIN_PX, LEFT_SHARE, MIN_PIECES, PAD_PX, PIECE_PX, _blocks, text_mask

ZONE_PX = 32  # changes this close to detected text may belong to its erase
CHANGED = 40  # a channel moved this much is a real change, not compression noise


def side(raw: np.ndarray, clean: np.ndarray, box, path: Path) -> None:
    x1, y1, x2, y2 = box
    tiles = [cv2.resize(i[y1:y2, x1:x2], None, fx=min(1.0, 320 / max(1, y2 - y1)), fy=min(1.0, 320 / max(1, y2 - y1))) for i in (raw, clean)]
    cv2.imwrite(str(path), np.concatenate([tiles[0], np.full((tiles[0].shape[0], 8, 3), 255, np.uint8), tiles[1]], axis=1),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def grade(raw_dir: Path, clean_dir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    kiuyha = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    totals = {"slices": 0, "missing": 0, "blocks": 0, "text_left": 0, "ghosts": 0, "damaged_slices": 0}
    damage, rows, crops = [], [], 0
    for path in sorted(raw_dir.glob("*.png")):
        raw = cv2.imread(str(path))
        clean = cv2.imread(str(clean_dir / path.name)) if (clean_dir / path.name).exists() else None
        totals["slices"] += 1
        boxes = kiuyha.text_boxes(raw)
        letters = text_mask(raw.shape[:2], boxes)
        if clean is None or clean.shape != raw.shape:
            totals["missing"] += 1
            clean = raw.copy()
        h, w = letters.shape
        row = {"slice": path.name, "blocks": 0, "text": 0, "ghost": 0}
        for block, (x1, y1, x2, y2) in _blocks(letters):
            bx1, by1, bx2, by2 = max(0, x1 - PAD_PX), max(0, y1 - PAD_PX), min(w, x2 + PAD_PX), min(h, y2 + PAD_PX)
            area = np.zeros((by2 - by1, bx2 - bx1), np.uint8)
            area[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1] = block
            area = cv2.dilate(area, np.ones((2 * PAD_PX + 1,) * 2, np.uint8)) > 0
            before = ctd_mask.probability(np.ascontiguousarray(raw[by1:by2, bx1:bx2]))
            after = ctd_mask.probability(np.ascontiguousarray(clean[by1:by2, bx1:bx2]))
            inked = (before > ctd_mask.THRESHOLD) & area
            _, _, stats, _ = cv2.connectedComponentsWithStats(inked.astype(np.uint8))
            if int((stats[1:, cv2.CC_STAT_AREA] >= PIECE_PX).sum()) < MIN_PIECES:
                continue
            row["blocks"] += 1
            ink = max(1, int(inked.sum()))
            left, faint = int(((after > ctd_mask.THRESHOLD) & area).sum()), int(((after > GHOST) & area).sum())
            kind = "text" if left >= max(LEFT_MIN_PX, LEFT_SHARE * ink) else "ghost" if faint >= max(GHOST_MIN_PX, GHOST_SHARE * ink) else ""
            if kind:
                row[kind] += 1
                crops += 1
                if crops <= 40:
                    side(raw, clean, (bx1, by1, bx2, by2), out / f"{kind}-{crops:02d}-{path.stem}.jpg")
        zone = letters.astype(np.uint8)
        for b in boxes:
            zone[b.y1:b.y2, b.x1:b.x2] = 1
        zone = cv2.dilate(zone, np.ones((2 * ZONE_PX + 1,) * 2, np.uint8)) > 0
        changed = (np.abs(raw.astype(np.int16) - clean.astype(np.int16)).max(axis=2) > CHANGED) & ~zone
        row["damage"] = round(float(changed.sum()) / max(1, int((~zone).sum())), 5)
        damage.append(row["damage"])
        if row["damage"] > 0.002:
            totals["damaged_slices"] += 1
            ys, xs = np.nonzero(changed)
            side(raw, clean, (max(0, int(xs.min()) - 20), max(0, int(ys.min()) - 20), min(w, int(xs.max()) + 20), min(h, int(ys.max()) + 20)),
                 out / f"damage-{path.stem}.jpg")
        for k in ("blocks", "text", "ghost"):
            totals[{"text": "text_left", "ghost": "ghosts"}.get(k, k)] += row[k]
        rows.append(row)
    totals["clean_share"] = round(1 - (totals["text_left"] + totals["ghosts"]) / max(1, totals["blocks"]), 3)
    totals["damage_mean"] = round(float(np.mean(damage)) if damage else 0.0, 5)
    report = {"totals": totals, "slices": rows}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return totals


if __name__ == "__main__":
    print(json.dumps(grade(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))))
