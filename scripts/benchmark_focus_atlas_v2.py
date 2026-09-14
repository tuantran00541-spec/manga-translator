from __future__ import annotations

from pathlib import Path

from app.parameters import DETECTOR_INPUT_SIZE
import scripts.benchmark_focus_atlas as base

SEAM_GUARD = 24
base.OUT = Path("benchmark-results/focus-atlas-v2/report.json")
base.ATLAS_GUARD = SEAM_GUARD


def _pair_layout(a: dict, b: dict, guard: int = SEAM_GUARD):
    aw, ah = int(a["resized_w"]), int(a["resized_h"])
    bw, bh = int(b["resized_w"]), int(b["resized_h"])
    size = int(DETECTOR_INPUT_SIZE)

    # Preserve each ROI's standalone resize scale. Only the padding/context is
    # compacted: content may touch the outer atlas boundary on the dimension
    # that already occupied 1024 px in the standalone letterbox. A seam of
    # letterbox-value pixels remains between independent ROIs.
    vertical_total = ah + guard + bh
    if max(aw, bw) <= size and vertical_total <= size:
        top = max(0, (size - vertical_total) // 2)
        return [
            (max(0, (size - aw) // 2), top),
            (max(0, (size - bw) // 2), top + ah + guard),
        ]

    horizontal_total = aw + guard + bw
    if max(ah, bh) <= size and horizontal_total <= size:
        left = max(0, (size - horizontal_total) // 2)
        return [
            (left, max(0, (size - ah) // 2)),
            (left + aw + guard, max(0, (size - bh) // 2)),
        ]
    return None


base._pair_layout = _pair_layout

if __name__ == "__main__":
    base.main()
