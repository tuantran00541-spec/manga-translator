from __future__ import annotations

import copy

import cv2
import numpy as np

_GEOMETRY_KEYS = ("x1", "y1", "x2", "y2")


def geometry_dict(box: dict) -> dict[str, int]:
    return {key: int(box[key]) for key in _GEOMETRY_KEYS}


def remap_local_mask_page_space(
    mask: np.ndarray | None,
    source_box: dict,
    target_box: dict,
) -> np.ndarray | None:
    """Move a box-local mask between geometries without scaling page artwork.

    Detector masks are defined in page coordinates, even though they are stored as
    arrays local to their boxes. When a user expands/crops/moves a box, resizing the
    old mask would distort glyph geometry. Instead, place the source mask back in
    page space and crop it into the target box.
    """
    if mask is None:
        return None

    sx1, sy1, sx2, sy2 = (int(source_box[key]) for key in _GEOMETRY_KEYS)
    tx1, ty1, tx2, ty2 = (int(target_box[key]) for key in _GEOMETRY_KEYS)
    source_w, source_h = sx2 - sx1, sy2 - sy1
    target_w, target_h = tx2 - tx1, ty2 - ty1
    if source_w <= 0 or source_h <= 0 or target_w <= 0 or target_h <= 0:
        return None

    source_mask = mask
    if source_mask.ndim == 3:
        source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2GRAY)
    if source_mask.shape[:2] != (source_h, source_w):
        source_mask = cv2.resize(
            source_mask, (source_w, source_h), interpolation=cv2.INTER_NEAREST
        )

    target_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
    ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return target_mask

    source_slice = source_mask[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1]
    target_mask[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1] = source_slice
    return target_mask


def reconcile_detector_geometry_override(new_box: dict, existing_box: dict) -> bool:
    """Apply a persisted detector-box geometry override to a fresh detection.

    This is intentionally used only before inpaint, while ``new_box`` still carries
    ``_mask_array``. The fresh detector mask is remapped into the user's edited
    geometry. ``existing_box`` is the per-job snapshot used by ``pipeline.py``;
    clearing its flag prevents the legacy pipeline branch from discarding the mask
    a second time. The canonical manifest is not mutated here.
    """
    if not existing_box.get("geometry_overridden") or "_mask_array" not in new_box:
        return False

    source_geometry = geometry_dict(new_box)
    target_geometry = geometry_dict(existing_box)
    new_box["_mask_array"] = remap_local_mask_page_space(
        new_box.get("_mask_array"), source_geometry, target_geometry
    )
    new_box.update(target_geometry)
    new_box["geometry_overridden"] = True

    anchor = existing_box.get("detector_anchor")
    new_box["detector_anchor"] = copy.deepcopy(anchor if isinstance(anchor, dict) else source_geometry)

    # ``_process_page`` builds old_by_id from this job-local existing_boxes copy
    # after stable ID assignment. Neutralize only that legacy branch so it cannot
    # replace the remapped mask with None.
    existing_box["geometry_overridden"] = False
    return True
